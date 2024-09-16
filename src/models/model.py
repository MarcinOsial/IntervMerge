import torch
from heads import get_classification_head
from utils.utils import load_weights
import torch.nn as nn
import wandb
from models.loreft_models import *
from utils.utils import make_functional
from configs.config import Config
import os
import errno
torch.set_printoptions(threshold=10_000)


def get_intervention_class(intervention_type):
    if intervention_type == 'default':
        return NoreftIntervention
    elif intervention_type == 'NodiReFT':
        return NodireftIntervention
    elif intervention_type == 'LobiReFT':
        return LobireftIntervention
    elif intervention_type == 'ConsReFT':
        return ConsreftIntervention
    elif intervention_type == 'DiReFT':
        return DireftIntervention
    elif intervention_type == 'LoReFT':
        return LoreftIntervention
    elif intervention_type == 'miniInterv_best_efficient': #one matrix: h + R^T(b) with shift
        return MiniInterventionParamEffic
    elif intervention_type == 'miniInterv_efficient': #one matrix: h + R^T(b)
        return MiniInterventionParam
    elif intervention_type == 'miniInterv': # default pattern (h + W2^T(W1h + b − W2h))
        return MiniInterv
    elif intervention_type == 'miniIntervMulti':  #hardcoded positions
        return MultiMiniInterv
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")


class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.interventions = nn.ModuleDict()
        _, self.names = make_functional(model)
        self.config = Config.get_instance()

        intervention_class = get_intervention_class(self.config.intervention_type) 

        # Initialize interventions based on the configuration
        for dataset_name, dataset_config in self.config.reft_config.items():
            if isinstance(dataset_config, str):  # This is a reference to a shared configuration
                dataset_config = self.config.reft_config[dataset_config]  # Redirect to the shared configuration

            self.interventions[dataset_name] = nn.ModuleDict()
            for layer_info in dataset_config['localization']:
                layer_index = str(layer_info['layer']) 
                if layer_index not in self.interventions[dataset_name]:
                    self.interventions[dataset_name][layer_index] = nn.ModuleList()
                for position in layer_info['positions']:
                    intervention = intervention_class(self.config.model_hidden_size, self.config.low_rank_dimension, dropout=dataset_config['dropout'], act_fn=dataset_config['act_fn'], layer_index=layer_index, dataset_name=dataset_name)
                    intervention.requires_grad_(True)
                    self.interventions[dataset_name][layer_index].append(intervention)


        self.interventions = self.interventions.to(self.config.device)   

        if self.config.layerwise:
            self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
            rlambdas = torch.ones(len(paramslist[0]), len(paramslist)-1) * self.config.prior
        if self.config.taskwise:
            self.pretrain_lambdas = torch.ones(1, 1)
            rlambdas = torch.ones(1, len(paramslist)-1) * self.config.prior

        if self.config.use_lambda_custom:
            rlambdas = self.config.custom_rlambdas
            
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        # print(str(self.lambdas_raw)[:100])
        
        self.classifier = []
        for dataset_name in self.config.exam_datasets:
            classification_head = get_classification_head(self.config, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to('cpu')) #self.config.device
            self.classifier.append(layer_name)

    def save(self, save_directory: str, custom_filename: str = "all_components.pt", include_model: bool = True) -> None:
        """
        Save all interventions along with their configuration in a single file, optionally including the model's trainable parameters.

        Args:
            save_directory (str): The directory to save the components.
            custom_filename (str): Custom filename for the saved file.
            include_model (bool): If True, save the model's trainable parameters as well.
        """
        if not isinstance(save_directory, str):
            raise ValueError("Expected save_directory to be a string")
        if not isinstance(include_model, bool):
            raise ValueError("Expected include_model to be a boolean")

        try:
            os.makedirs(save_directory, exist_ok=True)
        except OSError as e:
            if e.errno == errno.EACCES:
                msg = f"Permission denied: Unable to create directory {save_directory}."
                self.config.logger.error(msg)
                raise PermissionError(msg) from e
            else:
                msg = f"Failed to create directory {save_directory}: {e.strerror}"
                self.config.logger.error(msg)
                raise OSError(msg) from e
        
        # Prepare the dictionary to save
        save_dict = {
            "interventions": {},
            "lambdas": {},
            "config": self.config
        }
        
        # Collect state dicts for all interventions, integrating them into the config structure
        if self.config.train_intervention and self.config.use_intervention:
            for dataset_name, config in self.config.reft_config.items():
                if isinstance(self.config.reft_config[dataset_name], str):
                    config = self.config.reft_config.get(list(self.config.reft_config.keys())[0], {})
                
                intervention_config = {}
                for layer_info in config['localization']:
                    layer_index = str(layer_info['layer'])
                    positions = layer_info['positions']
                    component = layer_info['component']
                    
                    # Collect interventions for each position in the layer
                    interventions = []
                    for index_pos, pos in enumerate(positions):
                        intervention = self.interventions[dataset_name][layer_index][index_pos]
                        interventions.append(intervention.state_dict())

                    # Update the layer_info with the interventions' weights
                    intervention_config[layer_index] = {
                        'positions': positions,
                        'component': component,
                        'weights': interventions
                    }
                
                save_dict["interventions"][dataset_name] = intervention_config
        
        # Collect and save lambda parameters
        if self.lambdas_raw.requires_grad:
                save_dict['lambda'] = self.lambdas_raw

        # Optionally include the model's trainable parameters
        if include_model:
            model_state_dict = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            save_dict["model_trainable_params"] = model_state_dict
        
        # Save everything to a single file
        full_path = os.path.join(save_directory, custom_filename)
        torch.save(save_dict, full_path)
        msg = "Intervention have been successfully saved to: " + full_path
        print(msg)
        self.config.logger.info(msg)

    def load(self, load_directory: str, custom_filename: str = "all_components.pt", include_model: bool = False) -> None:
        """
        Load all interventions along with their configuration from a single file, optionally including the model's trainable parameters and lambdas.

        Args:
            load_directory (str): The directory from which to load the components.
            custom_filename (str): Custom filename for the loaded file.
            include_model (bool): If True, load the model's trainable parameters as well.
        """
        # Construct the full path to the file
        full_path = os.path.join(load_directory, custom_filename)

        # Load the dictionary from the file
        try:
            save_dict = torch.load(full_path)
        except FileNotFoundError:
            self.config.logger.error(f"File not found: {full_path}")
            raise
        except Exception as e:
            self.config.logger.error(f"Failed to load file {full_path}: {str(e)}")
            raise

        # Restore the configuration
        self.resume_config = save_dict["config"]

        # Restore interventions from the saved state dicts
        for dataset_name, layers in save_dict["interventions"].items():
            for layer_index, layer_info in layers.items():
                positions = layer_info['positions']
                component = layer_info['component']
                weights = layer_info['weights']

                for index_pos, pos in enumerate(positions):
                    intervention_state_dict = weights[index_pos]
                    intervention = self.interventions[dataset_name][layer_index][index_pos]
                    intervention.load_state_dict(intervention_state_dict)

        if 'lambda' in save_dict:
            self.lambdas_raw = save_dict['lambda']

    def set_lambdas_raw(self, new_lambdas):
        self.lambdas_raw.data = new_lambdas

    def get_lambdas_raw(self):
        return self.lambdas_raw.data

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        self.pretrain_lambdas = self.pretrain_lambdas.cuda()
        task_lambdas = task_lambdas.cuda()
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass


    def collect_trainable_params(self):
        params = {}
        if self.config.train_lambdas:
            if isinstance(self.lambdas_raw, torch.Tensor) and self.lambdas_raw.requires_grad:
                params.update({"lambdas": self.lambdas_raw})
        
        uniq = 0
        if self.config.train_intervention:
            for idx_dataset, dataset_name_al in enumerate(self.config.reft_config):
                if isinstance(self.config.reft_config[dataset_name_al], str): #shared params
                    continue
                for idx_conf, (k, v) in enumerate(self.config.reft_config[dataset_name_al].items()):
                    for num, dict_layer in enumerate(self.config.reft_config[dataset_name_al]['localization']):
                        for num_list_pos, idx_position in enumerate(self.config.reft_config[dataset_name_al]['localization'][num]['positions']):
                            key_params = 'layer_' + str(dict_layer['layer']) + "_pos_" + str(idx_position)
                            val_params = self.interventions[dataset_name_al][str(dict_layer['layer'])][num_list_pos]
                            my_small_dict = {}
                            my_small_dict[key_params] = val_params
                            if "intervention" not in params:
                                params["intervention"] = []
                            if "position" not in params:
                                params['position'] = []

                            params["intervention"].extend(list(my_small_dict[key_params].parameters()))
                            params["position"].extend([key_params, key_params, key_params])
                            uniq += 1
        return params

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name).to(self.config.device)
        return classification_head
    
    def move_classification_head_to_cpu(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        classification_head.to('cpu')

    def move_classification_head_to_gpu(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        classification_head.to('cuda:0')

    def get_image_encoder(self):
        alph = self.lambdas()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        if self.config.layerwise:
            params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        if self.config.taskwise:
            params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda() for p in params)
        load_weights(self.model, self.names, params)
        
        return self.model, self.interventions

    def forward(self, inp, dataset_name):
        alph = self.lambdas()

        if self.config.taskwise:
            params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        elif self.config.layerwise:
            params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
   
        feature = self.model(inp, self.interventions)
        if self.config.current_model == "teacher":
            layer_name = 'classifier_{}'.format(dataset_name)
            classification_head = getattr(self, layer_name)
            out = classification_head(feature)
            return out, feature

        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)

        return out, feature  
