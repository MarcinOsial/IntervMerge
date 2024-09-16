import os
import logging
import time
import wandb
from datetime import datetime
import torch
import argparse
import ast
import torch
from utils.utils import set_deterministic
from .merging_coefficients import *


class CustomNamespace(argparse.Namespace):
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
    
class Config:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls._parse_args()
        return cls._instance

    @staticmethod
    def _parse_args():
        parser = argparse.ArgumentParser(description='Process command-line arguments.')
        #Set once
        parser.add_argument('--base_path', default='/path/to/project', help='Base path for the project')
        parser.add_argument('--pretrained_checkpoint', default='zeroshot.pt', help='Filename of the pretrained checkpoint to use')
        parser.add_argument('--deterministic', action='store_true', help='Enable deterministic behavior')
        parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
        parser.add_argument('--cache_dir', default='cache_dir', help='Cache directory path')
        parser.add_argument('--data_location', default='data', help='Data directory path')
        parser.add_argument('--exam_datasets', nargs='+', default=['SUN397', 'Cars', 'RESISC45', 'SVHN', 'GTSRB', 'MNIST', 'DTD', 'EuroSAT'], help='List of datasets to be used for model merging')
        parser.add_argument('--save_checkpoints', default='checkpoints', help='Save checkpoints directory path')
        parser.add_argument('--logs', default='logs', help='Logs directory path')
        parser.add_argument('--device', default='cuda:0', help='Device for training')
        parser.add_argument('--saved_models', type=str, default='saved_models', help='Directory name for saved models')

        parser.add_argument('--model', default='ViT-B-32', help='Model name')
        parser.add_argument('--seed', type=int, default=1, help='Seed value for random number generators to ensure reproducibility')
        parser.add_argument('--iterations', type=int, default=500, help='Number of iterations for training')
        parser.add_argument('--eval_iterations', type=int, default=100, help='Number of iterations between evaluations')
        parser.add_argument('--save_model_flag', action='store_true', default=False, help='Enable saving of the model after training')
        parser.add_argument('--pruned', action='store_true', default=False, help='Enable pruning (TiesM/Adamerging++)')

        #Wandb
        parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
        parser.add_argument('--wandb_logs', default='wandb', help='Weights & Biases logs directory path')
        parser.add_argument('--wandb_entity', default='osialm', help='Weights & Biases entity')
        parser.add_argument('--wandb_project', default='intervmerge', help='Weights & Biases project name')
        parser.add_argument('--name', default='experiment_name', help='Name of the experiment for Weights & Biases logging')
        parser.add_argument('--group', default='experiment_group', help='Group name for organizing experiments in Weights & Biases')

        # Lambdas/Adamering
        parser.add_argument('--train_lambdas', action='store_true', default=False, help='Enable lambdas train')   
        parser.add_argument('--taskwise', action='store_true', help='Enable task-wise AdaMerging')
        parser.add_argument('--layerwise', action='store_true', help='Enable layer-wise AdaMerging')
        parser.add_argument('--prior', type=float, default=0.3, help='Prior value for lambda parameters')
        parser.add_argument('--use_lambda_custom', action='store_true', default=False, help='Enable the use of custom lambda values')
        
        #Interventions
        parser.add_argument('--use_intervention', action='store_true', default=False, help='Enable intervention training')
        parser.add_argument('--train_intervention', action='store_true', default=False, help='Enable intervention training')
        parser.add_argument('--reft_position', type=str, default='0', help="Position in token sequence for intervention. '0' is the class token.")
        parser.add_argument('--low_rank_dimension', type=int, default=4, help='Low rank dimension for the intervention')
        parser.add_argument('--num_layer', type=str, default='12', help="Number of layers to which the intervention will be applied. Eg. for one layer '1', for all '12'")
        parser.add_argument('--start_layer', type=str, default='11', help='Starting layer for intervention, counting from the last layer (11), to the first (0)')
        parser.add_argument('--config_type', type=str, default='one_token', choices=['one_token', 'patch_tokens', 'every_second', 'every_third', 'every_fourth'],
                    help='Type of localization configuration to use for token sequance')
        parser.add_argument('--intervention_type', type=str, default='default', choices=['default', 'NodiReFT', 'LobiReFT', 'ConsReFT', 'DiReFT', 'LoReFT', 'miniInterv_best_efficient', 'miniInterv_efficient', 'miniInterv', 'miniIntervMulti'], help='Type of intervention')
        parser.add_argument('--model_hidden_size', type=int, default=768, help='Hidden size of the model (e.g., 768 for ViT-B, 1024 for ViT-L, 64 for mini-intervention)')
        
        args = parser.parse_args(namespace=CustomNamespace())
        args.reft_position = int(args.reft_position)
        args.num_layer = int(args.num_layer)
        args.start_layer = int(args.start_layer)
        args.model_hidden_size = int(args.model_hidden_size)

        args.base_path = '/home/marcin.osial/IntervMerge/src'
        args.data_location = '/home/marcin.osial/AdaMerging/data'
        args.wandb_logs = os.path.join(args.base_path, args.wandb_logs)
        args.pretrained_checkpoint = os.path.join(args.base_path, args.save_checkpoints, args.model, args.pretrained_checkpoint)
        args.save_checkpoints = os.path.join(args.base_path, args.save_checkpoints, args.model)
        args.logs = os.path.join(args.base_path, args.logs)
        args.cache_dir = os.path.join(args.base_path, args.cache_dir)
        args.save_path = os.path.join(args.base_path, args.saved_models)

        selected_config = choose_localization_config(args)
        args.reft_config = {
            'SUN397': {
                'dropout': 0.0,
                'act_fn': 'linear',
                'localization': selected_config(args.start_layer, args.num_layer, args.reft_position)
            },
            'Cars': {
                'dropout': 0.0,
                'act_fn': 'linear',
                'localization': selected_config(args.start_layer, args.num_layer, args.reft_position)
            },
            'RESISC45': {
                'dropout': 0.0,
                'act_fn': 'linear',
                'localization':  selected_config(args.start_layer, args.num_layer, args.reft_position)
            },
            'EuroSAT': {
                'dropout': 0.0,
                'act_fn': 'linear',
                'localization': selected_config(args.start_layer, args.num_layer, args.reft_position)
            },
            'SVHN': {
                'dropout': 0.0,
                'act_fn': 'linear',
                'localization': selected_config(args.start_layer, args.num_layer, args.reft_position)
            },
            'GTSRB': {
                'dropout': 0.0,
                'act_fn': 'linear',
                'localization': selected_config(args.start_layer, args.num_layer, args.reft_position)
            },
            'MNIST': {
                'dropout': 0.0,
                'act_fn': 'linear',
                'localization': selected_config(args.start_layer, args.num_layer, args.reft_position)
            },
            'DTD': {
                'dropout': 0.0,
                'act_fn': 'linear',
                'localization': selected_config(args.start_layer, args.num_layer, args.reft_position)
            },
        }

        if args.deterministic:
            set_deterministic(args.seed)
        else:
            set_deterministic(None)

        custom_rlambdas = None
        args.custom_rlambdas = None

        if args.use_lambda_custom:
            if args.layerwise:
                if args.model == 'ViT-L-14':
                    custom_rlambdas_8 = custom_rlambdas_8_vitl14
                elif args.model == 'ViT-B-32' and not args.pruned:
                    custom_rlambdas_8 = custom_rlambdas_8_vitb32
                elif args.model == 'ViT-B-32' and args.pruned:
                    custom_rlambdas_8 = custom_rlambdas_8_vitb32_plus
                elif args.model == 'ViT-B-16':
                    custom_rlambdas_8 = custom_rlambdas_8_vitb16
                    
                custom_rlambdas = torch.stack(custom_rlambdas_8)
                args.custom_rlambdas = torch.tensor(custom_rlambdas)
            if args.taskwise:
                if args.model == 'ViT-B-32' and not args.pruned:
                    custom_rlambdas = custom_taskwise_8_vitb32

                custom_rlambdas = torch.stack(custom_rlambdas)
                args.custom_rlambdas = torch.tensor(custom_rlambdas, dtype=torch.float32)
        else:
            args.custom_rlambdas = None

        create_directories(args)
        args.logger = setup_logger(args)
        return args


def create_localization_config(start_layer, num_layers, reft_position):
    return [
        {'layer': layer, 'positions': [reft_position], 'component': 'attention'}
        for layer in range(start_layer, start_layer - num_layers, -1)
    ]

def create_localization_config_patch_token(start_layer, num_layers, reft_position=0):
    max_position=49
    config = []
    for reft_position in range(1, max_position + 1):
        for layer in range(start_layer, start_layer - num_layers, -1):
            config.append({
                'layer': layer,
                'positions': [reft_position],
                'component': 'attention'
            })
    return config

def create_localization_config_every_second_layer(start_layer, num_layers, reft_position):
    return [
        {'layer': layer, 'positions': [reft_position], 'component': 'attention'}
        for layer in range(start_layer, start_layer - num_layers, -1) if layer % 2 == 0
    ]

def create_localization_config_every_third_layer(start_layer, num_layers, reft_position):
    return [
        {'layer': layer, 'positions': [reft_position], 'component': 'attention'}
        for layer in range(start_layer, start_layer - num_layers, -1) if layer % 3 == 0
    ]

def create_localization_config_every_fourth_layer(start_layer, num_layers, reft_position):
    return [
        {'layer': layer, 'positions': [reft_position], 'component': 'attention'}
        for layer in range(start_layer, start_layer - num_layers, -1) if layer % 4 == 0
    ]

def choose_localization_config(args):
    if args.config_type == 'one_token':
        return create_localization_config
    elif args.config_type == 'patch_tokens':
        return create_localization_config_patch_token
    elif args.config_type == 'every_second_one_token':
        return create_localization_config_every_second_layer
    elif args.config_type == 'every_third_one_token':
        return create_localization_config_every_third_layer
    elif args.config_type == 'every_fourth_one_token':
        return create_localization_config_every_fourth_layer
    else:
        raise ValueError(f"Unknown configuration type: {args.config_type}")

def create_directories(args):
    paths_to_create = [
        args.cache_dir,
        args.data_location,
        args.logs,
        args.save_checkpoints,
        args.wandb_logs,
        args.saved_models
    ]
    for path in paths_to_create:
        os.makedirs(path, exist_ok=True)

def setup_logger(args):
    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    logger_name = f"ConfigLogger_{current_time}.log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Setting up file handler for logging
    file_handler = logging.FileHandler(os.path.join(args.logs, logger_name))
    file_handler.setLevel(logging.DEBUG)

    # Setting up console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Setting log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Adding handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

