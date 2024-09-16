import torch
import torch.nn as nn

import wandb
import inspect
import uuid
import torch.nn.functional as F
import os
from typing import Optional
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import patheffects as PathEffects
from configs.config import Config
import psutil

from PIL import Image as pil
from PIL import Image
from pkg_resources import parse_version
import torch
import torch.nn.functional as F

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, config):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.config = config
        self.inter_rep = {}

    def generate_custom_forward_resblocks(self, block):
        """
        Factory function to generate a custom forward function for a given block.
        """        
        layer_idx = self.layer_index
        component_layer = self.component_layer
        config = self.config
        reft_config = config.reft_config
        use_intervention = config.use_intervention

        def apply_intervention(inter_rep, interventions, component_key):        
            if interventions is not None and layer_idx in range(0, 12):
                if isinstance(self.config.reft_config[self.config.current_dataset], str):
                    dataset_config = reft_config.get(list(reft_config.keys())[0], {})
                    localization = dataset_config.get('localization', {})
                else:
                    localization = reft_config.get(config.current_dataset, {}).get('localization', {})
                intervention_positions_list = []
                for dict in localization:
                        if layer_idx == dict['layer']:
                            intervention_positions_list.extend(dict['positions'])
                processed_patches = []
                for idx in range(inter_rep.shape[0]):
                    if idx in intervention_positions_list:
                        uniq_num = intervention_positions_list.index(idx)
                        current_position = "layer_" + str(layer_idx) + "_position_" + str(idx) + "_" + str(config.current_dataset)                        
                        if use_intervention:
                            if config.current_model == 'student':
                                processed_patch = interventions[config.current_dataset][str(layer_idx)][uniq_num](inter_rep[idx:idx+1, :, :])
                            elif config.current_model == 'teacher':  
                                processed_patch = inter_rep[idx:idx+1, :, :]
                            elif config.current_model == 'mtl':
                                processed_patch = inter_rep[idx:idx+1, :, :]
                        else:
                            if config.current_model == 'student':
                                processed_patch = inter_rep[idx:idx+1, :, :]
                            elif config.current_model == 'teacher':
                                processed_patch = inter_rep[idx:idx+1, :, :]
                    else:
                        processed_patch = inter_rep[idx:idx+1, :, :]
                    processed_patches.append(processed_patch)
                processed_inter_rep = torch.cat(processed_patches, dim=0)
            else:
                processed_inter_rep = inter_rep
            return processed_inter_rep

        def custom_forward_resblocks(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, interventions=None):
            inter_rep_0 = block.ln_1(x)
            if "ln_1" in component_layer:
                inter_rep_0 = apply_intervention(inter_rep_0, interventions, "ln_1")

            inter_rep_1 = block.attention(inter_rep_0, attn_mask)
            if "attention" in component_layer:
                inter_rep_1 = apply_intervention(inter_rep_1, interventions, "attention")
 
            inter_rep_2 = x + inter_rep_1
            if "residual_1" in component_layer:
                inter_rep_2 = apply_intervention(inter_rep_2, interventions, "residual_1")

            inter_rep_3 = block.ln_2(inter_rep_2)
            if "ln_2" in component_layer:
                inter_rep_3 = apply_intervention(inter_rep_3, interventions, "ln_2")

            inter_rep_4 = inter_rep_2 + block.mlp(inter_rep_3)
            if "mlp" in component_layer:
                inter_rep_4 = apply_intervention(inter_rep_4, interventions, "mlp")

            return inter_rep_4
        
        def forward_wrapper(self, x, interventions=None):
            return custom_forward_resblocks(self, x, attn_mask=None, interventions=interventions)
        
        return forward_wrapper


    def forward(self, images, interventions=None):
        self.original_forward = self.model.model.visual.forward
        self.model.model.visual.forward = self.custom_forward.__get__(self.model.model.visual, type(self.model.model.visual))
        features = self.custom_forward(images, interventions)

        return features
    
    def get_components_for_layer(self):
        """
        Fetch all 'component' values for the current layer from the reft_config.
        
        Returns:
        - list: A list of 'component' values for the current layer.
        """
        reft_config = self.config.reft_config
        components = []

        if isinstance(self.config.reft_config[self.config.current_dataset], str):
            dataset_config = reft_config.get(list(reft_config.keys())[0], {})
        else:
        # Assuming the configuration is for a specific dataset, e.g., 'Cars'
            dataset_config = reft_config.get(self.config.current_dataset, {})

        # Extract intervention details for the current layer
        localization = dataset_config.get('localization', [])
        for intervention in localization:
            if intervention['layer'] == self.layer_index:
                components.append(intervention['component'])

        return components

    def custom_forward(self, x, interventions):
        x = self.model.model.visual.conv1(x) # shape = [*, width, grid, grid]
        # Reshape and permute operations
        x = x.reshape(x.shape[0], x.shape[1], -1) # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.model.visual.positional_embedding.to(x.dtype)
        x = self.model.model.visual.ln_pre(x)

        # Transformer layers
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Dynamically calculate and apply weights per layer
        for layer_index, block in enumerate(self.model.model.visual.transformer.resblocks):
            self.layer_path = f'image_encoder.model.model.visual.transformer.resblocks.{layer_index}'
            self.layer_index = layer_index
            self.component_layer = self.get_components_for_layer()
            
            custom_forward_rb = self.generate_custom_forward_resblocks(block)
            block.forward = custom_forward_rb.__get__(block, type(block))
            x = block(x, interventions=interventions)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.model.visual.ln_post(x[:, 0, :])

        if self.model.model.visual.proj is not None:
            x = x @ self.model.model.visual.proj

        # if self.config.phase == "vis" and self.config.vis_only_end:
        #     if self.config.current_model == "orig":
        #         self.save_rep_before_intervention(x, '10_' + str(self.config.current_dataset), self.config.unique_id)
        #     if self.config.current_model == "student":
        #         self.save_rep_after_intervention(x, '10_' + str(self.config.current_dataset), self.config.unique_id)
        #     if self.config.current_model == "teacher":
        #         self.save_rep_finetuned(x, '10_' + str(self.config.current_dataset), self.config.unique_id)
        #     if self.config.current_model == "mtl":
        #         self.save_rep_mtl(x, '10_' + str(self.config.current_dataset), self.config.unique_id)

        return x

