from transformers.activations import ACT2FN
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import math
import wandb
from datetime import datetime
from torch.nn.modules.module import _IncompatibleKeys
from configs.config import Config
import random


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""
    def __init__(self, n, m):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)
    

class MultiMiniInterv(nn.Module):
    """
    Default with: NoReFT(h) = h + W2^T(W1h + b − W2h)
    """
    def __init__(self, embed_dim, low_rank_dimension, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.low_rank_dimension = low_rank_dimension
        dtype = kwargs.get("dtype", torch.float32)
        
        self.proj_layers = nn.ModuleList([
            nn.Linear(embed_dim, low_rank_dimension, bias=True).to(dtype)
            for _ in range(4)
        ])
        self.learned_sources = nn.ModuleList([
            nn.Linear(embed_dim, low_rank_dimension).to(dtype)
            for _ in range(4)
        ])
        
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        self.layer_index = kwargs['layer_index']

        #Hardcoded positions
        self.start1, self.end1 = 0, 16
        self.start2, self.end2 = 752, 768
        self.start3, self.end3 = 184, 200
        self.start4, self.end4 = 568, 584


    def forward(self, base, source=None, subspaces=None):
        batch_size = base.shape[1]
        padded_base = torch.zeros(batch_size, 768, dtype=base.dtype, device=base.device)
        
        regions = [
            (self.start1, self.end1),
            (self.start2, self.end2),
            (self.start3, self.end3),
            (self.start4, self.end4)
        ]
        
        for i, (start, end) in enumerate(regions):
            sliced_base = base[:, :, start:end]
            reshaped_base = sliced_base.view(-1, self.embed_dim)
            proj_base = self.proj_layers[i](reshaped_base)
            sliced_output = torch.matmul(
                (self.act_fn(self.learned_sources[i](reshaped_base)) - proj_base),
                self.proj_layers[i].weight
            )
            padded_base[:, start:end] = sliced_output
        
        padded_base = padded_base.view(1, batch_size, 768)
        output = base + padded_base
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        state_dict = OrderedDict()
        for i, (proj, source) in enumerate(zip(self.proj_layers, self.learned_sources)):
            for k, v in proj.state_dict().items():
                state_dict[f'proj_layer_{i}.{k}'] = v
            for k, v in source.state_dict().items():
                state_dict[f'learned_source_{i}.{k}'] = v
        return state_dict

    def load_state_dict(self, state_dict, strict=False):
        for i in range(4):
            proj_state = {k.split(f'proj_layer_{i}.')[1]: v for k, v in state_dict.items() if f'proj_layer_{i}.' in k}
            source_state = {k.split(f'learned_source_{i}.')[1]: v for k, v in state_dict.items() if f'learned_source_{i}.' in k}
            self.proj_layers[i].load_state_dict(proj_state, strict=False)
            self.learned_sources[i].load_state_dict(source_state, strict=False)
        return

class MiniInterv(nn.Module):
    """
    Default with: NoReFT(h) = h + W2^T(W1h + b − W2h)
    """
    def __init__(self, embed_dim, low_rank_dimension, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.low_rank_dimension = low_rank_dimension
        dtype = kwargs.get("dtype", torch.float32)
        
        self.proj_layers = nn.ModuleList([
            nn.Linear(embed_dim, low_rank_dimension, bias=True).to(dtype)
            for _ in range(1)
        ])
        self.learned_sources = nn.ModuleList([
            nn.Linear(embed_dim, low_rank_dimension).to(dtype)
            for _ in range(1)
        ])
        
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        self.layer_index = kwargs['layer_index']

        #Hardcoded positions
        self.start1, self.end1 = 0, embed_dim


    def forward(self, base, source=None, subspaces=None):
        batch_size = base.shape[1]
        padded_base = torch.zeros(batch_size, 768, dtype=base.dtype, device=base.device)
        
        regions = [
            (self.start1, self.end1),
        ]
        
        for i, (start, end) in enumerate(regions):
            sliced_base = base[:, :, start:end]
            reshaped_base = sliced_base.view(-1, self.embed_dim)
            proj_base = self.proj_layers[i](reshaped_base)
            sliced_output = torch.matmul(
                (self.act_fn(self.learned_sources[i](reshaped_base)) - proj_base),
                self.proj_layers[i].weight
            )
            padded_base[:, start:end] = sliced_output
        
        padded_base = padded_base.view(1, batch_size, 768)
        output = base + padded_base
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        state_dict = OrderedDict()
        for i, (proj, source) in enumerate(zip(self.proj_layers, self.learned_sources)):
            for k, v in proj.state_dict().items():
                state_dict[f'proj_layer_{i}.{k}'] = v
            for k, v in source.state_dict().items():
                state_dict[f'learned_source_{i}.{k}'] = v
        return state_dict

    def load_state_dict(self, state_dict, strict=False):
        for i in range(1):
            proj_state = {k.split(f'proj_layer_{i}.')[1]: v for k, v in state_dict.items() if f'proj_layer_{i}.' in k}
            source_state = {k.split(f'learned_source_{i}.')[1]: v for k, v in state_dict.items() if f'learned_source_{i}.' in k}
            self.proj_layers[i].load_state_dict(proj_state, strict=False)
            self.learned_sources[i].load_state_dict(source_state, strict=False)
        return
    

class MiniInterventionParamEffic(torch.nn.Module):
    """
    Apply intervention to only part of representation vector. Default with unique position between layers.
    Parameter-efficient with h + R^T(b)
    """
    def __init__(self, embed_dim, low_rank_dimension, **kwargs):
        super().__init__()
        self.embed_dim = int(embed_dim)
        rotate_layer = LowRankRotateLayer(embed_dim, low_rank_dimension)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(low_rank_dimension), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.layer_index = int(kwargs["layer_index"])
        
        self.task_num = {
            'SUN397': 0,
            'Cars': 1,
            'RESISC45': 2,
            'EuroSAT': 3,
            'SVHN': 4,
            'GTSRB': 5,
            'MNIST': 6,
            'DTD': 7
        }
        self.dataset = kwargs["dataset_name"]
        self.current_dataset_number = self.task_num[self.dataset]

    def forward(
            self, base, source=None, subspaces=None
        ):
        
        learn_v = torch.matmul(
                self.learned_source, self.rotate_layer.weight.T
            )

        start = self.layer_index * self.embed_dim
        end = self.layer_index * self.embed_dim + self.embed_dim

        padded_vector = torch.zeros(768, dtype=learn_v.dtype, device=learn_v.device)
        padded_vector[start:end] = learn_v

        output = base + padded_vector
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        state_dict["learned_source"] = self.learned_source.data
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.data = state_dict["learned_source"]
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base.data[:, :overload_w_width] = overload_w
        return
    
class MiniInterventionParam(torch.nn.Module):
    """
    Parameter-efficient with h + R^T(b) without shift
    """
    def __init__(self, embed_dim, low_rank_dimension, **kwargs):
        super().__init__()
        self.embed_dim = int(embed_dim)
        rotate_layer = LowRankRotateLayer(embed_dim, low_rank_dimension)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(low_rank_dimension), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)


    def forward(
            self, base, source=None, subspaces=None
        ):
        
        learn_v = torch.matmul(
                self.learned_source, self.rotate_layer.weight.T
            )

        start = 0
        end = self.embed_dim # eg 64

        padded_vector = torch.zeros(768, dtype=learn_v.dtype, device=learn_v.device)
        padded_vector[start:end] = learn_v

        output = base + padded_vector
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        state_dict["learned_source"] = self.learned_source.data
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.data = state_dict["learned_source"]
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base.data[:, :overload_w_width] = overload_w
        return

class LoreftIntervention(torch.nn.Module):
    """
    LoReFT(h) = h + R^T(Wh + b - Rh)
    """
    def __init__(self, embed_dim, low_rank_dimension, **kwargs):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, low_rank_dimension)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            embed_dim, low_rank_dimension).to(torch.float32)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]

    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))
    
    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        return

class DireftIntervention(torch.nn.Module):
    """
    DiReFT(h) = h + R^T(Wh + b)
    """
    def __init__(self, embed_dim, low_rank_dimension, **kwargs):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, low_rank_dimension)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            embed_dim, low_rank_dimension).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.float32)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        cast_base = base.to(self.learned_source.weight.dtype)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(cast_base))).to(self.rotate_layer.weight.dtype), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))
    
    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        state_dict["learned_source"] = self.learned_source.data
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.data = state_dict["learned_source"]
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base.data[:, :overload_w_width] = overload_w
        return

class ConsreftIntervention(torch.nn.Module):
    """
    ConsReFT(h) = h + R^T(b − Rh)
    """
    def __init__(self, embed_dim, low_rank_dimension, **kwargs):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, low_rank_dimension)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(low_rank_dimension), requires_grad=True)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(base.dtype)

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        state_dict["learned_source"] = self.learned_source.data
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.data = state_dict["learned_source"]
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base.data[:, :overload_w_width] = overload_w
        return


class LobireftIntervention(torch.nn.Module):
    """
    LobiReFT(h) = h + R^T(b) 
    """
    def __init__(self, embed_dim, low_rank_dimension, **kwargs):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, low_rank_dimension)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(low_rank_dimension), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
    
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.learned_source, self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        state_dict["learned_source"] = self.learned_source.data
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.data = state_dict["learned_source"]
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base.data[:, :overload_w_width] = overload_w
        return

class NoreftIntervention(torch.nn.Module):
    """
    NoReFT(h) = h + W2^T(W1h + b − W2h)
    """
    def __init__(self, embed_dim, low_rank_dimension, **kwargs):
        super().__init__()
        self.proj_layer = torch.nn.Linear(
            embed_dim, low_rank_dimension, bias=True).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.float32)
        self.learned_source = torch.nn.Linear(
            embed_dim, low_rank_dimension).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.float32)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]

    def forward(
        self, base, source=None, subspaces=None
    ):
        proj_base = self.proj_layer(base)
        learned_source_output = self.learned_source(base)
        act_fn_output = self.act_fn(learned_source_output)
        diff = act_fn_output - proj_base
        matmul_output = torch.matmul(diff, self.proj_layer.weight)
        output = base + matmul_output
        final_output = self.dropout(output.to(base.dtype))

        return final_output

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.proj_layer.state_dict().items():
            state_dict['proj_layer.' + k] = v
        for k, v in self.learned_source.state_dict().items():
            state_dict['learned_source.' + k] = v
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Overwrite for data-efficiency.
        """
        missing_keys = []
        unexpected_keys = []

        proj_layer_state = {}
        learned_source_state = {}

        for k, v in state_dict.items():
            if k.startswith('proj_layer.'):
                proj_layer_state[k.split('proj_layer.')[1]] = v
            elif k.startswith('learned_source.'):
                learned_source_state[k.split('learned_source.')[1]] = v
            else:
                unexpected_keys.append(k)

        if not proj_layer_state:
            missing_keys.append('proj_layer')
        else:
            self.proj_layer.load_state_dict(proj_layer_state, strict=False)

        if not learned_source_state:
            missing_keys.append('learned_source')
        else:
            self.learned_source.load_state_dict(learned_source_state, strict=False)

        if strict:
            error_msgs = []
            if missing_keys:
                error_msgs.append(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                error_msgs.append(f"Unexpected keys: {unexpected_keys}")
            if error_msgs:
                raise RuntimeError("Error(s) in loading state_dict:\n\t{}".format("\n\t".join(error_msgs)))

        return _IncompatibleKeys(missing_keys, unexpected_keys)


class NodireftIntervention(torch.nn.Module):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """
    def __init__(self, embed_dim, low_rank_dimension, **kwargs):
        super().__init__()
        self.proj_layer = torch.nn.Linear(
            embed_dim, low_rank_dimension, bias=True).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.float32)
        self.learned_source = torch.nn.Linear(
            embed_dim, low_rank_dimension).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.float32)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.act_fn(self.learned_source(base)), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["proj_layer"] = self.proj_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)
        overload_w = state_dict["proj_layer"]
        overload_w_width = overload_w.shape[-1]
        self.proj_layer.weight[:,:overload_w_width] = overload_w
        return


