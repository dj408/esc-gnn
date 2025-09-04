"""
Centralized mappings for activation functions and related utilities.
This module contains all the mappings used throughout the codebase for
converting between different representations of activation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ATOM_WT_TO_IDX_MAP = {
    1: 0, # H
    6: 1, # C
    7: 2, # N
    8: 3, # O
    9: 4, # F
}

# String key -> torch.nn.functional function mapping
SCALAR_NONLIN_FN_MAP = {
    'relu': F.relu,
    'silu': F.silu,
    'tanh': F.tanh,
    'gelu': F.gelu,
}

# String key -> torch.nn.Module class mapping (for MLP layers)
MLP_NONLIN_MODULE_MAP = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'swish': nn.SiLU,  # Alias for backward compatibility
    'tanh': nn.Tanh,
    'gelu': nn.GELU,
}

# Function callable -> torch.nn.Module class mapping (for recombination layers)
FUNCTION_TO_MODULE_MAP = {
    F.relu: nn.ReLU,
    F.silu: nn.SiLU,
    F.tanh: nn.Tanh,
    F.gelu: nn.GELU,
}

# String key -> torch.nn.functional loss function mapping
LOSS_FN_MAP = {
    'mse': F.mse_loss,
    'l1': F.l1_loss,
    'huber': F.smooth_l1_loss,  # PyTorch's smooth_l1 is the Huber loss
}

# Vector nonlinearity mapping (for ESCGNN vector track)
VECTOR_NONLIN_FN_MAP = {
    'relu': F.relu,
    'silu': F.silu,
    'swish': F.silu,  # Alias for backward compatibility
    'tanh': F.tanh,
    'softplus': F.softplus,
}

# Gate function mapping (for ESCGNN vector gating)
GATE_FN_MAP = {
    'silu': F.silu,
    'swish': F.silu,  # Alias for backward compatibility
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    'tanh': F.tanh,
    'softplus': F.softplus,
}

# Gate initialization values mapping
GATE_INIT_MAP = {
    'silu': 0.75,      # smoother than ReLU, encourages non-zero scale
    'swish': 0.75,     # alias for backward compatibility
    'relu': 0.10,      # small positive -> ReLU(w) ~= 0.10
    'sigmoid': 0.0,    # sigmoid(0)=0.5 – neutral scaling
    'tanh': 0.10,      # tanh(0.10)=~0.10
    'softplus': 0.10,  # softplus(0.10)=~0.744
} 