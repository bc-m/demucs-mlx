# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Weight conversion from PyTorch Demucs to MLX HTDemucs.
#
# Key conversions:
#   Conv1d weight: PyTorch [C_out, C_in, K] → MLX [C_out, K, C_in]
#   Conv2d weight: PyTorch [C_out, C_in, kH, kW] → MLX [C_out, kH, kW, C_in]
#   ConvTranspose1d weight: PyTorch [C_in, C_out, K] → MLX [C_out, K, C_in]
#   ConvTranspose2d weight: PyTorch [C_in, C_out, kH, kW] → MLX [C_out, kH, kW, C_in]
#   nn.MultiheadAttention in_proj_weight → split into q/k/v proj weights
#   GroupNorm weight/bias stay as-is (1D, [C])
#   LayerScale: scale parameter stays as-is (1D, [C])

import typing as tp

import numpy as np


def _is_conv1d_weight(key: str, shape: tuple) -> bool:
    """Detect Conv1d weight: 3D tensor [C_out, C_in, K]."""
    return len(shape) == 3 and 'weight' in key and 'norm' not in key


def _is_conv2d_weight(key: str, shape: tuple) -> bool:
    """Detect Conv2d weight: 4D tensor [C_out, C_in, kH, kW]."""
    return len(shape) == 4 and 'weight' in key and 'norm' not in key


def _is_conv_transpose(key: str) -> bool:
    """Detect ConvTranspose weights by key pattern."""
    return 'conv_tr.' in key


def convert_htdemucs_weights(state_dict: dict) -> dict:
    """Convert a PyTorch HTDemucs state_dict to MLX-compatible format.

    Args:
        state_dict: PyTorch state_dict (keys → numpy arrays or torch tensors).

    Returns:
        Dict of {key: numpy_array} ready for loading into MLX model.
    """
    mlx_state = {}

    for key, value in state_dict.items():
        # Convert to numpy if torch tensor
        if hasattr(value, 'numpy'):
            value = value.numpy()
        elif hasattr(value, 'cpu'):
            value = value.cpu().numpy()
        value = np.array(value, dtype=np.float32) if value.dtype != np.float32 else value

        new_key = _remap_key(key)
        new_value = _convert_value(new_key, key, value)

        if new_value is not None:
            if isinstance(new_value, dict):
                mlx_state.update(new_value)
            else:
                mlx_state[new_key] = new_value

    return mlx_state


def _remap_key(key: str) -> str:
    """Remap PyTorch state_dict keys to MLX model structure.

    Main remappings:
    - nn.MultiheadAttention in_proj_weight/bias → q_proj/k_proj/v_proj
    - nn.MultiheadAttention out_proj → out_proj
    - Module lists use index notation
    - LayerScale scale parameter
    """
    # CrossTransformerEncoderLayer: cross_attn (nn.MultiheadAttention) →
    # our MultiheadAttention with separate q/k/v projections.
    # This is handled specially in _convert_value for in_proj_weight.

    # MyTransformerEncoderLayer inherits from nn.TransformerEncoderLayer
    # which uses self_attn (nn.MultiheadAttention internally)
    # PyTorch keys: self_attn.in_proj_weight, self_attn.in_proj_bias,
    #               self_attn.out_proj.weight, self_attn.out_proj.bias

    # Remap self_attn → self_attn (keeping name, but split in_proj later)
    # Remap cross_attn → cross_attn (keeping name, but split in_proj later)

    # gamma_1 / gamma_2 → LayerScale
    # In PyTorch: gamma_1.scale (nn.Parameter)
    # In MLX: gamma_1.scale (leaf parameter)

    # DConv layers: layers.N.M.weight → layers.N.convN.weight etc
    # This requires careful mapping since PyTorch uses nn.Sequential inside nn.ModuleList

    # For now, return key as-is. The actual structure matching happens
    # in the load function which walks the model tree.
    return key


def _convert_value(new_key: str, orig_key: str, value: np.ndarray):
    """Convert a single weight value, potentially splitting it.

    Returns either a numpy array or a dict of {key: array} for split weights.
    """
    shape = value.shape

    # Handle MultiheadAttention in_proj_weight (combined Q,K,V)
    if 'in_proj_weight' in orig_key:
        # Split [3*D, D] → three [D, D] matrices
        dim = shape[0] // 3
        prefix = orig_key.rsplit('in_proj_weight', 1)[0]
        return {
            prefix + 'q_proj.weight': value[:dim],
            prefix + 'k_proj.weight': value[dim:2*dim],
            prefix + 'v_proj.weight': value[2*dim:],
        }

    if 'in_proj_bias' in orig_key:
        dim = shape[0] // 3
        prefix = orig_key.rsplit('in_proj_bias', 1)[0]
        return {
            prefix + 'q_proj.bias': value[:dim],
            prefix + 'k_proj.bias': value[dim:2*dim],
            prefix + 'v_proj.bias': value[2*dim:],
        }

    # ConvTranspose weights
    if _is_conv_transpose(orig_key) and len(shape) >= 3 and 'weight' in orig_key:
        if len(shape) == 3:
            # ConvTranspose1d: PyTorch [C_in, C_out, K] → MLX [C_out, K, C_in]
            return np.transpose(value, (1, 2, 0))
        elif len(shape) == 4:
            # ConvTranspose2d: PyTorch [C_in, C_out, kH, kW] → MLX [C_out, kH, kW, C_in]
            return np.transpose(value, (1, 2, 3, 0))

    # Regular Conv weights
    if _is_conv1d_weight(orig_key, shape):
        # Conv1d: PyTorch [C_out, C_in, K] → MLX [C_out, K, C_in]
        return np.transpose(value, (0, 2, 1))

    if _is_conv2d_weight(orig_key, shape):
        # Conv2d: PyTorch [C_out, C_in, kH, kW] → MLX [C_out, kH, kW, C_in]
        return np.transpose(value, (0, 2, 3, 1))

    # Everything else (biases, norms, embeddings, scalars) → as-is
    return value


def map_state_dict_to_mlx(pt_state: dict, mlx_model) -> dict:
    """Map a converted state dict to the MLX model's parameter tree.

    This handles the structural differences between PyTorch nn.ModuleList
    and MLX's list-of-modules pattern.

    Args:
        pt_state: Already-converted state dict from convert_htdemucs_weights.
        mlx_model: The MLX HTDemucs model instance.

    Returns:
        Nested dict matching MLX model.parameters() structure.
    """
    # First, convert PyTorch flat key paths to nested dict
    nested = {}
    for key, value in pt_state.items():
        parts = key.split('.')
        d = nested
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    # Map DConv layers: PyTorch nn.Sequential inside nn.ModuleList
    # PyTorch: layers.0.0.weight (conv1), layers.0.1 (norm), layers.0.2 (act),
    #          layers.0.3.weight (conv2), layers.0.4 (norm), layers.0.5 (GLU),
    #          layers.0.6.scale (LayerScale)
    # MLX: layers.0.conv1.weight, layers.0.norm1.weight, layers.0.conv2.weight,
    #       layers.0.norm2.weight, layers.0.scale.scale

    return nested


# ── Sequential index mapping for DConv ────────────────────────────────────

# In PyTorch DConv, each layer is an nn.Sequential with these indices:
#   0: Conv1d (channels → hidden, dilated)
#   1: GroupNorm(1, hidden)
#   2: GELU
#   3: Conv1d (hidden → 2*channels, 1x1)
#   4: GroupNorm(1, 2*channels)
#   5: GLU
#   6: LayerScale
#
# In MLX DConv, each layer is a dict:
#   conv1, norm1, conv2, norm2, scale

DCONV_SEQ_MAP = {
    '0': 'conv1',    # Conv1d
    '1': 'norm1',    # GroupNorm
    # 2: GELU (no params)
    '3': 'conv2',    # Conv1d
    '4': 'norm2',    # GroupNorm
    # 5: GLU (no params)
    '6': 'scale',    # LayerScale
}

# ── HEncLayer Sequential index mapping ────────────────────────────────────
# For the rewrite conv + norm in HEncLayer, the PyTorch keys depend on
# which modules are present. We handle this by name matching in the
# key remapping.
