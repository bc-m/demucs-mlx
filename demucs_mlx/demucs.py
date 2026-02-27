# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – DConv residual branch and supporting modules.
# Only the modules needed for HTDemucs inference.

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from .transformer import LayerScale
from .utils import apply_conv1d, apply_group_norm, glu


class DConv(nn.Module):
    """Residual dilated convolution branches.

    Alternates dilated Conv1d layers with optional GLU gating.
    For HTDemucs, no attention or LSTM is used inside DConv.

    All tensors use ``[B, C, T]`` (channels-first) layout.
    """

    def __init__(self, channels: int, compress: float = 4, depth: int = 2,
                 init: float = 1e-4, norm: bool = True,
                 gelu: bool = True, kernel: int = 3, dilate: bool = True):
        super().__init__()
        assert kernel % 2 == 1
        self.channels = channels
        self.compress = compress
        self.depth_val = abs(depth)
        dilate = depth > 0

        hidden = int(channels / compress)

        self.layers = []
        for d in range(self.depth_val):
            dilation = 2 ** d if dilate else 1
            padding = dilation * (kernel // 2)
            # Each layer: conv1 → norm → act → conv2 → norm → GLU → LayerScale
            layer_mods = {}
            layer_mods['conv1'] = nn.Conv1d(
                channels, hidden, kernel, padding=padding, dilation=dilation)
            if norm:
                layer_mods['norm1'] = nn.GroupNorm(1, hidden, pytorch_compatible=True)
            else:
                layer_mods['norm1'] = None
            layer_mods['conv2'] = nn.Conv1d(hidden, 2 * channels, 1)
            if norm:
                layer_mods['norm2'] = nn.GroupNorm(1, 2 * channels, pytorch_compatible=True)
            else:
                layer_mods['norm2'] = None
            layer_mods['scale'] = LayerScale(channels, init)
            self.layers.append(layer_mods)

    def __call__(self, x: mx.array) -> mx.array:
        """x: [B, C, T] → [B, C, T]."""
        for mods in self.layers:
            # Conv1: [B, C, T] → [B, hidden, T]
            y = apply_conv1d(mods['conv1'], x)
            if mods['norm1'] is not None:
                y = apply_group_norm(mods['norm1'], y)
            y = nn.gelu(y)

            # Conv2: [B, hidden, T] → [B, 2*C, T]
            y = apply_conv1d(mods['conv2'], y)
            if mods['norm2'] is not None:
                y = apply_group_norm(mods['norm2'], y)
            # GLU on dim=1 (channel dim): [B, 2*C, T] → [B, C, T]
            y = glu(y, axis=1)

            # LayerScale
            y = mods['scale'](y)

            x = x + y
        return x
