# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Utility functions for Demucs.

import math
import typing as tp

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ── Tensor manipulation ──────────────────────────────────────────────────────

def center_trim(tensor: mx.array, reference: tp.Union[mx.array, int]) -> mx.array:
    """Center-trim ``tensor`` along the last dimension to match ``reference``.

    If the size difference is odd, the extra sample is removed on the right.
    """
    if isinstance(reference, mx.array):
        ref_size = reference.shape[-1]
    else:
        ref_size = reference
    delta = tensor.shape[-1] - ref_size
    if delta < 0:
        raise ValueError(f"tensor must be larger than reference. Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def pad1d(x: mx.array, paddings: tp.Tuple[int, int],
          mode: str = 'constant', value: float = 0.) -> mx.array:
    """Pad along the last dimension, handling reflect mode on small inputs.

    If ``mode='reflect'`` and the input is too small for the requested padding,
    extra zero-padding is inserted first so that reflect can succeed.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings

    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left,
                        padding_right - extra_pad_right)
            # Add zero-pad so that reflect can work
            pad_widths = [(0, 0)] * (x.ndim - 1) + [(extra_pad_left, extra_pad_right)]
            x = mx.pad(x, pad_widths)
            padding_left, padding_right = paddings

    if mode == 'constant':
        pad_widths = [(0, 0)] * (x.ndim - 1) + [(padding_left, padding_right)]
        return mx.pad(x, pad_widths, constant_values=value)
    elif mode == 'reflect':
        return _reflect_pad(x, padding_left, padding_right)
    else:
        raise ValueError(f"Unsupported pad mode: {mode}")


def _reflect_pad(x: mx.array, left: int, right: int) -> mx.array:
    """Reflect-pad the last dimension of x."""
    if left == 0 and right == 0:
        return x
    length = x.shape[-1]
    parts = []
    if left > 0:
        # Reflect from index 1..left (exclusive of boundary)
        parts.append(x[..., left:0:-1])
    parts.append(x)
    if right > 0:
        parts.append(x[..., -2:-2 - right:-1])
    return mx.concatenate(parts, axis=-1)


def unfold(x: mx.array, kernel_size: int, stride: int) -> mx.array:
    """Extract sliding frames from the last dimension.

    Input: ``[*shape, T]`` → Output: ``[*shape, n_frames, kernel_size]``.
    """
    *shape, length = x.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    if tgt_length > length:
        pad_widths = [(0, 0)] * (x.ndim - 1) + [(0, tgt_length - length)]
        x = mx.pad(x, pad_widths)

    # Gather frames using index arithmetic
    frame_starts = mx.arange(n_frames) * stride
    offsets = mx.arange(kernel_size)
    indices = frame_starts[:, None] + offsets[None, :]  # [n_frames, kernel_size]

    # Flatten spatial dims, index, then reshape
    flat = x.reshape(-1, x.shape[-1])  # [batch, tgt_length]
    frames = flat[:, indices]  # [batch, n_frames, kernel_size]
    return frames.reshape(*shape, n_frames, kernel_size)


# ── Conv / Norm application helpers ──────────────────────────────────────────
# Demucs uses PyTorch [B, C, T] and [B, C, H, W] layout.
# MLX convolutions expect [B, T, C] and [B, H, W, C].
# These helpers handle the transposition at each conv boundary.

def apply_conv1d(conv: nn.Conv1d, x: mx.array) -> mx.array:
    """Apply Conv1d: x [B, C, T] → [B, C', T']."""
    return conv(x.transpose(0, 2, 1)).transpose(0, 2, 1)


def apply_conv2d(conv: nn.Conv2d, x: mx.array) -> mx.array:
    """Apply Conv2d: x [B, C, H, W] → [B, C', H', W']."""
    return conv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


def apply_conv_tr1d(conv: nn.ConvTranspose1d, x: mx.array) -> mx.array:
    """Apply ConvTranspose1d: x [B, C, T] → [B, C', T']."""
    return conv(x.transpose(0, 2, 1)).transpose(0, 2, 1)


def apply_conv_tr2d(conv: nn.ConvTranspose2d, x: mx.array) -> mx.array:
    """Apply ConvTranspose2d: x [B, C, H, W] → [B, C', H', W']."""
    return conv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


def apply_group_norm(norm: nn.Module, x: mx.array) -> mx.array:
    """Apply GroupNorm to [B, C, ...] tensor (channels-first)."""
    if isinstance(norm, nn.Identity):
        return x
    if x.ndim == 3:
        # [B, C, T] → [B, T, C] → norm → [B, C, T]
        return norm(x.transpose(0, 2, 1)).transpose(0, 2, 1)
    elif x.ndim == 4:
        # [B, C, H, W] → [B, H, W, C] → norm → [B, C, H, W]
        return norm(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    else:
        return norm(x)


# ── Activation helpers ───────────────────────────────────────────────────────

def glu(x: mx.array, axis: int = 1) -> mx.array:
    """Gated Linear Unit: splits x on ``axis``, applies sigmoid to second half."""
    half = x.shape[axis] // 2
    slices_a = [slice(None)] * x.ndim
    slices_b = [slice(None)] * x.ndim
    slices_a[axis] = slice(None, half)
    slices_b[axis] = slice(half, None)
    a = x[tuple(slices_a)]
    b = x[tuple(slices_b)]
    return a * mx.sigmoid(b)
