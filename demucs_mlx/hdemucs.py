# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Hybrid Demucs encoder/decoder layers.
# HEncLayer, HDecLayer, ScaledEmbedding.

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from .demucs import DConv
from .utils import (apply_conv1d, apply_conv2d, apply_conv_tr1d,
                    apply_conv_tr2d, apply_group_norm, glu)


class ScaledEmbedding(nn.Module):
    """Embedding with a learnable scale (boost factor for learning rate)."""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 scale: float = 10., smooth: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.scale = scale
        # Note: smooth init and weight scaling are handled during weight loading.
        # At inference time we just use the loaded weights.

    def __call__(self, x: mx.array) -> mx.array:
        return self.embedding(x) * self.scale


class HEncLayer(nn.Module):
    """Encoder layer for the hybrid Demucs model.

    Works for both the frequency branch (Conv2d over freq×time) and
    the time branch (Conv1d over time).

    Tensors: ``[B, C, T]`` for time, ``[B, C, Fr, T]`` for freq.
    """

    def __init__(self, chin: int, chout: int, kernel_size: int = 8,
                 stride: int = 4, norm_groups: int = 1, empty: bool = False,
                 freq: bool = True, dconv: bool = True, norm: bool = True,
                 context: int = 0, dconv_kw: tp.Dict = {}, pad: bool = True,
                 rewrite: bool = True):
        super().__init__()
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.use_norm = norm
        self.pad_val = kernel_size // 4 if pad else 0

        if freq:
            ks = (kernel_size, 1)
            st = (stride, 1)
            pd = (self.pad_val, 0)
            self.conv = nn.Conv2d(chin, chout, ks, st, pd)
        else:
            self.conv = nn.Conv1d(chin, chout, kernel_size, stride,
                                  padding=self.pad_val)

        if empty:
            return

        if norm:
            self.norm1 = nn.GroupNorm(norm_groups, chout, pytorch_compatible=True)
        else:
            self.norm1 = None

        self.rewrite_conv = None
        self.norm2 = None
        if rewrite:
            if freq:
                # PyTorch uses scalar args that expand to both dims:
                # Conv2d(chout, 2*chout, 1+2*context, 1, context)
                # → kernel=(1+2*ctx, 1+2*ctx), padding=(ctx, ctx)
                ks = 1 + 2 * context
                self.rewrite_conv = nn.Conv2d(chout, 2 * chout,
                                              (ks, ks), (1, 1),
                                              (context, context))
            else:
                self.rewrite_conv = nn.Conv1d(chout, 2 * chout,
                                              1 + 2 * context, 1, context)
            if norm:
                self.norm2 = nn.GroupNorm(norm_groups, 2 * chout,
                                         pytorch_compatible=True)

        self.dconv_mod = None
        if dconv:
            self.dconv_mod = DConv(chout, **dconv_kw)

    def __call__(self, x: mx.array, inject: tp.Optional[mx.array] = None) -> mx.array:
        """
        Args:
            x: ``[B, C, Fr, T]`` for freq, ``[B, C, T]`` for time.
            inject: Time-branch injection when merging.
        """
        if not self.freq and x.ndim == 4:
            B, C, Fr, T = x.shape
            x = x.reshape(B, C * Fr, T)

        if not self.freq:
            le = x.shape[-1]
            if le % self.stride != 0:
                pad_r = self.stride - (le % self.stride)
                pad_widths = [(0, 0)] * (x.ndim - 1) + [(0, pad_r)]
                x = mx.pad(x, pad_widths)

        y = apply_conv2d(self.conv, x) if self.freq else apply_conv1d(self.conv, x)

        if self.empty:
            return y

        if inject is not None:
            if inject.ndim == 3 and y.ndim == 4:
                inject = inject[:, :, None, :]
            y = y + inject

        if self.norm1 is not None:
            y = apply_group_norm(self.norm1, y)
        y = nn.gelu(y)

        if self.dconv_mod is not None:
            if self.freq:
                B, C, Fr, T = y.shape
                # Fold freq into batch: [B*Fr, C, T]
                y = y.transpose(0, 2, 1, 3).reshape(B * Fr, C, T)
                y = self.dconv_mod(y)
                y = y.reshape(B, Fr, C, T).transpose(0, 2, 1, 3)
            else:
                y = self.dconv_mod(y)

        if self.rewrite_conv is not None:
            z = (apply_conv2d(self.rewrite_conv, y)
                 if self.freq
                 else apply_conv1d(self.rewrite_conv, y))
            if self.norm2 is not None:
                z = apply_group_norm(self.norm2, z)
            z = glu(z, axis=1)
        else:
            z = y
        return z


class HDecLayer(nn.Module):
    """Decoder layer for the hybrid Demucs model.

    Mirror of HEncLayer with transposed convolution for upsampling.
    """

    def __init__(self, chin: int, chout: int, last: bool = False,
                 kernel_size: int = 8, stride: int = 4,
                 norm_groups: int = 1, empty: bool = False,
                 freq: bool = True, dconv: bool = True, norm: bool = True,
                 context: int = 1, dconv_kw: tp.Dict = {}, pad: bool = True,
                 context_freq: bool = True, rewrite: bool = True):
        super().__init__()
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.last = last
        self.use_norm = norm
        self.pad_val = kernel_size // 4 if pad else 0
        self.context_freq = context_freq

        if freq:
            ks = (kernel_size, 1)
            st = (stride, 1)
            self.conv_tr = nn.ConvTranspose2d(chin, chout, ks, st)
        else:
            self.conv_tr = nn.ConvTranspose1d(chin, chout, kernel_size, stride)

        if norm:
            self.norm2 = nn.GroupNorm(norm_groups, chout, pytorch_compatible=True)
        else:
            self.norm2 = None

        if empty:
            return

        self.rewrite_conv = None
        self.norm1 = None
        if rewrite:
            if freq:
                # PyTorch uses scalar args that expand to both dims:
                # Conv2d(chin, 2*chin, 1+2*context, 1, context)
                # → kernel=(1+2*ctx, 1+2*ctx), padding=(ctx, ctx)
                ks = 1 + 2 * context
                if context_freq:
                    self.rewrite_conv = nn.Conv2d(chin, 2 * chin,
                                                  (ks, ks), (1, 1),
                                                  (context, context))
                else:
                    self.rewrite_conv = nn.Conv2d(chin, 2 * chin,
                                                  (1, ks), (1, 1),
                                                  (0, context))
            else:
                self.rewrite_conv = nn.Conv1d(chin, 2 * chin,
                                              2 * context + 1, 1, context)
            if norm:
                self.norm1 = nn.GroupNorm(norm_groups, 2 * chin,
                                         pytorch_compatible=True)

        self.dconv_mod = None
        if dconv:
            self.dconv_mod = DConv(chin, **dconv_kw)

    def __call__(self, x: mx.array, skip: tp.Optional[mx.array],
                 length: int) -> tp.Tuple[mx.array, mx.array]:
        """
        Args:
            x: Current features.
            skip: Skip connection from encoder.
            length: Target time length for the freq branch padding removal.

        Returns:
            (output, pre_transpose_features).
        """
        if self.freq and x.ndim == 3:
            B, C, T = x.shape
            x = x.reshape(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip

            if self.rewrite_conv is not None:
                y = (apply_conv2d(self.rewrite_conv, x)
                     if self.freq
                     else apply_conv1d(self.rewrite_conv, x))
                if self.norm1 is not None:
                    y = apply_group_norm(self.norm1, y)
                y = glu(y, axis=1)
            else:
                y = x

            if self.dconv_mod is not None:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.transpose(0, 2, 1, 3).reshape(B * Fr, C, T)
                    y = self.dconv_mod(y)
                    y = y.reshape(B, Fr, C, T).transpose(0, 2, 1, 3)
                else:
                    y = self.dconv_mod(y)
        else:
            y = x
            assert skip is None

        z = (apply_conv_tr2d(self.conv_tr, y)
             if self.freq
             else apply_conv_tr1d(self.conv_tr, y))

        if self.norm2 is not None:
            z = apply_group_norm(self.norm2, z)

        if self.freq:
            if self.pad_val:
                z = z[:, :, self.pad_val:-self.pad_val, :]
        else:
            z = z[:, :, self.pad_val:self.pad_val + length]

        if not self.last:
            z = nn.gelu(z)
        return z, y
