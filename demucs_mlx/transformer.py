# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Transformer modules for Demucs (CrossTransformerEncoder).
# Only implements the non-sparse path needed for htdemucs inference.

import math
import typing as tp

import mlx.core as mx
import mlx.nn as nn


# ── Positional embeddings ────────────────────────────────────────────────────

def create_sin_embedding(length: int, dim: int, shift: int = 0,
                         max_period: float = 10000.) -> mx.array:
    """Sinusoidal positional embedding.

    Returns: ``[1, length, dim]`` (batch-first format).
    """
    assert dim % 2 == 0
    pos = mx.arange(length).reshape(-1, 1) + shift  # [T, 1]
    half_dim = dim // 2
    adim = mx.arange(half_dim).reshape(1, -1)  # [1, D/2]
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    emb = mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)
    return emb[None, :, :]  # [1, T, D]


def create_2d_sin_embedding(d_model: int, height: int, width: int,
                            max_period: float = 10000.) -> mx.array:
    """2D sinusoidal positional embedding.

    Returns: ``[1, d_model, height, width]`` (channels-first).
    """
    if d_model % 4 != 0:
        raise ValueError(f"Cannot use sin/cos 2D encoding with odd dim={d_model}")

    pe = mx.zeros((d_model, height, width))
    d_half = d_model // 2

    div_term = mx.exp(mx.arange(0., d_half, 2) * -(math.log(max_period) / d_half))
    pos_w = mx.arange(0., width).reshape(-1, 1)   # [W, 1]
    pos_h = mx.arange(0., height).reshape(-1, 1)  # [H, 1]

    # Width sinusoids
    sin_w = mx.sin(pos_w * div_term)  # [W, d_half/2]
    cos_w = mx.cos(pos_w * div_term)  # [W, d_half/2]
    # Height sinusoids
    sin_h = mx.sin(pos_h * div_term)  # [H, d_half/2]
    cos_h = mx.cos(pos_h * div_term)  # [H, d_half/2]

    # Build pe: [d_model, height, width]
    # pe[0:d_half:2, :, :] = sin_w.T expanded over height
    # pe[1:d_half:2, :, :] = cos_w.T expanded over height
    # pe[d_half::2, :, :] = sin_h.T expanded over width
    # pe[d_half+1::2, :, :] = cos_h.T expanded over width
    #
    # Since mx.array is immutable, build each slice and stack.
    slices = []
    for i in range(d_half):
        if i % 2 == 0:
            # sin_w[:, i//2] → broadcast to [H, W]
            v = mx.broadcast_to(sin_w[:, i // 2].reshape(1, width),
                                (height, width))
        else:
            v = mx.broadcast_to(cos_w[:, i // 2].reshape(1, width),
                                (height, width))
        slices.append(v)
    for i in range(d_half):
        if i % 2 == 0:
            v = mx.broadcast_to(sin_h[:, i // 2].reshape(height, 1),
                                (height, width))
        else:
            v = mx.broadcast_to(cos_h[:, i // 2].reshape(height, 1),
                                (height, width))
        slices.append(v)
    pe = mx.stack(slices, axis=0)  # [d_model, H, W]
    return pe[None, :, :, :]  # [1, d_model, H, W]


# ── LayerScale ───────────────────────────────────────────────────────────────

class LayerScale(nn.Module):
    """Layer scale from Touvron et al 2021.

    Rescales residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0.,
                 channel_last: bool = False):
        """
        Args:
            channels: Number of channels.
            init: Initial scale value.
            channel_last: If True, input is ``[..., C]``; else ``[B, C, T]``.
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = mx.full((channels,), init)

    def __call__(self, x: mx.array) -> mx.array:
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


# ── MyGroupNorm (for batch-first transformer) ───────────────────────────────

class MyGroupNorm(nn.Module):
    """GroupNorm operating on [B, T, C] tensors.

    MLX GroupNorm expects channels as last dim, which matches [B, T, C].
    """
    def __init__(self, num_groups: int, num_channels: int,
                 eps: float = 1e-5, **kwargs):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C] — channels last, which MLX GroupNorm expects
        return self.norm(x)


# ── MultiheadAttention (standard, non-sparse) ───────────────────────────────

class MultiheadAttention(nn.Module):
    """Standard multi-head attention with batch_first support."""

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0., bias: bool = True,
                 batch_first: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def __call__(self, query: mx.array, key: mx.array, value: mx.array,
                 attn_mask: tp.Optional[mx.array] = None,
                 need_weights: bool = False) -> tp.Tuple[mx.array, None]:
        """
        Args:
            query: [B, T_q, C] if batch_first else [T_q, B, C]
            key:   [B, T_k, C] if batch_first else [T_k, B, C]
            value: [B, T_k, C] if batch_first else [T_k, B, C]
        """
        if not self.batch_first:
            query = query.transpose(1, 0, 2)  # [T, B, C] → [B, T, C]
            key = key.transpose(1, 0, 2)
            value = value.transpose(1, 0, 2)

        B, T_q, C = query.shape
        T_k = key.shape[1]

        q = self.q_proj(query).reshape(B, T_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, T_k, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, T_k, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = mx.softmax(scores, axis=-1)

        attn_out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T_q, C)
        out = self.out_proj(attn_out)

        if not self.batch_first:
            out = out.transpose(1, 0, 2)  # [B, T, C] → [T, B, C]
        return out, None


# ── Transformer encoder layer (self-attention + FFN) ─────────────────────────

class MyTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with optional LayerScale and GroupNorm.

    Operates in batch_first=True mode: input/output shape [B, T, C].
    """

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.,
                 activation=None, group_norm: int = 0,
                 norm_first: bool = False, norm_out: bool = False,
                 layer_norm_eps: float = 1e-5,
                 layer_scale: bool = False, init_values: float = 1e-4,
                 batch_first: bool = True, **kwargs):
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.use_gelu = True  # always gelu for htdemucs default

        # Norms
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.norm_out_mod = None
        if norm_first and norm_out:
            self.norm_out_mod = MyGroupNorm(
                num_groups=int(norm_out), num_channels=d_model)

        # LayerScale
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else None
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else None

    def _sa_block(self, x: mx.array) -> mx.array:
        return self.self_attn(x, x, x, need_weights=False)[0]

    def _ff_block(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.linear1(x))
        return self.linear2(x)

    def __call__(self, x: mx.array) -> mx.array:
        if self.norm_first:
            sa = self._sa_block(self.norm1(x))
            if self.gamma_1 is not None:
                sa = self.gamma_1(sa)
            x = x + sa

            ff = self._ff_block(self.norm2(x))
            if self.gamma_2 is not None:
                ff = self.gamma_2(ff)
            x = x + ff

            if self.norm_out_mod is not None:
                x = self.norm_out_mod(x)
        else:
            sa = self._sa_block(x)
            if self.gamma_1 is not None:
                sa = self.gamma_1(sa)
            x = self.norm1(x + sa)

            ff = self._ff_block(x)
            if self.gamma_2 is not None:
                ff = self.gamma_2(ff)
            x = self.norm2(x + ff)

        return x


# ── Cross-attention transformer layer ────────────────────────────────────────

class CrossTransformerEncoderLayer(nn.Module):
    """Cross-attention transformer layer.

    Query from one branch, key/value from the other.
    Operates in batch_first mode: [B, T, C].
    """

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.,
                 activation=None, layer_norm_eps: float = 1e-5,
                 layer_scale: bool = False, init_values: float = 1e-4,
                 norm_first: bool = False, group_norm: bool = False,
                 norm_out: bool = False, batch_first: bool = True,
                 **kwargs):
        super().__init__()
        self.norm_first = norm_first

        self.cross_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.use_gelu = True

        # Norms
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps)
            self.norm3 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.norm_out_mod = None
        if norm_first and norm_out:
            self.norm_out_mod = MyGroupNorm(
                num_groups=int(norm_out), num_channels=d_model)

        # LayerScale
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else None
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else None

    def _ca_block(self, q: mx.array, k: mx.array) -> mx.array:
        return self.cross_attn(q, k, k, need_weights=False)[0]

    def _ff_block(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.linear1(x))
        return self.linear2(x)

    def __call__(self, q: mx.array, k: mx.array) -> mx.array:
        if self.norm_first:
            ca = self._ca_block(self.norm1(q), self.norm2(k))
            if self.gamma_1 is not None:
                ca = self.gamma_1(ca)
            x = q + ca

            ff = self._ff_block(self.norm3(x))
            if self.gamma_2 is not None:
                ff = self.gamma_2(ff)
            x = x + ff

            if self.norm_out_mod is not None:
                x = self.norm_out_mod(x)
        else:
            ca = self._ca_block(q, k)
            if self.gamma_1 is not None:
                ca = self.gamma_1(ca)
            x = self.norm1(q + ca)

            ff = self._ff_block(x)
            if self.gamma_2 is not None:
                ff = self.gamma_2(ff)
            x = self.norm2(x + ff)

        return x


# ── CrossTransformerEncoder (freq ↔ time cross-attention) ────────────────────

class CrossTransformerEncoder(nn.Module):
    """Cross-transformer encoder alternating self-attention and cross-attention
    between a frequency branch and a time branch.

    Freq input: ``[B, C, Fr, T1]`` (channels-first spectrogram).
    Time input: ``[B, C, T2]`` (channels-first waveform features).
    """

    def __init__(self, dim: int, emb: str = "sin",
                 hidden_scale: float = 4.0, num_heads: int = 8,
                 num_layers: int = 6, cross_first: bool = False,
                 dropout: float = 0., max_positions: int = 1000,
                 norm_in: bool = True, norm_in_group: bool = False,
                 group_norm: int = False, norm_first: bool = False,
                 norm_out: bool = False, max_period: float = 10000.,
                 weight_decay: float = 0., lr: tp.Optional[float] = None,
                 layer_scale: bool = False, gelu: bool = True,
                 sin_random_shift: int = 0,
                 weight_pos_embed: float = 1.0,
                 # CAPE / sparse params (unused for inference)
                 cape_mean_normalize: bool = True,
                 cape_augment: bool = True,
                 cape_glob_loc_scale: list = [5000., 1., 1.4],
                 sparse_self_attn: bool = False,
                 sparse_cross_attn: bool = False,
                 mask_type: str = "diag",
                 mask_random_seed: int = 42,
                 sparse_attn_window: int = 500,
                 global_window: int = 50,
                 auto_sparsity: bool = False,
                 sparsity: float = 0.95):
        super().__init__()
        assert dim % num_heads == 0
        hidden_dim = int(dim * hidden_scale)

        self.num_layers = num_layers
        self.classic_parity = 1 if cross_first else 0
        self.emb = emb
        self.max_period = max_period
        self.weight_pos_embed = weight_pos_embed

        # Input norms
        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
            self.norm_in_t = nn.LayerNorm(dim)
        elif norm_in_group:
            self.norm_in = MyGroupNorm(int(norm_in_group), dim)
            self.norm_in_t = MyGroupNorm(int(norm_in_group), dim)
        else:
            self.norm_in = nn.Identity()
            self.norm_in_t = nn.Identity()

        # Scaled embedding (for emb="scaled")
        if emb == "scaled":
            self.position_embeddings = nn.Embedding(max_positions, dim)

        # Build layers
        self.layers = []
        self.layers_t = []

        kwargs_common = {
            "d_model": dim,
            "nhead": num_heads,
            "dim_feedforward": hidden_dim,
            "dropout": dropout,
            "group_norm": group_norm,
            "norm_first": norm_first,
            "norm_out": norm_out,
            "layer_scale": layer_scale,
            "batch_first": True,
        }

        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:
                self.layers.append(
                    MyTransformerEncoderLayer(**kwargs_common))
                self.layers_t.append(
                    MyTransformerEncoderLayer(**kwargs_common))
            else:
                self.layers.append(
                    CrossTransformerEncoderLayer(**kwargs_common))
                self.layers_t.append(
                    CrossTransformerEncoderLayer(**kwargs_common))

    def _get_pos_embedding(self, T: int, B: int, C: int) -> mx.array:
        """Get positional embedding. Returns [1, T, C]."""
        if self.emb == "sin":
            return create_sin_embedding(T, C, shift=0, max_period=self.max_period)
        elif self.emb == "scaled":
            pos = mx.arange(T)
            return self.position_embeddings(pos)[None, :, :]
        else:
            # Cape / other: fall back to sin for inference
            return create_sin_embedding(T, C, shift=0, max_period=self.max_period)

    def __call__(self, x: mx.array, xt: mx.array) -> tp.Tuple[mx.array, mx.array]:
        """
        Args:
            x:  Freq branch ``[B, C, Fr, T1]``.
            xt: Time branch ``[B, C, T2]``.

        Returns:
            Tuple of transformed (x, xt) in original shapes.
        """
        B, C, Fr, T1 = x.shape

        # 2D positional embedding for freq branch
        pos_emb_2d = create_2d_sin_embedding(C, Fr, T1, self.max_period)
        # [1, C, Fr, T1] → [1, T1*Fr, C]
        pos_emb_2d = pos_emb_2d.reshape(1, C, Fr * T1).transpose(0, 2, 1)

        # Reshape freq: [B, C, Fr, T1] → [B, T1*Fr, C]
        x = x.reshape(B, C, Fr * T1).transpose(0, 2, 1)
        x = self.norm_in(x)
        x = x + self.weight_pos_embed * pos_emb_2d

        # Time branch: [B, C, T2] → [B, T2, C]
        B, C, T2 = xt.shape
        xt = xt.transpose(0, 2, 1)
        pos_emb = self._get_pos_embedding(T2, B, C)
        xt = self.norm_in_t(xt)
        xt = xt + self.weight_pos_embed * pos_emb

        # Alternating self-attention and cross-attention
        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                # Self-attention (each branch independently)
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                # Cross-attention (freq ↔ time)
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        # Reshape back
        x = x.transpose(0, 2, 1).reshape(B, C, Fr, T1)
        xt = xt.transpose(0, 2, 1)
        return x, xt
