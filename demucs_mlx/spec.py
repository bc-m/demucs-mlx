# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – STFT/iSTFT via PyTorch bridge.
# Complex number operations (STFT, iSTFT, view_as_real, view_as_complex)
# are not supported natively in MLX, so we delegate to PyTorch.

import torch
import numpy as np


def spectro(x_np: np.ndarray, n_fft: int = 512,
            hop_length: int = None, pad: int = 0) -> np.ndarray:
    """STFT via PyTorch.

    Args:
        x_np: Real-valued numpy array of shape ``[*other, length]``.
        n_fft: FFT size.
        hop_length: Hop length (default: n_fft // 4).
        pad: Extra padding multiplier for n_fft.

    Returns:
        Complex numpy array of shape ``[*other, freqs, frames]``.
    """
    *other, length = x_np.shape
    x = torch.from_numpy(x_np.reshape(-1, length))
    z = torch.stft(
        x,
        n_fft * (1 + pad),
        hop_length or n_fft // 4,
        window=torch.hann_window(n_fft).to(x),
        win_length=n_fft,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode='reflect',
    )
    _, freqs, frames = z.shape
    return z.view(*other, freqs, frames).numpy()


def ispectro(z_np: np.ndarray, hop_length: int = None,
             length: int = None, pad: int = 0) -> np.ndarray:
    """Inverse STFT via PyTorch.

    Args:
        z_np: Complex numpy array of shape ``[*other, freqs, frames]``.
        hop_length: Hop length.
        length: Target output length.
        pad: Padding multiplier (must match ``spectro``).

    Returns:
        Real numpy array of shape ``[*other, length]``.
    """
    *other, freqs, frames = z_np.shape
    n_fft = 2 * freqs - 2
    z = torch.from_numpy(z_np.reshape(-1, freqs, frames))
    win_length = n_fft // (1 + pad)
    x = torch.istft(
        z,
        n_fft,
        hop_length,
        window=torch.hann_window(win_length).to(z.real),
        win_length=win_length,
        normalized=True,
        length=length,
        center=True,
    )
    _, out_length = x.shape
    return x.view(*other, out_length).numpy()
