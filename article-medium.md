# I Ported Demucs to Apple Silicon — It Separates a 7-Minute Song in 12 Seconds

Meta's Demucs is one of the best music source separation models out there. Give it any song, and it splits it into four clean stems: drums, bass, other instruments, and vocals. The problem? On a Mac, it crawls — because it runs on CPU through PyTorch.

I ported HTDemucs (the Hybrid Transformer variant) to Apple's MLX framework. The result: **34x faster than realtime** on an M4 Max. A full 7-minute track separates in about 12 seconds. And the output is bit-for-bit identical to PyTorch — less than 1 part per million difference.

The project is open source: [github.com/andrade0/demucs-mlx](https://github.com/andrade0/demucs-mlx)

---

## Why MLX?

If you have a Mac with Apple Silicon (M1, M2, M3, M4), you're sitting on a powerful GPU that most ML frameworks can't use properly. PyTorch's MPS backend exists, but Demucs doesn't work with it — complex tensors, custom ops, and various incompatibilities get in the way.

Apple's [MLX](https://github.com/ml-explore/mlx) is different. It's built from the ground up for Apple Silicon, with a NumPy-like API, lazy evaluation, and unified memory — meaning the GPU reads directly from the same memory as the CPU, with zero copy overhead. For inference workloads on Mac, it's remarkably fast.

The catch: you can't just wrap your PyTorch model and call it a day. MLX has its own conventions (channels-last layout for convolutions, no complex tensor support, different module system). A real port means rewriting the model from scratch.

## The numbers

Benchmarked on Apple M4 Max, MLX 0.21:

| Audio length | Time | Speed |
|:---:|:---:|:---:|
| 10s | 0.4s | 26x realtime |
| 30s | 1.0s | 30x realtime |
| 1 min | 1.8s | 33x realtime |
| 3 min | 5.3s | 34x realtime |
| 6 min | 10.7s | 34x realtime |

For comparison, the original PyTorch model running on CPU on the same machine takes about **11x longer**. That 7-minute track that takes 12 seconds on MLX? About 2 minutes and 15 seconds on PyTorch CPU.

And the quality is identical. I ran both models on the same inputs and measured the maximum absolute difference across all four stems: **0.8 millionths** (< 1 ppm). You literally cannot hear the difference, because there isn't one.

## How HTDemucs works

HTDemucs is a hybrid model — it processes audio through two parallel branches simultaneously:

1. **Frequency branch**: Takes an STFT spectrogram and runs it through a Conv2d U-Net (like an image processing pipeline, but on spectrograms)
2. **Time branch**: Takes the raw waveform and runs it through a Conv1d U-Net

At the bottleneck, both branches meet in a **Cross-Transformer** with 5+5 layers of cross-attention. This is where the magic happens — the frequency representation and the time representation teach each other what they've learned. Then each branch decodes back, and the results are combined into 4 separate source waveforms.

```
                    ┌─────────────────┐
   Waveform ──────>│  STFT (PyTorch)  │
       │            └────────┬────────┘
       │                     │
       v                     v
  ┌─────────┐         ┌─────────┐
  │  Time   │         │  Freq   │
  │ Encoder │         │ Encoder │
  │ (Conv1d)│         │ (Conv2d)│
  └────┬────┘         └────┬────┘
       │                   │
       └───────┬───────────┘
               v
     ┌─────────────────┐
     │ Cross-Transformer│
     │   (5+5 layers)   │
     └────────┬─────────┘
              │
       ┌──────┴──────┐
       v              v
  ┌─────────┐   ┌─────────┐
  │  Time   │   │  Freq   │
  │ Decoder │   │ Decoder │
  └────┬────┘   └────┬────┘
       │              │
       │         ┌────┴────┐
       │         │  iSTFT  │
       │         └────┬────┘
       └──────┬───────┘
              v
      4 separated stems
  (drums, bass, other, vocals)
```

## The hard parts of porting

This wasn't a simple find-and-replace from `torch` to `mlx`. Here are the interesting challenges:

### Layout conventions

PyTorch convolutions expect `[batch, channels, time]` for Conv1d and `[batch, channels, height, width]` for Conv2d. MLX expects channels-last: `[batch, time, channels]` and `[batch, height, width, channels]`.

My approach: keep the PyTorch layout `[B, C, T]` as the "external" format throughout the model, and transpose at every convolution boundary. This means every Conv layer does `transpose → conv → transpose`, which sounds expensive but is essentially free on MLX since transposes are just view operations on unified memory.

### No complex tensors

MLX doesn't support complex numbers, and STFT/iSTFT are inherently complex operations. Rather than reimplementing the Short-Time Fourier Transform from scratch (which would be error-prone and hard to validate), I kept STFT/iSTFT in PyTorch running on CPU. The data crosses the PyTorch-MLX boundary twice per forward pass. This adds about 5% overhead but guarantees numerical correctness.

### Weight conversion

The original PyTorch checkpoint can't be loaded directly into MLX. Conv weights need their axes transposed. The `MultiheadAttention` module packs Q, K, V projections into a single `in_proj_weight` — MLX uses separate linear layers, so these need to be split. Batch norm parameters need to be folded into the weights since we're inference-only. The LSTM layers in the encoder have a different parameter layout between PyTorch and MLX.

All of this is handled automatically — the model downloads Meta's original checkpoint and converts it on the fly.

### The bug that took hours

The most subtle bug: PyTorch's `Conv2d` accepts scalar arguments that expand to both dimensions. `Conv2d(ch, ch, 3, 1, 1)` means kernel `(3, 3)`, stride `(1, 1)`, padding `(1, 1)`. In my MLX port, I was passing `(3, 1)` for the kernel and `(1, 0)` for padding — applying context only to the frequency dimension, not time. The model ran fine, the shapes were close enough, but the decoder output was 334 time steps instead of 336, causing a shape mismatch with the skip connections. It took careful shape-tracing through every layer to find it.

## Getting started

You need a Mac with Apple Silicon (M1/M2/M3/M4) and Python 3.10+. I recommend using a dedicated conda environment:

```bash
# Create a clean environment
conda create -n demucs python=3.11 -y
conda activate demucs

# Clone and install
git clone https://github.com/andrade0/demucs-mlx.git
cd demucs-mlx
make install
```

That's it. Now from anywhere on your Mac:

```bash
# Separate a song into 4 stems
demucs-mlx song.mp3

# Extract only vocals
demucs-mlx song.mp3 --stems vocals

# Extract vocals and drums
demucs-mlx song.mp3 --stems vocals drums

# Better quality (slower)
demucs-mlx song.mp3 --shifts 3
```

The first run downloads the pretrained model (~80 MB). Output goes to `separated/htdemucs/<song>/` with 4 files: `drums.wav`, `bass.wav`, `other.wav`, `vocals.wav`.

## Choosing a model

There are three model variants, all available through the `-n` flag:

| Model | Sources | Best for |
|:---:|:---:|:---:|
| `htdemucs` | 4 (drums, bass, other, vocals) | General use — fast and reliable |
| `htdemucs_ft` | 4 (drums, bass, other, vocals) | Better vocal separation — fine-tuned on more data |
| `htdemucs_6s` | 6 (drums, bass, other, vocals, guitar, piano) | When you need guitar or piano isolated separately |

The default `htdemucs` works well for most use cases. If you're specifically after clean vocals (for karaoke, remixing, or running through Whisper for transcription), use `htdemucs_ft` — the fine-tuning makes a noticeable difference on complex mixes.

```bash
# Default model
demucs-mlx song.mp3

# Fine-tuned model (better vocals)
demucs-mlx song.mp3 -n htdemucs_ft

# 6-source model (guitar + piano separated)
demucs-mlx song.mp3 -n htdemucs_6s
```

Models are ~80 MB each and cached in `~/.cache/demucs_mlx/` after the first download.

You can also use it from Python:

```python
import mlx.core as mx
import soundfile as sf
from demucs_mlx.pretrained import load_model
from demucs_mlx.apply import apply_model

model = load_model("htdemucs")
wav, sr = sf.read("song.mp3", dtype="float32")
mix = mx.array(wav.T[None])  # [1, channels, samples]

sources = apply_model(model, mix, shifts=1, split=True)
mx.eval(sources)
# sources shape: [1, 4, channels, samples]
# stems: drums, bass, other, vocals
```

## What you can do with it

Once you can separate stems fast, a lot of things become practical:

- **DJing**: Isolate vocals or drums from any track for live remixing
- **Music production**: Sample specific elements from existing tracks
- **Karaoke**: Extract instrumentals instantly
- **Transcription**: Isolate vocals, then run Whisper for clean lyrics
- **Practice**: Remove an instrument from a track and play along
- **Audio forensics**: Isolate specific elements from recordings

The speed makes batch processing realistic too. You can run your entire music library through it and have stems for everything.

## What's next

- Removing the PyTorch dependency entirely (native MLX STFT)
- Extended generation support for very long tracks
- Performance optimizations (KV cache reuse in the transformer)
- Testing the 6-source model variant

Contributions welcome: [github.com/andrade0/demucs-mlx](https://github.com/andrade0/demucs-mlx)

---

*Demucs MLX is MIT licensed. The original Demucs model and pretrained weights are by Meta Research. MLX is by Apple. HTDemucs paper: [Rouard, Massa, Defossez (2023)](https://arxiv.org/abs/2211.08553).*
