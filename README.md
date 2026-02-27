# Demucs MLX

Music source separation on Apple Silicon, powered by [MLX](https://github.com/ml-explore/mlx).

A clean MLX port of Meta's [Hybrid Transformer Demucs](https://github.com/facebookresearch/demucs) (HTDemucs) for **inference-only** on Mac M1/M2/M3/M4. Separates any song into 4 stems: **drums**, **bass**, **other**, and **vocals**.

<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-MLX-black?logo=apple" alt="Apple Silicon MLX">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
</p>

## Performance

Benchmarked on Apple M4 Max with MLX 0.21:

| Audio length | Processing time | Speed |
|:---:|:---:|:---:|
| 10s | 0.4s | **26x** realtime |
| 30s | 1.0s | **30x** realtime |
| 1 min | 1.8s | **33x** realtime |
| 3 min | 5.3s | **34x** realtime |
| 6 min | 10.7s | **34x** realtime |

**11x faster** than PyTorch CPU on the same machine. A 7-minute track separates in about 12 seconds.

Numerical accuracy: max difference vs PyTorch is **< 1 part per million** (0.8 µ). The output is effectively identical.

## Quick start

### Install

```bash
# Clone the repo
git clone https://github.com/andrade0/demucs-mlx.git
cd demucs-mlx

# Install dependencies
pip install -r requirements.txt
```

### Separate a song

```bash
python separate.py song.mp3
```

That's it. The first run downloads the pretrained model (~80 MB, cached in `~/.cache/demucs_mlx/`). Output goes to `separated/htdemucs/<song>/`.

### More options

```bash
# Extract only vocals
python separate.py song.mp3 --stems vocals

# Extract vocals and drums
python separate.py song.mp3 --stems vocals drums

# Custom output directory
python separate.py song.mp3 -o my_stems/

# More shifts = better quality, slower (default: 1)
python separate.py song.mp3 --shifts 3

# Save as float32 WAV (default: int16)
python separate.py song.mp3 --float32
```

### Use from Python

```python
import mlx.core as mx
import soundfile as sf
from demucs_mlx.pretrained import load_model
from demucs_mlx.apply import apply_model

# Load model (downloads on first use)
model = load_model("htdemucs")

# Load audio
wav, sr = sf.read("song.mp3", dtype="float32")
mix = mx.array(wav.T[None])  # [1, channels, samples]

# Separate
sources = apply_model(model, mix, shifts=1, split=True)
mx.eval(sources)

# sources shape: [1, 4, channels, samples]
# stems: drums, bass, other, vocals
```

## How it works

HTDemucs is a hybrid model that processes audio in two parallel branches:

1. **Frequency branch** — STFT spectrogram through a Conv2d U-Net
2. **Time branch** — Raw waveform through a Conv1d U-Net

Both branches meet at a **Cross-Transformer** bottleneck that lets them exchange information, then decode back to 4 separate source waveforms.

```
                    ┌─────────────────┐
   Waveform ──────►│  STFT (PyTorch)  │
       │            └────────┬────────┘
       │                     │
       ▼                     ▼
  ┌─────────┐         ┌─────────┐
  │  Time   │         │  Freq   │
  │ Encoder │         │ Encoder │
  │ (Conv1d)│         │ (Conv2d)│
  └────┬────┘         └────┬────┘
       │                   │
       └───────┬───────────┘
               ▼
     ┌─────────────────┐
     │ Cross-Transformer│
     │   (5+5 layers)   │
     └────────┬─────────┘
              │
       ┌──────┴──────┐
       ▼              ▼
  ┌─────────┐   ┌─────────┐
  │  Time   │   │  Freq   │
  │ Decoder │   │ Decoder │
  └────┬────┘   └────┬────┘
       │              │
       │         ┌────┴────┐
       │         │  iSTFT  │
       │         └────┬────┘
       └──────┬───────┘
              ▼
      4 separated stems
  (drums, bass, other, vocals)
```

### MLX-specific design choices

- **Layout convention**: Tensors use PyTorch's `[B, C, T]` / `[B, C, H, W]` format externally; transpose at every Conv boundary since MLX expects channels-last
- **STFT/iSTFT**: Runs through PyTorch as a bridge (MLX doesn't support complex tensors). This adds ~5% overhead but keeps the code simple
- **Weight loading**: Downloads the original PyTorch checkpoint, converts Conv weights (transpose axes) and splits MultiheadAttention projections, then loads into the MLX model
- **No training code**: Inference-only, all batch norm / dropout / weight init stripped out

## Models

| Model | Description | Status |
|-------|-------------|--------|
| `htdemucs` | Default HTDemucs (4 sources) | Supported |
| `htdemucs_ft` | Fine-tuned on more data | Supported |
| `htdemucs_6s` | 6-source variant | Untested |

```bash
# Use the fine-tuned model
python separate.py song.mp3 -n htdemucs_ft
```

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+**
- Dependencies: `mlx`, `numpy`, `torch` (CPU), `soundfile`, `pyyaml`, `tqdm`
- Optional: `librosa` (only needed if input sample rate differs from 44.1 kHz)

> **Note on PyTorch**: PyTorch is used only for STFT/iSTFT and for loading the pretrained weights. It runs on CPU and adds minimal overhead. A future version may remove this dependency entirely.

## Project structure

```
demucs-mlx/
├── separate.py              # CLI entry point
├── demucs_mlx/
│   ├── htdemucs.py          # Main model (hybrid U-Net + transformer)
│   ├── hdemucs.py           # Encoder/decoder layers
│   ├── transformer.py       # Cross-attention transformer
│   ├── demucs.py            # DConv residual branches
│   ├── apply.py             # Inference pipeline (chunking, shifts)
│   ├── pretrained.py        # Model download & weight loading
│   ├── weight_convert.py    # PyTorch → MLX weight conversion
│   ├── spec.py              # STFT/iSTFT bridge
│   └── utils.py             # Tensor utilities
└── remote/                  # Model download configs
    ├── files.txt
    └── htdemucs.yaml
```

## License

MIT License — same as the [original Demucs](https://github.com/facebookresearch/demucs).

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) by Meta Research — the original PyTorch model and pretrained weights
- [MLX](https://github.com/ml-explore/mlx) by Apple — the framework that makes this fast on Apple Silicon
- HTDemucs paper: [Rouard, Massa, Défossez (2023)](https://arxiv.org/abs/2211.08553)
