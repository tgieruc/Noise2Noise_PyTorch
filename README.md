# Noise2Noise, unofficial PyTorch implementation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tgieruc/Noise2Noise_PyTorch/blob/main/Noise2Noise_PyTorch_demo.ipynb)
• [Paper](https://arxiv.org/abs/1803.04189)

This is a simple implementation of the [Noise2Noise](https://arxiv.org/abs/1803.04189) paper (Lehtinen et al., 2018). Trains a U-Net-style denoising network using pairs of noisy images — no clean targets needed.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Data

I used 32x32 images from ImageNet with random gaussian noise. The images are available in the form of pickle files for
PyTorch. Pretrained weights are also available
on [this Drive](https://drive.google.com/drive/folders/198AnkC5XJtQgJ76DNL3v5iApzZdTPQ6_?usp=sharing).

## Usage

### Training

```bash
uv run n2n-train --train-data train_data.pkl --epochs 10
```

Options:
- `--batch-size` — batch size (default: 8)
- `--lr` — learning rate (default: 0.001)
- `--num-workers` — data loading workers (default: 2)
- `--output` — path to save model (default: model_pytorch.pth)
- `--pretrained` — resume from pretrained weights

### Inference

```bash
uv run n2n-predict --model model_pytorch.pth --input val_data.pkl
```

Options:
- `--output` — save denoised images to a pickle file

### Notebook

A demo Jupyter Notebook is also available: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tgieruc/Noise2Noise_PyTorch/blob/main/Noise2Noise_PyTorch_demo.ipynb).

## Tests

```bash
uv run python -m pytest tests/
```

## Report

A full report on this implementation can be
found [here](https://github.com/tgieruc/Noise2Noise_PyTorch/blob/main/docs/Report_Noise2Noise_PyTorch.pdf).
