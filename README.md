[Colab](https://colab.research.google.com/drive/1ImErM9dHoWjyv1KykixcoV4_IBYZaEi0#scrollTo=L7IoUWkgbS7o)
• [Paper](https://arxiv.org/abs/1803.04189)

# Noise2Noise, unofficial PyTorch implementation

This is a simple implementation of the [Noise2Noise](https://arxiv.org/abs/1803.04189) paper (Lehtinen et al., 2018).

## Requirements

It requires torch, torchvision, numpy, matlplotlib and tqdm. It can be installed this way:

```bash
pip install -r requirements.txt
```

## Data

I used 32x32 images from ImageNet with random gaussian noise. The images are available in the form of pickle files for
PyTorch. Pretrained weights are also available
on [this Drive](https://drive.google.com/drive/folders/198AnkC5XJtQgJ76DNL3v5iApzZdTPQ6_?usp=sharing).

## Use

A demo Jupyter Notebook is available [here](https://github.com/tgieruc/Noise2Noise_PyTorch/blob/main/demo.ipynb).

## Report

A full report on this implementation can be
found [here](https://github.com/tgieruc/Noise2Noise_PyTorch/blob/main/Report_Noise2Noise_PyTorch.pdf).
