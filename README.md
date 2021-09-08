# Fast Transformer 

![PyPI](https://img.shields.io/pypi/v/fast-transformer-torch)
[![Lint Code Base](https://github.com/talipturkmen/Fast-Transformer-Pytorch/actions/workflows/linter.yml/badge.svg)](https://github.com/talipturkmen/Fast-Transformer-Pytorch/actions/workflows/linter.yml)
[![Upload Python Package](https://github.com/talipturkmen/Fast-Transformer-Pytorch/actions/workflows/python-publish.yml/badge.svg)](https://github.com/talipturkmen/Fast-Transformer-Pytorch/actions/workflows/python-publish.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5406025.svg)](https://doi.org/10.5281/zenodo.5406025)
![GitHub License](https://img.shields.io/github/license/talipturkmen/Fast-Transformer-Pytorch)
[![GitHub stars](https://img.shields.io/github/stars/talipturkmen/Fast-Transformer-Pytorch?style=social)](https://github.com/talipturkmen/Fast-Transformer-Pytorch/stargazers)

This repo implements [Fastformer: Additive Attention Can Be All You Need](https://arxiv.org/abs/2108.09084) by Wu et al. in 
Pytorch based on the implementation of [Rishit Dagli](https://github.com/Rishit-dagli/Fast-Transformer).
**Fast Transformer** is a Transformer variant based on additive attention that can handle long sequences 
efficiently with linear complexity. Fastformer is much more efficient than many existing Transformer models and can 
meanwhile achieve comparable or even better long text modeling performance.

![](https://github.com/talipturkmen/Fast-Transformer-Pytorch/blob/main/media/architecture.png)

## Installation

Run the following to install:

```sh
pip install fast-transformer-torch
```

## Developing fast-transformer

To install `fast-transformer-torch`, along with tools you need to develop and test, run the following in your virtualenv:

```sh
git clone https://github.com/talipturkmen/Fast-Transformer-Pytorch.git
# or clone your own fork

cd fast-transformer-torch
pip install -e .[dev]
```

## Usage

```python
from fast_transformer_torch import FastTransformer
import torch

mask = torch.ones([16, 4096], dtype=torch.bool)
model = FastTransformer(num_tokens = 20000,
                        dim = 512,
                        depth = 2,
                        max_seq_len = 4096,
                        absolute_pos_emb = True, # Absolute positional embeddings
                        mask = mask
                        )
x = torch.randint(0, 20000, (16, 4096))

logits = model(x) # (1, 4096, 20000)
```
