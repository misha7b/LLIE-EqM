# EqM+ for Low-Light Image Enhancement

EqM+ adapts [Equilibrium Matching](https://arxiv.org/abs/2510.02300) to paired low-light image enhancement: given a dark image and its well-lit target, it learns a descent field that iteratively moves the dark input toward the clean image.

## Method Summary

For a paired sample `(x_dark, x_gt)`, EqM+ samples an interpolated state between the low-light image and the target image, and trains a network `v_theta(x_gamma, x_dark)` to predict a scaled direction from `x_gt` back to `x_dark`. At inference time, enhancement starts from `x_dark` itself and applies a small number of gradient-descent steps in the learned field.

## Examples

<p align="center">
  <img src="img/8_comparison.png" alt="Comparison example 8" width="48%" />
  <img src="img/9_comparison.png" alt="Comparison example 9" width="48%" />
</p>

<p align="center">
  <img src="img/18_comparison.png" alt="Comparison example 18" width="48%" />
  <img src="img/101_comparison.png" alt="Comparison example 101" width="48%" />
</p>

## Endoscopy Benchmarks

Reported results on the EndoVis17 and EndoVis18 low-light surgical endoscopy datasets:

| Method | EndoVis17 PSNR | EndoVis17 SSIM | EndoVis18 PSNR | EndoVis18 SSIM |
| --- | ---: | ---: | ---: | ---: |
| LIME | 11.56 | 0.242 | 11.69 | 0.299 |
| DiffLL | 30.57 | 0.930 | 28.07 | 0.918 |
| LighTDiff | 34.28 | 0.957 | 31.99 | 0.949 |
| EqM+ (ours) | 30.20 | 0.934 | 28.51 | 0.926 |

EqM+ is competitive on both endoscopy benchmarks.

## Running The Code

Configuration options are in files `src/train.py` and `src/eval.py` respectively.

```bash
# Train EqM / EqM+
python -m src.train

# Evaluate on a paired dataset, a folder, or a single image
python -m src.eval



## Repository Layout

```text
src/
  train.py            Training loop
  eval.py             Inference and evaluation
  loss.py             EqM / EqM+ losses
  eqmnet.py           Default EqM+ backbone (`eqmnet2_small`)
datasets/
  lol_dataset.py      Paired dataset loader
img/
  eqmnet.svg          Model diagram
```

## Architecture

This repo supports two architectures:

**UNet** (`src/unet/`) — UNet adapted from [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

**EqMNet** (`src/eqmnet.py`) — custom architecture, 1.3M parameters.


<p align="center">
  <img src="img/eqmnet.svg" alt="EqM+ Architecture" width="50%" />
</p>





