# EqM+ for Low-Light Image Enhancement

This repository contains the EqM+ model from the report *On Flow-Based Generative Modelling and its Applications* (`flow_based_generative_modelling.pdf`, source in `main.tex`).

EqM+ adapts Equilibrium Matching to paired low-light image enhancement: given a dark image and its well-lit target, it learns a descent field that iteratively moves the dark input toward the clean image.

## Architecture

<p align="center">
  <img src="img/eqmnet.svg" alt="EqM+ Architecture" width="100%" />
</p>

## Method Summary

For a paired sample `(x_dark, x_gt)`, EqM+ samples an interpolated state between the low-light image and the target image, and trains a network `v_theta(x_gamma, x_dark)` to predict a scaled direction from `x_gt` back to `x_dark`. At inference time, enhancement starts from `x_dark` itself and applies a small number of gradient-descent steps in the learned field.

The implementation in this repo uses a compact EqMNet v2 backbone (`eqmnet2_small`, about `1.3M` parameters) with:

- a dual-input formulation based on the current iterate and the original dark image
- a lightweight dark-image encoder whose features are reused across descent steps
- multi-scale fusion with bottleneck self-attention and cross-attention
- post-upsampling smoothing to reduce checkerboard artifacts
- a truncated EqM decay with Charbonnier, FFT, and cosine-direction losses

In the report, EqM+ is evaluated on LOL / LOL-v2 and benchmarked on surgical endoscopy data.

## Endoscopy Benchmarks

Reported results on the EndoVis17 and EndoVis18 low-light surgical endoscopy datasets:

| Method | EndoVis17 PSNR | EndoVis17 SSIM | EndoVis18 PSNR | EndoVis18 SSIM |
| --- | ---: | ---: | ---: | ---: |
| LIME | 11.56 | 0.242 | 11.69 | 0.299 |
| DiffLL | 30.57 | 0.930 | 28.07 | 0.918 |
| LighTDiff | 34.28 | 0.957 | 31.99 | 0.949 |
| EqM+ (ours) | 30.20 | 0.934 | 28.51 | 0.926 |

EqM+ is competitive on both endoscopy benchmarks and clearly outperforms classical enhancement baselines in the reported experiments.

## Running The Code

The training and evaluation scripts are configuration-driven. Before running them, set dataset roots, checkpoint paths, and evaluation mode in the constants at the top of each script.

```bash
# Train EqM / EqM+
python -m src.train

# Evaluate on a paired dataset, a folder, or a single image
python -m src.eval
```

Current defaults in the code use:

- `AdamW` with cosine annealing in [`src/train.py`](src/train.py)
- the `eqmnet2_small` backbone in [`src/eqmnet.py`](src/eqmnet.py)
- `12` descent steps with step size `2e-2` in [`src/eval.py`](src/eval.py)
- the paired dataset loader in [`datasets/lol_dataset.py`](datasets/lol_dataset.py)

## Repository Layout

```text
src/
  train.py            Training loop
  eval.py             Inference and quantitative evaluation
  loss.py             EqM / EqM+ losses
  eqmnet.py           Default EqM+ backbone (`eqmnet2_small`)
datasets/
  lol_dataset.py      Paired dataset loader
img/
  eqmnet.svg          Model diagram
```

## Examples

<p align="center">
  <img src="img/8_comparison.png" alt="Comparison example 8" width="48%" />
  <img src="img/9_comparison.png" alt="Comparison example 9" width="48%" />
</p>

<p align="center">
  <img src="img/18_comparison.png" alt="Comparison example 18" width="48%" />
  <img src="img/101_comparison.png" alt="Comparison example 101" width="48%" />
</p>

## Reference

For the full derivation, experimental discussion, and additional qualitative results, see `flow_based_generative_modelling.pdf` and `main.tex`.
