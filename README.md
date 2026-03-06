# Equilibrium Matching for Low-Light Image Enhancement

Equilibrium Matching (EqM) applied to paired low-light/normal-light image data, using a U-Net to learn the energy gradient field. Generation is performed via iterative gradient descent.

## Usage

```bash
# Train on LOL dataset
python -m src.train

# Evaluate (edit config at top of eval.py to set mode/paths)
python -m src.eval
```

## Structure

```
src/
  train.py    Training loop
  eval.py     Inference & evaluation (single image, folder, or dataset with PSNR/SSIM)
  loss.py     EqM loss with configurable decay functions c(γ)
  unet/       U-Net architecture (from github.com/milesial/Pytorch-UNet)
datasets/
  paired_dataset.py   LOL dataset loader
```
