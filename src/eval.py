# src/eval.py

import os
import glob

import torch
from torchvision import transforms as T
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image

from src.unet.unet_model import UNet
from datasets.lol_dataset import LOLPairedDataset


# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH = "outputs/eqm_lol.pt"
SAVE_DIR = "outputs/eval"

# Generation
NUM_STEPS = 30
ETA = 2e-2
METHOD = "gd"        


MODE = "dataset" # single, folder, dataset
IMG_PATH = "datasets/custom/pool.png"       # used by "single"
IMG_DIR = "datasets/custom"                 # used by "folder"
DATA_ROOT = "datasets/LOL"                  # used by "dataset"

SAVE_STEPS = True      
SAVE_COMPARISON = True
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_gd(model, x_dark, num_steps=NUM_STEPS, eta=ETA, save_steps=False):
    y = x_dark.clone()
    x_k = x_dark.clone()
    history = [x_k.cpu().clone()] if save_steps else None

    for step in range(num_steps):
        pred_grad = model(torch.cat([x_k, y], dim=1))
        x_k = (x_k - eta * pred_grad).clamp(0.0, 1.0)

        if save_steps:
            history.append(x_k.cpu().clone())

        if step > num_steps // 2:
            eta *= 0.99

    return x_k, history

def generate(model, x_dark, save_steps=False):
    return generate_gd(model, x_dark, NUM_STEPS, ETA, save_steps=save_steps)



def load_model(model_path, device):
    model = UNet(n_channels=6, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    return model

def save_comparison(images, labels, path):
    widths, heights = zip(*(img.size for img in images))
    total_w = sum(widths)
    max_h = max(heights)
    canvas = Image.new("RGB", (total_w, max_h))
    x_offset = 0
    for img in images:
        canvas.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    canvas.save(path)
    
def save_outputs(out_dir, name, x_dark_pil, result_tensor, history=None,
                 gt_pil=None):
    to_pil = T.ToPILImage()
    os.makedirs(out_dir, exist_ok=True)

    result_pil = to_pil(result_tensor.squeeze(0).cpu())
    result_pil.save(os.path.join(out_dir, f"{name}_result.png"))

    if history is not None:
        steps_dir = os.path.join(out_dir, f"{name}_steps")
        os.makedirs(steps_dir, exist_ok=True)
        x_dark_pil.save(os.path.join(steps_dir, "input_dark.png"))
        for i, x_step in enumerate(history):
            to_pil(x_step.squeeze(0)).save(os.path.join(steps_dir, f"step_{i:02d}.png"))

    if SAVE_COMPARISON:
        images = [x_dark_pil, result_pil]
        if gt_pil is not None:
            images.append(gt_pil)
        save_comparison(images, None, os.path.join(out_dir, f"{name}_comparison.png"))


# ── Modes ──────────────────────────────────────────────────────────────

def run_single(model, device):
    """Enhance a single image."""
    assert os.path.exists(IMG_PATH), f"Image not found: {IMG_PATH}"
    x_dark_pil = Image.open(IMG_PATH).convert("RGB")
    x_dark = T.ToTensor()(x_dark_pil).unsqueeze(0).to(device)
    print(f"Input: {IMG_PATH} — {x_dark.shape}")

    result, history = generate(model, x_dark, save_steps=SAVE_STEPS)
    name = os.path.splitext(os.path.basename(IMG_PATH))[0]
    save_outputs(SAVE_DIR, name, x_dark_pil, result, history)
    print(f"Saved to {SAVE_DIR}")
    
    
def run_folder(model, device):
    """Enhance all images in a directory."""
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png"))
                   + glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    assert paths, f"No images found in {IMG_DIR}"
    print(f"Found {len(paths)} images in {IMG_DIR}")

    to_tensor = T.ToTensor()
    for path in paths:
        x_dark_pil = Image.open(path).convert("RGB")
        x_dark = to_tensor(x_dark_pil).unsqueeze(0).to(device)

        result, history = generate(model, x_dark, save_steps=SAVE_STEPS)
        name = os.path.splitext(os.path.basename(path))[0]
        save_outputs(SAVE_DIR, name, x_dark_pil, result, history)
        print(f"  {name} done")

    print(f"Saved to {SAVE_DIR}")
    
def run_dataset(model, device):
    """Evaluate on LOL with PSNR and SSIM metrics."""
    test_dataset = LOLPairedDataset(root_dir=DATA_ROOT, split="test")
    print(f"Evaluating on {len(test_dataset)} test pairs")

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    to_pil = T.ToPILImage()
    psnr_vals, ssim_vals = [], []

    for idx in range(len(test_dataset)):
        x_light, x_dark = test_dataset[idx]
        x_light = x_light.unsqueeze(0).to(device)
        x_dark = x_dark.unsqueeze(0).to(device)

        result, history = generate(model, x_dark, save_steps=SAVE_STEPS)

        psnr_val = psnr_fn(result, x_light).item()
        ssim_val = ssim_fn(result, x_light).item()
        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)

        # Save outputs
        name = f"{idx:03d}"
        x_dark_pil = to_pil(x_dark.squeeze(0).cpu())
        gt_pil = to_pil(x_light.squeeze(0).cpu())
        save_outputs(SAVE_DIR, name, x_dark_pil, result, history, gt_pil=gt_pil)

        print(f"  [{idx+1}/{len(test_dataset)}] PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

    avg_psnr = sum(psnr_vals) / len(psnr_vals)
    avg_ssim = sum(ssim_vals) / len(ssim_vals)
    print(f"\nResults — PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
    

# ── Main ──────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(MODEL_PATH, device)

    if MODE == "single":
        run_single(model, device)
    elif MODE == "folder":
        run_folder(model, device)
    elif MODE == "dataset":
        run_dataset(model, device)
    else:
        raise ValueError(f"Unknown mode: {MODE}")


if __name__ == "__main__":
    main()