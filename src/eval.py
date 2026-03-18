# src/eval.py

import os
import glob
import time
import math

import torch
from torchvision import transforms as T
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F
from PIL import Image

from src.unet.unet_model import UNet
from src.nafnet.nafnet_arch import NAFNet
from src.ddpm_unet import DDPMUNet
from src.tiny_unet.unet import TinyUNet
from src.ClaudesArchitecture import eqmnet_small, eqmnet_large
from src.eqmnet import eqmnet2_small

from datasets.lol_dataset import LOLPairedDataset


# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH = "checkpoints/claudenet_endo2_arch_best.pt"
SAVE_DIR = "outputs/custom_dark"

# Generation
NUM_STEPS = 12 #12
MOMENTUM = 0.9
ETA = 2e-2   #2e-2
METHOD = "gd"   # "gd" "heavy_ball" "nesterov" "big_little"
STOP_EPS = None


MODE = "dataset"  #"single" "folder" "dataset"
IMG_PATH = "datasets/custom/pool.png"       # used by "single"
IMG_DIR = "datasets/dark"              # used by "folder"
DATA_ROOT = "datasets/LOL"                  # used by "dataset"
#DATA_ROOT = "datasets/LOL"
#DATA_ROOT = "datasets/Lol-v2/Real_captured"

SAVE_STEPS = False
SAVE_COMPARISON = True
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_gd(model, x_dark, num_steps=NUM_STEPS, eta=ETA, save_steps=False,
                stop_eps=STOP_EPS):
    """Gradient Descent"""
    y = x_dark.clone()
    x_k = x_dark.clone()
    history = [x_k.cpu().clone()] if save_steps else None

    for step in range(num_steps):

        pred_grad = model(torch.cat([x_k, y], dim=1))

        # DDPM
        #gamma_val = torch.full((x_k.shape[0],), step / num_steps, device=x_k.device)
        #pred_grad = model(torch.cat([x_k, y], dim=1), gamma_val)

        x_new = (x_k - eta * pred_grad).clamp(0.0, 1.0)

        if stop_eps is not None:
            rel_step = (x_new - x_k).norm() / (x_k.norm() + 1e-8)
            if rel_step < stop_eps:
                x_k = x_new
                if save_steps:
                    history.append(x_k.cpu().clone())
                print(f"  GD early stop at step {step} (rel_step={rel_step:.2e})")
                break

        x_k = x_new

        if save_steps:
            history.append(x_k.cpu().clone())

        if step > num_steps // 2:
            eta *= 0.99

    return x_k, history

@torch.no_grad()
def generate_heavy_ball(model, x_dark, num_steps=NUM_STEPS, eta=ETA,
                        momentum=MOMENTUM, save_steps=False, stop_eps=None):
    """Heavy Ball"""
    y = x_dark.clone()
    x_k = x_dark.clone()
    v = torch.zeros_like(x_k)
    history = [x_k.cpu().clone()] if save_steps else None

    for step in range(num_steps):
        pred_grad = model(torch.cat([x_k, y], dim=1))
        v = momentum * v - eta * pred_grad
        x_k = (x_k + v).clamp(0.0, 1.0)

        if save_steps:
            history.append(x_k.cpu().clone())

    return x_k, history


@torch.no_grad()
def generate_nesterov(model, x_dark, num_steps=NUM_STEPS, eta=ETA,
                      momentum=MOMENTUM, save_steps=False, stop_eps=None):
    """Nesterov accelerated gradient"""
    y = x_dark.clone()
    x_k = x_dark.clone()
    v = torch.zeros_like(x_k)
    history = [x_k.cpu().clone()] if save_steps else None

    for step in range(num_steps):
        # Look-ahead position
        x_lookahead = (x_k + momentum * v).clamp(0.0, 1.0)
        pred_grad = model(torch.cat([x_lookahead, y], dim=1))

        v = momentum * v - eta * pred_grad
        x_k = (x_k + v).clamp(0.0, 1.0)

        if save_steps:
            history.append(x_k.cpu().clone())

    return x_k, history


@torch.no_grad()
def generate_big_little(model, x_dark, num_steps=NUM_STEPS, eta=ETA,
                        save_steps=False, stop_eps=None, ratio=0.1):
    """Big Step Little Step: alternates between a large and small learning rate."""
    y = x_dark.clone()
    x_k = x_dark.clone()
    history = [x_k.cpu().clone()] if save_steps else None

    for step in range(num_steps):
        lr = eta if step % 2 == 0 else eta * ratio

        pred_grad = model(torch.cat([x_k, y], dim=1))
        x_k = (x_k - lr * pred_grad).clamp(0.0, 1.0)

        if save_steps:
            history.append(x_k.cpu().clone())

    return x_k, history



@torch.no_grad()
def generate_gd_cosine(model, x_dark, num_steps=NUM_STEPS, eta=ETA,
                       save_steps=False, stop_eps=STOP_EPS):
    """Gradient Descent with cosine eta schedule — large steps early, small steps late."""
    y = x_dark.clone()
    x_k = x_dark.clone()
    history = [x_k.cpu().clone()] if save_steps else None

    for step in range(num_steps):
        eta_k = eta * 0.5 * (1 + math.cos(math.pi * step / num_steps))

        pred_grad = model(torch.cat([x_k, y], dim=1))
        x_new = (x_k - eta_k * pred_grad).clamp(0.0, 1.0)

        if stop_eps is not None:
            rel_step = (x_new - x_k).norm() / (x_k.norm() + 1e-8)
            if rel_step < stop_eps:
                x_k = x_new
                if save_steps:
                    history.append(x_k.cpu().clone())
                print(f"  GD-cosine early stop at step {step} (rel_step={rel_step:.2e})")
                break

        x_k = x_new

        if save_steps:
            history.append(x_k.cpu().clone())

    return x_k, history


GENERATION_METHODS = {
    "gd": generate_gd,
    "gd_cosine": generate_gd_cosine,
    "heavy_ball": generate_heavy_ball,
    "nesterov": generate_nesterov,
    "big_little": generate_big_little,
}


def generate(model, x_dark, save_steps=False):
    """Dispatch to the configured generation method."""
    fn = GENERATION_METHODS[METHOD]
    kwargs = dict(num_steps=NUM_STEPS, eta=ETA, save_steps=save_steps, stop_eps=STOP_EPS)
    if METHOD in ("heavy_ball", "nesterov"):
        kwargs["momentum"] = MOMENTUM
    return fn(model, x_dark, **kwargs)





def load_model(model_path, device):
    
    
    
    #model = UNet(n_channels=6, n_classes=3).to(device)
    #model = NAFNet(img_channel=6, out_channel=3, width=32, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]).to(device)
    #model = DDPMUNet(image_channels=6, out_channels=3, n_channels=64).to(device)
    #model = TinyUNet(in_ch=6, out_ch=3, base=16).to(device)
    #model = eqmnet_small().to(device)
    model = eqmnet2_small().to(device)
    
    obj = torch.load(model_path, map_location=device)
    
    if "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        print(f"Loaded checkpoint from {model_path} (epoch={obj.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(obj)
        print(f"Loaded weights from {model_path}")
    
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
    
    # dehaze
    #x_dark = 1.0 - x_dark

    # Warmup pass (excludes CUDA kernel launch overhead from timing)
    _ = model(torch.cat([x_dark, x_dark], dim=1))
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()

    result, history = generate(model, x_dark, save_steps=SAVE_STEPS)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000
    print(f"Inference time: {elapsed_ms:.1f} ms ({NUM_STEPS} steps, {elapsed_ms / NUM_STEPS:.1f} ms/step)")
    
    #result = 1.0 - result
    #if history is not None:
    #    history = [1.0 - h for h in history]
    
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
    """Evaluate on LOL with PSNR, SSIM, and LPIPS metrics."""
    test_dataset = LOLPairedDataset(root_dir=DATA_ROOT, split="test")
    
    #V2_SPLITS = {"train": "Train", "test": "Test"}
    #test_dataset = LOLPairedDataset(root_dir=DATA_ROOT, split="test",
    #                               splits=V2_SPLITS, low_folder="Input", high_folder="GT")
    
    print(f"Evaluating on {len(test_dataset)} test pairs")

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    to_pil = T.ToPILImage()
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for idx in range(len(test_dataset)):
        x_light, x_dark = test_dataset[idx]
        x_light = x_light.unsqueeze(0).to(device)
        x_dark = x_dark.unsqueeze(0).to(device)

        result, history = generate(model, x_dark, save_steps=SAVE_STEPS)

        psnr_val = psnr_fn(result, x_light).item()
        ssim_val = ssim_fn(result, x_light).item()
        lpips_val = lpips_fn(result, x_light).item()
        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
        lpips_vals.append(lpips_val)

        # Save outputs
        name = f"{idx:03d}"
        x_dark_pil = to_pil(x_dark.squeeze(0).cpu())
        gt_pil = to_pil(x_light.squeeze(0).cpu())
        save_outputs(SAVE_DIR, name, x_dark_pil, result, history, gt_pil=gt_pil)

        print(f"  [{idx+1}/{len(test_dataset)}] PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f}")

    avg_psnr = sum(psnr_vals) / len(psnr_vals)
    avg_ssim = sum(ssim_vals) / len(ssim_vals)
    avg_lpips = sum(lpips_vals) / len(lpips_vals)
    print(f"\nResults — PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
    

@torch.no_grad()
def evaluate_loader(model, loader, device, max_eval=None):
    model.eval()

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for i, (x_light, x_dark) in enumerate(loader):
        if max_eval and i >= max_eval:
            break
        x_light = x_light.to(device)
        x_dark = x_dark.to(device)

        result, _ = generate(model, x_dark, save_steps=False)

        psnr_vals.append(psnr_fn(result, x_light).item())
        ssim_vals.append(ssim_fn(result, x_light).item())
        lpips_vals.append(lpips_fn(result, x_light).item())

    return (sum(psnr_vals) / len(psnr_vals),
            sum(ssim_vals) / len(ssim_vals),
            sum(lpips_vals) / len(lpips_vals))

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
