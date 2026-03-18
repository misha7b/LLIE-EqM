# src/train.py

import os

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from src.unet.unet_model import UNet
#from src.nafnet.nafnet_arch import NAFNet
#from src.ddpm_unet import DDPMUNet
#from src.tiny_unet.unet import TinyUNet
#from src.ClaudesArchitecture import eqmnet_small, eqmnet_large
from src.eqmnet import eqmnet2_small

from src.loss import EqMLoss
from src.eval import evaluate_loader
from datasets.lol_dataset import LOLPairedDataset


# ── Config ────────────────────────────────────────────────────────────
BATCH_SIZE = 8
EPOCHS = 200
LR = 2e-4
LR_MIN = 1e-6
CHECKPOINT_DIR = "checkpoints"
DATA_ROOT = "datasets/LOL"

VAL_EVERY = 1
SAVE_EVERY = 50
MAX_VAL = 15  # None = evaluate all test images
# ──────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):

    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for i, (x_light, x_dark) in enumerate(pbar):
        x_light, x_dark = x_light.to(device), x_dark.to(device)

        optimizer.zero_grad()
        loss = criterion(model, x_light, x_dark)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=running_loss / (i + 1))

    return running_loss / len(loader)

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    

    # Data
    #train_dataset = LOLPairedDataset(root_dir=DATA_ROOT, split="train", patch_size = 256)
    
    '''
    train_dataset = ConcatDataset([
        LOLPairedDataset("datasets/Endovis17", split="train", patch_size=256),
        LOLPairedDataset("datasets/Endovis18", split="train", patch_size=256),
    ])
    '''
    
  
    V2_SPLITS = {"train": "Train", "test": "Test"}
    train_dataset = ConcatDataset([
    LOLPairedDataset("datasets/LOL", split="train", patch_size=256),
    LOLPairedDataset("datasets/Lol-v2/Real_captured", split="train", patch_size=256,
                     splits=V2_SPLITS, low_folder="Input", high_folder="GT"),
    LOLPairedDataset("datasets/Lol-v2/Synthetic", split="train", patch_size=256,
                     splits=V2_SPLITS, low_folder="Input", high_folder="GT"),
    ])

    
    
    test_dataset = LOLPairedDataset(root_dir=DATA_ROOT, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    # Model
    #model = UNet(n_channels=6, n_classes=3).to(device) 
    #model = NAFNet(img_channel=6, out_channel=3, width=32, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]).to(device)
    #model = DDPMUNet(image_channels=6, out_channels=3, n_channels=64).to(device)
    #model = TinyUNet(in_ch=6, out_ch=3, base=16).to(device)
    #model = eqmnet_small().to(device)
    #model = eqmnet_large().to(device)
    model = eqmnet2_small().to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)
    #criterion = EqMLoss(variant="truncated", loss_type="charbonnier")
    criterion = EqMLoss(variant="truncated", loss_type="charbonnier",
                      lambda_fft=0.2, lambda_cos=0.3)

    best_psnr = -float("inf")
    
    # Train
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        scheduler.step()
        
        if epoch % VAL_EVERY == 0:
            val_psnr, val_ssim, val_lpips = evaluate_loader(model, test_loader, device, max_eval=MAX_VAL)

            print(
                f"Epoch {epoch}/{EPOCHS} — "
                f"loss: {avg_loss:.6f} | "
                f"PSNR: {val_psnr:.2f} dB | "
                f"SSIM: {val_ssim:.4f} | "
                f"LPIPS: {val_lpips:.4f}"
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_path = os.path.join(CHECKPOINT_DIR, "claudenet_full_arch_best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": avg_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "val_lpips": val_lpips,
                }, best_path)
                print(f"Best checkpoint saved to {best_path}")
        else:
            print(f"Epoch {epoch}/{EPOCHS} — loss: {avg_loss:.6f}")

        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"claudenet_full_arch_ep{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_loss,
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
        

    # Save
    model_path = os.path.join(CHECKPOINT_DIR, "claudenet_full_arch.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved to {model_path}")


if __name__ == "__main__":
    main()
