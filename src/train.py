# src/train.py

import os

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from src.unet.unet_model import UNet
from src.nafnet.nafnet_arch import NAFNet
from src.loss import EqMLoss
from datasets.lol_dataset import LOLPairedDataset

# ── Config ────────────────────────────────────────────────────────────
BATCH_SIZE = 8
EPOCHS = 400
LR = 2e-4
LR_MIN = 1e-6
CHECKPOINT_DIR = "checkpoints"
DATA_ROOT = "datasets/LOL"
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
    
    V2_SPLITS = {"train": "Train", "test": "Test"}

    # Data
    #train_dataset = LOLPairedDataset(root_dir=DATA_ROOT, split="train", patch_size = None)
    
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
    model = UNet(n_channels=6, n_classes=3).to(device) 
    #model = NAFNet(img_channel=6, out_channel=3, width=32, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]).to(device)
    
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)
    criterion = EqMLoss(variant="truncated", loss_type="charbonnier")

    # Train
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        
        print(f"Epoch {epoch}/{EPOCHS} — loss: {avg_loss:.6f}")
        if epoch % 50 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"unet_ep{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
        

    # Save
    model_path = os.path.join(CHECKPOINT_DIR, "unet_400.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved to {model_path}")


if __name__ == "__main__":
    main()