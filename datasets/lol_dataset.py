# datasets/lol_dataset.py
"""Paired low-light dataset loader supporting LOL, LOL-v2"""

import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _list_images(directory):
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS))


class LOLPairedDataset(Dataset):
    """Paired low-light / normal-light dataset.

    During training: random crop, horizontal flip, vertical flip, random 90° rotation.
    During test: just convert to tensor.

  
    # LOL (our485/low, our485/high)
    LOLPairedDataset("datasets/LOL", split="train")

    # LOL-v2 Real (Train/Low, Train/Normal)
    LOLPairedDataset("datasets/Lol-v2/Real_captured", split="train",
                        splits={"train": "Train", "test": "Test"},
                        low_folder="Low", high_folder="Normal")
    """

    def __init__(self, root_dir="datasets/LOL", split="train", patch_size=256,
                 splits=None, low_folder="low", high_folder="high"):
        super().__init__()
        if splits is None:
            splits = {"train": "our485", "test": "eval15"}

        base = os.path.join(root_dir, splits[split])
        self.low_dir = os.path.join(base, low_folder)
        self.high_dir = os.path.join(base, high_folder)
        self.low_images = _list_images(self.low_dir)
        self.high_images = _list_images(self.high_dir)
        assert len(self.low_images) == len(self.high_images), (
            f"Mismatched pairs: {len(self.low_images)} low vs {len(self.high_images)} high"
        )
        self.split = split
        self.patch_size = patch_size

    def _augment(self, low, high):
        """Apply identical random augmentations to both images."""
        if self.patch_size is not None:
            i, j, h, w = T.RandomCrop.get_params(low, (self.patch_size, self.patch_size))
            low = TF.crop(low, i, j, h, w)
            high = TF.crop(high, i, j, h, w)

        if random.random() > 0.5:
            low = TF.hflip(low)
            high = TF.hflip(high)

        if random.random() > 0.5:
            low = TF.vflip(low)
            high = TF.vflip(high)

        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            low = TF.rotate(low, angle)
            high = TF.rotate(high, angle)

        return low, high

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low = Image.open(os.path.join(self.low_dir, self.low_images[idx])).convert("RGB")
        high = Image.open(os.path.join(self.high_dir, self.high_images[idx])).convert("RGB")

        if self.split == "train":
            low, high = self._augment(low, high)

        return TF.to_tensor(high), TF.to_tensor(low)