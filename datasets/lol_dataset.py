# datasets/paired_dataset.py
"""LOL (Low-Light) paired dataset loader."""

import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

SPLITS = {
    "train": "our485",
    "test": "eval15",
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _list_images(directory):
    """Return sorted list of image filenames in a directory."""
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS))


class LOLPairedDataset(Dataset):
    """Paired low-light / normal-light dataset from the LOL benchmark."""

    def __init__(self, root_dir="datasets/LOL", split="train"):
        super().__init__()
        base = os.path.join(root_dir, SPLITS[split])
        self.low_dir = os.path.join(base, "low")
        self.high_dir = os.path.join(base, "high")
        self.low_images = _list_images(self.low_dir)
        self.high_images = _list_images(self.high_dir)
        assert len(self.low_images) == len(self.high_images), (
            f"Mismatched pairs: {len(self.low_images)} low vs {len(self.high_images)} high"
        )
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low = Image.open(os.path.join(self.low_dir, self.low_images[idx])).convert("RGB")
        high = Image.open(os.path.join(self.high_dir, self.high_images[idx])).convert("RGB")
        return self.transform(high), self.transform(low)