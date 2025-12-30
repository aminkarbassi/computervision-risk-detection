from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class RoofDataset(Dataset):
    """
    Binary classification dataset:
    - roof -> 1
    - not_roof -> 0
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        class_to_label = {
            "roof": 1,
            "not_roof": 0,
        }

        for class_name, label in class_to_label.items():
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                raise ValueError(f"Missing directory: {class_dir}")

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    samples.append((img_path, label))

        if len(samples) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

