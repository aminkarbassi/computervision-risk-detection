import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image


class DetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transforms = transforms

        self.images = sorted(self.image_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        anno_path = self.annotation_dir / f"{img_path.stem}.json"

        image = Image.open(img_path).convert("RGB")

        with open(anno_path) as f:
            anno = json.load(f)

        boxes = torch.tensor(anno["boxes"], dtype=torch.float32)
        labels = torch.tensor(anno["labels"], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
