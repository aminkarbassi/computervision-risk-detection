import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.models.classifier import build_classifier
from src.data.transforms import get_val_transforms


def load_model(model_path: Path, device):
    model = build_classifier(
        num_classes=2,
        pretrained=False,
        freeze_backbone=False,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(image_path: Path, model, device):
    transform = get_val_transforms(image_size=224)

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label_map = {0: "not_roof", 1: "roof"}

    return label_map[pred.item()], confidence.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs/models/best_model.pt"),
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    model = load_model(args.model_path, device)
    label, confidence = predict(args.image_path, model, device)

    print(f"Prediction: {label} (confidence={confidence:.2f})")


if __name__ == "__main__":
    main()

