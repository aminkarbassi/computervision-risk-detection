import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor

from src.data.detection_dataset import DetectionDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    return model


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    return running_loss / len(dataloader)


def main():
    device = torch.device("cpu")

    train_ds = DetectionDataset(
        image_dir="data/detection_tiled/train/images",
        annotation_dir="data/detection_tiled/train/annotations",
        transforms=ToTensor(),
    )

    val_ds = DetectionDataset(
        image_dir="data/detection_tiled/val/images",
        annotation_dir="data/detection_tiled/val/annotations",
        transforms=ToTensor(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=2).to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    epochs = 10

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device
        )

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

    torch.save(
        model.state_dict(),
        "outputs/models/detection_model.pt",
    )


if __name__ == "__main__":
    main()
