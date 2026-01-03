import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.data.detection_dataset import DetectionDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    device = torch.device("cpu")

    dataset = DetectionDataset(
        image_dir="data/detection_tiled/train/images",
        annotation_dir="data/detection_tiled/train/annotations",
        transforms=ToTensor(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier head for our task (background + roof)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes=2,
    )
    
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    # ---- SMOKE TEST: ONE BATCH ONLY ----
    images, targets = next(iter(dataloader))
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    print("Smoke test successful. Loss:", losses.item())


if __name__ == "__main__":
    main()
