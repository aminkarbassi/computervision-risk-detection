import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import mlflow

from src.data.detection_dataset import DetectionDataset
from src.training.evaluation import match_predictions


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int):
    """
    Build Faster R-CNN with a custom detection head.
    """
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


@torch.no_grad()
def evaluate(model, dataloader, device, iou_threshold=0.5):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"].cpu()
            pred_scores = output["scores"].cpu()
            gt_boxes = target["boxes"].cpu()

            tp, fp, fn = match_predictions(
                pred_boxes,
                pred_scores,
                gt_boxes,
                iou_threshold=iou_threshold,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)

    return precision, recall


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

    mlflow.set_experiment("cv-risk-week5-detection")

    with mlflow.start_run():
        mlflow.log_params({
            "model": "fasterrcnn_resnet50_fpn",
            "num_classes": 1,
            "tile_size": 512,
            "iou_threshold": 0.5,
            "batch_size": 2,
            "optimizer": "SGD",
            "learning_rate": 0.005,
        })

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

            precision, recall = evaluate(
                model, val_loader, device
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_precision": precision,
                "val_recall": recall,
            }, step=epoch)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f}"
            )

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
        )

        torch.save(
            model.state_dict(),
            "outputs/models/detection_model.pt",
        )


if __name__ == "__main__":
    main()
