import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import hashlib
import subprocess

from pathlib import Path

from src.data.dataset import RoofDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.classifier import build_classifier
from src.training.utils import load_config


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def hash_directory(path: str) -> str:
    """
    Creates a lightweight hash of file paths + sizes.
    Enough to detect data changes without hashing full files.
    """
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

def main():
    config = load_config("configs/base.yaml")

    mlflow.set_experiment("cv-risk-week3")

    with mlflow.start_run():

        set_seed(config["training"]["seed"])
        device = torch.device("cpu")
        
        mlflow.log_params({
            "image_size": config["dataset"]["image_size"],
            "batch_size": config["dataset"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "epochs": config["training"]["epochs"],
            "freeze_backbone": config["training"]["freeze_backbone"],
            "pretrained": config["model"]["pretrained"],
        })

        mlflow.log_params({
            "train_data_path": config["paths"]["train_dir"],
            "val_data_path": config["paths"]["val_dir"],
        })

        mlflow.log_params({
            "train_data_hash": hash_directory(config["paths"]["train_dir"]),
            "val_data_hash": hash_directory(config["paths"]["val_dir"]),
            "git_commit": get_git_commit(),
        })

        train_ds = RoofDataset(
            config["paths"]["train_dir"],
            transform=get_train_transforms(config["dataset"]["image_size"]),
        )
        val_ds = RoofDataset(
            config["paths"]["val_dir"],
            transform=get_val_transforms(config["dataset"]["image_size"]),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config["dataset"]["batch_size"],
            shuffle=True,
            num_workers=config["dataset"]["num_workers"],
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=config["dataset"]["batch_size"],
            shuffle=False,
            num_workers=config["dataset"]["num_workers"],
        )

        model = build_classifier(
            num_classes=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
            freeze_backbone=config["training"]["freeze_backbone"],
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
        )

        best_val_acc = 0.0

        for epoch in range(config["training"]["epochs"]):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = eval_one_epoch(
                model, val_loader, criterion, device
            )

            print(
                f"Epoch {epoch + 1}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
            )
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)


            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "outputs/models/best_model.pt")
                mlflow.pytorch.log_model(model, artifact_path="model")


        print("Training complete. Best Val Acc:", best_val_acc)


if __name__ == "__main__":
    main()

