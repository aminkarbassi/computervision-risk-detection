import torch.nn as nn
from torchvision import models


def build_classifier(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    """
    Builds a ResNet18-based classifier.

    Args:
        num_classes: number of output classes
        pretrained: whether to use ImageNet weights
        freeze_backbone: whether to freeze convolutional layers
    """

    model = models.resnet18(pretrained=pretrained)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

