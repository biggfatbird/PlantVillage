"""ImageNet-initialized classifiers used as baselines."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_NAMES = ("resnet50", "efficientnet_b0")


def build_model(name: str, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
