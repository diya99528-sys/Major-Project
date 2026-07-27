"""
Melanoma Detection Model
-------------------------
Transfer-learning based CNN for binary classification of skin lesion images:
benign vs malignant (melanoma).

Architecture: ResNet18 backbone (ImageNet pretrained) with a custom
classification head. Transfer learning is justified because dermoscopic
image datasets (even the full ISIC set, ~25k-70k images) are small relative
to what's needed to train a deep CNN from scratch, and low-level visual
features (edges, textures, color gradients) learned from ImageNet transfer
well to lesion images.
"""

import torch
import torch.nn as nn
from torchvision import models


class MelanomaClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # benign, malignant
        )

    def forward(self, x):
        return self.backbone(x)


def build_model(pretrained=True, freeze_backbone=False):
    return MelanomaClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(1, 3, 128, 128)
    out = model(dummy)
    print("Output shape:", out.shape)  # (1, 2)
