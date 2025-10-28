from typing import Tuple

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """Convolutional network tailored for butterfly classification."""

    def __init__(self, num_classes: int, input_size: Tuple[int, int]):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flattened_features = self._infer_flattened_dim(input_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def _infer_flattened_dim(self, input_size: Tuple[int, int]) -> int:
        with torch.no_grad():
            sample = torch.zeros(1, 3, *input_size)
            features = self.features(sample)
            return int(features.numel() / features.size(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int, input_size: Tuple[int, int]) -> CustomCNN:
    return CustomCNN(num_classes=num_classes, input_size=input_size)
