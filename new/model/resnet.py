"""ResNet building blocks (full pre-activation variant)."""
from __future__ import annotations

import torch
import torch.nn as nn


class PreActResidualBlock(nn.Module):
    """Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv."""

    def __init__(self, n_filters: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        return x + out
