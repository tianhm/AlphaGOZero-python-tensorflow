"""AlphaGo Zero network: pre-activation ResNet tower + policy & value heads.

Input:  (B, n_planes, n, n)  float32 in [0, 1]
Output: (policy_logits: (B, n_actions), value: (B, 1) in [-1, 1])
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AGZNet(nn.Module):
    """Pre-activation ResNet with separate policy and value heads."""

    def __init__(
        self,
        n_planes: int,
        n_actions: int,
        n_resid_blocks: int = 19,
        n_filters: int = 256,
        board_size: int = 19,
    ) -> None:
        super().__init__()
        self.n_planes = n_planes
        self.n_actions = n_actions
        self.board_size = board_size
        n = board_size

        # Initial convolution.
        self.conv_input = nn.Conv2d(n_planes, n_filters, kernel_size=3, padding=1, bias=False)

        # Residual tower.
        self.residuals = nn.ModuleList(
            [self._make_block(n_filters) for _ in range(n_resid_blocks)]
        )

        # Policy head.
        self.policy_conv = nn.Conv2d(n_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * n * n, n_actions)

        # Value head.
        self.value_conv = nn.Conv2d(n_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(n * n, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def _make_block(self, n_filters: int) -> nn.Module:
        from .resnet import PreActResidualBlock
        return PreActResidualBlock(n_filters)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Tower.
        out = self.conv_input(x)
        for block in self.residuals:
            out = block(out)

        # Policy head.
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = torch.relu(p)
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head.
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = torch.relu(v)
        v = v.reshape(v.size(0), -1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
