"""Interpretable localizable feature selector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SelectorOutput:
    selected: torch.Tensor
    utility: torch.Tensor
    uncertainty: torch.Tensor
    channel_gates: torch.Tensor


class LocalizableFeatureSelector(nn.Module):
    """Group-gated 1x1 selector for raw VFM dense tokens."""

    def __init__(self, input_dim: int, output_dim: int = 64, group_size: int = 16):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or group_size <= 0:
            raise ValueError("input_dim, output_dim, and group_size must be positive")
        if input_dim % group_size != 0:
            raise ValueError("input_dim must be divisible by group_size")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.group_size = group_size
        self.group_count = input_dim // group_size
        self.group_logits = nn.Parameter(torch.zeros(self.group_count))
        self.projection = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
        self.utility_head = nn.Conv2d(output_dim, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(output_dim, 1, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> SelectorOutput:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape (B, C, H, W)")
        if tokens.shape[1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} channels, got {tokens.shape[1]}")
        group_gates = torch.sigmoid(self.group_logits)
        channel_gates = group_gates.repeat_interleave(self.group_size)
        gated = tokens * channel_gates.view(1, -1, 1, 1)
        selected = self.projection(gated)
        selected = F.normalize(selected, p=2, dim=1, eps=1e-6)
        utility = torch.sigmoid(self.utility_head(selected))
        uncertainty = F.softplus(self.uncertainty_head(selected))
        return SelectorOutput(
            selected=selected,
            utility=utility,
            uncertainty=uncertainty,
            channel_gates=group_gates,
        )
