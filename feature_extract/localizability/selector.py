"""Feature selector for POFD-FS hypothesis-localizability experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SelectorConfig:
    in_channels: int
    out_channels: int = 64
    group_size: int = 8
    hidden_dim: int = 128
    spatial_utility: bool = True
    uncertainty: bool = True
    l2_normalize: bool = True
    identity_init: bool = True


class LocalizationFeatureSelector(nn.Module):
    """Select a compact localization-usable subspace from frozen dense features.

    The module is intentionally small: a grouped channel gate controls which
    foundation-feature groups remain active, a 1x1 projection creates the
    compact localization feature, and lightweight heads expose spatial utility
    and uncertainty maps for ranking/calibration losses.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        *,
        group_size: int = 8,
        hidden_dim: int = 128,
        spatial_utility: bool = True,
        uncertainty: bool = True,
        l2_normalize: bool = True,
        identity_init: bool = True,
    ):
        super().__init__()
        self.config = SelectorConfig(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            group_size=int(group_size),
            hidden_dim=int(hidden_dim),
            spatial_utility=bool(spatial_utility),
            uncertainty=bool(uncertainty),
            l2_normalize=bool(l2_normalize),
            identity_init=bool(identity_init),
        )
        if self.config.in_channels <= 0 or self.config.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if self.config.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.config.in_channels % self.config.group_size != 0:
            raise ValueError("in_channels must be divisible by group_size")
        self.num_groups = self.config.in_channels // self.config.group_size

        gate_hidden = max(8, min(self.config.hidden_dim, self.num_groups * 4))
        self.channel_gate = nn.Sequential(
            nn.Linear(self.num_groups, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.num_groups),
        )
        self.proj = nn.Conv2d(self.config.in_channels, self.config.out_channels, kernel_size=1)
        self.utility_head = (
            nn.Sequential(
                nn.Conv2d(self.config.out_channels, max(8, self.config.out_channels // 2), kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(max(8, self.config.out_channels // 2), 1, kernel_size=1),
            )
            if self.config.spatial_utility
            else None
        )
        self.uncertainty_head = (
            nn.Sequential(
                nn.Conv2d(self.config.out_channels, max(8, self.config.out_channels // 2), kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(max(8, self.config.out_channels // 2), 1, kernel_size=1),
            )
            if self.config.uncertainty
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.config.identity_init and self.config.in_channels == self.config.out_channels:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            with torch.no_grad():
                diag = torch.arange(self.config.in_channels)
                self.proj.weight[diag, diag, 0, 0] = 1.0
        if self.config.identity_init:
            last_gate = self.channel_gate[-1]
            nn.init.zeros_(last_gate.weight)
            nn.init.constant_(last_gate.bias, 2.0)
            if self.utility_head is not None:
                last_utility = self.utility_head[-1]
                nn.init.zeros_(last_utility.weight)
                nn.init.constant_(last_utility.bias, 2.0)
            if self.uncertainty_head is not None:
                last_uncertainty = self.uncertainty_head[-1]
                nn.init.zeros_(last_uncertainty.weight)
                nn.init.zeros_(last_uncertainty.bias)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        if feature.ndim != 4:
            raise ValueError(f"feature must have shape (B,C,H,W), got {tuple(feature.shape)}")
        if feature.shape[1] != self.config.in_channels:
            raise ValueError(f"expected {self.config.in_channels} channels, got {feature.shape[1]}")

        bsz, channels, height, width = feature.shape
        grouped = feature.float().reshape(bsz, self.num_groups, self.config.group_size, height, width)
        group_descriptor = grouped.abs().mean(dim=(2, 3, 4))
        channel_gate = torch.sigmoid(self.channel_gate(group_descriptor))
        expanded_gate = channel_gate.repeat_interleave(self.config.group_size, dim=1).view(bsz, channels, 1, 1)
        gated_feature = feature.float() * expanded_gate.to(dtype=feature.dtype)
        z = self.proj(gated_feature.float())
        if self.config.l2_normalize:
            z = F.normalize(z, dim=1, eps=1.0e-6)

        if self.utility_head is None:
            utility = z.new_ones((bsz, 1, height, width))
        else:
            utility = torch.sigmoid(self.utility_head(z.float()))
        if self.uncertainty_head is None:
            uncertainty = z.new_ones((bsz, 1, height, width))
        else:
            uncertainty = F.softplus(self.uncertainty_head(z.float())) + 1.0e-6

        return {
            "z": z,
            "utility": utility,
            "uncertainty": uncertainty,
            "channel_gate": channel_gate,
            "gated_feature": gated_feature,
        }
