"""Phase-preserving RADIO-conditioned metric encoder for V6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value + self.net(value))


@dataclass(frozen=True)
class V6MetricEncoderConfig:
    radio_dim: int = 1280
    hidden_dim: int = 96
    output_dim: int = 64


class V6MetricEncoder(nn.Module):
    """Low-level spatial phase conditioned by RADIO-final context."""

    def __init__(
        self, config: V6MetricEncoderConfig = V6MetricEncoderConfig()
    ) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, hidden // 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, hidden // 2),
            nn.GELU(),
            ResidualBlock(hidden // 2),
            nn.Conv2d(
                hidden // 2, hidden, 3, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )
        self.radio_context = nn.Sequential(
            nn.Conv2d(int(config.radio_dim), hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            ResidualBlock(hidden),
        )
        self.film = nn.Conv2d(hidden, hidden * 2, 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )
        self.descriptor_head = nn.Conv2d(hidden, int(config.output_dim), 1)
        self.matchability_head = nn.Conv2d(hidden, 1, 1)
        self.null_head = nn.Conv2d(hidden, 1, 1)
        self.log_variance_head = nn.Conv2d(hidden, 1, 1)

    def forward(
        self, radio_final: torch.Tensor, rgb: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if radio_final.ndim != 4 or radio_final.shape[1] != int(
            self.config.radio_dim
        ):
            raise ValueError("radio_final must have shape (B,radio_dim,H/16,W/16)")
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape (B,3,H,W)")
        phase = self.rgb_stem(rgb.float())
        context = self.radio_context(radio_final.float())
        context = F.interpolate(
            context, size=phase.shape[-2:], mode="bilinear", align_corners=False
        )
        gamma, beta = torch.chunk(self.film(context), 2, dim=1)
        conditioned_phase = phase * (1.0 + torch.tanh(gamma)) + beta
        fused = self.fusion(torch.cat([conditioned_phase, context], dim=1))
        fine = F.normalize(self.descriptor_head(fused), dim=1)
        matchability_logits = self.matchability_head(fused)
        null_logits = self.null_head(fused)
        middle = F.normalize(
            F.avg_pool2d(fine, kernel_size=2, stride=2), dim=1
        )
        coarse = F.normalize(
            F.avg_pool2d(fine, kernel_size=4, stride=4), dim=1
        )
        return {
            "fine": fine,
            "middle": middle,
            "coarse": coarse,
            "matchability_logits": matchability_logits,
            "matchability": torch.sigmoid(matchability_logits),
            "null_logits": null_logits,
            "null_probability": torch.sigmoid(null_logits),
            "log_variance": torch.clamp(
                self.log_variance_head(fused), min=-6.0, max=6.0
            ),
        }


def probabilistic_metric_losses(
    output: Mapping[str, torch.Tensor],
    *,
    match_labels: torch.Tensor,
    displacement_error: torch.Tensor,
    displacement_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Supervise matchability, explicit null and calibrated uncertainty.

    ``match_labels`` must include both positives and negatives (wrong maplet,
    occluded, out-of-view or textureless).  ``displacement_error`` is squared
    EPE for visible matches and supplies heteroscedastic calibration.
    """

    labels = match_labels.float()
    if labels.shape != output["matchability_logits"].shape:
        raise ValueError("match_labels must match probability-head shape")
    valid = displacement_valid.bool()
    if displacement_error.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("displacement supervision shapes differ")
    matchability_loss = F.binary_cross_entropy_with_logits(
        output["matchability_logits"], labels
    )
    null_loss = F.binary_cross_entropy_with_logits(
        output["null_logits"], 1.0 - labels
    )
    log_variance = output["log_variance"]
    nll = 0.5 * (
        displacement_error.float() * torch.exp(-log_variance) + log_variance
    )
    uncertainty_loss = (
        nll[valid].mean() if torch.any(valid) else nll.sum() * 0.0
    )
    total = matchability_loss + null_loss + uncertainty_loss
    return {
        "total": total,
        "matchability": matchability_loss,
        "null": null_loss,
        "uncertainty": uncertainty_loss,
    }


def save_v6_metric_encoder(
    path: Path, model: V6MetricEncoder, metadata: Mapping[str, object]
) -> None:
    contract = dict(metadata)
    forbidden = (
        "uses_radio_intermediate",
        "uses_sfm_points",
        "uses_sfm_tracks",
        "uses_alike_descriptors",
        "uses_pairwise_image_matching",
    )
    if any(bool(contract.get(key, False)) for key in forbidden):
        raise ValueError("invalid V6 metric-encoder contract")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_type": "v6_radio_conditioned_metric_encoder",
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metadata": contract,
        },
        Path(path),
    )


def load_v6_metric_encoder(
    path: Path, *, device: str = "cpu"
) -> Tuple[V6MetricEncoder, Mapping[str, object]]:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("artifact_type") != "v6_radio_conditioned_metric_encoder":
        raise ValueError("not a V6 metric-encoder checkpoint")
    config = V6MetricEncoderConfig(**dict(payload["config"]))
    model = V6MetricEncoder(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata", {}))
