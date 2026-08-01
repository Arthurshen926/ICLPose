"""Query-side RADIO-final adapter trained on complete chart transforms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StructuredFrameAdapterConfig:
    feature_dim: int = 128

    def __post_init__(self) -> None:
        if int(self.feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")


class StructuredFrameAdapter(nn.Module):
    """Near-identity metric adapter; the map atlas remains fixed."""

    def __init__(
        self,
        config: StructuredFrameAdapterConfig = (
            StructuredFrameAdapterConfig()
        ),
    ) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Conv2d(
            int(config.feature_dim),
            int(config.feature_dim),
            kernel_size=1,
            bias=False,
        )
        with torch.no_grad():
            self.projection.weight.zero_()
            identity = torch.eye(int(config.feature_dim))
            self.projection.weight[:, :, 0, 0].copy_(identity)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 3:
            feature = feature[None]
        if (
            feature.ndim != 4
            or int(feature.shape[1]) != int(self.config.feature_dim)
        ):
            raise ValueError("feature must have shape (B,C,H,W)")
        return F.normalize(self.projection(feature), dim=1, eps=1e-6)


def save_structured_frame_adapter(
    path: Path,
    model: StructuredFrameAdapter,
    metadata: Mapping[str, object],
) -> None:
    payload = {
        "artifact_type": "v6_structured_frame_adapter",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(path))


def load_structured_frame_adapter(
    path: Path,
    *,
    device: str = "cpu",
) -> tuple[StructuredFrameAdapter, Mapping[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("artifact_type") != "v6_structured_frame_adapter":
        raise ValueError("not a V6 structured-frame adapter")
    model = StructuredFrameAdapter(
        StructuredFrameAdapterConfig(**dict(payload["config"]))
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    metadata = dict(payload.get("metadata") or {})
    forbidden = (
        "stores_mapping_rgb",
        "stores_mapping_image_ids",
        "stores_mapping_image_paths",
        "uses_sfm_points",
        "uses_sfm_tracks",
        "uses_alike_descriptors",
        "uses_radio_intermediate",
        "uses_pairwise_image_matching",
        "uses_point_correspondence_pnp",
    )
    if any(bool(metadata.get(key, False)) for key in forbidden):
        raise ValueError("structured-frame adapter violates map contract")
    return model, metadata
