"""RADIO-final projection trained for exact within-maplet surface location."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SurfaceSpatialProjectionConfig:
    input_dim: int = 1280
    output_dim: int = 128


class SurfaceSpatialProjection(nn.Module):
    """Origin-preserving linear subspace; no maplet-identity collapse head."""

    def __init__(
        self,
        config: SurfaceSpatialProjectionConfig = SurfaceSpatialProjectionConfig(),
        *,
        initial_projection: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        projection = torch.empty(
            int(config.output_dim), int(config.input_dim), dtype=torch.float32
        )
        if initial_projection is None:
            nn.init.orthogonal_(projection)
        else:
            initial = np.asarray(initial_projection, dtype=np.float32)
            if initial.shape != tuple(projection.shape):
                raise ValueError("initial spatial projection shape differs")
            projection.copy_(torch.from_numpy(initial))
        self.projection = nn.Parameter(projection)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.shape[-1] != int(self.config.input_dim):
            raise ValueError("RADIO descriptors have the wrong channel count")
        normalized = F.normalize(descriptors.float(), dim=-1)
        return F.normalize(
            F.linear(normalized, self.projection), dim=-1
        )


def save_surface_spatial_projection(
    path: Path,
    model: SurfaceSpatialProjection,
    metadata: Mapping[str, object],
) -> None:
    contract = dict(metadata)
    for key in (
        "stores_mapping_rgb",
        "stores_mapping_image_paths",
        "stores_mapping_image_ids",
        "uses_alike_descriptors",
        "uses_radio_intermediate",
        "uses_sfm_points",
        "uses_sfm_tracks",
        "uses_pairwise_image_matching",
    ):
        if bool(contract.get(key, False)):
            raise ValueError(f"surface spatial projection violates {key}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_type": "v6_exact_surface_spatial_projection",
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metadata": contract,
        },
        Path(path),
    )


def load_surface_spatial_projection(
    path: Path, *, device: str = "cpu"
) -> tuple[SurfaceSpatialProjection, Mapping[str, object]]:
    payload = torch.load(Path(path), map_location="cpu")
    if payload.get("artifact_type") != "v6_exact_surface_spatial_projection":
        raise ValueError("not a V6 surface spatial projection")
    model = SurfaceSpatialProjection(
        SurfaceSpatialProjectionConfig(**dict(payload["config"]))
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata", {}))
