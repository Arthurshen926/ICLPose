"""Fine metric adapter for detector-sampled RADIO-final surface features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SurfaceMetricFeatureMapperConfig:
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128


class SurfaceMetricFeatureMapper(nn.Module):
    """A pointwise adapter; spatial precision still comes from the detector."""

    def __init__(
        self,
        config: SurfaceMetricFeatureMapperConfig = SurfaceMetricFeatureMapperConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(int(config.input_dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(config.input_dim), int(config.hidden_dim)),
            nn.GELU(),
            nn.Linear(int(config.hidden_dim), int(config.output_dim)),
        )
        self.shortcut = nn.Linear(
            int(config.input_dim), int(config.output_dim), bias=False
        )
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if int(features.shape[-1]) != int(self.config.input_dim):
            raise ValueError("metric mapper input feature dimension differs")
        output = self.shortcut(features) + torch.tanh(self.scale) * self.mlp(
            self.input_norm(features)
        )
        return F.normalize(output, p=2, dim=-1, eps=1e-8)


@dataclass
class LoadedSurfaceMetricFeatureMapper:
    model: SurfaceMetricFeatureMapper
    device: str
    metadata: dict[str, object]

    def project_points(self, features: np.ndarray) -> np.ndarray:
        device = torch.device(
            self.device
            if torch.cuda.is_available() or not str(self.device).startswith("cuda")
            else "cpu"
        )
        with torch.no_grad():
            output = self.model.to(device).eval()(
                torch.as_tensor(features, dtype=torch.float32, device=device)
            )
        return output.cpu().numpy().astype(np.float32, copy=False)

    def project_map(self, feature_map: np.ndarray) -> np.ndarray:
        feature = np.asarray(feature_map, dtype=np.float32)
        channels, height, width = feature.shape
        points = feature.reshape(channels, -1).T
        return self.project_points(points).T.reshape(-1, height, width)


def save_surface_metric_feature_mapper(
    path: Path,
    model: SurfaceMetricFeatureMapper,
    metadata: dict[str, object],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_type": "surface_metric_feature_mapper",
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata),
        },
        Path(path),
    )


def load_surface_metric_feature_mapper(
    path: Path,
    *,
    device: str = "cpu",
) -> LoadedSurfaceMetricFeatureMapper:
    payload = torch.load(Path(path), map_location="cpu")
    if payload.get("artifact_type") != "surface_metric_feature_mapper":
        raise ValueError("not a surface metric feature mapper")
    metadata = dict(payload.get("metadata") or {})
    for key in (
        "uses_alike_descriptors",
        "uses_radio_intermediate",
        "uses_sfm_points",
        "uses_sfm_tracks",
    ):
        if bool(metadata.get(key, False)):
            raise ValueError(f"metric mapper violates contract: {key}")
    model = SurfaceMetricFeatureMapper(
        SurfaceMetricFeatureMapperConfig(**dict(payload["config"]))
    )
    model.load_state_dict(dict(payload["state_dict"]), strict=True)
    return LoadedSurfaceMetricFeatureMapper(
        model=model.eval(), device=str(device), metadata=metadata
    )
