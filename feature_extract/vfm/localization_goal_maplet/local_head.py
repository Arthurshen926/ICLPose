"""Regenerable metric head over the single stored canonical VFM field."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA = "goal_maplet_child_local_readout_head_v1"


class ChildLocalReadoutHead(nn.Module):
    """Small learned metric; it stores weights, never per-map embeddings."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.query_norm = nn.LayerNorm(self.dimension)
        self.map_norm = nn.LayerNorm(self.dimension)
        self.query_projection = nn.Linear(self.dimension, self.dimension, bias=False)
        self.map_projection = nn.Linear(self.dimension, self.dimension, bias=False)
        nn.init.eye_(self.query_projection.weight)
        nn.init.eye_(self.map_projection.weight)

    def encode_query(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_projection(self.query_norm(value)), dim=-1)

    def encode_map(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.map_projection(self.map_norm(value)), dim=-1)


@dataclass(frozen=True)
class ChildLocalHeadArtifact:
    model: ChildLocalReadoutHead
    metadata: Mapping[str, object]


def save_child_local_head(path: Path, model: ChildLocalReadoutHead, metadata: Mapping[str, object]) -> None:
    payload = {
        "artifact_type": SCHEMA,
        "config": {"dimension": int(model.dimension), "architecture": "dual_layernorm_linear_identity_init"},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": {
            "stored_downstream_embedding_count": 0,
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "stores_mapping_image_ids": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_point_correspondences": False,
            **dict(metadata),
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(path))


def load_child_local_head(path: Path, *, device: str = "cpu") -> ChildLocalHeadArtifact:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("artifact_type") != SCHEMA:
        raise ValueError("not a Goal-Maplet child-local head")
    metadata = dict(payload.get("metadata", {}))
    if int(metadata.get("stored_downstream_embedding_count", -1)) != 0:
        raise ValueError("child-local head contains stored downstream embeddings")
    model = ChildLocalReadoutHead(int(payload["config"]["dimension"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return ChildLocalHeadArtifact(model=model, metadata=metadata)
