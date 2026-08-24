"""Minimal trainable readout for conservative candidate pose transport.

This module deliberately stops before absolute pose regression.  A compact
query head turns the frozen RADIO grid plus camera-ray coordinates into a 32D
pose code and typed geometry observations.  A low-rank map field supplies
anonymous view-conditioned codes.  Candidate rendering and sparse transport
remain explicit downstream operations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


MODEL_SCHEMA = "goal_maplet_minimal_sparse_pose_transport_readout_v1"
MAP_FIELD_SEMANTICS = "canonical_code_plus_anonymous_view_scale_low_rank_v1"
DEPTH_SEMANTICS = (
    "ordinal_depth_v1",
    "centered_log_depth_v1",
    "metric_log_depth_with_uncertainty_v1",
)


@dataclass(frozen=True)
class MinimalPoseTransportConfig:
    radio_channels: int = 1280
    map_feature_dim: int = 128
    pose_code_dim: int = 32
    hidden_dim: int = 64
    map_view_rank: int = 4
    shared_query_map_projection: bool = False
    depth_semantics: str = "centered_log_depth_v1"
    zero_norm_threshold: float = 1.0e-8


@dataclass(frozen=True)
class QueryPoseHeadOutput:
    pose_code: torch.Tensor
    normal_camera: torch.Tensor
    relative_depth: torch.Tensor
    boundary: torch.Tensor
    confidence: torch.Tensor
    pose_code_valid: torch.Tensor
    normal_valid: torch.Tensor
    depth_valid: torch.Tensor
    boundary_valid: torch.Tensor
    normal_frame: str
    depth_semantics: str


def _validate_config(config: MinimalPoseTransportConfig) -> None:
    if int(config.radio_channels) <= 0:
        raise ValueError("radio_channels must be positive")
    if int(config.map_feature_dim) <= 0:
        raise ValueError("map_feature_dim must be positive")
    if not 2 <= int(config.pose_code_dim) <= 128:
        raise ValueError("pose_code_dim must lie in [2,128]")
    if int(config.hidden_dim) <= 0 or int(config.map_view_rank) <= 0:
        raise ValueError("hidden_dim and map_view_rank must be positive")
    if bool(config.shared_query_map_projection) and int(config.radio_channels) != int(
        config.map_feature_dim
    ):
        raise ValueError("shared query/map projection requires equal feature dimensions")
    if str(config.depth_semantics) not in DEPTH_SEMANTICS:
        raise ValueError("unknown pose transport depth semantics")
    if not np.isfinite(config.zero_norm_threshold) or float(config.zero_norm_threshold) <= 0.0:
        raise ValueError("zero_norm_threshold must be positive")


class MinimalPoseTransportReadout(nn.Module):
    """Low-capacity RADIO+ray query head and learned additive edge weights."""

    def __init__(self, config: MinimalPoseTransportConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        hidden = int(config.hidden_dim)
        self.stem = nn.Conv2d(int(config.radio_channels) + 2, hidden, kernel_size=1)
        self.body = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # code + normal + depth + boundary + four modality confidences
        self.head = nn.Conv2d(hidden, int(config.pose_code_dim) + 3 + 1 + 1 + 4, kernel_size=1)
        # The canonical map remains the single stored 128D RADIO-derived
        # field.  This small regenerable readout maps it into the same 32D
        # relative-pose code used by the query head; it does not create a
        # second stored map embedding.
        self.map_projection = nn.Linear(
            int(config.map_feature_dim), int(config.pose_code_dim), bias=False
        )
        # When the query is first projected by the frozen surface mapper, its
        # tokens and the canonical map already inhabit the same 128D space.
        # The same low-rank projection must then be applied on both sides;
        # otherwise a handful of pose labels would be asked to relearn the
        # entire RADIO-to-map alignment. A zero-initialized residual preserves
        # this shared-space identity at initialization while allowing a small
        # candidate-pose-specific correction to be learned later.
        if bool(config.shared_query_map_projection):
            self.query_pose_residual_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("query_pose_residual_scale", None)
        # feature, normal, depth, boundary, hierarchy and layout coefficients.
        # softplus makes every observed modality above-floor evidence; turning
        # a modality off therefore cannot add energy.
        self.edge_weight_unconstrained = nn.Parameter(torch.zeros(6))
        self.source_bias = nn.Parameter(torch.tensor(-0.5))
        self.unmatched_sink_logit = nn.Parameter(torch.tensor(0.0))

    def edge_weights(self) -> torch.Tensor:
        return F.softplus(self.edge_weight_unconstrained)

    def project_map_code(
        self, canonical_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.as_tensor(canonical_feature)
        if value.ndim < 2 or int(value.shape[-1]) != int(self.config.map_feature_dim):
            raise ValueError("canonical map features have the wrong final dimension")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError("canonical map features must be finite floating tensors")
        projected = self.map_projection(value)
        norm = torch.linalg.vector_norm(projected, dim=-1)
        threshold = float(self.config.zero_norm_threshold)
        valid = norm >= threshold
        return projected / norm[..., None].clamp_min(threshold), valid

    def forward(self, radio_final: torch.Tensor, ray_xy: torch.Tensor) -> QueryPoseHeadOutput:
        if radio_final.ndim != 4 or int(radio_final.shape[1]) != int(self.config.radio_channels):
            raise ValueError("radio_final must have shape [batch,radio_channels,H,W]")
        if ray_xy.shape != (radio_final.shape[0], 2, radio_final.shape[2], radio_final.shape[3]):
            raise ValueError("ray_xy must have shape [batch,2,H,W]")
        if (
            not torch.is_floating_point(radio_final)
            or not torch.is_floating_point(ray_xy)
            or ray_xy.dtype != radio_final.dtype
            or ray_xy.device != radio_final.device
        ):
            raise ValueError("RADIO and ray inputs must share a floating dtype and device")
        if not torch.isfinite(radio_final).all() or not torch.isfinite(ray_xy).all():
            raise ValueError("query pose-head inputs must be finite")
        value = self.head(self.body(self.stem(torch.cat([radio_final, ray_xy], dim=1))))
        dimension = int(self.config.pose_code_dim)
        code_raw = value[:, :dimension]
        if bool(self.config.shared_query_map_projection):
            shared_base = F.conv2d(
                radio_final, self.map_projection.weight[:, :, None, None]
            )
            code_raw = shared_base + self.query_pose_residual_scale * code_raw
        normal_raw = value[:, dimension : dimension + 3]
        relative_depth = value[:, dimension + 3]
        boundary = torch.sigmoid(value[:, dimension + 4])
        confidence = torch.sigmoid(value[:, dimension + 5 : dimension + 9])
        code_norm = torch.linalg.vector_norm(code_raw, dim=1)
        normal_norm = torch.linalg.vector_norm(normal_raw, dim=1)
        threshold = float(self.config.zero_norm_threshold)
        code_valid = code_norm >= threshold
        normal_valid = normal_norm >= threshold
        depth_valid = torch.isfinite(relative_depth)
        boundary_valid = torch.isfinite(boundary)
        pose_code = code_raw / code_norm[:, None].clamp_min(threshold)
        normal_camera = normal_raw / normal_norm[:, None].clamp_min(threshold)
        # Invalid vectors carry exactly zero confidence and cannot masquerade
        # as a neutral cosine observation.
        confidence = confidence.clone()
        confidence[:, 0] *= code_valid
        confidence[:, 1] *= normal_valid
        return QueryPoseHeadOutput(
            pose_code=pose_code,
            normal_camera=normal_camera,
            relative_depth=relative_depth,
            boundary=boundary,
            confidence=confidence,
            pose_code_valid=code_valid,
            normal_valid=normal_valid,
            depth_valid=depth_valid,
            boundary_valid=boundary_valid,
            normal_frame="camera",
            depth_semantics=str(self.config.depth_semantics),
        )


def anonymous_view_conditioned_map_code(
    canonical_code: torch.Tensor,
    view_scale_basis: torch.Tensor,
    view_scale_coordinate: torch.Tensor,
    *,
    zero_norm_threshold: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``normalize(mu + B phi(view,scale))`` without mode labels."""

    canonical = torch.as_tensor(canonical_code)
    basis = torch.as_tensor(
        view_scale_basis, device=canonical.device, dtype=canonical.dtype
    )
    coordinate = torch.as_tensor(
        view_scale_coordinate, device=canonical.device, dtype=canonical.dtype
    )
    if canonical.ndim != 2:
        raise ValueError("canonical_code must have shape [primitive,dimension]")
    if basis.ndim != 3 or basis.shape[0] != canonical.shape[0] or basis.shape[2] != canonical.shape[1]:
        raise ValueError("view_scale_basis must have shape [primitive,rank,dimension]")
    if coordinate.ndim != 3 or coordinate.shape[1:] != basis.shape[:2]:
        raise ValueError("view_scale_coordinate must have shape [batch,primitive,rank]")
    if any(not torch.isfinite(value).all() for value in (canonical, basis, coordinate)):
        raise ValueError("anonymous map pose field must be finite")
    value = canonical[None] + torch.einsum("bpr,prd->bpd", coordinate, basis)
    norm = torch.linalg.vector_norm(value, dim=2)
    threshold = float(zero_norm_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("zero_norm_threshold must be positive")
    valid = norm >= threshold
    return value / norm[..., None].clamp_min(threshold), valid


def pose_transport_model_content_sha256(model: MinimalPoseTransportReadout) -> str:
    """Hash config, architecture semantics and exact tensor bytes deterministically."""

    digest = hashlib.sha256()
    header = {
        "artifact_type": MODEL_SCHEMA,
        "config": asdict(model.config),
        "map_field_semantics": MAP_FIELD_SEMANTICS,
        "normal_frame": "camera",
        "uses_absolute_pose_regression": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def save_minimal_pose_transport_readout(
    model: MinimalPoseTransportReadout,
    path: Path,
    *,
    metadata: Mapping[str, object],
) -> None:
    content = pose_transport_model_content_sha256(model)
    payload = {
        "artifact_type": MODEL_SCHEMA,
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "model_content_sha256": content,
        "metadata": {
            **dict(metadata),
            "normal_frame": "camera",
            "map_field_semantics": MAP_FIELD_SEMANTICS,
            "uses_absolute_pose_regression": False,
            "uses_point_correspondences": False,
            "uses_pnp": False,
            "leave_one_view_out_required": True,
            "candidate_conditioned_sparse_transport": True,
            "production_eligible": False,
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(path))


def load_minimal_pose_transport_readout(
    path: Path, *, device: str = "cpu"
) -> tuple[MinimalPoseTransportReadout, dict[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("artifact_type") != MODEL_SCHEMA:
        raise ValueError("not a minimal sparse pose transport readout")
    model = MinimalPoseTransportReadout(
        MinimalPoseTransportConfig(**dict(payload["config"]))
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    expected = str(payload.get("model_content_sha256", ""))
    if pose_transport_model_content_sha256(model) != expected:
        raise ValueError("pose transport model content hash differs")
    metadata = dict(payload["metadata"])
    required = {
        "normal_frame": "camera",
        "map_field_semantics": MAP_FIELD_SEMANTICS,
        "uses_absolute_pose_regression": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "leave_one_view_out_required": True,
        "candidate_conditioned_sparse_transport": True,
        "production_eligible": False,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("pose transport model semantic contract differs")
    return model, metadata
