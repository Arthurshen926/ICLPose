"""Regenerable physical-instance readouts from one canonical RADIO field.

The map continues to store exactly one code per observed 2DGS primitive.  The
two role heads below are functions of that same code and are therefore map
readouts, not additional map embeddings.  RADIO downstream adaptors are used
only as offline teachers when this module is trained; no teacher feature is
part of the saved artifact or the deployment inputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


SCHEMA = "goal_maplet_physical_instance_readout_v1"


@dataclass(frozen=True)
class PhysicalInstanceReadoutConfig:
    feature_dim: int = 128
    hidden_dim: int = 256
    residual_scale: float = 0.25
    context_radius: int = 4
    local_radius: int = 1
    local_neighbour_mass: float = 1.0e-4


class _ResidualRoleHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, residual_scale: float) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.attention = nn.Sequential(
            nn.Linear(feature_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # The artifact starts from the legacy support kernel.  Training may add
        # position-aware residual attention, so every experiment has a stable
        # zero-update baseline rather than a random architecture.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.attention[-1].weight)
        nn.init.zeros_(self.attention[-1].bias)

    def transform(self, feature: torch.Tensor) -> torch.Tensor:
        base = F.normalize(feature, dim=-1)
        return F.normalize(
            base + self.residual_scale * self.residual(base), dim=-1,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        relative_xy: torch.Tensor,
        base_weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        value = self.transform(tokens)
        radius2 = torch.sum(relative_xy * relative_xy, dim=-1, keepdim=True)
        position = torch.cat(
            [relative_xy, radius2, relative_xy[..., :1] * relative_xy[..., 1:]], dim=-1,
        )
        delta = self.attention(torch.cat([F.normalize(tokens, dim=-1), position], dim=-1)).squeeze(-1)
        log_weight = torch.log(torch.clamp(base_weights, min=1.0e-12)) + delta
        has_support = torch.any(mask, dim=-1, keepdim=True)
        safe_mask = mask.clone()
        safe_mask[..., 0] |= ~has_support.squeeze(-1)
        log_weight = log_weight.masked_fill(~safe_mask, -torch.inf)
        weight = torch.softmax(log_weight, dim=-1)
        weight = weight * has_support.to(weight.dtype)
        return F.normalize(torch.sum(weight[..., None] * value, dim=-2), dim=-1)


class PhysicalInstanceReadout(nn.Module):
    """Position-preserving context/local heads from one canonical code."""

    def __init__(self, config: PhysicalInstanceReadoutConfig) -> None:
        super().__init__()
        self.config = config
        self.context = _ResidualRoleHead(
            config.feature_dim, config.hidden_dim, config.residual_scale,
        )
        self.local = _ResidualRoleHead(
            config.feature_dim, config.hidden_dim, config.residual_scale,
        )

    def role_head(self, role: str) -> _ResidualRoleHead:
        if role == "context":
            return self.context
        if role == "local":
            return self.local
        raise ValueError(f"unknown physical-instance role: {role}")

    def project_flat(self, feature: torch.Tensor, *, role: str) -> torch.Tensor:
        return self.role_head(role).transform(feature)

    def forward(
        self,
        tokens: torch.Tensor,
        relative_xy: torch.Tensor,
        base_weights: torch.Tensor,
        mask: torch.Tensor,
        *,
        role: str,
    ) -> torch.Tensor:
        return self.role_head(role)(tokens, relative_xy, base_weights, mask)

    def project_numpy(
        self, feature: np.ndarray, *, role: str, device: str | None = None,
    ) -> np.ndarray:
        target = device or next(self.parameters()).device.type
        tensor = torch.from_numpy(np.asarray(feature, dtype=np.float32)).to(target)
        with torch.inference_mode():
            output = self.project_flat(tensor, role=role).cpu().numpy()
        return output.astype(np.float32)


def _region_token_sets(
    feature_map: np.ndarray,
    token_xy: np.ndarray,
    *,
    role: str,
    config: PhysicalInstanceReadoutConfig,
    spatial_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature = np.asarray(feature_map, dtype=np.float32)
    xy = np.asarray(token_xy, dtype=np.int64).reshape(-1, 2)
    if feature.ndim != 3 or feature.shape[0] != int(config.feature_dim):
        raise ValueError("physical-instance feature map shape differs")
    height, width = int(feature.shape[1]), int(feature.shape[2])
    if np.any((xy[:, 0] < 0) | (xy[:, 0] >= width) | (xy[:, 1] < 0) | (xy[:, 1] >= height)):
        raise ValueError("physical-instance token coordinate is outside the feature map")
    radius = int(config.context_radius if role == "context" else config.local_radius)
    axis = np.arange(-radius, radius + 1, dtype=np.int64)
    dx, dy = np.meshgrid(axis, axis, indexing="xy")
    offset = np.stack([dx.reshape(-1), dy.reshape(-1)], axis=1)
    sample_xy = xy[:, None, :] + offset[None, :, :]
    mask = (
        (sample_xy[..., 0] >= 0) & (sample_xy[..., 0] < width)
        & (sample_xy[..., 1] >= 0) & (sample_xy[..., 1] < height)
    )
    safe = sample_xy.copy()
    safe[..., 0] = np.clip(safe[..., 0], 0, width - 1)
    safe[..., 1] = np.clip(safe[..., 1], 0, height - 1)
    if spatial_valid_mask is not None:
        valid_map = np.asarray(spatial_valid_mask, dtype=bool)
        if valid_map.shape != (height, width):
            raise ValueError("physical-instance spatial validity shape differs")
        mask &= valid_map[safe[..., 1], safe[..., 0]]
    tokens = feature[:, safe[..., 1], safe[..., 0]].transpose(1, 2, 0)
    relative = np.broadcast_to(
        offset.astype(np.float32)[None] / max(float(radius), 1.0),
        (xy.shape[0], offset.shape[0], 2),
    ).copy()
    base = np.zeros(mask.shape, dtype=np.float32)
    if role == "context":
        # Preserve the existing (1,3,5,9) support mixture before any learned
        # attention residual, including per-scale border clipping.
        for size, mixture in zip((1, 3, 5, 9), (0.4, 0.3, 0.2, 0.1)):
            local = mask & (np.max(np.abs(offset), axis=1)[None] <= size // 2)
            base += float(mixture) * local / np.maximum(np.sum(local, axis=1, keepdims=True), 1)
    elif role == "local":
        base[:] = float(config.local_neighbour_mass)
        center = np.flatnonzero(np.all(offset == 0, axis=1))
        base[:, int(center[0])] = 1.0
        base *= mask
        base /= np.maximum(np.sum(base, axis=1, keepdims=True), 1.0e-12)
    else:
        raise ValueError(f"unknown physical-instance role: {role}")
    return tokens.astype(np.float32), relative, base, mask


def encode_physical_instance_regions(
    model: PhysicalInstanceReadout,
    feature_map: np.ndarray,
    token_xy: np.ndarray,
    *,
    role: str,
    device: str,
    batch_size: int = 512,
    spatial_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    tokens, relative, weight, mask = _region_token_sets(
        feature_map, token_xy, role=role, config=model.config,
        spatial_valid_mask=spatial_valid_mask,
    )
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, tokens.shape[0], int(batch_size)):
            end = min(start + int(batch_size), tokens.shape[0])
            output.append(model(
                torch.from_numpy(tokens[start:end]).to(device),
                torch.from_numpy(relative[start:end]).to(device),
                torch.from_numpy(weight[start:end]).to(device),
                torch.from_numpy(mask[start:end]).to(device),
                role=role,
            ).cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def transform_canonical_field_for_role(
    model: PhysicalInstanceReadout,
    field,
    *,
    role: str,
    device: str,
):
    """Regenerate a role view of the single canonical primitive field.

    This object is intentionally ephemeral.  It prevents a query descriptor
    transformed by the local head from being compared with untransformed map
    primitive codes, while the serialized map remains the one canonical
    field.  The role view carries no teacher data or observation history.
    """

    from .canonical_field import CanonicalSurfaceField

    metadata = dict(field.metadata or {})
    metadata.pop("content_sha256", None)
    metadata.update(
        representation=f"regenerated_{role}_role_view_of_single_canonical_field",
        canonical_role_view=str(role),
        stored_feature_type_count=1,
        stored_downstream_embedding_count=0,
        teacher_embeddings_discarded_after_training=True,
    )
    return CanonicalSurfaceField(
        primitive_rows=field.primitive_rows,
        codes=model.project_numpy(field.codes, role=role, device=device),
        confidence=field.confidence,
        uncertainty=field.uncertainty,
        physical_map_sha256=field.physical_map_sha256,
        metadata=metadata,
    )


def save_physical_instance_readout(
    model: PhysicalInstanceReadout,
    path: Path,
    *,
    metadata: Mapping[str, object],
) -> None:
    contract = {
        **dict(metadata),
        "artifact_type": SCHEMA,
        "map_representation": "one_canonical_code_with_regenerable_role_readouts",
        "stored_map_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "stores_teacher_embeddings": False,
        "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False,
        "stores_mapping_image_ids": False,
        "uses_radio_intermediate": False,
        "uses_alike_descriptors": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": asdict(model.config),
        "metadata": contract,
    }, output)


def load_physical_instance_readout(
    path: Path, *, device: str = "cpu",
) -> tuple[PhysicalInstanceReadout, dict[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    metadata = dict(payload["metadata"])
    if metadata.get("artifact_type") != SCHEMA:
        raise ValueError("not a Goal-Maplet physical-instance readout")
    forbidden = (
        "stores_teacher_embeddings", "stores_mapping_rgb", "stores_mapping_image_paths",
        "stores_mapping_image_ids", "uses_radio_intermediate", "uses_alike_descriptors",
        "uses_sfm_points", "uses_sfm_tracks",
    )
    if any(bool(metadata.get(key, False)) for key in forbidden):
        raise ValueError("physical-instance readout violates the deployment contract")
    if int(metadata.get("stored_map_feature_type_count", -1)) != 1:
        raise ValueError("physical-instance readout requires one canonical map feature")
    model = PhysicalInstanceReadout(
        PhysicalInstanceReadoutConfig(**payload["config"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
