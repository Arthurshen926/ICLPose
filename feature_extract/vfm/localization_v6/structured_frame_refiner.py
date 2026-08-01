"""Structured RADIO-final chart-frame residual prediction.

The refiner consumes one complete chart correlation volume and predicts one
bounded regional transform plus an explicit usable/null logit.  Atlas cells
remain correlated evidence inside the model; they never become persistent
point identities or independent pose correspondences.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StructuredFrameRefinerConfig:
    feature_dim: int = 128
    correlation_radius: int = 3
    hidden_dim: int = 96
    transformer_layers: int = 3
    attention_heads: int = 4
    maximum_linear_residual: float = 0.45
    maximum_translation_cells: float = 3.5
    linear_parameterization: str = "additive"
    update_parameterization: str = "query_affine"

    def __post_init__(self) -> None:
        if (
            int(self.feature_dim) <= 0
            or int(self.correlation_radius) <= 0
            or int(self.hidden_dim) <= 0
            or int(self.transformer_layers) <= 0
            or int(self.attention_heads) <= 0
            or int(self.hidden_dim) % int(self.attention_heads)
            or float(self.maximum_linear_residual) <= 0.0
            or float(self.maximum_translation_cells) <= 0.0
            or str(self.linear_parameterization)
            not in {"additive", "matrix_exponential"}
            or str(self.update_parameterization)
            not in {"query_affine", "canonical_residual"}
        ):
            raise ValueError("invalid structured frame-refiner config")

    @property
    def correlation_cells(self) -> int:
        width = 2 * int(self.correlation_radius) + 1
        return width * width


class StructuredFrameRefiner(nn.Module):
    """Set transformer over a complete chart's local correlation field."""

    def __init__(
        self,
        config: StructuredFrameRefinerConfig = (
            StructuredFrameRefinerConfig()
        ),
    ) -> None:
        super().__init__()
        self.config = config
        input_dim = 2 * int(config.correlation_cells) + 4
        hidden = int(config.hidden_dim)
        self.token_embedding = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(config.attention_heads),
            dim_feedforward=hidden * 3,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=int(config.transformer_layers)
        )
        self.pool_logit = nn.Linear(hidden, 1)
        self.output_norm = nn.LayerNorm(hidden)
        self.transform_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        self.usable_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        # Start from an identity update.  Training must earn every geometric
        # correction instead of perturbing the established baseline at step 0.
        nn.init.zeros_(self.transform_head[-1].weight)
        nn.init.zeros_(self.transform_head[-1].bias)

    def forward(
        self,
        correlation: torch.Tensor,
        canonical_uv: torch.Tensor,
        candidate_xy_normalized: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if correlation.ndim != 3:
            raise ValueError("correlation must have shape (B,N,2P)")
        batch, cells, channels = correlation.shape
        expected = 2 * int(self.config.correlation_cells)
        if int(channels) != expected:
            raise ValueError("correlation patch size differs from config")
        if canonical_uv.shape != (batch, cells, 2):
            raise ValueError("canonical_uv differs from correlation")
        if candidate_xy_normalized.shape != (batch, cells, 2):
            raise ValueError("candidate coordinates differ from correlation")
        if valid_mask is None:
            valid = torch.ones(
                (batch, cells),
                dtype=torch.bool,
                device=correlation.device,
            )
        else:
            valid = valid_mask.to(
                device=correlation.device, dtype=torch.bool
            )
            if valid.shape != (batch, cells):
                raise ValueError("valid_mask differs from correlation")
        token = self.token_embedding(
            torch.cat(
                [
                    correlation,
                    canonical_uv,
                    candidate_xy_normalized,
                ],
                dim=2,
            )
        )
        token = self.transformer(token, src_key_padding_mask=~valid)
        pooling_logit = self.pool_logit(token)[..., 0]
        pooling_logit = torch.where(
            valid,
            pooling_logit,
            torch.full_like(pooling_logit, -torch.inf),
        )
        empty = ~torch.any(valid, dim=1)
        safe_logit = torch.where(
            empty[:, None], torch.zeros_like(pooling_logit), pooling_logit
        )
        weight = torch.softmax(safe_logit, dim=1)
        weight = torch.where(valid, weight, torch.zeros_like(weight))
        pooled = torch.sum(weight[..., None] * token, dim=1)
        pooled = self.output_norm(pooled)
        raw = self.transform_head(pooled)
        bounded_linear = float(
            self.config.maximum_linear_residual
        ) * torch.tanh(raw[:, :4]).reshape(batch, 2, 2)
        if str(self.config.update_parameterization) == "canonical_residual":
            # Units are query-feature cells per unit canonical coordinate.
            # Zero is the identity update because this matrix is added to the
            # existing chart projection, rather than replacing it.
            linear = bounded_linear
        elif str(self.config.linear_parameterization) == "matrix_exponential":
            # exp(M) spans the orientation-preserving component of GL(2):
            # unlike I+dM, it can represent the 90/180-degree orientation
            # ambiguities left by discrete chart search while remaining
            # nonsingular throughout optimization.
            linear = torch.matrix_exp(bounded_linear)
        else:
            identity = torch.eye(
                2, device=raw.device, dtype=raw.dtype
            )[None]
            linear = identity + bounded_linear
        translation = float(
            self.config.maximum_translation_cells
        ) * torch.tanh(raw[:, 4:6])
        return {
            "linear": linear,
            "translation": translation,
            "usable_logit": self.usable_head(pooled)[:, 0],
            "cell_weight": weight,
        }

    def apply_update(
        self,
        candidate_xy: torch.Tensor,
        linear: torch.Tensor,
        translation: torch.Tensor,
        canonical_uv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if str(self.config.update_parameterization) == "canonical_residual":
            if canonical_uv is None or canonical_uv.shape != candidate_xy.shape:
                raise ValueError(
                    "canonical residual update requires canonical_uv"
                )
            return (
                candidate_xy
                + torch.einsum("bnc,bdc->bnd", canonical_uv, linear)
                + translation[:, None]
            )
        # A chart residual is defined in the chart's local query-plane frame,
        # not around the feature-map origin.  Applying even a small rotation
        # around (0, 0) creates a center-dependent translation that cannot be
        # represented by the deliberately bounded translation head.
        center = torch.mean(candidate_xy, dim=1, keepdim=True)
        centered = candidate_xy - center
        return (
            torch.einsum("bnc,bdc->bnd", centered, linear)
            + center
            + translation[:, None]
        )


def structured_frame_correlation(
    query_feature: torch.Tensor,
    map_feature: torch.Tensor,
    map_mode_feature: torch.Tensor | None,
    map_mode_weight: torch.Tensor | None,
    map_mode_valid: torch.Tensor | None,
    candidate_xy: torch.Tensor,
    *,
    radius: int,
    mode_temperature: float = 0.07,
) -> torch.Tensor:
    """Build mean/mode RADIO correlation patches for complete charts."""

    if query_feature.ndim == 3:
        query_feature = query_feature[None]
    if map_feature.ndim != 3 or candidate_xy.ndim != 3:
        raise ValueError("map features and candidate_xy must be batched")
    batch, cells, channels = map_feature.shape
    if (
        int(query_feature.shape[1]) != int(channels)
        or candidate_xy.shape != (batch, cells, 2)
    ):
        raise ValueError("structured correlation inputs differ")
    if int(query_feature.shape[0]) == 1 and batch > 1:
        query_feature = query_feature.expand(batch, -1, -1, -1)
    if int(query_feature.shape[0]) != batch:
        raise ValueError("query batch differs from chart batch")
    device, dtype = query_feature.device, query_feature.dtype
    values = torch.arange(
        -int(radius), int(radius) + 1, device=device, dtype=dtype
    )
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    offsets = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    sample_xy = candidate_xy[..., None, :] + offsets[None, None]
    grid = sample_xy.clone()
    grid[..., 0] = (
        2.0 * (grid[..., 0] + 0.5) / query_feature.shape[3] - 1.0
    )
    grid[..., 1] = (
        2.0 * (grid[..., 1] + 0.5) / query_feature.shape[2] - 1.0
    )
    sampled = F.grid_sample(
        query_feature,
        grid.reshape(batch, cells, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    sampled = F.normalize(sampled, dim=3, eps=1e-6)
    mean = torch.einsum(
        "bnc,bnpc->bnp",
        F.normalize(map_feature, dim=2, eps=1e-6),
        sampled,
    )
    if map_mode_feature is None:
        mode = mean
    else:
        if (
            map_mode_weight is None
            or map_mode_valid is None
            or map_mode_feature.ndim != 4
            or map_mode_feature.shape[:2] != (batch, cells)
        ):
            raise ValueError("incomplete map appearance modes")
        mode_feature = F.normalize(map_mode_feature, dim=3, eps=1e-6)
        similarity = torch.einsum(
            "bnmc,bnpc->bnpm", mode_feature, sampled
        )
        logit = (
            similarity / float(mode_temperature)
            + torch.log(
                torch.clamp(map_mode_weight[:, :, None], min=1e-8)
            )
        )
        logit = torch.where(
            map_mode_valid[:, :, None],
            logit,
            torch.full_like(logit, -torch.inf),
        )
        has_mode = torch.any(map_mode_valid, dim=2)
        marginalized = float(mode_temperature) * torch.logsumexp(
            logit, dim=3
        )
        mode = torch.where(has_mode[..., None], marginalized, mean)
    return torch.cat([mean, mode], dim=2)


def save_structured_frame_refiner(
    path: Path,
    model: StructuredFrameRefiner,
    metadata: Mapping[str, object],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_type": "v6_structured_frame_refiner",
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata),
        },
        Path(path),
    )


def load_structured_frame_refiner(
    path: Path, *, device: str = "cpu"
) -> tuple[StructuredFrameRefiner, Mapping[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("artifact_type") != "v6_structured_frame_refiner":
        raise ValueError("not a V6 structured frame-refiner checkpoint")
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
        raise ValueError("structured frame refiner violates map contract")
    model = StructuredFrameRefiner(
        StructuredFrameRefinerConfig(**dict(payload["config"]))
    )
    model.load_state_dict(dict(payload["state_dict"]), strict=True)
    return model.to(device).eval(), metadata
