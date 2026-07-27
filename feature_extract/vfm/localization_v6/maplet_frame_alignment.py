"""Global canonical-chart to query alignment for V6 coarse pose.

The module estimates an explicit 2D projection of a metric surface chart.  It
does not turn a RADIO region center into a point correspondence and it does
not reuse mapping-camera poses.  Canonical chart features are correlated over
the complete query feature map across translation, in-plane rotation, and
anisotropic scale; chart-frame control points then provide regional geometric
constraints for coarse pose hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.query_to_3d_matching import (
    camera_matrix_and_distortion,
)


@dataclass(frozen=True)
class MapletFrameSearchConfig:
    """Bounded discrete global search in query-feature coordinates."""

    geometric_mean_sizes: tuple[float, ...] = (
        2.5,
        4.0,
        6.5,
        10.0,
        15.0,
    )
    aspect_multipliers: tuple[float, ...] = (0.55, 1.0, 1.8)
    rotation_degrees: tuple[float, ...] = (
        0.0,
        45.0,
        90.0,
        135.0,
        180.0,
        225.0,
        270.0,
        315.0,
    )
    shear_values: tuple[float, ...] = (-0.35, 0.0, 0.35)
    maximum_modes: int = 16
    per_transform_peaks: int = 2
    minimum_overlap_fraction: float = 0.60
    minimum_template_support_cells: float = 2.5
    matchability_floor: float = 0.10
    local_peak_temperature: float = 0.05
    posterior_temperature: float = 0.06
    diagnostic_null_score: float = 0.42
    spatial_mode_nms_radius_cells: float = 2.0
    affine_mode_nms_radius_cells: float = 2.0
    maximum_modes_per_spatial_cluster: int = 2
    maximum_appearance_modes: int = 2
    include_mean_appearance: bool = True
    background_similarity: float = 0.25
    support_score_power: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.geometric_mean_sizes
            or any(value <= 0.0 for value in self.geometric_mean_sizes)
            or not self.aspect_multipliers
            or any(value <= 0.0 for value in self.aspect_multipliers)
            or not self.rotation_degrees
            or not self.shear_values
            or int(self.maximum_modes) <= 0
            or int(self.per_transform_peaks) <= 0
            or float(self.spatial_mode_nms_radius_cells) < 0.0
            or float(self.affine_mode_nms_radius_cells) < 0.0
            or int(self.maximum_modes_per_spatial_cluster) <= 0
            or int(self.maximum_appearance_modes) < 0
            or not 0.0 <= float(self.support_score_power) <= 1.0
        ):
            raise ValueError("frame-search grids and limits must be positive")
        if not 0.0 < float(self.minimum_overlap_fraction) <= 1.0:
            raise ValueError("minimum_overlap_fraction must lie in (0,1]")
        if float(self.minimum_template_support_cells) <= 0.0:
            raise ValueError("minimum_template_support_cells must be positive")
        if (
            float(self.local_peak_temperature) <= 0.0
            or float(self.posterior_temperature) <= 0.0
        ):
            raise ValueError("frame-search temperatures must be positive")


@dataclass(frozen=True)
class MapletFrameMatch:
    """One chart projection mode in query-feature coordinates.

    ``canonical_to_query`` maps canonical tangent coordinates ``(u,v)`` in
    ``[-1,1]^2`` to query feature-cell coordinates.
    """

    chart_id: int
    canonical_to_query: np.ndarray
    query_center_xy: np.ndarray
    scale_xy: np.ndarray
    in_plane_rotation_deg: float
    covariance_xy: np.ndarray
    score: float
    probability: float
    null_probability: float
    support_fraction: float
    feature_level: str
    feature_stride: int
    canonical_homography: np.ndarray | None = None

    def __post_init__(self) -> None:
        transform = np.asarray(self.canonical_to_query, dtype=np.float64)
        center = np.asarray(self.query_center_xy, dtype=np.float64).reshape(-1)
        scale = np.asarray(self.scale_xy, dtype=np.float64).reshape(-1)
        covariance = np.asarray(self.covariance_xy, dtype=np.float64)
        homography = (
            None
            if self.canonical_homography is None
            else np.asarray(self.canonical_homography, dtype=np.float64)
        )
        if (
            transform.shape != (2, 3)
            or center.shape != (2,)
            or scale.shape != (2,)
            or covariance.shape != (2, 2)
            or not np.all(np.isfinite(transform))
            or not np.all(np.isfinite(covariance))
            or np.any(scale <= 0.0)
        ):
            raise ValueError("invalid maplet frame match")
        if homography is not None and (
            homography.shape != (3, 3)
            or not np.all(np.isfinite(homography))
            or abs(float(np.linalg.det(homography))) <= 1e-12
        ):
            raise ValueError("invalid canonical chart homography")
        object.__setattr__(
            self, "canonical_to_query", transform.astype(np.float32)
        )
        object.__setattr__(
            self, "query_center_xy", center.astype(np.float32)
        )
        object.__setattr__(self, "scale_xy", scale.astype(np.float32))
        object.__setattr__(
            self, "covariance_xy", covariance.astype(np.float32)
        )
        if homography is not None:
            object.__setattr__(
                self,
                "canonical_homography",
                homography.astype(np.float32),
            )

    def canonical_points_to_query(
        self, canonical_uv: np.ndarray
    ) -> np.ndarray:
        uv = np.asarray(canonical_uv, dtype=np.float64).reshape(-1, 2)
        if self.canonical_homography is not None:
            homogeneous = np.c_[uv, np.ones((uv.shape[0],))]
            warped = homogeneous @ np.asarray(
                self.canonical_homography, dtype=np.float64
            ).T
            denominator = np.where(
                np.abs(warped[:, 2:3]) > 1e-12,
                warped[:, 2:3],
                np.where(warped[:, 2:3] < 0.0, -1e-12, 1e-12),
            )
            return (
                warped[:, :2] / denominator
            ).astype(np.float32)
        return (
            uv @ self.canonical_to_query[:, :2].T
            + self.canonical_to_query[:, 2][None]
        ).astype(np.float32)

    def canonical_points_to_pixels(
        self, canonical_uv: np.ndarray
    ) -> np.ndarray:
        query = self.canonical_points_to_query(canonical_uv)
        return (
            (query + 0.5) * float(self.feature_stride) - 0.5
        ).astype(np.float32)


@dataclass(frozen=True)
class FramePoseHypothesis:
    pose_w2c: np.ndarray
    score: float
    source_chart_ids: tuple[int, ...]
    reprojection_error_px: float
    positive_depth: bool
    control_model: str = "frame_controls"


@dataclass(frozen=True)
class _Template:
    feature: torch.Tensor
    mask: torch.Tensor
    width: float
    height: float
    angle_deg: float
    linear: np.ndarray
    support: float


@dataclass(frozen=True)
class _Candidate:
    score: float
    center_xy: np.ndarray
    covariance_xy: np.ndarray
    width: float
    height: float
    angle_deg: float
    linear: np.ndarray
    support_fraction: float


def _affine_candidate_distance(
    first: _Candidate, second: _Candidate
) -> float:
    """Mean canonical-corner displacement between two linear frame modes."""

    corners = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    displacement = corners @ (
        np.asarray(first.linear, dtype=np.float64)
        - np.asarray(second.linear, dtype=np.float64)
    ).T
    return float(np.mean(np.linalg.norm(displacement, axis=1)))


def _select_distinct_candidates(
    candidates: Sequence[_Candidate],
    config: MapletFrameSearchConfig,
) -> list[_Candidate]:
    """Retain distinct joint translation/affine modes.

    Center-only NMS is invalid for frame alignment: several scale, rotation or
    shear hypotheses may share one correlation center.  Those alternatives
    must remain available to the regional pose solver.  A small per-location
    cap prevents one repeated peak from consuming the complete mode budget.
    """

    selected: list[_Candidate] = []
    spatial_radius = float(config.spatial_mode_nms_radius_cells)
    affine_radius = float(config.affine_mode_nms_radius_cells)
    spatial_cap = int(config.maximum_modes_per_spatial_cluster)
    for candidate in candidates:
        colocated = [
            retained
            for retained in selected
            if float(
                np.linalg.norm(
                    candidate.center_xy - retained.center_xy
                )
            )
            < spatial_radius
        ]
        if len(colocated) >= spatial_cap:
            continue
        duplicate = any(
            _affine_candidate_distance(candidate, retained)
            < affine_radius
            for retained in colocated
        )
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= int(config.maximum_modes):
            break
    return selected


def _odd_ceiling(value: float, *, minimum: int = 3) -> int:
    result = max(int(np.ceil(float(value))), int(minimum))
    return result if result % 2 == 1 else result + 1


def _render_template(
    feature: torch.Tensor,
    mask: torch.Tensor,
    *,
    width: float,
    height: float,
    angle_deg: float,
    shear: float,
) -> _Template | None:
    angle = np.deg2rad(float(angle_deg))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    linear = rotation @ np.asarray(
        [
            [float(width) * 0.5, float(shear) * float(height) * 0.5],
            [0.0, float(height) * 0.5],
        ],
        dtype=np.float64,
    )
    canonical_corners = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    projected_corners = canonical_corners @ linear.T
    span = np.ptp(projected_corners, axis=0)
    kernel_width = _odd_ceiling(float(span[0]) + 2.0)
    kernel_height = _odd_ceiling(float(span[1]) + 2.0)
    dx = torch.arange(
        kernel_width, device=feature.device, dtype=feature.dtype
    ) - (kernel_width - 1.0) * 0.5
    dy = torch.arange(
        kernel_height, device=feature.device, dtype=feature.dtype
    ) - (kernel_height - 1.0) * 0.5
    yy, xx = torch.meshgrid(dy, dx, indexing="ij")
    inverse = np.linalg.inv(linear)
    canonical_u = inverse[0, 0] * xx + inverse[0, 1] * yy
    canonical_v = inverse[1, 0] * xx + inverse[1, 1] * yy
    grid = torch.stack([canonical_u, canonical_v], dim=-1)[None]
    sampled_feature = F.grid_sample(
        feature[None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]
    sampled_mask = F.grid_sample(
        mask[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    inside = (canonical_u.abs() <= 1.0) & (canonical_v.abs() <= 1.0)
    sampled_mask = sampled_mask * inside.to(sampled_mask.dtype)
    sampled_feature = F.normalize(sampled_feature, dim=0, eps=1e-8)
    sampled_feature = sampled_feature * sampled_mask[None]
    support = float(sampled_mask.sum().item())
    if support <= 1e-6:
        return None
    return _Template(
        feature=sampled_feature,
        mask=sampled_mask,
        width=float(width),
        height=float(height),
        angle_deg=float(angle_deg) % 360.0,
        linear=linear.astype(np.float32),
        support=support,
    )


def _template_grid(
    atlas_feature: torch.Tensor,
    atlas_mask: torch.Tensor,
    physical_aspect: float,
    config: MapletFrameSearchConfig,
) -> list[_Template]:
    templates = []
    for size in config.geometric_mean_sizes:
        for multiplier in config.aspect_multipliers:
            aspect = float(
                np.clip(
                    float(physical_aspect) * float(multiplier), 0.20, 5.0
                )
            )
            width = float(size) * np.sqrt(aspect)
            height = float(size) / np.sqrt(aspect)
            for angle in config.rotation_degrees:
                for shear in config.shear_values:
                    template = _render_template(
                        atlas_feature,
                        atlas_mask,
                        width=width,
                        height=height,
                        angle_deg=float(angle),
                        shear=float(shear),
                    )
                    if (
                        template is not None
                        and template.support
                        >= float(config.minimum_template_support_cells)
                    ):
                        templates.append(template)
    return templates


def _atlas_appearance_sources(
    atlas: MapletFeatureAtlasBank,
    row: int,
    atlas_feature: torch.Tensor,
    atlas_mask: torch.Tensor,
    config: MapletFrameSearchConfig,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return the mean and strongest bounded anonymous appearance modes."""

    sources = []
    if bool(config.include_mean_appearance) or atlas.mode_features is None:
        sources.append((atlas_feature, atlas_mask))
    if (
        atlas.mode_features is None
        or int(config.maximum_appearance_modes) == 0
    ):
        return sources
    weights = np.asarray(atlas.mode_weights[row], dtype=np.float64)
    valid = np.asarray(atlas.mode_valid_mask[row], dtype=bool)
    source_score = np.sum(np.where(valid, weights, 0.0), axis=(1, 2))
    order = np.argsort(-source_score, kind="mergesort")[
        : int(config.maximum_appearance_modes)
    ]
    for mode in order.tolist():
        if float(source_score[mode]) <= 0.0:
            continue
        feature = torch.from_numpy(
            np.asarray(atlas.mode_features[row, mode], dtype=np.float32)
        ).to(device=atlas_feature.device, dtype=atlas_feature.dtype)
        mode_mask = (
            np.asarray(atlas.mode_valid_mask[row, mode], dtype=np.float32)
            * np.sqrt(
                np.maximum(
                    np.asarray(
                        atlas.mode_weights[row, mode], dtype=np.float32
                    ),
                    0.0,
                )
            )
        )
        mask = torch.from_numpy(mode_mask).to(
            device=atlas_mask.device, dtype=atlas_mask.dtype
        )
        sources.append((feature, atlas_mask * mask))
    return sources


def _same_padding(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
    left = (int(width) - 1) // 2
    right = int(width) - 1 - left
    top = (int(height) - 1) // 2
    bottom = int(height) - 1 - top
    return F.pad(value, (left, right, top, bottom))


def _local_peak_moments(
    response: torch.Tensor,
    x: int,
    y: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = response.shape
    x0, x1 = max(int(x) - 1, 0), min(int(x) + 2, width)
    y0, y1 = max(int(y) - 1, 0), min(int(y) + 2, height)
    patch = response[y0:y1, x0:x1]
    finite = torch.isfinite(patch)
    if not bool(torch.any(finite)):
        return np.asarray([x, y], dtype=np.float64), np.eye(2)
    safe = torch.where(finite, patch, torch.full_like(patch, -1e6))
    weight = torch.softmax(
        ((safe - torch.max(safe)) / float(temperature)).reshape(-1),
        dim=0,
    ).reshape_as(safe)
    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, device=response.device, dtype=response.dtype),
        torch.arange(x0, x1, device=response.device, dtype=response.dtype),
        indexing="ij",
    )
    coordinates = torch.stack([xx, yy], dim=-1)
    mean = torch.sum(weight[..., None] * coordinates, dim=(0, 1))
    residual = coordinates - mean
    covariance = torch.einsum(
        "hw,hwi,hwj->ij", weight, residual, residual
    )
    covariance = covariance + torch.eye(
        2, device=response.device, dtype=response.dtype
    ) * 0.05
    return (
        mean.detach().cpu().numpy().astype(np.float64),
        covariance.detach().cpu().numpy().astype(np.float64),
    )


def _angular_distance_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def align_maplet_frame_global(
    atlas: MapletFeatureAtlasBank,
    chart_id: int,
    query_feature: torch.Tensor,
    query_matchability: torch.Tensor | None = None,
    *,
    feature_level: str,
    feature_stride: int,
    config: MapletFrameSearchConfig = MapletFrameSearchConfig(),
    query_location_priors: np.ndarray | None = None,
    query_location_prior_weights: np.ndarray | None = None,
    location_prior_sigma_px: float = 96.0,
    location_prior_strength: float = 0.08,
    location_window_radius_px: float | None = None,
) -> tuple[MapletFrameMatch, ...]:
    """Globally align one canonical chart to a full query feature map."""

    if query_feature.ndim == 4:
        if query_feature.shape[0] != 1:
            raise ValueError("query_feature batch size must be one")
        query_feature = query_feature[0]
    if query_feature.ndim != 3:
        raise ValueError("query_feature must have shape (C,H,W)")
    rows = np.flatnonzero(atlas.maplet_ids == int(chart_id))
    if rows.size != 1:
        raise ValueError(f"unknown or duplicate metric chart ID: {chart_id}")
    row = int(rows[0])
    if query_feature.shape[0] != atlas.feature_dim:
        raise ValueError("query and metric-chart feature dimensions differ")
    device = query_feature.device
    dtype = query_feature.dtype
    atlas_feature = torch.from_numpy(atlas.features[row]).to(
        device=device, dtype=dtype
    )
    feature_norm = torch.linalg.vector_norm(atlas_feature, dim=0)
    atlas_mask = torch.from_numpy(
        (
            np.asarray(atlas.valid_mask[row], dtype=bool)
            & (np.asarray(atlas.support_count[row]) > 0)
        ).astype(np.float32)
    ).to(device=device, dtype=dtype)
    atlas_mask = atlas_mask * (feature_norm > 0.5).to(dtype)
    if float(atlas_mask.sum().item()) <= 0.0:
        return ()
    if query_matchability is None:
        matchability = torch.ones(
            query_feature.shape[-2:],
            device=device,
            dtype=dtype,
        )
    else:
        matchability = query_matchability
        if matchability.ndim == 4:
            matchability = matchability[0, 0]
        elif matchability.ndim == 3:
            matchability = matchability[0]
        if matchability.shape != query_feature.shape[-2:]:
            matchability = F.interpolate(
                matchability[None, None],
                size=query_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        matchability = matchability.to(device=device, dtype=dtype)
    matchability = torch.clamp(
        matchability, min=float(config.matchability_floor), max=1.0
    )
    location_log_prior = None
    location_allowed = None
    if query_location_priors is not None:
        prior_xy = np.asarray(
            query_location_priors, dtype=np.float32
        ).reshape(-1, 2)
        if prior_xy.size:
            prior_weight = (
                np.ones((prior_xy.shape[0],), dtype=np.float32)
                if query_location_prior_weights is None
                else np.asarray(
                    query_location_prior_weights, dtype=np.float32
                ).reshape(-1)
            )
            if prior_weight.shape != (prior_xy.shape[0],):
                raise ValueError("query location prior weights differ")
            prior_weight = np.maximum(prior_weight, 1e-8)
            prior_weight /= np.sum(prior_weight)
            prior_cells = torch.from_numpy(
                (prior_xy + 0.5) / float(feature_stride) - 0.5
            ).to(device=device, dtype=dtype)
            prior_log_weight = torch.log(
                torch.from_numpy(prior_weight).to(
                    device=device, dtype=dtype
                )
            )
            yy, xx = torch.meshgrid(
                torch.arange(
                    query_feature.shape[1], device=device, dtype=dtype
                ),
                torch.arange(
                    query_feature.shape[2], device=device, dtype=dtype
                ),
                indexing="ij",
            )
            distance2 = (
                (xx[None] - prior_cells[:, 0, None, None]) ** 2
                + (yy[None] - prior_cells[:, 1, None, None]) ** 2
            )
            sigma_cells = max(
                float(location_prior_sigma_px) / float(feature_stride), 1e-3
            )
            location_log_prior = torch.logsumexp(
                prior_log_weight[:, None, None]
                - 0.5 * distance2 / (sigma_cells * sigma_cells),
                dim=0,
            )
            location_log_prior -= torch.max(location_log_prior)
            if location_window_radius_px is not None:
                radius_cells = max(
                    float(location_window_radius_px)
                    / float(feature_stride),
                    0.0,
                )
                location_allowed = torch.any(
                    distance2 <= radius_cells * radius_cells, dim=0
                )
    physical_aspect = float(
        max(atlas.extents[row, 0], 1e-4)
        / max(atlas.extents[row, 1], 1e-4)
    )
    templates = [
        template
        for source_feature, source_mask in _atlas_appearance_sources(
            atlas,
            row,
            atlas_feature,
            atlas_mask,
            config,
        )
        for template in _template_grid(
            source_feature,
            source_mask,
            physical_aspect,
            config,
        )
    ]
    if not templates:
        return ()
    groups: dict[tuple[int, int], list[_Template]] = {}
    for template in templates:
        groups.setdefault(
            (int(template.mask.shape[0]), int(template.mask.shape[1])), []
        ).append(template)
    candidates: list[_Candidate] = []
    weighted_query = query_feature[None] * matchability[None, None]
    geometric_query = torch.ones_like(matchability)[None, None]
    for (kernel_height, kernel_width), values in groups.items():
        kernels = torch.stack([value.feature for value in values], dim=0)
        masks = torch.stack([value.mask for value in values], dim=0)[:, None]
        numerator = F.conv2d(
            _same_padding(weighted_query, kernel_height, kernel_width),
            kernels,
        )[0]
        denominator = F.conv2d(
            _same_padding(
                matchability[None, None], kernel_height, kernel_width
            ),
            masks,
        )[0]
        overlap = F.conv2d(
            _same_padding(geometric_query, kernel_height, kernel_width),
            masks,
        )[0]
        support = torch.as_tensor(
            [value.support for value in values],
            device=device,
            dtype=dtype,
        )[:, None, None]
        mean_similarity = numerator / torch.clamp(
            denominator, min=1e-6
        )
        # A one-cell accidental match must not outrank a coherent regional
        # layout.  This is the significance of the mean similarity above the
        # empirical background, with a sub-square-root support exponent to
        # account for spatially correlated VFM cells.
        response = (
            mean_similarity - float(config.background_similarity)
        ) * torch.clamp(overlap, min=1.0) ** float(
            config.support_score_power
        )
        if location_log_prior is not None:
            response = response + float(
                location_prior_strength
            ) * location_log_prior[None]
        valid = (
            overlap
            >= support * float(config.minimum_overlap_fraction)
        ) & (denominator > 1e-6)
        if location_allowed is not None:
            valid = valid & location_allowed[None]
        response = torch.where(
            valid, response, torch.full_like(response, -torch.inf)
        )
        flat = response.reshape(response.shape[0], -1)
        peak_count = min(
            int(config.per_transform_peaks), int(flat.shape[1])
        )
        peak_score, peak_index = torch.topk(
            flat, k=peak_count, dim=1, sorted=True
        )
        query_width = int(response.shape[2])
        for template_row, template in enumerate(values):
            for local_peak in range(peak_count):
                score = float(peak_score[template_row, local_peak].item())
                if not np.isfinite(score):
                    continue
                index = int(peak_index[template_row, local_peak].item())
                y, x = divmod(index, query_width)
                center, covariance = _local_peak_moments(
                    response[template_row],
                    x,
                    y,
                    float(config.local_peak_temperature),
                )
                overlap_fraction = float(
                    overlap[template_row, y, x].item()
                    / max(template.support, 1e-6)
                )
                candidates.append(
                    _Candidate(
                        score=score,
                        center_xy=center,
                        covariance_xy=covariance,
                        width=template.width,
                        height=template.height,
                        angle_deg=template.angle_deg,
                        linear=template.linear,
                        support_fraction=overlap_fraction,
                    )
                )
    candidates.sort(key=lambda value: -value.score)
    selected = _select_distinct_candidates(candidates, config)
    if not selected:
        return ()
    logits = np.asarray(
        [value.score for value in selected]
        + [float(config.diagnostic_null_score)],
        dtype=np.float64,
    ) / float(config.posterior_temperature)
    logits -= float(np.max(logits))
    probability = np.exp(logits)
    probability /= max(float(np.sum(probability)), 1e-12)
    null_probability = float(probability[-1])
    result = []
    for candidate, mode_probability in zip(selected, probability[:-1]):
        transform = np.c_[
            candidate.linear,
            candidate.center_xy,
        ].astype(np.float32)
        scale = np.linalg.norm(candidate.linear, axis=0)
        result.append(
            MapletFrameMatch(
                chart_id=int(chart_id),
                canonical_to_query=transform,
                query_center_xy=candidate.center_xy,
                scale_xy=scale,
                in_plane_rotation_deg=candidate.angle_deg,
                covariance_xy=candidate.covariance_xy,
                score=candidate.score,
                probability=float(mode_probability),
                null_probability=null_probability,
                support_fraction=candidate.support_fraction,
                feature_level=str(feature_level),
                feature_stride=int(feature_stride),
            )
        )
    return tuple(result)


def refine_maplet_frame_matches(
    atlas: MapletFeatureAtlasBank,
    matches: Sequence[MapletFrameMatch],
    query_feature: torch.Tensor,
    query_matchability: torch.Tensor | None = None,
    *,
    iterations: int = 40,
    maximum_translation_cells: float = 3.0,
    maximum_linear_fraction: float = 0.30,
    learning_rate: float = 0.08,
    background_similarity: float = 0.25,
    support_score_power: float = 0.0,
    appearance_temperature: float = 0.07,
) -> tuple[MapletFrameMatch, ...]:
    """Continuously refine discrete frame modes by regional feature sampling."""

    if not matches:
        return ()
    if query_feature.ndim == 4:
        if query_feature.shape[0] != 1:
            raise ValueError("query feature batch size must be one")
        query_feature = query_feature[0]
    if query_feature.ndim != 3:
        raise ValueError("query_feature must have shape (C,H,W)")
    if float(appearance_temperature) <= 0.0:
        raise ValueError("appearance_temperature must be positive")
    if int(iterations) <= 0:
        return tuple(matches)
    device, dtype = query_feature.device, query_feature.dtype
    if query_matchability is None:
        matchability = torch.ones(
            query_feature.shape[-2:], device=device, dtype=dtype
        )
    else:
        matchability = query_matchability
        if matchability.ndim == 4:
            matchability = matchability[0, 0]
        elif matchability.ndim == 3:
            matchability = matchability[0]
        if matchability.shape != query_feature.shape[-2:]:
            matchability = F.interpolate(
                matchability[None, None],
                size=query_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        matchability = matchability.to(device=device, dtype=dtype)
    row_by_id = {
        int(value): row
        for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    refined = []
    for match in matches:
        row = row_by_id.get(int(match.chart_id))
        if row is None:
            continue
        valid = (
            np.asarray(atlas.valid_mask[row], dtype=bool)
            & (np.asarray(atlas.support_count[row]) > 0)
            & (np.linalg.norm(atlas.features[row], axis=0) > 0.5)
        )
        y, x = np.nonzero(valid)
        if x.size < 3:
            refined.append(match)
            continue
        canonical = np.stack(
            [
                x / max(atlas.width - 1, 1) * 2.0 - 1.0,
                y / max(atlas.height - 1, 1) * 2.0 - 1.0,
            ],
            axis=1,
        ).astype(np.float32)
        map_feature = torch.from_numpy(
            atlas.features[row].transpose(1, 2, 0)[valid]
        ).to(device=device, dtype=dtype)
        if atlas.mode_features is not None:
            map_mode_feature = torch.from_numpy(
                atlas.mode_features[row].transpose(2, 3, 0, 1)[valid]
            ).to(device=device, dtype=dtype)
            map_mode_weight = torch.from_numpy(
                atlas.mode_weights[row].transpose(1, 2, 0)[valid]
            ).to(device=device, dtype=dtype)
            map_mode_valid = torch.from_numpy(
                atlas.mode_valid_mask[row].transpose(1, 2, 0)[valid]
            ).to(device=device)
        else:
            map_mode_feature = None
            map_mode_weight = None
            map_mode_valid = None
        canonical_tensor = torch.from_numpy(canonical).to(
            device=device, dtype=dtype
        )
        initial = torch.from_numpy(
            np.asarray(match.canonical_to_query, dtype=np.float32)
        ).to(device=device, dtype=dtype)
        raw = torch.zeros(
            (6,), device=device, dtype=dtype, requires_grad=True
        )
        optimizer = torch.optim.Adam([raw], lr=float(learning_rate))

        def transform_from_raw() -> torch.Tensor:
            translation = initial[:, 2] + float(
                maximum_translation_cells
            ) * torch.tanh(raw[:2])
            bounded = float(maximum_linear_fraction) * torch.tanh(
                raw[2:]
            )
            delta = torch.stack(
                [
                    torch.stack(
                        [torch.exp(bounded[0]), bounded[1]]
                    ),
                    torch.stack(
                        [bounded[2], torch.exp(bounded[3])]
                    ),
                ]
            )
            linear = initial[:, :2] @ delta
            return torch.cat([linear, translation[:, None]], dim=1)

        last_similarity = None
        for _iteration in range(int(iterations)):
            optimizer.zero_grad(set_to_none=True)
            transform = transform_from_raw()
            query_xy = (
                canonical_tensor @ transform[:, :2].T
                + transform[:, 2][None]
            )
            grid = torch.stack(
                [
                    (query_xy[:, 0] + 0.5)
                    / query_feature.shape[2]
                    * 2.0
                    - 1.0,
                    (query_xy[:, 1] + 0.5)
                    / query_feature.shape[1]
                    * 2.0
                    - 1.0,
                ],
                dim=1,
            )[None, :, None]
            sampled = F.grid_sample(
                query_feature[None],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0, :, :, 0].T
            sampled_matchability = F.grid_sample(
                matchability[None, None],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0, 0, :, 0]
            inside = (
                (query_xy[:, 0] >= 0.0)
                & (query_xy[:, 0] <= query_feature.shape[2] - 1)
                & (query_xy[:, 1] >= 0.0)
                & (query_xy[:, 1] <= query_feature.shape[1] - 1)
            )
            weight = sampled_matchability * inside.to(dtype)
            mean_cell_similarity = torch.sum(
                map_feature * sampled, dim=1
            )
            if map_mode_feature is not None:
                mode_similarity = torch.einsum(
                    "nkc,nc->nk", map_mode_feature, sampled
                )
                mode_logit = (
                    mode_similarity / float(appearance_temperature)
                    + torch.log(torch.clamp(map_mode_weight, min=1e-8))
                )
                mode_logit = torch.where(
                    map_mode_valid,
                    mode_logit,
                    torch.full_like(mode_logit, -torch.inf),
                )
                has_mode = torch.any(map_mode_valid, dim=1)
                cell_similarity = torch.where(
                    has_mode,
                    float(appearance_temperature)
                    * torch.logsumexp(mode_logit, dim=1),
                    mean_cell_similarity,
                )
            else:
                cell_similarity = mean_cell_similarity
            similarity = torch.sum(
                weight * cell_similarity
            ) / torch.clamp(torch.sum(weight), min=1e-6)
            loss = -similarity + 0.002 * torch.mean(raw * raw)
            loss.backward()
            optimizer.step()
            last_similarity = similarity.detach()
        transform_value = transform_from_raw().detach().cpu().numpy()
        linear = transform_value[:, :2]
        center = transform_value[:, 2]
        scale = np.linalg.norm(linear, axis=0)
        angle = float(
            np.rad2deg(np.arctan2(linear[1, 0], linear[0, 0])) % 360.0
        )
        projected_support = float(
            max(
                4.0
                * abs(float(np.linalg.det(linear)))
                * float(np.mean(valid)),
                1.0,
            )
        )
        mean_similarity = (
            float(last_similarity.item())
            if last_similarity is not None
            else float(match.score)
        )
        score = (
            mean_similarity - float(background_similarity)
        ) * projected_support ** float(support_score_power)
        refined.append(
            MapletFrameMatch(
                chart_id=int(match.chart_id),
                canonical_to_query=transform_value,
                query_center_xy=center,
                scale_xy=scale,
                in_plane_rotation_deg=angle,
                covariance_xy=match.covariance_xy,
                score=float(score),
                probability=float(match.probability),
                null_probability=float(match.null_probability),
                support_fraction=float(match.support_fraction),
                feature_level=match.feature_level,
                feature_stride=int(match.feature_stride),
            )
        )
    refined.sort(key=lambda value: -value.score)
    logits = np.asarray(
        [value.score for value in refined]
        + [float(background_similarity)],
        dtype=np.float64,
    )
    logits -= float(np.max(logits))
    posterior = np.exp(logits / 0.20)
    posterior /= max(float(np.sum(posterior)), 1e-12)
    null_probability = float(posterior[-1])
    return tuple(
        MapletFrameMatch(
            chart_id=value.chart_id,
            canonical_to_query=value.canonical_to_query,
            query_center_xy=value.query_center_xy,
            scale_xy=value.scale_xy,
            in_plane_rotation_deg=value.in_plane_rotation_deg,
            covariance_xy=value.covariance_xy,
            score=value.score,
            probability=float(probability),
            null_probability=null_probability,
            support_fraction=value.support_fraction,
            feature_level=value.feature_level,
            feature_stride=value.feature_stride,
        )
        for value, probability in zip(refined, posterior[:-1])
    )


def chart_canonical_control_points(
    atlas: MapletFeatureAtlasBank, chart_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical UV corners and their planar chart-frame 3D points."""

    rows = np.flatnonzero(atlas.maplet_ids == int(chart_id))
    if rows.size != 1:
        raise ValueError(f"unknown or duplicate metric chart ID: {chart_id}")
    row = int(rows[0])
    uv = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    xyz = (
        atlas.centers[row][None]
        + uv[:, 0:1] * atlas.extents[row, 0] * atlas.frames[row, 0][None]
        + uv[:, 1:2] * atlas.extents[row, 1] * atlas.frames[row, 1][None]
    )
    return uv.astype(np.float32), xyz.astype(np.float32)


def fit_oriented_chart_frame(
    canonical_uv: np.ndarray, query_xy: np.ndarray
) -> np.ndarray:
    """Fit the closest rotation plus independent canonical-axis scales."""

    uv = np.asarray(canonical_uv, dtype=np.float64).reshape(-1, 2)
    xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    if uv.shape != xy.shape or uv.shape[0] < 3:
        raise ValueError("frame fit needs at least three paired 2D controls")
    uv_center = np.mean(uv, axis=0)
    xy_center = np.mean(xy, axis=0)
    linear = np.linalg.lstsq(
        uv - uv_center, xy - xy_center, rcond=None
    )[0].T
    first_angle = np.arctan2(linear[1, 0], linear[0, 0])
    second_angle = np.arctan2(-linear[0, 1], linear[1, 1])
    mean_angle = np.arctan2(
        np.sin(first_angle) + np.sin(second_angle),
        np.cos(first_angle) + np.cos(second_angle),
    )
    cosine, sine = np.cos(mean_angle), np.sin(mean_angle)
    axis_u = np.asarray([cosine, sine])
    axis_v = np.asarray([-sine, cosine])
    scale_u = max(float(np.dot(linear[:, 0], axis_u)), 1e-4)
    scale_v = max(float(np.dot(linear[:, 1], axis_v)), 1e-4)
    fitted = np.asarray(
        [
            [cosine * scale_u, -sine * scale_v, 0.0],
            [sine * scale_u, cosine * scale_v, 0.0],
        ],
        dtype=np.float64,
    )
    fitted[:, 2] = xy_center - fitted[:, :2] @ uv_center
    return fitted.astype(np.float32)


def ground_truth_chart_frame(
    atlas: MapletFeatureAtlasBank,
    chart_id: int,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    feature_stride: int,
    feature_level: str = "oracle",
    model: str = "oriented_similarity",
) -> MapletFrameMatch | None:
    """Fit the search model to the chart's exact GT corner projections."""

    uv, xyz = chart_canonical_control_points(atlas, chart_id)
    pixels, depth = project_world_points(xyz, pose_w2c, camera)
    if np.any(depth <= 0.0) or not np.all(np.isfinite(pixels)):
        return None
    query = (pixels + 0.5) / float(feature_stride) - 0.5
    model_name = str(model)
    if model_name == "oriented_similarity":
        transform = fit_oriented_chart_frame(uv, query)
        homography = None
    elif model_name == "affine":
        design = np.c_[uv, np.ones((uv.shape[0],))]
        transform = np.linalg.lstsq(design, query, rcond=None)[0].T.astype(
            np.float32
        )
        homography = None
    elif model_name == "homography":
        homography = cv2.getPerspectiveTransform(
            uv.astype(np.float32), query.astype(np.float32)
        )
        transform = np.linalg.lstsq(
            np.c_[uv, np.ones((uv.shape[0],))], query, rcond=None
        )[0].T.astype(np.float32)
    else:
        raise ValueError(f"unknown ground-truth frame model: {model_name}")
    first = transform[:, 0]
    second = transform[:, 1]
    scale = np.asarray(
        [np.linalg.norm(first), np.linalg.norm(second)], dtype=np.float32
    )
    angle = float(np.rad2deg(np.arctan2(first[1], first[0])) % 360.0)
    center = transform[:, 2]
    return MapletFrameMatch(
        chart_id=int(chart_id),
        canonical_to_query=transform,
        query_center_xy=center,
        scale_xy=scale,
        in_plane_rotation_deg=angle,
        covariance_xy=np.eye(2, dtype=np.float32) * 1e-6,
        score=1.0,
        probability=1.0,
        null_probability=0.0,
        support_fraction=1.0,
        feature_level=str(feature_level),
        feature_stride=int(feature_stride),
        canonical_homography=homography,
    )


def frame_control_error_px(
    predicted: MapletFrameMatch, target: MapletFrameMatch
) -> float:
    uv = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float32,
    )
    predicted_xy = predicted.canonical_points_to_pixels(uv)
    target_xy = target.canonical_points_to_pixels(uv)
    return float(np.mean(np.linalg.norm(predicted_xy - target_xy, axis=1)))


def frame_parameter_errors(
    predicted: MapletFrameMatch, target: MapletFrameMatch
) -> dict[str, float]:
    stride = float(target.feature_stride)
    return {
        "center_error_px": float(
            np.linalg.norm(
                predicted.query_center_xy - target.query_center_xy
            )
            * stride
        ),
        "log_scale_error": float(
            np.mean(
                np.abs(
                    np.log(
                        np.maximum(predicted.scale_xy, 1e-6)
                        / np.maximum(target.scale_xy, 1e-6)
                    )
                )
            )
        ),
        "rotation_error_deg": _angular_distance_deg(
            predicted.in_plane_rotation_deg,
            target.in_plane_rotation_deg,
        ),
        "control_error_px": frame_control_error_px(predicted, target),
    }


def _pose_from_local_chart(
    atlas: MapletFeatureAtlasBank,
    match: MapletFrameMatch,
    camera: ColmapCamera,
) -> list[FramePoseHypothesis]:
    rows = np.flatnonzero(atlas.maplet_ids == int(match.chart_id))
    if rows.size != 1:
        return []
    row = int(rows[0])
    uv, _world = chart_canonical_control_points(atlas, match.chart_id)
    local = np.c_[
        uv[:, 0] * atlas.extents[row, 0],
        uv[:, 1] * atlas.extents[row, 1],
        np.zeros((uv.shape[0],), dtype=np.float32),
    ].astype(np.float64)
    pixels = match.canonical_points_to_pixels(uv).astype(np.float64)
    matrix, distortion = camera_matrix_and_distortion(camera)
    try:
        output = cv2.solvePnPGeneric(
            local,
            pixels,
            matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return []
    if not bool(output[0]):
        return []
    hypotheses = []
    for rvec, tvec in zip(output[1], output[2]):
        rotation_local, _ = cv2.Rodrigues(
            np.asarray(rvec, dtype=np.float64)
        )
        rotation_world = rotation_local @ atlas.frames[row].astype(
            np.float64
        )
        translation_world = np.asarray(
            tvec, dtype=np.float64
        ).reshape(3) - rotation_world @ atlas.centers[row].astype(
            np.float64
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation_world
        pose[:3, 3] = translation_world
        depth = (
            _world.astype(np.float64) @ rotation_world.T
            + translation_world[None]
        )[:, 2]
        projected, _ = project_world_points(_world, pose, camera)
        error = float(
            np.mean(np.linalg.norm(projected - pixels, axis=1))
        )
        hypotheses.append(
            FramePoseHypothesis(
                pose_w2c=pose,
                score=float(match.score),
                source_chart_ids=(int(match.chart_id),),
                reprojection_error_px=error,
                positive_depth=bool(np.all(depth > 0.0)),
            )
        )
    return hypotheses


def frame_matches_to_pose_hypotheses(
    atlas: MapletFeatureAtlasBank,
    matches: Sequence[MapletFrameMatch],
    camera: ColmapCamera,
    *,
    include_individual_poses: bool = True,
    include_grouped_pose: bool = True,
    include_grouped_center_pose: bool = False,
) -> tuple[FramePoseHypothesis, ...]:
    """Convert regional frame constraints to single- and multi-chart poses."""

    hypotheses = (
        [
            hypothesis
            for match in matches
            for hypothesis in _pose_from_local_chart(atlas, match, camera)
        ]
        if bool(include_individual_poses)
        else []
    )
    if include_grouped_pose and len(matches) >= 2:
        world_parts = []
        image_parts = []
        for match in matches:
            uv, world = chart_canonical_control_points(
                atlas, match.chart_id
            )
            world_parts.append(world.astype(np.float64))
            image_parts.append(
                match.canonical_points_to_pixels(uv).astype(np.float64)
            )
        world = np.concatenate(world_parts, axis=0)
        image = np.concatenate(image_parts, axis=0)
        origin = np.mean(world, axis=0)
        matrix, distortion = camera_matrix_and_distortion(camera)
        try:
            success, rvec, tvec = cv2.solvePnP(
                world - origin[None],
                image,
                matrix,
                distortion,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if success:
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        world - origin[None],
                        image,
                        matrix,
                        distortion,
                        rvec,
                        tvec,
                    )
                except cv2.error:
                    pass
                rotation, _ = cv2.Rodrigues(rvec)
                translation = np.asarray(tvec).reshape(3) - rotation @ origin
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = rotation
                pose[:3, 3] = translation
                projected, depth = project_world_points(world, pose, camera)
                error = float(
                    np.mean(np.linalg.norm(projected - image, axis=1))
                )
                hypotheses.append(
                    FramePoseHypothesis(
                        pose_w2c=pose,
                        score=float(
                            np.sum([match.score for match in matches])
                        ),
                        source_chart_ids=tuple(
                            int(match.chart_id) for match in matches
                        ),
                        reprojection_error_px=error,
                        positive_depth=bool(np.all(depth > 0.0)),
                        control_model="frame_controls",
                    )
                )
        except cv2.error:
            pass
    if include_grouped_center_pose and len(matches) >= 3:
        world = []
        image = []
        for match in matches:
            rows = np.flatnonzero(
                atlas.maplet_ids == int(match.chart_id)
            )
            if rows.size != 1:
                continue
            world.append(atlas.centers[int(rows[0])])
            image.append(match.canonical_points_to_pixels([[0.0, 0.0]])[0])
        if len(world) == len(matches):
            world_array = np.asarray(world, dtype=np.float64)
            image_array = np.asarray(image, dtype=np.float64)
            origin = np.mean(world_array, axis=0)
            matrix, distortion = camera_matrix_and_distortion(camera)
            try:
                output = cv2.solvePnPGeneric(
                    world_array - origin[None],
                    image_array,
                    matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
            except cv2.error:
                output = (False, (), ())
            if bool(output[0]):
                for rvec, tvec in zip(output[1], output[2]):
                    if len(matches) >= 4:
                        try:
                            rvec, tvec = cv2.solvePnPRefineLM(
                                world_array - origin[None],
                                image_array,
                                matrix,
                                distortion,
                                np.asarray(rvec, dtype=np.float64),
                                np.asarray(tvec, dtype=np.float64),
                            )
                        except cv2.error:
                            pass
                    rotation, _ = cv2.Rodrigues(
                        np.asarray(rvec, dtype=np.float64)
                    )
                    translation = (
                        np.asarray(tvec, dtype=np.float64).reshape(3)
                        - rotation @ origin
                    )
                    pose = np.eye(4, dtype=np.float64)
                    pose[:3, :3] = rotation
                    pose[:3, 3] = translation
                    projected, depth = project_world_points(
                        world_array, pose, camera
                    )
                    error = float(
                        np.mean(
                            np.linalg.norm(
                                projected - image_array, axis=1
                            )
                        )
                    )
                    hypotheses.append(
                        FramePoseHypothesis(
                            pose_w2c=pose,
                            score=float(
                                np.sum(
                                    [match.score for match in matches]
                                )
                            ),
                            source_chart_ids=tuple(
                                int(match.chart_id) for match in matches
                            ),
                            reprojection_error_px=error,
                            positive_depth=bool(np.all(depth > 0.0)),
                            control_model="chart_centers",
                        )
                    )
    hypotheses = [
        value
        for value in hypotheses
        if value.positive_depth
        and np.all(np.isfinite(value.pose_w2c))
        and np.isfinite(value.reprojection_error_px)
    ]
    hypotheses.sort(
        key=lambda value: (
            -value.score,
            value.reprojection_error_px,
            -len(value.source_chart_ids),
        )
    )
    return tuple(hypotheses)


def pose_hypotheses_for_mode_sets(
    atlas: MapletFeatureAtlasBank,
    mode_sets: Iterable[Sequence[MapletFrameMatch]],
    camera: ColmapCamera,
) -> tuple[FramePoseHypothesis, ...]:
    """Evaluate several chart-mode groups without collapsing multimodality."""

    result = []
    for matches in mode_sets:
        result.extend(
            frame_matches_to_pose_hypotheses(
                atlas,
                matches,
                camera,
                include_individual_poses=len(matches) == 1,
                include_grouped_pose=len(matches) >= 2,
                include_grouped_center_pose=len(matches) >= 3,
            )
        )
    result.sort(
        key=lambda value: (
            -value.score,
            value.reprojection_error_px,
            -len(value.source_chart_ids),
        )
    )
    return tuple(result)
