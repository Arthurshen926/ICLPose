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
from typing import Iterable, Mapping, Sequence

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
from feature_extract.vfm.localization_v6.se3_update import (
    projection_jacobian,
    se3_exp,
)
from feature_extract.vfm.localization_v6.structured_frame_refiner import (
    StructuredFrameRefiner,
    structured_frame_correlation,
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
    support_canonical_hull: np.ndarray | None = None
    control_covariance_px: np.ndarray | None = None
    identity_probability: float = 1.0

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
        support_hull = (
            None
            if self.support_canonical_hull is None
            else np.asarray(
                self.support_canonical_hull, dtype=np.float64
            ).reshape(-1, 2)
        )
        control_covariance = (
            None
            if self.control_covariance_px is None
            else np.asarray(
                self.control_covariance_px, dtype=np.float64
            )
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
        if support_hull is not None and (
            support_hull.shape[0] < 3
            or not np.all(np.isfinite(support_hull))
        ):
            raise ValueError("invalid canonical support hull")
        if control_covariance is not None and (
            control_covariance.shape != (8, 8)
            or not np.all(np.isfinite(control_covariance))
        ):
            raise ValueError("invalid chart-control covariance")
        if not 0.0 <= float(self.identity_probability) <= 1.0:
            raise ValueError("identity_probability must lie in [0,1]")
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
        if support_hull is not None:
            object.__setattr__(
                self,
                "support_canonical_hull",
                support_hull.astype(np.float32),
            )
        if control_covariance is not None:
            object.__setattr__(
                self,
                "control_covariance_px",
                control_covariance.astype(np.float32),
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


def rescale_maplet_frame_match(
    match: MapletFrameMatch,
    *,
    feature_stride: int,
    feature_level: str,
) -> MapletFrameMatch:
    """Express one frame mode on another feature lattice.

    Feature-cell coordinates refer to cell centres:

    ``pixel = (cell + 0.5) * stride - 0.5``.

    Consequently, changing stride is not a plain multiplication around the
    image origin.  This conversion preserves every canonical control point in
    image-pixel coordinates, including projective modes, while converting
    cell-domain uncertainty to the target lattice.  Pixel-domain control
    covariance is already resolution independent and is therefore retained.
    """

    source_stride = int(match.feature_stride)
    target_stride = int(feature_stride)
    if source_stride <= 0 or target_stride <= 0:
        raise ValueError("feature strides must be positive")
    ratio = float(source_stride) / float(target_stride)
    offset = 0.5 * (ratio - 1.0)

    transform = np.asarray(
        match.canonical_to_query, dtype=np.float64
    ).copy()
    transform[:, :2] *= ratio
    transform[:, 2] = transform[:, 2] * ratio + offset

    homography = None
    if match.canonical_homography is not None:
        lattice_transform = np.asarray(
            [
                [ratio, 0.0, offset],
                [0.0, ratio, offset],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        homography = lattice_transform @ np.asarray(
            match.canonical_homography, dtype=np.float64
        )
        if abs(float(homography[2, 2])) <= 1e-12:
            raise ValueError("rescaled frame homography is singular")
        homography /= float(homography[2, 2])

    return MapletFrameMatch(
        chart_id=int(match.chart_id),
        canonical_to_query=transform,
        query_center_xy=(
            np.asarray(match.query_center_xy, dtype=np.float64) * ratio
            + offset
        ),
        scale_xy=np.asarray(match.scale_xy, dtype=np.float64) * ratio,
        in_plane_rotation_deg=float(match.in_plane_rotation_deg),
        covariance_xy=(
            np.asarray(match.covariance_xy, dtype=np.float64) * ratio**2
        ),
        score=float(match.score),
        probability=float(match.probability),
        null_probability=float(match.null_probability),
        support_fraction=float(match.support_fraction),
        feature_level=str(feature_level),
        feature_stride=target_stride,
        canonical_homography=homography,
        support_canonical_hull=match.support_canonical_hull,
        control_covariance_px=match.control_covariance_px,
        identity_probability=float(match.identity_probability),
    )


def _refined_frame_posterior(
    original: Sequence[MapletFrameMatch],
    refined: Sequence[MapletFrameMatch],
    *,
    temperature: float,
) -> tuple[np.ndarray, float]:
    """Re-score refined modes while preserving the chart-null reference.

    Refinement contributes evidence to non-null frame modes; it does not
    observe a new negative chart.  Recover the common input null score from

    ``log(p_mode / p_null) = (score_mode - score_null) / temperature``

    and keep that score fixed.  A no-op refiner therefore leaves the input
    posterior unchanged instead of making every chart nearly certain.
    """

    values = tuple(refined)
    if not values:
        return np.zeros((0,), dtype=np.float64), 1.0
    scale = float(temperature)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("posterior temperature must be positive")
    inferred_null_scores = []
    for match in original:
        mode_probability = float(match.probability)
        null_probability = float(match.null_probability)
        if (
            np.isfinite(match.score)
            and np.isfinite(mode_probability)
            and np.isfinite(null_probability)
            and mode_probability > 0.0
            and null_probability > 0.0
        ):
            inferred_null_scores.append(
                float(match.score)
                - scale * np.log(mode_probability / null_probability)
            )
    # Synthetic oracle matches can omit a finite null posterior.  Production
    # global frame matches always provide one; zero remains a diagnostic-only
    # fallback for the former.
    null_score = (
        float(np.median(inferred_null_scores))
        if inferred_null_scores
        else 0.0
    )
    logits = np.asarray(
        [float(value.score) for value in values] + [null_score],
        dtype=np.float64,
    )
    logits /= scale
    logits -= float(np.max(logits))
    posterior = np.exp(logits)
    posterior /= max(float(np.sum(posterior)), 1e-12)
    return posterior[:-1], float(posterior[-1])


@dataclass(frozen=True)
class FramePoseHypothesis:
    pose_w2c: np.ndarray
    score: float
    source_chart_ids: tuple[int, ...]
    reprojection_error_px: float
    positive_depth: bool
    control_model: str = "chart_factor"
    seed_model: str = "unknown"
    factor_cost: float = float("inf")
    mode_support_count: int = 1
    mode_member_count: int = 1


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


def _support_hull_and_control_covariance(
    canonical_uv: np.ndarray,
    query_xy: np.ndarray,
    query_shape_hw: tuple[int, int],
    center_covariance_cells: np.ndarray,
    *,
    feature_stride: int,
) -> tuple[np.ndarray | None, np.ndarray, float]:
    """Summarize actually observed canonical support and correlated error.

    The four chart corners share a common frame prediction.  Their covariance
    is therefore block correlated; corners outside the observed canonical
    hull receive an additional extrapolation variance instead of being treated
    as four independent, equally precise point observations.
    """

    uv = np.asarray(canonical_uv, dtype=np.float64).reshape(-1, 2)
    xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    height, width = (int(query_shape_hw[0]), int(query_shape_hw[1]))
    inside = (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= width - 1.0)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= height - 1.0)
    )
    support_fraction = float(np.mean(inside)) if inside.size else 0.0
    hull = None
    if int(np.sum(inside)) >= 3:
        candidate_hull = cv2.convexHull(
            uv[inside].astype(np.float32)
        ).reshape(-1, 2)
        # Three or more visible samples may still be collinear (common for a
        # thin, partially observed chart). OpenCV then returns a one/two-point
        # hull, which is not a polygon. Treat it as unknown support and retain
        # the conservative extrapolation covariance instead of aborting the
        # entire localization query.
        if (
            candidate_hull.shape[0] >= 3
            and abs(
                float(
                    cv2.contourArea(
                        candidate_hull.astype(np.float32)
                    )
                )
            )
            > 1e-8
        ):
            hull = candidate_hull
    corners = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    shared = (
        np.asarray(center_covariance_cells, dtype=np.float64).reshape(2, 2)
        * float(feature_stride) ** 2
    )
    shared += np.eye(2, dtype=np.float64) * 0.25
    covariance = np.zeros((8, 8), dtype=np.float64)
    for first in range(4):
        for second in range(4):
            covariance[
                2 * first : 2 * first + 2,
                2 * second : 2 * second + 2,
            ] += shared
    base_variance = max(
        (0.35 * float(feature_stride)) ** 2
        / max(support_fraction, 0.10),
        1.0,
    )
    for row, corner in enumerate(corners):
        extrapolated = (
            hull is None
            or cv2.pointPolygonTest(
                hull.astype(np.float32),
                (float(corner[0]), float(corner[1])),
                False,
            )
            < 0.0
        )
        covariance[
            2 * row : 2 * row + 2,
            2 * row : 2 * row + 2,
        ] += np.eye(2) * base_variance * (9.0 if extrapolated else 1.0)
    covariance += np.eye(8, dtype=np.float64) * 1e-4
    return (
        None if hull is None else hull.astype(np.float32),
        covariance.astype(np.float32),
        support_fraction,
    )


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
    support_y, support_x = np.nonzero(
        np.asarray(atlas_mask.detach().cpu().numpy(), dtype=np.float32) > 0.0
    )
    canonical_support = np.stack(
        [
            support_x / max(atlas.width - 1, 1) * 2.0 - 1.0,
            support_y / max(atlas.height - 1, 1) * 2.0 - 1.0,
        ],
        axis=1,
    ).astype(np.float32)
    result = []
    for candidate, mode_probability in zip(selected, probability[:-1]):
        transform = np.c_[
            candidate.linear,
            candidate.center_xy,
        ].astype(np.float32)
        scale = np.linalg.norm(candidate.linear, axis=0)
        support_query = (
            canonical_support @ transform[:, :2].T
            + transform[:, 2][None]
        )
        support_hull, control_covariance, support_fraction = (
            _support_hull_and_control_covariance(
                canonical_support,
                support_query,
                tuple(query_feature.shape[-2:]),
                candidate.covariance_xy,
                feature_stride=int(feature_stride),
            )
        )
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
                support_fraction=min(
                    float(candidate.support_fraction), support_fraction
                ),
                feature_level=str(feature_level),
                feature_stride=int(feature_stride),
                support_canonical_hull=support_hull,
                control_covariance_px=control_covariance,
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
    enable_projective: bool = True,
    maximum_projective_fraction: float = 0.35,
    out_of_view_penalty: float = 0.20,
    posterior_temperature: float = 0.06,
) -> tuple[MapletFrameMatch, ...]:
    """Refine a chart projection with fixed-denominator projective evidence.

    Every canonical atlas cell remains in the objective.  Moving a difficult
    cell outside the query therefore incurs ``out_of_view_penalty`` rather
    than silently deleting it from the denominator.  The optional two
    projective parameters upgrade the discrete affine seed to a full planar
    homography.
    """

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
    if (
        float(maximum_projective_fraction) < 0.0
        or float(out_of_view_penalty) < 0.0
        or float(posterior_temperature) <= 0.0
    ):
        raise ValueError("invalid projective refinement parameters")
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
        initial_homography = (
            np.asarray(match.canonical_homography, dtype=np.float32)
            if match.canonical_homography is not None
            else np.vstack(
                [
                    np.asarray(
                        match.canonical_to_query, dtype=np.float32
                    ),
                    np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                ]
            )
        )
        initial_homography /= max(
            abs(float(initial_homography[2, 2])), 1e-8
        )
        initial = torch.from_numpy(initial_homography).to(
            device=device, dtype=dtype
        )
        parameter_count = 8 if bool(enable_projective) else 6
        raw = torch.zeros(
            (parameter_count,),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([raw], lr=float(learning_rate))

        def homography_from_raw() -> torch.Tensor:
            translation = initial[:2, 2] + float(
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
            linear = initial[:2, :2] @ delta
            if bool(enable_projective):
                projective = initial[2, :2] + float(
                    maximum_projective_fraction
                ) * torch.tanh(raw[6:8])
            else:
                projective = initial[2, :2]
            return torch.cat(
                [
                    torch.cat(
                        [linear, translation[:, None]], dim=1
                    ),
                    torch.cat(
                        [
                            projective,
                            torch.ones(
                                (1,), device=device, dtype=dtype
                            ),
                        ]
                    )[None],
                ],
                dim=0,
            )

        def project_canonical(homography: torch.Tensor) -> torch.Tensor:
            homogeneous = torch.cat(
                [
                    canonical_tensor,
                    torch.ones(
                        (canonical_tensor.shape[0], 1),
                        device=device,
                        dtype=dtype,
                    ),
                ],
                dim=1,
            )
            warped = homogeneous @ homography.T
            denominator = warped[:, 2]
            safe = torch.where(
                torch.abs(denominator) >= 0.20,
                denominator,
                torch.where(
                    denominator < 0.0,
                    torch.full_like(denominator, -0.20),
                    torch.full_like(denominator, 0.20),
                ),
            )
            return warped[:, :2] / safe[:, None]

        last_evidence = None
        for _iteration in range(int(iterations)):
            optimizer.zero_grad(set_to_none=True)
            homography = homography_from_raw()
            query_xy = project_canonical(homography)
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
            visible_weight = sampled_matchability * inside.to(dtype)
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
            # Fixed canonical denominator: an out-of-view cell contributes a
            # negative null term and cannot be optimized away.
            evidence = torch.mean(
                visible_weight
                * (cell_similarity - float(background_similarity))
                - (~inside).to(dtype) * float(out_of_view_penalty)
            )
            loss = -evidence + 0.002 * torch.mean(raw * raw)
            loss.backward()
            optimizer.step()
            last_evidence = evidence.detach()
        homography_value = (
            homography_from_raw().detach().cpu().numpy().astype(np.float64)
        )
        homography_value /= max(
            abs(float(homography_value[2, 2])), 1e-8
        )
        center = homography_value[:2, 2]
        # First-order affine frame at the canonical center, used only for
        # interpretable scale/rotation diagnostics.  Point transport consumes
        # the complete homography.
        linear = np.asarray(
            [
                [
                    homography_value[0, 0]
                    - center[0] * homography_value[2, 0],
                    homography_value[0, 1]
                    - center[0] * homography_value[2, 1],
                ],
                [
                    homography_value[1, 0]
                    - center[1] * homography_value[2, 0],
                    homography_value[1, 1]
                    - center[1] * homography_value[2, 1],
                ],
            ],
            dtype=np.float64,
        )
        transform_value = np.c_[linear, center]
        scale = np.linalg.norm(linear, axis=0)
        angle = float(
            np.rad2deg(np.arctan2(linear[1, 0], linear[0, 0])) % 360.0
        )
        homogeneous_support = np.c_[
            canonical.astype(np.float64),
            np.ones((canonical.shape[0],), dtype=np.float64),
        ]
        warped_support = homogeneous_support @ homography_value.T
        support_query = warped_support[:, :2] / np.where(
            np.abs(warped_support[:, 2:3]) > 1e-8,
            warped_support[:, 2:3],
            np.where(warped_support[:, 2:3] < 0.0, -1e-8, 1e-8),
        )
        support_hull, control_covariance, support_fraction = (
            _support_hull_and_control_covariance(
                canonical,
                support_query,
                tuple(query_feature.shape[-2:]),
                match.covariance_xy,
                feature_stride=int(match.feature_stride),
            )
        )
        fixed_evidence = (
            float(last_evidence.item())
            if last_evidence is not None
            else float(match.score)
        )
        # Retained only for backward-compatible ablations.  The production
        # default is zero because a probability score must not depend on chart
        # raster area.
        projected_support = max(
            4.0 * abs(float(np.linalg.det(linear))), 1.0
        )
        score = fixed_evidence * projected_support ** float(
            support_score_power
        )
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
                support_fraction=float(support_fraction),
                feature_level=match.feature_level,
                feature_stride=int(match.feature_stride),
                canonical_homography=homography_value,
                support_canonical_hull=support_hull,
                control_covariance_px=control_covariance,
                identity_probability=float(match.identity_probability),
            )
        )
    refined.sort(key=lambda value: -value.score)
    mode_probability, null_probability = _refined_frame_posterior(
        matches,
        refined,
        temperature=float(posterior_temperature),
    )
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
            canonical_homography=value.canonical_homography,
            support_canonical_hull=value.support_canonical_hull,
            control_covariance_px=value.control_covariance_px,
            identity_probability=value.identity_probability,
        )
        for value, probability in zip(refined, mode_probability)
    )


def _frame_homography(match: MapletFrameMatch) -> np.ndarray:
    homography = (
        np.asarray(match.canonical_homography, dtype=np.float64)
        if match.canonical_homography is not None
        else np.vstack(
            [
                np.asarray(match.canonical_to_query, dtype=np.float64),
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            ]
        )
    )
    scale = float(homography[2, 2])
    if not np.isfinite(scale) or abs(scale) <= 1e-8:
        raise ValueError("frame homography has an invalid scale")
    return homography / scale


def _project_homography_points(
    homography: np.ndarray, canonical_uv: np.ndarray
) -> np.ndarray:
    canonical = np.asarray(canonical_uv, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.c_[
        canonical, np.ones((canonical.shape[0],), dtype=np.float64)
    ]
    warped = homogeneous @ np.asarray(homography, dtype=np.float64).T
    denominator = warped[:, 2:3]
    safe = np.where(
        np.abs(denominator) > 1e-8,
        denominator,
        np.where(denominator < 0.0, -1e-8, 1e-8),
    )
    return warped[:, :2] / safe


def _sample_feature_candidates(
    query_feature: torch.Tensor, query_xy: torch.Tensor
) -> torch.Tensor:
    """Sample ``(N,K,2)`` query-cell coordinates as ``(N,K,C)``."""

    grid = query_xy.clone()
    grid[..., 0] = (
        2.0 * (grid[..., 0] + 0.5) / query_feature.shape[2] - 1.0
    )
    grid[..., 1] = (
        2.0 * (grid[..., 1] + 0.5) / query_feature.shape[1] - 1.0
    )
    sampled = F.grid_sample(
        query_feature[None],
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0]
    return sampled.permute(1, 2, 0)


def _local_flow_observations(
    query_feature: torch.Tensor,
    projected_xy: np.ndarray,
    map_feature: torch.Tensor,
    map_mode_feature: torch.Tensor | None,
    map_mode_weight: torch.Tensor | None,
    map_mode_valid: torch.Tensor | None,
    *,
    radius_cells: int,
    temperature: float,
    background_similarity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local multi-mode flow, confidence, similarity and entropy."""

    device, dtype = query_feature.device, query_feature.dtype
    radius = int(radius_cells)
    values = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    offsets = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    center = torch.from_numpy(
        np.asarray(projected_xy, dtype=np.float32)
    ).to(device=device, dtype=dtype)
    candidates = center[:, None] + offsets[None]
    sampled = F.normalize(
        _sample_feature_candidates(query_feature, candidates),
        dim=2,
        eps=1e-6,
    )
    mean_similarity = torch.einsum("nc,nkc->nk", map_feature, sampled)
    if map_mode_feature is not None:
        mode_similarity = torch.einsum(
            "nmc,nkc->nkm", map_mode_feature, sampled
        )
        mode_logit = (
            mode_similarity / float(temperature)
            + torch.log(torch.clamp(map_mode_weight[:, None], min=1e-8))
        )
        mode_logit = torch.where(
            map_mode_valid[:, None],
            mode_logit,
            torch.full_like(mode_logit, -torch.inf),
        )
        mode_score = float(temperature) * torch.logsumexp(
            mode_logit, dim=2
        )
        similarity = torch.where(
            torch.any(map_mode_valid, dim=1)[:, None],
            mode_score,
            mean_similarity,
        )
    else:
        similarity = mean_similarity
    inside = (
        (candidates[..., 0] >= 0.0)
        & (candidates[..., 0] <= query_feature.shape[2] - 1)
        & (candidates[..., 1] >= 0.0)
        & (candidates[..., 1] <= query_feature.shape[1] - 1)
    )
    logits = torch.where(
        inside,
        similarity / float(temperature),
        torch.full_like(similarity, -torch.inf),
    )
    finite_row = torch.any(torch.isfinite(logits), dim=1)
    safe_logits = torch.where(
        finite_row[:, None], logits, torch.zeros_like(logits)
    )
    probability = torch.softmax(safe_logits, dim=1)
    probability = torch.where(
        finite_row[:, None], probability, torch.zeros_like(probability)
    )
    peak_index = torch.argmax(probability, dim=1)
    peak_offset = offsets[peak_index]
    # Preserve one local displacement mode.  Averaging all repeated facade
    # peaks would create a virtual flow that no surface observation supports.
    delta = offsets[None] - peak_offset[:, None]
    neighbourhood = torch.max(torch.abs(delta), dim=2).values <= 1.0
    local_probability = probability * neighbourhood.to(dtype)
    local_probability /= torch.clamp(
        torch.sum(local_probability, dim=1, keepdim=True), min=1e-8
    )
    flow = torch.sum(local_probability[..., None] * offsets[None], dim=1)
    peak_similarity = similarity[
        torch.arange(similarity.shape[0], device=device), peak_index
    ]
    entropy = -torch.sum(
        probability * torch.log(torch.clamp(probability, min=1e-8)), dim=1
    )
    normalized_entropy = entropy / max(
        float(np.log(max(int(offsets.shape[0]), 2))), 1e-8
    )
    sorted_similarity = torch.topk(
        similarity, k=min(2, int(similarity.shape[1])), dim=1
    ).values
    margin = (
        sorted_similarity[:, 0] - sorted_similarity[:, 1]
        if sorted_similarity.shape[1] > 1
        else torch.ones_like(sorted_similarity[:, 0])
    )
    foreground = torch.sigmoid(
        (peak_similarity - float(background_similarity)) / 0.05
    )
    confidence = (
        foreground
        * torch.clamp(1.0 - normalized_entropy, min=0.0)
        * torch.sigmoid(margin / max(float(temperature), 1e-6))
        * finite_row.to(dtype)
    )
    return (
        flow.detach().cpu().numpy().astype(np.float64),
        confidence.detach().cpu().numpy().astype(np.float64),
        peak_similarity.detach().cpu().numpy().astype(np.float64),
        normalized_entropy.detach().cpu().numpy().astype(np.float64),
    )


def _chart_volume_residual_score(
    log_likelihood_ratio: torch.Tensor,
    canonical_uv: torch.Tensor,
    parameters: torch.Tensor,
    *,
    radius_cells: int,
    outlier_probability: float,
    cell_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score affine flow fields against all per-cell correlation volumes.

    ``parameters`` has shape ``(M,6)`` and represents a canonical affine
    residual ``flow = uv @ A.T + t``.  The score is a robust regional
    likelihood ratio: ambiguous cells contribute approximately zero, while a
    wrong peak is bounded by the explicit outlier component.
    """

    if (
        log_likelihood_ratio.ndim != 3
        or canonical_uv.ndim != 2
        or parameters.ndim != 2
        or parameters.shape[1] != 6
    ):
        raise ValueError("invalid structured chart-volume score inputs")
    modes = int(parameters.shape[0])
    cells = int(canonical_uv.shape[0])
    side = 2 * int(radius_cells) + 1
    if (
        log_likelihood_ratio.shape != (cells, side, side)
        or canonical_uv.shape != (cells, 2)
        or not 0.0 < float(outlier_probability) < 1.0
    ):
        raise ValueError("structured chart-volume shapes differ")
    linear = parameters[:, :4].reshape(modes, 2, 2)
    translation = parameters[:, 4:6]
    flow = torch.einsum("nc,mdc->mnd", canonical_uv, linear)
    flow = flow + translation[:, None]
    grid = (flow / float(radius_cells)).permute(1, 0, 2)[:, :, None]
    sampled = F.grid_sample(
        log_likelihood_ratio[:, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0, :, 0].T
    inside = torch.max(torch.abs(flow), dim=2).values <= float(radius_cells)
    log_outlier = float(np.log(float(outlier_probability)))
    robust = torch.logaddexp(
        sampled + float(np.log1p(-float(outlier_probability))),
        torch.full_like(sampled, log_outlier),
    )
    robust = torch.where(
        inside,
        robust,
        torch.full_like(robust, log_outlier),
    )
    if cell_weight is None:
        return torch.mean(robust, dim=1)
    weight = cell_weight.to(device=robust.device, dtype=robust.dtype)
    if weight.shape != (cells,):
        raise ValueError("chart-volume cell weight differs")
    weight = torch.clamp(weight, min=1e-4)
    return torch.sum(robust * weight[None], dim=1) / torch.sum(weight)


def _chart_volume_affine_proposals(
    score: np.ndarray,
    canonical_uv: np.ndarray,
    offsets: np.ndarray,
    *,
    cell_weight: np.ndarray | None = None,
    topk_per_cell: int,
    random_proposals: int,
    radius_cells: int,
    seed: int,
) -> np.ndarray:
    """Generate internal regional hypotheses from ambiguous local peaks."""

    values = np.asarray(score, dtype=np.float64)
    canonical = np.asarray(canonical_uv, dtype=np.float64)
    offset_values = np.asarray(offsets, dtype=np.float64)
    if (
        values.ndim != 2
        or canonical.shape != (values.shape[0], 2)
        or offset_values.shape != (values.shape[1], 2)
    ):
        raise ValueError("invalid chart-volume proposal inputs")
    cells = int(values.shape[0])
    topk = min(max(int(topk_per_cell), 1), int(values.shape[1]))
    order = np.argsort(-values, axis=1, kind="stable")[:, :topk]
    proposals: list[np.ndarray] = [
        np.zeros((6,), dtype=np.float64)
    ]

    # Constant translations are reliable initializers even when scale or
    # orientation still differs slightly.  Retain several separated modes;
    # repeated facades commonly make the best single translation ambiguous.
    aggregate = np.mean(values, axis=0)
    for offset_index in np.argsort(-aggregate, kind="stable")[
        : min(16, aggregate.size)
    ]:
        value = np.zeros((6,), dtype=np.float64)
        value[4:6] = offset_values[int(offset_index)]
        proposals.append(value)

    if cells >= 3 and int(random_proposals) > 0:
        rng = np.random.default_rng(int(seed))
        # Cells with a non-flat posterior carry more regional information.
        sorted_values = np.sort(values, axis=1)
        reliability = sorted_values[:, -1] - sorted_values[
            :, max(values.shape[1] - 5, 0)
        ]
        reliability = np.maximum(reliability, 1e-6)
        if cell_weight is not None:
            importance = np.asarray(cell_weight, dtype=np.float64).reshape(-1)
            if importance.shape != (cells,):
                raise ValueError("chart-volume proposal cell weight differs")
            reliability *= np.maximum(importance, 1e-4)
        reliability /= np.sum(reliability)
        corner_uv = np.asarray(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float64,
        )
        for _ in range(int(random_proposals)):
            rows = rng.choice(
                cells, size=3, replace=False, p=reliability
            )
            design = np.c_[
                canonical[rows], np.ones((3,), dtype=np.float64)
            ]
            if abs(float(np.linalg.det(design))) <= 1e-3:
                continue
            selected_offsets = []
            for row in rows.tolist():
                local_order = order[int(row)]
                local_score = values[int(row), local_order]
                probability = np.exp(
                    (local_score - np.max(local_score)) / 0.08
                )
                probability /= max(float(np.sum(probability)), 1e-12)
                selected_offsets.append(
                    offset_values[
                        int(rng.choice(local_order, p=probability))
                    ]
                )
            coefficients = np.linalg.solve(
                design, np.asarray(selected_offsets, dtype=np.float64)
            )
            linear = coefficients[:2].T
            translation = coefficients[2]
            corner_flow = corner_uv @ linear.T + translation[None]
            if (
                np.all(np.isfinite(corner_flow))
                and float(np.max(np.abs(corner_flow)))
                <= float(radius_cells) + 0.25
            ):
                proposals.append(
                    np.r_[linear.reshape(-1), translation]
                )
    result = np.stack(proposals).astype(np.float32)
    # Quantized de-duplication keeps the expensive continuous stage focused on
    # genuinely different chart hypotheses.
    key = np.round(result / 0.05).astype(np.int32)
    _unique, first = np.unique(key, axis=0, return_index=True)
    return result[np.sort(first)]


def estimate_chart_volume_residual(
    correlation: torch.Tensor,
    canonical_uv: torch.Tensor,
    *,
    candidate_matchability: torch.Tensor | None = None,
    radius_cells: int,
    topk_per_cell: int = 9,
    random_proposals: int = 384,
    retained_proposals: int = 12,
    iterations: int = 60,
    learning_rate: float = 0.08,
    correlation_temperature: float = 0.055,
    outlier_probability: float = 0.30,
    seed: int = 0,
    proposal_selection: str = "radio_union",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Find one chart-consistent residual in an ambiguous RADIO volume.

    Local peaks are proposal evidence only.  The returned displacement is the
    single affine field that maximizes a robust likelihood over the complete
    chart; no cell-level identity or correspondence leaves this function.
    """

    if correlation.ndim != 2 or canonical_uv.ndim != 2:
        raise ValueError("chart correlation and canonical coordinates required")
    selection_policy = str(proposal_selection)
    if selection_policy not in {"radio_union", "detector_guided"}:
        raise ValueError("unknown chart-volume proposal selection policy")
    if selection_policy == "detector_guided" and candidate_matchability is None:
        raise ValueError("detector-guided proposals require detector scores")
    radius = int(radius_cells)
    side = 2 * radius + 1
    patch_cells = side * side
    if (
        radius <= 0
        or correlation.shape[1] != 2 * patch_cells
        or canonical_uv.shape != (correlation.shape[0], 2)
        or int(correlation.shape[0]) < 4
        or int(retained_proposals) <= 0
        or int(iterations) < 0
        or float(correlation_temperature) <= 0.0
    ):
        raise ValueError("invalid chart-volume residual configuration")
    # View-conditioned appearance modes were measurably more informative than
    # the global mean on held-out replay.  A log-mean-exp fusion still lets
    # the mean rescue cells without a valid appearance mode.
    mean_score = correlation[:, :patch_cells]
    mode_score = correlation[:, patch_cells:]
    fused_score = float(correlation_temperature) * torch.logsumexp(
        torch.stack(
            [mean_score, mode_score], dim=0
        )
        / float(correlation_temperature),
        dim=0,
    ) - float(correlation_temperature) * float(np.log(2.0))
    cell_weight = None
    if candidate_matchability is not None:
        matchability = candidate_matchability.to(
            device=fused_score.device, dtype=fused_score.dtype
        )
        if matchability.shape != fused_score.shape:
            raise ValueError(
                "candidate matchability differs from chart correlation"
            )
        # ALIKE is a detector-only importance signal.  Collapse its local
        # patch to one weight per atlas cell so it can emphasize distinctive
        # regional evidence without inventing a preferred RADIO displacement.
        # Using it as an offset prior made a detector peak indistinguishable
        # from feature-match evidence and biased flow on repeated facades.
        cell_weight = torch.amax(
            torch.clamp(matchability, min=1e-4, max=1.0), dim=1
        )
    log_probability = F.log_softmax(
        fused_score / float(correlation_temperature), dim=1
    )
    log_likelihood_ratio = (
        log_probability + float(np.log(float(patch_cells)))
    ).reshape(-1, side, side)
    values = torch.arange(
        -radius,
        radius + 1,
        device=correlation.device,
        dtype=correlation.dtype,
    )
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    offsets = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    proposal_kwargs = {
        "cell_weight": (
            None
            if cell_weight is None
            else cell_weight.detach().cpu().numpy()
        ),
        "topk_per_cell": int(topk_per_cell),
        "radius_cells": radius,
        "seed": int(seed),
    }
    radio_proposals = (
        _chart_volume_affine_proposals(
            fused_score.detach().cpu().numpy(),
            canonical_uv.detach().cpu().numpy(),
            offsets.detach().cpu().numpy(),
            random_proposals=int(random_proposals),
            **proposal_kwargs,
        )
    )
    proposal_families = [radio_proposals]
    detector_guided_score = None
    detector_guided_proposals = None
    if candidate_matchability is not None:
        # Detector peaks may seed additional displacement hypotheses, but
        # never enter the RADIO log-probability or final residual score. This
        # preserves ALIKE's useful detector role without pretending that a
        # query-only heatmap is map/query correspondence evidence. Scaling by
        # the RADIO temperature turns matchability into a proposal prior only;
        # every retained proposal is subsequently ranked and optimized by the
        # unchanged RADIO regional likelihood above.
        detector_log_score = torch.log(
            torch.clamp(candidate_matchability, min=1e-4, max=1.0)
        )
        # Preserve two finite detector-seeded proposal families.  The weak
        # family only breaks nearly tied RADIO peaks; the strong family
        # recovers distinctive ALIKE locations when per-cell RADIO top-k is
        # too repetitive to seed the coherent regional optimum.  This is a
        # proposal schedule, not score fusion: both families are immediately
        # re-ranked and optimized by ``log_likelihood_ratio`` above, which is
        # computed from RADIO alone.
        for proposal_strength in (
            float(correlation_temperature),
            1.0,
        ):
            detector_proposal_score = (
                fused_score + proposal_strength * detector_log_score
            )
            family = _chart_volume_affine_proposals(
                    detector_proposal_score.detach().cpu().numpy(),
                    canonical_uv.detach().cpu().numpy(),
                    offsets.detach().cpu().numpy(),
                    random_proposals=0,
                    **proposal_kwargs,
                )
            proposal_families.append(family)
            if proposal_strength == 1.0:
                detector_guided_score = detector_proposal_score
                detector_guided_proposals = family
    if selection_policy == "detector_guided":
        if detector_guided_proposals is None or detector_guided_score is None:
            raise RuntimeError("detector-guided proposal family is unavailable")
        proposals = detector_guided_proposals
    else:
        proposals = np.concatenate(proposal_families, axis=0)
    proposal_key = np.round(proposals / 0.05).astype(np.int32)
    _unique, first = np.unique(
        proposal_key, axis=0, return_index=True
    )
    proposals = proposals[np.sort(first)]
    proposal_tensor = torch.from_numpy(proposals).to(
        device=correlation.device, dtype=correlation.dtype
    )
    with torch.no_grad():
        proposal_score = _chart_volume_residual_score(
            log_likelihood_ratio,
            canonical_uv,
            proposal_tensor,
            radius_cells=radius,
            outlier_probability=float(outlier_probability),
            cell_weight=cell_weight,
        )
        if selection_policy == "detector_guided":
            guided_log_probability = F.log_softmax(
                detector_guided_score
                / float(correlation_temperature),
                dim=1,
            )
            guided_log_likelihood_ratio = (
                guided_log_probability
                + float(np.log(float(patch_cells)))
            ).reshape(-1, side, side)
            proposal_rank_score = _chart_volume_residual_score(
                guided_log_likelihood_ratio,
                canonical_uv,
                proposal_tensor,
                radius_cells=radius,
                outlier_probability=float(outlier_probability),
                cell_weight=cell_weight,
            )
        else:
            proposal_rank_score = proposal_score
        keep = torch.argsort(proposal_rank_score, descending=True)[
            : min(int(retained_proposals), int(proposal_tensor.shape[0]))
        ]
        initial = proposal_tensor[keep].clone()
        # ALIKE may choose the finite starting basin, but the optimizer state,
        # returned evidence and every downstream probability remain RADIO.
        initial_score = proposal_score[keep].clone()
        zero_score = float(
            _chart_volume_residual_score(
                log_likelihood_ratio,
                canonical_uv,
                torch.zeros_like(proposal_tensor[:1]),
                radius_cells=radius,
                outlier_probability=float(outlier_probability),
                cell_weight=cell_weight,
            )[0].item()
        )
    if int(iterations) > 0:
        # Evaluation entry points intentionally run under ``no_grad`` for the
        # frozen VFM.  Re-enable gradients only for these six regional
        # geometry parameters; neither map nor query features are trainable.
        with torch.enable_grad():
            parameter = initial.detach().requires_grad_(True)
            optimizer = torch.optim.Adam(
                [parameter], lr=float(learning_rate)
            )
            best_parameter = initial.detach().clone()
            best_score = initial_score.detach().clone()
            corner = torch.tensor(
                [
                    [-1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                ],
                device=correlation.device,
                dtype=correlation.dtype,
            )
            for _ in range(int(iterations)):
                optimizer.zero_grad(set_to_none=True)
                score = _chart_volume_residual_score(
                    log_likelihood_ratio,
                    canonical_uv,
                    parameter,
                    radius_cells=radius,
                    outlier_probability=float(outlier_probability),
                    cell_weight=cell_weight,
                )
                linear = parameter[:, :4].reshape(-1, 2, 2)
                translation = parameter[:, 4:6]
                corner_flow = torch.einsum(
                    "kc,mdc->mkd", corner, linear
                ) + translation[:, None]
                overflow = F.relu(
                    torch.max(torch.abs(corner_flow), dim=2).values
                    - (float(radius) - 0.05)
                )
                loss = -torch.sum(score) + 0.20 * torch.sum(
                    overflow * overflow
                )
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    parameter[:, :4].clamp_(-float(radius), float(radius))
                    parameter[:, 4:6].clamp_(-float(radius), float(radius))
                    updated_score = _chart_volume_residual_score(
                        log_likelihood_ratio,
                        canonical_uv,
                        parameter,
                        radius_cells=radius,
                        outlier_probability=float(outlier_probability),
                        cell_weight=cell_weight,
                    )
                    improved = updated_score > best_score
                    best_score = torch.where(
                        improved, updated_score, best_score
                    )
                    best_parameter = torch.where(
                        improved[:, None],
                        parameter.detach(),
                        best_parameter,
                    )
    else:
        best_parameter = initial
        best_score = initial_score
    winner = int(torch.argmax(best_score).item())
    result = best_parameter[winner].detach().cpu().numpy()
    linear = result[:4].reshape(2, 2).astype(np.float64)
    translation = result[4:6].astype(np.float64)
    return linear, translation, float(best_score[winner].item()), zero_score


def refine_maplet_frame_matches_with_chart_volume(
    atlas: MapletFeatureAtlasBank,
    matches: Sequence[MapletFrameMatch],
    query_feature: torch.Tensor,
    query_matchability: torch.Tensor | None = None,
    *,
    radius_cells: int = 4,
    maximum_cells: int = 256,
    minimum_observations: int = 12,
    topk_per_cell: int = 9,
    random_proposals: int = 384,
    retained_proposals: int = 12,
    iterations: int = 60,
    maximum_corner_update_cells: float = 5.5,
    minimum_evidence_improvement: float = 0.0,
    posterior_temperature: float = 0.06,
    detector_guided_proposal_branch: bool = False,
) -> tuple[MapletFrameMatch, ...]:
    """Refine each chart as one structured RADIO correlation-volume factor.

    The local top-k peaks are used only to seed a chart-wide affine flow
    hypothesis.  A robust likelihood over all retained atlas cells selects
    and continuously refines one regional transform.  No sparse point match
    or persistent descriptor identity is produced.
    """

    if not matches:
        return ()
    if query_feature.ndim == 4:
        if int(query_feature.shape[0]) != 1:
            raise ValueError("query feature batch size must be one")
        query_feature = query_feature[0]
    if query_feature.ndim != 3:
        raise ValueError("query_feature must have shape (C,H,W)")
    if (
        int(radius_cells) <= 0
        or int(maximum_cells) < int(minimum_observations)
        or int(minimum_observations) < 4
        or int(topk_per_cell) <= 0
        or int(random_proposals) < 0
        or int(retained_proposals) <= 0
        or int(iterations) < 0
        or float(maximum_corner_update_cells) <= 0.0
        or float(posterior_temperature) <= 0.0
    ):
        raise ValueError("invalid chart-volume frame refinement parameters")
    query = F.normalize(query_feature, dim=0, eps=1e-6)
    device, dtype = query.device, query.dtype
    if query_matchability is None:
        matchability = None
    else:
        matchability = query_matchability
        if matchability.ndim == 4:
            matchability = matchability[0, 0]
        elif matchability.ndim == 3:
            matchability = matchability[0]
        if matchability.ndim != 2:
            raise ValueError("query_matchability must be a spatial map")
        if matchability.shape != query.shape[-2:]:
            matchability = F.interpolate(
                matchability[None, None],
                size=query.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        matchability = matchability.to(device=device, dtype=dtype).clamp(
            1e-4, 1.0
        )
    row_by_id = {
        int(value): row
        for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    canonical_corners = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    refined: list[MapletFrameMatch] = []
    for match_index, match in enumerate(matches):
        row = row_by_id.get(int(match.chart_id))
        if row is None:
            continue
        valid = (
            np.asarray(atlas.valid_mask[row], dtype=bool)
            & (np.asarray(atlas.support_count[row]) > 0)
            & (np.linalg.norm(atlas.features[row], axis=0) > 0.5)
        )
        y_all, x_all = np.nonzero(valid)
        if x_all.size < int(minimum_observations):
            refined.append(match)
            continue
        stride = max(
            int(np.ceil(np.sqrt(x_all.size / int(maximum_cells)))), 1
        )
        keep = (
            (x_all % stride == stride // 2)
            & (y_all % stride == stride // 2)
        )
        if int(np.sum(keep)) < int(minimum_observations):
            keep = np.arange(x_all.size) % stride == 0
        x, y = x_all[keep], y_all[keep]
        if x.size > int(maximum_cells):
            selection = np.linspace(
                0, x.size - 1, int(maximum_cells), dtype=np.int64
            )
            x, y = x[selection], y[selection]
        canonical = np.stack(
            [
                x / max(atlas.width - 1, 1) * 2.0 - 1.0,
                y / max(atlas.height - 1, 1) * 2.0 - 1.0,
            ],
            axis=1,
        ).astype(np.float64)
        initial = _frame_homography(match)
        projected = _project_homography_points(initial, canonical)
        radius = int(radius_cells)
        inside = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= -radius)
            & (projected[:, 0] <= query.shape[2] - 1 + radius)
            & (projected[:, 1] >= -radius)
            & (projected[:, 1] <= query.shape[1] - 1 + radius)
        )
        if int(np.sum(inside)) < int(minimum_observations):
            refined.append(match)
            continue
        x, y = x[inside], y[inside]
        canonical = canonical[inside]
        projected = projected[inside]
        map_feature = torch.from_numpy(
            atlas.features[row, :, y, x].copy()
        ).to(device=device, dtype=dtype)[None]
        if atlas.mode_features is not None:
            map_mode_feature = torch.from_numpy(
                atlas.mode_features[row, :, :, y, x].copy()
            ).to(device=device, dtype=dtype)[None]
            map_mode_weight = torch.from_numpy(
                atlas.mode_weights[row, :, y, x].copy()
            ).to(device=device, dtype=dtype)[None]
            map_mode_valid = torch.from_numpy(
                atlas.mode_valid_mask[row, :, y, x].copy()
            ).to(device=device)[None]
        else:
            map_mode_feature = None
            map_mode_weight = None
            map_mode_valid = None
        candidate = torch.from_numpy(
            projected.astype(np.float32)
        ).to(device=device, dtype=dtype)[None]
        canonical_tensor = torch.from_numpy(
            canonical.astype(np.float32)
        ).to(device=device, dtype=dtype)[None]
        correlation = structured_frame_correlation(
            query[None],
            map_feature,
            map_mode_feature,
            map_mode_weight,
            map_mode_valid,
            candidate,
            radius=radius,
        )[0]
        candidate_matchability = None
        if matchability is not None:
            offset_values = torch.arange(
                -radius, radius + 1, device=device, dtype=dtype
            )
            offset_y, offset_x = torch.meshgrid(
                offset_values, offset_values, indexing="ij"
            )
            offsets = torch.stack(
                [offset_x.reshape(-1), offset_y.reshape(-1)], dim=1
            )
            sample_xy = candidate[..., None, :] + offsets[None, None]
            grid = sample_xy.clone()
            grid[..., 0] = (
                2.0 * (grid[..., 0] + 0.5) / query.shape[2] - 1.0
            )
            grid[..., 1] = (
                2.0 * (grid[..., 1] + 0.5) / query.shape[1] - 1.0
            )
            candidate_matchability = F.grid_sample(
                matchability[None, None],
                grid.reshape(1, canonical.shape[0], -1, 2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0, 0].clamp(1e-4, 1.0)
        linear, translation, evidence, baseline_evidence = (
            estimate_chart_volume_residual(
                correlation,
                canonical_tensor[0],
                candidate_matchability=candidate_matchability,
                radius_cells=radius,
                topk_per_cell=int(topk_per_cell),
                random_proposals=int(random_proposals),
                retained_proposals=int(retained_proposals),
                iterations=int(iterations),
                seed=(
                    1000003 * int(match.chart_id)
                    + 9176 * int(match_index)
                    + int(round(float(match.query_center_xy[0]) * 31.0))
                    + int(round(float(match.query_center_xy[1]) * 37.0))
                ),
                proposal_selection=(
                    "detector_guided"
                    if bool(detector_guided_proposal_branch)
                    else "radio_union"
                ),
            )
        )
        old_corners = _project_homography_points(
            initial, canonical_corners
        )
        corrected_corners = (
            old_corners
            + canonical_corners @ linear.T
            + translation[None]
        )
        corner_update = np.linalg.norm(
            corrected_corners - old_corners, axis=1
        )
        if (
            (
                not bool(detector_guided_proposal_branch)
                and evidence
                < baseline_evidence + float(minimum_evidence_improvement)
            )
            or not np.all(np.isfinite(corrected_corners))
            or float(np.max(corner_update))
            > float(maximum_corner_update_cells)
        ):
            refined.append(match)
            continue
        candidate_homography = cv2.getPerspectiveTransform(
            canonical_corners.astype(np.float32),
            corrected_corners.astype(np.float32),
        ).astype(np.float64)
        if (
            not np.all(np.isfinite(candidate_homography))
            or abs(float(candidate_homography[2, 2])) <= 1e-8
        ):
            refined.append(match)
            continue
        candidate_homography /= float(candidate_homography[2, 2])
        epsilon = 1e-4
        origin_and_axes = _project_homography_points(
            candidate_homography,
            np.asarray(
                [[0.0, 0.0], [epsilon, 0.0], [0.0, epsilon]],
                dtype=np.float64,
            ),
        )
        center = origin_and_axes[0]
        local_linear = np.stack(
            [
                (origin_and_axes[1] - center) / epsilon,
                (origin_and_axes[2] - center) / epsilon,
            ],
            axis=1,
        )
        scale = np.linalg.norm(local_linear, axis=0)
        angle = float(
            np.rad2deg(
                np.arctan2(local_linear[1, 0], local_linear[0, 0])
            )
            % 360.0
        )
        support_query = _project_homography_points(
            candidate_homography, canonical
        )
        support_hull, control_covariance, support_fraction = (
            _support_hull_and_control_covariance(
                canonical,
                support_query,
                tuple(query.shape[-2:]),
                match.covariance_xy,
                feature_stride=int(match.feature_stride),
            )
        )
        evidence_gain = float(evidence - baseline_evidence)
        control_covariance = np.asarray(
            control_covariance, dtype=np.float64
        ) / max(float(np.exp(min(evidence_gain, 2.0))), 0.25)
        score = float(match.score) + float(posterior_temperature) * float(
            np.clip(evidence, -6.0, 6.0)
        )
        refined.append(
            MapletFrameMatch(
                chart_id=int(match.chart_id),
                canonical_to_query=np.c_[local_linear, center],
                query_center_xy=center,
                scale_xy=scale,
                in_plane_rotation_deg=angle,
                covariance_xy=match.covariance_xy,
                score=score,
                probability=float(match.probability),
                null_probability=float(match.null_probability),
                support_fraction=float(support_fraction),
                feature_level=match.feature_level,
                feature_stride=int(match.feature_stride),
                canonical_homography=candidate_homography,
                support_canonical_hull=support_hull,
                control_covariance_px=control_covariance,
                identity_probability=float(match.identity_probability),
            )
        )
    refined.sort(key=lambda value: -value.score)
    mode_probability, null_probability = _refined_frame_posterior(
        matches,
        refined,
        temperature=float(posterior_temperature),
    )
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
            canonical_homography=value.canonical_homography,
            support_canonical_hull=value.support_canonical_hull,
            control_covariance_px=value.control_covariance_px,
            identity_probability=value.identity_probability,
        )
        for value, probability in zip(refined, mode_probability)
    )


def refine_maplet_frame_matches_with_local_flow(
    atlas: MapletFeatureAtlasBank,
    matches: Sequence[MapletFrameMatch],
    query_feature: torch.Tensor,
    *,
    rounds: int = 2,
    radius_cells: int = 3,
    maximum_cells: int = 256,
    minimum_observations: int = 8,
    minimum_confidence: float = 0.015,
    temperature: float = 0.055,
    background_similarity: float = 0.25,
    ransac_threshold_cells: float = 0.85,
    maximum_corner_update_cells: float = 4.5,
    posterior_temperature: float = 0.06,
    update_model: str = "canonical_affine",
) -> tuple[MapletFrameMatch, ...]:
    """Refine complete chart modes from local RADIO displacement posteriors.

    Atlas cells provide a correlated regional flow field whose projective fit
    updates one chart homography.  They are not exported as stable point
    identities and are never treated as independent pose measurements.
    """

    if not matches:
        return ()
    if query_feature.ndim == 4:
        if int(query_feature.shape[0]) != 1:
            raise ValueError("query feature batch size must be one")
        query_feature = query_feature[0]
    if query_feature.ndim != 3:
        raise ValueError("query_feature must have shape (C,H,W)")
    if (
        int(rounds) <= 0
        or int(radius_cells) <= 0
        or int(maximum_cells) < 4
        or int(minimum_observations) < 4
        or float(temperature) <= 0.0
        or float(ransac_threshold_cells) <= 0.0
        or float(maximum_corner_update_cells) <= 0.0
        or str(update_model)
        not in {"canonical_affine", "query_affine", "homography"}
    ):
        raise ValueError("invalid local-flow frame refinement parameters")
    query_feature = F.normalize(query_feature, dim=0, eps=1e-6)
    device, dtype = query_feature.device, query_feature.dtype
    row_by_id = {
        int(value): row
        for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    canonical_corners = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
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
        y_all, x_all = np.nonzero(valid)
        if x_all.size < int(minimum_observations):
            refined.append(match)
            continue
        stride = max(
            int(np.ceil(np.sqrt(x_all.size / int(maximum_cells)))), 1
        )
        keep = (
            (x_all % stride == stride // 2)
            & (y_all % stride == stride // 2)
        )
        if int(np.sum(keep)) < int(minimum_observations):
            keep = np.arange(x_all.size) % stride == 0
        x, y = x_all[keep], y_all[keep]
        if x.size > int(maximum_cells):
            selection = np.linspace(
                0, x.size - 1, int(maximum_cells), dtype=np.int64
            )
            x, y = x[selection], y[selection]
        canonical = np.stack(
            [
                x / max(atlas.width - 1, 1) * 2.0 - 1.0,
                y / max(atlas.height - 1, 1) * 2.0 - 1.0,
            ],
            axis=1,
        ).astype(np.float64)
        map_feature = torch.from_numpy(
            atlas.features[row, :, y, x].copy()
        ).to(device=device, dtype=dtype)
        map_feature = F.normalize(map_feature, dim=1, eps=1e-6)
        if atlas.mode_features is not None:
            map_mode_feature = torch.from_numpy(
                atlas.mode_features[row, :, :, y, x]
                .copy()
            ).to(device=device, dtype=dtype)
            map_mode_feature = F.normalize(
                map_mode_feature, dim=2, eps=1e-6
            )
            map_mode_weight = torch.from_numpy(
                atlas.mode_weights[row, :, y, x].copy()
            ).to(device=device, dtype=dtype)
            map_mode_valid = torch.from_numpy(
                atlas.mode_valid_mask[row, :, y, x].copy()
            ).to(device=device)
        else:
            map_mode_feature = None
            map_mode_weight = None
            map_mode_valid = None
        initial = _frame_homography(match)
        current = initial.copy()
        final_confidence = np.zeros((canonical.shape[0],), dtype=np.float64)
        final_similarity = np.full(
            (canonical.shape[0],), -1.0, dtype=np.float64
        )
        final_entropy = np.ones((canonical.shape[0],), dtype=np.float64)
        accepted_rounds = 0
        inlier_fraction = 0.0
        for round_index in range(int(rounds)):
            projected = _project_homography_points(current, canonical)
            flow, confidence, similarity, entropy = _local_flow_observations(
                query_feature,
                projected,
                map_feature,
                map_mode_feature,
                map_mode_weight,
                map_mode_valid,
                radius_cells=max(int(radius_cells) - round_index, 1),
                temperature=float(temperature),
                background_similarity=float(background_similarity),
            )
            inside = (
                np.isfinite(projected).all(axis=1)
                & (projected[:, 0] >= 0.0)
                & (projected[:, 0] <= query_feature.shape[2] - 1)
                & (projected[:, 1] >= 0.0)
                & (projected[:, 1] <= query_feature.shape[1] - 1)
            )
            usable = (
                inside
                & np.isfinite(flow).all(axis=1)
                & (confidence >= float(minimum_confidence))
                & (similarity > float(background_similarity) - 0.10)
            )
            if int(np.sum(usable)) < int(minimum_observations):
                break
            order = np.argsort(-confidence[usable], kind="mergesort")
            usable_rows = np.flatnonzero(usable)[order]
            destination = projected[usable_rows] + flow[usable_rows]
            if str(update_model) == "homography":
                method = (
                    cv2.USAC_MAGSAC
                    if hasattr(cv2, "USAC_MAGSAC")
                    else cv2.RANSAC
                )
                candidate, inliers = cv2.findHomography(
                    canonical[usable_rows].astype(np.float32),
                    destination.astype(np.float32),
                    method=method,
                    ransacReprojThreshold=float(ransac_threshold_cells),
                    maxIters=2000,
                    confidence=0.995,
                )
                if candidate is None:
                    break
                candidate = np.asarray(candidate, dtype=np.float64)
            else:
                source = (
                    canonical[usable_rows]
                    if str(update_model) == "canonical_affine"
                    else projected[usable_rows]
                )
                affine_delta, inliers = cv2.estimateAffine2D(
                    source.astype(np.float32),
                    destination.astype(np.float32),
                    method=cv2.RANSAC,
                    ransacReprojThreshold=float(ransac_threshold_cells),
                    maxIters=2000,
                    confidence=0.995,
                    refineIters=10,
                )
                if affine_delta is None:
                    break
                affine_delta = np.asarray(
                    affine_delta, dtype=np.float64
                )
                if str(update_model) == "query_affine":
                    singular = np.linalg.svd(
                        affine_delta[:, :2], compute_uv=False
                    )
                    if (
                        float(np.linalg.det(affine_delta[:, :2])) <= 0.0
                        or float(np.min(singular)) < 0.50
                        or float(np.max(singular)) > 1.75
                    ):
                        break
                    candidate = (
                        np.vstack(
                            [
                                affine_delta,
                                np.asarray([0.0, 0.0, 1.0]),
                            ]
                        )
                        @ current
                    )
                else:
                    candidate = np.vstack(
                        [
                            affine_delta,
                            np.asarray([0.0, 0.0, 1.0]),
                        ]
                    )
            if not np.all(np.isfinite(candidate)):
                break
            candidate /= float(candidate[2, 2])
            initial_corners = _project_homography_points(
                initial, canonical_corners
            )
            candidate_corners = _project_homography_points(
                candidate, canonical_corners
            )
            if (
                not np.all(np.isfinite(candidate_corners))
                or float(
                    np.max(
                        np.linalg.norm(
                            candidate_corners - initial_corners, axis=1
                        )
                    )
                )
                > float(maximum_corner_update_cells)
            ):
                break
            inlier_mask = (
                np.ones((usable_rows.size,), dtype=bool)
                if inliers is None
                else np.asarray(inliers, dtype=bool).reshape(-1)
            )
            if int(np.sum(inlier_mask)) < int(minimum_observations):
                break
            current = candidate
            accepted_rounds += 1
            inlier_fraction = float(np.mean(inlier_mask))
            final_confidence = confidence
            final_similarity = similarity
            final_entropy = entropy
        if accepted_rounds == 0:
            refined.append(match)
            continue
        final_projected = _project_homography_points(current, canonical)
        (
            residual_flow,
            final_confidence,
            final_similarity,
            final_entropy,
        ) = _local_flow_observations(
            query_feature,
            final_projected,
            map_feature,
            map_mode_feature,
            map_mode_weight,
            map_mode_valid,
            radius_cells=1,
            temperature=float(temperature),
            background_similarity=float(background_similarity),
        )
        center = _project_homography_points(
            current, np.zeros((1, 2), dtype=np.float64)
        )[0]
        epsilon = 1e-4
        origin_and_axes = _project_homography_points(
            current,
            np.asarray(
                [[0.0, 0.0], [epsilon, 0.0], [0.0, epsilon]],
                dtype=np.float64,
            ),
        )
        linear = np.stack(
            [
                (origin_and_axes[1] - origin_and_axes[0]) / epsilon,
                (origin_and_axes[2] - origin_and_axes[0]) / epsilon,
            ],
            axis=1,
        )
        transform = np.c_[linear, center]
        scale = np.linalg.norm(linear, axis=0)
        angle = float(
            np.rad2deg(np.arctan2(linear[1, 0], linear[0, 0])) % 360.0
        )
        support_query = _project_homography_points(current, canonical)
        support_hull, control_covariance, support_fraction = (
            _support_hull_and_control_covariance(
                canonical,
                support_query,
                tuple(query_feature.shape[-2:]),
                match.covariance_xy,
                feature_stride=int(match.feature_stride),
            )
        )
        control_covariance = np.asarray(
            control_covariance, dtype=np.float64
        ) / max(inlier_fraction, 0.10)
        evidence_weight = np.maximum(final_confidence, 1e-6)
        evidence = float(
            np.sum(
                evidence_weight
                * (final_similarity - float(background_similarity))
            )
            / np.sum(evidence_weight)
        )
        residual_magnitude = np.linalg.norm(residual_flow, axis=1)
        score = (
            evidence
            - 0.10 * float(np.average(final_entropy, weights=evidence_weight))
            - 0.08
            * float(
                np.average(residual_magnitude, weights=evidence_weight)
            )
            + 0.05 * inlier_fraction
        )
        refined.append(
            MapletFrameMatch(
                chart_id=int(match.chart_id),
                canonical_to_query=transform,
                query_center_xy=center,
                scale_xy=scale,
                in_plane_rotation_deg=angle,
                covariance_xy=match.covariance_xy,
                score=score,
                probability=float(match.probability),
                null_probability=float(match.null_probability),
                support_fraction=float(support_fraction),
                feature_level=match.feature_level,
                feature_stride=int(match.feature_stride),
                canonical_homography=current,
                support_canonical_hull=support_hull,
                control_covariance_px=control_covariance,
                identity_probability=float(match.identity_probability),
            )
        )
    refined.sort(key=lambda value: -value.score)
    mode_probability, null_probability = _refined_frame_posterior(
        matches,
        refined,
        temperature=float(posterior_temperature),
    )
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
            canonical_homography=value.canonical_homography,
            support_canonical_hull=value.support_canonical_hull,
            control_covariance_px=value.control_covariance_px,
            identity_probability=value.identity_probability,
        )
        for value, probability in zip(refined, mode_probability)
    )


@torch.no_grad()
def refine_maplet_frame_matches_with_structured_refiner(
    atlas: MapletFeatureAtlasBank,
    matches: Sequence[MapletFrameMatch],
    query_feature: torch.Tensor,
    model: StructuredFrameRefiner,
    *,
    maximum_cells: int = 64,
    minimum_observations: int = 8,
    maximum_corner_update_cells: float = 4.5,
    training_positive_fraction: float = 0.75,
    posterior_temperature: float = 0.06,
) -> tuple[MapletFrameMatch, ...]:
    """Apply one learned, correlated chart-level RADIO residual.

    The network consumes the complete local correlation field and emits one
    query-plane affine update plus a chart-usable likelihood ratio.  Atlas
    cells never leave this regional factor as persistent point identities.
    """

    if not matches:
        return ()
    if query_feature.ndim == 4:
        if int(query_feature.shape[0]) != 1:
            raise ValueError("query feature batch size must be one")
        query_feature = query_feature[0]
    if query_feature.ndim != 3:
        raise ValueError("query_feature must have shape (C,H,W)")
    if (
        int(maximum_cells) < int(minimum_observations)
        or int(minimum_observations) < 4
        or float(maximum_corner_update_cells) <= 0.0
        or not 0.0 < float(training_positive_fraction) < 1.0
        or float(posterior_temperature) <= 0.0
    ):
        raise ValueError("invalid structured refinement parameters")
    if int(model.config.feature_dim) != int(atlas.feature_dim):
        raise ValueError("structured refiner feature dimension differs")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    query = F.normalize(
        query_feature.to(device=device, dtype=dtype), dim=0, eps=1e-6
    )
    row_by_id = {
        int(value): row
        for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    canonical_corners = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    prior_log_odds = float(
        np.log(
            float(training_positive_fraction)
            / (1.0 - float(training_positive_fraction))
        )
    )
    refined: list[MapletFrameMatch] = []
    for match in matches:
        row = row_by_id.get(int(match.chart_id))
        if row is None:
            continue
        valid = (
            np.asarray(atlas.valid_mask[row], dtype=bool)
            & (np.asarray(atlas.support_count[row]) > 0)
            & (np.linalg.norm(atlas.features[row], axis=0) > 0.5)
        )
        y_all, x_all = np.nonzero(valid)
        if x_all.size < int(minimum_observations):
            refined.append(match)
            continue
        # Deterministic spatial subsampling preserves complete-chart coverage
        # without letting dense atlas resolution masquerade as independent
        # evidence.
        stride = max(
            int(np.floor(np.sqrt(x_all.size / int(maximum_cells)))), 1
        )
        keep = (
            (x_all % stride == stride // 2)
            & (y_all % stride == stride // 2)
        )
        if int(np.sum(keep)) < int(minimum_observations):
            keep = np.arange(x_all.size) % stride == 0
        x, y = x_all[keep], y_all[keep]
        if x.size > int(maximum_cells):
            selection = np.linspace(
                0, x.size - 1, int(maximum_cells), dtype=np.int64
            )
            x, y = x[selection], y[selection]
        canonical = np.stack(
            [
                x / max(atlas.width - 1, 1) * 2.0 - 1.0,
                y / max(atlas.height - 1, 1) * 2.0 - 1.0,
            ],
            axis=1,
        ).astype(np.float64)
        initial = _frame_homography(match)
        projected = _project_homography_points(initial, canonical)
        radius = int(model.config.correlation_radius)
        inside = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= -radius)
            & (projected[:, 0] <= query.shape[2] - 1 + radius)
            & (projected[:, 1] >= -radius)
            & (projected[:, 1] <= query.shape[1] - 1 + radius)
        )
        if int(np.sum(inside)) < int(minimum_observations):
            refined.append(match)
            continue
        x, y = x[inside], y[inside]
        canonical = canonical[inside]
        projected = projected[inside]
        map_feature = torch.from_numpy(
            atlas.features[row, :, y, x].copy()
        ).to(device=device, dtype=dtype)[None]
        if atlas.mode_features is not None:
            map_mode_feature = torch.from_numpy(
                atlas.mode_features[row, :, :, y, x].copy()
            ).to(device=device, dtype=dtype)[None]
            map_mode_weight = torch.from_numpy(
                atlas.mode_weights[row, :, y, x].copy()
            ).to(device=device, dtype=dtype)[None]
            map_mode_valid = torch.from_numpy(
                atlas.mode_valid_mask[row, :, y, x].copy()
            ).to(device=device)[None]
        else:
            map_mode_feature = None
            map_mode_weight = None
            map_mode_valid = None
        candidate = torch.from_numpy(
            projected.astype(np.float32)
        ).to(device=device, dtype=dtype)[None]
        canonical_tensor = torch.from_numpy(
            canonical.astype(np.float32)
        ).to(device=device, dtype=dtype)[None]
        correlation = structured_frame_correlation(
            query[None],
            map_feature,
            map_mode_feature,
            map_mode_weight,
            map_mode_valid,
            candidate,
            radius=radius,
        )
        candidate_normalized = candidate.clone()
        candidate_normalized[..., 0] = (
            2.0 * (candidate_normalized[..., 0] + 0.5) / query.shape[2]
            - 1.0
        )
        candidate_normalized[..., 1] = (
            2.0 * (candidate_normalized[..., 1] + 0.5) / query.shape[1]
            - 1.0
        )
        prediction = model(
            correlation, canonical_tensor, candidate_normalized
        )
        linear = (
            prediction["linear"][0].detach().cpu().numpy().astype(np.float64)
        )
        translation = (
            prediction["translation"][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        usable_logit = float(prediction["usable_logit"][0].item())
        old_corners = _project_homography_points(
            initial, canonical_corners
        )
        if (
            str(model.config.update_parameterization)
            == "canonical_residual"
        ):
            corrected_corners = (
                old_corners
                + canonical_corners @ linear.T
                + translation[None]
            )
            candidate_homography = cv2.getPerspectiveTransform(
                canonical_corners.astype(np.float32),
                corrected_corners.astype(np.float32),
            ).astype(np.float64)
        else:
            center_before = np.mean(projected, axis=0)
            offset = (
                center_before + translation - linear @ center_before
            )
            query_update = np.asarray(
                [
                    [linear[0, 0], linear[0, 1], offset[0]],
                    [linear[1, 0], linear[1, 1], offset[1]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            candidate_homography = query_update @ initial
            if abs(float(np.linalg.det(linear))) <= 0.05:
                refined.append(match)
                continue
        if (
            not np.all(np.isfinite(candidate_homography))
            or abs(float(candidate_homography[2, 2])) <= 1e-8
        ):
            refined.append(match)
            continue
        candidate_homography /= float(candidate_homography[2, 2])
        new_corners = _project_homography_points(
            candidate_homography, canonical_corners
        )
        if (
            not np.all(np.isfinite(new_corners))
            or float(
                np.max(np.linalg.norm(new_corners - old_corners, axis=1))
            )
            > float(maximum_corner_update_cells)
        ):
            refined.append(match)
            continue
        epsilon = 1e-4
        origin_and_axes = _project_homography_points(
            candidate_homography,
            np.asarray(
                [[0.0, 0.0], [epsilon, 0.0], [0.0, epsilon]],
                dtype=np.float64,
            ),
        )
        center = origin_and_axes[0]
        local_linear = np.stack(
            [
                (origin_and_axes[1] - center) / epsilon,
                (origin_and_axes[2] - center) / epsilon,
            ],
            axis=1,
        )
        scale = np.linalg.norm(local_linear, axis=0)
        angle = float(
            np.rad2deg(
                np.arctan2(local_linear[1, 0], local_linear[0, 0])
            )
            % 360.0
        )
        support_query = _project_homography_points(
            candidate_homography, canonical
        )
        support_hull, control_covariance, support_fraction = (
            _support_hull_and_control_covariance(
                canonical,
                support_query,
                tuple(query.shape[-2:]),
                match.covariance_xy,
                feature_stride=int(match.feature_stride),
            )
        )
        usable_probability = float(
            torch.sigmoid(prediction["usable_logit"][0]).item()
        )
        control_covariance = np.asarray(
            control_covariance, dtype=np.float64
        ) / max(usable_probability, 0.10)
        # The BCE head includes the synthetic training prior.  Subtract it to
        # obtain a likelihood ratio, then express it in the same temperature
        # units as the established chart-mode score.
        score = float(match.score) + float(posterior_temperature) * float(
            np.clip(usable_logit - prior_log_odds, -6.0, 6.0)
        )
        refined.append(
            MapletFrameMatch(
                chart_id=int(match.chart_id),
                canonical_to_query=np.c_[local_linear, center],
                query_center_xy=center,
                scale_xy=scale,
                in_plane_rotation_deg=angle,
                covariance_xy=match.covariance_xy,
                score=score,
                probability=float(match.probability),
                null_probability=float(match.null_probability),
                support_fraction=float(support_fraction),
                feature_level=match.feature_level,
                feature_stride=int(match.feature_stride),
                canonical_homography=candidate_homography,
                support_canonical_hull=support_hull,
                control_covariance_px=control_covariance,
                identity_probability=float(match.identity_probability),
            )
        )
    refined.sort(key=lambda value: -value.score)
    mode_probability, null_probability = _refined_frame_posterior(
        matches,
        refined,
        temperature=float(posterior_temperature),
    )
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
            canonical_homography=value.canonical_homography,
            support_canonical_hull=value.support_canonical_hull,
            control_covariance_px=value.control_covariance_px,
            identity_probability=value.identity_probability,
        )
        for value, probability in zip(refined, mode_probability)
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
        support_canonical_hull=uv,
        control_covariance_px=np.eye(8, dtype=np.float32) * 1e-4,
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


def frame_log_likelihood_ratio(match: MapletFrameMatch) -> float:
    """Return a finite chart-frame/identity evidence ratio.

    Frame probabilities are still diagnostic until trajectory-isolated
    calibration is available, but using their mode-vs-null ratio is
    semantically preferable to adding raw cosine similarities.  The retrieval
    identity posterior enters once per chart.
    """

    mode = max(float(match.probability), 1e-8)
    null = max(float(match.null_probability), 1e-8)
    identity = max(float(match.identity_probability), 1e-8)
    return float(np.log(mode / null) + np.log(identity))


def marginal_frame_log_likelihood_ratio(
    matches: Sequence[MapletFrameMatch],
) -> float:
    """Marginalize mutually exclusive chart-frame modes against null."""

    values = np.asarray(
        [frame_log_likelihood_ratio(value) for value in matches],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("-inf")
    maximum = float(np.max(values))
    return float(maximum + np.log(np.sum(np.exp(values - maximum))))


def regional_frame_log_evidence(
    matches: Sequence[MapletFrameMatch],
) -> float:
    """Return chart-count-neutral evidence for one regional pose mode.

    Hypotheses supported by one, two, three, or four charts compete in the
    same coarse-pose queue.  Summing positive per-chart log odds gives larger
    chart sets an automatic score bonus before geometric consistency or
    held-out atlas evidence has been observed.  The mean retains evidence
    quality while support count is handled explicitly by pose-mode consensus.
    """

    values = np.asarray(
        [frame_log_likelihood_ratio(value) for value in matches],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("-inf")


def _control_covariance_for_match(
    match: MapletFrameMatch,
) -> np.ndarray:
    if match.control_covariance_px is not None:
        covariance = np.asarray(
            match.control_covariance_px, dtype=np.float64
        )
    else:
        center = (
            np.asarray(match.covariance_xy, dtype=np.float64)
            * float(match.feature_stride) ** 2
        )
        covariance = np.zeros((8, 8), dtype=np.float64)
        for first in range(4):
            for second in range(4):
                covariance[
                    2 * first : 2 * first + 2,
                    2 * second : 2 * second + 2,
                ] = center
        independent = (
            max(0.35 * float(match.feature_stride), 1.0)
            / max(float(match.support_fraction), 0.10)
        ) ** 2
        covariance += np.eye(8, dtype=np.float64) * independent
    return covariance + np.eye(8, dtype=np.float64) * 1e-4


def _refine_pose_with_chart_factors(
    atlas: MapletFeatureAtlasBank,
    matches: Sequence[MapletFrameMatch],
    seed_pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    iterations: int = 12,
    damping: float = 1e-3,
    robust_delta: float = 3.0,
) -> tuple[np.ndarray, float, bool, float]:
    """Robust SE(3) LM with one correlated block factor per chart."""

    factors = []
    for match in matches:
        uv, world = chart_canonical_control_points(atlas, match.chart_id)
        observed = match.canonical_points_to_pixels(uv).astype(np.float64)
        covariance = _control_covariance_for_match(match)
        try:
            information = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            information = np.linalg.pinv(covariance)
        factors.append(
            (
                world.astype(np.float64),
                observed,
                information,
                max(
                    float(match.probability)
                    / max(
                        float(match.probability)
                        + float(match.null_probability),
                        1e-8,
                    ),
                    1e-3,
                ),
            )
        )
    pose = np.asarray(seed_pose_w2c, dtype=np.float64).reshape(4, 4).copy()
    if not factors:
        return pose, float("inf"), False, float("inf")
    for _iteration in range(max(int(iterations), 0)):
        normal = np.eye(6, dtype=np.float64) * float(damping)
        right = np.zeros((6,), dtype=np.float64)
        for world, observed, information, confidence in factors:
            projected, jacobian = projection_jacobian(world, pose, camera)
            residual = (observed - projected).reshape(-1)
            block_jacobian = jacobian.reshape(-1, 6)
            normalized = float(
                np.sqrt(
                    max(
                        residual @ information @ residual
                        / max(residual.size, 1),
                        1e-12,
                    )
                )
            )
            robust = min(
                1.0, float(robust_delta) / max(normalized, 1e-8)
            )
            block_information = information * confidence * robust
            normal += block_jacobian.T @ block_information @ block_jacobian
            right += block_jacobian.T @ block_information @ residual
        try:
            delta = np.linalg.solve(normal, right)
        except np.linalg.LinAlgError:
            return pose, float("inf"), False, float("inf")
        rotation_norm = float(np.linalg.norm(delta[:3]))
        maximum_rotation = np.deg2rad(8.0)
        if rotation_norm > maximum_rotation:
            delta[:3] *= maximum_rotation / rotation_norm
        translation_norm = float(np.linalg.norm(delta[3:]))
        if translation_norm > 1.0:
            delta[3:] /= translation_norm
        pose = se3_exp(delta) @ pose
        if float(np.linalg.norm(delta)) < 1e-7:
            break
    errors = []
    costs = []
    positive = True
    for world, observed, information, _confidence in factors:
        projected, depth = project_world_points(world, pose, camera)
        residual = (observed - projected).reshape(-1)
        errors.extend(np.linalg.norm(observed - projected, axis=1).tolist())
        costs.append(
            float(
                np.sqrt(
                    max(
                        residual @ information @ residual
                        / max(residual.size, 1),
                        0.0,
                    )
                )
            )
        )
        positive = positive and bool(np.all(depth > 0.0))
    return (
        pose,
        float(np.mean(errors)) if errors else float("inf"),
        bool(positive and np.all(np.isfinite(pose))),
        float(np.mean(costs)) if costs else float("inf"),
    )


def _poses_materially_differ(
    first_w2c: np.ndarray,
    second_w2c: np.ndarray,
    *,
    translation_m: float = 1e-4,
    rotation_deg: float = 1e-3,
) -> bool:
    first = np.asarray(first_w2c, dtype=np.float64).reshape(4, 4)
    second = np.asarray(second_w2c, dtype=np.float64).reshape(4, 4)
    first_center = -first[:3, :3].T @ first[:3, 3]
    second_center = -second[:3, :3].T @ second[:3, 3]
    relative = first[:3, :3] @ second[:3, :3].T
    angle = float(
        np.degrees(
            np.arccos(
                np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
            )
        )
    )
    return bool(
        np.linalg.norm(first_center - second_center)
        > float(translation_m)
        or angle > float(rotation_deg)
    )


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
        pose, error, positive_depth, factor_cost = (
            _refine_pose_with_chart_factors(
                atlas, [match], pose, camera
            )
        )
        hypotheses.append(
            FramePoseHypothesis(
                pose_w2c=pose,
                score=float(
                    frame_log_likelihood_ratio(match)
                    - 0.5 * np.log1p(factor_cost)
                ),
                source_chart_ids=(int(match.chart_id),),
                reprojection_error_px=error,
                positive_depth=positive_depth,
                control_model="chart_factor",
                seed_model="IPPE",
                factor_cost=float(factor_cost),
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
    refine_grouped_pose: bool = True,
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
                seed_pose = np.eye(4, dtype=np.float64)
                seed_pose[:3, :3] = rotation
                seed_pose[:3, 3] = translation
                seed_pose, seed_error, seed_positive, seed_factor_cost = (
                    _refine_pose_with_chart_factors(
                        atlas, matches, seed_pose, camera, iterations=0
                    )
                )
                frame_evidence = regional_frame_log_evidence(matches)
                if not bool(refine_grouped_pose):
                    hypotheses.append(
                        FramePoseHypothesis(
                            pose_w2c=seed_pose,
                            score=float(
                                frame_evidence
                                - 0.5 * np.log1p(seed_factor_cost)
                            ),
                            source_chart_ids=tuple(
                                int(match.chart_id) for match in matches
                            ),
                            reprojection_error_px=seed_error,
                            positive_depth=seed_positive,
                            control_model="regional_chart_seed",
                            seed_model="EPNP_UNREFINED",
                            factor_cost=float(seed_factor_cost),
                        )
                    )
                else:
                    pose, error, positive_depth, factor_cost = (
                        _refine_pose_with_chart_factors(
                            atlas, matches, seed_pose, camera
                        )
                    )
                    if _poses_materially_differ(seed_pose, pose):
                        hypotheses.append(
                            FramePoseHypothesis(
                                pose_w2c=seed_pose,
                                score=float(
                                    frame_evidence
                                    - 0.5 * np.log1p(seed_factor_cost)
                                ),
                                source_chart_ids=tuple(
                                    int(match.chart_id)
                                    for match in matches
                                ),
                                reprojection_error_px=seed_error,
                                positive_depth=seed_positive,
                                control_model="regional_chart_seed",
                                seed_model="EPNP_UNREFINED",
                                factor_cost=float(seed_factor_cost),
                            )
                        )
                    hypotheses.append(
                        FramePoseHypothesis(
                            pose_w2c=pose,
                            score=float(
                                frame_evidence
                                - 0.5 * np.log1p(factor_cost)
                            ),
                            source_chart_ids=tuple(
                                int(match.chart_id) for match in matches
                            ),
                            reprojection_error_px=error,
                            positive_depth=positive_depth,
                            control_model="chart_factor",
                            seed_model="EPNP",
                            factor_cost=float(factor_cost),
                        )
                    )
        except cv2.error:
            pass
    if include_grouped_center_pose and len(matches) >= 3:
        # The projected chart centers are not independent landmark matches:
        # each is derived from a complete regional frame.  They are used only
        # to produce an alternative SE(3) basin when noisy chart scale/shear
        # makes an all-corner EPNP seed ill-conditioned. The seed and its
        # full-factor refinement are both retained when they differ: a local
        # optimizer is a proposal, not permission to erase a valid basin.
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
                    seed_pose = np.eye(4, dtype=np.float64)
                    seed_pose[:3, :3] = rotation
                    seed_pose[:3, 3] = translation
                    (
                        seed_pose,
                        seed_error,
                        seed_positive,
                        seed_factor_cost,
                    ) = _refine_pose_with_chart_factors(
                        atlas,
                        matches,
                        seed_pose,
                        camera,
                        iterations=0,
                    )
                    frame_evidence = regional_frame_log_evidence(matches)
                    if not bool(refine_grouped_pose):
                        hypotheses.append(
                            FramePoseHypothesis(
                                pose_w2c=seed_pose,
                                score=float(
                                    frame_evidence
                                    - 0.5 * np.log1p(seed_factor_cost)
                                ),
                                source_chart_ids=tuple(
                                    int(match.chart_id)
                                    for match in matches
                                ),
                                reprojection_error_px=seed_error,
                                positive_depth=seed_positive,
                                control_model="regional_chart_seed",
                                seed_model=(
                                    "SQPNP_CHART_CENTERS_UNREFINED"
                                ),
                                factor_cost=float(seed_factor_cost),
                            )
                        )
                    else:
                        pose, error, positive_depth, factor_cost = (
                            _refine_pose_with_chart_factors(
                                atlas, matches, seed_pose, camera
                            )
                        )
                        if _poses_materially_differ(seed_pose, pose):
                            hypotheses.append(
                                FramePoseHypothesis(
                                    pose_w2c=seed_pose,
                                    score=float(
                                        frame_evidence
                                        - 0.5
                                        * np.log1p(seed_factor_cost)
                                    ),
                                    source_chart_ids=tuple(
                                        int(match.chart_id)
                                        for match in matches
                                    ),
                                    reprojection_error_px=seed_error,
                                    positive_depth=seed_positive,
                                    control_model="regional_chart_seed",
                                    seed_model=(
                                        "SQPNP_CHART_CENTERS_UNREFINED"
                                    ),
                                    factor_cost=float(seed_factor_cost),
                                )
                            )
                        hypotheses.append(
                            FramePoseHypothesis(
                                pose_w2c=pose,
                                score=float(
                                    frame_evidence
                                    - 0.5 * np.log1p(factor_cost)
                                ),
                                source_chart_ids=tuple(
                                    int(match.chart_id)
                                    for match in matches
                                ),
                                reprojection_error_px=error,
                                positive_depth=positive_depth,
                                control_model="chart_factor",
                                seed_model="SQPNP_CHART_CENTERS",
                                factor_cost=float(factor_cost),
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
    *,
    refine_grouped_pose: bool = True,
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
                # An alternate regional-center seed protects candidate
                # coverage when noisy chart scale makes the corner seed
                # ill-conditioned. Atlas likelihood remains the final
                # estimator.
                include_grouped_center_pose=len(matches) >= 3,
                refine_grouped_pose=bool(refine_grouped_pose),
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


def pose_distribution_consensus_modes(
    hypotheses: Sequence[FramePoseHypothesis],
    *,
    translation_radius_m: float = 0.85,
    rotation_radius_deg: float = 5.0,
    minimum_support_sets: int = 2,
    maximum_source_charts: int = 6,
    maximum_modes: int = 32,
) -> tuple[FramePoseHypothesis, ...]:
    """Marginalize nearby regional-frame hypotheses into coarse SE(3) modes.

    A chart combination is one structured regional observation, regardless of
    how many solver seeds or near-duplicate frame modes it generated.  The
    function therefore gives every distinct unordered chart-support set one
    vote inside a local SE(3) neighbourhood, then computes an equal-mass
    chordal pose mean.  Raw frame scores are deliberately not used as weights:
    their scales differ with chart count and feature source.
    """

    values = tuple(hypotheses)
    if (
        not values
        or float(translation_radius_m) <= 0.0
        or float(rotation_radius_deg) <= 0.0
        or int(minimum_support_sets) < 2
        or int(maximum_source_charts) <= 0
        or int(maximum_modes) <= 0
    ):
        return ()
    poses = np.asarray(
        [
            np.asarray(value.pose_w2c, dtype=np.float64).reshape(4, 4)
            for value in values
        ]
    )
    rotations = poses[:, :3, :3]
    centers = -np.einsum(
        "nji,nj->ni", rotations, poses[:, :3, 3]
    )
    support_keys = [
        tuple(sorted({int(chart) for chart in value.source_chart_ids}))
        for value in values
    ]
    # A support set can have several genuine planar/IPPE pose branches. Using
    # only its first (frame-score-ranked) branch as a mode-discovery anchor
    # made a shared lower-ranked branch invisible even when several distinct
    # support sets agreed on it. Every branch may therefore seed a local
    # neighbourhood; probability mass is still deduplicated to one member per
    # unordered support set below, so solver duplicates cannot add votes.
    anchor_indices = list(range(len(values)))
    proposed = []
    translation_scale = float(translation_radius_m)
    rotation_scale = float(rotation_radius_deg)
    for anchor_index in anchor_indices:
        translation = np.linalg.norm(
            centers - centers[anchor_index][None], axis=1
        )
        relative = rotations @ rotations[anchor_index].T
        rotation = np.degrees(
            np.arccos(
                np.clip(
                    (np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5,
                    -1.0,
                    1.0,
                )
            )
        )
        rows = np.flatnonzero(
            (translation <= translation_scale)
            & (rotation <= rotation_scale)
        )
        # Retain the most central representative of each support set.  This
        # prevents IPPE/EPNP duplicates from manufacturing probability mass.
        member_by_support: dict[tuple[int, ...], tuple[float, int]] = {}
        for row in rows.tolist():
            normalized_distance = float(
                np.hypot(
                    translation[row] / translation_scale,
                    rotation[row] / rotation_scale,
                )
            )
            previous = member_by_support.get(support_keys[row])
            candidate = (normalized_distance, int(row))
            if previous is None or candidate < previous:
                member_by_support[support_keys[row]] = candidate
        if len(member_by_support) < int(minimum_support_sets):
            continue
        member_indices = [
            value[1] for value in member_by_support.values()
        ]
        member_centers = centers[member_indices]
        member_rotations = rotations[member_indices]
        matrix = np.mean(member_rotations, axis=0)
        left, _singular, right = np.linalg.svd(matrix)
        mean_rotation = (
            left
            @ np.diag(
                [1.0, 1.0, np.linalg.det(left @ right)]
            )
            @ right
        )
        mean_center = np.mean(member_centers, axis=0)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = mean_rotation
        pose[:3, 3] = -mean_rotation @ mean_center
        chart_frequency: dict[int, int] = {}
        for row in member_indices:
            for chart in support_keys[row]:
                chart_frequency[int(chart)] = (
                    chart_frequency.get(int(chart), 0) + 1
                )
        source_charts = tuple(
            chart
            for chart, _count in sorted(
                chart_frequency.items(),
                key=lambda item: (-item[1], item[0]),
            )[: int(maximum_source_charts)]
        )
        member_values = [values[row] for row in member_indices]
        finite_costs = [
            float(value.factor_cost)
            for value in member_values
            if np.isfinite(value.factor_cost)
        ]
        proposed.append(
            FramePoseHypothesis(
                pose_w2c=pose,
                score=float(
                    max(float(value.score) for value in member_values)
                ),
                source_chart_ids=source_charts,
                reprojection_error_px=float(
                    np.median(
                        [
                            float(value.reprojection_error_px)
                            for value in member_values
                        ]
                    )
                ),
                positive_depth=True,
                control_model="chart_distribution_consensus",
                seed_model="SE3_MODE_MEAN",
                factor_cost=(
                    float(np.median(finite_costs))
                    if finite_costs
                    else float("inf")
                ),
                mode_support_count=len(member_by_support),
                mode_member_count=int(rows.size),
            )
        )
    proposed.sort(
        key=lambda value: (
            -int(value.mode_support_count),
            float(value.reprojection_error_px),
            -float(value.score),
        )
    )
    result = []
    for candidate in proposed:
        candidate_pose = np.asarray(candidate.pose_w2c, dtype=np.float64)
        candidate_center = (
            -candidate_pose[:3, :3].T @ candidate_pose[:3, 3]
        )
        duplicate = False
        for retained in result:
            retained_pose = np.asarray(
                retained.pose_w2c, dtype=np.float64
            )
            retained_center = (
                -retained_pose[:3, :3].T @ retained_pose[:3, 3]
            )
            translation = float(
                np.linalg.norm(candidate_center - retained_center)
            )
            relative = (
                candidate_pose[:3, :3] @ retained_pose[:3, :3].T
            )
            rotation = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            (np.trace(relative) - 1.0) * 0.5,
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            if (
                translation < 0.5 * translation_scale
                and rotation < 0.5 * rotation_scale
            ):
                duplicate = True
                break
        if duplicate:
            continue
        result.append(candidate)
        if len(result) >= int(maximum_modes):
            break
    return tuple(result)


def factorized_pose_distribution_modes(
    hypotheses: Sequence[FramePoseHypothesis],
    consensus_modes: Sequence[FramePoseHypothesis],
    *,
    translation_radius_m: float = 1.5,
    rotation_radius_deg: float = 5.0,
    centers_per_mode: int = 6,
    maximum_source_charts: int = 6,
    maximum_modes: int = 192,
) -> tuple[FramePoseHypothesis, ...]:
    """Expose centre/rotation products from a structured planar mode.

    Near-planar chart factors often constrain orientation and camera centre
    with very different uncertainty. A chordal SE(3) mean can consequently
    have the best orientation while a lower-ranked member carries the better
    metric centre. This function preserves that structured uncertainty: each
    consensus rotation is paired with several pose-diverse member centres,
    and the rendered atlas later evaluates their *joint* consistency. No GT,
    mapping pose identity, point correspondence, or extra image is used.
    """

    values = tuple(hypotheses)
    modes = tuple(consensus_modes)
    if (
        not values
        or not modes
        or float(translation_radius_m) <= 0.0
        or float(rotation_radius_deg) <= 0.0
        or int(centers_per_mode) <= 0
        or int(maximum_source_charts) <= 0
        or int(maximum_modes) <= 0
    ):
        return ()
    value_poses = np.asarray(
        [
            np.asarray(value.pose_w2c, dtype=np.float64).reshape(4, 4)
            for value in values
        ]
    )
    value_rotations = value_poses[:, :3, :3]
    value_centers = -np.einsum(
        "nji,nj->ni", value_rotations, value_poses[:, :3, 3]
    )
    support_keys = [
        tuple(sorted({int(chart) for chart in value.source_chart_ids}))
        for value in values
    ]
    proposed = []
    for mode in modes:
        mode_pose = np.asarray(mode.pose_w2c, dtype=np.float64).reshape(4, 4)
        mode_rotation = mode_pose[:3, :3]
        mode_center = -mode_rotation.T @ mode_pose[:3, 3]
        translation = np.linalg.norm(
            value_centers - mode_center[None], axis=1
        )
        relative = value_rotations @ mode_rotation.T
        rotation = np.degrees(
            np.arccos(
                np.clip(
                    (np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5,
                    -1.0,
                    1.0,
                )
            )
        )
        rows = np.flatnonzero(
            (translation <= float(translation_radius_m))
            & (rotation <= float(rotation_radius_deg))
        )
        representative_by_support: dict[
            tuple[int, ...], tuple[float, int]
        ] = {}
        for row in rows.tolist():
            distance = float(
                np.hypot(
                    translation[row] / float(translation_radius_m),
                    rotation[row] / float(rotation_radius_deg),
                )
            )
            candidate = (distance, int(row))
            previous = representative_by_support.get(support_keys[row])
            if previous is None or candidate < previous:
                representative_by_support[support_keys[row]] = candidate
        members = sorted(representative_by_support.values())[
            : int(centers_per_mode)
        ]
        for _distance, row in members:
            center = value_centers[int(row)]
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = mode_rotation
            pose[:3, 3] = -mode_rotation @ center
            member = values[int(row)]
            source_charts = []
            target_source_count = min(
                max(len(member.source_chart_ids), 3),
                int(maximum_source_charts),
            )
            for chart in [
                *member.source_chart_ids,
                *mode.source_chart_ids,
            ]:
                if int(chart) not in source_charts:
                    source_charts.append(int(chart))
                if len(source_charts) >= target_source_count:
                    break
            proposed.append(
                FramePoseHypothesis(
                    pose_w2c=pose,
                    score=float(mode.score),
                    source_chart_ids=tuple(source_charts),
                    reprojection_error_px=float(
                        max(
                            mode.reprojection_error_px,
                            member.reprojection_error_px,
                        )
                    ),
                    positive_depth=bool(
                        mode.positive_depth and member.positive_depth
                    ),
                    control_model="chart_distribution_factorized",
                    seed_model="SE3_CONSENSUS_ROTATION_MEMBER_CENTER",
                    factor_cost=float(
                        max(mode.factor_cost, member.factor_cost)
                    ),
                    mode_support_count=int(mode.mode_support_count),
                    mode_member_count=int(mode.mode_member_count),
                )
            )
    proposed.sort(
        key=lambda value: (
            -int(value.mode_support_count),
            float(value.reprojection_error_px),
            -float(value.score),
        )
    )
    result = []
    for candidate in proposed:
        candidate_pose = np.asarray(candidate.pose_w2c, dtype=np.float64)
        candidate_center = (
            -candidate_pose[:3, :3].T @ candidate_pose[:3, 3]
        )
        duplicate = False
        for retained in result:
            retained_pose = np.asarray(retained.pose_w2c, dtype=np.float64)
            retained_center = (
                -retained_pose[:3, :3].T @ retained_pose[:3, 3]
            )
            translation = float(
                np.linalg.norm(candidate_center - retained_center)
            )
            relative = candidate_pose[:3, :3] @ retained_pose[:3, :3].T
            rotation = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            (np.trace(relative) - 1.0) * 0.5,
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            if translation < 0.15 and rotation < 1.0:
                duplicate = True
                break
        if duplicate:
            continue
        result.append(candidate)
        if len(result) >= int(maximum_modes):
            break
    return tuple(result)
