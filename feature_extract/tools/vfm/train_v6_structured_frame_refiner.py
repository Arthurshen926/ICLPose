"""Train the V6 complete-chart RADIO correlation refiner.

Training and validation views are trajectory-disjoint from each other and
from the atlas mapping trajectories.  Geometry supplies complete chart
projection targets; no point identity, ALIKE descriptor, SfM track, reference
image, or pairwise image matcher enters the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.tools.vfm.train_v6_structured_frame_adapter import (
    PreparedView,
    _prepare_views,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.maplet_frame_alignment import (
    MapletFrameSearchConfig,
    align_maplet_frame_global,
    ground_truth_chart_frame,
)
from feature_extract.vfm.localization_v6.structured_frame_refiner import (
    StructuredFrameRefiner,
    StructuredFrameRefinerConfig,
    load_structured_frame_refiner,
    save_structured_frame_refiner,
    structured_frame_correlation,
)
from feature_extract.vfm.localization_v6.surface_spatial_projection import (
    load_surface_spatial_projection,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio_atlas", required=True)
    parser.add_argument("--surface_mapper_checkpoint", default="")
    parser.add_argument("--frame_spatial_projection_checkpoint", default="")
    parser.add_argument("--train_contributor_dir", required=True)
    parser.add_argument("--validation_contributor_dir", required=True)
    parser.add_argument(
        "--train_trajectory_ids",
        nargs="*",
        default=("seq9", "seq10", "seq12", "seq14"),
    )
    parser.add_argument(
        "--validation_trajectory_ids", nargs="*", default=("seq11",)
    )
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--maximum_train_views", type=int, default=0)
    parser.add_argument("--maximum_validation_views", type=int, default=0)
    parser.add_argument("--validation_every", type=int, default=50)
    parser.add_argument("--batch_charts", type=int, default=8)
    parser.add_argument("--cells_per_chart", type=int, default=64)
    parser.add_argument(
        "--validation_episodes_per_view", type=int, default=2
    )
    parser.add_argument("--positive_fraction", type=float, default=0.75)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--correlation_radius", type=int, default=3)
    parser.add_argument(
        "--linear_parameterization",
        choices=("additive", "matrix_exponential"),
        default="additive",
    )
    parser.add_argument("--maximum_linear_log", type=float, default=3.2)
    parser.add_argument(
        "--update_parameterization",
        choices=("query_affine", "canonical_residual"),
        default="canonical_residual",
    )
    parser.add_argument(
        "--candidate_source",
        choices=("deployment_replay", "synthetic"),
        default="deployment_replay",
    )
    parser.add_argument("--replay_charts_per_view", type=int, default=4)
    parser.add_argument(
        "--replay_wrong_charts_per_view", type=int, default=1
    )
    parser.add_argument("--replay_modes_per_chart", type=int, default=8)
    parser.add_argument(
        "--replay_max_affine_residual_cells", type=float, default=0.75
    )
    parser.add_argument("--initial_checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=4171)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


@dataclass(frozen=True)
class DeploymentReplayMode:
    view_index: int
    chart_row: int
    view_chart_index: int
    canonical_homography: np.ndarray
    target_homography: np.ndarray | None
    usable: bool
    before_error_cells: float
    fitted_error_cells: float
    wrong_identity: bool


class _SurfaceSpatialProjectionMapper:
    """Expose the phase-preserving RADIO projection through mapper protocol."""

    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.model = model
        self.device = str(device)

    @torch.no_grad()
    def project(self, radio_final: np.ndarray) -> object:
        source = torch.from_numpy(
            np.asarray(radio_final, dtype=np.float32)
        ).to(self.device)
        if source.ndim != 3:
            raise ValueError("RADIO-final feature map must have shape (C,H,W)")
        projected = self.model(source.permute(1, 2, 0)).permute(2, 0, 1)
        return type(
            "ProjectedMeasurement",
            (),
            {
                "measurement_context": projected.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            },
        )()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_projection(
    target_xy: np.ndarray,
    *,
    usable: bool,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    target = np.asarray(target_xy, dtype=np.float32)
    center = np.mean(target, axis=0)
    if bool(usable):
        angle = rng.uniform(-np.deg2rad(18.0), np.deg2rad(18.0))
        log_scale = rng.uniform(-0.22, 0.22)
        anisotropy = rng.uniform(-0.12, 0.12)
        shear = rng.uniform(-0.12, 0.12)
        translation = rng.uniform(-2.75, 2.75, size=2)
    else:
        if rng.random() < 0.55:
            destination = np.asarray(
                [
                    rng.uniform(0.0, max(width - 1, 0)),
                    rng.uniform(0.0, max(height - 1, 0)),
                ],
                dtype=np.float32,
            )
            translation = destination - center
        else:
            radius = rng.uniform(4.5, 12.0)
            direction = rng.uniform(-np.pi, np.pi)
            translation = radius * np.asarray(
                [np.cos(direction), np.sin(direction)]
            )
        angle = rng.uniform(-np.pi, np.pi)
        log_scale = rng.uniform(-0.60, 0.60)
        anisotropy = rng.uniform(-0.30, 0.30)
        shear = rng.uniform(-0.30, 0.30)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float32
    )
    scale = np.asarray(
        [
            [np.exp(log_scale + anisotropy), shear],
            [0.0, np.exp(log_scale - anisotropy)],
        ],
        dtype=np.float32,
    )
    return (
        (target - center[None]) @ (rotation @ scale).T
        + center[None]
        + np.asarray(translation, dtype=np.float32)[None]
    ).astype(np.float32)


def _replay_search_config() -> MapletFrameSearchConfig:
    """Exactly mirror the deployed coarse chart search distribution."""

    return MapletFrameSearchConfig(
        geometric_mean_sizes=(0.9, 1.5, 2.5, 4.0, 6.5, 10.0, 15.0),
        maximum_modes=16,
        per_transform_peaks=4,
        minimum_template_support_cells=1.5,
        minimum_overlap_fraction=0.55,
        spatial_mode_nms_radius_cells=2.0,
        diagnostic_null_score=0.25,
        support_score_power=0.0,
    )


def _canonical_for_rows(
    rows: np.ndarray,
    atlas: MapletFeatureAtlasBank,
) -> tuple[int, np.ndarray]:
    cell_count = int(atlas.height * atlas.width)
    rows = np.asarray(rows, dtype=np.int64)
    chart_rows = rows // cell_count
    if rows.size == 0 or np.any(chart_rows != chart_rows[0]):
        raise ValueError("replay rows do not belong to one chart")
    local = rows % cell_count
    y = local // atlas.width
    x = local % atlas.width
    canonical = np.stack(
        [
            x / max(atlas.width - 1, 1) * 2.0 - 1.0,
            y / max(atlas.height - 1, 1) * 2.0 - 1.0,
        ],
        axis=1,
    ).astype(np.float32)
    return int(chart_rows[0]), canonical


def _target_feature_xy(view: PreparedView, xy: np.ndarray) -> np.ndarray:
    query_height, query_width = view.query_feature.shape[-2:]
    image_width, image_height = view.image_size_wh
    points = np.asarray(xy, dtype=np.float32)
    return np.stack(
        [
            (points[:, 0] + 0.5) * query_width / image_width - 0.5,
            (points[:, 1] + 0.5) * query_height / image_height - 0.5,
        ],
        axis=1,
    ).astype(np.float32)


def _match_homography(match: object) -> np.ndarray:
    homography = getattr(match, "canonical_homography", None)
    if homography is None:
        homography = np.vstack(
            [
                np.asarray(
                    getattr(match, "canonical_to_query"),
                    dtype=np.float64,
                ),
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            ]
        )
    value = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    if abs(float(value[2, 2])) <= 1e-8:
        raise ValueError("replay candidate homography is singular")
    return (value / float(value[2, 2])).astype(np.float32)


def _project_homography(
    homography: np.ndarray, canonical_uv: np.ndarray
) -> np.ndarray:
    uv = np.asarray(canonical_uv, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.c_[uv, np.ones((uv.shape[0],))]
    warped = homogeneous @ np.asarray(
        homography, dtype=np.float64
    ).T
    denominator = warped[:, 2:3]
    safe = np.where(
        np.abs(denominator) > 1e-8,
        denominator,
        np.where(denominator < 0.0, -1e-8, 1e-8),
    )
    return (warped[:, :2] / safe).astype(np.float32)


def _replay_correctability(
    candidate_xy: np.ndarray,
    target_xy: np.ndarray,
    canonical_uv: np.ndarray,
    *,
    maximum_fitted_error_cells: float,
    maximum_linear_residual: float,
    maximum_translation_cells: float = 3.5,
    linear_parameterization: str,
    update_parameterization: str,
) -> tuple[bool, float, float]:
    """Test whether the deployed bounded residual can represent the target."""

    candidate = np.asarray(candidate_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    canonical = np.asarray(canonical_uv, dtype=np.float64).reshape(-1, 2)
    if (
        candidate.shape != target.shape
        or canonical.shape != candidate.shape
        or candidate.shape[0] < 4
        or not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(target))
    ):
        return False, float("inf"), float("inf")
    if str(update_parameterization) == "canonical_residual":
        try:
            coefficients = np.linalg.lstsq(
                np.c_[canonical, np.ones((canonical.shape[0],))],
                target - candidate,
                rcond=None,
            )[0]
        except np.linalg.LinAlgError:
            return False, float("inf"), float("inf")
        linear = coefficients[:2].T
        translation = coefficients[2]
        fitted = (
            candidate
            + canonical @ linear.T
            + translation[None]
        )
        linear_representable = bool(
            np.all(np.isfinite(linear))
            and float(np.max(np.abs(linear)))
            <= float(maximum_linear_residual)
        )
    elif str(update_parameterization) == "query_affine":
        center = np.mean(candidate, axis=0)
        target_center = np.mean(target, axis=0)
        try:
            linear = np.linalg.lstsq(
                candidate - center[None],
                target - target_center[None],
                rcond=None,
            )[0].T
        except np.linalg.LinAlgError:
            return False, float("inf"), float("inf")
        translation = target_center - center
        fitted = (
            (candidate - center[None]) @ linear.T
            + center[None]
            + translation[None]
        )
        determinant = float(np.linalg.det(linear))
        if str(linear_parameterization) == "matrix_exponential":
            singular = np.linalg.svd(linear, compute_uv=False)
            scale_limit = float(
                np.exp(float(maximum_linear_residual))
            )
            linear_representable = bool(
                determinant > 0.0
                and float(np.min(singular)) >= 1.0 / scale_limit
                and float(np.max(singular)) <= scale_limit
            )
        elif str(linear_parameterization) == "additive":
            linear_representable = bool(
                abs(determinant) > 0.05
                and float(
                    np.max(
                        np.abs(
                            linear - np.eye(2, dtype=np.float64)
                        )
                    )
                )
                <= float(maximum_linear_residual)
            )
        else:
            raise ValueError("unknown replay linear parameterization")
    else:
        raise ValueError("unknown replay update parameterization")
    before = float(
        np.mean(np.linalg.norm(candidate - target, axis=1))
    )
    residual = float(
        np.mean(np.linalg.norm(fitted - target, axis=1))
    )
    representable = (
        np.all(np.isfinite(linear))
        and linear_representable
        and float(np.max(np.abs(translation)))
        <= float(maximum_translation_cells)
    )
    return (
        bool(
            representable
            and residual <= float(maximum_fitted_error_cells)
        ),
        before,
        residual,
    )


def _build_deployment_replay(
    views: Sequence[PreparedView],
    atlas: MapletFeatureAtlasBank,
    *,
    charts_per_view: int,
    wrong_charts_per_view: int,
    modes_per_chart: int,
    maximum_fitted_error_cells: float,
    maximum_linear_residual: float,
    linear_parameterization: str,
    update_parameterization: str,
    seed: int,
    device: str,
) -> tuple[list[DeploymentReplayMode], dict[str, object]]:
    """Replay the real global-search output without consulting test views."""

    if (
        int(charts_per_view) <= 0
        or int(wrong_charts_per_view) < 0
        or int(modes_per_chart) <= 0
        or float(maximum_fitted_error_cells) <= 0.0
    ):
        raise ValueError("invalid deployment replay parameters")
    rng = np.random.default_rng(int(seed))
    config = _replay_search_config()
    cell_count = int(atlas.height * atlas.width)
    atlas_usable = (
        np.mean(atlas.valid_mask, axis=(1, 2)) >= 0.01
    ) & (np.sum(atlas.support_count, axis=(1, 2)) > 0)
    all_chart_rows = np.flatnonzero(atlas_usable)
    replay: list[DeploymentReplayMode] = []
    correct_searches = 0
    wrong_searches = 0
    for view_index, view in enumerate(views):
        query = view.query_feature.to(device=device)
        order = np.argsort(
            -np.asarray(
                [rows.size for rows in view.chart_rows], dtype=np.int64
            ),
            kind="mergesort",
        )[: int(charts_per_view)]
        visible_chart_rows = set()
        for chart_index, rows in enumerate(view.chart_rows):
            chart_row, _canonical = _canonical_for_rows(rows, atlas)
            visible_chart_rows.add(int(chart_row))
        for chart_index in order.tolist():
            rows = np.asarray(
                view.chart_rows[int(chart_index)], dtype=np.int64
            )
            chart_row, _visible_canonical = _canonical_for_rows(
                rows, atlas
            )
            target_match = ground_truth_chart_frame(
                atlas,
                int(atlas.maplet_ids[chart_row]),
                view.pose_w2c,
                view.camera,
                feature_stride=16,
                feature_level="training_deployment_replay_gt",
                model="homography",
            )
            if target_match is None:
                continue
            target_homography = _match_homography(target_match)
            valid_local = np.flatnonzero(
                np.asarray(atlas.valid_mask[chart_row], bool)
                & (np.asarray(atlas.support_count[chart_row]) > 0)
            )
            full_rows = (
                int(chart_row) * cell_count + valid_local
            ).astype(np.int64)
            _full_chart_row, canonical = _canonical_for_rows(
                full_rows, atlas
            )
            target = _project_homography(
                target_homography, canonical
            )
            matches = align_maplet_frame_global(
                atlas,
                int(atlas.maplet_ids[chart_row]),
                query,
                feature_level="coarse",
                feature_stride=16,
                config=config,
            )
            correct_searches += 1
            for match in matches[: int(modes_per_chart)]:
                homography = _match_homography(match)
                candidate = _project_homography(
                    homography, canonical
                )
                usable, before, fitted = _replay_correctability(
                    candidate,
                    target,
                    canonical,
                    maximum_fitted_error_cells=float(
                        maximum_fitted_error_cells
                    ),
                    maximum_linear_residual=float(
                        maximum_linear_residual
                    ),
                    linear_parameterization=str(
                        linear_parameterization
                    ),
                    update_parameterization=str(
                        update_parameterization
                    ),
                )
                replay.append(
                    DeploymentReplayMode(
                        view_index=int(view_index),
                        chart_row=int(chart_row),
                        view_chart_index=int(chart_index),
                        canonical_homography=homography,
                        target_homography=target_homography,
                        usable=bool(usable),
                        before_error_cells=float(before),
                        fitted_error_cells=float(fitted),
                        wrong_identity=False,
                    )
                )
        wrong_pool = np.asarray(
            [
                row
                for row in all_chart_rows.tolist()
                if int(row) not in visible_chart_rows
            ],
            dtype=np.int64,
        )
        if wrong_pool.size and int(wrong_charts_per_view) > 0:
            wrong_rows = rng.choice(
                wrong_pool,
                size=min(int(wrong_charts_per_view), wrong_pool.size),
                replace=False,
            )
            for chart_row in wrong_rows.tolist():
                matches = align_maplet_frame_global(
                    atlas,
                    int(atlas.maplet_ids[int(chart_row)]),
                    query,
                    feature_level="coarse",
                    feature_stride=16,
                    config=config,
                )
                wrong_searches += 1
                for match in matches[: int(modes_per_chart)]:
                    replay.append(
                        DeploymentReplayMode(
                            view_index=int(view_index),
                            chart_row=int(chart_row),
                            view_chart_index=-1,
                            canonical_homography=_match_homography(match),
                            target_homography=None,
                            usable=False,
                            before_error_cells=float("inf"),
                            fitted_error_cells=float("inf"),
                            wrong_identity=True,
                        )
                    )
        print(
            json.dumps(
                {
                    "replay_progress": f"{view_index + 1}/{len(views)}",
                    "trajectory_id": view.trajectory_id,
                    "mode_count": len(replay),
                }
            ),
            flush=True,
        )
    positive = [value for value in replay if value.usable]
    correct = [value for value in replay if not value.wrong_identity]
    finite_before = [
        value.before_error_cells
        for value in positive
        if np.isfinite(value.before_error_cells)
    ]
    summary = {
        "view_count": int(len(views)),
        "mode_count": int(len(replay)),
        "positive_mode_count": int(len(positive)),
        "positive_fraction": float(
            len(positive) / max(len(replay), 1)
        ),
        "wrong_identity_mode_count": int(
            sum(value.wrong_identity for value in replay)
        ),
        "correct_search_count": int(correct_searches),
        "wrong_search_count": int(wrong_searches),
        "positive_before_error_median_cells": (
            float(np.median(finite_before)) if finite_before else None
        ),
        "correct_before_error_min_cells": (
            float(
                np.min(
                    [value.before_error_cells for value in correct]
                )
            )
            if correct
            else None
        ),
        "correct_fitted_error_min_cells": (
            float(
                np.min(
                    [value.fitted_error_cells for value in correct]
                )
            )
            if correct
            else None
        ),
    }
    print(json.dumps({"replay_summary": summary}), flush=True)
    if not positive or len(positive) == len(replay):
        raise ValueError("deployment replay lacks positive or null modes")
    return replay, summary


def _episode_arrays(
    view: PreparedView,
    atlas: MapletFeatureAtlasBank,
    *,
    batch_charts: int,
    cells_per_chart: int,
    positive_fraction: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    available = len(view.chart_rows)
    selected = rng.choice(
        available,
        size=int(batch_charts),
        replace=available < int(batch_charts),
    )
    cell_count = int(atlas.height * atlas.width)
    query_height, query_width = view.query_feature.shape[-2:]
    image_width, image_height = view.image_size_wh
    mean_rows = []
    mode_rows = []
    weight_rows = []
    valid_rows = []
    canonical_rows = []
    target_rows = []
    candidate_rows = []
    usable_rows = []
    for chart_index in selected.tolist():
        rows = np.asarray(view.chart_rows[int(chart_index)], dtype=np.int64)
        xy = np.asarray(view.chart_xy[int(chart_index)], dtype=np.float32)
        choice = rng.choice(
            rows.size,
            size=int(cells_per_chart),
            replace=rows.size < int(cells_per_chart),
        )
        rows = rows[choice]
        xy = xy[choice]
        chart_row = rows // cell_count
        local = rows % cell_count
        y = local // atlas.width
        x = local % atlas.width
        mean_rows.append(atlas.features[chart_row, :, y, x])
        if atlas.mode_features is not None:
            mode_rows.append(
                atlas.mode_features[chart_row, :, :, y, x]
            )
            weight_rows.append(atlas.mode_weights[chart_row, :, y, x])
            valid_rows.append(
                atlas.mode_valid_mask[chart_row, :, y, x]
            )
        canonical_rows.append(
            np.stack(
                [
                    x / max(atlas.width - 1, 1) * 2.0 - 1.0,
                    y / max(atlas.height - 1, 1) * 2.0 - 1.0,
                ],
                axis=1,
            ).astype(np.float32)
        )
        target = np.stack(
            [
                (xy[:, 0] + 0.5) * query_width / image_width - 0.5,
                (xy[:, 1] + 0.5) * query_height / image_height - 0.5,
            ],
            axis=1,
        ).astype(np.float32)
        usable = bool(rng.random() < float(positive_fraction))
        target_rows.append(target)
        candidate_rows.append(
            _candidate_projection(
                target,
                usable=usable,
                width=int(query_width),
                height=int(query_height),
                rng=rng,
            )
        )
        usable_rows.append(usable)
    result = {
        "map_feature": np.stack(mean_rows).astype(np.float32),
        "canonical_uv": np.stack(canonical_rows).astype(np.float32),
        "target_xy": np.stack(target_rows).astype(np.float32),
        "candidate_xy": np.stack(candidate_rows).astype(np.float32),
        "usable": np.asarray(usable_rows, dtype=np.float32),
    }
    if mode_rows:
        result.update(
            map_mode_feature=np.stack(mode_rows).astype(np.float32),
            map_mode_weight=np.stack(weight_rows).astype(np.float32),
            map_mode_valid=np.stack(valid_rows).astype(bool),
        )
    return result


def _loss_from_arrays(
    model: StructuredFrameRefiner,
    query_feature: torch.Tensor,
    arrays: Mapping[str, np.ndarray],
    *,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    dtype = torch.float32
    query = query_feature.to(device=device, dtype=dtype)
    if query.ndim == 3:
        query = query[None]
    map_feature = torch.from_numpy(arrays["map_feature"]).to(
        device=device, dtype=dtype
    )
    candidate = torch.from_numpy(arrays["candidate_xy"]).to(
        device=device, dtype=dtype
    )
    target = torch.from_numpy(arrays["target_xy"]).to(
        device=device, dtype=dtype
    )
    canonical = torch.from_numpy(arrays["canonical_uv"]).to(
        device=device, dtype=dtype
    )
    usable = torch.from_numpy(arrays["usable"]).to(
        device=device, dtype=dtype
    )
    modes = (
        torch.from_numpy(arrays["map_mode_feature"]).to(
            device=device, dtype=dtype
        )
        if "map_mode_feature" in arrays
        else None
    )
    weights = (
        torch.from_numpy(arrays["map_mode_weight"]).to(
            device=device, dtype=dtype
        )
        if "map_mode_weight" in arrays
        else None
    )
    mode_valid = (
        torch.from_numpy(arrays["map_mode_valid"]).to(device=device)
        if "map_mode_valid" in arrays
        else None
    )
    correlation = structured_frame_correlation(
        query,
        map_feature,
        modes,
        weights,
        mode_valid,
        candidate,
        radius=int(model.config.correlation_radius),
    )
    height, width = query.shape[-2:]
    candidate_normalized = candidate.clone()
    candidate_normalized[..., 0] = (
        2.0 * (candidate_normalized[..., 0] + 0.5) / width - 1.0
    )
    candidate_normalized[..., 1] = (
        2.0 * (candidate_normalized[..., 1] + 0.5) / height - 1.0
    )
    prediction = model(
        correlation, canonical, candidate_normalized
    )
    corrected = model.apply_update(
        candidate,
        prediction["linear"],
        prediction["translation"],
        canonical,
    )
    point_error = torch.linalg.vector_norm(corrected - target, dim=2)
    before_error = torch.mean(
        torch.linalg.vector_norm(candidate - target, dim=2), dim=1
    )
    after_error = torch.mean(point_error, dim=1)
    positive = usable > 0.5
    regression = (
        F.smooth_l1_loss(
            corrected[positive],
            target[positive],
            beta=0.25,
        )
        if bool(torch.any(positive))
        else torch.zeros((), device=corrected.device)
    )
    classification = F.binary_cross_entropy_with_logits(
        prediction["usable_logit"], usable
    )
    negative_motion = (
        torch.mean(
            (
                prediction["linear"][~positive]
                - torch.eye(
                    2, device=device, dtype=prediction["linear"].dtype
                )[None]
            )
            ** 2
        )
        + torch.mean(prediction["translation"][~positive] ** 2)
        if bool(torch.any(~positive))
        else torch.zeros((), device=corrected.device)
    )
    loss = regression + 0.30 * classification + 0.01 * negative_motion
    positive_count = max(int(torch.sum(positive).item()), 1)
    metrics = {
        "loss": float(loss.detach().item()),
        "regression_loss": float(regression.detach().item()),
        "usable_bce": float(classification.detach().item()),
        "positive_before_error_cells": float(
            torch.sum(before_error * positive).item() / positive_count
        ),
        "positive_after_error_cells": float(
            torch.sum(after_error * positive).item() / positive_count
        ),
        "positive_improvement_fraction": float(
            torch.sum(
                ((after_error < before_error) & positive).to(torch.float32)
            ).item()
            / positive_count
        ),
        "positive_recall_1cell": float(
            torch.sum(
                ((after_error <= 1.0) & positive).to(torch.float32)
            ).item()
            / positive_count
        ),
        "positive_recall_1_5cell": float(
            torch.sum(
                ((after_error <= 1.5) & positive).to(torch.float32)
            ).item()
            / positive_count
        ),
        "usable_accuracy": float(
            torch.mean(
                (
                    (prediction["usable_logit"] >= 0.0)
                    == positive
                ).to(torch.float32)
            ).item()
        ),
    }
    return loss, metrics


def _episode(
    model: StructuredFrameRefiner,
    view: PreparedView,
    atlas: MapletFeatureAtlasBank,
    *,
    batch_charts: int,
    cells_per_chart: int,
    positive_fraction: float,
    rng: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    arrays = _episode_arrays(
        view,
        atlas,
        batch_charts=int(batch_charts),
        cells_per_chart=int(cells_per_chart),
        positive_fraction=float(positive_fraction),
        rng=rng,
    )
    return _loss_from_arrays(
        model, view.query_feature[None], arrays, device=str(device)
    )


def _replay_episode_arrays(
    views: Sequence[PreparedView],
    atlas: MapletFeatureAtlasBank,
    replay: Sequence[DeploymentReplayMode],
    *,
    batch_charts: int,
    cells_per_chart: int,
    positive_fraction: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    positive_rows = np.asarray(
        [row for row, value in enumerate(replay) if value.usable],
        dtype=np.int64,
    )
    negative_rows = np.asarray(
        [row for row, value in enumerate(replay) if not value.usable],
        dtype=np.int64,
    )
    positive_count = min(
        max(int(round(int(batch_charts) * float(positive_fraction))), 1),
        int(batch_charts) - 1,
    )
    negative_count = int(batch_charts) - positive_count
    selected = np.concatenate(
        [
            rng.choice(
                positive_rows,
                size=positive_count,
                replace=positive_rows.size < positive_count,
            ),
            rng.choice(
                negative_rows,
                size=negative_count,
                replace=negative_rows.size < negative_count,
            ),
        ]
    )
    rng.shuffle(selected)
    cell_count = int(atlas.height * atlas.width)
    query_rows = []
    mean_rows = []
    mode_rows = []
    weight_rows = []
    valid_rows = []
    canonical_rows = []
    target_rows = []
    candidate_rows = []
    usable_rows = []
    for replay_row in selected.tolist():
        value = replay[int(replay_row)]
        view = views[int(value.view_index)]
        local = np.flatnonzero(
            np.asarray(atlas.valid_mask[int(value.chart_row)], bool)
            & (
                np.asarray(
                    atlas.support_count[int(value.chart_row)]
                )
                > 0
            )
        )
        source_rows = (
            int(value.chart_row) * cell_count + local
        ).astype(np.int64)
        choice = rng.choice(
            source_rows.size,
            size=int(cells_per_chart),
            replace=source_rows.size < int(cells_per_chart),
        )
        rows = source_rows[choice]
        chart_row, canonical = _canonical_for_rows(rows, atlas)
        if chart_row != int(value.chart_row):
            raise ValueError("replay chart lineage differs")
        local = rows % cell_count
        y = local // atlas.width
        x = local % atlas.width
        candidate = _project_homography(
            value.canonical_homography, canonical
        )
        target = (
            _project_homography(value.target_homography, canonical)
            if value.target_homography is not None
            else candidate.copy()
        )
        query_rows.append(view.query_feature)
        mean_rows.append(atlas.features[chart_row, :, y, x])
        if atlas.mode_features is not None:
            mode_rows.append(
                atlas.mode_features[chart_row, :, :, y, x]
            )
            weight_rows.append(atlas.mode_weights[chart_row, :, y, x])
            valid_rows.append(
                atlas.mode_valid_mask[chart_row, :, y, x]
            )
        canonical_rows.append(canonical)
        target_rows.append(target)
        candidate_rows.append(candidate)
        usable_rows.append(bool(value.usable))
    arrays: dict[str, np.ndarray] = {
        "map_feature": np.stack(mean_rows).astype(np.float32),
        "canonical_uv": np.stack(canonical_rows).astype(np.float32),
        "target_xy": np.stack(target_rows).astype(np.float32),
        "candidate_xy": np.stack(candidate_rows).astype(np.float32),
        "usable": np.asarray(usable_rows, dtype=np.float32),
    }
    if mode_rows:
        arrays.update(
            map_mode_feature=np.stack(mode_rows).astype(np.float32),
            map_mode_weight=np.stack(weight_rows).astype(np.float32),
            map_mode_valid=np.stack(valid_rows).astype(bool),
        )
    return torch.stack(query_rows), arrays


def _replay_episode(
    model: StructuredFrameRefiner,
    views: Sequence[PreparedView],
    atlas: MapletFeatureAtlasBank,
    replay: Sequence[DeploymentReplayMode],
    *,
    batch_charts: int,
    cells_per_chart: int,
    positive_fraction: float,
    rng: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    query, arrays = _replay_episode_arrays(
        views,
        atlas,
        replay,
        batch_charts=int(batch_charts),
        cells_per_chart=int(cells_per_chart),
        positive_fraction=float(positive_fraction),
        rng=rng,
    )
    return _loss_from_arrays(
        model, query, arrays, device=str(device)
    )


@torch.no_grad()
def _validate(
    model: StructuredFrameRefiner,
    views: Sequence[PreparedView],
    atlas: MapletFeatureAtlasBank,
    *,
    batch_charts: int,
    cells_per_chart: int,
    positive_fraction: float,
    episodes_per_view: int,
    seed: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    rows = []
    for view_index, view in enumerate(views):
        for episode in range(max(int(episodes_per_view), 1)):
            _loss, metrics = _episode(
                model,
                view,
                atlas,
                batch_charts=int(batch_charts),
                cells_per_chart=int(cells_per_chart),
                positive_fraction=float(positive_fraction),
                rng=np.random.default_rng(
                    int(seed) + 1000 * view_index + episode
                ),
                device=str(device),
            )
            rows.append(metrics)
    result = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
    result["selection_score"] = float(
        -result["positive_after_error_cells"]
        + 0.5 * result["positive_improvement_fraction"]
        + 0.5 * result["positive_recall_1cell"]
        + 0.10 * result["usable_accuracy"]
    )
    return result


@torch.no_grad()
def _validate_replay(
    model: StructuredFrameRefiner,
    views: Sequence[PreparedView],
    atlas: MapletFeatureAtlasBank,
    replay: Sequence[DeploymentReplayMode],
    *,
    batch_charts: int,
    cells_per_chart: int,
    positive_fraction: float,
    episode_count: int,
    seed: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    rows = []
    for episode in range(max(int(episode_count), 1)):
        _loss, metrics = _replay_episode(
            model,
            views,
            atlas,
            replay,
            batch_charts=int(batch_charts),
            cells_per_chart=int(cells_per_chart),
            positive_fraction=float(positive_fraction),
            rng=np.random.default_rng(int(seed) + episode),
            device=str(device),
        )
        rows.append(metrics)
    result = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
    result["selection_score"] = float(
        -result["positive_after_error_cells"]
        + 0.5 * result["positive_improvement_fraction"]
        + 0.5 * result["positive_recall_1cell"]
        + 0.10 * result["usable_accuracy"]
    )
    return result


def _metadata(
    atlas_path: Path,
    feature_transform_path: Path,
    feature_transform_kind: str,
    atlas: MapletFeatureAtlasBank,
    train_trajectories: Sequence[str],
    validation_trajectories: Sequence[str],
    best_step: int,
    best_metrics: Mapping[str, float],
    *,
    candidate_source: str,
    training_positive_fraction: float,
) -> dict[str, object]:
    result = {
        "vfm_layer": "radio_final",
        "training_objective": (
            "complete_chart_local_correlation_to_structured_affine_"
            "residual_and_explicit_null"
        ),
        "compatible_radio_atlas_sha256": _sha256(atlas_path),
        "query_feature_transform": str(feature_transform_kind),
        "query_feature_transform_sha256": _sha256(
            feature_transform_path
        ),
        "atlas_mapping_trajectory_ids": sorted(
            str(value)
            for value in (atlas.metadata or {}).get(
                "mapping_trajectory_ids", []
            )
        ),
        "training_trajectory_ids": list(train_trajectories),
        "validation_trajectory_ids": list(validation_trajectories),
        "best_step": int(best_step),
        "best_validation_metrics": dict(best_metrics),
        "candidate_source": str(candidate_source),
        "training_positive_fraction": float(training_positive_fraction),
        "stores_mapping_rgb": False,
        "stores_mapping_image_ids": False,
        "stores_mapping_image_paths": False,
        "uses_rgb_after_radio": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_pairwise_image_matching": False,
        "uses_point_correspondence_pnp": False,
    }
    if str(feature_transform_kind) == "surface_maplet_mapper":
        result["surface_mapper_checkpoint_sha256"] = _sha256(
            feature_transform_path
        )
    elif str(feature_transform_kind) == "surface_spatial_projection":
        result["frame_spatial_projection_checkpoint_sha256"] = _sha256(
            feature_transform_path
        )
    else:
        raise ValueError("unsupported frame feature transform")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_checkpoint)
    summary_path = Path(args.summary_json)
    if not bool(args.force) and (output.exists() or summary_path.exists()):
        raise FileExistsError("output exists; pass --force to replace it")
    atlas_path = Path(args.radio_atlas)
    has_mapper = bool(str(args.surface_mapper_checkpoint))
    has_spatial_projection = bool(
        str(args.frame_spatial_projection_checkpoint)
    )
    if has_mapper == has_spatial_projection:
        raise ValueError(
            "provide exactly one frame feature transform: surface mapper or "
            "surface spatial projection"
        )
    atlas = MapletFeatureAtlasBank.load_npz(atlas_path)
    if has_mapper:
        feature_transform_kind = "surface_maplet_mapper"
        feature_transform_path = Path(args.surface_mapper_checkpoint)
        mapper, _feature_transform_metadata = load_surface_maplet_mapper(
            feature_transform_path, device=str(args.device)
        )
    else:
        feature_transform_kind = "surface_spatial_projection"
        feature_transform_path = Path(
            args.frame_spatial_projection_checkpoint
        )
        projection, _feature_transform_metadata = (
            load_surface_spatial_projection(
                feature_transform_path, device=str(args.device)
            )
        )
        projection.eval()
        mapper = _SurfaceSpatialProjectionMapper(
            projection, str(args.device)
        )
    atlas_metadata = dict(atlas.metadata or {})
    if str(atlas_metadata.get("query_feature_transform", "")) != str(
        feature_transform_kind
    ):
        raise ValueError("frame feature transform differs from atlas")
    if str(
        atlas_metadata.get("query_feature_transform_sha256", "")
    ) != _sha256(feature_transform_path):
        raise ValueError("frame feature transform lineage differs from atlas")
    train_raw = _load_views(
        Path(args.train_contributor_dir),
        atlas,
        Path(args.image_root),
        tuple(str(value) for value in args.train_trajectory_ids),
    )
    validation_raw = _load_views(
        Path(args.validation_contributor_dir),
        atlas,
        Path(args.image_root),
        tuple(str(value) for value in args.validation_trajectory_ids),
    )
    # Apply explicit smoke/debug limits before projecting RADIO-final maps.
    # Projection is deterministic per view, so this is equivalent to slicing
    # the prepared prefix and avoids needlessly materializing every view.
    if int(args.maximum_train_views) > 0:
        train_raw = train_raw[: int(args.maximum_train_views)]
    if int(args.maximum_validation_views) > 0:
        validation_raw = validation_raw[
            : int(args.maximum_validation_views)
        ]
    train_views = _prepare_views(train_raw, atlas, mapper)
    validation_views = _prepare_views(validation_raw, atlas, mapper)
    # Prepared views retain only RADIO-final map features and geometry.  Drop
    # the temporary image-loading records before optimization so RGB never
    # becomes part of the learned artifact or its persistent working set.
    del train_raw, validation_raw
    if not train_views or not validation_views:
        raise ValueError("structured frame train/validation views are empty")
    train_trajectories = sorted(
        {value.trajectory_id for value in train_views}
    )
    validation_trajectories = sorted(
        {value.trajectory_id for value in validation_views}
    )
    mapping_trajectories = {
        str(value)
        for value in (atlas.metadata or {}).get(
            "mapping_trajectory_ids", []
        )
    }
    if (
        set(train_trajectories) & set(validation_trajectories)
        or set(train_trajectories) & mapping_trajectories
        or set(validation_trajectories) & mapping_trajectories
    ):
        raise ValueError("refiner train/validation/map trajectories overlap")
    candidate_source = str(args.candidate_source)
    train_replay: list[DeploymentReplayMode] = []
    validation_replay: list[DeploymentReplayMode] = []
    train_replay_summary: dict[str, object] | None = None
    validation_replay_summary: dict[str, object] | None = None
    if candidate_source == "deployment_replay":
        train_replay, train_replay_summary = _build_deployment_replay(
            train_views,
            atlas,
            charts_per_view=int(args.replay_charts_per_view),
            wrong_charts_per_view=int(
                args.replay_wrong_charts_per_view
            ),
            modes_per_chart=int(args.replay_modes_per_chart),
            maximum_fitted_error_cells=float(
                args.replay_max_affine_residual_cells
            ),
            maximum_linear_residual=float(args.maximum_linear_log),
            linear_parameterization=str(args.linear_parameterization),
            update_parameterization=str(args.update_parameterization),
            seed=int(args.seed) + 200000,
            device=str(args.device),
        )
        validation_replay, validation_replay_summary = (
            _build_deployment_replay(
                validation_views,
                atlas,
                charts_per_view=int(args.replay_charts_per_view),
                wrong_charts_per_view=int(
                    args.replay_wrong_charts_per_view
                ),
                modes_per_chart=int(args.replay_modes_per_chart),
                maximum_fitted_error_cells=float(
                    args.replay_max_affine_residual_cells
                ),
                maximum_linear_residual=float(
                    args.maximum_linear_log
                ),
                linear_parameterization=str(
                    args.linear_parameterization
                ),
                update_parameterization=str(
                    args.update_parameterization
                ),
                seed=int(args.seed) + 300000,
                device=str(args.device),
            )
        )
    if str(args.initial_checkpoint):
        initial_path = Path(args.initial_checkpoint)
        model, initial_metadata = load_structured_frame_refiner(
            initial_path, device=str(args.device)
        )
        if str(
            initial_metadata.get("compatible_radio_atlas_sha256", "")
        ) != _sha256(atlas_path):
            raise ValueError("initial refiner atlas differs")
        declared_kind = str(
            initial_metadata.get(
                "query_feature_transform", "surface_maplet_mapper"
            )
        )
        declared_sha256 = str(
            initial_metadata.get(
                "query_feature_transform_sha256",
                initial_metadata.get(
                    "surface_mapper_checkpoint_sha256", ""
                ),
            )
        )
        if (
            declared_kind != feature_transform_kind
            or declared_sha256 != _sha256(feature_transform_path)
        ):
            raise ValueError("initial refiner feature transform differs")
        if str(model.config.linear_parameterization) != str(
            args.linear_parameterization
        ):
            raise ValueError(
                "initial refiner linear parameterization differs"
            )
        if str(model.config.update_parameterization) != str(
            args.update_parameterization
        ):
            raise ValueError(
                "initial refiner update parameterization differs"
            )
    else:
        model = StructuredFrameRefiner(
            StructuredFrameRefinerConfig(
                feature_dim=int(atlas.feature_dim),
                correlation_radius=int(args.correlation_radius),
                hidden_dim=int(args.hidden_dim),
                transformer_layers=int(args.transformer_layers),
                attention_heads=int(args.attention_heads),
                maximum_linear_residual=float(
                    args.maximum_linear_log
                ),
                linear_parameterization=str(
                    args.linear_parameterization
                ),
                update_parameterization=str(
                    args.update_parameterization
                ),
            )
        ).to(str(args.device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=1e-4,
    )
    validation_seed = int(args.seed) + 100000

    def run_validation() -> dict[str, float]:
        if candidate_source == "deployment_replay":
            return _validate_replay(
                model,
                validation_views,
                atlas,
                validation_replay,
                batch_charts=int(args.batch_charts),
                cells_per_chart=int(args.cells_per_chart),
                positive_fraction=float(args.positive_fraction),
                episode_count=(
                    len(validation_views)
                    * int(args.validation_episodes_per_view)
                ),
                seed=validation_seed,
                device=str(args.device),
            )
        return _validate(
            model,
            validation_views,
            atlas,
            batch_charts=int(args.batch_charts),
            cells_per_chart=int(args.cells_per_chart),
            positive_fraction=float(args.positive_fraction),
            episodes_per_view=int(args.validation_episodes_per_view),
            seed=validation_seed,
            device=str(args.device),
        )

    baseline = run_validation()
    best_step = 0
    best_metrics = dict(baseline)
    best_score = float(baseline["selection_score"])
    save_structured_frame_refiner(
        output,
        model,
        _metadata(
            atlas_path,
            feature_transform_path,
            feature_transform_kind,
            atlas,
            train_trajectories,
            validation_trajectories,
            best_step,
            best_metrics,
            candidate_source=candidate_source,
            training_positive_fraction=float(args.positive_fraction),
        ),
    )
    history = [{"step": 0, "validation": baseline}]
    rng = np.random.default_rng(int(args.seed))
    for step in range(1, int(args.steps) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if candidate_source == "deployment_replay":
            loss, train_metrics = _replay_episode(
                model,
                train_views,
                atlas,
                train_replay,
                batch_charts=int(args.batch_charts),
                cells_per_chart=int(args.cells_per_chart),
                positive_fraction=float(args.positive_fraction),
                rng=rng,
                device=str(args.device),
            )
        else:
            view = train_views[
                int(rng.integers(0, len(train_views)))
            ]
            loss, train_metrics = _episode(
                model,
                view,
                atlas,
                batch_charts=int(args.batch_charts),
                cells_per_chart=int(args.cells_per_chart),
                positive_fraction=float(args.positive_fraction),
                rng=rng,
                device=str(args.device),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        if step == 1 or step % int(args.validation_every) == 0:
            validation = run_validation()
            history.append(
                {
                    "step": int(step),
                    "train": train_metrics,
                    "validation": validation,
                }
            )
            if float(validation["selection_score"]) > best_score:
                best_step = int(step)
                best_metrics = dict(validation)
                best_score = float(validation["selection_score"])
                save_structured_frame_refiner(
                    output,
                    model,
                    _metadata(
                        atlas_path,
                        feature_transform_path,
                        feature_transform_kind,
                        atlas,
                        train_trajectories,
                        validation_trajectories,
                        best_step,
                        best_metrics,
                        candidate_source=candidate_source,
                        training_positive_fraction=float(
                            args.positive_fraction
                        ),
                    ),
                )
            print(
                json.dumps(
                    {
                        "step": int(step),
                        "train": train_metrics,
                        "validation": validation,
                        "best_step": int(best_step),
                    }
                ),
                flush=True,
            )
    report = {
        "stage": "v6_structured_frame_refiner_training",
        "artifact": str(output),
        "artifact_sha256": _sha256(output),
        "baseline_validation_metrics": baseline,
        "best_step": int(best_step),
        "best_validation_metrics": best_metrics,
        "best_validation_score": float(best_score),
        "training_trajectory_ids": train_trajectories,
        "validation_trajectory_ids": validation_trajectories,
        "atlas_mapping_trajectory_ids": sorted(mapping_trajectories),
        "trajectory_disjoint": True,
        "candidate_source": candidate_source,
        "query_feature_transform": feature_transform_kind,
        "query_feature_transform_sha256": _sha256(
            feature_transform_path
        ),
        "training_replay": train_replay_summary,
        "validation_replay": validation_replay_summary,
        "history": history,
        "map_contract": {
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": False,
            "stores_mapping_image_paths": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_alike_descriptors": False,
            "uses_loftr": False,
            "uses_radio_intermediate": False,
            "uses_pairwise_image_matching": False,
            "uses_point_correspondence_pnp": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
