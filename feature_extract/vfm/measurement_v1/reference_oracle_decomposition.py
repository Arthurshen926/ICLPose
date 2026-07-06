"""Reference-topK coarse-only oracle decomposition.

This module operates on exported fixed-anchor match tables. It separates
retrieval/candidate quality, predicted coarse correspondences, validity
selection, oracle assignment, and continuous-measurement headroom without using
learned confidence, learned scorer, or fine heads.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


@dataclass(frozen=True)
class OracleDecompositionConfig:
    pnp_reprojection_error_px: float = 12.0
    pnp_iterations: int = 2000
    pnp_min_inliers: int = 6
    validity_thresholds_px: tuple[float, ...] = (2.0, 5.0, 10.0, 16.0)
    topk_values: tuple[int, ...] = (1, 3, 5, 10)


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _int_or_default(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _bool_or_false(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def match_from_table_row(row: Mapping[str, object], *, xy: np.ndarray | None = None, source: str = "match_table") -> QueryTo3DMatch:
    query_xy = (
        np.asarray(xy, dtype=np.float64).reshape(2)
        if xy is not None
        else np.asarray([float(row["query_x"]), float(row["query_y"])], dtype=np.float64)
    )
    return QueryTo3DMatch(
        token_index=_int_or_default(row.get("query_index")),
        xy=query_xy,
        track_id=_int_or_default(row.get("render_index")),
        xyz=np.asarray([float(row["world_x"]), float(row["world_y"]), float(row["world_z"])], dtype=np.float64),
        similarity=float(_float_or_none(row.get("similarity")) or 0.0),
        ratio=0.0,
        landmark_variance=0.0,
        source=source,
        pnp_soft_score=_float_or_none(row.get("confidence")),
        render_xy=np.asarray(
            [
                float(_float_or_none(row.get("render_x")) or 0.0),
                float(_float_or_none(row.get("render_y")) or 0.0),
            ],
            dtype=np.float64,
        ),
        base_render_index=None if row.get("base_render_index") in (None, "") else _int_or_default(row.get("base_render_index")),
        candidate_render_index=None
        if row.get("candidate_render_index") in (None, "")
        else _int_or_default(row.get("candidate_render_index")),
        candidate_id=None if row.get("candidate_id") in (None, "") else _int_or_default(row.get("candidate_id")),
        coarse_rank=None if row.get("coarse_rank") in (None, "") else _int_or_default(row.get("coarse_rank")),
        coarse_score=_float_or_none(row.get("coarse_score")),
        coarse_score_gap=_float_or_none(row.get("coarse_score_gap")),
        mutual_rank=None if row.get("mutual_rank") in (None, "") else _int_or_default(row.get("mutual_rank")),
        cell_delta_x=None if row.get("cell_delta_x") in (None, "") else _int_or_default(row.get("cell_delta_x")),
        cell_delta_y=None if row.get("cell_delta_y") in (None, "") else _int_or_default(row.get("cell_delta_y")),
    )


def project_visible_points(points_xyz: np.ndarray, pose_w2c: np.ndarray, camera: ColmapCamera) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_points = (pose[:3, :3] @ points.T).T + pose[:3, 3][None, :]
    projected = project_world_to_image(points, pose, camera)
    valid = (
        np.isfinite(projected).all(axis=1)
        & (camera_points[:, 2] > 1e-6)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < float(camera.width))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < float(camera.height))
    )
    return projected.astype(np.float64), valid


def coarse_cell_centers_for_pixels(
    xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    valid = (
        np.isfinite(coords).all(axis=1)
        & (coords[:, 0] >= 0.0)
        & (coords[:, 0] < float(int(image_width)))
        & (coords[:, 1] >= 0.0)
        & (coords[:, 1] < float(int(image_height)))
    )
    gx = np.floor(coords[:, 0] / max(float(image_width), 1.0) * float(grid_width)).astype(np.int64)
    gy = np.floor(coords[:, 1] / max(float(image_height), 1.0) * float(grid_height)).astype(np.int64)
    gx = np.clip(gx, 0, max(int(grid_width) - 1, 0))
    gy = np.clip(gy, 0, max(int(grid_height) - 1, 0))
    centers = np.stack(
        [
            (gx.astype(np.float64) + 0.5) * float(image_width) / float(grid_width),
            (gy.astype(np.float64) + 0.5) * float(image_height) / float(grid_height),
        ],
        axis=1,
    )
    return centers.astype(np.float64), valid


def _pose_row(
    *,
    query_id: str,
    candidate_rank: int,
    variant: str,
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    gt_pose_w2c: np.ndarray,
    config: OracleDecompositionConfig,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    pnp = estimate_pose_pnp_ransac(
        matches,
        camera,
        reprojection_error_px=float(config.pnp_reprojection_error_px),
        iterations=int(config.pnp_iterations),
        min_inliers=int(config.pnp_min_inliers),
    )
    err = pnp_pose_error(pnp.pose_w2c if pnp.success else None, gt_pose_w2c)
    row: dict[str, object] = {
        "query_id": str(query_id),
        "candidate_rank": int(candidate_rank),
        "variant": str(variant),
        "match_count": int(len(matches)),
        "pnp_success": bool(pnp.success),
        "pnp_inlier_count": int(pnp.inlier_count),
        "pnp_inlier_ratio": float(pnp.inlier_ratio),
        "translation_error_m": None if not np.isfinite(err.translation_m) else float(err.translation_m),
        "rotation_error_deg": None if not np.isfinite(err.rotation_deg) else float(err.rotation_deg),
    }
    if extra:
        row.update(dict(extra))
    return row


def decompose_candidate_rows(
    *,
    query_id: str,
    candidate_rank: int,
    rows: Sequence[Mapping[str, object]],
    camera: ColmapCamera,
    gt_pose_w2c: np.ndarray,
    config: OracleDecompositionConfig = OracleDecompositionConfig(),
) -> list[dict[str, object]]:
    values = list(rows)
    predicted_matches = [match_from_table_row(row, source="predicted_coarse") for row in values]
    out: list[dict[str, object]] = [
        _pose_row(
            query_id=query_id,
            candidate_rank=candidate_rank,
            variant="predicted",
            matches=predicted_matches,
            camera=camera,
            gt_pose_w2c=gt_pose_w2c,
            config=config,
        )
    ]
    gt_errors = np.asarray(
        [
            float(_float_or_none(row.get("gt_reproj_error_px")) or np.inf)
            for row in values
        ],
        dtype=np.float64,
    )
    projected: np.ndarray | None = None
    visible: np.ndarray | None = None
    if values:
        xyz = np.stack(
            [
                np.asarray([float(row["world_x"]), float(row["world_y"]), float(row["world_z"])], dtype=np.float64)
                for row in values
            ],
            axis=0,
        )
        projected, visible = project_visible_points(xyz, gt_pose_w2c, camera)
    for threshold in config.validity_thresholds_px:
        keep = np.isfinite(gt_errors) & (gt_errors <= float(threshold))
        valid_matches = [match for match, is_kept in zip(predicted_matches, keep) if bool(is_kept)]
        out.append(
            _pose_row(
                query_id=query_id,
                candidate_rank=candidate_rank,
                variant=f"oracle_validity_{float(threshold):g}px",
                matches=valid_matches,
                camera=camera,
                gt_pose_w2c=gt_pose_w2c,
                config=config,
                extra={"oracle_validity_threshold_px": float(threshold)},
            )
        )
        if visible is not None:
            visible_keep = keep & visible
            visible_matches = [
                match_from_table_row(row, source="oracle_visible_reprojection")
                for row, is_kept in zip(values, visible_keep)
                if bool(is_kept)
            ]
            out.append(
                _pose_row(
                    query_id=query_id,
                    candidate_rank=candidate_rank,
                    variant=f"oracle_visible_reprojection_{float(threshold):g}px",
                    matches=visible_matches,
                    camera=camera,
                    gt_pose_w2c=gt_pose_w2c,
                    config=config,
                    extra={
                        "oracle_validity_threshold_px": float(threshold),
                        "visible_anchor_count": int(np.count_nonzero(visible)),
                        "visibility_model": "positive_depth_in_image",
                        "z_buffer_strict": False,
                    },
                )
            )
    if not values:
        return out
    assert projected is not None
    assert visible is not None
    grid_width = _int_or_default(values[0].get("query_grid_width"), 0)
    grid_height = _int_or_default(values[0].get("query_grid_height"), 0)
    if grid_width > 0 and grid_height > 0:
        centers, center_valid = coarse_cell_centers_for_pixels(
            projected,
            image_width=int(camera.width),
            image_height=int(camera.height),
            grid_width=grid_width,
            grid_height=grid_height,
        )
        assignment_matches = [
            match_from_table_row(row, xy=centers[idx], source="oracle_assignment")
            for idx, row in enumerate(values)
            if bool(visible[idx] and center_valid[idx])
        ]
        out.append(
            _pose_row(
                query_id=query_id,
                candidate_rank=candidate_rank,
                variant="oracle_assignment",
                matches=assignment_matches,
                camera=camera,
                gt_pose_w2c=gt_pose_w2c,
                config=config,
                extra={"visible_anchor_count": int(np.count_nonzero(visible & center_valid))},
            )
        )
    continuous_matches = [
        match_from_table_row(row, xy=projected[idx], source="oracle_continuous")
        for idx, row in enumerate(values)
        if bool(visible[idx])
    ]
    out.append(
        _pose_row(
            query_id=query_id,
            candidate_rank=candidate_rank,
            variant="oracle_continuous",
            matches=continuous_matches,
            camera=camera,
            gt_pose_w2c=gt_pose_w2c,
            config=config,
            extra={"visible_anchor_count": int(np.count_nonzero(visible))},
        )
    )
    return out


def decompose_match_table(
    rows: Iterable[Mapping[str, object]],
    *,
    gt_pose_by_query: Mapping[str, np.ndarray],
    camera: ColmapCamera,
    config: OracleDecompositionConfig = OracleDecompositionConfig(),
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if not query_id:
            continue
        candidate_rank = _int_or_default(row.get("candidate_rank"), 0)
        grouped[(query_id, candidate_rank)].append(row)
    out: list[dict[str, object]] = []
    for (query_id, candidate_rank), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        gt_pose = gt_pose_by_query.get(query_id)
        if gt_pose is None:
            continue
        out.extend(
            decompose_candidate_rows(
                query_id=query_id,
                candidate_rank=candidate_rank,
                rows=values,
                camera=camera,
                gt_pose_w2c=gt_pose,
                config=config,
            )
        )
    return out


def summarize_decomposition_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    topk_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, object]:
    values = list(rows)
    variants = sorted({str(row.get("variant", "")) for row in values if row.get("variant")})
    queries = sorted({str(row.get("query_id", "")) for row in values if row.get("query_id")})
    summary: dict[str, object] = {
        "query_count": int(len(queries)),
        "candidate_pose_row_count": int(len(values)),
        "variants": variants,
    }
    by_query_variant: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in values:
        by_query_variant[(str(row.get("query_id", "")), str(row.get("variant", "")))].append(row)
    for variant in variants:
        variant_rows = [row for row in values if str(row.get("variant", "")) == variant]
        successful = [
            row
            for row in variant_rows
            if _bool_or_false(row.get("pnp_success")) and _float_or_none(row.get("translation_error_m")) is not None
        ]
        errors = np.asarray([float(row["translation_error_m"]) for row in successful], dtype=np.float64)
        summary[f"{variant}_solve_rate"] = float(len(successful) / len(variant_rows)) if variant_rows else None
        summary[f"{variant}_median_translation_error_m"] = float(np.median(errors)) if errors.size else None
        summary[f"{variant}_p90_translation_error_m"] = float(np.percentile(errors, 90.0)) if errors.size else None
        summary[f"{variant}_per_candidate_median_translation_error_m"] = summary[f"{variant}_median_translation_error_m"]
        summary[f"{variant}_per_candidate_p90_translation_error_m"] = summary[f"{variant}_p90_translation_error_m"]
        rank0_errors = []
        rank0_success_10cm_5deg = []
        for query_id in queries:
            rank0_rows = [
                row
                for row in by_query_variant.get((query_id, variant), [])
                if int(_int_or_default(row.get("candidate_rank"), 0)) == 0
                and _bool_or_false(row.get("pnp_success"))
                and _float_or_none(row.get("translation_error_m")) is not None
            ]
            if not rank0_rows:
                continue
            first = rank0_rows[0]
            rank0_t = float(first["translation_error_m"])
            rank0_r = float(_float_or_none(first.get("rotation_error_deg")) or np.inf)
            rank0_errors.append(rank0_t)
            rank0_success_10cm_5deg.append(rank0_t <= 0.10 and rank0_r <= 5.0)
        rank0_median = float(np.median(np.asarray(rank0_errors, dtype=np.float64))) if rank0_errors else None
        summary[f"{variant}_rank0_selected_by_existing_rule_median_translation_error_m"] = rank0_median
        summary[f"{variant}_rank0_selected_by_existing_rule_success_10cm_5deg"] = (
            float(np.mean(np.asarray(rank0_success_10cm_5deg, dtype=np.float64))) if rank0_success_10cm_5deg else None
        )
        for top_k in topk_values:
            best_errors = []
            success_10cm_5deg = []
            for query_id in queries:
                candidate_rows = [
                    row
                    for row in by_query_variant.get((query_id, variant), [])
                    if int(_int_or_default(row.get("candidate_rank"), 0)) < int(top_k)
                    and _bool_or_false(row.get("pnp_success"))
                    and _float_or_none(row.get("translation_error_m")) is not None
                ]
                if not candidate_rows:
                    continue
                best = min(candidate_rows, key=lambda row: float(row["translation_error_m"]))
                best_t = float(best["translation_error_m"])
                best_r = float(_float_or_none(best.get("rotation_error_deg")) or np.inf)
                best_errors.append(best_t)
                success_10cm_5deg.append(best_t <= 0.10 and best_r <= 5.0)
            suffix = f"best_at_{int(top_k)}"
            summary[f"{variant}_{suffix}_median_translation_error_m"] = (
                float(np.median(np.asarray(best_errors, dtype=np.float64))) if best_errors else None
            )
            summary[f"{variant}_{suffix}_success_10cm_5deg"] = (
                float(np.mean(np.asarray(success_10cm_5deg, dtype=np.float64))) if success_10cm_5deg else None
            )
            if rank0_median is not None and best_errors:
                best_median = float(np.median(np.asarray(best_errors, dtype=np.float64)))
                summary[f"{variant}_{suffix}_minus_rank0_median_translation_delta_m"] = best_median - rank0_median
            else:
                summary[f"{variant}_{suffix}_minus_rank0_median_translation_delta_m"] = None
    return summary
