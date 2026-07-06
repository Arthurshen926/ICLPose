from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image, run_pnp_solver_ablation
from feature_extract.vfm.rendered_keypoint_matching import backproject_depth_to_world


GEOMETRY_SOURCE_CHOICES = ("prefer_world_xyz", "force_backproject", "assert_consistent")


DENSE_DEPTH_FUSION_FIELDNAMES = [
    "query_id",
    "candidate_id",
    "render_pose_id",
    "match_index",
    "render_x",
    "render_y",
    "render_depth",
    "query_center_x",
    "query_center_y",
    "query_refined_x",
    "query_refined_y",
    "measurement_dx",
    "measurement_dy",
    "world_x",
    "world_y",
    "world_z",
    "world_xyz_source",
    "world_xyz_backproject_delta_m",
    "world_xyz_consistency_ok",
    "measurement_cov_xx",
    "measurement_cov_xy",
    "measurement_cov_yy",
    "measurement_sigma_px",
    "measurement_valid_prob",
    "radio_match_score",
    "local_cost_entropy",
    "measurement_search_radius_px",
    "query_gt_x",
    "query_gt_y",
    "gt_reproj_error_px",
    "depth_valid",
    "pnp_inlier",
]


def _safe_percentile(values: Sequence[float], percentile: float) -> float | None:
    finite = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.percentile(finite, float(percentile)))


def _safe_mean(values: Sequence[float]) -> float | None:
    finite = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _optional_bool(row: Mapping[str, object], name: str, *, default: bool = False) -> bool:
    value = row.get(name)
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return bool(default)


def _optional_float(row: Mapping[str, object], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            number = float(text)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return float(number)
    return None


def _optional_int(row: Mapping[str, object], *names: str, default: int = 0) -> int:
    value = _optional_float(row, *names)
    return int(value) if value is not None else int(default)


def _query_center_xy(row: Mapping[str, object]) -> np.ndarray | None:
    x = _optional_float(row, "query_center_x", "center_x", "query_x")
    y = _optional_float(row, "query_center_y", "center_y", "query_y")
    if x is None or y is None:
        return None
    return np.asarray([x, y], dtype=np.float64)


def _explicit_query_center_xy(row: Mapping[str, object]) -> np.ndarray | None:
    x = _optional_float(row, "query_center_x")
    y = _optional_float(row, "query_center_y")
    if x is None or y is None:
        return None
    return np.asarray([x, y], dtype=np.float64)


def _query_refined_xy(row: Mapping[str, object], center_xy: np.ndarray) -> np.ndarray | None:
    x = _optional_float(row, "query_refined_x", "query_pred_x", "query_x")
    y = _optional_float(row, "query_refined_y", "query_pred_y", "query_y")
    if x is not None and y is not None:
        return np.asarray([x, y], dtype=np.float64)
    dx = _optional_float(row, "measurement_dx", "pred_dx")
    dy = _optional_float(row, "measurement_dy", "pred_dy")
    if dx is not None and dy is not None:
        return center_xy + np.asarray([dx, dy], dtype=np.float64)
    return None


def _explicit_query_refined_xy(row: Mapping[str, object], center_xy: np.ndarray) -> np.ndarray | None:
    x = _optional_float(row, "query_refined_x")
    y = _optional_float(row, "query_refined_y")
    if x is not None and y is not None:
        return np.asarray([x, y], dtype=np.float64)
    dx = _optional_float(row, "measurement_dx")
    dy = _optional_float(row, "measurement_dy")
    if dx is not None and dy is not None:
        return center_xy + np.asarray([dx, dy], dtype=np.float64)
    return None


def _explicit_query_gt_xy(row: Mapping[str, object]) -> np.ndarray | None:
    x = _optional_float(row, "query_gt_x")
    y = _optional_float(row, "query_gt_y")
    if x is None or y is None:
        return None
    return np.asarray([x, y], dtype=np.float64)


def _render_xy_depth(row: Mapping[str, object]) -> tuple[np.ndarray | None, float | None]:
    x = _optional_float(row, "render_x")
    y = _optional_float(row, "render_y")
    depth = _optional_float(row, "render_depth", "depth")
    if x is None or y is None:
        return None, depth
    return np.asarray([x, y], dtype=np.float64), depth


def _world_xyz(row: Mapping[str, object]) -> np.ndarray | None:
    x = _optional_float(row, "world_x", "x")
    y = _optional_float(row, "world_y", "y")
    z = _optional_float(row, "world_z", "z")
    if x is None or y is None or z is None:
        return None
    xyz = np.asarray([x, y, z], dtype=np.float64)
    return xyz if np.isfinite(xyz).all() else None


def _backproject_row_world(
    row: Mapping[str, object],
    *,
    camera: ColmapCamera,
    render_pose_w2c: np.ndarray | None,
) -> tuple[np.ndarray | None, bool]:
    render_xy, depth = _render_xy_depth(row)
    if render_xy is None or depth is None or render_pose_w2c is None:
        return None, False
    if not (np.isfinite(float(depth)) and float(depth) > 1e-6):
        return None, False
    xyz, valid = backproject_depth_to_world(
        render_xy.reshape(1, 2),
        np.asarray([float(depth)], dtype=np.float64),
        camera,
        np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4),
    )
    if not bool(valid[0]) or not np.isfinite(xyz[0]).all():
        return None, False
    return xyz[0].astype(np.float64), True


def _resolve_world_xyz(
    row: Mapping[str, object],
    *,
    camera: ColmapCamera,
    render_pose_w2c: np.ndarray | None,
    geometry_source: str,
    world_xyz_consistency_threshold_m: float,
) -> tuple[np.ndarray | None, str, float | None, bool]:
    mode = str(geometry_source)
    if mode not in GEOMETRY_SOURCE_CHOICES:
        raise ValueError(f"geometry_source must be one of {GEOMETRY_SOURCE_CHOICES}, got {geometry_source!r}")
    provided = _world_xyz(row)
    backprojected, backproject_valid = _backproject_row_world(row, camera=camera, render_pose_w2c=render_pose_w2c)
    delta = None
    consistency_ok = True
    if provided is not None and backprojected is not None:
        delta = float(np.linalg.norm(provided - backprojected))
        consistency_ok = bool(delta <= float(world_xyz_consistency_threshold_m))
    if mode == "assert_consistent":
        if provided is None:
            raise ValueError("geometry_source='assert_consistent' requires world_x/world_y/world_z")
        if backprojected is None:
            raise ValueError("geometry_source='assert_consistent' requires valid render_x/render_y/render_depth and render pose")
        if not consistency_ok:
            raise ValueError(
                "world_xyz does not match render-depth backprojection: "
                f"delta={delta:.6g}m > threshold={float(world_xyz_consistency_threshold_m):.6g}m"
            )
        return provided, "world_xyz", delta, True
    if mode == "force_backproject":
        if backprojected is not None and backproject_valid:
            return backprojected, "backproject", delta, bool(consistency_ok)
        return None, "missing", delta, bool(consistency_ok)
    if provided is not None:
        return provided, "world_xyz", delta, bool(consistency_ok)
    if backprojected is not None and backproject_valid:
        return backprojected, "backproject", delta, bool(consistency_ok)
    return None, "missing", delta, bool(consistency_ok)


def _measurement_sigma_px(row: Mapping[str, object]) -> float | None:
    sigma = _optional_float(row, "measurement_sigma_px", "fine_offset_sigma_px", "patch_offset_sigma")
    if sigma is not None and sigma > 0.0:
        return float(sigma)
    xx = _optional_float(row, "measurement_cov_xx", "cov_xx")
    yy = _optional_float(row, "measurement_cov_yy", "cov_yy")
    if xx is not None and yy is not None and xx >= 0.0 and yy >= 0.0:
        return float(np.sqrt(max(0.5 * (xx + yy), 1e-12)))
    return None


def _measurement_valid_prob(row: Mapping[str, object]) -> float:
    value = _optional_float(row, "measurement_valid_prob", "p_valid", "confidence")
    if value is None:
        return 1.0
    return float(np.clip(value, 0.0, 1.0))


def dense_depth_measurement_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Summarize measurement quality against the query-center baseline.

    The summary is intentionally measurement-level. It does not claim that pose
    improves unless the downstream PnP ablation also improves.
    """

    table = list(rows)
    center_epe: list[float] = []
    measurement_epe: list[float] = []
    delta_norms: list[float] = []
    valid_probs: list[float] = []
    entropy_values: list[float] = []
    improved_count = 0
    paired_count = 0
    window_known_count = 0
    window_inside_count = 0
    depth_valid_count = 0
    for row in table:
        if _optional_bool(row, "depth_valid"):
            depth_valid_count += 1
        valid_probs.append(_measurement_valid_prob(row))
        entropy = _optional_float(row, "local_cost_entropy")
        if entropy is not None:
            entropy_values.append(float(entropy))
        center = _query_center_xy(row)
        if center is None:
            continue
        refined = _query_refined_xy(row, center)
        if refined is not None and np.isfinite(refined).all():
            delta_norms.append(float(np.linalg.norm(refined - center)))
        gt_x = _optional_float(row, "query_gt_x", "gt_query_x")
        gt_y = _optional_float(row, "query_gt_y", "gt_query_y")
        if gt_x is None or gt_y is None:
            continue
        target = np.asarray([gt_x, gt_y], dtype=np.float64)
        center_error = float(np.linalg.norm(center - target))
        center_epe.append(center_error)
        search_radius = _optional_float(row, "measurement_search_radius_px")
        if search_radius is not None and search_radius >= 0.0:
            window_known_count += 1
            if abs(float(center[0] - target[0])) <= search_radius and abs(float(center[1] - target[1])) <= search_radius:
                window_inside_count += 1
        if refined is None or not np.isfinite(refined).all():
            continue
        measurement_error = float(np.linalg.norm(refined - target))
        measurement_epe.append(measurement_error)
        paired_count += 1
        if measurement_error < center_error:
            improved_count += 1
    count = int(len(table))
    return {
        "row_count": count,
        "depth_valid_count": int(depth_valid_count),
        "depth_valid_rate": float(depth_valid_count / count) if count else 0.0,
        "measurement_valid_prob_mean": _safe_mean(valid_probs),
        "local_cost_entropy_mean": _safe_mean(entropy_values),
        "measurement_delta_norm_median_px": _safe_percentile(delta_norms, 50.0),
        "measurement_delta_norm_p90_px": _safe_percentile(delta_norms, 90.0),
        "center_epe_median_px": _safe_percentile(center_epe, 50.0),
        "center_epe_p90_px": _safe_percentile(center_epe, 90.0),
        "measurement_epe_median_px": _safe_percentile(measurement_epe, 50.0),
        "measurement_epe_p90_px": _safe_percentile(measurement_epe, 90.0),
        "measurement_improve_pair_count": int(paired_count),
        "measurement_improve_count": int(improved_count),
        "measurement_improve_ratio": float(improved_count / paired_count) if paired_count else None,
        "center_within_measurement_window_count": int(window_inside_count),
        "center_within_measurement_window_pair_count": int(window_known_count),
        "center_within_measurement_window_rate": float(window_inside_count / window_known_count) if window_known_count else None,
    }


def augment_rows_with_query_gt_projection(
    rows: Sequence[Mapping[str, object]],
    *,
    query_pose_w2c_by_id: Mapping[str, np.ndarray],
    camera: ColmapCamera,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach query_gt_x/query_gt_y by projecting each row's world point.

    This is required for older RADIO/MATCHA match tables produced before
    query-side GT target columns were exported.
    """

    output = [dict(row) for row in rows]
    grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
    already_present_count = 0
    missing_pose_count = 0
    missing_xyz_count = 0
    for index, row in enumerate(output):
        has_target = _optional_float(row, "query_gt_x", "gt_query_x") is not None and _optional_float(row, "query_gt_y", "gt_query_y") is not None
        if has_target and not bool(overwrite):
            already_present_count += 1
            continue
        query_id = str(row.get("query_id", ""))
        pose = query_pose_w2c_by_id.get(query_id)
        if pose is None:
            missing_pose_count += 1
            continue
        xyz = _world_xyz(row)
        if xyz is None:
            missing_xyz_count += 1
            continue
        grouped.setdefault(query_id, []).append((index, xyz))
    projected_count = 0
    for query_id, values in grouped.items():
        pose = np.asarray(query_pose_w2c_by_id[query_id], dtype=np.float64).reshape(4, 4)
        xyz = np.stack([item[1] for item in values], axis=0)
        xy = project_world_to_image(xyz, pose, camera)
        for (row_index, _xyz), point in zip(values, xy):
            output[row_index]["query_gt_x"] = float(point[0])
            output[row_index]["query_gt_y"] = float(point[1])
            projected_count += 1
    return output, {
        "input_count": int(len(output)),
        "projected_query_gt_count": int(projected_count),
        "already_present_query_gt_count": int(already_present_count),
        "missing_query_pose_count": int(missing_pose_count),
        "missing_world_xyz_count": int(missing_xyz_count),
    }


def dense_depth_rows_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    camera: ColmapCamera,
    render_pose_w2c: np.ndarray | None = None,
    geometry_source: str = "prefer_world_xyz",
    world_xyz_consistency_threshold_m: float = 1e-4,
) -> list[dict[str, Any]]:
    """Convert RADIO/MATCHA match-table rows into a dense-depth fusion table."""

    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        center = _query_center_xy(row)
        render_xy, depth = _render_xy_depth(row)
        world_xyz, world_source, world_delta, world_consistency_ok = _resolve_world_xyz(
            row,
            camera=camera,
            render_pose_w2c=render_pose_w2c,
            geometry_source=str(geometry_source),
            world_xyz_consistency_threshold_m=float(world_xyz_consistency_threshold_m),
        )
        refined = None if center is None else _query_refined_xy(row, center)
        depth_valid = bool(depth is not None and np.isfinite(depth) and float(depth) > 1e-6)
        if render_xy is not None and depth is not None and render_pose_w2c is not None:
            _xyz, xyz_valid = backproject_depth_to_world(render_xy.reshape(1, 2), np.asarray([depth], dtype=np.float64), camera, render_pose_w2c)
            depth_valid = depth_valid and bool(xyz_valid[0])
        sigma = _measurement_sigma_px(row)
        if center is None:
            center = np.asarray([np.nan, np.nan], dtype=np.float64)
        if refined is None:
            refined = np.asarray([np.nan, np.nan], dtype=np.float64)
        if render_xy is None:
            render_xy = np.asarray([np.nan, np.nan], dtype=np.float64)
        dxdy = refined - center
        gt_x = _optional_float(row, "query_gt_x", "gt_query_x")
        gt_y = _optional_float(row, "query_gt_y", "gt_query_y")
        gt_reproj_error = _optional_float(row, "gt_reproj_error_px")
        output.append(
            {
                "query_id": str(row.get("query_id", "")),
                "candidate_id": str(row.get("candidate_id", "")),
                "render_pose_id": str(row.get("render_pose_id", "")),
                "match_index": _optional_int(row, "match_index", "query_index", default=index),
                "render_x": float(render_xy[0]),
                "render_y": float(render_xy[1]),
                "render_depth": "" if depth is None else float(depth),
                "query_center_x": float(center[0]),
                "query_center_y": float(center[1]),
                "query_refined_x": float(refined[0]),
                "query_refined_y": float(refined[1]),
                "measurement_dx": float(dxdy[0]),
                "measurement_dy": float(dxdy[1]),
                "world_x": "" if world_xyz is None else float(world_xyz[0]),
                "world_y": "" if world_xyz is None else float(world_xyz[1]),
                "world_z": "" if world_xyz is None else float(world_xyz[2]),
                "world_xyz_source": str(world_source),
                "world_xyz_backproject_delta_m": "" if world_delta is None else float(world_delta),
                "world_xyz_consistency_ok": bool(world_consistency_ok),
                "measurement_cov_xx": "" if _optional_float(row, "measurement_cov_xx", "cov_xx") is None else float(_optional_float(row, "measurement_cov_xx", "cov_xx")),
                "measurement_cov_xy": "" if _optional_float(row, "measurement_cov_xy", "cov_xy") is None else float(_optional_float(row, "measurement_cov_xy", "cov_xy")),
                "measurement_cov_yy": "" if _optional_float(row, "measurement_cov_yy", "cov_yy") is None else float(_optional_float(row, "measurement_cov_yy", "cov_yy")),
                "measurement_sigma_px": "" if sigma is None else float(sigma),
                "measurement_valid_prob": _measurement_valid_prob(row),
                "radio_match_score": "" if _optional_float(row, "radio_match_score", "similarity") is None else float(_optional_float(row, "radio_match_score", "similarity")),
                "local_cost_entropy": "" if _optional_float(row, "local_cost_entropy") is None else float(_optional_float(row, "local_cost_entropy")),
                "measurement_search_radius_px": "" if _optional_float(row, "measurement_search_radius_px") is None else float(_optional_float(row, "measurement_search_radius_px")),
                "query_gt_x": "" if gt_x is None else float(gt_x),
                "query_gt_y": "" if gt_y is None else float(gt_y),
                "gt_reproj_error_px": "" if gt_reproj_error is None else float(gt_reproj_error),
                "depth_valid": bool(depth_valid),
                "pnp_inlier": _optional_bool(row, "pnp_inlier"),
            }
        )
    return output


def dense_depth_matches_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    camera: ColmapCamera,
    render_pose_w2c: np.ndarray | None = None,
    source: str = "radio_matcha_dense_depth_fusion",
    geometry_source: str = "prefer_world_xyz",
    world_xyz_consistency_threshold_m: float = 1e-4,
) -> tuple[list[QueryTo3DMatch], dict[str, Any]]:
    """Build PnP matches from render pixels/depth and refined query pixels."""

    table = dense_depth_rows_from_rows(
        rows,
        camera=camera,
        render_pose_w2c=render_pose_w2c,
        geometry_source=str(geometry_source),
        world_xyz_consistency_threshold_m=float(world_xyz_consistency_threshold_m),
    )
    matches: list[QueryTo3DMatch] = []
    invalid_depth_count = 0
    missing_geometry_count = 0
    depth_valid_count = int(sum(1 for row in table if _optional_bool(row, "depth_valid")))
    source_counts: dict[str, int] = {}
    world_deltas: list[float] = []
    mismatch_count = 0
    for row in table:
        source_name = str(row.get("world_xyz_source", ""))
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
        delta = _optional_float(row, "world_xyz_backproject_delta_m")
        if delta is not None:
            world_deltas.append(float(delta))
        if not _optional_bool(row, "world_xyz_consistency_ok", default=True):
            mismatch_count += 1
    for row in table:
        world_xyz = _world_xyz(row)
        if world_xyz is None and render_pose_w2c is None:
            missing_geometry_count += 1
            continue
        if world_xyz is None and not bool(row["depth_valid"]):
            invalid_depth_count += 1
            continue
        render_xy = np.asarray([row["render_x"], row["render_y"]], dtype=np.float64)
        depth = _optional_float(row, "render_depth")
        query_xy = np.asarray([row["query_refined_x"], row["query_refined_y"]], dtype=np.float64)
        if not np.isfinite(query_xy).all():
            continue
        if world_xyz is None:
            assert render_pose_w2c is not None
            if depth is None:
                invalid_depth_count += 1
                continue
            xyz, valid = backproject_depth_to_world(render_xy.reshape(1, 2), np.asarray([depth], dtype=np.float64), camera, render_pose_w2c)
            if not bool(valid[0]):
                invalid_depth_count += 1
                continue
            world_xyz = xyz[0].astype(np.float64)
        valid_prob = float(row["measurement_valid_prob"])
        sigma = row["measurement_sigma_px"]
        matches.append(
            QueryTo3DMatch(
                token_index=int(row["match_index"]),
                xy=query_xy,
                track_id=int(row["match_index"]),
                xyz=np.asarray(world_xyz, dtype=np.float64),
                similarity=float(row["radio_match_score"]) if row["radio_match_score"] != "" else valid_prob,
                ratio=1.0,
                landmark_variance=0.0,
                source=str(source),
                render_depth=depth,
                render_xy=render_xy,
                pnp_soft_score=valid_prob,
                measurement_sigma_px=None if sigma == "" else float(sigma),
            )
        )
    count = int(len(table))
    valid_count = int(len(matches))
    return matches, {
        "input_count": count,
        "valid_match_count": valid_count,
        "invalid_depth_count": int(invalid_depth_count),
        "missing_geometry_count": int(missing_geometry_count),
        "depth_valid_count": int(depth_valid_count),
        "depth_valid_fraction": float(depth_valid_count / count) if count else 0.0,
        "valid_match_fraction": float(valid_count / count) if count else 0.0,
        "geometry_source": str(geometry_source),
        "world_xyz_source_counts": source_counts,
        "world_xyz_backproject_delta_m_median": _safe_percentile(world_deltas, 50.0),
        "world_xyz_backproject_delta_m_p99": _safe_percentile(world_deltas, 99.0),
        "world_xyz_mismatch_count": int(mismatch_count),
        "world_xyz_consistency_threshold_m": float(world_xyz_consistency_threshold_m),
    }


def _legacy_gt_xy(row: Mapping[str, object]) -> np.ndarray | None:
    x = _optional_float(row, "query_gt_x", "gt_query_x")
    y = _optional_float(row, "query_gt_y", "gt_query_y")
    if x is None or y is None:
        return None
    return np.asarray([x, y], dtype=np.float64)


def _rows_for_variant(
    rows: Sequence[Mapping[str, object]],
    variant: str,
    *,
    strict_measurement_schema: bool = True,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    out: list[dict[str, object]] = []
    name = str(variant)
    oracle_replaced_count = 0
    oracle_out_of_window_count = 0
    for row in rows:
        item = dict(row)
        if name == "center":
            center = _explicit_query_center_xy(row) if strict_measurement_schema else _query_center_xy(row)
            if center is None and strict_measurement_schema:
                raise ValueError("center variant requires query_center_x/query_center_y under strict schema")
            if center is not None:
                item["query_refined_x"] = float(center[0])
                item["query_refined_y"] = float(center[1])
        elif name in {"measurement", "measurement_mean"}:
            center = _explicit_query_center_xy(row) if strict_measurement_schema else _query_center_xy(row)
            if center is None and strict_measurement_schema:
                raise ValueError("measurement variant requires query_center_x/query_center_y under strict schema")
            refined = None if center is None else (
                _explicit_query_refined_xy(row, center) if strict_measurement_schema else _query_refined_xy(row, center)
            )
            if refined is None and strict_measurement_schema:
                raise ValueError("measurement variant requires query_refined_x/query_refined_y or measurement_dx/measurement_dy")
            if refined is not None:
                item["query_refined_x"] = float(refined[0])
                item["query_refined_y"] = float(refined[1])
        elif name == "measurement_mode":
            center = _explicit_query_center_xy(row) if strict_measurement_schema else _query_center_xy(row)
            dx = _optional_float(row, "measurement_mode_dx", "measurement_peak_dx")
            dy = _optional_float(row, "measurement_mode_dy", "measurement_peak_dy")
            if center is None or dx is None or dy is None:
                raise ValueError("measurement_mode variant requires query_center_x/y and measurement_mode_dx/dy or measurement_peak_dx/dy")
            item["query_refined_x"] = float(center[0] + dx)
            item["query_refined_y"] = float(center[1] + dy)
        elif name in {"oracle", "oracle_all"}:
            gt = _explicit_query_gt_xy(row) if strict_measurement_schema else _legacy_gt_xy(row)
            if gt is None and strict_measurement_schema:
                raise ValueError("oracle variant requires query_gt_x/query_gt_y under strict schema")
            if gt is not None:
                item["query_refined_x"] = float(gt[0])
                item["query_refined_y"] = float(gt[1])
        elif name == "oracle_if_gt_in_window":
            center = _explicit_query_center_xy(row) if strict_measurement_schema else _query_center_xy(row)
            gt = _explicit_query_gt_xy(row) if strict_measurement_schema else _legacy_gt_xy(row)
            radius = _optional_float(row, "measurement_search_radius_px")
            if center is None or gt is None or radius is None:
                raise ValueError("oracle_if_gt_in_window requires query_center_x/y, query_gt_x/y, and measurement_search_radius_px")
            if abs(float(gt[0] - center[0])) <= float(radius) and abs(float(gt[1] - center[1])) <= float(radius):
                item["query_refined_x"] = float(gt[0])
                item["query_refined_y"] = float(gt[1])
                oracle_replaced_count += 1
            else:
                item["query_refined_x"] = float(center[0])
                item["query_refined_y"] = float(center[1])
                oracle_out_of_window_count += 1
        else:
            raise ValueError(f"unsupported dense-depth fusion variant: {variant}")
        out.append(item)
    return out, {
        "input_count": int(len(rows)),
        "strict_measurement_schema": bool(strict_measurement_schema),
        "oracle_replaced_count": int(oracle_replaced_count),
        "oracle_out_of_window_count": int(oracle_out_of_window_count),
    }


def dense_depth_pose_ablation_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    camera: ColmapCamera,
    render_pose_w2c: np.ndarray | None = None,
    gt_pose_w2c: np.ndarray | None = None,
    variants: Sequence[str] = ("center", "measurement", "oracle"),
    solvers: Sequence[str] = ("ransac", "weighted", "covariance", "oracle_uncertainty"),
    reprojection_error_px: float = 8.0,
    geometry_source: str = "prefer_world_xyz",
    world_xyz_consistency_threshold_m: float = 1e-4,
    strict_measurement_schema: bool = True,
) -> dict[str, Any]:
    """Evaluate dense-depth PnP variants from one RADIO/MATCHA match table."""

    variant_rows: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    variant_preparation: dict[str, dict[str, Any]] = {}
    for variant in variants:
        prepared, prep_summary = _rows_for_variant(
            rows,
            str(variant),
            strict_measurement_schema=bool(strict_measurement_schema),
        )
        matches, summary = dense_depth_matches_from_rows(
            prepared,
            camera=camera,
            render_pose_w2c=render_pose_w2c,
            source=f"radio_matcha_dense_depth_fusion:{variant}",
            geometry_source=str(geometry_source),
            world_xyz_consistency_threshold_m=float(world_xyz_consistency_threshold_m),
        )
        variant_rows[str(variant)] = run_pnp_solver_ablation(
            matches,
            camera,
            gt_pose_w2c=gt_pose_w2c,
            solvers=tuple(str(item) for item in solvers),
            reprojection_error_px=float(reprojection_error_px),
        )
        summaries[str(variant)] = summary
        variant_preparation[str(variant)] = prep_summary
    return {
        "stage": "radio_matcha_dense_depth_measurement_fusion",
        "input_count": int(len(rows)),
        "variants": variant_rows,
        "variant_preparation": variant_preparation,
        "match_summaries": summaries,
        "measurement_summary": dense_depth_measurement_summary(
            dense_depth_rows_from_rows(
                rows,
                camera=camera,
                render_pose_w2c=render_pose_w2c,
                geometry_source=str(geometry_source),
                world_xyz_consistency_threshold_m=float(world_xyz_consistency_threshold_m),
            )
        ),
    }
