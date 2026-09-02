"""Measure the plane-pose ceiling of a MoGe/reference explicit chart map.

This is deliberately an oracle-correspondence diagnostic.  Held reference
geometry and its camera pose are used only to attach each query plane to the
nearest map-plane region and to score the frozen pose estimate.  The solver
itself receives only matched plane equations and never receives the target
pose.  The same correspondences are evaluated with reference-query geometry
and pose-free MoGe3 query geometry, both with metric and one-scale solvers.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.evaluate_goal_maplet_moge_reference_held_render_control import (
    _normals,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_projective_aligned_charts_control import (
    _load_aligned_vertices,
)
from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import (
    StrictHeldRayInventory,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.planar_maplet_oracle import (
    solve_metric_translation,
    solve_rotation_from_plane_normals,
    solve_scaled_translation,
)
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    ProjectiveExactFaceSeamAuthority,
)
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import (
    QueryPlaneRegions,
    extract_query_plane_regions,
)


SCHEMA = "goal_maplet_chart_plane_pose_oracle_historical_held_v4"
MAXIMUM_MAP_ASSOCIATION_DISTANCE_M = 0.75
MINIMUM_REGION_ASSOCIATION_PIXELS = 40
MINIMUM_REGION_ASSOCIATION_COVERAGE = 0.20
MAXIMUM_ASSOCIATION_NORMAL_ANGLE_DEG = 30.0
MAXIMUM_MATCHED_PLANES = 16
MAXIMUM_POINT_CORRESPONDENCES = 4096
MINIMUM_POINT_CORRESPONDENCES = 32


def _camera_points(
    depth: np.ndarray,
    focal_xy: np.ndarray,
    principal_xy: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    direction = np.stack(
        (
            (xx - principal_xy[0]) / focal_xy[0],
            (yy - principal_xy[1]) / focal_xy[1],
            np.ones_like(xx, np.float64),
        ),
        axis=2,
    )
    return direction * depth[..., None]


def _rotation_error_deg(estimate_c2w: np.ndarray, target_c2w: np.ndarray) -> float:
    relative = estimate_c2w.T @ target_c2w[:3, :3]
    value = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def _map_surface_inventory(
    authority: ProjectiveExactFaceSeamAuthority,
    vertices_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all face-referenced stride-2 surface samples and their normals."""

    vertex_normals = _normals(vertices_world, authority.faces)
    face_referenced = np.zeros((len(vertices_world),), bool)
    face_referenced[np.unique(authority.faces)] = True
    if int(face_referenced.sum()) < 4:
        raise ValueError("map chart inventory contains fewer than four surface samples")
    return vertices_world[face_referenced], vertex_normals[face_referenced]


def _oracle_map_indices(
    tree: cKDTree,
    points_world: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.full(valid.shape, -1, np.int64)
    distance_image = np.full(valid.shape, np.inf, np.float64)
    distance, index = tree.query(points_world[valid], k=1, workers=-1)
    accepted = distance <= MAXIMUM_MAP_ASSOCIATION_DISTANCE_M
    flat = np.flatnonzero(valid)
    output.ravel()[flat[accepted]] = index[accepted]
    distance_image.ravel()[flat] = distance
    return output, distance_image


def _fit_local_map_plane(
    points: np.ndarray,
    reference_normal: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    center = np.mean(points, axis=0)
    covariance = (points - center).T @ (points - center) / len(points)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    normal = eigenvector[:, 0]
    if float(normal @ reference_normal) < 0.0:
        normal = -normal
    residual = np.abs((points - center) @ normal)
    return normal, float(normal @ center), float(np.quantile(residual, 0.90))


def _matched_planes(
    query: QueryPlaneRegions,
    nearest_index_image: np.ndarray,
    nearest_distance_image: np.ndarray,
    map_points: np.ndarray,
    map_vertex_normals: np.ndarray,
    target_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    candidates: list[tuple[int, int, float, np.ndarray, float, float, float]] = []
    cosine = float(np.cos(np.deg2rad(MAXIMUM_ASSOCIATION_NORMAL_ANGLE_DEG)))
    for query_row in range(len(query.normals_camera)):
        region = query.labels == query_row
        qn = np.asarray(query.normals_camera[query_row], np.float64)
        query_world = qn @ target_c2w[:3, :3].T
        accepted = region & (nearest_index_image >= 0)
        if accepted.any():
            candidate_index = nearest_index_image[accepted]
            angle_ok = np.abs(map_vertex_normals[candidate_index] @ query_world) >= cosine
            accepted_flat = np.flatnonzero(accepted)
            accepted.ravel()[accepted_flat[~angle_ok]] = False
        matched = int(accepted.sum())
        if matched < MINIMUM_REGION_ASSOCIATION_PIXELS:
            continue
        coverage = matched / max(int(region.sum()), 1)
        if coverage < MINIMUM_REGION_ASSOCIATION_COVERAGE:
            continue
        local_points = map_points[nearest_index_image[accepted]]
        mn, md, residual_p90 = _fit_local_map_plane(local_points, query_world)
        median_distance = float(np.median(nearest_distance_image[accepted]))
        candidates.append((query_row, matched, coverage, mn, md, residual_p90, median_distance))
    candidates = sorted(
        candidates,
        key=lambda value: (-value[1], value[0]),
    )[:MAXIMUM_MATCHED_PLANES]
    query_normal, query_offset, map_normal, map_offset, rows = [], [], [], [], []
    for query_row, matched, coverage, mn, md, residual_p90, median_distance in candidates:
        qn = np.asarray(query.normals_camera[query_row], np.float64)
        qd = float(query.offsets_camera[query_row])
        query_normal.append(qn)
        query_offset.append(qd)
        map_normal.append(mn)
        map_offset.append(md)
        rows.append({
            "query_plane": int(query_row),
            "matched_pixel_count": int(matched),
            "coverage": float(coverage),
            "nearest_map_distance_median_m": median_distance,
            "local_map_plane_residual_p90_m": residual_p90,
        })
    return (
        np.asarray(query_normal, np.float64).reshape(-1, 3),
        np.asarray(query_offset, np.float64),
        np.asarray(map_normal, np.float64).reshape(-1, 3),
        np.asarray(map_offset, np.float64),
        rows,
    )


def _robust_scaled_translation(
    map_normal: np.ndarray,
    map_offset: np.ndarray,
    query_offset: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, float, int, np.ndarray]:
    if len(query_offset) < 4:
        return np.full(3, np.nan), float("nan"), 0, np.zeros((len(query_offset),), bool)
    design = np.column_stack((map_normal, query_offset))
    best: tuple[tuple[int, float], np.ndarray, np.ndarray] | None = None
    combinations = list(itertools.combinations(range(len(query_offset)), 4))
    take = np.linspace(0, len(combinations) - 1, min(1024, len(combinations)), dtype=int)
    for index in take:
        subset = np.asarray(combinations[index], np.int64)
        if np.linalg.matrix_rank(design[subset]) < 4:
            continue
        solution = np.linalg.lstsq(design[subset], map_offset[subset], rcond=None)[0]
        if not 0.25 <= solution[3] <= 4.0:
            continue
        residual = np.abs(design @ solution - map_offset)
        key = (int(np.sum(residual <= 0.30)), -float(np.median(residual)))
        if best is None or key > best[0]:
            best = (key, solution, residual)
    if best is None:
        center, scale, rank, _ = solve_scaled_translation(
            map_normal, map_offset, query_offset, weight,
        )
        return center, scale, rank, np.ones((len(query_offset),), bool)
    inlier = best[2] <= 0.30
    if int(inlier.sum()) >= 4 and np.linalg.matrix_rank(design[inlier]) == 4:
        center, scale, rank, _ = solve_scaled_translation(
            map_normal[inlier], map_offset[inlier], query_offset[inlier], weight[inlier],
        )
        return center, scale, rank, inlier
    return best[1][:3], float(best[1][3]), int(np.linalg.matrix_rank(design)), inlier


def _umeyama(
    source: np.ndarray,
    target: np.ndarray,
    *,
    estimate_scale: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return c2w rotation, translation, and scale for target=s*R*source+t."""

    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Umeyama requires paired Nx3 points")
    if len(source) < 3:
        raise ValueError("Umeyama requires at least three points")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    parity = np.eye(3)
    parity[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ parity @ vt
    if estimate_scale:
        variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
        scale = float(np.sum(singular * np.diag(parity)) / max(variance, 1e-15))
    else:
        scale = 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    return rotation, translation, scale


def _robust_point_pose(
    source: np.ndarray,
    target: np.ndarray,
    *,
    estimate_scale: bool,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    if len(source) < 6:
        return (
            np.full((3, 3), np.nan),
            np.full((3,), np.nan),
            float("nan"),
            np.zeros((len(source),), bool),
            np.full((len(source),), np.nan),
        )
    inlier = np.ones((len(source),), bool)
    rotation = np.eye(3)
    translation = np.zeros(3)
    scale = 1.0
    rng = np.random.default_rng(260830)
    best: tuple[tuple[int, float], np.ndarray] | None = None
    for _ in range(min(384, max(32, len(source)))):
        subset = np.sort(rng.choice(len(source), size=3, replace=False))
        if (
            np.linalg.norm(np.cross(
                source[subset[1]] - source[subset[0]],
                source[subset[2]] - source[subset[0]],
            )) <= 1e-5
            or np.linalg.norm(np.cross(
                target[subset[1]] - target[subset[0]],
                target[subset[2]] - target[subset[0]],
            )) <= 1e-5
        ):
            continue
        candidate_rotation, candidate_translation, candidate_scale = _umeyama(
            source[subset], target[subset], estimate_scale=estimate_scale,
        )
        if not 0.25 <= candidate_scale <= 4.0:
            continue
        candidate = candidate_scale * (source @ candidate_rotation.T) + candidate_translation
        residual = np.linalg.norm(candidate - target, axis=1)
        candidate_inlier = residual <= 0.30
        key = (
            int(candidate_inlier.sum()),
            -float(np.median(residual[candidate_inlier])) if candidate_inlier.any() else -float("inf"),
        )
        if best is None or key > best[0]:
            best = (key, candidate_inlier)
    if best is not None and int(best[1].sum()) >= 6:
        inlier = best[1]
    for _ in range(6):
        if int(inlier.sum()) < 6:
            break
        rotation, translation, scale = _umeyama(
            source[inlier], target[inlier], estimate_scale=estimate_scale,
        )
        estimate = scale * (source @ rotation.T) + translation
        residual = np.linalg.norm(estimate - target, axis=1)
        quantile = float(np.quantile(residual[inlier], 0.70))
        threshold = min(0.50, max(0.15, quantile))
        updated = residual <= threshold
        if int(updated.sum()) < 6 or np.array_equal(updated, inlier):
            break
        inlier = updated
    if int(inlier.sum()) >= 6:
        rotation, translation, scale = _umeyama(
            source[inlier], target[inlier], estimate_scale=estimate_scale,
        )
    estimate = scale * (source @ rotation.T) + translation
    residual = np.linalg.norm(estimate - target, axis=1)
    return rotation, translation, scale, inlier, residual


def _point_pose_from_matched_planes(
    query: QueryPlaneRegions,
    query_points: np.ndarray,
    nearest_index_image: np.ndarray,
    nearest_distance_image: np.ndarray,
    map_points: np.ndarray,
    map_vertex_normals: np.ndarray,
    target_c2w: np.ndarray,
) -> dict[str, object]:
    plane_pixel = query.labels >= 0
    available = plane_pixel & (nearest_index_image >= 0) & np.isfinite(query_points).all(2)
    if available.any():
        region = query.labels[available]
        query_normal = query.normals_camera[region] @ target_c2w[:3, :3].T
        map_index = nearest_index_image[available]
        cosine = np.abs(np.sum(query_normal * map_vertex_normals[map_index], axis=1))
        flat = np.flatnonzero(available)
        available.ravel()[flat[cosine < np.cos(np.deg2rad(MAXIMUM_ASSOCIATION_NORMAL_ANGLE_DEG))]] = False
    flat = np.flatnonzero(available)
    if len(flat):
        map_index = nearest_index_image.ravel()[flat]
        distance = nearest_distance_image.ravel()[flat]
        # Retain one best query pixel per finite map sample so duplicates from
        # overlapping source charts cannot dominate the rigid fit.
        order = np.lexsort((flat, distance, map_index))
        ordered_map = map_index[order]
        first = np.r_[True, ordered_map[1:] != ordered_map[:-1]]
        flat = flat[order[first]]
    if len(flat) > MAXIMUM_POINT_CORRESPONDENCES:
        flat = flat[np.linspace(0, len(flat) - 1, MAXIMUM_POINT_CORRESPONDENCES, dtype=np.int64)]
    source = query_points.reshape(-1, 3)[flat]
    target_index = nearest_index_image.ravel()[flat]
    target = map_points[target_index] if len(flat) else np.empty((0, 3), np.float64)
    output: dict[str, object] = {
        "correspondence_count": int(len(source)),
        "usable_metric": False,
        "usable_scale_aware": False,
    }
    if len(source) < MINIMUM_POINT_CORRESPONDENCES:
        return output
    for name, with_scale in (("metric", False), ("scale_aware", True)):
        rotation, translation, scale, inlier, residual = _robust_point_pose(
            source, target, estimate_scale=with_scale,
        )
        output[name] = {
            "rotation_error_deg": _rotation_error_deg(rotation, target_c2w),
            "translation_error_m": float(np.linalg.norm(translation - target_c2w[:3, 3])),
            "estimated_scale": float(scale),
            "inlier_count": int(inlier.sum()),
            "residual_median_m": float(np.median(residual[inlier])) if inlier.any() else None,
            "residual_p90_m": float(np.quantile(residual[inlier], 0.90)) if inlier.any() else None,
        }
        output[f"usable_{name}"] = bool(
            np.isfinite(rotation).all() and np.isfinite(translation).all()
        )
    return output


def _solve(
    query: QueryPlaneRegions,
    nearest_index_image: np.ndarray,
    nearest_distance_image: np.ndarray,
    map_points: np.ndarray,
    map_vertex_normals: np.ndarray,
    target_c2w: np.ndarray,
) -> dict[str, object]:
    qn, qd, mn, md, matches = _matched_planes(
        query,
        nearest_index_image,
        nearest_distance_image,
        map_points,
        map_vertex_normals,
        target_c2w,
    )
    output: dict[str, object] = {
        "query_plane_count": int(len(query.normals_camera)),
        "matched_plane_count": int(len(matches)),
        "matches": matches,
        "usable_metric": False,
        "usable_scale_aware": False,
    }
    if len(matches) < 3:
        return output
    weight = np.asarray([row["matched_pixel_count"] for row in matches], np.float64)
    rotation, singular = solve_rotation_from_plane_normals(qn, mn, weight)
    center_metric, metric_rank, _ = solve_metric_translation(mn, md, qd, weight)
    center_scaled, scale, scale_rank, scale_inlier = _robust_scaled_translation(
        mn, md, qd, weight,
    )
    target_center = target_c2w[:3, 3]
    output.update({
        "normal_singular_values": singular.tolist(),
        "normal_rank": int(np.linalg.matrix_rank(mn)),
        "rotation_error_deg": _rotation_error_deg(rotation, target_c2w),
        "metric_rank": int(metric_rank),
        "metric_translation_error_m": float(np.linalg.norm(center_metric - target_center)),
        "scale_rank": int(scale_rank),
        "estimated_scale": float(scale),
        "scale_inlier_count": int(scale_inlier.sum()),
        "scale_translation_error_m": float(np.linalg.norm(center_scaled - target_center)),
        "usable_metric": bool(metric_rank == 3),
        "usable_scale_aware": bool(scale_rank == 4 and np.isfinite(center_scaled).all()),
    })
    return output


def _summary(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    usable = [row[prefix] for row in rows if row[prefix]["usable_scale_aware"]]
    metric = [row[prefix] for row in rows if row[prefix]["usable_metric"]]

    def values(name: str, source: list[dict[str, object]]) -> dict[str, float | None]:
        x = np.asarray([row[name] for row in source], np.float64)
        if not len(x):
            return {"median": None, "mean": None, "p90": None}
        return {
            "median": float(np.median(x)),
            "mean": float(np.mean(x)),
            "p90": float(np.quantile(x, 0.9)),
        }

    return {
        "query_count": len(rows),
        "metric_usable_count": len(metric),
        "scale_aware_usable_count": len(usable),
        "rotation_error_deg": values("rotation_error_deg", usable),
        "metric_translation_error_m": values("metric_translation_error_m", metric),
        "scale_translation_error_m": values("scale_translation_error_m", usable),
        "estimated_scale": values("estimated_scale", usable),
        "scale_aware_recall_2m45": float(np.mean([
            row["scale_translation_error_m"] <= 2.0 and row["rotation_error_deg"] <= 45.0
            for row in usable
        ])) if usable else 0.0,
        "scale_aware_recall_1m10": float(np.mean([
            row["scale_translation_error_m"] <= 1.0 and row["rotation_error_deg"] <= 10.0
            for row in usable
        ])) if usable else 0.0,
    }


def _point_summary(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    point_rows = [row[prefix] for row in rows]

    def mode_summary(mode: str) -> dict[str, object]:
        usable = [row[mode] for row in point_rows if row.get(f"usable_{mode}")]
        if not usable:
            return {
                "usable_count": 0,
                "full_query_recall_2m45": 0.0,
                "full_query_recall_1m10": 0.0,
                "conditional_recall_2m45": 0.0,
                "conditional_recall_1m10": 0.0,
            }

        def q(name: str) -> dict[str, float]:
            values = np.asarray([row[name] for row in usable], np.float64)
            return {
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "p90": float(np.quantile(values, 0.90)),
            }

        success_2m45 = [
            row["translation_error_m"] <= 2.0 and row["rotation_error_deg"] <= 45.0
            for row in usable
        ]
        success_1m10 = [
            row["translation_error_m"] <= 1.0 and row["rotation_error_deg"] <= 10.0
            for row in usable
        ]
        return {
            "usable_count": len(usable),
            "rotation_error_deg": q("rotation_error_deg"),
            "translation_error_m": q("translation_error_m"),
            "estimated_scale": q("estimated_scale"),
            "full_query_recall_2m45": float(np.sum(success_2m45) / len(rows)),
            "full_query_recall_1m10": float(np.sum(success_1m10) / len(rows)),
            "conditional_recall_2m45": float(np.mean(success_2m45)),
            "conditional_recall_1m10": float(np.mean(success_1m10)),
        }

    return {
        "query_count": len(rows),
        "metric": mode_summary("metric"),
        "scale_aware": mode_summary("scale_aware"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected_authority_content_sha256", required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--held_rays", type=Path, required=True)
    parser.add_argument("--moge3_query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite plane-pose oracle")
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority)
    if authority.metadata.get("content_sha256") != args.expected_authority_content_sha256:
        raise ValueError("plane-pose authority differs from pin")
    if (
        authority.metadata.get("moge_reference_only_topology") is not True
        or authority.metadata.get("dav2_geometry_consumed") is not False
        or authority.metadata.get("post_alignment_common_quarantine_control") is not True
    ):
        raise ValueError("plane-pose oracle requires quarantined MoGe/reference topology")
    held = StrictHeldRayInventory.load_npz(args.held_rays)
    vertices = _load_aligned_vertices(args.alignment, authority)
    map_points, map_vertex_normals = _map_surface_inventory(
        authority, vertices,
    )
    tree = cKDTree(map_points)
    moge_manifest_path = args.moge3_query / "manifest.json"
    moge_manifest = json.loads(moge_manifest_path.read_text())
    moge_rows = {row["name"]: row for row in moge_manifest["rows"]}
    if set(moge_rows) != set(held.view_names.astype(str)):
        raise ValueError("MoGe3 query inventory differs from held view inventory")

    rows: list[dict[str, object]] = []
    for view, name in enumerate(held.view_names.astype(str)):
        c2w = held.camera_to_world[view]
        reference_points = _camera_points(
            held.reference_depth_m[view], held.focal_xy[view], held.principal_xy[view],
        )
        reference_normal_camera = held.reference_normal_world[view] @ c2w[:3, :3]
        reference_query = extract_query_plane_regions(
            reference_points, reference_normal_camera, held.reference_valid[view],
        )
        reference_world = reference_points @ c2w[:3, :3].T + c2w[:3, 3]
        nearest_index, nearest_distance = _oracle_map_indices(
            tree, reference_world, held.reference_valid[view],
        )
        path = args.moge3_query / f"{name}.npz"
        if file_sha256(path) != moge_rows[name]["file_sha256"]:
            raise ValueError("MoGe3 query file differs from manifest")
        with np.load(path, allow_pickle=False) as data:
            points_moge = np.asarray(data["points_camera"], np.float64)
            normal_moge = np.asarray(data["normal_camera"], np.float64)
            valid_moge = np.asarray(data["valid"], bool)
        moge_query = extract_query_plane_regions(points_moge, normal_moge, valid_moge)
        common = held.reference_valid[view] & valid_moge
        scale_ratio = held.reference_depth_m[view][common] / points_moge[..., 2][common]
        depth_scale = float(np.median(scale_ratio)) if len(scale_ratio) else float("nan")
        raw_depth_error = np.abs(points_moge[..., 2][common] - held.reference_depth_m[view][common])
        scaled_depth_error = np.abs(
            depth_scale * points_moge[..., 2][common] - held.reference_depth_m[view][common]
        )
        rows.append({
            "name": name,
            "reference_valid_pixel_count": int(held.reference_valid[view].sum()),
            "oracle_map_associated_pixel_count": int(np.sum(nearest_index >= 0)),
            "oracle_map_associated_fraction": float(
                np.sum(nearest_index >= 0) / max(int(held.reference_valid[view].sum()), 1)
            ),
            "oracle_nearest_map_distance_median_m": float(np.median(
                nearest_distance[held.reference_valid[view]]
            )),
            "moge_common_pixel_count": int(common.sum()),
            "moge_depth_scale_median_reference_over_prediction": depth_scale,
            "moge_raw_depth_abs_error_median_m": float(np.median(raw_depth_error)) if len(raw_depth_error) else None,
            "moge_scaled_depth_abs_error_median_m": float(np.median(scaled_depth_error)) if len(scaled_depth_error) else None,
            "reference": _solve(
                reference_query,
                nearest_index,
                nearest_distance,
                map_points,
                map_vertex_normals,
                c2w,
            ),
            "moge3": _solve(
                moge_query,
                nearest_index,
                nearest_distance,
                map_points,
                map_vertex_normals,
                c2w,
            ),
            "reference_planar_support_point_pose": _point_pose_from_matched_planes(
                reference_query,
                reference_points,
                nearest_index,
                nearest_distance,
                map_points,
                map_vertex_normals,
                c2w,
            ),
            "moge3_planar_support_point_pose": _point_pose_from_matched_planes(
                moge_query,
                points_moge,
                nearest_index,
                nearest_distance,
                map_points,
                map_vertex_normals,
                c2w,
            ),
        })

    depth_scale = np.asarray([
        row["moge_depth_scale_median_reference_over_prediction"] for row in rows
    ], np.float64)
    pose_scale = np.asarray([
        row["moge3_planar_support_point_pose"].get("scale_aware", {}).get(
            "estimated_scale", np.nan,
        )
        for row in rows
    ], np.float64)
    scale_valid = np.isfinite(depth_scale) & np.isfinite(pose_scale)
    payload = {
        "artifact_type": SCHEMA,
        "authority_file_sha256": file_sha256(args.authority),
        "authority_content_sha256": args.expected_authority_content_sha256,
        "alignment_manifest_file_sha256": file_sha256(args.alignment / "manifest.json"),
        "held_rays_file_sha256": file_sha256(args.held_rays),
        "moge3_query_manifest_file_sha256": file_sha256(moge_manifest_path),
        "moge3_query_manifest_content_sha256": moge_manifest["content_sha256"],
        "map_surface_sample_count": int(len(map_points)),
        "map_plane_semantics": "query_footprint_local_PCA_plane_refit_from_oracle_nearest_atlas_samples",
        "correspondence_contract": {
            "oracle": True,
            "target_pose_used_only_for_map_region_correspondence_normal_sign_and_evaluation": True,
            "maximum_nearest_map_distance_m": MAXIMUM_MAP_ASSOCIATION_DISTANCE_M,
            "minimum_region_association_pixels": MINIMUM_REGION_ASSOCIATION_PIXELS,
            "minimum_region_association_coverage": MINIMUM_REGION_ASSOCIATION_COVERAGE,
            "maximum_association_normal_angle_deg": MAXIMUM_ASSOCIATION_NORMAL_ANGLE_DEG,
            "maximum_matched_planes": MAXIMUM_MATCHED_PLANES,
            "maximum_point_correspondences": MAXIMUM_POINT_CORRESPONDENCES,
            "minimum_point_correspondences": MINIMUM_POINT_CORRESPONDENCES,
        },
        "solver_contract": {
            "rotation": "weighted_Kabsch_query_plane_normal_to_world_plane_normal",
            "metric_translation": "weighted_plane_offset_least_squares_fixed_scale_1",
            "scale_aware_translation": "four_plane_deterministic_RANSAC_then_weighted_plane_offset_least_squares",
            "target_pose_not_passed_to_solver": True,
        },
        "held_inventory_previously_opened": True,
        "blind_or_preregistered_claim": False,
        "production_eligible": False,
        "promotion_eligible": False,
        "reference_summary": _summary(rows, "reference"),
        "moge3_summary": _summary(rows, "moge3"),
        "reference_planar_support_point_pose_summary": _point_summary(
            rows, "reference_planar_support_point_pose",
        ),
        "moge3_planar_support_point_pose_summary": _point_summary(
            rows, "moge3_planar_support_point_pose",
        ),
        "moge3_scale_consistency": {
            "view_count": int(scale_valid.sum()),
            "pearson_depth_ratio_vs_pose_scale": float(np.corrcoef(
                depth_scale[scale_valid], pose_scale[scale_valid],
            )[0, 1]) if scale_valid.sum() >= 2 else None,
            "absolute_difference_mean": float(np.mean(np.abs(
                depth_scale[scale_valid] - pose_scale[scale_valid]
            ))) if scale_valid.any() else None,
            "absolute_difference_median": float(np.median(np.abs(
                depth_scale[scale_valid] - pose_scale[scale_valid]
            ))) if scale_valid.any() else None,
        },
        "rows": rows,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "content_sha256": payload["content_sha256"],
        "reference_summary": payload["reference_summary"],
        "moge3_summary": payload["moge3_summary"],
        "reference_planar_support_point_pose_summary": payload[
            "reference_planar_support_point_pose_summary"
        ],
        "moge3_planar_support_point_pose_summary": payload[
            "moge3_planar_support_point_pose_summary"
        ],
        "moge3_scale_consistency": payload["moge3_scale_consistency"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
