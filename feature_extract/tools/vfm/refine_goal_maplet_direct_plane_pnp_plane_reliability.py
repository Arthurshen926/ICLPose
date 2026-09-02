"""Conservatively refine frozen plane-PnP poses using map-plane reliability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_budget_consensus import (
    _interpolate_pose,
)
from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_map_density import (
    _merge_correspondence,
    _robust_refine,
    _unique_token_inliers,
)
from feature_extract.tools.vfm.refine_goal_maplet_direct_plane_pnp_union_closure import (
    _load_primary,
    _normalized_pose_distance,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _scaled_intrinsics,
)
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import (
    PlaneVisibilityAtlas,
)


def _plane_reliability_weights(
    residual_rms_m: np.ndarray,
    *,
    residual_scale_m: float = 0.05,
    minimum: float = 0.5,
    maximum: float = 1.5,
) -> np.ndarray:
    """Return bounded, median-normalized inverse plane-residual weights."""
    residual = np.asarray(residual_rms_m, np.float64)
    if (
        residual.ndim != 1
        or not np.all(np.isfinite(residual))
        or np.any(residual < 0.0)
        or residual_scale_m <= 0.0
        or not 0.0 < minimum <= 1.0 <= maximum
    ):
        raise ValueError("plane reliability contract differs")
    if not len(residual):
        return np.ones(0, np.float64)
    weight = 1.0 / np.sqrt(1.0 + np.square(residual / float(residual_scale_m)))
    weight /= max(float(np.median(weight)), np.finfo(np.float64).eps)
    return np.clip(weight, float(minimum), float(maximum))


def _token_depth_dispersion(
    depth: np.ndarray,
    token_grid: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return mapping-only per-token mean depth and standard deviation.

    A token contains only roughly four rendered pixels at the active grid.
    Vectorized moments are both deterministic and far cheaper than millions
    of tiny median calls; a foreground/background mixture is deliberately
    treated as high uncertainty.
    """
    value = np.asarray(depth, np.float64)
    if value.ndim != 2:
        raise ValueError("mapping depth must be a 2D array")
    token_height, token_width = map(int, token_grid)
    yy, xx = np.indices(value.shape)
    owner_y = np.minimum(
        ((yy + 0.5) * token_height / value.shape[0]).astype(np.int64),
        token_height - 1,
    )
    owner_x = np.minimum(
        ((xx + 0.5) * token_width / value.shape[1]).astype(np.int64),
        token_width - 1,
    )
    owner = (owner_y * token_width + owner_x).reshape(-1)
    flat = value.reshape(-1)
    valid = np.isfinite(flat) & (flat > 0.0)
    size = token_height * token_width
    count = np.bincount(owner[valid], minlength=size).astype(np.float64)
    total = np.bincount(owner[valid], weights=flat[valid], minlength=size)
    total2 = np.bincount(owner[valid], weights=np.square(flat[valid]), minlength=size)
    center = np.full(size, np.nan, np.float64)
    dispersion = np.full(size, np.nan, np.float64)
    populated = count > 0.0
    center[populated] = total[populated] / count[populated]
    variance = total2[populated] / count[populated] - np.square(center[populated])
    dispersion[populated] = np.sqrt(np.maximum(variance, 0.0))
    return center, dispersion


def _mapping_depth_residuals(
    world_points: np.ndarray,
    pose_w2c: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    token_median_depth: np.ndarray,
    token_mad_depth: np.ndarray,
    token_grid: tuple[int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Measure local rendered-depth instability for frozen map 3D points."""
    world = np.asarray(world_points, np.float64)
    pose = np.asarray(pose_w2c, np.float64)
    K = np.asarray(camera_matrix, np.float64)
    mean_depth = np.asarray(token_median_depth, np.float64)
    depth_std = np.asarray(token_mad_depth, np.float64)
    if world.ndim != 2 or world.shape[1] != 3 or pose.shape != (4, 4) or K.shape != (3, 3):
        raise ValueError("mapping depth reliability geometry differs")
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    z = camera[:, 2]
    ideal = camera[:, :2] / np.maximum(z[:, None], np.finfo(np.float64).eps)
    radius2 = np.sum(np.square(ideal), axis=1)
    distorted = ideal * (1.0 + float(radial_k1) * radius2[:, None])
    pixel = np.c_[
        distorted[:, 0] * K[0, 0] + K[0, 2],
        distorted[:, 1] * K[1, 1] + K[1, 2],
    ]
    height, width = map(int, image_shape)
    token_height, token_width = map(int, token_grid)
    # Convert only finite projections.  Casting NaN/Inf directly to integers
    # is platform-warning-prone and can create an accidental valid token.
    finite_pixel = np.isfinite(pixel).all(axis=1)
    tx = np.full(len(pixel), -1, np.int64)
    ty = np.full(len(pixel), -1, np.int64)
    tx[finite_pixel] = np.floor(
        (pixel[finite_pixel, 0] + 0.5) * token_width / width
    ).astype(np.int64)
    ty[finite_pixel] = np.floor(
        (pixel[finite_pixel, 1] + 0.5) * token_height / height
    ).astype(np.int64)
    valid = (
        finite_pixel & np.isfinite(z) & (z > 0.0)
        & (tx >= 0) & (tx < token_width) & (ty >= 0) & (ty < token_height)
    )
    residual = np.zeros(len(world), np.float64)
    token = ty * token_width + tx
    rows = np.flatnonzero(valid)
    if len(rows):
        local_mean = mean_depth[token[rows]]
        local_std = depth_std[token[rows]]
        local_valid = np.isfinite(local_mean) & np.isfinite(local_std)
        selected = rows[local_valid]
        residual[selected] = np.hypot(
            local_std[local_valid], local_mean[local_valid] - z[selected],
        )
    return residual


def _query_relative_depth_dispersion(
    depth: np.ndarray,
    valid: np.ndarray,
    token_grid: tuple[int, int],
) -> np.ndarray:
    """Return scale-invariant per-token depth variation for query occlusion edges."""
    value = np.asarray(depth, np.float64)
    mask = np.asarray(valid, bool)
    if value.shape != mask.shape or value.ndim != 2:
        raise ValueError("query depth/valid arrays differ")
    masked = np.where(mask & np.isfinite(value) & (value > 0.0), value, np.nan)
    mean_depth, depth_std = _token_depth_dispersion(masked, token_grid)
    output = depth_std / np.maximum(mean_depth, np.finfo(np.float64).eps)
    output[~np.isfinite(mean_depth) | ~np.isfinite(depth_std)] = np.nan
    return output


def _weighted_candidate(
    pose: np.ndarray,
    world: np.ndarray,
    tokens: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    token_grid: tuple[int, int],
    weights: np.ndarray,
    spatial_balance: str = "none",
    spatial_group_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    current = np.asarray(pose, np.float64)
    rows, _ = _unique_token_inliers(
        current, world, tokens, camera_matrix, radial_k1, token_grid,
    )
    for _ in range(2):
        current = _robust_refine(
            current, rows, world, tokens, camera_matrix, radial_k1, token_grid,
            point_weights=weights, spatial_balance=spatial_balance,
            spatial_group_labels=spatial_group_labels,
        )
        rows, _ = _unique_token_inliers(
            current, world, tokens, camera_matrix, radial_k1, token_grid,
        )
    return current, int(len(rows))


def _damped_fraction(
    primary_ratio: float,
    *,
    maximum_primary_ratio: float,
    refinement_fraction: float,
    very_low_primary_ratio: float | None = None,
    very_low_refinement_fraction: float | None = None,
) -> float:
    if primary_ratio >= maximum_primary_ratio:
        return 0.0
    if (
        very_low_primary_ratio is not None
        and very_low_refinement_fraction is not None
        and primary_ratio < very_low_primary_ratio
    ):
        return float(very_low_refinement_fraction)
    return float(refinement_fraction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary_pose_inventory", type=Path, required=True)
    parser.add_argument("--top5_sparse_correspondence", type=Path, nargs="+", required=True)
    parser.add_argument("--top5_dense_correspondence", type=Path, nargs="+", required=True)
    parser.add_argument("--top10_sparse_correspondence", type=Path, nargs="+", required=True)
    parser.add_argument("--top10_dense_correspondence", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--supplemental_sparse_correspondence", type=Path, nargs="*", default=[],
        help=(
            "Optional pose-free sparse-map correspondence branch appended to the "
            "union support; it never replaces the four canonical branches."
        ),
    )
    parser.add_argument("--sparse_planar_map", type=Path, required=True)
    parser.add_argument("--dense_planar_map", type=Path, required=True)
    parser.add_argument("--sparse_visibility_atlas", type=Path)
    parser.add_argument("--dense_visibility_atlas", type=Path)
    parser.add_argument("--sparse_source_observation_bank", type=Path)
    parser.add_argument("--dense_source_observation_bank", type=Path)
    parser.add_argument("--sparse_mapping_contributors", type=Path)
    parser.add_argument("--dense_mapping_contributors", type=Path)
    parser.add_argument("--depth_dispersion_scale_m", type=float, default=0.10)
    parser.add_argument(
        "--query_moge3", type=Path, nargs="*", default=[],
        help="Optional MoGe3 query roots used only for scale-free token depth-edge weighting.",
    )
    parser.add_argument("--query_depth_edge_scale", type=float, default=0.05)
    parser.add_argument(
        "--spatial_balance",
        choices=(
            "none", "macrocell_equal_mass_3x5", "macrocell_equal_mass_4x6",
            "macrocell_equal_mass_4x6_shift_x",
            "macrocell_equal_mass_4x6_shift_y",
            "macrocell_equal_mass_4x6_shift_xy",
            "macrocell_equal_mass_5x8", "macrocell_equal_mass_6x10",
            "macrocell_equal_mass_6x10_shift_xy",
            "macrocell_equal_mass_8x12", "local_density_radius2",
            "query_region_equal_mass",
            "query_region_x_macrocell_4x6",
            "query_region_x_macrocell_6x10",
        ),
        default="none",
        help=(
            "Optional bounded image-space equal-mass weighting, combined with "
            "map-plane reliability only inside the low-support refinement gate."
        ),
    )
    parser.add_argument("--residual_scale_m", type=float, default=0.05)
    parser.add_argument("--maximum_primary_ratio", type=float, default=0.4)
    parser.add_argument("--maximum_pose_distance", type=float, default=0.75)
    parser.add_argument("--refinement_fraction", type=float, default=0.25)
    parser.add_argument("--very_low_primary_ratio", type=float)
    parser.add_argument("--very_low_refinement_fraction", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite plane-reliability refinement")
    if not 0.0 < args.maximum_primary_ratio < 1.0:
        raise ValueError("maximum primary ratio must lie in (0,1)")
    if args.maximum_pose_distance <= 0.0:
        raise ValueError("maximum pose distance must be positive")
    if not 0.0 < args.refinement_fraction < 1.0:
        raise ValueError("refinement fraction must lie in (0,1)")
    tiered = (
        args.very_low_primary_ratio is not None
        or args.very_low_refinement_fraction is not None
    )
    if tiered and not (
        args.very_low_primary_ratio is not None
        and args.very_low_refinement_fraction is not None
        and 0.0 < args.very_low_primary_ratio < args.maximum_primary_ratio
        and args.refinement_fraction < args.very_low_refinement_fraction < 1.0
    ):
        raise ValueError("very-low-support refinement tier differs")
    depth_reliability_arguments = (
        args.sparse_visibility_atlas,
        args.dense_visibility_atlas,
        args.sparse_mapping_contributors,
        args.dense_mapping_contributors,
        args.sparse_source_observation_bank,
        args.dense_source_observation_bank,
    )
    depth_reliability = any(value is not None for value in depth_reliability_arguments)
    if depth_reliability and (
        not all(value is not None for value in depth_reliability_arguments)
        or args.depth_dispersion_scale_m <= 0.0
    ):
        raise ValueError("mapping depth reliability inputs are incomplete")
    if depth_reliability and args.supplemental_sparse_correspondence:
        raise ValueError("depth reliability with supplemental correspondence is not implemented")
    if args.query_moge3 and args.query_depth_edge_scale <= 0.0:
        raise ValueError("query depth-edge scale must be positive")

    primary, primary_meta = _load_primary(args.primary_pose_inventory)
    paths_by_branch = [
        args.top5_sparse_correspondence,
        args.top5_dense_correspondence,
        args.top10_sparse_correspondence,
        args.top10_dense_correspondence,
    ]
    if args.supplemental_sparse_correspondence:
        paths_by_branch.append(args.supplemental_sparse_correspondence)
    loaded = [_merge_correspondence(paths) for paths in paths_by_branch]
    branches = [item[0] for item in loaded]
    branch_metadata = [item[1] for item in loaded]
    names = primary["names"].astype(str)
    for rows in branches:
        if not np.array_equal(
            names, np.asarray([str(row["name"].item()) for row in rows]),
        ):
            raise ValueError("plane-reliability query inventories differ")
    camera_hashes = {
        str(meta.get("query_camera_only_inventory_file_sha256"))
        for items in branch_metadata for meta in items
    }
    if camera_hashes != {
        str(primary_meta.get("query_camera_only_inventory_file_sha256"))
    }:
        raise ValueError("plane-reliability camera lineage differs")
    token_grids = {
        tuple(map(int, meta.get("token_grid", ())))
        for items in branch_metadata for meta in items
    }
    if len(token_grids) != 1:
        raise ValueError("plane-reliability token grids differ")
    token_grid = next(iter(token_grids))
    sparse_map = GeometryNativePlanarMap.load_npz(args.sparse_planar_map)
    dense_map = GeometryNativePlanarMap.load_npz(args.dense_planar_map)
    maps = [sparse_map, dense_map, sparse_map, dense_map]
    if args.supplemental_sparse_correspondence:
        maps.append(sparse_map)
    atlases = contributor_roots = None
    if depth_reliability:
        sparse_atlas, sparse_atlas_meta = PlaneVisibilityAtlas.load_npz(
            args.sparse_visibility_atlas
        )
        dense_atlas, dense_atlas_meta = PlaneVisibilityAtlas.load_npz(
            args.dense_visibility_atlas
        )
        bank_metadata = []
        for path in (
            args.sparse_source_observation_bank,
            args.dense_source_observation_bank,
        ):
            with np.load(path, allow_pickle=False) as data:
                bank_metadata.append(json.loads(str(data["metadata_json"].item())))
        for index, (path, meta, atlas_path, atlas_meta) in enumerate(zip(
            (args.sparse_source_observation_bank, args.dense_source_observation_bank),
            bank_metadata,
            (args.sparse_visibility_atlas, args.dense_visibility_atlas),
            (sparse_atlas_meta, dense_atlas_meta),
        )):
            expected_bank_hash = {
                str(item.get("source_observation_bank_file_sha256"))
                for branch_index in (index, index + 2)
                for item in branch_metadata[branch_index]
            }
            if (
                meta.get("artifact_type") != "goal_maplet_plane_pnp_observation_bank_v1"
                or expected_bank_hash != {file_sha256(path)}
                or meta.get("visibility_atlas_file_sha256") != file_sha256(atlas_path)
                or meta.get("visibility_atlas_content_sha256") != atlas_meta.get("content_sha256")
            ):
                raise ValueError("mapping depth reliability observation-bank lineage differs")
        atlases = (sparse_atlas, dense_atlas, sparse_atlas, dense_atlas)
        contributor_roots = (
            args.sparse_mapping_contributors,
            args.dense_mapping_contributors,
            args.sparse_mapping_contributors,
            args.dense_mapping_contributors,
        )
        if (
            tuple(sparse_atlas_meta.get("token_grid", ())) != token_grid
            or tuple(dense_atlas_meta.get("token_grid", ())) != token_grid
        ):
            raise ValueError("mapping depth reliability token grid differs")
    depth_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]] = {}
    contributor_hashes: dict[str, str] = {}
    moge_manifest_paths: list[Path] = []
    moge_manifest_metadata: list[dict[str, object]] = []
    moge_rows: dict[str, tuple[Path, dict[str, object]]] = {}
    for root in args.query_moge3:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        rows = {
            str(row["image_id"]).replace("/", "__") + ".npz": row
            for row in manifest.get("rows", [])
        }
        if (
            manifest.get("artifact_type") != "goal_maplet_moge_query_geometry_v2_manifest"
            or manifest.get("output_height") != 144
            or manifest.get("output_width") != 256
            or len(rows) != len(manifest.get("rows", []))
            or set(rows) & set(moge_rows)
        ):
            raise ValueError("query MoGe3 manifest differs")
        moge_rows.update({name: (root, row) for name, row in rows.items()})
        moge_manifest_paths.append(manifest_path)
        moge_manifest_metadata.append(manifest)
    if args.query_moge3 and set(names.tolist()) != set(moge_rows):
        raise ValueError("query MoGe3 inventory differs")
    query_edge_cache: dict[str, np.ndarray] = {}

    def query_edge_geometry(name: str) -> np.ndarray:
        if name in query_edge_cache:
            return query_edge_cache[name]
        root, row = moge_rows[name]
        path = root / name
        if file_sha256(path) != row.get("file_sha256"):
            raise ValueError("query MoGe3 bytes differ")
        with np.load(path, allow_pickle=False) as data:
            depth = np.asarray(data["depth_camera"], np.float64)
            valid = np.asarray(data["valid"], bool)
            embedded = json.loads(str(data["metadata_json"].item()))
        if (
            embedded.get("artifact_type") != "goal_maplet_moge_query_geometry_v2"
            or embedded.get("content_sha256") != row.get("content_sha256")
            or embedded.get("uses_ground_truth") is not False
            or embedded.get("uses_pose") is not False
        ):
            raise ValueError("query MoGe3 metadata differs")
        query_edge_cache[name] = _query_relative_depth_dispersion(
            depth, valid, token_grid,
        )
        return query_edge_cache[name]

    def depth_geometry(branch_index: int, atlas_row: int):
        atlas = atlases[branch_index]
        root = contributor_roots[branch_index]
        name = str(atlas.view_names[int(atlas_row)])
        key = (branch_index % 2, name)
        if key in depth_cache:
            return depth_cache[key]
        path = root / name
        with np.load(path, allow_pickle=False) as data:
            depth = np.asarray(data["dominant_depth"], np.float64)
            pose = np.asarray(data["pose_w2c"], np.float64)
            model_id = int(data["camera_model_id"])
            width = int(data["camera_width"])
            height = int(data["camera_height"])
            params = np.asarray(data["camera_params"], np.float64)
        K, k1 = _scaled_intrinsics(model_id, params, width, height)
        median, mad = _token_depth_dispersion(depth, token_grid)
        depth_cache[key] = depth, pose, K, k1, median, mad
        contributor_hashes[f"{branch_index % 2}:{name}"] = file_sha256(path)
        return depth_cache[key]

    output_pose = np.asarray(primary["pose_w2c"], np.float64).copy()
    output_branch = np.full(len(names), 70, np.int16)
    candidate_distance = np.full(len(names), np.inf, np.float64)
    candidate_inliers = np.zeros(len(names), np.int64)
    weight_minimum = np.ones(len(names), np.float64)
    weight_maximum = np.ones(len(names), np.float64)
    applied = 0
    for index, name in enumerate(names.tolist()):
        if not bool(primary["usable"][index]):
            continue
        rows = [branch[index] for branch in branches]
        first = rows[0]
        if any(
            not np.array_equal(first["camera_matrix"], row["camera_matrix"])
            or float(first["radial_k1"]) != float(row["radial_k1"])
            for row in rows[1:]
        ):
            raise ValueError(f"plane-reliability cameras differ for {name}")
        world = np.concatenate([row["world_points"] for row in rows], axis=0)
        tokens = np.concatenate([row["query_tokens"] for row in rows], axis=0)
        query_region_groups = np.concatenate([
            np.asarray(row["provenance"], np.int64)[:, 0] for row in rows
        ], axis=0)
        residual_rows = []
        for row, planar in zip(rows, maps):
            plane = np.asarray(row["provenance"], np.int64)[:, 1]
            if np.any(plane < 0) or np.any(plane >= len(planar.plane_ids)):
                raise ValueError("correspondence plane is outside reliability map")
            residual_rows.append(planar.residual_rms_m[plane])
        weights = _plane_reliability_weights(
            np.concatenate(residual_rows), residual_scale_m=float(args.residual_scale_m),
        )
        if args.query_moge3:
            edge_by_token = query_edge_geometry(name)
            edge = edge_by_token[tokens]
            finite = np.isfinite(edge)
            neutral = float(np.median(edge[finite])) if np.any(finite) else 0.0
            edge = np.where(finite, edge, neutral)
            query_weights = _plane_reliability_weights(
                edge, residual_scale_m=float(args.query_depth_edge_scale),
            )
            weights = np.sqrt(weights * query_weights)
        if depth_reliability:
            depth_residual_rows = []
            for branch_index, row in enumerate(rows):
                provenance = np.asarray(row["provenance"], np.int64)
                residual = np.zeros(len(provenance), np.float64)
                for atlas_row in np.unique(provenance[:, 2]).tolist():
                    selected = np.flatnonzero(provenance[:, 2] == int(atlas_row))
                    depth, pose, K, k1, median, mad = depth_geometry(
                        branch_index, int(atlas_row),
                    )
                    residual[selected] = _mapping_depth_residuals(
                        np.asarray(row["world_points"], np.float64)[selected],
                        pose, K, k1, median, mad, token_grid, depth.shape,
                    )
                depth_residual_rows.append(residual)
            depth_weights = _plane_reliability_weights(
                np.concatenate(depth_residual_rows),
                residual_scale_m=float(args.depth_dispersion_scale_m),
            )
            weights = np.sqrt(weights * depth_weights)
        candidate, inlier_count = _weighted_candidate(
            output_pose[index], world, tokens,
            np.asarray(first["camera_matrix"], np.float64),
            float(first["radial_k1"]), token_grid, weights,
            spatial_balance=str(args.spatial_balance),
            spatial_group_labels=query_region_groups,
        )
        distance = _normalized_pose_distance(output_pose[index], candidate)
        candidate_distance[index] = distance
        candidate_inliers[index] = inlier_count
        weight_minimum[index] = float(np.min(weights)) if len(weights) else 1.0
        weight_maximum[index] = float(np.max(weights)) if len(weights) else 1.0
        fraction = _damped_fraction(
            float(primary["selected_inlier_ratio"][index]),
            maximum_primary_ratio=float(args.maximum_primary_ratio),
            refinement_fraction=float(args.refinement_fraction),
            very_low_primary_ratio=args.very_low_primary_ratio,
            very_low_refinement_fraction=args.very_low_refinement_fraction,
        )
        if fraction > 0.0 and distance < args.maximum_pose_distance:
            output_pose[index] = _interpolate_pose(
                output_pose[index], candidate, fraction,
            )
            output_branch[index] = (
                72
                if args.very_low_primary_ratio is not None
                and float(primary["selected_inlier_ratio"][index])
                < float(args.very_low_primary_ratio)
                else 71
            )
            applied += 1

    arrays = {
        "names": names,
        "pose_w2c": output_pose,
        "usable": np.asarray(primary["usable"], bool),
        "selected_branch": output_branch,
        "selected_inlier_ratio": np.asarray(primary["selected_inlier_ratio"], np.float64),
        "selected_candidate_correspondence_count": np.asarray(
            primary["selected_candidate_correspondence_count"], np.int64,
        ),
        "selected_pnp_inlier_count": np.asarray(
            primary["selected_pnp_inlier_count"], np.int64,
        ),
        "reliability_candidate_pose_distance": candidate_distance,
        "reliability_candidate_pnp_inlier_count": candidate_inliers,
        "reliability_weight_minimum": weight_minimum,
        "reliability_weight_maximum": weight_maximum,
    }
    metadata = {
        "artifact_type": (
            "goal_maplet_direct_plane_pnp_query_depth_edge_reliability_tiered_damped_refinement_v8"
            if args.query_moge3 else
            "goal_maplet_direct_plane_pnp_plane_and_depth_reliability_tiered_damped_refinement_v3"
            if depth_reliability else
            "goal_maplet_direct_plane_pnp_spatial_plane_reliability_tiered_damped_refinement_v4"
            if tiered and args.spatial_balance != "none" else
            "goal_maplet_direct_plane_pnp_reliability_tiered_damped_refinement_v2"
            if tiered
            else "goal_maplet_direct_plane_pnp_reliability_damped_refinement_v1"
        ),
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "selection_rule": (
            "monotonic_two_tier_low_primary_support_bounded_plane_reliability_damping"
            if tiered
            else "low_primary_support_bounded_plane_reliability_candidate_quarter_step"
        ),
        "plane_reliability": (
            "median_normalized_1/sqrt(1+(residual_rms/residual_scale_m)^2)_clip_0.5_1.5"
        ),
        "mapping_depth_reliability": (
            "geometric_mean_with_median_normalized_local_token_depth_std_and_depth_residual"
            if depth_reliability else None
        ),
        "query_depth_edge_reliability": (
            "scale_invariant_per_token_depth_std_over_mean_geometric_mean_with_plane_weight"
            if args.query_moge3 else None
        ),
        "query_depth_edge_scale": (
            float(args.query_depth_edge_scale) if args.query_moge3 else None
        ),
        "query_moge3_manifest_file_sha256_in_order": [
            file_sha256(path) for path in moge_manifest_paths
        ],
        "query_moge3_manifest_content_sha256_in_order": [
            metadata.get("content_sha256") for metadata in moge_manifest_metadata
        ],
        "spatial_balance": str(args.spatial_balance),
        "depth_dispersion_scale_m": (
            float(args.depth_dispersion_scale_m) if depth_reliability else None
        ),
        "mapping_depth_contributor_file_sha256_by_density_and_name": dict(
            sorted(contributor_hashes.items())
        ),
        "sparse_visibility_atlas_file_sha256": (
            file_sha256(args.sparse_visibility_atlas) if depth_reliability else None
        ),
        "dense_visibility_atlas_file_sha256": (
            file_sha256(args.dense_visibility_atlas) if depth_reliability else None
        ),
        "sparse_source_observation_bank_file_sha256": (
            file_sha256(args.sparse_source_observation_bank) if depth_reliability else None
        ),
        "dense_source_observation_bank_file_sha256": (
            file_sha256(args.dense_source_observation_bank) if depth_reliability else None
        ),
        "residual_scale_m": float(args.residual_scale_m),
        "maximum_primary_ratio": float(args.maximum_primary_ratio),
        "maximum_pose_distance": float(args.maximum_pose_distance),
        "pose_distance": "translation_m/0.5 + rotation_deg/5",
        "refinement_fraction": float(args.refinement_fraction),
        "very_low_primary_ratio": args.very_low_primary_ratio,
        "very_low_refinement_fraction": args.very_low_refinement_fraction,
        "refinement_applied_count": int(applied),
        "primary_file_sha256": file_sha256(args.primary_pose_inventory),
        "primary_content_sha256": primary_meta.get("content_sha256"),
        "sparse_planar_map_file_sha256": file_sha256(args.sparse_planar_map),
        "dense_planar_map_file_sha256": file_sha256(args.dense_planar_map),
        "correspondence_file_sha256_by_branch": [
            [file_sha256(path) for path in paths] for paths in paths_by_branch
        ],
        "query_camera_only_inventory_file_sha256": primary_meta.get(
            "query_camera_only_inventory_file_sha256"
        ),
        "query_pose_or_ground_truth_read": False,
        "strict_runtime_phase_separation_eligible": True,
        "query_depth_or_scale_used_by_pose_solver": bool(args.query_moge3),
        "configuration_role": "historical_validation_adaptive_ablation_not_pristine_blind",
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary.npz")
    np.savez_compressed(
        temporary, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    print(json.dumps({**metadata, "output_file_sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
