"""Build a view-independent metric UV/RADIO atlas for finite planes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.chart_local_radio_projection import (
    load_chart_local_radio_projection,
    project_chart_local_radio,
)
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import (
    PlaneVisibilityAtlas,
)


def _fuse_plane_texels(
    uv: np.ndarray,
    features: np.ndarray,
    view_rows: np.ndarray,
    *,
    cell_size_m: float,
    minimum_views: int,
    view_directions_world: np.ndarray | None = None,
    observation_ranges_m: np.ndarray | None = None,
    surface_height_m: np.ndarray | None = None,
    geometry_uv_m: np.ndarray | None = None,
    geometry_covariance_m2: np.ndarray | None = None,
    plane_pixel_purity: np.ndarray | None = None,
    plane_depth_dispersion_m: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Fuse tokens view-first, then view-balanced within each metric texel."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    geometry_uv = uv if geometry_uv_m is None else np.asarray(geometry_uv_m, np.float64).reshape(-1, 2)
    features = np.asarray(features, np.float32)
    views = np.asarray(view_rows, np.int64).reshape(-1)
    with_geometry = view_directions_world is not None or observation_ranges_m is not None
    if with_geometry:
        if view_directions_world is None or observation_ranges_m is None:
            raise ValueError("both view direction and observation range are required")
        directions = np.asarray(view_directions_world, np.float64).reshape(-1, 3)
        ranges = np.asarray(observation_ranges_m, np.float64).reshape(-1)
        if len(directions) != len(uv) or len(ranges) != len(uv):
            raise ValueError("plane observation geometry differs in length")
    with_surface = surface_height_m is not None
    if with_surface:
        heights = np.asarray(surface_height_m, np.float64).reshape(-1)
        if len(heights) != len(uv) or not np.all(np.isfinite(heights)):
            raise ValueError("plane surface heights differ")
    with_uncertainty = any(
        value is not None for value in
        (geometry_covariance_m2, plane_pixel_purity, plane_depth_dispersion_m)
    )
    if with_uncertainty:
        if any(value is None for value in (geometry_covariance_m2, plane_pixel_purity, plane_depth_dispersion_m)):
            raise ValueError("all plane geometry uncertainty arrays are required together")
        covariance = np.asarray(geometry_covariance_m2, np.float64).reshape(-1, 3, 3)
        purity = np.asarray(plane_pixel_purity, np.float64).reshape(-1)
        dispersion = np.asarray(plane_depth_dispersion_m, np.float64).reshape(-1)
        if not (len(covariance) == len(purity) == len(dispersion) == len(uv)):
            raise ValueError("plane geometry uncertainty differs in length")
        if not (np.all(np.isfinite(covariance)) and np.all(np.isfinite(purity)) and np.all(np.isfinite(dispersion))):
            raise ValueError("plane geometry uncertainty must be finite")
        if np.any((purity < 0.0) | (purity > 1.0)) or np.any(dispersion < 0.0):
            raise ValueError("invalid plane geometry uncertainty")
    if not (len(uv) == len(geometry_uv) == len(features) == len(views)):
        raise ValueError("plane UV inputs differ in length")
    if not len(uv):
        base = (
            np.zeros((0, 2), np.float64),
            np.zeros((0, features.shape[1]), np.float32),
            np.zeros(0, np.uint16),
            np.zeros(0, np.uint32),
        )
        return (
            base
            + ((np.zeros((0, 3), np.float64), np.zeros(0, np.float64)) if with_geometry else ())
            + ((np.zeros(0, np.float64), np.zeros(0, np.float64)) if with_surface else ())
            + ((np.zeros((0, 3, 3), np.float64), np.zeros(0, np.float64), np.zeros(0, np.float64)) if with_uncertainty else ())
        )
    cells = np.floor(uv / float(cell_size_m)).astype(np.int64)
    # First average all tokens from one source view in one texel.  This keeps
    # image footprint/token density from becoming an implicit view weight.
    order = np.lexsort((views, cells[:, 1], cells[:, 0]))
    cells, views = cells[order], views[order]
    uv = uv[order]
    geometry_uv = geometry_uv[order]
    features = features[order]
    if with_geometry:
        directions, ranges = directions[order], ranges[order]
    if with_surface:
        heights = heights[order]
    if with_uncertainty:
        covariance, purity, dispersion = covariance[order], purity[order], dispersion[order]
    tokens_per_pair = np.ones(len(order), np.uint32)
    pair_start = np.r_[
        True,
        np.any(cells[1:] != cells[:-1], axis=1) | (views[1:] != views[:-1]),
    ]
    starts = np.flatnonzero(pair_start)
    pair_features = np.add.reduceat(features, starts, axis=0)
    pair_uv = np.add.reduceat(geometry_uv, starts, axis=0)
    if with_geometry:
        pair_directions = np.add.reduceat(directions, starts, axis=0)
        pair_ranges = np.add.reduceat(ranges, starts)
    if with_surface:
        pair_heights = np.add.reduceat(heights, starts)
        pair_height_second = np.add.reduceat(heights * heights, starts)
    if with_uncertainty:
        pair_covariance = np.add.reduceat(covariance, starts, axis=0)
        pair_purity = np.add.reduceat(purity, starts)
        pair_dispersion = np.add.reduceat(dispersion, starts)
    pair_tokens = np.add.reduceat(tokens_per_pair, starts)
    pair_features /= pair_tokens[:, None]
    pair_uv /= pair_tokens[:, None]
    if with_geometry:
        pair_directions /= np.maximum(np.linalg.norm(pair_directions, axis=1, keepdims=True), 1e-12)
        pair_ranges /= pair_tokens
    if with_surface:
        pair_heights /= pair_tokens
        pair_height_second /= pair_tokens
    if with_uncertainty:
        pair_covariance /= pair_tokens[:, None, None]
        pair_purity /= pair_tokens
        pair_dispersion /= pair_tokens
    pair_features /= np.maximum(np.linalg.norm(pair_features, axis=1, keepdims=True), 1e-8)
    pair_cells = cells[starts]

    # Then average the independently normalized view observations.
    cell_start = np.r_[True, np.any(pair_cells[1:] != pair_cells[:-1], axis=1)]
    starts = np.flatnonzero(cell_start)
    descriptor = np.add.reduceat(pair_features, starts, axis=0)
    texel_uv = np.add.reduceat(pair_uv, starts, axis=0)
    if with_geometry:
        texel_directions = np.add.reduceat(pair_directions, starts, axis=0)
        texel_ranges = np.add.reduceat(pair_ranges, starts)
    if with_surface:
        texel_heights = np.add.reduceat(pair_heights, starts)
        texel_height_second = np.add.reduceat(pair_height_second, starts)
    if with_uncertainty:
        texel_covariance = np.add.reduceat(pair_covariance, starts, axis=0)
        texel_purity = np.add.reduceat(pair_purity, starts)
        texel_dispersion = np.add.reduceat(pair_dispersion, starts)
    view_support = np.diff(np.r_[starts, len(pair_features)]).astype(np.uint16)
    token_support = np.add.reduceat(pair_tokens, starts).astype(np.uint32)
    descriptor /= np.maximum(np.linalg.norm(descriptor, axis=1, keepdims=True), 1e-8)
    texel_uv /= view_support[:, None]
    if with_geometry:
        texel_directions /= np.maximum(np.linalg.norm(texel_directions, axis=1, keepdims=True), 1e-12)
        texel_ranges /= view_support
    if with_surface:
        texel_heights /= view_support
        texel_height_second /= view_support
        texel_height_std = np.sqrt(np.maximum(texel_height_second - texel_heights * texel_heights, 0.0))
    if with_uncertainty:
        texel_covariance /= view_support[:, None, None]
        texel_purity /= view_support
        texel_dispersion /= view_support
    keep = view_support >= int(minimum_views)
    base = (texel_uv[keep], descriptor[keep], view_support[keep], token_support[keep])
    return (
        base
        + ((texel_directions[keep], texel_ranges[keep]) if with_geometry else ())
        + ((texel_heights[keep], texel_height_std[keep]) if with_surface else ())
        + ((texel_covariance[keep], texel_purity[keep], texel_dispersion[keep]) if with_uncertainty else ())
    )


def _fuse_plane_texel_prototypes(
    uv: np.ndarray,
    features: np.ndarray,
    view_rows: np.ndarray,
    *,
    cell_size_m: float,
    minimum_views: int,
    maximum_prototypes: int,
    view_directions_world: np.ndarray | None = None,
    observation_ranges_m: np.ndarray | None = None,
    surface_height_m: np.ndarray | None = None,
    geometry_uv_m: np.ndarray | None = None,
    geometry_covariance_m2: np.ndarray | None = None,
    plane_pixel_purity: np.ndarray | None = None,
    plane_depth_dispersion_m: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Keep deterministic diverse view prototypes at one shared metric texel."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    geometry_uv = uv if geometry_uv_m is None else np.asarray(geometry_uv_m, np.float64).reshape(-1, 2)
    features = np.asarray(features, np.float32)
    views = np.asarray(view_rows, np.int64).reshape(-1)
    with_geometry = view_directions_world is not None or observation_ranges_m is not None
    if with_geometry:
        if view_directions_world is None or observation_ranges_m is None:
            raise ValueError("both view direction and observation range are required")
        directions = np.asarray(view_directions_world, np.float64).reshape(-1, 3)
        ranges = np.asarray(observation_ranges_m, np.float64).reshape(-1)
        if len(directions) != len(uv) or len(ranges) != len(uv):
            raise ValueError("plane observation geometry differs in length")
    with_surface = surface_height_m is not None
    if with_surface:
        heights = np.asarray(surface_height_m, np.float64).reshape(-1)
        if len(heights) != len(uv) or not np.all(np.isfinite(heights)):
            raise ValueError("plane surface heights differ")
    with_uncertainty = any(
        value is not None for value in
        (geometry_covariance_m2, plane_pixel_purity, plane_depth_dispersion_m)
    )
    if with_uncertainty:
        if any(value is None for value in (geometry_covariance_m2, plane_pixel_purity, plane_depth_dispersion_m)):
            raise ValueError("all plane geometry uncertainty arrays are required together")
        covariance = np.asarray(geometry_covariance_m2, np.float64).reshape(-1, 3, 3)
        purity = np.asarray(plane_pixel_purity, np.float64).reshape(-1)
        dispersion = np.asarray(plane_depth_dispersion_m, np.float64).reshape(-1)
        if not (len(covariance) == len(purity) == len(dispersion) == len(uv)):
            raise ValueError("plane geometry uncertainty differs in length")
        if not (np.all(np.isfinite(covariance)) and np.all(np.isfinite(purity)) and np.all(np.isfinite(dispersion))):
            raise ValueError("plane geometry uncertainty must be finite")
        if np.any((purity < 0.0) | (purity > 1.0)) or np.any(dispersion < 0.0):
            raise ValueError("invalid plane geometry uncertainty")
    if not (len(uv) == len(geometry_uv) == len(features) == len(views)):
        raise ValueError("plane UV inputs differ in length")
    if not len(uv):
        base = (
            np.zeros((0, 2), np.float64), np.zeros((0, features.shape[1]), np.float32),
            np.zeros(0, np.int64), np.zeros(0, np.uint8),
            np.zeros(0, np.uint16), np.zeros(0, np.uint32),
        )
        return (
            base
            + ((np.zeros((0, 3), np.float64), np.zeros(0, np.float64)) if with_geometry else ())
            + ((np.zeros(0, np.float64), np.zeros(0, np.float64)) if with_surface else ())
            + ((np.zeros((0, 3, 3), np.float64), np.zeros(0, np.float64), np.zeros(0, np.float64)) if with_uncertainty else ())
        )
    cells = np.floor(uv / float(cell_size_m)).astype(np.int64)
    order = np.lexsort((views, cells[:, 1], cells[:, 0]))
    cells, views, features, uv = cells[order], views[order], features[order], uv[order]
    geometry_uv = geometry_uv[order]
    if with_geometry:
        directions, ranges = directions[order], ranges[order]
    if with_surface:
        heights = heights[order]
    if with_uncertainty:
        covariance, purity, dispersion = covariance[order], purity[order], dispersion[order]
    pair_start = np.r_[
        True, np.any(cells[1:] != cells[:-1], axis=1) | (views[1:] != views[:-1])
    ]
    starts = np.flatnonzero(pair_start)
    pair_tokens = np.diff(np.r_[starts, len(features)]).astype(np.uint32)
    pair_features = np.add.reduceat(features, starts, axis=0) / pair_tokens[:, None]
    pair_uv = np.add.reduceat(geometry_uv, starts, axis=0) / pair_tokens[:, None]
    if with_geometry:
        pair_directions = np.add.reduceat(directions, starts, axis=0)
        pair_directions /= np.maximum(np.linalg.norm(pair_directions, axis=1, keepdims=True), 1e-12)
        pair_ranges = np.add.reduceat(ranges, starts) / pair_tokens
    if with_surface:
        pair_heights = np.add.reduceat(heights, starts) / pair_tokens
        pair_height_second = np.add.reduceat(heights * heights, starts) / pair_tokens
        pair_height_std = np.sqrt(np.maximum(pair_height_second - pair_heights * pair_heights, 0.0))
    if with_uncertainty:
        pair_covariance = np.add.reduceat(covariance, starts, axis=0) / pair_tokens[:, None, None]
        pair_purity = np.add.reduceat(purity, starts) / pair_tokens
        pair_dispersion = np.add.reduceat(dispersion, starts) / pair_tokens
    pair_features /= np.maximum(np.linalg.norm(pair_features, axis=1, keepdims=True), 1e-8)
    pair_cells = cells[starts]
    cell_start = np.r_[True, np.any(pair_cells[1:] != pair_cells[:-1], axis=1)]
    cell_starts = np.flatnonzero(cell_start)
    cell_ends = np.r_[cell_starts[1:], len(pair_features)]
    output_uv, output_feature, output_identity, output_rank = [], [], [], []
    output_views, output_tokens = [], []
    output_directions, output_ranges = [], []
    output_heights, output_height_std = [], []
    output_covariance, output_purity, output_dispersion = [], [], []
    identity = 0
    for lo, hi in zip(cell_starts.tolist(), cell_ends.tolist()):
        support = hi - lo
        if support < int(minimum_views):
            continue
        value = pair_features[lo:hi]
        similarity = value @ value.T
        selected = [int(np.argmax(np.sum(similarity, axis=1)))]
        while len(selected) < min(int(maximum_prototypes), support):
            remaining = np.asarray([row for row in range(support) if row not in selected])
            maximum_similarity = np.max(similarity[remaining][:, selected], axis=1)
            selected.append(int(remaining[np.argmin(maximum_similarity)]))
        # The integer cell is identity/aggregation only.  Geometry preserves
        # the view-balanced metric observation rather than quantizing to the
        # cell centre (which can create up to sqrt(2)/2 cell of PnP error).
        for rank, local in enumerate(selected):
            # A feature mode is tied to the metric location actually observed
            # by that anonymous view-mode.  Assigning all diverse descriptors
            # to the cell mean silently creates false feature/geometry pairs
            # and was the main reason four-prototype atlases underperformed.
            output_uv.append(pair_uv[lo + local])
            output_feature.append(value[local])
            output_identity.append(identity)
            output_rank.append(rank)
            output_views.append(support)
            output_tokens.append(int(np.sum(pair_tokens[lo:hi])))
            if with_geometry:
                output_directions.append(pair_directions[lo + local])
                output_ranges.append(pair_ranges[lo + local])
            if with_surface:
                output_heights.append(pair_heights[lo + local])
                output_height_std.append(pair_height_std[lo + local])
            if with_uncertainty:
                output_covariance.append(pair_covariance[lo + local])
                output_purity.append(pair_purity[lo + local])
                output_dispersion.append(pair_dispersion[lo + local])
        identity += 1
    base = (
        np.asarray(output_uv, np.float64).reshape(-1, 2),
        np.asarray(output_feature, np.float32).reshape(-1, features.shape[1]),
        np.asarray(output_identity, np.int64), np.asarray(output_rank, np.uint8),
        np.asarray(output_views, np.uint16), np.asarray(output_tokens, np.uint32),
    )
    return (
        base
        + (
            (np.asarray(output_directions, np.float64).reshape(-1, 3), np.asarray(output_ranges, np.float64))
            if with_geometry else ()
        )
        + (
            (np.asarray(output_heights, np.float64), np.asarray(output_height_std, np.float64))
            if with_surface else ()
        )
        + (
            (
                np.asarray(output_covariance, np.float64).reshape(-1, 3, 3),
                np.asarray(output_purity, np.float64),
                np.asarray(output_dispersion, np.float64),
            )
            if with_uncertainty else ()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planar_map", type=Path, required=True)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--observation_bank", type=Path, required=True)
    parser.add_argument(
        "--radio_projection", type=Path,
        help="Optional mapping-only chart-local descriptor projection.",
    )
    parser.add_argument(
        "--geometry_observation_bank", type=Path,
        help="Optional aligned v2 bank supplying independent plane-specific geometry.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell_size_m", type=float, default=0.5)
    parser.add_argument("--minimum_views", type=int, default=2)
    parser.add_argument("--maximum_prototypes_per_texel", type=int, default=1)
    parser.add_argument(
        "--surface_height_policy",
        choices=("raw", "frozen_offset_fallback"),
        default="frozen_offset_fallback",
        help=(
            "Whether to retain the plane-labelled observation's signed normal residual "
            "or fall back to the ideal plane outside the map's frozen offset tolerance."
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite plane UV RADIO atlas")
    if (
        float(args.cell_size_m) <= 0.0 or int(args.minimum_views) < 1
        or int(args.maximum_prototypes_per_texel) < 1
    ):
        raise ValueError("invalid plane UV atlas configuration")

    planes = GeometryNativePlanarMap.load_npz(args.planar_map)
    with np.load(args.planar_map, allow_pickle=False) as data:
        planar_meta = json.loads(str(data["metadata_json"].item()))
    surface_height_limit_m = float(planar_meta.get("configuration", {}).get("offset_m", 0.0))
    if not np.isfinite(surface_height_limit_m) or surface_height_limit_m <= 0.0:
        raise ValueError("planar map lacks a positive frozen offset tolerance")
    visibility, visibility_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    if len(planes.plane_ids) != len(visibility.plane_offsets) - 1:
        raise ValueError("planar map and visibility atlas plane inventory differ")
    with np.load(args.observation_bank, allow_pickle=False) as data:
        bank_meta = json.loads(str(data["metadata_json"].item()))
        observation_offsets = np.asarray(data["observation_offsets"], np.int64)
        points_raw = np.asarray(data["world_points"])
        points = np.asarray(points_raw, np.float64)
        features = np.asarray(data["radio_features"], np.float32)
        bank_arrays = {
            "observation_offsets": observation_offsets,
            "token_ids": np.asarray(data["token_ids"]),
            "world_points": points_raw,
            "radio_features": np.asarray(data["radio_features"]),
        }
        if bank_meta.get("artifact_type") == "goal_maplet_plane_pnp_observation_bank_v2":
            for name in (
                "world_point_covariance_m2",
                "plane_pixel_purity",
                "plane_depth_dispersion_m",
            ):
                if name not in data:
                    raise ValueError(f"plane-specific observation bank lacks {name}")
                bank_arrays[name] = np.asarray(data[name])
    if (
        bank_meta.get("artifact_type") not in {
            "goal_maplet_plane_pnp_observation_bank_v1",
            "goal_maplet_plane_pnp_observation_bank_v2",
        }
        or bank_meta.get("visibility_atlas_content_sha256")
        != visibility_meta.get("content_sha256")
        or arrays_sha256(bank_arrays) != bank_meta.get("arrays_sha256")
        or len(observation_offsets) != len(visibility.view_names) + 1
    ):
        raise ValueError("plane observation bank contract differs")

    geometry_points = points
    geometry_meta = bank_meta
    geometry_covariance = (
        np.asarray(bank_arrays["world_point_covariance_m2"], np.float64)
        if "world_point_covariance_m2" in bank_arrays else None
    )
    geometry_purity = (
        np.asarray(bank_arrays["plane_pixel_purity"], np.float64)
        if "plane_pixel_purity" in bank_arrays else None
    )
    geometry_dispersion = (
        np.asarray(bank_arrays["plane_depth_dispersion_m"], np.float64)
        if "plane_depth_dispersion_m" in bank_arrays else None
    )
    if args.geometry_observation_bank is not None:
        with np.load(args.geometry_observation_bank, allow_pickle=False) as data:
            geometry_meta = json.loads(str(data["metadata_json"].item()))
            geometry_offsets = np.asarray(data["observation_offsets"], np.int64)
            geometry_tokens = np.asarray(data["token_ids"])
            geometry_points = np.asarray(data["world_points"], np.float64)
            required_geometry = {
                "world_point_covariance_m2", "plane_pixel_purity", "plane_depth_dispersion_m",
            }
            if not required_geometry.issubset(data.files):
                raise ValueError("independent geometry bank lacks uncertainty arrays")
            geometry_covariance = np.asarray(data["world_point_covariance_m2"], np.float64)
            geometry_purity = np.asarray(data["plane_pixel_purity"], np.float64)
            geometry_dispersion = np.asarray(data["plane_depth_dispersion_m"], np.float64)
        if (
            geometry_meta.get("artifact_type") != "goal_maplet_plane_pnp_observation_bank_v2"
            or geometry_meta.get("plane_specific_geometry") is not True
            or not np.array_equal(geometry_offsets, observation_offsets)
            or not np.array_equal(geometry_tokens, bank_arrays["token_ids"])
            or geometry_points.shape != points.shape
            or geometry_meta.get("visibility_atlas_content_sha256")
            != visibility_meta.get("content_sha256")
        ):
            raise ValueError("independent plane geometry bank is not row-aligned")
    if geometry_covariance is None or geometry_purity is None or geometry_dispersion is None:
        raise ValueError("V8 atlas construction requires a plane-specific v2 geometry bank")
    projection_meta = None
    if args.radio_projection is not None:
        projection_weight, projection_meta = load_chart_local_radio_projection(args.radio_projection)
        if projection_meta.get("observation_bank_content_sha256") != bank_meta.get("content_sha256"):
            raise ValueError("chart-local projection was not trained on the supplied observation bank")
        features = project_chart_local_radio(features, projection_weight)

    unique_views = {name: row for row, name in enumerate(sorted(set(visibility.view_names.astype(str))))}
    view_id = np.asarray([unique_views[str(name)] for name in visibility.view_names], np.int64)
    texel_offsets = [0]
    uv_rows: list[np.ndarray] = []
    point_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    view_rows: list[np.ndarray] = []
    token_rows: list[np.ndarray] = []
    identity_rows: list[np.ndarray] = []
    rank_rows: list[np.ndarray] = []
    direction_rows: list[np.ndarray] = []
    range_rows: list[np.ndarray] = []
    height_rows: list[np.ndarray] = []
    height_std_rows: list[np.ndarray] = []
    applied_height_rows: list[np.ndarray] = []
    height_valid_rows: list[np.ndarray] = []
    covariance_rows: list[np.ndarray] = []
    purity_rows: list[np.ndarray] = []
    dispersion_rows: list[np.ndarray] = []
    next_identity = 0
    for plane in range(len(planes.plane_ids)):
        token_uv, token_geometry_uv, token_feature = [], [], []
        token_view, token_direction, token_range, token_height = [], [], [], []
        token_covariance, token_purity, token_dispersion = [], [], []
        for observation in range(
            int(visibility.plane_offsets[plane]), int(visibility.plane_offsets[plane + 1])
        ):
            lo, hi = map(int, observation_offsets[observation : observation + 2])
            if hi == lo:
                continue
            token_uv.append((points[lo:hi] - planes.centers_world[plane]) @ planes.frames_world[plane, :2].T)
            token_geometry_uv.append(
                (geometry_points[lo:hi] - planes.centers_world[plane])
                @ planes.frames_world[plane, :2].T
            )
            token_feature.append(features[lo:hi])
            token_view.append(np.full(hi - lo, view_id[observation], np.int64))
            delta = visibility.centers_world[observation] - geometry_points[lo:hi]
            distance = np.linalg.norm(delta, axis=1)
            token_direction.append(delta / np.maximum(distance[:, None], 1e-12))
            token_range.append(distance)
            token_height.append(
                (geometry_points[lo:hi] - planes.centers_world[plane]) @ planes.normals_world[plane]
            )
            if geometry_covariance is not None:
                token_covariance.append(geometry_covariance[lo:hi])
                token_purity.append(geometry_purity[lo:hi])
                token_dispersion.append(geometry_dispersion[lo:hi])
        if token_uv:
            if int(args.maximum_prototypes_per_texel) == 1:
                (
                    uv, descriptor, support, token_count, direction, observation_range,
                    surface_height, surface_height_std,
                    prototype_covariance, prototype_purity, prototype_dispersion,
                ) = _fuse_plane_texels(
                    np.concatenate(token_uv), np.concatenate(token_feature),
                    np.concatenate(token_view), cell_size_m=float(args.cell_size_m),
                    minimum_views=int(args.minimum_views),
                    view_directions_world=np.concatenate(token_direction),
                    observation_ranges_m=np.concatenate(token_range),
                    surface_height_m=np.concatenate(token_height),
                    geometry_uv_m=np.concatenate(token_geometry_uv),
                    geometry_covariance_m2=np.concatenate(token_covariance),
                    plane_pixel_purity=np.concatenate(token_purity),
                    plane_depth_dispersion_m=np.concatenate(token_dispersion),
                )
                identity = np.arange(len(uv), dtype=np.int64)
                prototype_rank = np.zeros(len(uv), np.uint8)
            else:
                (
                    uv, descriptor, identity, prototype_rank, support, token_count,
                    direction, observation_range, surface_height, surface_height_std,
                    prototype_covariance, prototype_purity, prototype_dispersion,
                ) = (
                    _fuse_plane_texel_prototypes(
                        np.concatenate(token_uv), np.concatenate(token_feature),
                        np.concatenate(token_view), cell_size_m=float(args.cell_size_m),
                        minimum_views=int(args.minimum_views),
                        maximum_prototypes=int(args.maximum_prototypes_per_texel),
                        view_directions_world=np.concatenate(token_direction),
                        observation_ranges_m=np.concatenate(token_range),
                        surface_height_m=np.concatenate(token_height),
                        geometry_uv_m=np.concatenate(token_geometry_uv),
                        geometry_covariance_m2=np.concatenate(token_covariance),
                        plane_pixel_purity=np.concatenate(token_purity),
                        plane_depth_dispersion_m=np.concatenate(token_dispersion),
                    )
                )
        else:
            uv = np.zeros((0, 2), np.float64)
            descriptor = np.zeros((0, features.shape[1]), np.float32)
            support = np.zeros(0, np.uint16)
            token_count = np.zeros(0, np.uint32)
            identity = np.zeros(0, np.int64)
            prototype_rank = np.zeros(0, np.uint8)
            direction = np.zeros((0, 3), np.float64)
            observation_range = np.zeros(0, np.float64)
            surface_height = np.zeros(0, np.float64)
            surface_height_std = np.zeros(0, np.float64)
            prototype_covariance = np.zeros((0, 3, 3), np.float64)
            prototype_purity = np.zeros(0, np.float64)
            prototype_dispersion = np.zeros(0, np.float64)
        height_valid = np.abs(surface_height) <= surface_height_limit_m
        applied_height = (
            surface_height
            if args.surface_height_policy == "raw"
            else np.where(height_valid, surface_height, 0.0)
        )
        world = (
            planes.centers_world[plane]
            + uv[:, :1] * planes.frames_world[plane, 0]
            + uv[:, 1:] * planes.frames_world[plane, 1]
            + applied_height[:, None] * planes.normals_world[plane]
        )
        uv_rows.append(uv)
        point_rows.append(world)
        feature_rows.append(descriptor.astype(np.float16))
        view_rows.append(support)
        token_rows.append(token_count)
        identity_rows.append(identity + next_identity)
        rank_rows.append(prototype_rank)
        direction_rows.append(direction)
        range_rows.append(observation_range)
        height_rows.append(surface_height)
        height_std_rows.append(surface_height_std)
        applied_height_rows.append(applied_height)
        height_valid_rows.append(height_valid.astype(np.uint8))
        covariance_rows.append(prototype_covariance.astype(np.float32))
        purity_rows.append(prototype_purity.astype(np.float32))
        dispersion_rows.append(prototype_dispersion.astype(np.float32))
        next_identity += int(identity.max() + 1) if len(identity) else 0
        texel_offsets.append(texel_offsets[-1] + len(uv))

    arrays = {
        "plane_texel_offsets": np.asarray(texel_offsets, np.int64),
        "texel_uv_m": np.concatenate(uv_rows) if uv_rows else np.zeros((0, 2), np.float64),
        "world_points": np.concatenate(point_rows) if point_rows else np.zeros((0, 3), np.float64),
        "radio_features": np.concatenate(feature_rows) if feature_rows else np.zeros((0, features.shape[1]), np.float16),
        "view_support": np.concatenate(view_rows) if view_rows else np.zeros(0, np.uint16),
        "token_support": np.concatenate(token_rows) if token_rows else np.zeros(0, np.uint32),
        "texel_identity": np.concatenate(identity_rows) if identity_rows else np.zeros(0, np.int64),
        "prototype_rank": np.concatenate(rank_rows) if rank_rows else np.zeros(0, np.uint8),
        "prototype_view_direction_world": np.concatenate(direction_rows) if direction_rows else np.zeros((0, 3), np.float64),
        "prototype_observation_range_m": np.concatenate(range_rows) if range_rows else np.zeros(0, np.float64),
        "prototype_surface_height_m": np.concatenate(height_rows) if height_rows else np.zeros(0, np.float64),
        "prototype_surface_height_std_m": np.concatenate(height_std_rows) if height_std_rows else np.zeros(0, np.float64),
        "prototype_surface_height_applied_m": np.concatenate(applied_height_rows) if applied_height_rows else np.zeros(0, np.float64),
        "prototype_surface_height_valid": np.concatenate(height_valid_rows) if height_valid_rows else np.zeros(0, np.uint8),
        "prototype_world_covariance_m2": np.concatenate(covariance_rows) if covariance_rows else np.zeros((0, 3, 3), np.float32),
        "prototype_plane_pixel_purity": np.concatenate(purity_rows) if purity_rows else np.zeros(0, np.float32),
        "prototype_plane_depth_dispersion_m": np.concatenate(dispersion_rows) if dispersion_rows else np.zeros(0, np.float32),
    }
    metadata = {
        "artifact_type": "goal_maplet_metric_plane_uv_radio_atlas_v8",
        "plane_count": int(len(planes.plane_ids)),
        "prototype_count": int(len(arrays["texel_uv_m"])),
        "texel_count": int(next_identity),
        "cell_size_m": float(args.cell_size_m),
        "minimum_independent_mapping_views": int(args.minimum_views),
        "maximum_anonymous_view_prototypes_per_texel": int(args.maximum_prototypes_per_texel),
        "fusion": (
            "token_mean_per_view_then_l2_normalize_then_view_balanced_mean"
            if int(args.maximum_prototypes_per_texel) == 1
            else "view_cell_descriptors_then_deterministic_anonymous_diverse_prototypes"
        ),
        "metric_coordinate": (
            "view_balanced_mean_observed_uv_with_cell_used_only_as_identity"
            if int(args.maximum_prototypes_per_texel) == 1
            else "anonymous_view_mode_specific_observed_uv_with_cell_used_only_as_identity"
        ),
        "source_view_identity_retained_at_runtime": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "anonymous_prototype_nuisance_coordinates": "surface_to_camera_unit_direction_world_and_metric_range",
        "world_geometry": "finite_plane_center_plus_metric_uv_axes_plus_anonymous_mode_surface_height",
        "local_surface_coordinate": "signed_normal_residual_mean_per_anonymous_view_texel_mode",
        "local_surface_uncertainty": "within_view_texel_signed_height_standard_deviation_m",
        "surface_height_application": str(args.surface_height_policy),
        "surface_height_limit_m": surface_height_limit_m,
        "plane_specific_observation_geometry": bool(
            geometry_meta.get("artifact_type") == "goal_maplet_plane_pnp_observation_bank_v2"
            and geometry_meta.get("plane_specific_geometry") is True
        ),
        "geometry_uncertainty_available_in_source_bank": bool(
            geometry_meta.get("artifact_type") == "goal_maplet_plane_pnp_observation_bank_v2"
        ),
        "geometry_uncertainty_propagation": "token_to_view_texel_mean_then_anonymous_mode_binding",
        "appearance_geometry_decoupled": bool(args.geometry_observation_bank is not None),
        "radio_descriptor_dimension": int(features.shape[1]),
        "chart_local_radio_projection_file_sha256": (
            None if args.radio_projection is None else file_sha256(args.radio_projection)
        ),
        "chart_local_radio_projection_content_sha256": (
            None if projection_meta is None else projection_meta.get("content_sha256")
        ),
        "appearance_texel_assignment": "observation_bank_metric_uv",
        "prototype_world_geometry": "geometry_observation_bank_same_row_plane_specific_metric_point",
        "uses_query_pose_depth_or_ground_truth": False,
        "planar_map_file_sha256": file_sha256(args.planar_map),
        "visibility_atlas_file_sha256": file_sha256(args.visibility_atlas),
        "visibility_atlas_content_sha256": visibility_meta.get("content_sha256"),
        "observation_bank_file_sha256": file_sha256(args.observation_bank),
        "observation_bank_content_sha256": bank_meta.get("content_sha256"),
        "geometry_observation_bank_file_sha256": (
            None if args.geometry_observation_bank is None else file_sha256(args.geometry_observation_bank)
        ),
        "geometry_observation_bank_content_sha256": geometry_meta.get("content_sha256"),
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True))
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
