"""Build one conservative correspondence field between point and surface coordinates.

The V5 point-coordinate and V7 continuous chart-coordinate inventories carry
the same anonymous RADIO matches.  A fixed mapping-development fraction moves
each image and map coordinate from the point estimate toward the surface
estimate *before* PnP.  The disagreement between the two coordinate models is
retained as moment-matched uncertainty instead of being hidden by interpolating
two final poses.

This is a phase-1 operation: query poses, labels, contributor geometry, and
source RGB are neither accepted as inputs nor opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _load as _load_correspondences,
)
from feature_extract.tools.vfm.refine_goal_maplet_plane_uv_pose_by_view_geometry import (
    _load_atlas,
)
from feature_extract.tools.vfm.select_goal_maplet_cross_coordinate_surface_pose import (
    FIXED_QUARTER_SURFACE_FRACTION,
    _paired_contract,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


OUTPUT_SCHEMA = "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7"
NUMERICAL_COVARIANCE_FLOOR_M2 = 1e-8


def _moment_matched_scalar_variance(
    first_mean: np.ndarray,
    second_mean: np.ndarray,
    first_variance: np.ndarray,
    second_variance: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Per-axis variance of a two-component isotropic coordinate mixture."""
    first = np.asarray(first_mean, np.float64).reshape(-1, 2)
    second = np.asarray(second_mean, np.float64).reshape(-1, 2)
    var_first = np.asarray(first_variance, np.float64).reshape(-1)
    var_second = np.asarray(second_variance, np.float64).reshape(-1)
    alpha = float(fraction)
    if not (first.shape == second.shape and len(first) == len(var_first) == len(var_second)):
        raise ValueError("coordinate mixture arrays differ")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("coordinate mixture fraction differs")
    # The stored scalar denotes equal variance on x/y.  The trace of the
    # between-model covariance divided by two is its isotropic projection.
    between = 0.5 * np.sum(np.square(second - first), axis=1)
    output = (
        (1.0 - alpha) * var_first
        + alpha * var_second
        + alpha * (1.0 - alpha) * between
    )
    if np.any(~np.isfinite(output)) or np.any(output <= 0.0):
        raise ValueError("coordinate mixture variance is invalid")
    return output


def _moment_matched_world_covariance(
    first_mean: np.ndarray,
    second_mean: np.ndarray,
    second_covariance: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """World covariance of point-mass V5 and uncertain surface V7 mixture."""
    first = np.asarray(first_mean, np.float64).reshape(-1, 3)
    second = np.asarray(second_mean, np.float64).reshape(-1, 3)
    covariance = np.asarray(second_covariance, np.float64).reshape(-1, 3, 3)
    alpha = float(fraction)
    if not (first.shape == second.shape and len(first) == len(covariance)):
        raise ValueError("world-coordinate mixture arrays differ")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("world-coordinate mixture fraction differs")
    delta = second - first
    output = (
        alpha * covariance
        + alpha * (1.0 - alpha) * np.einsum("ni,nj->nij", delta, delta)
    )
    output = 0.5 * (output + np.swapaxes(output, 1, 2))
    if np.any(~np.isfinite(output)):
        raise ValueError("world-coordinate mixture covariance is invalid")
    # V7 covariances are serialized as float32 tangent projectors and can
    # acquire ~1e-9 negative eigenvalues.  Project them to a fixed 0.1 mm
    # numerical floor before the next float32 serialization.
    eigenvalue, eigenvector = np.linalg.eigh(output)
    eigenvalue = np.maximum(eigenvalue, NUMERICAL_COVARIANCE_FLOOR_M2)
    output = np.einsum("nij,nj,nkj->nik", eigenvector, eigenvalue, eigenvector)
    return output


def _positive_part_surface_gain(
    point_world: np.ndarray,
    surface_world: np.ndarray,
    surface_covariance: np.ndarray,
) -> np.ndarray:
    """Remove one calibrated RMS noise radius from each predicted map offset.

    This is the positive-part radial soft threshold.  It has no route- or
    query-tuned constant: a two-dimensional isotropic posterior has RMS radius
    sqrt(2 variance), obtained here from half the tangent covariance trace.
    """
    first = np.asarray(point_world, np.float64).reshape(-1, 3)
    second = np.asarray(surface_world, np.float64).reshape(-1, 3)
    covariance = np.asarray(surface_covariance, np.float64).reshape(-1, 3, 3)
    if not (first.shape == second.shape and len(first) == len(covariance)):
        raise ValueError("surface gain arrays differ")
    radius = np.linalg.norm(second - first, axis=1)
    variance = np.maximum(0.5 * np.trace(covariance, axis1=1, axis2=2), 0.0)
    noise_radius = np.sqrt(2.0 * variance)
    gain = np.maximum(0.0, 1.0 - noise_radius / np.maximum(radius, 1e-12))
    gain[radius <= 1e-12] = 0.0
    if np.any(~np.isfinite(gain)) or np.any((gain < 0.0) | (gain > 1.0)):
        raise ValueError("surface gain is invalid")
    return gain


def _variable_moment_matched_world_covariance(
    first_mean: np.ndarray,
    second_mean: np.ndarray,
    second_covariance: np.ndarray,
    gain: np.ndarray,
) -> np.ndarray:
    first = np.asarray(first_mean, np.float64).reshape(-1, 3)
    second = np.asarray(second_mean, np.float64).reshape(-1, 3)
    covariance = np.asarray(second_covariance, np.float64).reshape(-1, 3, 3)
    alpha = np.asarray(gain, np.float64).reshape(-1)
    if not (first.shape == second.shape and len(first) == len(covariance) == len(alpha)):
        raise ValueError("variable covariance arrays differ")
    delta = second - first
    output = (
        alpha[:, None, None] * covariance
        + (alpha * (1.0 - alpha))[:, None, None]
        * np.einsum("ni,nj->nij", delta, delta)
    )
    output = 0.5 * (output + np.swapaxes(output, 1, 2))
    eigenvalue, eigenvector = np.linalg.eigh(output)
    eigenvalue = np.maximum(eigenvalue, NUMERICAL_COVARIANCE_FLOOR_M2)
    return np.einsum("nij,nj,nkj->nik", eigenvector, eigenvalue, eigenvector)


def _blend(
    point: dict[str, np.ndarray],
    surface: dict[str, np.ndarray],
    atlas: dict[str, np.ndarray],
    *,
    fraction: float,
    coordinate_blend_scope: str = "joint_query_and_map",
) -> dict[str, np.ndarray]:
    alpha = float(fraction)
    if coordinate_blend_scope not in {
        "joint_query_and_map", "map_only", "map_uncertainty_soft_threshold",
    }:
        raise ValueError("coordinate blend scope differs")
    rows = np.asarray(point["prototype_atlas_row"], np.int64)
    if np.any((rows < 0) | (rows >= len(atlas["world_points"]))):
        raise ValueError("prototype atlas row lies outside supplied atlas")
    base_world = np.asarray(atlas["world_points"][rows], np.float64)
    point_world = np.asarray(point["world_points"], np.float64)
    if not np.array_equal(point_world, base_world):
        raise ValueError("point coordinates are not the supplied atlas prototypes")
    point_pixel = np.asarray(point["query_measurements_xy"], np.float64)
    surface_pixel = np.asarray(surface["query_measurements_xy"], np.float64)
    surface_world = np.asarray(surface["world_points"], np.float64)
    base_uv = np.asarray(atlas["texel_uv_m"][rows], np.float64)
    surface_uv = np.asarray(surface["prototype_chart_uv_measurement_m"], np.float64)
    cell_lower = np.asarray(surface["prototype_chart_uv_cell_lower_m"], np.float64)
    cell_size = 0.5

    output = {key: np.asarray(value).copy() for key, value in surface.items()}
    adaptive_gain = (
        _positive_part_surface_gain(
            point_world, surface_world,
            surface["prototype_centroid_covariance_world_m2"],
        )
        if coordinate_blend_scope == "map_uncertainty_soft_threshold" else None
    )
    map_gain = (
        adaptive_gain if adaptive_gain is not None else np.full(len(point_world), alpha)
    )
    output["world_points"] = (
        point_world + map_gain[:, None] * (surface_world - point_world)
    ).astype(np.float64)
    if coordinate_blend_scope == "joint_query_and_map":
        output["query_measurements_xy"] = (
            (1.0 - alpha) * point_pixel + alpha * surface_pixel
        ).astype(np.float32)
        output["query_measurement_variance_px2"] = _moment_matched_scalar_variance(
            point_pixel,
            surface_pixel,
            point["query_measurement_variance_px2"],
            surface["query_measurement_variance_px2"],
            alpha,
        ).astype(np.float32)
        output["correspondence_match_probability"] = (
            (1.0 - alpha) * np.asarray(point["correspondence_match_probability"], np.float64)
            + alpha * np.asarray(surface["correspondence_match_probability"], np.float64)
        ).astype(np.float32)
    else:
        output["query_measurements_xy"] = point_pixel.astype(np.float32)
        output["query_measurement_variance_px2"] = np.asarray(
            point["query_measurement_variance_px2"], np.float32,
        ).copy()
        output["correspondence_match_probability"] = np.asarray(
            point["correspondence_match_probability"], np.float32,
        ).copy()
    output["prototype_centroid_covariance_world_m2"] = (
        _variable_moment_matched_world_covariance(
            point_world, surface_world,
            surface["prototype_centroid_covariance_world_m2"], adaptive_gain,
        )
        if adaptive_gain is not None else
        _moment_matched_world_covariance(
            point_world, surface_world,
            surface["prototype_centroid_covariance_world_m2"], alpha,
        )
    ).astype(np.float32)
    output["prototype_chart_uv_measurement_m"] = (
        base_uv + map_gain[:, None] * (surface_uv - base_uv)
    ).astype(np.float64)
    output["prototype_chart_uv_cell_lower_m"] = cell_lower.astype(np.float64)
    if not np.array_equal(np.floor(base_uv / cell_size) * cell_size, cell_lower):
        raise ValueError("point and surface chart coordinates do not share a metric cell")
    if np.any(output["prototype_chart_uv_measurement_m"] < cell_lower) or np.any(
        output["prototype_chart_uv_measurement_m"] >= cell_lower + cell_size
    ):
        raise ValueError("blended chart coordinate escaped its original metric cell")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point_correspondences", type=Path, required=True)
    parser.add_argument("--surface_correspondences", type=Path, required=True)
    parser.add_argument("--plane_uv_atlas", type=Path, required=True)
    parser.add_argument(
        "--coordinate_blend_scope",
        choices=(
            "joint_query_and_map", "map_only", "map_uncertainty_soft_threshold",
        ),
        default="joint_query_and_map",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite blended correspondence inventory")

    point, point_meta = _load_correspondences(args.point_correspondences)
    surface, surface_meta = _load_correspondences(args.surface_correspondences)
    atlas, atlas_meta = _load_atlas(args.plane_uv_atlas)
    _paired_contract(point, point_meta, surface, surface_meta)
    if not (
        point_meta.get("plane_uv_atlas_content_sha256") == atlas_meta.get("content_sha256")
        and surface_meta.get("plane_uv_atlas_content_sha256") == atlas_meta.get("content_sha256")
        and point_meta.get("plane_uv_atlas_file_sha256") == file_sha256(args.plane_uv_atlas)
        and surface_meta.get("plane_uv_atlas_file_sha256") == file_sha256(args.plane_uv_atlas)
    ):
        raise ValueError("coordinate inventories do not bind the supplied atlas bytes")

    fraction = FIXED_QUARTER_SURFACE_FRACTION
    arrays = _blend(
        point, surface, atlas, fraction=fraction,
        coordinate_blend_scope=args.coordinate_blend_scope,
    )
    cell_size = float(surface_meta["chart_uv_metric_cell_size_m"])
    if abs(cell_size - 0.5) > 1e-12:
        raise ValueError("fixed correspondence blend requires the frozen 0.5m chart cell")
    metadata = {
        key: value for key, value in surface_meta.items()
        if key not in {"arrays_sha256", "content_sha256"}
    }
    metadata.update({
        "artifact_type": OUTPUT_SCHEMA,
        "query_measurement_semantics": (
            "mapping_only_joint_query_subtoken_and_continuous_chartUV_surface_coordinate_mean"
        ),
        "query_measurement_uncertainty_semantics": (
            "predicted_query_centroid_variance_px2_plus_tangent_chartUV_centroid_covariance_world_m2_not_surface_footprint"
        ),
        "continuous_map_coordinate_semantics": (
            "fixed_quarter_moment_matched_pointV5_to_cell_bounded_surfaceV7_coordinate_mean"
        ),
        "cross_coordinate_blend_fraction": float(fraction),
        "cross_coordinate_blend_scope": str(args.coordinate_blend_scope),
        "cross_coordinate_blend_policy": (
            "mapping_dev_frozen_smallest_one_eighth_grid_step_improving_seq10_strict_and_0p25_without_coarse_loss"
        ),
        "cross_coordinate_uncertainty_policy": (
            "two_component_moment_match_including_between_coordinate_model_disagreement"
        ),
        "centroid_covariance_numerical_floor_m2": NUMERICAL_COVARIANCE_FLOOR_M2,
        "map_coordinate_adaptive_gain_semantics": (
            "positive_part_remove_one_calibrated_2D_RMS_noise_radius_no_tuned_threshold"
            if args.coordinate_blend_scope == "map_uncertainty_soft_threshold" else None
        ),
        "query_pose_or_ground_truth_read": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "point_correspondence_file_sha256": file_sha256(args.point_correspondences),
        "point_correspondence_content_sha256": point_meta.get("content_sha256"),
        "surface_correspondence_file_sha256": file_sha256(args.surface_correspondences),
        "surface_correspondence_content_sha256": surface_meta.get("content_sha256"),
        "plane_uv_atlas_file_sha256": file_sha256(args.plane_uv_atlas),
        "plane_uv_atlas_content_sha256": atlas_meta.get("content_sha256"),
        "arrays_sha256": arrays_sha256(arrays),
    })
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    # Reload through the production correspondence contract before reporting.
    replay_arrays, replay_meta = _load_correspondences(args.output)
    if replay_meta.get("content_sha256") != metadata["content_sha256"] or any(
        not np.array_equal(arrays[key], replay_arrays[key])
        for key in arrays
    ):
        raise AssertionError("blended correspondence artifact does not replay")
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "content_sha256": metadata["content_sha256"],
        "query_count": int(len(arrays["names"])),
        "correspondence_count": int(len(arrays["world_points"])),
        "surface_fraction": float(fraction),
        "coordinate_blend_scope": str(args.coordinate_blend_scope),
        "adaptive_nonzero_fraction": (
            float(np.mean(_positive_part_surface_gain(
                point["world_points"], surface["world_points"],
                surface["prototype_centroid_covariance_world_m2"],
            ) > 0.0))
            if args.coordinate_blend_scope == "map_uncertainty_soft_threshold" else None
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
