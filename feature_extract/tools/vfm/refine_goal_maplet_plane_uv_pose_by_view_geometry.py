"""Refine canonical plane-UV poses using anonymous prototype view geometry.

The runtime atlas contains RADIO modes plus their observed surface UV, viewing
direction and metric range, but no source image, source RGB or source-view ID.
Starting from a frozen canonical-atlas pose, this stage removes hypotheses that
would require the surface to be seen from the opposite hemisphere or at more
than a three-fold range change, then runs a second grouped PnP.  Selection is
entirely label-free and must preserve at least 90% of the original support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _load,
    _score,
    _solve,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


MIN_DIRECTION_COSINE = 0.0
MIN_RANGE_RATIO = 1.0 / 3.0
MAX_RANGE_RATIO = 3.0
MIN_GLOBAL_SUPPORT_FRACTION = 0.90


def _load_atlas(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    base_keys = (
        "plane_texel_offsets", "texel_uv_m", "world_points", "radio_features",
        "view_support", "token_support", "texel_identity", "prototype_rank",
        "prototype_view_direction_world", "prototype_observation_range_m",
    )
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        keys = base_keys
        if metadata.get("artifact_type") in (
            "goal_maplet_metric_plane_uv_radio_atlas_v5",
            "goal_maplet_metric_plane_uv_radio_atlas_v6",
            "goal_maplet_metric_plane_uv_radio_atlas_v7",
            "goal_maplet_metric_plane_uv_radio_atlas_v8",
        ):
            keys += ("prototype_surface_height_m", "prototype_surface_height_std_m")
        if metadata.get("artifact_type") in (
            "goal_maplet_metric_plane_uv_radio_atlas_v6",
            "goal_maplet_metric_plane_uv_radio_atlas_v7",
            "goal_maplet_metric_plane_uv_radio_atlas_v8",
        ):
            keys += ("prototype_surface_height_applied_m", "prototype_surface_height_valid")
        if metadata.get("artifact_type") == "goal_maplet_metric_plane_uv_radio_atlas_v8":
            keys += (
                "prototype_world_covariance_m2", "prototype_plane_pixel_purity",
                "prototype_plane_depth_dispersion_m",
            )
        arrays = {key: np.asarray(data[key]) for key in keys}
    if (
        metadata.get("artifact_type") not in (
            "goal_maplet_metric_plane_uv_radio_atlas_v4",
            "goal_maplet_metric_plane_uv_radio_atlas_v5",
            "goal_maplet_metric_plane_uv_radio_atlas_v6",
            "goal_maplet_metric_plane_uv_radio_atlas_v7",
            "goal_maplet_metric_plane_uv_radio_atlas_v8",
        )
        or metadata.get("source_view_identity_retained_at_runtime") is not False
        or metadata.get("source_rgb_stored_or_consumed_at_runtime") is not False
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
    ):
        raise ValueError("plane UV view-geometry atlas contract differs")
    return arrays, metadata


def _load_initial(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    keys = ("names", "raw_inliers_pose_w2c", "raw_inliers_inlier_count")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in keys}
        # Replay the complete upstream hash, not merely the arrays consumed here.
        complete = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    if (
        metadata.get("artifact_type") != "goal_maplet_direct_plane_pnp_grouped_multihypothesis_v1"
        or metadata.get("query_pose_or_ground_truth_opened") is not False
        or arrays_sha256(complete) != metadata.get("arrays_sha256")
    ):
        raise ValueError("initial canonical plane pose inventory differs")
    return arrays, metadata


def _compatible_rows(
    pose_w2c: np.ndarray,
    world: np.ndarray,
    prototype_rows: np.ndarray,
    atlas_direction: np.ndarray,
    atlas_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, np.float64)
    center = -pose[:3, :3].T @ pose[:3, 3]
    delta = center - np.asarray(world, np.float64)
    query_range = np.linalg.norm(delta, axis=1)
    query_direction = delta / np.maximum(query_range[:, None], 1e-12)
    rows = np.asarray(prototype_rows, np.int64)
    cosine = np.sum(query_direction * atlas_direction[rows], axis=1)
    ratio = query_range / np.maximum(atlas_range[rows], 1e-12)
    compatible = np.flatnonzero(
        np.isfinite(cosine) & np.isfinite(ratio)
        & (cosine >= MIN_DIRECTION_COSINE)
        & (ratio >= MIN_RANGE_RATIO) & (ratio <= MAX_RANGE_RATIO)
    )
    return compatible, cosine, ratio


def _key(score: dict[str, object]) -> tuple[object, ...]:
    return (
        int(score["plane_capped_support"]), int(score["region_capped_support"]),
        int(score["inlier_count"]), -float(score["reprojection_median_px"] or 1e9),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_correspondences", type=Path, required=True)
    parser.add_argument("--initial_candidates", type=Path, required=True)
    parser.add_argument("--plane_uv_atlas", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output_frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_frozen_pose_inventory.exists():
        raise FileExistsError("refusing to overwrite view-geometry refinement")
    correspondence, corr_meta = _load(args.frozen_correspondences)
    if corr_meta.get("artifact_type") not in (
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v2",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v3",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4",
    ):
        raise ValueError("view-geometry refinement requires correspondence v2, v3, or v4")
    initial, initial_meta = _load_initial(args.initial_candidates)
    atlas, atlas_meta = _load_atlas(args.plane_uv_atlas)
    names = correspondence["names"].astype(str)
    if not np.array_equal(names, initial["names"].astype(str)):
        raise ValueError("initial pose and correspondence inventories differ")
    if corr_meta.get("plane_uv_atlas_content_sha256") != atlas_meta.get("content_sha256"):
        raise ValueError("correspondences do not bind the supplied view-geometry atlas")
    pose_out = np.asarray(initial["raw_inliers_pose_w2c"], np.float64).copy()
    accepted = np.zeros(len(names), bool)
    compatible_count = np.zeros(len(names), np.int64)
    output_inlier = np.asarray(initial["raw_inliers_inlier_count"], np.int64).copy()
    diagnostics: list[dict[str, object]] = []
    for index, name in enumerate(names.tolist()):
        pose = pose_out[index]
        row: dict[str, object] = {"name": name, "initial_pose_finite": bool(np.all(np.isfinite(pose)))}
        if not row["initial_pose_finite"]:
            diagnostics.append(row); continue
        lo, hi = map(int, correspondence["correspondence_offsets"][index:index + 2])
        world = correspondence["world_points"][lo:hi]
        tokens = correspondence["query_tokens"][lo:hi]
        measurements = (
            correspondence["query_measurements_xy"][lo:hi]
            if "query_measurements_xy" in correspondence else None
        )
        provenance = correspondence["provenance_region_plane_atlas_row"][lo:hi]
        prototype = correspondence["prototype_atlas_row"][lo:hi]
        K = correspondence["camera_matrices"][index]
        k1 = float(correspondence["radial_k1"][index])
        keep, cosine, ratio = _compatible_rows(
            pose, world, prototype, atlas["prototype_view_direction_world"],
            atlas["prototype_observation_range_m"],
        )
        compatible_count[index] = len(keep)
        initial_global = _score(pose, world, tokens, provenance, K, k1, measurements)
        initial_compatible = (
            _score(
                pose, world[keep], tokens[keep], provenance[keep], K, k1,
                None if measurements is None else measurements[keep],
            )
            if len(keep) else None
        )
        candidate = _solve(world, tokens, K, k1, keep, measurements)
        if candidate is not None:
            candidate_global = _score(candidate, world, tokens, provenance, K, k1, measurements)
            candidate_compatible = _score(
                candidate, world[keep], tokens[keep], provenance[keep], K, k1,
                None if measurements is None else measurements[keep],
            )
            use = bool(
                int(candidate_global["inlier_count"])
                >= int(np.floor(MIN_GLOBAL_SUPPORT_FRACTION * int(initial_global["inlier_count"])))
                and initial_compatible is not None
                and _key(candidate_compatible) > _key(initial_compatible)
            )
            if use:
                pose_out[index] = candidate
                output_inlier[index] = int(candidate_global["inlier_count"])
                accepted[index] = True
        row.update({
            "compatible_correspondence_count": int(len(keep)),
            "direction_cosine_median": None if not len(cosine) else float(np.median(cosine)),
            "range_ratio_median": None if not len(ratio) else float(np.median(ratio)),
            "initial_global_inlier_count": int(initial_global["inlier_count"]),
            "refinement_accepted": bool(accepted[index]),
        })
        diagnostics.append(row)
    arrays = {
        "names": correspondence["names"], "pose_w2c": pose_out,
        "usable": np.all(np.isfinite(pose_out), axis=(1, 2)),
        "candidate_correspondence_count": np.diff(correspondence["correspondence_offsets"]),
        "pnp_inlier_count": output_inlier,
        "view_geometry_refinement_accepted": accepted,
        "compatible_correspondence_count": compatible_count,
    }
    metadata = {
        "artifact_type": "goal_maplet_canonical_plane_uv_view_geometry_refined_pose_v1",
        "arrays_sha256": arrays_sha256(arrays), "query_count": int(len(names)),
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "source_view_identity_retained_at_runtime": False,
        "anonymous_view_geometry_gate": {
            "minimum_direction_cosine": MIN_DIRECTION_COSINE,
            "minimum_range_ratio": MIN_RANGE_RATIO,
            "maximum_range_ratio": MAX_RANGE_RATIO,
            "minimum_global_support_fraction": MIN_GLOBAL_SUPPORT_FRACTION,
        },
        "frozen_correspondence_file_sha256": file_sha256(args.frozen_correspondences),
        "frozen_correspondence_content_sha256": corr_meta.get("content_sha256"),
        "initial_candidate_file_sha256": file_sha256(args.initial_candidates),
        "initial_candidate_content_sha256": initial_meta.get("content_sha256"),
        "plane_uv_atlas_file_sha256": file_sha256(args.plane_uv_atlas),
        "plane_uv_atlas_content_sha256": atlas_meta.get("content_sha256"),
        "accepted_count": int(np.sum(accepted)), "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output_frozen_pose_inventory.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_frozen_pose_inventory, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    # Phase 2 starts only after the pose inventory is sealed.
    rows = []
    for index, name in enumerate(names.tolist()):
        with np.load(args.query_contributors / name, allow_pickle=False) as data:
            gt = np.asarray(data["pose_w2c"], np.float64)
        pose = pose_out[index]
        if np.all(np.isfinite(pose)):
            center = -pose[:3, :3].T @ pose[:3, 3]
            gt_center = -gt[:3, :3].T @ gt[:3, 3]
            translation = float(np.linalg.norm(center - gt_center))
            rotation = float(Rotation.from_matrix(pose[:3, :3] @ gt[:3, :3].T).magnitude() * 180.0 / np.pi)
        else:
            translation = rotation = float("inf")
        rows.append({**diagnostics[index], "translation_error_m": translation, "rotation_error_deg": rotation})
    translation = np.asarray([row["translation_error_m"] for row in rows])
    rotation = np.asarray([row["rotation_error_deg"] for row in rows])
    finite = np.isfinite(translation) & np.isfinite(rotation)
    report = {
        "artifact_type": "goal_maplet_canonical_plane_uv_view_geometry_refinement_evaluation_v1",
        "pose_frozen_before_query_pose_or_ground_truth_open": True,
        "frozen_pose_inventory_file_sha256": file_sha256(args.output_frozen_pose_inventory),
        "frozen_pose_inventory_content_sha256": metadata["content_sha256"],
        "query_count": int(len(rows)), "accepted_count": int(np.sum(accepted)),
        "median_translation_m": float(np.median(translation[finite])),
        "median_rotation_deg": float(np.median(rotation[finite])),
        "p90_translation_m": float(np.quantile(translation[finite], 0.9)),
        "p90_rotation_deg": float(np.quantile(rotation[finite], 0.9)),
        "recall_2m45": float(np.mean(finite & (translation <= 2.0) & (rotation <= 45.0))),
        "recall_1m10": float(np.mean(finite & (translation <= 1.0) & (rotation <= 10.0))),
        "recall_0p5m5": float(np.mean(finite & (translation <= 0.5) & (rotation <= 5.0))),
        "recall_0p25m2": float(np.mean(finite & (translation <= 0.25) & (rotation <= 2.0))),
        "recall_0p1m1": float(np.mean(finite & (translation <= 0.1) & (rotation <= 1.0))),
        "rows": rows, "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
