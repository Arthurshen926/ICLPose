"""Fuse V11 and local-correlation coordinates with a parameter-free Pareto gate.

Both correspondence inventories contain the same frozen RADIO/chart hypotheses.
For each physical hypothesis, the V12 coordinate replaces V11 only when its
mapping-calibrated image variance and chart-UV variance do not increase and its
mapping-trained match sigmoid does not decrease. The sigmoid has no separate
probability calibration; comparing it across heads is only a diagnostic. There is no query
pose, query ground truth, learned selector, threshold, or candidate-count gain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import load_mapping_subtoken_head
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


COMMON_ARRAYS = (
    "names", "correspondence_offsets", "query_tokens",
    "provenance_region_plane_atlas_row", "camera_matrices", "radial_k1",
    "prototype_atlas_row", "query_plane_visible_fraction", "radio_match_score",
    "prototype_world_covariance_m2", "prototype_plane_pixel_purity",
    "prototype_plane_depth_dispersion_m", "prototype_chart_uv_cell_lower_m",
)
COORDINATE_ARRAYS = (
    "world_points", "query_measurements_xy", "query_measurement_variance_px2",
    "correspondence_match_probability", "prototype_centroid_covariance_world_m2",
    "prototype_chart_uv_measurement_m",
)


def _pareto_local_coordinate_mask(
    baseline: dict[str, np.ndarray], local: dict[str, np.ndarray],
) -> np.ndarray:
    baseline_image = np.asarray(baseline["query_measurement_variance_px2"], np.float64)
    local_image = np.asarray(local["query_measurement_variance_px2"], np.float64)
    baseline_uv = 0.5 * np.trace(
        np.asarray(baseline["prototype_centroid_covariance_world_m2"], np.float64),
        axis1=1, axis2=2,
    )
    local_uv = 0.5 * np.trace(
        np.asarray(local["prototype_centroid_covariance_world_m2"], np.float64),
        axis1=1, axis2=2,
    )
    baseline_match = np.asarray(baseline["correspondence_match_probability"], np.float64)
    local_match = np.asarray(local["correspondence_match_probability"], np.float64)
    if not (
        baseline_image.shape == local_image.shape == baseline_uv.shape == local_uv.shape
        == baseline_match.shape == local_match.shape
        and np.all(np.isfinite(np.c_[
            baseline_image, local_image, baseline_uv, local_uv, baseline_match, local_match,
        ]))
        and np.all(np.c_[baseline_image, local_image, baseline_uv, local_uv] > 0.0)
        and np.all((np.c_[baseline_match, local_match] >= 0.0)
                   & (np.c_[baseline_match, local_match] <= 1.0))
    ):
        raise ValueError("coordinate uncertainty inputs differ")
    weak = (
        (local_image <= baseline_image) & (local_uv <= baseline_uv)
        & (local_match >= baseline_match)
    )
    strict = (
        (local_image < baseline_image) | (local_uv < baseline_uv)
        | (local_match > baseline_match)
    )
    return weak & strict


def _paired_contract(
    baseline: dict[str, np.ndarray], baseline_meta: dict[str, object],
    local: dict[str, np.ndarray], local_meta: dict[str, object],
) -> None:
    if (
        baseline_meta.get("artifact_type")
        != "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7"
        or local_meta.get("artifact_type")
        != "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7"
    ):
        raise ValueError("coordinate fusion requires surface-coordinate inventories")
    for name in COMMON_ARRAYS:
        if name not in baseline or name not in local or not np.array_equal(baseline[name], local[name]):
            raise ValueError(f"coordinate fusion pairing differs: {name}")
    for name in COORDINATE_ARRAYS:
        if name not in baseline or name not in local or baseline[name].shape != local[name].shape:
            raise ValueError(f"coordinate fusion arrays differ: {name}")
    for name in (
        "plane_uv_atlas_content_sha256", "plane_ranking_file_sha256",
        "query_camera_only_inventory_content_sha256", "homography_threshold_m",
        "hypotheses_per_query_token", "topk_planes", "query_support_policy",
    ):
        if baseline_meta.get(name) != local_meta.get(name):
            raise ValueError(f"coordinate fusion metadata differs: {name}")


def _fusion_metadata(baseline_meta, local_meta):
    """Do not advertise a mixed inventory as calibrated by one source head."""
    prefixes = ("mapping_subtoken_", "mapping_chart_uv_", "mapping_surface_")
    metadata = {
        key: value for key, value in baseline_meta.items()
        if key != "content_sha256" and not key.startswith(prefixes)
    }
    metadata["coordinate_fusion_source_head_metadata"] = {
        label: {key: value for key, value in source.items() if key.startswith(prefixes)}
        for label, source in (("0_v11", baseline_meta), ("1_v12", local_meta))
    }
    metadata["coordinate_fusion_row_source_array"] = "coordinate_source_head_index"
    metadata["coordinate_fusion_match_probability_calibrated"] = False
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_correspondences", type=Path, required=True)
    parser.add_argument("--local_correspondences", type=Path, required=True)
    parser.add_argument("--baseline_head", type=Path, required=True)
    parser.add_argument("--local_head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite fused correspondence inventory")
    baseline, baseline_meta = _load(args.baseline_correspondences)
    local, local_meta = _load(args.local_correspondences)
    _paired_contract(baseline, baseline_meta, local, local_meta)
    _, baseline_head = load_mapping_subtoken_head(args.baseline_head)
    _, local_head = load_mapping_subtoken_head(args.local_head)
    if (
        baseline_head.get("artifact_type")
        != "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_homography_context_head_v11"
        or local_head.get("artifact_type")
        != "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_local_correlation_head_v12"
        or local_head.get("local_radio_v11_relative_gate_pass") is not True
        or baseline_meta.get("mapping_subtoken_head_content_sha256")
        != baseline_head.get("content_sha256")
        or local_meta.get("mapping_subtoken_head_content_sha256")
        != local_head.get("content_sha256")
        or local_head.get("local_radio_reference_head_content_sha256")
        != baseline_head.get("content_sha256")
    ):
        raise ValueError("coordinate fusion head lineage differs")
    selected = _pareto_local_coordinate_mask(baseline, local)
    arrays = {name: np.asarray(value).copy() for name, value in baseline.items()}
    for name in COORDINATE_ARRAYS:
        arrays[name][selected] = local[name][selected]
    arrays["coordinate_source_head_index"] = selected.astype(np.uint8)
    metadata = _fusion_metadata(baseline_meta, local_meta)
    metadata.update(
        arrays_sha256=arrays_sha256(arrays),
        coordinate_fusion_semantics=(
            "per_physical_hypothesis_use_local_V12_only_if_mapping_calibrated_"
            "image_variance_AND_chartUV_variance_nondecrease_forbidden_AND_"
            "uncalibrated_match_sigmoid_nondecrease;otherwise_anchor_to_V11"
        ),
        coordinate_fusion_candidate_count_change=0,
        coordinate_fusion_continuous_weights=0,
        coordinate_fusion_query_pose_or_ground_truth_read=False,
        coordinate_fusion_selected_count=int(np.sum(selected)),
        coordinate_fusion_total_count=int(len(selected)),
        coordinate_fusion_selected_fraction=float(np.mean(selected)),
        coordinate_fusion_baseline_correspondence_file_sha256=file_sha256(
            args.baseline_correspondences,
        ),
        coordinate_fusion_baseline_correspondence_content_sha256=baseline_meta.get(
            "content_sha256",
        ),
        coordinate_fusion_local_correspondence_file_sha256=file_sha256(
            args.local_correspondences,
        ),
        coordinate_fusion_local_correspondence_content_sha256=local_meta.get("content_sha256"),
        coordinate_fusion_baseline_head_file_sha256=file_sha256(args.baseline_head),
        coordinate_fusion_baseline_head_content_sha256=baseline_head.get("content_sha256"),
        coordinate_fusion_local_head_file_sha256=file_sha256(args.local_head),
        coordinate_fusion_local_head_content_sha256=local_head.get("content_sha256"),
        production_eligible=False,
    )
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({
        "artifact_type": metadata["artifact_type"],
        "selected_count": int(np.sum(selected)),
        "total_count": int(len(selected)),
        "selected_fraction": float(np.mean(selected)),
        "file_sha256": file_sha256(args.output),
        "content_sha256": metadata["content_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
