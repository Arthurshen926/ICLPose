"""Post-label evaluation of a frozen direct-plane PnP selection artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


THRESHOLDS = ((0.1, 1.0), (0.25, 2.0), (0.5, 5.0), (1.0, 10.0), (2.0, 45.0))


def _summary(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    def good(row: dict[str, object], translation: float, rotation: float) -> bool:
        return bool(row["usable"]) and float(row["translation_error_m"]) <= translation and float(row["rotation_error_deg"]) <= rotation
    accepted = [row for row in rows if bool(row["usable"]) and float(row["selected_inlier_ratio"]) >= threshold]
    usable = [row for row in rows if bool(row["usable"])]
    branch_counts = {
        str(branch): int(sum(int(row["selected_branch"]) == branch for row in rows))
        for branch in sorted({int(row["selected_branch"]) for row in rows})
    }
    return {
        "query_count": int(len(rows)),
        "usable_count": int(len(usable)),
        "accepted_count": int(len(accepted)),
        "accepted_fraction": float(len(accepted) / len(rows)) if rows else 0.0,
        "raw_recall_2m45": float(sum(good(row, 2.0, 45.0) for row in rows) / len(rows)) if rows else 0.0,
        "raw_recall_1m10": float(sum(good(row, 1.0, 10.0) for row in rows) / len(rows)) if rows else 0.0,
        "accepted_precision_2m45": float(sum(good(row, 2.0, 45.0) for row in accepted) / len(accepted)) if accepted else 0.0,
        "accepted_precision_1m10": float(sum(good(row, 1.0, 10.0) for row in accepted) / len(accepted)) if accepted else 0.0,
        "selective_system_recall_2m45": float(sum(good(row, 2.0, 45.0) for row in accepted) / len(rows)) if rows else 0.0,
        "selective_system_recall_1m10": float(sum(good(row, 1.0, 10.0) for row in accepted) / len(rows)) if rows else 0.0,
        "rejected_bad_2m45_count": int(sum(not good(row, 2.0, 45.0) for row in rows if row not in accepted)),
        "rejected_good_2m45_count": int(sum(good(row, 2.0, 45.0) for row in rows if row not in accepted)),
        "accepted_bad_2m45_count": int(sum(not good(row, 2.0, 45.0) for row in accepted)),
        "threshold_hit_counts": {
            f"{translation:g}m_{rotation:g}deg": int(sum(
                good(row, translation, rotation) for row in rows
            ))
            for translation, rotation in THRESHOLDS
        },
        "selected_branch_counts": branch_counts,
        "selected_top10_count": int(sum(int(row["selected_branch"]) == 10 for row in rows)),
        "selected_h3_balanced_count": int(sum(int(row["selected_branch"]) == 30 for row in rows)),
        "selected_sift_count": int(sum(int(row["selected_branch"]) == 40 for row in rows)),
        "median_translation_m_among_usable": float(np.median([row["translation_error_m"] for row in usable])) if usable else None,
        "median_rotation_deg_among_usable": float(np.median([row["rotation_error_deg"] for row in usable])) if usable else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected_pose_inventory", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--pilot_ranking", type=Path, nargs="*", default=[])
    parser.add_argument("--confidence_threshold", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite selected PnP evaluation")
    with np.load(args.selected_pose_inventory, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("artifact_type") in (
            "goal_maplet_direct_plane_pnp_top5_top10_inlier_selected_v1",
            "goal_maplet_direct_plane_pnp_top5_top10_moge3_normal_selected_v1",
            "goal_maplet_dual_surface_geometry_consensus_selected_v1",
            "goal_maplet_dual_surface_probabilistic_fusion_selected_v1",
            "goal_maplet_cross_atlas_geometry_consensus_selected_v1",
            "goal_maplet_uncertainty_normalized_plane_pose_selected_v1",
            "goal_maplet_null_aware_marginalized_plane_pose_selected_v2",
        ):
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                *(
                    ("top5_normal_within_20deg", "top10_normal_within_20deg")
                    if metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_top5_top10_moge3_normal_selected_v1"
                    else (
                        (
                            ("cross_geometry_token_marginal_likelihood",)
                            if metadata.get("artifact_type")
                            == "goal_maplet_dual_surface_probabilistic_fusion_selected_v1"
                            else (
                                ("cross_atlas_geometry_inlier_ratio", "cross_atlas_geometry_inlier_count")
                                if metadata.get("artifact_type")
                                == "goal_maplet_cross_atlas_geometry_consensus_selected_v1"
                                else (
                                    ("cross_candidate_uncertainty_normalized_likelihood",)
                                    if metadata.get("artifact_type") in (
                                        "goal_maplet_uncertainty_normalized_plane_pose_selected_v1",
                                        "goal_maplet_null_aware_marginalized_plane_pose_selected_v2",
                                    )
                                    else ("cross_geometry_inlier_ratio", "cross_geometry_reprojection_median_px")
                                )
                            )
                        )
                        if metadata.get("artifact_type") in (
                            "goal_maplet_dual_surface_geometry_consensus_selected_v1",
                            "goal_maplet_dual_surface_probabilistic_fusion_selected_v1",
                            "goal_maplet_cross_atlas_geometry_consensus_selected_v1",
                            "goal_maplet_uncertainty_normalized_plane_pose_selected_v1",
                            "goal_maplet_null_aware_marginalized_plane_pose_selected_v2",
                        )
                        else ("top5_inlier_ratio", "top10_inlier_ratio")
                    )
                ),
            )
        elif metadata.get("artifact_type") == "goal_maplet_probabilistic_surface_pose_refinement_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "probabilistic_refinement_accepted", "unique_hypothesis_count",
                "mean_null_responsibility",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_top5_top10_h3_balanced_selected_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_source",
                "selected_unique_token_inlier_ratio", "top5_inlier_ratio",
                "top10_inlier_ratio", "h3_balanced_inlier_ratio",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_radio_sift_support_hybrid_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "baseline_inlier_ratio", "sift_inlier_count", "sift_candidate_count",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_radio_sift_global_context_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "baseline_inlier_ratio", "sift_inlier_count", "sift_candidate_count",
                "radio_global_context_score", "sift_global_context_score",
                "radio_eligible_mapping_view_count", "sift_eligible_mapping_view_count",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_radio_sift_spatial_context_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "baseline_inlier_ratio", "sift_inlier_count", "sift_candidate_count",
                "radio_spatial_inlier_count", "sift_spatial_inlier_count",
                "radio_spatial_mean_cosine", "sift_spatial_mean_cosine",
                "radio_nearby_mapping_view_count", "sift_nearby_mapping_view_count",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_map_density_inlier_selected_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "sparse_inlier_ratio", "dense_inlier_ratio",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_top5_top10_pose_consensus_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "top5_inlier_ratio", "top10_inlier_ratio",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_pose_medoid_fallback_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "override_distance_from_primary",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_damped_union_closure_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "closure_inlier_ratio", "closure_normalized_pose_distance",
                "closure_pnp_inlier_count",
            )
        elif metadata.get("artifact_type") in (
            "goal_maplet_direct_plane_pnp_reliability_damped_refinement_v1",
            "goal_maplet_direct_plane_pnp_reliability_tiered_damped_refinement_v2",
            "goal_maplet_direct_plane_pnp_plane_and_depth_reliability_tiered_damped_refinement_v3",
            "goal_maplet_direct_plane_pnp_spatial_plane_reliability_tiered_damped_refinement_v4",
            "goal_maplet_direct_plane_pnp_query_depth_edge_reliability_tiered_damped_refinement_v8",
        ):
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "reliability_candidate_pose_distance",
                "reliability_candidate_pnp_inlier_count",
                "reliability_weight_minimum", "reliability_weight_maximum",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_reliability_cascade_v5":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "cascade_selected_spatial", "spatial_candidate_pose_distance",
                "spatial_candidate_pnp_inlier_count",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_multiscale_reliability_cascade_v6":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "cascade_selected_spatial", "spatial_candidate_pose_distance",
                "spatial_candidate_pnp_inlier_count", "selected_spatial_inventory_index",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_multiscale_reliability_cascade_v7":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "cascade_selected_spatial", "spatial_candidate_pose_distance",
                "spatial_candidate_pnp_inlier_count", "selected_spatial_inventory_index",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_radio_sift_agreement_midpoint_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "sift_candidate_available", "sift_candidate_inlier_count",
                "radio_sift_agreement_translation_m", "radio_sift_agreement_rotation_deg",
                "radio_sift_midpoint_selected",
            )
        elif metadata.get("artifact_type") == "goal_maplet_direct_plane_pnp_hloc_low_support_fallback_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "hloc_pose_w2c", "hloc_fallback_selected",
            )
        elif metadata.get("artifact_type") == "goal_maplet_hloc_plane_agreement_midpoint_v1":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_confidence",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "hloc_pose_w2c", "plane_pose_w2c", "plane_usable",
                "hloc_plane_agreement_translation_m", "hloc_plane_agreement_rotation_deg",
                "hloc_plane_midpoint_selected",
            )
        elif metadata.get("artifact_type") == "goal_maplet_hloc_plane_tiered_agreement_v2":
            keys = (
                "names", "pose_w2c", "usable", "selected_branch", "selected_confidence",
                "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
                "hloc_pose_w2c", "plane_pose_w2c", "plane_usable",
                "hloc_plane_agreement_translation_m", "hloc_plane_agreement_rotation_deg",
                "hloc_plane_interpolation_fraction",
            )
        else:
            raise ValueError("selected PnP artifact type differs")
        arrays = {key: np.asarray(data[key]) for key in keys}
    raw_arrays_hash = arrays_sha256(arrays)
    if "selected_source" in arrays:
        arrays["selected_branch"] = arrays.pop("selected_source")
        arrays["selected_inlier_ratio"] = arrays.pop("selected_unique_token_inlier_ratio")
    if "selected_confidence" in arrays:
        arrays["selected_inlier_ratio"] = arrays.pop("selected_confidence")
    count = len(arrays["names"])
    if (
        metadata.get("query_pose_or_ground_truth_read", metadata.get("query_pose_or_ground_truth_opened")) is not False
        or raw_arrays_hash != metadata.get("arrays_sha256")
        or arrays["pose_w2c"].shape != (count, 4, 4)
    ):
        raise ValueError("selected PnP inventory differs")
    rows = []
    for index, name in enumerate(arrays["names"].astype(str).tolist()):
        usable = bool(arrays["usable"][index])
        row: dict[str, object] = {
            "name": name,
            "route": name.split("__", 1)[0],
            "usable": usable,
            "selected_branch": int(arrays["selected_branch"][index]),
            "selected_inlier_ratio": float(arrays["selected_inlier_ratio"][index]),
        }
        if usable:
            pose = np.asarray(arrays["pose_w2c"][index], np.float64)
            with np.load(args.query_contributors / name, allow_pickle=False) as data:
                gt = np.asarray(data["pose_w2c"], np.float64)
            center = -pose[:3, :3].T @ pose[:3, 3]
            gt_center = -gt[:3, :3].T @ gt[:3, 3]
            row["translation_error_m"] = float(np.linalg.norm(center - gt_center))
            row["rotation_error_deg"] = float(
                Rotation.from_matrix(pose[:3, :3] @ gt[:3, :3].T).magnitude()
                * 180.0 / np.pi
            )
        rows.append(row)
    pilot_names: set[str] = set()
    for path in args.pilot_ranking:
        pilot = json.loads(path.read_text())
        pilot_names.update(str(row["image"]) for row in pilot["rows"])
    complement = [row for row in rows if row["name"] not in pilot_names]
    report = {
        "artifact_type": "goal_maplet_selected_direct_plane_pnp_postlabel_evaluation_v1",
        "selected_pose_inventory_file_sha256": file_sha256(args.selected_pose_inventory),
        "selected_pose_inventory_content_sha256": metadata.get("content_sha256"),
        "selected_pose_inventory_artifact_type": metadata.get("artifact_type"),
        "selection_frozen_before_pose_labels_opened": True,
        "strict_runtime_phase_separation_eligible": bool(
            metadata.get("strict_runtime_phase_separation_eligible", False)
        ),
        "selected_confidence_semantics": metadata.get(
            "selected_confidence_semantics", "planar_pnp_inlier_ratio"
        ),
        "selected_confidence_calibrated_pose_quality": bool(
            metadata.get("selected_confidence_semantics") is None
        ),
        "confidence_threshold": float(args.confidence_threshold),
        "pilot_ranking_file_sha256_in_order": [file_sha256(path) for path in args.pilot_ranking],
        "pilot_query_count": int(len(pilot_names)),
        "pilot_excluded_from_complement": True,
        "all_query_summary": _summary(rows, float(args.confidence_threshold)),
        "complement_summary": _summary(complement, float(args.confidence_threshold)),
        "complement_route_summaries": {
            route: _summary([row for row in complement if row["route"] == route], float(args.confidence_threshold))
            for route in sorted(set(row["route"] for row in complement))
        },
        "evaluation_role": "locked-after-baseline-complement historical validation; not pristine blind test",
        "production_eligible": False,
        "rows": rows,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
