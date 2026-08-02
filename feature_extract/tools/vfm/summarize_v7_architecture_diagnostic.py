"""Consolidate V7 coarse-pose and Stage-C capture-basin decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization_v7.pose_signature import file_sha256


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _stage_c_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    initial_translation = np.asarray(
        [row["initial_translation_m"] for row in rows], dtype=np.float64
    )
    final_translation = np.asarray(
        [row["final_translation_m"] for row in rows], dtype=np.float64
    )
    initial_rotation = np.asarray(
        [row["initial_rotation_deg"] for row in rows], dtype=np.float64
    )
    final_rotation = np.asarray(
        [row["final_rotation_deg"] for row in rows], dtype=np.float64
    )
    return {
        "query_count": len(rows),
        "initial_translation_m": float(np.median(initial_translation)),
        "initial_rotation_deg": float(np.median(initial_rotation)),
        "initial_pixel_flow_median_px": float(
            payload["summary"]["initial_pixel_flow_median_px"]
        ),
        "final_translation_median_m": float(np.median(final_translation)),
        "final_translation_p90_m": float(
            np.quantile(final_translation, 0.90)
        ),
        "final_rotation_median_deg": float(np.median(final_rotation)),
        "final_rotation_p90_deg": float(np.quantile(final_rotation, 0.90)),
        "translation_improved_count_1mm": int(
            np.sum(final_translation < initial_translation - 1e-3)
        ),
        "translation_worsened_count_1mm": int(
            np.sum(final_translation > initial_translation + 1e-3)
        ),
        "rotation_improved_count_001deg": int(
            np.sum(final_rotation < initial_rotation - 0.01)
        ),
        "rotation_worsened_count_001deg": int(
            np.sum(final_rotation > initial_rotation + 0.01)
        ),
        "artifact_sha256": file_sha256(path),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    directory = Path(args.directory)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite V7 architecture dashboard")
    retrieval_path = directory / "pose_aware_retrieval_strict12.json"
    regression_path = directory / "pose_signature_regression_strict12.json"
    retrieval = json.loads(retrieval_path.read_text())
    regression = json.loads(regression_path.read_text())
    basin_names = (
        "stagec_basin_t0_r025.json",
        "stagec_basin_t0_r05.json",
        "stagec_basin_t0_r1.json",
        "stagec_basin_t0_r2.json",
        "stagec_basin_t02_r0.json",
        "stagec_basin_t05_r0.json",
        "stagec_basin_t10_r0.json",
        "stagec_basin_t15_r0.json",
        "stagec_basin_t20_r0.json",
        "stagec_basin_t30_r0.json",
    )
    stage_c = {
        name[len("stagec_basin_") : -len(".json")]: (
            _stage_c_summary(directory / name)
        )
        for name in basin_names
    }
    report = {
        "artifact_type": "v7_architecture_decision_dashboard",
        "coarse_pose": {
            "pose_aware_retrieval": retrieval["summary"],
            "mapping_pose_regression": regression["strict_test"],
            "heldout_mapping_regression": {
                "trajectory": regression["validation_trajectory"],
                "ridge_selected": regression["ridge_validation"][
                    str(regression["ridge_selected_alpha"])
                ],
                "extra_trees_selected": regression[
                    "extra_trees_validation"
                ][str(regression["extra_trees_selected_min_samples_leaf"])],
            },
        },
        "stage_c_capture_basin": stage_c,
        "decisions": {
            "remove_global_stage_b_now": False,
            "reason_stage_b": (
                "Current Stage-A Top-64 sufficient statistics produce no "
                "30cm/3deg strict candidate under voting, local regression, "
                "geometry KDE, PCA-ridge, or ExtraTrees."
            ),
            "stage_c_runtime_qualified": False,
            "reason_stage_c": (
                "Oracle identities yield zero meaningful translation "
                "improvements from both 5cm and 20cm starts; accepted "
                "updates can add rotation and translation drift."
            ),
            "retain_two_peer_map_layers": False,
            "recommended_map_structure": (
                "one hierarchical Maplet: context-rich retrieval parent plus "
                "geometry-owned disconnected child surface tiles"
            ),
            "next_gate": (
                "Redesign Stage-A region topology/signature and require "
                "trajectory-held-out coarse pose candidate coverage before "
                "any further Stage-C optimization."
            ),
        },
        "protocol": {
            "strict_query_count": 12,
            "strict_query_trajectories": retrieval["query_trajectory_ids"],
            "mapping_trajectories": retrieval["mapping_trajectory_ids"],
            "strict_query_mapping_overlap": retrieval[
                "strict_query_mapping_overlap"
            ],
            "mapping_rgb_stored": False,
            "mapping_image_ids_stored": False,
            "mapping_image_paths_stored": False,
            "mapping_observation_descriptors_stored": False,
            "query_map_feature_interaction_after_stage_a": False,
            "point_correspondence_pnp_used": False,
        },
        "artifact_sha256": {
            "pose_aware_retrieval": file_sha256(retrieval_path),
            "pose_signature_regression": file_sha256(regression_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
