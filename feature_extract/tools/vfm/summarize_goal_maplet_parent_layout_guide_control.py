"""Summarize the fixed first-four-query parent-layout historical control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


SCHEMA = "goal_maplet_parent_layout_first4_historical_control_summary_v1"
METRICS = ("region_2m_45deg", "loose_1m_10deg", "strict_0_5m_5deg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.artifact_dir).resolve()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite parent-layout control summary")
    files = []
    by_route: dict[str, object] = {}
    for route in ("seq12", "seq14"):
        scores, evaluations = [], []
        for query in range(4):
            stem = f"{route}_q{query}_parent_layout_unsigned_parent32_top512"
            score_npz = root / f"{stem}_v2.npz"
            score_json = root / f"{stem}_v2.json"
            evaluation_path = root / f"{stem}_phase2_v2.json"
            score_metadata = json.loads(score_json.read_text())
            evaluation = json.loads(evaluation_path.read_text())
            if (
                score_metadata["image_id"] != evaluation["image_id"]
                or file_sha256(score_npz) != evaluation["score_file_sha256"]
                or score_metadata["content_sha256"] != evaluation["score_content_sha256"]
            ):
                raise ValueError("parent-layout score/evaluation lineage differs")
            scores.append(score_metadata)
            evaluations.append(evaluation)
            files.extend([
                {
                    "role": "phase1_score",
                    "path": str(score_npz),
                    "file_sha256": file_sha256(score_npz),
                    "content_sha256": score_metadata["content_sha256"],
                },
                {
                    "role": "phase2_coverage",
                    "path": str(evaluation_path),
                    "file_sha256": file_sha256(evaluation_path),
                    "content_sha256": evaluation["content_sha256"],
                },
            ])
        rows = {}
        for budget in (1, 4, 8, 16, 32):
            selected = [
                next(
                    row for row in value["prefix_rows"]
                    if int(row["effective_prefix_budget"]) == budget
                )
                for value in evaluations
            ]
            rows[str(budget)] = {
                "compression_from_154880_factor_pairs": float(154880 / budget),
                "retained_hits": {
                    metric: int(sum(row["joint_support"][metric] for row in selected))
                    for metric in METRICS
                },
                "raw_achievable_hits": {
                    metric: int(sum(
                        value["raw_implicit_factor_support"][metric]
                        for value in evaluations
                    ))
                    for metric in METRICS
                },
                "distinct_position_factor_count_per_query": [
                    int(row["distinct_position_factor_count"]) for row in selected
                ],
                "distinct_orientation_factor_count_per_query": [
                    int(row["distinct_orientation_factor_count"]) for row in selected
                ],
                "mean_distinct_position_factor_count": float(np.mean([
                    row["distinct_position_factor_count"] for row in selected
                ])),
                "mean_distinct_orientation_factor_count": float(np.mean([
                    row["distinct_orientation_factor_count"] for row in selected
                ])),
            }
        elapsed = [float(value["elapsed_seconds"]) for value in scores]
        by_route[route] = {
            "query_indices": [0, 1, 2, 3],
            "query_selection": "fixed_first_four_query_prefix_not_label_selected",
            "elapsed_seconds": {
                "values": elapsed,
                "mean": float(np.mean(elapsed)),
                "minimum": float(np.min(elapsed)),
                "maximum": float(np.max(elapsed)),
            },
            "retention_by_prefix": rows,
            "first_hit_ranks": [value["first_hit_rank"] for value in evaluations],
        }

    batch_a = root / "seq14_q0_parent_layout_unsigned_parent32_top512_v2.npz"
    batch_b = root / (
        "seq14_q0_parent_layout_unsigned_parent32_top512_batch127_stability_v2.npz"
    )
    with np.load(batch_a, allow_pickle=False) as left, np.load(
        batch_b, allow_pickle=False,
    ) as right:
        names = sorted(set(left.files) - {"metadata_json"})
        batch_exact = (
            set(left.files) == set(right.files)
            and all(np.array_equal(left[name], right[name]) for name in names)
        )
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "artifact_directory": str(root),
        "sample_count": 8,
        "queries_per_route": 4,
        "factor_pair_count_per_query": 154880,
        "maximum_query_parents": 32,
        "routes": by_route,
        "real_batch_stability": {
            "query": "seq14/frame00001.png",
            "candidate_batch_sizes": [512, 127],
            "all_nonmetadata_arrays_bit_exact": bool(batch_exact),
            "batch512_file_sha256": file_sha256(batch_a),
            "batch127_file_sha256": file_sha256(batch_b),
        },
        "files": files,
        "phase_separation": {
            "phase1_inputs": (
                "factorized_factors_query_radio_parent_posterior_physical_geometry_"
                "intrinsics_only"
            ),
            "phase1_reads_query_pose_or_gt": False,
            "phase1_uses_pnp_or_hard_point_correspondence": False,
            "phase2_opens_frozen_direct_gt_labels": True,
        },
        "protocol_status": (
            "historical_v3_retrieval_control_not_strict_v4_seq10_calibrated"
        ),
        "seq12_is_calibration_leaky_posthoc_upper_bound": True,
        "seq14_is_cross_route_historical_control": True,
        "sample_is_sufficient_for_promotion": False,
        "must_rerun_on_v4_strict_pool_before_promotion": True,
        "k32_control_conclusion": (
            "all_raw_achievable_events_in_this_fixed_8_query_control_survive_"
            "154880_to_32_factor_pair_compression_4840x"
        ),
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output.resolve()),
        "content_sha256": report["content_sha256"],
        "real_batch_stability": report["real_batch_stability"],
        "k32_control_conclusion": report["k32_control_conclusion"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

