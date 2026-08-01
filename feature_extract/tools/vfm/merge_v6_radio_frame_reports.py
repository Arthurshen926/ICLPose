"""Merge disjoint strict-query shards from evaluate_v6_radio_frame_source."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from feature_extract.tools.vfm.evaluate_v6_maplet_frame_alignment import (
    _aggregate_chart_expansion,
    _aggregate_m1,
    _aggregate_pose_rows,
    _aggregate_retrieval,
)
from feature_extract.tools.vfm.evaluate_v6_radio_frame_source import (
    _aggregate_projective_gap,
    _aggregate_stage_c,
)


def _canonical_configuration(
    source: dict[str, object], candidate_policy: str
) -> dict[str, object]:
    """Normalize metadata-only policy labels across in-flight shards."""

    configuration = copy.deepcopy(source)
    if str(candidate_policy) == "preserve":
        return configuration
    if str(candidate_policy) != "pair_union4x4_v1":
        raise ValueError("unknown candidate policy")
    grouped = dict(configuration.get("grouped_frame_mode_search", {}))
    combinations = dict(grouped.get("combinations_per_chart_subset", {}))
    combinations["two_charts"] = (
        "one_all_pairs_plus_up_to_seven_complements_from_geometry_"
        "top4_union_likelihood_top4_within_finite_budget"
    )
    grouped["combinations_per_chart_subset"] = combinations
    configuration["grouped_frame_mode_search"] = grouped
    configuration["stage_c_raw_pose_family"] = (
        "leading_pair_supports_up_to_eight_modes_then_broad_"
        "pair_first_regional_supports_up_to_four_modes"
    )
    configuration["stage_c_pose_pool_family_allocation"] = (
        "raw_deep64_then_broad4mode_union_consensus32_"
        "factorized32_default_pool256"
    )
    configuration["candidate_policy_metadata_normalized_at_merge"] = True
    detector_enabled = bool(
        configuration.get("alike_detector_matchability", False)
    )
    configuration.setdefault(
        "alike_detector_global_frame_weighting", detector_enabled
    )
    configuration.setdefault(
        "alike_detector_global_proposal_only",
        bool(configuration["alike_detector_global_frame_weighting"]),
    )
    configuration.setdefault(
        "alike_detector_local_cell_reliability", detector_enabled
    )
    configuration.setdefault("alike_detector_offset_prior", False)
    if not bool(configuration.get("run_stage_c", False)):
        # These values are inactive in source-only shards and changed while
        # Stage C was being repaired. They are not a semantic shard mismatch.
        configuration["stage_c_prerank_pool"] = 256
    return configuration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--candidate_policy",
        choices=("preserve", "pair_union4x4_v1"),
        default="preserve",
        help=(
            "Explicitly normalize a metadata-only policy label when shards "
            "were launched while that label was being corrected."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("output exists; pass --force to replace it")
    reports = [
        json.loads(Path(value).read_text(encoding="utf-8"))
        for value in args.inputs
    ]
    if not reports:
        raise ValueError("no shard reports")
    reference = reports[0]
    for report in reports[1:]:
        if report["artifact_sha256"] != reference["artifact_sha256"]:
            raise ValueError("shard artifacts differ")
        first = _canonical_configuration(
            dict(reference["configuration"]), args.candidate_policy
        )
        second = _canonical_configuration(
            dict(report["configuration"]), args.candidate_policy
        )
        first.pop("query_shard_index", None)
        second.pop("query_shard_index", None)
        if first != second:
            raise ValueError("shard configurations differ")
    queries = sorted(
        [
            row
            for report in reports
            for row in report.get("queries", [])
        ],
        key=lambda value: str(value["image_id"]),
    )
    image_ids = [str(value["image_id"]) for value in queries]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("query shards overlap")
    m1 = [row for report in reports for row in report.get("m1_rows", [])]
    m1_refined = [
        row
        for report in reports
        for row in report.get("m1_refined_rows", [])
    ]
    composite = [
        row
        for report in reports
        for row in report.get("m1_composite_rows", [])
    ]
    composite_refined = [
        row
        for report in reports
        for row in report.get("m1_composite_refined_rows", [])
    ]
    retrieval = [
        row
        for report in reports
        for row in report.get("retrieval_rows", [])
    ]
    merged = dict(reference)
    merged["stage"] = "v6_radio_frame_source_merged_strict_query_shards"
    merged["query_count"] = len(queries)
    merged["queries"] = queries
    merged["m1_rows"] = m1
    merged["m1_refined_rows"] = m1_refined
    merged["m1_composite_rows"] = composite
    merged["m1_composite_refined_rows"] = composite_refined
    merged["retrieval_rows"] = retrieval
    merged["m1_correct_chart_global_correlation"] = _aggregate_m1(m1)
    merged["m1_correct_chart_local_refinement"] = _aggregate_m1(
        m1_refined
    )
    merged["m1_correct_composite_region_global_correlation"] = (
        _aggregate_m1(composite)
    )
    merged["m1_correct_composite_region_local_refinement"] = (
        _aggregate_m1(composite_refined)
    )
    merged["scene_evidence"] = (
        {"topq_nms": _aggregate_retrieval(retrieval)}
        if retrieval
        else None
    )
    for output_key, query_key in (
        (
            "m3_correct_metric_charts_predicted_frame_pose",
            "m3_correct_chart_pose_errors",
        ),
        (
            "m3_correct_metric_charts_raw_frame_pose",
            "m3_correct_chart_raw_pose_errors",
        ),
        (
            "m3_retrieved_metric_charts_predicted_frame_pose",
            "m3_retrieved_chart_pose_errors",
        ),
        (
            "m3_retrieved_se3_consensus_mode_pose",
            "m3_retrieved_se3_consensus_mode_pose_errors",
        ),
        (
            "m3_correct_composite_region_predicted_frame_pose",
            "m3_correct_region_pose_errors",
        ),
        (
            "d2_actual_retrieval_gt_frame_pose",
            "d2_retrieved_charts_gt_frame_pose_errors",
        ),
        (
            "d3_correct_identity_predicted_frame_oracle_mode_pose",
            "d3_correct_identity_oracle_mode_pose_errors",
        ),
        (
            "d4_actual_retrieval_predicted_frame_oracle_mode_pose",
            "d4_retrieved_predicted_frame_oracle_mode_pose_errors",
        ),
    ):
        merged[output_key] = _aggregate_pose_rows(queries, query_key)
    merged["d1_region_to_chart_coverage"] = _aggregate_chart_expansion(
        queries
    )
    merged["d6_gt_affine_vs_homography"] = _aggregate_projective_gap(m1)
    if bool(reference["configuration"].get("run_stage_c", False)):
        merged["stage_c_runtime_atlas_alignment"] = _aggregate_stage_c(
            queries
        )
    configuration = _canonical_configuration(
        dict(reference["configuration"]), args.candidate_policy
    )
    configuration["query_shard_index"] = "merged"
    configuration["merged_shard_count"] = len(reports)
    merged["configuration"] = configuration
    protocol = dict(reference["protocol"])
    protocol["strict_test"] = sorted(
        {
            value
            for report in reports
            for value in report["protocol"].get("strict_test", [])
        }
    )
    merged["protocol"] = protocol
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(merged, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "query_count": len(queries),
                "d1": merged["d1_region_to_chart_coverage"],
                "d2": merged["d2_actual_retrieval_gt_frame_pose"],
                "d3": merged[
                    "d3_correct_identity_predicted_frame_oracle_mode_pose"
                ],
                "d4": merged[
                    "d4_actual_retrieval_predicted_frame_oracle_mode_pose"
                ],
                "m3": merged[
                    "m3_retrieved_metric_charts_predicted_frame_pose"
                ],
                "stage_c": merged.get(
                    "stage_c_runtime_atlas_alignment"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
