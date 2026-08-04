"""Merge strict V8 shards into the requested A/B/C/D ablation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _pose(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    t = np.asarray([x["translation_m"] for x in values])
    r = np.asarray([x["rotation_deg"] for x in values])
    return {
        "translation_median_m": float(np.median(t)),
        "translation_p90_m": float(np.quantile(t, 0.9)),
        "rotation_median_deg": float(np.median(r)),
        "rotation_p90_deg": float(np.quantile(r, 0.9)),
        "within_1m_10deg": float(np.mean((t <= 1.0) & (r <= 10.0))),
        "within_30cm_3deg": float(np.mean((t <= 0.3) & (r <= 3.0))),
    }


def _retrieval(rows):
    count = np.asarray([row["labeled_region_count"] for row in rows], dtype=np.float64)
    result = {"labeled_region_count": int(np.sum(count))}
    for rank in (1, 5, 64):
        value = np.asarray([row[f"region_true_maplet_recall_at_{rank}"] for row in rows])
        result[f"region_true_maplet_recall_at_{rank}"] = float(np.sum(value * count) / np.sum(count))
    result["ground_truth_graph_rank_median"] = float(np.median([row["ground_truth_rank_among_proposals"] for row in rows]))
    result["ground_truth_graph_rank1_fraction"] = float(np.mean([row["ground_truth_rank_among_proposals"] == 1 for row in rows]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_shards", nargs="+", required=True)
    parser.add_argument("--student_shards", nargs="+", required=True)
    parser.add_argument("--student_training_summary", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    base = sum((json.loads(Path(x).read_text())["rows"] for x in args.base_shards), [])
    student = sum((json.loads(Path(x).read_text())["rows"] for x in args.student_shards), [])
    training = json.loads(Path(args.student_training_summary).read_text())
    report = {
        "stage": "v8_single_feature_structured_maplet_abcd",
        "strict_query_count": len(base),
        "A_current_single_feature_no_graph": {**_retrieval(base), **_pose(base, "baseline_top1")},
        "B_current_single_feature_graph": {**_retrieval(base), **_pose(base, "graph_refined_top1")},
        "C_multiteacher_single_student_no_graph": {**_retrieval(student), **_pose(student, "baseline_top1")},
        "D_multiteacher_single_student_graph": {**_retrieval(student), **_pose(student, "graph_refined_top1")},
        "trajectory_disjoint_student_validation": {
            "baseline": training["baseline_validation"],
            "student": training["student_validation"],
        },
        "storage_contract": {
            "map_feature_type_count": 1,
            "map_feature_dimension_all_variants": 128,
            "map_descriptor_component_count_all_variants": 2859,
            "physical_graph_stores_descriptors": False,
            "runtime_teacher_count": 0,
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids_or_paths": False,
            "uses_sfm_or_pnp": False,
        },
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
