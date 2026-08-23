"""Combine immutable full-token pattern-search shards and recompute closed gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


SCHEMA = "goal_maplet_fulltoken_surface_pattern_search_summary_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(value) for value in args.input]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite pattern-search summary")
    reports = [json.loads(path.read_text()) for path in paths]
    required_equal = (
        "artifact_type", "dataset_file_sha256", "physical_map_sha256",
        "canonical_field_sha256", "surface_mapper_file_sha256",
        "energy_semantics", "score_semantics", "search_semantics",
        "initialization_semantics",
    )
    for key in required_equal:
        if len({json.dumps(report.get(key), sort_keys=True) for report in reports}) != 1:
            raise ValueError(f"pattern-search shards disagree on {key}")
    rows = [row for report in reports for row in report["rows"]]
    image_ids = [str(row["image_id"]) for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("pattern-search shards contain duplicate queries")
    translation = np.asarray([row["final_translation_m"] for row in rows], dtype=np.float64)
    rotation = np.asarray([row["final_rotation_deg"] for row in rows], dtype=np.float64)
    initial_translation = np.asarray(
        [row["initial_translation_m"] for row in rows], dtype=np.float64
    )
    initial_rotation = np.asarray(
        [row["initial_rotation_deg"] for row in rows], dtype=np.float64
    )
    initial_joint = np.asarray([
        max(float(row["initial_translation_m"]) / 1.0, float(row["initial_rotation_deg"]) / 10.0)
        for row in rows
    ])
    final_joint = np.maximum(translation / 1.0, rotation / 10.0)
    strict = (translation <= 0.5 + 1.0e-6) & (rotation <= 5.0 + 1.0e-5)
    loose = (translation <= 1.0 + 1.0e-6) & (rotation <= 10.0 + 1.0e-5)
    initial_strict = (
        (initial_translation <= 0.5 + 1.0e-6)
        & (initial_rotation <= 5.0 + 1.0e-5)
    )
    initial_loose = (
        (initial_translation <= 1.0 + 1.0e-6)
        & (initial_rotation <= 10.0 + 1.0e-5)
    )
    initial_tier = initial_strict.astype(np.int8) * 2 + (
        (~initial_strict) & initial_loose
    ).astype(np.int8)
    final_tier = strict.astype(np.int8) * 2 + ((~strict) & loose).astype(np.int8)
    summary = {
        "artifact_type": SCHEMA,
        "source_reports": [
            {"path": str(path.resolve()), "file_sha256": file_sha256(path)} for path in paths
        ],
        **{key: reports[0].get(key) for key in required_equal[1:]},
        "query_count": len(rows),
        "image_ids": image_ids,
        "strict_capture_rate_closed_threshold": float(np.mean(strict)),
        "loose_capture_rate_closed_threshold": float(np.mean(loose)),
        "initial_strict_capture_rate_closed_threshold": float(np.mean(initial_strict)),
        "initial_loose_capture_rate_closed_threshold": float(np.mean(initial_loose)),
        "threshold_tier_improvement_rate": float(np.mean(final_tier > initial_tier)),
        "threshold_tier_degradation_rate": float(np.mean(final_tier < initial_tier)),
        "objective_drift_rate": float(np.mean(final_joint > initial_joint + 1.0e-9)),
        "median_final_translation_m": float(np.median(translation)),
        "median_final_rotation_deg": float(np.median(rotation)),
        "total_evaluated_pose_count": int(sum(row["evaluated_pose_count"] for row in rows)),
        "total_render_seconds": float(sum(row["render_seconds"] for row in rows)),
        "closed_threshold_tolerances": {"translation_m": 1.0e-6, "rotation_deg": 1.0e-5},
        "production_eligible": False,
        "claim": reports[0].get("claim"),
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "query_count", "strict_capture_rate_closed_threshold",
        "loose_capture_rate_closed_threshold", "objective_drift_rate",
        "initial_strict_capture_rate_closed_threshold",
        "initial_loose_capture_rate_closed_threshold",
        "threshold_tier_improvement_rate", "threshold_tier_degradation_rate",
        "median_final_translation_m", "median_final_rotation_deg",
        "total_render_seconds",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
