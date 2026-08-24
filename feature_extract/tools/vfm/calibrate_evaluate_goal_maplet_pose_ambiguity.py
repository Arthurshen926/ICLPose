"""Calibrate on one route and evaluate a conservative Goal-Maplet null gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pose_ambiguity_calibration import (
    calibrate_zero_false_accept_margin,
    evaluate_ambiguity_gate,
)


def _content_sha(payload: dict[str, object]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration_report", required=True)
    parser.add_argument("--evaluation_report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    calibration_path = Path(args.calibration_report).resolve()
    evaluation_path = Path(args.evaluation_report).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite ambiguity evaluation")
    calibration = json.loads(calibration_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    shared = (
        "artifact_type", "dataset_content_sha256", "physical_map_sha256",
        "canonical_field_sha256", "surface_mapper_file_sha256",
        "live_score_semantics", "search_semantics", "maximum_sweeps",
        "translation_radius_m", "rotation_radius_deg",
        "minimum_translation_radius_m", "minimum_rotation_radius_deg",
    )
    if any(calibration.get(key) != evaluation.get(key) for key in shared):
        raise ValueError("calibration and evaluation search contracts differ")
    if (
        calibration.get("initialization_semantics") != "gt_offset_1m10_diagnostic"
        or evaluation.get("initialization_semantics") != "gt_offset_1m10_diagnostic"
        or calibration.get("include_route") == evaluation.get("include_route")
        or calibration.get("view_conditioned_field_content_sha256") is not None
        or evaluation.get("view_conditioned_field_content_sha256") is not None
    ):
        raise ValueError("ambiguity evaluation requires disjoint canonical diagnostic routes")
    threshold = calibrate_zero_false_accept_margin(calibration["rows"])
    calibration_metrics = evaluate_ambiguity_gate(calibration["rows"], threshold)
    evaluation_metrics = evaluate_ambiguity_gate(evaluation["rows"], threshold)
    payload: dict[str, object] = {
        "artifact_type": "goal_maplet_pose_ambiguity_calibration_v1",
        "calibration_report": str(calibration_path),
        "calibration_report_file_sha256": file_sha256(calibration_path),
        "evaluation_report": str(evaluation_path),
        "evaluation_report_file_sha256": file_sha256(evaluation_path),
        "calibration_route": calibration["include_route"],
        "evaluation_route": evaluation["include_route"],
        "score_margin_semantics": "top_minus_second_distinct_physical_basin_score_v1",
        "acceptance_semantics": "strictly_greater_than_validation_maximum_loose_error_margin_v1",
        "score_margin_threshold": threshold,
        "calibration": calibration_metrics,
        "evaluation": evaluation_metrics,
        "claim": "finite_route_selective_pose_diagnostic_not_statistical_certificate",
        "production_eligible": False,
    }
    payload["content_sha256"] = _content_sha(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "threshold": threshold,
        "calibration": {key: value for key, value in calibration_metrics.items() if key != "rows"},
        "evaluation": {key: value for key, value in evaluation_metrics.items() if key != "rows"},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
