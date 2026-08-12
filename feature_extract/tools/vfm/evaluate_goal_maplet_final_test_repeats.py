"""Evaluate frozen official test output and three-run decision stability."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def _wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count == 0:
        return [float("nan"), float("nan")]
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    radius = z * sqrt(p * (1.0 - p) / count + z * z / (4.0 * count**2)) / denominator
    return [float(center - radius), float(center + radius)]


def _rate(successes: int, count: int) -> dict[str, object]:
    return {
        "count": int(successes), "total": int(count),
        "rate": float(successes / count) if count else float("nan"),
        "wilson95": _wilson(successes, count),
    }


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    translation = np.asarray([row["final_translation_m"] for row in rows], dtype=np.float64)
    rotation = np.asarray([row["final_rotation_deg"] for row in rows], dtype=np.float64)
    strict = (translation <= 0.5) & (rotation <= 5.0)
    loose = (translation <= 1.0) & (rotation <= 10.0)
    catastrophic = (translation > 2.0) | (rotation > 20.0)
    return {
        "query_count": len(rows),
        "strict_success": _rate(int(np.sum(strict)), len(rows)),
        "loose_success": _rate(int(np.sum(loose)), len(rows)),
        "catastrophic_failure": _rate(int(np.sum(catastrophic)), len(rows)),
        "translation_m": {
            "median": float(np.median(translation)),
            "p90": float(np.quantile(translation, 0.9)),
        },
        "rotation_deg": {
            "median": float(np.median(rotation)),
            "p90": float(np.quantile(rotation, 0.9)),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--selection_reports", nargs=3, required=True)
    parser.add_argument("--calibrated_reports", nargs=3, required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite final-test repeat evaluation")
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    expected = protocol["official_test"]
    selection_paths = [Path(value) for value in args.selection_reports]
    calibrated_paths = [Path(value) for value in args.calibrated_reports]
    selections = [json.loads(path.read_text()) for path in selection_paths]
    calibrated = [json.loads(path.read_text()) for path in calibrated_paths]
    selection_by_repeat = []
    calibrated_by_repeat = []
    frozen_hashes = set()
    calibrator_hashes = set()
    for index, (selection, probability) in enumerate(zip(selections, calibrated)):
        if (
            selection.get("artifact_type")
            != "goal_maplet_frozen_policy_deployment_report_v1"
            or not bool(selection.get("selection_uses_query_ground_truth") is False)
            or probability.get("artifact_type")
            != "goal_maplet_calibrated_selective_localization_v1"
        ):
            raise ValueError(f"repeat {index} violates frozen deployment semantics")
        if (
            str(probability.get("selection_report_sha256"))
            != file_sha256(selection_paths[index])
            or int(probability.get("query_count", -1)) != int(expected["count"])
        ):
            raise ValueError(f"repeat {index} calibrated the wrong selection report")
        frozen_path = Path(str(selection["frozen_configuration"]))
        frozen_sha = str(selection["frozen_configuration_sha256"])
        if not frozen_path.is_file() or file_sha256(frozen_path) != frozen_sha:
            raise ValueError(f"repeat {index} frozen configuration lineage changed")
        frozen_hashes.add(frozen_sha)
        calibrator_hashes.add(str(probability.get("calibrator_sha256")))
        selection_rows = {str(row["image_id"]): row for row in selection["rows"]}
        probability_rows = {str(row["image_id"]): row for row in probability["rows"]}
        if (
            len(selection_rows) != int(expected["count"])
            or set(selection_rows) != set(probability_rows)
            or ordered_id_sha256(selection_rows) != str(expected["image_ids_sha256"])
        ):
            raise ValueError(f"repeat {index} does not cover official test exactly")
        selection_by_repeat.append(selection_rows)
        calibrated_by_repeat.append(probability_rows)
    if len(frozen_hashes) != 1 or len(calibrator_hashes) != 1:
        raise ValueError("official-test repeats do not share one frozen configuration")
    frozen = json.loads(Path(str(selections[0]["frozen_configuration"])).read_text())
    if next(iter(calibrator_hashes)) != str(
        frozen.get("success_calibration", {}).get("sha256")
    ):
        raise ValueError("official-test calibrator differs from frozen configuration")
    image_ids = sorted(selection_by_repeat[0])
    pose_translation_delta = []
    pose_rotation_delta = []
    score_ranges = []
    strict_probability_ranges = []
    loose_probability_ranges = []
    winner_consistent = []
    acceptance_consistency: dict[str, list[bool]] = {}
    for image_id in image_ids:
        rows = [value[image_id] for value in selection_by_repeat]
        base_pose = np.asarray(rows[0]["pose_w2c"], dtype=np.float64)
        deviations = [
            pnp_pose_error(np.asarray(row["pose_w2c"], dtype=np.float64), base_pose)
            for row in rows[1:]
        ]
        pose_translation_delta.append(max([0.0] + [value.translation_m for value in deviations]))
        pose_rotation_delta.append(max([0.0] + [value.rotation_deg for value in deviations]))
        score_ranges.append(max(row["final_score"] for row in rows) - min(row["final_score"] for row in rows))
        winner_consistent.append(len({int(row["selected_union_candidate_index"]) for row in rows}) == 1)
        probabilities = [value[image_id] for value in calibrated_by_repeat]
        strict_probability_ranges.append(
            max(row["strict_success_probability"] for row in probabilities)
            - min(row["strict_success_probability"] for row in probabilities)
        )
        loose_probability_ranges.append(
            max(row["loose_success_probability"] for row in probabilities)
            - min(row["loose_success_probability"] for row in probabilities)
        )
        first = probabilities[0]["selective_acceptance"]
        for head, operating_points in first.items():
            for operating in operating_points:
                key = f"{head}/{operating}"
                decisions = [
                    bool(row["selective_acceptance"][head][operating])
                    for row in probabilities
                ]
                acceptance_consistency.setdefault(key, []).append(len(set(decisions)) == 1)

    first_rows = [selection_by_repeat[0][image_id] for image_id in image_ids]
    first_probability = calibrated_by_repeat[0]
    selective_risk = {}
    for head, operating_points in next(iter(first_probability.values()))["selective_acceptance"].items():
        success_field = (
            lambda row: float(row["final_translation_m"]) <= 0.5
            and float(row["final_rotation_deg"]) <= 5.0
        ) if head == "strict_0.5m_5deg" else (
            lambda row: float(row["final_translation_m"]) <= 1.0
            and float(row["final_rotation_deg"]) <= 10.0
        )
        selective_risk[head] = {}
        for operating in operating_points:
            accepted = [
                row for row in first_rows
                if bool(first_probability[str(row["image_id"])]["selective_acceptance"][head][operating])
            ]
            successes = sum(bool(success_field(row)) for row in accepted)
            selective_risk[head][operating] = {
                "coverage": _rate(len(accepted), len(first_rows)),
                "accepted_success": _rate(successes, len(accepted)),
            }
    payload = {
        "artifact_type": "goal_maplet_frozen_official_test_three_repeat_evaluation_v1",
        "protocol": str(protocol_path), "protocol_sha256": file_sha256(protocol_path),
        "official_test_evaluated_after_freeze": True,
        "repeat_count": 3, "query_count": len(image_ids),
        "primary_metrics_repeat0": _metrics(first_rows),
        "per_repeat_metrics": [
            _metrics([rows[image_id] for image_id in image_ids])
            for rows in selection_by_repeat
        ],
        "selective_risk_coverage_repeat0": selective_risk,
        "numeric_stability": {
            "winner_index_consistency_rate": float(np.mean(winner_consistent)),
            "acceptance_consistency_rate": {
                key: float(np.mean(values)) for key, values in acceptance_consistency.items()
            },
            "maximum_pose_translation_delta_m": float(max(pose_translation_delta)),
            "maximum_pose_rotation_delta_deg": float(max(pose_rotation_delta)),
            "maximum_final_score_range": float(max(score_ranges)),
            "maximum_strict_probability_range": float(max(strict_probability_ranges)),
            "maximum_loose_probability_range": float(max(loose_probability_ranges)),
        },
        "selection_reports": [
            {"path": str(path), "sha256": file_sha256(path)} for path in selection_paths
        ],
        "calibrated_reports": [
            {"path": str(path), "sha256": file_sha256(path)} for path in calibrated_paths
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"selection_reports", "calibrated_reports"}}, indent=2))


if __name__ == "__main__":
    main()
