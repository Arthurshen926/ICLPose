"""Apply the frozen all-official-train OOF success calibrator to selected poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.postselection_success_calibration import (
    postselection_feature_row,
    predict_sigmoid_logistic_payload,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--selection_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite calibrated success report")
    calibrator_path = Path(args.calibrator)
    selection_path = Path(args.selection_report)
    calibrator = json.loads(calibrator_path.read_text())
    selection = json.loads(selection_path.read_text())
    if calibrator.get("artifact_type") != (
        "goal_maplet_postselection_success_calibration_oof_v1"
    ):
        raise ValueError("unsupported post-selection success calibrator")
    if str(selection.get("postselection_evidence_contract", "")) != (
        "proposal_to_pruning_to_exact_ranking_to_refinement_to_winner_v1"
    ):
        raise ValueError("selection report lacks complete post-selection evidence")
    rows = list(selection.get("rows", []))
    if not rows:
        raise ValueError("selection report has no poses")
    features = np.stack([postselection_feature_row(row) for row in rows])
    probability = {}
    acceptance = {}
    for name, head in calibrator["heads"].items():
        model = head["final_all_official_train_oof_model"]
        probability[name] = predict_sigmoid_logistic_payload(model, features)
        acceptance[name] = {
            operating_name: probability[name] >= float(
                operating["final_deployment_threshold"]["probability_threshold"]
            )
            for operating_name, operating in head.get(
                "selective_operating_points", {}
            ).items()
        }
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append({
            "image_id": str(row["image_id"]),
            "pose_w2c": row["pose_w2c"],
            "strict_success_probability": float(
                probability["strict_0.5m_5deg"][index]
            ),
            "loose_success_probability": float(
                probability["loose_1m_10deg"][index]
            ),
            "selective_acceptance": {
                head_name: {
                    operating_name: bool(decision[index])
                    for operating_name, decision in head_acceptance.items()
                }
                for head_name, head_acceptance in acceptance.items()
            },
            "typed_null_diagnostics": row.get("typed_null_diagnostics"),
        })
    payload = {
        "artifact_type": "goal_maplet_calibrated_selective_localization_v1",
        "output_semantics": "pose_success_probability_not_pose_posterior",
        "calibrator": str(calibrator_path),
        "calibrator_sha256": file_sha256(calibrator_path),
        "selection_report": str(selection_path),
        "selection_report_sha256": file_sha256(selection_path),
        "query_count": len(output_rows),
        "rows": output_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
