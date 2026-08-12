import json

import numpy as np
import pytest

from feature_extract.tools.vfm.evaluate_goal_maplet_final_test_repeats import main
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


def _fixture(tmp_path):
    image_id = "seq3/frame00001.png"
    calibrator = tmp_path / "calibrator.json"
    calibrator.write_text("{}")
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({
        "success_calibration": {"sha256": file_sha256(calibrator)},
    }))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "official_test": {
            "count": 1, "image_ids_sha256": ordered_id_sha256([image_id]),
        },
    }))
    selections, probabilities = [], []
    for repeat in range(3):
        selection = tmp_path / f"selection{repeat}.json"
        selection.write_text(json.dumps({
            "artifact_type": "goal_maplet_frozen_policy_deployment_report_v1",
            "selection_uses_query_ground_truth": False,
            "frozen_configuration": str(frozen),
            "frozen_configuration_sha256": file_sha256(frozen),
            "rows": [{
                "image_id": image_id,
                "pose_w2c": np.eye(4).tolist(),
                "final_score": 0.5,
                "final_translation_m": 0.1,
                "final_rotation_deg": 1.0,
                "selected_union_candidate_index": 0,
            }],
        }))
        probability = tmp_path / f"probability{repeat}.json"
        probability.write_text(json.dumps({
            "artifact_type": "goal_maplet_calibrated_selective_localization_v1",
            "selection_report_sha256": file_sha256(selection),
            "calibrator_sha256": file_sha256(calibrator),
            "query_count": 1,
            "rows": [{
                "image_id": image_id,
                "strict_success_probability": 0.9,
                "loose_success_probability": 0.95,
                "selective_acceptance": {
                    "strict_0.5m_5deg": {"target": True},
                    "loose_1m_10deg": {"target": True},
                },
            }],
        }))
        selections.append(selection)
        probabilities.append(probability)
    return protocol, selections, probabilities


def test_final_test_repeat_evaluator_binds_one_frozen_lineage(tmp_path):
    protocol, selections, probabilities = _fixture(tmp_path)
    output = tmp_path / "output.json"
    main([
        "--protocol", str(protocol),
        "--selection_reports", *(str(value) for value in selections),
        "--calibrated_reports", *(str(value) for value in probabilities),
        "--output_json", str(output),
    ])
    result = json.loads(output.read_text())
    assert result["primary_metrics_repeat0"]["strict_success"]["count"] == 1
    assert result["numeric_stability"]["winner_index_consistency_rate"] == 1.0


def test_final_test_repeat_evaluator_rejects_mixed_selection_lineage(tmp_path):
    protocol, selections, probabilities = _fixture(tmp_path)
    payload = json.loads(probabilities[1].read_text())
    payload["selection_report_sha256"] = "wrong"
    probabilities[1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="wrong selection"):
        main([
            "--protocol", str(protocol),
            "--selection_reports", *(str(value) for value in selections),
            "--calibrated_reports", *(str(value) for value in probabilities),
            "--output_json", str(tmp_path / "output.json"),
        ])
