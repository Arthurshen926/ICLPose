import json

import numpy as np
import pytest

from feature_extract.tools.vfm.fit_v6_stage_c_score_calibration import (
    main as fit_score_calibration,
)
from feature_extract.tools.vfm.merge_v6_stage_c_reports import main
from feature_extract.vfm.localization_v6.stage_c_score_calibration import (
    FEATURE_NAMES,
    StageCScoreCalibration,
    candidate_rank_features,
)


def _report(indices, rows):
    return {
        "stage": "v6_stage_c_feature_atlas_replay",
        "image_id": "seq13/frame00001.png",
        "radio_atlas_sha256": "atlas",
        "region_chart_index": "index.npz",
        "frame_spatial_projection_checkpoint": "projection.pt",
        "source_report_sha256s": ["source"],
        "source_pool_mode": "complete",
        "pool_size_reconstructed": 4,
        "base_stride": 8,
        "rounds": 0,
        "render_charts": 12,
        "alike_detector_matchability": True,
        "query_gt_pose_component_diagnostic": False,
        "evaluated_pool_indices": indices,
        "stage_c_audit": {
            "broad_screen_input_count": len(indices),
            "broad_screen_selected_count": len(indices),
        },
        "rows": rows,
    }


def _row(index, score, translation):
    return {
        "replay_pool_index": index,
        "pose_mode_log_score": score,
        "atlas_score": score,
        "coarse_score": 0.0,
        "final_translation_m": translation,
        "final_rotation_deg": 1.0,
    }


def test_merge_stage_c_shards_ranks_without_gt(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "merged.json"
    first.write_text(json.dumps(_report([0, 2], [_row(0, 1.0, 0.1)])))
    # The more accurate row deliberately has the lower deployment score.
    second.write_text(json.dumps(_report([1, 3], [_row(1, 2.0, 1.0)])))
    main(
        [
            "--input_json",
            str(first),
            str(second),
            "--output_json",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())
    query = payload["query_results"][0]
    assert query["complete_pool_coverage"]
    assert query["rows"][0]["replay_pool_index"] == 1
    assert payload["summary"]["top1_translation_median_m"] == 1.0
    assert payload["summary"]["candidate_oracle_translation_median_m"] == 0.1


def test_candidate_rank_features_are_tie_aware_and_query_local():
    rows = [
        {name: value for name in FEATURE_NAMES}
        for value in (1.0, 3.0, 3.0)
    ]
    features = candidate_rank_features(rows)
    np.testing.assert_allclose(features[0], np.zeros(len(FEATURE_NAMES)))
    np.testing.assert_allclose(features[1], np.full(len(FEATURE_NAMES), 0.75))
    np.testing.assert_allclose(features[2], np.full(len(FEATURE_NAMES), 0.75))


def test_merge_can_apply_disjoint_stage_c_score_calibration(tmp_path):
    report_path = tmp_path / "report.json"
    output = tmp_path / "merged.json"
    calibration_path = tmp_path / "calibration.json"
    report = _report(
        [0, 1],
        [
            {
                **_row(0, 3.0, 1.0),
                "feature_atlas_score": 3.0,
                "final_fit_score": 1.0,
                "final_heldout_score": 1.0,
                "atlas_prerank_local_alignment_score": 1.0,
                "coarse_score": 1.0,
            },
            {
                **_row(1, 1.0, 0.1),
                "feature_atlas_score": 1.0,
                "final_fit_score": 3.0,
                "final_heldout_score": 3.0,
                "atlas_prerank_local_alignment_score": 3.0,
                "coarse_score": 3.0,
            },
        ],
    )
    report_path.write_text(json.dumps(report))
    StageCScoreCalibration(
        feature_names=FEATURE_NAMES,
        weights=np.asarray(
            [0.0, 1.0, *([0.0] * (len(FEATURE_NAMES) - 2))]
        ),
        metadata={
            "calibration_trajectory_ids": ["seq11"],
            "strict_holdout_trajectory_ids": ["seq13"],
            "radio_atlas_sha256": "atlas",
        },
    ).to_json(calibration_path)
    main(
        [
            "--input_json",
            str(report_path),
            "--output_json",
            str(output),
            "--score_calibration",
            str(calibration_path),
        ]
    )
    payload = json.loads(output.read_text())
    top = payload["query_results"][0]["rows"][0]
    assert top["replay_pool_index"] == 1
    assert top["calibrated_pose_score"] == 1.0


def _calibration_report(image_id, translation):
    rows = []
    for index, score in enumerate((1.0, 2.0)):
        row = _row(index, score, translation + 0.1 * index)
        row.update({name: score for name in FEATURE_NAMES})
        rows.append(row)
    report = _report([0, 1], rows)
    report.update(
        {
            "image_id": image_id,
            "deployable_result": False,
        }
    )
    return report


def test_score_calibration_rejects_one_training_trajectory(tmp_path):
    paths = []
    for frame in (1, 2):
        path = tmp_path / f"seq11_{frame}.json"
        path.write_text(
            json.dumps(
                _calibration_report(
                    f"seq11/frame{frame:05d}.png", 0.1
                )
            )
        )
        paths.append(path)
    with pytest.raises(ValueError, match="at least two calibration trajectories"):
        fit_score_calibration(
            [
                "--input_json",
                *(str(path) for path in paths),
                "--output_json",
                str(tmp_path / "calibration.json"),
                "--strict_holdout_trajectory_ids",
                "seq3",
            ]
        )


def test_score_calibration_rejects_all_negative_pools(tmp_path):
    paths = []
    for trajectory in ("seq9", "seq11"):
        path = tmp_path / f"{trajectory}.json"
        path.write_text(
            json.dumps(
                _calibration_report(
                    f"{trajectory}/frame00001.png", 1.0
                )
            )
        )
        paths.append(path)
    with pytest.raises(ValueError, match="too few 30 cm / 3 degree modes"):
        fit_score_calibration(
            [
                "--input_json",
                *(str(path) for path in paths),
                "--output_json",
                str(tmp_path / "calibration.json"),
                "--strict_holdout_trajectory_ids",
                "seq3",
            ]
        )
