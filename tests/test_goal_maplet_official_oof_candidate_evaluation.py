import json
from pathlib import Path

from feature_extract.tools.vfm.evaluate_goal_maplet_official_oof_candidates import (
    _candidate_summary,
    _stage_summary,
)


def _row(image_id: str, errors):
    modes = [
        {
            "rank": rank,
            "translation_m": translation,
            "rotation_deg": rotation,
        }
        for rank, (translation, rotation) in enumerate(errors, start=1)
    ]
    return {
        "image_id": image_id,
        "mode_details": {"actual_parent_actual_child": modes},
        "proposal_diagnostics": {
            "actual_parent_actual_child": {
                "topn_after_nms": {
                    "pose_count": len(modes),
                    "within_0_5m_5deg_count": sum(
                        translation <= 0.5 and rotation <= 5.0
                        for translation, rotation in errors
                    ),
                    "within_1m_10deg_count": sum(
                        translation <= 1.0 and rotation <= 10.0
                        for translation, rotation in errors
                    ),
                    "unique_pose_basin_count_0_5m_5deg": len(modes),
                    "unique_pose_basin_count_1m_10deg": len(modes),
                }
            }
        },
    }


def test_candidate_summary_counts_success_recall_and_empty_as_failure():
    rows = [
        _row("seq1/a.png", [(0.6, 6.0), (0.2, 2.0)]),
        _row("seq2/b.png", [(3.0, 1.0)]),
        _row("seq3/c.png", []),
    ]
    summary = _candidate_summary(rows)
    assert summary["query_count"] == 3
    assert summary["top1_strict_success"]["count"] == 0
    assert summary["top1_loose_success"]["count"] == 1
    assert summary["top1_catastrophic_failure"]["count"] == 2
    assert summary["topk_basin_recall"]["2"]["strict"]["count"] == 1
    assert summary["retained_basin_opportunity_capture"]["strict"][
        "available_correct_basin_count"
    ] == 1
    assert summary["retained_basin_opportunity_capture"]["strict"]["top1_capture"][
        "count"
    ] == 0


def test_stage_summary_reports_unique_basin_availability():
    rows = [_row("seq1/a.png", [(0.2, 2.0), (0.8, 8.0)])]
    summary = _stage_summary(rows, "topn_after_nms")
    assert summary["strict_basin_survival"]["count"] == 1
    assert summary["loose_basin_survival"]["count"] == 1
    assert summary["unique_pose_basin_count_0_5m_5deg"]["median"] == 2.0
