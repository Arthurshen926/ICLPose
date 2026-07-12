from __future__ import annotations

from feature_extract.vfm.measurement_v1.rgb_patch_diagnostics import (
    _candidate_geometry_evidence_metrics,
)


def _row(
    token: int,
    identity: str,
    rank: int,
    *,
    correct: bool,
    dustbin_probability: float,
    view: int,
) -> dict[str, object]:
    return {
        "source_query_row": token,
        "candidate_identity_key": identity,
        "candidate_measurement_rank": rank,
        "target_geometry_correct_2px": correct,
        "target_geometry_correct_5px": correct,
        "dustbin_probability": dustbin_probability,
        "support_view_probability": 0.5,
        "support_view_rank": view,
    }


def test_candidate_rgb_evidence_reports_rescue_and_harm() -> None:
    rows = [
        _row(1, "wrong-a", 1, correct=False, dustbin_probability=0.9, view=0),
        _row(1, "wrong-a", 1, correct=False, dustbin_probability=0.8, view=1),
        _row(1, "right-a", 2, correct=True, dustbin_probability=0.1, view=0),
        _row(1, "right-a", 2, correct=True, dustbin_probability=0.2, view=1),
        _row(2, "right-b", 1, correct=True, dustbin_probability=0.1, view=0),
        _row(2, "wrong-b", 2, correct=False, dustbin_probability=0.8, view=0),
    ]
    report = _candidate_geometry_evidence_metrics(rows)
    assert report is not None
    metrics = report["geometry_correct_2px"]["validity_mean"]
    assert metrics["frozen_selected_correct_rate"] == 0.5
    assert metrics["rgb_selected_correct_rate"] == 1.0
    assert metrics["rescue_eligible_count"] == 1
    assert metrics["rescued_count"] == 1
    assert metrics["harmed_count"] == 0
    assert metrics["net_correct_change"] == 1
