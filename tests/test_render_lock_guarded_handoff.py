from __future__ import annotations

import pytest

from feature_extract.tools.vfm.report_render_lock_guarded_handoff import summarize_guarded_handoff


def test_guarded_handoff_selects_candidate_only_inside_render_delta_band() -> None:
    baseline_rows = [
        {"query_id": "a", "translation_error_m": 0.30, "rotation_error_deg": 1.0},
        {"query_id": "b", "translation_error_m": 0.20, "rotation_error_deg": 1.0},
        {"query_id": "c", "translation_error_m": 0.24, "rotation_error_deg": 1.0},
    ]
    candidate_rows = [
        {
            "query_id": "a",
            "translation_error_m": 0.10,
            "rotation_error_deg": 1.0,
            "pnp_render_translation_delta_m": 0.25,
        },
        {
            "query_id": "b",
            "translation_error_m": 0.40,
            "rotation_error_deg": 1.0,
            "pnp_render_translation_delta_m": 0.25,
        },
        {
            "query_id": "c",
            "translation_error_m": 0.05,
            "rotation_error_deg": 1.0,
            "pnp_render_translation_delta_m": 0.50,
        },
    ]

    summary, rows = summarize_guarded_handoff(
        baseline_rows,
        candidate_rows,
        min_pnp_render_delta_m=0.20,
        max_pnp_render_delta_m=0.40,
    )

    assert [row["selected_source"] for row in rows] == ["candidate", "candidate", "baseline"]
    assert summary["query_count"] == 3
    assert summary["selected_candidate_count"] == 2
    assert summary["rescued_success_25cm_10deg_count"] == 1
    assert summary["broken_success_25cm_10deg_count"] == 1
    assert summary["success_25cm_10deg"] == pytest.approx(2 / 3)
    assert summary["median_translation_error_m"] == pytest.approx(0.24)
