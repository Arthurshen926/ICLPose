from __future__ import annotations

from feature_extract.vfm.coarse_proposal_pool_diagnostics import compare_pool_summaries, summarize_pool_rows


def test_summarize_pool_rows_uses_gt_reprojection_error_fallback() -> None:
    rows = [
        {"query_id": "q0", "gt_reproj_error_px": "1.5"},
        {"query_id": "q0", "gt_reproj_error_px": "4.5"},
        {"query_id": "q1", "gt_reproj_error_px": "8.0"},
    ]

    summary = summarize_pool_rows(rows, thresholds_px=(2.0, 5.0))

    assert summary["row_count"] == 3
    assert summary["query_count"] == 2
    assert summary["thresholds"]["2px"]["valid_count"] == 1
    assert summary["thresholds"]["5px"]["valid_count"] == 2
    assert summary["thresholds"]["5px"]["queries_with_valid_count"] == 1
    assert summary["thresholds"]["5px"]["valid_per_query_p50"] == 1.0


def test_summarize_pool_rows_computes_error_from_query_and_target_pixels() -> None:
    rows = [
        {
            "query_id": "q0",
            "query_x": "10.0",
            "query_y": "10.0",
            "query_gt_x": "11.0",
            "query_gt_y": "10.0",
        },
        {
            "query_id": "q0",
            "query_center_x": "10.0",
            "query_center_y": "10.0",
            "query_gt_x": "13.0",
            "query_gt_y": "14.0",
        },
    ]

    summary = summarize_pool_rows(rows, thresholds_px=(2.0, 5.0))

    assert summary["error_source_counts"]["xy_to_gt"] == 2
    assert summary["thresholds"]["2px"]["valid_count"] == 1
    assert summary["thresholds"]["5px"]["valid_count"] == 2


def test_compare_pool_summaries_reports_growth_factors() -> None:
    baseline = summarize_pool_rows(
        [{"query_id": "q0", "gt_reproj_error_px": "1.0"}, {"query_id": "q1", "gt_reproj_error_px": "9.0"}],
        thresholds_px=(2.0, 5.0),
    )
    current = summarize_pool_rows(
        [
            {"query_id": "q0", "gt_reproj_error_px": "1.0"},
            {"query_id": "q0", "gt_reproj_error_px": "3.0"},
            {"query_id": "q1", "gt_reproj_error_px": "1.5"},
            {"query_id": "q1", "gt_reproj_error_px": "4.0"},
        ],
        thresholds_px=(2.0, 5.0),
    )

    comparison = compare_pool_summaries(baseline, current)

    assert comparison["thresholds"]["2px"]["valid_count_growth"] == 2.0
    assert comparison["thresholds"]["5px"]["valid_count_growth"] == 4.0
    assert comparison["thresholds"]["5px"]["passes_3x_growth"] is True
