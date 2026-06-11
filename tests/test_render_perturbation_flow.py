from __future__ import annotations

import numpy as np

from feature_extract.vfm.render_perturbation_flow import flow_capture_stats


def test_flow_capture_stats_reports_threshold_and_same_cell_rates() -> None:
    source_xy = np.asarray(
        [
            [8.0, 8.0],
            [24.0, 8.0],
            [40.0, 8.0],
            [56.0, 8.0],
        ],
        dtype=np.float64,
    )
    target_xy = np.asarray(
        [
            [12.0, 8.0],  # 4 px, same 16 px cell
            [34.0, 8.0],  # 10 px, adjacent cell
            [58.0, 8.0],  # 18 px, adjacent cell
            [95.0, 8.0],  # 39 px, far
        ],
        dtype=np.float64,
    )

    stats = flow_capture_stats(
        source_xy,
        target_xy,
        image_width=128,
        image_height=64,
        grid_width=8,
        grid_height=4,
        thresholds_px=(8.0, 16.0, 32.0),
    )

    assert stats["flow_count"] == 4
    assert stats["flow_median_px"] == 14.0
    assert stats["flow_within_8px"] == 0.25
    assert stats["flow_within_16px"] == 0.5
    assert stats["flow_within_32px"] == 0.75
    assert stats["flow_within_same_cell"] == 0.25
    assert stats["flow_within_cell_radius_0"] == 0.25
    assert stats["flow_within_cell_radius_1"] == 0.75
    assert stats["flow_within_cell_radius_2"] == 1.0
    assert stats["flow_cell_radius_p90"] == 1.7000000000000002
