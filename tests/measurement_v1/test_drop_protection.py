from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.protect_selected_measurement_drop_actions import (
    protect_drop_actions,
)


def test_drop_protection_restores_inliers_coverage_and_minimum_count() -> None:
    action_rows = [
        {
            "policy_row_index": index,
            "query_id": "query.png",
            "track_id": 100 + index,
            "geometry_probability": 0.1 * (index + 1),
            "action": "DROP" if index < 5 else "KEEP",
        }
        for index in range(6)
    ]
    context = {
        index: {
            "policy_row_index": index,
            "query_id": "query.png",
            "track_id": 100 + index,
            "coarse_pose_inlier": index == 0,
        }
        for index in range(6)
    }
    query_xy = np.asarray(
        [[10.0, 10.0], [20.0, 20.0], [80.0, 20.0], [20.0, 80.0], [80.0, 80.0], [15.0, 15.0]],
        dtype=np.float32,
    )

    protected, reports = protect_drop_actions(
        action_rows=action_rows,
        context_by_policy_row=context,
        query_ids=np.asarray(["query.png"] * 6),
        query_xy=query_xy,
        selected_track_ids=np.arange(100, 106),
        pose_selection_scores=np.linspace(0.1, 0.6, 6),
        protect_coarse_pose_inliers=True,
        min_retained_matches=4,
        min_grid_cells=3,
        grid_rows=2,
        grid_cols=2,
        image_width=100,
        image_height=100,
    )

    assert protected[0]["action"] == "KEEP"
    assert "coarse_pose_inlier" in protected[0]["drop_protection_reason"]
    assert sum(row["action"] != "DROP" for row in protected) >= 4
    assert reports[0]["final_grid_cell_count"] >= 3
    assert reports[0]["restored_count"] >= 3
