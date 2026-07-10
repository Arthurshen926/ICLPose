from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.summarize_match_reprojection_residuals import (
    ResidualAuditConfig,
    summarize_match_reprojection_residuals,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera


def test_summarize_match_reprojection_residuals_counts_thresholds_and_oracle_tokens() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=80, params=(10.0, 10.0, 50.0, 40.0))
    pose = np.eye(4, dtype=np.float64)
    rows = [
        {
            "query_id": "q.png",
            "token_index": "7",
            "x": "50.0",
            "y": "40.0",
            "xyz": "[0.0, 0.0, 1.0]",
        },
        {
            "query_id": "q.png",
            "token_index": "7",
            "x": "55.0",
            "y": "40.0",
            "xyz": "[0.0, 0.0, 1.0]",
        },
        {
            "query_id": "q.png",
            "token_index": "8",
            "x": "70.0",
            "y": "40.0",
            "xyz": "[1.0, 0.0, 1.0]",
        },
    ]

    summary = summarize_match_reprojection_residuals(
        rows,
        pose_w2c_by_query={"q.png": pose},
        camera_by_query={"q.png": camera},
        config=ResidualAuditConfig(thresholds_px=(2.0, 5.0)),
    )

    assert summary["query_count"] == 1
    assert summary["processed_match_count"] == 3
    assert summary["valid_2px_count"] == 1
    assert summary["valid_5px_count"] == 2
    assert summary["mean_oracle_unique_token_count"] == 2.0
    assert summary["mean_oracle_unique_valid_2px_count_per_query"] == 1.0
    assert summary["mean_oracle_unique_valid_5px_count_per_query"] == 1.0
    assert summary["per_query"][0]["oracle_unique_token_count"] == 2
    assert summary["per_query"][0]["oracle_unique_valid_2px_count"] == 1
    assert summary["per_query"][0]["oracle_unique_valid_5px_count"] == 1
