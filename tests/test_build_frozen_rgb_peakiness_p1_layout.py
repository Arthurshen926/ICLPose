from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.build_frozen_rgb_peakiness_p1_layout import (
    _subset_layout,
    _validate_args,
    parse_args,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
)


def _layout() -> CandidatePoseRGBSpatialLayout:
    return CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray([11, 12, 21, 22], dtype=np.int64),
        query_ids=np.asarray(["q/a.png", "q/a.png", "q/b.png", "q/b.png"]),
        split_names=np.asarray(["train", "train", "validation", "validation"]),
        xy=np.asarray([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]], dtype=np.float32),
        point_sources=np.asarray(["p", "p", "p", "p"]),
        candidate_track_ids=np.asarray([[101], [102], [103], [104]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[0], [1], [2], [3]], dtype=np.int64),
        candidate_coarse_similarities=np.full((4, 1), 0.8, dtype=np.float32),
        candidate_prior_probabilities=np.full((4, 1), 0.8, dtype=np.float32),
        null_probabilities=np.full((4,), 0.2, dtype=np.float32),
        support_image_ids=np.asarray(
            [[["map/a.png"]], [["map/b.png"]], [["map/c.png"]], [["map/d.png"]]]
        ),
        support_xy=np.asarray(
            [[[[1.0, 1.0]]], [[[2.0, 2.0]]], [[[3.0, 3.0]]], [[[4.0, 4.0]]]],
            dtype=np.float32,
        ),
        support_view_valid=np.ones((4, 1, 1), dtype=bool),
        support_view_weights=np.ones((4, 1, 1), dtype=np.float32),
        support_coverage_counts=np.ones((4, 1, 1), dtype=np.int32),
        metadata={
            "format": "candidate_pose_rgb_spatial_layout_v1",
            "contains_ground_truth": False,
            "contains_target_errors": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "verification_points_sha256": "points",
            "maplet_support_index_sha256": "maplet",
            "support_geometry_index_sha256": "geometry",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
        },
    )


def test_subset_layout_keeps_only_selected_target_free_rows() -> None:
    layout = _layout()
    subset = _subset_layout(
        layout=layout,
        rows=np.asarray([1, 3], dtype=np.int64),
        metadata={
            **layout.metadata,
            "frozen_rgb_selector": {
                "selector_policy": "rgb_peakiness",
                "selection_before_train_target_join": True,
            },
        },
    )
    assert subset.source_point_ids.tolist() == [12, 22]
    assert subset.query_ids.tolist() == ["q/a.png", "q/b.png"]
    assert subset.metadata["frozen_rgb_selector"]["selection_before_train_target_join"] is True
    with pytest.raises(ValueError, match="rows are invalid"):
        _subset_layout(layout=layout, rows=np.asarray([1, 1]), metadata=layout.metadata)


def test_builder_rejects_non_rgb_selector_policy() -> None:
    base = [
        "--rgb-spatial-layout", "layout.npz",
        "--checkpoint", "checkpoint.pt",
        "--cross-layout-component-audit", "audit.json",
        "--radio-final-context-cache", "final.npz",
        "--radio-intermediate-context-cache", "intermediate.npz",
        "--alike-spatial-context-cache", "alike.npz",
        "--image-root", "images",
        "--output", "selected.npz",
        "--summary-json", "summary.json",
    ]
    assert _validate_args(parse_args(base)) == {
        "radio_final": 15,
        "radio_intermediate": 15,
        "alike": 13,
    }
    with pytest.raises(SystemExit):
        parse_args([*base, "--selector-policy", "uniform"])
