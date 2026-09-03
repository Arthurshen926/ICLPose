from __future__ import annotations

import copy

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_goal_maplet_sparse_first_render_equivalence import (
    _validate_sparse_rows,
)


def _report() -> dict[str, object]:
    return {
        "artifact_type": "goal_maplet_frozen_pnp_moge3_2dgs_render_consistency_v1",
        "query_count": 2,
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "query_moge3_role": "role",
        "depth_scale_fit": "scale",
        "affine_log_depth_fit": "affine",
        "raw_metric_depth_retained": True,
        "query_depth_changes_frozen_pose": False,
        "frozen_pose_inventory_file_sha256": "pose-file",
        "frozen_pose_inventory_content_sha256": "pose-content",
        "physical_map_file_sha256": "map-file",
        "physical_map_content_sha256": "map-content",
        "query_camera_inventory_file_sha256": "camera-file",
        "query_camera_inventory_content_sha256": "camera-content",
        "frozen_correspondence_file_sha256": "corr-file",
        "frozen_correspondence_content_sha256": "corr-content",
        "moge3_manifest_file_sha256_in_order": ["moge-file"],
        "moge3_manifest_content_sha256_in_order": ["moge-content"],
        "minimum_front_incidence": 0.05,
        "rows": [
            {"name": "a", "usable": True, "normal_within_20deg": 0.7},
            {"name": "b", "usable": True, "normal_within_20deg": 0.4},
        ],
    }


def test_sparse_rows_keep_required_and_omit_irrelevant_scores() -> None:
    full = _report()
    sparse = copy.deepcopy(full)
    sparse["rows"][0]["dense_score_evaluated"] = True
    sparse["rows"][1] = {"name": "b", "usable": True, "dense_score_evaluated": False}
    _validate_sparse_rows(full, sparse, np.asarray([True, False]))


def test_sparse_rows_reject_a_score_on_an_omitted_query() -> None:
    full = _report()
    sparse = copy.deepcopy(full)
    for row in sparse["rows"]:
        row["dense_score_evaluated"] = True
    with pytest.raises(ValueError, match="evaluated mask differs"):
        _validate_sparse_rows(full, sparse, np.asarray([True, False]))


def test_sparse_rows_reject_retained_score_drift() -> None:
    full = _report()
    sparse = copy.deepcopy(full)
    sparse["rows"][0]["dense_score_evaluated"] = True
    sparse["rows"][0]["normal_within_20deg"] = 0.6
    sparse["rows"][1] = {"name": "b", "usable": True, "dense_score_evaluated": False}
    with pytest.raises(ValueError, match="diagnostics differ"):
        _validate_sparse_rows(full, sparse, np.asarray([True, False]))
