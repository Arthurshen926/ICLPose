from __future__ import annotations

import json

import pytest

from feature_extract.tools.vfm.select_goal_maplet_direct_plane_pnp_by_moge3_normal import (
    _load_reports,
    _shared_lineage,
)


def _report() -> dict[str, object]:
    return {
        "artifact_type": "goal_maplet_frozen_pnp_moge3_2dgs_render_consistency_v1",
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "query_depth_changes_frozen_pose": False,
        "query_count": 1,
        "physical_map_content_sha256": "map",
        "physical_map_file_sha256": "map-file",
        "query_camera_inventory_content_sha256": "camera",
        "query_camera_inventory_file_sha256": "camera-file",
        "moge3_manifest_content_sha256_in_order": ["moge"],
        "moge3_manifest_file_sha256_in_order": ["moge-file"],
        "rows": [{"name": "seq__frame.png.npz", "usable": True, "normal_within_20deg": 0.8}],
    }


def test_normal_selector_accepts_only_frozen_label_free_report(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()))
    rows, reports = _load_reports([path])
    assert rows[0]["normal_within_20deg"] == 0.8
    assert _shared_lineage(reports)["physical_map_content_sha256"] == "map"
    bad = _report(); bad["query_pose_or_ground_truth_read"] = True
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        _load_reports([path])


def test_normal_selector_rejects_branch_lineage_mismatch() -> None:
    left = _report(); right = _report(); right["physical_map_content_sha256"] = "other"
    with pytest.raises(ValueError):
        _shared_lineage([left, right])
