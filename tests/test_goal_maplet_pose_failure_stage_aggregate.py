import json

import pytest

from feature_extract.tools.vfm.aggregate_goal_maplet_pose_failure_stages import aggregate
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256


def _write(path, split, name, category, success):
    report = {
        "artifact_type": "goal_maplet_pose_failure_stage_postlabel_audit_v1",
        "selection_or_training_eligible": False,
        "split_name": split,
        "query_count": 1,
        "coarse_threshold": {"translation_m": 2.0, "rotation_deg": 45.0},
        "oracle_definition": "fixed",
        "rows": [{
            "name": name,
            "final_failure_category": category,
            "selected_is_2m_45deg_hit": success,
            "oracle_candidate_row_count": 8,
            "oracle_physical_plane_count": 3,
            "minimum_candidate_gt_reprojection_px": 0.5,
        }],
    }
    report["content_sha256"] = canonical_json_sha256(report)
    path.write_text(json.dumps(report))


def test_aggregate_counts_and_binds_inputs(tmp_path):
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    _write(one, "a", "q.png", "selected_pose_coarse_success", True)
    _write(two, "b", "q.png", "hard_coordinate_or_pnp_initialization_failure", False)
    report = aggregate([one, two])
    assert report["query_count"] == 2
    assert report["selected_coarse_failure_count"] == 1
    assert report["failure_category_counts"] == {
        "hard_coordinate_or_pnp_initialization_failure": 1
    }
    assert report["initialization_failure_existing_support"]["oracle_candidate_row_count"]["median"] == 8.0


def test_aggregate_rejects_tampered_input(tmp_path):
    path = tmp_path / "one.json"
    _write(path, "a", "q.png", "selected_pose_coarse_success", True)
    report = json.loads(path.read_text())
    report["query_count"] = 2
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="canonical content hash"):
        aggregate([path])
