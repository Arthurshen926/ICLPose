from __future__ import annotations

import json

import pytest

from feature_extract.tools.vfm.evaluate_goal_maplet_factorized_parent_layout_guide_run import (
    _summary,
    _validated_phase1,
)
from feature_extract.tools.vfm.seal_goal_maplet_factorized_parent_layout_guide_run import (
    SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256


def _phase1_report() -> dict[str, object]:
    value: dict[str, object] = {
        "artifact_type": SCHEMA,
        "query_count": 1,
        "strict_v4_required": True,
        "strict_v4_seq10_calibrated_retrieval_confirmed": True,
        "score_before_label_contract": {
            "label_or_direct_dataset_argument_accepted": False,
            "query_pose_or_gt_argument_accepted": False,
            "contributor_argument_accepted": False,
            "contributor_file_bytes_hashed": False,
            "contributor_pose_member_opened": False,
            "camera_binding_is_intrinsics_and_image_id_only": True,
            "ranked_scores_are_frozen_by_file_and_content_hash": True,
        },
        "rows": [{"query_index": 0, "image_id": "seq14/frame.png"}],
    }
    value["content_sha256"] = canonical_json_sha256(value)
    return value


def test_phase1_manifest_accepts_only_strict_score_before_label_contract(tmp_path):
    report = _phase1_report()
    path = tmp_path / "phase1.json"
    path.write_text(json.dumps(report))
    assert _validated_phase1(path)["query_count"] == 1

    report["score_before_label_contract"][
        "query_pose_or_gt_argument_accepted"
    ] = True
    report.pop("content_sha256")
    report["content_sha256"] = canonical_json_sha256(report)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="score-before-label"):
        _validated_phase1(path)


def test_phase1_manifest_hash_and_nonempty_inventory_fail_closed(tmp_path):
    report = _phase1_report()
    report["content_sha256"] = "0" * 64
    path = tmp_path / "bad_hash.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="score-before-label"):
        _validated_phase1(path)

    report = _phase1_report()
    report["rows"] = []
    report["query_count"] = 0
    report.pop("content_sha256")
    report["content_sha256"] = canonical_json_sha256(report)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="inventory"):
        _validated_phase1(path)


def test_phase2_numeric_summary_is_deterministic_and_finite():
    result = _summary([1.0, 2.0, 3.0, 4.0])
    assert result == {
        "minimum": 1.0,
        "mean": 2.5,
        "median": 2.5,
        "p90": pytest.approx(3.7),
        "maximum": 4.0,
    }
    with pytest.raises(ValueError, match="empty"):
        _summary([])
