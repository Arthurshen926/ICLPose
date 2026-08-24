from __future__ import annotations

import json

from feature_extract.tools.vfm.evaluate_goal_maplet_global_all_parent_pose_support import (
    SCHEMA,
    _query_ids_from_pose_free_manifest,
    _validate_frozen_seq10_gate,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


def test_pose_free_manifest_selects_only_the_requested_route(tmp_path):
    path = tmp_path / "manifest.json"
    records = [
        {"image_id": f"seq10/frame{index:05d}.png"} for index in range(88)
    ] + [{"image_id": "seq12/frame00001.png"}]
    path.write_text(json.dumps({"records": records}))
    protocol = {
        "official_train": {
            "count": len(records),
            "token_manifest_sha256": file_sha256(path),
            "trajectory_counts": {"seq10": 88, "seq12": 1},
        }
    }
    selected = _query_ids_from_pose_free_manifest(path, "seq10", protocol)
    assert len(selected) == 88
    assert all(value.startswith("seq10/") for value in selected)


def test_held_gate_accepts_only_canonical_passing_seq10_report(tmp_path):
    path = tmp_path / "seq10.json"
    report = {
        "artifact_type": SCHEMA,
        "query_route": "seq10",
        "seq10_absolute_gate": {
            "decision": "GO",
            "threshold": 0.95,
            "required_hits": 84,
            "observed_hits": 88,
        },
    }
    report["content_sha256"] = canonical_json_sha256(report)
    path.write_text(json.dumps(report))
    assert _validate_frozen_seq10_gate(path)["seq10_absolute_gate"]["decision"] == "GO"

    report["seq10_absolute_gate"]["decision"] = "KILL"
    report["content_sha256"] = canonical_json_sha256({
        key: value for key, value in report.items() if key != "content_sha256"
    })
    path.write_text(json.dumps(report))
    try:
        _validate_frozen_seq10_gate(path)
    except ValueError as error:
        assert "passing frozen seq10 gate" in str(error)
    else:
        raise AssertionError("held evaluation accepted a killed seq10 gate")
