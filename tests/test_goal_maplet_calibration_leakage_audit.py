from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from feature_extract.tools.vfm.audit_goal_maplet_calibration_leakage import (
    build_invalidation_report,
)


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibration_leakage_audit_follows_cryptographic_dependency_closure(tmp_path):
    calibration = tmp_path / "calibration.json"
    _write(
        calibration,
        {
            "artifact_type": "goal_maplet_validity_calibration_v1",
            "content_sha256": "a" * 64,
            "metadata": {
                "fit_trajectory_ids": ["seq12"],
                "fit_image_ids": ["seq12/a.png"],
            },
        },
    )
    retrieval_shard = tmp_path / "query.npz"
    retrieval_shard.write_bytes(b"query")
    retrieval = tmp_path / "retrieval.json"
    _write(
        retrieval,
        {
            "artifact_type": "goal_maplet_pure_radio_retrieval_run_v1",
            "validity_calibration_sha256": "a" * 64,
            "query_count": 1,
            "rows": [
                {
                    "image_id": "seq14/a.png",
                    "artifact": str(retrieval_shard),
                    "artifact_sha256": _sha(retrieval_shard),
                }
            ],
        },
    )
    pool = tmp_path / "pool.json"
    _write(
        pool,
        {
            "artifact_type": "goal_maplet_pose_free_candidate_pool_v1",
            "content_sha256": "b" * 64,
            "retrieval_runs": [{"path": str(retrieval), "file_sha256": _sha(retrieval)}],
        },
    )
    labels = tmp_path / "labels.json"
    labels_npz = tmp_path / "labels.npz"
    labels_npz.write_bytes(b"labels")
    _write(
        labels,
        {
            "artifact_type": "goal_maplet_direct_pose_candidate_dataset_v1",
            "candidate_pool_content_sha256": "b" * 64,
            "content_sha256": "c" * 64,
            "output_npz": str(labels_npz),
        },
    )
    unrelated = tmp_path / "unrelated.json"
    _write(
        unrelated,
        {
            "artifact_type": "unrelated",
            "physical_map_sha256": "d" * 64,
        },
    )

    report = build_invalidation_report(
        calibration_path=calibration,
        scan_roots=[tmp_path],
        protected_query_routes=["seq12", "seq14"],
    )
    rows = {Path(str(row["path"])).name: row for row in report["affected_json_artifacts"]}
    assert set(rows) == {"retrieval.json", "pool.json", "labels.json"}
    assert rows["retrieval.json"]["dependency_depth"] == 1
    assert rows["pool.json"]["dependency_depth"] == 2
    assert rows["labels.json"]["dependency_depth"] == 3
    assert report["status"] == "invalid_protected_query_calibration_overlap"
    assert report["production_eligible"] is False
    owned = {Path(str(row["path"])).name for row in report["owned_materialized_artifacts"]}
    assert {"query.npz", "labels.npz"} <= owned


def test_calibration_leakage_audit_requires_protected_route_overlap(tmp_path):
    calibration = tmp_path / "calibration.json"
    _write(
        calibration,
        {
            "artifact_type": "goal_maplet_validity_calibration_v1",
            "content_sha256": "a" * 64,
            "metadata": {"fit_trajectory_ids": ["seq10"], "fit_image_ids": []},
        },
    )
    with pytest.raises(ValueError, match="protected-query route overlap"):
        build_invalidation_report(
            calibration_path=calibration,
            scan_roots=[tmp_path],
            protected_query_routes=["seq12", "seq14"],
        )
