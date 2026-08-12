import json
import sys

import pytest

from feature_extract.tools.vfm import merge_goal_maplet_configuration_evidence
from feature_extract.tools.vfm import merge_goal_maplet_pose_modes


def _write_json(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def _pose_shard(image_id: str, readout_sha: str) -> dict:
    return {
        "stage": "pose",
        "physical_map_sha256": "physical",
        "canonical_field_sha256": "field",
        "validity_calibration_sha256": "validity",
        "proposal_method": "graph",
        "maximum_modes": 32,
        "typed_graph_sha256": "graph",
        "render_identity_rerank": True,
        "identity_render_mode": "child_splat",
        "field_feature_contract_sha256": "contract",
        "physical_instance_readout_sha256": readout_sha,
        "proposal_seed_policy": "seed",
        "cascade_contract": {},
        "configuration_evidence_contract": None,
        "rows": [{"image_id": image_id, "modes": {}}],
    }


def test_pose_merge_preserves_physical_instance_lineage(tmp_path, monkeypatch):
    first = _write_json(tmp_path / "a.json", _pose_shard("seq/a.png", "readout"))
    second = _write_json(tmp_path / "b.json", _pose_shard("seq/b.png", "readout"))
    output = tmp_path / "merged.json"
    monkeypatch.setattr(sys, "argv", ["merge", "--inputs", first, second, "--output_json", str(output)])
    merge_goal_maplet_pose_modes.main()
    assert json.loads(output.read_text())["physical_instance_readout_sha256"] == "readout"


def test_pose_merge_rejects_mixed_physical_instance_lineage(tmp_path, monkeypatch):
    first = _write_json(tmp_path / "a.json", _pose_shard("seq/a.png", "readout-a"))
    second = _write_json(tmp_path / "b.json", _pose_shard("seq/b.png", "readout-b"))
    monkeypatch.setattr(
        sys, "argv", ["merge", "--inputs", first, second, "--output_json", str(tmp_path / "out.json")],
    )
    with pytest.raises(ValueError, match="physical_instance_readout_sha256"):
        merge_goal_maplet_pose_modes.main()


def test_pose_merge_rejects_mixed_surface_verification_contract(tmp_path, monkeypatch):
    first_payload = _pose_shard("seq/a.png", "readout")
    second_payload = _pose_shard("seq/b.png", "readout")
    first_payload["surface_verification_contract"] = {
        "role": "context", "maximum_modes": 16,
    }
    second_payload["surface_verification_contract"] = {
        "role": "local", "maximum_modes": 16,
    }
    first = _write_json(tmp_path / "a.json", first_payload)
    second = _write_json(tmp_path / "b.json", second_payload)
    monkeypatch.setattr(
        sys, "argv", ["merge", "--inputs", first, second, "--output_json", str(tmp_path / "out.json")],
    )
    with pytest.raises(ValueError, match="surface_verification_contract"):
        merge_goal_maplet_pose_modes.main()


def test_pose_merge_rejects_new_unlisted_lineage_fields(tmp_path, monkeypatch):
    first_payload = _pose_shard("seq/a.png", "readout")
    second_payload = _pose_shard("seq/b.png", "readout")
    first_payload["mapping_view_graph_sha256"] = "view-a"
    second_payload["mapping_view_graph_sha256"] = "view-b"
    first = _write_json(tmp_path / "a.json", first_payload)
    second = _write_json(tmp_path / "b.json", second_payload)
    monkeypatch.setattr(
        sys, "argv", ["merge", "--inputs", first, second, "--output_json", str(tmp_path / "out.json")],
    )
    with pytest.raises(ValueError, match="mapping_view_graph_sha256"):
        merge_goal_maplet_pose_modes.main()


def test_pose_merge_emits_standard_manifest_for_strict_freeze(tmp_path, monkeypatch):
    source_state = {
        "git_commit_sha": "commit",
        "git_tracked_worktree_dirty": True,
        "git_tracked_status_sha256": "a" * 64,
        "git_tracked_diff_sha256": "b" * 64,
        "untracked_source_file_count": 2,
        "untracked_source_files_sha256": "c" * 64,
        "repository_state_capture": "process_start_before_long_computation",
    }
    payloads = [
        _pose_shard("seq/a.png", "readout"),
        _pose_shard("seq/b.png", "readout"),
    ]
    for shard_index, payload in enumerate(payloads):
        configuration = {"parent_mode": "actual", "shard_index": shard_index, "shard_count": 2}
        payload["run_manifest"] = {
            "schema": "goal_maplet_run_manifest_v1",
            **source_state,
            "argv": ["python", "tool.py"],
            "configuration": configuration,
            "numeric_contract": {"feature_dtype": "float32"},
            "input_artifacts": {"map": "same", "queries": str(shard_index)},
            "candidate_counts": {
                payload["rows"][0]["image_id"]: {"raw": 10},
            },
            "device": "cuda:0",
        }
    paths = [
        _write_json(tmp_path / f"{index}.json", payload)
        for index, payload in enumerate(payloads)
    ]
    output = tmp_path / "merged.json"
    monkeypatch.setattr(
        sys, "argv", ["merge", "--inputs", *paths, "--output_json", str(output)],
    )
    merge_goal_maplet_pose_modes.main()
    manifest = json.loads(output.read_text())["run_manifest"]
    assert manifest["schema"] == "goal_maplet_run_manifest_v1"
    assert manifest["repository_state_capture"] == (
        "process_start_before_long_computation"
    )
    assert "shard_index" not in manifest["configuration"]
    assert set(manifest["candidate_counts"]) == {"seq/a.png", "seq/b.png"}


def _configuration_shard(image_id: str, readout_sha: str) -> dict:
    return {
        "stage": "configuration",
        "physical_map_sha256": "physical",
        "canonical_field_sha256": "field",
        "validity_calibration_sha256": "validity",
        "proposal_method": "graph",
        "maximum_modes": 32,
        "typed_graph_sha256": "graph",
        "field_feature_contract_sha256": "contract",
        "proposal_seed_policy": "seed",
        "rows": [{"image_id": image_id}],
        "configuration_evidence_contract": {
            "physical_instance_readout_sha256": readout_sha,
            "mode_relation_endpoint_state_policy": "mass_adaptive_g17_breadth_depth",
            "application_trajectories": [image_id.split("/", 1)[0]],
        },
    }


def test_configuration_merge_rejects_mixed_readout_lineage(tmp_path, monkeypatch):
    first = _write_json(tmp_path / "a.json", _configuration_shard("seq12/a.png", "readout-a"))
    second = _write_json(tmp_path / "b.json", _configuration_shard("seq14/b.png", "readout-b"))
    monkeypatch.setattr(
        sys, "argv", ["merge", "--inputs", first, second, "--output_json", str(tmp_path / "out.json")],
    )
    with pytest.raises(ValueError, match="physical_instance_readout_sha256"):
        merge_goal_maplet_configuration_evidence.main()
