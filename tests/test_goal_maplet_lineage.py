from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import (
    build_run_manifest,
    canonical_json_sha256,
)


def test_canonical_json_hash_is_key_order_invariant():
    assert canonical_json_sha256({"b": 2, "a": [1, 3]}) == canonical_json_sha256(
        {"a": [1, 3], "b": 2}
    )


def test_run_manifest_binds_configuration_queries_and_candidate_counts():
    source_state = {
        "git_commit_sha": "start-commit",
        "git_tracked_worktree_dirty": True,
        "git_tracked_status_sha256": "a" * 64,
        "git_tracked_diff_sha256": "b" * 64,
        "untracked_source_file_count": 3,
        "untracked_source_files_sha256": "c" * 64,
        "repository_state_capture": "process_start_before_long_computation",
    }
    manifest = build_run_manifest(
        repository_root=Path(__file__).resolve().parents[1],
        argv=["python", "tool.py", "--topk", "16"],
        configuration={"topk": 16, "device": "cuda:0"},
        input_artifacts={"physical_map": "abc", "optional": None},
        query_ids=["seq12/frame00001.png"],
        device="cuda:0",
        numeric_contract={"feature_dtype": "float32"},
        candidate_counts={"seq12/frame00001.png": {"raw": 100, "exact": 16}},
        repository_state_at_start=source_state,
    )
    assert manifest["schema"] == "goal_maplet_run_manifest_v1"
    assert len(manifest["configuration_sha256"]) == 64
    assert len(manifest["query_list_sha256"]) == 64
    assert len(manifest["git_tracked_status_sha256"]) == 64
    assert len(manifest["git_tracked_diff_sha256"]) == 64
    assert manifest["candidate_counts"]["seq12/frame00001.png"]["exact"] == 16
    assert manifest["input_artifacts"]["physical_map"] == "abc"
    assert manifest["git_commit_sha"] == "start-commit"
    assert manifest["git_tracked_diff_sha256"] == "b" * 64
    assert manifest["untracked_source_file_count"] == 3
    assert manifest["untracked_source_files_sha256"] == "c" * 64
    assert manifest["repository_state_capture"] == (
        "process_start_before_long_computation"
    )
