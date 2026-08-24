from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)


def _write(path: Path, *, schema: str, uses_pnp: bool = False) -> None:
    q, c, t, s, d = 1, 2, 4, 2, 3
    arrays = {
        "image_ids": np.asarray(["seq11/frame00001.png"]),
        "radio_final": np.zeros((q, 5, 1, t), dtype=np.float16),
        "source_child_rows": np.zeros((q, t, s), dtype=np.int32),
        "source_child_probabilities": np.full((q, t, s), 0.5, dtype=np.float32),
        "query_reliability": np.ones((q, t), dtype=np.float32),
        "token_xy": np.zeros((q, t, 2), dtype=np.int16),
        "candidate_poses_w2c": np.tile(np.eye(4), (q, c, 1, 1)),
        "translation_m": np.zeros((q, c), dtype=np.float32),
        "rotation_deg": np.zeros((q, c), dtype=np.float32),
        "candidate_valid": np.ones((q, c), dtype=bool),
        "target_child_rows": np.zeros((q, c, t, s), dtype=np.int32),
        "target_child_weights": np.full((q, c, t, s), 0.5, dtype=np.float32),
        "target_canonical_features": np.ones((q, c, t, s, d), dtype=np.float16),
        "target_modality_valid": np.ones((q, c, t, s, 4), dtype=bool),
    }
    metadata = {
        "artifact_type": schema,
        "content_sha256": arrays_sha256(arrays),
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": uses_pnp,
        "uses_absolute_pose_regression": False,
        "canonical_map_excludes_query_route": True,
    }
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True))
    )


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_pose_candidate_loader_replays_both_frozen_grid_versions(tmp_path: Path, version: str):
    path = tmp_path / "dataset.npz"
    _write(path, schema=f"goal_maplet_real_sparse_pose_transport_dataset_{version}")
    arrays, metadata = load_pose_candidate_dataset(path)
    assert metadata["artifact_type"].endswith(version)
    assert arrays["candidate_valid"].shape == (1, 2)


def test_pose_candidate_loader_does_not_weaken_method_boundary(tmp_path: Path):
    path = tmp_path / "dataset.npz"
    _write(path, schema="goal_maplet_real_sparse_pose_transport_dataset_v1", uses_pnp=True)
    with pytest.raises(ValueError, match="method boundary"):
        load_pose_candidate_dataset(path)


def test_pose_candidate_loader_rejects_rehashed_shape_drift(tmp_path: Path):
    path = tmp_path / "dataset.npz"
    _write(path, schema="goal_maplet_real_sparse_pose_transport_dataset_v2")
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    arrays["target_modality_valid"] = arrays["target_modality_valid"][..., 0]
    metadata["content_sha256"] = arrays_sha256(arrays)
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True))
    )
    with pytest.raises(ValueError, match="target validity grids"):
        load_pose_candidate_dataset(path)


def test_direct_pose_candidate_loader_has_explicit_no_rendered_grid_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "direct.npz"
    q, c = 2, 3
    arrays = {
        "image_ids": np.asarray(["seq3/frame00001.png", "seq5/frame00001.png"]),
        "radio_token_paths": np.asarray(["/tmp/radio-a.npz", "/tmp/radio-b.npz"]),
        "radio_file_sha256": np.asarray(["a" * 64, "b" * 64]),
        "contributor_paths": np.asarray(["/tmp/contributor-a.npz", "/tmp/contributor-b.npz"]),
        "contributor_file_sha256": np.asarray(["c" * 64, "d" * 64]),
        "candidate_poses_w2c": np.tile(np.eye(4), (q, c, 1, 1)),
        "translation_m": np.zeros((q, c), dtype=np.float32),
        "rotation_deg": np.zeros((q, c), dtype=np.float32),
        "candidate_valid": np.ones((q, c), dtype=bool),
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_pose_candidate_dataset_v1",
        "content_sha256": arrays_sha256(arrays),
        "candidate_pool_frozen_before_target_pose_opened": True,
        "canonical_map_excludes_query_route": True,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
    }
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="no rendered target grids"):
        load_pose_candidate_dataset(path)
    loaded, _ = load_pose_candidate_dataset(path, require_rendered_targets=False)
    assert loaded["candidate_poses_w2c"].shape == (q, c, 4, 4)


def test_direct_pose_candidate_loader_rejects_unfrozen_candidate_pool(
    tmp_path: Path,
) -> None:
    path = tmp_path / "direct.npz"
    arrays = {
        "image_ids": np.asarray(["seq3/frame00001.png"]),
        "radio_token_paths": np.asarray(["/tmp/radio.npz"]),
        "radio_file_sha256": np.asarray(["a" * 64]),
        "contributor_paths": np.asarray(["/tmp/contributor.npz"]),
        "contributor_file_sha256": np.asarray(["b" * 64]),
        "candidate_poses_w2c": np.tile(np.eye(4), (1, 2, 1, 1)),
        "translation_m": np.zeros((1, 2), dtype=np.float32),
        "rotation_deg": np.zeros((1, 2), dtype=np.float32),
        "candidate_valid": np.ones((1, 2), dtype=bool),
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_pose_candidate_dataset_v1",
        "content_sha256": arrays_sha256(arrays),
        "candidate_pool_frozen_before_target_pose_opened": False,
        "canonical_map_excludes_query_route": True,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
    }
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="not frozen"):
        load_pose_candidate_dataset(path, require_rendered_targets=False)


def test_controlled_training_inventory_cannot_masquerade_as_deployment_pool(
    tmp_path: Path,
) -> None:
    path = tmp_path / "controlled.npz"
    arrays = {
        "image_ids": np.asarray(["seq12/frame00001.png"]),
        "radio_token_paths": np.asarray(["/tmp/radio.npz"]),
        "radio_file_sha256": np.asarray(["a" * 64]),
        "contributor_paths": np.asarray(["/tmp/contributor.npz"]),
        "contributor_file_sha256": np.asarray(["b" * 64]),
        "candidate_poses_w2c": np.tile(np.eye(4), (1, 1, 1, 1)),
        "translation_m": np.zeros((1, 1), dtype=np.float32),
        "rotation_deg": np.zeros((1, 1), dtype=np.float32),
        "candidate_valid": np.ones((1, 1), dtype=bool),
    }
    metadata = {
        "artifact_type": "goal_maplet_controlled_pose_training_inventory_v1",
        "content_sha256": arrays_sha256(arrays),
        "uses_gt_for_training_candidate_generation": True,
        "deployment_candidate_pool": False,
        "production_eligible": False,
        "canonical_map_excludes_query_route": True,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
    }
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    loaded, loaded_metadata = load_pose_candidate_dataset(
        path, require_rendered_targets=False,
    )
    assert loaded["candidate_valid"].shape == (1, 1)
    assert loaded_metadata["deployment_candidate_pool"] is False

    metadata["deployment_candidate_pool"] = True
    metadata["content_sha256"] = arrays_sha256(arrays)
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="training inventory semantics"):
        load_pose_candidate_dataset(path, require_rendered_targets=False)
