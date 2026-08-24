from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.build_goal_maplet_sparse_pose_transport_dataset import (
    _load_contributors,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    _load_dataset,
)
from feature_extract.vfm.localization_goal_maplet.candidate_conditioned_pose_attribution import (
    pose_transport_hierarchy_content_sha256,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256
from feature_extract.vfm.localization_goal_maplet.pose_transport_hierarchy import (
    HIERARCHY_SEMANTICS,
)


def _write_dataset(path: Path, *, schema: str = "goal_maplet_real_sparse_pose_transport_dataset_v2"):
    parent = np.asarray([0, 0], dtype=np.int32)
    support = np.asarray([-1, -1], dtype=np.int32)
    offsets = np.asarray([0, 1, 2], dtype=np.int64)
    adjacency = np.asarray([1, 0], dtype=np.int32)
    arrays = {
        "image_ids": np.asarray(["seq11/frame00001.png"]),
        "candidate_valid": np.ones((1, 2), dtype=bool),
        "hierarchy_child_parent_ids": parent,
        "hierarchy_child_support_ids": support,
        "hierarchy_adjacency_offsets": offsets,
        "hierarchy_adjacency_child_rows": adjacency,
    }
    metadata = {
        "artifact_type": schema,
        "content_sha256": arrays_sha256(arrays),
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "canonical_map_excludes_query_route": True,
        "map_pose_field_semantics": "single_view_independent_canonical_field_control_v1",
        "view_conditioned_field_sha256": None,
        "view_conditioned_field_file_sha256": None,
        "hierarchy_semantics": HIERARCHY_SEMANTICS,
        "hierarchy_content_sha256": pose_transport_hierarchy_content_sha256(
            parent, support, offsets, adjacency
        ),
    }
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True))
    )
    return metadata


def test_v2_dataset_reopens_with_embedded_replayable_hierarchy(tmp_path: Path):
    path = tmp_path / "dataset.npz"
    expected = _write_dataset(path)
    arrays, metadata = _load_dataset(path)
    assert metadata["content_sha256"] == expected["content_sha256"]
    np.testing.assert_array_equal(arrays["hierarchy_child_parent_ids"], [0, 0])


def test_legacy_dataset_without_embedded_hierarchy_is_rejected(tmp_path: Path):
    path = tmp_path / "dataset.npz"
    _write_dataset(path, schema="goal_maplet_real_sparse_pose_transport_dataset_v1")
    with pytest.raises(ValueError, match="not a real sparse pose transport dataset"):
        _load_dataset(path)


def test_rehashed_but_false_hierarchy_lineage_is_rejected(tmp_path: Path):
    path = tmp_path / "dataset.npz"
    _write_dataset(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    arrays["hierarchy_child_parent_ids"] = np.asarray([0, 1], dtype=np.int32)
    metadata["content_sha256"] = arrays_sha256(arrays)
    # Deliberately retain the old hierarchy content hash while rehashing the
    # outer dataset, reproducing the historical insufficient-lineage failure.
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True))
    )
    with pytest.raises(ValueError, match="hierarchy lineage differs"):
        _load_dataset(path)


def test_view_conditioned_dataset_requires_both_field_hashes(tmp_path: Path):
    path = tmp_path / "dataset.npz"
    _write_dataset(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name]) for name in data.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    metadata["map_pose_field_semantics"] = (
        "candidate_pose_evaluated_low_rank_view_conditioned_canonical_field_v1"
    )
    metadata["view_conditioned_field_sha256"] = "a" * 64
    metadata["view_conditioned_field_file_sha256"] = None
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True))
    )
    with pytest.raises(ValueError, match="lacks field lineage"):
        _load_dataset(path)


def test_bounded_builder_opens_only_required_contributors(tmp_path: Path):
    for image_id in ("seq12/frame00001.png", "seq12/frame00002.png"):
        path = tmp_path / (image_id.replace("/", "__") + ".npz")
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps({"image_id": image_id})),
        )
    # A malformed unrelated file must not be opened by a bounded build.
    np.savez_compressed(tmp_path / "seq9__unrelated.png.npz", broken=np.asarray(1))
    result = _load_contributors(
        tmp_path, required_image_ids=["seq12/frame00002.png"],
    )
    assert list(result) == ["seq12/frame00002.png"]

    with pytest.raises(ValueError, match="missing contributor artifact"):
        _load_contributors(
            tmp_path, required_image_ids=["seq12/frame00999.png"],
        )
