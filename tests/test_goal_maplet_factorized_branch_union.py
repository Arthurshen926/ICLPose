from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.factorized_branch_union import (
    BRANCH_NAMES,
    SCHEMA,
    SEMANTICS,
    build_factorized_branch_union_arrays,
    load_factorized_branch_union,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256


def _branch(*, layout: bool) -> dict[str, np.ndarray]:
    seed_centers = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
         [4.0 if layout else 2.0, 0.0, 0.0],
         [5.0 if layout else 3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    poses = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
    poses[:, :3, 3] = -seed_centers
    angles = np.arange(64, dtype=np.float64)
    if layout:
        angles[32:] += 32.0
    radians = np.deg2rad(angles)
    rotations = np.zeros((64, 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = np.cos(radians)
    rotations[:, 0, 1] = -np.sin(radians)
    rotations[:, 1, 0] = np.sin(radians)
    rotations[:, 1, 1] = np.cos(radians)
    rotations[:, 2, 2] = 1.0
    return {
        "image_ids": np.asarray(["seq14/frame.png"]),
        "position_seed_candidate_ranks": np.arange(1, 5, dtype=np.int16)[None],
        "position_seed_poses_w2c": poses[None],
        "position_seed_centers_world": seed_centers[None],
        "position_offsets_camera": np.zeros((1, 3), dtype=np.float64),
        "position_centers_world": seed_centers[None, :, None, :],
        "orientation_rotations_w2c": rotations[None],
        "orientation_source_candidate_ranks": np.arange(
            1, 65, dtype=np.int16,
        )[None],
        "orientation_valid": np.ones((1, 64), dtype=bool),
    }


def test_factorized_branch_union_deduplicates_storage_but_keeps_branch_domains():
    union = build_factorized_branch_union_arrays(
        _branch(layout=False), _branch(layout=True),
    )
    assert int(np.sum(union["unique_position_seed_valid"])) == 6
    assert int(np.sum(union["unique_orientation_valid"])) == 96
    np.testing.assert_array_equal(
        union["branch_position_seed_to_unique"][0, 0], [0, 1, 2, 3],
    )
    np.testing.assert_array_equal(
        union["branch_position_seed_to_unique"][0, 1], [0, 1, 4, 5],
    )
    # Each branch has 4*64 seed/orientation pairs.  Their intersection has
    # 2 shared seeds * 32 shared orientations, hence union=448, not the
    # incorrect Cartesian hull 6*96=576.
    assert union["implicit_seed_orientation_pair_count_by_query"].tolist() == [448]
    assert union["implicit_lattice_pose_pair_count_by_query"].tolist() == [448]
    assert 448 < 6 * 96


def test_factorized_branch_union_rejects_query_or_lattice_drift():
    baseline = _branch(layout=False)
    layout = _branch(layout=True)
    layout["image_ids"] = np.asarray(["seq12/frame.png"])
    with pytest.raises(ValueError, match="query/lattice"):
        build_factorized_branch_union_arrays(baseline, layout)


def _metadata(arrays):
    seed_counts = np.sum(arrays["unique_position_seed_valid"], axis=1)
    orientation_counts = np.sum(arrays["unique_orientation_valid"], axis=1)
    pair_counts = arrays["implicit_seed_orientation_pair_count_by_query"]
    lattice_counts = arrays["implicit_lattice_pose_pair_count_by_query"]
    return {
        "artifact_type": SCHEMA,
        "semantics": SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "branch_names": list(BRANCH_NAMES),
        "query_count": int(arrays["image_ids"].size),
        "branch_position_seed_budget": 4,
        "branch_orientation_budget": 64,
        "position_offsets_per_seed": int(
            arrays["position_offsets_camera"].shape[0]
        ),
        "unique_position_seed_count_range": [
            int(np.min(seed_counts)), int(np.max(seed_counts)),
        ],
        "unique_orientation_count_range": [
            int(np.min(orientation_counts)), int(np.max(orientation_counts)),
        ],
        "implicit_seed_orientation_pair_count_range": [
            int(np.min(pair_counts)), int(np.max(pair_counts)),
        ],
        "implicit_lattice_pose_pair_count_range": [
            int(np.min(lattice_counts)), int(np.max(lattice_counts)),
        ],
        "domain_union_is_branch_or_not_cartesian_hull": True,
        "cross_branch_cartesian_products_included": False,
        "cartesian_pose_product_materialized": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
    }


def _save(path, arrays):
    metadata = _metadata(arrays)
    np.savez_compressed(
        path, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_factorized_branch_union_loader_checks_domain_count_and_geometry(tmp_path):
    arrays = build_factorized_branch_union_arrays(
        _branch(layout=False), _branch(layout=True),
    )
    path = tmp_path / "union.npz"
    _save(path, arrays)
    loaded, _ = load_factorized_branch_union(path)
    assert loaded["implicit_seed_orientation_pair_count_by_query"].tolist() == [448]

    tampered_count = {name: value.copy() for name, value in arrays.items()}
    tampered_count["implicit_seed_orientation_pair_count_by_query"][0] = 576
    _save(path, tampered_count)
    with pytest.raises(ValueError, match="valid prefixes"):
        load_factorized_branch_union(path)

    tampered_geometry = {name: value.copy() for name, value in arrays.items()}
    tampered_geometry["unique_position_centers_world"][0, 0, 0, 0] += 0.25
    _save(path, tampered_geometry)
    with pytest.raises(ValueError, match="geometry/padding"):
        load_factorized_branch_union(path)
