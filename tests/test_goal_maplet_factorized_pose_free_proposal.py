from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    SCHEMA,
    SEMANTICS,
    build_factorized_pose_free_proposal_arrays,
    camera_local_position_lattice_offsets,
    factorized_raw_coverage,
    load_factorized_pose_free_proposal,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256


def _pose(center, yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(yaw_deg))
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = -rotation @ np.asarray(center, dtype=np.float64)
    return value


def _pool() -> dict[str, object]:
    poses = [
        _pose((float(rank) * 20.0, 0.0, 0.0), float(rank) * 10.0)
        for rank in range(8)
    ]
    return {
        "maximum_modes": 8,
        "rows": [{
            "image_id": "seq14/frame00001.png",
            "mode_details": {"actual_parent_actual_child": [
                {"rank": rank + 1, "pose_w2c": pose.tolist()}
                for rank, pose in enumerate(poses)
            ]},
            # A generator bug that read arbitrary GT-shaped payloads would
            # make the two variants below differ.  The builder has no target
            # argument and consumes only the ranked pose-free details.
            "secret_query_gt_trap": {"pose_w2c": "must-not-be-read"},
        }],
    }


def _write_proposal(path: Path, arrays: dict[str, np.ndarray]) -> None:
    valid = np.asarray(arrays["orientation_valid"], dtype=bool)
    metadata = {
        "artifact_type": SCHEMA,
        "semantics": SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "query_count": int(arrays["image_ids"].size),
        "position_seed_count": int(arrays["position_seed_poses_w2c"].shape[1]),
        "positions_per_seed": int(arrays["position_offsets_camera"].shape[0]),
        "orientation_source_prefix_budget": int(
            arrays["orientation_rotations_w2c"].shape[1]
        ),
        "minimum_stored_orientation_count_per_query": int(
            np.min(np.sum(valid, axis=1))
        ),
        "maximum_stored_orientation_count_per_query": int(
            np.max(np.sum(valid, axis=1))
        ),
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
        "cartesian_pose_product_materialized": False,
        "position_lattice_occupancy_checked": False,
        "position_collision_free_space_certified": False,
        "raw_coverage_is_implicit_factor_support_upper_bound_only": True,
    }
    np.savez_compressed(
        path, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_factorized_builder_uses_camera_local_position_axes_and_no_pose_product():
    pool = _pool()
    # Make the first seed a 90-degree camera rotation so camera-local x is not
    # world x.
    first = _pose((0.0, 0.0, 0.0), 90.0)
    pool["rows"][0]["mode_details"]["actual_parent_actual_child"][0][
        "pose_w2c"
    ] = first.tolist()
    arrays = build_factorized_pose_free_proposal_arrays(
        pool, position_seed_count=4, orientation_budget=8,
    )
    offsets = arrays["position_offsets_camera"]
    row = int(np.flatnonzero(np.all(offsets == [2.0, 0.0, 0.0], axis=1))[0])
    # R_w2c.T @ [2,0,0] = [0,-2,0] for +90 degree yaw.
    np.testing.assert_allclose(
        arrays["position_centers_world"][0, 0, row], [0.0, -2.0, 0.0],
        atol=1.0e-12,
    )
    assert arrays["position_offsets_camera"].shape == (605, 3)
    assert arrays["position_centers_world"].shape == (1, 4, 605, 3)
    assert arrays["orientation_rotations_w2c"].shape == (1, 8, 3, 3)
    assert "candidate_poses_w2c" not in arrays
    assert "target_pose_w2c" not in arrays


def test_factorized_builder_is_invariant_to_gt_trap_and_stably_deduplicates_orientation():
    first = _pool()
    second = _pool()
    second["rows"][0]["secret_query_gt_trap"] = {"different": True}
    a = build_factorized_pose_free_proposal_arrays(
        first, position_seed_count=4, orientation_budget=8,
    )
    b = build_factorized_pose_free_proposal_arrays(
        second, position_seed_count=4, orientation_budget=8,
    )
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert a["orientation_source_candidate_ranks"].tolist() == [
        [1, 2, 3, 4, 5, 6, 7, 8]
    ]


def test_factorized_coverage_is_exact_for_implicit_product_without_materialization():
    position = np.asarray([[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]])
    rotations = np.stack([_pose((0, 0, 0), 0)[:3, :3], _pose((0, 0, 0), 20)[:3, :3]])[None]
    valid = np.ones((1, 2), dtype=bool)
    target = _pose((2.0, 0.0, 0.0), 20.0)[None]
    result = factorized_raw_coverage(
        position, rotations, valid, target,
        position_seed_budgets=(1,), orientation_budgets=(1, 2),
    )
    strict_at_one = result["rows"][0]["strict_0_5m_5deg"]
    strict_at_two = result["rows"][1]["strict_0_5m_5deg"]
    assert strict_at_one["hits"] == 0
    assert strict_at_two["hits"] == 1
    assert result["full_factor_position_only"]["le_0.5m"]["hits"] == 1
    assert result["full_factor_orientation_only"]["le_5deg"]["hits"] == 1


def test_frozen_development_lattice_has_requested_bounds_and_unique_origin():
    offsets = camera_local_position_lattice_offsets()
    np.testing.assert_array_equal(np.min(offsets, axis=0), [-10.0, -4.0, -10.0])
    np.testing.assert_array_equal(np.max(offsets, axis=0), [10.0, 4.0, 10.0])
    assert np.sum(np.all(offsets == 0.0, axis=1)) == 1


def test_strict_loader_validates_geometry_ranks_padding_and_exact_members(tmp_path: Path):
    arrays = build_factorized_pose_free_proposal_arrays(
        _pool(), position_seed_count=4, orientation_budget=8,
    )
    valid_path = tmp_path / "valid.npz"
    _write_proposal(valid_path, arrays)
    loaded, _ = load_factorized_pose_free_proposal(valid_path)
    assert np.array_equal(loaded["image_ids"], arrays["image_ids"])

    mutations = []
    bad_rotation = {name: value.copy() for name, value in arrays.items()}
    bad_rotation["position_seed_poses_w2c"][0, 0, 0, 0] = 2.0
    mutations.append(bad_rotation)
    bad_last_row = {name: value.copy() for name, value in arrays.items()}
    bad_last_row["position_seed_poses_w2c"][0, 0, 3, 0] = 0.1
    mutations.append(bad_last_row)
    bad_center = {name: value.copy() for name, value in arrays.items()}
    bad_center["position_seed_centers_world"][0, 0, 0] += 1.0
    mutations.append(bad_center)
    bad_rank = {name: value.copy() for name, value in arrays.items()}
    bad_rank["orientation_source_candidate_ranks"][0, 1] = 1
    mutations.append(bad_rank)
    bad_valid = {name: value.copy() for name, value in arrays.items()}
    bad_valid["orientation_valid"][0, 1] = False
    mutations.append(bad_valid)
    for index, mutation in enumerate(mutations):
        path = tmp_path / f"invalid_{index}.npz"
        _write_proposal(path, mutation)
        with pytest.raises(ValueError):
            load_factorized_pose_free_proposal(path)

    extra = {name: value.copy() for name, value in arrays.items()}
    extra["unexpected"] = np.asarray([1])
    extra_path = tmp_path / "extra.npz"
    _write_proposal(extra_path, extra)
    with pytest.raises(ValueError, match="exact schema"):
        load_factorized_pose_free_proposal(extra_path)

    duplicate_path = tmp_path / "duplicate.npz"
    duplicate_path.write_bytes(valid_path.read_bytes())
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate_path, mode="a") as archive:
            with zipfile.ZipFile(valid_path, mode="r") as source:
                archive.writestr("image_ids.npy", source.read("image_ids.npy"))
    with pytest.raises(ValueError, match="duplicate ZIP members"):
        load_factorized_pose_free_proposal(duplicate_path)


def test_builder_rejects_duplicate_query_identity_and_non_so3_pose():
    duplicate = _pool()
    duplicate["rows"].append(dict(duplicate["rows"][0]))
    with pytest.raises(ValueError):
        build_factorized_pose_free_proposal_arrays(
            duplicate, position_seed_count=4, orientation_budget=8,
        )
    bad = _pool()
    bad["rows"][0]["mode_details"]["actual_parent_actual_child"][0][
        "pose_w2c"
    ][0][0] = -1.0
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        build_factorized_pose_free_proposal_arrays(
            bad, position_seed_count=4, orientation_budget=8,
        )
