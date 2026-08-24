from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import (
    ALL_PARENT_UNION_SCHEMA,
    ALL_PARENT_UNION_SEMANTICS,
    all_parent_union_public_arrays,
    build_all_parent_union_support_arrays,
    build_global_physical_support_audit,
    global_physical_rectangle_aabb,
    load_all_parent_union_support,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256
from feature_extract.vfm.localization_goal_maplet.parent_geometry_pose_proposal import (
    orientation_cover_certificate,
)


class _Physical(SimpleNamespace):
    def member_slice(self, row):
        return slice(int(self.membership_offsets[row]), int(self.membership_offsets[row + 1]))


def _physical():
    angle = np.deg2rad(45.0)
    return _Physical(
        maplet_ids=np.asarray([10, 20], dtype=np.int64),
        primitive_centers=np.asarray([[0.0, 0.0, 0.0], [20.0, 4.0, 2.0]]),
        primitive_tangent1=np.asarray([
            [np.cos(angle), np.sin(angle), 0.0], [1.0, 0.0, 0.0],
        ]),
        primitive_tangent2=np.asarray([
            [-np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0],
        ]),
        primitive_scale1=np.asarray([2.0, 3.0]),
        primitive_scale2=np.asarray([1.0, 2.0]),
        membership_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        membership_primitive_rows=np.asarray([0, 1], dtype=np.int64),
    )


def test_global_aabb_is_single_union_box_of_exact_rotated_rectangles():
    lower, upper = global_physical_rectangle_aabb(_physical())
    rotated_extent = 3.0 / np.sqrt(2.0)
    assert np.allclose(lower, [-rotated_extent, -rotated_extent, 0.0])
    assert np.allclose(upper, [23.0, 4.0, 4.0])


def test_global_support_is_query_free_implicit_and_fails_cap_before_labels():
    report = build_global_physical_support_audit(
        _physical(), maximum_position_count=10,
    )
    assert report["uses_query_pose"] is False
    assert report["uses_query_ground_truth"] is False
    assert report["uses_query_retrieval"] is False
    assert report["query_independent_positions_shared"] is True
    assert report["cartesian_product_materialized"] is False
    assert report["implicit_pose_factor_count"] == report["position_count"] * 60
    assert report["cell_cover_radius_m"] == np.sqrt(3.0)
    assert report["position_count"] > 10
    assert report["structural_gate"] == {
        "decision": "KILL",
        "reason": "position_hard_cap_exceeded_before_label_read",
        "seq10_pose_labels_read": False,
    }


def test_global_support_passes_a_sufficiently_large_structural_cap():
    report = build_global_physical_support_audit(
        _physical(), maximum_position_count=10_000,
    )
    assert report["position_count_within_cap"] is True
    assert report["structural_gate"]["decision"] == "GO_TO_SEQ10_RAW_SUPPORT"
    assert report["ram_bytes"]["minimal_factored_indices_plus_orientation"] == (
        report["position_count"] * 3 * 4 + 60 * 3 * 3 * 8
    )


def test_all_parent_union_is_deduplicated_shared_and_cap_is_fail_closed():
    internal = build_all_parent_union_support_arrays(
        _physical(), maximum_position_count=10_000,
    )
    arrays = all_parent_union_public_arrays(internal)
    cells = arrays["cell_indices_world"]
    assert cells.shape[0] < int(internal["_raw_parent_cell_count"])
    assert np.unique(cells, axis=0).shape[0] == cells.shape[0]
    assert int(arrays["implicit_pose_factor_count"]) == cells.shape[0] * 60
    assert np.max(arrays["cell_parent_support_count"]) >= 2
    with np.testing.assert_raises_regex(ValueError, "hard cap exceeded"):
        build_all_parent_union_support_arrays(_physical(), maximum_position_count=10)


def test_all_parent_union_npz_round_trip_and_duplicate_member_trap(tmp_path):
    internal = build_all_parent_union_support_arrays(
        _physical(), maximum_position_count=10_000,
    )
    arrays = all_parent_union_public_arrays(internal)
    metadata = {
        "artifact_type": ALL_PARENT_UNION_SCHEMA,
        "semantics": ALL_PARENT_UNION_SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "uses_query_retrieval": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_mapping_camera_position_seed": False,
        "query_independent_positions_shared": True,
        "parent_retrieval_cuts_domain": False,
        "cartesian_product_materialized": False,
        "orientation_cover_certificate": orientation_cover_certificate(),
        "position_hard_cap": 10_000,
        "position_count": int(arrays["cell_indices_world"].shape[0]),
        "orientation_count": 60,
        "implicit_pose_factor_count": int(arrays["implicit_pose_factor_count"]),
    }
    path = tmp_path / "proposal.npz"
    np.savez_compressed(
        path, **arrays,
        metadata_json=np.asarray(__import__("json").dumps(metadata, sort_keys=True)),
    )
    loaded, loaded_metadata = load_all_parent_union_support(path)
    assert loaded_metadata == metadata
    assert arrays_sha256(loaded) == metadata["content_sha256"]

    import zipfile
    duplicate = tmp_path / "duplicate.npz"
    duplicate.write_bytes(path.read_bytes())
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "a") as archive:
            archive.writestr("maplet_ids.npy", b"duplicate")
    with np.testing.assert_raises_regex(ValueError, "members differ"):
        load_all_parent_union_support(duplicate)
