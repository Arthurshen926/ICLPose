from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.tools.vfm.build_goal_maplet_child_visibility_pose_atlas import (
    bind_selected_contributor_bytes,
    select_route_disjoint_contributors,
)
from feature_extract.tools.vfm.build_goal_maplet_direct_pose_candidate_dataset import (
    _frozen_pose_free_prefix,
)
from feature_extract.tools.vfm.build_goal_maplet_pose_free_visibility_candidate_pool import (
    _validate_route_disjoint_atlas,
)


def _detail(rank: int, x: float) -> dict[str, object]:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(x)
    return {"rank": rank, "pose_w2c": pose.tolist()}


def test_atlas_route_allowlist_filters_before_open_and_records_exact_inventory(tmp_path):
    for name in (
        "seq1__frame00001.png.npz",
        "seq1__frame00002.png.npz",
        "seq2__frame00001.png.npz",
        "seq12__frame00001.png.npz",
        "seq14__frame00001.png.npz",
    ):
        # Invalid NPZ bytes are intentional: selection must use filename
        # identity only and must not open pose-bearing contributor members.
        (tmp_path / name).write_bytes(b"not-an-npz")
    paths, audit = select_route_disjoint_contributors(tmp_path, ["seq1", "seq2"])
    assert [path.name for path in paths] == [
        "seq1__frame00001.png.npz",
        "seq1__frame00002.png.npz",
        "seq2__frame00001.png.npz",
    ]
    assert audit["route_allowlist_enforced"] is True
    assert audit["allowed_trajectories"] == ["seq1", "seq2"]
    assert audit["excluded_trajectories"] == ["seq12", "seq14"]
    assert audit["source_contributor_trajectory_counts"] == {"seq1": 2, "seq2": 1}
    assert audit["source_contributor_image_ids"] == [
        "seq1/frame00001.png", "seq1/frame00002.png", "seq2/frame00001.png",
    ]
    bound = bind_selected_contributor_bytes(paths, audit)
    assert bound["source_contributor_inventory_count"] == 3
    assert len(bound["source_contributor_inventory_sha256"]) == 64
    first_hash = bound["source_contributor_inventory_sha256"]
    paths[0].write_bytes(b"changed")
    assert bind_selected_contributor_bytes(paths, audit)[
        "source_contributor_inventory_sha256"
    ] != first_hash


def test_pool_route_disjoint_guard_rejects_query_route_or_tampered_inventory():
    metadata = {
        "route_allowlist_enforced": True,
        "allowed_trajectories": ["seq1", "seq2"],
        "source_contributor_trajectories": ["seq1", "seq2"],
        "source_contributor_trajectory_counts": {"seq1": 1, "seq2": 1},
        "source_contributor_image_ids": ["seq1/a.png", "seq2/b.png"],
        "source_contributor_inventory_count": 2,
        "source_contributor_inventory_sha256": "a" * 64,
        "source_contributor_inventory_semantics": (
            "ordered_image_id_resolved_path_file_sha256_v1"
        ),
        "coordinate_correct": True,
        "coordinate_contract": (
            "raw_simple_radial_equal_area_samples_inverse_warped_to_ideal_pinhole_"
            "contributor_grid_nearest_center_v1"
        ),
        "coordinate_transform_applied_before_visibility_aggregation": True,
        "coordinate_audit_count": 2,
        "coordinate_audits_sha256": "b" * 64,
    }
    from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256

    metadata["source_contributor_image_ids_sha256"] = canonical_json_sha256(
        metadata["source_contributor_image_ids"]
    )
    atlas = SimpleNamespace(metadata=metadata, view_count=2)
    audit = _validate_route_disjoint_atlas(atlas, query_route="seq14")
    assert audit["query_route_excluded_from_atlas"] is True
    with pytest.raises(ValueError, match="inconsistent"):
        _validate_route_disjoint_atlas(atlas, query_route="seq1")
    tampered = SimpleNamespace(
        metadata={**metadata, "source_contributor_image_ids": ["seq1/a.png", "seq14/x.png"]},
        view_count=2,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        _validate_route_disjoint_atlas(tampered, query_route="seq14")


def test_direct_prefix_keeps_pool_candidate_equal_to_later_gt_anchor():
    target = np.eye(4, dtype=np.float64)
    source = {
        "mode_details": {
            "actual_parent_actual_child": [_detail(1, 0.0), _detail(2, 2.0)]
        }
    }
    prefix = _frozen_pose_free_prefix(source, maximum_nonanchor=2)
    assert np.array_equal(prefix[0], target)
    joined = [target] + prefix
    assert len(joined) == 3
    assert np.array_equal(joined[0], joined[1])


def test_direct_prefix_rejects_duplicates_inside_pose_free_pool():
    source = {
        "mode_details": {
            "actual_parent_actual_child": [_detail(1, 1.0), _detail(2, 1.0)]
        }
    }
    with pytest.raises(ValueError, match="duplicate poses"):
        _frozen_pose_free_prefix(source, maximum_nonanchor=2)
