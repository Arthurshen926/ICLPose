import numpy as np

from feature_extract.vfm.localization_goal_maplet.geometry_complete_physical_map import (
    GEOMETRY_COMPLETE_PARTITION,
    build_geometry_complete_physical_map,
)
from test_goal_maplet_pure_retrieval import _physical


def test_geometry_complete_map_covers_every_primitive_once():
    source = _physical()
    result, audit = build_geometry_complete_physical_map(
        source, child_voxel_size_m=0.5, parent_voxel_size_m=2.0,
    )
    np.testing.assert_array_equal(result.primitive_ids, source.primitive_ids)
    assert np.all(
        np.bincount(
            result.membership_primitive_rows, minlength=result.primitive_ids.size
        ) == 1
    )
    assert np.all(
        np.bincount(
            result.child_member_primitive_rows, minlength=result.primitive_ids.size
        ) == 1
    )
    assert result.metadata["child_partition"] == GEOMETRY_COMPLETE_PARTITION
    assert result.metadata["uses_mapping_pose"] is False
    assert result.metadata["uses_query_ground_truth"] is False
    assert audit.unique_child_count == result.child_parent_rows.size


def test_geometry_complete_map_is_deterministic():
    source = _physical()
    first, first_audit = build_geometry_complete_physical_map(source)
    second, second_audit = build_geometry_complete_physical_map(source)
    assert first.content_sha256 == second.content_sha256
    assert first_audit.as_dict() == second_audit.as_dict()
