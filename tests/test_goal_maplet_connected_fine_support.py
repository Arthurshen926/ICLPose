import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.connected_fine_support import (
    connected_fine_support_components,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from test_goal_maplet_pure_retrieval import _physical


def _with_voxel_contract(physical, child_size=1.0):
    metadata = dict(physical.metadata)
    metadata.pop("content_sha256", None)
    metadata["child_voxel_size_m"] = child_size
    return GoalMapletPhysicalMap(
        **{
            **physical.__dict__,
            "metadata": metadata,
        }
    )


def test_connected_support_preserves_exact_child_union_and_is_deterministic():
    physical = _with_voxel_contract(_physical())
    selected = np.arange(physical.child_parent_rows.size, dtype=np.int64)
    left = connected_fine_support_components(selected, physical)
    right = connected_fine_support_components(selected[::-1], physical)
    np.testing.assert_array_equal(left.component_offsets, right.component_offsets)
    np.testing.assert_array_equal(left.component_child_rows, right.component_child_rows)
    assert set(left.component_child_rows.tolist()) == set(selected.tolist())
    assert 1 <= left.component_count <= selected.size
    assert np.sum(left.component_surface_area_m2) > 0.0


def test_empty_connected_support_is_typed_and_missing_voxel_contract_rejects():
    physical = _with_voxel_contract(_physical())
    empty = connected_fine_support_components(
        np.zeros(0, dtype=np.int64), physical
    )
    assert empty.component_count == 0
    with pytest.raises(ValueError, match="invalid connected fine-support input"):
        connected_fine_support_components(
            np.asarray([0], dtype=np.int64), _with_voxel_contract(_physical(), 0.0)
        )
