from dataclasses import replace

import pytest

from feature_extract.vfm.localization_goal_maplet.connected_support_audit import (
    audit_connected_support_carrier,
)
from test_goal_maplet_pure_retrieval import _physical, _result


def test_connected_support_carrier_preserves_child_area_and_score():
    raw = _physical()
    physical = replace(
        raw, metadata={**raw.metadata, "child_voxel_size_m": 1.0}
    )
    report = audit_connected_support_carrier(_result(physical), physical)
    assert report["child_union_preserved_exactly"] is True
    assert report["surface_area_preserved"] is True
    assert report["score_preserved"] is True
    assert report["connected_component_count"] >= 1


def test_connected_support_carrier_rejects_duplicate_child_rows():
    raw = _physical()
    physical = replace(
        raw, metadata={**raw.metadata, "child_voxel_size_m": 1.0}
    )
    retrieval = _result(physical)
    duplicate = type(retrieval)(**{
        **retrieval.__dict__,
        "scene_child_rows": retrieval.scene_child_rows[[0, 0]],
        "scene_child_scores": retrieval.scene_child_scores[[0, 0]],
    })
    with pytest.raises(ValueError, match="invalid scene child set"):
        audit_connected_support_carrier(duplicate, physical)
