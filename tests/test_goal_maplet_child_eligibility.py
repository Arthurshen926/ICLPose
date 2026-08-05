import numpy as np

from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.child_eligibility import build_child_geometry_eligibility
from test_goal_maplet_physical_map import _inputs


def test_child_eligibility_is_nested():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4)
    field = CanonicalSurfaceField(
        np.arange(physical.primitive_ids.size),
        np.eye(physical.primitive_ids.size, dtype=np.float32),
        np.ones(physical.primitive_ids.size, dtype=np.float32),
        np.zeros(physical.primitive_ids.size, dtype=np.float32),
        physical.content_sha256,
        metadata={"artifact_type": "goal_maplet_canonical_surface_field_v1"},
    )
    result = build_child_geometry_eligibility(physical, field)
    assert np.all(~result.proposal_qualified | result.retrieval_qualified)
    assert np.all(~result.refinement_qualified | result.proposal_qualified)
