import numpy as np

from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.child_local_likelihood import (
    predict_child_local_surface_likelihood,
)
from test_goal_maplet_physical_map import _inputs


def test_child_local_likelihood_is_conditioned_and_multimodal():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(
        maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4
    )
    codes = np.eye(physical.primitive_ids.size, dtype=np.float32)
    field = CanonicalSurfaceField(
        np.arange(physical.primitive_ids.size), codes,
        np.ones(physical.primitive_ids.size, dtype=np.float32),
        np.zeros(physical.primitive_ids.size, dtype=np.float32),
        physical.content_sha256,
        metadata={"artifact_type": "goal_maplet_canonical_surface_field_v1"},
    )
    child = 0
    start, end = physical.child_member_offsets[child : child + 2]
    primitive = physical.child_member_primitive_rows[int(start) : int(end)]
    target = int(primitive[-1])
    result = predict_child_local_surface_likelihood(
        codes[target, None], np.asarray([child]), physical, field,
        temperature=0.01, maximum_modes=4,
    )
    assert result.mode_primitive_rows[0, 0] == target
    np.testing.assert_allclose(result.map_points[0], physical.primitive_centers[target])
    assert np.isclose(np.sum(result.mode_probabilities[0]), 1.0, atol=1e-5)
    assert result.null_probabilities[0] == 0.0
