import numpy as np

from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.child_retrieval import retrieve_children_given_parents
from test_goal_maplet_physical_map import _inputs


def test_child_retrieval_is_conditioned_on_parent_identity():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4)
    descriptors = np.zeros((physical.child_parent_rows.size, 2), dtype=np.float32)
    descriptors[:, 1] = 1.0
    descriptors[0] = [1.0, 0.0]
    posterior = retrieve_children_given_parents(
        np.asarray([[1.0, 0.0]]),
        np.asarray([[7]]),
        np.asarray([[0.8]]),
        np.asarray([0.2]),
        descriptors,
        np.ones((descriptors.shape[0],)),
        physical,
        maximum_child_candidates=4,
        temperature=0.01,
    )
    assert posterior.candidate_child_rows[0, 0] == 0
    assert posterior.candidate_probabilities[0, 0] > 0.79
    np.testing.assert_allclose(posterior.null_probabilities[0], 0.2, atol=1e-5)
    assert posterior.conditional_parent_ids.shape == (1, 1)
    assert posterior.best_child_rows_by_parent[0, 0] == 0
    assert np.isfinite(posterior.conditional_parent_log_evidence[0, 0])
    assert posterior.best_child_probabilities_by_parent[0, 0] > 0.99
