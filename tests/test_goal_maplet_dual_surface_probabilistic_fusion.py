import numpy as np

from feature_extract.tools.vfm.select_goal_maplet_dual_surface_probabilistic_fusion import (
    _select,
    _token_marginal_likelihood,
)


def test_token_likelihood_does_not_double_count_hypotheses():
    pose = np.eye(4)
    world = np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    tokens = np.asarray([0, 0, 1])
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    score, per_token = _token_marginal_likelihood(pose, world, tokens, K, 0.0)
    assert per_token.shape == (2,)
    assert np.isclose(score, np.mean(per_token))
    assert per_token[0] == 1.0


def test_token_likelihood_consumes_explicit_subtoken_measurement():
    pose = np.eye(4)
    world = np.asarray([[0.0, 0.0, 2.0]])
    tokens = np.asarray([0])
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    centered, _ = _token_marginal_likelihood(pose, world, tokens, K, 0.0)
    shifted, _ = _token_marginal_likelihood(
        pose, world, tokens, K, 0.0, np.asarray([[0.5, 1.5]]),
    )
    assert centered == 1.0
    assert shifted < centered


def test_cross_scale_soft_selection_and_stable_tie():
    score = np.asarray([[[0.8, 0.6], [0.7, 0.8]], [[0.5, 0.5], [0.5, 0.5]]])
    assert _select(score).tolist() == [1, 0]
