import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from feature_extract.vfm.localization_goal_maplet.mode_relation import (
    EDGE_FAMILIES,
    FEATURE_NAMES,
    ModeRelationLikelihoodRatioArtifact,
    analytic_relation_score,
    build_sparse_relation_edges,
    exact_pair_log_marginal,
    family_preserving_child_shortlist,
    max_sum_forest,
    sum_product_forest,
)


def test_relation_edges_are_query_only_tree_and_disjoint_verification():
    xy = np.asarray([[0, 0], [10, 0], [20, 5], [100, 20], [60, 80]], dtype=np.float64)
    extent = np.full((5, 2), 2.0)
    descriptor = np.eye(5, dtype=np.float32)
    scale = np.asarray([8, 9, 12, 25, 10], dtype=np.float64)
    edges = build_sparse_relation_edges(xy, extent, descriptor, scale)
    assert edges.fit_left.size == 4
    fit = {tuple(sorted(edge)) for edge in zip(edges.fit_left, edges.fit_right)}
    verify = {tuple(sorted(edge)) for edge in zip(edges.verify_left, edges.verify_right)}
    assert fit.isdisjoint(verify)
    assert set(edges.fit_family.tolist()).issubset(set(range(len(EDGE_FAMILIES))))
    assert 0 in edges.fit_family


def test_analytic_relation_prefers_small_geometric_residual():
    feature = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
    index = {name: row for row, name in enumerate(FEATURE_NAMES)}
    feature[:, index["direction_cosine"]] = 1.0
    feature[:, index["depth_order_agreement"]] = 1.0
    feature[1, index["normalized_vector_residual"]] = 4.0
    score = analytic_relation_score(feature, np.ones((2,), dtype=bool))
    assert score[0] > score[1]


def test_max_sum_tree_is_exact_and_supports_heterogeneous_states():
    unary = [np.asarray([0.0, 1.0]), np.asarray([0.0, 0.5, 0.0]), np.asarray([0.2, 0.0])]
    left = np.asarray([0, 1])
    right = np.asarray([1, 2])
    pair = [
        np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
    ]
    result = max_sum_forest(unary, left, right, pair)
    brute = []
    for a in range(2):
        for b in range(3):
            for c in range(2):
                brute.append((unary[0][a] + unary[1][b] + unary[2][c] + pair[0][a, b] + pair[1][b, c], (a, b, c)))
    expected = max(brute)
    assert result.score == pytest.approx(expected[0])
    assert tuple(result.state_rows.tolist()) == expected[1]


def test_max_sum_rejects_cycles():
    unary = [np.zeros(2), np.zeros(2), np.zeros(2)]
    pair = [np.zeros((2, 2))] * 3
    with pytest.raises(ValueError, match="cycle"):
        max_sum_forest(unary, np.asarray([0, 1, 2]), np.asarray([1, 2, 0]), pair)
    with pytest.raises(ValueError, match="cycle"):
        sum_product_forest(unary, np.asarray([0, 1, 2]), np.asarray([1, 2, 0]), pair)


def test_sum_product_tree_matches_brute_force_partition_and_marginal():
    unary = [np.log(np.asarray([0.4, 0.6])), np.log(np.asarray([0.7, 0.2, 0.1]))]
    pair = [np.asarray([[0.0, -0.3, 0.2], [0.1, 0.4, -0.2]])]
    result = sum_product_forest(unary, np.asarray([0]), np.asarray([1]), pair)
    joint = unary[0][:, None] + unary[1][None, :] + pair[0]
    maximum = np.max(joint)
    expected_logz = maximum + np.log(np.sum(np.exp(joint - maximum)))
    expected_left = np.log(np.sum(np.exp(joint - expected_logz), axis=1))
    assert result.log_partition == pytest.approx(expected_logz)
    assert np.exp(result.node_log_marginals[0]) == pytest.approx(np.exp(expected_left))


def test_exact_nonadjacent_pair_marginal_matches_brute_force():
    unary = [
        np.log(np.asarray([0.4, 0.6])),
        np.log(np.asarray([0.7, 0.2, 0.1])),
        np.log(np.asarray([0.3, 0.7])),
    ]
    left, right = np.asarray([0, 1]), np.asarray([1, 2])
    pair = [
        np.asarray([[0.0, -0.3, 0.2], [0.1, 0.4, -0.2]]),
        np.asarray([[0.2, -0.1], [-0.4, 0.3], [0.0, 0.5]]),
    ]
    result = sum_product_forest(unary, left, right, pair)
    exact = exact_pair_log_marginal(result, unary, left, right, pair, 0, 2)
    joint = np.full((2, 2), -np.inf)
    for a in range(2):
        for c in range(2):
            joint[a, c] = np.log(np.sum(np.exp([
                unary[0][a] + unary[1][b] + unary[2][c]
                + pair[0][a, b] + pair[1][b, c]
                for b in range(3)
            ])))
    joint -= np.log(np.sum(np.exp(joint)))
    assert np.exp(exact) == pytest.approx(np.exp(joint))


def test_pair_marginal_is_stable_under_tree_reparameterization():
    unary = [np.asarray([0.2, -0.1]), np.asarray([0.0, 0.3]), np.asarray([-0.2, 0.4])]
    left, right = np.asarray([0, 1]), np.asarray([1, 2])
    pair = [np.asarray([[0.1, -0.2], [0.3, 0.0]]), np.asarray([[0.0, 0.2], [-0.1, 0.4]])]
    original = sum_product_forest(unary, left, right, pair)
    original_pair = exact_pair_log_marginal(original, unary, left, right, pair, 0, 2)
    delta = np.asarray([0.7, -0.4])
    changed_unary = [unary[0] + delta, unary[1].copy(), unary[2].copy()]
    changed_pair = [pair[0] - delta[:, None], pair[1].copy()]
    changed = sum_product_forest(changed_unary, left, right, changed_pair)
    changed_endpoint = exact_pair_log_marginal(
        changed, changed_unary, left, right, changed_pair, 0, 2,
    )
    assert changed.log_partition == pytest.approx(original.log_partition)
    assert np.exp(changed_endpoint) == pytest.approx(np.exp(original_pair))


def test_relation_supports_use_complete_link_and_quality_representative():
    # A overlaps B and B overlaps C, while A/C do not overlap enough.  The
    # retired connected component collapses all three; complete-link keeps C.
    xy = np.asarray([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [40.0, 0.0]])
    extent = np.asarray([[5.0, 5.0]] * 4)
    descriptor = np.eye(4, dtype=np.float32)
    scale = np.full((4,), 8.0)
    edges = build_sparse_relation_edges(
        xy, extent, descriptor, scale,
        query_priority=np.asarray([0.2, 0.9, 0.5, 0.4]),
    )
    assert np.unique(edges.support_cluster_rows).size > np.unique(
        edges.legacy_connected_cluster_rows,
    ).size
    # Highest-quality B represents the first complete-link support cluster.
    assert 1 in edges.representative_groups


def test_relation_artifact_contract_and_scalar_llr(tmp_path):
    x = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, 0] = [-2.0, -1.0, 1.0, 2.0]
    estimator = LogisticRegression(random_state=0).fit(x, np.asarray([1, 1, 0, 0]))
    artifact = ModeRelationLikelihoodRatioArtifact(
        estimator, 1.25, -0.5,
        {
            "artifact_type": "goal_maplet_mode_relation_likelihood_ratio_v1",
            "feature_names": list(FEATURE_NAMES),
            "pairing_contract": "same_image_same_query_edge_fixed_options_v1",
            "edge_contract": "query_only_fit_tree_disjoint_verify_v1",
        },
    )
    path = tmp_path / "relation.joblib"
    artifact.save(path)
    loaded = ModeRelationLikelihoodRatioArtifact.load(path)
    score = loaded.score_log_likelihood_ratio(x)
    assert score.shape == (4,)
    assert np.all(np.isfinite(score))
    assert loaded.score_log_likelihood_ratio(np.zeros((0, len(FEATURE_NAMES)))).shape == (0,)


def test_family_shortlist_prevents_one_parent_from_filling_every_slot():
    class Physical:
        child_parent_rows = np.asarray([0, 0, 0, 1, 2], dtype=np.int64)

    child = np.asarray([[0, 1, 2, 3, 4]], dtype=np.int64)
    probability = np.asarray([[0.40, 0.30, 0.20, 0.06, 0.04]])
    mask = family_preserving_child_shortlist(
        child, probability, Physical(), maximum_children=4, maximum_per_parent=2,
    )
    assert mask.tolist() == [[True, True, False, True, True]]
