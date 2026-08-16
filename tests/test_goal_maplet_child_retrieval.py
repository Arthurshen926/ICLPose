import numpy as np
from types import SimpleNamespace

from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.child_retrieval import (
    _rank_sparse_joint_topk,
    retrieve_children_given_parents,
)
from test_goal_maplet_physical_map import _inputs


def _reference_child_retrieval(
    local, parent_ids, parent_probability, parent_null, child_feature, coverage,
    physical, *, keep, temperature,
):
    local = local / np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1e-8)
    child_feature = child_feature / np.maximum(
        np.linalg.norm(child_feature, axis=1, keepdims=True), 1e-8
    )
    row_by_id = {int(value): row for row, value in enumerate(physical.maplet_ids)}
    rows = np.full((local.shape[0], keep), -1, dtype=np.int64)
    probability = np.zeros((local.shape[0], keep), dtype=np.float64)
    log_evidence = np.full(parent_ids.shape, -np.inf, dtype=np.float64)
    best_rows = np.full(parent_ids.shape, -1, dtype=np.int64)
    best_probability = np.zeros(parent_ids.shape, dtype=np.float64)
    alternatives = np.full(parent_ids.shape + (4,), -1, dtype=np.int64)
    alternative_probability = np.zeros(parent_ids.shape + (4,), dtype=np.float64)
    for token in range(local.shape[0]):
        accumulated = {}
        for slot, (parent_id, parent_mass) in enumerate(
            zip(parent_ids[token], parent_probability[token])
        ):
            parent = row_by_id.get(int(parent_id))
            if parent is None or parent_mass <= 0:
                continue
            start, end = physical.maplet_child_offsets[parent : parent + 2]
            children = np.arange(start, end, dtype=np.int64)
            children = children[coverage[children] > 0]
            if not children.size:
                continue
            scaled = child_feature[children] @ local[token] / temperature
            maximum = np.max(scaled)
            exponential = np.exp(scaled - maximum)
            conditional = exponential / np.sum(exponential)
            log_evidence[token, slot] = maximum + np.log(np.mean(exponential))
            order = np.argsort(-conditional, kind="stable")
            best_rows[token, slot] = children[order[0]]
            best_probability[token, slot] = conditional[order[0]]
            count = min(4, children.size)
            alternatives[token, slot, :count] = children[order[:count]]
            alternative_probability[token, slot, :count] = conditional[order[:count]]
            for child, value in zip(children, conditional):
                accumulated[int(child)] = accumulated.get(int(child), 0.0) + parent_mass * value
        ranked = sorted(accumulated.items(), key=lambda value: (-value[1], value[0]))[:keep]
        rows[token, : len(ranked)] = [value[0] for value in ranked]
        probability[token, : len(ranked)] = [value[1] for value in ranked]
    null = np.clip(np.maximum(parent_null, 1 - np.sum(probability, axis=1)), 0, 1)
    return rows, probability, null, log_evidence, best_rows, best_probability, alternatives, alternative_probability


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


def test_vectorized_child_retrieval_matches_scalar_probability_factorization():
    rng = np.random.default_rng(19)
    physical = SimpleNamespace(
        maplet_ids=np.asarray([10, 20], dtype=np.int64),
        maplet_child_offsets=np.asarray([0, 2, 5], dtype=np.int64),
        child_parent_rows=np.asarray([0, 0, 1, 1, 1], dtype=np.int64),
    )
    local = rng.normal(size=(7, 6)).astype(np.float32)
    child_feature = rng.normal(size=(5, 6)).astype(np.float32)
    parent_ids = np.asarray([[10, 20], [20, 10], [10, -1], [20, 10], [10, 20], [20, 10], [10, 20]])
    parent_probability = rng.uniform(0.05, 0.45, size=(7, 2))
    parent_null = 1.0 - np.sum(parent_probability, axis=1)
    coverage = np.asarray([1, 1, 1, 0, 1], dtype=np.float32)
    expected = _reference_child_retrieval(
        local, parent_ids, parent_probability, parent_null, child_feature,
        coverage, physical, keep=4, temperature=0.13,
    )
    actual = retrieve_children_given_parents(
        local, parent_ids, parent_probability, parent_null, child_feature,
        coverage, physical, maximum_child_candidates=4, temperature=0.13,
    )
    np.testing.assert_array_equal(actual.candidate_child_rows, expected[0])
    np.testing.assert_allclose(actual.candidate_probabilities, expected[1], atol=2e-7)
    np.testing.assert_allclose(actual.null_probabilities, expected[2], atol=2e-7)
    np.testing.assert_allclose(actual.conditional_parent_log_evidence, expected[3], atol=2e-6)
    np.testing.assert_array_equal(actual.best_child_rows_by_parent, expected[4])
    np.testing.assert_allclose(actual.best_child_probabilities_by_parent, expected[5], atol=2e-7)
    np.testing.assert_array_equal(actual.child_rows_by_parent, expected[6])
    np.testing.assert_allclose(actual.child_probabilities_by_parent, expected[7], atol=2e-7)


def test_sparse_topk_matches_global_lexsort_including_boundary_ties():
    token = np.repeat(np.arange(5, dtype=np.int64), [7, 3, 9, 1, 8])
    child = np.concatenate([
        np.arange(count, dtype=np.int64) for count in [7, 3, 9, 1, 8]
    ])
    probability = np.asarray(
        [0.4, 0.1, 0.4, 0.3, 0.2, 0.3, 0.4,
         0.2, 0.2, 0.1,
         0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2,
         0.9,
         0.1, 0.9, 0.4, 0.4, 0.4, 0.2, 0.8, 0.3],
        dtype=np.float64,
    )
    rows, scores = _rank_sparse_joint_topk(
        token, child, probability, token_count=5, keep=4,
    )
    for index in range(5):
        mask = token == index
        order = np.lexsort((child[mask], -probability[mask]))[:4]
        count = min(4, int(np.sum(mask)))
        np.testing.assert_array_equal(rows[index, :count], child[mask][order])
        np.testing.assert_array_equal(scores[index, :count], probability[mask][order])
        assert np.all(rows[index, count:] == -1)
