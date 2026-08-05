import numpy as np

from feature_extract.vfm.localization_goal_maplet.latent_configuration import (
    capacitated_assignment,
    correlated_support_clusters,
    effective_group_weights,
    independent_assignment,
)


def test_correlated_supports_have_unit_cluster_mass():
    cluster = correlated_support_clusters(
        np.asarray([[10.0, 10.0], [10.5, 10.0], [40.0, 40.0]]),
        np.asarray([[5.0, 5.0], [5.0, 5.0], [3.0, 3.0]]),
    )
    assert cluster[0] == cluster[1]
    assert cluster[2] != cluster[0]
    weight = effective_group_weights(cluster)
    np.testing.assert_allclose(np.bincount(cluster, weights=weight), 1.0)


def test_correlated_supports_do_not_merge_through_overlap_chain():
    cluster = correlated_support_clusters(
        np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        np.asarray([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        minimum_iou=0.50,
    )
    assert cluster[0] == cluster[1]
    assert cluster[2] != cluster[0]


def test_independent_assignment_keeps_null_groups_in_denominator():
    result = independent_assignment(
        group_count=3,
        option_group_rows=np.asarray([0, 0, 1]),
        option_child_rows=np.asarray([4, 5, 6]),
        option_primitive_rows=np.asarray([10, 11, 12]),
        option_factor_rows=np.asarray([0, 1, 2]),
        option_scores=np.asarray([2.0, 1.0, 4.0]),
        option_valid_mass=np.asarray([0.7, 0.6, 0.1]),
    )
    np.testing.assert_array_equal(result.assigned, [True, False, False])
    np.testing.assert_array_equal(result.child_rows, [4, -1, -1])
    np.testing.assert_allclose(result.valid_mass, [0.7, 0.0, 0.0])


def test_assignments_handle_all_geometry_invalid_as_explicit_null():
    arguments = {
        "group_count": 2,
        "option_group_rows": np.zeros((0,), dtype=np.int64),
        "option_child_rows": np.zeros((0,), dtype=np.int64),
        "option_primitive_rows": np.zeros((0,), dtype=np.int64),
        "option_factor_rows": np.zeros((0,), dtype=np.int64),
        "option_scores": np.zeros((0,), dtype=np.float64),
        "option_valid_mass": np.zeros((0,), dtype=np.float64),
    }
    independent = independent_assignment(**arguments)
    capacity = capacitated_assignment(
        **arguments, group_cluster_rows=np.asarray([0, 1]), child_capacities={},
    )
    assert not np.any(independent.assigned)
    assert not np.any(capacity.assigned)
    np.testing.assert_array_equal(independent.child_rows, [-1, -1])
    np.testing.assert_array_equal(capacity.primitive_rows, [-1, -1])


def test_capacitated_assignment_prevents_independent_physical_reuse():
    # Groups 0 and 1 are correlated and may share primitive 10.  Group 2 is
    # independent and must fall back to primitive 12.  Child 4 has capacity
    # one independent cluster, so group 3 must use child 6.
    result = capacitated_assignment(
        group_count=4,
        option_group_rows=np.asarray([0, 1, 2, 2, 3, 3]),
        option_child_rows=np.asarray([4, 4, 4, 5, 4, 6]),
        option_primitive_rows=np.asarray([10, 10, 10, 12, 13, 14]),
        option_factor_rows=np.asarray([0, 1, 2, 3, 4, 5]),
        option_scores=np.asarray([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        option_valid_mass=np.asarray([0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),
        group_cluster_rows=np.asarray([0, 0, 1, 2]),
        child_capacities={4: 1, 5: 1, 6: 1},
    )
    np.testing.assert_array_equal(result.primitive_rows, [10, 10, 12, 14])
    np.testing.assert_array_equal(result.child_rows, [4, 4, 5, 6])
    assert np.all(result.assigned)
