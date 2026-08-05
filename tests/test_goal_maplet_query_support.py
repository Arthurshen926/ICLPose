import numpy as np

from feature_extract.vfm.localization_goal_maplet.query_support import (
    aggregate_group_posteriors,
    aggregate_group_descriptors,
    all_token_coordinates,
    group_tokens_after_retrieval,
)


def test_grouping_is_after_identity_and_never_links_disconnected_repetition():
    _, xy = all_token_coordinates(2, 3)
    descriptor = np.tile([[1.0, 0.0]], (6, 1))
    identity = np.asarray([1, 1, 2, 3, 3, 1])
    grouped = group_tokens_after_retrieval(
        xy,
        descriptor,
        identity,
        token_height=2,
        token_width=3,
        image_width=300,
        image_height=200,
        descriptor_half_size_tokens=0.5,
    )
    sizes = np.diff(grouped.member_offsets)
    assert sorted(sizes.tolist()) == [1, 1, 2, 2]
    assert grouped.xy.shape == (4, 2)


def test_group_posterior_preserves_first_retrieval_and_averages_correlated_tokens():
    ids = np.asarray([[7, 8], [7, 9], [4, 7]])
    probability = np.asarray([[0.7, 0.2], [0.5, 0.3], [0.6, 0.2]])
    null = np.asarray([0.1, 0.2, 0.2])
    output_ids, output_probability, output_null = aggregate_group_posteriors(
        ids,
        probability,
        null,
        np.asarray([0, 2, 3]),
        np.asarray([0, 1, 2]),
        maximum_candidates=3,
    )
    assert output_ids[0].tolist() == [7, 9, 8]
    np.testing.assert_allclose(output_probability[0], [0.6, 0.15, 0.1])
    # The dropped posterior tail is correctly moved into unknown mass.
    np.testing.assert_allclose(output_null[0], 0.15)
    assert output_ids[1, 0] == 4


def test_group_descriptors_are_normalized_means():
    output = aggregate_group_descriptors(
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        np.asarray([0, 2, 3]),
        np.asarray([0, 1, 2]),
    )
    np.testing.assert_allclose(output[0], np.sqrt(0.5) * np.ones(2))
    np.testing.assert_allclose(output[1], [-1.0, 0.0])
