import numpy as np

from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
)
from feature_extract.vfm.localization_goal_maplet.multimodal_parent_retrieval import (
    AnonymousParentModeReadout,
    build_anonymous_parent_mode_readout,
    score_anonymous_parent_modes,
)
from feature_extract.vfm.localization_goal_maplet.retrieval import (
    retrieve_maplet_posterior_from_scores,
)
from test_goal_maplet_physical_map import _inputs


def _physical():
    geometry, maplets, region, poses = _inputs()
    return build_goal_maplet_physical_map(
        maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4
    )


def test_area_normalized_modes_are_invariant_to_identical_mode_duplication():
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    single = AnonymousParentModeReadout(
        descriptors=np.asarray([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        weights=np.asarray([[1.0, 0.0]], dtype=np.float32),
        parent_coverage=np.asarray([1.0], dtype=np.float32),
    )
    duplicate = AnonymousParentModeReadout(
        descriptors=np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32),
        weights=np.asarray([[0.5, 0.5]], dtype=np.float32),
        parent_coverage=np.asarray([1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        score_anonymous_parent_modes(query, single),
        score_anonymous_parent_modes(query, duplicate),
        atol=1e-7,
    )


def test_multimode_parent_keeps_one_physical_softmax_event():
    readout = AnonymousParentModeReadout(
        descriptors=np.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.8, 0.6], [0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        weights=np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype=np.float32),
        parent_coverage=np.asarray([1.0, 1.0], dtype=np.float32),
    )
    score = score_anonymous_parent_modes(
        np.asarray([[1.0, 0.0]], dtype=np.float32), readout, mode_temperature=0.03
    )
    assert score[0, 0] > score[0, 1]
    posterior = retrieve_maplet_posterior_from_scores(
        score,
        np.asarray([10, 20], dtype=np.int64),
        np.asarray([True, True]),
        maximum_candidates=2,
        temperature=0.07,
        null_similarity_center=0.0,
        null_similarity_scale=0.2,
    )
    assert posterior.candidate_ids.tolist() == [[10, 20]]
    assert np.unique(posterior.candidate_ids).size == 2


def test_mode_builder_is_deterministic_and_parent_normalized():
    physical = _physical()
    rows = np.arange(physical.primitive_ids.size, dtype=np.int64)
    codes = np.eye(max(rows.size, 2), dtype=np.float32)[rows, :2]
    codes[np.linalg.norm(codes, axis=1) == 0.0] = np.asarray([1.0, 1.0])
    field = CanonicalSurfaceField(
        primitive_rows=rows,
        codes=codes,
        confidence=np.ones(rows.size, dtype=np.float32),
        uncertainty=np.zeros(rows.size, dtype=np.float32),
        physical_map_sha256=physical.content_sha256,
        metadata={
            "artifact_type": "goal_maplet_canonical_surface_field_v1",
            "stored_downstream_embedding_count": 0,
        },
    )
    first = build_anonymous_parent_mode_readout(field, physical, maximum_modes=2)
    second = build_anonymous_parent_mode_readout(field, physical, maximum_modes=2)
    assert first.content_sha256 == second.content_sha256
    np.testing.assert_allclose(
        np.sum(first.weights, axis=1)[first.parent_coverage > 0.0], 1.0
    )
