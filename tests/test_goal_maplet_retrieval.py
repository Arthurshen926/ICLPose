import json

import numpy as np

from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalFieldFusionAccumulator
from feature_extract.vfm.localization_goal_maplet.retrieval import (
    ValidityCalibration,
    fit_validity_calibration,
    retrieve_maplet_posterior,
    retrieve_maplet_posterior_decomposed,
)
from test_goal_maplet_physical_map import _inputs


def test_validity_calibration_is_monotonic_and_roundtrips(tmp_path):
    score = np.linspace(0.0, 1.0, 200)
    target = 1.0 / (1.0 + np.exp(-(score - 0.65) / 0.07))
    calibration = fit_validity_calibration(score, target, metadata={"fold": "validation"})
    assert abs(calibration.center - 0.65) < 1e-3
    assert abs(calibration.scale - 0.07) < 1e-3
    assert np.all(np.diff(calibration.predict_valid(score)) > 0.0)
    path = tmp_path / "calibration.json"
    calibration.save_json(path)
    loaded = type(calibration).load_json(path)
    assert loaded.content_sha256 == calibration.content_sha256


def test_validity_calibration_loads_lineaged_training_report(tmp_path):
    calibration = ValidityCalibration(
        center=0.5,
        scale=0.2,
        metadata={"artifact_type": "goal_maplet_validity_calibration_v1", "fold": "a"},
    )
    path = tmp_path / "calibration_report.json"
    path.write_text(json.dumps({
        "center": calibration.center,
        "scale": calibration.scale,
        "calibration_metadata": calibration.metadata,
        "calibration_sha256": calibration.content_sha256,
        "stage": "calibrate_goal_maplet_validity",
    }))
    loaded = ValidityCalibration.load_json(path)
    assert loaded.content_sha256 == calibration.content_sha256


def test_exact_contributor_canonical_fusion_is_view_balanced():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4)
    accumulator = CanonicalFieldFusionAccumulator(physical.primitive_ids.size, 2)
    accumulator.add_view(np.asarray([0, 1]), np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([100.0, 1.0]))
    accumulator.add_view(np.asarray([0]), np.asarray([[0.0, 1.0]]), np.asarray([1.0]))
    field = accumulator.finalize(physical, metadata={"fusion": "test"})
    assert field.primitive_rows.tolist() == [0, 1]
    assert field.metadata["stored_downstream_embedding_count"] == 0
    # The footprint cap prevents the first close-up view from receiving 100x
    # the influence of the second view.
    assert field.codes[0, 1] > 0.2


def test_offline_teacher_quality_downweights_but_does_not_remove_view():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(
        maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4,
    )
    accumulator = CanonicalFieldFusionAccumulator(physical.primitive_ids.size, 2)
    accumulator.add_view(
        np.asarray([0]), np.asarray([[1.0, 0.0]]), np.asarray([1.0]),
        observation_quality=np.asarray([1.0]),
    )
    accumulator.add_view(
        np.asarray([0]), np.asarray([[0.0, 1.0]]), np.asarray([1.0]),
        observation_quality=np.asarray([0.25]),
    )
    field = accumulator.finalize(physical, metadata={"fusion": "teacher_quality_test"})
    assert field.primitive_rows.tolist() == [0]
    assert field.codes[0, 0] > field.codes[0, 1] > 0.0
    assert accumulator.view_count[0] == 2


def test_sparse_retrieval_separates_out_of_map_from_truncated_in_map_tail():
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    map_feature = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]], dtype=np.float32)
    posterior = retrieve_maplet_posterior_decomposed(
        query,
        map_feature,
        np.asarray([10, 11, 12]),
        np.ones((3,), dtype=bool),
        maximum_candidates=1,
        temperature=0.2,
        null_similarity_center=0.5,
        null_similarity_scale=0.1,
    )
    assert posterior.out_of_map_probabilities[0] < 0.01
    assert posterior.truncated_tail_probabilities[0] > 0.5
    np.testing.assert_allclose(
        posterior.candidate_probabilities.sum(axis=1)
        + posterior.out_of_map_probabilities
        + posterior.truncated_tail_probabilities,
        1.0,
        atol=1e-6,
    )
    ids, probability, legacy_null, best = retrieve_maplet_posterior(
        query,
        map_feature,
        np.asarray([10, 11, 12]),
        np.ones((3,), dtype=bool),
        maximum_candidates=1,
        temperature=0.2,
        null_similarity_center=0.5,
        null_similarity_scale=0.1,
    )
    np.testing.assert_array_equal(ids, posterior.candidate_ids)
    np.testing.assert_allclose(probability, posterior.candidate_probabilities)
    np.testing.assert_allclose(legacy_null, posterior.unresolved_probabilities)
    np.testing.assert_allclose(best, posterior.best_similarities)
