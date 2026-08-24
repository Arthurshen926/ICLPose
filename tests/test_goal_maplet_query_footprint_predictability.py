from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.build_goal_maplet_query_footprint_teacher_dataset import (
    SOURCE_SCHEMA,
    _estimated_source_uncompressed_bytes,
    _metric_log_depth,
    _retrieval_lineage,
    _select_direct_route_ids,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_query_footprint_predictability import (
    _apply_ridge,
    _assert_deployable_source_contract,
    _fit_fixed_ridge,
    _redistribute_token_mass,
    _teacher_token_targets,
    _token_features,
)


def _source_arrays() -> dict[str, np.ndarray]:
    return {
        "image_ids": np.asarray(["seq9/frame00000.png"]),
        "route_roles": np.asarray(["held"]),
        "radio_group_means": np.zeros((1, 2304, 32), dtype=np.float32),
        "source_child_rows": np.zeros((1, 2304, 2), dtype=np.int32),
        "source_child_probabilities": np.zeros((1, 2304, 2), dtype=np.float32),
        "query_reliability": np.ones((1, 2304), dtype=np.float32),
        "retrieval_content_sha256": np.asarray(["a" * 64]),
        "retrieval_file_sha256": np.asarray(["b" * 64]),
        "radio_file_sha256": np.asarray(["c" * 64]),
        "hierarchy_child_parent_ids": np.asarray([0], dtype=np.int32),
        "hierarchy_child_support_ids": np.asarray([0], dtype=np.int32),
    }


def _source_metadata() -> dict[str, object]:
    return {
        "artifact_type": SOURCE_SCHEMA,
        "contains_pose_or_GT_teacher_labels": False,
        "raw_RADIO_persisted": False,
        "retrieval_lineage": {"surface_mapper_file_sha256": "d" * 64},
    }


def test_direct_route_selection_matches_manifest_prefix_not_lexical_order() -> None:
    image_ids = [
        "seq8/frame00020.png", "seq9/frame00009.png",
        "seq8/frame00002.png", "seq8/frame00010.png",
    ]
    selected = _select_direct_route_ids(image_ids, set(image_ids), "seq8", 2)
    assert selected == ["seq8/frame00020.png", "seq8/frame00002.png"]
    with pytest.raises(ValueError):
        _select_direct_route_ids(image_ids, set(image_ids), "seq8", 4)


def test_v2_source_resource_bound_is_linear_and_sub_gib_for_1175_queries() -> None:
    one = _estimated_source_uncompressed_bytes(1, 16)
    large = _estimated_source_uncompressed_bytes(1175, 16)
    assert one > 8 * 1024 * 1024
    assert large < 1024 * 1024 * 1024
    assert large > one


def test_retrieval_lineage_requires_complete_coordinate_correct_chain() -> None:
    metadata = {
        "physical_map_file_sha256": "a" * 64,
        "canonical_field_file_sha256": "b" * 64,
        "canonical_field_coordinate_correct": True,
        "canonical_field_coordinate_contract": "contract",
        "surface_mapper_file_sha256": "c" * 64,
        "field_feature_contract_file_sha256": "d" * 64,
        "validity_calibration_file_sha256": "e" * 64,
        "parent_score_semantics": "parent",
        "parent_scene_ranking_semantics": "raw",
        "parent_mode_temperature": 0.03,
        "child_probability_semantics": "child",
    }
    assert _retrieval_lineage(metadata)["surface_mapper_file_sha256"] == "c" * 64
    metadata["canonical_field_coordinate_correct"] = False
    with pytest.raises(ValueError):
        _retrieval_lineage(metadata)


def test_metric_log_depth_is_absolute_and_masks_invalid_rows() -> None:
    rows = np.asarray([[0, 1, -1]], dtype=np.int64)
    pose = np.eye(4, dtype=np.float64)
    pose[2, 3] = 1.0
    centers = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -2.0]])
    depth, valid = _metric_log_depth(rows, pose, centers)
    assert valid.tolist() == [[True, False, False]]
    assert depth[0, 0] == pytest.approx(np.log(2.0))
    assert np.all(depth[~valid] == 0.0)


def test_deployable_source_contract_rejects_any_pose_or_teacher_member() -> None:
    arrays = _source_arrays()
    _assert_deployable_source_contract(arrays, _source_metadata())
    arrays["hidden_GT_pose"] = np.eye(4)
    with pytest.raises(ValueError):
        _assert_deployable_source_contract(arrays, _source_metadata())


def test_token_features_are_query_only_and_have_fixed_36_channels() -> None:
    radio = np.zeros((1, 1280, 36, 64), dtype=np.float32)
    radio[:, :40] = 2.0
    probability = np.zeros((1, 2304, 2), dtype=np.float32)
    probability[..., 0] = 0.6
    probability[..., 1] = 0.2
    reliability = np.full((1, 2304), 0.5, dtype=np.float32)
    features = _token_features(radio, probability, reliability)
    grouped = radio.reshape(1, 32, 40, 2304).mean(axis=2).transpose(0, 2, 1)
    grouped_features = _token_features(grouped, probability, reliability)
    assert features.shape == (1, 2304, 36)
    assert np.array_equal(features, grouped_features)
    assert np.allclose(features[..., 0], 0.8)
    assert np.allclose(features[..., 1], 0.5)
    assert np.allclose(features[..., 2], 0.75)
    assert np.allclose(features[..., 4], 2.0)
    assert np.allclose(features[..., 5:], 0.0)


def test_closed_form_ridge_improves_over_constant_and_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 20, 3))
    target = 0.3 + 0.4 * features[..., 0] - 0.2 * features[..., 2]
    model = _fit_fixed_ridge(features, target)
    repeated = _fit_fixed_ridge(features, target)
    prediction = _apply_ridge(model, features)
    assert all(np.array_equal(model[key], repeated[key]) for key in model)
    assert np.mean((prediction - target) ** 2) < 0.01 * np.var(target)


def test_mass_redistribution_preserves_support_and_clips_occupancy() -> None:
    probability = np.asarray([[[0.2, 0.6, 0.0], [0.0, 0.0, 0.0]]])
    result = _redistribute_token_mass(probability, np.asarray([[0.4, 2.0]]))
    assert result.shape == probability.shape
    assert np.allclose(result[0, 0], [0.1, 0.3, 0.0])
    assert np.all(result[0, 1] == 0.0)
    assert np.all((result > 0.0) <= (probability > 0.0))


def test_teacher_targets_use_projected_mass_and_weighted_metric_depth() -> None:
    weight = np.asarray([[[0.2, 0.3], [0.4, 0.0]]])
    depth = np.asarray([[[1.0, 3.0], [2.0, 9.0]]])
    valid = np.asarray([[[True, True], [True, False]]])
    occupancy, mean_depth, depth_valid = _teacher_token_targets(weight, depth, valid)
    assert np.allclose(occupancy, [[0.5, 0.4]])
    assert np.allclose(mean_depth, [[2.2, 2.0]])
    assert depth_valid.tolist() == [[True, True]]
