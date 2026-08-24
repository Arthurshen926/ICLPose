from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    _controlled_monotonic_pair_rows,
    _fixed_identity_scores_from_component_statistics,
    _full_6dof_direction_metrics,
    _identity_edge_gradient_mask,
    _mapper_supervision_audit,
    _metrics,
    _rankdata,
    _validate_scientific_dataset_contract,
)
from feature_extract.vfm.localization_goal_maplet.trainable_pose_transport import (
    MinimalPoseTransportConfig,
    MinimalPoseTransportReadout,
)


def _controlled_errors(query_count: int = 1) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray([
        0.0, 0.25, 0.25, 0.50, 0.50, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.50, 0.50, 1.0, 1.0, 2.0, 2.0,
    ], dtype=np.float32)
    rotation = np.asarray([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        5.0, 5.0, 10.0, 10.0, 15.0, 15.0,
        5.0, 5.0, 10.0, 10.0, 20.0, 20.0,
    ], dtype=np.float32)
    return (
        np.repeat(translation[None], query_count, axis=0),
        np.repeat(rotation[None], query_count, axis=0),
    )


def _minimal_scientific_arrays() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    query_count, candidate_count = 2, 19
    translation, rotation = _controlled_errors(query_count)
    arrays = {
        "image_ids": np.asarray([
            "seq12/frame00001.png", "seq12/frame00002.png",
        ]),
        "radio_final": np.zeros((query_count, 1), dtype=np.float32),
        "source_child_rows": np.zeros((query_count, 1), dtype=np.int32),
        "source_child_probabilities": np.ones((query_count, 1), dtype=np.float32),
        "query_reliability": np.ones((query_count, 1), dtype=np.float32),
        "token_xy": np.zeros((query_count, 1, 2), dtype=np.int16),
        "candidate_poses_w2c": np.repeat(
            np.eye(4, dtype=np.float64)[None, None],
            query_count * candidate_count, axis=0,
        ).reshape(query_count, candidate_count, 4, 4),
        "translation_m": translation,
        "rotation_deg": rotation,
        "candidate_valid": np.ones((query_count, candidate_count), dtype=bool),
    }
    for name in (
        "target_child_rows", "target_child_weights", "target_canonical_features",
        "target_normals_camera", "target_double_sided", "target_relative_depth",
        "target_boundary", "target_modality_valid", "target_modality_confidence",
    ):
        arrays[name] = np.zeros((query_count, candidate_count, 1), dtype=np.float32)
    metadata = {
        "query_route": "seq12",
        "query_count": query_count,
        "candidate_count": candidate_count,
        "map_training_routes": ["seq1", "seq11"],
        "candidate_zero_is_diagnostic_gt_anchor": True,
        "pose_errors_recomputed": True,
        "candidate_semantics": "controlled_local_oracle_v1",
        "controlled_candidates_are_gt_relative_oracle_diagnostic": True,
    }
    return arrays, metadata


def test_controlled_monotonic_pairs_preserve_six_independent_radial_paths():
    translation, rotation = _controlled_errors()
    pairs = _controlled_monotonic_pair_rows(
        np.ones(19, dtype=bool), translation[0], rotation[0],
        candidate_semantics="controlled_local_oracle_v1",
    )
    assert pairs.shape == (18, 3)
    assert [0, 1, 3] in pairs.tolist()
    assert [0, 2, 4] in pairs.tolist()
    # Equal-error +/- candidates must not acquire a label by array order.
    assert [0, 1, 2] not in pairs.tolist()
    assert [0, 7, 8] not in pairs.tolist()


def test_controlled_monotonic_contract_rejects_non_outward_path():
    translation, rotation = _controlled_errors()
    translation[0, 3] = translation[0, 1]
    with pytest.raises(ValueError, match="strictly away"):
        _controlled_monotonic_pair_rows(
            np.ones(19, dtype=bool), translation[0], rotation[0],
            candidate_semantics="controlled_local_oracle_v1",
        )


def test_embedded_full_6dof_paths_drive_dynamic_pairs_and_direction_metrics():
    paths = np.asarray([
        [0, 1, 2], [0, 3, 4], [0, 5, 6], [0, 7, 8],
    ], dtype=np.int64)
    translation = np.asarray([0.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0])
    rotation = np.zeros_like(translation)
    pairs = _controlled_monotonic_pair_rows(
        np.ones(9, dtype=bool), translation, rotation,
        candidate_semantics="controlled_medium_6dof_quadratic_oracle_v1",
        radial_paths=paths,
    )
    assert pairs.shape == (8, 3)
    np.testing.assert_array_equal(pairs[:2], [[0, 0, 1], [0, 1, 2]])

    scores = -translation[None]
    result = _full_6dof_direction_metrics(
        scores, np.ones_like(scores, dtype=bool), paths,
        np.asarray([-1, 0, 0, 0, 0, 1, 1, 1, 1]),
        np.asarray([0, -1, -1, 1, 1, -1, -1, 1, 1]),
        np.asarray([[0, -1], [0, 1]]),
        twist_order=("r_x", "r_y", "r_z", "t_x", "t_y", "t_z"),
    )
    assert result["direction_count"] == 2
    assert result["coordinate_axes"]["radial_pair_accuracy"] == 1.0
    assert result["pair_couplings"]["complete_path_rate"] == 1.0
    assert result["per_direction"][1]["label"] == "r_x+r_y"


def test_metrics_report_stage_and_controlled_directional_violations():
    translation, rotation = _controlled_errors()
    score = -np.maximum(translation, rotation / 15.0)
    metrics = _metrics(
        score, translation, rotation, np.ones((1, 19), dtype=bool),
        np.asarray(["seq12/frame00001.png"]),
        stage="medium", candidate_semantics="controlled_local_oracle_v1",
    )
    assert metrics["stage"] == "medium"
    assert metrics["controlled_radial_pair_count"] == 18
    assert metrics["controlled_radial_pair_accuracy"] == 1.0
    assert metrics["controlled_radial_complete_path_rate"] == 1.0
    assert metrics["directional_outward_drift_violation_rate"] == 0.0


def test_scientific_contract_checks_map_disjoint_route_and_gt_anchor():
    arrays, metadata = _minimal_scientific_arrays()
    audit = _validate_scientific_dataset_contract(arrays, metadata)
    assert audit["query_route"] == "seq12"
    assert audit["map_query_route_intersection"] == []
    assert audit["controlled_radial_stencil_complete"] is True

    overlapping = {**metadata, "map_training_routes": ["seq12"]}
    with pytest.raises(ValueError, match="canonical map training routes"):
        _validate_scientific_dataset_contract(arrays, overlapping)

    broken_arrays = {**arrays, "translation_m": arrays["translation_m"].copy()}
    broken_arrays["translation_m"][0, 0] = 0.1
    with pytest.raises(ValueError, match="zero-error GT anchor"):
        _validate_scientific_dataset_contract(broken_arrays, metadata)


def test_rankdata_assigns_average_rank_to_ties():
    np.testing.assert_allclose(_rankdata(np.asarray([1.0, 1.0, 3.0])), [0.5, 0.5, 2.0])


def test_identity_gradient_mask_exposes_only_feature_and_layout_relative_dof():
    mask = _identity_edge_gradient_mask(device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(mask, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]))


def test_fixed_identity_sufficient_statistics_preserve_score_and_gradient():
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=128, pose_code_dim=128, shared_query_map_projection=True,
    ))
    statistics = torch.tensor([
        [0.10, 0.0, 0.0, 0.0, 0.20, 0.30],
        [0.40, 0.0, 0.0, 0.0, 0.50, 0.60],
    ])
    score = _fixed_identity_scores_from_component_statistics(model, statistics)
    weights = model.edge_weights()
    expected = -1.0 + (statistics * weights[None]).sum(dim=1) / weights.sum()
    torch.testing.assert_close(score, expected)
    score.sum().backward()
    assert model.edge_weight_unconstrained.grad is not None
    assert torch.isfinite(model.edge_weight_unconstrained.grad).all()


def test_mapper_audit_recurses_and_detects_query_image_leakage():
    audit = _mapper_supervision_audit(
        {
            "training_images": ["seq1/frame1.png"],
            "initial_checkpoint_metadata": {
                "validation_images": ["seq12/frame00001.png"],
            },
        },
        query_image_ids=["seq12/frame00001.png", "seq12/frame00002.png"],
        query_route="seq12",
    )
    assert audit["query_route_in_mapper_supervision"] is True
    assert audit["exact_query_image_overlap"] == ["seq12/frame00001.png"]
    assert audit["strict_query_representation_route_disjoint"] is False
