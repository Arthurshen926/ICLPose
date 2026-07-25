from __future__ import annotations

import torch

from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CandidateHighresRGBMultiscalePrediction,
    CandidateHighresRGBScalePrediction,
)
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    CandidateMultiscalePhaseIdentityPrediction,
)
from feature_extract.vfm.localization.candidate_pose_evidence_weighter import (
    TARGET_FREE_EVIDENCE_FEATURE_NAMES,
    TargetFreePoseEvidenceWeighter,
    apply_target_free_feature_policy,
    build_target_free_pose_evidence_features,
    effective_sample_size,
    fit_target_free_feature_normalizer,
    relative_spatial_coverage_divergence,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    normalized_spatial_log_probabilities_with_dustbin,
)


def _runtime() -> CandidatePoseRGBSpatialRuntime:
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 0, 0]),
        query_xy=torch.tensor([[8.0, 8.0], [88.0, 8.0], [48.0, 40.0]]),
        support_image_indices=torch.tensor([[[1], [2]], [[1], [2]], [[1], [2]]]),
        support_xy=torch.full((3, 2, 1, 2), 12.0),
        support_view_valid=torch.ones((3, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((3, 2, 1)),
        candidate_probabilities=torch.full((3, 2), 0.45),
        null_probabilities=torch.full((3,), 0.10),
    )


def _phase() -> CandidateMultiscalePhaseIdentityPrediction:
    values = torch.tensor(
        [
            [[3.0], [-2.0]],
            [[0.0], [0.0]],
            [[1.5], [0.5]],
        ]
    )
    usable = torch.ones_like(values, dtype=torch.bool)
    return CandidateMultiscalePhaseIdentityPrediction(
        source_edge_log_likelihood_ratios={
            name: values.clone() for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
        },
        source_edge_usable={
            name: usable.clone() for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
        },
        edge_log_likelihood_ratios=values,
        edge_usable=usable,
        source_weights={name: 1.0 for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES},
    )


def _rgb() -> CandidateHighresRGBMultiscalePrediction:
    offsets = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    logits = torch.tensor(
        [
            [[[5.0, 0.0, 0.0, 0.0]], [[5.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]],
            [[[1.5, 1.0, 0.5, 0.0]], [[1.5, 1.0, 0.5, 0.0]]],
        ],
        dtype=torch.float32,
    )
    dustbin = torch.tensor([[[0.0], [0.0]], [[5.0], [5.0]], [[0.0], [0.0]]])
    joint = normalized_spatial_log_probabilities_with_dustbin(
        logits.reshape(-1, 4), dustbin.reshape(-1)
    ).reshape(3, 2, 1, 5)
    scale = CandidateHighresRGBScalePrediction(
        spatial_logits=logits,
        non_dustbin_logits=dustbin,
        joint_log_probabilities=joint,
        offsets_xy=offsets,
        edge_log_likelihood_ratios=torch.zeros((3, 2, 1)),
        edge_usable=torch.ones((3, 2, 1), dtype=torch.bool),
    )
    return CandidateHighresRGBMultiscalePrediction(sources={"fine": scale, "broad": scale})


def test_target_free_evidence_features_use_visual_statistics_only() -> None:
    features = build_target_free_pose_evidence_features(
        runtime=_runtime(), phase_prediction=_phase(), rgb_prediction=_rgb()
    )
    assert features.values.shape == (3, len(TARGET_FREE_EVIDENCE_FEATURE_NAMES))
    assert features.feature_names == TARGET_FREE_EVIDENCE_FEATURE_NAMES
    # Strong compact RGB mode and phase concentration beat the dustbin-dominated point.
    assert features.values[0, 0] > features.values[2, 0] > features.values[1, 0]
    assert features.values[0, -1] > features.values[2, -1] > features.values[1, -1]
    unavailable = build_target_free_pose_evidence_features(
        runtime=_runtime(),
        phase_prediction=_phase(),
        rgb_prediction=_rgb(),
        edge_availability_override=torch.zeros((3, 2, 1), dtype=torch.bool),
    )
    torch.testing.assert_close(unavailable.values, torch.zeros_like(unavailable.values))


def test_weighter_starts_conservative_and_is_token_permutation_equivariant() -> None:
    features = build_target_free_pose_evidence_features(
        runtime=_runtime(), phase_prediction=_phase(), rgb_prediction=_rgb()
    )
    center, scale = fit_target_free_feature_normalizer([features])
    model = TargetFreePoseEvidenceWeighter(
        feature_center=center,
        feature_scale=scale,
        hidden_dim=8,
        minimum_uniform_mass=0.50,
        maximum_uniform_mass=0.95,
        initial_uniform_mass=0.75,
    ).eval()
    prediction = model(features)
    torch.testing.assert_close(prediction.weights, torch.full((3,), 1.0 / 3.0))
    torch.testing.assert_close(prediction.uniform_mass, torch.tensor([0.75]), atol=1e-5, rtol=1e-5)
    permutation = torch.tensor([2, 0, 1])
    permuted = model(features.values.index_select(0, permutation))
    torch.testing.assert_close(permuted.weights, prediction.weights.index_select(0, permutation))
    torch.testing.assert_close(effective_sample_size(prediction.weights), torch.tensor(3.0))


def test_rgb_only_feature_policy_removes_the_falsified_phase_selector_signal() -> None:
    features = build_target_free_pose_evidence_features(
        runtime=_runtime(), phase_prediction=_phase(), rgb_prediction=_rgb()
    )
    rgb_only = apply_target_free_feature_policy(features=features, policy="rgb_only")
    torch.testing.assert_close(rgb_only.values[:, -1], torch.zeros((features.point_count,)))
    restored = apply_target_free_feature_policy(features=features, policy="rgb_plus_phase")
    torch.testing.assert_close(restored.values, features.values)


def test_relative_coverage_measures_collapse_against_detector_distribution() -> None:
    xy = torch.tensor([[5.0, 5.0], [95.0, 5.0], [5.0, 95.0], [95.0, 95.0]])
    uniform = torch.full((4,), 0.25)
    concentrated = torch.tensor([1.0, 0.0, 0.0, 0.0])
    divergence_uniform = relative_spatial_coverage_divergence(
        weights=uniform, xy=xy, image_size=(100, 100), grid_rows=2, grid_columns=2
    )
    divergence_concentrated = relative_spatial_coverage_divergence(
        weights=concentrated, xy=xy, image_size=(100, 100), grid_rows=2, grid_columns=2
    )
    torch.testing.assert_close(divergence_uniform, torch.tensor(0.0), atol=1e-6, rtol=0.0)
    assert divergence_concentrated > 1.0
