from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_FAMILIES,
    FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_REGION_MULTISOURCE_LAYOUT_PROFILE_NAMES,
    FULLTRACK_PER_VIEW_REGION_MULTISOURCE_POOL_PROFILE_NAMES,
    FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_PAIRED_CONTROL_FAMILIES,
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS,
    RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE,
    RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
    RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
    RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
    _cache_target_edges,
    _cached_batch_edges,
    _coarse_top1_stability_loss,
    FrozenFulltrackPerViewAppearanceFeatures,
    MonotonicTop1RelativeTop4Residual,
    SparseFulltrackPerViewResidual,
    fit_fixedprior_fulltrack_per_view_probe,
    fit_fixedprior_fulltrack_rawtop4_probe,
    fit_fulltrack_per_view_normalizer,
    fit_fulltrack_raw_top4_relative_normalizer,
    fulltrack_per_view_joint_edge_coverage,
    fulltrack_per_view_topk_aggregate,
    fulltrack_per_view_topk_mean,
    normalized_fulltrack_per_view_edges,
    normalized_fulltrack_raw_top4_relative_features,
    _normalized_edge_values,
    _batch_edges,
    predict_fixedprior_fulltrack_per_view_probe,
    predict_fixedprior_fulltrack_rawtop4_probe,
)
from feature_extract.vfm.localization.frozen_fulltrack_sparse_maplet_transport import (
    SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE,
    SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE,
)
from feature_extract.vfm.localization.fulltrack_aligned_layout_probe import (
    ALIGNED_LAYOUT_FEATURE_NAMES,
    FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
)


def _features() -> FrozenFulltrackPerViewAppearanceFeatures:
    """Two train and two validation rows with repeated source-row IDs."""

    count, candidates, edges_per_candidate = 4, 2, 2
    probabilities = np.asarray(
        [[0.2, 0.7], [0.7, 0.2], [0.2, 0.7], [0.7, 0.2]],
        dtype=np.float32,
    )
    correct_columns = (0, 1, 0, 1)
    scores: list[list[float]] = []
    for row, correct_column in enumerate(correct_columns):
        for column in range(candidates):
            value = 2.0 if column == correct_column else -2.0
            scores.extend(
                (
                    (
                        value,
                        value * 0.9,
                        value * 0.8,
                        value * 0.7,
                        value * 0.6,
                        value * 0.5,
                        value * 0.4,
                        value * 0.3,
                        value * 0.2,
                        value * 0.1,
                        value * 0.05,
                    ),
                    (
                        value * 0.8,
                        value * 0.72,
                        value * 0.64,
                        value * 0.56,
                        value * 0.48,
                        value * 0.4,
                        value * 0.32,
                        value * 0.24,
                        value * 0.16,
                        value * 0.08,
                        value * 0.04,
                    ),
                )
            )
    return FrozenFulltrackPerViewAppearanceFeatures(
        paths=(Path("/tmp/fulltrack-per-view-fixture.npz"),),
        query_ids=np.asarray(["train-a.png", "train-b.png", "val-a.png", "val-b.png"]),
        split_names=np.asarray(["train", "train", "validation", "validation"]),
        # A complete artifact merge repeats these indices across different
        # query images.  The identity invariant is (query_id, source_row).
        source_row_indices=np.asarray([0, 1, 0, 1], dtype=np.int64),
        xy=np.zeros((count, 2), dtype=np.float32),
        candidate_track_ids=np.asarray(
            [[10, 11], [12, 13], [14, 15], [16, 17]], dtype=np.int64
        ),
        candidate_probabilities=probabilities,
        null_probabilities=np.full((count,), 0.1, dtype=np.float32),
        candidate_support_observation_counts=np.full(
            (count, candidates), edges_per_candidate, dtype=np.int64
        ),
        edge_candidate_offsets=np.arange(
            0, count * candidates * edges_per_candidate + 1, edges_per_candidate, dtype=np.int64
        ),
        edge_geometry_rows=np.arange(count * candidates * edges_per_candidate, dtype=np.int64),
        edge_profile_scores=np.asarray(scores, dtype=np.float32),
        edge_profile_valid=np.ones(
            (count * candidates * edges_per_candidate, 11), dtype=bool
        ),
        profile_names=(
            "radio_final_context5",
            "radio_intermediate_center",
            "radio_intermediate_context5",
            "radio_intermediate_context9",
            "alike_center",
            "alike_context3",
            "alike_context5",
            "alike_context9",
            "radio_final_center",
            "radio_final_context3",
            "radio_intermediate_context13",
        ),
        artifact_metadata=({},),
        compatibility={
            "per_view_edge_feature_semantics": (
                FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS
            )
        },
    )


def _normalizer(features: FrozenFulltrackPerViewAppearanceFeatures):
    return fit_fulltrack_per_view_normalizer(
        features,
        profile_indices=np.asarray([0, 1], dtype=np.int64),
        train_rows=np.asarray([0, 1], dtype=np.int64),
    )


def _layout_features() -> FrozenFulltrackPerViewAppearanceFeatures:
    raw = _features()
    scores = np.repeat(
        raw.edge_profile_scores[:, :1], len(ALIGNED_LAYOUT_FEATURE_NAMES), axis=1
    )
    return FrozenFulltrackPerViewAppearanceFeatures(
        **{
            **raw.__dict__,
            "edge_profile_scores": scores,
            "edge_profile_valid": np.ones(scores.shape, dtype=bool),
            "profile_names": ALIGNED_LAYOUT_FEATURE_NAMES,
            "compatibility": {
                "per_view_edge_feature_semantics": (
                    FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS
                )
            },
        }
    )


def test_sparse_maplet_visual_control_families_select_the_same_profile_groups() -> None:
    """A matched topology control must not change the retained edge population."""

    expected = {
        "fixedprior_fulltrack_perview_sparse_maplet_final_near_mixture": (
            "radio_final_sparse_maplet_near",
        ),
        "fixedprior_fulltrack_perview_sparse_maplet_intermediate_pca256_near_mixture": (
            "radio_intermediate_pca256_sparse_maplet_near",
        ),
        "fixedprior_fulltrack_perview_sparse_maplet_alike_fpn_near_mixture": (
            "alike_fpn_sparse_maplet_near",
        ),
        "fixedprior_fulltrack_perview_sparse_maplet_multiscale_mixture": tuple(
            SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE
        ),
    }
    assert set(FULLTRACK_PER_VIEW_SPARSE_MAPLET_PAIRED_CONTROL_FAMILIES) == set(expected)
    for visual_name, profile_groups in expected.items():
        control_name = FULLTRACK_PER_VIEW_SPARSE_MAPLET_PAIRED_CONTROL_FAMILIES[visual_name]
        visual = FULLTRACK_PER_VIEW_FAMILIES[visual_name]
        control = FULLTRACK_PER_VIEW_FAMILIES[control_name]
        assert (
            visual.edge_feature_semantics
            == FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        )
        assert (
            control.edge_feature_semantics
            == FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS
        )
        expected_visual = tuple(
            name
            for profile in profile_groups
            for name in SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile]
        )
        expected_control = tuple(
            name
            for profile in profile_groups
            for name in SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE[profile]
        )
        assert visual.profile_names == expected_visual
        assert control.profile_names == expected_control
        assert visual.rank2_hard_pair_weight == 0.0
        assert control.rank2_hard_pair_weight == 0.0


def test_zero_initialized_per_view_probe_preserves_fixed_candidate_and_null_mass() -> None:
    features = _features()
    model = SparseFulltrackPerViewResidual(input_dim=2, hidden_dim=8)
    candidate, null, residual = predict_fixedprior_fulltrack_per_view_probe(
        model=model,
        features=features,
        normalizer=_normalizer(features),
        device=torch.device("cpu"),
        batch_size=2,
    )
    np.testing.assert_allclose(candidate, features.candidate_probabilities, atol=1e-6)
    np.testing.assert_array_equal(null, features.null_probabilities)
    np.testing.assert_allclose(residual, 0.0, atol=1e-7)


def test_bounded_linear_per_view_residual_preserves_mass_and_bounds_evidence() -> None:
    features = _features()
    cap = 0.25
    model = SparseFulltrackPerViewResidual(
        input_dim=2,
        hidden_dim=1,
        residual_architecture="linear",
        residual_cap=cap,
    )
    with torch.no_grad():
        model.evidence.bias.fill_(20.0)
    candidate, null, residual = predict_fixedprior_fulltrack_per_view_probe(
        model=model,
        features=features,
        normalizer=_normalizer(features),
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert np.max(np.abs(residual)) <= cap + 1e-6
    np.testing.assert_allclose(candidate.sum(axis=1) + null, 1.0, atol=1e-6)
    np.testing.assert_array_equal(null, features.null_probabilities)


def test_batched_normalization_matches_legacy_full_edge_path() -> None:
    features = _features()
    normalizer = _normalizer(features)
    legacy, legacy_valid = normalized_fulltrack_per_view_edges(features, normalizer)
    rows = np.asarray([0, 3, 5, 8], dtype=np.int64)
    batched, valid = _normalized_edge_values(
        features=features,
        normalizer=normalizer,
        edge_rows=rows,
    )
    np.testing.assert_array_equal(valid, legacy_valid[rows])
    np.testing.assert_allclose(batched, legacy[rows][legacy_valid[rows]], atol=1e-6)
    assert fulltrack_per_view_joint_edge_coverage(
        features, profile_indices=normalizer.profile_indices, edge_chunk_size=3
    ) == 1.0


def test_per_view_prediction_subset_keeps_source_row_order_and_mass() -> None:
    features = _features()
    model = SparseFulltrackPerViewResidual(input_dim=2, hidden_dim=8)
    normalizer = _normalizer(features)
    full = predict_fixedprior_fulltrack_per_view_probe(
        model=model,
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=2,
    )
    rows = np.asarray([3, 1], dtype=np.int64)
    subset = predict_fixedprior_fulltrack_per_view_probe(
        model=model,
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=2,
        rows=rows,
    )
    for complete, selected in zip(full, subset):
        np.testing.assert_allclose(selected, complete[rows], atol=1e-6)


def test_cached_target_edges_rebuild_the_same_shuffled_batch() -> None:
    features = _features()
    normalizer = _normalizer(features)
    cached_values, cached_columns = _cache_target_edges(
        features=features,
        normalizer=normalizer,
        target_rows=np.asarray([0, 1], dtype=np.int64),
        row_chunk_size=1,
    )
    values, candidates = _cached_batch_edges(
        cached_values=cached_values,
        cached_columns=cached_columns,
        selected_target_indices=np.asarray([1, 0], dtype=np.int64),
        candidate_count=features.candidate_count,
        feature_dim=len(normalizer.profile_indices),
    )
    expected_values, expected_candidates = _batch_edges(
        features=features,
        normalizer=normalizer,
        rows=np.asarray([1, 0], dtype=np.int64),
    )
    np.testing.assert_allclose(values, expected_values, atol=1e-6)
    np.testing.assert_array_equal(candidates, expected_candidates)


def test_per_view_mixture_is_invariant_to_support_edge_order() -> None:
    features = _features()
    scores = features.edge_profile_scores.copy()
    geometry = features.edge_geometry_rows.copy()
    for start, end in zip(
        features.edge_candidate_offsets[:-1], features.edge_candidate_offsets[1:]
    ):
        scores[start:end] = scores[start:end][::-1]
        geometry[start:end] = geometry[start:end][::-1]
    permuted = FrozenFulltrackPerViewAppearanceFeatures(
        **{
            **features.__dict__,
            "edge_profile_scores": scores,
            "edge_geometry_rows": geometry,
        }
    )
    torch.manual_seed(17)
    model = SparseFulltrackPerViewResidual(input_dim=2, hidden_dim=8)
    with torch.no_grad():
        model.evidence.weight.normal_()
        model.evidence.bias.normal_()
        model.view.weight.normal_()
        model.view.bias.normal_()
    normalizer = _normalizer(features)
    original = predict_fixedprior_fulltrack_per_view_probe(
        model=model,
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=4,
    )
    reordered = predict_fixedprior_fulltrack_per_view_probe(
        model=model,
        features=permuted,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=4,
    )
    for left, right in zip(original, reordered):
        np.testing.assert_allclose(left, right, atol=1e-6)


def test_missing_profile_edge_has_no_direct_candidate_residual() -> None:
    features = _features()
    scores = features.edge_profile_scores.copy()
    valid = features.edge_profile_valid.copy()
    # First row/candidate's two support observations have no jointly observed
    # descriptor family.  It must be omitted, not encoded as a negative cue.
    scores[:2] = np.nan
    valid[:2] = False
    missing = FrozenFulltrackPerViewAppearanceFeatures(
        **{
            **features.__dict__,
            "edge_profile_scores": scores,
            "edge_profile_valid": valid,
        }
    )
    model = SparseFulltrackPerViewResidual(input_dim=2, hidden_dim=8)
    with torch.no_grad():
        model.evidence.bias.fill_(1.0)
    candidate, null, residual = predict_fixedprior_fulltrack_per_view_probe(
        model=model,
        features=missing,
        normalizer=_normalizer(features),
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert residual[0, 0] == 0.0
    assert residual[0, 1] > 0.9
    np.testing.assert_allclose(candidate.sum(axis=1) + null, 1.0, atol=1e-6)


def test_train_only_per_view_probe_recovers_holdout_identity() -> None:
    features = _features()
    membership = np.asarray([[True, False], [False, True]], dtype=bool)
    model, normalizer, metadata = fit_fixedprior_fulltrack_per_view_probe(
        features=features,
        family="fixedprior_fulltrack_perview_multiscale",
        train_normalizer_rows=np.asarray([0, 1], dtype=np.int64),
        target_train_rows=np.asarray([0, 1], dtype=np.int64),
        target_candidate_membership=membership,
        device=torch.device("cpu"),
        epochs=140,
        batch_size=2,
        learning_rate=0.03,
        weight_decay=0.0,
        hidden_dim=8,
        seed=11,
    )
    candidate, null, _residual = predict_fixedprior_fulltrack_per_view_probe(
        model=model,
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert metadata["support_view_marginalization"] == (
        "learned_logsumexp_over_retained_real_observation_edges_v1"
    )
    assert np.argmax(candidate[2:], axis=1).tolist() == [0, 1]
    np.testing.assert_array_equal(null, features.null_probabilities)


def test_explicit_hard_pair_weight_is_recorded_as_a_train_only_loss_setting() -> None:
    features = _features()
    membership = np.asarray([[True, False], [False, True]], dtype=bool)
    _model, _normalizer, metadata = fit_fixedprior_fulltrack_per_view_probe(
        features=features,
        family="fixedprior_fulltrack_perview_alike",
        train_normalizer_rows=np.asarray([0, 1], dtype=np.int64),
        target_train_rows=np.asarray([0, 1], dtype=np.int64),
        target_candidate_membership=membership,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=2,
        learning_rate=0.03,
        weight_decay=0.0,
        hidden_dim=8,
        seed=13,
        rank2_hard_pair_weight=4.0,
    )
    assert metadata["rank2_hard_pair_weight"] == 4.0
    assert metadata["family_default_rank2_hard_pair_weight"] == 1.0


def test_zero_hard_pair_weight_is_reported_as_identity_nll_only() -> None:
    raw = _features()
    features = FrozenFulltrackPerViewAppearanceFeatures(
        **{
            **raw.__dict__,
            "edge_profile_scores": raw.edge_profile_scores[:, :2],
            "edge_profile_valid": raw.edge_profile_valid[:, :2],
            "profile_names": ("radio_final_global", "radio_final_summary"),
            "compatibility": {
                "per_view_edge_feature_semantics": (
                    FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_EDGE_FEATURE_SEMANTICS
                )
            },
        }
    )
    membership = np.asarray([[True, False], [False, True]], dtype=bool)
    _model, _normalizer_value, metadata = fit_fixedprior_fulltrack_per_view_probe(
        features=features,
        family="fixedprior_fulltrack_perview_globalcontext_mixture",
        train_normalizer_rows=np.asarray([0, 1], dtype=np.int64),
        target_train_rows=np.asarray([0, 1], dtype=np.int64),
        target_candidate_membership=membership,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=2,
        learning_rate=0.03,
        weight_decay=0.0,
        hidden_dim=8,
        seed=29,
    )
    assert metadata["rank2_hard_pair_weight"] == 0.0
    assert metadata["training_objective"] == "identity_nll_v1"


def test_region_context_families_do_not_learn_common_cell_fraction() -> None:
    selected = (
        "fixedprior_fulltrack_perview_region_multisource_mixture",
        "fixedprior_fulltrack_perview_region_multisource_pool_mixture",
    )
    for family in selected:
        assert all(
            not name.endswith("_common_cell_fraction")
            for name in FULLTRACK_PER_VIEW_FAMILIES[family].profile_names
        )
    assert len(FULLTRACK_PER_VIEW_REGION_MULTISOURCE_LAYOUT_PROFILE_NAMES) == 60
    assert len(FULLTRACK_PER_VIEW_REGION_MULTISOURCE_POOL_PROFILE_NAMES) == 6


def test_layout_family_uses_matching_semantics_and_symmetric_top1_stability() -> None:
    features = _layout_features()
    membership = np.asarray([[True, False], [False, True]], dtype=bool)
    _model, _normalizer_value, metadata = fit_fixedprior_fulltrack_per_view_probe(
        features=features,
        family="fixedprior_fulltrack_perview_aligned_layout_multiscale",
        train_normalizer_rows=np.asarray([0, 1], dtype=np.int64),
        target_train_rows=np.asarray([0, 1], dtype=np.int64),
        target_candidate_membership=membership,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=2,
        learning_rate=0.03,
        weight_decay=0.0,
        hidden_dim=8,
        seed=17,
    )
    assert metadata["coarse_top1_stability_weight"] == 8.0
    assert metadata["coarse_top1_correct_train_row_count"] == 0
    assert metadata["training_objective"] == (
        "identity_nll_plus_rank2_rescue_and_coarse_top1_stability_v1"
    )


def test_layout_nll_family_has_no_hard_pair_or_top1_stability_loss() -> None:
    features = _layout_features()
    membership = np.asarray([[True, False], [False, True]], dtype=bool)
    _model, _normalizer_value, metadata = fit_fixedprior_fulltrack_per_view_probe(
        features=features,
        family="fixedprior_fulltrack_perview_aligned_layout_alike_nll",
        train_normalizer_rows=np.asarray([0, 1], dtype=np.int64),
        target_train_rows=np.asarray([0, 1], dtype=np.int64),
        target_candidate_membership=membership,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=2,
        learning_rate=0.03,
        weight_decay=0.0,
        hidden_dim=8,
        seed=19,
    )
    assert metadata["rank2_hard_pair_weight"] == 0.0
    assert metadata["coarse_top1_stability_weight"] == 0.0
    assert metadata["training_objective"] == "identity_nll_v1"


def test_raw_top4_preserves_support_order_and_top1_relative_missing_neutrality() -> None:
    features = _features()
    profile_indices = np.asarray([4, 5, 6, 7], dtype=np.int64)
    values, valid = fulltrack_per_view_topk_mean(
        features, profile_indices=profile_indices, top_k=4
    )
    assert valid.all()
    # On row zero the first candidate is the generated correct one, so each
    # ALIKE top-4 statistic remains higher than its frozen top-one competitor.
    assert np.all(values[0, 0] > values[0, 1])
    normalizer = fit_fulltrack_raw_top4_relative_normalizer(
        features,
        profile_indices=profile_indices,
        train_rows=np.asarray([0, 1], dtype=np.int64),
    )
    relative, common = normalized_fulltrack_raw_top4_relative_features(
        features, normalizer
    )
    assert common.all()
    # Candidate one is frozen top-one on row zero and is the explicit zero
    # reference; no support count or missingness is encoded in that value.
    np.testing.assert_allclose(relative[0, 1], 0.0, atol=1e-7)
    assert np.all(relative[0, 0] > 0.0)


def test_monotonic_raw_top4_probe_preserves_mass_and_recovers_holdout_identity() -> None:
    features = _features()
    membership = np.asarray([[True, False], [False, True]], dtype=bool)
    initial = MonotonicTop1RelativeTop4Residual(input_dim=4)
    # The raw overlay begins close to the fixed posterior rather than injecting
    # softplus(0) ~= 0.693 of uncalibrated score for every profile.
    assert max(initial.weights().detach().cpu().tolist()) < 0.02
    model, normalizer, metadata = fit_fixedprior_fulltrack_rawtop4_probe(
        features=features,
        family="fixedprior_fulltrack_rawtop4_alike",
        train_normalizer_rows=np.asarray([0, 1], dtype=np.int64),
        target_train_rows=np.asarray([0, 1], dtype=np.int64),
        target_candidate_membership=membership,
        device=torch.device("cpu"),
        epochs=160,
        batch_size=2,
        learning_rate=0.03,
        weight_decay=0.0,
        seed=19,
    )
    assert metadata["architecture"] == "monotonic_top1_relative_raw_top4_overlay_v1"
    assert all(weight >= 0.0 for weight in metadata["monotonic_nonnegative_weights"])
    candidate, null, _residual = predict_fixedprior_fulltrack_rawtop4_probe(
        model=model,
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert isinstance(model, MonotonicTop1RelativeTop4Residual)
    assert all(weight >= 0.0 for weight in model.weights().detach().cpu().tolist())
    assert np.argmax(candidate[2:], axis=1).tolist() == [0, 1]
    np.testing.assert_array_equal(null, features.null_probabilities)


def test_positive_uplift_never_penalizes_a_below_top_one_raw_delta() -> None:
    model = MonotonicTop1RelativeTop4Residual(
        input_dim=2, initial_weight=1.0, positive_uplift=True
    )
    candidate, residual, _logits = model(
        relative_features=torch.tensor([[[2.0, -3.0], [-1.0, -2.0]]]),
        candidate_conditional_log_prior=torch.log(
            torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        ),
        candidate_mass=torch.tensor([1.0], dtype=torch.float32),
    )
    assert residual[0, 0] > 0.0
    assert residual[0, 1] == 0.0
    torch.testing.assert_close(candidate.sum(dim=1), torch.ones((1,)))


def test_coarse_top_one_stability_penalizes_only_verified_top_one_overrides() -> None:
    membership = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    loss = _coarse_top1_stability_loss(
        logits=torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32),
        membership=membership,
        coarse_top1_columns=torch.tensor([0, 0], dtype=torch.long),
    )
    # Only row zero has a correct frozen top-one, and its two-logit margin is
    # positive, so the result is the corresponding softplus penalty.
    torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(-2.0)))
    override = _coarse_top1_stability_loss(
        logits=torch.tensor([[0.0, 2.0], [0.0, 3.0]], dtype=torch.float32),
        membership=membership,
        coarse_top1_columns=torch.tensor([0, 0], dtype=torch.long),
    )
    assert override > loss


def test_positive_uplift_families_keep_representation_specific_profiles() -> None:
    expected = {
        "fixedprior_fulltrack_rawtop4_positive_uplift_alike": (
            4,
            RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
        ),
        "fixedprior_fulltrack_rawtop4_positive_uplift_alike_lower_envelope": (
            4,
            RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
        ),
        "fixedprior_fulltrack_rawtop4_positive_uplift_radio": (
            7,
            RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
        ),
        "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale": (
            11,
            RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
        ),
        "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale_tanh_cap": (
            11,
            RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE,
        ),
        "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale_tanh_cap_traincal": (
            11,
            RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
        ),
        "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale_tanh_cap_traincal_balanced": (
            11,
            RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
        ),
    }
    for family_name, (profile_count, architecture) in expected.items():
        family = FULLTRACK_PER_VIEW_FAMILIES[family_name]
        assert family.architecture == architecture
        assert family.rank2_hard_pair_weight == 8.0
        assert len(family.profile_names) == profile_count
        assert len(set(family.profile_names)) == profile_count
    assert (
        FULLTRACK_PER_VIEW_FAMILIES[
            "fixedprior_fulltrack_rawtop4_positive_uplift_alike_lower_envelope"
        ].raw_topk_aggregation
        == "lower_envelope"
    )
    assert (
        FULLTRACK_PER_VIEW_FAMILIES[
            "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale_tanh_cap"
        ].training_seed_key
        == "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale"
    )
    assert (
        FULLTRACK_PER_VIEW_FAMILIES[
            "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale_tanh_cap_traincal"
        ].calibration_during_train
        is True
    )
    balanced = FULLTRACK_PER_VIEW_FAMILIES[
        "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale_tanh_cap_traincal_balanced"
    ]
    assert balanced.calibration_during_train is True
    assert balanced.coarse_top1_stability_weight == 8.0


def test_lower_envelope_requires_each_retained_support_view_to_agree() -> None:
    features = _features()
    profile_indices = np.asarray([0, 1], dtype=np.int64)
    mean, mean_valid = fulltrack_per_view_topk_aggregate(
        features,
        profile_indices=profile_indices,
        top_k=2,
        aggregation="uniform_mean",
    )
    lower, lower_valid = fulltrack_per_view_topk_aggregate(
        features,
        profile_indices=profile_indices,
        top_k=2,
        aggregation="lower_envelope",
    )
    np.testing.assert_array_equal(mean_valid, lower_valid)
    assert np.all(lower[lower_valid] <= mean[mean_valid])
    assert np.any(lower[lower_valid] < mean[mean_valid])


def test_tanh_capped_uplift_preserves_mass_and_bounds_only_positive_evidence() -> None:
    model = MonotonicTop1RelativeTop4Residual(
        input_dim=2,
        initial_weight=10.0,
        positive_uplift=True,
        residual_cap=0.5,
    )
    candidate, residual, _logits = model(
        relative_features=torch.tensor([[[10.0, 10.0], [-2.0, -1.0]]]),
        candidate_conditional_log_prior=torch.log(
            torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        ),
        candidate_mass=torch.tensor([1.0], dtype=torch.float32),
    )
    assert 0.0 < residual[0, 0] <= 0.5
    assert residual[0, 1] == 0.0
    torch.testing.assert_close(candidate.sum(dim=1), torch.ones((1,)))


def test_traincal_bounded_probe_keeps_its_calibration_in_model_state() -> None:
    features = _features()
    membership = np.asarray([[True, False], [False, True]], dtype=bool)
    model, normalizer, metadata = fit_fixedprior_fulltrack_rawtop4_probe(
        features=features,
        family=(
            "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale_tanh_cap_traincal"
        ),
        train_normalizer_rows=np.asarray([0, 1], dtype=np.int64),
        target_train_rows=np.asarray([0, 1], dtype=np.int64),
        target_candidate_membership=membership,
        device=torch.device("cpu"),
        epochs=20,
        batch_size=2,
        learning_rate=0.03,
        weight_decay=0.0,
        seed=29,
        rank2_hard_pair_weight=8.0,
        postfit_residual_scale=0.6,
        postfit_residual_cap=0.5,
    )
    candidate, null, residual = predict_fixedprior_fulltrack_rawtop4_probe(
        model=model,
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert metadata["residual_calibration_applied_during_train"] is True
    assert metadata["postfit_cap_applied_after_train"] is False
    assert metadata["postfit_residual_cap"] == 0.5
    assert np.isclose(model.residual_scale.item(), 0.6)
    assert np.isclose(model.residual_cap.item(), 0.5)
    assert np.max(residual) <= 0.5 + 1e-6
    np.testing.assert_array_equal(null, features.null_probabilities)
    np.testing.assert_allclose(candidate.sum(axis=1) + null, 1.0, atol=2e-5)


def test_balanced_traincal_probe_records_coarse_top_one_stability_objective() -> None:
    features = _features()
    # The first row's frozen top-one is column one and is correct; the second
    # remains a rank-two rescue case.  This exercises both paired objectives.
    membership = np.asarray([[False, True], [False, True]], dtype=bool)
    _model, _normalizer, metadata = fit_fixedprior_fulltrack_rawtop4_probe(
        features=features,
        family=(
            "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale_tanh_cap_traincal_balanced"
        ),
        train_normalizer_rows=np.asarray([0, 1], dtype=np.int64),
        target_train_rows=np.asarray([0, 1], dtype=np.int64),
        target_candidate_membership=membership,
        device=torch.device("cpu"),
        epochs=20,
        batch_size=2,
        learning_rate=0.03,
        weight_decay=0.0,
        seed=29,
        rank2_hard_pair_weight=8.0,
        postfit_residual_scale=0.6,
        postfit_residual_cap=0.5,
    )
    assert metadata["coarse_top1_correct_train_pair_count"] == 1
    assert metadata["coarse_top1_stability_weight"] == 8.0
    assert (
        metadata["training_objective"]
        == "identity_nll_plus_rank2_rescue_and_coarse_top1_stability_v1"
    )
    assert np.isfinite(metadata["last_train_coarse_top1_stability_loss"])


def test_positive_uplift_raw_top4_probe_recovers_holdout_identity() -> None:
    features = _features()
    membership = np.asarray([[True, False], [False, True]], dtype=bool)
    model, normalizer, metadata = fit_fixedprior_fulltrack_rawtop4_probe(
        features=features,
        family="fixedprior_fulltrack_rawtop4_positive_uplift_alike",
        train_normalizer_rows=np.asarray([0, 1], dtype=np.int64),
        target_train_rows=np.asarray([0, 1], dtype=np.int64),
        target_candidate_membership=membership,
        device=torch.device("cpu"),
        epochs=160,
        batch_size=2,
        learning_rate=0.03,
        weight_decay=0.0,
        seed=23,
        rank2_hard_pair_weight=8.0,
        postfit_residual_scale=0.5,
    )
    candidate, null, _residual = predict_fixedprior_fulltrack_rawtop4_probe(
        model=model,
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert metadata["architecture"] == (
        "monotonic_top1_relative_positive_uplift_top4_overlay_v1"
    )
    assert metadata["candidate_evidence_transform"] == "relu_positive_relative_uplift_v1"
    assert metadata["postfit_residual_scale"] == 0.5
    np.testing.assert_allclose(
        metadata["monotonic_nonnegative_weights"],
        0.5 * np.asarray(metadata["raw_fit_monotonic_nonnegative_weights"]),
        rtol=1e-5,
        atol=1e-6,
    )
    assert model.positive_uplift is True
    assert np.argmax(candidate[2:], axis=1).tolist() == [0, 1]
    np.testing.assert_array_equal(null, features.null_probabilities)
