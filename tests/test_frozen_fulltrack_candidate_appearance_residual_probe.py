from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.localization.frozen_fulltrack_candidate_appearance_residual_probe import (
    FixedCandidateMassLinearResidual,
    FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES,
    FrozenFulltrackAppearanceFeatures,
    fit_fixedprior_fulltrack_linear_residual,
    fit_fulltrack_appearance_feature_normalizer,
    fit_fixedprior_fulltrack_summary_top4_probe,
    fit_fulltrack_summary_top4_relative_normalizer,
    fulltrack_feature_indices_for_family,
    normalized_fulltrack_appearance_model_input,
    normalized_fulltrack_summary_top4_relative_features,
    predict_fixedprior_fulltrack_linear_residual,
    predict_fixedprior_fulltrack_summary_top4_probe,
    _radio_intermediate_projection_contract,
    select_rank2_to_top1_wrong_training_pairs,
)


def _features() -> FrozenFulltrackAppearanceFeatures:
    names = (
        "radio_intermediate_context9__uniform_top4_mean_ncc",
        "alike_context5__uniform_top4_mean_ncc",
    )
    count, candidates = 4, 2
    candidate_probability = np.asarray(
        [[0.2, 0.7], [0.7, 0.2], [0.2, 0.7], [0.7, 0.2]],
        dtype=np.float32,
    )
    values = np.full((count, candidates, len(names)), -2.0, dtype=np.float32)
    # Candidate zero is correct on even rows, candidate one on odd rows.  The
    # frozen prior therefore ranks the wrong identity first in every row.
    for row, correct in enumerate((0, 1, 0, 1)):
        values[row, correct] = 2.0
    return FrozenFulltrackAppearanceFeatures(
        paths=(Path("/tmp/fulltrack-summary-fixture.npz"),),
        query_ids=np.asarray(["train-a.png", "train-b.png", "val-a.png", "val-b.png"]),
        split_names=np.asarray(["train", "train", "validation", "validation"]),
        source_row_indices=np.arange(count, dtype=np.int64),
        xy=np.zeros((count, 2), dtype=np.float32),
        candidate_track_ids=np.asarray([[10, 11], [12, 13], [14, 15], [16, 17]]),
        candidate_probabilities=candidate_probability,
        null_probabilities=np.full((count,), 0.1, dtype=np.float32),
        candidate_summary_features=values,
        candidate_summary_feature_valid=np.ones_like(values, dtype=bool),
        feature_names=names,
        profile_names=("radio_intermediate_context9", "alike_context5"),
        artifact_metadata=({},),
        compatibility={},
    )


def test_intermediate_projection_contract_keeps_explicit_override_lineage() -> None:
    metadata = {
        "radio_intermediate_projection_override": {
            "enabled": True,
            "source_cache": "/tmp/context-pca64.npz",
            "source_cache_sha256": "source-cache",
            "source_projection_dim": 64,
            "override_cache": "/tmp/context-pca256.npz",
            "override_cache_sha256": "override-cache",
            "override_projection_dim": 256,
            "same_source_context_sha256": "shared-context",
            "same_pca_training_manifest_sha256": "shared-pca-fit",
        }
    }
    contract = _radio_intermediate_projection_contract(
        metadata, path=Path("/tmp/fixture.npz")
    )
    assert contract["override_projection_dim"] == 256
    assert contract["source_projection_dim"] == 64

    metadata["radio_intermediate_projection_override"]["override_projection_dim"] = 64
    with np.testing.assert_raises_regex(ValueError, "intermediate PCA contract"):
        _radio_intermediate_projection_contract(metadata, path=Path("/tmp/fixture.npz"))


def test_zero_residual_preserves_candidate_and_null_probability_mass() -> None:
    features = _features()
    normalizer = fit_fulltrack_appearance_feature_normalizer(
        features,
        feature_indices=np.asarray([0, 1], dtype=np.int64),
        train_rows=np.asarray([0, 1], dtype=np.int64),
    )
    candidate, null, residual = predict_fixedprior_fulltrack_linear_residual(
        model=FixedCandidateMassLinearResidual(input_dim=4),
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=2,
    )
    np.testing.assert_allclose(candidate, features.candidate_probabilities, atol=1e-6)
    np.testing.assert_array_equal(null, features.null_probabilities)
    np.testing.assert_allclose(residual, 0.0, atol=1e-7)


def test_train_rank2_hardpair_residual_recovers_holdout_identity_without_null_shift() -> None:
    features = _features()
    membership = np.asarray([[True, False], [False, True]], dtype=bool)
    model, normalizer, metadata = fit_fixedprior_fulltrack_linear_residual(
        features=features,
        family="fixedprior_fulltrack_multiscale_rank2hard",
        train_normalizer_rows=np.asarray([0, 1], dtype=np.int64),
        target_train_rows=np.asarray([0, 1], dtype=np.int64),
        target_candidate_membership=membership,
        device=torch.device("cpu"),
        epochs=120,
        batch_size=2,
        learning_rate=0.1,
        weight_decay=0.0,
        seed=11,
    )
    candidate, null, _residual = predict_fixedprior_fulltrack_linear_residual(
        model=model,
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert metadata["rank2_to_top1_wrong_train_pair_count"] == 2
    assert metadata["null_handling"] == "input_null_probability_exactly_preserved_v1"
    np.testing.assert_array_equal(null, features.null_probabilities)
    assert np.argmax(candidate[2:], axis=1).tolist() == [0, 1]


def test_missing_summary_feature_is_finite_neutral_value_with_availability_flag() -> None:
    features = _features()
    values = features.candidate_summary_features.copy()
    valid = features.candidate_summary_feature_valid.copy()
    values[0, 0, 0] = np.nan
    valid[0, 0, 0] = False
    features = FrozenFulltrackAppearanceFeatures(
        **{
            **features.__dict__,
            "candidate_summary_features": values,
            "candidate_summary_feature_valid": valid,
        }
    )
    normalizer = fit_fulltrack_appearance_feature_normalizer(
        features,
        feature_indices=np.asarray([0], dtype=np.int64),
        train_rows=np.asarray([0, 1], dtype=np.int64),
    )
    model_input = normalized_fulltrack_appearance_model_input(features, normalizer)
    assert np.all(np.isfinite(model_input))
    assert model_input[0, 0].tolist() == [0.0, 0.0]


def test_top1_relative_input_uses_only_jointly_observed_evidence() -> None:
    features = _features()
    normalizer = fit_fulltrack_appearance_feature_normalizer(
        features,
        feature_indices=np.asarray([0], dtype=np.int64),
        train_rows=np.asarray([0, 1], dtype=np.int64),
        input_mode="top1_relative_common_evidence",
    )
    model_input = normalized_fulltrack_appearance_model_input(features, normalizer)
    # Row zero has frozen candidate one as top-1.  Candidate zero's raw score is
    # higher, while the reference candidate is explicitly assigned zero.
    assert model_input.shape == (4, 2, 1)
    assert model_input[0, 0, 0] > 0.0
    assert model_input[0, 1, 0] == 0.0

    values = features.candidate_summary_features.copy()
    valid = features.candidate_summary_feature_valid.copy()
    values[0, 1, 0] = np.nan
    valid[0, 1, 0] = False
    missing_reference = FrozenFulltrackAppearanceFeatures(
        **{
            **features.__dict__,
            "candidate_summary_features": values,
            "candidate_summary_feature_valid": valid,
        }
    )
    neutral = normalized_fulltrack_appearance_model_input(
        missing_reference, normalizer
    )
    # No shared support evidence is unknown, hence no candidate residual input.
    assert neutral[0, 0, 0] == 0.0


def test_global_context_family_cannot_be_mislabeled_as_local_radio_features() -> None:
    names = (
        "radio_final_global__uniform_mean_cosine",
        "radio_final_summary__uniform_mean_cosine",
    )
    indices = fulltrack_feature_indices_for_family(
        "fixedprior_fulltrack_globalcontext_top1relative_nll",
        feature_names=names,
        profile_names=("radio_final_global", "radio_final_summary"),
    )
    assert indices.tolist() == [0, 1]
    with np.testing.assert_raises_regex(ValueError, "incompatible"):
        fulltrack_feature_indices_for_family(
            "fixedprior_fulltrack_radio_nll",
            feature_names=names,
            profile_names=("radio_final_global", "radio_final_summary"),
        )


def test_rank2_training_pair_selection_uses_only_frozen_prior_and_train_labels() -> None:
    selection = select_rank2_to_top1_wrong_training_pairs(
        candidate_probabilities=np.asarray(
            [[0.2, 0.7, 0.1], [0.5, 0.3, 0.2], [0.7, 0.2, 0.1]],
            dtype=np.float32,
        ),
        candidate_labels=np.asarray(
            [[True, False, False], [False, True, False], [True, False, False]],
            dtype=bool,
        ),
    )
    assert selection.row_positions.tolist() == [0, 1]
    assert selection.positive_columns.tolist() == [0, 1]
    assert selection.negative_columns.tolist() == [1, 0]


def _summary_top4_features() -> FrozenFulltrackAppearanceFeatures:
    family = (
        "fixedprior_fulltrack_summarytop4_positive_uplift_multiscale_tanh_cap_"
        "traincal_balanced"
    )
    spec = FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES[family]
    names = tuple(
        f"{profile}__uniform_top4_mean_ncc" for profile in spec.profile_names
    )
    correct_columns = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    # The first two train and both validation rows have a wrong frozen top-one;
    # the remaining train rows exercise the top-one stability term.
    candidate_probability = np.asarray(
        [
            [0.2, 0.7],
            [0.7, 0.2],
            [0.7, 0.2],
            [0.2, 0.7],
            [0.2, 0.7],
            [0.7, 0.2],
        ],
        dtype=np.float32,
    )
    values = np.full((len(correct_columns), 2, len(names)), -2.0, dtype=np.float32)
    values[np.arange(len(correct_columns)), correct_columns] = 2.0
    return FrozenFulltrackAppearanceFeatures(
        paths=(Path("/tmp/fulltrack-summary-top4-fixture.npz"),),
        query_ids=np.asarray(
            ["train-a.png", "train-b.png", "train-c.png", "train-d.png", "val-a.png", "val-b.png"]
        ),
        split_names=np.asarray(
            ["train", "train", "train", "train", "validation", "validation"]
        ),
        source_row_indices=np.arange(len(correct_columns), dtype=np.int64),
        xy=np.zeros((len(correct_columns), 2), dtype=np.float32),
        candidate_track_ids=np.arange(
            len(correct_columns) * 2, dtype=np.int64
        ).reshape(len(correct_columns), 2),
        candidate_probabilities=candidate_probability,
        null_probabilities=np.full((len(correct_columns),), 0.1, dtype=np.float32),
        candidate_summary_features=values,
        candidate_summary_feature_valid=np.ones_like(values, dtype=bool),
        feature_names=names,
        profile_names=spec.profile_names,
        artifact_metadata=({},),
        compatibility={},
    )


def test_summary_top4_probe_is_fixed_mass_and_recovers_holdout_identity() -> None:
    family = (
        "fixedprior_fulltrack_summarytop4_positive_uplift_multiscale_tanh_cap_"
        "traincal_balanced"
    )
    features = _summary_top4_features()
    membership = np.asarray(
        [[True, False], [False, True], [True, False], [False, True]], dtype=bool
    )
    model, normalizer, metadata = fit_fixedprior_fulltrack_summary_top4_probe(
        features=features,
        family=family,
        train_normalizer_rows=np.asarray([0, 1, 2, 3], dtype=np.int64),
        target_train_rows=np.asarray([0, 1, 2, 3], dtype=np.int64),
        target_candidate_membership=membership,
        device=torch.device("cpu"),
        epochs=120,
        batch_size=4,
        learning_rate=0.1,
        seed=23,
    )
    candidate, null, residual = predict_fixedprior_fulltrack_summary_top4_probe(
        model=model,
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=6,
    )
    assert metadata["per_view_model"] is False
    assert metadata["coarse_top1_correct_train_pair_count"] == 2
    np.testing.assert_array_equal(null, features.null_probabilities)
    np.testing.assert_allclose(candidate.sum(axis=1) + null, 1.0, atol=1e-6)
    assert np.all(residual >= 0.0)
    assert np.argmax(candidate[4:], axis=1).tolist() == [0, 1]


def test_summary_top4_missing_topone_reference_is_exactly_neutral() -> None:
    family = (
        "fixedprior_fulltrack_summarytop4_positive_uplift_multiscale_tanh_cap_"
        "traincal_balanced"
    )
    features = _summary_top4_features()
    indices = np.arange(len(features.feature_names), dtype=np.int64)
    normalizer = fit_fulltrack_summary_top4_relative_normalizer(
        features, feature_indices=indices, train_rows=np.asarray([0, 1, 2, 3])
    )
    values = features.candidate_summary_features.copy()
    valid = features.candidate_summary_feature_valid.copy()
    # Row zero's frozen top-one is column one.  Losing it makes all candidate
    # comparisons unknown for that profile, rather than an implicit penalty.
    values[0, 1, 0] = np.nan
    valid[0, 1, 0] = False
    missing_reference = FrozenFulltrackAppearanceFeatures(
        **{
            **features.__dict__,
            "candidate_summary_features": values,
            "candidate_summary_feature_valid": valid,
        }
    )
    relative, common = normalized_fulltrack_summary_top4_relative_features(
        missing_reference, normalizer
    )
    assert common[0, :, 0].tolist() == [False, False]
    assert relative[0, :, 0].tolist() == [0.0, 0.0]
