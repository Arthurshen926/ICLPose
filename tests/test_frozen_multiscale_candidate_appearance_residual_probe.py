from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.localization.frozen_multiscale_candidate_appearance_residual_probe import (
    FROZEN_APPEARANCE_RESIDUAL_FAMILIES,
    FixedPriorPerViewLinearResidual,
    FrozenAppearanceProbeFeatures,
    fit_appearance_feature_normalizer,
    fit_fixedprior_linear_residual,
    normalized_appearance_model_input,
    predict_fixedprior_linear_residual,
)


def _features() -> FrozenAppearanceProbeFeatures:
    names = tuple(
        dict.fromkeys(
            name
            for family in FROZEN_APPEARANCE_RESIDUAL_FAMILIES.values()
            for name in family
        )
    )
    count, candidates, views = 4, 2, 2
    probability = np.asarray(
        [[0.2, 0.7], [0.7, 0.2], [0.2, 0.7], [0.7, 0.2]], dtype=np.float32
    )
    # The expected identity is candidate 0 on even rows and candidate 1 on odd
    # rows, while the immutable base prior ranks the other candidate first.
    scores = np.full((count, candidates, views, len(names)), -2.0, dtype=np.float32)
    for row, correct in enumerate((0, 1, 0, 1)):
        scores[row, correct] = 2.0
    return FrozenAppearanceProbeFeatures(
        paths=(Path("/tmp/frozen-appearance-fixture.npz"),),
        query_ids=np.asarray(["train-a.png", "train-b.png", "val-a.png", "val-b.png"]),
        split_names=np.asarray(["train", "train", "validation", "validation"]),
        source_row_indices=np.arange(count, dtype=np.int64),
        xy=np.zeros((count, 2), dtype=np.float32),
        candidate_track_ids=np.asarray([[10, 11], [12, 13], [14, 15], [16, 17]]),
        candidate_probabilities=probability,
        null_probabilities=np.full((count,), 0.1, dtype=np.float32),
        candidate_view_weights=np.full((count, candidates, views), 0.5, dtype=np.float32),
        candidate_view_scores=scores,
        candidate_view_usable=np.ones_like(scores, dtype=bool),
        feature_names=names,
        metadata={},
    )


def test_zero_residual_exactly_preserves_fixed_candidate_and_null_priors() -> None:
    features = _features()
    indices = np.arange(len(features.feature_names), dtype=np.int64)
    normalizer = fit_appearance_feature_normalizer(
        features, feature_indices=indices, train_rows=np.asarray([0, 1])
    )
    model = FixedPriorPerViewLinearResidual(input_dim=len(indices) * 2)
    candidate, null, residual = predict_fixedprior_linear_residual(
        model=model,
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert np.allclose(candidate, features.candidate_probabilities, atol=1e-6)
    assert np.allclose(null, features.null_probabilities, atol=1e-6)
    assert np.allclose(residual, 0.0, atol=1e-7)


def test_train_only_residual_can_use_per_view_appearance_without_changing_null() -> None:
    features = _features()
    target_membership = np.asarray(
        [[True, False, False], [False, True, False]], dtype=bool
    )
    model, normalizer, metadata = fit_fixedprior_linear_residual(
        features=features,
        family="fixedprior_multiscale",
        train_rows=np.asarray([0, 1], dtype=np.int64),
        target_membership=target_membership,
        device=torch.device("cpu"),
        epochs=120,
        batch_size=2,
        learning_rate=0.1,
        weight_decay=0.0,
        seed=7,
    )
    candidate, null, _residual = predict_fixedprior_linear_residual(
        model=model,
        features=features,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert metadata["null_logit"] == "immutable_base_null_prior"
    # The null *logit* is immutable. Its normalized posterior may change when
    # visual residuals add evidence to candidates; zero-residual equivalence is
    # covered by the preceding test.
    assert np.all(null < features.null_probabilities)
    assert np.argmax(candidate[2:], axis=1).tolist() == [0, 1]


def test_missing_view_features_are_explicit_neutral_input_not_nonfinite() -> None:
    features = _features()
    scores = features.candidate_view_scores.copy()
    scores[0, 0, 0] = np.nan
    usable = features.candidate_view_usable.copy()
    usable[0, 0, 0] = False
    features = FrozenAppearanceProbeFeatures(
        **{
            **features.__dict__,
            "candidate_view_scores": scores,
            "candidate_view_usable": usable,
        }
    )
    normalizer = fit_appearance_feature_normalizer(
        features,
        feature_indices=np.asarray([0], dtype=np.int64),
        train_rows=np.asarray([0, 1]),
    )
    model_input = normalized_appearance_model_input(features, normalizer)
    assert np.all(np.isfinite(model_input))
    # Raw value plus finite mask: the missing view receives an explicit zero
    # standardized value and a zero availability flag.
    assert model_input[0, 0, 0].tolist() == [0.0, 0.0]
