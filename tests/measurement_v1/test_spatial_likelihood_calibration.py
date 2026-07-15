from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.vfm.measurement_v1.spatial_likelihood_calibration import (
    CandidateSpatialLikelihoodCalibration,
    fit_dustbin_platt_scaling,
    fit_spatial_temperature,
    load_candidate_spatial_likelihood_calibration,
    spatial_target_nll,
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def test_dustbin_platt_scaling_improves_true_binary_nll() -> None:
    rng = np.random.default_rng(81)
    raw_logits = rng.normal(size=6000)
    raw_probabilities = _sigmoid(raw_logits)
    true_probabilities = _sigmoid(0.45 * raw_logits + 0.7)
    labels = rng.random(len(raw_logits)) < true_probabilities

    fitted = fit_dustbin_platt_scaling(raw_probabilities, labels)

    assert 0.25 < fitted["logit_scale"] < 0.7
    assert 0.4 < fitted["logit_bias"] < 1.0
    assert fitted["calibrated_metrics"]["nll"] < fitted["raw_metrics"]["nll"]
    assert fitted["calibrated_metrics"]["ece"] < fitted["raw_metrics"]["ece"]


def test_spatial_temperature_recovers_softened_categorical_density() -> None:
    rng = np.random.default_rng(82)
    xs = np.linspace(-2.0, 2.0, 5)
    offsets = np.stack(np.meshgrid(xs, xs, indexing="xy"), axis=-1).reshape(-1, 2)
    logits = rng.normal(size=(2500, len(offsets))) * 2.0
    true_temperature = 2.4
    scaled = logits / true_temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    sampled = np.asarray(
        [rng.choice(len(offsets), p=row) for row in probabilities], dtype=np.int64
    )
    targets = offsets[sampled]

    fitted = fit_spatial_temperature(logits, offsets, targets)

    assert 1.8 < fitted["temperature"] < 3.1
    assert fitted["calibrated_nll"] < fitted["raw_nll"]
    assert np.isclose(
        fitted["calibrated_nll"],
        spatial_target_nll(
            logits, offsets, targets, temperature=fitted["temperature"]
        ),
    )


def test_spatial_calibration_enforces_manifest_and_production_contract(tmp_path) -> None:
    calibration = CandidateSpatialLikelihoodCalibration(
        spatial_temperature=2.0,
        dustbin_logit_scale=0.5,
        dustbin_logit_bias=0.25,
        measurement_checkpoint_sha256="checkpoint",
        search_radius_px=6.0,
        step_px=0.5,
        context_radius_px=12.0,
        query_source="real_pair",
        support_patch_warp="none",
        fit_query_manifest_sha256="fit",
        audit_query_manifest_sha256="audit",
        production_eligible=True,
        base_model_query_overlap_count=0,
    )
    metadata = {
        "measurement_checkpoint_sha256": "checkpoint",
        "search_radius_px": 6.0,
        "step_px": 0.5,
        "context_radius_px": 12.0,
        "query_source": "real_pair",
        "support_patch_warp": "none",
    }
    calibration.validate_spatial_metadata(metadata)
    local, dustbin = calibration.apply(
        np.asarray([[0.0, -2.0]], dtype=np.float32),
        np.asarray([0.2], dtype=np.float32),
    )
    np.testing.assert_allclose(local, [[0.0, -1.0]])
    assert 0.0 < float(dustbin[0]) < 1.0
    with pytest.raises(ValueError, match="incompatible"):
        calibration.validate_spatial_metadata({**metadata, "step_px": 1.0})

    diagnostic = CandidateSpatialLikelihoodCalibration(
        **{
            **calibration.__dict__,
            "production_eligible": False,
            "base_model_query_overlap_count": 3,
        }
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(diagnostic.to_dict()))
    with pytest.raises(ValueError, match="diagnostic"):
        load_candidate_spatial_likelihood_calibration(
            path, verify_artifact_manifest=False
        )
    loaded = load_candidate_spatial_likelihood_calibration(
        path, require_production=False, verify_artifact_manifest=False
    )
    assert not loaded.production_eligible
