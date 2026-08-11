from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.joint_phase_geometry_likelihood import (
    COMPONENT_NAMES,
    JointPhaseGeometryLikelihood,
    candidate_measurements,
    load_joint_phase_geometry_likelihood,
)


def _phase(value: float) -> dict[str, float]:
    return {
        "jacobian_observability": value,
        "missing_fraction_mean": 0.1,
        "mixed_surface_fraction": 0.2,
    }


def test_joint_likelihood_combines_typed_monotonic_evidence() -> None:
    measurements = candidate_measurements(
        np.asarray([0.1, 0.2]),
        np.asarray([0.2, 0.7]),
        [_phase(0.4), _phase(0.8)],
        [{"score": 0.3}, {"score": 0.9}],
    )
    policy = JointPhaseGeometryLikelihood(
        weights=np.ones(4), scale_floors=np.full(4, 0.01), metadata={},
    )
    probability = policy.posterior(measurements)
    assert probability.shape == (2,)
    assert probability[1] > probability[0]
    assert np.sum(probability) == pytest.approx(1.0)


def test_joint_likelihood_treats_unrenderable_geometry_as_zero_evidence() -> None:
    measurements = candidate_measurements(
        np.asarray([0.1]), np.asarray([0.2]), [_phase(0.4)], [{"score": None}],
    )
    assert measurements[0, 3] == 0.0


def test_joint_likelihood_v2_conserves_candidate_and_typed_null_mass() -> None:
    measurements = candidate_measurements(
        np.asarray([0.1, 0.2]),
        np.asarray([0.2, 0.7]),
        [_phase(0.4), _phase(0.8)],
        [{"score": 0.3}, {"score": 0.9}],
    )
    policy = JointPhaseGeometryLikelihood(
        weights=np.ones(4),
        scale_floors=np.full(4, 0.01),
        metadata={},
        null_logit=20.0,
    )
    probability, null = policy.posterior_with_null(measurements)
    assert float(np.sum(probability)) + null == pytest.approx(1.0)
    assert null > float(np.max(probability))


def test_joint_likelihood_loader_fails_closed_on_contract(tmp_path) -> None:
    payload = {
        "artifact_type": "goal_maplet_joint_phase_geometry_likelihood_v1",
        "component_names": list(COMPONENT_NAMES),
        "normalization": "per_query_median_iqr_with_fixed_floors_v1",
        "weights": [1.0, 1.0, 1.0, 1.0],
        "scale_floors": [0.05, 0.02, 0.02, 0.02],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload))
    assert load_joint_phase_geometry_likelihood(path).weights.shape == (4,)
    payload["component_names"][0] = "untyped_score"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="contract differs"):
        load_joint_phase_geometry_likelihood(path)


def test_joint_likelihood_v2_loader_requires_typed_null(tmp_path) -> None:
    payload = {
        "artifact_type": "goal_maplet_joint_phase_geometry_likelihood_v2",
        "component_names": list(COMPONENT_NAMES),
        "normalization": "per_query_median_iqr_with_fixed_floors_v1",
        "weights": [1.0, 1.0, 1.0, 1.0],
        "scale_floors": [0.05, 0.02, 0.02, 0.02],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="typed null"):
        load_joint_phase_geometry_likelihood(path)
