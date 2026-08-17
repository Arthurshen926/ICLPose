from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.se3_local_quadratic import (
    complete_quadratic_probe_coordinates,
    fit_complete_local_se3_quadratic,
    fit_local_se3_quadratic_least_squares,
    left_retract_pose_w2c,
    minimal_quadratic_probe_coordinates,
)


def test_full_quadratic_recovers_cross_terms_and_bias():
    probes = complete_quadratic_probe_coordinates()
    hessian = np.diag(np.arange(1.0, 7.0))
    hessian[0, 3] = hessian[3, 0] = 0.35
    hessian[1, 5] = hessian[5, 1] = -0.20
    gradient = np.asarray([0.1, -0.2, 0.05, 0.1, 0.0, -0.05])
    scores = {
        name: float(2.0 - gradient @ x - 0.5 * x @ hessian @ x)
        for name, x in probes.items()
    }
    fitted = fit_complete_local_se3_quadratic(scores)
    np.testing.assert_allclose(fitted.loss_hessian, hessian, atol=1e-12)
    np.testing.assert_allclose(fitted.loss_gradient, gradient, atol=1e-12)
    np.testing.assert_allclose(
        fitted.predicted_bias_normalized,
        -np.linalg.solve(hessian, gradient),
        atol=1e-12,
    )
    assert fitted.positive_definite
    assert np.isfinite(fitted.hessian_condition_number)


def test_axiswise_maxima_do_not_imply_positive_definite_full_hessian():
    probes = complete_quadratic_probe_coordinates()
    hessian = np.eye(6)
    hessian[0, 1] = hessian[1, 0] = 2.0
    scores = {
        name: float(-0.5 * x @ hessian @ x) for name, x in probes.items()
    }
    fitted = fit_complete_local_se3_quadratic(scores)
    assert np.all(np.diag(fitted.loss_hessian) > 0.0)
    assert not fitted.positive_definite
    assert fitted.hessian_eigenvalues[0] < 0.0
    assert np.isinf(fitted.hessian_condition_number)


def test_quadratic_requires_exact_probe_inventory():
    probes = complete_quadratic_probe_coordinates()
    probes.pop("tx+")
    with pytest.raises(ValueError, match="probe set differs"):
        fit_complete_local_se3_quadratic({key: 0.0 for key in probes})


def test_minimal_design_recovers_same_quadratic():
    probes = minimal_quadratic_probe_coordinates()
    hessian = np.diag(np.linspace(1.0, 2.0, 6))
    hessian[2, 4] = hessian[4, 2] = 0.3
    gradient = np.linspace(-0.2, 0.2, 6)
    scores = {
        name: float(3.0 - gradient @ x - 0.5 * x @ hessian @ x)
        for name, x in probes.items()
    }
    fitted = fit_local_se3_quadratic_least_squares(probes, scores)
    np.testing.assert_allclose(fitted.loss_hessian, hessian, atol=1e-11)
    np.testing.assert_allclose(fitted.loss_gradient, gradient, atol=1e-11)
    assert fitted.fit_root_mean_square_error < 1e-12


def test_left_se3_retraction_is_single_coupled_group_operation_and_immutable():
    pose = np.eye(4, dtype=np.float64)
    coordinate = np.asarray([1.0, -0.5, 0.25, 0.0, 0.0, 1.0])
    pose_before, coordinate_before = pose.copy(), coordinate.copy()
    result = left_retract_pose_w2c(
        pose, coordinate, translation_step_m=0.5, rotation_step_degrees=10.0,
    )
    np.testing.assert_array_equal(pose, pose_before)
    np.testing.assert_array_equal(coordinate, coordinate_before)
    np.testing.assert_allclose(result[:3, :3] @ result[:3, :3].T, np.eye(3), atol=1e-12)
    assert np.linalg.det(result[:3, :3]) == pytest.approx(1.0)
    # Coupling through the SE(3) left Jacobian means this is not a sequential
    # raw translation followed by an unrelated rotation probe.
    assert not np.allclose(result[:3, 3], coordinate[:3] * 0.5)
    np.testing.assert_array_equal(result[3], [0.0, 0.0, 0.0, 1.0])


def test_left_se3_retraction_pure_translation_and_rotation_limits():
    pose = np.eye(4, dtype=np.float64)
    translated = left_retract_pose_w2c(
        pose, [1, 2, 3, 0, 0, 0],
        translation_step_m=0.25, rotation_step_degrees=5.0,
    )
    np.testing.assert_allclose(translated[:3, 3], [0.25, 0.5, 0.75])
    rotated = left_retract_pose_w2c(
        pose, [0, 0, 0, 1, 0, 0],
        translation_step_m=0.25, rotation_step_degrees=5.0,
    )
    np.testing.assert_allclose(rotated[:3, 3], 0.0)
    assert np.degrees(np.arccos(np.clip((np.trace(rotated[:3, :3]) - 1) / 2, -1, 1))) == pytest.approx(5.0)
