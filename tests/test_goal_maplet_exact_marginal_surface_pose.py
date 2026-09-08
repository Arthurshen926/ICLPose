import cv2
import numpy as np

from feature_extract.tools.vfm.refine_goal_maplet_exact_marginal_surface_pose import _objective, _image_covariance
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import _fixed_reprojection_sigma_px


def test_pose_free_reference_keeps_every_token_with_deterministic_physical_ties():
    from feature_extract.tools.vfm.refine_goal_maplet_exact_marginal_surface_pose import _pose_free_reference
    rows = _pose_free_reference(np.asarray([4, 4, 8, 8, 9]), np.asarray([2, 1, 3, 2, 1]),
                               np.asarray([6, 5, 4, 3, 2]), np.asarray([.8, .8, .9, .7, -.2]))
    np.testing.assert_array_equal(rows, [1, 2, 4])


def _problem():
    camera = np.asarray([[.2, .3, 3.], [.5, -.2, 4.], [-.3, .5, 3.5]])
    K = np.asarray([[120., 0, 128], [0, 115., 72.], [0, 0, 1.]])
    pixel = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), K, np.asarray([.03, 0., 0., 0., 0.]))[0].reshape(-1, 2)
    pixel += np.asarray([[.5, -.1], [-.4, .2], [.1, -.3]])
    cov = np.asarray([[[2., .3], [.3, 1.]], [[1., .2], [.2, 3.]], [[2., -.2], [-.2, 2.]]])
    return camera, pixel, K, .03, np.linalg.inv(cov), np.linalg.slogdet(cov)[1]


def test_exact_mixture_gradient_matches_finite_differences_with_radial_distortion():
    camera, pixel, K, k1, inv, logdet = _problem()
    args = (camera, pixel, K, k1, inv, logdet, np.asarray([0, 0, 1]),
            np.asarray([.4, .6, 1.]), np.asarray([.7, .8, .9]), np.asarray([.8, 1.2]))
    x = np.asarray([.002, -.003, .001, .01, -.01, .005])
    _, grad, _ = _objective(x, *args)
    numeric = []
    for j in range(6):
        step = np.zeros(6); step[j] = 1e-6
        numeric.append((_objective(x + step, *args)[0] - _objective(x - step, *args)[0]) / 2e-6)
    np.testing.assert_allclose(grad, numeric, atol=2e-6, rtol=2e-6)


def test_duplicate_component_splits_mass_without_objective_or_gradient_bonus():
    camera, pixel, K, k1, inv, logdet = _problem()
    indices = np.asarray([0, 1, 2, 0])
    original = _objective(np.zeros(6), camera, pixel, K, k1, inv, logdet,
        np.asarray([0, 0, 1]), np.asarray([.4, .6, 1.]), np.full(3, .8), np.ones(2))
    duplicate = _objective(np.zeros(6), camera[indices], pixel[indices], K, k1, inv[indices], logdet[indices],
        np.asarray([0, 0, 1, 0]), np.asarray([.2, .6, 1., .2]), np.full(4, .8), np.ones(2))
    for a, b in zip(original, duplicate):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_one_hot_no_null_reduces_to_weighted_gaussian_reprojection():
    camera, pixel, K, k1, inv, logdet = _problem()
    loss, _, mass = _objective(np.zeros(6), camera, pixel, K, k1, inv, logdet,
        np.arange(3), np.ones(3), np.ones(3), np.asarray([1., 2., 3.]))
    projected = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), K, np.asarray([k1, 0., 0., 0., 0.]))[0].reshape(-1, 2)
    residual = projected - pixel
    expected = .5 * (np.einsum("ni,nij,nj->n", residual, inv, residual) + logdet) + np.log(2 * np.pi)
    np.testing.assert_allclose(loss, np.average(expected, weights=[1, 2, 3]), atol=1e-12)
    np.testing.assert_allclose(mass, 3.)


def test_all_null_has_constant_density_zero_gradient_and_zero_match_mass():
    camera, pixel, K, k1, inv, logdet = _problem()
    loss, gradient, mass = _objective(np.zeros(6), camera, pixel, K, k1, inv, logdet,
        np.arange(3), np.ones(3), np.zeros(3), np.ones(3))
    np.testing.assert_allclose(loss, np.log(256. * 144.))
    np.testing.assert_array_equal(gradient, np.zeros(6))
    assert mass == 0.


def test_full_covariance_is_spd_and_isotropic_trace_matches_existing_sigma():
    world, _, K, k1, _, _ = _problem()
    pose = np.eye(4)
    centroid = np.tile(np.diag([.001, .004, .0002]), (3, 1, 1))
    purity = np.asarray([.3, .7, 1.]); qvar = np.asarray([.5, 1., 2.])
    full = _image_covariance(pose, world, centroid, qvar, purity, K, k1, False)
    scalar = _image_covariance(pose, world, centroid, qvar, purity, K, k1, True)
    sigma = _fixed_reprojection_sigma_px(pose, world, np.zeros((3, 3, 3)), purity, np.zeros(3), K,
        query_measurement_variance_px2=qvar, centroid_covariance_world_m2=centroid,
        include_map_footprint_scatter=False, radial_k1=k1)
    assert np.all(np.linalg.eigvalsh(full) > 0.)
    np.testing.assert_allclose(scalar[:, 0, 0], sigma**2, rtol=1e-12)
    assert not np.allclose(full[:, 0, 0], full[:, 1, 1])
