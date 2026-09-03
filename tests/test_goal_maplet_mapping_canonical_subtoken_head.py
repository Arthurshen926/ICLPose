import numpy as np

from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import (
    _apply_coordinate_calibration,
    _closed_form_coordinate_affine,
    _closed_form_coordinate_shrinkage,
    _diverse_mode_indices,
    _fit_prototype_dataset,
    _isotropic_gaussian_nll,
    _isotropic_mixture_nll,
    _isotropic_variance_scale,
    _mixture_offset_metrics,
    _mixture_variance_scale,
    _surface_geometric_context,
    _mapping_homography_projection,
    _undistort_simple_radial_xy,
)


def test_validation_prototype_excludes_validation_route():
    # identity0 has fit observations rows0/1 and validation row2.
    reps=np.asarray([0,1,2],np.int64); ids=np.zeros(3,np.int64)
    obs=np.asarray([0,1,2],np.int64); routes=np.asarray(["seq1","seq2","seq9"])
    world=np.asarray([[-.04,-.04,2],[-.02,-.04,2],[-.03,-.04,2]],np.float64)
    feat=np.zeros((3,2),np.float32); feat[0]=[1,0]; feat[1]=[0,1]; feat[2]=[-1,0]
    token=np.asarray([64*17+31]*3,np.int64)
    pose=np.repeat(np.eye(4)[None],3,axis=0)
    K=np.repeat(np.asarray([[100,0,127.5],[0,100,71.5],[0,0,1.]])[None],3,axis=0)
    out=_fit_prototype_dataset(reps,ids,obs,routes,{"seq1","seq2"},"seq9",world,feat,token,pose,K,np.zeros(3))
    assert len(out["validation_query_rows"])==1
    np.testing.assert_allclose(out["validation_map_features"][0],[2**-.5,2**-.5],atol=1e-6)
    assert not np.allclose(out["validation_map_features"][0],feat[2])


def test_diverse_modes_match_medoid_then_farthest_rule():
    feature = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0], [0.0, 1.0]], np.float32,
    )
    selected = _diverse_mode_indices(feature, 3)
    assert selected.tolist() == [3, 0, 2]


def test_mapping_homography_projection_recovers_affine_chart_coordinates():
    token = np.asarray([0, 10, 640, 650, 1300], np.int64)
    xy = np.c_[token % 64, token // 64].astype(np.float64)
    uv = np.c_[0.2 * xy[:, 0] + 1.0, -0.3 * xy[:, 1] + 2.0]
    world = np.c_[uv, np.zeros(len(uv))]
    projected, valid = _mapping_homography_projection(
        np.arange(len(token)), world,
        observation=np.zeros(len(token), np.int64), token_ids=token,
        plane_rows=np.zeros(len(token), np.int64),
        plane_centers_world=np.zeros((1, 3)),
        plane_frames_world=np.asarray([[[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]]]),
    )
    assert valid.all()
    np.testing.assert_allclose(projected, uv, atol=1e-12)


def test_coordinate_shrinkage_is_closed_form_and_bounded():
    mean = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    target = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    assert _closed_form_coordinate_shrinkage(mean, target) == 0.5
    assert _closed_form_coordinate_shrinkage(mean, 3.0 * mean) == 1.0
    assert _closed_form_coordinate_shrinkage(mean, -mean) == 0.0


def test_variance_scale_is_closed_form_and_improves_nll():
    mean = np.zeros((2, 2))
    target = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    raw = np.full(2, 4.0)
    scale = _isotropic_variance_scale(mean, target, raw)
    assert scale == 0.125
    assert _isotropic_gaussian_nll(mean, target, scale * raw) < _isotropic_gaussian_nll(mean, target, raw)


def test_coordinate_affine_recovers_axis_coupling_and_bias():
    mean = np.asarray([[-1.0, -0.5], [0.0, 0.0], [0.5, 1.0], [1.0, -1.0]])
    expected_matrix = np.asarray([[0.7, 0.1], [-0.2, 0.8]])
    expected_bias = np.asarray([0.05, -0.08])
    target = mean @ expected_matrix + expected_bias
    matrix, bias = _closed_form_coordinate_affine(mean, target)
    np.testing.assert_allclose(matrix, expected_matrix, atol=1e-12)
    np.testing.assert_allclose(bias, expected_bias, atol=1e-12)
    np.testing.assert_allclose(
        _apply_coordinate_calibration(
            mean, shrinkage=1.0, affine_matrix=matrix, affine_bias=bias,
        ),
        target,
        atol=1e-12,
    )


def test_mixture_likelihood_rewards_a_correct_low_weight_mode():
    target = np.asarray([[0.2, -0.1], [-0.2, 0.1]])
    means = np.asarray([
        [[0.2, -0.1], [-0.3, 0.3]],
        [[0.3, -0.3], [-0.2, 0.1]],
    ])
    variance = np.full((2, 2), 0.01)
    logits = np.zeros((2, 2))
    good = _isotropic_mixture_nll(means, target, variance, logits)
    bad = _isotropic_mixture_nll(np.zeros_like(means), target, variance, logits)
    assert good < bad
    metrics = _mixture_offset_metrics(target, means, variance, logits)
    assert metrics["best_mode_oracle_error_m"]["median"] == 0.0
    assert metrics["mean_effective_mode_count"] == 2.0


def test_mixture_variance_scale_is_positive_and_finite():
    target = np.asarray([[0.1, 0.0], [-0.1, 0.0]])
    means = np.zeros((2, 2, 2))
    variance = np.full((2, 2), 0.25)
    logits = np.zeros((2, 2))
    scale = _mixture_variance_scale(means, target, variance, logits)
    assert np.isfinite(scale) and scale > 0.0


def test_simple_radial_undistortion_replays_forward_model():
    ideal = np.asarray([[0.1, 0.2], [-0.3, 0.05]])
    radial = np.asarray([0.1, -0.05])
    distorted = ideal * (1.0 + radial[:, None] * np.sum(ideal * ideal, axis=1, keepdims=True))
    matrix = np.repeat(np.asarray([[100.0, 0.0, 127.5], [0.0, 100.0, 71.5], [0.0, 0.0, 1.0]])[None], 2, axis=0)
    pixel = distorted * matrix[:, (0, 1), (0, 1)] + matrix[:, (0, 1), (2, 2)]
    np.testing.assert_allclose(_undistort_simple_radial_xy(pixel, matrix, radial), ideal, atol=1e-12)


def test_surface_context_contains_cell_phase_and_valid_incidence():
    token = np.asarray([64 * 17 + 31])
    observation = np.asarray([0])
    plane = np.asarray([0])
    pose = np.eye(4)[None]
    matrix = np.asarray([[[100.0, 0.0, 127.5], [0.0, 100.0, 71.5], [0.0, 0.0, 1.0]]])
    context = _surface_geometric_context(
        np.asarray([0]), np.asarray([[0.10, -0.10, 2.0]]),
        observation=observation, token_ids=token, plane_rows=plane,
        poses_w2c=pose, camera_matrices=matrix, radial_coefficients=np.zeros(1),
        plane_centers_world=np.asarray([[0.0, 0.0, 2.0]]),
        plane_frames_world=np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
    )
    np.testing.assert_allclose(context[0, 2:4], [-0.6, 0.6], atol=1e-6)
    assert 0.0 <= context[0, 4] <= 1.0
