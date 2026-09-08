import numpy as np
import pytest


def test_source_isolated_centers_remove_same_view_and_balance_remaining_views(monkeypatch):
    from feature_extract.tools.vfm import train_goal_maplet_mapping_canonical_subtoken_head as module
    monkeypatch.setattr(module, "_project_world_to_pixel", lambda world, *args:
                        (np.tile([1.5, 1.5], (len(world), 1)), np.ones(len(world))))
    world = np.c_[np.arange(5) * .01, np.zeros(5), np.ones(5)]
    data = module._fit_prototype_dataset(
        np.arange(5), np.zeros(5, np.int64), np.arange(5),
        np.asarray(["fit"] * 4 + ["val"]), {"fit"}, "val", world,
        np.tile([1., 0.], (5, 1)), np.zeros(5, np.int64),
        np.tile(np.eye(4), (5, 1, 1)), np.tile(np.eye(3), (5, 1, 1)), np.zeros(5),
        source_view_per_observation=np.asarray([0, 0, 1, 2, 3]),
    )
    rows = data["fit_query_rows"]
    np.testing.assert_allclose(data["fit_map_world"][rows == 0, 0], [.025])
    np.testing.assert_allclose(data["fit_map_world"][rows == 2, 0], [.0175])
    np.testing.assert_allclose(data["validation_map_world"][:, 0], [(.005 + .02 + .03) / 3])


def test_view_cell_mean_uses_all_tokens_without_cross_observation_mixing():
    from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import (
        _view_cell_mean_features,
    )
    raw = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], np.float32)
    original = raw.copy()
    pooled = _view_cell_mean_features(raw, np.eye(2, dtype=np.float32),
                                     np.asarray([0, 0, 0, 1]),
                                     np.asarray([0, 0, 1, 0]), np.asarray([0, 2, 3]))
    np.testing.assert_allclose(pooled[0], np.asarray([1, 1]) / np.sqrt(2), atol=1e-7)
    np.testing.assert_allclose(pooled[2], [-1, 0])
    np.testing.assert_allclose(pooled[3], [0, -1])
    np.testing.assert_array_equal(pooled[1], [0, 0])
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("minimum_views", [1, 2])
def test_local_neighbors_exclude_all_rows_from_query_source_view(minimum_views):
    from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import (
        _mapping_local_feature_grid,
    )
    grid, valid = _mapping_local_feature_grid(
        np.asarray([0]), np.asarray([[1, 0]], np.float32), np.asarray([[0.25, 0.25, 0]]),
        leave_query_observation_out=True, neighbor_policy="atlas_modes_mean",
        minimum_neighbor_views=minimum_views,
        representative_rows=np.arange(4), representative_identity=np.asarray([0, 1, 1, 1]),
        identity_keys_plane_cell=np.asarray([[0, 0, 0], [0, 1, 0]]),
        observation=np.asarray([0, 0, 0, 1]), route_per_observation=np.asarray(["fit", "fit"]),
        fit_routes={"fit"}, projected_features=np.asarray([[1, 0], [1, 0], [1, 0], [0, 1]], np.float32),
        plane_rows=np.zeros(4, np.int64), plane_centers_world=np.zeros((1, 3)),
        plane_frames_world=np.asarray([np.eye(3)]),
    )
    assert bool(valid[0, 5]) == (minimum_views == 1)
    np.testing.assert_array_equal(grid[0, 5], [0, 1] if minimum_views == 1 else [0, 0])

from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import (
    _apply_coordinate_calibration,
    _closed_form_coordinate_affine,
    _closed_form_coordinate_shrinkage,
    _diverse_mode_indices,
    _fit_prototype_dataset,
    _isotropic_gaussian_nll,
    _isotropic_mixture_nll,
    _isotropic_variance_scale,
    _local_radio_correlation_volume,
    _local_correlation_reference_gate,
    _mapping_local_feature_grid,
    _project_query_local_feature_grid,
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


def test_local_correlation_volume_masks_missing_query_and_map_cells():
    query = np.zeros((1, 9, 2), np.float32)
    mapping = np.zeros_like(query)
    query[0, 4] = [1.0, 0.0]
    mapping[0, 4] = [0.5, np.sqrt(0.75)]
    qvalid = np.zeros((1, 9), bool); qvalid[0, 4] = True
    mvalid = np.zeros((1, 9), bool); mvalid[0, 4] = True
    context = _local_radio_correlation_volume(query, mapping, qvalid, mvalid)
    assert context.shape == (1, 99)
    assert context[0, 4 * 9 + 4] == np.float32(0.5)
    assert np.count_nonzero(context[0, :81]) == 1
    np.testing.assert_array_equal(context[0, 81:90], qvalid[0])
    np.testing.assert_array_equal(context[0, 90:99], mvalid[0])


def test_query_local_grid_stays_inside_same_plane_observation():
    # Tokens 65/66 share an observation. Token65 from another observation must
    # not be used even though it has the same grid location.
    observation = np.asarray([0, 0, 1], np.int64)
    token = np.asarray([65, 66, 65], np.int64)
    raw = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], np.float32)
    grid, valid = _project_query_local_feature_grid(
        np.asarray([0]), observation=observation, token_ids=token,
        raw_features=raw, projection=np.eye(2, dtype=np.float32),
    )
    assert valid[0, 4] and valid[0, 5]
    np.testing.assert_allclose(grid[0, 4], [1.0, 0.0])
    np.testing.assert_allclose(grid[0, 5], [0.0, 1.0])


@pytest.mark.parametrize("neighbor_policy", ["canonical_mean", "atlas_modes_mean"])
def test_mapping_local_grid_excludes_entire_fit_query_observation(neighbor_policy):
    # Identity1's neighbor has observations0/1. Query observation0 must be
    # removed, leaving only observation1's [0,1] descriptor.
    grid, valid = _mapping_local_feature_grid(
        np.asarray([0]), np.asarray([[1.0, 0.0]], np.float32),
        np.asarray([[0.25, 0.25, 0.0]]),
        leave_query_observation_out=True,
        neighbor_policy=neighbor_policy,
        representative_rows=np.arange(4),
        representative_identity=np.asarray([0, 0, 1, 1]),
        identity_keys_plane_cell=np.asarray([[0, 0, 0], [0, 1, 0]]),
        observation=np.asarray([0, 1, 0, 1]),
        route_per_observation=np.asarray(["seq1", "seq2"]),
        fit_routes={"seq1", "seq2"},
        projected_features=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], np.float32,
        ),
        plane_rows=np.zeros(4, np.int64),
        plane_centers_world=np.zeros((1, 3)),
        plane_frames_world=np.asarray([np.eye(3)]),
    )
    assert valid[0, 4] and valid[0, 5]
    np.testing.assert_allclose(grid[0, 5], [0.0, 1.0], atol=1e-7)


def test_mapping_local_modes_match_runtime_aggregation_and_exclude_validation():
    from feature_extract.tools.vfm.build_goal_maplet_plane_uv_radio_correspondences import (
        _atlas_local_cell_feature_lookup,
    )
    from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import (
        _diverse_mode_indices,
    )
    features = np.asarray([[1, 0], [1, 0], [1, 0], [0, 1], [-0.8, 0.2], [0.3, -1],
                           [0.7, 0.7]], np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    grid, valid = _mapping_local_feature_grid(
        np.asarray([6]), features[6:7], np.asarray([[0.25, 0.25, 0]]),
        leave_query_observation_out=False, neighbor_policy="atlas_modes_mean",
        representative_rows=np.arange(7), representative_identity=np.ones(7, np.int64),
        identity_keys_plane_cell=np.asarray([[0, 0, 0], [0, 1, 0]]),
        observation=np.arange(7), route_per_observation=np.asarray(["fit"] * 6 + ["val"]),
        fit_routes={"fit"}, projected_features=features, plane_rows=np.zeros(7, np.int64),
        plane_centers_world=np.zeros((1, 3)), plane_frames_world=np.asarray([np.eye(3)]),
    )
    selected = features[:6][_diverse_mode_indices(features[:6], 4)].astype(np.float16)
    lookup = _atlas_local_cell_feature_lookup(
        np.asarray([0, 4]), np.tile([0.75, 0.25], (4, 1)), np.zeros(4, np.int64), selected, 0.5,
    )
    assert valid[0, 5]
    np.testing.assert_allclose(grid[0, 5], lookup[(0, 1, 0)], atol=1e-7)


def test_local_reference_gate_requires_every_v11_metric_to_be_nondecreasing():
    def metrics(value: float):
        return {
            "predicted_error_px": {"median": value, "p90": value},
            "predicted_isotropic_gaussian_nll": value,
            "positive_match_probability_mean": 0.7,
            "negative_match_probability_mean": 0.3,
            "uncertainty_quantile_mean_error_px": [0.1, 0.2, 0.3, 0.4],
            "chart_uv_offset": {
                "predicted_error_m": {"median": value, "p90": value},
                "predicted_isotropic_gaussian_nll": value,
                "uncertainty_quantile_mean_error_m": [0.1, 0.2, 0.3, 0.4],
            },
        }
    passed, summary = _local_correlation_reference_gate(metrics(0.5), metrics(0.6))
    assert passed and all(summary["criteria"].values())
    worse = metrics(0.5)
    worse["chart_uv_offset"]["predicted_error_m"]["p90"] = 0.7
    passed, summary = _local_correlation_reference_gate(worse, metrics(0.6))
    assert not passed and not summary["criteria"]["chart_uv_p90_nonincrease"]
