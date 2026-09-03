import json

import numpy as np

from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _load_pose_candidate,
    _projection_variance_px2,
    _select,
    _uncertainty_normalized_token_likelihood,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256


def test_projection_covariance_and_density_penalize_uncertainty():
    xyz = np.asarray([[0.0, 0.0, 2.0]])
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    assert _projection_variance_px2(xyz, np.zeros((1, 3, 3)), np.eye(3), K)[0] == 0.0
    pose = np.eye(4)
    token = np.asarray([0])
    clean, _ = _uncertainty_normalized_token_likelihood(
        pose, xyz, token, np.zeros((1, 3, 3)), np.ones(1), K, 0.0,
    )
    uncertain, _ = _uncertainty_normalized_token_likelihood(
        pose, xyz, token, np.eye(3)[None], np.ones(1), K, 0.0,
    )
    assert clean == 1.0
    assert uncertain < clean


def test_simple_radial_covariance_jacobian_matches_finite_difference():
    xyz = np.asarray([[0.7, -0.4, 3.2]])
    covariance = np.asarray([[[0.04, 0.01, 0.0], [0.01, 0.03, 0.002], [0.0, 0.002, 0.02]]])
    K = np.asarray([[310.0, 0.0, 127.5], [0.0, 305.0, 71.5], [0.0, 0.0, 1.0]])
    k1 = -0.08
    analytic = _projection_variance_px2(
        xyz, covariance, np.eye(3), K, radial_k1=k1,
    )[0]

    def project(point):
        x, y = point[:2] / point[2]
        scale = 1.0 + k1 * (x * x + y * y)
        return np.asarray([K[0, 0] * x * scale + K[0, 2], K[1, 1] * y * scale + K[1, 2]])

    epsilon = 1e-6
    jacobian = np.column_stack([
        (project(xyz[0] + epsilon * np.eye(3)[axis])
         - project(xyz[0] - epsilon * np.eye(3)[axis])) / (2.0 * epsilon)
        for axis in range(3)
    ])
    expected = 0.5 * np.trace(jacobian @ covariance[0] @ jacobian.T)
    np.testing.assert_allclose(analytic, expected, rtol=1e-9, atol=1e-9)


def test_purity_and_per_token_marginalization():
    pose = np.eye(4)
    xyz = np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    score, token = _uncertainty_normalized_token_likelihood(
        pose, xyz, np.asarray([0, 0]), np.zeros((2, 3, 3)), np.asarray([0.5, 0.8]), K, 0.0,
    )
    assert token.tolist() == [0.8]
    assert score == 0.8


def test_selection_is_stable():
    score = np.asarray([[[0.5, 0.5], [0.6, 0.6]], [[0.5, 0.5], [0.5, 0.5]]])
    assert _select(score).tolist() == [1, 0]


def test_explicit_null_is_invariant_to_duplicate_hypotheses():
    pose = np.eye(4)
    xyz = np.asarray([[0.0, 0.0, 2.0]])
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    kwargs = dict(
        pose_w2c=pose, covariance_world_m2=np.zeros((1, 3, 3)),
        plane_purity=np.ones(1), camera_matrix=K, radial_k1=0.0,
        query_measurement_variance_px2=np.asarray([1.0]),
        correspondence_match_probability=np.asarray([0.75]),
        explicit_null_marginalization=True,
    )
    one, _ = _uncertainty_normalized_token_likelihood(
        world_points=xyz, query_tokens=np.asarray([0]), **kwargs,
    )
    two, _ = _uncertainty_normalized_token_likelihood(
        world_points=np.repeat(xyz, 2, axis=0), query_tokens=np.asarray([0, 0]),
        covariance_world_m2=np.zeros((2, 3, 3)), plane_purity=np.ones(2),
        pose_w2c=pose, camera_matrix=K, radial_k1=0.0,
        query_measurement_variance_px2=np.ones(2),
        correspondence_match_probability=np.full(2, 0.75),
        explicit_null_marginalization=True,
    )
    assert one == two


def test_explicit_null_downweights_a_low_match_hypothesis():
    pose = np.eye(4)
    xyz = np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    high, _ = _uncertainty_normalized_token_likelihood(
        pose, xyz[:1], np.asarray([0]), np.zeros((1, 3, 3)), np.ones(1), K, 0.0,
        query_measurement_variance_px2=np.ones(1),
        correspondence_match_probability=np.asarray([0.9]),
        explicit_null_marginalization=True,
    )
    low, _ = _uncertainty_normalized_token_likelihood(
        pose, xyz[1:], np.asarray([0]), np.zeros((1, 3, 3)), np.ones(1), K, 0.0,
        query_measurement_variance_px2=np.ones(1),
        correspondence_match_probability=np.asarray([0.1]),
        explicit_null_marginalization=True,
    )
    assert high > low


def test_geometry_consensus_is_a_valid_frozen_refinement_initializer(tmp_path):
    arrays = {
        "names": np.asarray(["query"]),
        "pose_w2c": np.eye(4)[None],
        "usable": np.asarray([True]),
    }
    metadata = {
        "artifact_type": "goal_maplet_coordinate_pose_geometry_consensus_v1",
        "query_pose_or_ground_truth_read": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "arrays_sha256": arrays_sha256(arrays),
    }
    path = tmp_path / "consensus.npz"
    np.savez_compressed(
        path, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    loaded, loaded_metadata = _load_pose_candidate(path)
    np.testing.assert_array_equal(loaded["pose_w2c"], arrays["pose_w2c"])
    assert loaded_metadata["artifact_type"] == metadata["artifact_type"]
