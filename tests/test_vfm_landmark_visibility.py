import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera, ColmapTrackObservation
from feature_extract.vfm.landmark_visibility import (
    LandmarkVisibilityIndex,
    coverage_balanced_track_observations,
    count_projected_landmarks,
    filter_landmarks_by_visibility,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def _obs(track_id: int, image_id: str, x: float = 0.0) -> ColmapTrackObservation:
    return ColmapTrackObservation(
        track_id=track_id,
        image_id=image_id,
        point2d_idx=track_id,
        xy=(x, 1.0),
        xyz=np.asarray([x, 0.0, 5.0], dtype=np.float64),
        track_length=2,
        reprojection_error=0.1,
        image_width=100,
        image_height=100,
    )


def test_visibility_index_filters_bank_and_reports_full_counts(tmp_path) -> None:
    visibility = LandmarkVisibilityIndex.from_observations(
        [_obs(1, "ref_a"), _obs(2, "ref_a"), _obs(4, "ref_a"), _obs(3, "ref_b")]
    )
    path = tmp_path / "visibility.npz"
    visibility.save_npz(path)
    loaded = LandmarkVisibilityIndex.load_npz(path)
    index = LandmarkMapIndex(
        track_ids=np.asarray([2, 3], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]], dtype=np.float64),
        features=np.ones((2, 4), dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.ones((2,), dtype=np.int64),
        observation_image_ids=((), ()),
    )

    subset, gate = filter_landmarks_by_visibility(index, loaded, ["ref_a"])

    assert loaded.visible_tracks(["ref_a"]) == {1, 2, 4}
    assert subset.track_ids.tolist() == [2]
    assert gate["full_visible_tracks"] == 3
    assert gate["bank_visible_tracks"] == 1
    assert gate["bank_visibility_coverage"] == 1 / 3


def test_projected_landmark_count_uses_pose_and_camera() -> None:
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 3], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [100.0, 0.0, 5.0], [0.0, 0.0, -1.0]], dtype=np.float64),
        features=np.ones((3, 4), dtype=np.float32),
        mean_variances=np.zeros((3,), dtype=np.float32),
        observation_counts=np.ones((3,), dtype=np.int64),
        observation_image_ids=((), (), ()),
    )
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))

    assert count_projected_landmarks(index, np.eye(4, dtype=np.float64), camera) == 1


def test_coverage_balanced_sampling_covers_late_images_not_prefix_only() -> None:
    observations = []
    for track_id in range(10):
        observations.extend([_obs(track_id, "early_a"), _obs(track_id, "early_b")])
    for track_id in range(10, 14):
        observations.extend([_obs(track_id, "late_a"), _obs(track_id, "late_b")])

    sampled = coverage_balanced_track_observations(
        observations,
        max_observations=8,
        observations_per_track=2,
        min_track_observations=2,
        seed=0,
    )

    sampled_images = {obs.image_id for obs in sampled}
    sampled_tracks = {obs.track_id for obs in sampled}
    assert "late_a" in sampled_images or "late_b" in sampled_images
    assert any(track_id >= 10 for track_id in sampled_tracks)
    assert len(sampled) <= 8
