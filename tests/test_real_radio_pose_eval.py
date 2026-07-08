from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.localization.pose_eval import build_support_observation_index


def _obs(image_id: str, track_id: int, xy: tuple[float, float]) -> ColmapTrackObservation:
    return ColmapTrackObservation(
        track_id=track_id,
        image_id=image_id,
        point2d_idx=track_id,
        xy=xy,
        xyz=np.asarray([float(track_id), 0.0, 4.0], dtype=np.float64),
        track_length=3,
        reprojection_error=0.25,
        camera_id=1,
        image_width=100,
        image_height=80,
    )


def test_support_observation_index_chooses_nearest_within_radius() -> None:
    index = build_support_observation_index(
        [
            _obs("seq/r.png", 11, (10.0, 10.0)),
            _obs("seq/r.png", 12, (15.0, 10.0)),
            _obs("seq/other.png", 99, (10.0, 10.0)),
        ]
    )

    match = index.nearest("seq/r.png", np.asarray([14.0, 10.0], dtype=np.float32), max_distance_px=2.0)

    assert match is not None
    assert match.observation.track_id == 12
    assert match.distance_px == 1.0


def test_support_observation_index_respects_radius_and_image_id() -> None:
    index = build_support_observation_index([_obs("seq/r.png", 11, (10.0, 10.0))])

    assert index.nearest("seq/r.png", np.asarray([30.0, 30.0], dtype=np.float32), max_distance_px=4.0) is None
    assert index.nearest("seq/missing.png", np.asarray([10.0, 10.0], dtype=np.float32), max_distance_px=4.0) is None
