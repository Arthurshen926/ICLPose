import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)


def test_project_world_points_simple_radial_center_and_radius():
    camera = ColmapCamera(
        camera_id=1,
        model_id=2,
        width=100,
        height=80,
        params=(50.0, 50.0, 40.0, 0.1),
    )
    pixels, depth = project_world_points(
        np.asarray([[0, 0, 2], [1, 0, 2]], dtype=np.float64),
        np.eye(4),
        camera,
    )
    assert np.allclose(pixels[0], [50.0, 40.0])
    assert np.isclose(pixels[1, 0], 50.0 + 50.0 * 0.5 * 1.025)
    assert np.allclose(depth, [2.0, 2.0])
