import numpy as np
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.build_detector_weighted_2dgs_surface_field import (
    _assign_ray_disk_candidates,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera


def test_ray_disk_assignment_requires_plane_and_footprint_compatibility():
    camera = ColmapCamera(
        camera_id=1,
        model_id=2,
        width=100,
        height=80,
        params=(50.0, 50.0, 40.0, 0.0),
    )
    centers = np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64)
    common = dict(
        clean_tree=cKDTree(centers),
        clean_indices=np.asarray([7]),
        clean_centers=centers,
        clean_normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        clean_tangent1=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        clean_tangent2=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        clean_scale1=np.asarray([0.1], dtype=np.float32),
        clean_scale2=np.asarray([0.1], dtype=np.float32),
        clean_opacity=np.asarray([1.0], dtype=np.float32),
        camera=camera,
        pose_w2c=np.eye(4),
        candidate_count=1,
        maximum_candidate_center_m=1.0,
        maximum_plane_depth_residual_m=0.03,
        maximum_disk_sigma=3.0,
        minimum_ray_normal_cosine=0.05,
    )
    rows, xyz, valid, stats = _assign_ray_disk_candidates(
        np.asarray([[50.0, 40.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 2.01]], dtype=np.float64),
        **common,
    )
    assert rows.tolist() == [0]
    assert valid.tolist() == [True]
    assert np.allclose(xyz, centers)
    assert stats["accepted"] == 1

    rows, _xyz, valid, stats = _assign_ray_disk_candidates(
        np.asarray([[70.0, 40.0]], dtype=np.float32),
        np.asarray([[0.8, 0.0, 2.0]], dtype=np.float64),
        **common,
    )
    assert rows.tolist() == [-1]
    assert valid.tolist() == [False]
    assert stats["ray_disk_rejected"] == 1
