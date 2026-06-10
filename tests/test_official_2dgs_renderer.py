from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.official_2dgs_renderer import load_official_2dgs_source_from_ply, scaled_camera_matrix


def _write_ply(path, rows, dtype):
    plyfile = pytest.importorskip("plyfile")
    elements = np.zeros((len(rows),), dtype=dtype)
    for idx, row in enumerate(rows):
        elements[idx] = tuple(row)
    plyfile.PlyData([plyfile.PlyElement.describe(elements, "vertex")]).write(path)


def test_load_official_2dgs_source_accepts_complete_2dgs_ply(tmp_path) -> None:
    path = tmp_path / "complete_2dgs.ply"
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        ("loc_0", "f4"),
    ]
    _write_ply(
        path,
        [[0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.5, 0.25, 0.125, 2.0, -2.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.75]],
        dtype,
    )

    source = load_official_2dgs_source_from_ply(path)

    assert source.gaussian_count == 1
    assert source.sh_degree == 0
    assert source.log_scales_2d.shape == (1, 2)
    assert source.rotations.shape == (1, 4)
    assert source.sh_features.shape == (1, 1, 3)
    assert np.allclose(source.loc_features, [[0.75]])


def test_load_official_2dgs_source_rejects_single_scale_proxy_ply(tmp_path) -> None:
    path = tmp_path / "proxy.ply"
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    _write_ply(path, [[0.0, 0.0, 4.0, 0.5, 0.25, 0.125, 2.0, -2.0, 1.0, 0.0, 0.0, 0.0]], dtype)

    with pytest.raises(ValueError, match="exactly two scale"):
        load_official_2dgs_source_from_ply(path)


def test_scaled_camera_matrix_scales_intrinsics_to_render_size() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=1000, height=500, params=(800.0, 700.0, 500.0, 250.0))

    matrix = scaled_camera_matrix(camera, 500, 250)

    assert np.allclose(matrix, [[400.0, 0.0, 250.0], [0.0, 350.0, 125.0], [0.0, 0.0, 1.0]])
