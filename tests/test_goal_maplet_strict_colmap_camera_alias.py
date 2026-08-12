from feature_extract.tools.vfm import build_stage_h2_raw_gaussian_anchor_map as builder
import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera, ColmapImageObservation


def test_flattened_strict_colmap_image_has_cambridge_alias(monkeypatch, tmp_path):
    camera = ColmapCamera(1, 1, 10, 8, (5.0, 5.0, 4.0, 3.0))
    image = ColmapImageObservation(
        image_id=1, qvec=np.asarray([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3), camera_id=1,
        image_name="seq2__frame00001.png",
        xys=np.zeros((0, 2)), point3d_ids=np.zeros((0,), dtype=np.int64),
    )
    monkeypatch.setattr(builder, "read_colmap_cameras_binary", lambda _path: {1: camera})
    monkeypatch.setattr(builder, "read_colmap_images_binary", lambda _path: {1: image})
    result = builder._load_camera_by_image(str(tmp_path))
    assert result["seq2/frame00001.png"] is camera
