import struct
import subprocess
import sys
import json

import numpy as np

from feature_extract.vfm.colmap_tracks import (
    ColmapCamera,
    ColmapImageObservation,
    load_colmap_track_observations,
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
    read_colmap_images_binary,
    scale_colmap_camera,
    write_colmap_cameras_binary,
    write_colmap_images_binary,
)


def _write_cameras_bin(path):
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, 1, 100, 80))
        handle.write(struct.pack("<dddd", 50.0, 50.0, 50.0, 40.0))


def _write_images_bin(path):
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 2))
        for image_id, name, xy in [
            (1, "seq1/frame0001.png", (10.0, 20.0)),
            (2, "seq1/frame0002.png", (30.0, 40.0)),
        ]:
            handle.write(struct.pack("<i", image_id))
            handle.write(struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0))
            handle.write(struct.pack("<ddd", 0.0, 0.0, 0.0))
            handle.write(struct.pack("<i", 1))
            handle.write(name.encode("utf8") + b"\x00")
            handle.write(struct.pack("<Q", 1))
            handle.write(struct.pack("<ddq", xy[0], xy[1], 7))


def _write_points3d_bin(path):
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<QdddBBBd", 7, 1.0, 2.0, 3.0, 255, 0, 0, 0.25))
        handle.write(struct.pack("<Q", 2))
        handle.write(struct.pack("<ii", 1, 0))
        handle.write(struct.pack("<ii", 2, 0))


def test_load_colmap_track_observations_from_binary_model(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_cameras_bin(model_dir / "cameras.bin")
    _write_images_bin(model_dir / "images.bin")
    _write_points3d_bin(model_dir / "points3D.bin")

    observations = load_colmap_track_observations(model_dir, min_track_length=2)

    assert len(observations) == 2
    assert observations[0].track_id == 7
    assert observations[0].image_id == "seq1/frame0001.png"
    assert observations[0].point2d_idx == 0
    assert observations[0].xy == (10.0, 20.0)
    assert observations[0].camera_id == 1
    assert observations[0].image_width == 100
    assert observations[0].image_height == 80
    assert observations[0].track_length == 2
    np.testing.assert_allclose(observations[0].xyz, np.asarray([1.0, 2.0, 3.0]))
    assert observations[1].image_id == "seq1/frame0002.png"


def test_load_colmap_track_observations_filters_short_tracks(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_cameras_bin(model_dir / "cameras.bin")
    _write_images_bin(model_dir / "images.bin")
    _write_points3d_bin(model_dir / "points3D.bin")

    assert load_colmap_track_observations(model_dir, min_track_length=3) == []


def test_read_colmap_image_camera_ids_discards_pose_payload(tmp_path):
    image_path = tmp_path / "images.bin"
    _write_images_bin(image_path)

    image_camera_ids = read_colmap_image_camera_ids_binary(image_path)

    assert image_camera_ids == {
        "seq1/frame0001.png": 1,
        "seq1/frame0002.png": 1,
    }


def test_export_colmap_track_observations_cli_writes_jsonl_and_summary(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_cameras_bin(model_dir / "cameras.bin")
    _write_images_bin(model_dir / "images.bin")
    _write_points3d_bin(model_dir / "points3D.bin")
    output_jsonl = tmp_path / "tracks.jsonl"
    summary_json = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.export_colmap_track_observations",
            "--model_dir",
            str(model_dir),
            "--min_track_length",
            "2",
            "--output_jsonl",
            str(output_jsonl),
            "--summary_json",
            str(summary_json),
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    summary = json.loads(summary_json.read_text())

    assert len(rows) == 2
    assert rows[0]["track_id"] == 7
    assert rows[0]["image_id"] == "seq1/frame0001.png"
    assert rows[0]["image_width"] == 100
    assert rows[0]["image_height"] == 80
    assert summary["observation_count"] == 2
    assert summary["available_observation_count"] == 2
    assert summary["track_count"] == 1


def test_scaled_colmap_binary_roundtrip_preserves_pose_and_scales_pixels(tmp_path):
    camera = ColmapCamera(
        camera_id=4,
        model_id=2,
        width=1920,
        height=1080,
        params=(1669.05, 960.0, 540.0, 0.03),
    )
    scaled = scale_colmap_camera(camera, width=1024, height=576)
    image = ColmapImageObservation(
        image_id=7,
        image_name="query.png",
        camera_id=4,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0]),
        tvec=np.asarray([1.0, 2.0, 3.0]),
        xys=np.asarray([[960.0, 540.0], [480.0, 270.0]]),
        point3d_ids=np.asarray([11, -1], dtype=np.int64),
    )
    camera_path = tmp_path / "cameras.bin"
    image_path = tmp_path / "images.bin"
    write_colmap_cameras_binary({4: scaled}, camera_path)
    write_colmap_images_binary(
        {7: image},
        image_path,
        xy_scale_by_camera_id={4: (1024.0 / 1920.0, 576.0 / 1080.0)},
    )

    loaded_camera = read_colmap_cameras_binary(camera_path)[4]
    loaded_image = read_colmap_images_binary(image_path)[7]
    assert (loaded_camera.width, loaded_camera.height) == (1024, 576)
    np.testing.assert_allclose(
        loaded_camera.params,
        [890.16, 512.0, 288.0, 0.03],
    )
    np.testing.assert_allclose(loaded_image.xys, [[512.0, 288.0], [256.0, 144.0]])
    np.testing.assert_allclose(loaded_image.qvec, image.qvec)
    np.testing.assert_allclose(loaded_image.tvec, image.tvec)
    assert loaded_image.point3d_ids.tolist() == [11, -1]
