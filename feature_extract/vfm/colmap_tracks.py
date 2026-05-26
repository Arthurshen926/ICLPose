"""COLMAP track-observation loading for VFM mapability experiments."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: Tuple[float, ...]


@dataclass(frozen=True)
class ColmapImageObservation:
    image_id: int
    image_name: str
    camera_id: int
    xys: np.ndarray
    point3d_ids: np.ndarray


@dataclass(frozen=True)
class ColmapPoint3D:
    point3d_id: int
    xyz: np.ndarray
    error: float
    track: Tuple[Tuple[int, int], ...]

    @property
    def track_length(self) -> int:
        return len(self.track)


@dataclass(frozen=True)
class ColmapTrackObservation:
    track_id: int
    image_id: str
    point2d_idx: int
    xy: Tuple[float, float]
    xyz: np.ndarray
    track_length: int
    reprojection_error: float
    camera_id: int | None = None
    image_width: int | None = None
    image_height: int | None = None


def _read_exact(handle: BinaryIO, num_bytes: int) -> bytes:
    payload = handle.read(num_bytes)
    if len(payload) != num_bytes:
        raise ValueError("unexpected end of COLMAP binary file")
    return payload


def _read_c_string(handle: BinaryIO) -> str:
    chunks = []
    while True:
        char = _read_exact(handle, 1)
        if char == b"\x00":
            return b"".join(chunks).decode("utf8")
        chunks.append(char)


_CAMERA_MODEL_PARAM_COUNTS = {
    0: 3,  # SIMPLE_PINHOLE
    1: 4,  # PINHOLE
    2: 4,  # SIMPLE_RADIAL
    3: 5,  # RADIAL
    4: 8,  # OPENCV
    5: 8,  # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,  # FOV
    8: 4,  # SIMPLE_RADIAL_FISHEYE
    9: 5,  # RADIAL_FISHEYE
    10: 12,  # THIN_PRISM_FISHEYE
}


def read_colmap_cameras_binary(path: Path) -> Dict[int, ColmapCamera]:
    """Read camera dimensions from a COLMAP `cameras.bin` file."""

    cameras: Dict[int, ColmapCamera] = {}
    with Path(path).open("rb") as handle:
        (camera_count,) = struct.unpack("<Q", _read_exact(handle, 8))
        for _ in range(camera_count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", _read_exact(handle, 24))
            if model_id not in _CAMERA_MODEL_PARAM_COUNTS:
                raise ValueError(f"unsupported COLMAP camera model id: {model_id}")
            param_count = _CAMERA_MODEL_PARAM_COUNTS[model_id]
            params = struct.unpack("<" + "d" * param_count, _read_exact(handle, 8 * param_count))
            cameras[int(camera_id)] = ColmapCamera(
                camera_id=int(camera_id),
                model_id=int(model_id),
                width=int(width),
                height=int(height),
                params=tuple(float(value) for value in params),
            )
    return cameras


def read_colmap_images_binary(path: Path) -> Dict[int, ColmapImageObservation]:
    """Read the image observations from a COLMAP `images.bin` file."""

    images: Dict[int, ColmapImageObservation] = {}
    with Path(path).open("rb") as handle:
        (image_count,) = struct.unpack("<Q", _read_exact(handle, 8))
        for _ in range(image_count):
            (image_id,) = struct.unpack("<i", _read_exact(handle, 4))
            _read_exact(handle, 8 * 4)  # qvec
            _read_exact(handle, 8 * 3)  # tvec
            (camera_id,) = struct.unpack("<i", _read_exact(handle, 4))
            image_name = _read_c_string(handle)
            (point2d_count,) = struct.unpack("<Q", _read_exact(handle, 8))
            xys = np.zeros((point2d_count, 2), dtype=np.float64)
            point3d_ids = np.zeros((point2d_count,), dtype=np.int64)
            for idx in range(point2d_count):
                x, y, point3d_id = struct.unpack("<ddq", _read_exact(handle, 24))
                xys[idx] = (x, y)
                point3d_ids[idx] = point3d_id
            images[int(image_id)] = ColmapImageObservation(
                image_id=int(image_id),
                image_name=image_name,
                camera_id=int(camera_id),
                xys=xys,
                point3d_ids=point3d_ids,
            )
    return images


def read_colmap_points3d_binary(path: Path) -> Dict[int, ColmapPoint3D]:
    """Read 3D points and their image tracks from a COLMAP `points3D.bin` file."""

    points: Dict[int, ColmapPoint3D] = {}
    with Path(path).open("rb") as handle:
        (point_count,) = struct.unpack("<Q", _read_exact(handle, 8))
        for _ in range(point_count):
            point_id, x, y, z, _r, _g, _b, error = struct.unpack(
                "<QdddBBBd", _read_exact(handle, 43)
            )
            (track_length,) = struct.unpack("<Q", _read_exact(handle, 8))
            track = []
            for _track_idx in range(track_length):
                image_id, point2d_idx = struct.unpack("<ii", _read_exact(handle, 8))
                track.append((int(image_id), int(point2d_idx)))
            points[int(point_id)] = ColmapPoint3D(
                point3d_id=int(point_id),
                xyz=np.asarray([x, y, z], dtype=np.float64),
                error=float(error),
                track=tuple(track),
            )
    return points


def load_colmap_track_observations(
    model_dir: Path,
    min_track_length: int = 2,
) -> list[ColmapTrackObservation]:
    """Load per-image 2D observations for valid COLMAP 3D tracks."""

    if min_track_length <= 0:
        raise ValueError("min_track_length must be positive")
    model_dir = Path(model_dir)
    cameras = {}
    if (model_dir / "cameras.bin").exists():
        cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    points = read_colmap_points3d_binary(model_dir / "points3D.bin")

    observations: list[ColmapTrackObservation] = []
    for point in points.values():
        if point.track_length < min_track_length:
            continue
        for image_id, point2d_idx in point.track:
            image = images.get(image_id)
            if image is None:
                continue
            if point2d_idx < 0 or point2d_idx >= image.xys.shape[0]:
                continue
            if int(image.point3d_ids[point2d_idx]) != point.point3d_id:
                continue
            xy = image.xys[point2d_idx]
            camera = cameras.get(image.camera_id)
            observations.append(
                ColmapTrackObservation(
                    track_id=point.point3d_id,
                    image_id=image.image_name,
                    point2d_idx=int(point2d_idx),
                    xy=(float(xy[0]), float(xy[1])),
                    xyz=point.xyz.astype(np.float64, copy=False),
                    track_length=point.track_length,
                    reprojection_error=point.error,
                    camera_id=image.camera_id,
                    image_width=None if camera is None else camera.width,
                    image_height=None if camera is None else camera.height,
                )
            )
    observations.sort(key=lambda item: (item.track_id, item.image_id, item.point2d_idx))
    return observations
