"""COLMAP track-observation loading for VFM mapability experiments."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: Tuple[float, ...]


def colmap_camera_focal_lengths(camera: ColmapCamera) -> tuple[float, float]:
    """Return ``(fx, fy)`` according to the COLMAP camera model contract.

    In the single-focal models, ``params[1]`` is a principal-point coordinate,
    not a second focal length.  Keeping this parsing in one place prevents
    projected-scale and splat-footprint code from silently depending on the
    number and ordering of camera parameters.
    """

    model_id = int(camera.model_id)
    params = tuple(float(value) for value in camera.params)
    single_focal_models = {0, 2, 3, 8, 9}
    dual_focal_models = {1, 4, 5, 6, 7, 10}
    if model_id in single_focal_models:
        if len(params) < 1:
            raise ValueError("single-focal COLMAP camera has no focal length")
        focal = float(params[0])
        return focal, focal
    if model_id in dual_focal_models:
        if len(params) < 2:
            raise ValueError("dual-focal COLMAP camera has incomplete focal lengths")
        return float(params[0]), float(params[1])
    raise ValueError(f"unsupported COLMAP camera model id: {camera.model_id}")


@dataclass(frozen=True)
class ColmapImageObservation:
    image_id: int
    image_name: str
    camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
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
    camera_center: np.ndarray | None = None
    viewing_ray: np.ndarray | None = None


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64).reshape(4)
    norm = max(float(np.linalg.norm(q)), 1e-12)
    qw, qx, qy, qz = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw],
            [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw],
            [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


def camera_center_from_qvec_tvec(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation = qvec_to_rotmat(qvec)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    return (-rotation.T @ translation).astype(np.float64)


def viewing_ray_from_camera_center(point_xyz: np.ndarray, camera_center: np.ndarray) -> np.ndarray:
    ray = np.asarray(point_xyz, dtype=np.float64).reshape(3) - np.asarray(camera_center, dtype=np.float64).reshape(3)
    norm = max(float(np.linalg.norm(ray)), 1e-12)
    return (ray / norm).astype(np.float64)


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
            qvec = np.asarray(struct.unpack("<dddd", _read_exact(handle, 8 * 4)), dtype=np.float64)
            tvec = np.asarray(struct.unpack("<ddd", _read_exact(handle, 8 * 3)), dtype=np.float64)
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
                qvec=qvec,
                tvec=tvec,
                xys=xys,
                point3d_ids=point3d_ids,
            )
    return images


def read_colmap_image_camera_ids_binary(path: Path) -> Dict[str, int]:
    """Read only image-name to camera-ID ownership, discarding all pose targets.

    Inference-only evaluators need the camera intrinsics associated with a
    query image but must not retain its COLMAP qvec/tvec.  Keeping this parser
    separate makes that boundary explicit and auditable.
    """

    image_camera_ids: Dict[str, int] = {}
    with Path(path).open("rb") as handle:
        (image_count,) = struct.unpack("<Q", _read_exact(handle, 8))
        for _ in range(image_count):
            _image_id = struct.unpack("<i", _read_exact(handle, 4))[0]
            _read_exact(handle, 8 * 7)  # qvec and tvec are intentionally discarded.
            (camera_id,) = struct.unpack("<i", _read_exact(handle, 4))
            image_name = _read_c_string(handle)
            (point2d_count,) = struct.unpack("<Q", _read_exact(handle, 8))
            handle.seek(int(point2d_count) * 24, 1)
            if image_name in image_camera_ids:
                raise ValueError(f"duplicate COLMAP image name: {image_name}")
            image_camera_ids[str(image_name)] = int(camera_id)
    return image_camera_ids


def scale_colmap_camera(
    camera: ColmapCamera,
    *,
    width: int,
    height: int,
) -> ColmapCamera:
    """Scale a COLMAP camera while preserving its normalized ray geometry."""

    target_width = int(width)
    target_height = int(height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("scaled camera dimensions must be positive")
    scale_x = float(target_width) / float(camera.width)
    scale_y = float(target_height) / float(camera.height)
    params = list(float(value) for value in camera.params)
    if int(camera.model_id) in {0, 2, 3, 8, 9}:
        if not np.isclose(scale_x, scale_y, rtol=1e-7, atol=1e-9):
            raise ValueError(
                "single-focal COLMAP cameras require aspect-preserving scaling"
            )
        params[0] *= scale_x
        params[1] *= scale_x
        params[2] *= scale_y
    elif int(camera.model_id) in {1, 4, 5, 6, 7, 10}:
        params[0] *= scale_x
        params[1] *= scale_y
        params[2] *= scale_x
        params[3] *= scale_y
    else:
        raise ValueError(f"unsupported COLMAP camera model id: {camera.model_id}")
    return ColmapCamera(
        camera_id=int(camera.camera_id),
        model_id=int(camera.model_id),
        width=target_width,
        height=target_height,
        params=tuple(params),
    )


def write_colmap_cameras_binary(
    cameras: Mapping[int, ColmapCamera], path: Path
) -> None:
    """Write the camera subset of a COLMAP binary model."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera_id in sorted(cameras):
            camera = cameras[int(camera_id)]
            expected = _CAMERA_MODEL_PARAM_COUNTS.get(int(camera.model_id))
            if expected is None or len(camera.params) != int(expected):
                raise ValueError("camera parameters do not match its COLMAP model")
            handle.write(
                struct.pack(
                    "<iiQQ",
                    int(camera.camera_id),
                    int(camera.model_id),
                    int(camera.width),
                    int(camera.height),
                )
            )
            handle.write(
                struct.pack(
                    "<" + "d" * len(camera.params),
                    *(float(value) for value in camera.params),
                )
            )


def write_colmap_images_binary(
    images: Mapping[int, ColmapImageObservation],
    path: Path,
    *,
    xy_scale_by_camera_id: Mapping[int, tuple[float, float]] | None = None,
) -> None:
    """Write COLMAP images, optionally scaling every stored 2D observation."""

    scales = {} if xy_scale_by_camera_id is None else xy_scale_by_camera_id
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image_id in sorted(images):
            image = images[int(image_id)]
            scale_x, scale_y = scales.get(int(image.camera_id), (1.0, 1.0))
            xys = np.asarray(image.xys, dtype=np.float64).reshape(-1, 2).copy()
            point3d_ids = np.asarray(image.point3d_ids, dtype=np.int64).reshape(-1)
            if len(xys) != len(point3d_ids):
                raise ValueError("COLMAP image xy and point3D arrays differ")
            xys[:, 0] *= float(scale_x)
            xys[:, 1] *= float(scale_y)
            handle.write(struct.pack("<i", int(image.image_id)))
            handle.write(
                struct.pack("<dddd", *(float(value) for value in image.qvec))
            )
            handle.write(
                struct.pack("<ddd", *(float(value) for value in image.tvec))
            )
            handle.write(struct.pack("<i", int(image.camera_id)))
            handle.write(str(image.image_name).encode("utf8") + b"\x00")
            handle.write(struct.pack("<Q", len(xys)))
            for xy, point3d_id in zip(xys, point3d_ids):
                handle.write(
                    struct.pack(
                        "<ddq",
                        float(xy[0]),
                        float(xy[1]),
                        int(point3d_id),
                    )
                )


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
            camera_center = camera_center_from_qvec_tvec(image.qvec, image.tvec)
            viewing_ray = viewing_ray_from_camera_center(point.xyz, camera_center)
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
                    camera_center=camera_center,
                    viewing_ray=viewing_ray,
                )
            )
    observations.sort(key=lambda item: (item.track_id, item.image_id, item.point2d_idx))
    return observations
