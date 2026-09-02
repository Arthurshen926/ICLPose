"""Coordinate-correct RADIO-final fields on explicit surface-chart atlases.

The module deliberately separates two identities:

* :class:`SourceViewChartRadioField` is an offline, view-bound observation
  artifact.  It keeps every RADIO token in its original 2-D layout and the
  exact raw-SIMPLE_RADIAL -> ideal-chart coordinate transform.
* :class:`CanonicalChartRadioField` is a deployment-oriented, physical
  surface-family artifact.  It contains view-balanced fused codes and
  anonymous view prototypes, but no source-view names or token inventory.

No query image, query pose, pose ground truth, local feature or Gaussian is
used by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from .explicit_chart_atlas import ExplicitChartAtlas
from .lineage import arrays_sha256, canonical_json_sha256
from .retrieval_surface_metrics import inverse_simple_radial
from .chart_surface_families import CanonicalSurfaceFamilyCarrier


SOURCE_SCHEMA = "goal_maplet_source_view_chart_radio_uv_field_v2"
FAMILY_SCHEMA = "goal_maplet_canonical_surface_family_layout_v1"
CANONICAL_SCHEMA = "goal_maplet_canonical_chart_radio_field_v1"
COORDINATE_CONTRACT = (
    "raw_simple_radial_radio_token_endpoints_inverse_warped_by_normalized_"
    "camera_rays_to_lowres_ideal_chart_uv_align_corners_true_v2"
)


def chart_name_to_image_id(name: str) -> str:
    """Convert a flattened chart filename without guessing its route."""

    value = str(name)
    fields = value.split("__", 1)
    if (
        len(fields) != 2
        or not fields[0]
        or not fields[1]
        or "/" in fields[0]
        or "\\" in fields[0]
        or "/" in fields[1]
        or "\\" in fields[1]
    ):
        raise ValueError(f"chart name lacks a strict flattened image ID: {value}")
    return f"{fields[0]}/{fields[1]}"


def _artifact_content_sha256(metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> str:
    basis = {
        str(key): value
        for key, value in dict(metadata).items()
        if str(key) not in {"arrays_sha256", "content_sha256"}
    }
    basis["arrays_sha256"] = arrays_sha256(arrays)
    return canonical_json_sha256(basis)


def _sealed_metadata(metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    result = {
        str(key): value
        for key, value in dict(metadata).items()
        if str(key) not in {"arrays_sha256", "content_sha256"}
    }
    result["arrays_sha256"] = arrays_sha256(arrays)
    result["content_sha256"] = canonical_json_sha256(result)
    return result


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> dict[str, object]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sealed = _sealed_metadata(metadata, arrays)
    temporary = output.with_name(output.stem + ".temporary.npz")
    np.savez_compressed(
        temporary,
        **{name: np.asarray(value) for name, value in arrays.items()},
        metadata_json=np.asarray(json.dumps(sealed, sort_keys=True)),
    )
    temporary.replace(output)
    return sealed


def _load_npz_arrays(path: Path, names: tuple[str, ...]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in names}
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if arrays_sha256(arrays) != str(metadata.get("arrays_sha256", "")):
        raise ValueError("artifact arrays differ from their lineage hash")
    if _artifact_content_sha256(metadata, arrays) != str(metadata.get("content_sha256", "")):
        raise ValueError("artifact metadata differs from its content hash")
    return arrays, metadata


@dataclass(frozen=True)
class RawSimpleRadialCamera:
    width: int
    height: int
    focal: float
    cx: float
    cy: float
    k1: float

    def validated(self) -> "RawSimpleRadialCamera":
        values = np.asarray([self.focal, self.cx, self.cy, self.k1], dtype=np.float64)
        if int(self.width) <= 1 or int(self.height) <= 1 or float(self.focal) <= 0.0:
            raise ValueError("raw camera canvas/focal is invalid")
        if np.any(~np.isfinite(values)):
            raise ValueError("raw camera parameters must be finite")
        # SIMPLE_RADIAL must remain monotone on the complete source canvas.
        corners = np.asarray(
            [[0.0, 0.0], [self.width - 1.0, 0.0], [0.0, self.height - 1.0],
             [self.width - 1.0, self.height - 1.0]],
            dtype=np.float64,
        )
        radius2 = np.sum(
            np.square((corners - np.asarray([self.cx, self.cy])) / float(self.focal)),
            axis=1,
        )
        if np.min(1.0 + 3.0 * float(self.k1) * radius2) <= 1e-8:
            raise ValueError("SIMPLE_RADIAL is non-monotone on the camera canvas")
        return self

    @classmethod
    def from_manifest_row(cls, row: Mapping[str, object]) -> "RawSimpleRadialCamera":
        model = int(row["model_id"])
        params = np.asarray(row["params"], dtype=np.float64).reshape(-1)
        if model == 2 and params.size == 4:
            focal, cx, cy, k1 = params.tolist()
        elif model == 0 and params.size == 3:
            focal, cx, cy = params.tolist()
            k1 = 0.0
        else:
            raise ValueError("chart RADIO attachment requires SIMPLE_PINHOLE/SIMPLE_RADIAL")
        return cls(
            width=int(row["width"]), height=int(row["height"]),
            focal=float(focal), cx=float(cx), cy=float(cy), k1=float(k1),
        ).validated()

    def raw_xy_to_ideal_uv(self, raw_xy: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_xy, dtype=np.float64)
        if raw.ndim < 1 or raw.shape[-1] != 2 or np.any(~np.isfinite(raw)):
            raise ValueError("raw coordinates must be finite (...,2)")
        distorted = (raw - np.asarray([self.cx, self.cy])) / float(self.focal)
        ideal = inverse_simple_radial(distorted, float(self.k1))
        ideal_xy = ideal * float(self.focal) + np.asarray([self.cx, self.cy])
        return ideal_xy / np.asarray([self.width - 1.0, self.height - 1.0])

    def ideal_uv_to_raw_xy(self, ideal_uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(ideal_uv, dtype=np.float64)
        if uv.ndim < 1 or uv.shape[-1] != 2 or np.any(~np.isfinite(uv)):
            raise ValueError("ideal chart UV must be finite (...,2)")
        ideal_xy = uv * np.asarray([self.width - 1.0, self.height - 1.0])
        undistorted = (ideal_xy - np.asarray([self.cx, self.cy])) / float(self.focal)
        radius2 = np.sum(np.square(undistorted), axis=-1, keepdims=True)
        distorted = undistorted * (1.0 + float(self.k1) * radius2)
        return distorted * float(self.focal) + np.asarray([self.cx, self.cy])


@dataclass(frozen=True)
class IdealChartCamera:
    """Ideal-pinhole camera defined on the actual low-resolution chart grid."""

    width: int
    height: int
    focal: float
    cx: float
    cy: float

    def validated(self) -> "IdealChartCamera":
        values = np.asarray([self.focal, self.cx, self.cy], dtype=np.float64)
        if int(self.width) <= 1 or int(self.height) <= 1 or float(self.focal) <= 0.0:
            raise ValueError("ideal chart camera canvas/focal is invalid")
        if np.any(~np.isfinite(values)):
            raise ValueError("ideal chart camera parameters must be finite")
        return self

    def raw_xy_to_chart_uv(
        self, raw_camera: RawSimpleRadialCamera, raw_xy: np.ndarray,
    ) -> np.ndarray:
        raw = np.asarray(raw_xy, dtype=np.float64)
        distorted = (raw - np.asarray([raw_camera.cx, raw_camera.cy])) / float(raw_camera.focal)
        ray_xy = inverse_simple_radial(distorted, float(raw_camera.k1))
        chart_xy = ray_xy * float(self.focal) + np.asarray([self.cx, self.cy])
        return chart_xy / np.asarray([self.width - 1.0, self.height - 1.0])

    def chart_uv_to_raw_xy(
        self, raw_camera: RawSimpleRadialCamera, chart_uv: np.ndarray,
    ) -> np.ndarray:
        uv = np.asarray(chart_uv, dtype=np.float64)
        if uv.ndim < 1 or uv.shape[-1] != 2 or np.any(~np.isfinite(uv)):
            raise ValueError("chart UV must be finite (...,2)")
        chart_xy = uv * np.asarray([self.width - 1.0, self.height - 1.0])
        ray_xy = (chart_xy - np.asarray([self.cx, self.cy])) / float(self.focal)
        radius2 = np.sum(np.square(ray_xy), axis=-1, keepdims=True)
        distorted = ray_xy * (1.0 + float(raw_camera.k1) * radius2)
        return distorted * float(raw_camera.focal) + np.asarray([raw_camera.cx, raw_camera.cy])

    def chart_uv_to_raw_ideal_xy(
        self, raw_camera: RawSimpleRadialCamera, chart_uv: np.ndarray,
    ) -> np.ndarray:
        uv = np.asarray(chart_uv, dtype=np.float64)
        chart_xy = uv * np.asarray([self.width - 1.0, self.height - 1.0])
        ray_xy = (chart_xy - np.asarray([self.cx, self.cy])) / float(self.focal)
        return ray_xy * float(raw_camera.focal) + np.asarray([raw_camera.cx, raw_camera.cy])


_SOURCE_ARRAY_NAMES = (
    "view_names", "view_token_offsets", "view_token_shapes", "token_xy",
    "token_chart_uv", "token_chart_uv_valid", "token_codes",
    "chart_vertex_offsets", "vertex_chart_uv", "vertex_raw_xy", "vertex_token_rows",
    "vertex_token_weights", "vertex_coordinate_valid",
    "vertex_geometry_confidence", "vertex_view_cosine",
    "vertex_observation_weight", "camera_centers_world", "camera_poses_c2w",
    "raw_camera_parameters", "chart_camera_parameters",
)


@dataclass(frozen=True)
class SourceViewChartRadioField:
    """Complete view-bound RADIO inventory and its chart-UV attachment."""

    view_names: np.ndarray
    view_token_offsets: np.ndarray
    view_token_shapes: np.ndarray
    token_xy: np.ndarray
    token_chart_uv: np.ndarray
    token_chart_uv_valid: np.ndarray
    token_codes: np.ndarray
    chart_vertex_offsets: np.ndarray
    vertex_chart_uv: np.ndarray
    vertex_raw_xy: np.ndarray
    vertex_token_rows: np.ndarray
    vertex_token_weights: np.ndarray
    vertex_coordinate_valid: np.ndarray
    vertex_geometry_confidence: np.ndarray
    vertex_view_cosine: np.ndarray
    vertex_observation_weight: np.ndarray
    camera_centers_world: np.ndarray
    camera_poses_c2w: np.ndarray
    raw_camera_parameters: np.ndarray
    chart_camera_parameters: np.ndarray
    atlas_content_sha256: str
    metadata: Mapping[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in _SOURCE_ARRAY_NAMES}

    @property
    def content_sha256(self) -> str:
        return _artifact_content_sha256(self.metadata, self.arrays())

    @property
    def feature_dim(self) -> int:
        return int(np.asarray(self.token_codes).shape[1])

    def validated(self) -> "SourceViewChartRadioField":
        a = self.arrays()
        view_count = int(a["view_names"].size)
        token_count = int(a["token_codes"].shape[0])
        vertex_count = int(a["vertex_raw_xy"].shape[0])
        if self.metadata.get("artifact_type") != SOURCE_SCHEMA:
            raise ValueError("wrong source-view chart RADIO schema")
        if self.metadata.get("coordinate_contract") != COORDINATE_CONTRACT:
            raise ValueError("source-view field has an unknown coordinate contract")
        if bool(self.metadata.get("uses_query_or_ground_truth", True)):
            raise ValueError("source-view chart RADIO field used query/ground truth")
        if a["view_token_offsets"].shape != (view_count + 1,) or a["view_token_shapes"].shape != (view_count, 2):
            raise ValueError("invalid source-view token offsets/shapes")
        if a["chart_vertex_offsets"].shape != (view_count + 1,):
            raise ValueError("source views and charts are not one-to-one")
        if (
            int(a["view_token_offsets"][0]) != 0
            or int(a["view_token_offsets"][-1]) != token_count
            or int(a["chart_vertex_offsets"][0]) != 0
            or int(a["chart_vertex_offsets"][-1]) != vertex_count
            or np.any(np.diff(a["view_token_offsets"]) <= 0)
            or np.any(np.diff(a["chart_vertex_offsets"]) <= 0)
        ):
            raise ValueError("source-view offsets are invalid/empty")
        expected_counts = np.prod(a["view_token_shapes"].astype(np.int64), axis=1)
        if not np.array_equal(np.diff(a["view_token_offsets"]), expected_counts):
            raise ValueError("a source view dropped or duplicated RADIO tokens")
        if a["token_codes"].ndim != 2 or int(a["token_codes"].shape[1]) <= 0:
            raise ValueError("RADIO token inventory must have shape (N,D)")
        if a["token_xy"].shape != (token_count, 2) or a["token_chart_uv"].shape != (token_count, 2):
            raise ValueError("token coordinate arrays differ from complete inventory")
        if a["token_chart_uv_valid"].shape != (token_count,):
            raise ValueError("invalid token chart-UV validity")
        for view in range(view_count):
            lo, hi = map(int, a["view_token_offsets"][view:view + 2])
            height, width = map(int, a["view_token_shapes"][view])
            yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            expected = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
            if not np.array_equal(a["token_xy"][lo:hi], expected):
                raise ValueError("RADIO token layout is not complete y-major (x,y)")
        vertex_shapes = {
            "vertex_chart_uv": (vertex_count, 2),
            "vertex_raw_xy": (vertex_count, 2),
            "vertex_token_rows": (vertex_count, 4),
            "vertex_token_weights": (vertex_count, 4),
            "vertex_coordinate_valid": (vertex_count,),
            "vertex_geometry_confidence": (vertex_count,),
            "vertex_view_cosine": (vertex_count,),
            "vertex_observation_weight": (vertex_count,),
        }
        for name, shape in vertex_shapes.items():
            if a[name].shape != shape:
                raise ValueError(f"invalid {name} shape")
        valid = a["vertex_coordinate_valid"].astype(bool)
        rows = a["vertex_token_rows"].astype(np.int64)
        weights = a["vertex_token_weights"].astype(np.float64)
        if np.any(rows[valid] < 0) or np.any(rows[valid] >= token_count):
            raise ValueError("valid chart vertex samples outside RADIO inventory")
        if np.any(rows[~valid] != -1) or np.any(weights[~valid] != 0.0):
            raise ValueError("invalid chart vertex retains RADIO evidence")
        if not np.allclose(np.sum(weights[valid], axis=1), 1.0, atol=2e-6):
            raise ValueError("valid bilinear RADIO weights do not conserve mass")
        if a["camera_centers_world"].shape != (view_count, 3):
            raise ValueError("source-view camera centers differ")
        if a["camera_poses_c2w"].shape != (view_count, 4, 4):
            raise ValueError("source-view mapping camera poses differ")
        if not np.allclose(a["camera_centers_world"], a["camera_poses_c2w"][:, :3, 3], atol=1e-10):
            raise ValueError("source-view camera centers and poses differ")
        if a["raw_camera_parameters"].shape != (view_count, 6):
            raise ValueError("source-view raw camera parameters differ")
        if a["chart_camera_parameters"].shape != (view_count, 5):
            raise ValueError("source-view low-resolution chart cameras differ")
        # The complete token->chart transform must replay without consulting
        # any path or mutable external camera object.
        for view in range(view_count):
            width, height, focal, cx, cy, k1 = a["raw_camera_parameters"][view]
            camera = RawSimpleRadialCamera(
                int(width), int(height), float(focal), float(cx), float(cy), float(k1),
            ).validated()
            chart_width, chart_height, chart_focal, chart_cx, chart_cy = a[
                "chart_camera_parameters"
            ][view]
            chart_camera = IdealChartCamera(
                int(chart_width), int(chart_height), float(chart_focal),
                float(chart_cx), float(chart_cy),
            ).validated()
            token_height, token_width = map(int, a["view_token_shapes"][view])
            lo, hi = map(int, a["view_token_offsets"][view:view + 2])
            replay = chart_camera.raw_xy_to_chart_uv(
                camera, _token_grid_raw_xy(camera, token_height, token_width)
            ).reshape(-1, 2)
            if not np.allclose(replay, a["token_chart_uv"][lo:hi], atol=2e-7, rtol=0.0):
                raise ValueError("stored token->chart coordinate transform does not replay")
            replay_token_valid = (
                (replay[:, 0] >= 0.0) & (replay[:, 0] <= 1.0)
                & (replay[:, 1] >= 0.0) & (replay[:, 1] <= 1.0)
            )
            if not np.array_equal(
                replay_token_valid, a["token_chart_uv_valid"][lo:hi].astype(bool),
            ):
                raise ValueError("stored token chart-UV validity does not replay")
            vlo, vhi = map(int, a["chart_vertex_offsets"][view:view + 2])
            replay_raw = chart_camera.chart_uv_to_raw_xy(
                camera, a["vertex_chart_uv"][vlo:vhi]
            )
            if not np.allclose(replay_raw, a["vertex_raw_xy"][vlo:vhi], atol=2e-5, rtol=0.0):
                raise ValueError("stored chart-vertex->raw coordinate transform does not replay")
            replay_vertex_valid = (
                np.isfinite(replay_raw).all(axis=1)
                & (replay_raw[:, 0] >= 0.0) & (replay_raw[:, 0] <= camera.width - 1.0)
                & (replay_raw[:, 1] >= 0.0) & (replay_raw[:, 1] <= camera.height - 1.0)
            )
            if not np.array_equal(
                replay_vertex_valid, a["vertex_coordinate_valid"][vlo:vhi].astype(bool),
            ):
                raise ValueError("stored chart-vertex coordinate validity does not replay")
        finite_names = [name for name in a if name not in {"view_names"}]
        if any(np.issubdtype(a[name].dtype, np.number) and np.any(~np.isfinite(a[name])) for name in finite_names):
            raise ValueError("source-view chart RADIO artifact contains nonfinite values")
        if np.any(a["vertex_observation_weight"] < 0.0) or np.any(a["vertex_observation_weight"] > 1.0 + 1e-6):
            raise ValueError("source observation weights must be probabilities")
        if str(self.metadata.get("atlas_content_sha256", "")) != str(self.atlas_content_sha256):
            raise ValueError("source field atlas lineage differs")
        declared = str(self.metadata.get("content_sha256", ""))
        if declared and declared != self.content_sha256:
            raise ValueError("source-view field content hash differs")
        return self

    def sampled_vertex_codes(self, *, block_size: int = 1024) -> np.ndarray:
        """Bilinearly sample and L2-normalize RADIO codes at all chart vertices."""

        self.validated()
        count = int(self.vertex_token_rows.shape[0])
        output = np.zeros((count, self.feature_dim), dtype=np.float32)
        for begin in range(0, count, int(block_size)):
            end = min(begin + int(block_size), count)
            rows = np.asarray(self.vertex_token_rows[begin:end], dtype=np.int64)
            safe = np.maximum(rows, 0)
            weight = np.asarray(self.vertex_token_weights[begin:end], dtype=np.float32)
            value = np.sum(self.token_codes[safe] * weight[..., None], axis=1)
            norm = np.linalg.norm(value, axis=1, keepdims=True)
            value = value / np.maximum(norm, 1e-8)
            value[~np.asarray(self.vertex_coordinate_valid[begin:end], dtype=bool)] = 0.0
            output[begin:end] = value
        return output

    def save_npz(self, path: Path) -> dict[str, object]:
        self.validated()
        return _save_npz(path, self.arrays(), self.metadata)

    @classmethod
    def load_npz(cls, path: Path) -> "SourceViewChartRadioField":
        arrays, metadata = _load_npz_arrays(path, _SOURCE_ARRAY_NAMES)
        return cls(
            atlas_content_sha256=str(metadata["atlas_content_sha256"]),
            metadata=metadata, **arrays,
        ).validated()


def _token_grid_raw_xy(camera: RawSimpleRadialCamera, height: int, width: int) -> np.ndarray:
    if int(height) <= 1 or int(width) <= 1:
        raise ValueError("RADIO token grids must have at least 2x2 endpoints")
    y = np.arange(int(height), dtype=np.float64) * (camera.height - 1.0) / (int(height) - 1.0)
    x = np.arange(int(width), dtype=np.float64) * (camera.width - 1.0) / (int(width) - 1.0)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.stack([xx, yy], axis=-1)


def _vertex_bilinear_attachment(
    uv: np.ndarray,
    camera: RawSimpleRadialCamera,
    chart_camera: IdealChartCamera,
    *,
    token_height: int,
    token_width: int,
    token_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = chart_camera.chart_uv_to_raw_xy(camera, uv)
    pos = raw / np.asarray([camera.width - 1.0, camera.height - 1.0])
    pos *= np.asarray([token_width - 1.0, token_height - 1.0])
    valid = (
        np.isfinite(pos).all(axis=1)
        & (pos[:, 0] >= 0.0) & (pos[:, 0] <= token_width - 1.0)
        & (pos[:, 1] >= 0.0) & (pos[:, 1] <= token_height - 1.0)
    )
    safe = np.clip(pos, [0.0, 0.0], [token_width - 1.0, token_height - 1.0])
    x0 = np.floor(safe[:, 0]).astype(np.int64)
    y0 = np.floor(safe[:, 1]).astype(np.int64)
    x1 = np.minimum(x0 + 1, token_width - 1)
    y1 = np.minimum(y0 + 1, token_height - 1)
    wx = safe[:, 0] - x0
    wy = safe[:, 1] - y0
    rows = np.stack(
        [y0 * token_width + x0, y0 * token_width + x1,
         y1 * token_width + x0, y1 * token_width + x1],
        axis=1,
    ) + int(token_offset)
    weights = np.stack(
        [(1.0 - wx) * (1.0 - wy), wx * (1.0 - wy),
         (1.0 - wx) * wy, wx * wy],
        axis=1,
    )
    rows[~valid] = -1
    weights[~valid] = 0.0
    return raw, rows.astype(np.int64), weights.astype(np.float32), valid


def attach_radio_to_chart_atlas(
    atlas: ExplicitChartAtlas,
    *,
    radio_by_view: Mapping[str, np.ndarray],
    raw_camera_by_view: Mapping[str, RawSimpleRadialCamera],
    chart_camera_by_view: Mapping[str, IdealChartCamera],
    camera_pose_c2w_by_view: Mapping[str, np.ndarray],
    atlas_content_sha256: str,
    lineage_metadata: Mapping[str, object],
) -> SourceViewChartRadioField:
    """Attach full raw-image RADIO layouts to source charts.

    ``radio_by_view`` values must be channel-first ``(D,H,W)`` tensors.  The
    tensors are preserved exactly; no saliency selection, pooling or PCA is
    allowed at this layer.
    """

    atlas = atlas.validated()
    names = np.asarray(atlas.chart_names).astype(str)
    expected = set(names.tolist())
    if (
        set(radio_by_view) != expected
        or set(raw_camera_by_view) != expected
        or set(chart_camera_by_view) != expected
        or set(camera_pose_c2w_by_view) != expected
    ):
        raise ValueError("source chart, RADIO, raw/chart-camera and pose inventories differ")
    token_offsets = [0]
    token_shapes, token_xy, token_uv, token_uv_valid, token_codes = [], [], [], [], []
    raw_all, rows_all, weights_all, coordinate_valid_all = [], [], [], []
    vertex_warp_displacement_px = []
    geometry_confidence_all, view_cosine_all, observation_weight_all = [], [], []
    centers, poses, raw_camera_parameters, chart_camera_parameters = [], [], [], []
    feature_dim: int | None = None
    for view, name in enumerate(names.tolist()):
        radio = np.asarray(radio_by_view[name])
        if radio.ndim != 3 or np.any(~np.isfinite(radio)):
            raise ValueError("RADIO-final tensor must be finite channel-first (D,H,W)")
        channels, token_height, token_width = map(int, radio.shape)
        if feature_dim is None:
            feature_dim = channels
        if channels != feature_dim:
            raise ValueError("RADIO feature dimensions differ across source views")
        camera = raw_camera_by_view[name].validated()
        chart_camera = chart_camera_by_view[name].validated()
        raw_camera_parameters.append(
            [camera.width, camera.height, camera.focal, camera.cx, camera.cy, camera.k1]
        )
        chart_camera_parameters.append(
            [chart_camera.width, chart_camera.height, chart_camera.focal,
             chart_camera.cx, chart_camera.cy]
        )
        pose = np.asarray(camera_pose_c2w_by_view[name], dtype=np.float64)
        if pose.shape != (4, 4) or np.any(~np.isfinite(pose)) or not np.allclose(
            pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10,
        ):
            raise ValueError("mapping camera c2w must be a finite homogeneous pose")
        center = pose[:3, 3]
        centers.append(center)
        poses.append(pose)
        grid_raw = _token_grid_raw_xy(camera, token_height, token_width)
        grid_uv = chart_camera.raw_xy_to_chart_uv(camera, grid_raw)
        grid_valid = (
            (grid_uv[..., 0] >= 0.0) & (grid_uv[..., 0] <= 1.0)
            & (grid_uv[..., 1] >= 0.0) & (grid_uv[..., 1] <= 1.0)
        )
        yy, xx = np.meshgrid(np.arange(token_height), np.arange(token_width), indexing="ij")
        token_xy.append(np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.int16))
        token_uv.append(grid_uv.reshape(-1, 2).astype(np.float32))
        token_uv_valid.append(grid_valid.reshape(-1))
        token_codes.append(np.transpose(radio, (1, 2, 0)).reshape(-1, channels).astype(np.float32, copy=False))
        token_shapes.append((token_height, token_width))
        token_offsets.append(token_offsets[-1] + token_height * token_width)

        vlo, vhi = map(int, atlas.chart_vertex_offsets[view:view + 2])
        raw, rows, weights, coordinate_valid = _vertex_bilinear_attachment(
            np.asarray(atlas.uv[vlo:vhi], dtype=np.float64), camera,
            chart_camera,
            token_height=token_height, token_width=token_width,
            token_offset=token_offsets[-2],
        )
        naive_raw = chart_camera.chart_uv_to_raw_ideal_xy(
            camera, np.asarray(atlas.uv[vlo:vhi], dtype=np.float64)
        )
        vertex_warp_displacement_px.append(np.linalg.norm(raw - naive_raw, axis=1))
        vertex = np.asarray(atlas.vertices_world[vlo:vhi], dtype=np.float64)
        normal = np.asarray(atlas.normals_world[vlo:vhi], dtype=np.float64)
        direction = center[None, :] - vertex
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        normal_norm = np.linalg.norm(normal, axis=1)
        normal_unit = normal / np.maximum(normal_norm[:, None], 1e-12)
        cosine = np.clip(np.sum(normal_unit * direction, axis=1), 0.0, 1.0)
        cosine[normal_norm < 0.5] = 0.0
        confidence = np.maximum(np.asarray(atlas.confidence[vlo:vhi], dtype=np.float64), 0.0)
        positive = confidence[confidence > 0.0]
        scale = float(np.quantile(positive, 0.9)) if positive.size else 1.0
        geometry_quality = np.clip(confidence / max(scale, 1e-12), 0.0, 1.0)
        observation = coordinate_valid.astype(np.float64) * geometry_quality * cosine
        # Coordinate lineage is tiny relative to RADIO and remains float64 so
        # the forward warp replays at sub-pixel numerical precision.
        raw_all.append(raw.astype(np.float64))
        rows_all.append(rows)
        weights_all.append(weights)
        coordinate_valid_all.append(coordinate_valid)
        geometry_confidence_all.append(geometry_quality.astype(np.float32))
        view_cosine_all.append(cosine.astype(np.float32))
        observation_weight_all.append(observation.astype(np.float32))
    arrays = {
        "view_names": names,
        "view_token_offsets": np.asarray(token_offsets, dtype=np.int64),
        "view_token_shapes": np.asarray(token_shapes, dtype=np.int16),
        "token_xy": np.concatenate(token_xy),
        "token_chart_uv": np.concatenate(token_uv),
        "token_chart_uv_valid": np.concatenate(token_uv_valid),
        "token_codes": np.concatenate(token_codes),
        "chart_vertex_offsets": np.asarray(atlas.chart_vertex_offsets, dtype=np.int64),
        "vertex_chart_uv": np.asarray(atlas.uv, dtype=np.float64).copy(),
        "vertex_raw_xy": np.concatenate(raw_all),
        "vertex_token_rows": np.concatenate(rows_all),
        "vertex_token_weights": np.concatenate(weights_all),
        "vertex_coordinate_valid": np.concatenate(coordinate_valid_all),
        "vertex_geometry_confidence": np.concatenate(geometry_confidence_all),
        "vertex_view_cosine": np.concatenate(view_cosine_all),
        "vertex_observation_weight": np.concatenate(observation_weight_all),
        "camera_centers_world": np.asarray(centers, dtype=np.float64),
        "camera_poses_c2w": np.asarray(poses, dtype=np.float64),
        "raw_camera_parameters": np.asarray(raw_camera_parameters, dtype=np.float64),
        "chart_camera_parameters": np.asarray(chart_camera_parameters, dtype=np.float64),
    }
    displacement = np.concatenate(vertex_warp_displacement_px)
    metadata = {
        "artifact_type": SOURCE_SCHEMA,
        "representation": "complete_source_view_radio_final_layout_attached_to_explicit_chart_uv",
        "coordinate_contract": COORDINATE_CONTRACT,
        "token_layout": "complete_y_major_no_selection_no_pooling_no_pca",
        "sampling": "bilinear_raw_radio_endpoint_grid_align_corners_true",
        "visibility": "mapping_view_front_facing_cosine_times_per_chart_robust_geometry_confidence",
        "source_identity_role": "offline_observation_only_not_canonical_surface_identity",
        "vfm_layer": "radio_final",
        "feature_dimension": int(arrays["token_codes"].shape[1]),
        "feature_storage_dtype": str(arrays["token_codes"].dtype),
        "complete_token_count": int(arrays["token_codes"].shape[0]),
        "complete_token_feature_uncompressed_bytes": int(arrays["token_codes"].nbytes),
        "feature_projection": "none_source_authority_preserves_raw_radio_final",
        "uses_mapping_camera_pose": True,
        "uses_query_or_ground_truth": False,
        "uses_gaussian_or_2dgs": False,
        "atlas_content_sha256": str(atlas_content_sha256),
        "mean_vertex_ideal_to_raw_warp_displacement_px": float(np.mean(displacement)),
        "maximum_vertex_ideal_to_raw_warp_displacement_px": float(np.max(displacement, initial=0.0)),
        **dict(lineage_metadata),
    }
    return SourceViewChartRadioField(
        atlas_content_sha256=str(atlas_content_sha256), metadata=metadata, **arrays,
    ).validated()


_FAMILY_ARRAY_NAMES = (
    "family_keys", "family_node_offsets", "node_points_world", "node_normals_world",
    "vertex_family_rows", "vertex_node_rows", "vertex_assignment_weight",
)


@dataclass(frozen=True)
class CanonicalSurfaceFamilyLayout:
    """Explicit source-vertex -> canonical-family/node assignment."""

    family_keys: np.ndarray
    family_node_offsets: np.ndarray
    node_points_world: np.ndarray
    node_normals_world: np.ndarray
    vertex_family_rows: np.ndarray
    vertex_node_rows: np.ndarray
    vertex_assignment_weight: np.ndarray
    atlas_content_sha256: str
    metadata: Mapping[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in _FAMILY_ARRAY_NAMES}

    @property
    def content_sha256(self) -> str:
        return _artifact_content_sha256(self.metadata, self.arrays())

    def validated(self) -> "CanonicalSurfaceFamilyLayout":
        a = self.arrays()
        family_count = int(a["family_keys"].size)
        node_count = int(a["node_points_world"].shape[0])
        vertex_count = int(a["vertex_node_rows"].size)
        if self.metadata.get("artifact_type") != FAMILY_SCHEMA:
            raise ValueError("wrong canonical surface-family schema")
        if a["family_node_offsets"].shape != (family_count + 1,):
            raise ValueError("invalid family-node offsets")
        if (
            int(a["family_node_offsets"][0]) != 0
            or int(a["family_node_offsets"][-1]) != node_count
            or np.any(np.diff(a["family_node_offsets"]) <= 0)
        ):
            raise ValueError("canonical surface families must contain nodes")
        if a["node_points_world"].shape != (node_count, 3) or a["node_normals_world"].shape != (node_count, 3):
            raise ValueError("invalid canonical node geometry")
        if a["vertex_family_rows"].shape != (vertex_count,) or a["vertex_assignment_weight"].shape != (vertex_count,):
            raise ValueError("invalid source-vertex family assignment")
        assigned = a["vertex_node_rows"] >= 0
        if np.any(a["vertex_node_rows"][assigned] >= node_count):
            raise ValueError("source vertex assigned outside canonical nodes")
        if np.any(a["vertex_family_rows"][assigned] < 0):
            raise ValueError("assigned source vertex lacks a canonical family")
        if np.any(a["vertex_family_rows"][~assigned] != -1):
            raise ValueError("unassigned source vertex retains a canonical family")
        if np.any(a["vertex_assignment_weight"][~assigned] != 0.0):
            raise ValueError("unassigned source vertex retains assignment weight")
        node_family = np.repeat(np.arange(family_count), np.diff(a["family_node_offsets"]))
        if not np.array_equal(
            a["vertex_family_rows"][assigned], node_family[a["vertex_node_rows"][assigned]],
        ):
            raise ValueError("source vertex family and canonical node family differ")
        if np.any(a["vertex_assignment_weight"] < 0.0) or np.any(a["vertex_assignment_weight"] > 1.0 + 1e-6):
            raise ValueError("canonical assignment weights must be probabilities")
        if any(np.any(~np.isfinite(a[name])) for name in ("node_points_world", "node_normals_world", "vertex_assignment_weight")):
            raise ValueError("canonical family layout contains nonfinite geometry")
        if len(set(a["family_keys"].astype(str).tolist())) != family_count:
            raise ValueError("canonical surface family keys must be unique")
        if str(self.metadata.get("atlas_content_sha256", "")) != str(self.atlas_content_sha256):
            raise ValueError("canonical family atlas lineage differs")
        declared = str(self.metadata.get("content_sha256", ""))
        if declared and declared != self.content_sha256:
            raise ValueError("canonical family content hash differs")
        return self

    def save_npz(self, path: Path) -> dict[str, object]:
        self.validated()
        return _save_npz(path, self.arrays(), self.metadata)

    @classmethod
    def load_npz(cls, path: Path) -> "CanonicalSurfaceFamilyLayout":
        arrays, metadata = _load_npz_arrays(path, _FAMILY_ARRAY_NAMES)
        return cls(
            atlas_content_sha256=str(metadata["atlas_content_sha256"]),
            metadata=metadata, **arrays,
        ).validated()


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(int(count), dtype=np.int64)
        self.rank = np.zeros((int(count),), dtype=np.int8)

    def find(self, value: int) -> int:
        row = int(value)
        while int(self.parent[row]) != row:
            self.parent[row] = self.parent[int(self.parent[row])]
            row = int(self.parent[row])
        return row

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if int(self.rank[a]) < int(self.rank[b]):
            a, b = b, a
        self.parent[b] = a
        if int(self.rank[a]) == int(self.rank[b]):
            self.rank[a] += 1


def build_diagnostic_metric_surface_families(
    atlas: ExplicitChartAtlas,
    *,
    atlas_content_sha256: str,
    maximum_cross_chart_distance_m: float = 0.5,
    maximum_cross_chart_normal_degrees: float = 60.0,
    maximum_intra_family_edge_m: float = 1.0,
    maximum_intra_family_normal_degrees: float = 45.0,
) -> CanonicalSurfaceFamilyLayout:
    """Build a bounded CPU diagnostic assignment, not a production ontology.

    Cross-view nodes are mutual-nearest, distance/normal-gated vertex groups.
    Surface families are connected components of those nodes through existing
    chart mesh edges.  The function exists to validate the RADIO interface
    before the independently audited production canonicalizer is available.
    """

    atlas = atlas.validated()
    if float(maximum_cross_chart_distance_m) <= 0.0 or float(maximum_intra_family_edge_m) <= 0.0:
        raise ValueError("diagnostic family distance gates must be positive")
    xyz = np.asarray(atlas.vertices_world, dtype=np.float64)
    normal = np.asarray(atlas.normals_world, dtype=np.float64)
    norm = np.linalg.norm(normal, axis=1)
    unit = normal / np.maximum(norm[:, None], 1e-12)
    valid_normal = norm >= 0.5
    vertex_dsu = _DisjointSet(len(xyz))
    cross_pair_count = 0
    cross_cos = np.cos(np.deg2rad(float(maximum_cross_chart_normal_degrees)))
    chart_count = int(atlas.chart_names.size)
    for left in range(chart_count):
        l0, l1 = map(int, atlas.chart_vertex_offsets[left:left + 2])
        left_rows = np.arange(l0, l1, dtype=np.int64)[valid_normal[l0:l1]]
        if left_rows.size == 0:
            continue
        for right in range(left + 1, chart_count):
            r0, r1 = map(int, atlas.chart_vertex_offsets[right:right + 2])
            right_rows = np.arange(r0, r1, dtype=np.int64)[valid_normal[r0:r1]]
            if right_rows.size == 0:
                continue
            right_tree = cKDTree(xyz[right_rows])
            distance, local_right = right_tree.query(xyz[left_rows], k=1)
            left_tree = cKDTree(xyz[left_rows])
            _, reverse_left = left_tree.query(xyz[right_rows], k=1)
            mutual = reverse_left[np.asarray(local_right, dtype=np.int64)] == np.arange(left_rows.size)
            target = right_rows[np.asarray(local_right, dtype=np.int64)]
            cosine = np.abs(np.sum(unit[left_rows] * unit[target], axis=1))
            keep = mutual & (distance <= float(maximum_cross_chart_distance_m)) & (cosine >= cross_cos)
            for a, b in zip(left_rows[keep].tolist(), target[keep].tolist()):
                vertex_dsu.union(a, b)
                cross_pair_count += 1
    roots = np.asarray([vertex_dsu.find(row) for row in range(len(xyz))], dtype=np.int64)
    unique_roots = sorted(set(roots.tolist()), key=lambda root: int(np.min(np.flatnonzero(roots == root))))
    old_node_by_root = {root: row for row, root in enumerate(unique_roots)}
    old_vertex_node = np.asarray([old_node_by_root[int(root)] for root in roots], dtype=np.int64)
    old_node_count = len(unique_roots)
    node_xyz = np.zeros((old_node_count, 3), dtype=np.float64)
    node_normal = np.zeros((old_node_count, 3), dtype=np.float64)
    for node in range(old_node_count):
        rows = np.flatnonzero(old_vertex_node == node)
        weight = np.maximum(np.asarray(atlas.confidence[rows], dtype=np.float64), 1e-6)
        node_xyz[node] = np.sum(weight[:, None] * xyz[rows], axis=0) / np.sum(weight)
        normal_rows = rows[valid_normal[rows]]
        if normal_rows.size:
            reference = unit[normal_rows[0]]
            aligned = unit[normal_rows] * np.where(
                np.sum(unit[normal_rows] * reference[None, :], axis=1) >= 0.0, 1.0, -1.0,
            )[:, None]
            value = np.sum(aligned, axis=0)
            node_normal[node] = value / max(float(np.linalg.norm(value)), 1e-12)
    family_dsu = _DisjointSet(old_node_count)
    intra_cos = np.cos(np.deg2rad(float(maximum_intra_family_normal_degrees)))
    for face in np.asarray(atlas.faces, dtype=np.int64):
        for a_vertex, b_vertex in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            a, b = int(old_vertex_node[a_vertex]), int(old_vertex_node[b_vertex])
            if a == b:
                continue
            na, nb = node_normal[a], node_normal[b]
            if np.linalg.norm(na) < 0.5 or np.linalg.norm(nb) < 0.5:
                continue
            if (
                np.linalg.norm(node_xyz[a] - node_xyz[b]) <= float(maximum_intra_family_edge_m)
                and abs(float(np.dot(na, nb))) >= intra_cos
            ):
                family_dsu.union(a, b)
    family_roots = np.asarray([family_dsu.find(row) for row in range(old_node_count)], dtype=np.int64)
    root_order = sorted(
        set(family_roots.tolist()),
        key=lambda root: int(np.min(np.flatnonzero(family_roots == root))),
    )
    old_family_by_root = {root: row for row, root in enumerate(root_order)}
    old_node_family = np.asarray([old_family_by_root[int(root)] for root in family_roots], dtype=np.int64)
    new_order = np.concatenate([np.flatnonzero(old_node_family == row) for row in range(len(root_order))])
    new_by_old = np.empty(old_node_count, dtype=np.int64)
    new_by_old[new_order] = np.arange(old_node_count, dtype=np.int64)
    vertex_node = new_by_old[old_vertex_node]
    family_sizes = np.bincount(old_node_family, minlength=len(root_order))
    family_offsets = np.r_[0, np.cumsum(family_sizes)].astype(np.int64)
    vertex_family = old_node_family[old_vertex_node]
    reordered_xyz = node_xyz[new_order]
    reordered_normal = node_normal[new_order]
    assignment_distance = np.linalg.norm(xyz - reordered_xyz[vertex_node], axis=1)
    assignment = np.exp(-np.square(assignment_distance / float(maximum_cross_chart_distance_m)))
    arrays = {
        "family_keys": np.asarray([f"surface-family-{row:05d}" for row in range(len(root_order))]),
        "family_node_offsets": family_offsets,
        "node_points_world": reordered_xyz,
        "node_normals_world": reordered_normal,
        "vertex_family_rows": vertex_family.astype(np.int32),
        "vertex_node_rows": vertex_node,
        "vertex_assignment_weight": assignment.astype(np.float32),
    }
    chart_row = np.repeat(np.arange(chart_count), np.diff(atlas.chart_vertex_offsets))
    node_view_count = np.zeros(old_node_count, dtype=np.int32)
    for node in range(old_node_count):
        node_view_count[node] = np.unique(chart_row[vertex_node == node]).size
    metadata = {
        "artifact_type": FAMILY_SCHEMA,
        "representation": "diagnostic_mutual_nearest_metric_nodes_and_mesh_connected_surface_families",
        "diagnostic_only": True,
        "production_eligible": False,
        "canonical_identity_is_source_view_id": False,
        "atlas_content_sha256": str(atlas_content_sha256),
        "maximum_cross_chart_distance_m": float(maximum_cross_chart_distance_m),
        "maximum_cross_chart_normal_degrees": float(maximum_cross_chart_normal_degrees),
        "maximum_intra_family_edge_m": float(maximum_intra_family_edge_m),
        "maximum_intra_family_normal_degrees": float(maximum_intra_family_normal_degrees),
        "mutual_cross_chart_pair_count": int(cross_pair_count),
        "multi_view_node_count": int(np.sum(node_view_count >= 2)),
        "uses_query_or_ground_truth": False,
    }
    return CanonicalSurfaceFamilyLayout(
        atlas_content_sha256=str(atlas_content_sha256), metadata=metadata, **arrays,
    ).validated()


def build_carrier_constrained_surface_family_layout(
    atlas: ExplicitChartAtlas,
    carrier: CanonicalSurfaceFamilyCarrier,
    *,
    atlas_content_sha256: str,
    carrier_content_sha256: str | None = None,
    maximum_cross_chart_distance_m: float = 0.5,
    maximum_cross_chart_normal_degrees: float = 60.0,
) -> CanonicalSurfaceFamilyLayout:
    """Attach source vertices to nodes *inside* an authoritative family carrier.

    The carrier fixes physical family identity.  This adapter is allowed to
    form metric feature-sampling nodes only between parameterizations already
    assigned to the same carrier family; it can never merge two families.
    Mesh-crease boundary vertices may occur in more than one patch.  Such a
    vertex receives deterministic single ownership from the largest eligible
    incident patch (then lowest patch ID), while the ambiguity count is kept
    in lineage.  Vertices absent from all carrier patches remain explicitly
    unassigned with node/family ``-1`` and zero weight.

    This is a RADIO association layout, not a replacement for the carrier's
    still-missing shared canonical mesh.  Accordingly it is never marked as a
    deployment/pose-ready geometry artifact.
    """

    atlas = atlas.validated()
    carrier = carrier.validated()
    if float(maximum_cross_chart_distance_m) <= 0.0:
        raise ValueError("carrier-constrained node distance must be positive")
    if not 0.0 < float(maximum_cross_chart_normal_degrees) <= 90.0:
        raise ValueError("carrier-constrained node normal gate is invalid")
    observed_carrier_hash = str(carrier.metadata.get("content_sha256", ""))
    if carrier_content_sha256 is not None and str(carrier_content_sha256) != observed_carrier_hash:
        raise ValueError("surface-family carrier content differs from runner authority")
    if len(observed_carrier_hash) != 64:
        raise ValueError("surface-family carrier lacks a sealed content hash")
    if str(carrier.metadata.get("source_atlas_content_sha256", "")) != str(
        atlas_content_sha256
    ):
        raise ValueError("surface-family carrier and explicit atlas differ")
    chart_count = int(atlas.chart_names.size)
    if int(carrier.metadata.get("parameterization_count", -1)) != chart_count:
        raise ValueError("surface-family carrier parameterization inventory differs")

    vertex_count = int(atlas.vertices_world.shape[0])
    patch_count = int(carrier.patch_count)
    candidates: list[list[int]] = [[] for _ in range(vertex_count)]
    for patch in range(patch_count):
        parameterization = int(carrier.patch_parameterization_ids[patch])
        if parameterization < 0 or parameterization >= chart_count:
            raise ValueError("surface-family patch parameterization is outside atlas")
        lo, hi = map(int, atlas.chart_vertex_offsets[parameterization:parameterization + 2])
        begin, end = map(int, carrier.patch_vertex_offsets[patch:patch + 2])
        rows = np.asarray(carrier.patch_vertex_indices[begin:end], dtype=np.int64)
        if np.any(rows < lo) or np.any(rows >= hi):
            raise ValueError("surface-family patch vertices cross parameterization boundaries")
        for vertex in rows.tolist():
            candidates[int(vertex)].append(patch)

    vertex_family = np.full((vertex_count,), -1, dtype=np.int32)
    boundary_ambiguous = 0
    for vertex, patch_ids in enumerate(candidates):
        if not patch_ids:
            continue
        families = {int(carrier.patch_family_ids[patch]) for patch in patch_ids}
        boundary_ambiguous += int(len(families) > 1)
        chosen = max(
            patch_ids,
            key=lambda patch: (
                int(bool(carrier.patch_geometry_eligible[patch])),
                float(carrier.patch_area_m2[patch]),
                -int(patch),
            ),
        )
        vertex_family[vertex] = int(carrier.patch_family_ids[chosen])

    xyz = np.asarray(atlas.vertices_world, dtype=np.float64)
    normal = np.asarray(atlas.normals_world, dtype=np.float64)
    normal_norm = np.linalg.norm(normal, axis=1)
    normal_unit = normal / np.maximum(normal_norm[:, None], 1e-12)
    valid_normal = normal_norm >= 0.5
    chart_rows = np.repeat(np.arange(chart_count), np.diff(atlas.chart_vertex_offsets))
    union = _DisjointSet(vertex_count)
    threshold = float(maximum_cross_chart_distance_m)
    minimum_cosine = np.cos(np.deg2rad(float(maximum_cross_chart_normal_degrees)))
    accepted_pairs = 0
    for family in range(carrier.family_count):
        family_rows = np.flatnonzero((vertex_family == family) & valid_normal)
        parameterizations = np.unique(chart_rows[family_rows])
        for left_offset, left_parameterization in enumerate(parameterizations[:-1]):
            left_rows = family_rows[chart_rows[family_rows] == left_parameterization]
            for right_parameterization in parameterizations[left_offset + 1:]:
                right_rows = family_rows[chart_rows[family_rows] == right_parameterization]
                if not left_rows.size or not right_rows.size:
                    continue
                right_tree = cKDTree(xyz[right_rows])
                distance, local_right = right_tree.query(xyz[left_rows], k=1)
                left_tree = cKDTree(xyz[left_rows])
                _, reverse_left = left_tree.query(xyz[right_rows], k=1)
                local_right = np.asarray(local_right, dtype=np.int64)
                mutual = reverse_left[local_right] == np.arange(left_rows.size)
                targets = right_rows[local_right]
                cosine = np.abs(np.sum(normal_unit[left_rows] * normal_unit[targets], axis=1))
                keep = mutual & (distance <= threshold) & (cosine >= minimum_cosine)
                for first, second in zip(left_rows[keep].tolist(), targets[keep].tolist()):
                    union.union(int(first), int(second))
                    accepted_pairs += 1

    family_node_offsets = [0]
    node_points: list[np.ndarray] = []
    node_normals: list[np.ndarray] = []
    vertex_nodes = np.full((vertex_count,), -1, dtype=np.int64)
    node_view_counts: list[int] = []
    for family in range(carrier.family_count):
        rows = np.flatnonzero(vertex_family == family)
        if not rows.size:
            raise ValueError("surface-family carrier produced a family without owned atlas vertices")
        roots = np.asarray([union.find(int(row)) for row in rows], dtype=np.int64)
        ordered_roots = sorted(
            set(roots.tolist()),
            key=lambda root: int(np.min(rows[roots == root])),
        )
        for root in ordered_roots:
            members = rows[roots == root]
            weight = np.maximum(np.asarray(atlas.confidence[members], dtype=np.float64), 1e-6)
            point = np.sum(weight[:, None] * xyz[members], axis=0) / np.sum(weight)
            normals = normal_unit[members[valid_normal[members]]]
            if normals.size:
                reference = normals[0]
                aligned = normals * np.where(normals @ reference >= 0.0, 1.0, -1.0)[:, None]
                fused_normal = np.sum(aligned, axis=0)
                fused_normal /= max(float(np.linalg.norm(fused_normal)), 1e-12)
            else:
                fused_normal = np.zeros((3,), dtype=np.float64)
            node = len(node_points)
            vertex_nodes[members] = node
            node_points.append(point)
            node_normals.append(fused_normal)
            node_view_counts.append(int(np.unique(chart_rows[members]).size))
        family_node_offsets.append(len(node_points))

    node_points_array = np.asarray(node_points, dtype=np.float64).reshape(-1, 3)
    node_normals_array = np.asarray(node_normals, dtype=np.float64).reshape(-1, 3)
    assignment_weight = np.zeros((vertex_count,), dtype=np.float32)
    assigned = vertex_nodes >= 0
    assignment_distance = np.linalg.norm(
        xyz[assigned] - node_points_array[vertex_nodes[assigned]], axis=1,
    )
    assignment_weight[assigned] = np.exp(-np.square(assignment_distance / threshold)).astype(
        np.float32
    )
    arrays = {
        "family_keys": np.asarray(
            [f"canonical-surface-family-{row:05d}" for row in range(carrier.family_count)]
        ),
        "family_node_offsets": np.asarray(family_node_offsets, dtype=np.int64),
        "node_points_world": node_points_array,
        "node_normals_world": node_normals_array,
        "vertex_family_rows": vertex_family,
        "vertex_node_rows": vertex_nodes,
        "vertex_assignment_weight": assignment_weight,
    }
    interface_eligible = bool(carrier.metadata.get("family_canonicalization_gate_pass", False))
    metadata = {
        "artifact_type": FAMILY_SCHEMA,
        "representation": "carrier_family_constrained_mutual_metric_radio_sampling_nodes",
        "diagnostic_only": False,
        "production_eligible": False,
        "feature_interface_eligible": interface_eligible,
        "production_blocker": "canonical_surface_family_carrier_has_no_shared_fused_mesh",
        "canonical_geometry_fused": False,
        "canonical_identity_is_source_view_id": False,
        "runtime_candidate_unit": "canonical_surface_family",
        "atlas_content_sha256": str(atlas_content_sha256),
        "surface_family_carrier_schema": str(carrier.metadata.get("artifact_type", "")),
        "surface_family_carrier_content_sha256": observed_carrier_hash,
        "carrier_family_canonicalization_gate_pass": interface_eligible,
        "maximum_cross_chart_distance_m": threshold,
        "maximum_cross_chart_normal_degrees": float(maximum_cross_chart_normal_degrees),
        "cross_parameterization_metric_node_pair_count": int(accepted_pairs),
        "multi_view_node_count": int(np.sum(np.asarray(node_view_counts) >= 2)),
        "unassigned_atlas_vertex_count": int(np.sum(~assigned)),
        "multi_family_boundary_vertex_count": int(boundary_ambiguous),
        "boundary_ownership_contract": (
            "largest_geometry_eligible_incident_patch_then_area_then_lowest_patch_id"
        ),
        "family_merge_authority": "input_carrier_only_adapter_never_merges_carrier_families",
        "uses_query_or_ground_truth": False,
    }
    return CanonicalSurfaceFamilyLayout(
        atlas_content_sha256=str(atlas_content_sha256), metadata=metadata, **arrays,
    ).validated()


_CANONICAL_ARRAY_NAMES = (
    "family_keys", "family_node_offsets", "node_points_world", "node_normals_world",
    "codes", "confidence", "uncertainty", "view_count", "effective_view_count",
    "prototype_offsets", "prototype_codes", "prototype_weights", "prototype_view_directions",
)


@dataclass(frozen=True)
class CanonicalChartRadioField:
    """View-balanced RADIO-final field on canonical physical surface nodes."""

    family_keys: np.ndarray
    family_node_offsets: np.ndarray
    node_points_world: np.ndarray
    node_normals_world: np.ndarray
    codes: np.ndarray
    confidence: np.ndarray
    uncertainty: np.ndarray
    view_count: np.ndarray
    effective_view_count: np.ndarray
    prototype_offsets: np.ndarray
    prototype_codes: np.ndarray
    prototype_weights: np.ndarray
    prototype_view_directions: np.ndarray
    source_field_content_sha256: str
    family_layout_content_sha256: str
    metadata: Mapping[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in _CANONICAL_ARRAY_NAMES}

    @property
    def content_sha256(self) -> str:
        return _artifact_content_sha256(self.metadata, self.arrays())

    def validated(self) -> "CanonicalChartRadioField":
        a = self.arrays()
        family_count = int(a["family_keys"].size)
        node_count = int(a["codes"].shape[0])
        feature_dim = int(a["codes"].shape[1]) if a["codes"].ndim == 2 else 0
        prototype_count = int(a["prototype_codes"].shape[0])
        if self.metadata.get("artifact_type") != CANONICAL_SCHEMA:
            raise ValueError("wrong canonical chart RADIO schema")
        if bool(self.metadata.get("stores_source_view_names", True)):
            raise ValueError("canonical chart field retains source-view identities")
        if bool(self.metadata.get("uses_query_or_ground_truth", True)):
            raise ValueError("canonical chart field used query/ground truth")
        if a["family_node_offsets"].shape != (family_count + 1,) or int(a["family_node_offsets"][-1]) != node_count:
            raise ValueError("canonical family/node inventory differs")
        if a["node_points_world"].shape != (node_count, 3) or a["node_normals_world"].shape != (node_count, 3):
            raise ValueError("canonical node geometry differs")
        for name in ("confidence", "uncertainty", "view_count", "effective_view_count"):
            if a[name].shape != (node_count,):
                raise ValueError(f"canonical {name} differs")
        if a["prototype_offsets"].shape != (node_count + 1,) or int(a["prototype_offsets"][-1]) != prototype_count:
            raise ValueError("canonical anonymous prototype offsets differ")
        if a["prototype_codes"].shape != (prototype_count, feature_dim) or a["prototype_weights"].shape != (prototype_count,) or a["prototype_view_directions"].shape != (prototype_count, 3):
            raise ValueError("canonical anonymous prototype arrays differ")
        if any(np.any(~np.isfinite(a[name])) for name in _CANONICAL_ARRAY_NAMES if name != "family_keys"):
            raise ValueError("canonical chart RADIO field contains nonfinite values")
        if np.any(a["confidence"] < 0.0) or np.any(a["confidence"] > 1.0 + 1e-6) or np.any(a["uncertainty"] < 0.0):
            raise ValueError("canonical confidence/uncertainty is invalid")
        norm = np.linalg.norm(a["codes"], axis=1)
        observed = a["view_count"] > 0
        if np.any(np.abs(norm[observed] - 1.0) > 2e-5) or np.any(norm[~observed] > 1e-8):
            raise ValueError("canonical RADIO codes violate normalization/missingness")
        if str(self.metadata.get("source_field_content_sha256", "")) != str(self.source_field_content_sha256) or str(self.metadata.get("family_layout_content_sha256", "")) != str(self.family_layout_content_sha256):
            raise ValueError("canonical field source lineage differs")
        declared = str(self.metadata.get("content_sha256", ""))
        if declared and declared != self.content_sha256:
            raise ValueError("canonical chart field content hash differs")
        return self

    def save_npz(self, path: Path) -> dict[str, object]:
        self.validated()
        return _save_npz(path, self.arrays(), self.metadata)

    @classmethod
    def load_npz(cls, path: Path) -> "CanonicalChartRadioField":
        arrays, metadata = _load_npz_arrays(path, _CANONICAL_ARRAY_NAMES)
        return cls(
            source_field_content_sha256=str(metadata["source_field_content_sha256"]),
            family_layout_content_sha256=str(metadata["family_layout_content_sha256"]),
            metadata=metadata, **arrays,
        ).validated()


def fuse_canonical_chart_radio_field(
    source: SourceViewChartRadioField,
    layout: CanonicalSurfaceFamilyLayout,
    *,
    maximum_anonymous_prototypes: int = 4,
    minimum_observation_weight: float = 1e-4,
) -> CanonicalChartRadioField:
    """Fuse chart vertices via view-balanced canonical physical nodes.

    Vertices are first fused inside each ``(canonical node, source view)``.
    Only then are view records fused across views.  Consequently a dense
    source chart cannot dominate a sparse chart merely by contributing more
    vertices.  Prototype rows retain feature and viewing direction but drop
    the source-view identity.
    """

    source = source.validated()
    layout = layout.validated()
    if str(source.atlas_content_sha256) != str(layout.atlas_content_sha256):
        raise ValueError("source RADIO and canonical family atlases differ")
    vertex_count = int(source.vertex_token_rows.shape[0])
    if int(layout.vertex_node_rows.size) != vertex_count:
        raise ValueError("canonical family layout and chart vertex inventories differ")
    if int(maximum_anonymous_prototypes) <= 0 or float(minimum_observation_weight) < 0.0:
        raise ValueError("canonical fusion prototype/weight configuration is invalid")
    descriptors = source.sampled_vertex_codes()
    view_count_total = int(source.view_names.size)
    vertex_view = np.repeat(np.arange(view_count_total), np.diff(source.chart_vertex_offsets))
    node_count = int(layout.node_points_world.shape[0])
    raw_weight = (
        np.asarray(source.vertex_observation_weight, dtype=np.float64)
        * np.asarray(layout.vertex_assignment_weight, dtype=np.float64)
    )
    valid = (
        (raw_weight >= float(minimum_observation_weight))
        & (np.linalg.norm(descriptors, axis=1) > 0.5)
        & (np.asarray(layout.vertex_node_rows, dtype=np.int64) >= 0)
    )
    keys = layout.vertex_node_rows[valid].astype(np.int64) * view_count_total + vertex_view[valid]
    rows = np.flatnonzero(valid)
    order = np.argsort(keys, kind="stable")
    keys, rows = keys[order], rows[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1] if keys.size else np.zeros((0,), dtype=np.int64)
    view_nodes, view_ids, view_codes, view_weights, view_directions = [], [], [], [], []
    for index, begin in enumerate(starts.tolist()):
        end = int(starts[index + 1]) if index + 1 < starts.size else int(keys.size)
        selected = rows[begin:end]
        weight = raw_weight[selected]
        code = np.sum(weight[:, None] * descriptors[selected], axis=0)
        code /= max(float(np.linalg.norm(code)), 1e-12)
        view = int(vertex_view[selected[0]])
        center = np.asarray(source.camera_centers_world[view], dtype=np.float64)
        point = np.asarray(layout.node_points_world[int(layout.vertex_node_rows[selected[0]])], dtype=np.float64)
        direction = center - point
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        view_nodes.append(int(layout.vertex_node_rows[selected[0]]))
        view_ids.append(view)
        view_codes.append(code.astype(np.float32))
        # Mean, not sum: chart sampling density cannot increase view authority.
        view_weights.append(float(np.mean(weight)))
        view_directions.append(direction.astype(np.float32))
    view_nodes_array = np.asarray(view_nodes, dtype=np.int64)
    view_ids_array = np.asarray(view_ids, dtype=np.int64)
    view_codes_array = np.asarray(view_codes, dtype=np.float32).reshape(-1, source.feature_dim)
    view_weights_array = np.asarray(view_weights, dtype=np.float64)
    view_directions_array = np.asarray(view_directions, dtype=np.float32).reshape(-1, 3)
    codes = np.zeros((node_count, source.feature_dim), dtype=np.float32)
    confidence = np.zeros((node_count,), dtype=np.float32)
    uncertainty = np.zeros((node_count,), dtype=np.float32)
    contributing_views = np.zeros((node_count,), dtype=np.int16)
    effective_views = np.zeros((node_count,), dtype=np.float32)
    prototype_offsets = [0]
    prototype_codes, prototype_weights, prototype_directions = [], [], []
    for node in range(node_count):
        selected = np.flatnonzero(view_nodes_array == node)
        if selected.size == 0:
            prototype_offsets.append(prototype_offsets[-1])
            continue
        # A construction bug must never silently emit duplicate records from one view.
        if np.unique(view_ids_array[selected]).size != selected.size:
            raise ValueError("canonical fusion retained duplicate source-view node records")
        weight = np.clip(view_weights_array[selected], 0.0, 1.0)
        value = np.sum(weight[:, None] * view_codes_array[selected], axis=0)
        value /= max(float(np.linalg.norm(value)), 1e-12)
        agreement = np.clip(view_codes_array[selected] @ value, -1.0, 1.0)
        effective = float(np.sum(weight) / max(float(np.max(weight)), 1e-12))
        support = 1.0 - np.exp(-float(np.sum(weight)))
        consensus = float(np.sum(weight * (agreement + 1.0) * 0.5) / max(float(np.sum(weight)), 1e-12))
        codes[node] = value.astype(np.float32)
        confidence[node] = float(np.clip(support * consensus, 0.0, 1.0))
        uncertainty[node] = float(np.sum(weight * (1.0 - agreement)) / max(float(np.sum(weight)), 1e-12))
        contributing_views[node] = int(selected.size)
        effective_views[node] = effective
        # Source IDs determine a stable tie-break only; they are not serialized.
        ranked = selected[np.lexsort((view_ids_array[selected], -weight))]
        retained = ranked[: int(maximum_anonymous_prototypes)]
        retained_weight = np.clip(view_weights_array[retained], 0.0, 1.0)
        prototype_codes.append(view_codes_array[retained])
        prototype_weights.append(retained_weight.astype(np.float32))
        prototype_directions.append(view_directions_array[retained])
        prototype_offsets.append(prototype_offsets[-1] + retained.size)
    if prototype_codes:
        prototype_code_array = np.concatenate(prototype_codes)
        prototype_weight_array = np.concatenate(prototype_weights)
        prototype_direction_array = np.concatenate(prototype_directions)
    else:
        prototype_code_array = np.zeros((0, source.feature_dim), dtype=np.float32)
        prototype_weight_array = np.zeros((0,), dtype=np.float32)
        prototype_direction_array = np.zeros((0, 3), dtype=np.float32)
    arrays = {
        "family_keys": np.asarray(layout.family_keys).copy(),
        "family_node_offsets": np.asarray(layout.family_node_offsets, dtype=np.int64).copy(),
        "node_points_world": np.asarray(layout.node_points_world, dtype=np.float64).copy(),
        "node_normals_world": np.asarray(layout.node_normals_world, dtype=np.float64).copy(),
        "codes": codes,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "view_count": contributing_views,
        "effective_view_count": effective_views,
        "prototype_offsets": np.asarray(prototype_offsets, dtype=np.int64),
        "prototype_codes": prototype_code_array,
        "prototype_weights": prototype_weight_array,
        "prototype_view_directions": prototype_direction_array,
    }
    metadata = {
        "artifact_type": CANONICAL_SCHEMA,
        "representation": "view_balanced_radio_final_on_canonical_surface_family_nodes",
        "vfm_layer": "radio_final",
        "feature_dimension": int(source.feature_dim),
        "feature_storage_dtype": "float32",
        "canonical_node_count": int(node_count),
        "anonymous_prototype_count": int(prototype_code_array.shape[0]),
        "canonical_feature_uncompressed_bytes": int(codes.nbytes + prototype_code_array.nbytes),
        "feature_projection": "none_full_1280d_interface_gate",
        "fusion_order": "vertices_to_unique_node_view_records_then_views_to_canonical_node",
        "view_density_authority": "mean_vertex_quality_per_view_not_vertex_count",
        "anonymous_prototypes": "feature_and_view_direction_without_source_view_identity",
        "stores_source_view_names": False,
        "stores_complete_source_token_layout": False,
        "source_lineage_retained_by_hash_only": True,
        "source_field_content_sha256": source.content_sha256,
        "family_layout_content_sha256": layout.content_sha256,
        "maximum_anonymous_prototypes": int(maximum_anonymous_prototypes),
        "minimum_observation_weight": float(minimum_observation_weight),
        "uses_mapping_camera_pose": True,
        "uses_query_or_ground_truth": False,
        "uses_gaussian_or_2dgs": False,
        "production_eligible": bool(layout.metadata.get("production_eligible", False)),
        "feature_interface_eligible": bool(
            layout.metadata.get("feature_interface_eligible", False)
        ),
    }
    return CanonicalChartRadioField(
        source_field_content_sha256=source.content_sha256,
        family_layout_content_sha256=layout.content_sha256,
        metadata=metadata,
        **arrays,
    ).validated()


__all__ = [
    "SOURCE_SCHEMA", "FAMILY_SCHEMA", "CANONICAL_SCHEMA", "COORDINATE_CONTRACT",
    "RawSimpleRadialCamera", "IdealChartCamera", "SourceViewChartRadioField",
    "CanonicalSurfaceFamilyLayout", "CanonicalChartRadioField",
    "chart_name_to_image_id", "attach_radio_to_chart_atlas",
    "build_diagnostic_metric_surface_families",
    "build_carrier_constrained_surface_family_layout",
    "fuse_canonical_chart_radio_field",
]
