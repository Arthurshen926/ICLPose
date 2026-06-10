"""Official 2DGS RGB/depth rendering adapter.

This module intentionally does not reuse the diagnostic soft splat renderer.
It loads a full 2DGS-style Gaussian PLY with two in-plane scales and rotations,
then calls ``gsplat.rasterization_2dgs`` for RGB+expected-depth rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-value))


def _sorted_property_names(names: Iterable[str], prefix: str) -> list[str]:
    selected = [str(name) for name in names if str(name).startswith(prefix)]
    return sorted(selected, key=lambda item: int(item.split("_")[-1]))


def _infer_sh_degree(rest_count: int) -> int:
    if rest_count == 0:
        return 0
    if rest_count % 3 != 0:
        raise ValueError("f_rest_* property count must be divisible by 3")
    coeff_count = 1 + rest_count // 3
    root = int(round(np.sqrt(coeff_count)))
    if root * root != coeff_count:
        raise ValueError("f_rest_* property count does not form a valid SH basis")
    return root - 1


@dataclass(frozen=True)
class Official2DGSSource:
    xyz: np.ndarray
    sh_features: np.ndarray
    opacity_logits: np.ndarray
    log_scales_2d: np.ndarray
    rotations: np.ndarray
    loc_features: np.ndarray | None
    sh_degree: int
    path: str

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float32)
        sh_features = np.asarray(self.sh_features, dtype=np.float32)
        opacity_logits = np.asarray(self.opacity_logits, dtype=np.float32).reshape(-1)
        log_scales_2d = np.asarray(self.log_scales_2d, dtype=np.float32)
        rotations = np.asarray(self.rotations, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        if sh_features.ndim != 3 or sh_features.shape[0] != xyz.shape[0] or sh_features.shape[2] != 3:
            raise ValueError("sh_features must have shape (N, K, 3)")
        if opacity_logits.shape[0] != xyz.shape[0]:
            raise ValueError("opacity_logits must have shape (N,)")
        if log_scales_2d.shape != (xyz.shape[0], 2):
            raise ValueError("full 2DGS rendering requires exactly two scale channels per Gaussian")
        if rotations.shape != (xyz.shape[0], 4):
            raise ValueError("rotations must have shape (N, 4)")
        loc_features = None if self.loc_features is None else np.asarray(self.loc_features, dtype=np.float32)
        if loc_features is not None and (loc_features.ndim != 2 or loc_features.shape[0] != xyz.shape[0]):
            raise ValueError("loc_features must have shape (N, C)")
        expected_coeffs = (int(self.sh_degree) + 1) ** 2
        if sh_features.shape[1] != expected_coeffs:
            raise ValueError("sh_features coefficient count does not match sh_degree")
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "sh_features", sh_features)
        object.__setattr__(self, "opacity_logits", opacity_logits)
        object.__setattr__(self, "log_scales_2d", log_scales_2d)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "loc_features", loc_features)

    @property
    def gaussian_count(self) -> int:
        return int(self.xyz.shape[0])


def load_official_2dgs_source_from_ply(path: Path, *, sh_degree: int | None = None) -> Official2DGSSource:
    """Load a complete 2DGS PLY.

    The loader rejects proxy point clouds that only contain one scale channel,
    because those cannot be rendered with the 2DGS surface rasterizer.
    """

    try:
        from plyfile import PlyData
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("plyfile is required to load 2DGS PLY files") from exc
    ply = PlyData.read(Path(path))
    vertex = ply.elements[0]
    names = vertex.data.dtype.names or ()
    required = {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"2DGS PLY is missing required properties: {missing}")
    scale_names = _sorted_property_names(names, "scale_")
    if len(scale_names) != 2:
        raise ValueError(
            f"full 2DGS rendering requires exactly two scale_* properties, found {len(scale_names)}"
        )
    rot_names = _sorted_property_names(names, "rot_")
    if len(rot_names) != 4:
        raise ValueError(f"full 2DGS rendering requires exactly four rot_* properties, found {len(rot_names)}")
    rest_names = _sorted_property_names(names, "f_rest_")
    degree = _infer_sh_degree(len(rest_names)) if sh_degree is None else int(sh_degree)
    expected_rest = 3 * ((degree + 1) ** 2 - 1)
    if len(rest_names) != expected_rest:
        raise ValueError(
            f"sh_degree={degree} expects {expected_rest} f_rest_* properties, found {len(rest_names)}"
        )
    xyz = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in ("x", "y", "z")], axis=1)
    opacity = np.asarray(vertex["opacity"], dtype=np.float32)
    scales = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in scale_names], axis=1)
    rotations = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in rot_names], axis=1)
    coeff_count = (degree + 1) ** 2
    sh = np.zeros((xyz.shape[0], coeff_count, 3), dtype=np.float32)
    sh[:, 0, 0] = np.asarray(vertex["f_dc_0"], dtype=np.float32)
    sh[:, 0, 1] = np.asarray(vertex["f_dc_1"], dtype=np.float32)
    sh[:, 0, 2] = np.asarray(vertex["f_dc_2"], dtype=np.float32)
    if rest_names:
        rest = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in rest_names], axis=1)
        sh[:, 1:, :] = rest.reshape(xyz.shape[0], 3, coeff_count - 1).transpose(0, 2, 1)
    loc_names = _sorted_property_names(names, "loc_")
    loc = None
    if loc_names:
        loc = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in loc_names], axis=1)
    return Official2DGSSource(
        xyz=xyz,
        sh_features=sh,
        opacity_logits=opacity,
        log_scales_2d=scales,
        rotations=rotations,
        loc_features=loc,
        sh_degree=degree,
        path=str(path),
    )


def scaled_camera_matrix(camera: ColmapCamera, width: int, height: int) -> np.ndarray:
    matrix, _distortion = camera_matrix_and_distortion(camera)
    scaled = np.asarray(matrix, dtype=np.float32).copy()
    sx = float(width) / max(float(camera.width), 1.0)
    sy = float(height) / max(float(camera.height), 1.0)
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def render_official_2dgs_rgb_depth(
    source: Official2DGSSource,
    *,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    width: int,
    height: int,
    device: str = "cuda",
    near_plane: float = 0.01,
    far_plane: float = 10000.0,
    background: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render RGB, expected depth and alpha using gsplat's 2DGS rasterizer."""

    try:
        import torch
        from gsplat import rasterization_2dgs
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("gsplat with rasterization_2dgs is required for official 2DGS rendering") from exc
    if source.gaussian_count == 0:
        return (
            np.zeros((int(height), int(width), 3), dtype=np.float32),
            np.zeros((int(height), int(width)), dtype=np.float32),
            np.zeros((int(height), int(width)), dtype=np.float32),
        )
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("official 2DGS rendering requested CUDA, but CUDA is unavailable")
    means = torch.as_tensor(source.xyz, dtype=torch.float32, device=torch_device)
    quats = torch.as_tensor(source.rotations, dtype=torch.float32, device=torch_device)
    quats = torch.nn.functional.normalize(quats, dim=1)
    scales2d = torch.as_tensor(source.log_scales_2d, dtype=torch.float32, device=torch_device).exp()
    scales = torch.cat([scales2d, torch.ones((scales2d.shape[0], 1), dtype=scales2d.dtype, device=torch_device)], dim=1)
    opacities = torch.sigmoid(torch.as_tensor(source.opacity_logits, dtype=torch.float32, device=torch_device).reshape(-1))
    colors = torch.as_tensor(source.sh_features, dtype=torch.float32, device=torch_device)
    viewmat = torch.as_tensor(np.asarray(pose_w2c, dtype=np.float32).reshape(4, 4), dtype=torch.float32, device=torch_device)
    k_matrix = torch.as_tensor(scaled_camera_matrix(camera, int(width), int(height)), dtype=torch.float32, device=torch_device)
    bg = torch.as_tensor(background, dtype=torch.float32, device=torch_device).reshape(4)
    rendered, alphas, _normals, _surf_normals, _distort, _median_depth, _info = rasterization_2dgs(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None],
        Ks=k_matrix[None],
        width=int(width),
        height=int(height),
        packed=False,
        sh_degree=int(source.sh_degree),
        backgrounds=bg[None],
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        render_mode="RGB+ED",
    )
    image = rendered[0].detach().cpu().numpy().astype(np.float32, copy=False)
    rgb = np.clip(image[..., :3], 0.0, 1.0)
    depth = image[..., 3].astype(np.float32, copy=False) if image.shape[-1] > 3 else np.zeros((int(height), int(width)), dtype=np.float32)
    alpha = alphas[0].detach().cpu().numpy().astype(np.float32, copy=False)
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    return rgb, depth, alpha
