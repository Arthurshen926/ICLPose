"""Resident-GPU batched full-geometry renderer for soft surface pose energy.

This is the first production-candidate batching seam.  Geometry and canonical
payload are materialized once; camera projection and tile intersection are
batched.  The deterministic token/child reducer is intentionally shared with
the scalar authority until a GPU reducer passes the same equivalence gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np

from feature_extract.vfm.vfm_2dgs_mapping import (
    SurfaceElementMap,
    _composite_sorted_packed_hits,
    _intrinsic_matrix,
    _surface_element_quaternions_and_scales,
)

from .canonical_field import CanonicalSurfaceField
from .physical_map import DOUBLE_SIDED, GoalMapletPhysicalMap
from .surface_renderer import (
    RenderedSoftChildMixture,
    dominant_child_owner,
    _raw_to_ideal_token_warp,
    _reduce_soft_child_token_hits,
)


@dataclass(frozen=True)
class ResidentRendererBatchAudit:
    batch_size: int
    projection_tile_raster_seconds: float
    device_to_host_seconds: float
    depth_sort_composite_seconds: float
    raw_token_gather_seconds: float
    child_identity_reduction_seconds: float
    feature_reduction_seconds: float
    typed_finalize_seconds: float
    direct_parent_reduction_seconds: float
    child_and_feature_reduction_seconds: float
    raster_seconds: float
    host_reduction_seconds: float
    total_seconds: float
    packed_hit_count: int
    remapped_hit_count: int
    resident_geometry_bytes: int
    gpu_child_reducer_implemented: bool
    production_speed_gate_passed: bool


@dataclass(frozen=True)
class ResidentSoftSurfaceBatch:
    rendered: tuple[RenderedSoftChildMixture, ...]
    audit: ResidentRendererBatchAudit


class FrozenSoftSurfaceSceneGPU:
    """Hash-bound static scene with batched camera projection on one GPU."""

    def __init__(
        self,
        physical: GoalMapletPhysicalMap,
        field: CanonicalSurfaceField,
        *,
        device: str = "cuda",
    ) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch is required for resident surface rendering") from exc
        self._torch = torch
        self.device = torch.device(str(device))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FrozenSoftSurfaceSceneGPU requires CUDA")
        if field.physical_map_sha256 != physical.content_sha256:
            raise ValueError("canonical field and physical map lineage differ")
        self.physical = physical
        self.field = field
        elements = SurfaceElementMap(
            element_ids=physical.primitive_ids,
            parent_gaussian_indices=physical.primitive_ids,
            centers=physical.primitive_centers,
            tangent1=physical.primitive_tangent1,
            tangent2=physical.primitive_tangent2,
            normals=physical.primitive_normals,
            scale1=physical.primitive_scale1,
            scale2=physical.primitive_scale2,
            opacity=physical.primitive_opacity,
            area=np.pi * physical.primitive_scale1 * physical.primitive_scale2,
            adjacency=tuple(),
            metadata={"representation": "goal_maplet_resident_full_clean_scene_v1"},
        )
        quats, scales = _surface_element_quaternions_and_scales(elements)
        self.means = torch.as_tensor(
            physical.primitive_centers, dtype=torch.float32, device=self.device
        ).contiguous()
        self.quats = torch.as_tensor(
            quats, dtype=torch.float32, device=self.device
        ).contiguous()
        self.scales = torch.as_tensor(
            scales, dtype=torch.float32, device=self.device
        ).contiguous()
        self.opacities = torch.as_tensor(
            physical.primitive_opacity, dtype=torch.float32, device=self.device
        ).reshape(-1).clamp(0.0, 1.0).contiguous()
        self.normals = torch.as_tensor(
            physical.primitive_normals, dtype=torch.float32, device=self.device
        ).contiguous()
        self.double_sided = torch.as_tensor(
            np.asarray(physical.primitive_sidedness, dtype=np.uint8) == DOUBLE_SIDED,
            dtype=torch.bool, device=self.device,
        ).contiguous()
        child_owner = np.array(dominant_child_owner(physical), dtype=np.int64, copy=True)
        parent_owner = np.full(child_owner.shape, -1, dtype=np.int64)
        owned = child_owner >= 0
        parent_owner[owned] = np.asarray(
            physical.child_parent_rows, dtype=np.int64
        )[child_owner[owned]]
        field_row = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
        field_row[field.primitive_rows] = np.arange(field.primitive_rows.size, dtype=np.int64)
        self.stable_primitive_ids = torch.as_tensor(
            physical.primitive_ids, dtype=torch.int64, device=self.device
        ).contiguous()
        self.child_owner = torch.as_tensor(
            child_owner, dtype=torch.int64, device=self.device
        ).contiguous()
        self.parent_owner = torch.as_tensor(
            parent_owner, dtype=torch.int64, device=self.device
        ).contiguous()
        self.field_row_by_primitive = torch.as_tensor(
            field_row, dtype=torch.int64, device=self.device
        ).contiguous()
        codes = np.asarray(field.codes, dtype=np.float32)
        self.normalized_codes = codes / np.maximum(
            np.linalg.norm(codes, axis=1, keepdims=True), 1e-8
        )
        self.canonical_codes = torch.as_tensor(
            self.normalized_codes, dtype=torch.float32, device=self.device
        ).contiguous()
        self.field_confidence = torch.as_tensor(
            field.confidence, dtype=torch.float32, device=self.device
        ).contiguous()
        self.field_uncertainty = torch.as_tensor(
            field.uncertainty, dtype=torch.float32, device=self.device
        ).contiguous()
        self._resident_geometry_bytes = int(sum(
            value.numel() * value.element_size()
            for value in (
                self.means, self.quats, self.scales, self.opacities,
                self.normals, self.double_sided,
                self.stable_primitive_ids, self.child_owner, self.parent_owner,
                self.field_row_by_primitive, self.canonical_codes,
                self.field_confidence, self.field_uncertainty,
            )
        ))

    def _batch_ideal_hits(
        self,
        poses_w2c: np.ndarray,
        camera,
        *,
        render_width: int,
        render_height: int,
        minimum_incidence: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, float]]:
        torch = self._torch
        try:
            from gsplat.cuda._wrapper import (
                fully_fused_projection_2dgs,
                isect_offset_encode,
                isect_tiles,
                rasterize_to_indices_in_range_2dgs,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("gsplat 2DGS CUDA wrappers are required") from exc
        pose = np.asarray(poses_w2c, dtype=np.float32)
        if pose.ndim != 3 or pose.shape[1:] != (4, 4) or pose.shape[0] == 0:
            raise ValueError("poses_w2c must have shape [batch,4,4]")
        if np.any(~np.isfinite(pose)):
            raise ValueError("poses_w2c must be finite")
        batch = int(pose.shape[0])
        viewmats = torch.as_tensor(pose, dtype=torch.float32, device=self.device).contiguous()
        k = _intrinsic_matrix(camera, int(render_width), int(render_height))
        ks = torch.as_tensor(
            np.broadcast_to(k, (batch, 3, 3)).copy(),
            dtype=torch.float32, device=self.device,
        ).contiguous()
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        radii, means2d, depths, transforms, _camera_normals = fully_fused_projection_2dgs(
            self.means, self.quats, self.scales, viewmats, ks,
            int(render_width), int(render_height), packed=False,
        )
        rotation = viewmats[:, :3, :3]
        translation = viewmats[:, :3, 3]
        camera_center = -torch.bmm(rotation.transpose(1, 2), translation[..., None])[..., 0]
        view = camera_center[:, None, :] - self.means[None, :, :]
        view = view / torch.linalg.vector_norm(view, dim=2, keepdim=True).clamp_min(1e-12)
        signed = torch.sum(self.normals[None, :, :] * view, dim=2)
        incidence = torch.where(
            self.double_sided[None, :], torch.abs(signed), torch.clamp_min(signed, 0.0)
        )
        front = incidence >= float(minimum_incidence)
        radii = torch.where(front, radii, torch.zeros_like(radii)).contiguous()
        per_camera_opacity = (
            self.opacities[None, :].expand(batch, -1) * front.to(torch.float32)
        ).contiguous()
        tile_size = 16
        tile_width = math.ceil(int(render_width) / tile_size)
        tile_height = math.ceil(int(render_height) / tile_size)
        _tiles, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            packed=False, n_cameras=batch,
        )
        offsets = isect_offset_encode(isect_ids, batch, tile_width, tile_height)
        transmittance = torch.ones(
            (batch, int(render_height), int(render_width)),
            dtype=torch.float32, device=self.device,
        )
        gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range_2dgs(
            0, 1_000_000_000, transmittance, means2d, transforms,
            per_camera_opacity, int(render_width), int(render_height), tile_size,
            offsets, flatten_ids,
        )
        packed_hit_count = int(gs_ids.numel())
        if packed_hit_count == 0:
            torch.cuda.synchronize(self.device)
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32), 0,
                {
                    "projection_tile_raster_seconds": time.perf_counter() - started,
                    "device_to_host_seconds": 0.0,
                    "depth_sort_composite_seconds": 0.0,
                },
            )
        px = (pixel_ids % int(render_width)).to(torch.float32) + 0.5
        py = (pixel_ids // int(render_width)).to(torch.float32) + 0.5
        delta = torch.stack([px, py], dim=-1) - means2d[camera_ids, gs_ids]
        transform = transforms[camera_ids, gs_ids]
        h_u = -transform[..., 0, :3] + transform[..., 2, :3] * px[..., None]
        h_v = -transform[..., 1, :3] + transform[..., 2, :3] * py[..., None]
        tmp = torch.cross(h_u, h_v, dim=-1)
        denominator = tmp[..., 2]
        denominator = torch.where(
            torch.abs(denominator) < 1e-12,
            torch.where(denominator >= 0.0, 1e-12, -1e-12), denominator,
        )
        u = tmp[..., 0] / denominator
        v = tmp[..., 1] / denominator
        sigma3 = u * u + v * v
        sigma2 = 2.0 * torch.sum(delta * delta, dim=1)
        sigma = 0.5 * torch.minimum(sigma3, sigma2)
        alpha = torch.clamp(
            per_camera_opacity[camera_ids, gs_ids] * torch.exp(-sigma), max=0.999
        )
        hit_depth = depths[camera_ids, gs_ids]
        global_pixel = camera_ids.to(torch.int64) * (
            int(render_width) * int(render_height)
        ) + pixel_ids.to(torch.int64)
        torch.cuda.synchronize(self.device)
        projection_seconds = time.perf_counter() - started
        transfer_started = time.perf_counter()
        arrays = [
            global_pixel.detach().cpu().numpy(),
            hit_depth.detach().cpu().numpy(),
            gs_ids.detach().cpu().numpy(),
            alpha.detach().cpu().numpy(),
        ]
        torch.cuda.synchronize(self.device)
        transfer_seconds = time.perf_counter() - transfer_started
        global_pixel_np, depth_np, rows_np, alpha_np = arrays
        composite_started = time.perf_counter()
        order = np.lexsort((rows_np, depth_np, global_pixel_np))
        packed = np.stack([
            global_pixel_np[order].astype(np.float64),
            depth_np[order].astype(np.float64),
            rows_np[order].astype(np.float64),
            alpha_np[order].astype(np.float64),
        ], axis=1)
        out_pixel, out_row, out_weight = _composite_sorted_packed_hits(packed)
        composite_seconds = time.perf_counter() - composite_started
        return out_pixel, out_row, out_weight, packed_hit_count, {
            "projection_tile_raster_seconds": float(projection_seconds),
            "device_to_host_seconds": float(transfer_seconds),
            "depth_sort_composite_seconds": float(composite_seconds),
        }

    @staticmethod
    def _batch_token_remap(
        global_ideal_pixels: np.ndarray,
        primitive_rows: np.ndarray,
        contribution: np.ndarray,
        camera,
        *,
        batch_size: int,
        token_width: int,
        token_height: int,
        supersample_factor: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        factor = int(supersample_factor)
        render_pixels = int(token_width) * factor * int(token_height) * factor
        token_pixels = int(token_width) * int(token_height)
        warp = _raw_to_ideal_token_warp(
            camera, token_width=int(token_width), token_height=int(token_height),
            supersample_factor=factor,
        )
        sources = (
            np.arange(int(batch_size), dtype=np.int64)[:, None] * render_pixels
            + warp.source_ideal_pixel_ids[None, :]
        ).reshape(-1)
        destinations = (
            np.arange(int(batch_size), dtype=np.int64)[:, None] * token_pixels
            + warp.destination_token_pixel_ids[None, :]
        ).reshape(-1)
        hits = np.asarray(global_ideal_pixels, dtype=np.int64).reshape(-1)
        if hits.shape != np.asarray(primitive_rows).reshape(-1).shape or hits.shape != np.asarray(contribution).reshape(-1).shape:
            raise ValueError("batched token-remap hit arrays differ")
        if np.any(hits[1:] < hits[:-1]):
            raise ValueError("batched token remap requires globally sorted pixel hits")
        left = np.searchsorted(hits, sources, side="left")
        right = np.searchsorted(hits, sources, side="right")
        keep = right > left
        left, right, destinations = left[keep], right[keep], destinations[keep]
        if destinations.size == 0:
            return (
                np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
            )
        counts = right - left
        expanded_left = np.repeat(left, counts)
        group_origin = np.repeat(np.cumsum(counts) - counts, counts)
        selected = expanded_left + (
            np.arange(int(np.sum(counts)), dtype=np.int64) - group_origin
        )
        return (
            np.repeat(destinations, counts),
            np.asarray(primitive_rows, dtype=np.int64)[selected],
            (
                np.asarray(contribution, dtype=np.float32)[selected]
                / float(factor * factor)
            ).astype(np.float32),
        )

    def render_exact_batch(
        self,
        poses_w2c: np.ndarray,
        camera,
        *,
        width: int = 64,
        height: int = 36,
        selected_child_rows: np.ndarray | None = None,
        top_l: int = 4,
        coordinate_supersample_factor: int = 4,
        minimum_incidence: float = 0.05,
        minimum_feature_alpha: float = 1e-4,
        alpha_conservation_tolerance: float = 2e-5,
    ) -> ResidentSoftSurfaceBatch:
        pose = np.asarray(poses_w2c, dtype=np.float64)
        batch = int(pose.shape[0]) if pose.ndim else 0
        if batch <= 0:
            raise ValueError("exact resident batch cannot be empty")
        started = time.perf_counter()
        factor = int(coordinate_supersample_factor)
        ideal_pixel, primitive, weight, packed_count, timings = self._batch_ideal_hits(
            pose, camera, render_width=int(width) * factor,
            render_height=int(height) * factor,
            minimum_incidence=float(minimum_incidence),
        )
        gather_started = time.perf_counter()
        token, primitive, weight = self._batch_token_remap(
            ideal_pixel, primitive, weight, camera, batch_size=batch,
            token_width=int(width), token_height=int(height), supersample_factor=factor,
        )
        gather_seconds = time.perf_counter() - gather_started
        reduction_started = time.perf_counter()
        token_count = int(width) * int(height)
        rendered: list[RenderedSoftChildMixture] = []
        reduction_timings: list[dict[str, float]] = []
        for row in range(batch):
            mask = (token >= row * token_count) & (token < (row + 1) * token_count)
            row_timing: dict[str, float] = {}
            rendered.append(_reduce_soft_child_token_hits(
                self.physical, self.field,
                token_pixel_ids=token[mask] - row * token_count,
                primitive_rows=primitive[mask], contribution=weight[mask],
                width=int(width), height=int(height),
                normalized_codes=self.normalized_codes,
                selected_child_rows=selected_child_rows, top_l=int(top_l),
                minimum_feature_alpha=float(minimum_feature_alpha),
                alpha_conservation_tolerance=float(alpha_conservation_tolerance),
                timing_sink=row_timing,
            ))
            reduction_timings.append(row_timing)
        host_seconds = time.perf_counter() - reduction_started
        total_seconds = time.perf_counter() - started
        raster_seconds = (
            timings["projection_tile_raster_seconds"]
            + timings["device_to_host_seconds"]
            + timings["depth_sort_composite_seconds"]
        )
        return ResidentSoftSurfaceBatch(
            rendered=tuple(rendered),
            audit=ResidentRendererBatchAudit(
                batch_size=batch, raster_seconds=float(raster_seconds),
                projection_tile_raster_seconds=float(timings["projection_tile_raster_seconds"]),
                device_to_host_seconds=float(timings["device_to_host_seconds"]),
                depth_sort_composite_seconds=float(timings["depth_sort_composite_seconds"]),
                raw_token_gather_seconds=float(gather_seconds),
                child_identity_reduction_seconds=float(sum(
                    row["child_identity_reduction_seconds"] for row in reduction_timings
                )),
                feature_reduction_seconds=float(sum(
                    row["feature_reduction_seconds"] for row in reduction_timings
                )),
                typed_finalize_seconds=float(sum(
                    row["typed_finalize_seconds"] for row in reduction_timings
                )),
                direct_parent_reduction_seconds=float(sum(
                    row["direct_parent_reduction_seconds"] for row in reduction_timings
                )),
                child_and_feature_reduction_seconds=float(host_seconds),
                host_reduction_seconds=float(host_seconds), total_seconds=float(total_seconds),
                packed_hit_count=int(packed_count), remapped_hit_count=int(weight.size),
                resident_geometry_bytes=self._resident_geometry_bytes,
                gpu_child_reducer_implemented=False,
                production_speed_gate_passed=False,
            ),
        )
