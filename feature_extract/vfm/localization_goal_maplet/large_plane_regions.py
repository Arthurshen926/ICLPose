"""Large finite plane regions for outdoor chart/query geometry.

Unlike :mod:`query_plane_regions`, this extractor does not force a plane to be
one image connected component.  A facade interrupted by windows, vegetation,
or missing depth is represented by one finite *support union*.  The geometry
still owns an explicit pixel mask; disconnected support is never filled in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LargePlaneRegions:
    labels: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray
    pixel_counts: np.ndarray
    component_counts: np.ndarray
    residual_rms: np.ndarray
    residual_p95: np.ndarray

    def validated(self) -> "LargePlaneRegions":
        labels = np.asarray(self.labels)
        count = int(np.asarray(self.normals).shape[0])
        if labels.ndim != 2 or labels.dtype.kind not in "iu":
            raise ValueError("large-plane labels must be an integer image")
        if np.any((labels < -1) | (labels >= count)):
            raise ValueError("large-plane label is outside the plane inventory")
        for name, shape in (
            ("normals", (count, 3)),
            ("offsets", (count,)),
            ("pixel_counts", (count,)),
            ("component_counts", (count,)),
            ("residual_rms", (count,)),
            ("residual_p95", (count,)),
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid large-plane {name}")
        if count and np.max(np.abs(np.linalg.norm(self.normals, axis=1) - 1.0)) > 1e-8:
            raise ValueError("large-plane normals are not unit length")
        if np.any(self.pixel_counts <= 0) or np.any(self.component_counts <= 0):
            raise ValueError("large-plane support must be non-empty")
        for row in range(count):
            if int(np.sum(labels == row)) != int(self.pixel_counts[row]):
                raise ValueError("large-plane count differs from its label mask")
        return self


def _component_count(mask: np.ndarray) -> int:
    # Small dependency-free 8-connected component counter.  It is diagnostic
    # only: components remain a support union and are never hole-filled.
    mask = np.asarray(mask, bool)
    visited = np.zeros(mask.shape, bool)
    count = 0
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        if visited[y, x]:
            continue
        count += 1
        visited[y, x] = True
        stack = [(int(y), int(x))]
        while stack:
            cy, cx = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return count


def _fit(points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    center = np.mean(points, axis=0)
    covariance = (points - center).T @ (points - center) / max(len(points), 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    reference = np.mean(normals, axis=0)
    if float(normal @ reference) < 0.0:
        normal = -normal
    normal /= max(float(np.linalg.norm(normal)), 1e-15)
    offset = float(normal @ center)
    # Canonical sign makes cross-view plane comparisons deterministic.
    pivot = int(np.argmax(np.abs(normal)))
    if normal[pivot] < 0.0:
        normal = -normal
        offset = -offset
    residual = np.abs(points @ normal - offset)
    return normal, offset, residual


def extract_large_plane_regions(
    points: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
    *,
    minimum_pixels: int = 160,
    maximum_planes: int = 16,
    maximum_hypotheses: int = 128,
    distance_ratio: float = 0.005,
    normal_degrees: float = 25.0,
) -> LargePlaneRegions:
    """Extract deterministic large plane support unions.

    RANSAC proposals are scored on *all* remaining pixels rather than one
    connected component.  Plane support may therefore contain several image
    components, while every retained pixel must independently pass the metric
    point-to-plane and normal tests.
    """
    points = np.asarray(points, np.float64)
    normals = np.asarray(normals, np.float64)
    valid = np.asarray(valid, bool)
    if points.ndim != 3 or points.shape[-1] != 3 or normals.shape != points.shape or valid.shape != points.shape[:2]:
        raise ValueError("large-plane inputs have incompatible shapes")
    valid = valid & np.isfinite(points).all(2) & np.isfinite(normals).all(2)
    length = np.linalg.norm(normals, axis=2)
    valid &= length > 1e-8
    normals = normals / np.maximum(length[..., None], 1e-15)
    if not valid.any():
        return LargePlaneRegions(
            np.full(valid.shape, -1, np.int32), np.zeros((0, 3)), np.zeros((0,)),
            np.zeros((0,), np.int64), np.zeros((0,), np.int64),
            np.zeros((0,)), np.zeros((0,)),
        ).validated()
    depth = np.linalg.norm(points[valid], axis=1)
    threshold = float(np.clip(distance_ratio * np.median(depth), 0.03, 0.20))
    cosine = float(np.cos(np.deg2rad(normal_degrees)))
    flat_points = points.reshape(-1, 3)
    flat_normals = normals.reshape(-1, 3)
    remaining = valid.copy()
    labels = np.full(valid.shape, -1, np.int32)
    output: list[tuple[np.ndarray, float, int, int, float, float]] = []
    for _ in range(maximum_planes):
        candidate = np.flatnonzero(remaining.ravel())
        if candidate.size < minimum_pixels:
            break
        seed_rows = candidate[np.linspace(
            0, candidate.size - 1, min(maximum_hypotheses, candidate.size), dtype=np.int64,
        )]
        best_mask = None
        best_seed = None
        for seed in seed_rows:
            normal = flat_normals[seed]
            offset = float(normal @ flat_points[seed])
            mask = remaining & (
                np.abs((flat_points @ normal - offset).reshape(valid.shape)) <= threshold
            ) & (
                np.abs(flat_normals @ normal).reshape(valid.shape) >= cosine
            )
            if best_mask is None or int(mask.sum()) > int(best_mask.sum()):
                best_mask, best_seed = mask, int(seed)
        if best_mask is None or int(best_mask.sum()) < minimum_pixels:
            break
        mask = best_mask
        for _ in range(4):
            normal, offset, _ = _fit(points[mask], normals[mask])
            updated = remaining & (
                np.abs((flat_points @ normal - offset).reshape(valid.shape)) <= threshold
            ) & (
                np.abs(flat_normals @ normal).reshape(valid.shape) >= cosine
            )
            if np.array_equal(mask, updated):
                break
            mask = updated
        if int(mask.sum()) < minimum_pixels:
            remaining.ravel()[best_seed] = False
            continue
        normal, offset, residual = _fit(points[mask], normals[mask])
        labels[mask] = len(output)
        output.append((
            normal, offset, int(mask.sum()), _component_count(mask),
            float(np.sqrt(np.mean(residual ** 2))), float(np.quantile(residual, 0.95)),
        ))
        remaining[mask] = False
    return LargePlaneRegions(
        labels=labels,
        normals=np.asarray([row[0] for row in output], np.float64).reshape(-1, 3),
        offsets=np.asarray([row[1] for row in output], np.float64),
        pixel_counts=np.asarray([row[2] for row in output], np.int64),
        component_counts=np.asarray([row[3] for row in output], np.int64),
        residual_rms=np.asarray([row[4] for row in output], np.float64),
        residual_p95=np.asarray([row[5] for row in output], np.float64),
    ).validated()


__all__ = ["LargePlaneRegions", "extract_large_plane_regions"]
