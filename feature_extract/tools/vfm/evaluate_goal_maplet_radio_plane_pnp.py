"""PlanarReloc-style RADIO plane retrieval, within-plane matching, and PnP.

MoGe3 is used only to segment query planes.  Query depth and its scale are not
used by the pose solver.  Mapping RADIO tokens are lifted to world points by
the frozen 2DGS contributor depth and mapping camera, then query 2D pixels and
map 3D points are passed to PnP-RANSAC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import (
    PlaneVisibilityAtlas,
    normalized_token_counts,
)
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
)
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    inverse_simple_radial,
)


TOPK_PLANES = 5
TOP_SOURCE_VIEWS = 4
MINIMUM_PLANE_TOKEN_PIXELS = 8
MINIMUM_TOKEN_COSINE = 0.40
MINIMUM_TOKEN_MARGIN = 0.02
MINIMUM_HOMOGRAPHY_MATCHES = 6
MINIMUM_PNP_POINTS = 6
SOURCE_SUBTOKEN_DEPTH_TOLERANCE_ABS_M = 0.5
SOURCE_SUBTOKEN_DEPTH_TOLERANCE_REL = 0.05
PLANE_BALANCE_QUADRATIC_MASS_CAP = 8


def _records(paths: list[Path] | tuple[Path, ...]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in paths:
        rows = json.loads(path.read_text()).get("records", [])
        if not rows:
            raise ValueError("RADIO manifest is empty")
        for row in rows:
            image_id = str(row["image_id"])
            if image_id in result:
                raise ValueError("RADIO manifests contain duplicate image IDs")
            result[image_id] = row
    return result


def _image_id(name: str) -> str:
    value = str(name)
    if value.endswith(".npz"):
        value = value[:-4]
    return value.replace("__", "/", 1)


def _radio(name: str, records: dict[str, dict[str, object]]) -> np.ndarray:
    image_id = _image_id(name)
    if image_id not in records:
        raise ValueError(f"RADIO manifest lacks {image_id}")
    row = records[image_id]
    path = Path(str(row["token_path"]))
    if file_sha256(path) != row.get("checksum"):
        raise ValueError("RADIO token bytes differ")
    with np.load(path, allow_pickle=False) as data:
        feature = np.asarray(data["radio_final"], np.float32)
    if feature.ndim != 3 or feature.shape[0] != 1280:
        raise ValueError("RADIO feature grid differs")
    feature = feature.reshape(1280, -1).T
    return feature / np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-8)


def _scaled_intrinsics(
    model_id: int,
    params: np.ndarray,
    source_width: int,
    source_height: int,
    width: int = 256,
    height: int = 144,
) -> tuple[np.ndarray, float]:
    value = np.asarray(params, np.float64).reshape(-1)
    if int(model_id) == 0:  # SIMPLE_PINHOLE
        fx = fy = value[0]; cx, cy = value[1:3]; k1 = 0.0
    elif int(model_id) == 1:  # PINHOLE
        fx, fy, cx, cy = value[:4]; k1 = 0.0
    elif int(model_id) == 2:  # SIMPLE_RADIAL
        fx = fy = value[0]; cx, cy, k1 = value[1:4]
    else:
        raise ValueError("unsupported PnP camera model")
    sx, sy = width / float(source_width), height / float(source_height)
    # OpenCV integer pixel coordinates refer to pixel centers.  This is the
    # exact resize phase for an area-resized image.
    scaled = np.asarray([
        [fx * sx, 0.0, (cx + 0.5) * sx - 0.5],
        [0.0, fy * sy, (cy + 0.5) * sy - 0.5],
        [0.0, 0.0, 1.0],
    ])
    return scaled, float(k1)


def _token_world_points(
    contributor: Path,
    token_ids: np.ndarray,
    token_grid: tuple[int, int] = (36, 64),
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(contributor, allow_pickle=False) as data:
        depth = np.asarray(data["dominant_depth"], np.float64)
        pose = np.asarray(data["pose_w2c"], np.float64)
        model_id = int(data["camera_model_id"])
        source_width = int(data["camera_width"])
        source_height = int(data["camera_height"])
        params = np.asarray(data["camera_params"], np.float64)
    if depth.shape != (144, 256):
        raise ValueError("mapping depth grid differs")
    K, k1 = _scaled_intrinsics(model_id, params, source_width, source_height)
    rows = []
    keep = []
    rotation, translation = pose[:3, :3], pose[:3, 3]
    center = -rotation.T @ translation
    token_height, token_width = map(int, token_grid)
    pixel_y, pixel_x = np.indices(depth.shape)
    owner_y = np.minimum(((pixel_y + 0.5) * token_height / depth.shape[0]).astype(np.int64), token_height - 1)
    owner_x = np.minimum(((pixel_x + 0.5) * token_width / depth.shape[1]).astype(np.int64), token_width - 1)
    for token in np.asarray(token_ids, np.int64).tolist():
        ty, tx = divmod(int(token), token_width)
        yy, xx = np.nonzero((owner_y == ty) & (owner_x == tx) & np.isfinite(depth) & (depth > 0.0))
        if not len(xx):
            continue
        z = depth[yy, xx]
        pixel = np.stack((xx, yy), axis=1).astype(np.float64)
        distorted = np.stack(
            ((pixel[:, 0] - K[0, 2]) / K[0, 0], (pixel[:, 1] - K[1, 2]) / K[1, 1]),
            axis=1,
        )
        ideal = inverse_simple_radial(distorted, k1)
        camera = np.c_[ideal * z[:, None], z]
        world = camera @ rotation + center
        rows.append(np.median(world, axis=0))
        keep.append(token)
    return np.asarray(rows, np.float64).reshape(-1, 3), np.asarray(keep, np.int64)


def _region_tokens(labels: np.ndarray, region: int, token_grid: tuple[int, int] = (36, 64)) -> np.ndarray:
    return _region_token_support(labels, region, token_grid=token_grid)[0]


def _region_token_support(
    labels: np.ndarray,
    region: int,
    token_grid: tuple[int, int] = (36, 64),
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed tokens and their real visible-pixel fractions.

    Carrier gaps stay label ``-1`` and therefore have exactly zero weight.  A
    downstream descriptor can downweight partially occluded boundary tokens
    without ever assigning evidence to an inferred pixel.
    """
    mask = np.asarray(labels == int(region), np.uint8)
    count = normalized_token_counts(mask, token_grid).reshape(-1)
    token = np.flatnonzero(count >= MINIMUM_PLANE_TOKEN_PIXELS)
    return token, np.asarray(count[token], np.float32) / 16.0


def _mutual_matches(query: np.ndarray, mapping: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(query) == 0 or len(mapping) < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    cosine = query @ mapping.T
    top2 = np.argpartition(cosine, -2, axis=1)[:, -2:]
    value = np.take_along_axis(cosine, top2, axis=1)
    order = np.argsort(value, axis=1)
    best = top2[np.arange(len(query)), order[:, 1]]
    score = cosine[np.arange(len(query)), best]
    second = value[np.arange(len(query)), order[:, 0]]
    reverse = np.argmax(cosine, axis=0)
    keep = (
        (reverse[best] == np.arange(len(query)))
        & (score >= MINIMUM_TOKEN_COSINE)
        & ((score - second) >= MINIMUM_TOKEN_MARGIN)
    )
    return np.flatnonzero(keep), best[keep], score[keep]


def _homography_filter(
    query_token: np.ndarray,
    map_token: np.ndarray,
    token_width: int = 64,
) -> np.ndarray:
    homography, mask, _, _ = _fit_token_homography(
        query_token, map_token, token_width=token_width,
    )
    return np.zeros(len(query_token), bool) if homography is None else mask


def _fit_token_homography(
    query_token: np.ndarray,
    map_token: np.ndarray,
    token_width: int,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    query_xy = np.c_[query_token % token_width, query_token // token_width].astype(np.float64)
    map_xy = np.c_[map_token % token_width, map_token // token_width].astype(np.float64)
    if len(query_token) < MINIMUM_HOMOGRAPHY_MATCHES:
        return None, np.zeros(len(query_token), bool), query_xy, map_xy
    cv2.setRNGSeed(260831)
    homography, mask = cv2.findHomography(
        query_xy, map_xy, cv2.RANSAC, 2.0, maxIters=2000, confidence=0.995,
    )
    if homography is None or mask is None:
        return None, np.zeros(len(query_token), bool), query_xy, map_xy
    return np.asarray(homography, np.float64), mask.reshape(-1).astype(bool), query_xy, map_xy


def _homography_filter_with_query_projection(
    query_token: np.ndarray,
    map_token: np.ndarray,
    token_width: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RANSAC inliers and map-token centers projected into query-token space."""
    keep, projected_query, _ = _homography_filter_with_bidirectional_projections(
        query_token, map_token, token_width=token_width,
    )
    return keep & np.all(np.isfinite(projected_query), axis=1), projected_query


def _homography_filter_with_bidirectional_projections(
    query_token: np.ndarray,
    map_token: np.ndarray,
    token_width: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inliers plus continuous query and mapping token coordinates.

    The same frozen RANSAC homography supplies both directions.  The forward
    projection is useful for recovering a sub-token source point from the
    mapping-only 2DGS depth without changing the query measurement.
    """
    homography, keep, query_xy, map_xy = _fit_token_homography(
        query_token, map_token, token_width=token_width,
    )
    if homography is None:
        invalid = np.full((len(query_token), 2), np.nan)
        return np.zeros(len(query_token), bool), invalid.copy(), invalid
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        projected_query = np.full((len(query_token), 2), np.nan)
    else:
        projected_query = cv2.perspectiveTransform(
            map_xy.reshape(-1, 1, 2).astype(np.float64), inverse,
        ).reshape(-1, 2)
    projected_map = cv2.perspectiveTransform(
        query_xy.reshape(-1, 1, 2).astype(np.float64),
        homography,
    ).reshape(-1, 2)
    return keep, projected_query, projected_map


def _token_pixels(query_token: np.ndarray, token_grid: tuple[int, int]) -> np.ndarray:
    token_height, token_width = map(int, token_grid)
    token = np.asarray(query_token, np.int64)
    return np.c_[
        (token % token_width + 0.5) * 256.0 / token_width - 0.5,
        (token // token_width + 0.5) * 144.0 / token_height - 0.5,
    ].astype(np.float64)


def _homography_source_world_points(
    depth: np.ndarray,
    pose_w2c: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    projected_map_xy: np.ndarray,
    reference_world_points: np.ndarray,
    token_grid: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Lift continuous mapping-token positions through local 2DGS depth.

    Depth interpolation is restricted to the four adjacent pixels whose depth
    agrees with the original token-median surface.  This reduces tangential
    token quantization while failing closed at depth discontinuities.
    """
    depth = np.asarray(depth, np.float64)
    pose = np.asarray(pose_w2c, np.float64)
    K = np.asarray(camera_matrix, np.float64)
    map_xy = np.asarray(projected_map_xy, np.float64)
    reference = np.asarray(reference_world_points, np.float64)
    if depth.ndim != 2 or pose.shape != (4, 4) or K.shape != (3, 3):
        raise ValueError("mapping sub-token geometry arrays differ")
    if map_xy.ndim != 2 or map_xy.shape[1] != 2 or reference.shape != (len(map_xy), 3):
        raise ValueError("mapping sub-token rows differ")
    token_height, token_width = map(int, token_grid)
    pixel = np.c_[
        (map_xy[:, 0] + 0.5) * depth.shape[1] / token_width - 0.5,
        (map_xy[:, 1] + 0.5) * depth.shape[0] / token_height - 0.5,
    ]
    rotation, translation = pose[:3, :3], pose[:3, 3]
    reference_depth = (reference @ rotation.T + translation)[:, 2]
    output = reference.copy()
    used = np.zeros(len(reference), bool)
    for row, ((x, y), z_reference) in enumerate(zip(pixel, reference_depth)):
        if not np.isfinite(x + y + z_reference) or z_reference <= 0.0:
            continue
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        candidates: list[tuple[float, float, float]] = []
        tolerance = max(
            SOURCE_SUBTOKEN_DEPTH_TOLERANCE_ABS_M,
            SOURCE_SUBTOKEN_DEPTH_TOLERANCE_REL * float(z_reference),
        )
        for yy in (y0, y0 + 1):
            for xx in (x0, x0 + 1):
                if not (0 <= yy < depth.shape[0] and 0 <= xx < depth.shape[1]):
                    continue
                value = float(depth[yy, xx])
                if not np.isfinite(value) or value <= 0.0 or abs(value - z_reference) > tolerance:
                    continue
                weight = max(0.0, 1.0 - abs(x - xx)) * max(0.0, 1.0 - abs(y - yy))
                if weight > 0.0:
                    candidates.append((weight, value, abs(value - z_reference)))
        if not candidates:
            continue
        weight = np.asarray([item[0] for item in candidates], np.float64)
        values = np.asarray([item[1] for item in candidates], np.float64)
        z = float(np.sum(weight * values) / np.sum(weight))
        distorted = np.asarray([[(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1]]])
        ideal = inverse_simple_radial(distorted, float(radial_k1))[0]
        camera = np.asarray([ideal[0] * z, ideal[1] * z, z], np.float64)
        output[row] = (camera - translation) @ rotation
        used[row] = True
    return output, used


def _homography_source_plane_points(
    pose_w2c: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    projected_map_xy: np.ndarray,
    token_grid: tuple[int, int],
    plane_normal_world: np.ndarray,
    plane_offset_world: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect continuous source-image rays with one retrieved finite plane.

    Unlike the legacy token-median lift, this measurement never mixes depth
    pixels from an occluder or an adjacent surface into the retrieved plane.
    The returned points are temporary samples of the continuous plane; source
    token IDs remain appearance observations rather than persistent 3D
    landmarks.  Finite support is supplied upstream by the exact visibility
    atlas, while invalid/behind-camera/near-parallel intersections fail closed.
    """

    pose = np.asarray(pose_w2c, np.float64)
    K = np.asarray(camera_matrix, np.float64)
    map_xy = np.asarray(projected_map_xy, np.float64)
    normal = np.asarray(plane_normal_world, np.float64).reshape(3)
    if pose.shape != (4, 4) or K.shape != (3, 3):
        raise ValueError("mapping plane-intersection camera arrays differ")
    if map_xy.ndim != 2 or map_xy.shape[1] != 2:
        raise ValueError("mapping plane-intersection coordinates must be Nx2")
    normal_norm = float(np.linalg.norm(normal))
    if (
        not np.all(np.isfinite(pose))
        or not np.all(np.isfinite(K))
        or abs(float(K[0, 0])) <= 0.0
        or abs(float(K[1, 1])) <= 0.0
        or not np.isfinite(normal_norm)
        or normal_norm <= 0.0
        or not np.isfinite(plane_offset_world)
    ):
        raise ValueError("mapping plane-intersection authority is invalid")
    normal = normal / normal_norm
    offset = float(plane_offset_world) / normal_norm
    token_height, token_width = map(int, token_grid)
    finite_coordinate = np.all(np.isfinite(map_xy), axis=1)
    pixel = np.full((len(map_xy), 2), np.nan, np.float64)
    pixel[finite_coordinate] = np.c_[
        (map_xy[finite_coordinate, 0] + 0.5) * 256.0 / token_width - 0.5,
        (map_xy[finite_coordinate, 1] + 0.5) * 144.0 / token_height - 0.5,
    ]
    distorted = np.full((len(map_xy), 2), np.nan, np.float64)
    distorted[finite_coordinate] = np.c_[
        (pixel[finite_coordinate, 0] - K[0, 2]) / K[0, 0],
        (pixel[finite_coordinate, 1] - K[1, 2]) / K[1, 1],
    ]
    ideal = np.full_like(distorted, np.nan)
    ideal[finite_coordinate] = inverse_simple_radial(
        distorted[finite_coordinate], float(radial_k1)
    )
    ray_camera = np.c_[ideal, np.ones(len(ideal), np.float64)]
    rotation, translation = pose[:3, :3], pose[:3, 3]
    center_world = -rotation.T @ translation
    ray_world = ray_camera @ rotation
    denominator = ray_world @ normal
    numerator = offset - float(center_world @ normal)
    depth = np.full(len(map_xy), np.nan, np.float64)
    nonparallel = finite_coordinate & (np.abs(denominator) > 1e-10)
    depth[nonparallel] = numerator / denominator[nonparallel]
    valid = nonparallel & np.isfinite(depth) & (depth > 0.0)
    output = np.full((len(map_xy), 3), np.nan, np.float64)
    output[valid] = center_world + depth[valid, None] * ray_world[valid]
    valid &= np.all(np.isfinite(output), axis=1)
    # Recheck the actual plane equation after all coordinate conversions.
    if np.any(valid):
        residual = np.abs(output[valid] @ normal - offset)
        if float(np.max(residual)) > 1e-7:
            raise ValueError("mapping ray-plane intersection does not replay")
    return output, valid


def _project_points_to_plane(
    points: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    value = np.asarray(points, np.float64)
    unit = np.asarray(normal, np.float64).reshape(3)
    norm = float(np.linalg.norm(unit))
    if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
        raise ValueError("plane projection points must be finite Nx3")
    if not np.isfinite(norm) or norm <= 0.0 or not np.isfinite(offset):
        raise ValueError("plane projection authority is invalid")
    unit = unit / norm
    signed = value @ unit - float(offset) / norm
    return value - signed[:, None] * unit[None, :]


def _choose_token_hypotheses(
    query_tokens: np.ndarray,
    world_points: np.ndarray,
    match_scores: np.ndarray,
    provenance: np.ndarray,
    atlas_view_names: np.ndarray,
    *,
    maximum_per_token: int,
    selection: str,
    consensus_radius_m: float,
) -> np.ndarray:
    """Choose distinct 3D hypotheses without pose/GT access."""
    tokens = np.asarray(query_tokens, np.int64)
    points = np.asarray(world_points, np.float64)
    scores = np.asarray(match_scores, np.float64)
    rows = np.asarray(provenance, np.int64).reshape(-1, 3)
    names = np.asarray(atlas_view_names).astype(str)
    chosen: list[int] = []
    for token_id in np.unique(tokens):
        candidates = np.flatnonzero(tokens == token_id)
        if selection == "score":
            order = candidates[np.argsort(-scores[candidates], kind="stable")]
        elif selection == "multiview_consensus":
            support = []
            for index in candidates.tolist():
                same_plane = rows[candidates, 1] == rows[index, 1]
                nearby = np.linalg.norm(points[candidates] - points[index], axis=1) <= float(
                    consensus_radius_m
                )
                atlas_rows = rows[candidates[same_plane & nearby], 2]
                support.append(len(set(names[atlas_rows].tolist())))
            support = np.asarray(support, np.int64)
            local = np.lexsort((candidates, -scores[candidates], -support))
            order = candidates[local]
        else:
            raise ValueError("unsupported token hypothesis selection")
        selected_for_token: list[int] = []
        for index in order.tolist():
            # Multiple mapping views supporting the same physical point form
            # one hypothesis, not separate votes in downstream PnP.
            redundant = any(
                rows[index, 1] == rows[prior, 1]
                and np.linalg.norm(points[index] - points[prior]) <= float(consensus_radius_m)
                for prior in selected_for_token
            )
            if redundant:
                continue
            selected_for_token.append(index)
            if len(selected_for_token) == int(maximum_per_token):
                break
        chosen.extend(selected_for_token)
    return np.asarray(chosen, np.int64)


def _pnp(
    world: np.ndarray,
    query_token: np.ndarray,
    K: np.ndarray,
    k1: float,
    token_grid: tuple[int, int] = (36, 64),
    query_pixel: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    if len(world) < MINIMUM_PNP_POINTS:
        return None, np.zeros(0, np.int64)
    pixel = (
        _token_pixels(query_token, token_grid)
        if query_pixel is None else np.asarray(query_pixel, np.float64)
    )
    if pixel.shape != (len(world), 2) or not np.all(np.isfinite(pixel)):
        raise ValueError("query pixel coordinates must be finite Nx2")
    distortion = np.asarray([k1, 0.0, 0.0, 0.0, 0.0], np.float64)
    cv2.setRNGSeed(260831)
    ok, rvec, tvec, inlier = cv2.solvePnPRansac(
        world.astype(np.float64), pixel.astype(np.float64), K, distortion,
        iterationsCount=4000, reprojectionError=4.0, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inlier is None or len(inlier) < MINIMUM_PNP_POINTS:
        return None, np.zeros(0, np.int64)
    selected = inlier.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(world[selected], pixel[selected], K, distortion, rvec, tvec)
    rotation = cv2.Rodrigues(rvec)[0]
    pose = np.eye(4, dtype=np.float64); pose[:3, :3] = rotation; pose[:3, 3] = tvec.reshape(3)
    return pose, selected


def _plane_balance_weights(
    plane_ids: np.ndarray,
    *,
    quadratic_mass_cap: int = PLANE_BALANCE_QUADRATIC_MASS_CAP,
) -> np.ndarray:
    """Bound the quadratic leverage of any one retrieved finite plane.

    Sparse planes are never amplified.  A dense plane keeps at most ``cap``
    units of squared weight, after which all weights are RMS-normalized.  This
    prevents one facade patch with many repeated token observations from
    drowning the independent orientation/translation evidence of other planes.
    """

    values = np.asarray(plane_ids, np.int64).reshape(-1)
    if int(quadratic_mass_cap) < 1:
        raise ValueError("plane balance mass cap must be positive")
    if not len(values):
        return np.ones(0, np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    weight = np.sqrt(
        np.minimum(counts[inverse], int(quadratic_mass_cap))
        / counts[inverse].astype(np.float64)
    )
    return weight / max(float(np.sqrt(np.mean(weight * weight))), 1e-12)


def _plane_balanced_surface_refine(
    pose_w2c: np.ndarray,
    inlier_rows: np.ndarray,
    world_points: np.ndarray,
    query_pixels: np.ndarray,
    plane_ids: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly refine a PnP seed as balanced finite-surface reprojection.

    The 3D inputs are temporary samples on retrieved plane surfaces.  The
    optimizer keeps the PnP data association frozen, caps each plane's mass,
    and uses a Huber loss.  It cannot manufacture support for a missing plane
    and falls back to the seed if the fixed, label-free inlier support drops.
    """

    pose = np.asarray(pose_w2c, np.float64)
    selected = np.asarray(inlier_rows, np.int64).reshape(-1)
    world = np.asarray(world_points, np.float64)
    pixel = np.asarray(query_pixels, np.float64)
    planes = np.asarray(plane_ids, np.int64).reshape(-1)
    K = np.asarray(camera_matrix, np.float64)
    if (
        pose.shape != (4, 4)
        or world.ndim != 2 or world.shape[1] != 3
        or pixel.shape != (len(world), 2)
        or planes.shape != (len(world),)
        or np.any(selected < 0) or np.any(selected >= len(world))
    ):
        raise ValueError("plane-balanced surface refinement arrays differ")
    if len(selected) < MINIMUM_PNP_POINTS or len(np.unique(planes[selected])) < 2:
        return pose.copy(), selected.copy()
    weights = _plane_balance_weights(planes[selected])
    distortion = np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64)
    initial = np.r_[cv2.Rodrigues(pose[:3, :3])[0].reshape(3), pose[:3, 3]]

    def residual(parameter: np.ndarray) -> np.ndarray:
        projected, _ = cv2.projectPoints(
            world[selected], parameter[:3], parameter[3:], K, distortion,
        )
        return ((projected.reshape(-1, 2) - pixel[selected]) * weights[:, None]).reshape(-1)

    solution = least_squares(
        residual,
        initial,
        method="trf",
        loss="huber",
        f_scale=2.0,
        max_nfev=100,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not solution.success or not np.all(np.isfinite(solution.x)):
        return pose.copy(), selected.copy()
    initial_projected, _ = cv2.projectPoints(
        world[selected], initial[:3], initial[3:], K, distortion,
    )
    refined_projected, _ = cv2.projectPoints(
        world[selected], solution.x[:3], solution.x[3:], K, distortion,
    )
    initial_median = float(np.median(np.linalg.norm(
        initial_projected.reshape(-1, 2) - pixel[selected], axis=1,
    )))
    refined_median = float(np.median(np.linalg.norm(
        refined_projected.reshape(-1, 2) - pixel[selected], axis=1,
    )))
    # The balanced objective is a regularizer, not permission to degrade the
    # ordinary geometric fit.  This monotonic, label-free acceptance test is
    # what makes the refinement safe as a drop-in backend.
    if refined_median > initial_median + 1e-12:
        return pose.copy(), selected.copy()
    refined = np.eye(4, dtype=np.float64)
    refined[:3, :3] = cv2.Rodrigues(solution.x[:3])[0]
    refined[:3, 3] = solution.x[3:]
    camera = world @ refined[:3, :3].T + refined[:3, 3]
    projected, _ = cv2.projectPoints(
        world, solution.x[:3], solution.x[3:], K, distortion,
    )
    error = np.linalg.norm(projected.reshape(-1, 2) - pixel, axis=1)
    final = np.flatnonzero((camera[:, 2] > 0.0) & np.isfinite(error) & (error <= 4.0))
    # Refinement is not allowed to trade away substantial support merely to
    # lower a robust objective on a small subset.
    if len(final) < max(MINIMUM_PNP_POINTS, int(np.floor(0.9 * len(selected)))):
        return pose.copy(), selected.copy()
    return refined, final


def _pose_diagnostics(
    pose: np.ndarray | None,
    inlier: np.ndarray,
    world: np.ndarray,
    query_token: np.ndarray,
    matched_rows: list[tuple[int, int, int]],
    K: np.ndarray,
    k1: float,
    token_grid: tuple[int, int] = (36, 64),
    atlas_view_names: np.ndarray | None = None,
    query_pixel: np.ndarray | None = None,
) -> dict[str, object]:
    """Pose-free confidence signals computed before any GT member is opened."""
    if pose is None or len(inlier) == 0:
        return {
            "inlier_query_hull_fraction": 0.0,
            "inlier_query_bbox_fraction": 0.0,
            "inlier_query_covariance_eigenvalues": [0.0, 0.0],
            "inlier_world_spread_eigenvalues_m": [0.0, 0.0, 0.0],
            "inlier_reprojection_median_px": None,
            "inlier_reprojection_p90_px": None,
            "inlier_region_count": 0,
            "inlier_plane_count": 0,
            "inlier_source_view_count": 0,
        }
    selected = np.asarray(inlier, np.int64)
    if query_pixel is None:
        pixel = _token_pixels(np.asarray(query_token, np.int64)[selected], token_grid)
    else:
        all_pixel = np.asarray(query_pixel, np.float64)
        if all_pixel.shape != (len(world), 2) or not np.all(np.isfinite(all_pixel)):
            raise ValueError("query pixel coordinates must be finite Nx2")
        pixel = all_pixel[selected]
    hull = cv2.convexHull(pixel.astype(np.float32))
    hull_fraction = float(cv2.contourArea(hull) / (256.0 * 144.0)) if len(hull) >= 3 else 0.0
    extent = np.ptp(pixel, axis=0) if len(pixel) else np.zeros(2)
    bbox_fraction = float(np.prod(extent) / (256.0 * 144.0))
    normalized = pixel / np.asarray([256.0, 144.0])
    covariance = np.cov(normalized, rowvar=False, bias=True) if len(normalized) >= 2 else np.zeros((2, 2))
    image_eigen = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    selected_world = np.asarray(world, np.float64)[selected]
    world_covariance = (
        np.cov(selected_world, rowvar=False, bias=True)
        if len(selected_world) >= 2 else np.zeros((3, 3))
    )
    world_eigen = np.sqrt(np.maximum(np.linalg.eigvalsh(world_covariance), 0.0))
    rotation = np.asarray(pose, np.float64)[:3, :3]
    translation = np.asarray(pose, np.float64)[:3, 3]
    projected, _ = cv2.projectPoints(
        selected_world,
        cv2.Rodrigues(rotation)[0],
        translation,
        K,
        np.asarray([k1, 0.0, 0.0, 0.0, 0.0], np.float64),
    )
    residual = np.linalg.norm(projected.reshape(-1, 2) - pixel, axis=1)
    provenance = [matched_rows[int(index)] for index in selected.tolist()]
    source_view_identity = (
        [str(np.asarray(atlas_view_names)[row[2]]) for row in provenance]
        if atlas_view_names is not None else [str(row[2]) for row in provenance]
    )
    return {
        "inlier_query_hull_fraction": hull_fraction,
        "inlier_query_bbox_fraction": bbox_fraction,
        "inlier_query_covariance_eigenvalues": image_eigen.tolist(),
        "inlier_world_spread_eigenvalues_m": world_eigen.tolist(),
        "inlier_reprojection_median_px": float(np.median(residual)),
        "inlier_reprojection_p90_px": float(np.quantile(residual, 0.9)),
        "inlier_region_count": int(len(set(row[0] for row in provenance))),
        "inlier_plane_count": int(len(set(row[1] for row in provenance))),
        "inlier_source_view_count": int(len(set(source_view_identity))),
    }


def _camera_inventory(path: Path) -> tuple[dict[str, tuple[int, int, int, np.ndarray]], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {name: np.asarray(data[name]) for name in (
            "names", "camera_model_id", "camera_width", "camera_height",
            "camera_params", "source_contributor_file_sha256",
        )}
    if (
        metadata.get("artifact_type") != "goal_maplet_query_camera_only_inventory_v1"
        or metadata.get("pose_or_ground_truth_member_read") is not False
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
    ):
        raise ValueError("query camera-only inventory contract differs")
    rows = {
        str(name): (int(model), int(width), int(height), np.asarray(params, np.float64))
        for name, model, width, height, params in zip(
            arrays["names"], arrays["camera_model_id"], arrays["camera_width"],
            arrays["camera_height"], arrays["camera_params"],
        )
    }
    if len(rows) != len(arrays["names"]):
        raise ValueError("query camera inventory names are duplicated")
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--query_plane_dir", type=Path, required=True)
    parser.add_argument("--sparse_occlusion_carrier_dir", type=Path)
    parser.add_argument("--correspondence_report", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--contributors", type=Path)
    parser.add_argument("--mapping_contributors", type=Path)
    parser.add_argument("--query_contributors", type=Path)
    parser.add_argument("--source_observation_bank", type=Path)
    parser.add_argument("--planar_map", type=Path)
    parser.add_argument(
        "--source_point_geometry",
        choices=("rendered_depth", "plane_projected"),
        default="rendered_depth",
    )
    parser.add_argument(
        "--source_subtoken_geometry",
        choices=("token_median", "homography_depth", "homography_plane"),
        default="token_median",
    )
    parser.add_argument("--mapping_depth_contributors", type=Path)
    parser.add_argument("--mapping_camera_contributors", type=Path)
    parser.add_argument("--official_test_map_disjoint", action="store_true")
    parser.add_argument("--query_camera_inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output_frozen_pose_inventory", type=Path)
    parser.add_argument("--output_frozen_correspondence_inventory", type=Path)
    parser.add_argument("--topk_planes", type=int, default=TOPK_PLANES)
    parser.add_argument("--top_source_views", type=int, default=TOP_SOURCE_VIEWS)
    parser.add_argument("--hypotheses_per_query_token", type=int, default=1)
    parser.add_argument(
        "--hypothesis_selection", choices=("score", "multiview_consensus"), default="score"
    )
    parser.add_argument("--hypothesis_consensus_radius_m", type=float, default=0.5)
    parser.add_argument(
        "--query_pixel_mode",
        choices=("token_center", "homography_projection"),
        default="token_center",
    )
    parser.add_argument(
        "--pose_solver",
        choices=("standard_pnp", "plane_balanced_surface"),
        default="standard_pnp",
    )
    parser.add_argument("--homography_projection_fraction", type=float, default=1.0)
    args = parser.parse_args()
    if int(args.top_source_views) < 1:
        raise ValueError("top source views must be positive")
    if int(args.hypotheses_per_query_token) < 1:
        raise ValueError("hypotheses per query token must be positive")
    if not 0.0 <= float(args.homography_projection_fraction) <= 1.0:
        raise ValueError("homography projection fraction must lie in [0,1]")
    homography_projection_fraction = (
        float(args.homography_projection_fraction)
        if args.query_pixel_mode == "homography_projection" else 0.0
    )
    mapping_contributors = args.mapping_contributors or args.contributors
    query_contributors = args.query_contributors or args.contributors
    if mapping_contributors is None or query_contributors is None:
        raise ValueError("mapping and query contributor roots are required")
    if args.source_subtoken_geometry == "homography_depth":
        if args.mapping_depth_contributors is None:
            raise ValueError("homography source depth requires mapping depth contributors")
    elif args.source_subtoken_geometry == "homography_plane":
        if args.planar_map is None:
            raise ValueError("homography source plane intersections require the planar map")
        if args.mapping_camera_contributors is None:
            raise ValueError("homography source plane intersections require mapping camera contributors")
        if args.mapping_depth_contributors is not None:
            raise ValueError("homography source plane intersections do not consume depth")
    elif args.mapping_depth_contributors is not None:
        raise ValueError("mapping depth contributors supplied without homography source depth")
    if (
        args.source_subtoken_geometry != "homography_plane"
        and args.mapping_camera_contributors is not None
    ):
        raise ValueError("mapping camera contributors supplied without homography plane geometry")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite RADIO plane PnP diagnostic")
    atlas, atlas_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    token_grid = tuple(map(int, atlas.token_pixel_counts.shape[1:]))
    if list(token_grid) != list(atlas_meta.get("token_grid", (36, 64))):
        raise ValueError("visibility token grid metadata differs")
    planar_map = None
    if args.source_point_geometry == "plane_projected" or args.source_subtoken_geometry == "homography_plane":
        if args.planar_map is None:
            raise ValueError("plane-projected source points require the planar map")
        if file_sha256(args.planar_map) != atlas_meta.get("planar_map_file_sha256"):
            raise ValueError("plane-projection map differs from visibility atlas")
        planar_map = GeometryNativePlanarMap.load_npz(args.planar_map)
    elif args.planar_map is not None:
        raise ValueError("planar map supplied without plane-projected source geometry")
    bank_arrays = bank_meta = None
    if args.source_observation_bank is not None:
        with np.load(args.source_observation_bank, allow_pickle=False) as data:
            bank_meta = json.loads(str(data["metadata_json"].item()))
            bank_arrays = {name: np.asarray(data[name]) for name in (
                "observation_offsets", "token_ids", "world_points", "radio_features",
            )}
        if (
            bank_meta.get("artifact_type") != "goal_maplet_plane_pnp_observation_bank_v1"
            or bank_meta.get("visibility_atlas_content_sha256") != atlas_meta.get("content_sha256")
            or list(bank_meta.get("token_grid", (36, 64))) != list(token_grid)
            or arrays_sha256(bank_arrays) != bank_meta.get("arrays_sha256")
            or len(bank_arrays["observation_offsets"]) != len(atlas.view_names) + 1
        ):
            raise ValueError("source observation bank contract differs")
    ranking = json.loads(args.correspondence_report.read_text())
    if list(ranking.get("token_grid", (36, 64))) != list(token_grid):
        raise ValueError("plane ranking token grid differs")
    ranking_is_label_free = (
        ranking.get("artifact_type") in (
            "goal_maplet_pose_free_radio_to_physical_plane_ranking_v2",
            "goal_maplet_direct_radio_to_finite_plane_ranking_v1",
            "goal_maplet_direct_radio_to_finite_plane_ranking_v2",
            "goal_maplet_context_fused_direct_radio_to_finite_plane_ranking_v1",
            "goal_maplet_context_fused_direct_radio_to_finite_plane_ranking_v2",
        )
        and ranking.get("uses_pose_or_ground_truth") is False
        and ranking.get("contains_postlabel_fields") is False
    )
    ranking_semantics = (
        "direct query-region RADIO to finite-plane observation descriptors"
        if ranking.get("artifact_type") in (
            "goal_maplet_direct_radio_to_finite_plane_ranking_v1",
            "goal_maplet_direct_radio_to_finite_plane_ranking_v2",
            "goal_maplet_context_fused_direct_radio_to_finite_plane_ranking_v1",
            "goal_maplet_context_fused_direct_radio_to_finite_plane_ranking_v2",
        )
        else f"pose-free RADIO child-to-plane top{int(args.topk_planes)}"
    )
    radio_records = _records(args.radio_manifest)
    camera_rows = camera_meta = None
    if args.query_camera_inventory is not None:
        camera_rows, camera_meta = _camera_inventory(args.query_camera_inventory)
    source_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    mapping_depth_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    mapping_depth_hashes: dict[str, str] = {}
    feature_cache: dict[str, np.ndarray] = {}
    query_plane_inventory = {
        str(row["name"]): row for row in ranking.get("query_plane_inventory", [])
    }
    strict_query_plane_lineage = ranking.get("artifact_type") in (
        "goal_maplet_direct_radio_to_finite_plane_ranking_v2",
        "goal_maplet_context_fused_direct_radio_to_finite_plane_ranking_v2",
    )
    if strict_query_plane_lineage and len(query_plane_inventory) != len(ranking.get("rows", [])):
        raise ValueError("strict plane ranking has an incomplete query-plane inventory")
    descriptor_weighting = str(ranking.get("query_region_descriptor_weighting", "uniform"))
    if descriptor_weighting not in ("uniform", "visible_pixel_fraction"):
        raise ValueError("unsupported query-region descriptor weighting")
    sparse_carrier = bool(ranking.get("sparse_occlusion_carrier", False))
    candidate_sharing = bool(ranking.get("sparse_occlusion_candidate_sharing", False))
    carrier_inventory = {
        str(row["name"]): row
        for row in ranking.get("candidate_sharing_carrier_inventory", [])
    }
    if candidate_sharing:
        if args.sparse_occlusion_carrier_dir is None or sparse_carrier:
            raise ValueError("candidate-sharing ranking requires a separate carrier inventory")
        carrier_manifest_path = args.sparse_occlusion_carrier_dir / "manifest.json"
        carrier_manifest = json.loads(carrier_manifest_path.read_text())
        if (
            file_sha256(carrier_manifest_path)
            != ranking.get("candidate_sharing_carrier_manifest_file_sha256")
            or carrier_manifest.get("sparse_occlusion_carrier") is not True
            or carrier_manifest.get("observed_support_only") is not True
            or carrier_manifest.get("hidden_pixel_count_added") != 0
            or len(carrier_inventory) != len(ranking.get("rows", []))
        ):
            raise ValueError("candidate-sharing carrier lineage differs")
    elif args.sparse_occlusion_carrier_dir is not None:
        raise ValueError("carrier directory supplied to a ranking without candidate sharing")

    def source_observation(plane: int, atlas_row: int):
        key = (int(plane), int(atlas_row))
        if key in source_cache:
            return source_cache[key]
        name = str(atlas.view_names[atlas_row])
        if bank_arrays is not None:
            lo, hi = map(int, bank_arrays["observation_offsets"][atlas_row:atlas_row + 2])
            tokens = np.asarray(bank_arrays["token_ids"][lo:hi], np.int64)
            points = np.asarray(bank_arrays["world_points"][lo:hi], np.float64)
            features = np.asarray(bank_arrays["radio_features"][lo:hi], np.float32)
        else:
            tokens = np.flatnonzero(atlas.token_pixel_counts[atlas_row].reshape(-1) >= MINIMUM_PLANE_TOKEN_PIXELS)
            points, tokens = _token_world_points(
                mapping_contributors / name, tokens, token_grid=token_grid,
            )
            if name not in feature_cache:
                feature_cache[name] = _radio(name, radio_records)
            features = feature_cache[name][tokens]
        if planar_map is not None:
            if int(plane) < 0 or int(plane) >= len(planar_map.plane_ids):
                raise ValueError("visibility plane row is outside planar map")
            points = _project_points_to_plane(
                points,
                planar_map.normals_world[int(plane)],
                float(planar_map.offsets_world[int(plane)]),
            )
        descriptor = np.mean(features, axis=0) if len(features) else np.zeros(1280, np.float32)
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-8)
        source_cache[key] = points, tokens, features, descriptor
        return source_cache[key]

    def mapping_depth_geometry(name: str):
        if name in mapping_depth_cache:
            return mapping_depth_cache[name]
        path = args.mapping_depth_contributors / name
        with np.load(path, allow_pickle=False) as data:
            depth = np.asarray(data["dominant_depth"], np.float64)
            pose = np.asarray(data["pose_w2c"], np.float64)
            model_id = int(data["camera_model_id"])
            source_width = int(data["camera_width"])
            source_height = int(data["camera_height"])
            params = np.asarray(data["camera_params"], np.float64)
        if depth.shape != (144, 256) or pose.shape != (4, 4):
            raise ValueError("mapping depth contributor geometry differs")
        K, k1 = _scaled_intrinsics(model_id, params, source_width, source_height)
        mapping_depth_cache[name] = depth, pose, K, k1
        mapping_depth_hashes[name] = file_sha256(path)
        return mapping_depth_cache[name]

    mapping_camera_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    mapping_camera_hashes: dict[str, str] = {}

    def mapping_camera_geometry(name: str):
        if name in mapping_camera_cache:
            return mapping_camera_cache[name]
        path = args.mapping_camera_contributors / name
        with np.load(path, allow_pickle=False) as data:
            pose = np.asarray(data["pose_w2c"], np.float64)
            model_id = int(data["camera_model_id"])
            source_width = int(data["camera_width"])
            source_height = int(data["camera_height"])
            params = np.asarray(data["camera_params"], np.float64)
        if pose.shape != (4, 4):
            raise ValueError("mapping contributor camera geometry differs")
        K, k1 = _scaled_intrinsics(model_id, params, source_width, source_height)
        mapping_camera_cache[name] = pose, K, k1
        mapping_camera_hashes[name] = file_sha256(path)
        return mapping_camera_cache[name]

    frozen = []
    frozen_correspondences: list[dict[str, object]] = []
    for query_row in ranking["rows"]:
        name = str(query_row["image"])
        query_plane_path = args.query_plane_dir / name
        query_planes, query_plane_meta = QueryPlaneRegions.load_npz(query_plane_path)
        if strict_query_plane_lineage:
            row = query_plane_inventory.get(name)
            if (
                row is None
                or row.get("file_sha256") != file_sha256(query_plane_path)
                or row.get("content_sha256") != query_plane_meta.get("content_sha256")
                or bool(query_plane_meta.get("sparse_occlusion_carrier", False)) != sparse_carrier
            ):
                raise ValueError("query-plane bytes differ from strict ranking lineage")
        if candidate_sharing:
            carrier_path = args.sparse_occlusion_carrier_dir / name
            carrier_planes, carrier_meta = QueryPlaneRegions.load_npz(carrier_path)
            carrier_row = carrier_inventory.get(name)
            if (
                carrier_row is None
                or carrier_row.get("file_sha256") != file_sha256(carrier_path)
                or carrier_row.get("content_sha256") != carrier_meta.get("content_sha256")
                or not np.array_equal(query_planes.labels >= 0, carrier_planes.labels >= 0)
            ):
                raise ValueError("candidate-sharing carrier bytes/support differ")
            for record in query_row["regions"]:
                base_region = int(record["region"])
                values = np.unique(carrier_planes.labels[query_planes.labels == base_region])
                if len(values) != 1 or int(values[0]) != int(record.get("carrier_region", -1)):
                    raise ValueError("candidate-sharing region mapping differs")
        query_feature = _radio(name, radio_records)
        if query_feature.shape[0] != int(np.prod(token_grid)):
            raise ValueError("query RADIO and visibility token grids differ")
        source_points, selection_source_points = [], []
        query_tokens, query_pixels, match_scores, match_rows = [], [], [], []
        for record in query_row["regions"]:
            region = int(record["region"])
            qtoken, qweight = _region_token_support(
                query_planes.labels, region, token_grid=token_grid,
            )
            if len(qtoken) < 2:
                continue
            qfeature = query_feature[qtoken]
            qdescriptor = (
                np.average(qfeature, axis=0, weights=qweight)
                if descriptor_weighting == "visible_pixel_fraction"
                else np.mean(qfeature, axis=0)
            ); qdescriptor /= max(float(np.linalg.norm(qdescriptor)), 1e-8)
            for plane_rank, plane in enumerate(record["top10"][: int(args.topk_planes)]):
                lo, hi = map(int, atlas.plane_offsets[int(plane):int(plane) + 2])
                candidates = []
                for atlas_row in range(lo, hi):
                    points, stoken, sfeature, descriptor = source_observation(int(plane), atlas_row)
                    if len(points) >= 2:
                        candidates.append((float(qdescriptor @ descriptor), atlas_row, points, stoken, sfeature))
                for view_score, atlas_row, points, stoken, sfeature in sorted(
                    candidates, reverse=True
                )[: int(args.top_source_views)]:
                    qi, si, cosine = _mutual_matches(qfeature, sfeature)
                    keep, projected_query_xy, projected_map_xy = _homography_filter_with_bidirectional_projections(
                        qtoken[qi], stoken[si], token_width=token_grid[1],
                    )
                    if args.query_pixel_mode == "homography_projection":
                        keep &= np.all(np.isfinite(projected_query_xy), axis=1)
                    if args.source_subtoken_geometry in ("homography_depth", "homography_plane"):
                        keep &= np.all(np.isfinite(projected_map_xy), axis=1)
                    matched_points = np.asarray(points[si], np.float64)
                    if args.source_subtoken_geometry == "homography_depth":
                        source_name = str(atlas.view_names[int(atlas_row)])
                        depth, source_pose, source_K, source_k1 = mapping_depth_geometry(source_name)
                        matched_points, _ = _homography_source_world_points(
                            depth, source_pose, source_K, source_k1,
                            projected_map_xy, matched_points, token_grid,
                        )
                    elif args.source_subtoken_geometry == "homography_plane":
                        source_name = str(atlas.view_names[int(atlas_row)])
                        source_pose, source_K, source_k1 = mapping_camera_geometry(source_name)
                        matched_points, plane_intersection_valid = _homography_source_plane_points(
                            source_pose,
                            source_K,
                            source_k1,
                            projected_map_xy,
                            token_grid,
                            planar_map.normals_world[int(plane)],
                            float(planar_map.offsets_world[int(plane)]),
                        )
                        keep &= plane_intersection_valid
                    for local in np.flatnonzero(keep).tolist():
                        query_token = int(qtoken[qi[local]])
                        query_xy = np.asarray(
                            [query_token % token_grid[1], query_token // token_grid[1]],
                            np.float64,
                        )
                        fraction = homography_projection_fraction
                        continuous_xy = (
                            query_xy
                            if fraction == 0.0 else
                            (1.0 - fraction) * query_xy
                            + fraction * projected_query_xy[local]
                        )
                        query_pixel = np.asarray([
                            (continuous_xy[0] + 0.5) * 256.0 / token_grid[1] - 0.5,
                            (continuous_xy[1] + 0.5) * 144.0 / token_grid[0] - 0.5,
                        ], np.float64)
                        source_points.append(matched_points[local])
                        # Candidate identity/deduplication remains frozen to
                        # the original token-median geometry.  A sub-token
                        # coordinate may refine PnP, but may not silently
                        # change which 3D hypothesis was selected.
                        selection_source_points.append(points[si[local]])
                        query_tokens.append(query_token)
                        query_pixels.append(query_pixel)
                        match_scores.append(float(cosine[local] - 0.02 * plane_rank + 0.01 * view_score))
                        match_rows.append((region, int(plane), int(atlas_row)))
        # Keep a fixed number of best 3D hypotheses per query token before
        # PnP.  The default remains the original one-hypothesis contract.
        if match_scores:
            chosen = _choose_token_hypotheses(
                np.asarray(query_tokens, np.int64),
                np.asarray(selection_source_points, np.float64),
                np.asarray(match_scores, np.float64),
                np.asarray(match_rows, np.int64),
                atlas.view_names,
                maximum_per_token=int(args.hypotheses_per_query_token),
                selection=str(args.hypothesis_selection),
                consensus_radius_m=float(args.hypothesis_consensus_radius_m),
            ).tolist()
            world = np.asarray(source_points, np.float64)[chosen]
            token = np.asarray(query_tokens, np.int64)[chosen]
            pixel = np.asarray(query_pixels, np.float64)[chosen]
            selected_rows = [match_rows[index] for index in chosen]
        else:
            world = np.zeros((0, 3), np.float64)
            token = np.zeros(0, np.int64)
            pixel = np.zeros((0, 2), np.float64)
            selected_rows = []
        contributor = query_contributors / name
        if camera_rows is None:
            with np.load(contributor, allow_pickle=False) as data:
                model_id = int(data["camera_model_id"]); params = np.asarray(data["camera_params"], np.float64)
                source_width = int(data["camera_width"]); source_height = int(data["camera_height"])
        else:
            if name not in camera_rows:
                raise ValueError("query camera-only inventory lacks a query")
            model_id, source_width, source_height, params = camera_rows[name]
        K, k1 = _scaled_intrinsics(model_id, params, source_width, source_height)
        frozen_correspondences.append({
            "name": name,
            "world_points": world,
            "query_tokens": token,
            "query_pixels": pixel,
            "provenance": np.asarray(selected_rows, np.int64).reshape(-1, 3),
            "camera_matrix": K,
            "radial_k1": float(k1),
        })
        pose, inlier = _pnp(
            world, token, K, k1, token_grid=token_grid, query_pixel=pixel,
        )
        if pose is not None and args.pose_solver == "plane_balanced_surface":
            pose, inlier = _plane_balanced_surface_refine(
                pose,
                inlier,
                world,
                pixel,
                np.asarray(selected_rows, np.int64).reshape(-1, 3)[:, 1],
                K,
                k1,
            )
        diagnostics = _pose_diagnostics(
            pose, inlier, world, token, selected_rows, K, k1,
            token_grid=token_grid, atlas_view_names=atlas.view_names,
            query_pixel=pixel,
        )
        frozen.append({
            "name": name,
            "candidate_correspondence_count": int(len(world)),
            "pnp_inlier_count": int(len(inlier)),
            "pose_w2c": None if pose is None else pose.tolist(),
            "matched_plane_count": int(len(set((selected_rows[i][0], selected_rows[i][1]) for i in inlier.tolist()))) if pose is not None else 0,
            "query_plane_count": int(len(query_planes.normals_camera)),
            "query_plane_coverage_fraction": float(np.mean(query_planes.labels >= 0)),
            **diagnostics,
        })

    if args.output_frozen_correspondence_inventory is not None:
        if args.output_frozen_correspondence_inventory.exists():
            raise FileExistsError("refusing to overwrite frozen PnP correspondences")
        offsets = np.zeros(len(frozen_correspondences) + 1, np.int64)
        for index, row in enumerate(frozen_correspondences):
            offsets[index + 1] = offsets[index] + len(row["query_tokens"])
        correspondence_arrays = {
            "names": np.asarray([row["name"] for row in frozen_correspondences]),
            "correspondence_offsets": offsets,
            "world_points": np.concatenate(
                [row["world_points"] for row in frozen_correspondences], axis=0
            ).astype(np.float64),
            "query_tokens": np.concatenate(
                [row["query_tokens"] for row in frozen_correspondences], axis=0
            ).astype(np.int64),
            "provenance_region_plane_atlas_row": np.concatenate(
                [row["provenance"] for row in frozen_correspondences], axis=0
            ).astype(np.int64),
            "camera_matrices": np.asarray(
                [row["camera_matrix"] for row in frozen_correspondences], np.float64
            ),
            "radial_k1": np.asarray(
                [row["radial_k1"] for row in frozen_correspondences], np.float64
            ),
        }
        if args.query_pixel_mode == "homography_projection":
            correspondence_arrays["query_pixels"] = np.concatenate(
                [row["query_pixels"] for row in frozen_correspondences], axis=0
            ).astype(np.float64)
        correspondence_metadata = {
            "artifact_type": (
                "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v2"
                if (
                    args.query_pixel_mode == "homography_projection"
                    or args.source_point_geometry == "plane_projected"
                    or args.source_subtoken_geometry in ("homography_depth", "homography_plane")
                )
                else "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v1"
            ),
            "arrays_sha256": arrays_sha256(correspondence_arrays),
            "query_count": int(len(frozen_correspondences)),
            "correspondence_count": int(offsets[-1]),
            "pose_or_ground_truth_opened": False,
            "query_depth_or_scale_used": False,
            "sparse_occlusion_carrier": sparse_carrier,
            "query_region_descriptor_weighting": descriptor_weighting,
            "hypotheses_per_query_token": int(args.hypotheses_per_query_token),
            "one_best_3d_hypothesis_per_query_token": int(args.hypotheses_per_query_token) == 1,
            "hypothesis_selection": str(args.hypothesis_selection),
            "hypothesis_consensus_radius_m": float(args.hypothesis_consensus_radius_m),
            "query_pixel_mode": str(args.query_pixel_mode),
            "homography_projection_fraction": homography_projection_fraction,
            "source_point_geometry": str(args.source_point_geometry),
            "source_subtoken_geometry": str(args.source_subtoken_geometry),
            "pose_solver": str(args.pose_solver),
            "plane_balance_quadratic_mass_cap": int(PLANE_BALANCE_QUADRATIC_MASS_CAP),
            "mapping_depth_contributor_file_sha256_by_name": dict(sorted(mapping_depth_hashes.items())),
            "mapping_camera_contributor_file_sha256_by_name": dict(sorted(mapping_camera_hashes.items())),
            "planar_map_file_sha256": (
                None if args.planar_map is None else file_sha256(args.planar_map)
            ),
            "topk_planes": int(args.topk_planes),
            "top_source_views": int(args.top_source_views),
            "token_grid": list(token_grid),
            "correspondence_report_file_sha256": file_sha256(args.correspondence_report),
            "query_camera_only_inventory_file_sha256": (
                None if args.query_camera_inventory is None
                else file_sha256(args.query_camera_inventory)
            ),
            "source_observation_bank_file_sha256": (
                None if args.source_observation_bank is None
                else file_sha256(args.source_observation_bank)
            ),
        }
        correspondence_metadata["content_sha256"] = canonical_json_sha256(
            correspondence_metadata
        )
        args.output_frozen_correspondence_inventory.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_frozen_correspondence_inventory.with_name(
            args.output_frozen_correspondence_inventory.name + ".temporary.npz"
        )
        np.savez_compressed(
            temporary,
            **correspondence_arrays,
            metadata_json=np.asarray(json.dumps(correspondence_metadata, sort_keys=True)),
        )
        temporary.replace(args.output_frozen_correspondence_inventory)

    # Persist candidate poses before opening any pose-bearing query member.
    # This artifact is the only legal input to later render-consistency gates:
    # it cannot inherit GT through the Phase-2 evaluation report.
    frozen_pose_meta = None
    if args.output_frozen_pose_inventory is not None:
        if args.output_frozen_pose_inventory.exists():
            raise FileExistsError("refusing to overwrite frozen PnP pose inventory")
        arrays = {
            "names": np.asarray([row["name"] for row in frozen]),
            "pose_w2c": np.asarray([
                np.full((4, 4), np.nan, np.float64)
                if row["pose_w2c"] is None else np.asarray(row["pose_w2c"], np.float64)
                for row in frozen
            ], np.float64),
            "usable": np.asarray([row["pose_w2c"] is not None for row in frozen], bool),
            "candidate_correspondence_count": np.asarray([
                row["candidate_correspondence_count"] for row in frozen
            ], np.int64),
            "pnp_inlier_count": np.asarray([
                row["pnp_inlier_count"] for row in frozen
            ], np.int64),
        }
        frozen_pose_meta = {
            "artifact_type": "goal_maplet_frozen_direct_plane_pnp_pose_inventory_v1",
            "arrays_sha256": arrays_sha256(arrays),
            "query_count": int(len(frozen)),
            "pose_frozen_before_query_pose_or_ground_truth_open": True,
            "query_depth_or_scale_used_by_pose_solver": False,
            "query_moge3_role": (
                "plane_segmentation_and_sparse_foreground_candidate_sharing_only"
                if candidate_sharing else
                "plane_segmentation_and_sparse_foreground_carrier_only"
                if sparse_carrier else "plane_segmentation_only"
            ),
            "token_grid": list(token_grid),
            "top_source_views": int(args.top_source_views),
            "query_pixel_mode": str(args.query_pixel_mode),
            "homography_projection_fraction": homography_projection_fraction,
            "source_point_geometry": str(args.source_point_geometry),
            "source_subtoken_geometry": str(args.source_subtoken_geometry),
            "pose_solver": str(args.pose_solver),
            "plane_balance_quadratic_mass_cap": int(PLANE_BALANCE_QUADRATIC_MASS_CAP),
            "mapping_depth_contributor_file_sha256_by_name": dict(sorted(mapping_depth_hashes.items())),
            "mapping_camera_contributor_file_sha256_by_name": dict(sorted(mapping_camera_hashes.items())),
            "planar_map_file_sha256": (
                None if args.planar_map is None else file_sha256(args.planar_map)
            ),
            "correspondence_report_file_sha256": file_sha256(args.correspondence_report),
            "query_camera_only_inventory_file_sha256": (
                None if args.query_camera_inventory is None
                else file_sha256(args.query_camera_inventory)
            ),
            "source_observation_bank_file_sha256": (
                None if args.source_observation_bank is None
                else file_sha256(args.source_observation_bank)
            ),
        }
        frozen_pose_meta["content_sha256"] = canonical_json_sha256(frozen_pose_meta)
        args.output_frozen_pose_inventory.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_frozen_pose_inventory.with_name(
            args.output_frozen_pose_inventory.name + ".temporary.npz"
        )
        np.savez_compressed(
            temporary,
            **arrays,
            metadata_json=np.asarray(json.dumps(frozen_pose_meta, sort_keys=True)),
        )
        temporary.replace(args.output_frozen_pose_inventory)

    # Phase 2: only now open the pose-bearing arrays for evaluation.
    rows = []
    for row in frozen:
        output = {k: v for k, v in row.items() if k != "pose_w2c"}
        output["usable"] = row["pose_w2c"] is not None
        if row["pose_w2c"] is not None:
            pose = np.asarray(row["pose_w2c"], np.float64)
            with np.load(query_contributors / row["name"], allow_pickle=False) as data:
                gt = np.asarray(data["pose_w2c"], np.float64)
            center = -pose[:3, :3].T @ pose[:3, 3]
            gt_center = -gt[:3, :3].T @ gt[:3, 3]
            output["translation_error_m"] = float(np.linalg.norm(center - gt_center))
            output["rotation_error_deg"] = float(
                Rotation.from_matrix(pose[:3, :3] @ gt[:3, :3].T).magnitude() * 180.0 / np.pi
            )
        rows.append(output)
    usable = [row for row in rows if row["usable"]]
    translation = np.asarray([row["translation_error_m"] for row in usable])
    rotation = np.asarray([row["rotation_error_deg"] for row in usable])
    report = {
        "artifact_type": "goal_maplet_radio_plane_2d3d_pnp_control_v2",
        "token_grid": list(token_grid),
        "query_count": len(rows),
        "usable_count": len(usable),
        "query_depth_or_scale_used_by_pose_solver": False,
        "query_moge3_role": (
            "plane_segmentation_and_sparse_foreground_candidate_sharing_only"
            if candidate_sharing else
            "plane_segmentation_and_sparse_foreground_carrier_only"
            if sparse_carrier else "plane_segmentation_only"
        ),
        "sparse_occlusion_carrier": sparse_carrier,
        "sparse_occlusion_candidate_sharing": candidate_sharing,
        "query_region_descriptor_weighting": descriptor_weighting,
        "plane_retrieval": ranking_semantics,
        "within_plane_matching": "mutual RADIO token + homography RANSAC",
        "query_pixel_mode": str(args.query_pixel_mode),
        "homography_projection_fraction": homography_projection_fraction,
        "map_3d_source": (
            "2DGS contributor dominant depth projected onto its frozen finite plane"
            if args.source_point_geometry == "plane_projected"
            else "2DGS contributor dominant depth at mapping tokens"
        ),
        "source_point_geometry": str(args.source_point_geometry),
        "source_subtoken_geometry": str(args.source_subtoken_geometry),
        "pose_solver": str(args.pose_solver),
        "plane_balance_quadratic_mass_cap": int(PLANE_BALANCE_QUADRATIC_MASS_CAP),
        "mapping_depth_contributor_file_sha256_by_name": dict(sorted(mapping_depth_hashes.items())),
        "mapping_camera_contributor_file_sha256_by_name": dict(sorted(mapping_camera_hashes.items())),
        "planar_map_file_sha256": (
            None if args.planar_map is None else file_sha256(args.planar_map)
        ),
        "gt_pose_opened_after_correspondences_and_pose_frozen": True,
        "plane_ranking_label_free_contract_verified": bool(ranking_is_label_free),
        "pose_confidence_diagnostics_frozen_before_gt": True,
        "topk_planes": int(args.topk_planes),
        "top_source_views": int(args.top_source_views),
        "query_pose_bearing_contributor_opened_for_intrinsics_before_pose_freeze": camera_rows is None,
        "query_camera_only_inventory_file_sha256": None if args.query_camera_inventory is None else file_sha256(args.query_camera_inventory),
        "query_camera_only_inventory_content_sha256": None if camera_meta is None else camera_meta.get("content_sha256"),
        "strict_runtime_phase_separation_eligible": camera_rows is not None and ranking_is_label_free,
        "mapping_and_query_contributor_roots_separated": mapping_contributors.resolve() != query_contributors.resolve(),
        "source_observation_bank_file_sha256": None if args.source_observation_bank is None else file_sha256(args.source_observation_bank),
        "source_observation_bank_content_sha256": None if bank_meta is None else bank_meta.get("content_sha256"),
        "frozen_pose_inventory_file_sha256": (
            None if args.output_frozen_pose_inventory is None
            else file_sha256(args.output_frozen_pose_inventory)
        ),
        "frozen_pose_inventory_content_sha256": (
            None if frozen_pose_meta is None else frozen_pose_meta.get("content_sha256")
        ),
        "official_test_map_disjoint_declared": bool(args.official_test_map_disjoint),
        "full_train_2dgs_may_have_consumed_query_route_images": not bool(args.official_test_map_disjoint),
        "mapping_observation_query_route_overlap": bool(
            set(str(name).split("__", 1)[0] for name in atlas.view_names.astype(str))
            & set(str(row["image"]).split("__", 1)[0] for row in ranking["rows"])
        ),
        "correspondence_report_file_sha256": file_sha256(args.correspondence_report),
        "radio_manifest_file_sha256_in_order": [file_sha256(path) for path in args.radio_manifest],
        "visibility_atlas_file_sha256": file_sha256(args.visibility_atlas),
        "visibility_atlas_content_sha256": atlas_meta.get("content_sha256"),
        "production_eligible": False,
        "median_translation_m": float(np.median(translation)) if len(translation) else None,
        "median_rotation_deg": float(np.median(rotation)) if len(rotation) else None,
        "full_query_recall_2m45": float(np.sum((translation <= 2) & (rotation <= 45)) / len(rows)) if rows else 0.0,
        "full_query_recall_1m10": float(np.sum((translation <= 1) & (rotation <= 10)) / len(rows)) if rows else 0.0,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
