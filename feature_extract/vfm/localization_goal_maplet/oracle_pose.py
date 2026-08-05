"""Oracle observation layers for isolating Goal-Maplet pose information loss."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion

from .pfir import ContributorLabels, _primitive_to_maplet_links
from .physical_map import GoalMapletPhysicalMap


@dataclass(frozen=True)
class TokenOracleEvidence:
    xy_px: np.ndarray
    clean_xyz: np.ndarray
    clean_mass: np.ndarray
    owned_xyz: np.ndarray
    owned_mass: np.ndarray
    parent_rows: np.ndarray
    parent_mass: np.ndarray
    parent_purity: np.ndarray
    child_rows: np.ndarray
    child_mass: np.ndarray
    child_purity: np.ndarray
    child_local_xyz: np.ndarray


def intersect_rays_with_primitive_planes(
    xy_px: np.ndarray,
    primitive_rows: np.ndarray,
    physical: GoalMapletPhysicalMap,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift pixels to their contributor surfel planes under an oracle pose."""

    import cv2

    pixels = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)
    rows = np.asarray(primitive_rows, dtype=np.int64).reshape(-1)
    if rows.shape != (pixels.shape[0],):
        raise ValueError("pixel and primitive arrays differ")
    matrix, distortion = camera_matrix_and_distortion(camera)
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), matrix, distortion).reshape(-1, 2)
    ray_camera = np.column_stack([normalized, np.ones((normalized.shape[0],), dtype=np.float64)])
    ray_camera /= np.maximum(np.linalg.norm(ray_camera, axis=1, keepdims=True), 1e-12)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_center = -pose[:3, :3].T @ pose[:3, 3]
    ray_world = ray_camera @ pose[:3, :3]
    normal = physical.primitive_normals[rows]
    center = physical.primitive_centers[rows]
    denominator = np.sum(normal * ray_world, axis=1)
    numerator = np.sum(normal * (center - camera_center[None]), axis=1)
    safe_denominator = np.where(
        np.abs(denominator) >= 1e-8,
        denominator,
        np.where(denominator >= 0.0, 1e-8, -1e-8),
    )
    distance = numerator / safe_denominator
    point = camera_center[None] + distance[:, None] * ray_world
    valid = np.isfinite(distance) & (distance > 0.0) & (np.abs(denominator) >= 1e-5)
    return point, valid


def _child_by_parent_primitive(physical: GoalMapletPhysicalMap) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for child_row, parent_row in enumerate(physical.child_parent_rows.tolist()):
        start, end = int(physical.child_member_offsets[child_row]), int(physical.child_member_offsets[child_row + 1])
        for primitive_row in physical.child_member_primitive_rows[start:end].tolist():
            key = (int(parent_row), int(primitive_row))
            if key in result:
                raise ValueError("a primitive belongs to multiple children of one parent")
            result[key] = int(child_row)
    return result


def token_oracle_evidence(
    labels: ContributorLabels,
    physical: GoalMapletPhysicalMap,
    token_xy: np.ndarray,
    *,
    token_height: int,
    token_width: int,
    image_height: int,
    image_width: int,
    camera: ColmapCamera | None = None,
) -> TokenOracleEvidence:
    """Associate each token's central cell with exact physical hierarchy."""

    xy = np.asarray(token_xy, dtype=np.int64).reshape(-1, 2)
    height, width, topk = labels.topk_primitive_ids.shape
    primitive_row_by_id = {int(value): row for row, value in enumerate(physical.primitive_ids.tolist())}
    link_offsets, link_rows, link_weights = _primitive_to_maplet_links(physical)
    child_by_key = _child_by_parent_primitive(physical)
    count = xy.shape[0]
    clean_xyz = np.zeros((count, 3), dtype=np.float64)
    clean_mass = np.zeros((count,), dtype=np.float64)
    owned_xyz = np.zeros((count, 3), dtype=np.float64)
    owned_mass = np.zeros((count,), dtype=np.float64)
    parent_rows = np.full((count,), -1, dtype=np.int64)
    parent_mass = np.zeros((count,), dtype=np.float64)
    parent_purity = np.zeros((count,), dtype=np.float64)
    child_rows = np.full((count,), -1, dtype=np.int64)
    child_mass = np.zeros((count,), dtype=np.float64)
    child_purity = np.zeros((count,), dtype=np.float64)
    child_local_xyz = np.zeros((count, 3), dtype=np.float64)
    evidence_xy_px = (xy.astype(np.float64) + 0.5) * np.asarray(
        [image_width / float(token_width), image_height / float(token_height)], dtype=np.float64
    )
    for token, (x, y) in enumerate(xy.tolist()):
        px0 = int(np.floor(x * width / float(token_width)))
        px1 = int(np.ceil((x + 1) * width / float(token_width)))
        py0 = int(np.floor(y * height / float(token_height)))
        py1 = int(np.ceil((y + 1) * height / float(token_height)))
        ids = labels.topk_primitive_ids[py0:py1, px0:px1].reshape(-1, topk)
        values = labels.topk_weights[py0:py1, px0:px1].reshape(-1, topk)
        grid_y, grid_x = np.mgrid[py0:py1, px0:px1]
        contributor_xy = np.stack(
            [
                np.repeat((grid_x.reshape(-1) + 0.5) * image_width / width, topk),
                np.repeat((grid_y.reshape(-1) + 0.5) * image_height / height, topk),
            ],
            axis=1,
        )
        flat_ids = ids.reshape(-1)
        flat_values = values.reshape(-1).astype(np.float64)
        flat_rows = np.asarray(
            [primitive_row_by_id.get(int(value), -1) for value in flat_ids.tolist()], dtype=np.int64
        )
        valid_contributor = (flat_rows >= 0) & (flat_values > 0.0)
        intersection = np.zeros((flat_rows.size, 3), dtype=np.float64)
        if np.any(valid_contributor):
            selected = np.flatnonzero(valid_contributor)
            if camera is None:
                intersection[selected] = physical.primitive_centers[flat_rows[selected]]
            else:
                point, ray_valid = intersect_rays_with_primitive_planes(
                    contributor_xy[selected], flat_rows[selected], physical, labels.pose_w2c, camera
                )
                intersection[selected] = point
                valid_contributor[selected[~ray_valid]] = False
        primitive_mass: dict[int, float] = {}
        primitive_xyz: dict[int, np.ndarray] = {}
        primitive_xy: dict[int, np.ndarray] = {}
        for row, value, point, pixel in zip(
            flat_rows[valid_contributor].tolist(),
            flat_values[valid_contributor].tolist(),
            intersection[valid_contributor],
            contributor_xy[valid_contributor],
        ):
            primitive_mass[int(row)] = primitive_mass.get(int(row), 0.0) + float(value)
            primitive_xyz[int(row)] = primitive_xyz.get(int(row), np.zeros((3,), dtype=np.float64)) + float(value) * point
            primitive_xy[int(row)] = primitive_xy.get(int(row), np.zeros((2,), dtype=np.float64)) + float(value) * pixel
        if not primitive_mass:
            continue
        primitive_rows = np.asarray(list(primitive_mass), dtype=np.int64)
        masses = np.asarray([primitive_mass[int(row)] for row in primitive_rows], dtype=np.float64)
        clean_mass[token] = float(np.sum(masses))
        clean_xyz[token] = np.sum([primitive_xyz[int(row)] for row in primitive_rows], axis=0) / clean_mass[token]
        evidence_xy_px[token] = np.sum([primitive_xy[int(row)] for row in primitive_rows], axis=0) / clean_mass[token]
        parent_score: dict[int, float] = {}
        owned_numerator = np.zeros((3,), dtype=np.float64)
        for primitive_row, mass in zip(primitive_rows.tolist(), masses.tolist()):
            start, end = int(link_offsets[primitive_row]), int(link_offsets[primitive_row + 1])
            if end <= start:
                continue
            owned_mass[token] += float(mass)
            owned_numerator += primitive_xyz[int(primitive_row)]
            for parent, link_weight in zip(link_rows[start:end].tolist(), link_weights[start:end].tolist()):
                parent_score[int(parent)] = parent_score.get(int(parent), 0.0) + float(mass) * float(link_weight)
        if owned_mass[token] > 0.0:
            owned_xyz[token] = owned_numerator / owned_mass[token]
        if not parent_score:
            continue
        parent = min(parent_score, key=lambda row: (-parent_score[row], row))
        parent_rows[token] = int(parent)
        parent_mass[token] = float(parent_score[parent])
        parent_purity[token] = float(parent_score[parent] / max(sum(parent_score.values()), 1e-12))
        child_score: dict[int, float] = {}
        child_numerator: dict[int, np.ndarray] = {}
        for primitive_row, mass in zip(primitive_rows.tolist(), masses.tolist()):
            start, end = int(link_offsets[primitive_row]), int(link_offsets[primitive_row + 1])
            parent_link = [
                float(weight)
                for row, weight in zip(link_rows[start:end].tolist(), link_weights[start:end].tolist())
                if int(row) == int(parent)
            ]
            child = child_by_key.get((int(parent), int(primitive_row)))
            if child is None or not parent_link:
                continue
            value = float(mass) * parent_link[0]
            child_score[child] = child_score.get(child, 0.0) + value
            child_numerator[child] = (
                child_numerator.get(child, np.zeros((3,), dtype=np.float64))
                + float(parent_link[0]) * primitive_xyz[int(primitive_row)]
            )
        if child_score:
            child = min(child_score, key=lambda row: (-child_score[row], row))
            child_rows[token] = int(child)
            child_mass[token] = float(child_score[child])
            child_purity[token] = float(child_score[child] / max(sum(child_score.values()), 1e-12))
            child_local_xyz[token] = child_numerator[child] / max(child_score[child], 1e-12)
    return TokenOracleEvidence(
        xy_px=evidence_xy_px,
        clean_xyz=clean_xyz,
        clean_mass=clean_mass,
        owned_xyz=owned_xyz,
        owned_mass=owned_mass,
        parent_rows=parent_rows,
        parent_mass=parent_mass,
        parent_purity=parent_purity,
        child_rows=child_rows,
        child_mass=child_mass,
        child_purity=child_purity,
        child_local_xyz=child_local_xyz,
    )


def grouped_oracle_correspondences(
    evidence: TokenOracleEvidence,
    physical: GoalMapletPhysicalMap,
    member_offsets: np.ndarray,
    member_token_indices: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return layered region correspondences for one grouping policy."""

    offsets = np.asarray(member_offsets, dtype=np.int64).reshape(-1)
    members = np.asarray(member_token_indices, dtype=np.int64).reshape(-1)
    output_xy: dict[str, list[np.ndarray]] = {name: [] for name in ("clean_surface", "owned_surface", "parent_center", "child_center", "child_local")}
    output_xyz: dict[str, list[np.ndarray]] = {name: [] for name in output_xy}

    def append_weighted(name: str, rows: np.ndarray, mass: np.ndarray, xyz: np.ndarray) -> None:
        valid = mass[rows] > 0.0
        if not np.any(valid):
            return
        selected = rows[valid]
        weight = mass[selected]
        output_xy[name].append(np.average(evidence.xy_px[selected], axis=0, weights=weight))
        output_xyz[name].append(np.average(xyz[selected], axis=0, weights=weight))

    for group in range(offsets.size - 1):
        rows = members[int(offsets[group]) : int(offsets[group + 1])]
        append_weighted("clean_surface", rows, evidence.clean_mass, evidence.clean_xyz)
        append_weighted("owned_surface", rows, evidence.owned_mass, evidence.owned_xyz)
        valid_parent = rows[evidence.parent_rows[rows] >= 0]
        if valid_parent.size == 0:
            continue
        parent_scores: dict[int, float] = {}
        for row in valid_parent.tolist():
            parent = int(evidence.parent_rows[row])
            parent_scores[parent] = parent_scores.get(parent, 0.0) + float(evidence.parent_mass[row])
        parent = min(parent_scores, key=lambda value: (-parent_scores[value], value))
        parent_rows = valid_parent[evidence.parent_rows[valid_parent] == parent]
        parent_weight = evidence.parent_mass[parent_rows]
        output_xy["parent_center"].append(np.average(evidence.xy_px[parent_rows], axis=0, weights=parent_weight))
        output_xyz["parent_center"].append(physical.maplet_centers[parent])
        valid_child = parent_rows[evidence.child_rows[parent_rows] >= 0]
        if valid_child.size == 0:
            continue
        child_scores: dict[int, float] = {}
        for row in valid_child.tolist():
            child = int(evidence.child_rows[row])
            child_scores[child] = child_scores.get(child, 0.0) + float(evidence.child_mass[row])
        child = min(child_scores, key=lambda value: (-child_scores[value], value))
        child_rows = valid_child[evidence.child_rows[valid_child] == child]
        child_weight = evidence.child_mass[child_rows]
        child_xy = np.average(evidence.xy_px[child_rows], axis=0, weights=child_weight)
        output_xy["child_center"].append(child_xy)
        output_xyz["child_center"].append(physical.child_centers[child])
        output_xy["child_local"].append(child_xy)
        output_xyz["child_local"].append(np.average(evidence.child_local_xyz[child_rows], axis=0, weights=child_weight))
    return {
        name: (
            np.asarray(output_xy[name], dtype=np.float64).reshape(-1, 2),
            np.asarray(output_xyz[name], dtype=np.float64).reshape(-1, 3),
        )
        for name in output_xy
    }
