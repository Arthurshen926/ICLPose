"""Patch-overlap SE(3) pose optimization for patch-positive 2D-3D anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.patch_to_3d_matching import TokenPatchBox
from feature_extract.vfm.patch_to_3d_matching import token_patch_boxes
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatch
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, Vfm2DgsAnchorMap


@dataclass(frozen=True)
class PatchOverlapSupport:
    points_xyz: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xyz, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_xyz must have shape (N, 3)")
        if weights.shape[0] != points.shape[0]:
            raise ValueError("weights must have one value per support point")
        if points.shape[0] == 0:
            raise ValueError("support must contain at least one point")
        object.__setattr__(self, "points_xyz", points)
        object.__setattr__(self, "weights", np.clip(weights, 0.0, None))


@dataclass(frozen=True)
class PatchOverlapConfig:
    iterations: int = 60
    lr: float = 0.05
    sigma: float = 0.5
    epsilon: float = 1e-6
    max_anchors_per_patch: int = 20
    depth_weight: float = 0.01
    depth_min: float = 1e-4
    center_weight: float = 0.0
    center_inner_fraction: float = 0.0
    area_weight: float = 0.0
    min_normalized_area: float = 0.0
    max_normalized_area: float = 2.0
    init_regularization_weight: float = 0.0
    optimize_rotation: bool = True
    optimize_translation: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if int(self.iterations) < 0:
            raise ValueError("iterations must be non-negative")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.sigma) <= 0.0:
            raise ValueError("sigma must be positive")
        if float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive")
        if int(self.max_anchors_per_patch) <= 0:
            raise ValueError("max_anchors_per_patch must be positive")
        if float(self.depth_weight) < 0.0:
            raise ValueError("depth_weight must be non-negative")
        if float(self.depth_min) < 0.0:
            raise ValueError("depth_min must be non-negative")
        if float(self.center_weight) < 0.0:
            raise ValueError("center_weight must be non-negative")
        if not 0.0 <= float(self.center_inner_fraction) <= 1.0:
            raise ValueError("center_inner_fraction must be in [0, 1]")
        if float(self.area_weight) < 0.0:
            raise ValueError("area_weight must be non-negative")
        if float(self.min_normalized_area) < 0.0:
            raise ValueError("min_normalized_area must be non-negative")
        if float(self.max_normalized_area) <= 0.0:
            raise ValueError("max_normalized_area must be positive")
        if float(self.max_normalized_area) < float(self.min_normalized_area):
            raise ValueError("max_normalized_area must be >= min_normalized_area")
        if float(self.init_regularization_weight) < 0.0:
            raise ValueError("init_regularization_weight must be non-negative")
        if not bool(self.optimize_rotation) and not bool(self.optimize_translation):
            raise ValueError("at least one of optimize_rotation/optimize_translation must be true")


@dataclass(frozen=True)
class PatchOverlapResult:
    success: bool
    pose_w2c: np.ndarray
    initial_energy: float
    final_energy: float
    gt_energy: float | None = None
    iterations: int = 0
    mean_containment: float = 0.0


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for patch-overlap optimization") from exc
    return torch


def _skew(vec):
    torch = _torch()
    x, y, z = vec[0], vec[1], vec[2]
    zero = torch.zeros((), dtype=vec.dtype, device=vec.device)
    return torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        ),
        dim=0,
    )


def _so3_exp(omega):
    torch = _torch()
    theta = torch.linalg.norm(omega)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    K = _skew(omega)
    theta2 = theta * theta
    small = theta < 1e-8
    a = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / torch.clamp(theta, min=1e-12))
    b = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / torch.clamp(theta2, min=1e-12))
    return eye + a * K + b * (K @ K)


def _se3_left_update(xi, init_pose):
    torch = _torch()
    rotation_delta = _so3_exp(xi[:3])
    translation_delta = xi[3:6]
    init = torch.as_tensor(init_pose, dtype=xi.dtype, device=xi.device)
    pose = torch.eye(4, dtype=xi.dtype, device=xi.device)
    pose[:3, :3] = rotation_delta @ init[:3, :3]
    pose[:3, 3] = rotation_delta @ init[:3, 3] + translation_delta
    return pose


def _camera_params(camera: ColmapCamera) -> tuple[float, float, float, float, float]:
    if camera.model_id == 0:
        f, cx, cy = camera.params[:3]
        return float(f), float(f), float(cx), float(cy), 0.0
    if camera.model_id == 1:
        fx, fy, cx, cy = camera.params[:4]
        return float(fx), float(fy), float(cx), float(cy), 0.0
    if camera.model_id == 2:
        f, cx, cy, k = camera.params[:4]
        return float(f), float(f), float(cx), float(cy), float(k)
    raise ValueError(f"unsupported camera model id: {camera.model_id}")


def _project_points(points, pose, camera: ColmapCamera):
    torch = _torch()
    fx, fy, cx, cy, k = _camera_params(camera)
    cam = points @ pose[:3, :3].T + pose[:3, 3]
    z = cam[:, 2]
    safe_z = torch.clamp(z, min=1e-6)
    x = cam[:, 0] / safe_z
    y = cam[:, 1] / safe_z
    if abs(float(k)) > 0.0:
        r2 = x * x + y * y
        scale = 1.0 + float(k) * r2
        x = x * scale
        y = y * scale
    u = torch.stack((float(fx) * x + float(cx), float(fy) * y + float(cy)), dim=1)
    return u, z


def _project_points_numpy(points: np.ndarray, pose_w2c: np.ndarray, camera: ColmapCamera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy, cx, cy, k = _camera_params(camera)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    cam = points @ pose[:3, :3].T + pose[:3, 3]
    z = cam[:, 2]
    safe_z = np.maximum(z, 1e-6)
    x = cam[:, 0] / safe_z
    y = cam[:, 1] / safe_z
    if abs(float(k)) > 0.0:
        r2 = x * x + y * y
        scale = 1.0 + float(k) * r2
        x = x * scale
        y = y * scale
    xy = np.stack((float(fx) * x + float(cx), float(fy) * y + float(cy)), axis=1)
    visible = z > 1e-6
    visible &= (xy[:, 0] >= 0.0) & (xy[:, 0] <= float(camera.width - 1))
    visible &= (xy[:, 1] >= 0.0) & (xy[:, 1] <= float(camera.height - 1))
    return xy.astype(np.float64), z.astype(np.float64), visible.astype(bool)


def _soft_patch_indicator(projected_xy, patch: TokenPatchBox, sigma: float):
    torch = _torch()
    hx = max(float(patch.x1) - float(patch.x0), 1e-6) * 0.5
    hy = max(float(patch.y1) - float(patch.y0), 1e-6) * 0.5
    cx = float(patch.center[0])
    cy = float(patch.center[1])
    px = (projected_xy[:, 0] - cx) / hx
    py = (projected_xy[:, 1] - cy) / hy
    dx = torch.clamp(torch.abs(px) - 1.0, min=0.0)
    dy = torch.clamp(torch.abs(py) - 1.0, min=0.0)
    d2 = dx * dx + dy * dy
    return torch.exp(-d2 / (2.0 * float(sigma) * float(sigma)))


def _normalized_patch_xy(projected_xy, patch: TokenPatchBox):
    torch = _torch()
    hx = max(float(patch.x1) - float(patch.x0), 1e-6) * 0.5
    hy = max(float(patch.y1) - float(patch.y0), 1e-6) * 0.5
    cx = float(patch.center[0])
    cy = float(patch.center[1])
    return torch.stack(((projected_xy[:, 0] - cx) / hx, (projected_xy[:, 1] - cy) / hy), dim=1)


def _support_shape_terms(projected_xy, weights, patch: TokenPatchBox, config: PatchOverlapConfig):
    torch = _torch()
    normalized = _normalized_patch_xy(projected_xy, patch)
    mean = torch.sum(weights[:, None] * normalized, dim=0)
    center_loss = torch.zeros((), dtype=projected_xy.dtype, device=projected_xy.device)
    if float(config.center_weight) > 0.0:
        if float(config.center_inner_fraction) > 0.0:
            outside = torch.clamp(torch.abs(mean) - float(config.center_inner_fraction), min=0.0)
            center_loss = torch.sum(outside * outside)
        else:
            center_loss = torch.sum(mean * mean)

    area_loss = torch.zeros((), dtype=projected_xy.dtype, device=projected_xy.device)
    if float(config.area_weight) > 0.0 and normalized.shape[0] >= 2:
        centered = normalized - mean[None, :]
        cov = centered.T @ (centered * weights[:, None])
        eye = torch.eye(2, dtype=projected_xy.dtype, device=projected_xy.device)
        area = torch.sqrt(torch.clamp(torch.det(cov + float(config.epsilon) * eye), min=float(config.epsilon)))
        below = torch.clamp(float(config.min_normalized_area) - area, min=0.0)
        above = torch.clamp(area - float(config.max_normalized_area), min=0.0)
        area_loss = below * below + above * above
    return center_loss, area_loss


def _group_matches_by_patch(
    matches: Sequence[QueryTo3DMatch],
    config: PatchOverlapConfig,
) -> dict[int, list[QueryTo3DMatch]]:
    grouped: dict[int, list[QueryTo3DMatch]] = {}
    for match in matches:
        grouped.setdefault(int(match.token_index), []).append(match)
    capped = {}
    for token_index, values in grouped.items():
        ordered = sorted(
            values,
            key=lambda item: (
                0.0 if item.landmark_quality is None else float(item.landmark_quality),
                int(item.observation_count or 0),
                float(item.similarity),
            ),
            reverse=True,
        )
        capped[int(token_index)] = ordered[: int(config.max_anchors_per_patch)]
    return capped


def _support_for_match(
    match: QueryTo3DMatch,
    supports: Mapping[int, PatchOverlapSupport],
) -> PatchOverlapSupport:
    support = supports.get(int(match.track_id))
    if support is not None:
        return support
    return PatchOverlapSupport(
        points_xyz=np.asarray(match.xyz, dtype=np.float64).reshape(1, 3),
        weights=np.ones((1,), dtype=np.float64),
    )


def _patch_overlap_loss(
    matches: Sequence[QueryTo3DMatch],
    patch_boxes: Mapping[int, TokenPatchBox],
    supports: Mapping[int, PatchOverlapSupport],
    camera: ColmapCamera,
    pose,
    config: PatchOverlapConfig,
):
    torch = _torch()
    grouped = _group_matches_by_patch(matches, config)
    losses = []
    containments = []
    depth_terms = []
    center_terms = []
    area_terms = []
    for token_index, patch_matches in grouped.items():
        patch = patch_boxes.get(int(token_index))
        if patch is None:
            continue
        patch_losses = []
        for match in patch_matches:
            support = _support_for_match(match, supports)
            points = torch.as_tensor(support.points_xyz, dtype=pose.dtype, device=pose.device)
            weights = torch.as_tensor(support.weights, dtype=pose.dtype, device=pose.device)
            weights = weights / torch.clamp(torch.sum(weights), min=1e-12)
            projected_xy, z = _project_points(points, pose, camera)
            indicator = _soft_patch_indicator(projected_xy, patch, float(config.sigma))
            containment = torch.sum(weights * indicator)
            containments.append(containment)
            patch_losses.append(-torch.log(containment + float(config.epsilon)))
            if float(config.depth_weight) > 0.0:
                depth_terms.append(torch.mean(torch.clamp(float(config.depth_min) - z, min=0.0) ** 2))
            center_loss, area_loss = _support_shape_terms(projected_xy, weights, patch, config)
            if float(config.center_weight) > 0.0:
                center_terms.append(center_loss)
            if float(config.area_weight) > 0.0:
                area_terms.append(area_loss)
        if patch_losses:
            losses.append(torch.stack(patch_losses).mean())
    if not losses:
        zero = torch.zeros((), dtype=pose.dtype, device=pose.device)
        return zero, zero
    loss = torch.stack(losses).sum()
    if depth_terms and float(config.depth_weight) > 0.0:
        loss = loss + float(config.depth_weight) * torch.stack(depth_terms).mean()
    if center_terms and float(config.center_weight) > 0.0:
        loss = loss + float(config.center_weight) * torch.stack(center_terms).mean()
    if area_terms and float(config.area_weight) > 0.0:
        loss = loss + float(config.area_weight) * torch.stack(area_terms).mean()
    mean_containment = torch.stack(containments).mean() if containments else torch.zeros((), dtype=pose.dtype, device=pose.device)
    return loss, mean_containment


def patch_overlap_energy_numpy(
    matches: Sequence[QueryTo3DMatch],
    patch_boxes: Mapping[int, TokenPatchBox],
    supports: Mapping[int, PatchOverlapSupport],
    camera: ColmapCamera,
    pose_w2c: np.ndarray,
    config: PatchOverlapConfig | None = None,
) -> float:
    torch = _torch()
    cfg = config or PatchOverlapConfig(iterations=0)
    pose = torch.as_tensor(np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4), dtype=torch.float64, device=cfg.device)
    with torch.no_grad():
        loss, _containment = _patch_overlap_loss(matches, patch_boxes, supports, camera, pose, cfg)
    return float(loss.detach().cpu())


def patch_overlap_pose_optimize(
    matches: Sequence[QueryTo3DMatch],
    patch_boxes: Mapping[int, TokenPatchBox],
    supports: Mapping[int, PatchOverlapSupport],
    camera: ColmapCamera,
    init_pose_w2c: np.ndarray,
    config: PatchOverlapConfig | None = None,
    gt_pose_w2c: np.ndarray | None = None,
) -> PatchOverlapResult:
    torch = _torch()
    cfg = config or PatchOverlapConfig()
    init_pose = np.asarray(init_pose_w2c, dtype=np.float64).reshape(4, 4)
    if not matches:
        return PatchOverlapResult(False, init_pose.copy(), 0.0, 0.0, iterations=0)
    xi = torch.zeros((6,), dtype=torch.float64, device=cfg.device, requires_grad=True)
    xi_mask = torch.ones((6,), dtype=torch.float64, device=cfg.device)
    if not bool(cfg.optimize_rotation):
        xi_mask[:3] = 0.0
    if not bool(cfg.optimize_translation):
        xi_mask[3:6] = 0.0
    optimizer = torch.optim.Adam([xi], lr=float(cfg.lr))
    init_pose_t = torch.as_tensor(init_pose, dtype=torch.float64, device=cfg.device)
    with torch.no_grad():
        initial_loss, initial_containment = _patch_overlap_loss(matches, patch_boxes, supports, camera, init_pose_t, cfg)
    best_pose = init_pose.copy()
    best_energy = float(initial_loss.detach().cpu())
    best_containment = float(initial_containment.detach().cpu())
    for _iter in range(int(cfg.iterations)):
        optimizer.zero_grad()
        active_xi = xi * xi_mask
        pose = _se3_left_update(active_xi, init_pose)
        loss, containment = _patch_overlap_loss(matches, patch_boxes, supports, camera, pose, cfg)
        if float(cfg.init_regularization_weight) > 0.0:
            loss = loss + float(cfg.init_regularization_weight) * torch.sum(active_xi * active_xi)
        loss.backward()
        optimizer.step()
        energy = float(loss.detach().cpu())
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_containment = float(containment.detach().cpu())
            best_pose = _se3_left_update((xi.detach() * xi_mask), init_pose).detach().cpu().numpy()
    gt_energy = None
    if gt_pose_w2c is not None:
        gt_energy = patch_overlap_energy_numpy(matches, patch_boxes, supports, camera, gt_pose_w2c, cfg)
    return PatchOverlapResult(
        success=bool(np.isfinite(best_energy)),
        pose_w2c=best_pose,
        initial_energy=float(initial_loss.detach().cpu()),
        final_energy=float(best_energy),
        gt_energy=gt_energy,
        iterations=int(cfg.iterations),
        mean_containment=float(best_containment),
    )


def build_vfm_2dgs_patch_overlap_supports(
    anchor_map: Vfm2DgsAnchorMap,
    surface_elements: SurfaceElementMap,
    max_samples_per_anchor: int = 32,
) -> dict[int, PatchOverlapSupport]:
    """Build track-id keyed support samples for VFM-2DGS semidense anchors.

    View-bin semidense export uses track id ``-100_000_000 - anchor_id`` for
    all descriptors from the same 3D anchor. We also register the raw anchor id
    for diagnostic callers that operate directly on the anchor map.
    """

    max_samples = max(int(max_samples_per_anchor), 1)
    row_by_element = {int(element_id): int(row) for row, element_id in enumerate(surface_elements.element_ids.tolist())}
    supports: dict[int, PatchOverlapSupport] = {}
    for anchor_row, anchor_id in enumerate(anchor_map.anchor_ids.astype(np.int64).tolist()):
        start = int(anchor_map.support_offsets[anchor_row])
        end = int(anchor_map.support_offsets[anchor_row + 1])
        element_ids = anchor_map.support_element_ids[start:end].astype(np.int64, copy=False)
        weights = anchor_map.support_weights[start:end].astype(np.float64, copy=False)
        row_weight_pairs = [
            (row_by_element[int(element_id)], float(weight))
            for element_id, weight in zip(element_ids.tolist(), weights.tolist())
            if int(element_id) in row_by_element
        ]
        rows = [pair[0] for pair in row_weight_pairs]
        if not rows:
            points = np.asarray(anchor_map.centers[anchor_row], dtype=np.float64).reshape(1, 3)
            support_weights = np.ones((1,), dtype=np.float64)
        else:
            rows_arr = np.asarray(rows, dtype=np.int64)
            valid_weights = np.asarray([pair[1] for pair in row_weight_pairs], dtype=np.float64)
            if rows_arr.shape[0] > max_samples:
                order = np.argsort(-valid_weights, kind="mergesort")[:max_samples]
                rows_arr = rows_arr[order]
                valid_weights = valid_weights[order]
            points = np.asarray(surface_elements.centers[rows_arr], dtype=np.float64)
            support_weights = np.asarray(valid_weights, dtype=np.float64)
            if float(np.sum(support_weights)) <= 1e-12:
                support_weights = np.ones((points.shape[0],), dtype=np.float64)
        support = PatchOverlapSupport(points_xyz=points, weights=support_weights)
        supports[int(anchor_id)] = support
        supports[int(-100_000_000 - int(anchor_id))] = support
    return supports


def oracle_patch_overlap_matches(
    index: LandmarkMapIndex,
    supports: Mapping[int, PatchOverlapSupport],
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    token_width: int,
    token_height: int,
    patch_scale: float = 1.0,
    min_support_containment: float = 0.5,
    max_per_token: int = 1,
) -> tuple[list[QueryTo3DMatch], dict[int, TokenPatchBox]]:
    """Build GT support-to-patch oracle matches for patch-overlap pose tests.

    Unlike centroid-based patch positives, this assigns an anchor to the query
    patch that contains the largest weighted fraction of its projected 2DGS
    support samples under the GT pose.
    """

    if int(max_per_token) <= 0:
        raise ValueError("max_per_token must be positive")
    if not 0.0 <= float(min_support_containment) <= 1.0:
        raise ValueError("min_support_containment must be in [0, 1]")
    boxes = token_patch_boxes(
        int(token_width),
        int(token_height),
        int(camera.width),
        int(camera.height),
        scale=float(patch_scale),
    )
    patch_boxes = {int(box.token_index): box for box in boxes}
    if len(index) == 0:
        return [], patch_boxes
    candidates_by_token: dict[int, list[tuple[float, QueryTo3DMatch]]] = {}
    for row, track_id in enumerate(index.track_ids.astype(np.int64).tolist()):
        support = supports.get(int(track_id))
        if support is None:
            support = PatchOverlapSupport(
                points_xyz=np.asarray(index.xyz[row], dtype=np.float64).reshape(1, 3),
                weights=np.ones((1,), dtype=np.float64),
            )
        xy, _z, visible = _project_points_numpy(support.points_xyz, pose_w2c, camera)
        if not np.any(visible):
            continue
        weights = np.asarray(support.weights, dtype=np.float64).reshape(-1)
        weights = np.where(visible, np.clip(weights, 0.0, None), 0.0)
        total_weight = float(np.sum(weights))
        if total_weight <= 1e-12:
            continue
        token_weights: dict[int, float] = {}
        for point_xy, weight in zip(xy, weights):
            if float(weight) <= 0.0:
                continue
            x_idx = int(
                round(
                    np.clip(point_xy[0] / max(float(camera.width - 1), 1.0), 0.0, 1.0)
                    * max(int(token_width) - 1, 0)
                )
            )
            y_idx = int(
                round(
                    np.clip(point_xy[1] / max(float(camera.height - 1), 1.0), 0.0, 1.0)
                    * max(int(token_height) - 1, 0)
                )
            )
            for yy in range(max(0, y_idx - 1), min(int(token_height), y_idx + 2)):
                for xx in range(max(0, x_idx - 1), min(int(token_width), x_idx + 2)):
                    token_index = yy * int(token_width) + xx
                    patch = patch_boxes[token_index]
                    if patch.contains(point_xy):
                        token_weights[token_index] = token_weights.get(token_index, 0.0) + float(weight)
        if not token_weights:
            continue
        token_index, best_weight = max(token_weights.items(), key=lambda item: item[1])
        containment = float(best_weight / total_weight)
        if containment < float(min_support_containment):
            continue
        patch = patch_boxes[int(token_index)]
        candidates_by_token.setdefault(int(token_index), []).append(
            (
                containment,
                QueryTo3DMatch(
                    token_index=int(token_index),
                    xy=np.asarray(patch.center, dtype=np.float64).reshape(2),
                    track_id=int(track_id),
                    xyz=np.asarray(index.xyz[row], dtype=np.float64).reshape(3),
                    similarity=containment,
                    ratio=0.0,
                    landmark_variance=float(index.mean_variances[row]),
                    source="oracle_patch_overlap",
                    observation_count=int(index.observation_counts[row]),
                    landmark_reprojection_error=None
                    if index.reprojection_errors is None
                    else float(index.reprojection_errors[row]),
                    landmark_ambiguity=None
                    if index.feature_ambiguities is None
                    else float(index.feature_ambiguities[row]),
                    similarity_margin=containment,
                ),
            )
        )
    matches: list[QueryTo3DMatch] = []
    for token_index in sorted(candidates_by_token):
        ordered = sorted(candidates_by_token[int(token_index)], key=lambda item: item[0], reverse=True)
        matches.extend(match for _score, match in ordered[: int(max_per_token)])
    return matches, patch_boxes
