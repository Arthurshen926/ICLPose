"""Pose-conditioned sparse VFM surface likelihood.

This is deliberately not a retrieval-posterior score.  A candidate pose first
renders the physical child surfaces into the query token grid.  The canonical
map descriptor selected by that rendering is then compared with the VFM query
descriptor at the *same* token.  Consequently, changing pose changes which
map feature is observed at each query location.

The map still stores one canonical primitive field.  Parent/child descriptors
are regenerable readouts of that field and are only runtime caches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera

from .physical_map import GoalMapletPhysicalMap


@dataclass(frozen=True)
class SparseVFMPoseEvidence:
    scores: np.ndarray
    raw_cosine_scores: np.ndarray
    marginal_centered_scores: np.ndarray
    log_partition_llr_scores: np.ndarray
    context_scores: np.ndarray
    local_scores: np.ndarray
    rendered_coverage: np.ndarray
    feature_coverage: np.ndarray


@dataclass(frozen=True)
class SparsePrimitiveVFMEvidence:
    scores: np.ndarray
    fixed_grid_scores: np.ndarray
    visible_sample_mean_scores: np.ndarray
    rendered_coverage: np.ndarray
    feature_coverage: np.ndarray
    sample_count: int
    primitives_per_child: int


def _camera_parameters(camera: ColmapCamera) -> tuple[float, float, float, float, float]:
    if int(camera.model_id) == 0:
        focal, cx, cy = camera.params[:3]
        return float(focal), float(focal), float(cx), float(cy), 0.0
    if int(camera.model_id) == 1:
        fx, fy, cx, cy = camera.params[:4]
        return float(fx), float(fy), float(cx), float(cy), 0.0
    if int(camera.model_id) == 2:
        focal, cx, cy, radial = camera.params[:4]
        return float(focal), float(focal), float(cx), float(cy), float(radial)
    raise ValueError(f"unsupported camera model id for sparse VFM render: {camera.model_id}")


def _normalized_tensor(values, torch, device):
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    return tensor / torch.clamp(torch.linalg.vector_norm(tensor, dim=-1, keepdim=True), min=1e-8)


def _marginal_log_partition(query, map_descriptor, temperature: float, torch) -> object:
    """Log E_map exp(cosine/tau), evaluated without a learned scene prior."""

    value = query @ map_descriptor.T / max(float(temperature), 1e-6)
    return torch.logsumexp(value, dim=1) - float(np.log(max(int(map_descriptor.shape[0]), 1)))


def _render_child_owner_batch(
    physical: GoalMapletPhysicalMap,
    poses_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    token_height: int,
    token_width: int,
    device: str,
    maximum_splat_radius_tokens: int = 2,
):
    """Batched conservative child z-buffer at VFM token resolution.

    The exact primitive renderer remains the final verifier.  This bounded
    child-surface raster is a high-recall screen for thousands of poses.  All
    children participate in the z-buffer, including children without a map
    code, so missing field support cannot reveal an occluded surface.
    """

    import torch

    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    pose = torch.as_tensor(poses_w2c, dtype=torch.float32, device=torch_device)
    center = torch.as_tensor(physical.child_centers, dtype=torch.float32, device=torch_device)
    normal = torch.as_tensor(physical.child_normals, dtype=torch.float32, device=torch_device)
    extent = torch.as_tensor(physical.child_extents, dtype=torch.float32, device=torch_device)
    parent = torch.as_tensor(physical.child_parent_rows, dtype=torch.long, device=torch_device)
    sidedness = torch.as_tensor(physical.maplet_sidedness, dtype=torch.uint8, device=torch_device)
    batch_size, child_count = int(pose.shape[0]), int(center.shape[0])
    pixel_count = int(token_height) * int(token_width)
    rotation = pose[:, :3, :3]
    translation = pose[:, :3, 3]
    camera_xyz = torch.einsum("bij,nj->bni", rotation, center) + translation[:, None, :]
    depth = camera_xyz[:, :, 2]
    normalized_xy = camera_xyz[:, :, :2] / torch.clamp(depth[:, :, None], min=1e-6)
    fx, fy, cx, cy, radial = _camera_parameters(camera)
    if radial != 0.0:
        radius_squared = torch.sum(torch.square(normalized_xy), dim=2)
        normalized_xy = normalized_xy * (1.0 + float(radial) * radius_squared)[:, :, None]
    projected_x = (float(fx) * normalized_xy[:, :, 0] + float(cx)) * (
        float(token_width) / float(camera.width)
    )
    projected_y = (float(fy) * normalized_xy[:, :, 1] + float(cy)) * (
        float(token_height) / float(camera.height)
    )

    camera_center = -torch.einsum("bji,bj->bi", rotation, translation)
    view = camera_center[:, None, :] - center[None, :, :]
    view = view / torch.clamp(torch.linalg.vector_norm(view, dim=2, keepdim=True), min=1e-8)
    incidence = torch.sum(normal[None, :, :] * view, dim=2)
    front = (sidedness[parent][None, :] == 2) | (incidence >= 0.02)
    focal = 0.5 * (float(fx) + float(fy))
    radius_world = torch.clamp(torch.linalg.vector_norm(extent[:, :2], dim=1), min=0.02)
    radius_pixel = float(focal) * radius_world[None, :] / torch.clamp(depth, min=1e-4)
    radius_token = radius_pixel * 0.5 * (
        float(token_width) / float(camera.width) + float(token_height) / float(camera.height)
    )
    radius_token = torch.clamp(
        radius_token, min=0.6, max=float(max(int(maximum_splat_radius_tokens), 1)),
    )
    valid_child = front & (depth > 0.05)

    maximum_radius = max(int(maximum_splat_radius_tokens), 1)
    offset_y, offset_x = torch.meshgrid(
        torch.arange(-maximum_radius, maximum_radius + 1, device=torch_device),
        torch.arange(-maximum_radius, maximum_radius + 1, device=torch_device),
        indexing="ij",
    )
    offset_x = offset_x.reshape(1, 1, -1)
    offset_y = offset_y.reshape(1, 1, -1)
    base_x = torch.floor(projected_x).to(torch.long)[:, :, None]
    base_y = torch.floor(projected_y).to(torch.long)[:, :, None]
    pixel_x = base_x + offset_x
    pixel_y = base_y + offset_y
    dx = (pixel_x.to(torch.float32) + 0.5 - projected_x[:, :, None]) / radius_token[:, :, None]
    dy = (pixel_y.to(torch.float32) + 0.5 - projected_y[:, :, None]) / radius_token[:, :, None]
    inside = (torch.square(dx) + torch.square(dy)) <= 1.0
    valid = (
        valid_child[:, :, None] & inside
        & (pixel_x >= 0) & (pixel_x < int(token_width))
        & (pixel_y >= 0) & (pixel_y < int(token_height))
    )
    batch_offset = torch.arange(batch_size, device=torch_device)[:, None, None] * pixel_count
    global_pixel = batch_offset + pixel_y * int(token_width) + pixel_x
    flat_valid = valid.reshape(-1)
    selected_pixel = global_pixel.reshape(-1)[flat_valid]
    selected_depth = depth[:, :, None].expand_as(global_pixel).reshape(-1)[flat_valid]
    selected_child = torch.arange(child_count, device=torch_device)[None, :, None].expand_as(
        global_pixel
    ).reshape(-1)[flat_valid]

    zbuffer = torch.full(
        (batch_size * pixel_count,), float("inf"), dtype=torch.float32, device=torch_device,
    )
    if selected_pixel.numel():
        zbuffer.scatter_reduce_(0, selected_pixel, selected_depth, reduce="amin", include_self=True)
    # Select a deterministic child among numerically tied closest splats.
    closest = selected_depth <= zbuffer[selected_pixel] + 1e-5
    owner = torch.full(
        (batch_size * pixel_count,), child_count, dtype=torch.long, device=torch_device,
    )
    if torch.any(closest):
        owner.scatter_reduce_(
            0, selected_pixel[closest], selected_child[closest], reduce="amin", include_self=True,
        )
    owner[owner == child_count] = -1
    return owner.reshape(batch_size, pixel_count)


def score_pose_conditioned_sparse_vfm(
    poses_w2c: np.ndarray,
    query_context_descriptors: np.ndarray,
    query_local_descriptors: np.ndarray,
    map_parent_descriptors: np.ndarray,
    map_child_descriptors: np.ndarray,
    parent_descriptor_valid: np.ndarray,
    child_descriptor_valid: np.ndarray,
    physical: GoalMapletPhysicalMap,
    camera: ColmapCamera,
    *,
    token_height: int,
    token_width: int,
    temperature: float = 0.07,
    batch_size: int = 16,
    maximum_splat_radius_tokens: int = 2,
    score_semantics: str = "marginal_centered",
    allowed_parent_rows: np.ndarray | None = None,
    query_token_mask: np.ndarray | None = None,
    device: str = "cuda",
) -> SparseVFMPoseEvidence:
    """Evaluate direct same-token VFM likelihood ratios for many poses.

    For each token the null is the marginal distribution over valid map
    surfaces.  Therefore a ubiquitous feature has little evidence even when
    its raw cosine is high, while a distinctive pose-consistent feature has a
    positive log-likelihood ratio.  Missing rendered features contribute zero
    LLR and remain in the fixed full-grid denominator.
    """

    import torch

    poses = np.asarray(poses_w2c, dtype=np.float64).reshape(-1, 4, 4)
    token_count = int(token_height) * int(token_width)
    query_context = np.asarray(query_context_descriptors, dtype=np.float32)
    query_local = np.asarray(query_local_descriptors, dtype=np.float32)
    parent_map = np.asarray(map_parent_descriptors, dtype=np.float32)
    child_map = np.asarray(map_child_descriptors, dtype=np.float32)
    parent_valid = np.asarray(parent_descriptor_valid, dtype=bool).reshape(-1)
    child_valid = np.asarray(child_descriptor_valid, dtype=bool).reshape(-1)
    if (
        query_context.ndim != 2 or query_context.shape[0] != token_count
        or query_local.ndim != 2 or query_local.shape[0] != token_count
        or parent_map.ndim != 2 or parent_map.shape[0] != physical.maplet_ids.size
        or child_map.ndim != 2 or child_map.shape[0] != physical.child_centers.shape[0]
        or parent_valid.shape != (parent_map.shape[0],)
        or child_valid.shape != (child_map.shape[0],)
        or query_context.shape[1] != parent_map.shape[1]
        or query_local.shape[1] != child_map.shape[1]
        or not np.any(parent_valid) or not np.any(child_valid)
    ):
        raise ValueError("sparse VFM query/map descriptor contract differs")
    allowed = None
    if allowed_parent_rows is not None:
        allowed = np.asarray(allowed_parent_rows, dtype=np.int64)
        if allowed.ndim != 2 or allowed.shape[0] != poses.shape[0]:
            raise ValueError("allowed parent rows must have one row per pose")
        if np.any(allowed >= parent_map.shape[0]):
            raise ValueError("allowed parent row is outside the physical map")
    planned_mask = None
    if query_token_mask is not None:
        planned_mask = np.asarray(query_token_mask, dtype=bool)
        if planned_mask.shape != (poses.shape[0], token_count):
            raise ValueError("query token mask must match pose and token counts")
        if np.any(np.sum(planned_mask, axis=1) == 0):
            raise ValueError("every pose must plan at least one query token")
    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    query_context_t = _normalized_tensor(query_context, torch, torch_device)
    query_local_t = _normalized_tensor(query_local, torch, torch_device)
    parent_map_t = _normalized_tensor(parent_map, torch, torch_device)
    child_map_t = _normalized_tensor(child_map, torch, torch_device)
    valid_parent_rows = torch.as_tensor(np.flatnonzero(parent_valid), dtype=torch.long, device=torch_device)
    valid_child_rows = torch.as_tensor(np.flatnonzero(child_valid), dtype=torch.long, device=torch_device)
    context_null = _marginal_log_partition(
        query_context_t, parent_map_t[valid_parent_rows], float(temperature), torch,
    )
    local_null = _marginal_log_partition(
        query_local_t, child_map_t[valid_child_rows], float(temperature), torch,
    )
    parent_by_child = torch.as_tensor(
        physical.child_parent_rows, dtype=torch.long, device=torch_device,
    )
    parent_valid_t = torch.as_tensor(parent_valid, dtype=torch.bool, device=torch_device)
    child_valid_t = torch.as_tensor(child_valid, dtype=torch.bool, device=torch_device)

    if str(score_semantics) not in ("raw_cosine", "marginal_centered", "log_partition_llr"):
        raise ValueError(f"unknown sparse VFM score semantics: {score_semantics}")
    context_mean = torch.mean(parent_map_t[valid_parent_rows], dim=0)
    local_mean = torch.mean(child_map_t[valid_child_rows], dim=0)
    context_marginal_mean = query_context_t @ context_mean
    local_marginal_mean = query_local_t @ local_mean
    all_score, raw_score, centered_score, llr_score = [], [], [], []
    context_score, local_score, rendered_coverage, feature_coverage = [], [], [], []
    with torch.no_grad():
        for start in range(0, poses.shape[0], max(int(batch_size), 1)):
            stop = min(start + max(int(batch_size), 1), poses.shape[0])
            owner = _render_child_owner_batch(
                physical, poses[start:stop], camera,
                token_height=int(token_height), token_width=int(token_width),
                maximum_splat_radius_tokens=int(maximum_splat_radius_tokens),
                device=str(torch_device),
            )
            rendered = owner >= 0
            safe_owner = torch.clamp(owner, min=0)
            rendered_parent = parent_by_child[safe_owner]
            feature_valid = rendered & child_valid_t[safe_owner] & parent_valid_t[rendered_parent]
            if allowed is not None:
                allowed_t = torch.as_tensor(
                    allowed[start:stop], dtype=torch.long, device=torch_device,
                )
                identity_allowed = torch.any(
                    rendered_parent[:, :, None] == allowed_t[:, None, :], dim=2,
                )
                feature_valid &= identity_allowed
            if planned_mask is not None:
                planned_t = torch.as_tensor(
                    planned_mask[start:stop], dtype=torch.bool, device=torch_device,
                )
                feature_valid &= planned_t
                denominator = torch.sum(planned_t, dim=1).to(torch.float32)
                rendered_for_coverage = rendered & planned_t
            else:
                denominator = torch.full(
                    (stop - start,), float(token_count),
                    dtype=torch.float32, device=torch_device,
                )
                rendered_for_coverage = rendered
            context_cosine = torch.sum(
                query_context_t[None, :, :] * parent_map_t[rendered_parent], dim=2,
            )
            local_cosine = torch.sum(
                query_local_t[None, :, :] * child_map_t[safe_owner], dim=2,
            )
            context_llr = context_cosine / max(float(temperature), 1e-6) - context_null[None, :]
            local_llr = local_cosine / max(float(temperature), 1e-6) - local_null[None, :]
            zero = torch.zeros((), dtype=torch.float32, device=torch_device)
            context_raw = torch.where(feature_valid, context_cosine, zero)
            local_raw = torch.where(feature_valid, local_cosine, zero)
            context_centered = torch.where(
                feature_valid, context_cosine - context_marginal_mean[None, :], zero,
            )
            local_centered = torch.where(
                feature_valid, local_cosine - local_marginal_mean[None, :], zero,
            )
            context_llr = torch.where(feature_valid, context_llr, zero)
            local_llr = torch.where(feature_valid, local_llr, zero)
            raw_value = 0.5 * (
                torch.sum(context_raw, dim=1) + torch.sum(local_raw, dim=1)
            ) / denominator
            centered_value = 0.5 * (
                torch.sum(context_centered, dim=1) + torch.sum(local_centered, dim=1)
            ) / denominator
            context_value = torch.sum(context_llr, dim=1) / denominator
            local_value = torch.sum(local_llr, dim=1) / denominator
            # The two readout roles are equally scaled log evidence from the
            # same canonical field.  Averaging prevents double-counting while
            # retaining both VFM context and child-scale spatial evidence.
            llr_value = 0.5 * (context_value + local_value)
            combined = {
                "raw_cosine": raw_value,
                "marginal_centered": centered_value,
                "log_partition_llr": llr_value,
            }[str(score_semantics)]
            all_score.append(combined.cpu().numpy())
            raw_score.append(raw_value.cpu().numpy())
            centered_score.append(centered_value.cpu().numpy())
            llr_score.append(llr_value.cpu().numpy())
            context_score.append(context_value.cpu().numpy())
            local_score.append(local_value.cpu().numpy())
            rendered_coverage.append(
                (torch.sum(rendered_for_coverage, dim=1) / denominator).cpu().numpy()
            )
            feature_coverage.append(
                (torch.sum(feature_valid, dim=1) / denominator).cpu().numpy()
            )
    return SparseVFMPoseEvidence(
        scores=np.concatenate(all_score).astype(np.float64) if all_score else np.zeros((0,)),
        raw_cosine_scores=np.concatenate(raw_score).astype(np.float64) if raw_score else np.zeros((0,)),
        marginal_centered_scores=(
            np.concatenate(centered_score).astype(np.float64) if centered_score else np.zeros((0,))
        ),
        log_partition_llr_scores=(
            np.concatenate(llr_score).astype(np.float64) if llr_score else np.zeros((0,))
        ),
        context_scores=np.concatenate(context_score).astype(np.float64) if context_score else np.zeros((0,)),
        local_scores=np.concatenate(local_score).astype(np.float64) if local_score else np.zeros((0,)),
        rendered_coverage=np.concatenate(rendered_coverage).astype(np.float64) if rendered_coverage else np.zeros((0,)),
        feature_coverage=np.concatenate(feature_coverage).astype(np.float64) if feature_coverage else np.zeros((0,)),
    )


def _stratified_primitive_samples(
    physical: GoalMapletPhysicalMap,
    field_primitive_rows: np.ndarray,
    field_confidence: np.ndarray,
    *,
    primitives_per_child: int,
) -> np.ndarray:
    """Choose spatially spread real primitive rows inside every child.

    This is a runtime index over the sole canonical field, not another map
    embedding.  Farthest-point sampling in the child tangent plane preserves
    within-maplet VFM phase that descriptor averaging destroys.
    """

    from .surface_renderer import dominant_child_owner

    field_rows = np.asarray(field_primitive_rows, dtype=np.int64).reshape(-1)
    confidence = np.asarray(field_confidence, dtype=np.float64).reshape(-1)
    if field_rows.shape != confidence.shape:
        raise ValueError("canonical primitive confidence differs")
    owner = dominant_child_owner(physical)[field_rows]
    order = np.argsort(owner, kind="stable")
    sorted_owner = owner[order]
    chosen: list[np.ndarray] = []
    maximum = max(int(primitives_per_child), 1)
    for child in np.unique(sorted_owner[sorted_owner >= 0]).tolist():
        start = int(np.searchsorted(sorted_owner, child, side="left"))
        stop = int(np.searchsorted(sorted_owner, child, side="right"))
        field_indices = order[start:stop]
        if field_indices.size <= maximum:
            chosen.append(field_indices)
            continue
        primitive_rows = field_rows[field_indices]
        delta = physical.primitive_centers[primitive_rows] - physical.child_centers[int(child)]
        frame = physical.child_frames[int(child)]
        uv = np.stack((delta @ frame[:, 0], delta @ frame[:, 1]), axis=1)
        scale = np.maximum(np.asarray(physical.child_extents[int(child), :2]), 1e-4)
        uv /= scale[None, :]
        quality = (
            np.maximum(confidence[field_indices], 1e-4)
            * np.maximum(physical.primitive_opacity[primitive_rows], 1e-4)
            * np.sqrt(np.maximum(
                physical.primitive_scale1[primitive_rows]
                * physical.primitive_scale2[primitive_rows],
                1e-10,
            ))
        )
        quality /= max(float(np.max(quality)), 1e-8)
        selected = [int(np.argmax(quality))]
        minimum_distance = np.sum(np.square(uv - uv[selected[0]]), axis=1)
        for _ in range(1, maximum):
            score = minimum_distance * (0.5 + 0.5 * quality)
            score[np.asarray(selected, dtype=np.int64)] = -1.0
            next_row = int(np.argmax(score))
            selected.append(next_row)
            minimum_distance = np.minimum(
                minimum_distance,
                np.sum(np.square(uv - uv[next_row]), axis=1),
            )
        chosen.append(field_indices[np.asarray(selected, dtype=np.int64)])
    return (
        np.concatenate(chosen).astype(np.int64)
        if chosen else np.zeros((0,), dtype=np.int64)
    )


def _render_primitive_sample_owner_batch(
    physical: GoalMapletPhysicalMap,
    primitive_rows: np.ndarray,
    poses_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    token_height: int,
    token_width: int,
    device: str,
    maximum_splat_radius_tokens: int,
):
    import torch

    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    rows = np.asarray(primitive_rows, dtype=np.int64).reshape(-1)
    pose = torch.as_tensor(poses_w2c, dtype=torch.float32, device=torch_device)
    center = torch.as_tensor(physical.primitive_centers[rows], dtype=torch.float32, device=torch_device)
    normal = torch.as_tensor(physical.primitive_normals[rows], dtype=torch.float32, device=torch_device)
    sidedness = torch.as_tensor(
        physical.primitive_sidedness[rows], dtype=torch.uint8, device=torch_device,
    )
    scale1 = torch.as_tensor(physical.primitive_scale1[rows], dtype=torch.float32, device=torch_device)
    scale2 = torch.as_tensor(physical.primitive_scale2[rows], dtype=torch.float32, device=torch_device)
    batch_size, sample_count = int(pose.shape[0]), int(center.shape[0])
    pixel_count = int(token_height) * int(token_width)
    rotation, translation = pose[:, :3, :3], pose[:, :3, 3]
    camera_xyz = torch.einsum("bij,nj->bni", rotation, center) + translation[:, None, :]
    depth = camera_xyz[:, :, 2]
    normalized_xy = camera_xyz[:, :, :2] / torch.clamp(depth[:, :, None], min=1e-6)
    fx, fy, cx, cy, radial = _camera_parameters(camera)
    if radial != 0.0:
        radius_squared = torch.sum(torch.square(normalized_xy), dim=2)
        normalized_xy = normalized_xy * (1.0 + float(radial) * radius_squared)[:, :, None]
    projected_x = (float(fx) * normalized_xy[:, :, 0] + float(cx)) * (
        float(token_width) / float(camera.width)
    )
    projected_y = (float(fy) * normalized_xy[:, :, 1] + float(cy)) * (
        float(token_height) / float(camera.height)
    )
    camera_center = -torch.einsum("bji,bj->bi", rotation, translation)
    view = camera_center[:, None, :] - center[None, :, :]
    view /= torch.clamp(torch.linalg.vector_norm(view, dim=2, keepdim=True), min=1e-8)
    incidence = torch.sum(normal[None, :, :] * view, dim=2)
    front = (sidedness[None, :] == 2) | (incidence >= 0.02)
    focal = 0.5 * (float(fx) + float(fy))
    radius_world = torch.sqrt(torch.clamp(scale1 * scale2, min=1e-10))
    radius_pixel = float(focal) * radius_world[None, :] / torch.clamp(depth, min=1e-4)
    radius_token = radius_pixel * 0.5 * (
        float(token_width) / float(camera.width) + float(token_height) / float(camera.height)
    )
    valid_sample = front & (depth > 0.05)
    maximum_radius = int(maximum_splat_radius_tokens)
    if maximum_radius <= 0:
        pixel_x = torch.floor(projected_x).long()[:, :, None]
        pixel_y = torch.floor(projected_y).long()[:, :, None]
        valid = (
            valid_sample[:, :, None]
            & (pixel_x >= 0) & (pixel_x < int(token_width))
            & (pixel_y >= 0) & (pixel_y < int(token_height))
        )
    else:
        radius_token = torch.clamp(radius_token, min=0.6, max=float(maximum_radius))
        offset_y, offset_x = torch.meshgrid(
            torch.arange(-maximum_radius, maximum_radius + 1, device=torch_device),
            torch.arange(-maximum_radius, maximum_radius + 1, device=torch_device),
            indexing="ij",
        )
        offset_x, offset_y = offset_x.reshape(1, 1, -1), offset_y.reshape(1, 1, -1)
        pixel_x = torch.floor(projected_x).long()[:, :, None] + offset_x
        pixel_y = torch.floor(projected_y).long()[:, :, None] + offset_y
        dx = (pixel_x.float() + 0.5 - projected_x[:, :, None]) / radius_token[:, :, None]
        dy = (pixel_y.float() + 0.5 - projected_y[:, :, None]) / radius_token[:, :, None]
        valid = (
            valid_sample[:, :, None] & ((torch.square(dx) + torch.square(dy)) <= 1.0)
            & (pixel_x >= 0) & (pixel_x < int(token_width))
            & (pixel_y >= 0) & (pixel_y < int(token_height))
        )
    global_pixel = (
        torch.arange(batch_size, device=torch_device)[:, None, None] * pixel_count
        + pixel_y * int(token_width) + pixel_x
    )
    flat_valid = valid.reshape(-1)
    selected_pixel = global_pixel.reshape(-1)[flat_valid]
    selected_depth = depth[:, :, None].expand_as(global_pixel).reshape(-1)[flat_valid]
    selected_sample = torch.arange(sample_count, device=torch_device)[None, :, None].expand_as(
        global_pixel
    ).reshape(-1)[flat_valid]
    zbuffer = torch.full(
        (batch_size * pixel_count,), float("inf"), dtype=torch.float32, device=torch_device,
    )
    if selected_pixel.numel():
        zbuffer.scatter_reduce_(0, selected_pixel, selected_depth, reduce="amin", include_self=True)
    closest = selected_depth <= zbuffer[selected_pixel] + 1e-5
    owner = torch.full(
        (batch_size * pixel_count,), sample_count, dtype=torch.long, device=torch_device,
    )
    if torch.any(closest):
        owner.scatter_reduce_(
            0, selected_pixel[closest], selected_sample[closest], reduce="amin", include_self=True,
        )
    owner[owner == sample_count] = -1
    return owner.reshape(batch_size, pixel_count)


def score_pose_conditioned_sparse_primitives(
    poses_w2c: np.ndarray,
    query_canonical_descriptors: np.ndarray,
    field_primitive_rows: np.ndarray,
    field_codes: np.ndarray,
    field_confidence: np.ndarray,
    physical: GoalMapletPhysicalMap,
    camera: ColmapCamera,
    *,
    token_height: int,
    token_width: int,
    primitives_per_child: int = 8,
    batch_size: int = 8,
    maximum_splat_radius_tokens: int = 1,
    score_semantics: str = "visible_sample_mean",
    allowed_parent_rows: np.ndarray | None = None,
    query_token_mask: np.ndarray | None = None,
    device: str = "cuda",
) -> SparsePrimitiveVFMEvidence:
    """Approximate exact canonical rendering with spatial real-code samples."""

    import torch

    poses = np.asarray(poses_w2c, dtype=np.float64).reshape(-1, 4, 4)
    query = np.asarray(query_canonical_descriptors, dtype=np.float32)
    field_rows = np.asarray(field_primitive_rows, dtype=np.int64).reshape(-1)
    codes = np.asarray(field_codes, dtype=np.float32)
    token_count = int(token_height) * int(token_width)
    if query.shape[0] != token_count or codes.shape[0] != field_rows.size:
        raise ValueError("sparse primitive query/canonical field differs")
    if int(primitives_per_child) > 0:
        selected_field_rows = _stratified_primitive_samples(
            physical, field_rows, np.asarray(field_confidence),
            primitives_per_child=int(primitives_per_child),
        )
        if selected_field_rows.size == 0:
            raise ValueError("canonical field has no eligible primitive samples")
        selected_primitive_rows = field_rows[selected_field_rows]
        selected_codes = codes[selected_field_rows]
        field_index_by_sample = np.arange(selected_field_rows.size, dtype=np.int64)
    else:
        selected_primitive_rows = np.arange(physical.primitive_ids.size, dtype=np.int64)
        selected_codes = codes
        field_index_by_sample = np.full(
            (selected_primitive_rows.size,), -1, dtype=np.int64,
        )
        field_index_by_sample[field_rows] = np.arange(field_rows.size, dtype=np.int64)
    from .visibility import dominant_maplet_owner
    parent_by_sample = dominant_maplet_owner(physical)[selected_primitive_rows]
    allowed = None
    if allowed_parent_rows is not None:
        allowed = np.asarray(allowed_parent_rows, dtype=np.int64)
        if allowed.ndim != 2 or allowed.shape[0] != poses.shape[0]:
            raise ValueError("allowed primitive parent rows must match poses")
    planned_mask = None
    if query_token_mask is not None:
        planned_mask = np.asarray(query_token_mask, dtype=bool)
        if planned_mask.shape != (poses.shape[0], token_count):
            raise ValueError("primitive query token mask differs")
        if np.any(np.sum(planned_mask, axis=1) == 0):
            raise ValueError("every primitive pose must plan query tokens")
    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    query_t = _normalized_tensor(query, torch, torch_device)
    code_t = _normalized_tensor(selected_codes, torch, torch_device)
    field_index_t = torch.as_tensor(
        field_index_by_sample, dtype=torch.long, device=torch_device,
    )
    parent_by_sample_t = torch.as_tensor(
        parent_by_sample, dtype=torch.long, device=torch_device,
    )
    if str(score_semantics) not in ("fixed_grid", "visible_sample_mean"):
        raise ValueError(f"unknown sparse primitive score semantics: {score_semantics}")
    scores, fixed_scores, visible_mean_scores, coverage, feature_coverage = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, poses.shape[0], max(int(batch_size), 1)):
            stop = min(start + max(int(batch_size), 1), poses.shape[0])
            owner = _render_primitive_sample_owner_batch(
                physical, selected_primitive_rows, poses[start:stop], camera,
                token_height=int(token_height), token_width=int(token_width),
                device=str(torch_device),
                maximum_splat_radius_tokens=int(maximum_splat_radius_tokens),
            )
            rendered = owner >= 0
            safe_owner = torch.clamp(owner, min=0)
            field_index = field_index_t[safe_owner]
            valid = rendered & (field_index >= 0)
            if allowed is not None:
                allowed_t = torch.as_tensor(
                    allowed[start:stop], dtype=torch.long, device=torch_device,
                )
                rendered_parent = parent_by_sample_t[safe_owner]
                valid &= torch.any(
                    rendered_parent[:, :, None] == allowed_t[:, None, :], dim=2,
                )
            if planned_mask is not None:
                planned_t = torch.as_tensor(
                    planned_mask[start:stop], dtype=torch.bool, device=torch_device,
                )
                valid &= planned_t
                denominator = torch.sum(planned_t, dim=1).to(torch.float32)
                rendered_for_coverage = rendered & planned_t
            else:
                denominator = torch.full(
                    (stop - start,), float(token_count),
                    dtype=torch.float32, device=torch_device,
                )
                rendered_for_coverage = rendered
            cosine = torch.sum(
                query_t[None, :, :] * code_t[torch.clamp(field_index, min=0)], dim=2,
            )
            cosine = torch.where(valid, cosine, torch.zeros((), device=torch_device))
            fixed = torch.sum(cosine, dim=1) / denominator
            visible_mean = torch.sum(cosine, dim=1) / torch.clamp(
                torch.sum(valid, dim=1), min=1,
            )
            selected = {
                "fixed_grid": fixed,
                "visible_sample_mean": visible_mean,
            }[str(score_semantics)]
            scores.append(selected.cpu().numpy())
            fixed_scores.append(fixed.cpu().numpy())
            visible_mean_scores.append(visible_mean.cpu().numpy())
            coverage.append(
                (torch.sum(rendered_for_coverage, dim=1) / denominator).cpu().numpy()
            )
            feature_coverage.append(
                (torch.sum(valid, dim=1) / denominator).cpu().numpy()
            )
    return SparsePrimitiveVFMEvidence(
        scores=np.concatenate(scores).astype(np.float64),
        fixed_grid_scores=np.concatenate(fixed_scores).astype(np.float64),
        visible_sample_mean_scores=np.concatenate(visible_mean_scores).astype(np.float64),
        rendered_coverage=np.concatenate(coverage).astype(np.float64),
        feature_coverage=np.concatenate(feature_coverage).astype(np.float64),
        sample_count=int(selected_primitive_rows.size),
        primitives_per_child=int(primitives_per_child),
    )
