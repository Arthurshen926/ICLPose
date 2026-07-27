"""Continuous SE(3) alignment of RADIO-final query features to 2DGS surfels.

The optimizer samples the query feature field at projections of disconnected
surfel patches selected by maplet retrieval.  Those samples are a rendering of
the selected feature field, but no point is treated as a persistent matching
anchor and no PnP correspondences are formed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_feature_field import SurfaceFeatureField


@dataclass(frozen=True)
class ContinuousSurfaceAlignmentConfig:
    maximum_maplets: int = 12
    maximum_samples: int = 2048
    maximum_samples_per_maplet: int = 256
    minimum_depth_m: float = 0.10
    pyramid_scales: tuple[float, ...] = (1.0,)
    outer_iterations_per_scale: int = 3
    lbfgs_iterations: int = 20
    robust_delta: float = 0.20
    similarity_threshold: float = 0.30
    similarity_temperature: float = 0.10
    maximum_step_rotation_deg: float = 1.0
    maximum_step_translation_m: float = 0.05
    optimizer: str = "trust_region"
    trust_region_iterations: int = 12
    detector_evidence_floor: float = 0.15
    detector_evidence_weight: float = 0.35
    detector_radio_gate: float = 0.05
    minimum_confidence: float = 0.02

    def __post_init__(self) -> None:
        if (
            int(self.maximum_maplets) <= 0
            or int(self.maximum_samples) <= 0
            or int(self.maximum_samples_per_maplet) <= 0
            or int(self.outer_iterations_per_scale) <= 0
            or int(self.lbfgs_iterations) <= 0
            or float(self.minimum_depth_m) <= 0.0
            or float(self.robust_delta) <= 0.0
            or float(self.similarity_temperature) <= 0.0
            or float(self.maximum_step_rotation_deg) <= 0.0
            or float(self.maximum_step_translation_m) <= 0.0
            or int(self.trust_region_iterations) <= 0
        ):
            raise ValueError("continuous alignment limits must be positive")
        if not self.pyramid_scales or any(
            not 0.0 < float(value) <= 1.0 for value in self.pyramid_scales
        ):
            raise ValueError("pyramid scales must lie in (0, 1]")
        if str(self.optimizer) not in {"trust_region", "lbfgs"}:
            raise ValueError("optimizer must be trust_region or lbfgs")
        if not 0.0 <= float(self.detector_evidence_floor) <= 1.0:
            raise ValueError("detector_evidence_floor must be in [0, 1]")
        if float(self.detector_evidence_weight) < 0.0:
            raise ValueError("detector_evidence_weight must be non-negative")


@dataclass(frozen=True)
class SurfaceAlignmentResult:
    pose_w2c: np.ndarray
    initial_score: float
    final_score: float
    selected_maplet_ids: np.ndarray
    sample_count: int
    converged: bool
    diagnostics: dict[str, object]


def _camera_parameters(camera: ColmapCamera) -> tuple[float, float, float, float]:
    if int(camera.model_id) == 0:
        f, cx, cy = camera.params[:3]
        return float(f), float(cx), float(cy), 0.0
    if int(camera.model_id) == 1:
        fx, fy, cx, cy = camera.params[:4]
        if abs(float(fx) - float(fy)) > 1e-6:
            raise ValueError("continuous aligner currently requires square pixels")
        return float(fx), float(cx), float(cy), 0.0
    if int(camera.model_id) == 2:
        f, cx, cy, k = camera.params[:4]
        return float(f), float(cx), float(cy), float(k)
    raise ValueError(f"unsupported camera model {camera.model_id}")


def project_world_points(
    xyz_world: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with the Cambridge SIMPLE_RADIAL calibration."""

    xyz = np.asarray(xyz_world, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera_xyz[:, 2]
    denominator = np.where(
        np.abs(depth) >= 1e-6,
        depth,
        np.where(depth < 0.0, -1e-6, 1e-6),
    )
    normalized = camera_xyz[:, :2] / denominator[:, None]
    f, cx, cy, k = _camera_parameters(camera)
    radial = 1.0 + float(k) * np.sum(normalized * normalized, axis=1)
    pixels = np.stack(
        [
            float(f) * normalized[:, 0] * radial + float(cx),
            float(f) * normalized[:, 1] * radial + float(cy),
        ],
        axis=1,
    )
    return pixels.astype(np.float32), depth.astype(np.float32)


def select_visible_maplets(
    field: SurfaceFeatureField,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    maximum_maplets: int,
    minimum_depth_m: float = 0.10,
) -> np.ndarray:
    """Oracle-free visibility helper; runtime may instead pass retrieved IDs."""

    pixels, depth = project_world_points(field.centers, pose_w2c, camera)
    visible = (
        (depth > float(minimum_depth_m))
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= float(camera.width - 1))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= float(camera.height - 1))
    )
    ids, counts = np.unique(field.owner_maplet_ids[visible], return_counts=True)
    if ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-counts, kind="mergesort")[: int(maximum_maplets)]
    return ids[order].astype(np.int64, copy=False)


def select_render_samples(
    field: SurfaceFeatureField,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    maplet_ids: np.ndarray,
    *,
    feature_height: int,
    feature_width: int,
    config: ContinuousSurfaceAlignmentConfig,
) -> np.ndarray:
    """Hard visibility pass used between differentiable optimization steps."""

    eligible = np.isin(field.owner_maplet_ids, np.asarray(maplet_ids, dtype=np.int64))
    eligible &= field.confidence >= float(config.minimum_confidence)
    rows = np.flatnonzero(eligible)
    if rows.size == 0:
        return rows
    pixels, depth = project_world_points(field.centers[rows], pose_w2c, camera)
    visible = (
        (depth > float(config.minimum_depth_m))
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= float(camera.width - 1))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= float(camera.height - 1))
    )
    rows = rows[visible]
    pixels = pixels[visible]
    depth = depth[visible]
    if rows.size == 0:
        return rows

    grid_x = np.clip(
        np.rint(pixels[:, 0] * (feature_width - 1) / max(camera.width - 1, 1)),
        0,
        feature_width - 1,
    ).astype(np.int64)
    grid_y = np.clip(
        np.rint(pixels[:, 1] * (feature_height - 1) / max(camera.height - 1, 1)),
        0,
        feature_height - 1,
    ).astype(np.int64)
    cell = grid_y * int(feature_width) + grid_x
    # Frontmost surfel per feature pixel approximates a depth-buffered feature render.
    order = np.lexsort((-field.confidence[rows], depth, cell))
    sorted_cell = cell[order]
    keep = np.ones(order.size, dtype=bool)
    keep[1:] = sorted_cell[1:] != sorted_cell[:-1]
    rows = rows[order[keep]]

    balanced: list[int] = []
    per_maplet = int(config.maximum_samples_per_maplet)
    for maplet_id in np.asarray(maplet_ids, dtype=np.int64).tolist():
        local = rows[field.owner_maplet_ids[rows] == int(maplet_id)]
        if local.size > per_maplet:
            order = np.argsort(-field.confidence[local], kind="mergesort")[:per_maplet]
            local = local[order]
        balanced.extend(local.tolist())
    selected = np.asarray(balanced, dtype=np.int64)
    if selected.size > int(config.maximum_samples):
        order = np.argsort(-field.confidence[selected], kind="mergesort")
        selected = selected[order[: int(config.maximum_samples)]]
    return selected


def build_detector_heatmap(
    xy: np.ndarray,
    scores: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    output_width: int | None = None,
    output_height: int | None = None,
    sigma_px: float = 4.0,
) -> np.ndarray:
    """Rasterize detector responses only; no local descriptors are accepted."""

    points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if points.shape[0] != values.shape[0]:
        raise ValueError("detector points and scores must align")
    width = int(output_width or image_width)
    height = int(output_height or image_height)
    if width <= 0 or height <= 0 or float(sigma_px) <= 0.0:
        raise ValueError("detector heatmap dimensions and sigma must be positive")
    heat = np.zeros((height, width), dtype=np.float32)
    x = np.clip(
        np.rint(points[:, 0] * (width - 1) / max(int(image_width) - 1, 1)),
        0,
        width - 1,
    ).astype(np.int64)
    y = np.clip(
        np.rint(points[:, 1] * (height - 1) / max(int(image_height) - 1, 1)),
        0,
        height - 1,
    ).astype(np.int64)
    normalized_score = values - float(np.min(values, initial=0.0))
    scale = float(np.quantile(normalized_score, 0.90)) if values.size else 1.0
    normalized_score = np.clip(normalized_score / max(scale, 1e-8), 0.0, 1.0)
    np.maximum.at(heat, (y, x), normalized_score)
    import cv2

    sigma = float(sigma_px) * float(width) / max(float(image_width), 1.0)
    heat = cv2.GaussianBlur(
        heat,
        (0, 0),
        sigmaX=max(sigma, 0.5),
        sigmaY=max(sigma, 0.5),
        borderType=cv2.BORDER_REPLICATE,
    )
    maximum = float(np.max(heat, initial=0.0))
    if maximum > 0.0:
        heat /= maximum
    return heat.astype(np.float32, copy=False)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind()
    zero = x * 0.0
    return torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )


def _so3_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    theta2 = torch.sum(rotation_vector * rotation_vector)
    theta = torch.sqrt(torch.clamp(theta2, min=1e-16))
    safe_theta = torch.clamp(theta, min=1e-8)
    safe_theta2 = torch.clamp(theta2, min=1e-16)
    skew = _skew(rotation_vector)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    a = torch.where(
        theta < 1e-4,
        1.0 - theta2 / 6.0,
        torch.sin(theta) / safe_theta,
    )
    b = torch.where(
        theta < 1e-4,
        0.5 - theta2 / 24.0,
        (1.0 - torch.cos(theta)) / safe_theta2,
    )
    return identity + a * skew + b * (skew @ skew)


def _compose_delta(delta: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
    rotation = _so3_exp(delta[:3])
    output = torch.eye(4, dtype=delta.dtype, device=delta.device)
    output[:3, :3] = rotation
    output[:3, 3] = delta[3:]
    return output @ base_pose


def _alignment_loss(
    delta: torch.Tensor,
    base_pose: torch.Tensor,
    xyz: torch.Tensor,
    map_features: torch.Tensor,
    weights: torch.Tensor,
    query_feature: torch.Tensor,
    camera: ColmapCamera,
    robust_delta: float,
    similarity_threshold: float = 0.30,
    similarity_temperature: float = 0.10,
    detector_heatmap: torch.Tensor | None = None,
    detector_evidence_floor: float = 0.15,
    detector_evidence_weight: float = 0.35,
    detector_radio_gate: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose = _compose_delta(delta, base_pose)
    camera_xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera_xyz[:, 2]
    normalized = camera_xyz[:, :2] / torch.clamp(depth[:, None], min=1e-4)
    f, cx, cy, k = _camera_parameters(camera)
    radial = 1.0 + float(k) * torch.sum(normalized * normalized, dim=1)
    pixel_x = float(f) * normalized[:, 0] * radial + float(cx)
    pixel_y = float(f) * normalized[:, 1] * radial + float(cy)
    grid = torch.stack(
        [
            2.0 * pixel_x / max(int(camera.width) - 1, 1) - 1.0,
            2.0 * pixel_y / max(int(camera.height) - 1, 1) - 1.0,
        ],
        dim=1,
    )
    sampled = F.grid_sample(
        query_feature[None],
        grid.reshape(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, :, 0, :].T
    sampled = F.normalize(sampled, p=2, dim=1, eps=1e-8)
    cosine = torch.sum(sampled * map_features, dim=1)
    valid = (
        (depth > 0.05)
        & (torch.abs(grid[:, 0]) <= 1.0)
        & (torch.abs(grid[:, 1]) <= 1.0)
    )
    # A null mixture is essential: many surfels in a retrieved region are not
    # repeatable in a particular view.  Maximizing every cosine lets those
    # outliers rotate the camera toward a spurious average.  Softplus is a
    # smooth inlier evidence term; similarities below the null threshold have
    # exponentially vanishing influence.
    del robust_delta  # Kept in the signature for artifact compatibility.
    temperature = float(similarity_temperature)
    evidence = temperature * F.softplus(
        (cosine - float(similarity_threshold)) / temperature
    )
    if detector_heatmap is not None:
        detector = F.grid_sample(
            detector_heatmap.reshape(1, 1, *detector_heatmap.shape[-2:]),
            grid.reshape(1, 1, -1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0, 0, :]
        floor = float(detector_evidence_floor)
        radio_gate = floor + (1.0 - floor) * torch.sigmoid(
            (cosine - float(detector_radio_gate)) / 0.10
        )
        evidence = evidence + float(detector_evidence_weight) * detector * radio_gate
    active_weight = weights * valid.to(weights.dtype)
    score = torch.sum(active_weight * evidence) / torch.clamp(
        torch.sum(active_weight), min=1e-8
    )
    return -score, score


def score_surface_alignment(
    field: SurfaceFeatureField,
    query_feature: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    sample_rows: np.ndarray,
    *,
    device: str = "cuda",
    robust_delta: float = 0.20,
    detector_heatmap: np.ndarray | None = None,
    detector_evidence_floor: float = 0.15,
    detector_evidence_weight: float = 0.35,
    detector_radio_gate: float = 0.05,
) -> float:
    if np.asarray(sample_rows).size == 0:
        return float("-inf")
    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    with torch.no_grad():
        _loss, score = _alignment_loss(
            torch.zeros(6, device=torch_device),
            torch.as_tensor(pose_w2c, dtype=torch.float32, device=torch_device),
            torch.as_tensor(field.centers[sample_rows], dtype=torch.float32, device=torch_device),
            torch.as_tensor(field.features[sample_rows], dtype=torch.float32, device=torch_device),
            torch.as_tensor(field.confidence[sample_rows], dtype=torch.float32, device=torch_device),
            torch.as_tensor(query_feature, dtype=torch.float32, device=torch_device),
            camera,
            robust_delta,
            detector_heatmap=(
                None
                if detector_heatmap is None
                else torch.as_tensor(
                    detector_heatmap, dtype=torch.float32, device=torch_device
                )
            ),
            detector_evidence_floor=float(detector_evidence_floor),
            detector_evidence_weight=float(detector_evidence_weight),
            detector_radio_gate=float(detector_radio_gate),
        )
    return float(score.detach().cpu().item())


def _align_surface_feature_field_lbfgs(
    field: SurfaceFeatureField,
    query_feature: np.ndarray,
    initial_pose_w2c: np.ndarray,
    camera: ColmapCamera,
    selected_maplet_ids: np.ndarray,
    *,
    config: ContinuousSurfaceAlignmentConfig = ContinuousSurfaceAlignmentConfig(),
    device: str = "cuda",
    detector_heatmap: np.ndarray | None = None,
) -> SurfaceAlignmentResult:
    """Optimize a pose directly against selected disconnected maplet patches."""

    feature = np.asarray(query_feature, dtype=np.float32)
    if feature.ndim != 3 or int(feature.shape[0]) != field.feature_dim:
        raise ValueError("query feature dimensions do not match surface field")
    selected_ids = np.asarray(selected_maplet_ids, dtype=np.int64).reshape(-1)
    if selected_ids.size == 0:
        raise ValueError("at least one selected maplet is required")
    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    current_pose = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4).copy()
    initial_score = float("-inf")
    final_score = float("-inf")
    trace: list[dict[str, object]] = []
    last_rows = np.zeros((0,), dtype=np.int64)
    detector_tensor = (
        None
        if detector_heatmap is None
        else torch.as_tensor(
            detector_heatmap, dtype=torch.float32, device=torch_device
        )
    )

    for scale in config.pyramid_scales:
        target_h = max(2, int(round(feature.shape[1] * float(scale))))
        target_w = max(2, int(round(feature.shape[2] * float(scale))))
        source = torch.as_tensor(feature[None], dtype=torch.float32, device=torch_device)
        query_level = F.interpolate(
            source, size=(target_h, target_w), mode="bilinear", align_corners=True
        )[0]
        query_level = F.normalize(query_level, p=2, dim=0, eps=1e-8)
        for outer in range(int(config.outer_iterations_per_scale)):
            rows = select_render_samples(
                field,
                current_pose,
                camera,
                selected_ids,
                feature_height=target_h,
                feature_width=target_w,
                config=config,
            )
            last_rows = rows
            if rows.size < 6:
                trace.append(
                    {
                        "scale": float(scale),
                        "outer": int(outer),
                        "sample_count": int(rows.size),
                        "status": "insufficient_visible_surface",
                    }
                )
                continue
            xyz = torch.as_tensor(
                field.centers[rows], dtype=torch.float32, device=torch_device
            )
            map_features = torch.as_tensor(
                field.features[rows], dtype=torch.float32, device=torch_device
            )
            weights = torch.as_tensor(
                field.confidence[rows], dtype=torch.float32, device=torch_device
            )
            base_pose = torch.as_tensor(
                current_pose, dtype=torch.float32, device=torch_device
            )
            unconstrained_delta = torch.zeros(
                6, dtype=torch.float32, device=torch_device, requires_grad=True
            )

            def bounded_delta() -> torch.Tensor:
                rotation_limit = float(config.maximum_step_rotation_deg) * np.pi / 180.0
                return torch.cat(
                    [
                        rotation_limit * torch.tanh(unconstrained_delta[:3]),
                        float(config.maximum_step_translation_m)
                        * torch.tanh(unconstrained_delta[3:]),
                    ]
                )

            with torch.no_grad():
                _before_loss, before_score = _alignment_loss(
                    bounded_delta(),
                    base_pose,
                    xyz,
                    map_features,
                    weights,
                    query_level,
                    camera,
                    config.robust_delta,
                    config.similarity_threshold,
                    config.similarity_temperature,
                    detector_tensor,
                    config.detector_evidence_floor,
                    config.detector_evidence_weight,
                    config.detector_radio_gate,
                )
                if not np.isfinite(initial_score):
                    initial_score = float(before_score.detach().cpu().item())
            optimizer = torch.optim.LBFGS(
                [unconstrained_delta],
                lr=0.75,
                max_iter=int(config.lbfgs_iterations),
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss, _score = _alignment_loss(
                    bounded_delta(),
                    base_pose,
                    xyz,
                    map_features,
                    weights,
                    query_level,
                    camera,
                    config.robust_delta,
                    config.similarity_threshold,
                    config.similarity_temperature,
                    detector_tensor,
                    config.detector_evidence_floor,
                    config.detector_evidence_weight,
                    config.detector_radio_gate,
                )
                loss.backward()
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                _after_loss, after_score = _alignment_loss(
                    bounded_delta(),
                    base_pose,
                    xyz,
                    map_features,
                    weights,
                    query_level,
                    camera,
                    config.robust_delta,
                    config.similarity_threshold,
                    config.similarity_temperature,
                    detector_tensor,
                    config.detector_evidence_floor,
                    config.detector_evidence_weight,
                    config.detector_radio_gate,
                )
                actual_delta = bounded_delta()
                candidate = _compose_delta(actual_delta, base_pose).detach().cpu().numpy()
            before_value = float(before_score.detach().cpu().item())
            after_value = float(after_score.detach().cpu().item())
            accepted = bool(np.isfinite(after_value) and after_value >= before_value)
            if accepted:
                current_pose = candidate.astype(np.float64)
                final_score = after_value
            trace.append(
                {
                    "scale": float(scale),
                    "outer": int(outer),
                    "sample_count": int(rows.size),
                    "score_before": before_value,
                    "score_after": after_value,
                    "accepted": accepted,
                    "delta_rotation_deg": float(
                        np.linalg.norm(actual_delta[:3].detach().cpu().numpy())
                        * 180.0
                        / np.pi
                    ),
                    "delta_translation_m": float(
                        np.linalg.norm(actual_delta[3:].detach().cpu().numpy())
                    ),
                }
            )
    if not np.isfinite(final_score):
        final_score = initial_score
    return SurfaceAlignmentResult(
        pose_w2c=current_pose,
        initial_score=float(initial_score),
        final_score=float(final_score),
        selected_maplet_ids=selected_ids,
        sample_count=int(last_rows.size),
        converged=bool(np.isfinite(final_score) and last_rows.size >= 6),
        diagnostics={"optimization_trace": trace},
    )


def _left_pose_step(
    pose_w2c: np.ndarray,
    axis: int,
    amount: float,
) -> np.ndarray:
    step = np.eye(4, dtype=np.float64)
    if int(axis) < 3:
        vector = np.zeros((3,), dtype=np.float64)
        vector[int(axis)] = float(amount)
        theta = abs(float(amount))
        if theta > 0.0:
            skew = np.asarray(
                [
                    [0.0, -vector[2], vector[1]],
                    [vector[2], 0.0, -vector[0]],
                    [-vector[1], vector[0], 0.0],
                ],
                dtype=np.float64,
            )
            step[:3, :3] = (
                np.eye(3)
                + np.sin(theta) / theta * skew
                + (1.0 - np.cos(theta)) / (theta * theta) * (skew @ skew)
            )
    else:
        step[int(axis) - 3, 3] = float(amount)
    return step @ np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)


def _align_surface_feature_field_trust_region(
    field: SurfaceFeatureField,
    query_feature: np.ndarray,
    initial_pose_w2c: np.ndarray,
    camera: ColmapCamera,
    selected_maplet_ids: np.ndarray,
    *,
    config: ContinuousSurfaceAlignmentConfig,
    device: str,
    detector_heatmap: np.ndarray | None,
) -> SurfaceAlignmentResult:
    """Bounded coordinate trust region with disjoint fit/validation surfels.

    RADIO-final is smooth and locally non-quadratic.  An unconstrained LM step
    can exploit a false feature mode.  This small six-dimensional search keeps
    the continuous SE(3) objective but only accepts a step when both disjoint
    surfel partitions support it.
    """

    feature = np.asarray(query_feature, dtype=np.float32)
    selected_ids = np.asarray(selected_maplet_ids, dtype=np.int64).reshape(-1)
    current = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4).copy()
    rotation_step = np.deg2rad(float(config.maximum_step_rotation_deg))
    translation_step = float(config.maximum_step_translation_m)
    trace: list[dict[str, object]] = []
    initial_score = float("-inf")
    current_score = float("-inf")
    last_rows = np.zeros((0,), dtype=np.int64)

    for iteration in range(int(config.trust_region_iterations)):
        rows = select_render_samples(
            field,
            current,
            camera,
            selected_ids,
            feature_height=int(feature.shape[1]),
            feature_width=int(feature.shape[2]),
            config=config,
        )
        last_rows = rows
        if rows.size < 12:
            trace.append(
                {
                    "iteration": int(iteration),
                    "status": "insufficient_visible_surface",
                    "sample_count": int(rows.size),
                }
            )
            break
        heldout = (field.source_indices[rows] % 5) == 0
        if np.sum(heldout) < 4 or np.sum(~heldout) < 6:
            heldout = (np.arange(rows.size) % 5) == 0
        fit_rows = rows[~heldout]
        validation_rows = rows[heldout]
        fit_before = score_surface_alignment(
            field,
            feature,
            current,
            camera,
            fit_rows,
            device=device,
            detector_heatmap=detector_heatmap,
            detector_evidence_floor=config.detector_evidence_floor,
            detector_evidence_weight=config.detector_evidence_weight,
            detector_radio_gate=config.detector_radio_gate,
        )
        validation_before = score_surface_alignment(
            field,
            feature,
            current,
            camera,
            validation_rows,
            device=device,
            detector_heatmap=detector_heatmap,
            detector_evidence_floor=config.detector_evidence_floor,
            detector_evidence_weight=config.detector_evidence_weight,
            detector_radio_gate=config.detector_radio_gate,
        )
        all_before = score_surface_alignment(
            field,
            feature,
            current,
            camera,
            rows,
            device=device,
            detector_heatmap=detector_heatmap,
            detector_evidence_floor=config.detector_evidence_floor,
            detector_evidence_weight=config.detector_evidence_weight,
            detector_radio_gate=config.detector_radio_gate,
        )
        if not np.isfinite(initial_score):
            initial_score = all_before
        candidates: list[tuple[float, float, float, int, int, np.ndarray]] = []
        for axis in range(6):
            amount = rotation_step if axis < 3 else translation_step
            for sign in (-1, 1):
                candidate = _left_pose_step(current, axis, sign * amount)
                fit_score = score_surface_alignment(
                    field,
                    feature,
                    candidate,
                    camera,
                    fit_rows,
                    device=device,
                    detector_heatmap=detector_heatmap,
                    detector_evidence_floor=config.detector_evidence_floor,
                    detector_evidence_weight=config.detector_evidence_weight,
                    detector_radio_gate=config.detector_radio_gate,
                )
                validation_score = score_surface_alignment(
                    field,
                    feature,
                    candidate,
                    camera,
                    validation_rows,
                    device=device,
                    detector_heatmap=detector_heatmap,
                    detector_evidence_floor=config.detector_evidence_floor,
                    detector_evidence_weight=config.detector_evidence_weight,
                    detector_radio_gate=config.detector_radio_gate,
                )
                candidates.append(
                    (
                        fit_score + validation_score,
                        fit_score,
                        validation_score,
                        axis,
                        sign,
                        candidate,
                    )
                )
        candidates.sort(key=lambda row: row[0], reverse=True)
        _combined, fit_after, validation_after, axis, sign, candidate = candidates[0]
        fit_margin = max(1e-4, 0.005 * abs(float(fit_before)))
        validation_margin = max(5e-5, 0.002 * abs(float(validation_before)))
        accepted = bool(
            fit_after > fit_before + fit_margin
            and validation_after > validation_before + validation_margin
        )
        if accepted:
            current = candidate
            current_score = score_surface_alignment(
                field,
                feature,
                current,
                camera,
                rows,
                device=device,
                detector_heatmap=detector_heatmap,
                detector_evidence_floor=config.detector_evidence_floor,
                detector_evidence_weight=config.detector_evidence_weight,
                detector_radio_gate=config.detector_radio_gate,
            )
        else:
            rotation_step *= 0.5
            translation_step *= 0.5
            current_score = all_before
        trace.append(
            {
                "iteration": int(iteration),
                "sample_count": int(rows.size),
                "fit_count": int(fit_rows.size),
                "heldout_count": int(validation_rows.size),
                "score_before": float(all_before),
                "fit_score_before": float(fit_before),
                "heldout_score_before": float(validation_before),
                "fit_score_candidate": float(fit_after),
                "heldout_score_candidate": float(validation_after),
                "candidate_axis": int(axis),
                "candidate_sign": int(sign),
                "accepted": accepted,
                "rotation_step_deg": float(np.rad2deg(rotation_step)),
                "translation_step_m": float(translation_step),
            }
        )
        if np.rad2deg(rotation_step) < 0.05 and translation_step < 0.002:
            break
    return SurfaceAlignmentResult(
        pose_w2c=current,
        initial_score=float(initial_score),
        final_score=float(current_score),
        selected_maplet_ids=selected_ids,
        sample_count=int(last_rows.size),
        converged=bool(np.isfinite(current_score) and last_rows.size >= 12),
        diagnostics={"optimization_trace": trace, "optimizer": "trust_region"},
    )


def align_surface_feature_field(
    field: SurfaceFeatureField,
    query_feature: np.ndarray,
    initial_pose_w2c: np.ndarray,
    camera: ColmapCamera,
    selected_maplet_ids: np.ndarray,
    *,
    config: ContinuousSurfaceAlignmentConfig = ContinuousSurfaceAlignmentConfig(),
    device: str = "cuda",
    detector_heatmap: np.ndarray | None = None,
) -> SurfaceAlignmentResult:
    if str(config.optimizer) == "trust_region":
        return _align_surface_feature_field_trust_region(
            field,
            query_feature,
            initial_pose_w2c,
            camera,
            selected_maplet_ids,
            config=config,
            device=device,
            detector_heatmap=detector_heatmap,
        )
    return _align_surface_feature_field_lbfgs(
        field,
        query_feature,
        initial_pose_w2c,
        camera,
        selected_maplet_ids,
        config=config,
        device=device,
        detector_heatmap=detector_heatmap,
    )
