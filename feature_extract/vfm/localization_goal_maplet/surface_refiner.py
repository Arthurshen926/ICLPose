"""Plugin continuous surface-coordinate refiner for Goal-Maplet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization_v6.local_correlation import (
    build_local_correlation_query_cache,
    local_correlation_distribution,
)
from feature_extract.vfm.localization_v6.se3_update import solve_correlation_se3_update
from feature_extract.vfm.localization_v6.se3_update import se3_exp

from .canonical_field import CanonicalSurfaceField
from .local_head import ChildLocalReadoutHead
from .physical_map import GoalMapletPhysicalMap
from .surface_renderer import render_canonical_surface_field


@dataclass(frozen=True)
class SurfaceRefinementResult:
    pose_w2c: np.ndarray
    success: bool
    rounds_completed: int
    history: tuple[dict[str, float | int | bool], ...]


def canonical_alignment_score(rendered, query_feature: np.ndarray) -> float:
    """Fixed-denominator same-pixel canonical-field agreement."""

    query = np.asarray(query_feature, dtype=np.float32)
    query = query / np.maximum(np.linalg.norm(query, axis=0, keepdims=True), 1e-8)
    cosine = np.sum(query * np.asarray(rendered.feature, dtype=np.float32), axis=0)
    valid = np.asarray(rendered.mask, dtype=bool)
    if not np.any(valid):
        return float("-inf")
    # The denominator is the complete query grid: losing surface coverage can
    # never improve the score by retaining only a few easy pixels.
    positive = np.clip(cosine, -1.0, 1.0)
    return float(np.sum(positive[valid]) / float(valid.size))


def apply_local_readout(
    query_feature: np.ndarray,
    field: CanonicalSurfaceField,
    local_head: ChildLocalReadoutHead | None,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Regenerate query/map local codes without storing another map field."""

    if local_head is None:
        return np.asarray(query_feature, dtype=np.float32), None
    import torch

    query = np.asarray(query_feature, dtype=np.float32)
    if query.ndim != 3 or query.shape[0] != field.feature_dim:
        raise ValueError("query feature and canonical field dimensions differ")
    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    model = local_head.to(torch_device).eval()
    with torch.no_grad():
        query_flat = torch.as_tensor(
            query.transpose(1, 2, 0).reshape(-1, query.shape[0]),
            device=torch_device,
        )
        query_local = model.encode_query(query_flat).cpu().numpy().astype(np.float32)
        map_chunks = []
        for start in range(0, field.codes.shape[0], 65536):
            values = torch.as_tensor(field.codes[start : start + 65536], device=torch_device)
            map_chunks.append(model.encode_map(values).cpu().numpy().astype(np.float32))
    query_local = query_local.reshape(query.shape[1], query.shape[2], query.shape[0]).transpose(2, 0, 1)
    return query_local, np.concatenate(map_chunks, axis=0)


def refine_pose_with_canonical_surface(
    query_feature: np.ndarray,
    initial_pose_w2c: np.ndarray,
    camera,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    *,
    selected_child_rows: np.ndarray | None = None,
    local_head: ChildLocalReadoutHead | None = None,
    rounds: int = 5,
    radius: int = 4,
    temperature: float = 0.07,
    null_logit: float = 0.0,
    maximum_translation_step_m: float = 0.12,
    maximum_rotation_step_deg: float = 2.0,
    maximum_points: int = 2048,
    displacement_prior_sigma_cells: float | Sequence[float] = (1.0, 0.7, 0.5, 0.35),
    minimum_score_improvement: float = 1e-5,
    render_supersample_factor: int = 1,
    acceptance_supersample_factor: int | None = None,
    device: str = "cuda",
) -> SurfaceRefinementResult:
    """Render exact local surfaces, correlate RADIO codes, update joint SE(3)."""

    query = np.asarray(query_feature, dtype=np.float32)
    if query.ndim != 3:
        raise ValueError("query feature must have shape [C,H,W]")
    query, feature_codes = apply_local_readout(query, field, local_head, device=str(device))
    pose = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4).copy()
    cache = build_local_correlation_query_cache(
        query, radius=int(radius), background_samples=512, device=str(device)
    )
    history: list[dict[str, float | int | bool]] = []
    success = False
    sigma_schedule = (
        [float(displacement_prior_sigma_cells)]
        if np.isscalar(displacement_prior_sigma_cells)
        else [float(value) for value in displacement_prior_sigma_cells]
    )
    if not sigma_schedule or any(value <= 0.0 for value in sigma_schedule):
        raise ValueError("invalid displacement-prior schedule")
    rendered = render_canonical_surface_field(
        physical,
        field,
        pose,
        camera,
        width=int(query.shape[2]),
        height=int(query.shape[1]),
        selected_child_rows=selected_child_rows,
        feature_codes=feature_codes,
        device=str(device),
        supersample_factor=int(render_supersample_factor),
    )
    acceptance_factor = (
        int(render_supersample_factor)
        if acceptance_supersample_factor is None
        else int(acceptance_supersample_factor)
    )
    if acceptance_factor <= 0:
        raise ValueError("acceptance_supersample_factor must be positive")
    score_rendered = (
        rendered
        if acceptance_factor == int(render_supersample_factor)
        else render_canonical_surface_field(
            physical,
            field,
            pose,
            camera,
            width=int(query.shape[2]),
            height=int(query.shape[1]),
            selected_child_rows=selected_child_rows,
            feature_codes=feature_codes,
            device=str(device),
            supersample_factor=acceptance_factor,
        )
    )
    current_score = canonical_alignment_score(score_rendered, query)
    for iteration in range(max(int(rounds), 0)):
        sigma = sigma_schedule[min(iteration, len(sigma_schedule) - 1)]
        correlation = local_correlation_distribution(
            rendered,
            query,
            radius=int(radius),
            temperature=float(temperature),
            null_logit=float(null_logit),
            uncertainty_temperature_scale=1.0,
            uncertainty_null_scale=1.0,
            maximum_points=int(maximum_points),
            background_samples=512,
            device=str(device),
            query_cache=cache,
            displacement_prior_sigma_cells=sigma,
        )
        update = solve_correlation_se3_update(
            correlation,
            pose,
            camera,
            maximum_null_probability=0.85,
            maximum_entropy=5.0,
            minimum_matchability=0.0,
            minimum_variance_px2=1.0,
            damping=1e-2,
            robust_delta=3.0,
            iterations=4,
            minimum_points=16,
            displacement_scale_xy=(
                float(camera.width) / float(query.shape[2]),
                float(camera.height) / float(query.shape[1]),
            ),
            maximum_translation_step_m=float(maximum_translation_step_m),
            maximum_rotation_step_deg=float(maximum_rotation_step_deg),
            dominant_mode_conditioning=True,
            mode_radius_cells=1.5,
            balance_maplet_weights=True,
        )
        visible = 1.0 - correlation.null_probability
        candidate_score = float("-inf")
        reverse_score = float("-inf")
        accepted = False
        accepted_direction = "none"
        candidate_rendered = None
        candidate_pose = None
        if update.success:
            candidate_pose = update.updated_pose_w2c
            candidate_rendered = render_canonical_surface_field(
                physical,
                field,
                candidate_pose,
                camera,
                width=int(query.shape[2]),
                height=int(query.shape[1]),
                selected_child_rows=selected_child_rows,
                feature_codes=feature_codes,
                device=str(device),
                supersample_factor=int(render_supersample_factor),
            )
            candidate_score_rendered = (
                candidate_rendered
                if acceptance_factor == int(render_supersample_factor)
                else render_canonical_surface_field(
                    physical,
                    field,
                    candidate_pose,
                    camera,
                    width=int(query.shape[2]),
                    height=int(query.shape[1]),
                    selected_child_rows=selected_child_rows,
                    feature_codes=feature_codes,
                    device=str(device),
                    supersample_factor=acceptance_factor,
                )
            )
            candidate_score = canonical_alignment_score(candidate_score_rendered, query)
            accepted = bool(candidate_score >= current_score + float(minimum_score_improvement))
            if accepted:
                accepted_direction = "forward"
            else:
                # A flow-to-SE(3) sign error or a poor local linearization can
                # be detected without GT by evaluating the inverse trust-region
                # step under the same fixed-denominator objective.
                reverse_pose = se3_exp(-np.asarray(update.delta, dtype=np.float64)) @ pose
                reverse_rendered = render_canonical_surface_field(
                    physical,
                    field,
                    reverse_pose,
                    camera,
                    width=int(query.shape[2]),
                    height=int(query.shape[1]),
                    selected_child_rows=selected_child_rows,
                    feature_codes=feature_codes,
                    device=str(device),
                    supersample_factor=int(render_supersample_factor),
                )
                reverse_score_rendered = (
                    reverse_rendered
                    if acceptance_factor == int(render_supersample_factor)
                    else render_canonical_surface_field(
                        physical,
                        field,
                        reverse_pose,
                        camera,
                        width=int(query.shape[2]),
                        height=int(query.shape[1]),
                        selected_child_rows=selected_child_rows,
                        feature_codes=feature_codes,
                        device=str(device),
                        supersample_factor=acceptance_factor,
                    )
                )
                reverse_score = canonical_alignment_score(reverse_score_rendered, query)
                if reverse_score >= current_score + float(minimum_score_improvement):
                    candidate_pose = reverse_pose
                    candidate_rendered = reverse_rendered
                    candidate_score = reverse_score
                    accepted = True
                    accepted_direction = "reverse"
        history.append({
            "iteration": int(iteration),
            "displacement_prior_sigma_cells": float(sigma),
            "rendered_pixel_count": int(np.sum(rendered.mask)),
            "correlation_point_count": int(correlation.xyz.shape[0]),
            "mean_visible_probability": float(np.mean(visible)) if visible.size else 0.0,
            "median_flow_cells": (
                float(np.median(np.linalg.norm(correlation.mean_displacement, axis=1)))
                if correlation.mean_displacement.size else 0.0
            ),
            "used_point_count": int(update.used_point_count),
            "condition_number": float(update.condition_number),
            "residual_rms_px": float(update.residual_rms_px),
            "translation_step_m": float(np.linalg.norm(update.delta[3:])),
            "rotation_step_deg": float(np.degrees(np.linalg.norm(update.delta[:3]))),
            "update_success": bool(update.success),
            "alignment_score": float(current_score),
            "candidate_alignment_score": float(candidate_score),
            "reverse_alignment_score": float(reverse_score),
            "accepted": bool(accepted),
            "accepted_direction": str(accepted_direction),
        })
        if not update.success:
            continue
        if accepted:
            pose = candidate_pose
            rendered = candidate_rendered
            current_score = candidate_score
            success = True
        if np.linalg.norm(update.delta[3:]) < 1e-4 and np.linalg.norm(update.delta[:3]) < 1e-5:
            break
    return SurfaceRefinementResult(
        pose_w2c=pose,
        success=success,
        rounds_completed=len(history),
        history=tuple(history),
    )
