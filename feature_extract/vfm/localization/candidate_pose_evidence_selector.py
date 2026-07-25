"""Target-free static token selection for fixed-top-L pose evidence.

The local phase/RGB likelihoods emit one candidate-marginalized score per
query token and pose hypothesis.  Averaging every token is robust but can
wash out the few tokens that carry a distinctive physical observation.  This
module derives a *static* query-token selection from target-free visual
outputs before any pose projection is evaluated:

* RADIO phase confidence measures how concentrated its fixed-candidate
  identity posterior is;
* RGB quality measures non-dustbin, low-entropy local spatial modes;
* spatial-diverse top-K selection prevents one facade patch from consuming
  the full token budget.

The selector deliberately has no pose, residual, track identifier, candidate
rank, coarse score, or supervision input.  A caller must freeze the selected
weights before scoring correct/wrong pose hypotheses or visual controls.
"""

from __future__ import annotations

import math

import torch

from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CandidateHighresRGBMultiscalePrediction,
)
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    CandidateMultiscalePhaseIdentityPrediction,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    fixed_candidate_view_mixture_log_ratio,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
)


CANDIDATE_POSE_EVIDENCE_SELECTOR_FORMAT = "candidate_pose_evidence_selector_v2"


def _runtime_and_edge_layout(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    edge_values: torch.Tensor,
    edge_usable: torch.Tensor,
) -> tuple[CandidatePoseRGBSpatialRuntime, torch.Tensor, torch.Tensor]:
    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("static token selector requires a target-free runtime")
    values = torch.as_tensor(edge_values, dtype=torch.float32)
    usable = torch.as_tensor(edge_usable, dtype=torch.bool, device=values.device)
    active = runtime.to(values.device)
    if (
        values.shape != tuple(active.support_image_indices.shape)
        or usable.shape != values.shape
        or values.ndim != 3
        or not torch.isfinite(values).all()
    ):
        raise ValueError("static token selector edge layout is invalid")
    return active, values, usable


def runtime_visual_edge_availability(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    phase_prediction: CandidateMultiscalePhaseIdentityPrediction,
    rgb_prediction: CandidateHighresRGBMultiscalePrediction,
    phase_source_name: str = "radio_final",
    rgb_source_name: str = "fine",
) -> torch.Tensor:
    """Return only the normal runtime's visual-source intersection.

    Appearance-deranged controls are useful for a paired evaluation denominator
    but must never influence the selector that would run at inference.  This
    helper is intentionally limited to one normal phase prediction and one
    normal RGB prediction.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("runtime visual availability requires a target-free runtime")
    if not isinstance(phase_prediction, CandidateMultiscalePhaseIdentityPrediction) or not isinstance(
        rgb_prediction, CandidateHighresRGBMultiscalePrediction
    ):
        raise ValueError("runtime visual availability requires target-free visual predictions")
    phase_source = str(phase_source_name)
    rgb_source = str(rgb_source_name)
    if phase_source not in phase_prediction.source_edge_usable or rgb_source not in rgb_prediction.sources:
        raise ValueError("runtime visual availability source is invalid")
    phase_usable = phase_prediction.source_edge_usable[phase_source]
    rgb_usable = rgb_prediction.sources[rgb_source].edge_usable.to(device=phase_usable.device)
    active = runtime.to(phase_usable.device)
    if phase_usable.shape != tuple(active.support_image_indices.shape) or rgb_usable.shape != phase_usable.shape:
        raise ValueError("runtime visual availability layout differs from target-free runtime")
    return phase_usable & rgb_usable


def phase_identity_confidence(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateMultiscalePhaseIdentityPrediction,
    source_name: str = "radio_final",
    edge_availability_override: torch.Tensor | None = None,
    include_fixed_candidate_prior: bool = False,
) -> torch.Tensor:
    """Return a pose-independent concentration score for every query token.

    The source LLR is first marginalized over its fixed support views.  A
    conditional softmax over the frozen candidate set gives a confidence above
    its uniform baseline.  Missing phase/RGB-common edges receive zero score,
    rather than becoming a padded-image or prior-only selection shortcut.
    """

    if not isinstance(prediction, CandidateMultiscalePhaseIdentityPrediction):
        raise ValueError("phase selector requires a phase identity prediction")
    source = str(source_name)
    if source not in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
        raise ValueError("phase selector source is invalid")
    active, values, usable = _runtime_and_edge_layout(
        runtime=runtime,
        edge_values=prediction.source_edge_log_likelihood_ratios[source],
        edge_usable=prediction.source_edge_usable[source],
    )
    if edge_availability_override is not None:
        override = torch.as_tensor(
            edge_availability_override, dtype=torch.bool, device=values.device
        )
        if override.shape != usable.shape:
            raise ValueError("phase selector availability override has the wrong layout")
        usable = usable & override
    _point, candidate = fixed_candidate_view_mixture_log_ratio(
        edge_log_likelihood_ratios=values.unsqueeze(0),
        edge_usable=usable.unsqueeze(0),
        candidate_view_weights=active.candidate_view_weights,
        candidate_probabilities=active.candidate_probabilities,
        null_probabilities=active.null_probabilities,
        missing_edge_log_likelihood_ratio=0.0,
    )
    candidate_values = candidate[0]
    if include_fixed_candidate_prior:
        nonnull_mass = active.candidate_probabilities.sum(dim=1, keepdim=True)
        logits = torch.where(
            active.candidate_probabilities > 0.0,
            torch.log(
                active.candidate_probabilities / nonnull_mass.clamp_min(torch.finfo(torch.float32).tiny)
            )
            + candidate_values,
            torch.full_like(candidate_values, -torch.inf),
        )
    else:
        logits = torch.where(
            active.candidate_probabilities > 0.0,
            candidate_values,
            torch.full_like(candidate_values, -torch.inf),
        )
    conditional = torch.softmax(logits, dim=1)
    candidate_count = int(conditional.shape[1])
    coverage = (usable & (active.candidate_view_weights > 0.0)).reshape(
        active.point_count, -1
    ).any(dim=1)
    baseline = 1.0 / float(candidate_count)
    confidence = (conditional.max(dim=1).values - baseline).clamp_min(0.0)
    return torch.where(coverage, confidence, torch.zeros_like(confidence))


def rgb_spatial_mode_quality(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateHighresRGBMultiscalePrediction,
    source_name: str = "fine",
    edge_availability_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return target-free RGB mode quality for every query token.

    A useful local measurement should allocate non-dustbin mass to a compact
    spatial mode.  This is a quality selector only: it does not evaluate a
    pose-projected offset and therefore cannot leak hypothesis geometry.
    """

    if not isinstance(prediction, CandidateHighresRGBMultiscalePrediction):
        raise ValueError("RGB selector requires a high-resolution RGB prediction")
    source = str(source_name)
    if source not in prediction.sources:
        raise ValueError("RGB selector source is invalid")
    scale = prediction.sources[source]
    active, _values, usable = _runtime_and_edge_layout(
        runtime=runtime,
        edge_values=scale.edge_log_likelihood_ratios,
        edge_usable=scale.edge_usable,
    )
    if edge_availability_override is not None:
        override = torch.as_tensor(
            edge_availability_override, dtype=torch.bool, device=usable.device
        )
        if override.shape != usable.shape:
            raise ValueError("RGB selector availability override has the wrong layout")
        usable = usable & override
    joint = scale.joint_log_probabilities.to(device=usable.device, dtype=torch.float32)
    local = torch.exp(joint[..., :-1])
    non_dustbin = local.sum(dim=-1)
    conditional = local / non_dustbin.unsqueeze(-1).clamp_min(torch.finfo(local.dtype).tiny)
    entropy = -(conditional * torch.log(conditional.clamp_min(torch.finfo(local.dtype).tiny))).sum(dim=-1)
    maximum_entropy = math.log(float(local.shape[-1]))
    compactness = 1.0 - entropy / max(maximum_entropy, 1e-12)
    edge_quality = (non_dustbin * compactness.clamp(0.0, 1.0)).where(
        usable, torch.zeros_like(non_dustbin)
    )
    weighted = edge_quality * active.candidate_view_weights
    candidate_quality = weighted.sum(dim=2)
    coverage = (usable & (active.candidate_view_weights > 0.0)).reshape(
        active.point_count, -1
    ).any(dim=1)
    quality = candidate_quality.max(dim=1).values
    return torch.where(coverage, quality, torch.zeros_like(quality))


def combine_selector_confidences(
    *, phase_confidence: torch.Tensor, rgb_quality: torch.Tensor
) -> torch.Tensor:
    """Conservatively combine two correlated static selector signals."""

    phase = torch.as_tensor(phase_confidence, dtype=torch.float32)
    rgb = torch.as_tensor(rgb_quality, dtype=torch.float32, device=phase.device)
    if (
        phase.ndim != 1
        or rgb.shape != phase.shape
        or len(phase) == 0
        or not torch.isfinite(phase).all()
        or not torch.isfinite(rgb).all()
        or torch.any(phase < 0.0)
        or torch.any(rgb < 0.0)
    ):
        raise ValueError("static selector confidences are invalid")
    phase_normalized = phase / phase.amax().clamp_min(torch.finfo(phase.dtype).tiny)
    rgb_normalized = rgb / rgb.amax().clamp_min(torch.finfo(rgb.dtype).tiny)
    # A geometric mean rewards agreement without treating the two RGB-derived
    # visual signals as independent likelihood factors.
    return torch.sqrt(phase_normalized.clamp_min(0.0) * rgb_normalized.clamp_min(0.0))


def spatial_diverse_topk_weights(
    *,
    xy: torch.Tensor,
    scores: torch.Tensor,
    image_size: tuple[int, int],
    top_k: int,
    grid_rows: int = 4,
    grid_columns: int = 4,
) -> torch.Tensor:
    """Select a fixed top-K score set with one token per occupied grid cell.

    The result is a binary weight vector and is fixed before any pose score is
    evaluated.  Ties resolve by original point index, making diagnostics
    reproducible across devices.
    """

    coordinates = torch.as_tensor(xy, dtype=torch.float32)
    values = torch.as_tensor(scores, dtype=torch.float32, device=coordinates.device).reshape(-1)
    width, height = (int(image_size[0]), int(image_size[1]))
    count = len(values)
    if (
        coordinates.shape != (count, 2)
        or count == 0
        or int(top_k) < 0
        or int(grid_rows) <= 0
        or int(grid_columns) <= 0
        or width <= 1
        or height <= 1
        or not torch.isfinite(coordinates).all()
        or not torch.isfinite(values).all()
    ):
        raise ValueError("spatial-diverse selector inputs are invalid")
    target = count if int(top_k) == 0 else min(int(top_k), count)
    if target == count:
        return torch.ones((count,), dtype=torch.float32, device=values.device)
    columns = torch.floor(coordinates[:, 0] / float(width) * int(grid_columns)).to(torch.long)
    rows = torch.floor(coordinates[:, 1] / float(height) * int(grid_rows)).to(torch.long)
    columns = columns.clamp(0, int(grid_columns) - 1)
    rows = rows.clamp(0, int(grid_rows) - 1)
    cells = rows * int(grid_columns) + columns
    # Stable sorting makes equal confidence scores deterministic by index.
    order = torch.argsort(values, descending=True, stable=True)
    chosen: list[int] = []
    occupied: set[int] = set()
    for index in order.tolist():
        cell = int(cells[index].item())
        if cell not in occupied:
            chosen.append(int(index))
            occupied.add(cell)
    if len(chosen) > target:
        chosen = chosen[:target]
    selected = set(chosen)
    if len(chosen) < target:
        for index in order.tolist():
            if int(index) not in selected:
                chosen.append(int(index))
                selected.add(int(index))
                if len(chosen) == target:
                    break
    if len(chosen) != target:
        raise RuntimeError("spatial-diverse selector could not fill top-K")
    weights = torch.zeros((count,), dtype=torch.float32, device=values.device)
    weights[torch.as_tensor(chosen, dtype=torch.long, device=values.device)] = 1.0
    return weights


def score_reweighted_selector_weights(
    *,
    selected_weights: torch.Tensor,
    scores: torch.Tensor,
    floor: float = 0.10,
    power: float = 1.0,
) -> torch.Tensor:
    """Reweight a fixed selection using only its target-free confidence score.

    A positive floor prevents a single overconfident patch from becoming an
    implicit hard argmax.  The returned weights remain static and may be used
    unchanged for every pose hypothesis and control branch.
    """

    selected = torch.as_tensor(selected_weights, dtype=torch.float32)
    values = torch.as_tensor(scores, dtype=torch.float32, device=selected.device).reshape(-1)
    lower = float(floor)
    exponent = float(power)
    if (
        selected.ndim != 1
        or values.shape != selected.shape
        or len(selected) == 0
        or not torch.isfinite(selected).all()
        or not torch.isfinite(values).all()
        or torch.any(selected < 0.0)
        or torch.any(values < 0.0)
        or not math.isfinite(lower)
        or not math.isfinite(exponent)
        or lower <= 0.0
        or lower > 1.0
        or exponent <= 0.0
        or not bool(selected.sum() > 0.0)
    ):
        raise ValueError("selector score reweighting inputs are invalid")
    active = selected > 0.0
    maximum = values[active].amax().clamp_min(torch.finfo(values.dtype).tiny)
    normalized = values / maximum
    factor = lower + (1.0 - lower) * normalized.clamp(0.0, 1.0)
    return selected * factor.pow(exponent)


def blend_selector_with_uniform_mass(
    *, selector_weights: torch.Tensor, uniform_mass: float
) -> torch.Tensor:
    """Mix a static selector with an equal-mass all-token fallback.

    The two terms are each normalized before mixing.  This is important: a
    binary top-12 vector otherwise has much less total mass than an all-token
    vector and an apparent ``50/50`` blend would silently be almost uniform.
    The result has unit mass and is still entirely target-free and
    pose-independent.
    """

    weights = torch.as_tensor(selector_weights, dtype=torch.float32).reshape(-1)
    mass = float(uniform_mass)
    if (
        len(weights) == 0
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0.0)
        or not bool(weights.sum() > 0.0)
        or not math.isfinite(mass)
        or mass < 0.0
        or mass > 1.0
    ):
        raise ValueError("uniform selector mixture inputs are invalid")
    selected = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    uniform = torch.full_like(selected, 1.0 / float(len(selected)))
    return mass * uniform + (1.0 - mass) * selected


def aggregate_static_point_log_likelihood_ratios(
    *, point_log_likelihood_ratios: torch.Tensor, selector_weights: torch.Tensor
) -> torch.Tensor:
    """Aggregate fixed per-point evidence with a pre-pose static selector."""

    points = torch.as_tensor(point_log_likelihood_ratios, dtype=torch.float32)
    weights = torch.as_tensor(selector_weights, dtype=torch.float32, device=points.device)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or weights.shape != (points.shape[1],)
        or not torch.isfinite(points).all()
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0.0)
        or not bool(weights.sum() > 0.0)
    ):
        raise ValueError("static selector aggregation inputs are invalid")
    return (points * weights.unsqueeze(0)).sum(dim=1) / weights.sum()
