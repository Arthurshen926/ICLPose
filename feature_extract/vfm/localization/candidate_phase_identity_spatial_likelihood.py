"""Strict hybrid of candidate identity and candidate-specific RGB spatial evidence.

This module joins two target-free visual predictions only *after* both network
forwards have completed:

``RADIO-final phase identity``
    asks whether a frozen query point and frozen support observation describe
    the same landmark candidate.

``real-RGB local density``
    evaluates the candidate-specific spatial mode at an externally supplied
    frozen pose projection.

The identity term is deliberately not allowed to reward an invalid or
out-of-window projection by itself.  A hybrid edge is usable only where the
selected phase source and RGB spatial mode are both available at the same
candidate/view projection.  Missing evidence is exactly neutral, preserves
the fixed support-view mass, and never consults a learned dustbin outside the
RGB density window.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CANDIDATE_HIGHRES_RGB_SCORE_PRESETS,
    CandidateHighresRGBMultiscalePrediction,
    highres_rgb_edge_log_likelihood_ratio_at_pose_projection,
)
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    CandidateMultiscalePhaseIdentityPrediction,
    phase_identity_point_block_derangement_shift,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    bounded_log_likelihood_ratio,
    fixed_candidate_view_mixture_log_ratio,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    continuous_joint_log_probability_at_offsets,
)


CANDIDATE_PHASE_IDENTITY_SPATIAL_LIKELIHOOD_FORMAT = (
    "candidate_phase_identity_spatial_likelihood_v1"
)


@dataclass(frozen=True)
class CandidatePhaseIdentitySpatialPoseScore:
    """Fixed-top-L pose score from one strict identity/spatial hybrid."""

    pose_log_likelihood_ratios: torch.Tensor
    point_log_likelihood_ratios: torch.Tensor
    candidate_log_likelihood_ratios: torch.Tensor
    edge_log_likelihood_ratios: torch.Tensor
    edge_usable: torch.Tensor
    phase_source_name: str
    spatial_source_name: str
    identity_weight: float
    spatial_weight: float

    def __post_init__(self) -> None:
        pose = torch.as_tensor(self.pose_log_likelihood_ratios, dtype=torch.float32)
        point = torch.as_tensor(self.point_log_likelihood_ratios, dtype=torch.float32)
        candidate = torch.as_tensor(self.candidate_log_likelihood_ratios, dtype=torch.float32)
        edge = torch.as_tensor(self.edge_log_likelihood_ratios, dtype=torch.float32)
        usable = torch.as_tensor(self.edge_usable, dtype=torch.bool, device=edge.device)
        identity_weight, spatial_weight = resolve_phase_spatial_weights(
            identity_weight=float(self.identity_weight), spatial_weight=float(self.spatial_weight)
        )
        if (
            pose.ndim != 1
            or point.ndim != 2
            or candidate.ndim != 3
            or edge.ndim != 4
            or usable.shape != edge.shape
            or pose.shape != (point.shape[0],)
            or candidate.shape[:2] != point.shape
            or edge.shape[:3] != candidate.shape
            or self.phase_source_name not in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
            or self.spatial_source_name not in CANDIDATE_HIGHRES_RGB_SCORE_PRESETS
            or self.spatial_source_name == "zero"
            or not torch.isfinite(pose).all()
            or not torch.isfinite(point).all()
            or not torch.isfinite(candidate).all()
            or not torch.isfinite(edge).all()
        ):
            raise ValueError("phase/spatial hybrid pose score is invalid")
        object.__setattr__(self, "pose_log_likelihood_ratios", pose)
        object.__setattr__(self, "point_log_likelihood_ratios", point)
        object.__setattr__(self, "candidate_log_likelihood_ratios", candidate)
        object.__setattr__(self, "edge_log_likelihood_ratios", edge)
        object.__setattr__(self, "edge_usable", usable)
        object.__setattr__(self, "identity_weight", identity_weight)
        object.__setattr__(self, "spatial_weight", spatial_weight)


def resolve_phase_spatial_weights(*, identity_weight: float, spatial_weight: float) -> tuple[float, float]:
    """Normalize a conservative convex combination of correlated visual LLRs."""

    identity = float(identity_weight)
    spatial = float(spatial_weight)
    if (
        not math.isfinite(identity)
        or not math.isfinite(spatial)
        or identity < 0.0
        or spatial < 0.0
        or identity <= 0.0
        or spatial <= 0.0
    ):
        raise ValueError("phase/spatial hybrid requires positive finite source weights")
    total = identity + spatial
    return identity / total, spatial / total


def phase_identity_spatial_edge_log_likelihood_ratios(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    phase_prediction: CandidateMultiscalePhaseIdentityPrediction,
    spatial_prediction: CandidateHighresRGBMultiscalePrediction,
    candidate_projection_offsets_xy: torch.Tensor,
    candidate_projection_valid: torch.Tensor,
    phase_source_name: str,
    spatial_source_name: str = "fine",
    identity_weight: float = 1.0,
    spatial_weight: float = 1.0,
    edge_availability_override: torch.Tensor | None = None,
    missing_edge_log_likelihood_ratio: float = 0.0,
    max_abs_spatial_log_likelihood_ratio: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float]]:
    """Score strict phase-plus-spatial evidence at frozen pose projections.

    The phase prediction has shape ``[point, candidate, view]`` and is
    independent of pose.  The RGB density has shape
    ``[hypothesis, point, candidate, view]`` after projection.  Their
    intersection is intentional: otherwise an invalid RGB projection could
    inherit a positive identity score and become a false pose reward.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("phase/spatial edge scoring requires a target-free runtime")
    if not isinstance(phase_prediction, CandidateMultiscalePhaseIdentityPrediction) or not isinstance(
        spatial_prediction, CandidateHighresRGBMultiscalePrediction
    ):
        raise ValueError("phase/spatial edge scoring requires target-free visual predictions")
    phase_source = str(phase_source_name)
    spatial_source = str(spatial_source_name)
    if phase_source not in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
        raise ValueError("phase/spatial phase source is invalid")
    if spatial_source not in CANDIDATE_HIGHRES_RGB_SCORE_PRESETS or spatial_source == "zero":
        raise ValueError("phase/spatial RGB source is invalid")
    identity, spatial = resolve_phase_spatial_weights(
        identity_weight=float(identity_weight), spatial_weight=float(spatial_weight)
    )
    phase_values = phase_prediction.source_edge_log_likelihood_ratios[phase_source]
    phase_usable = phase_prediction.source_edge_usable[phase_source]
    device = phase_values.device
    active = runtime.to(device)
    if (
        phase_values.shape != tuple(active.support_image_indices.shape)
        or phase_usable.shape != phase_values.shape
        or spatial_prediction.edge_shape != tuple(active.support_image_indices.shape)
        or spatial_prediction.sources["fine"].joint_log_probabilities.device != device
    ):
        raise ValueError("phase/spatial visual predictions do not match the fixed runtime")
    spatial_values, spatial_usable, _ = highres_rgb_edge_log_likelihood_ratio_at_pose_projection(
        prediction=spatial_prediction,
        candidate_projection_offsets_xy=candidate_projection_offsets_xy,
        candidate_projection_valid=candidate_projection_valid,
        source=spatial_source,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
        max_abs_log_likelihood_ratio=float(max_abs_spatial_log_likelihood_ratio),
    )
    if spatial_values.shape[1:] != phase_values.shape or spatial_usable.shape != spatial_values.shape:
        raise ValueError("phase/spatial projected RGB edge layout is invalid")
    phase_available = phase_usable.to(device=device).unsqueeze(0).expand_as(spatial_usable)
    if edge_availability_override is None:
        availability = torch.ones_like(phase_usable, dtype=torch.bool, device=device)
    else:
        availability = torch.as_tensor(
            edge_availability_override, dtype=torch.bool, device=device
        )
        if availability.shape != phase_usable.shape:
            raise ValueError("phase/spatial edge availability override has the wrong layout")
    common = phase_available & spatial_usable & availability.unsqueeze(0)
    combined = identity * phase_values.unsqueeze(0) + spatial * spatial_values
    missing = float(missing_edge_log_likelihood_ratio)
    if not math.isfinite(missing):
        raise ValueError("phase/spatial hybrid missing value is invalid")
    return torch.where(common, combined, torch.full_like(combined, missing)), common, (identity, spatial)


def score_candidate_phase_identity_spatial_batch(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    phase_prediction: CandidateMultiscalePhaseIdentityPrediction,
    spatial_prediction: CandidateHighresRGBMultiscalePrediction,
    candidate_projection_offsets_xy: torch.Tensor,
    candidate_projection_valid: torch.Tensor,
    phase_source_name: str,
    spatial_source_name: str = "fine",
    identity_weight: float = 1.0,
    spatial_weight: float = 1.0,
    edge_availability_override: torch.Tensor | None = None,
    missing_edge_log_likelihood_ratio: float = 0.0,
    max_abs_spatial_log_likelihood_ratio: float = 6.0,
) -> CandidatePhaseIdentitySpatialPoseScore:
    """Marginalize strict hybrid edges under immutable candidate/null mass."""

    edge, usable, weights = phase_identity_spatial_edge_log_likelihood_ratios(
        runtime=runtime,
        phase_prediction=phase_prediction,
        spatial_prediction=spatial_prediction,
        candidate_projection_offsets_xy=candidate_projection_offsets_xy,
        candidate_projection_valid=candidate_projection_valid,
        phase_source_name=phase_source_name,
        spatial_source_name=spatial_source_name,
        identity_weight=float(identity_weight),
        spatial_weight=float(spatial_weight),
        edge_availability_override=edge_availability_override,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
        max_abs_spatial_log_likelihood_ratio=float(max_abs_spatial_log_likelihood_ratio),
    )
    active = runtime.to(edge.device)
    point, candidate = fixed_candidate_view_mixture_log_ratio(
        edge_log_likelihood_ratios=edge,
        edge_usable=usable,
        candidate_view_weights=active.candidate_view_weights,
        candidate_probabilities=active.candidate_probabilities,
        null_probabilities=active.null_probabilities,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
    )
    return CandidatePhaseIdentitySpatialPoseScore(
        pose_log_likelihood_ratios=point.mean(dim=1),
        point_log_likelihood_ratios=point,
        candidate_log_likelihood_ratios=candidate,
        edge_log_likelihood_ratios=edge,
        edge_usable=usable,
        phase_source_name=str(phase_source_name),
        spatial_source_name=str(spatial_source_name),
        identity_weight=weights[0],
        spatial_weight=weights[1],
    )


def selected_candidate_phase_identity_spatial_log_likelihood_ratios(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    phase_prediction: CandidateMultiscalePhaseIdentityPrediction,
    spatial_prediction: CandidateHighresRGBMultiscalePrediction,
    point_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    offsets_xy: torch.Tensor,
    projection_valid: torch.Tensor | None = None,
    phase_source_name: str,
    spatial_source_name: str = "fine",
    identity_weight: float = 1.0,
    spatial_weight: float = 1.0,
    edge_availability_override: torch.Tensor | None = None,
    missing_edge_log_likelihood_ratio: float = 0.0,
    max_abs_spatial_log_likelihood_ratio: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-view hybrid scores for direct train-only hard-repeat rows.

    The selected rows are not inputs to either visual encoder.  They are used
    only after the normal target-free phase and RGB predictions have been
    emitted, which makes this suitable for correct-versus-coherent-wrong
    margin supervision and paired controls.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("selected phase/spatial scoring requires a target-free runtime")
    if not isinstance(phase_prediction, CandidateMultiscalePhaseIdentityPrediction) or not isinstance(
        spatial_prediction, CandidateHighresRGBMultiscalePrediction
    ):
        raise ValueError("selected phase/spatial scoring requires target-free visual predictions")
    phase_source = str(phase_source_name)
    spatial_source = str(spatial_source_name)
    if phase_source not in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES or spatial_source not in {
        "fine",
        "broad",
    }:
        raise ValueError("selected phase/spatial source is invalid")
    identity, spatial = resolve_phase_spatial_weights(
        identity_weight=float(identity_weight), spatial_weight=float(spatial_weight)
    )
    device = spatial_prediction.sources["fine"].joint_log_probabilities.device
    active = runtime.to(device)
    points = torch.as_tensor(point_indices, dtype=torch.long, device=device).reshape(-1)
    candidates = torch.as_tensor(candidate_indices, dtype=torch.long, device=device).reshape(-1)
    offsets = torch.as_tensor(offsets_xy, dtype=torch.float32, device=device)
    valid = (
        torch.ones((len(points),), dtype=torch.bool, device=device)
        if projection_valid is None
        else torch.as_tensor(projection_valid, dtype=torch.bool, device=device).reshape(-1)
    )
    if (
        len(points) == 0
        or candidates.shape != points.shape
        or offsets.shape != (len(points), 2)
        or valid.shape != points.shape
        or torch.any(points < 0)
        or torch.any(points >= active.point_count)
        or torch.any(candidates < 0)
        or torch.any(candidates >= active.candidate_count)
        or not torch.isfinite(offsets).all()
    ):
        raise ValueError("selected phase/spatial rows are invalid")
    phase_values = phase_prediction.source_edge_log_likelihood_ratios[phase_source].to(device)
    phase_usable = phase_prediction.source_edge_usable[phase_source].to(device)
    scale = spatial_prediction.sources[spatial_source]
    if (
        phase_values.shape != tuple(active.support_image_indices.shape)
        or phase_usable.shape != phase_values.shape
        or scale.edge_usable.shape != phase_values.shape
    ):
        raise ValueError("selected phase/spatial visual layouts differ")
    selected_joint = scale.joint_log_probabilities[points, candidates].unsqueeze(1)
    local, in_window = continuous_joint_log_probability_at_offsets(
        joint_log_probabilities=selected_joint,
        offsets_xy=scale.offsets_xy,
        query_offsets_xy=offsets.reshape(1, len(points), 1, 2),
    )
    category_count = int(selected_joint.shape[-1] - 1)
    spatial_raw = (
        local[0, :, 0]
        + math.log(2.0 * float(category_count))
        + scale.edge_log_likelihood_ratios[points, candidates]
    )
    spatial_values = bounded_log_likelihood_ratio(
        spatial_raw, max_abs_log_ratio=float(max_abs_spatial_log_likelihood_ratio)
    )
    spatial_usable = (
        valid[:, None]
        & in_window[0, :, 0]
        & scale.edge_usable[points, candidates]
    )
    if edge_availability_override is None:
        availability = torch.ones_like(phase_usable, dtype=torch.bool, device=device)
    else:
        availability = torch.as_tensor(
            edge_availability_override, dtype=torch.bool, device=device
        )
        if availability.shape != phase_usable.shape:
            raise ValueError("selected phase/spatial availability override has the wrong layout")
    common = spatial_usable & phase_usable[points, candidates] & availability[points, candidates]
    missing = float(missing_edge_log_likelihood_ratio)
    if not math.isfinite(missing):
        raise ValueError("selected phase/spatial missing value is invalid")
    values = identity * phase_values[points, candidates] + spatial * spatial_values
    values = torch.where(common, values, torch.full_like(values, missing))
    view_weights = active.candidate_view_weights[points, candidates]
    log_weights = torch.where(
        view_weights > 0.0,
        torch.log(view_weights),
        torch.full_like(view_weights, -torch.inf),
    )
    score = torch.logsumexp(values + log_weights, dim=1)
    candidate_usable = valid & torch.any(common & (view_weights > 0.0), dim=1)
    return score, candidate_usable


def permute_support_patches_with_phase_identity_point_blocks(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    support_patches: torch.Tensor,
    shift: int = 1,
) -> torch.Tensor:
    """Derange RGB support appearance with the exact phase-control point shift.

    Candidate slot, support-view slot, all geometry, and fixed posterior mass
    stay in their original locations.  Only raw support image content moves
    across distant query-point blocks.  Hybrid scoring later intersects the
    phase and RGB availability masks, so a moved invalid patch cannot become
    positive evidence.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("phase-block RGB permutation requires a target-free runtime")
    patches = torch.as_tensor(support_patches)
    expected = tuple(int(value) for value in runtime.support_image_indices.shape)
    if patches.ndim < 6 or tuple(int(value) for value in patches.shape[:3]) != expected:
        raise ValueError("phase-block RGB support patch layout is invalid")
    amount = phase_identity_point_block_derangement_shift(
        point_count=runtime.point_count, shift=int(shift)
    )
    result = torch.roll(patches, shifts=int(amount), dims=0)
    if torch.equal(result, patches):
        raise ValueError("phase-block RGB support derangement did not alter appearance")
    return result
