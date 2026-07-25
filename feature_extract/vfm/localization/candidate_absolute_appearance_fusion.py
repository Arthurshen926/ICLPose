"""Target-free candidate-prior conditioning with absolute visual evidence.

The two visual branches intentionally have different probabilistic roles:

* RADIO-final phase is an absolute, candidate-specific *identity* signal.  It
  reweights the frozen top-L candidate/null prior before pose scoring.
* Fine RGB is a local spatial likelihood.  It is the only term evaluated at
  an externally supplied pose projection.

An identity score is invariant to a pose hypothesis when the fixed candidate
layout is unchanged.  It must therefore never be added directly to a
pose-hypothesis score: doing so cannot improve correct-vs-wrong pose ranking
and can dilute the RGB spatial term.  This module keeps that separation
explicit while retaining a target-free runtime boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CandidateHighresRGBMultiscalePrediction,
    highres_rgb_edge_log_likelihood_ratio_at_pose_projection,
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


CANDIDATE_ABSOLUTE_APPEARANCE_FUSION_FORMAT = (
    "candidate_absolute_appearance_radiofinal_identity_prior_rgbfine_spatial_v2"
)
CANDIDATE_ABSOLUTE_APPEARANCE_FUSION_MODES = frozenset(
    {"fused", "rgb_only", "phase_identity_only"}
)


@dataclass(frozen=True)
class CandidateAbsoluteAppearanceEvidence:
    """Target-free output of the two visual experts before pose scoring."""

    phase_prediction: CandidateMultiscalePhaseIdentityPrediction
    rgb_prediction: CandidateHighresRGBMultiscalePrediction
    phase_source_name: str = "radio_final"

    def __post_init__(self) -> None:
        phase = self.phase_prediction
        rgb = self.rgb_prediction
        source = str(self.phase_source_name)
        if (
            not isinstance(phase, CandidateMultiscalePhaseIdentityPrediction)
            or not isinstance(rgb, CandidateHighresRGBMultiscalePrediction)
            or source not in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
            or source != "radio_final"
            or tuple(int(value) for value in phase.edge_usable.shape)
            != tuple(int(value) for value in rgb.edge_shape)
        ):
            raise ValueError("absolute appearance evidence has incompatible visual experts")
        object.__setattr__(self, "phase_source_name", source)


@dataclass(frozen=True)
class CandidateAbsoluteAppearancePoseScore:
    """Pose score plus the target-free candidate-prior conditioning state."""

    pose_log_likelihood_ratios: torch.Tensor
    point_log_likelihood_ratios: torch.Tensor
    candidate_log_likelihood_ratios: torch.Tensor
    edge_log_likelihood_ratios: torch.Tensor
    edge_usable: torch.Tensor
    phase_candidate_log_likelihood_ratios: torch.Tensor
    phase_candidate_usable: torch.Tensor
    conditioned_candidate_probabilities: torch.Tensor
    conditioned_null_probabilities: torch.Tensor
    phase_prior_strength: float
    mode: str

    def __post_init__(self) -> None:
        pose = torch.as_tensor(self.pose_log_likelihood_ratios, dtype=torch.float32)
        point = torch.as_tensor(self.point_log_likelihood_ratios, dtype=torch.float32, device=pose.device)
        candidate = torch.as_tensor(
            self.candidate_log_likelihood_ratios, dtype=torch.float32, device=pose.device
        )
        edge = torch.as_tensor(
            self.edge_log_likelihood_ratios, dtype=torch.float32, device=pose.device
        )
        usable = torch.as_tensor(self.edge_usable, dtype=torch.bool, device=pose.device)
        phase = torch.as_tensor(
            self.phase_candidate_log_likelihood_ratios, dtype=torch.float32, device=pose.device
        )
        phase_usable = torch.as_tensor(self.phase_candidate_usable, dtype=torch.bool, device=pose.device)
        probabilities = torch.as_tensor(
            self.conditioned_candidate_probabilities, dtype=torch.float32, device=pose.device
        )
        null = torch.as_tensor(self.conditioned_null_probabilities, dtype=torch.float32, device=pose.device)
        strength = float(self.phase_prior_strength)
        mode = str(self.mode)
        if (
            pose.ndim != 1
            or point.ndim != 2
            or candidate.ndim != 3
            or edge.ndim != 4
            or usable.shape != edge.shape
            or pose.shape != (point.shape[0],)
            or candidate.shape[:2] != point.shape
            or edge.shape[:3] != candidate.shape
            or phase.shape != probabilities.shape
            or phase_usable.shape != phase.shape
            or probabilities.shape != candidate.shape[1:]
            or null.shape != (probabilities.shape[0],)
            or mode not in CANDIDATE_ABSOLUTE_APPEARANCE_FUSION_MODES
            or not math.isfinite(strength)
            or strength < 0.0
            or not torch.isfinite(pose).all()
            or not torch.isfinite(point).all()
            or not torch.isfinite(candidate).all()
            or not torch.isfinite(edge).all()
            or not torch.isfinite(phase).all()
            or not torch.isfinite(probabilities).all()
            or not torch.isfinite(null).all()
            or torch.any(probabilities < 0.0)
            or torch.any(null < 0.0)
            or torch.any(torch.abs(probabilities.sum(dim=1) + null - 1.0) > 1e-4)
        ):
            raise ValueError("absolute appearance pose score is invalid")
        object.__setattr__(self, "pose_log_likelihood_ratios", pose)
        object.__setattr__(self, "point_log_likelihood_ratios", point)
        object.__setattr__(self, "candidate_log_likelihood_ratios", candidate)
        object.__setattr__(self, "edge_log_likelihood_ratios", edge)
        object.__setattr__(self, "edge_usable", usable)
        object.__setattr__(self, "phase_candidate_log_likelihood_ratios", phase)
        object.__setattr__(self, "phase_candidate_usable", phase_usable)
        object.__setattr__(self, "conditioned_candidate_probabilities", probabilities)
        object.__setattr__(self, "conditioned_null_probabilities", null)
        object.__setattr__(self, "phase_prior_strength", strength)
        object.__setattr__(self, "mode", mode)


def _candidate_view_log_likelihood_ratio(
    *,
    edge_log_likelihood_ratios: torch.Tensor,
    edge_usable: torch.Tensor,
    candidate_view_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize support views without mixing candidate or null mass."""

    edges = torch.as_tensor(edge_log_likelihood_ratios, dtype=torch.float32)
    usable = torch.as_tensor(edge_usable, dtype=torch.bool, device=edges.device)
    weights = torch.as_tensor(candidate_view_weights, dtype=torch.float32, device=edges.device)
    if (
        edges.ndim != 4
        or usable.shape != edges.shape
        or weights.shape != edges.shape[1:]
        or not torch.isfinite(edges).all()
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0.0)
    ):
        raise ValueError("candidate view likelihood inputs are invalid")
    view_mass = weights.sum(dim=2)
    positive = view_mass > 0.0
    if torch.any(torch.abs(view_mass[positive] - 1.0) > 1e-4):
        raise ValueError("candidate view weights must sum to one")
    effective = torch.where(usable, edges, torch.zeros_like(edges))
    safe_log_weights = torch.where(
        weights > 0.0,
        torch.log(weights),
        torch.full_like(weights, -torch.inf),
    )
    values = torch.logsumexp(effective + safe_log_weights.unsqueeze(0), dim=3)
    values = torch.where(positive.unsqueeze(0), values, torch.zeros_like(values))
    candidate_usable = (usable & (weights > 0.0).unsqueeze(0)).any(dim=3) & positive.unsqueeze(0)
    return values, candidate_usable


class CandidateAbsoluteAppearanceFusion(nn.Module):
    """Use RADIO identity only to condition the fixed candidate/null mixture.

    ``phase_prior_strength`` is one globally shared bounded scalar.  It cannot
    inspect an individual pose, residual, track id, candidate rank, or label.
    The RGB term remains the sole hypothesis-projected visual likelihood.
    """

    def __init__(
        self,
        *,
        initial_phase_prior_strength: float = 1.0,
        max_phase_prior_strength: float = 2.0,
    ) -> None:
        super().__init__()
        initial = float(initial_phase_prior_strength)
        maximum = float(max_phase_prior_strength)
        if (
            not math.isfinite(initial)
            or not math.isfinite(maximum)
            or maximum <= 0.0
            or initial <= 0.0
            or initial >= maximum
        ):
            raise ValueError("phase prior strength must be finite and strictly within its bound")
        self.max_phase_prior_strength = maximum
        initial_fraction = initial / maximum
        self.phase_prior_logit = nn.Parameter(
            torch.tensor(math.log(initial_fraction / (1.0 - initial_fraction)), dtype=torch.float32)
        )

    def learned_phase_prior_strength(self) -> torch.Tensor:
        """Return the differentiable, globally bounded RADIO prior scale."""

        return self.max_phase_prior_strength * torch.sigmoid(self.phase_prior_logit)

    def calibration_state(self) -> dict[str, torch.Tensor]:
        return {"radio_final_phase_prior_strength": self.learned_phase_prior_strength()}

    def forward(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        phase_prediction: CandidateMultiscalePhaseIdentityPrediction,
        rgb_prediction: CandidateHighresRGBMultiscalePrediction,
    ) -> CandidateAbsoluteAppearanceEvidence:
        """Validate target-free visual outputs before any pose is supplied."""

        if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
            raise ValueError("absolute appearance fusion requires a target-free runtime")
        evidence = CandidateAbsoluteAppearanceEvidence(
            phase_prediction=phase_prediction,
            rgb_prediction=rgb_prediction,
        )
        if tuple(int(value) for value in runtime.support_view_valid.shape) != tuple(
            int(value) for value in evidence.phase_prediction.edge_usable.shape
        ):
            raise ValueError("absolute appearance evidence differs from the fixed runtime")
        return evidence

    def phase_candidate_log_likelihood_ratios(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        evidence: CandidateAbsoluteAppearanceEvidence,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return target-free candidate identity LLRs after fixed-view pooling."""

        if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
            raise ValueError("phase candidate scoring requires a target-free runtime")
        if not isinstance(evidence, CandidateAbsoluteAppearanceEvidence):
            raise ValueError("phase candidate scoring requires target-free evidence")
        device = evidence.rgb_prediction.sources["fine"].joint_log_probabilities.device
        active = runtime.to(device)
        values = evidence.phase_prediction.source_edge_log_likelihood_ratios[
            evidence.phase_source_name
        ].to(device=device, dtype=torch.float32)
        usable = evidence.phase_prediction.source_edge_usable[evidence.phase_source_name].to(
            device=device, dtype=torch.bool
        )
        if values.shape != active.support_view_valid.shape or usable.shape != values.shape:
            raise ValueError("absolute appearance phase layout differs from the fixed runtime")
        candidate_values, candidate_usable = _candidate_view_log_likelihood_ratio(
            edge_log_likelihood_ratios=values.unsqueeze(0),
            edge_usable=(usable & active.support_view_valid).unsqueeze(0),
            candidate_view_weights=active.candidate_view_weights,
        )
        positive = active.candidate_probabilities > 0.0
        return (
            torch.where(positive, candidate_values[0], torch.zeros_like(candidate_values[0])),
            candidate_usable[0] & positive,
        )

    def phase_conditioned_candidate_probabilities(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        evidence: CandidateAbsoluteAppearanceEvidence,
        phase_prior_strength: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reweight candidates with RADIO identity while preserving explicit null.

        The all-zero/missing phase case returns the frozen runtime probabilities
        exactly.  This makes phase ablations and the structural-zero control a
        real no-op instead of a softmax-rounding approximation.
        """

        phase_values, phase_usable = self.phase_candidate_log_likelihood_ratios(
            runtime=runtime,
            evidence=evidence,
        )
        device = phase_values.device
        active = runtime.to(device)
        if phase_prior_strength is None:
            strength = self.learned_phase_prior_strength().to(device=device, dtype=torch.float32)
        else:
            strength = torch.as_tensor(phase_prior_strength, dtype=torch.float32, device=device)
            if strength.ndim != 0 or not bool(torch.isfinite(strength)) or float(strength.detach().item()) < 0.0:
                raise ValueError("phase prior strength override is invalid")
            if float(strength.detach().item()) > self.max_phase_prior_strength + 1e-6:
                raise ValueError("phase prior strength override exceeds its calibrated bound")
        if float(strength.detach().item()) == 0.0:
            return (
                active.candidate_probabilities,
                active.null_probabilities,
                phase_values,
                phase_usable,
                strength,
            )
        positive = active.candidate_probabilities > 0.0
        adjustment = torch.where(phase_usable, strength * phase_values, torch.zeros_like(phase_values))
        candidate_logits = torch.where(
            positive,
            torch.log(active.candidate_probabilities) + adjustment,
            torch.full_like(adjustment, -torch.inf),
        )
        null_logits = torch.where(
            active.null_probabilities > 0.0,
            torch.log(active.null_probabilities),
            torch.full_like(active.null_probabilities, -torch.inf),
        )
        conditioned = torch.softmax(torch.cat((candidate_logits, null_logits.unsqueeze(1)), dim=1), dim=1)
        phase_changes_row = (
            phase_usable & positive & (phase_values != 0.0)
        ).any(dim=1)
        candidate_probabilities = torch.where(
            phase_changes_row.unsqueeze(1),
            conditioned[:, :-1],
            active.candidate_probabilities,
        )
        null_probabilities = torch.where(
            phase_changes_row,
            conditioned[:, -1],
            active.null_probabilities,
        )
        return candidate_probabilities, null_probabilities, phase_values, phase_usable, strength

    def candidate_identity_log_likelihood_ratios(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        evidence: CandidateAbsoluteAppearanceEvidence,
        candidate_projection_offsets_xy: torch.Tensor,
        candidate_projection_valid: torch.Tensor,
        mode: str = "fused",
        phase_prior_strength: float | torch.Tensor | None = None,
        missing_edge_log_likelihood_ratio: float = 0.0,
        max_abs_log_likelihood_ratio: float = 6.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score individual candidates for train-only identity/repeat audits.

        This direct edge diagnostic is distinct from hypothesis scoring.  The
        fused value combines candidate identity and local RGB evidence at a
        supplied observation offset, while :meth:`score` uses phase only to
        condition candidate mass before evaluating a pose-projected RGB term.
        """

        selected_mode = str(mode)
        if selected_mode not in CANDIDATE_ABSOLUTE_APPEARANCE_FUSION_MODES:
            raise ValueError("absolute appearance identity mode is invalid")
        phase_values, phase_usable = self.phase_candidate_log_likelihood_ratios(
            runtime=runtime,
            evidence=evidence,
        )
        if selected_mode == "phase_identity_only":
            batch = int(torch.as_tensor(candidate_projection_offsets_xy).shape[0])
            if batch <= 0:
                raise ValueError("absolute appearance identity projection batch is invalid")
            return (
                phase_values.unsqueeze(0).expand(batch, -1, -1),
                phase_usable.unsqueeze(0).expand(batch, -1, -1),
            )
        rgb_values, rgb_usable, _ = highres_rgb_edge_log_likelihood_ratio_at_pose_projection(
            prediction=evidence.rgb_prediction,
            candidate_projection_offsets_xy=candidate_projection_offsets_xy,
            candidate_projection_valid=candidate_projection_valid,
            source="fine",
            missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
            max_abs_log_likelihood_ratio=float(max_abs_log_likelihood_ratio),
        )
        active = runtime.to(rgb_values.device)
        rgb_candidate, rgb_candidate_usable = _candidate_view_log_likelihood_ratio(
            edge_log_likelihood_ratios=rgb_values,
            edge_usable=rgb_usable,
            candidate_view_weights=active.candidate_view_weights,
        )
        if selected_mode == "rgb_only":
            return rgb_candidate, rgb_candidate_usable
        _, _, _, _, strength = self.phase_conditioned_candidate_probabilities(
            runtime=runtime,
            evidence=evidence,
            phase_prior_strength=phase_prior_strength,
        )
        phase_expanded = phase_values.unsqueeze(0).expand_as(rgb_candidate)
        phase_usable_expanded = phase_usable.unsqueeze(0).expand_as(rgb_candidate_usable)
        values = rgb_candidate + strength * torch.where(
            phase_usable_expanded, phase_expanded, torch.zeros_like(phase_expanded)
        )
        return values, rgb_candidate_usable | phase_usable_expanded

    def score(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        evidence: CandidateAbsoluteAppearanceEvidence,
        candidate_projection_offsets_xy: torch.Tensor,
        candidate_projection_valid: torch.Tensor,
        mode: str = "fused",
        phase_prior_strength: float | torch.Tensor | None = None,
        missing_edge_log_likelihood_ratio: float = 0.0,
        max_abs_log_likelihood_ratio: float = 6.0,
    ) -> CandidateAbsoluteAppearancePoseScore:
        """Score externally projected poses with RGB under a phase-conditioned prior.

        An unavailable or out-of-window RGB projection contributes exactly the
        fixed neutral factor.  RADIO changes only the candidate/null mixture;
        therefore it cannot create a pose-specific reward on its own.
        """

        if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
            raise ValueError("absolute appearance scoring requires a target-free runtime")
        if not isinstance(evidence, CandidateAbsoluteAppearanceEvidence):
            raise ValueError("absolute appearance scoring requires target-free evidence")
        selected_mode = str(mode)
        if selected_mode not in CANDIDATE_ABSOLUTE_APPEARANCE_FUSION_MODES:
            raise ValueError("absolute appearance pose mode is invalid")
        device = evidence.rgb_prediction.sources["fine"].joint_log_probabilities.device
        offsets = torch.as_tensor(candidate_projection_offsets_xy, dtype=torch.float32, device=device)
        valid = torch.as_tensor(candidate_projection_valid, dtype=torch.bool, device=device)
        active = runtime.to(device)
        if (
            offsets.ndim != 4
            or offsets.shape[-1] != 2
            or valid.shape != offsets.shape[:-1]
            or offsets.shape[0] <= 0
            or offsets.shape[1:3] != active.support_view_valid.shape[:2]
            or not torch.isfinite(offsets).all()
        ):
            raise ValueError("absolute appearance pose projection inputs are invalid")
        effective_strength: float | torch.Tensor | None = phase_prior_strength
        if selected_mode == "rgb_only":
            effective_strength = 0.0
        candidate_probabilities, null_probabilities, phase_values, phase_usable, strength = (
            self.phase_conditioned_candidate_probabilities(
                runtime=runtime,
                evidence=evidence,
                phase_prior_strength=effective_strength,
            )
        )
        if selected_mode == "phase_identity_only":
            edge_values = torch.zeros(
                (*offsets.shape[:-1], active.support_view_valid.shape[2]),
                dtype=torch.float32,
                device=device,
            )
            edge_usable = torch.zeros_like(edge_values, dtype=torch.bool)
            point_values = torch.zeros((offsets.shape[0], active.point_count), dtype=torch.float32, device=device)
            candidate_values = phase_values.unsqueeze(0).expand(offsets.shape[0], -1, -1)
        else:
            edge_values, edge_usable, _ = highres_rgb_edge_log_likelihood_ratio_at_pose_projection(
                prediction=evidence.rgb_prediction,
                candidate_projection_offsets_xy=offsets,
                candidate_projection_valid=valid,
                source="fine",
                missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
                max_abs_log_likelihood_ratio=float(max_abs_log_likelihood_ratio),
            )
            point_values, candidate_values = fixed_candidate_view_mixture_log_ratio(
                edge_log_likelihood_ratios=edge_values,
                edge_usable=edge_usable,
                candidate_view_weights=active.candidate_view_weights,
                candidate_probabilities=candidate_probabilities,
                null_probabilities=null_probabilities,
                missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
            )
        return CandidateAbsoluteAppearancePoseScore(
            pose_log_likelihood_ratios=point_values.mean(dim=1),
            point_log_likelihood_ratios=point_values,
            candidate_log_likelihood_ratios=candidate_values,
            edge_log_likelihood_ratios=edge_values,
            edge_usable=edge_usable,
            phase_candidate_log_likelihood_ratios=phase_values,
            phase_candidate_usable=phase_usable,
            conditioned_candidate_probabilities=candidate_probabilities,
            conditioned_null_probabilities=null_probabilities,
            phase_prior_strength=float(strength.detach().item()),
            mode=selected_mode,
        )


def phase_conditioned_candidate_posterior_nll(
    *,
    fusion: CandidateAbsoluteAppearanceFusion,
    runtime: CandidatePoseRGBSpatialRuntime,
    evidence: CandidateAbsoluteAppearanceEvidence,
    observed_candidate_mask: torch.Tensor,
    target_dustbin: torch.Tensor,
    target_supervised: torch.Tensor,
    balance_observed_and_null: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise the exact candidate/null posterior used by v2 pose scoring.

    The visual encoders have already run when this function is called.  It is
    therefore a train-only target join, not a target-bearing runtime input.
    In contrast with an edge-level identity loss, this objective directly
    trains the posterior that RADIO uses to reweight the frozen top-L mixture.
    """

    if not isinstance(fusion, CandidateAbsoluteAppearanceFusion):
        raise ValueError("phase posterior NLL requires an absolute appearance fusion module")
    candidates, null, _, usable, _ = fusion.phase_conditioned_candidate_probabilities(
        runtime=runtime,
        evidence=evidence,
    )
    observed = torch.as_tensor(observed_candidate_mask, dtype=torch.bool, device=candidates.device)
    dustbin = torch.as_tensor(target_dustbin, dtype=torch.bool, device=candidates.device).reshape(-1)
    supervised = torch.as_tensor(target_supervised, dtype=torch.bool, device=candidates.device).reshape(-1)
    if (
        observed.shape != candidates.shape
        or dustbin.shape != (len(candidates),)
        or supervised.shape != dustbin.shape
        or torch.any(observed & ~supervised[:, None])
        or torch.any(dustbin & ~supervised)
        or torch.any(observed.sum(dim=1) > 1)
        or torch.any(observed.any(dim=1) & dustbin)
    ):
        raise ValueError("phase-conditioned posterior identity target contract is invalid")
    labels = torch.where(
        dustbin,
        torch.full((len(candidates),), candidates.shape[1], dtype=torch.long, device=candidates.device),
        observed.to(dtype=torch.long).argmax(dim=1),
    )
    observed_rows = observed.any(dim=1)
    target_candidate_usable = usable.gather(
        1, labels[:, None].clamp_max(usable.shape[1] - 1)
    ).squeeze(1)
    active = supervised & torch.where(
        observed_rows,
        target_candidate_usable & (usable.sum(dim=1) >= 2),
        usable.any(dim=1),
    )
    probabilities = torch.cat((candidates, null[:, None]), dim=1)
    logits = torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny))
    if not bool(active.any()):
        return logits.sum() * 0.0, {
            "posterior_identity_active": 0.0,
            "posterior_identity_observed_active": 0.0,
            "posterior_identity_null_active": 0.0,
            "posterior_identity_top1": 0.0,
            "posterior_identity_nll": 0.0,
        }
    individual = F.nll_loss(logits[active], labels[active], reduction="none")
    active_observed = observed_rows[active]
    if bool(balance_observed_and_null) and bool(active_observed.any()) and bool((~active_observed).any()):
        loss = 0.5 * (
            individual[active_observed].mean() + individual[~active_observed].mean()
        )
    else:
        loss = individual.mean()
    top1 = logits.argmax(dim=1)
    return loss, {
        "posterior_identity_active": float(active.sum().item()),
        "posterior_identity_observed_active": float((active & observed_rows).sum().item()),
        "posterior_identity_null_active": float((active & ~observed_rows).sum().item()),
        "posterior_identity_top1": float((top1[active] == labels[active]).float().mean().item()),
        "posterior_identity_nll": float(loss.detach().item()),
    }


def phase_conditioned_candidate_posterior_margin_loss(
    *,
    fusion: CandidateAbsoluteAppearanceFusion,
    runtime: CandidatePoseRGBSpatialRuntime,
    evidence: CandidateAbsoluteAppearanceEvidence,
    point_indices: torch.Tensor,
    positive_candidate_indices: torch.Tensor,
    negative_candidate_indices: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Push exact hard-repeat pairs apart in the posterior used at runtime."""

    value = float(margin)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("phase-conditioned posterior margin is invalid")
    candidates, _, _, usable, _ = fusion.phase_conditioned_candidate_probabilities(
        runtime=runtime,
        evidence=evidence,
    )
    points = torch.as_tensor(point_indices, dtype=torch.long, device=candidates.device).reshape(-1)
    positive = torch.as_tensor(
        positive_candidate_indices, dtype=torch.long, device=candidates.device
    ).reshape(-1)
    negative = torch.as_tensor(
        negative_candidate_indices, dtype=torch.long, device=candidates.device
    ).reshape(-1)
    if (
        len(points) == 0
        or positive.shape != points.shape
        or negative.shape != points.shape
        or torch.any(points < 0)
        or torch.any(points >= len(candidates))
        or torch.any(positive < 0)
        or torch.any(positive >= candidates.shape[1])
        or torch.any(negative < 0)
        or torch.any(negative >= candidates.shape[1])
        or torch.any(positive == negative)
    ):
        raise ValueError("phase-conditioned posterior hard-repeat indices are invalid")
    active = usable[points, positive] & usable[points, negative]
    log_probabilities = torch.log(candidates.clamp_min(torch.finfo(candidates.dtype).tiny))
    gaps = log_probabilities[points, positive] - log_probabilities[points, negative]
    if not bool(active.any()):
        return gaps.sum() * 0.0, {
            "posterior_hard_repeat_active": 0.0,
            "posterior_hard_repeat_gap": 0.0,
            "posterior_hard_repeat_win": 0.0,
        }
    active_gaps = gaps[active]
    loss = F.softplus(torch.as_tensor(value, dtype=active_gaps.dtype, device=active_gaps.device) - active_gaps).mean()
    return loss, {
        "posterior_hard_repeat_active": float(active.sum().item()),
        "posterior_hard_repeat_gap": float(active_gaps.detach().mean().item()),
        "posterior_hard_repeat_win": float((active_gaps.detach() > 0.0).float().mean().item()),
    }
