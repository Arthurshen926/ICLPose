"""Target-free continuous aggregation for candidate pose evidence.

This is not the RADIO feature mapper and it does not rerank landmarks.  Given
the frozen candidate-specific visual likelihood outputs for one query, it
produces a continuous weight for every query token before a pose projection is
evaluated.  Runtime inputs are restricted to:

* RGB local-mode shape statistics;
* phase-context identity statistics from frozen RADIO/ALIKE branches; and
* fixed support-view validity/mass already present in the target-free layout.

It deliberately excludes pose matrices, projected offsets, residuals, track
identities, candidate ranks, coarse scores, labels, and supervision.  A
training caller may join correct/coherent-wrong pose scores only after this
module has emitted its target-free features and weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import nn

from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CandidateHighresRGBMultiscalePrediction,
)
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CandidateMultiscalePhaseIdentityPrediction,
)
from feature_extract.vfm.localization.candidate_pose_evidence_selector import (
    phase_identity_confidence,
    rgb_spatial_mode_quality,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
)


CANDIDATE_POSE_EVIDENCE_WEIGHTER_FORMAT = "candidate_pose_evidence_weighter_v1"
TARGET_FREE_EVIDENCE_FEATURE_NAMES = (
    "rgb_mode_quality_top1",
    "rgb_mode_quality_top1_minus_top2",
    "rgb_mode_quality_candidate_mean",
    "rgb_mode_quality_candidate_concentration",
    "rgb_non_dustbin_top1",
    "rgb_non_dustbin_top1_minus_top2",
    "rgb_local_peak_top1",
    "phase_identity_concentration",
)
TARGET_FREE_EVIDENCE_FEATURE_POLICIES = (
    "rgb_only",
    "rgb_plus_phase",
)


@dataclass(frozen=True)
class TargetFreePoseEvidenceFeatures:
    """Per-token visual summaries with no pose or supervision fields."""

    values: torch.Tensor
    feature_names: tuple[str, ...] = TARGET_FREE_EVIDENCE_FEATURE_NAMES

    def __post_init__(self) -> None:
        features = torch.as_tensor(self.values, dtype=torch.float32)
        names = tuple(str(name) for name in self.feature_names)
        if (
            features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] != len(TARGET_FREE_EVIDENCE_FEATURE_NAMES)
            or names != TARGET_FREE_EVIDENCE_FEATURE_NAMES
            or not torch.isfinite(features).all()
        ):
            raise ValueError("target-free pose evidence features are invalid")
        object.__setattr__(self, "values", features)
        object.__setattr__(self, "feature_names", names)

    @property
    def point_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class TargetFreePoseEvidenceWeightPrediction:
    """A normalized static token distribution and its uniform fallback mass."""

    weights: torch.Tensor
    local_weights: torch.Tensor
    uniform_mass: torch.Tensor
    local_logits: torch.Tensor

    def __post_init__(self) -> None:
        weights = torch.as_tensor(self.weights, dtype=torch.float32).reshape(-1)
        local = torch.as_tensor(self.local_weights, dtype=torch.float32, device=weights.device).reshape(-1)
        uniform = torch.as_tensor(self.uniform_mass, dtype=torch.float32, device=weights.device).reshape(-1)
        logits = torch.as_tensor(self.local_logits, dtype=torch.float32, device=weights.device).reshape(-1)
        if (
            len(weights) == 0
            or local.shape != weights.shape
            or logits.shape != weights.shape
            or uniform.shape != (1,)
            or not torch.isfinite(weights).all()
            or not torch.isfinite(local).all()
            or not torch.isfinite(uniform).all()
            or not torch.isfinite(logits).all()
            or torch.any(weights < 0.0)
            or torch.any(local < 0.0)
            or not bool(torch.allclose(weights.sum(), torch.ones((), device=weights.device), atol=1e-5, rtol=1e-5))
            or not bool(torch.allclose(local.sum(), torch.ones((), device=weights.device), atol=1e-5, rtol=1e-5))
            or bool(uniform[0] < 0.0)
            or bool(uniform[0] > 1.0)
        ):
            raise ValueError("target-free pose evidence weights are invalid")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "local_weights", local)
        object.__setattr__(self, "uniform_mass", uniform)
        object.__setattr__(self, "local_logits", logits)


def _candidate_top_two(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return candidate top-1/top-2 without relying on candidate rank input."""

    candidate_values = torch.as_tensor(values, dtype=torch.float32)
    if candidate_values.ndim != 2 or candidate_values.shape[0] == 0 or candidate_values.shape[1] == 0:
        raise ValueError("candidate evidence statistics require nonempty candidate values")
    top = torch.topk(candidate_values, k=min(2, candidate_values.shape[1]), dim=1).values
    first = top[:, 0]
    second = top[:, 1] if top.shape[1] == 2 else torch.zeros_like(first)
    return first, second


def _candidate_concentration(values: torch.Tensor) -> torch.Tensor:
    """Return entropy concentration over visual candidate quality only."""

    candidate_values = torch.as_tensor(values, dtype=torch.float32)
    total = candidate_values.sum(dim=1)
    probabilities = candidate_values / total[:, None].clamp_min(torch.finfo(candidate_values.dtype).tiny)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(torch.finfo(candidate_values.dtype).tiny))).sum(
        dim=1
    )
    maximum = math.log(float(candidate_values.shape[1]))
    concentration = 1.0 - entropy / max(maximum, 1e-12)
    return torch.where(total > 0.0, concentration.clamp(0.0, 1.0), torch.zeros_like(total))


def build_target_free_pose_evidence_features(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    phase_prediction: CandidateMultiscalePhaseIdentityPrediction,
    rgb_prediction: CandidateHighresRGBMultiscalePrediction,
    edge_availability_override: torch.Tensor | None = None,
    phase_source_name: str = "radio_final",
    rgb_source_name: str = "fine",
) -> TargetFreePoseEvidenceFeatures:
    """Summarize frozen visual outputs before any pose projection is supplied."""

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("pose evidence features require a target-free runtime")
    if not isinstance(phase_prediction, CandidateMultiscalePhaseIdentityPrediction) or not isinstance(
        rgb_prediction, CandidateHighresRGBMultiscalePrediction
    ):
        raise ValueError("pose evidence features require target-free visual predictions")
    source = str(rgb_source_name)
    if source not in rgb_prediction.sources:
        raise ValueError("pose evidence RGB source is invalid")
    scale = rgb_prediction.sources[source]
    device = scale.joint_log_probabilities.device
    active = runtime.to(device)
    usable = scale.edge_usable.to(device=device)
    if usable.shape != tuple(active.support_image_indices.shape):
        raise ValueError("pose evidence RGB layout differs from target-free runtime")
    if edge_availability_override is not None:
        override = torch.as_tensor(edge_availability_override, dtype=torch.bool, device=device)
        if override.shape != usable.shape:
            raise ValueError("pose evidence availability override has the wrong layout")
        usable = usable & override
    joint = scale.joint_log_probabilities.to(device=device, dtype=torch.float32)
    local = torch.exp(joint[..., :-1])
    non_dustbin = local.sum(dim=-1)
    conditional = local / non_dustbin.unsqueeze(-1).clamp_min(torch.finfo(local.dtype).tiny)
    entropy = -(conditional * torch.log(conditional.clamp_min(torch.finfo(local.dtype).tiny))).sum(dim=-1)
    compactness = 1.0 - entropy / max(math.log(float(local.shape[-1])), 1e-12)
    edge_quality = torch.where(
        usable,
        non_dustbin * compactness.clamp(0.0, 1.0),
        torch.zeros_like(non_dustbin),
    )
    edge_non_dustbin = torch.where(usable, non_dustbin, torch.zeros_like(non_dustbin))
    edge_peak = torch.where(usable, local.amax(dim=-1), torch.zeros_like(non_dustbin))
    view_weights = active.candidate_view_weights
    candidate_quality = (edge_quality * view_weights).sum(dim=2)
    candidate_non_dustbin = (edge_non_dustbin * view_weights).sum(dim=2)
    candidate_peak = (edge_peak * view_weights).sum(dim=2)
    quality_top1, quality_top2 = _candidate_top_two(candidate_quality)
    non_dustbin_top1, non_dustbin_top2 = _candidate_top_two(candidate_non_dustbin)
    peak_top1, _unused_peak_top2 = _candidate_top_two(candidate_peak)
    del _unused_peak_top2
    phase = phase_identity_confidence(
        runtime=runtime,
        prediction=phase_prediction,
        source_name=str(phase_source_name),
        edge_availability_override=usable,
        include_fixed_candidate_prior=False,
    ).to(device=device)
    # Keep this assertion tied to the standalone RGB selector implementation;
    # otherwise the two branches could silently disagree on what "quality" is.
    standalone_quality = rgb_spatial_mode_quality(
        runtime=runtime,
        prediction=rgb_prediction,
        source_name=source,
        edge_availability_override=usable,
    ).to(device=device)
    if not torch.allclose(quality_top1, standalone_quality, atol=1e-5, rtol=1e-5):
        raise RuntimeError("pose evidence RGB quality drifted from the static selector")
    values = torch.stack(
        [
            quality_top1,
            (quality_top1 - quality_top2).clamp_min(0.0),
            candidate_quality.mean(dim=1),
            _candidate_concentration(candidate_quality),
            non_dustbin_top1,
            (non_dustbin_top1 - non_dustbin_top2).clamp_min(0.0),
            peak_top1,
            phase,
        ],
        dim=1,
    )
    return TargetFreePoseEvidenceFeatures(values=values)


def fit_target_free_feature_normalizer(
    feature_values: Iterable[torch.Tensor | TargetFreePoseEvidenceFeatures],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a train-partition-only visual feature normalizer."""

    rows: list[torch.Tensor] = []
    for item in feature_values:
        values = item.values if isinstance(item, TargetFreePoseEvidenceFeatures) else torch.as_tensor(item)
        values = torch.as_tensor(values, dtype=torch.float32)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] != len(TARGET_FREE_EVIDENCE_FEATURE_NAMES)
            or not torch.isfinite(values).all()
        ):
            raise ValueError("target-free feature normalizer inputs are invalid")
        rows.append(values.detach().cpu())
    if not rows:
        raise ValueError("target-free feature normalizer requires at least one query")
    stacked = torch.cat(rows, dim=0)
    center = stacked.mean(dim=0)
    scale = stacked.std(dim=0, unbiased=False).clamp_min(1e-4)
    return center, scale


def apply_target_free_feature_policy(
    *,
    features: TargetFreePoseEvidenceFeatures,
    policy: str,
) -> TargetFreePoseEvidenceFeatures:
    """Return a declared target-free feature subset without changing layout.

    ``phase_identity_concentration`` was falsified as a standalone static
    selector signal.  Keeping the fixed column layout while replacing it with
    a constant in the RGB-only policy makes the checkpoint schema explicit and
    prevents a small weighter from recovering that known-bad shortcut.
    """

    if not isinstance(features, TargetFreePoseEvidenceFeatures):
        raise ValueError("target-free feature policy requires evidence features")
    name = str(policy).strip().lower()
    if name not in TARGET_FREE_EVIDENCE_FEATURE_POLICIES:
        raise ValueError("target-free evidence feature policy is invalid")
    if name == "rgb_plus_phase":
        return features
    values = features.values.clone()
    phase_index = TARGET_FREE_EVIDENCE_FEATURE_NAMES.index("phase_identity_concentration")
    values[:, phase_index] = 0.0
    return TargetFreePoseEvidenceFeatures(values=values)


class TargetFreePoseEvidenceWeighter(nn.Module):
    """Learn a smooth, query-conditioned token distribution with a fallback.

    ``uniform_mass`` is predicted from pooled target-free visual statistics and
    bounded away from zero.  The local distribution is continuous over every
    token, not a hard top-K identity choice.  Its output can therefore be used
    unchanged for every pose hypothesis and visual counterfactual.
    """

    def __init__(
        self,
        *,
        feature_center: torch.Tensor,
        feature_scale: torch.Tensor,
        hidden_dim: int = 32,
        minimum_uniform_mass: float = 0.50,
        maximum_uniform_mass: float = 0.95,
        initial_uniform_mass: float = 0.75,
        max_abs_local_logit: float = 4.0,
    ) -> None:
        super().__init__()
        center = torch.as_tensor(feature_center, dtype=torch.float32).reshape(-1)
        scale = torch.as_tensor(feature_scale, dtype=torch.float32).reshape(-1)
        minimum = float(minimum_uniform_mass)
        maximum = float(maximum_uniform_mass)
        initial = float(initial_uniform_mass)
        logit_bound = float(max_abs_local_logit)
        if (
            center.shape != (len(TARGET_FREE_EVIDENCE_FEATURE_NAMES),)
            or scale.shape != center.shape
            or not torch.isfinite(center).all()
            or not torch.isfinite(scale).all()
            or torch.any(scale <= 0.0)
            or int(hidden_dim) < 4
            or not all(math.isfinite(value) for value in (minimum, maximum, initial, logit_bound))
            or minimum < 0.0
            or maximum > 1.0
            or minimum >= maximum
            or not minimum <= initial <= maximum
            or logit_bound <= 0.0
        ):
            raise ValueError("target-free pose evidence weighter configuration is invalid")
        dimension = int(len(center))
        self.register_buffer("feature_center", center, persistent=True)
        self.register_buffer("feature_scale", scale, persistent=True)
        self.local_network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.query_network = nn.Sequential(
            nn.LayerNorm(3 * dimension),
            nn.Linear(3 * dimension, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        # Start from a conservative uniform distribution.  The final layers
        # alone receive the first nonzero gradient, avoiding an accidental
        # pretraining-dependent selector at epoch zero.
        nn.init.zeros_(self.local_network[-1].weight)
        nn.init.zeros_(self.local_network[-1].bias)
        nn.init.zeros_(self.query_network[-1].weight)
        normalized_initial = (initial - minimum) / (maximum - minimum)
        nn.init.constant_(self.query_network[-1].bias, math.log(normalized_initial / (1.0 - normalized_initial)))
        self.minimum_uniform_mass = minimum
        self.maximum_uniform_mass = maximum
        self.max_abs_local_logit = logit_bound

    @property
    def feature_dim(self) -> int:
        return int(self.feature_center.numel())

    def forward(
        self, features: torch.Tensor | TargetFreePoseEvidenceFeatures
    ) -> TargetFreePoseEvidenceWeightPrediction:
        values = features.values if isinstance(features, TargetFreePoseEvidenceFeatures) else torch.as_tensor(features)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.feature_center.device)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] != self.feature_dim
            or not torch.isfinite(values).all()
        ):
            raise ValueError("target-free pose evidence weighter inputs are invalid")
        normalized = (values - self.feature_center) / self.feature_scale
        local_logits = self.local_network(normalized).reshape(-1).clamp(
            min=-self.max_abs_local_logit, max=self.max_abs_local_logit
        )
        local_weights = torch.softmax(local_logits, dim=0)
        pooled = torch.cat(
            [
                normalized.mean(dim=0),
                normalized.std(dim=0, unbiased=False),
                normalized.amax(dim=0),
            ],
            dim=0,
        )
        uniform_fraction = torch.sigmoid(self.query_network(pooled.unsqueeze(0)).reshape(1))
        uniform_mass = self.minimum_uniform_mass + (
            self.maximum_uniform_mass - self.minimum_uniform_mass
        ) * uniform_fraction
        uniform = torch.full_like(local_weights, 1.0 / float(len(local_weights)))
        weights = uniform_mass * uniform + (1.0 - uniform_mass) * local_weights
        return TargetFreePoseEvidenceWeightPrediction(
            weights=weights,
            local_weights=local_weights,
            uniform_mass=uniform_mass,
            local_logits=local_logits,
        )


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    """Return the standard inverse-squared-mass effective token count."""

    values = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
    if (
        len(values) == 0
        or not torch.isfinite(values).all()
        or torch.any(values < 0.0)
        or not bool(values.sum() > 0.0)
    ):
        raise ValueError("effective sample size weights are invalid")
    normalized = values / values.sum().clamp_min(torch.finfo(values.dtype).tiny)
    return 1.0 / normalized.square().sum().clamp_min(torch.finfo(values.dtype).tiny)


def relative_spatial_coverage_divergence(
    *,
    weights: torch.Tensor,
    xy: torch.Tensor,
    image_size: tuple[int, int],
    grid_rows: int = 4,
    grid_columns: int = 4,
) -> torch.Tensor:
    """Penalize collapse relative to the query's own detector distribution."""

    values = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
    coordinates = torch.as_tensor(xy, dtype=torch.float32, device=values.device)
    width, height = int(image_size[0]), int(image_size[1])
    if (
        len(values) == 0
        or coordinates.shape != (len(values), 2)
        or width <= 1
        or height <= 1
        or int(grid_rows) <= 0
        or int(grid_columns) <= 0
        or not torch.isfinite(values).all()
        or not torch.isfinite(coordinates).all()
        or torch.any(values < 0.0)
        or not bool(values.sum() > 0.0)
    ):
        raise ValueError("relative spatial coverage inputs are invalid")
    normalized = values / values.sum().clamp_min(torch.finfo(values.dtype).tiny)
    columns = torch.floor(coordinates[:, 0] / float(width) * int(grid_columns)).to(torch.long)
    rows = torch.floor(coordinates[:, 1] / float(height) * int(grid_rows)).to(torch.long)
    cells = rows.clamp(0, int(grid_rows) - 1) * int(grid_columns) + columns.clamp(
        0, int(grid_columns) - 1
    )
    cell_count = int(grid_rows) * int(grid_columns)
    selected_mass = torch.zeros((cell_count,), dtype=normalized.dtype, device=normalized.device)
    selected_mass.scatter_add_(0, cells, normalized)
    baseline_count = torch.zeros_like(selected_mass)
    baseline_count.scatter_add_(0, cells, torch.ones_like(normalized))
    occupied = baseline_count > 0.0
    baseline_mass = baseline_count[occupied] / baseline_count[occupied].sum().clamp_min(1.0)
    active_mass = selected_mass[occupied]
    return torch.sum(
        active_mass * torch.log(active_mass.clamp_min(torch.finfo(values.dtype).tiny) / baseline_mass)
    )
