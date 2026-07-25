"""Constrained group-aware linear probes for frozen candidate-edge features.

The probe is deliberately diagnostic-only.  It operates on target-free visual
edge features *after* a training caller has joined coherent-wrong group
membership.  It contains one linear weight vector, no candidate rank, coarse
score, pose, residual, or landmark identifier.  Its objective matches the
actual hard-pose failure mode: every point must defeat its strongest coherent
wrong candidate and the resulting margins must agree across one pose group.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


CANDIDATE_EDGE_GROUP_LINEAR_PROBE_FORMAT = "candidate_edge_group_softmin_linear_probe_v1"


@dataclass(frozen=True)
class CandidateEdgeGroupLinearBatch:
    """Flattened target-side grouping over frozen pair-difference features.

    ``edge_features`` are already oriented as ``correct - coherent_wrong``.
    ``edge_to_point`` assigns multiple wrong candidates for the same source
    point to one soft-min reduction, and ``point_to_pose`` combines those
    point margins into a coherent pose group.
    """

    edge_features: torch.Tensor
    edge_to_point: torch.Tensor
    point_to_pose: torch.Tensor

    def __post_init__(self) -> None:
        features = torch.as_tensor(self.edge_features, dtype=torch.float32)
        edge_to_point = torch.as_tensor(self.edge_to_point, dtype=torch.long).reshape(-1)
        point_to_pose = torch.as_tensor(self.point_to_pose, dtype=torch.long).reshape(-1)
        if (
            features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] == 0
            or edge_to_point.shape != (len(features),)
            or len(point_to_pose) == 0
            or torch.any(edge_to_point < 0)
            or torch.any(edge_to_point >= len(point_to_pose))
            or torch.any(point_to_pose < 0)
            or not torch.isfinite(features).all()
        ):
            raise ValueError("candidate-edge group linear batch is invalid")
        point_edge_count = torch.bincount(edge_to_point, minlength=len(point_to_pose))
        pose_count = int(torch.max(point_to_pose).item()) + 1
        pose_point_count = torch.bincount(point_to_pose, minlength=pose_count)
        if torch.any(point_edge_count <= 0) or torch.any(pose_point_count <= 0):
            raise ValueError("candidate-edge group linear batch has an empty reduction group")
        object.__setattr__(self, "edge_features", features)
        object.__setattr__(self, "edge_to_point", edge_to_point)
        object.__setattr__(self, "point_to_pose", point_to_pose)

    @property
    def edge_count(self) -> int:
        return int(len(self.edge_features))

    @property
    def point_count(self) -> int:
        return int(len(self.point_to_pose))

    @property
    def pose_count(self) -> int:
        return int(torch.max(self.point_to_pose).item()) + 1

    @property
    def feature_dimension(self) -> int:
        return int(self.edge_features.shape[1])

    def to(self, device: torch.device | str) -> "CandidateEdgeGroupLinearBatch":
        return CandidateEdgeGroupLinearBatch(
            edge_features=self.edge_features.to(device),
            edge_to_point=self.edge_to_point.to(device),
            point_to_pose=self.point_to_pose.to(device),
        )


@dataclass(frozen=True)
class CandidateEdgeGroupLinearProbe:
    """A target-free visual linear score with train-fold-only normalization."""

    feature_scale: torch.Tensor
    weights: torch.Tensor
    softmin_temperature: float

    def __post_init__(self) -> None:
        scale = torch.as_tensor(self.feature_scale, dtype=torch.float32).reshape(-1)
        weights = torch.as_tensor(self.weights, dtype=torch.float32).reshape(-1)
        temperature = float(self.softmin_temperature)
        if (
            len(scale) == 0
            or weights.shape != scale.shape
            or torch.any(scale <= 0.0)
            or not torch.isfinite(scale).all()
            or not torch.isfinite(weights).all()
            or not math.isfinite(temperature)
            or temperature <= 0.0
        ):
            raise ValueError("candidate-edge group linear probe is invalid")
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "softmin_temperature", temperature)

    def to(self, device: torch.device | str) -> "CandidateEdgeGroupLinearProbe":
        return CandidateEdgeGroupLinearProbe(
            feature_scale=self.feature_scale.to(device),
            weights=self.weights.to(device),
            softmin_temperature=float(self.softmin_temperature),
        )

    def score(self, edge_features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(edge_features, dtype=torch.float32, device=self.weights.device)
        if (
            features.ndim != 2
            or features.shape[1] != len(self.weights)
            or not torch.isfinite(features).all()
        ):
            raise ValueError("candidate-edge group linear score inputs are invalid")
        return (features / self.feature_scale) @ self.weights


def _segment_logsumexp(values: torch.Tensor, groups: torch.Tensor, group_count: int) -> torch.Tensor:
    """Stable logsumexp over a dense contiguous segment ID vector."""

    scores = torch.as_tensor(values, dtype=torch.float32)
    ids = torch.as_tensor(groups, dtype=torch.long, device=scores.device).reshape(-1)
    count = int(group_count)
    if (
        scores.ndim != 1
        or ids.shape != scores.shape
        or count <= 0
        or torch.any(ids < 0)
        or torch.any(ids >= count)
        or not torch.isfinite(scores).all()
    ):
        raise ValueError("candidate-edge group logsumexp inputs are invalid")
    maximum = torch.full((count,), -torch.inf, dtype=scores.dtype, device=scores.device)
    maximum.scatter_reduce_(0, ids, scores, reduce="amax", include_self=True)
    if torch.any(~torch.isfinite(maximum)):
        raise RuntimeError("candidate-edge group logsumexp has an empty segment")
    sumexp = torch.zeros((count,), dtype=scores.dtype, device=scores.device)
    sumexp.scatter_add_(0, ids, torch.exp(scores - maximum.index_select(0, ids)))
    return maximum + torch.log(sumexp.clamp_min(torch.finfo(scores.dtype).tiny))


def group_softmin_pose_gaps(
    *,
    edge_margins: torch.Tensor,
    batch: CandidateEdgeGroupLinearBatch,
    temperature: float,
) -> torch.Tensor:
    """Return one differentiable strongest-wrong margin per coherent pose.

    The point soft-min is normalized by its number of wrong-candidate edges,
    so duplicate equivalent negatives do not add a constant reward/penalty.
    As ``temperature`` approaches zero it converges to the exact minimum edge
    margin used by the runtime coherent-wrong audit.
    """

    margins = torch.as_tensor(edge_margins, dtype=torch.float32, device=batch.edge_features.device)
    tau = float(temperature)
    if (
        margins.shape != (batch.edge_count,)
        or not torch.isfinite(margins).all()
        or not math.isfinite(tau)
        or tau <= 0.0
    ):
        raise ValueError("candidate-edge softmin pose inputs are invalid")
    point_count = batch.point_count
    logsum = _segment_logsumexp(-margins / tau, batch.edge_to_point, point_count)
    edge_counts = torch.bincount(batch.edge_to_point, minlength=point_count).to(dtype=margins.dtype)
    point_margin = -tau * (logsum - torch.log(edge_counts.clamp_min(1.0)))
    pose_count = batch.pose_count
    pose_sum = torch.zeros((pose_count,), dtype=margins.dtype, device=margins.device)
    pose_sum.scatter_add_(0, batch.point_to_pose, point_margin)
    point_counts = torch.bincount(batch.point_to_pose, minlength=pose_count).to(dtype=margins.dtype)
    return pose_sum / point_counts.clamp_min(1.0)


def fit_candidate_edge_group_linear_probe(
    *,
    batch: CandidateEdgeGroupLinearBatch,
    device: torch.device | str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    softmin_temperature: float,
    margin: float = 0.0,
    seed: int = 20260722,
) -> tuple[CandidateEdgeGroupLinearProbe, dict[str, float]]:
    """Fit one L2-regularized linear group-softmin margin probe.

    All inputs are frozen visual edge features.  The only train-only values in
    this routine are the group reductions supplied by ``batch`` after visual
    extraction.  The fixed number of steps avoids held-fold early stopping or
    any validation-derived hyperparameter selection.
    """

    count = int(epochs)
    lr = float(learning_rate)
    decay = float(weight_decay)
    tau = float(softmin_temperature)
    target_margin = float(margin)
    if (
        count <= 0
        or not all(math.isfinite(value) for value in (lr, decay, tau, target_margin))
        or lr <= 0.0
        or decay < 0.0
        or tau <= 0.0
    ):
        raise ValueError("candidate-edge group probe optimization configuration is invalid")
    active = batch.to(device)
    features = active.edge_features
    scale = torch.sqrt(torch.mean(features.square(), dim=0).clamp_min(1e-8))
    normalized = features / scale
    # A diagonal train-fold direction gives the one-layer model a stable
    # starting point without adding a learned hidden representation.
    initial = normalized.mean(dim=0)
    if float(torch.linalg.vector_norm(initial).item()) <= 1e-8:
        initial = torch.ones_like(initial)
    initial = initial / torch.linalg.vector_norm(initial).clamp_min(1e-8)
    generator = torch.Generator(device=normalized.device)
    generator.manual_seed(int(seed))
    # Deterministic zero-mean jitter only breaks exact feature symmetries.
    weights = torch.nn.Parameter(initial + 1e-5 * torch.randn(
        initial.shape, generator=generator, device=initial.device, dtype=initial.dtype
    ))
    optimizer = torch.optim.AdamW([weights], lr=lr, weight_decay=decay)
    loss_value = 0.0
    gaps = torch.empty((0,), dtype=torch.float32, device=normalized.device)
    for _ in range(count):
        optimizer.zero_grad(set_to_none=True)
        edge_margins = normalized @ weights
        gaps = group_softmin_pose_gaps(
            edge_margins=edge_margins, batch=active, temperature=tau
        )
        loss = F.softplus(torch.as_tensor(target_margin, device=gaps.device) - gaps).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("candidate-edge group probe loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([weights], max_norm=10.0)
        optimizer.step()
        loss_value = float(loss.detach().item())
    with torch.no_grad():
        edge_margins = normalized @ weights
        gaps = group_softmin_pose_gaps(edge_margins=edge_margins, batch=active, temperature=tau)
        final_loss = F.softplus(
            torch.as_tensor(target_margin, device=gaps.device) - gaps
        ).mean()
    probe = CandidateEdgeGroupLinearProbe(
        feature_scale=scale.detach().cpu(),
        weights=weights.detach().cpu(),
        softmin_temperature=tau,
    )
    return probe, {
        "epoch_count": float(count),
        "last_step_loss": float(loss_value),
        "final_group_softmin_loss": float(final_loss.item()),
        "final_group_softmin_mean_gap": float(gaps.mean().item()),
        "final_group_softmin_win_fraction": float((gaps > 0.0).float().mean().item()),
        "weight_l2": float(torch.linalg.vector_norm(weights.detach()).item()),
        "feature_scale_min": float(scale.min().item()),
        "feature_scale_max": float(scale.max().item()),
    }
