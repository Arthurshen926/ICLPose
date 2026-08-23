"""Training contracts for candidate-conditioned sparse pose transport.

The objective trains a *relative basin energy*, never an absolute-pose head.
It combines a continuous pose-error listwise target, ordered pair margins,
explicit monotonic paths, known optimizer-drift negatives and optional
transport attribution KL.  Map pose-code observations can be subtracted with
the leave-one-view-out sufficient-statistics helper to prevent self-copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F


TRAINING_SEMANTICS = "sparse_pose_transport_energy_landscape_training_v1"
LEAVE_ONE_VIEW_OUT_SEMANTICS = "pose_field_sufficient_statistics_leave_one_view_out_v1"


@dataclass(frozen=True)
class PoseTransportTrainingConfig:
    stage: str
    soft_target_temperature: float = 0.25
    score_temperature: float = 0.25
    pairwise_margin: float = 0.10
    monotonic_margin: float = 0.05
    drift_margin: float = 0.10
    attribution_weight: float = 1.0
    listwise_weight: float = 1.0
    pairwise_weight: float = 0.5
    monotonic_weight: float = 0.5
    drift_weight: float = 1.0


@dataclass(frozen=True)
class LeaveOneViewOutPoseField:
    mean: np.ndarray
    weight: np.ndarray
    valid: np.ndarray
    semantics: str = LEAVE_ONE_VIEW_OUT_SEMANTICS


_STAGE_SCALE = {
    "coarse": (2.0, 45.0),
    "medium": (1.0, 15.0),
    "fine": (0.5, 5.0),
}


def normalized_joint_pose_error(
    translation_m: torch.Tensor,
    rotation_deg: torch.Tensor,
    *,
    stage: str,
) -> torch.Tensor:
    """Return the stage-normalized joint SE(3) error ``max(t/r_t,r/r_R)``."""

    if str(stage) not in _STAGE_SCALE:
        raise ValueError("pose transport stage must be coarse, medium, or fine")
    translation = torch.as_tensor(translation_m)
    rotation = torch.as_tensor(rotation_deg, device=translation.device, dtype=translation.dtype)
    if translation.shape != rotation.shape or translation.ndim != 2:
        raise ValueError("pose errors must both have shape [batch,candidate]")
    if not torch.isfinite(translation).all() or not torch.isfinite(rotation).all():
        raise ValueError("pose errors must be finite")
    if torch.any(translation < 0.0) or torch.any(rotation < 0.0):
        raise ValueError("pose errors must be nonnegative")
    translation_radius, rotation_radius = _STAGE_SCALE[str(stage)]
    return torch.maximum(
        translation / float(translation_radius), rotation / float(rotation_radius)
    )


def leave_one_view_out_pose_field(
    total_feature_sum: np.ndarray,
    total_weight: np.ndarray,
    held_feature_sum: np.ndarray,
    held_weight: np.ndarray,
    *,
    minimum_weight: float = 1.0e-6,
) -> LeaveOneViewOutPoseField:
    """Subtract one query view from pose-field sufficient statistics.

    The function consumes only additive sums and weights.  It therefore does
    not require storing all view features and makes the no-self-copy rule
    numerically testable.  Rows with no residual evidence are invalid rather
    than silently falling back to the held observation.
    """

    total_sum = np.asarray(total_feature_sum, dtype=np.float64)
    held_sum = np.asarray(held_feature_sum, dtype=np.float64)
    total_mass = np.asarray(total_weight, dtype=np.float64)
    held_mass = np.asarray(held_weight, dtype=np.float64)
    if total_sum.ndim < 2 or held_sum.shape != total_sum.shape:
        raise ValueError("feature sufficient statistics differ")
    expected_weight_shape = total_sum.shape[:-1]
    if total_mass.shape != expected_weight_shape or held_mass.shape != expected_weight_shape:
        raise ValueError("weight sufficient statistics differ")
    if any(np.any(~np.isfinite(value)) for value in (total_sum, held_sum, total_mass, held_mass)):
        raise ValueError("leave-one-view-out statistics must be finite")
    tolerance = 2.0e-6
    if (
        np.any(total_mass < 0.0) or np.any(held_mass < 0.0)
        or np.any(held_mass > total_mass + tolerance)
    ):
        raise ValueError("held view is not a subset of total pose-field statistics")
    residual_weight = total_mass - held_mass
    residual_sum = total_sum - held_sum
    # A negative residual beyond floating accumulation tolerance indicates
    # lineage mismatch. Feature sums themselves may legitimately be signed.
    if np.any(residual_weight < -tolerance):
        raise ValueError("negative residual pose-field weight")
    residual_weight = np.maximum(residual_weight, 0.0)
    threshold = float(minimum_weight)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_weight must be positive")
    valid = residual_weight >= threshold
    mean = np.zeros_like(residual_sum, dtype=np.float64)
    np.divide(
        residual_sum,
        np.maximum(residual_weight[..., None], threshold),
        out=mean,
        where=valid[..., None],
    )
    return LeaveOneViewOutPoseField(
        mean=mean.astype(np.float32),
        weight=residual_weight.astype(np.float32),
        valid=valid,
    )


def _validate_config(config: PoseTransportTrainingConfig) -> None:
    if str(config.stage) not in _STAGE_SCALE:
        raise ValueError("pose transport stage must be coarse, medium, or fine")
    positive = (
        config.soft_target_temperature,
        config.score_temperature,
        config.pairwise_margin,
        config.monotonic_margin,
        config.drift_margin,
    )
    weights = (
        config.attribution_weight,
        config.listwise_weight,
        config.pairwise_weight,
        config.monotonic_weight,
        config.drift_weight,
    )
    if any(not np.isfinite(value) or float(value) <= 0.0 for value in positive):
        raise ValueError("training temperatures and margins must be positive")
    if any(not np.isfinite(value) or float(value) < 0.0 for value in weights):
        raise ValueError("training loss weights must be nonnegative")


def _indexed_margin_loss(
    score: torch.Tensor,
    rows: torch.Tensor | None,
    *,
    candidate_valid: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if rows is None:
        return score.sum() * 0.0
    index = torch.as_tensor(rows, device=score.device, dtype=torch.long)
    if index.ndim != 2 or index.shape[1] != 3:
        raise ValueError("ordered pair rows must have shape [pair,3]")
    if index.numel() == 0:
        return score.sum() * 0.0
    batch, first, second = index.unbind(dim=1)
    if (
        torch.any(batch < 0) or torch.any(batch >= score.shape[0])
        or torch.any(first < 0) or torch.any(first >= score.shape[1])
        or torch.any(second < 0) or torch.any(second >= score.shape[1])
    ):
        raise ValueError("ordered pair index is out of range")
    if not torch.all(candidate_valid[batch, first] & candidate_valid[batch, second]):
        raise ValueError("ordered pairs must reference valid candidates")
    # Standard rows mean first=better, second=worse. Drift rows mean
    # first=earlier/better-error, second=later/worse-error and use the same
    # desired ordering even if the current objective ranked them oppositely.
    difference = score[batch, first] - score[batch, second]
    return torch.mean(F.relu(float(margin) - difference))


def pose_transport_energy_landscape_loss(
    candidate_score: torch.Tensor,
    translation_error_m: torch.Tensor,
    rotation_error_deg: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    config: PoseTransportTrainingConfig,
    attribution_probability: torch.Tensor | None = None,
    attribution_target: torch.Tensor | None = None,
    monotonic_pairs: torch.Tensor | None = None,
    drift_negative_pairs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Mapping[str, float]]:
    """Train score/error consistency for a frozen same-query candidate set.

    ``monotonic_pairs`` and ``drift_negative_pairs`` contain rows
    ``[batch, better_error_candidate, worse_error_candidate]``.  Drift rows
    are mined from prior optimizer trajectories and remain fixed labels; they
    are not selected from the score being optimized in this call.
    """

    _validate_config(config)
    score = torch.as_tensor(candidate_score)
    valid = torch.as_tensor(candidate_valid, device=score.device, dtype=torch.bool)
    if score.ndim != 2 or valid.shape != score.shape or not torch.isfinite(score).all():
        raise ValueError("candidate score/validity arrays differ")
    if not torch.all(torch.any(valid, dim=1)):
        raise ValueError("every query requires at least one valid candidate")
    error = normalized_joint_pose_error(
        translation_error_m, rotation_error_deg, stage=str(config.stage)
    ).to(device=score.device, dtype=score.dtype)
    if error.shape != score.shape:
        raise ValueError("pose errors differ from candidate scores")
    masked_error = error.masked_fill(~valid, torch.inf)
    target_logits = -masked_error / float(config.soft_target_temperature)
    target = torch.softmax(target_logits, dim=1).detach()
    score_logits = (score / float(config.score_temperature)).masked_fill(~valid, -torch.inf)
    log_probability = torch.log_softmax(score_logits, dim=1)
    listwise = -torch.sum(target * log_probability.masked_fill(~valid, 0.0), dim=1).mean()

    pair_losses = []
    for batch in range(score.shape[0]):
        indices = torch.nonzero(valid[batch], as_tuple=False).reshape(-1)
        for left_position in range(indices.numel()):
            for right_position in range(left_position + 1, indices.numel()):
                left = indices[left_position]
                right = indices[right_position]
                delta = error[batch, right] - error[batch, left]
                if torch.abs(delta) <= 1.0e-8:
                    continue
                better, worse = (left, right) if delta > 0 else (right, left)
                adaptive_margin = float(config.pairwise_margin) * torch.clamp(
                    torch.abs(delta), min=0.25, max=2.0
                )
                pair_losses.append(F.relu(adaptive_margin - (score[batch, better] - score[batch, worse])))
    pairwise = torch.stack(pair_losses).mean() if pair_losses else score.sum() * 0.0
    monotonic = _indexed_margin_loss(
        score, monotonic_pairs, candidate_valid=valid,
        margin=float(config.monotonic_margin),
    )
    drift = _indexed_margin_loss(
        score, drift_negative_pairs, candidate_valid=valid,
        margin=float(config.drift_margin),
    )

    attribution = score.sum() * 0.0
    if (attribution_probability is None) != (attribution_target is None):
        raise ValueError("attribution probability and target must be provided together")
    if attribution_probability is not None:
        prediction = torch.as_tensor(
            attribution_probability, device=score.device, dtype=score.dtype
        )
        teacher = torch.as_tensor(attribution_target, device=score.device, dtype=score.dtype)
        if prediction.shape != teacher.shape or prediction.ndim != 2:
            raise ValueError("attribution distributions must share shape [source,state]")
        if (
            not torch.isfinite(prediction).all() or not torch.isfinite(teacher).all()
            or torch.any(prediction < 0.0) or torch.any(teacher < 0.0)
        ):
            raise ValueError("attribution distributions are invalid")
        if not torch.allclose(prediction.sum(dim=1), torch.ones_like(prediction[:, 0]), atol=2e-5):
            raise ValueError("attribution prediction must include sink and sum to one")
        if not torch.allclose(teacher.sum(dim=1), torch.ones_like(teacher[:, 0]), atol=2e-5):
            raise ValueError("attribution target must include sink and sum to one")
        attribution = torch.mean(
            torch.sum(
                teacher * (
                    torch.log(torch.clamp(teacher, min=1.0e-8))
                    - torch.log(torch.clamp(prediction, min=1.0e-8))
                ),
                dim=1,
            )
        )

    total = (
        float(config.attribution_weight) * attribution
        + float(config.listwise_weight) * listwise
        + float(config.pairwise_weight) * pairwise
        + float(config.monotonic_weight) * monotonic
        + float(config.drift_weight) * drift
    )
    return total, {
        "training_semantics": TRAINING_SEMANTICS,
        "attribution_kl": float(attribution.detach().cpu()),
        "listwise_soft_target_nll": float(listwise.detach().cpu()),
        "pairwise_margin": float(pairwise.detach().cpu()),
        "monotonic_path": float(monotonic.detach().cpu()),
        "drift_negative": float(drift.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
