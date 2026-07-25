"""Candidate-specific visual pose log-likelihood-ratio primitives.

This module is intentionally separate from candidate identity rerankers.  A
network may only emit per-candidate, per-support-view visual edge LLRs.  The
fixed candidate prior, explicit null mass, and fixed support-view weights are
then marginalized outside the network.  No pose matrix, residual, target,
track ID, or coarse score is an encoder input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

CANDIDATE_POSE_LLR_FORMAT = "candidate_specific_pose_llr_v3"
CANDIDATE_POSE_LLR_SCORE_FORMAT = "candidate_specific_pose_llr_scores_v3"
POSE_LLR_SCALE_WINDOWS = {
    "radio_final": 9,
    "radio_intermediate": 13,
    "alike": 13,
}


def _crop_subpixel_grid_tokens(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a masked crop at a continuous image-space anchor.

    Candidate poses commonly differ by fewer pixels than one RADIO/ALIKE grid
    cell.  Integer anchor lookup would give those poses identical visual
    inputs, so this LLR-only path uses bilinear samples on the frozen feature
    grid.  Out-of-grid crop tokens remain explicitly masked rather than being
    treated as zero-valued visual evidence.
    """

    if (
        image_grids.ndim != 4
        or image_grids.shape[1] != image_grids.shape[2]
        or image_sizes.shape != (image_grids.shape[0], 2)
        or int(window_size) <= 0
        or int(window_size) % 2 != 1
        or int(window_size) > int(image_grids.shape[1])
    ):
        raise ValueError("subpixel crop inputs are invalid")
    indices = torch.as_tensor(
        image_indices, dtype=torch.long, device=image_grids.device
    ).reshape(-1)
    coordinates = torch.as_tensor(xy, dtype=torch.float32, device=image_grids.device)
    if (
        coordinates.shape != (len(indices), 2)
        or torch.any(indices < 0)
        or torch.any(indices >= len(image_grids))
        or not torch.isfinite(coordinates).all()
    ):
        raise ValueError("subpixel crop coordinates are invalid")
    grid_size = int(image_grids.shape[1])
    selected_sizes = image_sizes.index_select(0, indices).to(dtype=torch.float32)
    # ``align_corners=True`` maps real image endpoints to real descriptor-grid
    # endpoints, so an in-image border projection retains its observed tokens.
    extent = float(grid_size - 1)
    center_columns = coordinates[:, 0] / (selected_sizes[:, 0] - 1.0).clamp_min(1.0)
    center_rows = coordinates[:, 1] / (selected_sizes[:, 1] - 1.0).clamp_min(1.0)
    center_columns = center_columns * extent
    center_rows = center_rows * extent
    radius = int(window_size) // 2
    offsets = torch.arange(
        -radius, radius + 1, dtype=torch.float32, device=image_grids.device
    )
    rows = center_rows[:, None] + offsets[None, :]
    columns = center_columns[:, None] + offsets[None, :]
    valid = (
        (rows[:, :, None] >= 0.0)
        & (rows[:, :, None] <= extent)
        & (columns[:, None, :] >= 0.0)
        & (columns[:, None, :] <= extent)
    )
    if grid_size == 1:
        grid_x = torch.zeros_like(columns)
        grid_y = torch.zeros_like(rows)
    else:
        grid_x = 2.0 * columns / extent - 1.0
        grid_y = 2.0 * rows / extent - 1.0
    sampling_grid = torch.stack(
        [
            grid_x[:, None, :].expand(-1, int(window_size), -1),
            grid_y[:, :, None].expand(-1, -1, int(window_size)),
        ],
        dim=-1,
    ).to(dtype=image_grids.dtype)
    selected = image_grids.index_select(0, indices).permute(0, 3, 1, 2)
    crop = F.grid_sample(
        selected,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    crop = crop.permute(0, 2, 3, 1).reshape(
        len(indices), int(window_size) ** 2, image_grids.shape[-1]
    )
    return crop, valid.reshape(len(indices), int(window_size) ** 2)


def grouped_hypothesis_semantic_manifest(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Extract the immutable generation semantics of a target-free pose bank.

    Row count and query shard membership deliberately do not appear here: a
    train-only source and a held-out scorer need different query rows.  Every
    candidate/proposal input and every grouped-PnP control remains part of the
    manifest, because changing either changes the hard-pose distribution that
    a pose LLR is asked to rank.
    """

    if (
        not isinstance(metadata, Mapping)
        or metadata.get("format") != "grouped_pose_hypotheses_inference_only_v1"
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_generation") is not False
        or not isinstance(metadata.get("candidate_pose_evidence_version"), str)
        or not str(metadata.get("candidate_pose_evidence_version", "")).strip()
        or not isinstance(metadata.get("inputs"), Mapping)
        or not isinstance(metadata.get("grouped_config"), Mapping)
    ):
        raise ValueError("grouped hypothesis metadata cannot form a semantic lineage")
    payload = {
        "format": str(metadata["format"]),
        "candidate_pose_evidence_version": str(
            metadata["candidate_pose_evidence_version"]
        ),
        "inputs": metadata["inputs"],
        "grouped_config": metadata["grouped_config"],
    }
    try:
        canonical_json = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        canonical_payload = json.loads(canonical_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("grouped hypothesis semantic lineage is not JSON canonical") from exc
    return {
        "semantic_manifest": canonical_payload,
        "semantic_hash": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    }


def validate_grouped_hypothesis_semantic_match(
    *, expected: Mapping[str, object], observed: Mapping[str, object]
) -> None:
    """Reject a train/held-out hypothesis distribution mismatch before scoring."""

    expected_manifest = expected.get("semantic_manifest") if isinstance(expected, Mapping) else None
    observed_manifest = observed.get("semantic_manifest") if isinstance(observed, Mapping) else None
    expected_hash = expected.get("semantic_hash") if isinstance(expected, Mapping) else None
    observed_hash = observed.get("semantic_hash") if isinstance(observed, Mapping) else None
    if (
        not isinstance(expected_manifest, Mapping)
        or not isinstance(observed_manifest, Mapping)
        or not isinstance(expected_hash, str)
        or not isinstance(observed_hash, str)
    ):
        raise ValueError("grouped hypothesis semantic lineage is malformed")
    if expected_hash != observed_hash or expected_manifest != observed_manifest:
        raise ValueError("grouped hypothesis semantic lineage differs")


def validate_serialized_grouped_hypothesis_semantic_lineage(
    lineage: Mapping[str, object],
) -> dict[str, object]:
    """Validate a lineage previously emitted by the target-free builder.

    A serialized lineage intentionally retains only the canonical semantic
    manifest.  The two target-free source-metadata guards are therefore
    reconstructed here solely to reuse the canonicalizer, then the resulting
    manifest and hash must exactly match the serialized values.
    """

    manifest = lineage.get("semantic_manifest") if isinstance(lineage, Mapping) else None
    if not isinstance(manifest, Mapping):
        raise ValueError("grouped hypothesis semantic lineage is malformed")
    source_metadata = {
        **dict(manifest),
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
    }
    observed = grouped_hypothesis_semantic_manifest(source_metadata)
    validate_grouped_hypothesis_semantic_match(expected=lineage, observed=observed)
    return observed


def bounded_log_likelihood_ratio(
    raw_log_likelihood_ratio: torch.Tensor,
    *,
    max_abs_log_ratio: float,
) -> torch.Tensor:
    """Bound an edge LLR without changing the neutral zero point."""

    cap = float(max_abs_log_ratio)
    if not torch.isfinite(torch.as_tensor(cap)) or cap <= 0.0:
        raise ValueError("max_abs_log_ratio must be finite and positive")
    values = torch.as_tensor(raw_log_likelihood_ratio)
    if not torch.isfinite(values).all():
        raise ValueError("raw log-likelihood ratios must be finite")
    return cap * torch.tanh(values / cap)


def fixed_candidate_view_mixture_log_ratio(
    *,
    edge_log_likelihood_ratios: torch.Tensor,
    edge_usable: torch.Tensor,
    candidate_view_weights: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    null_probabilities: torch.Tensor,
    missing_edge_log_likelihood_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize bounded visual edges under an immutable top-L posterior.

    ``edge_usable=False`` always receives the same fixed missing factor.  It
    therefore cannot make an out-of-bounds projection look better than one
    with actual visual evidence.  The explicit null remains neutral.
    """

    # The posterior is a probability contract, not a mixed-precision feature
    # tensor.  In particular, a 20-way prior can lose enough FP16 mass to
    # invalidate an otherwise exact explicit-null denominator.  Keep the
    # whole log-domain mixture in FP32 while preserving gradients to edges.
    edges = torch.as_tensor(edge_log_likelihood_ratios, dtype=torch.float32)
    usable = torch.as_tensor(edge_usable, dtype=torch.bool, device=edges.device)
    weights = torch.as_tensor(candidate_view_weights, dtype=torch.float32, device=edges.device)
    candidates = torch.as_tensor(candidate_probabilities, dtype=torch.float32, device=edges.device)
    null = torch.as_tensor(null_probabilities, dtype=torch.float32, device=edges.device)
    missing = float(missing_edge_log_likelihood_ratio)
    if not torch.isfinite(torch.as_tensor(missing)):
        raise ValueError("missing edge log-likelihood ratio must be finite")
    if edges.ndim != 4:
        raise ValueError("edge log-likelihood ratios must have shape [batch, point, candidate, view]")
    batch, point_count, candidate_count, view_count = edges.shape
    if (
        usable.shape != edges.shape
        or weights.shape != (point_count, candidate_count, view_count)
        or candidates.shape != (point_count, candidate_count)
        or null.shape != (point_count,)
        or not torch.isfinite(edges).all()
        or not torch.isfinite(weights).all()
        or not torch.isfinite(candidates).all()
        or not torch.isfinite(null).all()
        or torch.any(weights < 0.0)
        or torch.any(candidates < 0.0)
        or torch.any(null < 0.0)
    ):
        raise ValueError("fixed candidate/view mixture inputs are invalid")
    candidate_mass = candidates.sum(dim=1) + null
    if torch.any(torch.abs(candidate_mass - 1.0) > 1e-4):
        raise ValueError("candidate and null probabilities must sum to one")
    positive_candidates = candidates > 0.0
    view_mass = weights.sum(dim=2)
    if torch.any(torch.abs(view_mass[positive_candidates] - 1.0) > 1e-4) or torch.any(
        view_mass[~positive_candidates] > 1e-6
    ):
        raise ValueError("fixed support-view weights must preserve candidate mass")

    effective_edges = torch.where(
        usable,
        edges,
        torch.full_like(edges, missing),
    )
    safe_log_weights = torch.where(
        weights > 0.0,
        torch.log(weights),
        torch.full_like(weights, -torch.inf),
    )
    candidate_log_ratio = torch.logsumexp(
        effective_edges + safe_log_weights.unsqueeze(0), dim=3
    )
    candidate_log_ratio = torch.where(
        positive_candidates.unsqueeze(0),
        candidate_log_ratio,
        torch.zeros_like(candidate_log_ratio),
    )
    log_terms = torch.cat(
        [
            torch.where(
                candidates > 0.0,
                torch.log(candidates) + candidate_log_ratio,
                torch.full_like(candidate_log_ratio, -torch.inf),
            ),
            torch.where(
                null > 0.0,
                torch.log(null),
                torch.full_like(null, -torch.inf),
            ).reshape(1, point_count, 1).expand(batch, -1, -1),
        ],
        dim=2,
    )
    point_log_ratio = torch.logsumexp(log_terms, dim=2)
    return point_log_ratio, candidate_log_ratio


def pairwise_pose_margin_loss(
    *,
    correct_scores: torch.Tensor,
    coherent_wrong_scores: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train a score gap, not a candidate identity class posterior."""

    correct = torch.as_tensor(correct_scores)
    wrong = torch.as_tensor(coherent_wrong_scores, device=correct.device, dtype=correct.dtype)
    value = float(margin)
    if (
        correct.ndim != 1
        or wrong.shape != correct.shape
        or len(correct) == 0
        or not torch.isfinite(correct).all()
        or not torch.isfinite(wrong).all()
        or not torch.isfinite(torch.as_tensor(value))
        or value < 0.0
    ):
        raise ValueError("pairwise pose margin inputs are invalid")
    gaps = correct - wrong
    loss = F.softplus(torch.as_tensor(value, dtype=correct.dtype, device=correct.device) - gaps).mean()
    metrics = {
        "pair_count": float(len(gaps)),
        "pairwise_correct_win_fraction": float((gaps > 0.0).to(dtype=torch.float32).mean().item()),
        "pairwise_mean_correct_minus_wrong": float(gaps.detach().mean().item()),
    }
    return loss, metrics


def query_grouped_pose_margin_loss(
    *,
    correct_scores: torch.Tensor,
    coherent_wrong_scores: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Margin loss against each query's strongest coherent wrong pose.

    A query can have several target-mined wrong modes.  Reducing them to the
    maximum score before computing the loss gives every query equal weight and
    directly trains the ranking failure that matters at inference.
    """

    correct = torch.as_tensor(correct_scores)
    wrong = torch.as_tensor(
        coherent_wrong_scores, device=correct.device, dtype=correct.dtype
    )
    value = float(margin)
    if (
        correct.ndim != 1
        or wrong.ndim != 2
        or wrong.shape[0] != len(correct)
        or wrong.shape[1] == 0
        or len(correct) == 0
        or not torch.isfinite(correct).all()
        or not torch.isfinite(wrong).all()
        or not torch.isfinite(torch.as_tensor(value))
        or value < 0.0
    ):
        raise ValueError("query-grouped pose margin inputs are invalid")
    hardest_wrong = wrong.max(dim=1).values
    gaps = correct - hardest_wrong
    loss = F.softplus(
        torch.as_tensor(value, dtype=correct.dtype, device=correct.device) - gaps
    ).mean()
    metrics = {
        "query_count": float(len(gaps)),
        "query_correct_win_fraction": float(
            (gaps > 0.0).to(dtype=torch.float32).mean().item()
        ),
        "query_mean_correct_minus_hardest_wrong": float(gaps.detach().mean().item()),
    }
    return loss, metrics


def query_grouped_pose_soft_hard_margin_loss(
    *,
    correct_scores: torch.Tensor,
    coherent_wrong_scores: torch.Tensor,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Margin against a normalized soft maximum over every wrong pose mode.

    ``max``-only supervision gives no gradient to all but one wrong mode.  A
    temperature-controlled log-mean-exp retains the score scale of the usual
    max-margin loss while assigning nonzero gradient to every finite mode.
    The normalization by mode count is essential: adding more frozen wrong
    hypotheses must not change the target score merely through cardinality.
    """

    correct = torch.as_tensor(correct_scores)
    wrong = torch.as_tensor(
        coherent_wrong_scores, device=correct.device, dtype=correct.dtype
    )
    value = float(margin)
    tau = float(temperature)
    if (
        correct.ndim != 1
        or wrong.ndim != 2
        or wrong.shape[0] != len(correct)
        or wrong.shape[1] == 0
        or len(correct) == 0
        or not torch.isfinite(correct).all()
        or not torch.isfinite(wrong).all()
        or not math.isfinite(value)
        or not math.isfinite(tau)
        or value < 0.0
        or tau <= 0.0
    ):
        raise ValueError("query-grouped soft-hard pose margin inputs are invalid")
    mode_count = int(wrong.shape[1])
    scaled = wrong / tau
    soft_weights = torch.softmax(scaled, dim=1)
    soft_hard_wrong = tau * (
        torch.logsumexp(scaled, dim=1) - math.log(float(mode_count))
    )
    gaps = correct - soft_hard_wrong
    loss = F.softplus(
        torch.as_tensor(value, dtype=correct.dtype, device=correct.device) - gaps
    ).mean()
    entropy = -torch.sum(
        soft_weights * torch.log(soft_weights.clamp_min(torch.finfo(soft_weights.dtype).tiny)),
        dim=1,
    )
    metrics = {
        "query_count": float(len(gaps)),
        "query_soft_hard_correct_win_fraction": float(
            (gaps > 0.0).to(dtype=torch.float32).mean().item()
        ),
        "query_mean_correct_minus_soft_hard_wrong": float(gaps.detach().mean().item()),
        "query_soft_hard_effective_wrong_mode_count": float(
            torch.exp(entropy.detach()).mean().item()
        ),
    }
    return loss, metrics


def query_grouped_pose_permutation_soft_hard_margin_loss(
    *,
    normal_correct_scores: torch.Tensor,
    normal_wrong_scores: torch.Tensor,
    permuted_correct_scores: torch.Tensor,
    permuted_wrong_scores: torch.Tensor,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require the real support pairing to improve a full-pool pose ranking.

    Both branches score the same frozen query points, candidate priors, and
    train-only correct/wrong pose projections.  The only changed input is the
    support-view appearance.  Unlike a hardest-wrong control, the normalized
    log-mean-exp aggregates every coherent wrong mode, so the loss cannot be
    satisfied by moving just one accidental winner.
    """

    target_margin = float(margin)
    tau = float(temperature)
    if (
        not math.isfinite(target_margin)
        or not math.isfinite(tau)
        or target_margin < 0.0
        or tau <= 0.0
    ):
        raise ValueError("query-grouped permutation soft-hard margin is invalid")

    def soft_hard_gap(
        correct_scores: torch.Tensor,
        wrong_scores: torch.Tensor,
        *,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        correct = torch.as_tensor(correct_scores)
        wrong = torch.as_tensor(wrong_scores, device=correct.device, dtype=correct.dtype)
        if (
            correct.ndim != 1
            or wrong.ndim != 2
            or wrong.shape[0] != len(correct)
            or wrong.shape[1] == 0
            or len(correct) == 0
            or not torch.isfinite(correct).all()
            or not torch.isfinite(wrong).all()
        ):
            raise ValueError(f"{name} query-grouped soft-hard pose scores are invalid")
        scaled = wrong / tau
        weights = torch.softmax(scaled, dim=1)
        soft_hard_wrong = tau * (
            torch.logsumexp(scaled, dim=1) - math.log(float(wrong.shape[1]))
        )
        entropy = -torch.sum(
            weights
            * torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny)),
            dim=1,
        )
        return correct - soft_hard_wrong, torch.exp(entropy)

    normal_gap, normal_effective_modes = soft_hard_gap(
        normal_correct_scores, normal_wrong_scores, name="normal"
    )
    permuted_gap, permuted_effective_modes = soft_hard_gap(
        permuted_correct_scores, permuted_wrong_scores, name="permuted"
    )
    if normal_gap.shape != permuted_gap.shape:
        raise ValueError("normal and permuted query groups must have the same shape")
    visual_delta = normal_gap - permuted_gap
    loss = F.softplus(
        torch.as_tensor(target_margin, dtype=normal_gap.dtype, device=normal_gap.device)
        - visual_delta
    ).mean()
    metrics = {
        "query_count": float(len(normal_gap)),
        "normal_mean_correct_minus_soft_hard_wrong": float(normal_gap.detach().mean().item()),
        "permuted_mean_correct_minus_soft_hard_wrong": float(
            permuted_gap.detach().mean().item()
        ),
        "normal_minus_permuted_soft_hard_gap": float(visual_delta.detach().mean().item()),
        "normal_soft_hard_effective_wrong_mode_count": float(
            normal_effective_modes.detach().mean().item()
        ),
        "permuted_soft_hard_effective_wrong_mode_count": float(
            permuted_effective_modes.detach().mean().item()
        ),
        "permutation_soft_hard_margin_loss": float(loss.detach().item()),
    }
    return loss, metrics


def validate_target_free_pose_llr_score_metadata(metadata: Mapping[str, object]) -> None:
    """Reject score artifacts that would blur diagnostic and PnP protocols."""

    required = {
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
    }
    if not isinstance(metadata, Mapping) or any(
        metadata.get(key) is not value for key, value in required.items()
    ):
        raise ValueError("pose-LLR score metadata violates the target-free diagnostic contract")


@dataclass(frozen=True)
class CandidatePoseLLRRuntime:
    """Static target-free candidate/support layout for a query point set."""

    query_image_indices: torch.Tensor
    support_image_indices: torch.Tensor
    support_xy: torch.Tensor
    support_view_valid: torch.Tensor
    candidate_view_weights: torch.Tensor
    candidate_probabilities: torch.Tensor
    null_probabilities: torch.Tensor

    def __post_init__(self) -> None:
        query = torch.as_tensor(self.query_image_indices, dtype=torch.long).reshape(-1)
        support_indices = torch.as_tensor(self.support_image_indices, dtype=torch.long)
        support_xy = torch.as_tensor(self.support_xy, dtype=torch.float32)
        support_valid = torch.as_tensor(self.support_view_valid, dtype=torch.bool)
        view_weights = torch.as_tensor(self.candidate_view_weights, dtype=torch.float32)
        candidates = torch.as_tensor(self.candidate_probabilities, dtype=torch.float32)
        null = torch.as_tensor(self.null_probabilities, dtype=torch.float32).reshape(-1)
        if (
            len(query) == 0
            or support_indices.ndim != 3
            or support_xy.shape != (*support_indices.shape, 2)
            or support_valid.shape != support_indices.shape
            or view_weights.shape != support_indices.shape
            or candidates.shape != support_indices.shape[:2]
            or null.shape != (len(query),)
            or query.shape != (support_indices.shape[0],)
            or torch.any(query < 0)
            or torch.any(support_indices < 0)
            or not torch.isfinite(support_xy).all()
            or not torch.isfinite(view_weights).all()
            or not torch.isfinite(candidates).all()
            or not torch.isfinite(null).all()
            or torch.any(view_weights < 0.0)
            or torch.any(candidates < 0.0)
            or torch.any(null < 0.0)
            or torch.any(support_valid & (support_indices < 0))
        ):
            raise ValueError("candidate pose-LLR runtime arrays are invalid")
        if torch.any(torch.abs(candidates.sum(dim=1) + null - 1.0) > 1e-4):
            raise ValueError("candidate pose-LLR runtime priors must sum to one")
        positive = candidates > 0.0
        view_mass = view_weights.sum(dim=2)
        if (
            torch.any(torch.abs(view_mass[positive] - 1.0) > 1e-4)
            or torch.any(view_mass[~positive] > 1e-6)
            or torch.any(positive & ~torch.any(support_valid, dim=2))
        ):
            raise ValueError("candidate pose-LLR runtime view weights are invalid")
        object.__setattr__(self, "query_image_indices", query)
        object.__setattr__(self, "support_image_indices", support_indices)
        object.__setattr__(self, "support_xy", support_xy)
        object.__setattr__(self, "support_view_valid", support_valid)
        object.__setattr__(self, "candidate_view_weights", view_weights)
        object.__setattr__(self, "candidate_probabilities", candidates)
        object.__setattr__(self, "null_probabilities", null)

    def to(self, device: torch.device | str) -> "CandidatePoseLLRRuntime":
        """Move only fixed, target-free runtime tensors to a scoring device."""

        return CandidatePoseLLRRuntime(
            query_image_indices=self.query_image_indices.to(device=device),
            support_image_indices=self.support_image_indices.to(device=device),
            support_xy=self.support_xy.to(device=device),
            support_view_valid=self.support_view_valid.to(device=device),
            candidate_view_weights=self.candidate_view_weights.to(device=device),
            candidate_probabilities=self.candidate_probabilities.to(device=device),
            null_probabilities=self.null_probabilities.to(device=device),
        )


@dataclass(frozen=True)
class CandidatePoseLLRScore:
    """Target-free score tensors emitted for fixed pose hypotheses."""

    pose_log_likelihood_ratios: torch.Tensor
    point_log_likelihood_ratios: torch.Tensor
    candidate_log_likelihood_ratios: torch.Tensor
    edge_log_likelihood_ratios: torch.Tensor
    edge_usable: torch.Tensor


class _FullContextCropEncoder(nn.Module):
    """Encode one masked query/support crop without pose metadata.

    Real image crops close to an image boundary contain valid visual tokens and
    clamped tensor addresses for the remainder.  The latter are padding, not
    repeated visual evidence.  They are zeroed before the convolutional
    context branch and excluded from both pooling operations.
    """

    def __init__(self, descriptor_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.query_projection = nn.Linear(int(descriptor_dim), int(hidden_dim), bias=False)
        self.support_projection = nn.Linear(int(descriptor_dim), int(hidden_dim), bias=False)
        self.context = nn.Sequential(
            nn.Conv2d(int(hidden_dim) * 4, int(hidden_dim), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * 5),
            nn.Linear(int(hidden_dim) * 5, int(hidden_dim)),
            nn.GELU(),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_valid: torch.Tensor,
        support_valid: torch.Tensor,
        *,
        window_size: int,
    ) -> torch.Tensor:
        if (
            query_tokens.ndim != 3
            or support_tokens.shape != query_tokens.shape
            or query_tokens.shape[1] != int(window_size) ** 2
            or query_valid.shape != query_tokens.shape[:2]
            or support_valid.shape != query_tokens.shape[:2]
        ):
            raise ValueError("full-context crop encoder inputs are invalid")
        pair_valid = torch.as_tensor(
            query_valid, dtype=torch.bool, device=query_tokens.device
        ) & torch.as_tensor(support_valid, dtype=torch.bool, device=query_tokens.device)
        if torch.any(~torch.any(pair_valid, dim=1)):
            raise ValueError("full-context crop encoder received an empty paired crop")
        query = self.query_projection(query_tokens.to(dtype=torch.float32))
        support = self.support_projection(support_tokens.to(dtype=torch.float32))
        count = len(query)
        width = int(window_size)
        pair_mask = pair_valid.reshape(count, 1, width, width).to(dtype=query.dtype)
        fused = torch.cat(
            [query, support, query * support, torch.abs(query - support)], dim=2
        ).reshape(count, width, width, -1).permute(0, 3, 1, 2)
        encoded = self.context(fused * pair_mask) * pair_mask
        pooled_sum = F.adaptive_avg_pool2d(encoded, (2, 2))
        pooled_mass = F.adaptive_avg_pool2d(pair_mask, (2, 2))
        pooled = torch.where(
            pooled_mass > 0.0,
            pooled_sum / pooled_mass.clamp_min(torch.finfo(encoded.dtype).eps),
            torch.zeros_like(pooled_sum),
        ).reshape(count, -1)
        maximum = encoded.masked_fill(~pair_valid.reshape(count, 1, width, width), torch.finfo(encoded.dtype).min)
        maximum = maximum.amax(dim=(2, 3))
        return self.output(torch.cat([pooled, maximum], dim=1))


def _alike_shift_correlations(
    query_tokens: torch.Tensor,
    support_tokens: torch.Tensor,
    *,
    window_size: int,
    query_valid: torch.Tensor,
    support_valid: torch.Tensor,
) -> torch.Tensor:
    """Return local ALIKE correlations normalized over real paired tokens."""

    if (
            query_tokens.ndim != 3
            or support_tokens.shape != query_tokens.shape
            or query_tokens.shape[1] != int(window_size) ** 2
            or query_valid.shape != query_tokens.shape[:2]
            or support_valid.shape != query_tokens.shape[:2]
        ):
            raise ValueError("ALIKE shift-correlation inputs are invalid")
    width = int(window_size)
    query = F.normalize(query_tokens.to(dtype=torch.float32), p=2, dim=2).reshape(
        len(query_tokens), width, width, -1
    )
    support = F.normalize(support_tokens.to(dtype=torch.float32), p=2, dim=2).reshape(
        len(support_tokens), width, width, -1
    )
    query_mask = torch.as_tensor(
        query_valid, dtype=torch.bool, device=query.device
    ).reshape(len(query), width, width)
    support_mask = torch.as_tensor(
        support_valid, dtype=torch.bool, device=query.device
    ).reshape(len(query), width, width)
    correlations: list[torch.Tensor] = []
    for row_shift in (-1, 0, 1):
        for column_shift in (-1, 0, 1):
            query_rows = slice(max(row_shift, 0), width + min(row_shift, 0))
            support_rows = slice(max(-row_shift, 0), width - max(row_shift, 0))
            query_columns = slice(max(column_shift, 0), width + min(column_shift, 0))
            support_columns = slice(max(-column_shift, 0), width - max(column_shift, 0))
            values = torch.sum(
                query[:, query_rows, query_columns]
                * support[:, support_rows, support_columns],
                dim=3,
            )
            valid = (
                query_mask[:, query_rows, query_columns]
                & support_mask[:, support_rows, support_columns]
            )
            mass = valid.sum(dim=(1, 2))
            correlations.append(
                torch.where(
                    mass > 0,
                    torch.sum(values * valid.to(dtype=values.dtype), dim=(1, 2))
                    / mass.to(dtype=values.dtype).clamp_min(1.0),
                    torch.zeros_like(mass, dtype=values.dtype),
                )
            )
    return torch.stack(correlations, dim=1)


class CandidateSpecificPoseLLR(nn.Module):
    """Candidate/view LLR from full real-image context crops.

    The only inputs accepted by :meth:`forward_edge_log_likelihood_ratios` are
    image-grid addresses and crop coordinates.  Pose projection happens in the
    caller and no pose matrix, residual, target, track identity, or coarse
    posterior reaches this neural encoder.
    """

    def __init__(
        self,
        *,
        sources: Mapping[str, torch.Tensor],
        image_sizes: torch.Tensor,
        hidden_dim: int = 32,
        max_abs_log_ratio: float = 3.0,
        edge_chunk_size: int = 256,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if set(sources) != set(POSE_LLR_SCALE_WINDOWS):
            raise ValueError("candidate pose-LLR source set is incomplete")
        if int(hidden_dim) < 4 or int(edge_chunk_size) <= 0:
            raise ValueError("candidate pose-LLR hidden dimension or chunk size is invalid")
        sizes = torch.as_tensor(image_sizes, dtype=torch.float32)
        if sizes.ndim != 2 or sizes.shape[1] != 2 or torch.any(sizes <= 1.0):
            raise ValueError("candidate pose-LLR image sizes are invalid")
        encoders: dict[str, _FullContextCropEncoder] = {}
        for name, window in POSE_LLR_SCALE_WINDOWS.items():
            grid = torch.as_tensor(sources[name])
            if (
                grid.ndim != 4
                or grid.shape[0] != len(sizes)
                or grid.shape[1] != grid.shape[2]
                or grid.shape[1] < int(window)
                or grid.shape[3] <= 0
                or not torch.isfinite(grid).all()
            ):
                raise ValueError(f"candidate pose-LLR {name} source grid is invalid")
            norms = torch.linalg.vector_norm(grid.to(dtype=torch.float32), dim=-1)
            if torch.max(torch.abs(norms - 1.0)) > 5e-3:
                raise ValueError(f"candidate pose-LLR {name} descriptors are not normalized")
            self.register_buffer(f"_{name}_grid", grid.to(dtype=torch.float32), persistent=False)
            encoders[name] = _FullContextCropEncoder(int(grid.shape[3]), int(hidden_dim))
        self.encoders = nn.ModuleDict(encoders)
        self.register_buffer("_image_sizes", sizes, persistent=False)
        self.max_abs_log_ratio = float(max_abs_log_ratio)
        if self.max_abs_log_ratio <= 0.0:
            raise ValueError("candidate pose-LLR maximum edge score must be positive")
        self.edge_chunk_size = int(edge_chunk_size)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.edge_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * len(POSE_LLR_SCALE_WINDOWS) + 9),
            nn.Linear(int(hidden_dim) * len(POSE_LLR_SCALE_WINDOWS) + 9, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        final = self.edge_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def device(self) -> torch.device:
        return self._image_sizes.device

    def _edge_chunk(
        self,
        *,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_scales: list[torch.Tensor] = []
        alike_correlations: torch.Tensor | None = None
        all_valid = torch.ones((len(query_xy),), dtype=torch.bool, device=self.device)
        for name, window in POSE_LLR_SCALE_WINDOWS.items():
            grid = getattr(self, f"_{name}_grid")
            query_crop, query_valid = _crop_subpixel_grid_tokens(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=query_image_indices,
                xy=query_xy,
                window_size=int(window),
            )
            support_crop, support_valid = _crop_subpixel_grid_tokens(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=support_image_indices,
                xy=support_xy,
                window_size=int(window),
            )
            paired_tokens = torch.any(query_valid & support_valid, dim=1)
            all_valid &= paired_tokens
            # Fully out-of-image candidate projections must remain fixed
            # neutral edges.  Give the encoder one zero-valued safe token so
            # it can execute its batch path; ``all_valid`` below removes this
            # synthetic entry from both the score mixture and its gradients.
            safe_query_valid = query_valid.clone()
            safe_support_valid = support_valid.clone()
            empty = ~paired_tokens
            safe_query_valid[empty, 0] = True
            safe_support_valid[empty, 0] = True
            encoded_scales.append(
                self.encoders[name](
                    query_crop,
                    support_crop,
                    safe_query_valid,
                    safe_support_valid,
                    window_size=int(window),
                )
            )
            if name == "alike":
                alike_correlations = _alike_shift_correlations(
                    query_crop,
                    support_crop,
                    window_size=int(window),
                    query_valid=query_valid,
                    support_valid=support_valid,
                )
        if alike_correlations is None:
            raise RuntimeError("candidate pose-LLR ALIKE spatial branch was not constructed")
        raw = self.edge_head(torch.cat([*encoded_scales, alike_correlations], dim=1)).reshape(-1)
        return bounded_log_likelihood_ratio(
            raw, max_abs_log_ratio=self.max_abs_log_ratio
        ), all_valid

    def forward_edge_log_likelihood_ratios(
        self,
        *,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
        declared_usable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score independent candidate/view image crops in memory-bounded chunks."""

        query_indices = torch.as_tensor(query_image_indices, dtype=torch.long, device=self.device).reshape(-1)
        query_coordinates = torch.as_tensor(query_xy, dtype=torch.float32, device=self.device)
        support_indices = torch.as_tensor(support_image_indices, dtype=torch.long, device=self.device).reshape(-1)
        support_coordinates = torch.as_tensor(support_xy, dtype=torch.float32, device=self.device)
        usable = torch.as_tensor(declared_usable, dtype=torch.bool, device=self.device).reshape(-1)
        count = len(query_indices)
        if (
            count == 0
            or query_coordinates.shape != (count, 2)
            or support_indices.shape != (count,)
            or support_coordinates.shape != (count, 2)
            or usable.shape != (count,)
            or torch.any(query_indices < 0)
            or torch.any(support_indices < 0)
            or torch.any(query_indices >= len(self._image_sizes))
            or torch.any(support_indices >= len(self._image_sizes))
            or not torch.isfinite(query_coordinates).all()
            or not torch.isfinite(support_coordinates).all()
        ):
            raise ValueError("candidate pose-LLR edge crop inputs are invalid")
        llrs: list[torch.Tensor] = []
        crop_usable: list[torch.Tensor] = []
        for begin in range(0, count, self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, count)
            chunk_query_indices = query_indices[begin:end]
            chunk_query_coordinates = query_coordinates[begin:end]
            chunk_support_indices = support_indices[begin:end]
            chunk_support_coordinates = support_coordinates[begin:end]
            if self.training and self.activation_checkpointing and torch.is_grad_enabled():
                part, valid = checkpoint(
                    self._edge_chunk_from_tensors,
                    chunk_query_indices,
                    chunk_query_coordinates,
                    chunk_support_indices,
                    chunk_support_coordinates,
                    use_reentrant=False,
                )
            else:
                part, valid = self._edge_chunk(
                    query_image_indices=chunk_query_indices,
                    query_xy=chunk_query_coordinates,
                    support_image_indices=chunk_support_indices,
                    support_xy=chunk_support_coordinates,
                )
            llrs.append(part)
            crop_usable.append(valid & usable[begin:end])
        return torch.cat(llrs, dim=0), torch.cat(crop_usable, dim=0)

    def _edge_chunk_from_tensors(
        self,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Positional checkpoint adapter for one candidate/view crop chunk."""

        return self._edge_chunk(
            query_image_indices=query_image_indices,
            query_xy=query_xy,
            support_image_indices=support_image_indices,
            support_xy=support_xy,
        )

    def forward(
        self,
        runtime: CandidatePoseLLRRuntime,
        candidate_query_xy: torch.Tensor,
        candidate_projection_valid: torch.Tensor,
        missing_edge_log_likelihood_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Return one target-free pose LLR per input hypothesis batch.

        Keeping this public path on ``nn.Module.forward`` is required for DDP
        to register gradient synchronization around the image-context encoder.
        """

        return score_candidate_pose_batch(
            model=self,
            runtime=runtime,
            candidate_query_xy=candidate_query_xy,
            candidate_projection_valid=candidate_projection_valid,
            missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
        ).pose_log_likelihood_ratios


def score_candidate_pose_batch(
    *,
    model: CandidateSpecificPoseLLR,
    runtime: CandidatePoseLLRRuntime,
    candidate_query_xy: torch.Tensor,
    candidate_projection_valid: torch.Tensor,
    missing_edge_log_likelihood_ratio: float = 0.0,
) -> CandidatePoseLLRScore:
    """Score frozen candidates under caller-provided projected query positions."""

    active = runtime.to(model.device)
    projected_xy = torch.as_tensor(candidate_query_xy, dtype=torch.float32, device=model.device)
    projected_valid = torch.as_tensor(
        candidate_projection_valid, dtype=torch.bool, device=model.device
    )
    if (
        projected_xy.ndim != 4
        or projected_xy.shape[1:] != (*active.candidate_probabilities.shape, 2)
        or projected_valid.shape != projected_xy.shape[:3]
        or not torch.isfinite(projected_xy).all()
    ):
        raise ValueError("candidate pose-LLR projected query positions are invalid")
    batch, point_count, candidate_count, view_count = (
        int(projected_xy.shape[0]),
        int(projected_xy.shape[1]),
        int(projected_xy.shape[2]),
        int(active.support_view_valid.shape[2]),
    )
    query_indices = active.query_image_indices.reshape(1, point_count, 1, 1).expand(
        batch, -1, candidate_count, view_count
    ).reshape(-1)
    query_xy = projected_xy.unsqueeze(3).expand(-1, -1, -1, view_count, -1).reshape(-1, 2)
    support_indices = active.support_image_indices.reshape(1, point_count, candidate_count, view_count).expand(
        batch, -1, -1, -1
    ).reshape(-1)
    support_xy = active.support_xy.reshape(1, point_count, candidate_count, view_count, 2).expand(
        batch, -1, -1, -1, -1
    ).reshape(-1, 2)
    declared_usable = (
        projected_valid.unsqueeze(3)
        & active.support_view_valid.reshape(1, point_count, candidate_count, view_count)
    ).reshape(-1)
    edge_llr, edge_usable = model.forward_edge_log_likelihood_ratios(
        query_image_indices=query_indices,
        query_xy=query_xy,
        support_image_indices=support_indices,
        support_xy=support_xy,
        declared_usable=declared_usable,
    )
    edge_llr = edge_llr.reshape(batch, point_count, candidate_count, view_count)
    edge_usable = edge_usable.reshape(batch, point_count, candidate_count, view_count)
    point_llr, candidate_llr = fixed_candidate_view_mixture_log_ratio(
        edge_log_likelihood_ratios=edge_llr,
        edge_usable=edge_usable,
        candidate_view_weights=active.candidate_view_weights,
        candidate_probabilities=active.candidate_probabilities,
        null_probabilities=active.null_probabilities,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
    )
    return CandidatePoseLLRScore(
        pose_log_likelihood_ratios=point_llr.mean(dim=1),
        point_log_likelihood_ratios=point_llr,
        candidate_log_likelihood_ratios=candidate_llr,
        edge_log_likelihood_ratios=edge_llr,
        edge_usable=edge_usable,
    )
