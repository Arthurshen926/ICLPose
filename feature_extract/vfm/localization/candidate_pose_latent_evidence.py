"""Current-layout latent identity and pose-alignment visual evidence.

The older candidate-pose LLR averages every fixed top-L candidate over every
verification token.  That is a poor fit when only a small subset of tokens
contains a retrievable physical landmark.  This module keeps the same fixed
top-L/support/null contract but separates two target-free runtime quantities:

* a candidate identity posterior evaluated at the observed query token; and
* a candidate-specific alignment LLR evaluated at a pose projection.

The identity posterior produces static token weights before any pose is read.
The pose path then marginalizes the immutable candidate/view mixture and uses
those static weights.  Training-only labels can supervise both heads, but the
runtime model never accepts pose targets, residuals, track IDs, or coarse
scores as neural inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from feature_extract.vfm.localization.candidate_pose_llr import (
    POSE_LLR_SCALE_WINDOWS,
    CandidatePoseLLRRuntime,
    _FullContextCropEncoder,
    _alike_shift_correlations,
    _crop_subpixel_grid_tokens,
    bounded_log_likelihood_ratio,
)


CANDIDATE_POSE_LATENT_EVIDENCE_FORMAT = "candidate_pose_latent_evidence_v1"


def _candidate_view_log_mixture(
    *,
    edge_log_values: torch.Tensor,
    edge_usable: torch.Tensor,
    candidate_view_weights: torch.Tensor,
    missing_edge_log_value: float = 0.0,
) -> torch.Tensor:
    """Marginalize fixed real support views without turning missingness into a cue."""

    values = torch.as_tensor(edge_log_values, dtype=torch.float32)
    usable = torch.as_tensor(edge_usable, dtype=torch.bool, device=values.device)
    weights = torch.as_tensor(
        candidate_view_weights, dtype=torch.float32, device=values.device
    )
    missing = float(missing_edge_log_value)
    if (
        values.ndim != 4
        or usable.shape != values.shape
        or weights.shape != values.shape[1:]
        or not torch.isfinite(values).all()
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0.0)
        or not torch.isfinite(torch.as_tensor(missing))
    ):
        raise ValueError("candidate/view log-mixture inputs are invalid")
    view_mass = weights.sum(dim=2)
    active = view_mass > 0.0
    if torch.any(torch.abs(view_mass[active] - 1.0) > 1e-4) or torch.any(
        view_mass[~active] > 1e-6
    ):
        raise ValueError("candidate/view weights must preserve each active candidate")
    effective = torch.where(usable, values, torch.full_like(values, missing))
    safe_log_weights = torch.where(
        weights > 0.0,
        torch.log(weights),
        torch.full_like(weights, -torch.inf),
    )
    mixed = torch.logsumexp(effective + safe_log_weights.unsqueeze(0), dim=3)
    return torch.where(active.unsqueeze(0), mixed, torch.zeros_like(mixed))


def candidate_pose_point_log_mixture(
    *,
    candidate_alignment: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    null_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Marginalize a fixed soft candidate posterior for each pose and point.

    ``candidate_probabilities`` may be the learned identity posterior evaluated
    at the observed query coordinate.  It is fixed before a pose is read, so
    this remains a soft top-L mixture rather than pose-specific candidate
    reselection or a hard identity decision.
    """

    alignment = torch.as_tensor(candidate_alignment, dtype=torch.float32)
    candidates = torch.as_tensor(
        candidate_probabilities, dtype=torch.float32, device=alignment.device
    )
    null = torch.as_tensor(
        null_probabilities, dtype=torch.float32, device=alignment.device
    ).reshape(-1)
    if (
        alignment.ndim != 3
        or alignment.shape[0] == 0
        or candidates.shape != alignment.shape[1:]
        or null.shape != (alignment.shape[1],)
        or not torch.isfinite(alignment).all()
        or not torch.isfinite(candidates).all()
        or not torch.isfinite(null).all()
        or torch.any(candidates < 0.0)
        or torch.any(null < 0.0)
        or torch.any(torch.abs(candidates.sum(dim=1) + null - 1.0) > 1e-4)
    ):
        raise ValueError("latent evidence candidate point mixture inputs are invalid")
    candidate_terms = torch.where(
        candidates > 0.0,
        torch.log(candidates).unsqueeze(0) + alignment,
        torch.full_like(alignment, -torch.inf),
    )
    null_terms = torch.where(
        null > 0.0,
        torch.log(null),
        torch.full_like(null, -torch.inf),
    ).reshape(1, len(null), 1).expand(len(alignment), -1, -1)
    return torch.logsumexp(torch.cat((candidate_terms, null_terms), dim=2), dim=2)


def identity_posterior_from_residual(
    *,
    candidate_residual: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    null_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply visual residuals while exactly preserving fixed non-null/null mass."""

    residual = torch.as_tensor(candidate_residual, dtype=torch.float32)
    candidate = torch.as_tensor(
        candidate_probabilities, dtype=torch.float32, device=residual.device
    )
    null = torch.as_tensor(
        null_probabilities, dtype=torch.float32, device=residual.device
    ).reshape(-1)
    if (
        residual.ndim != 2
        or candidate.shape != residual.shape
        or null.shape != (len(residual),)
        or len(residual) == 0
        or not torch.isfinite(residual).all()
        or not torch.isfinite(candidate).all()
        or not torch.isfinite(null).all()
        or torch.any(candidate < 0.0)
        or torch.any(null < 0.0)
    ):
        raise ValueError("identity posterior inputs are invalid")
    nonnull_mass = candidate.sum(dim=1)
    if torch.any(nonnull_mass <= 0.0) or torch.any(
        torch.abs(nonnull_mass + null - 1.0) > 1e-4
    ):
        raise ValueError("identity posterior prior mass is invalid")
    conditional_log_prior = torch.where(
        candidate > 0.0,
        torch.log(candidate / nonnull_mass[:, None]),
        torch.full_like(candidate, -torch.inf),
    )
    conditional = torch.softmax(conditional_log_prior + residual, dim=1)
    posterior = conditional * nonnull_mass[:, None]
    if torch.any(~torch.isfinite(posterior)) or torch.any(
        torch.abs(posterior.sum(dim=1) + null - 1.0) > 1e-4
    ):
        raise RuntimeError("identity posterior did not preserve probability mass")
    return posterior, null, conditional


def identity_selector_weights(
    conditional_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return a pose-independent confidence weight from a fixed identity posterior."""

    conditional = torch.as_tensor(conditional_probabilities, dtype=torch.float32)
    if (
        conditional.ndim != 2
        or conditional.shape[0] == 0
        or conditional.shape[1] < 2
        or not torch.isfinite(conditional).all()
        or torch.any(conditional < 0.0)
        or torch.any(torch.abs(conditional.sum(dim=1) - 1.0) > 1e-4)
    ):
        raise ValueError("identity selector posterior is invalid")
    # The maximum conditional identity probability is target-free, fixed before
    # hypothesis scoring, and has a nonzero uniform baseline of 1 / top-L.
    return conditional.max(dim=1).values


def weighted_pose_log_likelihood_ratio(
    *,
    point_log_likelihood_ratios: torch.Tensor,
    selector_weights: torch.Tensor,
) -> torch.Tensor:
    """Aggregate candidate-marginalized point evidence with static token weights."""

    points = torch.as_tensor(point_log_likelihood_ratios, dtype=torch.float32)
    weights = torch.as_tensor(selector_weights, dtype=torch.float32, device=points.device)
    if (
        points.ndim != 2
        or weights.shape != (points.shape[1],)
        or points.shape[0] == 0
        or not torch.isfinite(points).all()
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0.0)
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError("weighted pose likelihood inputs are invalid")
    return (points * weights[None, :]).sum(dim=1) / weights.sum()


def same_track_alignment_margin_loss(
    *,
    correct_alignment: torch.Tensor,
    coherent_wrong_alignment: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train directly on same-track correct versus coherent-wrong alignment."""

    correct = torch.as_tensor(correct_alignment, dtype=torch.float32)
    wrong = torch.as_tensor(
        coherent_wrong_alignment, dtype=torch.float32, device=correct.device
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
        raise ValueError("same-track alignment margin inputs are invalid")
    hardest_wrong = wrong.max(dim=1).values
    gap = correct - hardest_wrong
    loss = torch.nn.functional.softplus(
        torch.as_tensor(value, dtype=correct.dtype, device=correct.device) - gap
    ).mean()
    return loss, {
        "pair_count": float(len(gap)),
        "correct_win_fraction": float((gap > 0.0).float().mean().item()),
        "mean_correct_minus_hardest_wrong": float(gap.detach().mean().item()),
    }


def direct_candidate_alignment_scores(
    *,
    candidate_alignment: torch.Tensor,
    target_classes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average the train-only true-candidate edge score for every pose.

    A registered landmark whose track is absent from the frozen top-L has the
    explicit fixed-null class ``candidate_count``.  It cannot identify an
    edge, just as an unlabelled point cannot, so both are excluded here.  This
    helper is deliberately target-bearing and may only be called by training
    or train-only diagnostics, never by the target-free scorer.
    """

    alignment = torch.as_tensor(candidate_alignment, dtype=torch.float32)
    targets = torch.as_tensor(
        target_classes, dtype=torch.long, device=alignment.device
    ).reshape(-1)
    if (
        alignment.ndim != 3
        or alignment.shape[0] == 0
        or alignment.shape[1] == 0
        or alignment.shape[2] <= 1
        or targets.shape != (alignment.shape[1],)
        or not torch.isfinite(alignment).all()
        or torch.any(targets < -1)
        or torch.any(targets > alignment.shape[2])
    ):
        raise ValueError("direct candidate alignment inputs are invalid")
    candidate_count = int(alignment.shape[2])
    exact = (targets >= 0) & (targets < candidate_count)
    if not torch.any(exact):
        raise ValueError("direct candidate alignment has no exact top-L target")
    point_indices = torch.arange(
        alignment.shape[1], device=alignment.device, dtype=torch.long
    )[exact]
    candidate_indices = targets[exact]
    direct_edges = alignment[:, point_indices, candidate_indices]
    return direct_edges.mean(dim=1), exact


@dataclass(frozen=True)
class CandidateIdentityPosterior:
    """Static candidate identity posterior produced before any pose is scored."""

    candidate_probabilities: torch.Tensor
    null_probabilities: torch.Tensor
    conditional_probabilities: torch.Tensor
    candidate_residual: torch.Tensor
    selector_weights: torch.Tensor
    edge_usable: torch.Tensor


@dataclass(frozen=True)
class CandidatePoseLatentScore:
    """Pose-conditioned alignment evidence after fixed candidate marginalization."""

    pose_log_likelihood_ratios: torch.Tensor
    point_log_likelihood_ratios: torch.Tensor
    candidate_log_likelihood_ratios: torch.Tensor
    edge_log_likelihood_ratios: torch.Tensor
    edge_usable: torch.Tensor


class CandidatePoseLatentEvidence(nn.Module):
    """Shared visual encoder with identity and pose-alignment heads.

    The model is agnostic to track identifiers and ground-truth data.  All
    candidate priors and support weights are passed through the explicit runtime
    contract and are only used by fixed marginalizers outside neural heads.
    """

    def __init__(
        self,
        *,
        sources: Mapping[str, torch.Tensor],
        image_sizes: torch.Tensor,
        hidden_dim: int = 32,
        max_abs_identity_residual: float = 3.0,
        max_abs_alignment_log_ratio: float = 3.0,
        edge_chunk_size: int = 256,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if set(sources) != set(POSE_LLR_SCALE_WINDOWS):
            raise ValueError("latent evidence source set is incomplete")
        if int(hidden_dim) < 4 or int(edge_chunk_size) <= 0:
            raise ValueError("latent evidence hidden dimension or chunk size is invalid")
        sizes = torch.as_tensor(image_sizes, dtype=torch.float32)
        if sizes.ndim != 2 or sizes.shape[1] != 2 or torch.any(sizes <= 1.0):
            raise ValueError("latent evidence image sizes are invalid")
        encoders: dict[str, _FullContextCropEncoder] = {}
        for name, window in POSE_LLR_SCALE_WINDOWS.items():
            grid = torch.as_tensor(sources[name], dtype=torch.float32)
            if (
                grid.ndim != 4
                or grid.shape[0] != len(sizes)
                or grid.shape[1] != grid.shape[2]
                or grid.shape[1] < int(window)
                or grid.shape[3] <= 0
                or not torch.isfinite(grid).all()
            ):
                raise ValueError(f"latent evidence {name} source grid is invalid")
            norms = torch.linalg.vector_norm(grid, dim=-1)
            if torch.max(torch.abs(norms - 1.0)) > 5e-3:
                raise ValueError(f"latent evidence {name} descriptors are not normalized")
            self.register_buffer(f"_{name}_grid", grid, persistent=False)
            encoders[name] = _FullContextCropEncoder(int(grid.shape[3]), int(hidden_dim))
        self.encoders = nn.ModuleDict(encoders)
        self.register_buffer("_image_sizes", sizes, persistent=False)
        representation_dim = int(hidden_dim) * len(POSE_LLR_SCALE_WINDOWS) + 9
        self.identity_head = self._head(representation_dim, int(hidden_dim))
        self.alignment_head = self._head(representation_dim, int(hidden_dim))
        self.max_abs_identity_residual = float(max_abs_identity_residual)
        self.max_abs_alignment_log_ratio = float(max_abs_alignment_log_ratio)
        if (
            self.max_abs_identity_residual <= 0.0
            or self.max_abs_alignment_log_ratio <= 0.0
        ):
            raise ValueError("latent evidence output caps must be positive")
        self.edge_chunk_size = int(edge_chunk_size)
        self.activation_checkpointing = bool(activation_checkpointing)

    @staticmethod
    def _head(input_dim: int, hidden_dim: int) -> nn.Sequential:
        head = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        final = head[-1]
        assert isinstance(final, nn.Linear)
        # A small nonzero output layer lets the shared crop encoder receive a
        # gradient on the first step, unlike an exactly-zero LLR head.
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final.bias)
        return head

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            paired = torch.any(query_valid & support_valid, dim=1)
            all_valid &= paired
            # Empty crops remain neutral after the declared usable mask is
            # applied below.  One synthetic zero token only keeps the encoder
            # numerically well-defined for this batch element.
            safe_query_valid = query_valid.clone()
            safe_support_valid = support_valid.clone()
            empty = ~paired
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
            raise RuntimeError("latent evidence ALIKE branch was not constructed")
        representation = torch.cat([*encoded_scales, alike_correlations], dim=1)
        identity = bounded_log_likelihood_ratio(
            self.identity_head(representation).reshape(-1),
            max_abs_log_ratio=self.max_abs_identity_residual,
        )
        alignment = bounded_log_likelihood_ratio(
            self.alignment_head(representation).reshape(-1),
            max_abs_log_ratio=self.max_abs_alignment_log_ratio,
        )
        return identity, alignment, all_valid

    def _edge_chunk_from_tensors(
        self,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._edge_chunk(
            query_image_indices=query_image_indices,
            query_xy=query_xy,
            support_image_indices=support_image_indices,
            support_xy=support_xy,
        )

    def _edge_outputs(
        self,
        *,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
        declared_usable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_indices = torch.as_tensor(
            query_image_indices, dtype=torch.long, device=self.device
        ).reshape(-1)
        query_coordinates = torch.as_tensor(
            query_xy, dtype=torch.float32, device=self.device
        )
        support_indices = torch.as_tensor(
            support_image_indices, dtype=torch.long, device=self.device
        ).reshape(-1)
        support_coordinates = torch.as_tensor(
            support_xy, dtype=torch.float32, device=self.device
        )
        usable = torch.as_tensor(
            declared_usable, dtype=torch.bool, device=self.device
        ).reshape(-1)
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
            raise ValueError("latent evidence edge inputs are invalid")
        identity_parts: list[torch.Tensor] = []
        alignment_parts: list[torch.Tensor] = []
        usable_parts: list[torch.Tensor] = []
        for begin in range(0, count, self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, count)
            chunk = (
                query_indices[begin:end],
                query_coordinates[begin:end],
                support_indices[begin:end],
                support_coordinates[begin:end],
            )
            if self.training and self.activation_checkpointing and torch.is_grad_enabled():
                identity, alignment, crop_usable = checkpoint(
                    self._edge_chunk_from_tensors,
                    *chunk,
                    use_reentrant=False,
                )
            else:
                identity, alignment, crop_usable = self._edge_chunk(
                    query_image_indices=chunk[0],
                    query_xy=chunk[1],
                    support_image_indices=chunk[2],
                    support_xy=chunk[3],
                )
            identity_parts.append(identity)
            alignment_parts.append(alignment)
            usable_parts.append(crop_usable & usable[begin:end])
        return (
            torch.cat(identity_parts, dim=0),
            torch.cat(alignment_parts, dim=0),
            torch.cat(usable_parts, dim=0),
        )

    def _edge_outputs_for_positions(
        self,
        *,
        runtime: CandidatePoseLLRRuntime,
        candidate_query_xy: torch.Tensor,
        candidate_projection_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = runtime.to(self.device)
        projected = torch.as_tensor(
            candidate_query_xy, dtype=torch.float32, device=self.device
        )
        projected_valid = torch.as_tensor(
            candidate_projection_valid, dtype=torch.bool, device=self.device
        )
        if (
            projected.ndim != 4
            or projected.shape[1:] != (*active.candidate_probabilities.shape, 2)
            or projected_valid.shape != projected.shape[:3]
            or not torch.isfinite(projected).all()
        ):
            raise ValueError("latent evidence candidate positions are invalid")
        batch, point_count, candidate_count, view_count = (
            int(projected.shape[0]),
            int(projected.shape[1]),
            int(projected.shape[2]),
            int(active.support_view_valid.shape[2]),
        )
        query_indices = active.query_image_indices.reshape(1, point_count, 1, 1).expand(
            batch, -1, candidate_count, view_count
        ).reshape(-1)
        query_xy = projected.unsqueeze(3).expand(
            -1, -1, -1, view_count, -1
        ).reshape(-1, 2)
        support_indices = active.support_image_indices.reshape(
            1, point_count, candidate_count, view_count
        ).expand(batch, -1, -1, -1).reshape(-1)
        support_xy = active.support_xy.reshape(
            1, point_count, candidate_count, view_count, 2
        ).expand(batch, -1, -1, -1, -1).reshape(-1, 2)
        declared = (
            projected_valid.unsqueeze(3)
            & active.support_view_valid.reshape(1, point_count, candidate_count, view_count)
        ).reshape(-1)
        identity, alignment, usable = self._edge_outputs(
            query_image_indices=query_indices,
            query_xy=query_xy,
            support_image_indices=support_indices,
            support_xy=support_xy,
            declared_usable=declared,
        )
        shape = (batch, point_count, candidate_count, view_count)
        return identity.reshape(shape), alignment.reshape(shape), usable.reshape(shape)

    def identity_posterior(
        self,
        *,
        runtime: CandidatePoseLLRRuntime,
        observed_xy: torch.Tensor,
    ) -> CandidateIdentityPosterior:
        """Score fixed candidates at observed query coordinates before pose scoring."""

        active = runtime.to(self.device)
        observed = torch.as_tensor(observed_xy, dtype=torch.float32, device=self.device)
        point_count, candidate_count = active.candidate_probabilities.shape
        if observed.shape != (point_count, 2) or not torch.isfinite(observed).all():
            raise ValueError("latent identity observed coordinates are invalid")
        positions = observed.reshape(1, point_count, 1, 2).expand(
            1, -1, candidate_count, -1
        )
        declared = torch.ones(
            (1, point_count, candidate_count), dtype=torch.bool, device=self.device
        )
        identity_edges, _alignment, usable = self._edge_outputs_for_positions(
            runtime=active,
            candidate_query_xy=positions,
            candidate_projection_valid=declared,
        )
        residual = _candidate_view_log_mixture(
            edge_log_values=identity_edges,
            edge_usable=usable,
            candidate_view_weights=active.candidate_view_weights,
        ).squeeze(0)
        candidate, null, conditional = identity_posterior_from_residual(
            candidate_residual=residual,
            candidate_probabilities=active.candidate_probabilities,
            null_probabilities=active.null_probabilities,
        )
        return CandidateIdentityPosterior(
            candidate_probabilities=candidate,
            null_probabilities=null,
            conditional_probabilities=conditional,
            candidate_residual=residual,
            selector_weights=identity_selector_weights(conditional),
            edge_usable=usable.squeeze(0),
        )

    def forward(
        self,
        *,
        runtime: CandidatePoseLLRRuntime,
        observed_xy: torch.Tensor,
        candidate_query_xy: torch.Tensor,
        candidate_projection_valid: torch.Tensor,
        alignment_selector_mask: torch.Tensor | None = None,
    ) -> tuple[CandidateIdentityPosterior, CandidatePoseLatentScore, torch.Tensor]:
        """Run the train-time identity-to-pose chain through one DDP forward.

        ``alignment_selector_mask`` is a training-loss mask only.  It is
        applied after the identity visual encoder has produced its posterior,
        and is never accepted by target-free scoring.  The identity posterior
        is detached before candidate mixing and token weighting so pose-margin
        gradients cannot turn selector learning into a pose-label shortcut.
        """

        identity = self.identity_posterior(runtime=runtime, observed_xy=observed_xy)
        selector = identity.selector_weights.detach()
        if alignment_selector_mask is None:
            active_mask = torch.ones_like(selector, dtype=torch.bool)
        else:
            active_mask = torch.as_tensor(
                alignment_selector_mask, dtype=torch.bool, device=self.device
            ).reshape(-1)
            if active_mask.shape != selector.shape:
                raise ValueError("latent evidence alignment selector mask is invalid")
        masked_selector = selector * active_mask.to(dtype=selector.dtype)
        has_alignment_tokens = masked_selector.sum() > 0.0
        # Keep the alignment branch attached to DDP even for a training query
        # with no exact top-L identity.  Its loss is explicitly zeroed by the
        # caller using ``has_alignment_tokens``.
        pose_selector = torch.where(
            has_alignment_tokens,
            masked_selector,
            selector,
        )
        pose = self.pose_log_likelihood_ratios(
            runtime=runtime,
            candidate_query_xy=candidate_query_xy,
            candidate_projection_valid=candidate_projection_valid,
            selector_weights=pose_selector,
            static_candidate_probabilities=identity.candidate_probabilities.detach(),
            static_null_probabilities=identity.null_probabilities.detach(),
        )
        return identity, pose, has_alignment_tokens

    def pose_log_likelihood_ratios(
        self,
        *,
        runtime: CandidatePoseLLRRuntime,
        candidate_query_xy: torch.Tensor,
        candidate_projection_valid: torch.Tensor,
        selector_weights: torch.Tensor,
        static_candidate_probabilities: torch.Tensor | None = None,
        static_null_probabilities: torch.Tensor | None = None,
    ) -> CandidatePoseLatentScore:
        """Score pose projections with static identity-derived soft evidence."""

        active = runtime.to(self.device)
        _identity, alignment_edges, usable = self._edge_outputs_for_positions(
            runtime=active,
            candidate_query_xy=candidate_query_xy,
            candidate_projection_valid=candidate_projection_valid,
        )
        candidate_alignment = _candidate_view_log_mixture(
            edge_log_values=alignment_edges,
            edge_usable=usable,
            candidate_view_weights=active.candidate_view_weights,
        )
        if (static_candidate_probabilities is None) != (static_null_probabilities is None):
            raise ValueError("latent evidence static candidate and null posterior must be paired")
        candidates = (
            active.candidate_probabilities
            if static_candidate_probabilities is None
            else torch.as_tensor(
                static_candidate_probabilities,
                dtype=torch.float32,
                device=self.device,
            )
        )
        null = (
            active.null_probabilities
            if static_null_probabilities is None
            else torch.as_tensor(
                static_null_probabilities,
                dtype=torch.float32,
                device=self.device,
            )
        )
        point = candidate_pose_point_log_mixture(
            candidate_alignment=candidate_alignment,
            candidate_probabilities=candidates,
            null_probabilities=null,
        )
        pose = weighted_pose_log_likelihood_ratio(
            point_log_likelihood_ratios=point,
            selector_weights=torch.as_tensor(selector_weights, device=self.device),
        )
        return CandidatePoseLatentScore(
            pose_log_likelihood_ratios=pose,
            point_log_likelihood_ratios=point,
            candidate_log_likelihood_ratios=candidate_alignment,
            edge_log_likelihood_ratios=alignment_edges,
            edge_usable=usable,
        )
