"""Candidate-conditioned maplet assignment with an explicit dustbin."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.localization.local_assignment_matcher import log_optimal_transport


@dataclass(frozen=True)
class CandidateMapletMatcherConfig:
    query_input_dim: int
    support_input_dim: int
    static_input_dim: int
    descriptor_dim: int = 64
    model_dim: int = 96
    num_heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    sinkhorn_iterations: int = 10
    descriptor_prior_scale: float = 5.0
    descriptor_prior_center: float = 0.65
    candidate_set_layers: int = 1
    candidate_prior_index: int = 1
    candidate_prior_scale: float = 20.0
    static_feature_mean: tuple[float, ...] | None = None
    static_feature_scale: tuple[float, ...] | None = None
    geometry_validity_enabled: bool = False
    decoupled_candidate_heads: bool = False
    candidate_view_marginalization_enabled: bool = False
    geometry_validity_thresholds_px: tuple[float, ...] = (1.0, 2.0, 5.0)
    rescue_policy_enabled: bool = False
    rescue_candidate_threshold_px: float = 5.0
    rescue_baseline_invalid_threshold_px: float = 5.0

    def __post_init__(self) -> None:
        for name in ("query_input_dim", "support_input_dim", "static_input_dim", "descriptor_dim", "model_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.descriptor_dim) > min(int(self.query_input_dim), int(self.support_input_dim)):
            raise ValueError("descriptor_dim exceeds an input feature dimension")
        if int(self.num_heads) <= 0 or int(self.model_dim) % int(self.num_heads) != 0:
            raise ValueError("num_heads must divide model_dim")
        if (
            int(self.layers) <= 0
            or int(self.sinkhorn_iterations) <= 0
            or int(self.candidate_set_layers) <= 0
        ):
            raise ValueError(
                "layers, sinkhorn_iterations, and candidate_set_layers must be positive"
            )
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.descriptor_prior_scale) <= 0.0:
            raise ValueError("descriptor_prior_scale must be positive")
        if int(self.candidate_prior_index) < 0 or int(self.candidate_prior_index) >= int(
            self.static_input_dim
        ):
            raise ValueError("candidate_prior_index is outside static input features")
        if float(self.candidate_prior_scale) <= 0.0:
            raise ValueError("candidate_prior_scale must be positive")
        if (self.static_feature_mean is None) != (self.static_feature_scale is None):
            raise ValueError("static feature mean and scale must be configured together")
        if self.static_feature_mean is not None and self.static_feature_scale is not None:
            if (
                len(self.static_feature_mean) != int(self.static_input_dim)
                or len(self.static_feature_scale) != int(self.static_input_dim)
            ):
                raise ValueError("static normalization dimensions differ from static_input_dim")
            if not all(math.isfinite(float(value)) for value in self.static_feature_mean):
                raise ValueError("static feature means must be finite")
            if not all(
                math.isfinite(float(value)) and float(value) > 0.0
                for value in self.static_feature_scale
            ):
                raise ValueError("static feature scales must be finite and positive")
        thresholds = tuple(float(value) for value in self.geometry_validity_thresholds_px)
        if (
            len(thresholds) != 3
            or any(not math.isfinite(value) or value <= 0.0 for value in thresholds)
            or tuple(sorted(thresholds)) != thresholds
            or len(set(thresholds)) != len(thresholds)
        ):
            raise ValueError(
                "geometry validity requires three strictly increasing positive thresholds"
            )
        if (
            not math.isfinite(float(self.rescue_candidate_threshold_px))
            or not math.isfinite(float(self.rescue_baseline_invalid_threshold_px))
            or float(self.rescue_candidate_threshold_px) <= 0.0
            or float(self.rescue_baseline_invalid_threshold_px) <= 0.0
            or float(self.rescue_candidate_threshold_px)
            > float(self.rescue_baseline_invalid_threshold_px)
        ):
            raise ValueError(
                "rescue thresholds must be positive and candidate <= baseline-invalid"
            )
        if bool(self.decoupled_candidate_heads) and not bool(
            self.geometry_validity_enabled
        ):
            raise ValueError(
                "decoupled_candidate_heads requires geometry validity supervision"
            )
        if bool(self.candidate_view_marginalization_enabled) and not bool(
            self.decoupled_candidate_heads
        ):
            raise ValueError(
                "candidate view marginalization requires decoupled candidate heads"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateMapletBatch:
    query_features: torch.Tensor
    query_mask: torch.Tensor
    support_features: torch.Tensor
    support_mask: torch.Tensor
    static_features: torch.Tensor
    target_track_indices: torch.Tensor | None
    candidate_labels: torch.Tensor | None
    edge_indices: torch.Tensor
    anchor_residuals_px: torch.Tensor | None = None
    candidate_visible: torch.Tensor | None = None

    def validate(self) -> None:
        query = self.query_features
        support = self.support_features
        if query.ndim != 3 or support.ndim != 3:
            raise ValueError("query and support features must have shape (B, N, C)")
        batch_size = int(query.shape[0])
        if int(support.shape[0]) != batch_size:
            raise ValueError("query and support batch sizes differ")
        if self.query_mask.shape != query.shape[:2] or self.support_mask.shape != support.shape[:2]:
            raise ValueError("node masks must match feature prefixes")
        if self.static_features.ndim != 2 or int(self.static_features.shape[0]) != batch_size:
            raise ValueError("static_features must have shape (B, C)")
        if (self.target_track_indices is None) != (self.candidate_labels is None):
            raise ValueError(
                "assignment targets and candidate labels must be provided together"
            )
        if self.target_track_indices is not None and self.candidate_labels is not None:
            if self.target_track_indices.shape != self.query_mask.shape:
                raise ValueError("target_track_indices must match query nodes")
            if self.candidate_labels.reshape(-1).shape[0] != batch_size:
                raise ValueError("candidate_labels must contain one value per episode")
        if self.edge_indices.reshape(-1).shape[0] != batch_size:
            raise ValueError("edge_indices must contain one value per episode")
        if (self.anchor_residuals_px is None) != (self.candidate_visible is None):
            raise ValueError(
                "anchor residual and candidate visibility supervision must be provided together"
            )
        if self.anchor_residuals_px is not None and self.candidate_visible is not None:
            residuals = self.anchor_residuals_px.reshape(-1)
            visible = self.candidate_visible.reshape(-1).bool()
            if residuals.shape[0] != batch_size or visible.shape[0] != batch_size:
                raise ValueError("geometry supervision must contain one value per episode")
            if torch.any(torch.isnan(residuals)) or torch.any(residuals < 0.0):
                raise ValueError(
                    "anchor residuals must be non-negative or positive infinity"
                )
            if not torch.equal(torch.isfinite(residuals), visible):
                raise ValueError("candidate visibility and anchor residuals disagree")
        if torch.any(torch.sum(self.query_mask.bool(), dim=1) <= 0):
            raise ValueError("every episode requires at least one query node")
        if torch.any(torch.sum(self.support_mask.bool(), dim=1) <= 0):
            raise ValueError("every episode requires at least one support node")
        if not torch.all(self.query_mask[:, 0].bool()) or not torch.all(self.support_mask[:, 0].bool()):
            raise ValueError("node zero must be the candidate anchor on both sides")
        if self.target_track_indices is not None:
            for batch_index in range(batch_size):
                query_count = int(torch.sum(self.query_mask[batch_index].bool()).item())
                support_count = int(torch.sum(self.support_mask[batch_index].bool()).item())
                targets = self.target_track_indices[batch_index, :query_count]
                if torch.any(targets < 0) or torch.any(targets > support_count):
                    raise ValueError(
                        "valid query targets must reference a support node or its dustbin"
                    )
            if torch.any(self.target_track_indices[~self.query_mask.bool()] != -1):
                raise ValueError("padded query targets must be -1")

    def to(self, device: torch.device | str) -> "CandidateMapletBatch":
        return CandidateMapletBatch(
            query_features=self.query_features.to(device),
            query_mask=self.query_mask.to(device),
            support_features=self.support_features.to(device),
            support_mask=self.support_mask.to(device),
            static_features=self.static_features.to(device),
            target_track_indices=(
                None
                if self.target_track_indices is None
                else self.target_track_indices.to(device)
            ),
            candidate_labels=(
                None
                if self.candidate_labels is None
                else self.candidate_labels.to(device)
            ),
            edge_indices=self.edge_indices.to(device),
            anchor_residuals_px=(
                None
                if self.anchor_residuals_px is None
                else self.anchor_residuals_px.to(device)
            ),
            candidate_visible=(
                None if self.candidate_visible is None else self.candidate_visible.to(device)
            ),
        )


def log_optimal_transport_batched(
    scores: torch.Tensor,
    query_dustbin_scores: torch.Tensor,
    support_dustbin_scores: torch.Tensor,
    corner_score: torch.Tensor,
    query_mask: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    iterations: int,
) -> torch.Tensor:
    """Masked batched form of rectangular SuperGlue optimal transport."""

    if scores.ndim != 3:
        raise ValueError("scores must have shape (B, Nq, Nt)")
    batch_size, query_size, support_size = scores.shape
    if query_dustbin_scores.shape != (batch_size, query_size):
        raise ValueError("query dustbin scores have an incompatible shape")
    if support_dustbin_scores.shape != (batch_size, support_size):
        raise ValueError("support dustbin scores have an incompatible shape")
    if query_mask.shape != (batch_size, query_size) or support_mask.shape != (
        batch_size,
        support_size,
    ):
        raise ValueError("transport masks have incompatible shapes")
    query_valid = query_mask.bool()
    support_valid = support_mask.bool()
    query_counts = torch.sum(query_valid, dim=1).to(dtype=scores.dtype)
    support_counts = torch.sum(support_valid, dim=1).to(dtype=scores.dtype)
    if torch.any(query_counts <= 0) or torch.any(support_counts <= 0):
        raise ValueError("every transport sample requires non-empty node sets")
    negative = torch.tensor(-1e9, dtype=scores.dtype, device=scores.device)
    pair_mask = query_valid.unsqueeze(2) & support_valid.unsqueeze(1)
    pair_scores = scores.masked_fill(~pair_mask, negative)
    query_bin = query_dustbin_scores.masked_fill(~query_valid, negative).unsqueeze(2)
    support_bin = support_dustbin_scores.masked_fill(~support_valid, negative).unsqueeze(1)
    corner = corner_score.reshape(1, 1, 1).expand(batch_size, 1, 1)
    couplings = torch.cat(
        [
            torch.cat([pair_scores, query_bin], dim=2),
            torch.cat([support_bin, corner], dim=2),
        ],
        dim=1,
    )
    total = query_counts + support_counts
    norm = -torch.log(total)
    log_mu = torch.full(
        (batch_size, query_size + 1),
        float(negative.item()),
        dtype=scores.dtype,
        device=scores.device,
    )
    log_nu = torch.full(
        (batch_size, support_size + 1),
        float(negative.item()),
        dtype=scores.dtype,
        device=scores.device,
    )
    log_mu[:, :query_size] = torch.where(query_valid, norm[:, None], negative)
    log_mu[:, query_size] = torch.log(support_counts) + norm
    log_nu[:, :support_size] = torch.where(support_valid, norm[:, None], negative)
    log_nu[:, support_size] = torch.log(query_counts) + norm
    mu_valid = torch.cat(
        [query_valid, torch.ones((batch_size, 1), dtype=torch.bool, device=scores.device)], dim=1
    )
    nu_valid = torch.cat(
        [support_valid, torch.ones((batch_size, 1), dtype=torch.bool, device=scores.device)], dim=1
    )
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(int(iterations)):
        row_lse = torch.logsumexp(couplings + v.unsqueeze(1), dim=2)
        u = torch.where(mu_valid, log_mu - row_lse, torch.zeros_like(log_mu))
        column_lse = torch.logsumexp(couplings + u.unsqueeze(2), dim=1)
        v = torch.where(nu_valid, log_nu - column_lse, torch.zeros_like(log_nu))
    return couplings + u.unsqueeze(2) + v.unsqueeze(1) - norm[:, None, None]


class _CrossContextBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_self = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.support_self = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.query_cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.support_cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.query_norm1 = nn.LayerNorm(dim)
        self.support_norm1 = nn.LayerNorm(dim)
        self.query_norm2 = nn.LayerNorm(dim)
        self.support_norm2 = nn.LayerNorm(dim)
        self.query_ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.support_ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.query_norm3 = nn.LayerNorm(dim)
        self.support_norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_mask: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_padding = ~query_mask.bool()
        support_padding = ~support_mask.bool()
        query_self, _ = self.query_self(query, query, query, key_padding_mask=query_padding, need_weights=False)
        support_self, _ = self.support_self(
            support, support, support, key_padding_mask=support_padding, need_weights=False
        )
        query = self.query_norm1(query + query_self)
        support = self.support_norm1(support + support_self)
        query_cross, _ = self.query_cross(
            query, support, support, key_padding_mask=support_padding, need_weights=False
        )
        support_cross, _ = self.support_cross(
            support, query, query, key_padding_mask=query_padding, need_weights=False
        )
        query = self.query_norm2(query + query_cross)
        support = self.support_norm2(support + support_cross)
        query = self.query_norm3(query + self.query_ffn(query))
        support = self.support_norm3(support + self.support_ffn(support))
        query = query * query_mask.unsqueeze(2).to(dtype=query.dtype)
        support = support * support_mask.unsqueeze(2).to(dtype=support.dtype)
        return query, support


class _CandidateSetBlock(nn.Module):
    """Contextualize mutually exclusive candidates from one global top-L row."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            values,
            values,
            values,
            key_padding_mask=~mask.bool(),
            need_weights=False,
        )
        values = self.norm1(values + attended)
        values = self.norm2(values + self.ffn(values))
        return values * mask.unsqueeze(2).to(dtype=values.dtype)


class CandidateMapletMatcher(nn.Module):
    """Resolve one global candidate against a real support-view maplet."""

    def __init__(self, config: CandidateMapletMatcherConfig) -> None:
        super().__init__()
        self.config = config
        dim = int(config.model_dim)
        static_mean = (
            torch.zeros((int(config.static_input_dim),), dtype=torch.float32)
            if config.static_feature_mean is None
            else torch.tensor(config.static_feature_mean, dtype=torch.float32)
        )
        static_scale = (
            torch.ones((int(config.static_input_dim),), dtype=torch.float32)
            if config.static_feature_scale is None
            else torch.tensor(config.static_feature_scale, dtype=torch.float32)
        )
        self.register_buffer("static_feature_mean", static_mean, persistent=False)
        self.register_buffer("static_feature_scale", static_scale, persistent=False)

        def encoder(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(dim, dim),
            )

        self.query_encoder = encoder(int(config.query_input_dim))
        self.support_encoder = encoder(int(config.support_input_dim))
        self.context_blocks = nn.ModuleList(
            [
                _CrossContextBlock(dim, int(config.num_heads), float(config.dropout))
                for _ in range(int(config.layers))
            ]
        )
        self.query_pair = nn.Linear(dim, dim)
        self.support_pair = nn.Linear(dim, dim)
        self.query_dustbin = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.support_dustbin = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.corner_score = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.prior_log_scale = nn.Parameter(
            torch.tensor(float(math.log(config.descriptor_prior_scale)), dtype=torch.float32)
        )
        self.prior_center = nn.Parameter(torch.tensor(float(config.descriptor_prior_center), dtype=torch.float32))
        candidate_dim = dim * 4 + int(config.static_input_dim)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
        )
        self.raw_candidate_head = nn.Linear(dim, 1)
        self.support_view_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.candidate_set_blocks = nn.ModuleList(
            [
                _CandidateSetBlock(dim, int(config.num_heads), float(config.dropout))
                for _ in range(int(config.candidate_set_layers))
            ]
        )
        self.set_candidate_head = nn.Linear(dim, 1)
        if bool(config.candidate_view_marginalization_enabled):
            self.candidate_view_context: nn.Linear | None = nn.Linear(dim, dim)
            self.candidate_view_norm: nn.LayerNorm | None = nn.LayerNorm(dim)
            self.candidate_view_head: nn.Linear | None = nn.Linear(dim, 1)
            nn.init.zeros_(self.candidate_view_head.weight)
            nn.init.zeros_(self.candidate_view_head.bias)
        else:
            self.candidate_view_context = None
            self.candidate_view_norm = None
            self.candidate_view_head = None
        self.candidate_prior_log_scale = nn.Parameter(
            torch.tensor(float(math.log(config.candidate_prior_scale)), dtype=torch.float32)
        )
        self.set_dustbin_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        if bool(config.geometry_validity_enabled):
            self.geometry_logit_gap_head: nn.Linear | None = nn.Linear(dim, 2)
            self.geometry_base_head: nn.Linear | None = (
                nn.Linear(dim, 1) if bool(config.decoupled_candidate_heads) else None
            )
            self.candidate_visibility_head: nn.Linear | None = nn.Linear(dim, 1)
            self.geometry_candidate_set_blocks = nn.ModuleList(
                [
                    _CandidateSetBlock(dim, int(config.num_heads), float(config.dropout))
                    for _ in range(int(config.candidate_set_layers))
                ]
                if bool(config.decoupled_candidate_heads)
                else []
            )
            self.visibility_candidate_set_blocks = nn.ModuleList(
                [
                    _CandidateSetBlock(dim, int(config.num_heads), float(config.dropout))
                    for _ in range(int(config.candidate_set_layers))
                ]
                if bool(config.decoupled_candidate_heads)
                else []
            )
            nn.init.zeros_(self.geometry_logit_gap_head.weight)
            nn.init.zeros_(self.geometry_logit_gap_head.bias)
            if self.geometry_base_head is not None:
                nn.init.zeros_(self.geometry_base_head.weight)
                nn.init.zeros_(self.geometry_base_head.bias)
            nn.init.zeros_(self.candidate_visibility_head.weight)
            nn.init.zeros_(self.candidate_visibility_head.bias)
        else:
            self.geometry_logit_gap_head = None
            self.geometry_base_head = None
            self.candidate_visibility_head = None
            self.geometry_candidate_set_blocks = nn.ModuleList()
            self.visibility_candidate_set_blocks = nn.ModuleList()
        if bool(config.rescue_policy_enabled):
            self.rescue_candidate_head: nn.Sequential | None = nn.Sequential(
                nn.Linear(dim * 4, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, 1),
            )
            self.rescue_keep_head: nn.Sequential | None = nn.Sequential(
                nn.Linear(dim * 3, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, 1),
            )
        else:
            self.rescue_candidate_head = None
            self.rescue_keep_head = None
        nn.init.zeros_(self.raw_candidate_head.weight)
        nn.init.zeros_(self.raw_candidate_head.bias)
        nn.init.zeros_(self.set_candidate_head.weight)
        nn.init.zeros_(self.set_candidate_head.bias)

    def aggregate_candidate_views(
        self,
        candidate_embeddings: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Learn a posterior over real support views for each candidate."""

        if candidate_embeddings.ndim != 3:
            raise ValueError("candidate_embeddings must have shape (B, V, C)")
        if view_mask is None:
            valid = torch.ones(
                candidate_embeddings.shape[:2],
                dtype=torch.bool,
                device=candidate_embeddings.device,
            )
        else:
            valid = view_mask.bool()
            if valid.shape != candidate_embeddings.shape[:2]:
                raise ValueError("view_mask must match candidate view dimensions")
        if torch.any(torch.sum(valid, dim=1) <= 0):
            raise ValueError("every candidate requires at least one support view")
        logits = self.support_view_head(candidate_embeddings)[:, :, 0]
        logits = logits.masked_fill(~valid, -1e4)
        weights = torch.softmax(logits, dim=1)
        aggregated = torch.sum(candidate_embeddings * weights.unsqueeze(2), dim=1)
        return aggregated, weights

    def resolve_candidate_sets(
        self,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        candidate_prior_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Resolve a complete top-L candidate set and predict a learned no-match bin."""

        if candidate_embeddings.ndim != 3:
            raise ValueError("candidate_embeddings must have shape (G, L, C)")
        if candidate_mask is None:
            valid = torch.ones(
                candidate_embeddings.shape[:2],
                dtype=torch.bool,
                device=candidate_embeddings.device,
            )
        else:
            valid = candidate_mask.bool()
            if valid.shape != candidate_embeddings.shape[:2]:
                raise ValueError("candidate_mask must match candidate set dimensions")
        if torch.any(torch.sum(valid, dim=1) <= 0):
            raise ValueError("every candidate set requires at least one valid candidate")
        contextual = candidate_embeddings
        for block in self.candidate_set_blocks:
            contextual = block(contextual, valid)
        logits = self.set_candidate_head(contextual)[:, :, 0]
        if candidate_prior_scores is not None:
            prior = candidate_prior_scores.to(dtype=logits.dtype)
            if prior.shape != logits.shape:
                raise ValueError("candidate_prior_scores must match candidate set dimensions")
            weights = valid.to(dtype=prior.dtype)
            prior_center = torch.sum(prior * weights, dim=1, keepdim=True) / torch.clamp(
                torch.sum(weights, dim=1, keepdim=True), min=1.0
            )
            prior_scale = torch.clamp(
                torch.exp(self.candidate_prior_log_scale), min=0.1, max=100.0
            )
            logits = logits + prior_scale * (prior - prior_center)
        logits = logits.masked_fill(~valid, -1e4)
        weights = valid.to(dtype=contextual.dtype)
        mean_pool = torch.sum(contextual * weights.unsqueeze(2), dim=1) / torch.clamp(
            torch.sum(weights, dim=1, keepdim=True), min=1.0
        )
        max_pool = contextual.masked_fill(~valid.unsqueeze(2), -1e4).amax(dim=1)
        dustbin_logits = self.set_dustbin_head(
            torch.cat([mean_pool, max_pool], dim=1)
        )[:, 0]
        output = {
            "candidate_logits": logits,
            "dustbin_logits": dustbin_logits,
            "candidate_embeddings": contextual,
        }
        if self.rescue_candidate_head is not None:
            if self.rescue_keep_head is None or candidate_prior_scores is None:
                raise RuntimeError(
                    "rescue policy requires both heads and candidate prior scores"
                )
            baseline_indices = torch.argmax(
                candidate_prior_scores.masked_fill(~valid, -1e4), dim=1
            )
            baseline = contextual[
                torch.arange(len(contextual), device=contextual.device),
                baseline_indices,
            ]
            baseline_expanded = baseline.unsqueeze(1).expand_as(contextual)
            rescue_input = torch.cat(
                [
                    contextual,
                    baseline_expanded,
                    contextual - baseline_expanded,
                    contextual * baseline_expanded,
                ],
                dim=2,
            )
            rescue_logits = self.rescue_candidate_head(rescue_input)[:, :, 0]
            rescue_logits = rescue_logits.masked_fill(~valid, -1e4)
            keep_logits = self.rescue_keep_head(
                torch.cat([baseline, mean_pool, max_pool], dim=1)
            )[:, 0]
            output["rescue_candidate_logits"] = rescue_logits
            output["rescue_keep_logits"] = keep_logits
            output["baseline_candidate_indices"] = baseline_indices
        if self.geometry_logit_gap_head is not None:
            if self.candidate_visibility_head is None:
                raise RuntimeError("geometry validity heads are only partially configured")
            geometry_contextual = candidate_embeddings
            visibility_contextual = candidate_embeddings
            for block in self.geometry_candidate_set_blocks:
                geometry_contextual = block(geometry_contextual, valid)
            for block in self.visibility_candidate_set_blocks:
                visibility_contextual = block(visibility_contextual, valid)
            gaps = F.softplus(self.geometry_logit_gap_head(geometry_contextual))
            geometry_middle = (
                logits
                if self.geometry_base_head is None
                else self.geometry_base_head(geometry_contextual)[:, :, 0].masked_fill(
                    ~valid, -1e4
                )
            )
            geometry_logits = torch.stack(
                [
                    geometry_middle - gaps[:, :, 0],
                    geometry_middle,
                    geometry_middle + gaps[:, :, 1],
                ],
                dim=2,
            )
            visibility_logits = self.candidate_visibility_head(visibility_contextual)[:, :, 0]
            visibility_logits = visibility_logits.masked_fill(~valid, -1e4)
            output["geometry_validity_logits"] = geometry_logits
            output["candidate_visibility_logits"] = visibility_logits
        return output

    def resolve_candidate_view_sets(
        self,
        candidate_view_embeddings: torch.Tensor,
        support_view_probabilities: torch.Tensor,
        *,
        support_view_mask: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        candidate_prior_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Resolve top-L identity while marginalizing rather than averaging views."""

        if self.candidate_view_head is None:
            raise RuntimeError("candidate view marginalization is not enabled")
        if self.candidate_view_context is None or self.candidate_view_norm is None:
            raise RuntimeError("candidate view marginalization modules are incomplete")
        if candidate_view_embeddings.ndim != 4:
            raise ValueError(
                "candidate_view_embeddings must have shape (G, L, V, C)"
            )
        group_count, candidate_count, view_count, _ = candidate_view_embeddings.shape
        if support_view_probabilities.shape != (group_count, candidate_count, view_count):
            raise ValueError("support-view probabilities are not aligned")
        view_valid = (
            torch.ones_like(support_view_probabilities, dtype=torch.bool)
            if support_view_mask is None
            else support_view_mask.bool()
        )
        if view_valid.shape != support_view_probabilities.shape:
            raise ValueError("support-view mask is not aligned")
        if torch.any(torch.sum(view_valid, dim=2) <= 0):
            raise ValueError("every candidate requires at least one support view")
        view_weights = torch.where(
            view_valid,
            support_view_probabilities.clamp_min(0.0),
            torch.zeros_like(support_view_probabilities),
        )
        view_weights = view_weights / view_weights.sum(dim=2, keepdim=True).clamp_min(
            1e-8
        )
        marginal = torch.sum(
            candidate_view_embeddings * view_weights.unsqueeze(3), dim=2
        )
        output = self.resolve_candidate_sets(
            marginal,
            candidate_mask=candidate_mask,
            candidate_prior_scores=candidate_prior_scores,
        )
        contextual = output["candidate_embeddings"]
        view_contextual = self.candidate_view_norm(
            candidate_view_embeddings
            + self.candidate_view_context(contextual).unsqueeze(2)
        )
        view_residual = self.candidate_view_head(view_contextual)[:, :, :, 0]
        view_residual = view_residual.masked_fill(~view_valid, -1e4)
        candidate_residual = self.set_candidate_head(contextual)[:, :, 0] + torch.logsumexp(
            torch.log(view_weights.clamp_min(1e-8)) + view_residual, dim=2
        )
        valid = (
            torch.ones(
                (group_count, candidate_count),
                dtype=torch.bool,
                device=candidate_view_embeddings.device,
            )
            if candidate_mask is None
            else candidate_mask.bool()
        )
        if candidate_prior_scores is not None:
            prior = candidate_prior_scores.to(dtype=candidate_residual.dtype)
            if prior.shape != candidate_residual.shape:
                raise ValueError("candidate prior scores are not aligned")
            prior_weights = valid.to(dtype=prior.dtype)
            prior_center = torch.sum(
                prior * prior_weights, dim=1, keepdim=True
            ) / torch.clamp(torch.sum(prior_weights, dim=1, keepdim=True), min=1.0)
            prior_scale = torch.clamp(
                torch.exp(self.candidate_prior_log_scale), min=0.1, max=100.0
            )
            candidate_residual = candidate_residual + prior_scale * (
                prior - prior_center
            )
        output["candidate_logits"] = candidate_residual.masked_fill(~valid, -1e4)
        output["candidate_view_logits"] = view_residual
        output["support_view_probabilities"] = view_weights
        return output

    def forward(self, batch: CandidateMapletBatch) -> dict[str, object]:
        batch.validate()
        query_mask = batch.query_mask.bool()
        support_mask = batch.support_mask.bool()
        query = self.query_encoder(batch.query_features)
        support = self.support_encoder(batch.support_features)
        query = query * query_mask.unsqueeze(2).to(dtype=query.dtype)
        support = support * support_mask.unsqueeze(2).to(dtype=support.dtype)
        for block in self.context_blocks:
            query, support = block(query, support, query_mask, support_mask)

        query_pair = F.normalize(self.query_pair(query), p=2, dim=2)
        support_pair = F.normalize(self.support_pair(support), p=2, dim=2)
        contextual_scores = torch.einsum("bqd,btd->bqt", query_pair, support_pair)
        descriptor_dim = int(self.config.descriptor_dim)
        query_descriptors = F.normalize(batch.query_features[:, :, :descriptor_dim], p=2, dim=2)
        support_descriptors = F.normalize(batch.support_features[:, :, :descriptor_dim], p=2, dim=2)
        descriptor_scores = torch.einsum("bqd,btd->bqt", query_descriptors, support_descriptors)
        prior_scale = torch.clamp(torch.exp(self.prior_log_scale), min=0.1, max=50.0)
        pair_logits = contextual_scores + prior_scale * (descriptor_scores - self.prior_center)
        pair_mask = query_mask.unsqueeze(2) & support_mask.unsqueeze(1)
        pair_logits = pair_logits.masked_fill(~pair_mask, -1e4)
        query_dustbin = self.query_dustbin(query)[:, :, 0]
        support_dustbin = self.support_dustbin(support)[:, :, 0]

        transport_batch = log_optimal_transport_batched(
            pair_logits.float(),
            query_dustbin.float(),
            support_dustbin.float(),
            self.corner_score.float(),
            query_mask,
            support_mask,
            iterations=int(self.config.sinkhorn_iterations),
        )
        query_log_probabilities = []
        for batch_index in range(int(query.shape[0])):
            query_count = int(torch.sum(query_mask[batch_index]).item())
            support_count = int(torch.sum(support_mask[batch_index]).item())
            transport = torch.cat(
                [
                    transport_batch[batch_index, :query_count, :support_count],
                    transport_batch[
                        batch_index,
                        :query_count,
                        int(transport_batch.shape[2]) - 1 : int(transport_batch.shape[2]),
                    ],
                ],
                dim=1,
            )
            query_log_probabilities.append(F.log_softmax(transport, dim=1))

        query_weights = query_mask.to(dtype=query.dtype)
        support_weights = support_mask.to(dtype=support.dtype)
        query_pool = torch.sum(query * query_weights.unsqueeze(2), dim=1) / torch.clamp(
            torch.sum(query_weights, dim=1, keepdim=True), min=1.0
        )
        support_pool = torch.sum(support * support_weights.unsqueeze(2), dim=1) / torch.clamp(
            torch.sum(support_weights, dim=1, keepdim=True), min=1.0
        )
        normalized_static = (
            batch.static_features - self.static_feature_mean.unsqueeze(0)
        ) / self.static_feature_scale.unsqueeze(0)
        candidate_input = torch.cat(
            [query[:, 0], support[:, 0], query_pool, support_pool, normalized_static], dim=1
        )
        candidate_embeddings = self.candidate_encoder(candidate_input)
        candidate_logits = (
            self.raw_candidate_head(candidate_embeddings)[:, 0] + pair_logits[:, 0, 0]
        )
        return {
            "pair_logits": pair_logits,
            "pair_mask": pair_mask,
            "query_log_probabilities": query_log_probabilities,
            "log_transport": transport_batch,
            "candidate_logits": candidate_logits,
            "candidate_embeddings": candidate_embeddings,
            "descriptor_scores": descriptor_scores,
        }


def candidate_maplet_assignment_loss(
    output: Mapping[str, object],
    batch: CandidateMapletBatch,
    *,
    assignment_weight: float = 1.0,
    pair_weight: float = 0.25,
    candidate_weight: float = 1.0,
    matched_query_weight: float = 3.0,
    candidate_pos_weight: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch.validate()
    log_probabilities = output["query_log_probabilities"]
    if not isinstance(log_probabilities, list):
        raise TypeError("query_log_probabilities must be a list")
    pair_logits = output["pair_logits"]
    candidate_logits = output["candidate_logits"]
    if not isinstance(pair_logits, torch.Tensor) or not isinstance(candidate_logits, torch.Tensor):
        raise TypeError("matcher outputs contain invalid tensors")
    assignment_losses = []
    pair_targets = torch.zeros_like(pair_logits)
    matched_count = 0
    valid_query_count = 0
    for batch_index, query_log_probability in enumerate(log_probabilities):
        query_count = int(query_log_probability.shape[0])
        support_count = int(query_log_probability.shape[1] - 1)
        targets = batch.target_track_indices[batch_index, :query_count].long()
        nll = -query_log_probability[torch.arange(query_count, device=targets.device), targets]
        matched = targets < support_count
        weights = torch.where(
            matched,
            torch.full_like(nll, float(matched_query_weight)),
            torch.ones_like(nll),
        )
        assignment_losses.append(torch.sum(nll * weights) / torch.clamp(torch.sum(weights), min=1.0))
        matched_rows = torch.nonzero(matched, as_tuple=False).reshape(-1)
        if matched_rows.numel():
            pair_targets[batch_index, matched_rows, targets[matched_rows]] = 1.0
        matched_count += int(torch.sum(matched).item())
        valid_query_count += query_count
    assignment_loss = torch.mean(torch.stack(assignment_losses))
    pair_mask = output["pair_mask"]
    if not isinstance(pair_mask, torch.Tensor):
        raise TypeError("pair_mask must be a tensor")
    valid_pair_logits = pair_logits[pair_mask]
    valid_pair_targets = pair_targets[pair_mask]
    positive_pairs = torch.sum(valid_pair_targets)
    negative_pairs = torch.tensor(float(valid_pair_targets.numel()), device=positive_pairs.device) - positive_pairs
    pair_pos_weight = torch.clamp(negative_pairs / torch.clamp(positive_pairs, min=1.0), min=1.0, max=50.0)
    pair_loss = F.binary_cross_entropy_with_logits(
        valid_pair_logits,
        valid_pair_targets,
        pos_weight=pair_pos_weight,
    )
    candidate_loss = F.binary_cross_entropy_with_logits(
        candidate_logits,
        batch.candidate_labels.float(),
        pos_weight=torch.tensor(float(candidate_pos_weight), device=candidate_logits.device),
    )
    loss = (
        float(assignment_weight) * assignment_loss
        + float(pair_weight) * pair_loss
        + float(candidate_weight) * candidate_loss
    )
    with torch.no_grad():
        candidate_predictions = candidate_logits >= 0.0
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "assignment_loss": float(assignment_loss.detach().cpu().item()),
            "pair_loss": float(pair_loss.detach().cpu().item()),
            "candidate_loss": float(candidate_loss.detach().cpu().item()),
            "matched_query_rate": float(matched_count / max(valid_query_count, 1)),
            "candidate_positive_rate": float(torch.mean(batch.candidate_labels.float()).cpu().item()),
            "candidate_accuracy": float(
                torch.mean((candidate_predictions == batch.candidate_labels.bool()).float()).cpu().item()
            ),
        }
    return loss, metrics


def set_valued_candidate_loss(
    candidate_logits: torch.Tensor,
    dustbin_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-positive top-L classification with real no-match rows."""

    if candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must have shape (G, L)")
    group_count, candidate_count = candidate_logits.shape
    if dustbin_logits.shape != (group_count,):
        raise ValueError("dustbin_logits must contain one value per candidate group")
    positives = positive_mask.bool()
    if positives.shape != candidate_logits.shape:
        raise ValueError("positive_mask must match candidate_logits")
    if candidate_mask is None:
        valid = torch.ones_like(positives)
    else:
        valid = candidate_mask.bool()
        if valid.shape != candidate_logits.shape:
            raise ValueError("candidate_mask must match candidate_logits")
    if torch.any(torch.sum(valid, dim=1) <= 0):
        raise ValueError("every candidate group requires a valid candidate")
    if torch.any(positives & ~valid):
        raise ValueError("positive candidates must also be valid")
    negative = torch.tensor(-1e4, dtype=candidate_logits.dtype, device=candidate_logits.device)
    valid_logits = candidate_logits.masked_fill(~valid, negative)
    denominator = torch.logsumexp(
        torch.cat([valid_logits, dustbin_logits[:, None]], dim=1), dim=1
    )
    has_positive = torch.any(positives, dim=1)
    positive_numerator = torch.logsumexp(
        valid_logits.masked_fill(~positives, negative), dim=1
    )
    numerator = torch.where(has_positive, positive_numerator, dustbin_logits)
    loss = torch.mean(denominator - numerator)

    with torch.no_grad():
        classes = torch.argmax(
            torch.cat([valid_logits, dustbin_logits[:, None]], dim=1), dim=1
        )
        joint_probabilities = torch.softmax(
            torch.cat([valid_logits, dustbin_logits[:, None]], dim=1), dim=1
        )
        candidate_classes = torch.argmax(valid_logits, dim=1)
        dustbin_argmax = classes == int(candidate_count)
        predicts_dustbin = joint_probabilities[:, candidate_count] >= 0.5
        selected_labels = positives[
            torch.arange(group_count, device=classes.device),
            torch.clamp(classes, max=max(candidate_count - 1, 0)),
        ]
        correct = torch.where(
            has_positive, ~dustbin_argmax & selected_labels, dustbin_argmax
        )
        mappable_count = int(torch.sum(has_positive).item())
        no_match_count = int(torch.sum(~has_positive).item())
        candidate_rank_correct = positives[
            torch.arange(group_count, device=classes.device), candidate_classes
        ]
        metrics = {
            "candidate_set_loss": float(loss.detach().cpu().item()),
            "candidate_set_accuracy": float(torch.mean(correct.float()).cpu().item()),
            "candidate_set_top1_accuracy_mappable": (
                0.0
                if mappable_count == 0
                else float(torch.mean(correct[has_positive].float()).cpu().item())
            ),
            "candidate_rank_top1_accuracy_mappable": (
                0.0
                if mappable_count == 0
                else float(
                    torch.mean(candidate_rank_correct[has_positive].float()).cpu().item()
                )
            ),
            "candidate_set_no_match_accuracy": (
                0.0
                if no_match_count == 0
                else float(torch.mean(correct[~has_positive].float()).cpu().item())
            ),
            "candidate_set_mappable_rate": float(torch.mean(has_positive.float()).cpu().item()),
            "candidate_set_predicted_dustbin_rate": float(
                torch.mean(predicts_dustbin.float()).cpu().item()
            ),
            "candidate_set_dustbin_argmax_rate": float(
                torch.mean(dustbin_argmax.float()).cpu().item()
            ),
        }
    return loss, metrics


def geometry_validity_loss(
    geometry_logits: torch.Tensor,
    visibility_logits: torch.Tensor,
    residuals_px: torch.Tensor,
    candidate_visible: torch.Tensor,
    *,
    thresholds_px: tuple[float, ...] = (1.0, 2.0, 5.0),
    candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Proper binary losses for nested geometric events and query visibility."""

    if geometry_logits.ndim != 3:
        raise ValueError("geometry_logits must have shape (G, L, T)")
    group_count, candidate_count, threshold_count = geometry_logits.shape
    thresholds = tuple(float(value) for value in thresholds_px)
    if len(thresholds) != int(threshold_count):
        raise ValueError("geometry thresholds do not match geometry logits")
    if residuals_px.shape != (group_count, candidate_count):
        raise ValueError("residuals_px must match candidate group dimensions")
    if visibility_logits.shape != residuals_px.shape or candidate_visible.shape != residuals_px.shape:
        raise ValueError("visibility tensors must match candidate group dimensions")
    if candidate_mask is None:
        valid = torch.ones_like(residuals_px, dtype=torch.bool)
    else:
        valid = candidate_mask.bool()
        if valid.shape != residuals_px.shape:
            raise ValueError("candidate_mask must match candidate group dimensions")
    if torch.any(torch.sum(valid, dim=1) <= 0):
        raise ValueError("every geometry group requires a valid candidate")
    residuals = residuals_px.to(dtype=geometry_logits.dtype)
    visible = candidate_visible.bool()
    if torch.any(torch.isnan(residuals)) or torch.any(residuals < 0.0):
        raise ValueError("geometry residuals must be non-negative or positive infinity")
    if not torch.equal(torch.isfinite(residuals), visible):
        raise ValueError("geometry residuals and visibility targets disagree")
    threshold_tensor = torch.tensor(
        thresholds, dtype=geometry_logits.dtype, device=geometry_logits.device
    )
    targets = residuals.unsqueeze(2) <= threshold_tensor.reshape(1, 1, -1)
    valid_geometry = valid.unsqueeze(2).expand_as(targets)
    geometry_loss = F.binary_cross_entropy_with_logits(
        geometry_logits[valid_geometry], targets[valid_geometry].to(dtype=geometry_logits.dtype)
    )
    visibility_loss = F.binary_cross_entropy_with_logits(
        visibility_logits[valid], visible[valid].to(dtype=visibility_logits.dtype)
    )

    with torch.no_grad():
        probabilities = torch.sigmoid(geometry_logits)
        metrics = {
            "geometry_validity_loss": float(geometry_loss.detach().cpu().item()),
            "candidate_visibility_loss": float(visibility_loss.detach().cpu().item()),
            "candidate_visibility_rate": float(torch.mean(visible[valid].float()).cpu().item()),
            "candidate_visibility_accuracy": float(
                torch.mean(((visibility_logits[valid] >= 0.0) == visible[valid]).float())
                .cpu()
                .item()
            ),
            "geometry_monotonic_violation_rate": float(
                torch.mean(
                    (
                        (probabilities[:, :, 0] > probabilities[:, :, 1])
                        | (probabilities[:, :, 1] > probabilities[:, :, 2])
                    )[valid].float()
                )
                .cpu()
                .item()
            ),
        }
        for threshold_index, threshold in enumerate(thresholds):
            tag = f"p{int(round(threshold)):02d}px"
            threshold_targets = targets[:, :, threshold_index]
            threshold_probabilities = probabilities[:, :, threshold_index]
            metrics[f"geometry_{tag}_positive_rate"] = float(
                torch.mean(threshold_targets[valid].float()).cpu().item()
            )
            metrics[f"geometry_{tag}_brier"] = float(
                torch.mean(
                    (
                        threshold_probabilities[valid]
                        - threshold_targets[valid].to(dtype=threshold_probabilities.dtype)
                    )
                    ** 2
                )
                .cpu()
                .item()
            )
            masked_probabilities = threshold_probabilities.masked_fill(~valid, -1.0)
            selected = torch.argmax(masked_probabilities, dim=1)
            has_positive = torch.any(threshold_targets & valid, dim=1)
            if torch.any(has_positive):
                selected_positive = threshold_targets[
                    torch.arange(group_count, device=selected.device), selected
                ]
                metrics[f"geometry_{tag}_rank_top1_accuracy_mappable"] = float(
                    torch.mean(selected_positive[has_positive].float()).cpu().item()
                )
            else:
                metrics[f"geometry_{tag}_rank_top1_accuracy_mappable"] = 0.0
    return geometry_loss, visibility_loss, metrics


def rescue_policy_targets(
    residuals_px: torch.Tensor,
    baseline_candidate_indices: torch.Tensor,
    *,
    candidate_threshold_px: float,
    baseline_invalid_threshold_px: float,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mark candidates that can rescue a geometrically invalid coarse baseline."""

    if residuals_px.ndim != 2:
        raise ValueError("residuals_px must have shape (G, L)")
    group_count, candidate_count = residuals_px.shape
    baseline_indices = baseline_candidate_indices.long().reshape(-1)
    if baseline_indices.shape[0] != group_count:
        raise ValueError("baseline candidate indices must contain one value per group")
    if torch.any(baseline_indices < 0) or torch.any(baseline_indices >= candidate_count):
        raise ValueError("baseline candidate index is outside the candidate set")
    candidate_threshold = float(candidate_threshold_px)
    baseline_threshold = float(baseline_invalid_threshold_px)
    if (
        not math.isfinite(candidate_threshold)
        or not math.isfinite(baseline_threshold)
        or candidate_threshold <= 0.0
        or baseline_threshold <= 0.0
        or candidate_threshold > baseline_threshold
    ):
        raise ValueError("invalid rescue policy thresholds")
    residuals = residuals_px
    if torch.any(torch.isnan(residuals)) or torch.any(residuals < 0.0):
        raise ValueError("residuals must be non-negative or positive infinity")
    valid = (
        torch.ones_like(residuals, dtype=torch.bool)
        if candidate_mask is None
        else candidate_mask.bool()
    )
    if valid.shape != residuals.shape:
        raise ValueError("candidate_mask must match residuals_px")
    if torch.any(torch.sum(valid, dim=1) <= 0):
        raise ValueError("every rescue group requires a valid candidate")
    rows = torch.arange(group_count, device=residuals.device)
    if torch.any(~valid[rows, baseline_indices]):
        raise ValueError("baseline candidate must be valid")
    baseline_residuals = residuals[rows, baseline_indices]
    baseline_invalid = baseline_residuals > baseline_threshold
    targets = (
        valid
        & baseline_invalid.unsqueeze(1)
        & (residuals <= candidate_threshold)
    )
    targets[rows, baseline_indices] = False
    return targets


def rescue_policy_loss(
    rescue_candidate_logits: torch.Tensor,
    rescue_keep_logits: torch.Tensor,
    residuals_px: torch.Tensor,
    baseline_candidate_indices: torch.Tensor,
    *,
    candidate_threshold_px: float,
    baseline_invalid_threshold_px: float,
    candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Set-valued rescue action loss with KEEP represented by the dustbin class."""

    targets = rescue_policy_targets(
        residuals_px,
        baseline_candidate_indices,
        candidate_threshold_px=float(candidate_threshold_px),
        baseline_invalid_threshold_px=float(baseline_invalid_threshold_px),
        candidate_mask=candidate_mask,
    )
    loss, set_metrics = set_valued_candidate_loss(
        rescue_candidate_logits,
        rescue_keep_logits,
        targets,
        candidate_mask=candidate_mask,
    )
    metrics = {
        name.replace("candidate_set_", "rescue_").replace(
            "candidate_rank_", "rescue_candidate_rank_"
        ): value
        for name, value in set_metrics.items()
    }
    with torch.no_grad():
        metrics["rescue_policy_loss"] = float(loss.detach().cpu().item())
        metrics["rescue_candidate_positive_rate"] = float(
            torch.mean(targets.float()).cpu().item()
        )
        metrics["rescue_group_positive_rate"] = float(
            torch.mean(torch.any(targets, dim=1).float()).cpu().item()
        )
    return loss, metrics


def candidate_maplet_group_loss(
    model: CandidateMapletMatcher,
    outputs_by_view: list[Mapping[str, object]],
    batches_by_view: list[CandidateMapletBatch],
    *,
    candidate_group_size: int,
    assignment_weight: float = 1.0,
    pair_weight: float = 0.25,
    candidate_aux_weight: float = 0.25,
    candidate_set_weight: float = 1.0,
    matched_query_weight: float = 3.0,
    candidate_pos_weight: float = 3.0,
    geometry_validity_weight: float = 0.0,
    candidate_visibility_weight: float = 0.0,
    rescue_policy_weight: float = 0.0,
    support_view_dropout: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train all support views and resolve each complete mutually exclusive top-L set."""

    if not outputs_by_view or len(outputs_by_view) != len(batches_by_view):
        raise ValueError("outputs_by_view and batches_by_view must be non-empty and aligned")
    group_size = int(candidate_group_size)
    if group_size < 2:
        raise ValueError("candidate_group_size must be at least two")
    reference_edges = batches_by_view[0].edge_indices.long()
    batch_size = int(reference_edges.numel())
    if batch_size % group_size != 0:
        raise ValueError("candidate episode batch does not contain complete top-L groups")
    edge_matrix = reference_edges.reshape(-1, group_size)
    group_ids = torch.div(edge_matrix[:, 0], group_size, rounding_mode="floor")
    expected = group_ids[:, None] * group_size + torch.arange(
        group_size, device=reference_edges.device
    )[None]
    if not torch.equal(edge_matrix, expected):
        raise ValueError("candidate episode batch is not ordered as complete top-L groups")

    episode_losses = []
    episode_metrics: list[dict[str, float]] = []
    embeddings = []
    for output, batch in zip(outputs_by_view, batches_by_view):
        if not torch.equal(batch.edge_indices.long(), reference_edges):
            raise ValueError("support-view batches contain different candidate edges")
        loss, metrics = candidate_maplet_assignment_loss(
            output,
            batch,
            assignment_weight=float(assignment_weight),
            pair_weight=float(pair_weight),
            candidate_weight=float(candidate_aux_weight),
            matched_query_weight=float(matched_query_weight),
            candidate_pos_weight=float(candidate_pos_weight),
        )
        candidate_embeddings = output.get("candidate_embeddings")
        if not isinstance(candidate_embeddings, torch.Tensor):
            raise TypeError("matcher output is missing candidate_embeddings")
        episode_losses.append(loss)
        episode_metrics.append(metrics)
        embeddings.append(candidate_embeddings)
    episode_loss = torch.mean(torch.stack(episode_losses))
    view_embeddings = torch.stack(embeddings, dim=1)
    dropout = float(support_view_dropout)
    if not 0.0 <= dropout < 1.0:
        raise ValueError("support_view_dropout must be in [0, 1)")
    view_mask = torch.ones(
        view_embeddings.shape[:2], dtype=torch.bool, device=view_embeddings.device
    )
    if model.training and dropout > 0.0 and view_embeddings.shape[1] > 1:
        view_mask = torch.rand(
            view_embeddings.shape[:2], device=view_embeddings.device
        ) >= dropout
        empty = ~torch.any(view_mask, dim=1)
        if torch.any(empty):
            replacement = torch.randint(
                int(view_embeddings.shape[1]),
                (int(torch.sum(empty).item()),),
                device=view_embeddings.device,
            )
            view_mask[torch.nonzero(empty, as_tuple=False)[:, 0], replacement] = True
    aggregated, view_weights = model.aggregate_candidate_views(
        view_embeddings, view_mask=view_mask
    )
    prior_scores = batches_by_view[0].static_features[
        :, int(model.config.candidate_prior_index)
    ].reshape(-1, group_size)
    if bool(model.config.candidate_view_marginalization_enabled):
        resolved = model.resolve_candidate_view_sets(
            view_embeddings.reshape(
                -1,
                group_size,
                int(view_embeddings.shape[1]),
                int(view_embeddings.shape[2]),
            ),
            view_weights.reshape(-1, group_size, int(view_weights.shape[1])),
            support_view_mask=view_mask.reshape(
                -1, group_size, int(view_mask.shape[1])
            ),
            candidate_prior_scores=prior_scores,
        )
    else:
        resolved = model.resolve_candidate_sets(
            aggregated.reshape(-1, group_size, int(aggregated.shape[1])),
            candidate_prior_scores=prior_scores,
        )
    positive_mask = batches_by_view[0].candidate_labels.bool().reshape(-1, group_size)
    set_loss, set_metrics = set_valued_candidate_loss(
        resolved["candidate_logits"], resolved["dustbin_logits"], positive_mask
    )
    total = episode_loss + float(candidate_set_weight) * set_loss
    geometry_metrics: dict[str, float] = {}
    rescue_metrics: dict[str, float] = {}
    residual_supervision_required = bool(
        float(geometry_validity_weight) > 0.0
        or float(candidate_visibility_weight) > 0.0
        or float(rescue_policy_weight) > 0.0
    )
    residuals_px: torch.Tensor | None = None
    visible: torch.Tensor | None = None
    if residual_supervision_required:
        reference_batch = batches_by_view[0]
        if reference_batch.anchor_residuals_px is None or reference_batch.candidate_visible is None:
            raise ValueError("geometry-aware losses require residual supervision")
        for batch in batches_by_view[1:]:
            if batch.anchor_residuals_px is None or batch.candidate_visible is None:
                raise ValueError("all support views require residual supervision")
            if not torch.equal(
                batch.anchor_residuals_px, reference_batch.anchor_residuals_px
            ) or not torch.equal(batch.candidate_visible, reference_batch.candidate_visible):
                raise ValueError("support views disagree on candidate geometry supervision")
        residuals_px = reference_batch.anchor_residuals_px.reshape(-1, group_size)
        visible = reference_batch.candidate_visible.reshape(-1, group_size)

    if float(geometry_validity_weight) > 0.0 or float(candidate_visibility_weight) > 0.0:
        geometry_logits = resolved.get("geometry_validity_logits")
        visibility_logits = resolved.get("candidate_visibility_logits")
        if not isinstance(geometry_logits, torch.Tensor) or not isinstance(
            visibility_logits, torch.Tensor
        ):
            raise ValueError("geometry validity loss requires enabled model heads")
        if residuals_px is None or visible is None:
            raise RuntimeError("geometry residual supervision was not initialized")
        expected_middle = residuals_px <= float(
            model.config.geometry_validity_thresholds_px[1]
        )
        if not torch.equal(expected_middle, positive_mask):
            raise ValueError(
                "middle geometry threshold must reproduce candidate positive labels"
            )
        geometry_loss, visibility_loss, geometry_metrics = geometry_validity_loss(
            geometry_logits,
            visibility_logits,
            residuals_px,
            visible,
            thresholds_px=tuple(model.config.geometry_validity_thresholds_px),
        )
        total = (
            total
            + float(geometry_validity_weight) * geometry_loss
            + float(candidate_visibility_weight) * visibility_loss
        )
    if float(rescue_policy_weight) > 0.0:
        rescue_candidate_logits = resolved.get("rescue_candidate_logits")
        rescue_keep_logits = resolved.get("rescue_keep_logits")
        baseline_indices = resolved.get("baseline_candidate_indices")
        if (
            not isinstance(rescue_candidate_logits, torch.Tensor)
            or not isinstance(rescue_keep_logits, torch.Tensor)
            or not isinstance(baseline_indices, torch.Tensor)
        ):
            raise ValueError("rescue policy loss requires enabled model heads")
        if residuals_px is None:
            raise RuntimeError("rescue residual supervision was not initialized")
        rescue_loss, rescue_metrics = rescue_policy_loss(
            rescue_candidate_logits,
            rescue_keep_logits,
            residuals_px,
            baseline_indices,
            candidate_threshold_px=float(model.config.rescue_candidate_threshold_px),
            baseline_invalid_threshold_px=float(
                model.config.rescue_baseline_invalid_threshold_px
            ),
        )
        total = total + float(rescue_policy_weight) * rescue_loss

    metrics = {
        name: float(np_mean([values[name] for values in episode_metrics]))
        for name in episode_metrics[0]
    }
    metrics.update(set_metrics)
    metrics.update(geometry_metrics)
    metrics.update(rescue_metrics)
    metrics["episode_loss"] = float(episode_loss.detach().cpu().item())
    metrics["loss"] = float(total.detach().cpu().item())
    with torch.no_grad():
        entropy = -torch.sum(
            view_weights * torch.log(torch.clamp(view_weights, min=1e-8)), dim=1
        )
        metrics["support_view_entropy"] = float(torch.mean(entropy).cpu().item())
        metrics["support_view_max_probability"] = float(
            torch.mean(torch.max(view_weights, dim=1).values).cpu().item()
        )
    return total, metrics


def np_mean(values: list[float]) -> float:
    """Small dependency-free mean helper for detached metric dictionaries."""

    return float(sum(float(value) for value in values) / max(len(values), 1))
