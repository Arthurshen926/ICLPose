"""Query-conditioned support selection and partial landmark assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LocalAssignmentMatcherConfig:
    query_input_dim: int
    track_input_dim: int
    support_input_dim: int
    edge_input_dim: int
    model_dim: int = 128
    num_heads: int = 4
    query_layers: int = 1
    track_layers: int = 1
    dropout: float = 0.1
    sinkhorn_iterations: int = 20
    edge_prior_feature_index: int = 0
    edge_prior_scale: float = 10.0
    edge_prior_center: float = 0.8
    query_dustbin_initial_bias: float = -0.3

    def __post_init__(self) -> None:
        for name in ("query_input_dim", "track_input_dim", "support_input_dim", "edge_input_dim", "model_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.num_heads) <= 0 or int(self.model_dim) % int(self.num_heads) != 0:
            raise ValueError("num_heads must divide model_dim")
        if int(self.query_layers) <= 0 or int(self.track_layers) <= 0:
            raise ValueError("transformer layer counts must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if int(self.sinkhorn_iterations) <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if not 0 <= int(self.edge_prior_feature_index) < int(self.edge_input_dim):
            raise ValueError("edge_prior_feature_index must reference an edge input column")
        if float(self.edge_prior_scale) <= 0.0:
            raise ValueError("edge_prior_scale must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LocalAssignmentEpisode:
    query_features: torch.Tensor
    track_features: torch.Tensor
    edge_query_indices: torch.Tensor
    edge_track_indices: torch.Tensor
    edge_features: torch.Tensor
    support_features: torch.Tensor
    support_mask: torch.Tensor
    target_track_indices: torch.Tensor
    query_rows: torch.Tensor
    candidate_columns: torch.Tensor

    def validate(self) -> None:
        query = self.query_features
        tracks = self.track_features
        edge_query = self.edge_query_indices.reshape(-1)
        edge_track = self.edge_track_indices.reshape(-1)
        edge_features = self.edge_features
        support = self.support_features
        support_mask = self.support_mask
        targets = self.target_track_indices.reshape(-1)
        if query.ndim != 2 or tracks.ndim != 2:
            raise ValueError("query_features and track_features must have shape (N, C)")
        edge_count = int(edge_query.numel())
        if edge_track.numel() != edge_count or edge_features.shape[0] != edge_count:
            raise ValueError("edge arrays must contain the same number of rows")
        if support.ndim != 3 or support.shape[0] != edge_count:
            raise ValueError("support_features must have shape (E, M, C)")
        if support_mask.shape != support.shape[:2]:
            raise ValueError("support_mask must have shape (E, M)")
        if targets.shape[0] != query.shape[0]:
            raise ValueError("target_track_indices must contain one target per query node")
        if torch.any(edge_query < 0) or torch.any(edge_query >= int(query.shape[0])):
            raise ValueError("edge query index out of range")
        if torch.any(edge_track < 0) or torch.any(edge_track >= int(tracks.shape[0])):
            raise ValueError("edge track index out of range")
        if torch.any(targets < 0) or torch.any(targets > int(tracks.shape[0])):
            raise ValueError("target track index out of range; N tracks denotes dustbin")
        if self.query_rows.reshape(-1).shape[0] != query.shape[0]:
            raise ValueError("query_rows must contain one global row per query node")
        if self.candidate_columns.reshape(-1).shape[0] != edge_count:
            raise ValueError("candidate_columns must contain one column per edge")

    def to(self, device: torch.device | str) -> "LocalAssignmentEpisode":
        return LocalAssignmentEpisode(
            query_features=self.query_features.to(device),
            track_features=self.track_features.to(device),
            edge_query_indices=self.edge_query_indices.to(device),
            edge_track_indices=self.edge_track_indices.to(device),
            edge_features=self.edge_features.to(device),
            support_features=self.support_features.to(device),
            support_mask=self.support_mask.to(device),
            target_track_indices=self.target_track_indices.to(device),
            query_rows=self.query_rows.to(device),
            candidate_columns=self.candidate_columns.to(device),
        )


def log_sinkhorn_iterations(
    scores: torch.Tensor,
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(int(iterations)):
        u = log_mu - torch.logsumexp(scores + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(scores + u.unsqueeze(1), dim=0)
    return scores + u.unsqueeze(1) + v.unsqueeze(0)


def log_optimal_transport(
    scores: torch.Tensor,
    query_dustbin_scores: torch.Tensor,
    track_dustbin_scores: torch.Tensor,
    corner_score: torch.Tensor,
    *,
    iterations: int,
) -> torch.Tensor:
    """SuperGlue-style rectangular optimal transport with explicit dustbins."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape (Nq, Nt)")
    query_count, track_count = int(scores.shape[0]), int(scores.shape[1])
    if query_count <= 0 or track_count <= 0:
        raise ValueError("optimal transport requires non-empty query and track sets")
    query_bin = query_dustbin_scores.reshape(query_count, 1)
    track_bin = track_dustbin_scores.reshape(1, track_count)
    corner = corner_score.reshape(1, 1)
    couplings = torch.cat(
        [
            torch.cat([scores, query_bin], dim=1),
            torch.cat([track_bin, corner], dim=1),
        ],
        dim=0,
    )
    total = torch.tensor(float(query_count + track_count), dtype=scores.dtype, device=scores.device)
    norm = -torch.log(total)
    log_mu = torch.cat(
        [
            norm.expand(query_count),
            (torch.log(torch.tensor(float(track_count), dtype=scores.dtype, device=scores.device)) + norm).reshape(1),
        ]
    )
    log_nu = torch.cat(
        [
            norm.expand(track_count),
            (torch.log(torch.tensor(float(query_count), dtype=scores.dtype, device=scores.device)) + norm).reshape(1),
        ]
    )
    return log_sinkhorn_iterations(couplings, log_mu, log_nu, int(iterations)) - norm


class LocalAssignmentMatcher(nn.Module):
    """Set-to-set matcher with learned support posterior and dustbin assignment."""

    def __init__(self, config: LocalAssignmentMatcherConfig) -> None:
        super().__init__()
        self.config = config
        dim = int(config.model_dim)

        def encoder(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(int(input_dim), dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(dim, dim),
            )

        self.query_encoder = encoder(int(config.query_input_dim))
        self.track_encoder = encoder(int(config.track_input_dim))
        query_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(config.num_heads),
            dim_feedforward=dim * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        track_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(config.num_heads),
            dim_feedforward=dim * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_context = nn.TransformerEncoder(query_layer, num_layers=int(config.query_layers))
        self.track_context = nn.TransformerEncoder(track_layer, num_layers=int(config.track_layers))
        self.query_to_track = nn.MultiheadAttention(
            dim,
            int(config.num_heads),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.track_to_query = nn.MultiheadAttention(
            dim,
            int(config.num_heads),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.query_cross_norm = nn.LayerNorm(dim)
        self.track_cross_norm = nn.LayerNorm(dim)
        self.support_key = nn.Linear(int(config.support_input_dim), dim)
        self.support_value = nn.Linear(int(config.support_input_dim), dim)
        self.support_query = nn.Linear(dim, dim)
        edge_dim = dim * 5 + int(config.edge_input_dim)
        self.edge_head = nn.Sequential(
            nn.Linear(edge_dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.edge_head[-1].weight)
        nn.init.zeros_(self.edge_head[-1].bias)
        self.edge_prior_log_scale = nn.Parameter(
            torch.tensor(float(np.log(config.edge_prior_scale)), dtype=torch.float32)
        )
        self.edge_prior_center = nn.Parameter(torch.tensor(float(config.edge_prior_center), dtype=torch.float32))
        no_match_input_dim = int(config.edge_input_dim) * 4
        self.no_match_head = nn.Linear(no_match_input_dim, 1)
        self.track_dustbin_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.no_match_head.weight)
        nn.init.constant_(self.no_match_head.bias, float(config.query_dustbin_initial_bias))
        nn.init.zeros_(self.track_dustbin_head[-1].weight)
        nn.init.zeros_(self.track_dustbin_head[-1].bias)
        self.corner_score = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, episode: LocalAssignmentEpisode) -> dict[str, torch.Tensor]:
        episode.validate()
        query = self.query_encoder(episode.query_features)
        tracks = self.track_encoder(episode.track_features)
        query = self.query_context(query.unsqueeze(0))[0]
        tracks = self.track_context(tracks.unsqueeze(0))[0]
        query_cross, _ = self.query_to_track(
            query.unsqueeze(0),
            tracks.unsqueeze(0),
            tracks.unsqueeze(0),
            need_weights=False,
        )
        track_cross, _ = self.track_to_query(
            tracks.unsqueeze(0),
            query.unsqueeze(0),
            query.unsqueeze(0),
            need_weights=False,
        )
        query = self.query_cross_norm(query + query_cross[0])
        tracks = self.track_cross_norm(tracks + track_cross[0])

        edge_query = query[episode.edge_query_indices.long()]
        edge_track = tracks[episode.edge_track_indices.long()]
        support_keys = self.support_key(episode.support_features)
        support_values = self.support_value(episode.support_features)
        support_logits = torch.sum(
            support_keys * self.support_query(edge_query).unsqueeze(1),
            dim=2,
        ) / float(self.config.model_dim) ** 0.5
        support_logits = support_logits.masked_fill(~episode.support_mask.bool(), -1e4)
        support_weights = torch.softmax(support_logits, dim=1)
        support_weights = support_weights * episode.support_mask.to(dtype=support_weights.dtype)
        support_weights = support_weights / torch.clamp(support_weights.sum(dim=1, keepdim=True), min=1e-8)
        support_context = torch.sum(support_values * support_weights.unsqueeze(2), dim=1)

        edge_input = torch.cat(
            [
                edge_query,
                edge_track,
                edge_query * edge_track,
                torch.abs(edge_query - edge_track),
                support_context,
                episode.edge_features,
            ],
            dim=1,
        )
        edge_residual = self.edge_head(edge_input)[:, 0]
        prior = episode.edge_features[:, int(self.config.edge_prior_feature_index)]
        prior_scale = torch.clamp(torch.exp(self.edge_prior_log_scale), min=1.0, max=100.0)
        edge_logits = edge_residual + prior_scale * (prior - self.edge_prior_center)
        score_matrix = torch.full(
            (int(query.shape[0]), int(tracks.shape[0])),
            -1e4,
            dtype=edge_logits.dtype,
            device=edge_logits.device,
        )
        score_matrix[episode.edge_query_indices.long(), episode.edge_track_indices.long()] = edge_logits
        evidence_rows = []
        for query_index in range(int(query.shape[0])):
            values = episode.edge_features[episode.edge_query_indices.long() == int(query_index)]
            if int(values.shape[0]) == 0:
                raise ValueError("every query node must contain at least one proposal edge")
            top_values = torch.topk(values, k=min(2, int(values.shape[0])), dim=0).values
            top1 = top_values[0]
            top2 = top_values[1] if int(top_values.shape[0]) > 1 else top_values[0]
            evidence_rows.append(
                torch.cat(
                    [
                        torch.max(values, dim=0).values,
                        torch.mean(values, dim=0),
                        torch.std(values, dim=0, unbiased=False),
                        top1 - top2,
                    ],
                    dim=0,
                )
            )
        no_match_evidence = torch.stack(evidence_rows, dim=0)
        query_dustbin = self.no_match_head(no_match_evidence)[:, 0]
        track_dustbin = self.track_dustbin_head(tracks)[:, 0]
        log_transport = log_optimal_transport(
            score_matrix,
            query_dustbin,
            track_dustbin,
            self.corner_score,
            iterations=int(self.config.sinkhorn_iterations),
        )
        query_log_probabilities = F.log_softmax(log_transport[: query.shape[0], :], dim=1)
        return {
            "edge_logits": edge_logits,
            "support_weights": support_weights,
            "score_matrix": score_matrix,
            "log_transport": log_transport,
            "query_log_probabilities": query_log_probabilities,
            "no_match_logits": query_dustbin,
            "no_match_evidence": no_match_evidence,
        }


def local_assignment_loss(
    output: Mapping[str, torch.Tensor],
    episode: LocalAssignmentEpisode,
    *,
    pair_loss_weight: float = 0.2,
    no_match_loss_weight: float = 1.0,
    positive_assignment_weight: float = 1.0,
    balance_no_match_classes: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_probabilities = output["query_log_probabilities"]
    targets = episode.target_track_indices.long()
    assignment_rows = F.nll_loss(
        log_probabilities,
        targets,
        reduction="none",
    )
    positive_nodes = targets != int(episode.track_features.shape[0])
    requested_positive_weight = float(positive_assignment_weight)
    if requested_positive_weight <= 0.0:
        positive_count_nodes = torch.sum(positive_nodes)
        null_count_nodes = int(targets.numel()) - positive_count_nodes
        requested_positive_weight = float(
            torch.clamp(
                null_count_nodes
                / torch.clamp(positive_count_nodes, min=1),
                min=1.0,
                max=50.0,
            ).detach().cpu().item()
        )
    assignment_weights = torch.where(
        positive_nodes,
        torch.full_like(
            assignment_rows,
            requested_positive_weight,
        ),
        torch.ones_like(assignment_rows),
    )
    assignment_loss = torch.sum(
        assignment_rows * assignment_weights
    ) / torch.clamp(torch.sum(assignment_weights), min=1.0)
    edge_targets = (
        episode.edge_track_indices.long()
        == targets[episode.edge_query_indices.long()]
    ).to(dtype=output["edge_logits"].dtype)
    positive_count = torch.sum(edge_targets)
    negative_count = torch.tensor(float(edge_targets.numel()), device=edge_targets.device) - positive_count
    pos_weight = torch.clamp(negative_count / torch.clamp(positive_count, min=1.0), min=1.0, max=50.0)
    pair_loss = F.binary_cross_entropy_with_logits(
        output["edge_logits"],
        edge_targets,
        pos_weight=pos_weight,
    )
    no_match_targets = (targets == int(episode.track_features.shape[0])).to(
        dtype=output["no_match_logits"].dtype
    )
    no_match_rows = F.binary_cross_entropy_with_logits(
        output["no_match_logits"],
        no_match_targets,
        reduction="none",
    )
    if bool(balance_no_match_classes):
        non_null_count = torch.sum(positive_nodes)
        null_count = int(targets.numel()) - non_null_count
        non_null_weight = torch.clamp(
            null_count / torch.clamp(non_null_count, min=1),
            min=1.0,
            max=50.0,
        ).to(dtype=no_match_rows.dtype)
        no_match_weights = torch.where(
            positive_nodes,
            non_null_weight,
            torch.ones_like(no_match_rows),
        )
        no_match_loss = torch.sum(
            no_match_rows * no_match_weights
        ) / torch.clamp(torch.sum(no_match_weights), min=1.0)
    else:
        no_match_loss = torch.mean(no_match_rows)
    loss = (
        assignment_loss
        + float(pair_loss_weight) * pair_loss
        + float(no_match_loss_weight) * no_match_loss
    )
    with torch.no_grad():
        predictions = torch.argmax(log_probabilities, dim=1)
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "assignment_loss": float(assignment_loss.detach().cpu().item()),
            "pair_loss": float(pair_loss.detach().cpu().item()),
            "no_match_loss": float(no_match_loss.detach().cpu().item()),
            "positive_assignment_weight": float(
                requested_positive_weight
            ),
            "assignment_accuracy": float(torch.mean((predictions == targets).float()).cpu().item()),
            "dustbin_target_rate": float(torch.mean((targets == int(episode.track_features.shape[0])).float()).cpu().item()),
            "positive_edge_count": float(positive_count.cpu().item()),
        }
    return loss, metrics
