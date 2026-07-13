"""Whole-image graph refinement for fixed per-token candidate evidence."""

from __future__ import annotations

import torch
from torch import nn


class WholeImageCandidateGraph(nn.Module):
    """Refine candidate/null logits using query and 3D-neighborhood context."""

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0 or int(model_dim) % int(heads) != 0:
            raise ValueError("invalid graph dimensions")
        self.input_encoder = nn.Sequential(
            nn.Linear(int(input_dim), int(model_dim)),
            nn.LayerNorm(int(model_dim)),
            nn.GELU(),
        )
        self.candidate_attention = nn.MultiheadAttention(
            int(model_dim), int(heads), dropout=float(dropout), batch_first=True
        )
        self.candidate_norm = nn.LayerNorm(int(model_dim))
        self.query_position = nn.Sequential(
            nn.Linear(2, int(model_dim)), nn.GELU(), nn.Linear(int(model_dim), int(model_dim))
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(model_dim),
            nhead=int(heads),
            dim_feedforward=int(model_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_graph = nn.TransformerEncoder(encoder_layer, num_layers=int(layers))
        self.query_to_candidate = nn.Linear(int(model_dim), int(model_dim))
        self.neighbor_to_candidate = nn.Linear(int(model_dim), int(model_dim))
        self.fusion_norm = nn.LayerNorm(int(model_dim))
        self.candidate_residual_head = nn.Linear(int(model_dim), 1)
        self.null_residual_head = nn.Sequential(
            nn.LayerNorm(int(model_dim)), nn.Linear(int(model_dim), 1)
        )
        self.prior_log_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.candidate_residual_head.weight)
        nn.init.zeros_(self.candidate_residual_head.bias)
        nn.init.zeros_(self.null_residual_head[-1].weight)
        nn.init.zeros_(self.null_residual_head[-1].bias)

    def forward(
        self,
        candidate_features: torch.Tensor,
        query_xy_normalized: torch.Tensor,
        neighbor_indices: torch.Tensor,
        candidate_prior_probability: torch.Tensor,
        null_prior_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_features.ndim != 4:
            raise ValueError("candidate_features must have shape (B, Q, L, F)")
        batch, query_count, candidate_count, _ = candidate_features.shape
        if query_xy_normalized.shape != (batch, query_count, 2):
            raise ValueError("query coordinates are not aligned")
        if neighbor_indices.shape[:3] != (batch, query_count, candidate_count):
            raise ValueError("neighbor indices are not aligned")
        if candidate_prior_probability.shape != (batch, query_count, candidate_count):
            raise ValueError("candidate prior is not aligned")
        if null_prior_probability.shape != (batch, query_count):
            raise ValueError("null prior is not aligned")

        encoded = self.input_encoder(candidate_features)
        within = encoded.reshape(batch * query_count, candidate_count, -1)
        attended, _ = self.candidate_attention(within, within, within, need_weights=False)
        within = self.candidate_norm(within + attended).reshape(
            batch, query_count, candidate_count, -1
        )
        weights = candidate_prior_probability / candidate_prior_probability.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-8)
        query_nodes = torch.sum(within * weights.unsqueeze(3), dim=2)
        query_nodes = query_nodes + self.query_position(query_xy_normalized)
        query_context = self.query_graph(query_nodes)

        flat = within.reshape(batch, query_count * candidate_count, -1)
        neighbor_count = neighbor_indices.shape[-1]
        feature_dim = flat.shape[-1]
        gather_indices = neighbor_indices.reshape(batch, -1, 1).expand(
            -1, -1, feature_dim
        )
        neighbor_context_tensor = torch.gather(flat, 1, gather_indices).reshape(
            batch,
            query_count,
            candidate_count,
            neighbor_count,
            feature_dim,
        ).mean(dim=3)
        fused = self.fusion_norm(
            within
            + self.query_to_candidate(query_context).unsqueeze(2)
            + self.neighbor_to_candidate(neighbor_context_tensor)
        )
        residual = self.candidate_residual_head(fused)[:, :, :, 0]
        null_residual = self.null_residual_head(query_context)[:, :, 0]
        scale = torch.exp(self.prior_log_scale).clamp(0.1, 10.0)
        candidate_logits = residual + scale * torch.log(
            candidate_prior_probability.clamp_min(1e-8)
        )
        null_logits = null_residual + scale * torch.log(
            null_prior_probability.clamp_min(1e-8)
        )
        return candidate_logits, null_logits


class WholeImageLatentCandidateGraph(nn.Module):
    """Refine identity while retaining a mixture over support-view latents."""

    def __init__(
        self,
        latent_dim: int,
        scalar_dim: int,
        *,
        model_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(int(latent_dim), int(scalar_dim)) <= 0:
            raise ValueError("latent and scalar dimensions must be positive")
        if int(model_dim) % int(heads) != 0:
            raise ValueError("model_dim must be divisible by heads")
        self.view_encoder = nn.Sequential(
            nn.Linear(int(latent_dim) + int(scalar_dim), int(model_dim)),
            nn.LayerNorm(int(model_dim)),
            nn.GELU(),
        )
        self.view_attention = nn.MultiheadAttention(
            int(model_dim), int(heads), dropout=float(dropout), batch_first=True
        )
        self.view_norm = nn.LayerNorm(int(model_dim))
        self.candidate_attention = nn.MultiheadAttention(
            int(model_dim), int(heads), dropout=float(dropout), batch_first=True
        )
        self.candidate_norm = nn.LayerNorm(int(model_dim))
        self.query_position = nn.Sequential(
            nn.Linear(2, int(model_dim)),
            nn.GELU(),
            nn.Linear(int(model_dim), int(model_dim)),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(model_dim),
            nhead=int(heads),
            dim_feedforward=int(model_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_graph = nn.TransformerEncoder(encoder_layer, num_layers=int(layers))
        self.candidate_to_view = nn.Linear(int(model_dim), int(model_dim))
        self.query_to_view = nn.Linear(int(model_dim), int(model_dim))
        self.neighbor_to_view = nn.Linear(int(model_dim), int(model_dim))
        self.fusion_norm = nn.LayerNorm(int(model_dim))
        self.view_residual_head = nn.Linear(int(model_dim), 1)
        self.null_residual_head = nn.Sequential(
            nn.LayerNorm(int(model_dim)), nn.Linear(int(model_dim), 1)
        )
        self.prior_log_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.view_residual_head.weight)
        nn.init.zeros_(self.view_residual_head.bias)
        nn.init.zeros_(self.null_residual_head[-1].weight)
        nn.init.zeros_(self.null_residual_head[-1].bias)

    def forward(
        self,
        candidate_view_latents: torch.Tensor,
        candidate_scalar_features: torch.Tensor,
        support_view_probability: torch.Tensor,
        support_view_mask: torch.Tensor,
        query_xy_normalized: torch.Tensor,
        neighbor_indices: torch.Tensor,
        candidate_prior_probability: torch.Tensor,
        null_prior_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_view_latents.ndim != 5:
            raise ValueError("candidate_view_latents must have shape (B, Q, L, V, D)")
        batch, query_count, candidate_count, view_count, _ = candidate_view_latents.shape
        if candidate_scalar_features.shape[:3] != (batch, query_count, candidate_count):
            raise ValueError("candidate scalar features are not aligned")
        if support_view_probability.shape != (batch, query_count, candidate_count, view_count):
            raise ValueError("support-view probabilities are not aligned")
        if support_view_mask.shape != support_view_probability.shape:
            raise ValueError("support-view mask is not aligned")
        if query_xy_normalized.shape != (batch, query_count, 2):
            raise ValueError("query coordinates are not aligned")
        if neighbor_indices.shape[:3] != (batch, query_count, candidate_count):
            raise ValueError("neighbor indices are not aligned")
        if candidate_prior_probability.shape != (batch, query_count, candidate_count):
            raise ValueError("candidate prior is not aligned")
        if null_prior_probability.shape != (batch, query_count):
            raise ValueError("null prior is not aligned")
        valid_views = support_view_mask.bool()
        if torch.any(torch.sum(valid_views, dim=3) <= 0):
            raise ValueError("every candidate requires a valid support view")
        view_weights = torch.where(
            valid_views,
            support_view_probability.clamp_min(0.0),
            torch.zeros_like(support_view_probability),
        )
        view_weights = view_weights / view_weights.sum(dim=3, keepdim=True).clamp_min(1e-8)

        scalar = candidate_scalar_features.unsqueeze(3).expand(
            -1, -1, -1, view_count, -1
        )
        encoded = self.view_encoder(torch.cat([candidate_view_latents, scalar], dim=4))
        flat_views = encoded.reshape(batch * query_count * candidate_count, view_count, -1)
        flat_mask = valid_views.reshape(batch * query_count * candidate_count, view_count)
        attended, _ = self.view_attention(
            flat_views,
            flat_views,
            flat_views,
            key_padding_mask=~flat_mask,
            need_weights=False,
        )
        encoded = self.view_norm(flat_views + attended).reshape_as(encoded)
        encoded = encoded * valid_views.unsqueeze(4).to(dtype=encoded.dtype)
        candidate_nodes = torch.sum(encoded * view_weights.unsqueeze(4), dim=3)

        within = candidate_nodes.reshape(batch * query_count, candidate_count, -1)
        attended_candidates, _ = self.candidate_attention(
            within, within, within, need_weights=False
        )
        candidate_context = self.candidate_norm(within + attended_candidates).reshape_as(
            candidate_nodes
        )
        candidate_weights = candidate_prior_probability / candidate_prior_probability.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-8)
        query_nodes = torch.sum(
            candidate_context * candidate_weights.unsqueeze(3), dim=2
        ) + self.query_position(query_xy_normalized)
        query_context = self.query_graph(query_nodes)

        flat_candidates = candidate_context.reshape(
            batch, query_count * candidate_count, -1
        )
        feature_dim = flat_candidates.shape[-1]
        neighbor_count = neighbor_indices.shape[-1]
        gather_indices = neighbor_indices.reshape(batch, -1, 1).expand(
            -1, -1, feature_dim
        )
        neighbor_context = torch.gather(
            flat_candidates, 1, gather_indices
        ).reshape(
            batch,
            query_count,
            candidate_count,
            neighbor_count,
            feature_dim,
        ).mean(dim=3)
        fused = self.fusion_norm(
            encoded
            + self.candidate_to_view(candidate_context).unsqueeze(3)
            + self.query_to_view(query_context).unsqueeze(2).unsqueeze(3)
            + self.neighbor_to_view(neighbor_context).unsqueeze(3)
        )
        view_residual = self.view_residual_head(fused)[:, :, :, :, 0]
        view_residual = view_residual.masked_fill(~valid_views, -1e4)
        candidate_residual = torch.logsumexp(
            torch.log(view_weights.clamp_min(1e-8)) + view_residual, dim=3
        )
        scale = torch.exp(self.prior_log_scale).clamp(0.1, 10.0)
        candidate_logits = candidate_residual + scale * torch.log(
            candidate_prior_probability.clamp_min(1e-8)
        )
        null_logits = self.null_residual_head(query_context)[:, :, 0] + scale * torch.log(
            null_prior_probability.clamp_min(1e-8)
        )
        return candidate_logits, null_logits


def set_identity_nll(
    candidate_logits: torch.Tensor,
    null_logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    if candidate_logits.shape != positive_mask.shape:
        raise ValueError("positive mask must match candidate logits")
    all_logits = torch.cat([candidate_logits, null_logits.unsqueeze(2)], dim=2)
    denominator = torch.logsumexp(all_logits, dim=2)
    negative = torch.full_like(candidate_logits, -1e4)
    positive_numerator = torch.logsumexp(
        torch.where(positive_mask.bool(), candidate_logits, negative), dim=2
    )
    numerator = torch.where(
        torch.any(positive_mask.bool(), dim=2), positive_numerator, null_logits
    )
    return torch.mean(denominator - numerator)
