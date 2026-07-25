"""Target-free, permutation-equivariant candidate-group identity likelihood.

The localizer's fixed global top-L pool is a *set* of mutually exclusive
landmark identities for each query point.  A per-edge scalar can say that a
candidate looks plausible, but it cannot explicitly compare several visually
similar windows in that set.  This module keeps the full two-dimensional
RADIO-final, RADIO-intermediate, and ALIKE phase fields until after fixed
support-view aggregation, then scores candidates with a permutation-equivariant
group head.

The visual forward consumes only the frozen query/support image layout and
descriptor grids.  It deliberately has no pose, projection offset, residual,
track id, candidate rank, coarse score, or label input.  Exact-track and
coherent-wrong labels are joined only by the train-only losses below.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    PhaseIdentitySourceConfig,
    _edge_layout,
    _phase_field_features,
    canonical_registered_identity_or_null_targets,
    phase_identity_point_block_derangement_shift,
    resolve_phase_identity_source_configs,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    bounded_log_likelihood_ratio,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    _crop_subpixel_grid_tokens,
)


CANDIDATE_GROUP_PHASE_IDENTITY_LLR_FORMAT = "candidate_group_phase_identity_llr_v1"


@dataclass(frozen=True)
class CandidateGroupPhaseIdentityPrediction:
    """Candidate/null identity evidence emitted before any target is joined."""

    candidate_log_likelihood_ratios: torch.Tensor
    candidate_usable: torch.Tensor
    null_log_likelihood_ratios: torch.Tensor
    source_edge_usable: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        candidate = torch.as_tensor(self.candidate_log_likelihood_ratios, dtype=torch.float32)
        usable = torch.as_tensor(self.candidate_usable, dtype=torch.bool, device=candidate.device)
        null = torch.as_tensor(self.null_log_likelihood_ratios, dtype=torch.float32, device=candidate.device)
        source_usable = {
            str(name): torch.as_tensor(value, dtype=torch.bool, device=candidate.device)
            for name, value in self.source_edge_usable.items()
        }
        if (
            candidate.ndim != 2
            or candidate.shape[0] == 0
            or candidate.shape[1] < 2
            or usable.shape != candidate.shape
            or null.shape != (candidate.shape[0],)
            or set(source_usable) != set(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES)
            or any(value.ndim != 3 or value.shape[:2] != candidate.shape for value in source_usable.values())
            or not torch.isfinite(candidate).all()
            or not torch.isfinite(null).all()
        ):
            raise ValueError("candidate-group identity prediction is invalid")
        object.__setattr__(self, "candidate_log_likelihood_ratios", candidate)
        object.__setattr__(self, "candidate_usable", usable)
        object.__setattr__(self, "null_log_likelihood_ratios", null)
        object.__setattr__(self, "source_edge_usable", source_usable)


class _PhaseFeatureEmbedder(nn.Module):
    """Project one source's entire phase field into an edge embedding."""

    def __init__(self, *, feature_dim: int, embedding_dim: int) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(embedding_dim) < 4:
            raise ValueError("phase feature embedder dimensions are invalid")
        self.network = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(embedding_dim)),
            nn.GELU(),
            nn.Linear(int(embedding_dim), int(embedding_dim)),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[1] == 0 or not torch.isfinite(tensor).all():
            raise ValueError("phase feature embedder input is invalid")
        return self.network(tensor)


class CandidateGroupPhaseIdentityLLR(nn.Module):
    """Compare top-L candidates as an unordered visual set.

    There are intentionally no candidate-position embeddings, candidate-score
    features, or view-selection logits.  The only cross-candidate operation is
    a masked mean of fixed-view visual embeddings, making the output exactly
    equivariant to a candidate-order permutation.
    """

    def __init__(
        self,
        *,
        sources: Mapping[str, torch.Tensor],
        image_sizes: torch.Tensor,
        source_configs: Mapping[str, PhaseIdentitySourceConfig] | None = None,
        source_embedding_dim: int = 48,
        group_embedding_dim: int = 96,
        max_abs_log_ratio: float = 4.0,
        source_storage_dtype: torch.dtype = torch.float16,
        learn_null_log_likelihood: bool = False,
    ) -> None:
        super().__init__()
        configs = resolve_phase_identity_source_configs(source_configs)
        sizes = torch.as_tensor(image_sizes, dtype=torch.float32)
        source_map = {str(name): value for name, value in sources.items()}
        if (
            set(source_map) != set(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES)
            or sizes.ndim != 2
            or sizes.shape[0] == 0
            or sizes.shape[1] != 2
            or torch.any(sizes <= 1.0)
            or int(source_embedding_dim) < 4
            or int(group_embedding_dim) < 4
            or not math.isfinite(float(max_abs_log_ratio))
            or float(max_abs_log_ratio) <= 0.0
            or source_storage_dtype not in {torch.float16, torch.float32}
        ):
            raise ValueError("candidate-group identity model configuration is invalid")
        embedders: dict[str, nn.Module] = {}
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
            grid = torch.as_tensor(source_map[name], dtype=source_storage_dtype)
            config = configs[name]
            if (
                grid.ndim != 4
                or grid.shape[0] != len(sizes)
                or grid.shape[1] != grid.shape[2]
                or grid.shape[1] < int(config.window_size)
                or grid.shape[3] <= 0
                or not torch.isfinite(grid).all()
            ):
                raise ValueError(f"candidate-group {name} source grid is invalid")
            norms = torch.linalg.vector_norm(grid.float(), dim=-1)
            if not torch.isfinite(norms).all() or torch.max(torch.abs(norms - 1.0)).item() > 1e-2:
                raise ValueError(f"candidate-group {name} descriptors are not normalized")
            self.register_buffer(f"_{name}_grid", grid, persistent=False)
            embedders[name] = _PhaseFeatureEmbedder(
                feature_dim=int(config.feature_dimension),
                embedding_dim=int(source_embedding_dim),
            )
        input_dim = int(source_embedding_dim) * len(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES)
        self.source_embedders = nn.ModuleDict(embedders)
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(group_embedding_dim)),
            nn.GELU(),
            nn.Linear(int(group_embedding_dim), int(group_embedding_dim)),
            nn.GELU(),
        )
        self.candidate_head = nn.Sequential(
            nn.LayerNorm(int(group_embedding_dim) * 3),
            nn.Linear(int(group_embedding_dim) * 3, int(group_embedding_dim)),
            nn.GELU(),
            nn.Linear(int(group_embedding_dim), 1),
        )
        self.null_head = nn.Sequential(
            nn.LayerNorm(int(group_embedding_dim)),
            nn.Linear(int(group_embedding_dim), int(group_embedding_dim)),
            nn.GELU(),
            nn.Linear(int(group_embedding_dim), 1),
        )
        # The trainable expert must start as an exact frozen-prior no-op.
        # Random candidate/null logits would immediately scramble top-L
        # identity ranks before a single visual supervision update has been
        # earned, defeating the promotion-control interpretation.
        candidate_final = self.candidate_head[-1]
        null_final = self.null_head[-1]
        assert isinstance(candidate_final, nn.Linear) and isinstance(null_final, nn.Linear)
        nn.init.zeros_(candidate_final.weight)
        nn.init.zeros_(candidate_final.bias)
        nn.init.zeros_(null_final.weight)
        nn.init.zeros_(null_final.bias)
        self.learn_null_log_likelihood = bool(learn_null_log_likelihood)
        if not self.learn_null_log_likelihood:
            # Candidate ranking is the first gate.  A learned null head is a
            # separate calibration problem and must not suppress a correct
            # top-L candidate before candidate identity itself is validated.
            for parameter in self.null_head.parameters():
                parameter.requires_grad_(False)
        self.register_buffer("_image_sizes", sizes, persistent=False)
        self.source_configs = configs
        self.max_abs_log_ratio = float(max_abs_log_ratio)

    @property
    def device(self) -> torch.device:
        return self._image_sizes.device

    def _validate_runtime(self, runtime: CandidatePoseRGBSpatialRuntime) -> CandidatePoseRGBSpatialRuntime:
        if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
            raise ValueError("candidate-group identity likelihood requires a target-free runtime")
        active = runtime.to(self.device)
        image_count = len(self._image_sizes)
        if (
            torch.any(active.query_image_indices < 0)
            or torch.any(active.query_image_indices >= image_count)
            or torch.any(active.support_image_indices < 0)
            or torch.any(active.support_image_indices >= image_count)
        ):
            raise ValueError("candidate-group identity runtime image index is invalid")
        return active

    def _source_edge_features(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        source_name: str,
        edge_to_point: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
        support_permutation_shift: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full phase features and real-crop masks for one visual source."""

        config = self.source_configs[source_name]
        grid = getattr(self, f"_{source_name}_grid")
        query_crop, query_valid = _crop_subpixel_grid_tokens(
            image_grids=grid,
            image_sizes=self._image_sizes,
            image_indices=runtime.query_image_indices,
            xy=runtime.query_xy,
            window_size=int(config.window_size),
        )
        support_crop, support_valid = _crop_subpixel_grid_tokens(
            image_grids=grid,
            image_sizes=self._image_sizes,
            image_indices=support_image_indices,
            xy=support_xy,
            window_size=int(config.window_size),
        )
        query_crop = query_crop.index_select(0, edge_to_point)
        query_valid = query_valid.index_select(0, edge_to_point)
        shift = int(support_permutation_shift)
        if shift:
            point_count = runtime.point_count
            candidate_count = runtime.candidate_count
            view_count = runtime.support_view_count
            if point_count < 2:
                raise ValueError("candidate-group support derangement requires at least two points")
            amount = phase_identity_point_block_derangement_shift(
                point_count=point_count, shift=shift
            )
            crop_shape = (point_count, candidate_count, view_count, *support_crop.shape[1:])
            valid_shape = (point_count, candidate_count, view_count, support_valid.shape[1])
            support_crop = torch.roll(
                support_crop.reshape(crop_shape), shifts=int(amount), dims=0
            ).reshape_as(support_crop)
            support_valid = torch.roll(
                support_valid.reshape(valid_shape), shifts=int(amount), dims=0
            ).reshape_as(support_valid)
        features, usable = _phase_field_features(
            query_tokens=query_crop,
            support_tokens=support_crop,
            query_valid=query_valid,
            support_valid=support_valid,
            config=config,
        )
        shape = (runtime.point_count, runtime.candidate_count, runtime.support_view_count)
        return features.reshape(*shape, -1), usable.reshape(shape) & runtime.support_view_valid

    def forward(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        support_permutation_shift: int = 0,
        zero_appearance: bool = False,
    ) -> CandidateGroupPhaseIdentityPrediction:
        """Score one immutable top-L candidate set without pose-dependent inputs."""

        active = self._validate_runtime(runtime)
        shape = (active.point_count, active.candidate_count, active.support_view_count)
        if bool(zero_appearance):
            return CandidateGroupPhaseIdentityPrediction(
                candidate_log_likelihood_ratios=torch.zeros(
                    shape[:2], dtype=torch.float32, device=self.device
                ),
                candidate_usable=torch.any(active.support_view_valid, dim=2),
                null_log_likelihood_ratios=torch.zeros(
                    (active.point_count,), dtype=torch.float32, device=self.device
                ),
                source_edge_usable={
                    name: active.support_view_valid.to(device=self.device)
                    for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
                },
            )
        edge_to_point, _, _, support_image_indices = _edge_layout(active)
        support_xy = active.support_xy.reshape(-1, 2)
        source_usable: dict[str, torch.Tensor] = {}
        source_embeddings: list[torch.Tensor] = []
        weights = active.candidate_view_weights.to(dtype=torch.float32)
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
            features, usable = self._source_edge_features(
                runtime=active,
                source_name=name,
                edge_to_point=edge_to_point,
                support_image_indices=support_image_indices,
                support_xy=support_xy,
                support_permutation_shift=int(support_permutation_shift),
            )
            embedding = self.source_embedders[name](features.reshape(-1, features.shape[-1])).reshape(
                *shape, -1
            )
            # Missing support views retain their fixed mass at a neutral zero
            # embedding.  We never renormalize weights or learn availability.
            pooled = torch.sum(
                weights.unsqueeze(-1)
                * torch.where(usable.unsqueeze(-1), embedding, torch.zeros_like(embedding)),
                dim=2,
            )
            source_embeddings.append(pooled)
            source_usable[name] = usable
        candidate_usable = torch.stack(
            [
                torch.any(source_usable[name] & (weights > 0.0), dim=2)
                for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
            ],
            dim=0,
        ).all(dim=0)
        candidate_input = torch.cat(source_embeddings, dim=2)
        candidate_embedding = self.candidate_encoder(candidate_input)
        masked_embedding = torch.where(
            candidate_usable.unsqueeze(-1), candidate_embedding, torch.zeros_like(candidate_embedding)
        )
        count = candidate_usable.sum(dim=1, keepdim=True).to(dtype=masked_embedding.dtype)
        group_embedding = masked_embedding.sum(dim=1) / count.clamp_min(1.0)
        context = group_embedding[:, None, :].expand_as(candidate_embedding)
        relative = torch.cat(
            (candidate_embedding, context, candidate_embedding - context), dim=2
        )
        candidate_llr = bounded_log_likelihood_ratio(
            self.candidate_head(relative).squeeze(-1), max_abs_log_ratio=self.max_abs_log_ratio
        )
        candidate_llr = torch.where(candidate_usable, candidate_llr, torch.zeros_like(candidate_llr))
        if self.learn_null_log_likelihood:
            null_llr = bounded_log_likelihood_ratio(
                self.null_head(group_embedding).squeeze(-1), max_abs_log_ratio=self.max_abs_log_ratio
            )
        else:
            null_llr = torch.zeros(
                (active.point_count,), dtype=candidate_llr.dtype, device=candidate_llr.device
            )
        # A point with no fully visual candidate must not receive a learned
        # null reward from the all-zero aggregate; its evidence is neutral.
        null_llr = torch.where(count.squeeze(1) > 0.0, null_llr, torch.zeros_like(null_llr))
        return CandidateGroupPhaseIdentityPrediction(
            candidate_log_likelihood_ratios=candidate_llr,
            candidate_usable=candidate_usable,
            null_log_likelihood_ratios=null_llr,
            source_edge_usable=source_usable,
        )


def candidate_group_identity_plus_null_logits(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateGroupPhaseIdentityPrediction,
    candidate_prior_logit_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fixed candidate/null mass only after target-free visual inference."""

    weight = float(candidate_prior_logit_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("candidate-group identity prior weight is invalid")
    active = runtime.to(prediction.candidate_log_likelihood_ratios.device)
    candidate = prediction.candidate_log_likelihood_ratios
    usable = prediction.candidate_usable
    null = prediction.null_log_likelihood_ratios
    prior = active.candidate_probabilities.to(dtype=candidate.dtype)
    null_prior = active.null_probabilities.to(dtype=candidate.dtype)
    if (
        candidate.shape != prior.shape
        or usable.shape != candidate.shape
        or null.shape != null_prior.shape
        or torch.any(prior < 0.0)
        or torch.any(null_prior < 0.0)
        or torch.any(torch.abs(prior.sum(dim=1) + null_prior - 1.0) > 1e-4)
    ):
        raise ValueError("candidate-group fixed candidate/null mass is invalid")
    if weight == 0.0:
        candidate_logits = torch.where(usable, candidate, torch.zeros_like(candidate))
        null_logits = null
    else:
        candidate_logits = weight * torch.log(prior.clamp_min(torch.finfo(candidate.dtype).tiny))
        candidate_logits = candidate_logits + torch.where(usable, candidate, torch.zeros_like(candidate))
        null_logits = weight * torch.log(null_prior.clamp_min(torch.finfo(candidate.dtype).tiny)) + null
    return torch.cat((candidate_logits, null_logits[:, None]), dim=1), usable


def candidate_group_identity_probabilities(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateGroupPhaseIdentityPrediction,
    candidate_prior_logit_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized fixed-top-L candidate and explicit-null probabilities."""

    logits, _ = candidate_group_identity_plus_null_logits(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=candidate_prior_logit_weight,
    )
    probabilities = torch.softmax(logits, dim=1)
    return probabilities[:, :-1], probabilities[:, -1]


def exact_group_identity_or_null_cross_entropy(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateGroupPhaseIdentityPrediction,
    observed_candidate_mask: torch.Tensor,
    candidate_dustbin_mask: torch.Tensor,
    candidate_supervised_mask: torch.Tensor,
    candidate_prior_logit_weight: float = 1.0,
    balance_observed_and_null: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train exact candidate-or-null posterior after the visual forward."""

    observed, dustbin, supervised = canonical_registered_identity_or_null_targets(
        observed_candidate_mask=observed_candidate_mask,
        candidate_dustbin_mask=candidate_dustbin_mask,
        candidate_supervised_mask=candidate_supervised_mask,
    )
    logits, _ = candidate_group_identity_plus_null_logits(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=candidate_prior_logit_weight,
    )
    observed = observed.to(device=logits.device)
    dustbin = dustbin.to(device=logits.device)
    supervised = supervised.to(device=logits.device)
    if (
        observed.shape != logits[:, :-1].shape
        or dustbin.shape != (len(logits),)
        or supervised.shape != (len(logits),)
    ):
        raise ValueError("candidate-group exact identity target shape is invalid")
    labels = torch.where(
        dustbin,
        torch.full((len(dustbin),), logits.shape[1] - 1, dtype=torch.long, device=logits.device),
        observed.to(dtype=torch.long).argmax(dim=1),
    )
    observed_rows = observed.any(dim=1)
    _, candidate_usable = candidate_group_identity_plus_null_logits(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=candidate_prior_logit_weight,
    )
    target_candidate_usable = candidate_usable.gather(
        1, labels[:, None].clamp_max(candidate_usable.shape[1] - 1)
    ).squeeze(1)
    active_mask = supervised & torch.where(
        observed_rows,
        target_candidate_usable & (candidate_usable.sum(dim=1) >= 2),
        candidate_usable.any(dim=1),
    )
    active = torch.nonzero(active_mask, as_tuple=False).reshape(-1)
    if len(active) == 0:
        return logits.sum() * 0.0, {"identity_active": 0.0, "identity_loss": 0.0, "identity_top1": 0.0}
    losses = F.cross_entropy(logits.index_select(0, active), labels.index_select(0, active), reduction="none")
    if bool(balance_observed_and_null):
        active_targets = labels.index_select(0, active)
        observed_mask = active_targets != logits.shape[1] - 1
        null_mask = ~observed_mask
        weights = torch.ones_like(losses)
        if bool(observed_mask.any()) and bool(null_mask.any()):
            weights = torch.where(
                observed_mask,
                torch.full_like(losses, 0.5 / observed_mask.to(dtype=losses.dtype).mean()),
                torch.full_like(losses, 0.5 / null_mask.to(dtype=losses.dtype).mean()),
            )
        loss = (weights * losses).mean()
    else:
        loss = losses.mean()
    predicted = logits.index_select(0, active).argmax(dim=1)
    return loss, {
        "identity_active": float(len(active)),
        "identity_observed_active": float((active_mask & observed_rows).sum().item()),
        "identity_null_active": float((active_mask & ~observed_rows).sum().item()),
        "identity_loss": float(loss.detach().item()),
        "identity_top1": float((predicted == labels.index_select(0, active)).float().mean().item()),
    }


def current_group_hard_repeat_identity_margin_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateGroupPhaseIdentityPrediction,
    point_indices: torch.Tensor,
    positive_candidate_indices: torch.Tensor,
    negative_candidate_indices: torch.Tensor,
    margin: float,
    candidate_prior_logit_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require exact candidate identity to beat its coherent wrong alternative."""

    required_margin = float(margin)
    if not math.isfinite(required_margin) or required_margin < 0.0:
        raise ValueError("candidate-group hard-repeat margin is invalid")
    logits, usable = candidate_group_identity_plus_null_logits(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=candidate_prior_logit_weight,
    )
    points = torch.as_tensor(point_indices, dtype=torch.long, device=logits.device).reshape(-1)
    positive = torch.as_tensor(positive_candidate_indices, dtype=torch.long, device=logits.device).reshape(-1)
    negative = torch.as_tensor(negative_candidate_indices, dtype=torch.long, device=logits.device).reshape(-1)
    if (
        points.shape != positive.shape
        or points.shape != negative.shape
        or torch.any(points < 0)
        or torch.any(points >= len(logits))
        or torch.any(positive < 0)
        or torch.any(positive >= logits.shape[1] - 1)
        or torch.any(negative < 0)
        or torch.any(negative >= logits.shape[1] - 1)
        or torch.any(positive == negative)
    ):
        raise ValueError("candidate-group hard-repeat indices are invalid")
    active = usable[points, positive] & usable[points, negative]
    if not bool(active.any()):
        zero = logits.sum() * 0.0
        return zero, {
            "hard_repeat_active": 0.0,
            "hard_repeat_margin_loss": 0.0,
            "hard_repeat_mean_gap": 0.0,
            "hard_repeat_win_fraction": 0.0,
        }
    gaps = logits[points[active], positive[active]] - logits[points[active], negative[active]]
    loss = F.softplus(torch.as_tensor(required_margin, device=logits.device) - gaps).mean()
    return loss, {
        "hard_repeat_active": float(len(gaps)),
        "hard_repeat_margin_loss": float(loss.detach().item()),
        "hard_repeat_mean_gap": float(gaps.detach().mean().item()),
        "hard_repeat_win_fraction": float((gaps.detach() > 0.0).float().mean().item()),
    }
