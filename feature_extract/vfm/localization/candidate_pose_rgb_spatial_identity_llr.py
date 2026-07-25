"""Candidate/view identity LLR from real RGB and multi-scale image context.

The local RGB density used by the existing measurement path answers where a
selected support observation aligns inside a query patch.  It must not also be
asked to decide whether two repeated facade observations have the same 3-D
identity.  This module provides that separate probability factor.

The runtime contract is deliberately narrow: a caller supplies a fixed
``CandidatePoseRGBSpatialRuntime`` plus query/support RGB patches.  The model
receives no pose, reprojection residual, landmark ID, coarse score, rank, or
train target.  It emits one bounded visual LLR per candidate/support-view
edge; fixed priors and the explicit null are marginalized outside this module.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from feature_extract.vfm.localization.candidate_pose_llr import (
    bounded_log_likelihood_ratio,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    _BidirectionalCrossAttentionContextCropEncoder,
    _crop_subpixel_grid_absolute_coordinates,
    _crop_subpixel_grid_tokens,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    TexturePatchEncoder,
)


CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_V2_FORMAT = (
    "candidate_pose_rgb_spatial_identity_llr_v2"
)
CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT = (
    "candidate_pose_rgb_spatial_identity_llr_v3"
)
CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES = (
    "radio_final",
    "radio_intermediate",
    "alike",
)
CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES = (
    *CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES,
    "rgb",
)
CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_CONTEXT_WINDOWS = {
    "radio_final": 15,
    "radio_intermediate": 15,
    "alike": 21,
}


def resolve_candidate_pose_rgb_spatial_identity_context_windows(
    windows: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Return the fixed odd context window for every visual scale."""

    values = (
        CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_CONTEXT_WINDOWS
        if windows is None
        else dict(windows)
    )
    if set(values) != set(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES):
        raise ValueError("identity LLR context-window scale set is incomplete")
    output = {str(name): int(value) for name, value in values.items()}
    if any(value < 3 or value % 2 != 1 for value in output.values()):
        raise ValueError("identity LLR context windows must be odd and at least three")
    return output


def resolve_candidate_pose_rgb_spatial_identity_visual_source_scales(
    *,
    visual_content_scale: float = 1.0,
    visual_source_scales: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Resolve target-free per-source appearance controls for diagnostics.

    The normal runtime leaves ``visual_source_scales`` unset, so every source
    receives ``visual_content_scale`` exactly as in the original API.  An
    optional mapping is intentionally limited to appearance scaling: source
    coordinates, validity, fixed priors, and candidate/view layout remain
    untouched.  This gives the audit path a clean way to isolate RADIO-final,
    RADIO-intermediate, ALIKE, and RGB evidence without creating a new model
    or allowing target-bearing inputs into ``forward``.
    """

    base = float(visual_content_scale)
    if not math.isfinite(base) or not 0.0 <= base <= 1.0:
        raise ValueError("identity LLR visual content scale is invalid")
    overrides = {} if visual_source_scales is None else dict(visual_source_scales)
    if not set(overrides).issubset(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES):
        raise ValueError("identity LLR visual source-scale set is invalid")
    output: dict[str, float] = {}
    for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES:
        value = float(overrides.get(name, 1.0))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("identity LLR visual source scale is invalid")
        output[name] = base * value
    return output


@dataclass(frozen=True)
class CandidatePoseRGBSpatialIdentityLLREdgePrediction:
    """Target-free candidate/view identity evidence for a fixed P1 layout.

    ``support_view_logits`` are deliberately separate from the identity LLR.
    They define a learned posterior over the fixed support observations only
    after the caller applies the immutable maplet coverage mass.  The scalar
    null residual is likewise emitted from the visual candidate set, never
    from a pose, a residual, a track label, or a coarse score.

    The optional component fields expose the independently calibrated evidence
    factors used to form ``edge_log_likelihood_ratios``.  They are diagnostic
    outputs, not additional runtime inputs.  Missing component values retain
    the v2-compatible neutral interpretation used by hand-built predictions
    in the marginalization helpers.
    """

    edge_log_likelihood_ratios: torch.Tensor
    edge_usable: torch.Tensor
    support_view_logits: torch.Tensor | None = None
    point_null_log_likelihood_ratios: torch.Tensor | None = None
    rgb_identity_edge_log_likelihood_ratios: torch.Tensor | None = None
    context_coherence_edge_log_likelihood_ratios: torch.Tensor | None = None
    rgb_edge_usable: torch.Tensor | None = None
    context_edge_usable: torch.Tensor | None = None

    def __post_init__(self) -> None:
        llrs = torch.as_tensor(self.edge_log_likelihood_ratios)
        usable = torch.as_tensor(self.edge_usable, dtype=torch.bool, device=llrs.device)
        view_logits = (
            torch.zeros_like(llrs)
            if self.support_view_logits is None
            else torch.as_tensor(self.support_view_logits, dtype=llrs.dtype, device=llrs.device)
        )
        null_llrs = (
            torch.zeros((len(llrs),), dtype=llrs.dtype, device=llrs.device)
            if self.point_null_log_likelihood_ratios is None
            else torch.as_tensor(
                self.point_null_log_likelihood_ratios, dtype=llrs.dtype, device=llrs.device
            ).reshape(-1)
        )
        rgb_llrs = (
            torch.zeros_like(llrs)
            if self.rgb_identity_edge_log_likelihood_ratios is None
            else torch.as_tensor(
                self.rgb_identity_edge_log_likelihood_ratios,
                dtype=llrs.dtype,
                device=llrs.device,
            )
        )
        context_llrs = (
            torch.zeros_like(llrs)
            if self.context_coherence_edge_log_likelihood_ratios is None
            else torch.as_tensor(
                self.context_coherence_edge_log_likelihood_ratios,
                dtype=llrs.dtype,
                device=llrs.device,
            )
        )
        source_masks_supplied = (
            self.rgb_edge_usable is not None or self.context_edge_usable is not None
        )
        if not source_masks_supplied:
            rgb_usable = usable
            context_usable = usable
        else:
            rgb_usable = (
                torch.zeros_like(usable)
                if self.rgb_edge_usable is None
                else torch.as_tensor(
                    self.rgb_edge_usable, dtype=torch.bool, device=llrs.device
                )
            )
            context_usable = (
                torch.zeros_like(usable)
                if self.context_edge_usable is None
                else torch.as_tensor(
                    self.context_edge_usable, dtype=torch.bool, device=llrs.device
                )
            )
        if (
            llrs.ndim != 3
            or usable.shape != llrs.shape
            or view_logits.shape != llrs.shape
            or rgb_llrs.shape != llrs.shape
            or context_llrs.shape != llrs.shape
            or rgb_usable.shape != llrs.shape
            or context_usable.shape != llrs.shape
            or null_llrs.shape != (len(llrs),)
            or len(llrs) == 0
            or llrs.shape[1] == 0
            or llrs.shape[2] == 0
            or not torch.isfinite(llrs).all()
            or not torch.isfinite(view_logits).all()
            or not torch.isfinite(rgb_llrs).all()
            or not torch.isfinite(context_llrs).all()
            or not torch.isfinite(null_llrs).all()
            or torch.any(rgb_usable & ~usable)
            or torch.any(context_usable & ~usable)
            or (source_masks_supplied and not torch.equal(usable, rgb_usable | context_usable))
        ):
            raise ValueError("identity LLR edge prediction is invalid")
        object.__setattr__(self, "edge_log_likelihood_ratios", llrs)
        object.__setattr__(self, "edge_usable", usable)
        object.__setattr__(self, "support_view_logits", view_logits)
        object.__setattr__(self, "point_null_log_likelihood_ratios", null_llrs)
        object.__setattr__(self, "rgb_identity_edge_log_likelihood_ratios", rgb_llrs)
        object.__setattr__(
            self,
            "context_coherence_edge_log_likelihood_ratios",
            context_llrs,
        )
        object.__setattr__(self, "rgb_edge_usable", rgb_usable)
        object.__setattr__(self, "context_edge_usable", context_usable)


@dataclass(frozen=True)
class CandidatePoseRGBSpatialIdentityLLREdgeRepresentation:
    """Frozen target-free visual representation before the scalar LLR heads.

    This is an inference-only diagnostic boundary for probing whether the
    current candidate-edge visual representation contains any separable
    correct-versus-repeat signal at all.  It intentionally exposes no pose,
    residual, target, landmark ID, coarse score, or candidate rank.  Any
    training-only hard-repeat labels must be joined by the caller *after*
    this representation has been produced.

    ``context_source_features`` preserves the independently encoded RADIO
    final, RADIO intermediate, and ALIKE context factors.  ``rgb_pair_features``
    is the high-resolution FPN patch-pair factor before the learned RGB
    projector.  The remaining fields are the exact projected and
    candidate-set-relative tensors consumed by :meth:`forward`.
    """

    context_source_features: Mapping[str, torch.Tensor]
    rgb_pair_features: torch.Tensor
    context_embeddings: torch.Tensor
    rgb_embeddings: torch.Tensor
    edge_embeddings: torch.Tensor
    relative_features: torch.Tensor
    edge_usable: torch.Tensor
    rgb_edge_usable: torch.Tensor
    context_edge_usable: torch.Tensor

    def __post_init__(self) -> None:
        sources = {str(name): torch.as_tensor(value) for name, value in self.context_source_features.items()}
        expected_sources = set(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES)
        rgb_pair = torch.as_tensor(self.rgb_pair_features)
        context = torch.as_tensor(self.context_embeddings)
        rgb = torch.as_tensor(self.rgb_embeddings, dtype=context.dtype, device=context.device)
        edges = torch.as_tensor(self.edge_embeddings, dtype=context.dtype, device=context.device)
        relative = torch.as_tensor(self.relative_features, dtype=context.dtype, device=context.device)
        usable = torch.as_tensor(self.edge_usable, dtype=torch.bool, device=context.device)
        rgb_usable = torch.as_tensor(self.rgb_edge_usable, dtype=torch.bool, device=context.device)
        context_usable = torch.as_tensor(
            self.context_edge_usable, dtype=torch.bool, device=context.device
        )
        base_shape = context.shape[:3]
        source_valid = (
            set(sources) == expected_sources
            and all(
                value.ndim == 4
                and value.shape[:3] == base_shape
                and value.shape[3] > 0
                and torch.isfinite(value).all()
                for value in sources.values()
            )
        )
        if (
            context.ndim != 4
            or context.shape[3] == 0
            or rgb.shape != context.shape
            or rgb_pair.ndim != 4
            or rgb_pair.shape[:3] != base_shape
            or rgb_pair.shape[3] == 0
            or edges.shape != (*base_shape, context.shape[3] + rgb.shape[3])
            or relative.shape != (*base_shape, edges.shape[3] * 4)
            or usable.shape != base_shape
            or rgb_usable.shape != base_shape
            or context_usable.shape != base_shape
            or len(base_shape) != 3
            or base_shape[0] == 0
            or base_shape[1] == 0
            or base_shape[2] == 0
            or not source_valid
            or not torch.isfinite(rgb_pair).all()
            or not torch.isfinite(context).all()
            or not torch.isfinite(rgb).all()
            or not torch.isfinite(edges).all()
            or not torch.isfinite(relative).all()
            or torch.any(rgb_usable & ~usable)
            or torch.any(context_usable & ~usable)
            or not torch.equal(usable, rgb_usable | context_usable)
        ):
            raise ValueError("identity LLR edge representation is invalid")
        object.__setattr__(self, "context_source_features", sources)
        object.__setattr__(self, "rgb_pair_features", rgb_pair)
        object.__setattr__(self, "context_embeddings", context)
        object.__setattr__(self, "rgb_embeddings", rgb)
        object.__setattr__(self, "edge_embeddings", edges)
        object.__setattr__(self, "relative_features", relative)
        object.__setattr__(self, "edge_usable", usable)
        object.__setattr__(self, "rgb_edge_usable", rgb_usable)
        object.__setattr__(self, "context_edge_usable", context_usable)


def candidate_pose_rgb_spatial_identity_component_edge_usable(
    *,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    component: str,
) -> torch.Tensor:
    """Return the target-free availability mask for one independent expert."""

    fields = {
        "rgb_identity": prediction.rgb_edge_usable,
        "context_coherence": prediction.context_edge_usable,
    }
    value = fields.get(str(component))
    if value is None:
        raise ValueError("identity LLR component name is invalid")
    return torch.as_tensor(
        value,
        dtype=torch.bool,
        device=prediction.edge_log_likelihood_ratios.device,
    )


def support_view_log_probabilities(
    *,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    runtime: CandidatePoseRGBSpatialRuntime,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a learned posterior over usable fixed support observations.

    The learned logits can choose between visible support views, but cannot
    invent candidate mass.  Immutable coverage weights provide the base
    measure, while unavailable views keep their coverage mass as a neutral
    missing-evidence term in :func:`marginalize_candidate_pose_rgb_spatial_identity_llr`.
    """

    logits = torch.as_tensor(prediction.support_view_logits, dtype=torch.float32)
    usable = torch.as_tensor(prediction.edge_usable, dtype=torch.bool, device=logits.device)
    active = runtime.to(logits.device)
    weights = active.candidate_view_weights.to(dtype=logits.dtype)
    if (
        logits.shape != usable.shape
        or weights.shape != logits.shape
        or torch.any(weights < 0.0)
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("identity LLR support-view posterior inputs are invalid")
    active_view = usable & (weights > 0.0)
    available_mass = torch.where(active_view, weights, torch.zeros_like(weights)).sum(dim=2)
    has_available = available_mass > 0.0
    base_logits = torch.where(
        active_view,
        logits + torch.log(weights.clamp_min(torch.finfo(logits.dtype).tiny)),
        torch.full_like(logits, -torch.inf),
    )
    # ``logsumexp([-inf, ...])`` is a valid forward value but its backward
    # domain is undefined when every support view is unavailable.  Replacing
    # those rows with a fixed one-slot fallback keeps the exact output domain
    # (all returned views remain -inf) while preventing -inf - -inf from
    # contaminating candidate-set gradients.
    empty_fallback = torch.full_like(base_logits, -torch.inf)
    empty_fallback[:, :, 0] = 0.0
    normalizer_inputs = torch.where(
        has_available[:, :, None], base_logits, empty_fallback
    )
    normalizer = torch.logsumexp(normalizer_inputs, dim=2, keepdim=True)
    normalized = base_logits - normalizer
    # Keeping invalid values at -inf makes the support-view domain explicit;
    # they are never supplied to a learned head or silently renormalized.
    normalized = torch.where(active_view, normalized, torch.full_like(normalized, -torch.inf))
    return normalized, available_mass, has_available


def marginalize_candidate_pose_rgb_spatial_identity_llr(
    *,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    runtime: CandidatePoseRGBSpatialRuntime,
    missing_edge_log_likelihood_ratio: float = 0.0,
) -> torch.Tensor:
    """Marginalize candidate views while preserving missing mass neutrally.

    The learned view posterior is normalized only over usable observations.
    Any fixed maplet mass on unavailable views is retained as the caller's
    neutral missing factor.  Consequently a support crop boundary cannot make
    a candidate more plausible simply by concentrating all of its mass on one
    surviving view.
    """

    missing = float(missing_edge_log_likelihood_ratio)
    edges = torch.as_tensor(prediction.edge_log_likelihood_ratios, dtype=torch.float32)
    usable = torch.as_tensor(prediction.edge_usable, dtype=torch.bool, device=edges.device)
    active = runtime.to(edges.device)
    weights = active.candidate_view_weights.to(dtype=torch.float32)
    if (
        weights.shape != edges.shape
        or usable.shape != edges.shape
        or not math.isfinite(missing)
        or torch.any(weights < 0.0)
    ):
        raise ValueError("identity LLR view marginalization inputs are invalid")
    view_log_probabilities, available_mass, has_available = support_view_log_probabilities(
        prediction=prediction,
        runtime=active,
    )
    learned_inputs = edges + view_log_probabilities
    empty_fallback = torch.full_like(learned_inputs, -torch.inf)
    empty_fallback[:, :, 0] = 0.0
    learned_inputs = torch.where(has_available[:, :, None], learned_inputs, empty_fallback)
    learned_term = torch.logsumexp(learned_inputs, dim=2)
    missing_mass = (weights.sum(dim=2) - available_mass).clamp_min(0.0)
    missing_term = torch.where(
        missing_mass > 0.0,
        torch.log(missing_mass.clamp_min(torch.finfo(edges.dtype).tiny)) + missing,
        torch.full_like(missing_mass, -torch.inf),
    )
    learned_term = torch.where(
        has_available,
        torch.log(available_mass.clamp_min(torch.finfo(edges.dtype).tiny)) + learned_term,
        torch.full_like(learned_term, -torch.inf),
    )
    candidate_mass = weights.sum(dim=2)
    has_candidate_mass = candidate_mass > 0.0
    # A padded candidate can have neither observed nor missing support mass.
    # Its score is ignored downstream, but feeding logaddexp(-inf, -inf) into
    # autograd would still create NaN gradients through an otherwise masked
    # row.  Use a constant neutral pair before restoring the ignored zero.
    result = torch.logaddexp(
        torch.where(has_candidate_mass, learned_term, torch.zeros_like(learned_term)),
        torch.where(has_candidate_mass, missing_term, torch.zeros_like(missing_term)),
    )
    result = torch.where(has_candidate_mass, result, torch.zeros_like(result))
    positive_candidate = active.candidate_probabilities > 0.0
    if positive_candidate.shape != result.shape:
        raise ValueError("identity LLR candidate probabilities are misaligned")
    return torch.where(positive_candidate, result, torch.zeros_like(result))


def _masked_shift_correlations(
    *,
    query_tokens: torch.Tensor,
    support_tokens: torch.Tensor,
    query_valid: torch.Tensor,
    support_valid: torch.Tensor,
    window_size: int,
    radius: int,
) -> torch.Tensor:
    """Return fixed local phase correlations and valid-pair fractions."""

    if (
        query_tokens.ndim != 3
        or support_tokens.shape != query_tokens.shape
        or query_tokens.shape[1] != int(window_size) ** 2
        or query_valid.shape != query_tokens.shape[:2]
        or support_valid.shape != query_tokens.shape[:2]
        or int(radius) < 0
    ):
        raise ValueError("identity LLR shift-correlation inputs are invalid")
    width = int(window_size)
    query = F.normalize(query_tokens.float(), dim=2).reshape(len(query_tokens), width, width, -1)
    support = F.normalize(support_tokens.float(), dim=2).reshape(
        len(support_tokens), width, width, -1
    )
    query_mask = torch.as_tensor(query_valid, dtype=torch.bool, device=query.device).reshape(
        len(query), width, width
    )
    support_mask = torch.as_tensor(
        support_valid, dtype=torch.bool, device=query.device
    ).reshape(len(query), width, width)
    outputs: list[torch.Tensor] = []
    max_mass = float(width * width)
    for row_shift in range(-int(radius), int(radius) + 1):
        for column_shift in range(-int(radius), int(radius) + 1):
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
            cosine = torch.where(
                mass > 0,
                torch.sum(values * valid.to(dtype=values.dtype), dim=(1, 2))
                / mass.to(dtype=values.dtype).clamp_min(1.0),
                torch.zeros_like(mass, dtype=values.dtype),
            )
            outputs.extend((cosine, mass.to(dtype=cosine.dtype) / max_mass))
    return torch.stack(outputs, dim=1)


def _rgb_pair_features(
    *, query_features: torch.Tensor, support_features: torch.Tensor
) -> torch.Tensor:
    """Encode high-resolution absolute patch agreement without a pose offset.

    The FPN output is pooled into a fixed 4x4 layout.  Aligned regional
    correlations retain absolute patch phase, while the 5x5 shift summary
    retains a small amount of viewpoint tolerance.  This is an identity
    feature; it is intentionally not a local spatial density.
    """

    if (
        query_features.ndim != 4
        or support_features.shape != query_features.shape
        or query_features.shape[1] == 0
        or not torch.isfinite(query_features).all()
        or not torch.isfinite(support_features).all()
    ):
        raise ValueError("identity LLR RGB FPN features are invalid")
    query = F.normalize(query_features.float(), dim=1)
    support = F.normalize(support_features.float(), dim=1)
    query_mean = torch.mean(query, dim=(2, 3))
    support_mean = torch.mean(support, dim=(2, 3))
    query_max = torch.amax(query, dim=(2, 3))
    support_max = torch.amax(support, dim=(2, 3))
    pair = torch.cat(
        [
            query_mean,
            support_mean,
            query_mean * support_mean,
            torch.abs(query_mean - support_mean),
            query_max * support_max,
            torch.abs(query_max - support_max),
        ],
        dim=1,
    )
    query_grid = F.normalize(F.adaptive_avg_pool2d(query, (4, 4)), dim=1)
    support_grid = F.normalize(F.adaptive_avg_pool2d(support, (4, 4)), dim=1)
    aligned = torch.sum(query_grid * support_grid, dim=1).reshape(len(query), -1)
    shifted: list[torch.Tensor] = []
    for row_shift in range(-2, 3):
        for column_shift in range(-2, 3):
            query_rows = slice(max(row_shift, 0), 4 + min(row_shift, 0))
            support_rows = slice(max(-row_shift, 0), 4 - max(row_shift, 0))
            query_columns = slice(max(column_shift, 0), 4 + min(column_shift, 0))
            support_columns = slice(max(-column_shift, 0), 4 - max(column_shift, 0))
            values = torch.sum(
                query_grid[:, :, query_rows, query_columns]
                * support_grid[:, :, support_rows, support_columns],
                dim=1,
            )
            shifted.append(values.mean(dim=(1, 2)))
    shift_values = torch.stack(shifted, dim=1)
    summary = torch.stack(
        [
            aligned.mean(dim=1),
            aligned.amax(dim=1),
            aligned.std(dim=1, unbiased=False),
            shift_values.amax(dim=1),
        ],
        dim=1,
    )
    return torch.cat([pair, aligned, shift_values, summary], dim=1)


class CandidatePoseRGBSpatialIdentityLLR(nn.Module):
    """Candidate-specific identity evidence with separate spatial semantics.

    ``forward`` returns an edge LLR only.  It does not consume a pose or a
    local projected offset, so callers remain free to use the output as a
    candidate identity factor while a separate module models pixel density.
    """

    def __init__(
        self,
        *,
        sources: Mapping[str, torch.Tensor],
        image_sizes: torch.Tensor,
        rgb_context_radius_px: float = 48.0,
        rgb_step_px: float = 1.0,
        texture_feature_dim: int = 32,
        hidden_dim: int = 48,
        max_abs_log_ratio: float = 4.0,
        edge_chunk_size: int = 128,
        activation_checkpointing: bool = False,
        context_windows: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        if set(sources) != set(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES):
            raise ValueError("identity LLR source set is incomplete")
        radius = float(rgb_context_radius_px)
        step = float(rgb_step_px)
        if (
            not all(
                math.isfinite(value)
                for value in (radius, step, float(max_abs_log_ratio))
            )
            or radius <= 0.0
            or step <= 0.0
            or int(texture_feature_dim) <= 0
            or int(hidden_dim) < 4
            or int(edge_chunk_size) <= 0
            or float(max_abs_log_ratio) <= 0.0
        ):
            raise ValueError("identity LLR configuration is invalid")
        patch_side = int(round(2.0 * radius / step)) + 1
        if patch_side < 5 or patch_side % 2 != 1 or not math.isclose(
            (patch_side - 1) * step,
            2.0 * radius,
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError("identity LLR RGB patch geometry is invalid")
        sizes = torch.as_tensor(image_sizes, dtype=torch.float32)
        if sizes.ndim != 2 or sizes.shape[1] != 2 or torch.any(sizes <= 1.0):
            raise ValueError("identity LLR image sizes are invalid")
        windows = resolve_candidate_pose_rgb_spatial_identity_context_windows(context_windows)
        encoders: dict[str, nn.Module] = {}
        global_projections: dict[str, nn.Module] = {}
        descriptor_dims: dict[str, int] = {}
        for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES:
            grid = torch.as_tensor(sources[name], dtype=torch.float32)
            if (
                grid.ndim != 4
                or grid.shape[0] != len(sizes)
                or grid.shape[1] != grid.shape[2]
                or grid.shape[1] < int(windows[name])
                or grid.shape[3] <= 0
                or not torch.isfinite(grid).all()
            ):
                raise ValueError(f"identity LLR {name} source grid is invalid")
            norms = torch.linalg.vector_norm(grid, dim=-1)
            if torch.max(torch.abs(norms - 1.0)) > 5e-3:
                raise ValueError(f"identity LLR {name} descriptors are not normalized")
            descriptor_dims[name] = int(grid.shape[3])
            self.register_buffer(f"_{name}_grid", grid, persistent=False)
            encoders[name] = _BidirectionalCrossAttentionContextCropEncoder(
                int(grid.shape[3]), int(hidden_dim), absolute_coordinates=True
            )
            global_projections[name] = nn.Linear(int(grid.shape[3]), int(hidden_dim), bias=False)
        self.context_encoders = nn.ModuleDict(encoders)
        self.global_projections = nn.ModuleDict(global_projections)
        self.context_windows = windows
        self.register_buffer("_image_sizes", sizes, persistent=False)
        self.texture_encoder = TexturePatchEncoder(
            feature_dim=int(texture_feature_dim),
            hidden_dim=max(int(texture_feature_dim), int(hidden_dim)),
            input_mode="rgb_graygrad",
            encoder_arch="fpn",
        )
        context_shift_dim = (2 * 2 + 1) ** 2 * 2
        context_feature_dim = (
            int(hidden_dim) * len(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES)
            + (2 * int(hidden_dim) + 1)
            * len(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES)
            + context_shift_dim
        )
        rgb_feature_dim = int(texture_feature_dim) * 6 + 16 + 25 + 4
        self.context_projector = nn.Sequential(
            nn.LayerNorm(context_feature_dim),
            nn.Linear(context_feature_dim, int(hidden_dim)),
            nn.GELU(),
        )
        self.rgb_projector = nn.Sequential(
            nn.LayerNorm(rgb_feature_dim),
            nn.Linear(rgb_feature_dim, int(hidden_dim)),
            nn.GELU(),
        )
        self.edge_embedding_dim = int(hidden_dim) * 2
        # A facade repeat is inherently relative: a support observation can be
        # visually plausible in isolation yet less plausible than another
        # fixed top-L candidate for the same query anchor.  The set summary is
        # permutation-equivariant and contains no rank, coarse score, pose, or
        # target field.  It only gives each visual edge a target-free reference
        # distribution over that query's already-frozen candidate/view set.
        self.relative_edge_feature_dim = self.edge_embedding_dim * 4

        def make_edge_head() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(self.relative_edge_feature_dim),
                nn.Linear(self.relative_edge_feature_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 1),
            )

        # Keep the v2 module name so a previously gate-approved broad
        # observation initializer can still populate shared layers.  The v3
        # runtime deliberately does not consume this fused head: source audits
        # showed it entangles RGB identity evidence with multiscale structural
        # coherence and prevents either factor from passing its own control.
        self.edge_head = make_edge_head()
        self.rgb_identity_head = make_edge_head()
        self.context_coherence_head = make_edge_head()
        self.support_view_head = nn.Sequential(
            nn.LayerNorm(self.relative_edge_feature_dim),
            nn.Linear(self.relative_edge_feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        # Mean/max pooling is invariant to candidate/view slot ordering.  This
        # is a target-free residual over the immutable explicit null prior.
        self.null_head = nn.Sequential(
            nn.LayerNorm(self.relative_edge_feature_dim * 2),
            nn.Linear(self.relative_edge_feature_dim * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        for head in (
            self.edge_head,
            self.rgb_identity_head,
            self.context_coherence_head,
            self.support_view_head,
            self.null_head,
        ):
            final = head[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        # This retained v2 state slot is intentionally never consulted by the
        # v3 forward path.  Freeze it at construction so every caller,
        # including broad-pretraining utilities, is DDP-safe by default.
        for parameter in self.edge_head.parameters():
            parameter.requires_grad_(False)
        self.rgb_context_radius_px = radius
        self.rgb_step_px = step
        self.patch_side = patch_side
        self.max_abs_log_ratio = float(max_abs_log_ratio)
        self.edge_chunk_size = int(edge_chunk_size)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.descriptor_dimensions = descriptor_dims

    @property
    def device(self) -> torch.device:
        return self._image_sizes.device

    def _scalar_head_autocast_context(self):
        """Keep probability-producing heads out of FP16 gradient overflow.

        The image encoders and crop projectors remain AMP-friendly.  The edge
        LLR, view posterior, and null residual are subsequently marginalized
        with logarithms and exponentials, though.  Returning an FP16 scalar
        here forces a scaled FP32 likelihood gradient back through a half
        output, which can overflow before GradScaler sees it and silently skip
        every optimizer step.  These heads are tiny, so evaluating them in
        FP32 is both stable and negligible for throughput.
        """

        return torch.cuda.amp.autocast(enabled=False) if self.device.type == "cuda" else nullcontext()

    def _rgb_window_usable(
        self, *, image_indices: torch.Tensor, centers_xy: torch.Tensor
    ) -> torch.Tensor:
        indices = torch.as_tensor(image_indices, dtype=torch.long, device=self.device).reshape(-1)
        centers = torch.as_tensor(centers_xy, dtype=torch.float32, device=self.device)
        if (
            centers.shape != (len(indices), 2)
            or len(indices) == 0
            or torch.any(indices < 0)
            or torch.any(indices >= len(self._image_sizes))
            or not torch.isfinite(centers).all()
        ):
            raise ValueError("identity LLR RGB ownership or centers are invalid")
        sizes = self._image_sizes.index_select(0, indices).to(dtype=torch.float32)
        radius = float(self.rgb_context_radius_px)
        return (
            (centers[:, 0] >= radius)
            & (centers[:, 1] >= radius)
            & (centers[:, 0] <= sizes[:, 0] - 1.0 - radius)
            & (centers[:, 1] <= sizes[:, 1] - 1.0 - radius)
        )

    def _context_chunk_components(
        self,
        *,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
        visual_source_scales: Mapping[str, float],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Encode each context scale while retaining its target-free factor.

        The public scalar path still concatenates these values in its original
        order.  Keeping the pieces separate here permits a frozen
        representation audit to test whether any individual scale contains
        usable hard-repeat information before another learned scalar head is
        introduced.
        """

        scales = dict(visual_source_scales)
        if set(scales) != set(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES):
            raise ValueError("identity LLR visual source scales are incomplete")
        encoded: dict[str, torch.Tensor] = {}
        globals_pair: dict[str, torch.Tensor] = {}
        alike_shift: torch.Tensor | None = None
        all_valid = torch.ones((len(query_xy),), dtype=torch.bool, device=self.device)
        for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES:
            scale = float(scales[name])
            grid = getattr(self, f"_{name}_grid")
            window = int(self.context_windows[name])
            query_crop, query_valid = _crop_subpixel_grid_tokens(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=query_image_indices,
                xy=query_xy,
                window_size=window,
            )
            support_crop, support_valid = _crop_subpixel_grid_tokens(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=support_image_indices,
                xy=support_xy,
                window_size=window,
            )
            # The zero-content path is an audit-only coordinate control.  It
            # retains valid masks and absolute crop coordinates while removing
            # all descriptor appearance values.
            query_crop = query_crop * scale
            support_crop = support_crop * scale
            paired = torch.any(query_valid & support_valid, dim=1)
            all_valid &= paired
            safe_query_valid = query_valid.clone()
            safe_support_valid = support_valid.clone()
            empty = ~paired
            safe_query_valid[empty, 0] = True
            safe_support_valid[empty, 0] = True
            query_absolute = _crop_subpixel_grid_absolute_coordinates(
                image_sizes=self._image_sizes,
                image_indices=query_image_indices,
                xy=query_xy,
                grid_size=int(grid.shape[1]),
                window_size=window,
            )
            support_absolute = _crop_subpixel_grid_absolute_coordinates(
                image_sizes=self._image_sizes,
                image_indices=support_image_indices,
                xy=support_xy,
                grid_size=int(grid.shape[1]),
                window_size=window,
            )
            encoded[name] = self.context_encoders[name](
                query_crop,
                support_crop,
                safe_query_valid,
                safe_support_valid,
                window_size=window,
                query_absolute_coordinates=query_absolute,
                support_absolute_coordinates=support_absolute,
            )
            query_global = F.normalize(
                grid.index_select(0, query_image_indices).mean(dim=(1, 2)) * scale, dim=1
            )
            support_global = F.normalize(
                grid.index_select(0, support_image_indices).mean(dim=(1, 2)) * scale, dim=1
            )
            query_projected = self.global_projections[name](query_global)
            support_projected = self.global_projections[name](support_global)
            globals_pair[name] = torch.cat(
                [
                    query_projected * support_projected,
                    torch.abs(query_projected - support_projected),
                    torch.sum(query_global * support_global, dim=1, keepdim=True),
                ],
                dim=1,
            )
            if name == "alike":
                alike_shift = _masked_shift_correlations(
                    query_tokens=query_crop,
                    support_tokens=support_crop,
                    query_valid=query_valid,
                    support_valid=support_valid,
                    window_size=window,
                    radius=2,
                )
        if alike_shift is None:
            raise RuntimeError("identity LLR ALIKE spatial context was not constructed")
        return encoded, globals_pair, alike_shift, all_valid

    def _context_chunk(
        self,
        *,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
        visual_source_scales: Mapping[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the scalar path's original concatenated context tensor."""

        encoded, globals_pair, alike_shift, all_valid = self._context_chunk_components(
            query_image_indices=query_image_indices,
            query_xy=query_xy,
            support_image_indices=support_image_indices,
            support_xy=support_xy,
            visual_source_scales=visual_source_scales,
        )
        return (
            torch.cat(
                [
                    *(encoded[name] for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES),
                    *(globals_pair[name] for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES),
                    alike_shift,
                ],
                dim=1,
            ),
            all_valid,
        )

    def _edge_chunk(
        self,
        *,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
        query_texture_features: torch.Tensor,
        support_rgb_patches: torch.Tensor,
        visual_source_scales: Mapping[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, context_valid = self._context_chunk(
            query_image_indices=query_image_indices,
            query_xy=query_xy,
            support_image_indices=support_image_indices,
            support_xy=support_xy,
            visual_source_scales=visual_source_scales,
        )
        if (
            query_texture_features.ndim != 4
            or query_texture_features.shape[0] != len(query_xy)
            or support_rgb_patches.shape
            != (len(query_xy), 3, self.patch_side, self.patch_side)
            or not torch.isfinite(query_texture_features).all()
            or not torch.isfinite(support_rgb_patches).all()
        ):
            raise ValueError("identity LLR edge RGB inputs are invalid")
        support_texture = self.texture_encoder(
            support_rgb_patches * float(visual_source_scales["rgb"])
        )
        rgb = _rgb_pair_features(
            query_features=query_texture_features,
            support_features=support_texture,
        )
        return (
            self.context_projector(context),
            self.rgb_projector(rgb),
            context_valid,
        )

    def _edge_chunk_from_tensors(
        self,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
        query_texture_features: torch.Tensor,
        support_rgb_patches: torch.Tensor,
        visual_source_scales: Mapping[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._edge_chunk(
            query_image_indices=query_image_indices,
            query_xy=query_xy,
            support_image_indices=support_image_indices,
            support_xy=support_xy,
            query_texture_features=query_texture_features,
            support_rgb_patches=support_rgb_patches,
            visual_source_scales=visual_source_scales,
        )

    def _candidate_set_relative_features(
        self, *, edge_embeddings: torch.Tensor, edge_usable: torch.Tensor
    ) -> torch.Tensor:
        """Build candidate-set-equivariant visual features for every edge.

        The operation is equivariant to a complete candidate-slot
        permutation.  It is intentionally applied before scalar view
        marginalization: each support observation retains its own identity
        score and view posterior, while the candidate set only supplies an
        appearance reference distribution.
        """

        embeddings = torch.as_tensor(edge_embeddings, dtype=torch.float32)
        usable = torch.as_tensor(edge_usable, dtype=torch.bool, device=embeddings.device)
        if (
            embeddings.ndim != 4
            or embeddings.shape[:3] != usable.shape
            or embeddings.shape[3] != self.edge_embedding_dim
            or len(embeddings) == 0
            or not torch.isfinite(embeddings).all()
        ):
            raise ValueError("identity LLR candidate-set edge embeddings are invalid")
        mask = usable.unsqueeze(-1).to(dtype=embeddings.dtype)
        count = mask.sum(dim=(1, 2), keepdim=True)
        mean = (embeddings * mask).sum(dim=(1, 2), keepdim=True) / count.clamp_min(1.0)
        centered = embeddings - mean
        variance = (centered.square() * mask).sum(dim=(1, 2), keepdim=True) / count.clamp_min(
            1.0
        )
        # Normalizing per feature makes the head sensitive to small, local
        # candidate differences instead of spending its capacity on image-wide
        # appearance shifts.  The clamp is a numerical guard for degenerate
        # candidate sets and is not a learned confidence shortcut.
        standardized = (centered / torch.sqrt(variance + 1e-4)).clamp(-6.0, 6.0)
        relative_features = torch.cat(
            [embeddings, mean.expand_as(embeddings), centered, standardized], dim=3
        )
        return relative_features

    def _source_masked_relative_features(
        self, *, relative_features: torch.Tensor, source: str
    ) -> torch.Tensor:
        """Keep exactly one visual factor in the relative candidate-set input.

        Every relative-feature group has the same ``[context, rgb]`` embedding
        layout.  Keeping the original dimensionality, rather than constructing
        a second feature layout, makes the scalar heads directly comparable and
        lets v2 shared projectors remain load-compatible.  The masked half is
        fixed at zero before the expert head, so no RGB/context cross-feature
        can enter either expert likelihood.
        """

        features = torch.as_tensor(relative_features, dtype=torch.float32)
        if (
            features.ndim != 4
            or features.shape[3] != self.relative_edge_feature_dim
            or not torch.isfinite(features).all()
        ):
            raise ValueError("identity LLR relative source features are invalid")
        half = self.edge_embedding_dim // 2
        if half <= 0 or self.edge_embedding_dim != half * 2:
            raise RuntimeError("identity LLR source embedding layout is invalid")
        if source == "rgb":
            source_mask = torch.cat(
                [
                    torch.zeros((half,), dtype=features.dtype, device=features.device),
                    torch.ones((half,), dtype=features.dtype, device=features.device),
                ]
            )
        elif source == "context":
            source_mask = torch.cat(
                [
                    torch.ones((half,), dtype=features.dtype, device=features.device),
                    torch.zeros((half,), dtype=features.dtype, device=features.device),
                ]
            )
        else:
            raise ValueError("identity LLR source expert is invalid")
        return features * source_mask.repeat(4).view(1, 1, 1, -1)

    def _candidate_set_relative_edge_raw(
        self,
        *,
        relative_features: torch.Tensor,
        edge_usable: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        """Emit an unbounded scalar LLR residual from one fixed feature view."""

        features = torch.as_tensor(relative_features, dtype=torch.float32)
        usable = torch.as_tensor(edge_usable, dtype=torch.bool, device=features.device)
        if (
            features.ndim != 4
            or features.shape[:3] != usable.shape
            or features.shape[3] != self.relative_edge_feature_dim
            or not torch.isfinite(features).all()
        ):
            raise ValueError("identity LLR relative edge features are invalid")
        with self._scalar_head_autocast_context():
            raw = head(features.float()).squeeze(3)
        if raw.shape != usable.shape or not torch.isfinite(raw).all():
            raise ValueError("identity LLR scalar edge head is invalid")
        return torch.where(usable, raw, torch.zeros_like(raw))

    def _candidate_set_relative_edge_llrs(
        self,
        *,
        relative_features: torch.Tensor,
        edge_usable: torch.Tensor,
        head: nn.Module | None = None,
    ) -> torch.Tensor:
        """Emit one bounded LLR per unaveraged edge from a scalar head.

        The default preserves the old private helper semantics for diagnostics
        and checkpoint migration.  V3 ``forward`` invokes the raw helper for
        the two independent experts and bounds their sum only once.
        """

        raw = self._candidate_set_relative_edge_raw(
            relative_features=relative_features,
            edge_usable=edge_usable,
            head=self.edge_head if head is None else head,
        )
        values = bounded_log_likelihood_ratio(raw, max_abs_log_ratio=self.max_abs_log_ratio)
        return torch.where(
            torch.as_tensor(edge_usable, dtype=torch.bool, device=raw.device),
            values,
            torch.zeros_like(values),
        )

    def _candidate_set_null_log_likelihood_ratios(
        self, *, relative_features: torch.Tensor, edge_usable: torch.Tensor
    ) -> torch.Tensor:
        """Emit a candidate-permutation-invariant visual null residual."""

        features = torch.as_tensor(relative_features, dtype=torch.float32)
        usable = torch.as_tensor(edge_usable, dtype=torch.bool, device=features.device)
        if (
            features.ndim != 4
            or features.shape[:3] != usable.shape
            or features.shape[3] != self.relative_edge_feature_dim
            or not torch.isfinite(features).all()
        ):
            raise ValueError("identity LLR null features are invalid")
        mask = usable.unsqueeze(-1)
        count = mask.sum(dim=(1, 2)).to(dtype=features.dtype)
        mean = (features * mask.to(dtype=features.dtype)).sum(dim=(1, 2)) / count.clamp_min(1.0)
        maximum = features.masked_fill(~mask, -torch.inf).amax(dim=(1, 2))
        maximum = torch.where(count > 0.0, maximum, torch.zeros_like(maximum))
        summary = torch.cat([mean, maximum], dim=1)
        with self._scalar_head_autocast_context():
            raw = self.null_head(summary.float()).reshape(-1)
            return bounded_log_likelihood_ratio(raw, max_abs_log_ratio=self.max_abs_log_ratio)

    def forward_edge_representation(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        query_rgb_patches: torch.Tensor,
        support_rgb_patches: torch.Tensor,
        visual_content_scale: float = 1.0,
        visual_source_scales: Mapping[str, float] | None = None,
    ) -> CandidatePoseRGBSpatialIdentityLLREdgeRepresentation:
        """Emit frozen target-free edge features before any scalar LLR head.

        This intentionally mirrors the visual portion of :meth:`forward`
        without invoking the identity, support-view, or null scalar heads.
        It is reserved for evaluation-mode diagnostics: using it while the
        model is training would create a second partially-checkpointed forward
        path with different activation-memory semantics.
        """

        if self.training:
            raise RuntimeError("edge representation extraction requires model.eval()")
        active = runtime.to(self.device)
        source_scales = resolve_candidate_pose_rgb_spatial_identity_visual_source_scales(
            visual_content_scale=float(visual_content_scale),
            visual_source_scales=visual_source_scales,
        )
        point_count = active.point_count
        candidate_count = active.candidate_count
        view_count = active.support_view_count
        query_patches = torch.as_tensor(
            query_rgb_patches, dtype=torch.float32, device=self.device
        )
        support_patches = torch.as_tensor(
            support_rgb_patches, dtype=torch.float32, device=self.device
        )
        expected_query = (point_count, 3, self.patch_side, self.patch_side)
        expected_support = (
            point_count,
            candidate_count,
            view_count,
            3,
            self.patch_side,
            self.patch_side,
        )
        if (
            query_patches.shape != expected_query
            or support_patches.shape != expected_support
            or not torch.isfinite(query_patches).all()
            or not torch.isfinite(support_patches).all()
        ):
            raise ValueError("identity LLR RGB patches do not match the fixed runtime")
        query_texture = self.texture_encoder(query_patches * source_scales["rgb"])
        edge_count = point_count * candidate_count * view_count
        edge_to_point = torch.arange(point_count, device=self.device).repeat_interleave(
            candidate_count * view_count
        )
        flat_support = support_patches.reshape(edge_count, 3, self.patch_side, self.patch_side)
        flat_query_indices = active.query_image_indices.index_select(0, edge_to_point)
        flat_query_xy = active.query_xy.index_select(0, edge_to_point)
        flat_support_indices = active.support_image_indices.reshape(-1)
        flat_support_xy = active.support_xy.reshape(-1, 2)
        encoded_parts = {
            name: [] for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES
        }
        global_parts = {
            name: [] for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES
        }
        alike_shift_parts: list[torch.Tensor] = []
        context_parts: list[torch.Tensor] = []
        rgb_pair_parts: list[torch.Tensor] = []
        context_valid_parts: list[torch.Tensor] = []
        for begin in range(0, edge_count, self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, edge_count)
            query_indices = flat_query_indices[begin:end]
            query_xy = flat_query_xy[begin:end]
            support_indices = flat_support_indices[begin:end]
            support_xy = flat_support_xy[begin:end]
            encoded, globals_pair, alike_shift, context_valid = self._context_chunk_components(
                query_image_indices=query_indices,
                query_xy=query_xy,
                support_image_indices=support_indices,
                support_xy=support_xy,
                visual_source_scales=source_scales,
            )
            for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES:
                encoded_parts[name].append(encoded[name])
                global_parts[name].append(globals_pair[name])
            alike_shift_parts.append(alike_shift)
            context_parts.append(
                torch.cat(
                    [
                        *(encoded[name] for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES),
                        *(globals_pair[name] for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES),
                        alike_shift,
                    ],
                    dim=1,
                )
            )
            support_texture = self.texture_encoder(
                flat_support[begin:end] * source_scales["rgb"]
            )
            rgb_pair_parts.append(
                _rgb_pair_features(
                    query_features=query_texture.index_select(
                        0, edge_to_point[begin:end]
                    ),
                    support_features=support_texture,
                )
            )
            context_valid_parts.append(context_valid)
        context_valid = torch.cat(context_valid_parts, dim=0).reshape(
            point_count, candidate_count, view_count
        )
        query_usable = self._rgb_window_usable(
            image_indices=active.query_image_indices, centers_xy=active.query_xy
        )
        support_usable = self._rgb_window_usable(
            image_indices=active.support_image_indices.reshape(-1),
            centers_xy=active.support_xy.reshape(-1, 2),
        ).reshape(point_count, candidate_count, view_count)
        context_usable = active.support_view_valid & context_valid
        rgb_usable = (
            active.support_view_valid & query_usable[:, None, None] & support_usable
        )
        usable = context_usable | rgb_usable
        context_raw = torch.cat(context_parts, dim=0)
        rgb_pair_raw = torch.cat(rgb_pair_parts, dim=0)
        context_embeddings = self.context_projector(context_raw).reshape(
            point_count, candidate_count, view_count, -1
        )
        rgb_embeddings = self.rgb_projector(rgb_pair_raw).reshape(
            point_count, candidate_count, view_count, -1
        )
        context_embeddings = torch.where(
            context_usable[..., None], context_embeddings, torch.zeros_like(context_embeddings)
        )
        rgb_embeddings = torch.where(
            rgb_usable[..., None], rgb_embeddings, torch.zeros_like(rgb_embeddings)
        )
        context_source_features: dict[str, torch.Tensor] = {}
        for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_SCALES:
            source_raw = torch.cat(
                [torch.cat(encoded_parts[name], dim=0), torch.cat(global_parts[name], dim=0)],
                dim=1,
            )
            if name == "alike":
                source_raw = torch.cat([source_raw, torch.cat(alike_shift_parts, dim=0)], dim=1)
            source_raw = source_raw.reshape(point_count, candidate_count, view_count, -1)
            context_source_features[name] = torch.where(
                context_usable[..., None], source_raw, torch.zeros_like(source_raw)
            )
        rgb_pair_features = rgb_pair_raw.reshape(point_count, candidate_count, view_count, -1)
        rgb_pair_features = torch.where(
            rgb_usable[..., None], rgb_pair_features, torch.zeros_like(rgb_pair_features)
        )
        edge_embeddings = torch.cat([context_embeddings, rgb_embeddings], dim=3)
        if edge_embeddings.shape[3] != self.edge_embedding_dim:
            raise RuntimeError("identity LLR source embedding dimensions drifted")
        relative_features = self._candidate_set_relative_features(
            edge_embeddings=edge_embeddings, edge_usable=usable
        )
        return CandidatePoseRGBSpatialIdentityLLREdgeRepresentation(
            context_source_features=context_source_features,
            rgb_pair_features=rgb_pair_features,
            context_embeddings=context_embeddings,
            rgb_embeddings=rgb_embeddings,
            edge_embeddings=edge_embeddings,
            relative_features=relative_features,
            edge_usable=usable,
            rgb_edge_usable=rgb_usable,
            context_edge_usable=context_usable,
        )

    def forward(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        query_rgb_patches: torch.Tensor,
        support_rgb_patches: torch.Tensor,
        visual_content_scale: float = 1.0,
        visual_source_scales: Mapping[str, float] | None = None,
    ) -> CandidatePoseRGBSpatialIdentityLLREdgePrediction:
        """Emit target-free candidate/view identity LLRs for one point batch."""

        active = runtime.to(self.device)
        source_scales = resolve_candidate_pose_rgb_spatial_identity_visual_source_scales(
            visual_content_scale=float(visual_content_scale),
            visual_source_scales=visual_source_scales,
        )
        point_count = active.point_count
        candidate_count = active.candidate_count
        view_count = active.support_view_count
        query_patches = torch.as_tensor(
            query_rgb_patches, dtype=torch.float32, device=self.device
        )
        support_patches = torch.as_tensor(
            support_rgb_patches, dtype=torch.float32, device=self.device
        )
        expected_query = (point_count, 3, self.patch_side, self.patch_side)
        expected_support = (
            point_count,
            candidate_count,
            view_count,
            3,
            self.patch_side,
            self.patch_side,
        )
        if (
            query_patches.shape != expected_query
            or support_patches.shape != expected_support
            or not torch.isfinite(query_patches).all()
            or not torch.isfinite(support_patches).all()
        ):
            raise ValueError("identity LLR RGB patches do not match the fixed runtime")
        query_texture = self.texture_encoder(query_patches * source_scales["rgb"])
        edge_count = point_count * candidate_count * view_count
        edge_to_point = torch.arange(point_count, device=self.device).repeat_interleave(
            candidate_count * view_count
        )
        flat_support = support_patches.reshape(edge_count, 3, self.patch_side, self.patch_side)
        flat_query_indices = active.query_image_indices.index_select(0, edge_to_point)
        flat_query_xy = active.query_xy.index_select(0, edge_to_point)
        flat_support_indices = active.support_image_indices.reshape(-1)
        flat_support_xy = active.support_xy.reshape(-1, 2)
        context_embedding_parts: list[torch.Tensor] = []
        rgb_embedding_parts: list[torch.Tensor] = []
        context_valid_parts: list[torch.Tensor] = []
        for begin in range(0, edge_count, self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, edge_count)
            tensors = (
                flat_query_indices[begin:end],
                flat_query_xy[begin:end],
                flat_support_indices[begin:end],
                flat_support_xy[begin:end],
                query_texture.index_select(0, edge_to_point[begin:end]),
                flat_support[begin:end],
            )
            if self.training and self.activation_checkpointing and torch.is_grad_enabled():
                # Keep the checkpoint inputs tensor-only.  The fixed diagnostic
                # source scales are a closure value rather than a trainable
                # tensor and therefore cannot affect gradient semantics.
                context_part, rgb_part, valid = checkpoint(
                    lambda *values: self._edge_chunk_from_tensors(
                        *values,
                        visual_source_scales=source_scales,
                    ),
                    *tensors,
                    use_reentrant=False,
                )
            else:
                context_part, rgb_part, valid = self._edge_chunk(
                    query_image_indices=tensors[0],
                    query_xy=tensors[1],
                    support_image_indices=tensors[2],
                    support_xy=tensors[3],
                    query_texture_features=tensors[4],
                    support_rgb_patches=tensors[5],
                    visual_source_scales=source_scales,
                )
            context_valid_parts.append(valid)
            context_embedding_parts.append(context_part)
            rgb_embedding_parts.append(rgb_part)
        context_valid = torch.cat(context_valid_parts, dim=0).reshape(
            point_count, candidate_count, view_count
        )
        query_usable = self._rgb_window_usable(
            image_indices=active.query_image_indices, centers_xy=active.query_xy
        )
        support_usable = self._rgb_window_usable(
            image_indices=active.support_image_indices.reshape(-1),
            centers_xy=active.support_xy.reshape(-1, 2),
        ).reshape(point_count, candidate_count, view_count)
        context_usable = active.support_view_valid & context_valid
        rgb_usable = (
            active.support_view_valid & query_usable[:, None, None] & support_usable
        )
        usable = context_usable | rgb_usable
        context_embeddings = torch.cat(context_embedding_parts, dim=0).reshape(
            point_count, candidate_count, view_count, -1
        )
        rgb_embeddings = torch.cat(rgb_embedding_parts, dim=0).reshape(
            point_count, candidate_count, view_count, -1
        )
        # An unavailable source is neutral for its own likelihood factor.  It
        # must not invalidate the other source: RGB crop boundaries are not a
        # reason to discard a valid RADIO/ALIKE observation, and conversely.
        context_embeddings = torch.where(
            context_usable[..., None], context_embeddings, torch.zeros_like(context_embeddings)
        )
        rgb_embeddings = torch.where(
            rgb_usable[..., None], rgb_embeddings, torch.zeros_like(rgb_embeddings)
        )
        edge_embeddings = torch.cat([context_embeddings, rgb_embeddings], dim=3)
        if edge_embeddings.shape[3] != self.edge_embedding_dim:
            raise RuntimeError("identity LLR source embedding dimensions drifted")
        relative_features = self._candidate_set_relative_features(
            edge_embeddings=edge_embeddings, edge_usable=usable
        )
        rgb_raw = self._candidate_set_relative_edge_raw(
            relative_features=self._source_masked_relative_features(
                relative_features=relative_features,
                source="rgb",
            ),
            edge_usable=usable,
            head=self.rgb_identity_head,
        )
        context_raw = self._candidate_set_relative_edge_raw(
            relative_features=self._source_masked_relative_features(
                relative_features=relative_features,
                source="context",
            ),
            edge_usable=usable,
            head=self.context_coherence_head,
        )
        rgb_raw = torch.where(rgb_usable, rgb_raw, torch.zeros_like(rgb_raw))
        context_raw = torch.where(
            context_usable, context_raw, torch.zeros_like(context_raw)
        )
        with self._scalar_head_autocast_context():
            rgb_llrs = bounded_log_likelihood_ratio(
                rgb_raw, max_abs_log_ratio=self.max_abs_log_ratio
            )
            context_llrs = bounded_log_likelihood_ratio(
                context_raw, max_abs_log_ratio=self.max_abs_log_ratio
            )
            # These are independent evidence factors, so add in raw log-space
            # and bound once.  Adding already-bounded outputs would change the
            # likelihood calibration near saturation.
            llrs = bounded_log_likelihood_ratio(
                rgb_raw + context_raw,
                max_abs_log_ratio=self.max_abs_log_ratio,
            )
        with self._scalar_head_autocast_context():
            support_view_logits = self.support_view_head(relative_features.float()).squeeze(3)
        point_null_llrs = self._candidate_set_null_log_likelihood_ratios(
            relative_features=relative_features, edge_usable=usable
        )
        # Keep unusable rows fixed at the neutral LLR even if their padded
        # tensor path happened to produce a finite value.
        return CandidatePoseRGBSpatialIdentityLLREdgePrediction(
            edge_log_likelihood_ratios=torch.where(usable, llrs, torch.zeros_like(llrs)),
            edge_usable=usable,
            support_view_logits=torch.where(
                usable, support_view_logits, torch.zeros_like(support_view_logits)
            ),
            point_null_log_likelihood_ratios=point_null_llrs,
            rgb_identity_edge_log_likelihood_ratios=torch.where(
                rgb_usable,
                rgb_llrs,
                torch.zeros_like(rgb_llrs),
            ),
            context_coherence_edge_log_likelihood_ratios=torch.where(
                context_usable,
                context_llrs,
                torch.zeros_like(context_llrs),
            ),
            rgb_edge_usable=rgb_usable,
            context_edge_usable=context_usable,
        )
