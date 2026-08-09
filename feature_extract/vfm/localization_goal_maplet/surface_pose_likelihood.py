"""View/geometry-conditioned typed likelihood for frozen surface poses.

The model consumes runtime evidence produced by rendering the *single* stored
canonical RADIO field.  Geometry channels are pose-conditioned and ephemeral;
they are never serialized as another map embedding.  Candidate scores are
normalized jointly with a typed null hypothesis, so the output is a posterior
over one frozen candidate set rather than an independently calibrated cosine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


SCHEMA = "goal_maplet_surface_pose_likelihood_v1"
EVENT_NAMES = (
    "surface_match",
    "wrong_phase",
    "occluded",
    "field_missing",
    "query_unmapped_dynamic",
    "unresolved",
)
FEATURE_NAMES = (
    "cosine",
    "one_minus_cosine",
    "absolute_difference_mean",
    "absolute_difference_std",
    "absolute_difference_max",
    "query_norm",
    "query_contrast",
    "render_contrast",
    "parent_boundary",
    "child_boundary",
    "log_depth",
    "depth_gradient",
    "normal_camera_x",
    "normal_camera_y",
    "normal_camera_z",
    "incidence",
    "log_projected_scale",
    "primitive_uncertainty",
    "render_valid",
    "surface_visible",
    "field_missing",
    "grid_x",
    "grid_y",
)


@dataclass(frozen=True)
class SurfacePoseLikelihoodConfig:
    feature_dim: int = len(FEATURE_NAMES)
    hidden_dim: int = 64
    residual_scale: float = 0.5
    null_hidden_dim: int = 32
    candidate_conditioned_null: bool = False
    candidate_contrast_scale: float = 0.0
    candidate_disagreement_weight: bool = False
    event_log_likelihood: tuple[float, ...] = (0.0, -4.0, -0.8, -1.2, -1.0, -1.5)


def _neighbour_contrast(feature: np.ndarray) -> np.ndarray:
    value = np.asarray(feature, dtype=np.float32)
    value = value / np.maximum(np.linalg.norm(value, axis=0, keepdims=True), 1.0e-8)
    height, width = value.shape[1:]
    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    horizontal = 1.0 - np.sum(value[:, :, 1:] * value[:, :, :-1], axis=0)
    vertical = 1.0 - np.sum(value[:, 1:, :] * value[:, :-1, :], axis=0)
    total[:, 1:] += horizontal
    total[:, :-1] += horizontal
    count[:, 1:] += 1.0
    count[:, :-1] += 1.0
    total[1:, :] += vertical
    total[:-1, :] += vertical
    count[1:, :] += 1.0
    count[:-1, :] += 1.0
    return total / np.maximum(count, 1.0)


def _scalar_gradient(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    gx = np.zeros_like(source)
    gy = np.zeros_like(source)
    gx[:, 1:] = np.abs(source[:, 1:] - source[:, :-1]) * (mask[:, 1:] & mask[:, :-1])
    gy[1:, :] = np.abs(source[1:, :] - source[:-1, :]) * (mask[1:, :] & mask[:-1, :])
    return np.sqrt(gx * gx + gy * gy)


def _identity_boundary(identity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    value = np.asarray(identity, dtype=np.int64)
    mask = np.asarray(valid, dtype=bool)
    boundary = np.zeros(mask.shape, dtype=np.float32)
    horizontal = mask[:, 1:] & mask[:, :-1] & (value[:, 1:] != value[:, :-1])
    vertical = mask[1:, :] & mask[:-1, :] & (value[1:, :] != value[:-1, :])
    boundary[:, 1:] += horizontal
    boundary[:, :-1] += horizontal
    boundary[1:, :] += vertical
    boundary[:-1, :] += vertical
    return np.clip(boundary / 4.0, 0.0, 1.0)


def extract_surface_likelihood_features(
    query_feature: np.ndarray,
    rendered,
    pose_w2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-token evidence, typed diagnostic targets and query summary."""

    query = np.asarray(query_feature, dtype=np.float32)
    render = np.asarray(rendered.feature, dtype=np.float32)
    if query.shape != render.shape or query.ndim != 3:
        raise ValueError("query and rendered feature maps must have shape [C,H,W]")
    query_norm = np.linalg.norm(query, axis=0)
    query_unit = query / np.maximum(query_norm[None], 1.0e-8)
    render_unit = render / np.maximum(np.linalg.norm(render, axis=0, keepdims=True), 1.0e-8)
    valid = np.asarray(rendered.mask, dtype=bool)
    visibility = (
        valid
        if getattr(rendered, "visibility", None) is None
        else np.asarray(rendered.visibility, dtype=bool)
    )
    field_missing = (
        visibility & ~valid
        if getattr(rendered, "field_missing", None) is None
        else np.asarray(rendered.field_missing, dtype=bool)
    )
    cosine = np.sum(query_unit * render_unit, axis=0)
    cosine = np.where(valid, np.clip(cosine, -1.0, 1.0), 0.0)
    difference = np.abs(query_unit - render_unit)
    difference[:, ~valid] = 0.0
    query_contrast = _neighbour_contrast(query_unit)
    render_contrast = _neighbour_contrast(render_unit) * valid
    identity = (
        np.asarray(rendered.maplet_id, dtype=np.int64)
        if rendered.maplet_id is not None
        else np.full(valid.shape, -1, dtype=np.int64)
    )
    parent_boundary = _identity_boundary(identity, valid)
    child_identity = (
        np.asarray(rendered.child_id, dtype=np.int64)
        if getattr(rendered, "child_id", None) is not None
        else identity
    )
    child_boundary = _identity_boundary(child_identity, valid)
    depth = np.asarray(rendered.depth, dtype=np.float32)
    log_depth = np.log1p(np.maximum(depth, 0.0))
    depth_gradient = _scalar_gradient(log_depth, valid)
    normal = np.asarray(rendered.normal, dtype=np.float32)
    rotation = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)[:3, :3]
    normal_camera = normal @ rotation.T
    incidence = (
        np.abs(normal_camera[..., 2])
        if getattr(rendered, "incidence", None) is None
        else np.asarray(rendered.incidence, dtype=np.float32)
    )
    projected_scale = (
        np.zeros(valid.shape, dtype=np.float32)
        if getattr(rendered, "projected_scale", None) is None
        else np.asarray(rendered.projected_scale, dtype=np.float32)
    )
    uncertainty = np.asarray(rendered.uncertainty, dtype=np.float32)
    height, width = valid.shape
    grid_y, grid_x = np.mgrid[:height, :width]
    grid_x = (grid_x.astype(np.float32) + 0.5) / max(float(width), 1.0) * 2.0 - 1.0
    grid_y = (grid_y.astype(np.float32) + 0.5) / max(float(height), 1.0) * 2.0 - 1.0
    channels = (
        cosine,
        1.0 - cosine,
        np.mean(difference, axis=0),
        np.std(difference, axis=0),
        np.max(difference, axis=0),
        query_norm,
        query_contrast,
        render_contrast,
        parent_boundary,
        child_boundary,
        log_depth,
        depth_gradient,
        normal_camera[..., 0],
        normal_camera[..., 1],
        normal_camera[..., 2],
        incidence,
        np.log1p(np.maximum(projected_scale, 0.0)),
        uncertainty,
        valid.astype(np.float32),
        visibility.astype(np.float32),
        field_missing.astype(np.float32),
        grid_x,
        grid_y,
    )
    feature = np.stack(channels, axis=-1).astype(np.float32)
    if feature.shape[-1] != len(FEATURE_NAMES) or np.any(~np.isfinite(feature)):
        raise ValueError("invalid surface-likelihood token evidence")

    # Typed targets are diagnostic/regularizing labels, never a replacement
    # for the same-query listwise pose objective.  Their precedence makes the
    # known null events mutually exclusive.
    target = np.full(valid.shape, EVENT_NAMES.index("unresolved"), dtype=np.uint8)
    target[~visibility] = EVENT_NAMES.index("query_unmapped_dynamic")
    target[field_missing] = EVENT_NAMES.index("field_missing")
    geometric = valid & (incidence < 0.15)
    target[geometric] = EVENT_NAMES.index("occluded")
    comparable = valid & ~geometric
    target[comparable & (cosine < 0.35)] = EVENT_NAMES.index("wrong_phase")
    target[comparable & (cosine >= 0.35)] = EVENT_NAMES.index("surface_match")
    summary = np.asarray(
        [
            float(np.mean(query_contrast)),
            float(np.std(query_contrast)),
            float(np.linalg.norm(np.mean(query_unit, axis=(1, 2)))),
            float(np.percentile(query_contrast, 90.0)),
        ],
        dtype=np.float32,
    )
    return feature.reshape(-1, feature.shape[-1]), target.reshape(-1), summary


class ViewGeometryConditionedSurfaceLikelihood(nn.Module):
    """Typed token likelihood plus same-query candidate normalization."""

    def __init__(self, config: SurfacePoseLikelihoodConfig) -> None:
        super().__init__()
        if int(config.feature_dim) != len(FEATURE_NAMES):
            raise ValueError("surface-likelihood feature contract differs")
        self.config = config
        self.token_residual = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, len(EVENT_NAMES)),
        )
        null_feature_dim = 8 if bool(config.candidate_conditioned_null) else 4
        self.null_head = nn.Sequential(
            nn.Linear(null_feature_dim, config.null_hidden_dim),
            nn.GELU(),
            nn.Linear(config.null_hidden_dim, 1),
        )
        nn.init.zeros_(self.token_residual[-1].weight)
        nn.init.zeros_(self.token_residual[-1].bias)
        nn.init.zeros_(self.null_head[-1].weight)
        nn.init.constant_(self.null_head[-1].bias, -1.0)
        self.score_scale_unconstrained = nn.Parameter(torch.tensor(2.0))

    def _semantic_logits(self, feature: torch.Tensor) -> torch.Tensor:
        index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
        cosine = feature[..., index["cosine"]]
        valid = feature[..., index["render_valid"]]
        visibility = feature[..., index["surface_visible"]]
        missing = feature[..., index["field_missing"]]
        incidence = feature[..., index["incidence"]]
        uncertainty = feature[..., index["primitive_uncertainty"]]
        parent_boundary = feature[..., index["parent_boundary"]]
        child_boundary = feature[..., index["child_boundary"]]
        boundary = 0.5 * (parent_boundary + child_boundary)
        return torch.stack(
            [
                4.0 * (cosine - 0.35) + 1.5 * valid - 2.0 * missing,
                4.0 * (0.35 - cosine) + valid,
                3.0 * (1.0 - incidence) * valid + 0.5 * boundary,
                5.0 * missing,
                3.0 * (1.0 - visibility),
                2.0 * uncertainty + 0.5 * (1.0 - valid),
            ],
            dim=-1,
        )

    def forward(
        self,
        token_feature: torch.Tensor,
        query_summary: torch.Tensor,
        candidate_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return candidate score, null score and typed token posterior.

        ``token_feature`` has shape ``[B,K,N,F]`` and the candidate set K is
        frozen before this method is called.  The N-token mean is the fixed
        full-query denominator.
        """

        if token_feature.ndim != 4 or token_feature.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("token_feature must have shape [B,K,N,F]")
        if query_summary.shape != (token_feature.shape[0], 4):
            raise ValueError("query_summary must have shape [B,4]")
        if candidate_valid is None:
            candidate_valid = torch.ones(
                token_feature.shape[:2], dtype=torch.bool, device=token_feature.device,
            )
        else:
            if candidate_valid.shape != token_feature.shape[:2]:
                raise ValueError("candidate validity shape differs")
            if not torch.all(torch.any(candidate_valid, dim=1)):
                raise ValueError("every query requires at least one candidate")
        event_logits = self._semantic_logits(token_feature)
        cosine = token_feature[..., FEATURE_NAMES.index("cosine")]
        valid_count = torch.sum(candidate_valid, dim=1).to(token_feature.dtype).clamp(min=1.0)
        candidate_mean_cosine = torch.sum(
            cosine * candidate_valid[..., None].to(cosine.dtype), dim=1,
        ) / valid_count[:, None]
        relative_cosine = cosine - candidate_mean_cosine[:, None]
        if float(self.config.candidate_contrast_scale) != 0.0:
            contrast = float(self.config.candidate_contrast_scale) * relative_cosine
            event_logits[..., EVENT_NAMES.index("surface_match")] += contrast
            event_logits[..., EVENT_NAMES.index("wrong_phase")] -= contrast
        event_logits = event_logits + float(self.config.residual_scale) * self.token_residual(token_feature)
        log_event = F.log_softmax(event_logits, dim=-1)
        utility = torch.as_tensor(
            self.config.event_log_likelihood,
            dtype=token_feature.dtype,
            device=token_feature.device,
        )
        token_log_likelihood = torch.logsumexp(log_event + utility, dim=-1)
        score_scale = F.softplus(self.score_scale_unconstrained) + 1.0e-4
        token_weight = torch.ones_like(token_log_likelihood)
        if bool(self.config.candidate_disagreement_weight):
            centered_cosine = relative_cosine * candidate_valid[..., None].to(cosine.dtype)
            disagreement = torch.sqrt(
                torch.sum(centered_cosine * centered_cosine, dim=1) / valid_count[:, None]
                + 1.0e-8
            )
            disagreement /= torch.mean(disagreement, dim=1, keepdim=True).clamp(min=1.0e-4)
            shared_weight = torch.clamp(0.25 + disagreement, min=0.25, max=3.0)
            shared_weight /= torch.mean(shared_weight, dim=1, keepdim=True).clamp(min=1.0e-4)
            token_weight = shared_weight[:, None].expand_as(token_log_likelihood)
        candidate_score = score_scale * torch.mean(token_weight * token_log_likelihood, dim=-1)
        candidate_score = candidate_score.masked_fill(~candidate_valid, -torch.inf)
        null_feature = query_summary
        if bool(self.config.candidate_conditioned_null):
            finite_score = torch.where(candidate_valid, candidate_score, torch.zeros_like(candidate_score))
            count = torch.sum(candidate_valid, dim=1).to(candidate_score.dtype).clamp(min=1.0)
            score_mean = torch.sum(finite_score, dim=1) / count
            centered = torch.where(
                candidate_valid, finite_score - score_mean[:, None], torch.zeros_like(candidate_score),
            )
            score_std = torch.sqrt(torch.sum(centered * centered, dim=1) / count + 1.0e-8)
            score_max = torch.max(candidate_score, dim=1).values
            log_count = torch.log(count)
            null_feature = torch.cat([
                query_summary,
                score_mean[:, None],
                score_max[:, None],
                score_std[:, None],
                log_count[:, None],
            ], dim=1)
        null_score = self.null_head(null_feature).squeeze(-1)
        return candidate_score, null_score, torch.softmax(event_logits, dim=-1)

    def posterior(
        self,
        token_feature: torch.Tensor,
        query_summary: torch.Tensor,
        candidate_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate, null, event = self(token_feature, query_summary, candidate_valid)
        normalized = torch.softmax(torch.cat([candidate, null[:, None]], dim=1), dim=1)
        return normalized[:, :-1], normalized[:, -1], event


def listwise_surface_pose_loss(
    model: ViewGeometryConditionedSurfaceLikelihood,
    token_feature: torch.Tensor,
    query_summary: torch.Tensor,
    target_index: torch.Tensor,
    typed_target: torch.Tensor | None = None,
    typed_sample_weight: torch.Tensor | None = None,
    candidate_valid: torch.Tensor | None = None,
    *,
    typed_weight: float = 0.05,
) -> tuple[torch.Tensor, Mapping[str, float]]:
    candidate, null, event = model(token_feature, query_summary, candidate_valid)
    logits = torch.cat([candidate, null[:, None]], dim=1)
    loss_listwise = F.cross_entropy(logits, target_index)
    loss_typed = torch.zeros((), device=logits.device)
    if typed_target is not None:
        if typed_target.shape != event.shape[:-1]:
            raise ValueError("typed target shape differs")
        typed_values = F.nll_loss(
            torch.log(torch.clamp(event, min=1.0e-8)).reshape(-1, len(EVENT_NAMES)),
            typed_target.reshape(-1),
            reduction="none",
        )
        if typed_sample_weight is not None:
            if typed_sample_weight.shape != typed_target.shape:
                raise ValueError("typed sample weight shape differs")
            weight = typed_sample_weight.reshape(-1)
            loss_typed = torch.sum(weight * typed_values) / torch.clamp(torch.sum(weight), min=1.0e-8)
        else:
            loss_typed = torch.mean(typed_values)
    loss = loss_listwise + float(typed_weight) * loss_typed
    return loss, {
        "listwise_nll": float(loss_listwise.detach().cpu()),
        "typed_nll": float(loss_typed.detach().cpu()),
    }


def save_surface_pose_likelihood(
    model: ViewGeometryConditionedSurfaceLikelihood,
    path: Path,
    *,
    metadata: Mapping[str, object],
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(model.config),
        "feature_names": FEATURE_NAMES,
        "event_names": EVENT_NAMES,
        "metadata": {
            **dict(metadata),
            "artifact_type": SCHEMA,
            "stored_map_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "runtime_geometry_is_ephemeral": True,
            "stores_mapping_rgb": False,
            "uses_point_correspondences": False,
            "uses_pnp": False,
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(path))


def load_surface_pose_likelihood(
    path: Path,
    *,
    device: str = "cpu",
) -> tuple[ViewGeometryConditionedSurfaceLikelihood, dict[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("surface-likelihood feature schema differs")
    if tuple(payload.get("event_names", ())) != EVENT_NAMES:
        raise ValueError("surface-likelihood event schema differs")
    metadata = dict(payload["metadata"])
    if metadata.get("artifact_type") != SCHEMA:
        raise ValueError("not a Goal-Maplet surface pose likelihood")
    if int(metadata.get("stored_map_feature_type_count", -1)) != 1:
        raise ValueError("surface likelihood violates the single-field contract")
    model = ViewGeometryConditionedSurfaceLikelihood(
        SurfacePoseLikelihoodConfig(**payload["config"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata
