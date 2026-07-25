"""Target-free multi-scale real-RGB candidate spatial likelihood.

This module is intentionally narrower than the historical combined RGB / RADIO
likelihood.  It has two *real RGB only* spatial sources:

``fine``
    A one-pixel local density that retains the measurement resolution needed by
    downstream pose refinement.

``broad``
    A lower-resolution density with a substantially larger raw-image context.
    It is meant to distinguish coherent repeated facade structure, not to
    replace the fine local measurement.

Both sources receive exactly the same target-free P1 runtime: a query anchor,
the frozen global top-L tracks, and fixed SfM support observations.  Pose
projections, residuals, labels, track IDs, candidate ranks, and coarse scores
are never inputs to the visual encoder.  A caller may evaluate the emitted
density at one or more pose projections *after* the visual forward pass.

The two scales are deliberately emitted and scored independently.  Their RGB
inputs are correlated, so ``combined`` is a conservative convex blend of their
edge log likelihood ratios, not an unjustified product of two likelihoods.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.localization.candidate_pose_llr import (
    bounded_log_likelihood_ratio,
    fixed_candidate_view_mixture_log_ratio,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    continuous_joint_log_probability_at_offsets,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    normalized_spatial_log_probabilities_with_dustbin,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    TexturePatchEncoder,
    cost_volume_quality_features,
    local_offset_grid,
)


CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT = (
    "candidate_highres_rgb_multiscale_likelihood_v1"
)
CANDIDATE_HIGHRES_RGB_SOURCES = ("fine", "broad")
CANDIDATE_HIGHRES_RGB_SCORE_PRESETS = ("fine", "broad", "combined", "zero")


@dataclass(frozen=True)
class CandidateHighresRGBScalePrediction:
    """One target-free RGB density source for every candidate/support edge."""

    spatial_logits: torch.Tensor
    non_dustbin_logits: torch.Tensor
    joint_log_probabilities: torch.Tensor
    offsets_xy: torch.Tensor
    edge_log_likelihood_ratios: torch.Tensor
    edge_usable: torch.Tensor

    def __post_init__(self) -> None:
        spatial = torch.as_tensor(self.spatial_logits, dtype=torch.float32)
        non_dustbin = torch.as_tensor(self.non_dustbin_logits, dtype=torch.float32)
        joint = torch.as_tensor(self.joint_log_probabilities, dtype=torch.float32)
        offsets = torch.as_tensor(self.offsets_xy, dtype=torch.float32)
        edge_llr = torch.as_tensor(self.edge_log_likelihood_ratios, dtype=torch.float32)
        usable = torch.as_tensor(self.edge_usable, dtype=torch.bool)
        if (
            spatial.ndim != 4
            or spatial.shape[0] == 0
            or spatial.shape[1] == 0
            or spatial.shape[2] == 0
            or spatial.shape[3] < 4
            or non_dustbin.shape != spatial.shape[:-1]
            or joint.shape != (*spatial.shape[:-1], spatial.shape[-1] + 1)
            or offsets.shape != (spatial.shape[-1], 2)
            or edge_llr.shape != spatial.shape[:-1]
            or usable.shape != spatial.shape[:-1]
            or not torch.isfinite(spatial).all()
            or not torch.isfinite(non_dustbin).all()
            or not torch.isfinite(joint).all()
            or not torch.isfinite(offsets).all()
            or not torch.isfinite(edge_llr).all()
        ):
            raise ValueError("high-resolution RGB scale prediction is invalid")
        _validate_regular_offset_grid(offsets)
        if torch.any(torch.abs(torch.exp(joint).sum(dim=-1) - 1.0) > 1e-4):
            raise ValueError("high-resolution RGB scale density is not normalized")
        object.__setattr__(self, "spatial_logits", spatial)
        object.__setattr__(self, "non_dustbin_logits", non_dustbin)
        object.__setattr__(self, "joint_log_probabilities", joint)
        object.__setattr__(self, "offsets_xy", offsets)
        object.__setattr__(self, "edge_log_likelihood_ratios", edge_llr)
        object.__setattr__(self, "edge_usable", usable)


@dataclass(frozen=True)
class CandidateHighresRGBMultiscalePrediction:
    """Target-free fine and broad RGB edge likelihoods."""

    sources: Mapping[str, CandidateHighresRGBScalePrediction]

    def __post_init__(self) -> None:
        source_map = {str(name): value for name, value in self.sources.items()}
        if set(source_map) != set(CANDIDATE_HIGHRES_RGB_SOURCES):
            raise ValueError("high-resolution RGB prediction source set is incomplete")
        reference_shape: tuple[int, int, int] | None = None
        for name in CANDIDATE_HIGHRES_RGB_SOURCES:
            value = source_map[name]
            if not isinstance(value, CandidateHighresRGBScalePrediction):
                raise ValueError("high-resolution RGB prediction source has an invalid type")
            shape = tuple(int(item) for item in value.edge_usable.shape)
            if reference_shape is None:
                reference_shape = shape
            elif shape != reference_shape:
                raise ValueError("high-resolution RGB source edge layouts differ")
        object.__setattr__(self, "sources", source_map)

    @property
    def edge_shape(self) -> tuple[int, int, int]:
        return tuple(int(item) for item in self.sources["fine"].edge_usable.shape)


@dataclass(frozen=True)
class CandidateHighresRGBPoseScore:
    """Fixed top-L / fixed-support-view pose score from real RGB only."""

    pose_log_likelihood_ratios: torch.Tensor
    point_log_likelihood_ratios: torch.Tensor
    candidate_log_likelihood_ratios: torch.Tensor
    edge_log_likelihood_ratios: torch.Tensor
    edge_usable: torch.Tensor
    source_weights: Mapping[str, float]
    candidate_prior_temperature: float = 1.0

    def __post_init__(self) -> None:
        pose = torch.as_tensor(self.pose_log_likelihood_ratios, dtype=torch.float32)
        point = torch.as_tensor(self.point_log_likelihood_ratios, dtype=torch.float32)
        candidate = torch.as_tensor(self.candidate_log_likelihood_ratios, dtype=torch.float32)
        edge = torch.as_tensor(self.edge_log_likelihood_ratios, dtype=torch.float32)
        usable = torch.as_tensor(self.edge_usable, dtype=torch.bool)
        weights = {str(name): float(value) for name, value in self.source_weights.items()}
        temperature = float(self.candidate_prior_temperature)
        if (
            pose.ndim != 1
            or point.ndim != 2
            or candidate.ndim != 3
            or edge.ndim != 4
            or usable.shape != edge.shape
            or pose.shape != (point.shape[0],)
            or candidate.shape[:2] != point.shape
            or edge.shape[:3] != candidate.shape
            or set(weights) != set(CANDIDATE_HIGHRES_RGB_SOURCES)
            or any(value < 0.0 or not math.isfinite(value) for value in weights.values())
            or not math.isfinite(temperature)
            or temperature <= 0.0
            or not torch.isfinite(pose).all()
            or not torch.isfinite(point).all()
            or not torch.isfinite(candidate).all()
            or not torch.isfinite(edge).all()
        ):
            raise ValueError("high-resolution RGB pose score is invalid")
        object.__setattr__(self, "pose_log_likelihood_ratios", pose)
        object.__setattr__(self, "point_log_likelihood_ratios", point)
        object.__setattr__(self, "candidate_log_likelihood_ratios", candidate)
        object.__setattr__(self, "edge_log_likelihood_ratios", edge)
        object.__setattr__(self, "edge_usable", usable)
        object.__setattr__(self, "source_weights", weights)
        object.__setattr__(self, "candidate_prior_temperature", temperature)


def _validate_regular_offset_grid(offsets_xy: torch.Tensor) -> tuple[int, float, float, float]:
    """Validate the Cartesian grid convention used by continuous scoring."""

    offsets = torch.as_tensor(offsets_xy, dtype=torch.float32)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or len(offsets) == 0:
        raise ValueError("high-resolution RGB local offset grid is invalid")
    side = int(round(math.sqrt(int(len(offsets)))))
    if side * side != int(len(offsets)) or side < 2:
        raise ValueError("high-resolution RGB local offset grid must be square")
    xs = torch.unique(offsets[:, 0], sorted=True)
    ys = torch.unique(offsets[:, 1], sorted=True)
    if len(xs) != side or len(ys) != side:
        raise ValueError("high-resolution RGB local offset grid is not Cartesian")
    step_x = float((xs[1] - xs[0]).item())
    step_y = float((ys[1] - ys[0]).item())
    if (
        step_x <= 0.0
        or step_y <= 0.0
        or not math.isclose(step_x, step_y, rel_tol=1e-5, abs_tol=1e-5)
    ):
        raise ValueError("high-resolution RGB local offset grid step is invalid")
    expected_x, expected_y = torch.meshgrid(xs, ys, indexing="xy")
    expected = torch.stack([expected_x.reshape(-1), expected_y.reshape(-1)], dim=1)
    if not torch.allclose(offsets.cpu(), expected.cpu(), atol=1e-5, rtol=1e-5):
        raise ValueError("high-resolution RGB local offset grid ordering is invalid")
    return side, float(xs[0].item()), float(ys[0].item()), step_x


def _source_weights(
    *,
    source: str = "combined",
    source_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Resolve an explicit, conservative RGB-source blend.

    The two outputs share RGB pixels and an encoder, therefore their scores are
    averaged rather than multiplied.  A zero source preset is an exact neutral
    visual control with no dependency on image content.
    """

    if source_weights is not None:
        values = {str(name): float(value) for name, value in source_weights.items()}
        if set(values) != set(CANDIDATE_HIGHRES_RGB_SOURCES):
            raise ValueError("high-resolution RGB source weights are incomplete")
    else:
        name = str(source).strip().lower()
        if name not in CANDIDATE_HIGHRES_RGB_SCORE_PRESETS:
            raise ValueError("high-resolution RGB score preset is invalid")
        if name == "fine":
            values = {"fine": 1.0, "broad": 0.0}
        elif name == "broad":
            values = {"fine": 0.0, "broad": 1.0}
        elif name == "combined":
            values = {"fine": 0.5, "broad": 0.5}
        else:
            values = {"fine": 0.0, "broad": 0.0}
    if any(value < 0.0 or not math.isfinite(value) for value in values.values()):
        raise ValueError("high-resolution RGB source weights are invalid")
    total = float(sum(values.values()))
    if total <= 0.0:
        return {name: 0.0 for name in CANDIDATE_HIGHRES_RGB_SOURCES}
    return {name: float(values[name]) / total for name in CANDIDATE_HIGHRES_RGB_SOURCES}


def temper_candidate_prior_probabilities(
    *,
    candidate_probabilities: torch.Tensor,
    null_probabilities: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Temper a fixed top-L prior without collapsing it into an argmax.

    Candidate likelihoods are still marginalized over every positive top-L
    entry.  Temperature changes only the conditional distribution after the
    explicit null mass is removed, then restores that exact null complement.
    It is therefore a calibrated mixture parameter, not a learned identity
    shortcut and not an encoder input.
    """

    values = torch.as_tensor(candidate_probabilities, dtype=torch.float32)
    null = torch.as_tensor(null_probabilities, dtype=torch.float32, device=values.device)
    value = float(temperature)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] == 0
        or null.shape != (values.shape[0],)
        or not torch.isfinite(values).all()
        or not torch.isfinite(null).all()
        or torch.any(values < 0.0)
        or torch.any(null < 0.0)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError("high-resolution RGB candidate-prior temperature inputs are invalid")
    candidate_mass = values.sum(dim=1)
    if torch.any(torch.abs(candidate_mass + null - 1.0) > 1e-4):
        raise ValueError("high-resolution RGB candidate/null prior mass is invalid")
    if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-8):
        return values
    positive_mass = candidate_mass > 1e-8
    if not bool(positive_mass.all()):
        raise ValueError("high-resolution RGB candidate prior cannot temper zero candidate mass")
    conditional = values / candidate_mass.unsqueeze(1)
    positive = values > 0.0
    tempered = torch.where(
        positive,
        conditional.clamp_min(torch.finfo(conditional.dtype).tiny).pow(1.0 / value),
        torch.zeros_like(conditional),
    )
    tempered = tempered / tempered.sum(dim=1, keepdim=True).clamp_min(1e-12)
    output = tempered * candidate_mass.unsqueeze(1)
    if (
        torch.any((output > 0.0) != positive)
        or torch.any(torch.abs(output.sum(dim=1) + null - 1.0) > 1e-4)
    ):
        raise RuntimeError("high-resolution RGB candidate-prior tempering drifted")
    return output


def _regular_grouped_template_cost_volume_logits(
    *,
    query_features: torch.Tensor,
    support_features: torch.Tensor,
    search_radius_px: float,
    context_radius_px: float,
    feature_step_px: float,
    temperature: float,
) -> torch.Tensor:
    """Fast regular-grid template correlation, retaining autocast precision.

    The shared measurement helper supports fractional offsets and multi-scale
    template resizing, so it intentionally promotes every correlation to
    FP32.  This new branch needs neither: fine uses one-pixel shifts and broad
    uses two-pixel shifts on a downsampled feature grid.  Keeping the grouped
    convolutions in the current autocast dtype materially improves RTX 3090
    throughput while the final density/calibration remains FP32.
    """

    query = torch.as_tensor(query_features)
    support = torch.as_tensor(support_features, device=query.device)
    search = float(search_radius_px)
    context = float(context_radius_px)
    step = float(feature_step_px)
    scale = float(temperature)
    if (
        query.ndim != 4
        or support.shape != query.shape
        or query.shape[0] == 0
        or query.shape[1] == 0
        or not query.is_floating_point()
        or not support.is_floating_point()
        or not all(math.isfinite(value) for value in (search, context, step, scale))
        or search <= 0.0
        or context <= 0.0
        or step <= 0.0
        or scale <= 0.0
    ):
        raise ValueError("regular RGB cost-volume inputs are invalid")
    search_steps = int(round(search / step))
    context_steps = int(round(context / step))
    crop_steps = int(round((search + context) / step))
    if (
        search_steps < 1
        or context_steps < 1
        or not math.isclose(search_steps * step, search, rel_tol=1e-5, abs_tol=1e-5)
        or not math.isclose(context_steps * step, context, rel_tol=1e-5, abs_tol=1e-5)
        or int(query.shape[2]) != 2 * crop_steps + 1
        or int(query.shape[3]) != 2 * crop_steps + 1
    ):
        raise ValueError("regular RGB cost-volume geometry is invalid")
    batch, channels, height, width = (int(value) for value in query.shape)
    template_side = 2 * context_steps + 1
    center = crop_steps
    template = support[
        :,
        :,
        center - context_steps : center + context_steps + 1,
        center - context_steps : center + context_steps + 1,
    ]
    if template.shape[2:] != (template_side, template_side):
        raise RuntimeError("regular RGB cost-volume template geometry drifted")
    grouped_query = query.reshape(1, batch * channels, height, width)
    kernel = template.reshape(batch, channels, template_side, template_side)
    norm_kernel = torch.ones(
        (batch, channels, template_side, template_side),
        dtype=query.dtype,
        device=query.device,
    )
    query_norm = torch.sqrt(
        F.conv2d(grouped_query.square(), norm_kernel, groups=batch).clamp_min(
            torch.finfo(query.dtype).tiny
        )
    )
    numerator = F.conv2d(grouped_query, kernel, groups=batch)
    support_norm = torch.linalg.vector_norm(kernel.reshape(batch, -1), dim=1).clamp_min(
        torch.finfo(query.dtype).tiny
    )
    correlation = numerator / (
        query_norm * support_norm.reshape(1, batch, 1, 1)
    ).clamp_min(torch.finfo(query.dtype).tiny)
    expected_side = 2 * search_steps + 1
    if correlation.shape != (1, batch, expected_side, expected_side):
        raise RuntimeError("regular RGB cost-volume output geometry drifted")
    return correlation.reshape(batch, -1) * scale


class _RGBEdgeCalibration(nn.Module):
    """Calibrate a raw visual cost volume without geometry or identity inputs."""

    def __init__(self, *, hidden_dim: int, max_abs_edge_log_ratio: float) -> None:
        super().__init__()
        width = int(hidden_dim)
        cap = float(max_abs_edge_log_ratio)
        if width < 4 or not math.isfinite(cap) or cap <= 0.0:
            raise ValueError("RGB edge calibration configuration is invalid")
        self.network = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        # Initial output is exactly the uncalibrated density: raw temperature,
        # 0.5 non-dustbin mass, and zero scalar LLR.  Unlike a zero projection
        # layer, all three heads retain nonzero gradients at the first step.
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.max_abs_edge_log_ratio = cap

    def forward(self, raw_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.as_tensor(raw_logits, dtype=torch.float32)
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 4:
            raise ValueError("RGB edge calibration expects a cost-volume matrix")
        values = self.network(cost_volume_quality_features(logits))
        log_temperature = values[:, 0].clamp(min=-2.0, max=2.0)
        calibrated = logits * torch.exp(log_temperature).unsqueeze(1)
        non_dustbin = values[:, 1]
        edge_llr = torch.tanh(values[:, 2]) * self.max_abs_edge_log_ratio
        return calibrated, non_dustbin, edge_llr


class CandidateHighresRGBMultiscaleLikelihood(nn.Module):
    """Fine plus broad raw-RGB local likelihood with target-free inference.

    ``fine`` uses a 1-pixel grid for local measurement.  ``broad`` downsamples
    the shared FPN feature map by two and uses a larger context template on a
    2-pixel grid.  Both branches are candidate-specific template searches and
    are evaluated at pose projections only outside this module.
    """

    def __init__(
        self,
        *,
        image_sizes: torch.Tensor,
        fine_search_radius_px: float = 8.0,
        fine_context_radius_px: float = 12.0,
        fine_step_px: float = 1.0,
        broad_search_radius_px: float = 8.0,
        broad_context_radius_px: float = 24.0,
        broad_feature_step_px: float = 2.0,
        broad_output_step_px: float = 2.0,
        texture_feature_dim: int = 32,
        hidden_dim: int = 32,
        edge_chunk_size: int = 128,
        rgb_temperature: float = 10.0,
        max_abs_edge_log_ratio: float = 3.0,
        texture_input_mode: str = "rgb_graygrad",
    ) -> None:
        super().__init__()
        sizes = torch.as_tensor(image_sizes, dtype=torch.float32)
        if sizes.ndim != 2 or sizes.shape[0] == 0 or sizes.shape[1] != 2 or torch.any(sizes <= 1.0):
            raise ValueError("high-resolution RGB image sizes are invalid")
        values = (
            float(fine_search_radius_px),
            float(fine_context_radius_px),
            float(fine_step_px),
            float(broad_search_radius_px),
            float(broad_context_radius_px),
            float(broad_feature_step_px),
            float(broad_output_step_px),
            float(rgb_temperature),
            float(max_abs_edge_log_ratio),
        )
        if (
            not all(math.isfinite(value) for value in values)
            or min(values[:7]) <= 0.0
            or float(rgb_temperature) <= 0.0
            or float(max_abs_edge_log_ratio) <= 0.0
            or int(texture_feature_dim) <= 0
            or int(hidden_dim) < 4
            or int(edge_chunk_size) <= 0
        ):
            raise ValueError("high-resolution RGB likelihood configuration is invalid")
        if not math.isclose(float(fine_step_px), 1.0, abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError("fine RGB density must retain one-pixel output spacing")
        if not math.isclose(
            float(broad_feature_step_px), float(broad_output_step_px), abs_tol=1e-6, rel_tol=1e-6
        ):
            raise ValueError("broad RGB density currently requires aligned feature/output spacing")
        if not math.isclose(float(broad_feature_step_px), 2.0, abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError("broad RGB density currently requires two-pixel feature spacing")
        fine_patch_side = _patch_side(
            search_radius_px=float(fine_search_radius_px),
            context_radius_px=float(fine_context_radius_px),
            step_px=float(fine_step_px),
        )
        broad_patch_side = _patch_side(
            search_radius_px=float(broad_search_radius_px),
            context_radius_px=float(broad_context_radius_px),
            step_px=float(broad_feature_step_px),
        )
        full_patch_radius = max(
            float(fine_search_radius_px) + float(fine_context_radius_px),
            float(broad_search_radius_px) + float(broad_context_radius_px),
        )
        full_patch_side = int(round(2.0 * full_patch_radius)) + 1
        if (
            full_patch_side % 2 != 1
            or full_patch_side < fine_patch_side
            or int(round(2.0 * full_patch_radius / float(broad_feature_step_px))) + 1
            != broad_patch_side
        ):
            raise ValueError("high-resolution RGB multi-scale patch geometry is invalid")
        fine_offsets = local_offset_grid(
            search_radius_px=float(fine_search_radius_px),
            step_px=float(fine_step_px),
            device=None,
            dtype=torch.float32,
        )
        broad_offsets = local_offset_grid(
            search_radius_px=float(broad_search_radius_px),
            step_px=float(broad_output_step_px),
            device=None,
            dtype=torch.float32,
        )
        _validate_regular_offset_grid(fine_offsets)
        _validate_regular_offset_grid(broad_offsets)
        self.register_buffer("_image_sizes", sizes, persistent=False)
        self.register_buffer("_fine_offsets_xy", fine_offsets, persistent=False)
        self.register_buffer("_broad_offsets_xy", broad_offsets, persistent=False)
        self.texture_encoder = TexturePatchEncoder(
            feature_dim=int(texture_feature_dim),
            hidden_dim=max(int(texture_feature_dim), int(hidden_dim)),
            input_mode=str(texture_input_mode),
            encoder_arch="fpn",
        )
        self.calibrators = nn.ModuleDict(
            {
                name: _RGBEdgeCalibration(
                    hidden_dim=int(hidden_dim),
                    max_abs_edge_log_ratio=float(max_abs_edge_log_ratio),
                )
                for name in CANDIDATE_HIGHRES_RGB_SOURCES
            }
        )
        self.fine_search_radius_px = float(fine_search_radius_px)
        self.fine_context_radius_px = float(fine_context_radius_px)
        self.fine_step_px = float(fine_step_px)
        self.broad_search_radius_px = float(broad_search_radius_px)
        self.broad_context_radius_px = float(broad_context_radius_px)
        self.broad_feature_step_px = float(broad_feature_step_px)
        self.broad_output_step_px = float(broad_output_step_px)
        self.full_patch_radius_px = full_patch_radius
        self.patch_side = full_patch_side
        self.fine_patch_side = fine_patch_side
        self.broad_patch_side = broad_patch_side
        self.edge_chunk_size = int(edge_chunk_size)
        self.rgb_temperature = float(rgb_temperature)
        self.max_abs_edge_log_ratio = float(max_abs_edge_log_ratio)

    @property
    def device(self) -> torch.device:
        return self._image_sizes.device

    def _validate_runtime(self, runtime: CandidatePoseRGBSpatialRuntime) -> CandidatePoseRGBSpatialRuntime:
        if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
            raise ValueError("high-resolution RGB likelihood requires a target-free runtime")
        active = runtime.to(self.device)
        if (
            torch.any(active.query_image_indices >= len(self._image_sizes))
            or torch.any(active.support_image_indices >= len(self._image_sizes))
        ):
            raise ValueError("high-resolution RGB runtime image index is out of range")
        return active

    def _window_usable(
        self,
        *,
        image_indices: torch.Tensor,
        centers_xy: torch.Tensor,
        radius_px: float,
    ) -> torch.Tensor:
        indices = torch.as_tensor(image_indices, dtype=torch.long, device=self.device).reshape(-1)
        centers = torch.as_tensor(centers_xy, dtype=torch.float32, device=self.device)
        radius = float(radius_px)
        if (
            len(indices) == 0
            or centers.shape != (len(indices), 2)
            or torch.any(indices < 0)
            or torch.any(indices >= len(self._image_sizes))
            or not torch.isfinite(centers).all()
            or not math.isfinite(radius)
            or radius < 0.0
        ):
            raise ValueError("high-resolution RGB window ownership is invalid")
        sizes = self._image_sizes.index_select(0, indices)
        return (
            (centers[:, 0] >= radius)
            & (centers[:, 1] >= radius)
            & (centers[:, 0] <= sizes[:, 0] - 1.0 - radius)
            & (centers[:, 1] <= sizes[:, 1] - 1.0 - radius)
        )

    def _scale_features(self, features: torch.Tensor, *, source: str) -> torch.Tensor:
        name = str(source)
        if features.ndim != 4 or features.shape[2:] != (self.patch_side, self.patch_side):
            raise ValueError("high-resolution RGB FPN feature geometry is invalid")
        if name == "fine":
            start = (self.patch_side - self.fine_patch_side) // 2
            stop = start + self.fine_patch_side
            return features[:, :, start:stop, start:stop]
        if name == "broad":
            return F.interpolate(
                features,
                size=(self.broad_patch_side, self.broad_patch_side),
                mode="bilinear",
                align_corners=True,
            )
        raise ValueError("high-resolution RGB source is invalid")

    def _cost_volume(
        self,
        *,
        source: str,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        name = str(source)
        if name == "fine":
            logits = _regular_grouped_template_cost_volume_logits(
                query_features=query_features,
                support_features=support_features,
                search_radius_px=self.fine_search_radius_px,
                context_radius_px=self.fine_context_radius_px,
                feature_step_px=self.fine_step_px,
                temperature=self.rgb_temperature,
            )
            expected = self._fine_offsets_xy
        elif name == "broad":
            logits = _regular_grouped_template_cost_volume_logits(
                query_features=query_features,
                support_features=support_features,
                search_radius_px=self.broad_search_radius_px,
                context_radius_px=self.broad_context_radius_px,
                feature_step_px=self.broad_feature_step_px,
                temperature=self.rgb_temperature,
            )
            expected = self._broad_offsets_xy
        else:
            raise ValueError("high-resolution RGB source is invalid")
        if logits.shape != (len(query_features), len(expected)):
            raise RuntimeError("high-resolution RGB cost volume emitted an unexpected grid")
        calibrated, non_dustbin, edge_llr = self.calibrators[name](logits)
        return calibrated, non_dustbin, edge_llr

    def _neutral_prediction(
        self, *, runtime: CandidatePoseRGBSpatialRuntime
    ) -> CandidateHighresRGBMultiscalePrediction:
        point_count = runtime.point_count
        candidate_count = runtime.candidate_count
        view_count = runtime.support_view_count
        source_values: dict[str, CandidateHighresRGBScalePrediction] = {}
        for name, offsets in (
            ("fine", self._fine_offsets_xy),
            ("broad", self._broad_offsets_xy),
        ):
            spatial = torch.zeros(
                (point_count, candidate_count, view_count, len(offsets)),
                dtype=torch.float32,
                device=self.device,
            )
            non_dustbin = torch.zeros(
                (point_count, candidate_count, view_count),
                dtype=torch.float32,
                device=self.device,
            )
            joint = normalized_spatial_log_probabilities_with_dustbin(
                spatial.reshape(-1, spatial.shape[-1]), non_dustbin.reshape(-1)
            ).reshape(point_count, candidate_count, view_count, -1)
            source_values[name] = CandidateHighresRGBScalePrediction(
                spatial_logits=spatial,
                non_dustbin_logits=non_dustbin,
                joint_log_probabilities=joint,
                offsets_xy=offsets,
                edge_log_likelihood_ratios=non_dustbin,
                edge_usable=torch.zeros_like(runtime.support_view_valid),
            )
        return CandidateHighresRGBMultiscalePrediction(sources=source_values)

    def forward(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        query_rgb_patches: torch.Tensor,
        support_rgb_patches: torch.Tensor,
        zero_appearance: bool = False,
        active_sources: Sequence[str] | None = None,
    ) -> CandidateHighresRGBMultiscalePrediction:
        """Emit visual evidence once, before any pose projection is supplied.

        ``zero_appearance`` is a structural null-image control.  It returns an
        exact neutral prediction rather than allowing encoder biases or crop
        border masks to masquerade as visual evidence.
        """

        active = self._validate_runtime(runtime)
        enabled_sources = (
            set(CANDIDATE_HIGHRES_RGB_SOURCES)
            if active_sources is None
            else {str(name) for name in active_sources}
        )
        if not enabled_sources or not enabled_sources.issubset(CANDIDATE_HIGHRES_RGB_SOURCES):
            raise ValueError("high-resolution RGB active source set is invalid")
        point_count = active.point_count
        candidate_count = active.candidate_count
        view_count = active.support_view_count
        query = torch.as_tensor(query_rgb_patches, dtype=torch.float32, device=self.device)
        support = torch.as_tensor(support_rgb_patches, dtype=torch.float32, device=self.device)
        if (
            query.shape != (point_count, 3, self.patch_side, self.patch_side)
            or support.shape
            != (point_count, candidate_count, view_count, 3, self.patch_side, self.patch_side)
            or not torch.isfinite(query).all()
            or not torch.isfinite(support).all()
        ):
            raise ValueError("high-resolution RGB patches do not match the fixed runtime")
        if bool(zero_appearance):
            return self._neutral_prediction(runtime=active)
        edge_count = point_count * candidate_count * view_count
        query_features = self.texture_encoder(query)
        edge_to_point = torch.arange(point_count, device=self.device).repeat_interleave(
            candidate_count * view_count
        )
        flat_support = support.reshape(edge_count, 3, self.patch_side, self.patch_side)
        spatial_parts: dict[str, list[torch.Tensor]] = {
            name: [] for name in CANDIDATE_HIGHRES_RGB_SOURCES
        }
        dustbin_parts: dict[str, list[torch.Tensor]] = {
            name: [] for name in CANDIDATE_HIGHRES_RGB_SOURCES
        }
        edge_llr_parts: dict[str, list[torch.Tensor]] = {
            name: [] for name in CANDIDATE_HIGHRES_RGB_SOURCES
        }
        for begin in range(0, edge_count, self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, edge_count)
            query_chunk = query_features.index_select(0, edge_to_point[begin:end])
            support_chunk = self.texture_encoder(flat_support[begin:end])
            for name in CANDIDATE_HIGHRES_RGB_SOURCES:
                if name not in enabled_sources:
                    continue
                calibrated, non_dustbin, edge_llr = self._cost_volume(
                    source=name,
                    query_features=self._scale_features(query_chunk, source=name),
                    support_features=self._scale_features(support_chunk, source=name),
                )
                spatial_parts[name].append(calibrated)
                dustbin_parts[name].append(non_dustbin)
                edge_llr_parts[name].append(edge_llr)
        neutral_sources = self._neutral_prediction(runtime=active).sources
        source_values: dict[str, CandidateHighresRGBScalePrediction] = {}
        for name, offsets, context_radius in (
            ("fine", self._fine_offsets_xy, self.fine_context_radius_px),
            ("broad", self._broad_offsets_xy, self.broad_context_radius_px),
        ):
            if name not in enabled_sources:
                source_values[name] = neutral_sources[name]
                continue
            spatial = torch.cat(spatial_parts[name], dim=0).reshape(
                point_count, candidate_count, view_count, -1
            )
            non_dustbin = torch.cat(dustbin_parts[name], dim=0).reshape(
                point_count, candidate_count, view_count
            )
            edge_llr = torch.cat(edge_llr_parts[name], dim=0).reshape(
                point_count, candidate_count, view_count
            )
            joint = normalized_spatial_log_probabilities_with_dustbin(
                spatial.reshape(-1, spatial.shape[-1]), non_dustbin.reshape(-1)
            ).reshape(point_count, candidate_count, view_count, -1)
            radius = max(
                self.fine_search_radius_px if name == "fine" else self.broad_search_radius_px,
                0.0,
            ) + float(context_radius)
            query_usable = self._window_usable(
                image_indices=active.query_image_indices,
                centers_xy=active.query_xy,
                radius_px=radius,
            )
            support_usable = self._window_usable(
                image_indices=active.support_image_indices.reshape(-1),
                centers_xy=active.support_xy.reshape(-1, 2),
                radius_px=radius,
            ).reshape(point_count, candidate_count, view_count)
            usable = active.support_view_valid & query_usable[:, None, None] & support_usable
            source_values[name] = CandidateHighresRGBScalePrediction(
                spatial_logits=spatial,
                non_dustbin_logits=non_dustbin,
                joint_log_probabilities=joint,
                offsets_xy=offsets,
                edge_log_likelihood_ratios=edge_llr,
                edge_usable=usable,
            )
        return CandidateHighresRGBMultiscalePrediction(sources=source_values)


def _patch_side(*, search_radius_px: float, context_radius_px: float, step_px: float) -> int:
    radius = float(search_radius_px) + float(context_radius_px)
    side = int(round(2.0 * radius / float(step_px))) + 1
    if side < 3 or side % 2 != 1 or not math.isclose(
        float(side - 1) * float(step_px), 2.0 * radius, rel_tol=1e-5, abs_tol=1e-5
    ):
        raise ValueError("high-resolution RGB patch geometry is not integral")
    return side


def _edge_log_likelihood_ratio_at_pose_projection(
    *,
    scale: CandidateHighresRGBScalePrediction,
    candidate_projection_offsets_xy: torch.Tensor,
    candidate_projection_valid: torch.Tensor,
    missing_edge_log_likelihood_ratio: float,
    max_abs_log_likelihood_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one RGB source at external pose projections only."""

    missing = float(missing_edge_log_likelihood_ratio)
    cap = float(max_abs_log_likelihood_ratio)
    if not math.isfinite(missing) or not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("high-resolution RGB projection constants are invalid")
    offsets = torch.as_tensor(
        candidate_projection_offsets_xy,
        dtype=torch.float32,
        device=scale.joint_log_probabilities.device,
    )
    valid = torch.as_tensor(
        candidate_projection_valid,
        dtype=torch.bool,
        device=scale.joint_log_probabilities.device,
    )
    if (
        offsets.ndim != 4
        or offsets.shape[-1] != 2
        or valid.shape != offsets.shape[:-1]
        or offsets.shape[1:3] != scale.edge_usable.shape[:2]
        or not torch.isfinite(offsets).all()
    ):
        raise ValueError("high-resolution RGB pose projection inputs are invalid")
    local_log_probability, in_window = continuous_joint_log_probability_at_offsets(
        joint_log_probabilities=scale.joint_log_probabilities,
        offsets_xy=scale.offsets_xy,
        query_offsets_xy=offsets,
    )
    category_count = int(scale.joint_log_probabilities.shape[-1] - 1)
    neutral = -math.log(2.0 * float(category_count))
    raw = local_log_probability - neutral + scale.edge_log_likelihood_ratios.unsqueeze(0)
    bounded = bounded_log_likelihood_ratio(raw, max_abs_log_ratio=cap)
    usable = valid.unsqueeze(-1) & in_window & scale.edge_usable.unsqueeze(0)
    return torch.where(usable, bounded, torch.full_like(bounded, missing)), usable


def highres_rgb_edge_log_likelihood_ratio_at_pose_projection(
    *,
    prediction: CandidateHighresRGBMultiscalePrediction,
    candidate_projection_offsets_xy: torch.Tensor,
    candidate_projection_valid: torch.Tensor,
    source: str = "combined",
    source_weights: Mapping[str, float] | None = None,
    missing_edge_log_likelihood_ratio: float = 0.0,
    max_abs_log_likelihood_ratio: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, float]]:
    """Evaluate selected RGB density sources before any candidate mixture.

    This is the public edge-level counterpart of
    :func:`score_candidate_highres_rgb_multiscale_batch`.  It keeps frozen
    candidate/view mass outside the visual model so a later, separately
    calibrated identity factor can be combined only at the same externally
    projected candidate location.  An out-of-window projection remains the
    fixed missing value and never receives learned dustbin evidence.
    """

    if not isinstance(prediction, CandidateHighresRGBMultiscalePrediction):
        raise ValueError("high-resolution RGB edge scoring requires a visual prediction")
    offsets = torch.as_tensor(
        candidate_projection_offsets_xy,
        dtype=torch.float32,
        device=prediction.sources["fine"].joint_log_probabilities.device,
    )
    valid = torch.as_tensor(
        candidate_projection_valid,
        dtype=torch.bool,
        device=offsets.device,
    )
    if (
        offsets.ndim != 4
        or offsets.shape[-1] != 2
        or valid.shape != offsets.shape[:-1]
        or offsets.shape[0] == 0
        or offsets.shape[1:3] != prediction.edge_shape[:2]
        or not torch.isfinite(offsets).all()
    ):
        raise ValueError("high-resolution RGB edge pose projection inputs are invalid")
    weights = _source_weights(source=source, source_weights=source_weights)
    edge_total = torch.zeros(
        (offsets.shape[0], *prediction.edge_shape), dtype=torch.float32, device=offsets.device
    )
    usable_total = torch.zeros_like(edge_total, dtype=torch.bool)
    for name in CANDIDATE_HIGHRES_RGB_SOURCES:
        weight = float(weights[name])
        if weight <= 0.0:
            continue
        values, usable = _edge_log_likelihood_ratio_at_pose_projection(
            scale=prediction.sources[name],
            candidate_projection_offsets_xy=offsets,
            candidate_projection_valid=valid,
            missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
            max_abs_log_likelihood_ratio=float(max_abs_log_likelihood_ratio),
        )
        edge_total = edge_total + weight * values
        usable_total = usable_total | usable
    return edge_total, usable_total, weights


def highres_rgb_candidate_identity_log_likelihood_ratios(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateHighresRGBMultiscalePrediction,
    source: str = "combined",
    source_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, float]]:
    """Return a target-free candidate appearance LLR before pose evaluation.

    The high-resolution branch emits a visual quality LLR for every fixed
    ``(query point, top-L candidate, support view)`` edge.  This helper
    marginalizes only the immutable support-view mass, retaining missing views
    as a neutral factor.  It deliberately does not receive projected offsets,
    a pose, a residual, or an identity target, so it can be audited as a
    standalone candidate-identity expert.

    Fine and broad sources share RGB content and are therefore combined as the
    same conservative convex LLR blend used by the pose scorer, not as an
    independent likelihood product.  ``source='zero'`` is an exact neutral
    visual control and returns zero LLRs with no usable candidates.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("high-resolution RGB identity scoring requires a target-free runtime")
    if not isinstance(prediction, CandidateHighresRGBMultiscalePrediction):
        raise ValueError("high-resolution RGB identity scoring requires a visual prediction")
    device = prediction.sources["fine"].edge_log_likelihood_ratios.device
    active = runtime.to(device)
    if prediction.edge_shape != tuple(int(value) for value in active.support_image_indices.shape):
        raise ValueError("high-resolution RGB identity runtime and prediction layouts differ")
    weights = _source_weights(source=source, source_weights=source_weights)
    edge_total = torch.zeros(prediction.edge_shape, dtype=torch.float32, device=device)
    edge_usable = torch.zeros(prediction.edge_shape, dtype=torch.bool, device=device)
    for name in CANDIDATE_HIGHRES_RGB_SOURCES:
        source_weight = float(weights[name])
        if source_weight <= 0.0:
            continue
        scale = prediction.sources[name]
        values = scale.edge_log_likelihood_ratios.to(device=device, dtype=torch.float32)
        usable = scale.edge_usable.to(device=device)
        # A missing support edge retains its prescribed fixed view mass at a
        # neutral factor.  Renormalizing onto the surviving views would allow
        # a candidate to discard contrary or unavailable evidence.
        edge_total = edge_total + source_weight * torch.where(
            usable, values, torch.zeros_like(values)
        )
        edge_usable = edge_usable | usable
    view_weights = active.candidate_view_weights.to(dtype=edge_total.dtype)
    if (
        view_weights.shape != edge_total.shape
        or torch.any(view_weights < 0.0)
        or not torch.isfinite(view_weights).all()
    ):
        raise ValueError("high-resolution RGB identity fixed view weights are invalid")
    log_view_weights = torch.where(
        view_weights > 0.0,
        torch.log(view_weights),
        torch.full_like(view_weights, -torch.inf),
    )
    candidate_llr = torch.logsumexp(log_view_weights + edge_total, dim=2)
    candidate_usable = torch.any(edge_usable & (view_weights > 0.0), dim=2)
    candidate_llr = torch.where(candidate_usable, candidate_llr, torch.zeros_like(candidate_llr))
    if not torch.isfinite(candidate_llr).all():
        raise RuntimeError("high-resolution RGB candidate identity mixture is non-finite")
    return candidate_llr, candidate_usable, weights


def highres_rgb_candidate_identity_plus_null_logits(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateHighresRGBMultiscalePrediction,
    source: str = "combined",
    source_weights: Mapping[str, float] | None = None,
    candidate_prior_logit_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, float]]:
    """Combine target-free RGB candidate evidence with fixed top-L/null mass.

    The explicit null retains its frozen prior mass.  Visual evidence changes
    candidate logits only after the target-free forward; it never makes an
    unsupported candidate or a missing view into a learned positive signal.
    """

    prior_weight = float(candidate_prior_logit_weight)
    if not math.isfinite(prior_weight) or prior_weight < 0.0:
        raise ValueError("high-resolution RGB identity prior weight is invalid")
    candidate_llr, candidate_usable, weights = highres_rgb_candidate_identity_log_likelihood_ratios(
        runtime=runtime,
        prediction=prediction,
        source=source,
        source_weights=source_weights,
    )
    active = runtime.to(candidate_llr.device)
    prior = active.candidate_probabilities.to(dtype=candidate_llr.dtype)
    null = active.null_probabilities.to(dtype=candidate_llr.dtype)
    if (
        prior.shape != candidate_llr.shape
        or null.shape != (active.point_count,)
        or torch.any(prior < 0.0)
        or torch.any(null < 0.0)
        or torch.any(torch.abs(prior.sum(dim=1) + null - 1.0) > 1e-4)
    ):
        raise ValueError("high-resolution RGB identity fixed candidate/null mass is invalid")
    if prior_weight == 0.0:
        candidate_logits = candidate_llr
        null_logits = torch.zeros_like(null)
    else:
        candidate_logits = candidate_llr + prior_weight * torch.log(
            prior.clamp_min(torch.finfo(prior.dtype).tiny)
        )
        null_logits = prior_weight * torch.log(null.clamp_min(torch.finfo(null.dtype).tiny))
    return torch.cat((candidate_logits, null_logits[:, None]), dim=1), candidate_usable, weights


def score_candidate_highres_rgb_multiscale_batch(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateHighresRGBMultiscalePrediction,
    candidate_projection_offsets_xy: torch.Tensor,
    candidate_projection_valid: torch.Tensor,
    source: str = "combined",
    source_weights: Mapping[str, float] | None = None,
    candidate_prior_temperature: float = 1.0,
    missing_edge_log_likelihood_ratio: float = 0.0,
    max_abs_log_likelihood_ratio: float = 6.0,
) -> CandidateHighresRGBPoseScore:
    """Score fixed global top-L candidates under caller supplied poses."""

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("high-resolution RGB scoring requires a target-free runtime")
    if not isinstance(prediction, CandidateHighresRGBMultiscalePrediction):
        raise ValueError("high-resolution RGB scoring requires a visual prediction")
    device = prediction.sources["fine"].joint_log_probabilities.device
    active = runtime.to(device)
    if prediction.edge_shape != tuple(int(value) for value in active.support_image_indices.shape):
        raise ValueError("high-resolution RGB prediction and runtime layouts differ")
    tempered_candidates = temper_candidate_prior_probabilities(
        candidate_probabilities=active.candidate_probabilities,
        null_probabilities=active.null_probabilities,
        temperature=float(candidate_prior_temperature),
    )
    edge_total, usable_total, weights = highres_rgb_edge_log_likelihood_ratio_at_pose_projection(
        prediction=prediction,
        candidate_projection_offsets_xy=candidate_projection_offsets_xy,
        candidate_projection_valid=candidate_projection_valid,
        source=source,
        source_weights=source_weights,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
        max_abs_log_likelihood_ratio=float(max_abs_log_likelihood_ratio),
    )
    point_llr, candidate_llr = fixed_candidate_view_mixture_log_ratio(
        edge_log_likelihood_ratios=edge_total,
        edge_usable=usable_total,
        candidate_view_weights=active.candidate_view_weights,
        candidate_probabilities=tempered_candidates,
        null_probabilities=active.null_probabilities,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
    )
    return CandidateHighresRGBPoseScore(
        pose_log_likelihood_ratios=point_llr.mean(dim=1),
        point_log_likelihood_ratios=point_llr,
        candidate_log_likelihood_ratios=candidate_llr,
        edge_log_likelihood_ratios=edge_total,
        edge_usable=usable_total,
        source_weights=weights,
        candidate_prior_temperature=float(candidate_prior_temperature),
    )


def selected_candidate_highres_rgb_log_likelihood_ratio_at_offsets(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateHighresRGBMultiscalePrediction,
    point_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    offsets_xy: torch.Tensor,
    projection_valid: torch.Tensor | None = None,
    source: str = "combined",
    source_weights: Mapping[str, float] | None = None,
    missing_edge_log_likelihood_ratio: float = 0.0,
    max_abs_log_likelihood_ratio: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-view candidate score for direct train-only hard-repeat margins."""

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("selected high-resolution RGB scoring requires a target-free runtime")
    if not isinstance(prediction, CandidateHighresRGBMultiscalePrediction):
        raise ValueError("selected high-resolution RGB scoring requires a visual prediction")
    device = prediction.sources["fine"].joint_log_probabilities.device
    active = runtime.to(device)
    points = torch.as_tensor(point_indices, dtype=torch.long, device=device).reshape(-1)
    candidates = torch.as_tensor(candidate_indices, dtype=torch.long, device=device).reshape(-1)
    offsets = torch.as_tensor(offsets_xy, dtype=torch.float32, device=device)
    if projection_valid is None:
        valid = torch.ones((len(points),), dtype=torch.bool, device=device)
    else:
        valid = torch.as_tensor(projection_valid, dtype=torch.bool, device=device).reshape(-1)
    if (
        len(points) == 0
        or candidates.shape != points.shape
        or offsets.shape != (len(points), 2)
        or valid.shape != points.shape
        or torch.any(points < 0)
        or torch.any(points >= active.point_count)
        or torch.any(candidates < 0)
        or torch.any(candidates >= active.candidate_count)
        or not torch.isfinite(offsets).all()
    ):
        raise ValueError("selected high-resolution RGB score inputs are invalid")
    weights = _source_weights(source=source, source_weights=source_weights)
    edge_total: torch.Tensor | None = None
    usable_total: torch.Tensor | None = None
    for name in CANDIDATE_HIGHRES_RGB_SOURCES:
        source_weight = float(weights[name])
        if source_weight <= 0.0:
            continue
        scale = prediction.sources[name]
        selected_joint = scale.joint_log_probabilities[points, candidates].unsqueeze(1)
        projected = offsets.reshape(1, len(points), 1, 2)
        local, in_window = continuous_joint_log_probability_at_offsets(
            joint_log_probabilities=selected_joint,
            offsets_xy=scale.offsets_xy,
            query_offsets_xy=projected,
        )
        category_count = int(selected_joint.shape[-1] - 1)
        raw = (
            local[0, :, 0]
            + math.log(2.0 * float(category_count))
            + scale.edge_log_likelihood_ratios[points, candidates]
        )
        bounded = bounded_log_likelihood_ratio(
            raw, max_abs_log_ratio=float(max_abs_log_likelihood_ratio)
        )
        source_usable = (
            valid[:, None]
            & in_window[0, :, 0]
            & scale.edge_usable[points, candidates]
        )
        values = torch.where(
            source_usable,
            bounded,
            torch.full_like(bounded, float(missing_edge_log_likelihood_ratio)),
        )
        if edge_total is None:
            edge_total = source_weight * values
            usable_total = source_usable
        else:
            edge_total = edge_total + source_weight * values
            assert usable_total is not None
            usable_total = usable_total | source_usable
    if edge_total is None:
        edge_total = torch.zeros(
            (len(points), active.support_view_count), dtype=torch.float32, device=device
        )
        usable_total = torch.zeros_like(edge_total, dtype=torch.bool)
    assert usable_total is not None
    view_weights = active.candidate_view_weights[points, candidates]
    log_view_weights = torch.where(
        view_weights > 0.0,
        torch.log(view_weights),
        torch.full_like(view_weights, -torch.inf),
    )
    score = torch.logsumexp(edge_total + log_view_weights, dim=1)
    candidate_usable = valid & torch.any(usable_total & (view_weights > 0.0), dim=1)
    return score, candidate_usable


def highres_rgb_spatial_density_nll(
    *,
    scale_prediction: CandidateHighresRGBScalePrediction,
    target_offsets_xy: torch.Tensor,
    target_dustbin: torch.Tensor,
    target_supervised: torch.Tensor | None = None,
    dustbin_weight: float = 1.0,
    balance_observed_and_dustbin: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train an RGB density from train-only correct-pose targets.

    Dustbin supervision calibrates in-window visual confidence.  It is never
    consumed as positive evidence for an out-of-window pose projection.
    """

    weight = float(dustbin_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("high-resolution RGB dustbin weight is invalid")
    device = scale_prediction.joint_log_probabilities.device
    targets = torch.as_tensor(target_offsets_xy, dtype=torch.float32, device=device)
    dustbin = torch.as_tensor(target_dustbin, dtype=torch.bool, device=device)
    supervised = (
        torch.ones_like(dustbin)
        if target_supervised is None
        else torch.as_tensor(target_supervised, dtype=torch.bool, device=device)
    )
    expected = scale_prediction.joint_log_probabilities.shape[:2]
    if (
        targets.shape != (*expected, 2)
        or dustbin.shape != expected
        or supervised.shape != expected
        or torch.any(dustbin & ~supervised)
        or not torch.isfinite(targets).all()
    ):
        raise ValueError("high-resolution RGB density targets are invalid")
    local, in_window = continuous_joint_log_probability_at_offsets(
        joint_log_probabilities=scale_prediction.joint_log_probabilities,
        offsets_xy=scale_prediction.offsets_xy,
        query_offsets_xy=targets.unsqueeze(0),
    )
    dustbin_log_probability = scale_prediction.joint_log_probabilities[..., -1].unsqueeze(0)
    target_log_probability = torch.where(
        dustbin.unsqueeze(0).unsqueeze(-1), dustbin_log_probability, local
    )
    observed_usable = scale_prediction.edge_usable.unsqueeze(0) & in_window
    dustbin_usable = scale_prediction.edge_usable.unsqueeze(0)
    usable = torch.where(
        dustbin.unsqueeze(0).unsqueeze(-1), dustbin_usable, observed_usable
    ) & supervised.unsqueeze(0).unsqueeze(-1)
    observed = usable & ~dustbin.unsqueeze(0).unsqueeze(-1)
    dustbin_active = usable & dustbin.unsqueeze(0).unsqueeze(-1)
    negative = -target_log_probability
    if not bool(usable.any()):
        loss = scale_prediction.spatial_logits.sum() * 0.0
    elif bool(balance_observed_and_dustbin):
        values: list[torch.Tensor] = []
        denominator = 0.0
        if bool(observed.any()):
            values.append(negative[observed].mean())
            denominator += 1.0
        if bool(dustbin_active.any()) and weight > 0.0:
            values.append(weight * negative[dustbin_active].mean())
            denominator += weight
        loss = (
            torch.stack(values).sum() / max(denominator, 1e-12)
            if values
            else scale_prediction.spatial_logits.sum() * 0.0
        )
    else:
        per_edge_weight = torch.where(
            dustbin.unsqueeze(0).unsqueeze(-1),
            torch.full_like(negative, weight),
            torch.ones_like(negative),
        )
        loss = (negative * per_edge_weight)[usable].mean()
    return loss, {
        "active_edges": float(usable.sum().item()),
        "observed_edges": float(observed.sum().item()),
        "dustbin_edges": float(dustbin_active.sum().item()),
        "mean_nll": float(loss.detach().item()),
    }


def highres_rgb_target_free_point_quality(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateHighresRGBMultiscalePrediction,
    source: str = "combined",
) -> dict[str, torch.Tensor]:
    """Return target-free point quality for optional post-forward selection.

    This helper has no pose projection input.  A caller that uses it for a
    point budget must freeze the selected point rows before joining correct or
    wrong train-only pose targets, and must reuse those rows for permutation
    controls.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("high-resolution RGB quality requires a target-free runtime")
    if not isinstance(prediction, CandidateHighresRGBMultiscalePrediction):
        raise ValueError("high-resolution RGB quality requires a visual prediction")
    weights = _source_weights(source=source)
    device = prediction.sources["fine"].joint_log_probabilities.device
    active = runtime.to(device)
    point_count = active.point_count
    peakiness = torch.zeros((point_count,), dtype=torch.float32, device=device)
    non_dustbin = torch.zeros_like(peakiness)
    edge_llr = torch.zeros_like(peakiness)
    for name in CANDIDATE_HIGHRES_RGB_SOURCES:
        weight = float(weights[name])
        if weight <= 0.0:
            continue
        scale = prediction.sources[name]
        spatial = torch.exp(scale.joint_log_probabilities[..., :-1])
        mass = spatial.sum(dim=-1).clamp_min(torch.finfo(spatial.dtype).tiny)
        conditional = spatial / mass.unsqueeze(-1)
        entropy = -torch.sum(
            conditional * torch.log(conditional.clamp_min(torch.finfo(conditional.dtype).tiny)), dim=-1
        )
        source_peakiness = 1.0 - entropy / math.log(float(conditional.shape[-1]))
        mixture = (
            active.candidate_probabilities.unsqueeze(-1)
            * active.candidate_view_weights
            * scale.edge_usable.to(dtype=spatial.dtype)
        )
        mixture_mass = mixture.sum(dim=(1, 2)).clamp_min(torch.finfo(spatial.dtype).tiny)
        peakiness = peakiness + weight * (mixture * source_peakiness).sum(dim=(1, 2)) / mixture_mass
        non_dustbin = non_dustbin + weight * (
            mixture * mass
        ).sum(dim=(1, 2)) / mixture_mass
        edge_llr = edge_llr + weight * (
            mixture * scale.edge_log_likelihood_ratios
        ).sum(dim=(1, 2)) / mixture_mass
    return {
        "peakiness": peakiness,
        "non_dustbin_probability": non_dustbin,
        "edge_log_likelihood_ratio": edge_llr,
    }
