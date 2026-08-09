"""Dual-band phase evidence regenerated from one canonical VFM field.

The identity band uses the existing contextual readout.  The phase band never
stores another map descriptor: it compares the canonical mapper field and its
local directional differences directly on the fixed rendered/query grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DualBandPhaseEvidence:
    mapper_cosine: float
    context_cosine: float
    horizontal_phase: float
    vertical_phase: float
    horizontal_phase_step2: float
    vertical_phase_step2: float
    diagonal_down_phase: float
    diagonal_up_phase: float
    horizontal_phase_visible: float
    vertical_phase_visible: float
    horizontal_phase_step2_visible: float
    vertical_phase_step2_visible: float
    diagonal_down_phase_visible: float
    diagonal_up_phase_visible: float
    horizontal_observability: float
    vertical_observability: float
    horizontal_step2_observability: float
    vertical_step2_observability: float
    diagonal_down_observability: float
    diagonal_up_observability: float
    jacobian_phase: float
    jacobian_phase_visible: float
    jacobian_observability: float
    jacobian_carrier_mass: float
    jacobian_log_scale_agreement: float
    feature_fraction_mean: float
    visibility_fraction_mean: float
    missing_fraction_mean: float
    background_fraction_mean: float
    dominant_surface_fraction_mean: float
    mixed_surface_fraction: float
    coverage: float

    @property
    def phase_score(self) -> float:
        return 0.5 * (float(self.horizontal_phase) + float(self.vertical_phase))

    @property
    def phase_visible(self) -> float:
        """Phase agreement conditioned on an edge being jointly observable."""

        return 0.5 * (
            float(self.horizontal_phase_visible) + float(self.vertical_phase_visible)
        )

    @property
    def phase_observability(self) -> float:
        """Fraction of horizontal/vertical grid edges carrying phase evidence."""

        return 0.5 * (
            float(self.horizontal_observability) + float(self.vertical_observability)
        )

    @property
    def legacy_dual_band_score(self) -> float:
        """Historical fixed mixture; active runtime must load a policy."""

        # Identity remains present, but phase-sensitive evidence receives most
        # of the mass.  All terms use a fixed full-grid/edge denominator, so a
        # pose cannot improve merely by rendering fewer difficult tokens.
        return (
            0.50 * float(self.mapper_cosine)
            + 0.20 * float(self.context_cosine)
            + 0.15 * float(self.horizontal_phase)
            + 0.15 * float(self.vertical_phase)
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "mapper_cosine": float(self.mapper_cosine),
            "context_cosine": float(self.context_cosine),
            "horizontal_phase": float(self.horizontal_phase),
            "vertical_phase": float(self.vertical_phase),
            "horizontal_phase_step2": float(self.horizontal_phase_step2),
            "vertical_phase_step2": float(self.vertical_phase_step2),
            "diagonal_down_phase": float(self.diagonal_down_phase),
            "diagonal_up_phase": float(self.diagonal_up_phase),
            "horizontal_phase_visible": float(self.horizontal_phase_visible),
            "vertical_phase_visible": float(self.vertical_phase_visible),
            "horizontal_phase_step2_visible": float(self.horizontal_phase_step2_visible),
            "vertical_phase_step2_visible": float(self.vertical_phase_step2_visible),
            "diagonal_down_phase_visible": float(self.diagonal_down_phase_visible),
            "diagonal_up_phase_visible": float(self.diagonal_up_phase_visible),
            "horizontal_observability": float(self.horizontal_observability),
            "vertical_observability": float(self.vertical_observability),
            "horizontal_step2_observability": float(self.horizontal_step2_observability),
            "vertical_step2_observability": float(self.vertical_step2_observability),
            "diagonal_down_observability": float(self.diagonal_down_observability),
            "diagonal_up_observability": float(self.diagonal_up_observability),
            "jacobian_phase": float(self.jacobian_phase),
            "jacobian_phase_visible": float(self.jacobian_phase_visible),
            "jacobian_observability": float(self.jacobian_observability),
            "jacobian_carrier_mass": float(self.jacobian_carrier_mass),
            "jacobian_log_scale_agreement": float(
                self.jacobian_log_scale_agreement
            ),
            "feature_fraction_mean": float(self.feature_fraction_mean),
            "visibility_fraction_mean": float(self.visibility_fraction_mean),
            "missing_fraction_mean": float(self.missing_fraction_mean),
            "background_fraction_mean": float(self.background_fraction_mean),
            "dominant_surface_fraction_mean": float(
                self.dominant_surface_fraction_mean
            ),
            "mixed_surface_fraction": float(self.mixed_surface_fraction),
            "phase_score": float(self.phase_score),
            "phase_visible": float(self.phase_visible),
            "phase_observability": float(self.phase_observability),
            "coverage": float(self.coverage),
            "legacy_dual_band_score": float(self.legacy_dual_band_score),
        }


@dataclass(frozen=True)
class PhaseReadoutPolicy:
    component_names: tuple[str, ...]
    standardizer_mean: np.ndarray
    standardizer_scale: np.ndarray
    coefficient: np.ndarray
    metadata: dict[str, object]

    def score(self, evidence: DualBandPhaseEvidence) -> float:
        values = evidence.as_dict()
        feature = np.asarray([values[name] for name in self.component_names], dtype=np.float64)
        normalized = (feature - self.standardizer_mean) / np.maximum(
            self.standardizer_scale, 1.0e-12,
        )
        return float(np.sum(normalized * self.coefficient))


def load_phase_readout_policy(path: Path) -> PhaseReadoutPolicy:
    payload = json.loads(Path(path).read_text())
    artifact_type = str(payload.get("artifact_type", ""))
    if artifact_type not in {
        "goal_maplet_phase_readout_policy_v1",
        "goal_maplet_phase_readout_policy_v2",
    }:
        raise ValueError("not a Goal-Maplet phase-readout policy")
    names = tuple(str(value) for value in payload.get("component_names", ()))
    mean = np.asarray(payload.get("standardizer_mean", ()), dtype=np.float64)
    scale = np.asarray(payload.get("standardizer_scale", ()), dtype=np.float64)
    coefficient = np.asarray(payload.get("coefficient", ()), dtype=np.float64)
    if not names or mean.shape != scale.shape or mean.shape != coefficient.shape or mean.shape != (len(names),):
        raise ValueError("invalid phase-readout policy dimensions")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("invalid phase-readout standardizer")
    allowed = set(DualBandPhaseEvidence.__dataclass_fields__) | {
        "phase_score", "phase_visible", "phase_observability",
        "legacy_dual_band_score",
    }
    if any(name not in allowed for name in names):
        raise ValueError("phase-readout policy requests an unknown component")
    if artifact_type == "goal_maplet_phase_readout_policy_v2":
        if str(payload.get("role", "")) != "phase_residual_only":
            raise ValueError("phase policy v2 must declare phase_residual_only role")
        if names != ("jacobian_phase_visible",):
            raise ValueError(
                "phase-residual policy v2 must consume exactly one orientation-equivariant "
                "conditional phase; identity and observability belong to the joint energy"
            )
        if coefficient[0] <= 0.0:
            raise ValueError("phase-residual policy v2 must be monotonic in phase agreement")
    return PhaseReadoutPolicy(names, mean, scale, coefficient, payload)


def _normalize_map(value: np.ndarray) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32)
    if feature.ndim != 3:
        raise ValueError("phase-preserving readout expects [C,H,W] features")
    return feature / np.maximum(np.linalg.norm(feature, axis=0, keepdims=True), 1.0e-8)


def _fixed_grid_cosine(
    query: np.ndarray,
    rendered: np.ndarray,
    valid: np.ndarray,
) -> float:
    query_unit = _normalize_map(query)
    render_unit = _normalize_map(rendered)
    mask = np.asarray(valid, dtype=bool)
    if mask.shape != query_unit.shape[1:] or render_unit.shape != query_unit.shape:
        raise ValueError("phase-preserving feature grids differ")
    cosine = np.clip(np.sum(query_unit * render_unit, axis=0), -1.0, 1.0)
    return float(np.sum(cosine[mask]) / max(float(mask.size), 1.0))


@dataclass(frozen=True)
class DirectionalPhaseStatistics:
    """Separate phase compatibility from whether phase is observable."""

    fixed_grid_score: float
    conditional_score: float
    observability: float
    informative_edge_count: int
    grid_edge_count: int


@dataclass(frozen=True)
class JacobianPhaseStatistics:
    """Common-orientation-equivariant VFM differential evidence."""

    fixed_grid_score: float
    conditional_score: float
    observability: float
    query_carrier_mass: float
    informative_token_count: int
    log_scale_agreement: float


def _directional_phase_statistics_normalized(
    query_unit: np.ndarray,
    render_unit: np.ndarray,
    valid: np.ndarray,
    *,
    delta_y: int,
    delta_x: int,
    edge_selector: np.ndarray | None = None,
) -> DirectionalPhaseStatistics:
    mask = np.asarray(valid, dtype=bool)
    dy, dx = int(delta_y), int(delta_x)
    height, width = mask.shape
    if dy == 0 and dx == 0:
        raise ValueError("directional phase offset must fit the spatial grid")
    if abs(dy) >= height or abs(dx) >= width:
        return DirectionalPhaseStatistics(0.0, 0.0, 0.0, 0, 0)
    source_y = slice(max(-dy, 0), min(height - dy, height))
    target_y = slice(max(dy, 0), min(height + dy, height))
    source_x = slice(max(-dx, 0), min(width - dx, width))
    target_x = slice(max(dx, 0), min(width + dx, width))
    query_delta = query_unit[:, target_y, target_x] - query_unit[:, source_y, source_x]
    render_delta = render_unit[:, target_y, target_x] - render_unit[:, source_y, source_x]
    edge_valid = mask[target_y, target_x] & mask[source_y, source_x]
    selected_edge = np.ones(edge_valid.shape, dtype=bool)
    if edge_selector is not None:
        selector = np.asarray(edge_selector, dtype=bool)
        if selector.shape != mask.shape:
            raise ValueError("phase edge selector differs from the feature grid")
        selected_edge = selector[target_y, target_x] & selector[source_y, source_x]
        edge_valid &= selected_edge
    query_norm = np.linalg.norm(query_delta, axis=0)
    render_norm = np.linalg.norm(render_delta, axis=0)
    informative = edge_valid & (query_norm >= 1.0e-4) & (render_norm >= 1.0e-4)
    cosine = np.sum(query_delta * render_delta, axis=0)
    cosine /= np.maximum(query_norm * render_norm, 1.0e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    informative_count = int(np.count_nonzero(informative))
    grid_count = int(np.count_nonzero(selected_edge))
    numerator = float(np.sum(cosine[informative]))
    # Keep the historical fixed-grid score for an exact replay of G19-B, but
    # expose its two distinct factors.  A downstream energy can now decide
    # independently whether low visibility is weak evidence or negative
    # evidence instead of silently baking that decision into phase agreement.
    return DirectionalPhaseStatistics(
        fixed_grid_score=numerator / max(float(grid_count), 1.0),
        conditional_score=numerator / max(float(informative_count), 1.0),
        observability=float(informative_count) / max(float(grid_count), 1.0),
        informative_edge_count=informative_count,
        grid_edge_count=grid_count,
    )


def directional_phase_statistics(
    query: np.ndarray,
    rendered: np.ndarray,
    valid: np.ndarray,
    *,
    delta_y: int,
    delta_x: int,
    edge_selector: np.ndarray | None = None,
) -> DirectionalPhaseStatistics:
    """Return conditional phase and observability for one image-grid offset."""

    query_unit = _normalize_map(query)
    render_unit = _normalize_map(rendered)
    if query_unit.shape != render_unit.shape:
        raise ValueError("phase-preserving feature grids differ")
    if np.asarray(valid).shape != query_unit.shape[1:]:
        raise ValueError("phase-preserving validity grid differs")
    return _directional_phase_statistics_normalized(
        query_unit, render_unit, valid, delta_y=delta_y, delta_x=delta_x,
        edge_selector=edge_selector,
    )


def _feature_jacobian(feature: np.ndarray) -> np.ndarray:
    """Central VFM derivative tensor with image-axis components last."""

    value = _normalize_map(feature)
    channels, height, width = value.shape
    gradient = np.zeros((channels, height, width, 2), dtype=np.float32)
    if width > 1:
        gradient[:, :, 0, 0] = value[:, :, 1] - value[:, :, 0]
        gradient[:, :, -1, 0] = value[:, :, -1] - value[:, :, -2]
    if width > 2:
        gradient[:, :, 1:-1, 0] = 0.5 * (value[:, :, 2:] - value[:, :, :-2])
    if height > 1:
        gradient[:, 0, :, 1] = value[:, 1, :] - value[:, 0, :]
        gradient[:, -1, :, 1] = value[:, -1, :] - value[:, -2, :]
    if height > 2:
        gradient[:, 1:-1, :, 1] = 0.5 * (value[:, 2:, :] - value[:, :-2, :])
    return gradient


def jacobian_phase_statistics(
    query: np.ndarray,
    rendered: np.ndarray,
    *,
    feature_fraction: np.ndarray | None = None,
    dominant_surface_fraction: np.ndarray | None = None,
) -> JacobianPhaseStatistics:
    """Compare VFM Jacobians with query-only carrier weights.

    The Frobenius product is invariant when a common image rotation changes
    both derivative bases.  Candidate-dependent visibility is returned as a
    separate scalar rather than being interpreted as phase disagreement.
    """

    query_gradient = _feature_jacobian(query)
    render_gradient = _feature_jacobian(rendered)
    if query_gradient.shape != render_gradient.shape:
        raise ValueError("phase-preserving feature grids differ")
    shape = query_gradient.shape[1:3]
    feature_mass = (
        np.ones(shape, dtype=np.float32)
        if feature_fraction is None
        else np.asarray(feature_fraction, dtype=np.float32)
    )
    purity = (
        np.ones(shape, dtype=np.float32)
        if dominant_surface_fraction is None
        else np.asarray(dominant_surface_fraction, dtype=np.float32)
    )
    if feature_mass.shape != shape or purity.shape != shape:
        raise ValueError("fractional phase support differs from the feature grid")
    if np.any(~np.isfinite(feature_mass)) or np.any(~np.isfinite(purity)):
        raise ValueError("fractional phase support is not finite")
    support = np.clip(feature_mass, 0.0, 1.0) * np.clip(purity, 0.0, 1.0)
    query_norm = np.linalg.norm(query_gradient, axis=(0, 3))
    render_norm = np.linalg.norm(render_gradient, axis=(0, 3))
    carrier = query_norm.astype(np.float64)
    informative = (query_norm >= 1.0e-4) & (render_norm >= 1.0e-4) & (support > 0.0)
    cosine = np.sum(query_gradient * render_gradient, axis=(0, 3))
    cosine /= np.maximum(query_norm * render_norm, 1.0e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    carrier_total = float(np.sum(carrier))
    observed_weight = carrier * support * informative
    observed_total = float(np.sum(observed_weight))
    conditional = float(np.sum(observed_weight * cosine) / max(observed_total, 1.0e-12))
    observability = observed_total / max(carrier_total, 1.0e-12)
    log_scale_residual = np.abs(np.log(
        np.maximum(render_norm, 1.0e-4) / np.maximum(query_norm, 1.0e-4)
    ))
    # A separate orientation-invariant metric measurement.  It is not part of
    # the phase policy: phase answers "same spatial direction?", whereas this
    # term measures the projected spatial scale needed for surface-normal
    # motion.  Larger (closer to zero) is better.
    log_scale_agreement = -float(
        np.sum(observed_weight * np.minimum(log_scale_residual, 5.0))
        / max(observed_total, 1.0e-12)
    )
    return JacobianPhaseStatistics(
        fixed_grid_score=conditional * observability,
        conditional_score=conditional,
        observability=observability,
        query_carrier_mass=carrier_total / max(float(carrier.size), 1.0),
        informative_token_count=int(np.count_nonzero(informative)),
        log_scale_agreement=log_scale_agreement,
    )


def _fractional_observation_means(
    valid: np.ndarray,
    *,
    feature_fraction: np.ndarray | None,
    visibility_fraction: np.ndarray | None,
    missing_fraction: np.ndarray | None,
    background_fraction: np.ndarray | None,
    dominant_surface_fraction: np.ndarray | None,
    mixed_surface: np.ndarray | None,
) -> dict[str, float]:
    spatial_shape = np.asarray(valid).shape

    def fraction(value: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        if result.shape != spatial_shape or np.any(~np.isfinite(result)):
            raise ValueError(f"fractional {name} grid differs from the feature grid")
        return np.clip(result, 0.0, 1.0)

    feature_mass = fraction(
        np.asarray(valid, dtype=np.float32)
        if feature_fraction is None else feature_fraction,
        "feature",
    )
    visibility_mass = fraction(
        feature_mass if visibility_fraction is None else visibility_fraction,
        "visibility",
    )
    missing_mass = fraction(
        np.zeros(spatial_shape, dtype=np.float32)
        if missing_fraction is None else missing_fraction,
        "missing",
    )
    background_mass = fraction(
        1.0 - visibility_mass
        if background_fraction is None else background_fraction,
        "background",
    )
    dominant_mass = fraction(
        np.ones(spatial_shape, dtype=np.float32)
        if dominant_surface_fraction is None else dominant_surface_fraction,
        "dominant-surface",
    )
    mixed = fraction(
        np.zeros(spatial_shape, dtype=np.float32)
        if mixed_surface is None else mixed_surface,
        "mixed-surface",
    )
    return {
        "feature_fraction_mean": float(np.mean(feature_mass)),
        "visibility_fraction_mean": float(np.mean(visibility_mass)),
        "missing_fraction_mean": float(np.mean(missing_mass)),
        "background_fraction_mean": float(np.mean(background_mass)),
        "dominant_surface_fraction_mean": float(np.mean(dominant_mass)),
        "mixed_surface_fraction": float(np.mean(mixed)),
    }


def orientation_equivariant_phase_evidence(
    query_mapper: np.ndarray,
    rendered_mapper: np.ndarray,
    valid: np.ndarray,
    *,
    feature_fraction: np.ndarray | None = None,
    visibility_fraction: np.ndarray | None = None,
    missing_fraction: np.ndarray | None = None,
    background_fraction: np.ndarray | None = None,
    dominant_surface_fraction: np.ndarray | None = None,
    mixed_surface: np.ndarray | None = None,
) -> DualBandPhaseEvidence:
    """Minimal active G20 phase path; legacy identity/directions are not computed."""

    jacobian = jacobian_phase_statistics(
        query_mapper,
        rendered_mapper,
        feature_fraction=(
            np.asarray(valid, dtype=np.float32)
            if feature_fraction is None else feature_fraction
        ),
        dominant_surface_fraction=dominant_surface_fraction,
    )
    observation = _fractional_observation_means(
        valid,
        feature_fraction=feature_fraction,
        visibility_fraction=visibility_fraction,
        missing_fraction=missing_fraction,
        background_fraction=background_fraction,
        dominant_surface_fraction=dominant_surface_fraction,
        mixed_surface=mixed_surface,
    )
    return DualBandPhaseEvidence(
        mapper_cosine=0.0,
        context_cosine=0.0,
        horizontal_phase=0.0,
        vertical_phase=0.0,
        horizontal_phase_step2=0.0,
        vertical_phase_step2=0.0,
        diagonal_down_phase=0.0,
        diagonal_up_phase=0.0,
        horizontal_phase_visible=0.0,
        vertical_phase_visible=0.0,
        horizontal_phase_step2_visible=0.0,
        vertical_phase_step2_visible=0.0,
        diagonal_down_phase_visible=0.0,
        diagonal_up_phase_visible=0.0,
        horizontal_observability=0.0,
        vertical_observability=0.0,
        horizontal_step2_observability=0.0,
        vertical_step2_observability=0.0,
        diagonal_down_observability=0.0,
        diagonal_up_observability=0.0,
        jacobian_phase=jacobian.fixed_grid_score,
        jacobian_phase_visible=jacobian.conditional_score,
        jacobian_observability=jacobian.observability,
        jacobian_carrier_mass=jacobian.query_carrier_mass,
        jacobian_log_scale_agreement=jacobian.log_scale_agreement,
        **observation,
        coverage=float(np.mean(np.asarray(valid, dtype=bool))),
    )


def dual_band_phase_evidence(
    query_mapper: np.ndarray,
    rendered_mapper: np.ndarray,
    query_context: np.ndarray,
    rendered_context: np.ndarray,
    valid: np.ndarray,
    *,
    feature_fraction: np.ndarray | None = None,
    visibility_fraction: np.ndarray | None = None,
    missing_fraction: np.ndarray | None = None,
    background_fraction: np.ndarray | None = None,
    dominant_surface_fraction: np.ndarray | None = None,
    mixed_surface: np.ndarray | None = None,
) -> DualBandPhaseEvidence:
    """Compare identity and spatial phase without discrete correspondences."""

    mapper = _fixed_grid_cosine(query_mapper, rendered_mapper, valid)
    context = _fixed_grid_cosine(query_context, rendered_context, valid)
    query_unit = _normalize_map(query_mapper)
    render_unit = _normalize_map(rendered_mapper)
    horizontal = _directional_phase_statistics_normalized(
        query_unit, render_unit, valid, delta_y=0, delta_x=1,
    )
    vertical = _directional_phase_statistics_normalized(
        query_unit, render_unit, valid, delta_y=1, delta_x=0,
    )
    horizontal_step2 = _directional_phase_statistics_normalized(
        query_unit, render_unit, valid, delta_y=0, delta_x=2,
    )
    vertical_step2 = _directional_phase_statistics_normalized(
        query_unit, render_unit, valid, delta_y=2, delta_x=0,
    )
    diagonal_down = _directional_phase_statistics_normalized(
        query_unit, render_unit, valid, delta_y=1, delta_x=1,
    )
    diagonal_up = _directional_phase_statistics_normalized(
        query_unit, render_unit, valid, delta_y=1, delta_x=-1,
    )
    jacobian = jacobian_phase_statistics(
        query_mapper,
        rendered_mapper,
        feature_fraction=(
            np.asarray(valid, dtype=np.float32)
            if feature_fraction is None else feature_fraction
        ),
        dominant_surface_fraction=dominant_surface_fraction,
    )
    observation = _fractional_observation_means(
        valid,
        feature_fraction=feature_fraction,
        visibility_fraction=visibility_fraction,
        missing_fraction=missing_fraction,
        background_fraction=background_fraction,
        dominant_surface_fraction=dominant_surface_fraction,
        mixed_surface=mixed_surface,
    )
    return DualBandPhaseEvidence(
        mapper_cosine=mapper,
        context_cosine=context,
        horizontal_phase=horizontal.fixed_grid_score,
        vertical_phase=vertical.fixed_grid_score,
        horizontal_phase_step2=horizontal_step2.fixed_grid_score,
        vertical_phase_step2=vertical_step2.fixed_grid_score,
        diagonal_down_phase=diagonal_down.fixed_grid_score,
        diagonal_up_phase=diagonal_up.fixed_grid_score,
        horizontal_phase_visible=horizontal.conditional_score,
        vertical_phase_visible=vertical.conditional_score,
        horizontal_phase_step2_visible=horizontal_step2.conditional_score,
        vertical_phase_step2_visible=vertical_step2.conditional_score,
        diagonal_down_phase_visible=diagonal_down.conditional_score,
        diagonal_up_phase_visible=diagonal_up.conditional_score,
        horizontal_observability=horizontal.observability,
        vertical_observability=vertical.observability,
        horizontal_step2_observability=horizontal_step2.observability,
        vertical_step2_observability=vertical_step2.observability,
        diagonal_down_observability=diagonal_down.observability,
        diagonal_up_observability=diagonal_up.observability,
        jacobian_phase=jacobian.fixed_grid_score,
        jacobian_phase_visible=jacobian.conditional_score,
        jacobian_observability=jacobian.observability,
        jacobian_carrier_mass=jacobian.query_carrier_mass,
        jacobian_log_scale_agreement=jacobian.log_scale_agreement,
        **observation,
        coverage=float(np.mean(np.asarray(valid, dtype=bool))),
    )
