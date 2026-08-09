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
    coverage: float

    @property
    def phase_score(self) -> float:
        return 0.5 * (float(self.horizontal_phase) + float(self.vertical_phase))

    @property
    def score(self) -> float:
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
            "phase_score": float(self.phase_score),
            "coverage": float(self.coverage),
            "dual_band_score": float(self.score),
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
    if payload.get("artifact_type") != "goal_maplet_phase_readout_policy_v1":
        raise ValueError("not a Goal-Maplet phase-readout policy")
    names = tuple(str(value) for value in payload.get("component_names", ()))
    mean = np.asarray(payload.get("standardizer_mean", ()), dtype=np.float64)
    scale = np.asarray(payload.get("standardizer_scale", ()), dtype=np.float64)
    coefficient = np.asarray(payload.get("coefficient", ()), dtype=np.float64)
    if not names or mean.shape != scale.shape or mean.shape != coefficient.shape or mean.shape != (len(names),):
        raise ValueError("invalid phase-readout policy dimensions")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("invalid phase-readout standardizer")
    allowed = set(DualBandPhaseEvidence.__dataclass_fields__) | {"phase_score", "dual_band_score"}
    if any(name not in allowed for name in names):
        raise ValueError("phase-readout policy requests an unknown component")
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


def _directional_phase(
    query_unit: np.ndarray,
    render_unit: np.ndarray,
    valid: np.ndarray,
    *,
    delta_y: int,
    delta_x: int,
) -> float:
    mask = np.asarray(valid, dtype=bool)
    dy, dx = int(delta_y), int(delta_x)
    height, width = mask.shape
    if dy == 0 and dx == 0:
        raise ValueError("directional phase offset must fit the spatial grid")
    if abs(dy) >= height or abs(dx) >= width:
        return 0.0
    source_y = slice(max(-dy, 0), min(height - dy, height))
    target_y = slice(max(dy, 0), min(height + dy, height))
    source_x = slice(max(-dx, 0), min(width - dx, width))
    target_x = slice(max(dx, 0), min(width + dx, width))
    query_delta = query_unit[:, target_y, target_x] - query_unit[:, source_y, source_x]
    render_delta = render_unit[:, target_y, target_x] - render_unit[:, source_y, source_x]
    edge_valid = mask[target_y, target_x] & mask[source_y, source_x]
    query_norm = np.linalg.norm(query_delta, axis=0)
    render_norm = np.linalg.norm(render_delta, axis=0)
    informative = edge_valid & (query_norm >= 1.0e-4) & (render_norm >= 1.0e-4)
    cosine = np.sum(query_delta * render_delta, axis=0)
    cosine /= np.maximum(query_norm * render_norm, 1.0e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    # The denominator is every edge in the fixed grid, not just visible or
    # high-contrast edges.  Missing support therefore contributes zero.
    return float(np.sum(cosine[informative]) / max(float(edge_valid.size), 1.0))


def dual_band_phase_evidence(
    query_mapper: np.ndarray,
    rendered_mapper: np.ndarray,
    query_context: np.ndarray,
    rendered_context: np.ndarray,
    valid: np.ndarray,
) -> DualBandPhaseEvidence:
    """Compare identity and spatial phase without discrete correspondences."""

    mapper = _fixed_grid_cosine(query_mapper, rendered_mapper, valid)
    context = _fixed_grid_cosine(query_context, rendered_context, valid)
    query_unit = _normalize_map(query_mapper)
    render_unit = _normalize_map(rendered_mapper)
    horizontal = _directional_phase(
        query_unit, render_unit, valid, delta_y=0, delta_x=1,
    )
    vertical = _directional_phase(
        query_unit, render_unit, valid, delta_y=1, delta_x=0,
    )
    horizontal_step2 = _directional_phase(
        query_unit, render_unit, valid, delta_y=0, delta_x=2,
    )
    vertical_step2 = _directional_phase(
        query_unit, render_unit, valid, delta_y=2, delta_x=0,
    )
    diagonal_down = _directional_phase(
        query_unit, render_unit, valid, delta_y=1, delta_x=1,
    )
    diagonal_up = _directional_phase(
        query_unit, render_unit, valid, delta_y=1, delta_x=-1,
    )
    return DualBandPhaseEvidence(
        mapper_cosine=mapper,
        context_cosine=context,
        horizontal_phase=horizontal,
        vertical_phase=vertical,
        horizontal_phase_step2=horizontal_step2,
        vertical_phase_step2=vertical_step2,
        diagonal_down_phase=diagonal_down,
        diagonal_up_phase=diagonal_up,
        coverage=float(np.mean(np.asarray(valid, dtype=bool))),
    )
