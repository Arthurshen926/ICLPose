"""Low-capacity risk gate for primitive-VFM SE(3) refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrimitiveRefinementGate:
    maximum_posterior: float
    minimum_phase_margin: float

    def select(self, posterior_max: float, phase_margin: float) -> bool:
        return bool(
            float(posterior_max) <= float(self.maximum_posterior)
            and float(phase_margin) >= float(self.minimum_phase_margin)
        )


@dataclass(frozen=True)
class PrimitiveRefinementGateSample:
    posterior_max: float
    phase_margin: float
    initial_translation_m: float
    initial_rotation_deg: float
    refined_translation_m: float
    refined_rotation_deg: float


def _metrics(
    samples: list[PrimitiveRefinementGateSample], gate: PrimitiveRefinementGate
) -> tuple[int, int, int, float, int]:
    values: list[tuple[float, float, bool]] = []
    for sample in samples:
        selected = gate.select(sample.posterior_max, sample.phase_margin)
        values.append(
            (
                sample.refined_translation_m if selected else sample.initial_translation_m,
                sample.refined_rotation_deg if selected else sample.initial_rotation_deg,
                selected,
            )
        )
    strict = sum(t <= 0.5 and r <= 5.0 for t, r, _ in values)
    loose = sum(t <= 1.0 and r <= 10.0 for t, r, _ in values)
    catastrophic = sum(t > 2.0 or r > 20.0 for t, r, _ in values)
    risk = sum(min(t / 0.5 + r / 5.0, 20.0) for t, r, _ in values)
    selected_count = sum(selected for _, _, selected in values)
    return strict, loose, -catastrophic, -float(risk), -selected_count


def fit_primitive_refinement_gate(
    samples: list[PrimitiveRefinementGateSample],
) -> PrimitiveRefinementGate:
    """Fit a two-threshold monotone gate with deterministic conservative ties."""

    if not samples:
        raise ValueError("primitive refinement gate requires calibration samples")
    posterior = sorted({float(sample.posterior_max) for sample in samples})
    margin = sorted({float(sample.phase_margin) for sample in samples})
    candidates = [PrimitiveRefinementGate(-1.0, float("inf"))]
    candidates.extend(
        PrimitiveRefinementGate(p + 1.0e-9, m - 1.0e-9)
        for p in posterior
        for m in margin
    )
    # max() is stable.  The no-refinement policy is first and therefore wins
    # an exact tie; the final tuple component additionally prefers fewer calls.
    return max(candidates, key=lambda gate: _metrics(samples, gate))


def primitive_refinement_gate_metrics(
    samples: list[PrimitiveRefinementGateSample], gate: PrimitiveRefinementGate
) -> dict[str, float | int]:
    strict, loose, negative_catastrophic, negative_risk, negative_count = _metrics(
        samples, gate
    )
    return {
        "query_count": int(len(samples)),
        "strict_count": int(strict),
        "within_1m_10deg_count": int(loose),
        "catastrophic_count": int(-negative_catastrophic),
        "clipped_normalized_risk_sum": float(-negative_risk),
        "selected_refinement_count": int(-negative_count),
    }
