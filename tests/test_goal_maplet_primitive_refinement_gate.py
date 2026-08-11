from feature_extract.vfm.localization_goal_maplet.primitive_refinement_gate import (
    PrimitiveRefinementGateSample,
    fit_primitive_refinement_gate,
    primitive_refinement_gate_metrics,
)


def _sample(posterior, margin, initial_t, refined_t):
    return PrimitiveRefinementGateSample(
        posterior_max=posterior,
        phase_margin=margin,
        initial_translation_m=initial_t,
        initial_rotation_deg=1.0,
        refined_translation_m=refined_t,
        refined_rotation_deg=1.0,
    )


def test_gate_learns_ambiguity_and_margin_conjunction():
    samples = [
        _sample(0.30, 0.02, 0.60, 0.40),
        _sample(0.30, 0.001, 0.40, 0.60),
        _sample(0.80, 0.02, 0.40, 0.60),
    ]
    gate = fit_primitive_refinement_gate(samples)
    assert gate.select(0.30, 0.02)
    assert not gate.select(0.30, 0.001)
    assert not gate.select(0.80, 0.02)
    metrics = primitive_refinement_gate_metrics(samples, gate)
    assert metrics["strict_count"] == 3


def test_gate_defaults_to_no_refinement_when_nothing_improves():
    samples = [_sample(0.2, 0.02, 0.3, 0.7), _sample(0.8, 0.01, 0.2, 0.8)]
    gate = fit_primitive_refinement_gate(samples)
    assert not any(gate.select(sample.posterior_max, sample.phase_margin) for sample in samples)
