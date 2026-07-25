from __future__ import annotations

import pytest

from feature_extract.tools.vfm.audit_candidate_multiscale_phase_identity_llr import (
    promotable_phase_identity_sources,
)
from feature_extract.tools.vfm.train_candidate_multiscale_phase_identity_llr import GATE_FORMAT


def _gate(*, radio_final: bool, combined: bool = False) -> dict[str, object]:
    sources: dict[str, object] = {}
    values = {
        "radio_final": radio_final,
        "radio_intermediate": False,
        "alike": False,
    }
    for name, passed in values.items():
        sources[name] = {"gate": {"passed": passed}}
    sources["combined"] = {"gate": {"passed": combined}}
    names = [name for name, passed in values.items() if passed]
    return {
        "format": GATE_FORMAT,
        "sources": sources,
        "independent_source_passes": values,
        "promotable_source_names": names,
        "combined_passed": combined,
        "passed": bool(combined or names),
    }


def test_source_specific_promotion_survives_a_failed_equal_weight_blend() -> None:
    assert promotable_phase_identity_sources(_gate(radio_final=True)) == ("radio_final",)


def test_source_specific_promotion_rejects_a_stale_declared_source() -> None:
    gate = _gate(radio_final=True)
    gate["promotable_source_names"] = []
    with pytest.raises(ValueError, match="disagrees"):
        promotable_phase_identity_sources(gate)


def test_source_specific_promotion_rejects_inconsistent_aggregate_state() -> None:
    gate = _gate(radio_final=False)
    gate["passed"] = True
    with pytest.raises(ValueError, match="aggregate"):
        promotable_phase_identity_sources(gate)
