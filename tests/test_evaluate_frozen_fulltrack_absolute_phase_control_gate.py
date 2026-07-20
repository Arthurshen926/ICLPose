from __future__ import annotations

import json

from feature_extract.tools.vfm.evaluate_frozen_fulltrack_absolute_phase_control_gate import (
    POSITION_CONTROL_FAMILY,
    VISUAL_FAMILIES,
    evaluate_absolute_phase_visual_control_gate,
)


def _family(*, passed: bool, ap_lift: float, hard_gap: float, wilson: float, wins: int, losses: int) -> dict:
    return {
        "predeclared_incremental_gate": {"passed": passed},
        "exact_registered_identity": {
            "baseline": {"candidate_pair_average_precision": 0.5},
            "probe": {"candidate_pair_average_precision": 0.5 + ap_lift},
        },
        "rank2_to_top1_wrong_hard_pairs": {
            "probe_pairwise": {
                "median_correct_minus_wrong": hard_gap,
                "win_rate_wilson95_lower": wilson,
            },
            "paired_rank": {"rank_win_count": wins, "rank_loss_count": losses},
        },
    }


def test_absolute_phase_visual_control_gate_requires_visual_improvement(tmp_path) -> None:
    families = {
        POSITION_CONTROL_FAMILY: _family(
            passed=False, ap_lift=0.01, hard_gap=-0.1, wilson=0.35, wins=7, losses=8
        )
    }
    for index, name in enumerate(VISUAL_FAMILIES):
        families[name] = _family(
            passed=index == 0,
            ap_lift=0.04,
            hard_gap=0.2,
            wilson=0.55,
            wins=12,
            losses=4,
        )
    source = tmp_path / "candidate_audit.json"
    output = tmp_path / "control_gate.json"
    source.write_text(
        json.dumps(
            {
                "stage": "audit_frozen_fulltrack_per_view_candidate_probe",
                "families": families,
            }
        )
    )
    result = evaluate_absolute_phase_visual_control_gate(
        validation_audit_summary=source, output_json=output
    )
    assert result["any_visual_family_passed"] is True
    assert result["visual_families"][VISUAL_FAMILIES[0]]["passed"] is True
    assert result["visual_families"][VISUAL_FAMILIES[1]]["passed"] is False
