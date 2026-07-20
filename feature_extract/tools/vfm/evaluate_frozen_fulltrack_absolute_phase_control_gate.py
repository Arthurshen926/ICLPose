"""Apply the predeclared visual-versus-position-control promotion gate.

The per-view candidate audit evaluates each train-frozen family separately.
This small post-audit guard prevents a full-image phase feature from being
treated as visual evidence when a matched anchor-mask position control explains
the same validation improvement.  A pass only permits the next frozen
hypothesis-ranking audit; it never promotes a pose or changes the main solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


VISUAL_FAMILIES = (
    "fixedprior_fulltrack_perview_absolute_phase_final_mixture",
    "fixedprior_fulltrack_perview_absolute_phase_intermediate_pca256_mixture",
    "fixedprior_fulltrack_perview_absolute_phase_alike_fpn_mixture",
    "fixedprior_fulltrack_perview_absolute_phase_multiscale_mixture",
)
POSITION_CONTROL_FAMILY = (
    "fixedprior_fulltrack_perview_absolute_phase_position_control_mixture"
)
ARTIFACT_FORMAT = "frozen_fulltrack_absolute_phase_visual_control_gate_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-audit-summary", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def _number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} is not numeric")
    return float(value)


def _metrics(family: Mapping[str, Any]) -> dict[str, float]:
    identity = family.get("exact_registered_identity")
    hard = family.get("rank2_to_top1_wrong_hard_pairs")
    if not isinstance(identity, Mapping) or not isinstance(hard, Mapping):
        raise ValueError("candidate audit family lacks validation metrics")
    baseline = identity.get("baseline")
    probe = identity.get("probe")
    hard_probe = hard.get("probe_pairwise")
    hard_paired = hard.get("paired_rank")
    if not all(isinstance(item, Mapping) for item in (baseline, probe, hard_probe, hard_paired)):
        raise ValueError("candidate audit family metrics are malformed")
    return {
        "candidate_ap_lift": _number(
            probe.get("candidate_pair_average_precision"), context="probe AP"
        )
        - _number(baseline.get("candidate_pair_average_precision"), context="baseline AP"),
        "hard_pair_correct_minus_wrong": _number(
            hard_probe.get("median_correct_minus_wrong"), context="hard-pair gap"
        ),
        "hard_pair_wilson95_lower": _number(
            hard_probe.get("win_rate_wilson95_lower"), context="hard-pair Wilson"
        ),
        "hard_pair_rank_win_minus_loss": _number(
            hard_paired.get("rank_win_count"), context="hard-pair rank wins"
        )
        - _number(hard_paired.get("rank_loss_count"), context="hard-pair rank losses"),
    }


def evaluate_absolute_phase_visual_control_gate(
    *, validation_audit_summary: Path, output_json: Path
) -> dict[str, Any]:
    source = Path(validation_audit_summary)
    output = Path(output_json)
    if output.exists():
        raise FileExistsError("refusing to overwrite absolute-phase control gate")
    payload = json.loads(source.read_text())
    if not isinstance(payload, Mapping) or payload.get("stage") != (
        "audit_frozen_fulltrack_per_view_candidate_probe"
    ):
        raise ValueError("input is not a frozen full-track validation candidate audit")
    families = payload.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("candidate audit lacks family results")
    required = {*VISUAL_FAMILIES, POSITION_CONTROL_FAMILY}
    if required.difference(families):
        raise ValueError("candidate audit lacks an absolute-phase visual/control family")
    control = families[POSITION_CONTROL_FAMILY]
    if not isinstance(control, Mapping):
        raise ValueError("position-control family is malformed")
    control_gate = control.get("predeclared_incremental_gate")
    if not isinstance(control_gate, Mapping) or not isinstance(control_gate.get("passed"), bool):
        raise ValueError("position-control family lacks the frozen candidate gate")
    control_metrics = _metrics(control)
    results: dict[str, Any] = {}
    for name in VISUAL_FAMILIES:
        family = families[name]
        if not isinstance(family, Mapping):
            raise ValueError("absolute-phase visual family is malformed")
        candidate_gate = family.get("predeclared_incremental_gate")
        if not isinstance(candidate_gate, Mapping) or not isinstance(
            candidate_gate.get("passed"), bool
        ):
            raise ValueError("absolute-phase visual family lacks the frozen candidate gate")
        visual_metrics = _metrics(family)
        checks = {
            "visual_candidate_gate_passed": bool(candidate_gate["passed"]),
            "position_control_candidate_gate_failed": not bool(control_gate["passed"]),
            "visual_ap_lift_exceeds_position_control": (
                visual_metrics["candidate_ap_lift"]
                > control_metrics["candidate_ap_lift"]
            ),
            "visual_hard_pair_gap_exceeds_position_control": (
                visual_metrics["hard_pair_correct_minus_wrong"]
                > control_metrics["hard_pair_correct_minus_wrong"]
            ),
            "visual_hard_pair_wilson_exceeds_position_control": (
                visual_metrics["hard_pair_wilson95_lower"]
                > control_metrics["hard_pair_wilson95_lower"]
            ),
            "visual_hard_pair_rank_wins_exceed_position_control": (
                visual_metrics["hard_pair_rank_win_minus_loss"]
                > control_metrics["hard_pair_rank_win_minus_loss"]
            ),
        }
        results[name] = {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "visual_metrics": visual_metrics,
        }
    result = {
        "format": ARTIFACT_FORMAT,
        "stage": "evaluate_frozen_fulltrack_absolute_phase_control_gate",
        "validation_audit_summary": str(source),
        "position_control_family": POSITION_CONTROL_FAMILY,
        "position_control_candidate_gate_passed": bool(control_gate["passed"]),
        "position_control_metrics": control_metrics,
        "visual_families": results,
        "any_visual_family_passed": bool(any(item["passed"] for item in results.values())),
        "policy": (
            "a pass permits only a separate frozen-hypothesis pose-ranking audit; "
            "it never promotes an inference pose or trains a pose solver"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate_absolute_phase_visual_control_gate(
        validation_audit_summary=Path(args.validation_audit_summary),
        output_json=Path(args.output_json),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
