"""Apply the predeclared cross-split gate to a frozen S1 candidate audit.

This gate is deliberately stricter than a single AP increase.  A candidate
appearance model is allowed to trigger expensive frozen pose scoring only if
both held-out splits improve its geometric and strict-identity diagnostics
without a paired-rank regression.  It never selects a feature family from the
test split; callers supply the already predeclared primary family explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_CANDIDATE_SPECIFIC_ATTRIBUTION_FAMILIES = frozenset(
    {
        "structured_candidate_specific_context_only",
        "cost_volume_candidate_specific_context_only",
        "cost_volume_radio_final_context_only",
        "wide_fullcorr_candidate_specific_context_only",
        "wide_fullcorr_radio_final_context_only",
        "global_context_candidate_specific_context_only",
        "global_context_support8_candidate_specific_context_only",
        "landmark_region_prototype_candidate_specific_context_only",
        "highres_landmark_region_prototype_candidate_specific_context_only",
        "multisource_landmark_region_candidate_specific_context_only",
        "multisource_landmark_region_candidate_specific_appearance_only",
        "dense_alike_local_mode_candidate_specific_context_only",
        "anchor_global_layout_context_only",
        "absolute_global_transport_context_only",
        "context_attention_multiscale_context_only",
    }
)

# A composite family can look useful solely because its legacy anchor is
# useful.  These predeclared families therefore cannot enter pose scoring
# unless the matching context-only counterfactual passed the same gate.
_REQUIRED_CANDIDATE_SPECIFIC_CONTROL_BY_PRIMARY = {
    "cost_volume_radio_final_large_context": "cost_volume_radio_final_context_only",
    "wide_fullcorr_radio_final_large_context": "wide_fullcorr_radio_final_context_only",
    "anchor_global_layout_with_anchor": "anchor_global_layout_context_only",
    "absolute_global_transport_with_anchor": "absolute_global_transport_context_only",
    "context_attention_multiscale_with_anchor": "context_attention_multiscale_context_only",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_summary", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--gate_mode",
        choices=("joint", "identity_only"),
        default="joint",
        help=(
            "joint requires one posterior to improve both geometry and identity; "
            "identity_only validates a separately supervised exact-track head"
        ),
    )
    parser.add_argument(
        "--candidate_specific_family",
        default="",
        help=(
            "Optional predeclared context-only attribution control. When set, "
            "both this family and --family must pass before pose scoring."
        ),
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _finite_metric(payload: Mapping[str, Any], key: str, *, context: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{context} lacks finite {key}")
    return float(value)


def _split_gate(split: Mapping[str, Any], *, split_name: str) -> dict[str, Any]:
    geometry = split.get("geometry_set")
    identity = split.get("exact_registered_identity")
    if not isinstance(geometry, Mapping) or not isinstance(identity, Mapping):
        raise ValueError(f"{split_name} audit lacks geometry/identity sections")
    geometry_base = geometry.get("baseline")
    geometry_probe = geometry.get("probe")
    geometry_paired = geometry.get("paired_rank")
    identity_base = identity.get("baseline")
    identity_probe = identity.get("probe")
    identity_paired = identity.get("paired_rank")
    if not all(
        isinstance(value, Mapping)
        for value in (
            geometry_base,
            geometry_probe,
            geometry_paired,
            identity_base,
            identity_probe,
            identity_paired,
        )
    ):
        raise ValueError(f"{split_name} audit has an incomplete baseline/probe contract")
    geometry_ap_delta = _finite_metric(
        geometry_probe, "candidate_pair_average_precision", context=split_name
    ) - _finite_metric(geometry_base, "candidate_pair_average_precision", context=split_name)
    geometry_nll_delta = _finite_metric(
        geometry_probe, "group_target_nll", context=split_name
    ) - _finite_metric(geometry_base, "group_target_nll", context=split_name)
    geometry_top1_delta = _finite_metric(
        geometry_probe, "top1_geometry_valid_rate_given_positive", context=split_name
    ) - _finite_metric(
        geometry_base, "top1_geometry_valid_rate_given_positive", context=split_name
    )
    identity_ap_delta = _finite_metric(
        identity_probe, "exact_candidate_pair_average_precision", context=split_name
    ) - _finite_metric(
        identity_base, "exact_candidate_pair_average_precision", context=split_name
    )
    geometry_wins = int(geometry_paired.get("rank_win_count", -1))
    geometry_losses = int(geometry_paired.get("rank_loss_count", -1))
    identity_wins = int(identity_paired.get("rank_win_count", -1))
    identity_losses = int(identity_paired.get("rank_loss_count", -1))
    if min(geometry_wins, geometry_losses, identity_wins, identity_losses) < 0:
        raise ValueError(f"{split_name} audit lacks paired rank counts")
    checks = {
        "geometry_ap_strictly_improved": geometry_ap_delta > 0.0,
        "geometry_set_nll_strictly_improved": geometry_nll_delta < 0.0,
        "geometry_top1_not_worse": geometry_top1_delta >= 0.0,
        "strict_identity_ap_strictly_improved": identity_ap_delta > 0.0,
        "geometry_paired_rank_wins_not_less_than_losses": geometry_wins >= geometry_losses,
        "identity_paired_rank_wins_not_less_than_losses": identity_wins >= identity_losses,
    }
    return {
        "deltas": {
            "geometry_candidate_ap": geometry_ap_delta,
            "geometry_set_nll": geometry_nll_delta,
            "geometry_top1": geometry_top1_delta,
            "strict_identity_ap": identity_ap_delta,
        },
        "paired_counts": {
            "geometry_rank_wins": geometry_wins,
            "geometry_rank_losses": geometry_losses,
            "identity_rank_wins": identity_wins,
            "identity_rank_losses": identity_losses,
        },
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _identity_split_gate(split: Mapping[str, Any], *, split_name: str) -> dict[str, Any]:
    """Gate a dedicated exact-track head without conflating it with geometry."""

    identity = split.get("exact_registered_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"{split_name} audit lacks exact registered identity")
    baseline = identity.get("baseline")
    probe = identity.get("probe")
    paired = identity.get("paired_rank")
    if not all(isinstance(value, Mapping) for value in (baseline, probe, paired)):
        raise ValueError(f"{split_name} identity audit has an incomplete contract")
    identity_ap_delta = _finite_metric(
        probe, "exact_candidate_pair_average_precision", context=split_name
    ) - _finite_metric(
        baseline, "exact_candidate_pair_average_precision", context=split_name
    )
    identity_top1_delta = _finite_metric(
        probe, "top1_exact_rate_given_retrieved", context=split_name
    ) - _finite_metric(
        baseline, "top1_exact_rate_given_retrieved", context=split_name
    )
    median_rank_delta = _finite_metric(
        probe, "median_exact_rank_when_retrieved", context=split_name
    ) - _finite_metric(
        baseline, "median_exact_rank_when_retrieved", context=split_name
    )
    p90_rank_delta = _finite_metric(
        probe, "p90_exact_rank_when_retrieved", context=split_name
    ) - _finite_metric(
        baseline, "p90_exact_rank_when_retrieved", context=split_name
    )
    wins = int(paired.get("rank_win_count", -1))
    losses = int(paired.get("rank_loss_count", -1))
    if min(wins, losses) < 0:
        raise ValueError(f"{split_name} identity audit lacks paired rank counts")
    checks = {
        "strict_identity_ap_strictly_improved": identity_ap_delta > 0.0,
        "identity_top1_not_worse": identity_top1_delta >= 0.0,
        "identity_median_rank_not_worse": median_rank_delta <= 0.0,
        "identity_p90_rank_not_worse": p90_rank_delta <= 0.0,
        "identity_paired_rank_wins_not_less_than_losses": wins >= losses,
    }
    return {
        "deltas": {
            "strict_identity_ap": identity_ap_delta,
            "identity_top1": identity_top1_delta,
            "identity_median_rank": median_rank_delta,
            "identity_p90_rank": p90_rank_delta,
        },
        "paired_counts": {"identity_rank_wins": wins, "identity_rank_losses": losses},
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _family_gate(
    audit: Mapping[str, Any],
    *,
    family: str,
    protocol_checks: Mapping[str, bool],
    gate_mode: str,
) -> dict[str, Any]:
    families = audit.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("candidate audit lacks family section")
    family_payload = families.get(str(family))
    if not isinstance(family_payload, Mapping) or not isinstance(
        family_payload.get("splits"), Mapping
    ):
        raise ValueError(f"candidate audit has no requested family {family!r}")
    split_gate = _split_gate if str(gate_mode) == "joint" else _identity_split_gate
    split_results = {
        split_name: split_gate(
            family_payload["splits"].get(split_name, {}), split_name=split_name
        )
        for split_name in ("validation", "test")
    }
    return {
        "family": str(family),
        "splits": split_results,
        "pass": bool(all(protocol_checks.values()) and all(
            result["pass"] for result in split_results.values()
        )),
    }


def evaluate_multiscale_candidate_probe_gate(
    audit: Mapping[str, Any],
    *,
    family: str,
    candidate_specific_family: str = "",
    gate_mode: str = "joint",
) -> dict[str, Any]:
    mode = str(gate_mode)
    if mode not in {"joint", "identity_only"}:
        raise ValueError("candidate probe gate mode is unsupported")
    protocol = audit.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("candidate audit lacks protocol section")
    soft_global_context_allowed = bool(
        protocol.get("whole_image_summary_or_global_used") is False
        or (
            protocol.get("whole_image_summary_or_global_used") is True
            and protocol.get("soft_global_context_factor_used") is True
            and protocol.get("global_context_hard_retrieval_or_candidate_reselection") is False
        )
    )
    protocol_checks = {
        "train_only_fit": bool(
            protocol.get("fit_uses_train_targets_only") is True
            or protocol.get("fit_uses_train_geometric_targets_only") is True
        ),
        "validation_test_joined_after_prediction": protocol.get(
            "prediction_artifact_frozen_before_validation_test_label_join"
        ) is True,
        "test_not_used_for_model_selection": protocol.get("test_used_for_model_selection") is False,
        # A global descriptor remains forbidden unless it is explicitly the
        # fixed candidate/support-view soft factor.  This preserves the
        # no-retrieval protocol while allowing the post-S1e absolute-context
        # diagnostic from entering the same frozen gate.
        "no_retrieval_or_global_summary": protocol.get("image_retrieval_or_submap_used") is False
        and soft_global_context_allowed,
    }
    if mode == "identity_only":
        protocol_checks["registered_identity_train_only"] = bool(
            protocol.get("fit_uses_train_registered_identity_targets_only") is True
            and protocol.get("training_supervision_mode") == "registered_track_identity"
            and protocol.get("prediction_probability_semantics")
            == "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one"
        )
    primary = _family_gate(
        audit,
        family=str(family),
        protocol_checks=protocol_checks,
        gate_mode=mode,
    )
    control_name = str(candidate_specific_family).strip()
    required_control_name = _REQUIRED_CANDIDATE_SPECIFIC_CONTROL_BY_PRIMARY.get(
        str(family)
    )
    if required_control_name and not control_name:
        raise ValueError(
            f"{family!r} requires the predeclared candidate-specific attribution "
            f"control {required_control_name!r}"
        )
    if required_control_name and control_name != required_control_name:
        raise ValueError(
            f"{family!r} requires attribution control {required_control_name!r}, "
            f"not {control_name!r}"
        )
    if control_name and control_name not in _CANDIDATE_SPECIFIC_ATTRIBUTION_FAMILIES:
        raise ValueError(
            "candidate-specific attribution control must be a predeclared "
            "context-only family"
        )
    attribution_control = (
        None
        if not control_name
        else _family_gate(
            audit,
            family=control_name,
            protocol_checks=protocol_checks,
            gate_mode=mode,
        )
    )
    passed = bool(primary["pass"] and (
        attribution_control is None or attribution_control["pass"]
    ))
    return {
        "stage": "frozen_multiscale_candidate_probe_cross_split_gate",
        "family": str(family),
        "gate_mode": mode,
        "protocol_checks": protocol_checks,
        "required_candidate_specific_attribution_control": required_control_name,
        "splits": primary["splits"],
        "candidate_specific_attribution_control": attribution_control,
        "pass": passed,
        "next_action": (
            (
                "run_fixed_hypothesis_pose_rank_gate_with_identity_prior"
                if mode == "identity_only"
                else "run_fixed_hypothesis_pose_rank_gate"
            )
            if passed
            else "stop_before_pose_scoring_and_report_nonseparable_absolute_evidence"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.audit_summary)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output}")
    audit = json.loads(source.read_text())
    result = evaluate_multiscale_candidate_probe_gate(
        audit,
        family=str(args.family),
        candidate_specific_family=str(args.candidate_specific_family),
        gate_mode=str(args.gate_mode),
    )
    result["audit_summary"] = str(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
