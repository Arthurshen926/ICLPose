"""Gate a spatial-pyramid visual probe against its matched mask control.

The two probes are fitted independently on sequence-grouped train OOF folds.
This tool does not select a pose model: it only decides whether the visual
artifact contains enough *train-only* evidence beyond the original-crop-mask
control to justify one frozen validation candidate audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ARTIFACT_FORMAT = "frozen_fulltrack_spatial_pyramid_visual_control_gate_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-audit-summary", required=True)
    parser.add_argument("--control-audit-summary", required=True)
    parser.add_argument("--visual-family", required=True)
    parser.add_argument("--control-family", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--minimum-hard-pairs", type=int, default=50)
    parser.add_argument("--minimum-visual-hard-wilson-lower", type=float, default=0.50)
    parser.add_argument(
        "--minimum-visual-over-control-wilson-lower",
        type=float,
        default=0.02,
    )
    return parser.parse_args(argv)


def _number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    return float(value)


def _count(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return int(value)


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _load_train_audit(path: Path, *, family: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text())
    audit = _mapping(payload, context="audit summary")
    if (
        audit.get("stage") != "audit_frozen_fulltrack_per_view_candidate_probe"
        or audit.get("audit_split") != "train"
        or audit.get("diagnostic_only") is not True
        or audit.get("promotion_allowed") is not False
    ):
        raise ValueError("control gate requires a train-only diagnostic candidate audit")
    protocol = _mapping(audit.get("protocol"), context="audit protocol")
    required_protocol = {
        "audit_split_is_train_development_diagnostic": True,
        "model_fit_or_selection": False,
        "prediction_frozen_before_validation_target_join": True,
        "fixed_global_top_l": 20,
        "candidate_reselection": False,
        "all_real_sfm_support_observations_retained": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "pose_scoring": False,
    }
    if any(protocol.get(key) != value for key, value in required_protocol.items()):
        raise ValueError("candidate audit violates the frozen S1 protocol")
    families = _mapping(audit.get("families"), context="audit families")
    metrics = _mapping(families.get(str(family)), context=f"family {family!r}")
    gate = _mapping(metrics.get("predeclared_incremental_gate"), context="family gate")
    if gate.get("not_applicable") is not True or gate.get("passed") is not False:
        raise ValueError("train-only candidate audit unexpectedly claims promotion eligibility")
    return audit, metrics


def _metrics(family: Mapping[str, Any]) -> dict[str, float | int]:
    identity = _mapping(family.get("exact_registered_identity"), context="identity metrics")
    rank2 = _mapping(family.get("exact_registered_rank2_to_l"), context="rank2 metrics")
    hard = _mapping(family.get("rank2_to_top1_wrong_hard_pairs"), context="hard metrics")
    identity_base = _mapping(identity.get("baseline"), context="identity baseline")
    identity_probe = _mapping(identity.get("probe"), context="identity probe")
    identity_paired = _mapping(identity.get("paired_rank"), context="identity paired")
    rank2_base = _mapping(rank2.get("baseline"), context="rank2 baseline")
    rank2_probe = _mapping(rank2.get("probe"), context="rank2 probe")
    rank2_paired = _mapping(rank2.get("paired_rank"), context="rank2 paired")
    hard_probe = _mapping(hard.get("probe_pairwise"), context="hard probe")
    hard_paired = _mapping(hard.get("paired_rank"), context="hard paired")
    return {
        "baseline_identity_ap": _number(
            identity_base.get("candidate_pair_average_precision"), context="baseline identity AP"
        ),
        "baseline_identity_top1": _number(
            identity_base.get("top1_positive_rate_given_positive"), context="baseline identity top1"
        ),
        "baseline_identity_p90_rank": _number(
            identity_base.get("p90_first_positive_rank"), context="baseline identity P90 rank"
        ),
        "identity_ap_lift": _number(
            identity_probe.get("candidate_pair_average_precision"), context="probe identity AP"
        )
        - _number(identity_base.get("candidate_pair_average_precision"), context="baseline identity AP"),
        "identity_top1_lift": _number(
            identity_probe.get("top1_positive_rate_given_positive"), context="probe identity top1"
        )
        - _number(identity_base.get("top1_positive_rate_given_positive"), context="baseline identity top1"),
        "identity_p90_rank_delta": _number(
            identity_probe.get("p90_first_positive_rank"), context="probe identity P90 rank"
        )
        - _number(identity_base.get("p90_first_positive_rank"), context="baseline identity P90 rank"),
        "identity_rank_win_minus_loss": _count(
            identity_paired.get("rank_win_count"), context="identity rank wins"
        )
        - _count(identity_paired.get("rank_loss_count"), context="identity rank losses"),
        "baseline_rank2_median_rank": _number(
            rank2_base.get("median_first_positive_rank"), context="baseline rank2 median rank"
        ),
        "baseline_rank2_p90_rank": _number(
            rank2_base.get("p90_first_positive_rank"), context="baseline rank2 P90 rank"
        ),
        "rank2_median_rank_delta": _number(
            rank2_probe.get("median_first_positive_rank"), context="probe rank2 median rank"
        )
        - _number(rank2_base.get("median_first_positive_rank"), context="baseline rank2 median rank"),
        "rank2_p90_rank_delta": _number(
            rank2_probe.get("p90_first_positive_rank"), context="probe rank2 P90 rank"
        )
        - _number(rank2_base.get("p90_first_positive_rank"), context="baseline rank2 P90 rank"),
        "rank2_rank_win_minus_loss": _count(
            rank2_paired.get("rank_win_count"), context="rank2 wins"
        )
        - _count(rank2_paired.get("rank_loss_count"), context="rank2 losses"),
        "hard_pair_count": _count(hard_probe.get("usable_pair_count"), context="hard pair count"),
        "hard_pair_wilson95_lower": _number(
            hard_probe.get("win_rate_wilson95_lower"), context="hard pair Wilson lower"
        ),
        "hard_pair_correct_minus_wrong": _number(
            hard_probe.get("median_correct_minus_wrong"), context="hard pair median gap"
        ),
        "hard_rank_win_minus_loss": _count(
            hard_paired.get("rank_win_count"), context="hard rank wins"
        )
        - _count(hard_paired.get("rank_loss_count"), context="hard rank losses"),
    }


def _same_baseline(visual: Mapping[str, float | int], control: Mapping[str, float | int]) -> bool:
    keys = (
        "baseline_identity_ap",
        "baseline_identity_top1",
        "baseline_identity_p90_rank",
        "baseline_rank2_median_rank",
        "baseline_rank2_p90_rank",
        "hard_pair_count",
    )
    return all(abs(float(visual[key]) - float(control[key])) <= 1e-12 for key in keys)


def evaluate_paired_visual_control_gate(
    *,
    visual_audit_summary: Path,
    control_audit_summary: Path,
    visual_family: str,
    control_family: str,
    output_json: Path,
    artifact_format: str,
    stage: str,
    minimum_hard_pairs: int = 50,
    minimum_visual_hard_wilson_lower: float = 0.50,
    minimum_visual_over_control_wilson_lower: float = 0.02,
) -> dict[str, Any]:
    """Gate one target-free visual family against its paired non-visual control."""

    if (
        not str(artifact_format).strip()
        or not str(stage).strip()
        or int(minimum_hard_pairs) <= 0
        or not 0.0 <= float(minimum_visual_hard_wilson_lower) <= 1.0
        or float(minimum_visual_over_control_wilson_lower) < 0.0
    ):
        raise ValueError("paired visual/control gate configuration is invalid")
    output = Path(output_json)
    if output.exists():
        raise FileExistsError("refusing to overwrite paired visual/control gate")
    visual_audit, visual_family_payload = _load_train_audit(
        Path(visual_audit_summary), family=str(visual_family)
    )
    control_audit, control_family_payload = _load_train_audit(
        Path(control_audit_summary), family=str(control_family)
    )
    if (
        visual_audit.get("identity_supervision_colmap_images_sha256")
        != control_audit.get("identity_supervision_colmap_images_sha256")
        or visual_audit.get("row_count") != control_audit.get("row_count")
        or visual_audit.get("query_count") != control_audit.get("query_count")
    ):
        raise ValueError("visual and control OOF audits do not share the same train protocol")
    visual = _metrics(visual_family_payload)
    control = _metrics(control_family_payload)
    same_baseline = _same_baseline(visual, control)
    checks = {
        "same_immutable_baseline": same_baseline,
        "visual_identity_ap_positive": float(visual["identity_ap_lift"]) > 0.0,
        "visual_identity_ap_exceeds_control": float(visual["identity_ap_lift"])
        > float(control["identity_ap_lift"]),
        "visual_identity_top1_not_worse": float(visual["identity_top1_lift"]) >= 0.0,
        "visual_identity_p90_rank_not_worse": float(visual["identity_p90_rank_delta"]) <= 0.0,
        "visual_identity_rank_wins_exceed_losses": int(visual["identity_rank_win_minus_loss"]) > 0,
        "visual_rank2_median_not_worse": float(visual["rank2_median_rank_delta"]) <= 0.0,
        "visual_rank2_p90_not_worse": float(visual["rank2_p90_rank_delta"]) <= 0.0,
        "visual_rank2_wins_exceed_losses": int(visual["rank2_rank_win_minus_loss"]) > 0,
        "hard_pair_support_sufficient": int(visual["hard_pair_count"]) >= int(minimum_hard_pairs),
        "visual_hard_gap_positive": float(visual["hard_pair_correct_minus_wrong"]) > 0.0,
        "visual_hard_wilson_passes": float(visual["hard_pair_wilson95_lower"])
        > float(minimum_visual_hard_wilson_lower),
        "visual_hard_wins_exceed_losses": int(visual["hard_rank_win_minus_loss"]) > 0,
        "visual_hard_gap_exceeds_control": float(visual["hard_pair_correct_minus_wrong"])
        > float(control["hard_pair_correct_minus_wrong"]),
        "visual_hard_wilson_exceeds_control": float(visual["hard_pair_wilson95_lower"])
        >= float(control["hard_pair_wilson95_lower"])
        + float(minimum_visual_over_control_wilson_lower),
        "visual_hard_rank_wins_exceed_control": int(visual["hard_rank_win_minus_loss"])
        > int(control["hard_rank_win_minus_loss"]),
        # A mask control is disqualifying only if it itself satisfies the
        # complete hard-pair criterion.  Having a few paired-rank rescues is
        # not sufficient: a control whose correct-vs-wrong pairwise evidence
        # remains below chance must not be mislabeled as independently valid.
        "mask_control_does_not_independently_pass_hard_pair": not (
            int(control["hard_pair_count"]) >= int(minimum_hard_pairs)
            and float(control["hard_pair_correct_minus_wrong"]) > 0.0
            and float(control["hard_pair_wilson95_lower"])
            > float(minimum_visual_hard_wilson_lower)
            and int(control["hard_rank_win_minus_loss"]) > 0
        ),
    }
    result = {
        "format": str(artifact_format),
        "stage": str(stage),
        "visual_audit_summary": str(Path(visual_audit_summary)),
        "control_audit_summary": str(Path(control_audit_summary)),
        "visual_family": str(visual_family),
        "control_family": str(control_family),
        "thresholds": {
            "minimum_hard_pairs": int(minimum_hard_pairs),
            "minimum_visual_hard_wilson_lower": float(minimum_visual_hard_wilson_lower),
            "minimum_visual_over_control_wilson_lower": float(
                minimum_visual_over_control_wilson_lower
            ),
        },
        "visual_metrics": visual,
        "control_metrics": control,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "promotion_allowed": False,
        "policy": (
            "a pass only permits one frozen validation candidate audit for the predeclared "
            "visual family; it does not rank poses, change inference, or train a pose solver"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def evaluate_spatial_pyramid_visual_control_gate(
    *,
    visual_audit_summary: Path,
    control_audit_summary: Path,
    visual_family: str,
    control_family: str,
    output_json: Path,
    minimum_hard_pairs: int = 50,
    minimum_visual_hard_wilson_lower: float = 0.50,
    minimum_visual_over_control_wilson_lower: float = 0.02,
) -> dict[str, Any]:
    """Backward-compatible spatial-pyramid wrapper around the generic gate."""

    return evaluate_paired_visual_control_gate(
        visual_audit_summary=visual_audit_summary,
        control_audit_summary=control_audit_summary,
        visual_family=visual_family,
        control_family=control_family,
        output_json=output_json,
        artifact_format=ARTIFACT_FORMAT,
        stage="evaluate_frozen_fulltrack_spatial_pyramid_control_gate",
        minimum_hard_pairs=minimum_hard_pairs,
        minimum_visual_hard_wilson_lower=minimum_visual_hard_wilson_lower,
        minimum_visual_over_control_wilson_lower=minimum_visual_over_control_wilson_lower,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate_spatial_pyramid_visual_control_gate(
        visual_audit_summary=Path(args.visual_audit_summary),
        control_audit_summary=Path(args.control_audit_summary),
        visual_family=args.visual_family,
        control_family=args.control_family,
        output_json=Path(args.output_json),
        minimum_hard_pairs=args.minimum_hard_pairs,
        minimum_visual_hard_wilson_lower=args.minimum_visual_hard_wilson_lower,
        minimum_visual_over_control_wilson_lower=args.minimum_visual_over_control_wilson_lower,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
