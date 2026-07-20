from __future__ import annotations

import json
from pathlib import Path

from feature_extract.tools.vfm.evaluate_frozen_fulltrack_spatial_pyramid_control_gate import (
    evaluate_spatial_pyramid_visual_control_gate,
)


VISUAL_FAMILY = "fixedprior_fulltrack_perview_spatial_pyramid_multiscale_mixture"
CONTROL_FAMILY = "fixedprior_fulltrack_perview_spatial_pyramid_mask_control_multiscale_mixture"


def _family(*, visual: bool) -> dict[str, object]:
    ap = 0.57 if visual else 0.41
    top1 = 0.54 if visual else 0.40
    p90 = 7.0 if visual else 10.0
    rank2_median = 2.0 if visual else 5.0
    rank2_p90 = 8.0 if visual else 12.0
    hard_wilson = 0.64 if visual else 0.42
    hard_gap = 0.30 if visual else -0.05
    return {
        "predeclared_incremental_gate": {
            "not_applicable": True,
            "passed": False,
        },
        "exact_registered_identity": {
            "baseline": {
                "candidate_pair_average_precision": 0.40,
                "top1_positive_rate_given_positive": 0.40,
                "p90_first_positive_rank": 10.0,
            },
            "probe": {
                "candidate_pair_average_precision": ap,
                "top1_positive_rate_given_positive": top1,
                "p90_first_positive_rank": p90,
            },
            "paired_rank": {
                "rank_win_count": 30 if visual else 5,
                "rank_loss_count": 3 if visual else 9,
            },
        },
        "exact_registered_rank2_to_l": {
            "baseline": {
                "median_first_positive_rank": 5.0,
                "p90_first_positive_rank": 12.0,
            },
            "probe": {
                "median_first_positive_rank": rank2_median,
                "p90_first_positive_rank": rank2_p90,
            },
            "paired_rank": {
                "rank_win_count": 25 if visual else 4,
                "rank_loss_count": 2 if visual else 8,
            },
        },
        "rank2_to_top1_wrong_hard_pairs": {
            "probe_pairwise": {
                "usable_pair_count": 80,
                "win_rate_wilson95_lower": hard_wilson,
                "median_correct_minus_wrong": hard_gap,
            },
            "paired_rank": {
                "rank_win_count": 22 if visual else 3,
                "rank_loss_count": 2 if visual else 7,
            },
        },
    }


def _audit(*, family: str, visual: bool) -> dict[str, object]:
    return {
        "stage": "audit_frozen_fulltrack_per_view_candidate_probe",
        "audit_split": "train",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "row_count": 100,
        "query_count": 12,
        "identity_supervision_colmap_images_sha256": "imageshash",
        "protocol": {
            "audit_split_is_train_development_diagnostic": True,
            "model_fit_or_selection": False,
            "prediction_frozen_before_validation_target_join": True,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "all_real_sfm_support_observations_retained": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "pose_scoring": False,
        },
        "families": {family: _family(visual=visual)},
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_visual_control_gate_passes_only_train_oof_visual_evidence(tmp_path: Path) -> None:
    result = evaluate_spatial_pyramid_visual_control_gate(
        visual_audit_summary=_write(
            tmp_path / "visual.json", _audit(family=VISUAL_FAMILY, visual=True)
        ),
        control_audit_summary=_write(
            tmp_path / "control.json", _audit(family=CONTROL_FAMILY, visual=False)
        ),
        visual_family=VISUAL_FAMILY,
        control_family=CONTROL_FAMILY,
        output_json=tmp_path / "gate.json",
    )
    assert result["passed"] is True
    assert result["promotion_allowed"] is False
    assert "validation candidate audit" in str(result["policy"])


def test_visual_control_gate_rejects_a_mask_control_that_explains_the_hard_gain(
    tmp_path: Path,
) -> None:
    visual = _audit(family=VISUAL_FAMILY, visual=True)
    control = _audit(family=CONTROL_FAMILY, visual=True)
    result = evaluate_spatial_pyramid_visual_control_gate(
        visual_audit_summary=_write(tmp_path / "visual.json", visual),
        control_audit_summary=_write(tmp_path / "control.json", control),
        visual_family=VISUAL_FAMILY,
        control_family=CONTROL_FAMILY,
        output_json=tmp_path / "gate.json",
    )
    assert result["passed"] is False
    assert result["checks"]["mask_control_does_not_independently_pass_hard_pair"] is False


def test_visual_control_gate_does_not_misclassify_noisy_control_rank_rescues(
    tmp_path: Path,
) -> None:
    """A control below chance on pairs is not an independent hard-pair pass."""

    visual = _audit(family=VISUAL_FAMILY, visual=True)
    control = _audit(family=CONTROL_FAMILY, visual=False)
    control_family = control["families"][CONTROL_FAMILY]
    control_family["rank2_to_top1_wrong_hard_pairs"]["paired_rank"] = {
        "rank_win_count": 20,
        "rank_loss_count": 2,
    }
    result = evaluate_spatial_pyramid_visual_control_gate(
        visual_audit_summary=_write(tmp_path / "visual.json", visual),
        control_audit_summary=_write(tmp_path / "control.json", control),
        visual_family=VISUAL_FAMILY,
        control_family=CONTROL_FAMILY,
        output_json=tmp_path / "gate.json",
    )
    assert result["passed"] is True
    assert result["checks"]["mask_control_does_not_independently_pass_hard_pair"] is True
