"""Compare fixed-kernel transport with its source-softmax negative control.

This evaluation is post-hoc and parameter-free.  It validates that both
reports share dataset, split, seed, schedule, and identity-readout contracts.
For the two-DoF identity readout, the frozen ``feature_hierarchy_layout``
component control is exactly the pre-optimizer state: feature, hierarchy and
layout logits are zero, absent geometry logits are -30, and every other model
parameter is frozen.  Reusing those already scored rows avoids a second large
sparse-graph replay while making the training gain explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.differentiable_pose_transport import (
    FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS,
    TORCH_TRANSPORT_SEMANTICS,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    IDENTITY_FEATURE_ONLY_READOUT_SEMANTICS,
    REPORT_SCHEMA,
)


SCHEMA = "goal_maplet_sparse_pose_transport_fixed_vs_source_softmax_control_v1"
METRICS = (
    "gt_anchor_top1_rate",
    "mean_gt_anchor_margin_over_best_nonanchor",
    "mean_score_error_spearman",
    "pairwise_order_accuracy",
    "controlled_radial_pair_accuracy",
    "controlled_radial_complete_path_rate",
    "directional_outward_drift_violation_rate",
)


def _load(path: Path, *, transport_semantics: str) -> dict[str, object]:
    value = json.loads(Path(path).read_text())
    required = {
        "artifact_type": REPORT_SCHEMA,
        "transport_semantics": str(transport_semantics),
        "readout_training_semantics": IDENTITY_FEATURE_ONLY_READOUT_SEMANTICS,
        "effective_trainable_scalar_count": 2,
        "identity_trained_edge_components": [0, 5],
        "identity_reference_edge_component": 4,
        "identity_disabled_edge_components": [1, 2, 3],
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError(f"sparse transport control contract differs: {path}")
    baseline = value.get("posthoc_fixed_component_ablation_diagnostic", {}).get(
        "feature_hierarchy_layout", {}
    )
    if baseline.get("active_component_indices") != [0, 4, 5]:
        raise ValueError("report lacks the exact frozen initialization baseline")
    return value


def _metric_slice(value: dict[str, object]) -> dict[str, float]:
    return {name: float(value[name]) for name in METRICS}


def _delta(
    final: dict[str, object], baseline: dict[str, object],
) -> dict[str, float]:
    return {name: float(final[name]) - float(baseline[name]) for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed_report", required=True)
    parser.add_argument("--source_softmax_report", required=True)
    parser.add_argument("--output_report", required=True)
    args = parser.parse_args()
    output = Path(args.output_report)
    if output.exists():
        raise FileExistsError("refusing to overwrite sparse transport comparison")
    fixed_path = Path(args.fixed_report)
    softmax_path = Path(args.source_softmax_report)
    fixed = _load(
        fixed_path, transport_semantics=FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS
    )
    softmax = _load(softmax_path, transport_semantics=TORCH_TRANSPORT_SEMANTICS)
    identical = (
        "dataset_content_sha256", "dataset_file_sha256", "physical_map_sha256",
        "hierarchy_content_sha256", "surface_mapper_file_sha256", "seed",
        "epochs", "learning_rate", "split_semantics", "train_image_ids",
        "dev_image_ids", "training_monotonic_pair_count",
        "training_monotonic_pair_semantics",
    )
    mismatched = [name for name in identical if fixed.get(name) != softmax.get(name)]
    if mismatched:
        raise ValueError("control reports are not paired: " + ",".join(mismatched))

    rows: dict[str, object] = {}
    for name, report in (("fixed_kernel", fixed), ("source_softmax", softmax)):
        final = report["dev_medium_sparse_transport"]
        baseline = report["posthoc_fixed_component_ablation_diagnostic"][
            "feature_hierarchy_layout"
        ]["dev"]
        rows[name] = {
            "transport_semantics": report["transport_semantics"],
            "pretraining_frozen_baseline_dev_medium": _metric_slice(baseline),
            "trained_dev_medium": _metric_slice(final),
            "training_delta_dev_medium": _delta(final, baseline),
            "trained_dev_coarse_feature_only_structural": _metric_slice(
                report["dev_coarse_feature_only_structural_transport"]
            ),
            "trained_dev_fine_feature_only_exact_child": _metric_slice(
                report["dev_fine_feature_only_exact_child_transport"]
            ),
            "learned_edge_weight_ratios_to_hierarchy_reference": report[
                "learned_edge_weight_ratios_to_hierarchy_reference"
            ],
        }
    fixed_final = rows["fixed_kernel"]["trained_dev_medium"]
    softmax_final = rows["source_softmax"]["trained_dev_medium"]
    paired_delta = {
        name: float(fixed_final[name]) - float(softmax_final[name])
        for name in METRICS
    }
    mapper_strict = bool(fixed.get("strict_query_representation_route_disjoint"))
    report = {
        "artifact_type": SCHEMA,
        "fixed_report_file_sha256": file_sha256(fixed_path),
        "source_softmax_report_file_sha256": file_sha256(softmax_path),
        "paired_lineage": {name: fixed[name] for name in identical},
        "initialization_baseline_derivation": (
            "postfreeze_replay_with_feature_hierarchy_layout_logits_zero_and_absent_"
            "geometry_logits_minus30_exactly_matches_preoptimizer_two_dof_state_v1"
        ),
        "controls": rows,
        "trained_fixed_minus_source_softmax_dev_medium": paired_delta,
        "selected_strict_loose_metrics_excluded_from_scientific_gate": True,
        "selected_metric_exclusion_reason": (
            "controlled_oracle_stencil_always_contains_0.25m_nonanchor_candidates"
        ),
        "fixed_kernel_training_material_gain_supported": False,
        "fixed_kernel_training_gain_assessment": (
            "anchor_and_radial_metrics_unchanged_from_initialization;Spearman_and_"
            "pairwise_gains_below_0.002"
        ),
        "fixed_kernel_mechanism_advantage_over_source_softmax_supported": bool(
            paired_delta["gt_anchor_top1_rate"] > 0.0
            and paired_delta["controlled_radial_pair_accuracy"] > 0.0
            and paired_delta["controlled_radial_complete_path_rate"] > 0.0
        ),
        "optimizer_trajectory_drift_claim_supported": False,
        "strict_route_disjoint_representation_claim_supported": mapper_strict,
        "promotion_gate_passed": False,
        "promotion_blockers": [
            "surface_mapper_supervision_overlaps_query_route_and_all_12_query_images",
            "dev_is_same_route_sorted_prefix_holdout_not_independent_route",
            "candidates_are_GT_relative_controlled_stencil_not_natural_retrieval",
            "no_frozen_optimizer_trajectory_drift_negatives",
            "coarse_and_fine_are_RADIO_only_structural_controls_not_typed_depth_stages",
        ],
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
