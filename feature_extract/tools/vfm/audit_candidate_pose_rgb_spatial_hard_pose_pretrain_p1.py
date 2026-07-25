"""Audit a gate-approved hard-pose visual initializer on frozen P1 rows.

This is deliberately a train-only transfer audit.  The model first emits
target-free edge densities from fixed P1 query/support observations; only then
does the evaluator join correct and coherent-wrong projection offsets from the
train-only P1 target artifact.  It never invokes PnP, external pose evaluation,
rendering, image retrieval, or submaps.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    CHECKPOINT_FORMAT as RGB_SPATIAL_LIKELIHOOD_CHECKPOINT_FORMAT,
    _DistributedState,
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _evaluate_inner_validation_components,
    _finalize_distributed,
    _initialize_distributed,
    _partition_train_queries_for_inner_validation,
    _query_batch_from_group,
    _slice_runtime,
    _select_group_points,
    _source_table,
    build_train_query_groups,
    full_pose_hard_target_initializer_parent_hashes,
    load_context_observation_pretrain_initialization_checkpoint,
    load_hard_pose_pretrain_initialization_checkpoint,
    load_observation_pretrain_initialization_checkpoint,
    load_target_free_initialization_checkpoint,
    training_gate_decision,
    validate_rgb_coordinate_bridge,
    validate_training_layout_and_targets,
)
from feature_extract.tools.vfm.train_candidate_pose_context_identity_l0 import (
    load_context_identity_l0_initialization_checkpoint,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    query_grouped_pose_margin_loss,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
    CandidatePoseRGBSpatialLikelihood,
    candidate_pose_rgb_spatial_score_component_prediction,
    resolve_candidate_pose_rgb_spatial_context_encoder_arch,
    resolve_candidate_pose_rgb_spatial_context_windows,
    runtime_from_target_free_layout,
    score_candidate_pose_rgb_spatial_batch,
    permute_runtime_support_appearance,
    permute_support_patch_appearance,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_evidence import (
    EVIDENCE_LAYER_AUDIT_FORMAT,
    summarize_candidate_pose_rgb_spatial_evidence_layers,
    summarize_registered_observation_candidate_oracle,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_selector import (
    TARGET_FREE_SELECTOR_POLICIES,
    select_target_free_spatial_quota,
    selector_input_from_target_free_layout,
    slice_target_free_edge_prediction,
    summarize_target_free_point_selection,
    target_free_rgb_point_quality,
    target_free_selector_scores,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_source_headers,
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


AUDIT_FORMAT = "candidate_pose_rgb_spatial_checkpoint_p1_transfer_audit_v4"
CROSS_LAYOUT_COMPONENT_AUDIT_FORMAT = (
    "candidate_pose_rgb_spatial_checkpoint_p1_transfer_audit_v4"
)
SCORE_COMPONENTS = {
    "combined",
    "learned_spatial_no_context",
    "rgb_cost_volume",
    "rgb_cost_volume_with_dustbin",
    "spatial_residual",
    "context_only",
}

# These checkpoints deliberately contain only the RADIO/ALIKE context
# component.  Replaying one through the RGB density path would make random,
# uninitialized spatial heads participate in an audit that is supposed to
# measure context transfer only.
_CONTEXT_ONLY_CHECKPOINT_KINDS = frozenset(
    {"context_observation_pretrain", "context_identity_l0"}
)


def _checkpoint_kind_requires_context_only_replay(checkpoint_kind: str) -> bool:
    """Return whether a checkpoint may be audited only via its context LLR."""

    return str(checkpoint_kind) in _CONTEXT_ONLY_CHECKPOINT_KINDS


def _p1_point_selection_contract(
    *,
    layout,
    groups: Mapping[str, object],
    query_ids: Sequence[str],
    max_points_per_query: int,
) -> dict[str, object]:
    """Describe whether this replay can use every frozen target-free row.

    The historic P1 audit used ``_select_group_points`` to preserve scarce
    registered observations.  That is valid for fitting diagnostics, but it
    is not a runtime selector because it reads train-only targets.  A layout
    created by the frozen RGB selector is different: its rows were selected
    before target join and can be replayed as-is.  We only recognize that
    stronger contract when this audit keeps *all* rows in every evaluated
    query; any second-stage subsampling falls back to the conservative
    historic designation.
    """

    selected_queries = tuple(str(query_id) for query_id in query_ids)
    if not selected_queries or len(set(selected_queries)) != len(selected_queries):
        raise ValueError("P1 point-selection audit query IDs are invalid")
    unresolved = [query_id for query_id in selected_queries if query_id not in groups]
    if unresolved:
        raise ValueError("P1 point-selection audit query group is unresolved")
    point_counts: list[int] = []
    for query_id in selected_queries:
        group = groups[query_id]
        point_count = int(getattr(group, "point_count", -1))
        if point_count <= 0:
            raise ValueError("P1 point-selection audit group is invalid")
        point_counts.append(point_count)
    max_points = int(max_points_per_query)
    keeps_all_rows = max_points <= 0 or all(max_points >= count for count in point_counts)

    selector = layout.metadata.get("frozen_rgb_selector")
    required_exclusions = {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
    }
    selector_exclusions = (
        set(str(value) for value in selector.get("selection_excludes", ()))
        if isinstance(selector, Mapping)
        else set()
    )
    frozen_target_free_layout = bool(
        isinstance(selector, Mapping)
        and selector.get("format") == "frozen_rgb_peakiness_p1_subset_layout_v1"
        and selector.get("runtime_layout_target_free") is True
        and selector.get("selection_before_train_target_join") is True
        and required_exclusions.issubset(selector_exclusions)
    )
    runtime_eligible = bool(frozen_target_free_layout and keeps_all_rows)
    if runtime_eligible:
        mode = "frozen_target_free_rgb_selector_layout_all_rows_v1"
    elif frozen_target_free_layout:
        mode = "frozen_target_free_layout_with_second_stage_subsampling_v1"
    else:
        mode = "train_target_observation_preserving_sampler_v1"
    return {
        "mode": mode,
        "runtime_eligible": runtime_eligible,
        "frozen_target_free_layout": frozen_target_free_layout,
        "selection_before_train_target_join": bool(
            isinstance(selector, Mapping)
            and selector.get("selection_before_train_target_join") is True
        ),
        "keeps_all_frozen_layout_rows": keeps_all_rows,
        "max_points_per_query": max_points,
        "evaluated_query_count": len(selected_queries),
        "minimum_group_point_count": min(point_counts),
        "maximum_group_point_count": max(point_counts),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument(
        "--registered-identity-targets",
        default="",
        help=(
            "Optional registered exact-track target artifact for the edge-evidence "
            "oracle. It is accepted only by --edge-evidence-layers and never enters "
            "the target-free model forward."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint to replay on the frozen P1 train-only transfer protocol.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=(
            "hard_pose_pretrain",
            "observation_pretrain",
            "context_observation_pretrain",
            "context_identity_l0",
            "p1_checkpoint",
            "p1_cross_layout_curriculum_checkpoint",
        ),
        default="hard_pose_pretrain",
        help="Strict checkpoint lineage contract used before the replay audit.",
    )
    parser.add_argument(
        "--hard-pose-pretrain-checkpoint",
        default="",
        help="Deprecated alias for --checkpoint with --checkpoint-kind hard_pose_pretrain.",
    )
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--search-radius-px", type=float, default=None)
    parser.add_argument("--context-radius-px", type=float, default=12.0)
    parser.add_argument("--step-px", type=float, default=1.0)
    parser.add_argument("--texture-feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-abs-context-log-ratio", type=float, default=3.0)
    parser.add_argument("--max-abs-pose-log-ratio", type=float, default=6.0)
    parser.add_argument("--edge-chunk-size", type=int, default=256)
    parser.add_argument(
        "--rgb-cost-volume-only",
        action="store_true",
        help=(
            "Replay the strict RGB-only FPN/template-cost-volume path. This is "
            "required for checkpoints whose metadata declares rgb_cost_volume_only."
        ),
    )
    parser.add_argument("--radio-final-context-window", type=int, default=9)
    parser.add_argument("--radio-intermediate-context-window", type=int, default=9)
    parser.add_argument("--alike-context-window", type=int, default=13)
    parser.add_argument(
        "--context-encoder-arch",
        default="conv_v1",
        help="Explicit context encoder architecture required to match the frozen checkpoint.",
    )
    parser.add_argument("--max-points-per-query", type=int, default=64)
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=0)
    parser.add_argument("--pose-margin", type=float, default=0.25)
    parser.add_argument("--missing-edge-log-likelihood-ratio", type=float, default=0.0)
    parser.add_argument("--permutation-control-shift", type=int, default=1)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-normal-gap", type=float, default=0.05)
    parser.add_argument("--minimum-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--rgb-cache-gb", type=float, default=10.0)
    parser.add_argument(
        "--rgb-cache-dtype",
        choices=("float16", "uint8"),
        default="float16",
        help="Full-image RGB cache storage for target-free replay.",
    )
    parser.add_argument(
        "--allow-ineligible-diagnostic-checkpoint",
        action="store_true",
        help=(
            "Allow a failed-gate target-free checkpoint only for this train-only audit; "
            "the result remains ineligible for P1 or PnP promotion."
        ),
    )
    parser.add_argument(
        "--cross-layout-component-audit",
        default="",
        help=(
            "Required only for p1_cross_layout_curriculum_checkpoint. A prior frozen "
            "RGB-cost-volume component audit of the source curriculum checkpoint; "
            "it prevents a combined-only source result from being transferred."
        ),
    )
    parser.add_argument(
        "--score-components",
        default="combined,learned_spatial_no_context,rgb_cost_volume,rgb_cost_volume_with_dustbin,spatial_residual,context_only",
        help="Comma-separated target-free score components to audit.",
    )
    parser.add_argument(
        "--edge-evidence-layers",
        action="store_true",
        help=(
            "After target-free frozen scoring, join train-only pose/observation targets "
            "to decompose edge, view, candidate/null, and point aggregation evidence. "
            "The result is diagnostic-only and cannot promote the checkpoint."
        ),
    )
    parser.add_argument(
        "--target-free-selector-sweep",
        action="store_true",
        help=(
            "Score every frozen P1 anchor, then compare strict target-free fixed-point "
            "selectors before joining train-only pose targets. This corrects the historic "
            "supervision-preserving P1 sampler and is diagnostic-only."
        ),
    )
    parser.add_argument(
        "--target-free-selector-policies",
        default="uniform,coarse_margin,support_coverage,rgb_peakiness,coarse_margin_rgb_peakiness",
        help="Comma-separated target-free point-selector policies for --target-free-selector-sweep.",
    )
    parser.add_argument(
        "--target-free-selector-point-budgets",
        default="32,64,96",
        help="Comma-separated point budgets for --target-free-selector-sweep.",
    )
    parser.add_argument("--target-free-selector-grid-rows", type=int, default=4)
    parser.add_argument("--target-free-selector-grid-columns", type=int, default=4)
    parser.add_argument(
        "--gate-component",
        default="combined",
        choices=tuple(sorted(SCORE_COMPONENTS)),
        help="Component used for the frozen P1 transfer gate; it must be audited explicitly.",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> dict[str, int]:
    values = (
        float(args.context_radius_px),
        float(args.step_px),
        float(args.max_abs_context_log_ratio),
        float(args.max_abs_pose_log_ratio),
        float(args.pose_margin),
        float(args.missing_edge_log_likelihood_ratio),
        float(args.minimum_win_fraction),
        float(args.minimum_normal_gap),
        float(args.minimum_visual_gap_delta),
        float(args.rgb_cache_gb),
    )
    if (
        int(args.texture_feature_dim) <= 0
        or int(args.hidden_dim) < 4
        or int(args.edge_chunk_size) <= 0
        or 0 < int(args.max_points_per_query) < 4
        or int(args.inner_validation_fold_count) < 2
        or int(args.inner_validation_fold_index) < 0
        or int(args.permutation_control_shift) <= 0
        or not all(math.isfinite(value) for value in values)
        or float(args.context_radius_px) <= 0.0
        or float(args.step_px) <= 0.0
        or float(args.max_abs_context_log_ratio) <= 0.0
        or float(args.max_abs_pose_log_ratio) <= 0.0
        or float(args.pose_margin) < 0.0
        or not 0.0 <= float(args.minimum_win_fraction) <= 1.0
        or float(args.minimum_normal_gap) < 0.0
        or float(args.minimum_visual_gap_delta) < 0.0
        or float(args.rgb_cache_gb) <= 0.0
        or int(args.target_free_selector_grid_rows) <= 0
        or int(args.target_free_selector_grid_columns) <= 0
    ):
        raise ValueError("hard-pose P1 transfer audit arguments are invalid")
    if bool(args.target_free_selector_sweep):
        _parse_target_free_selector_policies(args.target_free_selector_policies)
        _parse_target_free_selector_budgets(args.target_free_selector_point_budgets)
    if str(args.registered_identity_targets).strip() and not bool(args.edge_evidence_layers):
        raise ValueError(
            "registered identity targets are valid only with --edge-evidence-layers"
        )
    return resolve_candidate_pose_rgb_spatial_context_windows(
        {
            "radio_final": int(args.radio_final_context_window),
            "radio_intermediate": int(args.radio_intermediate_context_window),
            "alike": int(args.alike_context_window),
        }
    )


def _parse_score_components(value: str) -> tuple[str, ...]:
    components = tuple(
        dict.fromkeys(
            str(component).strip().lower()
            for component in str(value).split(",")
            if str(component).strip()
        )
    )
    if not components or not set(components).issubset(SCORE_COMPONENTS):
        raise ValueError("hard-pose P1 transfer score components are invalid")
    return components


def _parse_target_free_selector_policies(value: str) -> tuple[str, ...]:
    policies = tuple(
        dict.fromkeys(
            str(policy).strip().lower()
            for policy in str(value).split(",")
            if str(policy).strip()
        )
    )
    if not policies or not set(policies).issubset(TARGET_FREE_SELECTOR_POLICIES):
        raise ValueError("target-free selector policy list is invalid")
    return policies


def _parse_target_free_selector_budgets(value: str) -> tuple[int, ...]:
    try:
        budgets = tuple(
            dict.fromkeys(
                int(part.strip()) for part in str(value).split(",") if part.strip()
            )
        )
    except ValueError as error:
        raise ValueError("target-free selector point budgets are invalid") from error
    if not budgets or any(budget < 4 for budget in budgets):
        raise ValueError("target-free selector point budgets must be at least four")
    return budgets


def _checkpoint_request(args: argparse.Namespace) -> tuple[Path, str]:
    """Resolve one explicit initializer while keeping the legacy CLI safe."""

    generic = str(args.checkpoint).strip()
    legacy = str(args.hard_pose_pretrain_checkpoint).strip()
    if generic and legacy and Path(generic) != Path(legacy):
        raise ValueError("P1 transfer audit received conflicting checkpoint paths")
    path = Path(generic or legacy) if (generic or legacy) else None
    kind = str(args.checkpoint_kind)
    if path is None:
        raise ValueError("P1 transfer audit requires --checkpoint")
    if legacy and kind != "hard_pose_pretrain":
        raise ValueError("legacy hard-pose checkpoint alias requires hard_pose_pretrain kind")
    if bool(args.allow_ineligible_diagnostic_checkpoint) and kind not in {
        "hard_pose_pretrain",
        "context_identity_l0",
    }:
        raise ValueError(
            "diagnostic ineligible override is valid only for hard-pose or P1 context L0 checkpoints"
        )
    component_audit = str(args.cross_layout_component_audit).strip()
    if kind == "p1_cross_layout_curriculum_checkpoint" and not component_audit:
        raise ValueError("cross-layout curriculum audit requires --cross-layout-component-audit")
    if kind != "p1_cross_layout_curriculum_checkpoint" and component_audit:
        raise ValueError(
            "--cross-layout-component-audit is valid only for a cross-layout curriculum checkpoint"
        )
    return path, kind


def _load_cross_layout_curriculum_checkpoint(
    *,
    path: Path,
    model: CandidatePoseRGBSpatialLikelihood,
    destination_layout,
    source_cache_paths: Mapping[str, Path],
    source_image_manifest_sha256: str,
    rgb_coordinate_bridge: Mapping[str, object],
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    texture_feature_dim: int,
    hidden_dim: int,
    max_abs_context_log_ratio: float,
    context_windows: Mapping[str, int],
    context_encoder_arch: str,
    component_audit_path: Path,
) -> dict[str, object]:
    """Load a passed curriculum checkpoint on a distinct real P1 layout.

    This is intentionally narrower than normal P1 fine-tuning.  It permits a
    *frozen diagnostic replay* from a train-only observation-anchor curriculum
    to the real detector-anchor layout only after an independent RGB-only
    support-permutation audit has passed.  It never accepts target arrays as
    model inputs and never marks the resulting score as PnP-eligible.
    """

    checkpoint_path = Path(path)
    audit_path = Path(component_audit_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"cross-layout curriculum checkpoint is absent: {checkpoint_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"cross-layout component audit is absent: {audit_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("cross-layout curriculum checkpoint is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        payload.get("format") != RGB_SPATIAL_LIKELIHOOD_CHECKPOINT_FORMAT
        or not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("model_format") != CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("pose_or_ground_truth_used_by_runtime_scorer") is not False
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("raw_scores_must_not_feed_pnp") is not True
        or metadata.get("train_only_inner_gate_passed") is not True
        or metadata.get("holdout_evaluation_allowed") is not True
        or int(metadata.get("fixed_candidate_top_k", -1))
        != int(destination_layout.candidate_count)
        or int(metadata.get("fixed_support_view_count", -1))
        != int(destination_layout.support_view_count)
    ):
        raise ValueError("cross-layout curriculum checkpoint is not target-free compatible")
    lineage = metadata.get("lineage")
    config = metadata.get("config")
    inputs = metadata.get("inputs")
    if (
        not isinstance(lineage, Mapping)
        or not isinstance(config, Mapping)
        or not isinstance(inputs, Mapping)
    ):
        raise ValueError("cross-layout curriculum checkpoint lacks lineage/config")
    source_layout_input = inputs.get("rgb_spatial_layout")
    if not isinstance(source_layout_input, Mapping):
        raise ValueError("cross-layout curriculum checkpoint lacks source layout lineage")
    source_layout_path = Path(str(source_layout_input.get("path", "")))
    expected_source_layout_sha = str(lineage.get("layout_sha256", ""))
    if (
        not source_layout_path.is_file()
        or not expected_source_layout_sha
        or str(source_layout_input.get("sha256", "")) != expected_source_layout_sha
        or file_sha256_short(source_layout_path) != expected_source_layout_sha
    ):
        raise ValueError("cross-layout curriculum source layout lineage differs")
    source_layout = load_candidate_pose_rgb_spatial_layout(source_layout_path)
    if (
        source_layout.metadata.get("training_only_anchor_layout") is not True
        or source_layout.metadata.get("runtime_scorer_must_not_load_this_layout") is not True
        or set(source_layout.split_names.tolist()) != {"train"}
        or source_layout.candidate_count != destination_layout.candidate_count
        or source_layout.support_view_count != destination_layout.support_view_count
        or str(source_layout.metadata.get("projection_space_id", ""))
        != str(destination_layout.metadata.get("projection_space_id", ""))
        or str(source_layout.metadata.get("descriptor_space_id", ""))
        != str(destination_layout.metadata.get("descriptor_space_id", ""))
        or destination_layout.metadata.get("training_only_anchor_layout") is True
    ):
        raise ValueError("cross-layout curriculum/source destination layout contract differs")
    expected_cache_inputs = {
        "radio_final": "radio_final_context_cache",
        "radio_intermediate": "radio_intermediate_context_cache",
        "alike": "alike_spatial_context_cache",
    }
    if set(source_cache_paths) != set(expected_cache_inputs):
        raise ValueError("cross-layout curriculum source-cache set is incomplete")
    for source_name, input_name in expected_cache_inputs.items():
        entry = inputs.get(input_name)
        if (
            not isinstance(entry, Mapping)
            or str(entry.get("sha256", ""))
            != file_sha256_short(Path(source_cache_paths[source_name]))
        ):
            raise ValueError("cross-layout curriculum source cache lineage differs")
    if str(lineage.get("source_image_manifest_sha256", "")) != str(
        source_image_manifest_sha256
    ):
        raise ValueError("cross-layout curriculum image manifest lineage differs")
    try:
        serialized_bridge = json.dumps(
            lineage.get("rgb_coordinate_bridge"), sort_keys=True, separators=(",", ":")
        )
        expected_bridge = json.dumps(
            dict(rgb_coordinate_bridge), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("cross-layout curriculum RGB coordinate bridge is invalid") from error
    if serialized_bridge != expected_bridge:
        raise ValueError("cross-layout curriculum RGB coordinate bridge differs")
    expected_floats = {
        "search_radius_px": float(search_radius_px),
        "context_radius_px": float(context_radius_px),
        "step_px": float(step_px),
        "max_abs_context_log_ratio": float(max_abs_context_log_ratio),
    }
    for name, expected in expected_floats.items():
        try:
            observed = float(config[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("cross-layout curriculum checkpoint config is incomplete") from error
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("cross-layout curriculum checkpoint config differs")
    if (
        int(config.get("texture_feature_dim", -1)) != int(texture_feature_dim)
        or int(config.get("hidden_dim", -1)) != int(hidden_dim)
        or config.get("rgb_cost_volume_only") is not False
    ):
        raise ValueError("cross-layout curriculum checkpoint model config differs")
    try:
        observed_windows = resolve_candidate_pose_rgb_spatial_context_windows(
            config.get("context_windows")
        )
        expected_windows = resolve_candidate_pose_rgb_spatial_context_windows(context_windows)
        observed_context_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            config.get("context_encoder_arch", "conv_v1")
        )
        expected_context_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            context_encoder_arch
        )
    except (TypeError, ValueError) as error:
        raise ValueError("cross-layout curriculum context configuration is invalid") from error
    if observed_windows != expected_windows or observed_context_arch != expected_context_arch:
        raise ValueError("cross-layout curriculum context configuration differs")
    try:
        component_audit = json.loads(audit_path.read_text())
    except (OSError, ValueError, TypeError) as error:
        raise ValueError("cross-layout component audit is malformed") from error
    audited_checkpoint = component_audit.get("checkpoint") if isinstance(component_audit, Mapping) else None
    component_metrics = (
        component_audit.get("component_metrics") if isinstance(component_audit, Mapping) else None
    )
    audit_config = component_audit.get("config") if isinstance(component_audit, Mapping) else None
    audit_gate = (
        component_audit.get("zero_shot_transfer_gate")
        if isinstance(component_audit, Mapping)
        else None
    )
    rgb_component = (
        component_metrics.get("rgb_cost_volume_with_dustbin")
        if isinstance(component_metrics, Mapping)
        else None
    )
    if (
        component_audit.get("format") != CROSS_LAYOUT_COMPONENT_AUDIT_FORMAT
        or not isinstance(audited_checkpoint, Mapping)
        or str(audited_checkpoint.get("sha256", "")) != file_sha256_short(checkpoint_path)
        or not isinstance(audit_config, Mapping)
        or audit_config.get("gate_component") != "rgb_cost_volume_with_dustbin"
        or not isinstance(audit_gate, Mapping)
        or audit_gate.get("passed") is not True
        or not isinstance(rgb_component, Mapping)
    ):
        raise ValueError("cross-layout component audit does not prove RGB-only source evidence")
    try:
        rgb_gate = training_gate_decision(
            rgb_component,
            minimum_win_fraction=float(audit_gate["minimum_win_fraction"]),
            minimum_normal_gap=float(audit_gate["minimum_normal_gap"]),
            minimum_visual_gap_delta=float(audit_gate["minimum_visual_gap_delta"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("cross-layout component audit RGB gate is malformed") from error
    if rgb_gate.get("passed") is not True:
        raise ValueError("cross-layout component audit RGB-only gate did not pass")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("cross-layout curriculum state dict is incompatible") from error
    return {
        "kind": "cross_layout_train_only_curriculum_to_real_detector_anchor_diagnostic",
        "path": str(checkpoint_path),
        "sha256": file_sha256_short(checkpoint_path),
        "selected_epoch": int(
            metadata.get("training", {}).get("inner_validation", {}).get("selected_epoch", -1)
        )
        if isinstance(metadata.get("training"), Mapping)
        else -1,
        "source_training_layout": {
            "path": str(source_layout_path),
            "sha256": expected_source_layout_sha,
            "training_only_anchor_layout": True,
        },
        "component_audit": {
            "path": str(audit_path),
            "sha256": file_sha256_short(audit_path),
            "rgb_cost_volume_with_dustbin_gate": rgb_gate,
        },
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
    }


def _distributed_output_conflict(*, state: _DistributedState, output: Path) -> bool:
    """Avoid an output collision leaving a peer DDP rank at a barrier."""

    conflict = bool(output.exists()) if state.rank == 0 else False
    if state.enabled:
        decision = torch.tensor([int(conflict)], dtype=torch.int64, device=state.device)
        distributed.broadcast(decision, src=0)
        conflict = bool(int(decision.item()))
    return conflict


def _summarize_edge_evidence_rows(
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Aggregate fixed-query diagnostic rows without concealing row variation."""

    if not rows:
        raise ValueError("RGB spatial edge evidence has no query rows")
    query_ids = [str(row.get("query_id", "")) for row in rows]
    if any(not query_id for query_id in query_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("RGB spatial edge evidence query rows are invalid")
    metric_rows = [row.get("metrics") for row in rows]
    if any(not isinstance(metrics, dict) for metrics in metric_rows):
        raise ValueError("RGB spatial edge evidence metric rows are invalid")
    typed_metrics = [dict(metrics) for metrics in metric_rows if isinstance(metrics, dict)]
    metric_names = sorted(set().union(*(set(metrics) for metrics in typed_metrics)))
    if any(set(metrics) != set(metric_names) for metrics in typed_metrics):
        raise ValueError("RGB spatial edge evidence metric schemas differ by query")
    per_query_mean: dict[str, float] = {}
    per_query_median: dict[str, float] = {}
    sums: dict[str, float] = {}
    for name in metric_names:
        values = np.asarray([float(metrics[name]) for metrics in typed_metrics], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("RGB spatial edge evidence metrics are non-finite")
        per_query_mean[name] = float(values.mean())
        per_query_median[name] = float(np.median(values))
        if name.endswith("_count"):
            sums[name] = float(values.sum())
    return {
        "query_count": int(len(rows)),
        "per_query_mean": per_query_mean,
        "per_query_median": per_query_median,
        "sum_counts": sums,
    }


def _registered_identity_masks_by_source(
    *,
    registered_identity_targets: object,
    expected_source_point_ids: np.ndarray,
    expected_query_ids: np.ndarray,
    candidate_count: int,
) -> dict[int, np.ndarray]:
    """Join an explicit exact-track artifact for an oracle-only audit.

    Full-pose target artifacts encode geometric projection visibility, where
    multiple candidates may reasonably be true.  They are not physical-track
    labels.  This helper accepts only the separately serialized exact-track
    artifact and keeps the join out of the visual forward path.
    """

    metadata = getattr(registered_identity_targets, "metadata", None)
    source_ids = np.asarray(
        getattr(registered_identity_targets, "source_point_ids", ()), dtype=np.int64
    ).reshape(-1)
    query_ids = np.asarray(
        getattr(registered_identity_targets, "query_ids", ()), dtype=str
    ).reshape(-1)
    observed = np.asarray(
        getattr(registered_identity_targets, "spatial_target_observed", ()), dtype=bool
    )
    expected_sources = np.asarray(expected_source_point_ids, dtype=np.int64).reshape(-1)
    expected_queries = np.asarray(expected_query_ids, dtype=str).reshape(-1)
    if (
        not isinstance(metadata, Mapping)
        or str(metadata.get("format", ""))
        != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT
        or str(metadata.get("spatial_supervision_mode", ""))
        != "registered_exact_identity"
        or str(metadata.get("spatial_target_semantics", ""))
        != "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
        or len(source_ids) == 0
        or len(source_ids) != len(np.unique(source_ids))
        or source_ids.shape != query_ids.shape
        or expected_sources.shape != expected_queries.shape
        or len(expected_sources) != len(np.unique(expected_sources))
        or observed.shape != (len(source_ids), int(candidate_count))
        or np.any(observed.sum(axis=1) > 1)
    ):
        raise ValueError("edge-evidence registered identity targets are invalid")
    expected_query_by_source = {
        int(source_id): str(query_id)
        for source_id, query_id in zip(expected_sources.tolist(), expected_queries.tolist())
    }
    identity_query_by_source = {
        int(source_id): str(query_id)
        for source_id, query_id in zip(source_ids.tolist(), query_ids.tolist())
    }
    if (
        set(identity_query_by_source) != set(expected_query_by_source)
        or any(
            identity_query_by_source[source_id] != expected_query_by_source[source_id]
            for source_id in expected_query_by_source
        )
    ):
        raise ValueError("edge-evidence registered identity target layout differs")
    return {
        int(source_id): np.asarray(observed[row], dtype=bool).copy()
        for row, source_id in enumerate(source_ids.tolist())
    }


@torch.no_grad()
def _evaluate_inner_validation_edge_evidence_layers(
    *,
    model: torch.nn.Module,
    groups: Mapping[str, object],
    complete_runtime: object,
    query_ids: Sequence[str],
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    radius_px: float,
    step_px: float,
    cache: TensorImageLRUCache,
    state: _DistributedState,
    max_points_per_query: int,
    seed: int,
    missing_edge_log_likelihood_ratio: float,
    max_abs_pose_log_ratio: float,
    amp_enabled: bool,
    prediction_transform: object,
    rgb_cost_volume_only: bool,
    registered_identity_masks_by_source: Mapping[int, np.ndarray] | None,
) -> dict[str, object]:
    """Audit score layers after a frozen target-free forward pass.

    ``groups`` contains train-only targets, but this function deliberately
    creates RGB patches, runs the model, and obtains both pose scores before it
    passes a target field into either evidence summary.  This preserves the
    runtime boundary even in a diagnostic replay.
    """

    if not query_ids:
        raise ValueError("RGB spatial edge evidence has no inner-validation queries")
    model.eval()
    local_rows: list[dict[str, object]] = []
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        if group is None:
            raise ValueError("RGB spatial edge evidence query group is unresolved")
        positions = _select_group_points(
            group=group,
            max_points=int(max_points_per_query),
            seed=int(seed),
        )
        batch = _query_batch_from_group(
            group=group,
            complete_runtime=complete_runtime,
            point_positions=positions,
            device=state.device,
        )
        query_patches, support_patches = _crop_runtime_rgb_patches(
            runtime=batch.runtime,
            image_ids=image_ids,
            image_root=image_root,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=float(radius_px),
            step_px=float(step_px),
            cache=cache,
            device=state.device,
        )
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            base_prediction = model(
                runtime=batch.runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                rgb_cost_volume_only=bool(rgb_cost_volume_only),
            )
            prediction = (
                base_prediction
                if prediction_transform is None
                else prediction_transform(base_prediction)
            )
            correct_score = score_candidate_pose_rgb_spatial_batch(
                runtime=batch.runtime,
                prediction=prediction,
                candidate_projection_offsets_xy=batch.correct_projection_offsets_xy.unsqueeze(0),
                candidate_projection_valid=batch.correct_projection_valid.unsqueeze(0),
                missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
            )
            wrong_score = score_candidate_pose_rgb_spatial_batch(
                runtime=batch.runtime,
                prediction=prediction,
                candidate_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                candidate_projection_valid=batch.wrong_projection_valid,
                missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
            )
        # These calls are deliberately after all model and target-free scoring.
        metrics = summarize_candidate_pose_rgb_spatial_evidence_layers(
            correct=correct_score,
            wrong=wrong_score,
            correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
            wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
            candidate_probabilities=batch.runtime.candidate_probabilities,
            null_probabilities=batch.runtime.null_probabilities,
        )
        if registered_identity_masks_by_source is None:
            metrics["registered_observation_oracle_available"] = 0.0
        else:
            source_ids = np.asarray(group.source_point_ids[positions], dtype=np.int64)
            try:
                observed_candidate_mask = np.stack(
                    [registered_identity_masks_by_source[int(source_id)] for source_id in source_ids],
                    axis=0,
                )
            except KeyError as error:
                raise ValueError(
                    "edge-evidence registered identity target misses a selected source"
                ) from error
            if observed_candidate_mask.shape != (
                len(source_ids), int(batch.runtime.candidate_count)
            ):
                raise ValueError("edge-evidence registered identity mask shape differs")
            metrics.update(
                summarize_registered_observation_candidate_oracle(
                    correct=correct_score,
                    wrong=wrong_score,
                    observed_candidate_mask=torch.from_numpy(observed_candidate_mask).to(
                        device=state.device
                    ),
                )
            )
            metrics["registered_observation_oracle_available"] = 1.0
        local_rows.append({"query_id": str(query_id), "metrics": metrics})
    if state.enabled:
        gathered: list[list[dict[str, object]] | None] = [None] * state.world_size
        distributed.all_gather_object(gathered, local_rows)
        rows = [row for part in gathered if part is not None for row in part]
    else:
        rows = local_rows
    rows = sorted(rows, key=lambda row: str(row["query_id"]))
    if len(rows) != len(query_ids):
        raise RuntimeError("RGB spatial edge evidence did not cover every inner-validation query")
    return {
        "format": EVIDENCE_LAYER_AUDIT_FORMAT,
        "target_joined_after_network_prediction": True,
        "oracle_metrics_diagnostic_only": True,
        "runtime_scorer_remains_target_free": True,
        "rows": rows,
        "summary": _summarize_edge_evidence_rows(rows),
    }


def _target_free_selector_pose_metrics(
    *,
    runtime: object,
    prediction: object,
    group: object,
    positions: np.ndarray,
    device: torch.device,
    pose_margin: float,
    missing_edge_log_likelihood_ratio: float,
    max_abs_pose_log_ratio: float,
) -> dict[str, float]:
    """Join pose targets only after one target-free selector has run.

    ``positions`` must have been produced from the immutable layout and visual
    density alone.  This function is intentionally the first place in the
    selector sweep that reads the train-only projection arrays.
    """

    selected = np.asarray(positions, dtype=np.int64).reshape(-1)
    if len(selected) == 0:
        raise ValueError("target-free selector pose metrics require selected points")
    selected_runtime = _slice_runtime(runtime, selected)
    selected_prediction = slice_target_free_edge_prediction(
        prediction=prediction, positions=selected
    )
    correct_offsets = torch.from_numpy(
        np.asarray(group.correct_projection_offsets_xy[selected], dtype=np.float32)
    ).to(device=device)
    correct_valid = torch.from_numpy(
        np.asarray(group.correct_projection_valid[selected], dtype=bool)
    ).to(device=device)
    wrong_offsets = torch.from_numpy(
        np.asarray(group.wrong_projection_offsets_xy[:, selected], dtype=np.float32)
    ).to(device=device)
    wrong_valid = torch.from_numpy(
        np.asarray(group.wrong_projection_valid[:, selected], dtype=bool)
    ).to(device=device)
    correct = score_candidate_pose_rgb_spatial_batch(
        runtime=selected_runtime,
        prediction=selected_prediction,
        candidate_projection_offsets_xy=correct_offsets.unsqueeze(0),
        candidate_projection_valid=correct_valid.unsqueeze(0),
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    ).pose_log_likelihood_ratios
    wrong = score_candidate_pose_rgb_spatial_batch(
        runtime=selected_runtime,
        prediction=selected_prediction,
        candidate_projection_offsets_xy=wrong_offsets,
        candidate_projection_valid=wrong_valid,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    ).pose_log_likelihood_ratios
    loss, metrics = query_grouped_pose_margin_loss(
        correct_scores=correct,
        coherent_wrong_scores=wrong.reshape(1, -1),
        margin=float(pose_margin),
    )
    return {
        "query_grouped_loss": float(loss.item()),
        "mean_correct_minus_hardest_wrong": float(
            metrics["query_mean_correct_minus_hardest_wrong"]
        ),
        "correct_win_fraction": float(metrics["query_correct_win_fraction"]),
        "hardest_wrong_mode_count": float(len(wrong)),
    }


def _summarize_target_free_selector_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    minimum_win_fraction: float,
    minimum_normal_gap: float,
    minimum_visual_gap_delta: float,
) -> dict[str, object]:
    """Aggregate target-free selector rows into strict fixed/end-to-end gates."""

    if not rows:
        raise ValueError("target-free selector sweep produced no rows")
    entry_keys = tuple(sorted(set().union(*(set(dict(row["entries"])) for row in rows))))
    if not entry_keys or any(set(dict(row["entries"])) != set(entry_keys) for row in rows):
        raise ValueError("target-free selector sweep rows do not share a policy grid")
    output: dict[str, object] = {}
    for key in entry_keys:
        entries = [dict(dict(row["entries"])[key]) for row in rows]
        def mean(path: str) -> float:
            values = np.asarray(
                [float(dict(entry[path])["mean_correct_minus_hardest_wrong"]) for entry in entries],
                dtype=np.float64,
            )
            return float(values.mean())
        def win(path: str) -> float:
            values = np.asarray(
                [float(dict(entry[path])["correct_win_fraction"]) for entry in entries], dtype=np.float64
            )
            return float(values.mean())
        def loss(path: str) -> float:
            values = np.asarray(
                [float(dict(entry[path])["query_grouped_loss"]) for entry in entries], dtype=np.float64
            )
            return float(values.mean())
        normal_gap = mean("normal")
        fixed_gap = mean("permuted_fixed_selection")
        end_to_end_gap = mean("permuted_end_to_end_selection")
        normal = {
            "normal_query_grouped_loss": loss("normal"),
            "normal_mean_correct_minus_hardest_wrong": normal_gap,
            "normal_correct_win_fraction": win("normal"),
            "permuted_query_grouped_loss": loss("permuted_fixed_selection"),
            "permuted_mean_correct_minus_hardest_wrong": fixed_gap,
            "permuted_correct_win_fraction": win("permuted_fixed_selection"),
        }
        end_to_end = {
            **normal,
            "permuted_query_grouped_loss": loss("permuted_end_to_end_selection"),
            "permuted_mean_correct_minus_hardest_wrong": end_to_end_gap,
            "permuted_correct_win_fraction": win("permuted_end_to_end_selection"),
        }
        normal_summary = [dict(entry["normal_selection"]) for entry in entries]
        output[key] = {
            "query_count": int(len(entries)),
            "normal": normal,
            "permuted_fixed_selection": {
                "query_grouped_loss": loss("permuted_fixed_selection"),
                "mean_correct_minus_hardest_wrong": fixed_gap,
                "correct_win_fraction": win("permuted_fixed_selection"),
            },
            "permuted_end_to_end_selection": {
                "query_grouped_loss": loss("permuted_end_to_end_selection"),
                "mean_correct_minus_hardest_wrong": end_to_end_gap,
                "correct_win_fraction": win("permuted_end_to_end_selection"),
            },
            "visual_gap_delta_fixed_selection": float(normal_gap - fixed_gap),
            "visual_gap_delta_end_to_end_selection": float(normal_gap - end_to_end_gap),
            "fixed_selection_gate": training_gate_decision(
                normal,
                minimum_win_fraction=float(minimum_win_fraction),
                minimum_normal_gap=float(minimum_normal_gap),
                minimum_visual_gap_delta=float(minimum_visual_gap_delta),
            ),
            "end_to_end_selection_gate": training_gate_decision(
                end_to_end,
                minimum_win_fraction=float(minimum_win_fraction),
                minimum_normal_gap=float(minimum_normal_gap),
                minimum_visual_gap_delta=float(minimum_visual_gap_delta),
            ),
            "normal_selection_mean": {
                "selected_quality_mean": float(
                    np.mean([float(item["selected_quality_mean"]) for item in normal_summary])
                ),
                "occupied_grid_cells": float(
                    np.mean([float(item["occupied_grid_cells"]) for item in normal_summary])
                ),
            },
        }
    return output


@torch.no_grad()
def _evaluate_inner_validation_target_free_selector_sweep(
    *,
    model: torch.nn.Module,
    layout: object,
    groups: Mapping[str, object],
    complete_runtime: object,
    query_ids: Sequence[str],
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    radius_px: float,
    step_px: float,
    cache: TensorImageLRUCache,
    state: _DistributedState,
    policies: Sequence[str],
    point_budgets: Sequence[int],
    grid_rows: int,
    grid_columns: int,
    pose_margin: float,
    missing_edge_log_likelihood_ratio: float,
    max_abs_pose_log_ratio: float,
    amp_enabled: bool,
    permutation_control_shift: int,
    prediction_transform: object,
    context_only: bool,
    rgb_cost_volume_only: bool,
    minimum_win_fraction: float,
    minimum_normal_gap: float,
    minimum_visual_gap_delta: float,
) -> dict[str, object]:
    """Audit selector policies without using target-aware point sampling.

    Each query first runs one visual forward on all 192 frozen anchors.  A
    policy can inspect only the layout and post-encoder density shape, then the
    evaluator joins train-only correct/wrong projections for scoring.  The
    normal and support-permuted passes both select points independently, while
    a fixed-selection permutation control isolates score evidence from a
    selector that happens to choose different points under permuted RGB.
    """

    if (
        not query_ids
        or not policies
        or not point_budgets
        or (bool(context_only) and bool(rgb_cost_volume_only))
    ):
        raise ValueError("target-free selector sweep has no query/policy/budget")
    if any(policy not in TARGET_FREE_SELECTOR_POLICIES for policy in policies):
        raise ValueError("target-free selector sweep policy is invalid")
    model.eval()
    local_rows: list[dict[str, object]] = []
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        if group is None:
            raise ValueError("target-free selector sweep query group is unresolved")
        full_positions = np.arange(group.point_count, dtype=np.int64)
        if any(int(budget) > len(full_positions) for budget in point_budgets):
            raise ValueError("target-free selector point budget exceeds the frozen P1 pool")
        # The selector sees only this layout-derived table and target-free runtime.
        selector_input = selector_input_from_target_free_layout(
            layout=layout, rows=group.layout_rows
        )
        full_runtime = _slice_runtime(complete_runtime, group.layout_rows)
        if bool(context_only):
            query_patches = None
            support_patches = None
        else:
            query_patches, support_patches = _crop_runtime_rgb_patches(
                runtime=full_runtime,
                image_ids=image_ids,
                image_root=image_root,
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=float(radius_px),
                step_px=float(step_px),
                cache=cache,
                device=state.device,
            )
        permuted_runtime = permute_runtime_support_appearance(
            full_runtime, shift=int(permutation_control_shift)
        )
        if bool(context_only):
            permuted_support_patches = None
        else:
            if support_patches is None:
                raise RuntimeError("RGB selector-sweep support patches are unexpectedly absent")
            permuted_support_patches = permute_support_patch_appearance(
                runtime=permuted_runtime,
                support_patches=support_patches,
                shift=int(permutation_control_shift),
            )
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            normal_base = model(
                runtime=full_runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                context_only=bool(context_only),
                rgb_cost_volume_only=bool(rgb_cost_volume_only),
            )
            permuted_base = model(
                runtime=permuted_runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=permuted_support_patches,
                context_only=bool(context_only),
                rgb_cost_volume_only=bool(rgb_cost_volume_only),
            )
        normal_prediction = (
            normal_base if prediction_transform is None else prediction_transform(normal_base)
        )
        permuted_prediction = (
            permuted_base if prediction_transform is None else prediction_transform(permuted_base)
        )
        normal_rgb_quality = target_free_rgb_point_quality(
            runtime=full_runtime, prediction=normal_prediction
        )
        permuted_rgb_quality = target_free_rgb_point_quality(
            runtime=permuted_runtime, prediction=permuted_prediction
        )
        entries: dict[str, object] = {}
        for policy in policies:
            normal_scores = target_free_selector_scores(
                selector_input=selector_input,
                policy=str(policy),
                rgb_quality=normal_rgb_quality,
            )
            permuted_scores = target_free_selector_scores(
                selector_input=selector_input,
                policy=str(policy),
                rgb_quality=permuted_rgb_quality,
            )
            for budget in point_budgets:
                normal_positions = select_target_free_spatial_quota(
                    selector_input=selector_input,
                    quality_scores=normal_scores,
                    point_budget=int(budget),
                    grid_rows=int(grid_rows),
                    grid_columns=int(grid_columns),
                    image_size=coordinate_image_size,
                )
                permuted_positions = select_target_free_spatial_quota(
                    selector_input=selector_input,
                    quality_scores=permuted_scores,
                    point_budget=int(budget),
                    grid_rows=int(grid_rows),
                    grid_columns=int(grid_columns),
                    image_size=coordinate_image_size,
                )
                entry_key = f"{policy}@{int(budget)}"
                entries[entry_key] = {
                    "policy": str(policy),
                    "point_budget": int(budget),
                    "normal": _target_free_selector_pose_metrics(
                        runtime=full_runtime,
                        prediction=normal_prediction,
                        group=group,
                        positions=normal_positions,
                        device=state.device,
                        pose_margin=float(pose_margin),
                        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
                        max_abs_pose_log_ratio=float(max_abs_pose_log_ratio),
                    ),
                    "permuted_fixed_selection": _target_free_selector_pose_metrics(
                        runtime=permuted_runtime,
                        prediction=permuted_prediction,
                        group=group,
                        positions=normal_positions,
                        device=state.device,
                        pose_margin=float(pose_margin),
                        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
                        max_abs_pose_log_ratio=float(max_abs_pose_log_ratio),
                    ),
                    "permuted_end_to_end_selection": _target_free_selector_pose_metrics(
                        runtime=permuted_runtime,
                        prediction=permuted_prediction,
                        group=group,
                        positions=permuted_positions,
                        device=state.device,
                        pose_margin=float(pose_margin),
                        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
                        max_abs_pose_log_ratio=float(max_abs_pose_log_ratio),
                    ),
                    "normal_selection": summarize_target_free_point_selection(
                        selector_input=selector_input,
                        selected_positions=normal_positions,
                        quality_scores=normal_scores,
                        grid_rows=int(grid_rows),
                        grid_columns=int(grid_columns),
                        image_size=coordinate_image_size,
                    ),
                    "permuted_selection": summarize_target_free_point_selection(
                        selector_input=selector_input,
                        selected_positions=permuted_positions,
                        quality_scores=permuted_scores,
                        grid_rows=int(grid_rows),
                        grid_columns=int(grid_columns),
                        image_size=coordinate_image_size,
                    ),
                }
        local_rows.append({"query_id": str(query_id), "entries": entries})
    if state.enabled:
        gathered: list[list[dict[str, object]] | None] = [None] * state.world_size
        distributed.all_gather_object(gathered, local_rows)
        rows = [row for part in gathered if part is not None for row in part]
    else:
        rows = local_rows
    rows = sorted(rows, key=lambda row: str(row["query_id"]))
    if len(rows) != len(query_ids):
        raise RuntimeError("target-free selector sweep did not cover every inner-validation query")
    return {
        "format": "candidate_pose_rgb_spatial_target_free_selector_audit_v1",
        "target_joined_only_after_network_prediction": True,
        "selector_inputs_target_free": True,
        "historic_supervision_preserving_sampler_used": False,
        "hypothesis_dependent_active_normalization_used": False,
        "per_query": rows,
        "summary": _summarize_target_free_selector_rows(
            rows=rows,
            minimum_win_fraction=float(minimum_win_fraction),
            minimum_normal_gap=float(minimum_normal_gap),
            minimum_visual_gap_delta=float(minimum_visual_gap_delta),
        ),
    }


def audit_candidate_pose_rgb_spatial_hard_pose_pretrain_p1(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Run the frozen P1 transfer audit without updating any model weight."""

    context_windows = _validate_args(args)
    context_encoder_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
        args.context_encoder_arch
    )
    score_components = _parse_score_components(args.score_components)
    selector_policies = (
        _parse_target_free_selector_policies(args.target_free_selector_policies)
        if bool(args.target_free_selector_sweep)
        else ()
    )
    selector_budgets = (
        _parse_target_free_selector_budgets(args.target_free_selector_point_budgets)
        if bool(args.target_free_selector_sweep)
        else ()
    )
    gate_component = str(args.gate_component).strip().lower()
    if gate_component not in score_components:
        raise ValueError("P1 transfer gate component must be included in score components")
    checkpoint_path, checkpoint_kind = _checkpoint_request(args)
    context_only_checkpoint = _checkpoint_kind_requires_context_only_replay(checkpoint_kind)
    if context_only_checkpoint:
        if (
            gate_component != "context_only"
            or score_components != ("context_only",)
            or bool(args.rgb_cost_volume_only)
        ):
            raise ValueError(
                "context-only P1 replay must audit only context_only without RGB cost-volume mode"
            )
        if bool(args.target_free_selector_sweep) and any(
            "rgb" in str(policy) for policy in selector_policies
        ):
            raise ValueError("context-only P1 selector replay requires static non-RGB policies")
    state = _initialize_distributed(str(args.device))
    try:
        output_path = Path(args.output)
        if not bool(args.force) and _distributed_output_conflict(
            state=state, output=output_path
        ):
            raise FileExistsError(f"refusing to overwrite existing P1 transfer audit: {output_path}")
        if state.enabled:
            distributed.barrier()
        random.seed(int(args.seed) + state.rank)
        np.random.seed(int(args.seed) + state.rank)
        torch.manual_seed(int(args.seed) + state.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed) + state.rank)
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:  # pragma: no cover - torch compatibility
            pass

        layout_path = Path(args.rgb_spatial_layout)
        targets_path = Path(args.training_targets)
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        targets = load_candidate_pose_rgb_spatial_training_targets(targets_path)
        validate_training_layout_and_targets(
            layout=layout,
            targets=targets,
            layout_sha256=file_sha256_short(layout_path),
        )
        registered_identity_targets_path = (
            Path(str(args.registered_identity_targets))
            if str(args.registered_identity_targets).strip()
            else None
        )
        registered_identity_masks: dict[int, np.ndarray] | None = None
        if registered_identity_targets_path is not None:
            registered_identity_targets = load_candidate_pose_rgb_spatial_training_targets(
                registered_identity_targets_path
            )
            validate_training_layout_and_targets(
                layout=layout,
                targets=registered_identity_targets,
                layout_sha256=file_sha256_short(layout_path),
            )
            registered_identity_masks = _registered_identity_masks_by_source(
                registered_identity_targets=registered_identity_targets,
                expected_source_point_ids=np.asarray(targets.source_point_ids, dtype=np.int64),
                expected_query_ids=np.asarray(targets.query_ids).astype(str),
                candidate_count=int(layout.candidate_count),
            )
        target_radius = float(targets.metadata["spatial_search_radius_px"])
        search_radius = (
            target_radius if args.search_radius_px is None else float(args.search_radius_px)
        )
        if not math.isclose(search_radius, target_radius, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("P1 transfer audit search radius differs from train-only targets")
        groups = build_train_query_groups(layout=layout, targets=targets)
        all_query_ids = tuple(sorted(groups))
        _inner_train, inner_validation = _partition_train_queries_for_inner_validation(
            query_ids=all_query_ids,
            fold_count=int(args.inner_validation_fold_count),
            fold_index=int(args.inner_validation_fold_index),
        )
        point_selection_contract = _p1_point_selection_contract(
            layout=layout,
            groups=groups,
            query_ids=inner_validation,
            max_points_per_query=int(args.max_points_per_query),
        )

        if bool(args.rgb_cost_volume_only):
            source_headers = load_context_attention_source_headers(
                radio_final_context_cache=Path(args.radio_final_context_cache),
                radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
                alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
                expected_radio_checkpoint="",
            )
            image_ids = source_headers.image_ids
            image_sizes = source_headers.image_sizes
            source_tensors = None
            context_source_dimensions = source_headers.descriptor_dimensions
            source_metadata = source_headers.metadata_by_name["radio_final"]
        else:
            sources = load_context_attention_sources(
                radio_final_context_cache=Path(args.radio_final_context_cache),
                radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
                alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
                expected_radio_checkpoint="",
                require_equal_descriptor_dimensions=False,
            )
            image_ids, image_sizes, source_tensors = _source_table(sources)
            context_source_dimensions = None
            source_metadata = sources[0].metadata
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("hard-pose P1 transfer audit requires common image dimensions")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=source_metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        context_cache_paths = {
            "radio_final": Path(args.radio_final_context_cache),
            "radio_intermediate": Path(args.radio_intermediate_context_cache),
            "alike": Path(args.alike_spatial_context_cache),
        }
        model = CandidatePoseRGBSpatialLikelihood(
            sources=source_tensors,
            context_source_dimensions=context_source_dimensions,
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            search_radius_px=search_radius,
            context_radius_px=float(args.context_radius_px),
            step_px=float(args.step_px),
            texture_feature_dim=int(args.texture_feature_dim),
            hidden_dim=int(args.hidden_dim),
            max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
            edge_chunk_size=int(args.edge_chunk_size),
            activation_checkpointing=False,
            context_windows=context_windows,
            context_encoder_arch=context_encoder_arch,
        )
        if checkpoint_kind == "hard_pose_pretrain":
            initialization = load_hard_pose_pretrain_initialization_checkpoint(
                path=checkpoint_path,
                model=model,
                source_cache_paths=context_cache_paths,
                source_image_manifest_sha256=str(
                    source_metadata.get("source_image_manifest_sha256", "")
                ),
                rgb_coordinate_bridge=rgb_bridge,
                search_radius_px=search_radius,
                context_radius_px=float(args.context_radius_px),
                step_px=float(args.step_px),
                texture_feature_dim=int(args.texture_feature_dim),
                hidden_dim=int(args.hidden_dim),
                max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
                context_windows=context_windows,
                context_encoder_arch=context_encoder_arch,
                allow_diagnostic_ineligible=bool(args.allow_ineligible_diagnostic_checkpoint),
            )
        elif checkpoint_kind == "observation_pretrain":
            initialization = load_observation_pretrain_initialization_checkpoint(
                path=checkpoint_path,
                model=model,
                source_cache_paths=context_cache_paths,
                source_image_manifest_sha256=str(
                    source_metadata.get("source_image_manifest_sha256", "")
                ),
                rgb_coordinate_bridge=rgb_bridge,
                search_radius_px=search_radius,
                context_radius_px=float(args.context_radius_px),
                step_px=float(args.step_px),
                texture_feature_dim=int(args.texture_feature_dim),
                hidden_dim=int(args.hidden_dim),
                max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
                context_windows=context_windows,
                context_encoder_arch=context_encoder_arch,
            )
        elif checkpoint_kind == "context_observation_pretrain":
            if gate_component != "context_only":
                raise ValueError(
                    "context-observation P1 transfer audit must gate the context_only component"
                )
            initialization = load_context_observation_pretrain_initialization_checkpoint(
                path=checkpoint_path,
                model=model,
                source_cache_paths=context_cache_paths,
                source_image_manifest_sha256=str(
                    source_metadata.get("source_image_manifest_sha256", "")
                ),
                search_radius_px=search_radius,
                context_radius_px=float(args.context_radius_px),
                step_px=float(args.step_px),
                texture_feature_dim=int(args.texture_feature_dim),
                hidden_dim=int(args.hidden_dim),
                max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
                context_windows=context_windows,
                context_encoder_arch=context_encoder_arch,
            )
        elif checkpoint_kind == "context_identity_l0":
            initialization = load_context_identity_l0_initialization_checkpoint(
                path=checkpoint_path,
                model=model,
                layout_sha256=file_sha256_short(layout_path),
                targets_sha256=file_sha256_short(targets_path),
                candidate_count=int(layout.candidate_count),
                support_view_count=int(layout.support_view_count),
                source_cache_paths=context_cache_paths,
                source_image_manifest_sha256=str(
                    source_metadata.get("source_image_manifest_sha256", "")
                ),
                search_radius_px=search_radius,
                context_radius_px=float(args.context_radius_px),
                step_px=float(args.step_px),
                texture_feature_dim=int(args.texture_feature_dim),
                hidden_dim=int(args.hidden_dim),
                max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
                context_windows=context_windows,
                context_encoder_arch=context_encoder_arch,
                allow_diagnostic_ineligible=bool(args.allow_ineligible_diagnostic_checkpoint),
            )
        elif checkpoint_kind == "p1_cross_layout_curriculum_checkpoint":
            initialization = _load_cross_layout_curriculum_checkpoint(
                path=checkpoint_path,
                model=model,
                destination_layout=layout,
                source_cache_paths=context_cache_paths,
                source_image_manifest_sha256=str(
                    source_metadata.get("source_image_manifest_sha256", "")
                ),
                rgb_coordinate_bridge=rgb_bridge,
                search_radius_px=search_radius,
                context_radius_px=float(args.context_radius_px),
                step_px=float(args.step_px),
                texture_feature_dim=int(args.texture_feature_dim),
                hidden_dim=int(args.hidden_dim),
                max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
                context_windows=context_windows,
                context_encoder_arch=context_encoder_arch,
                component_audit_path=Path(args.cross_layout_component_audit),
            )
        else:
            initialization = load_target_free_initialization_checkpoint(
                path=checkpoint_path,
                model=model,
                layout_sha256=file_sha256_short(layout_path),
                targets_sha256=file_sha256_short(targets_path),
                candidate_count=int(layout.candidate_count),
                support_view_count=int(layout.support_view_count),
                search_radius_px=search_radius,
                context_radius_px=float(args.context_radius_px),
                step_px=float(args.step_px),
                texture_feature_dim=int(args.texture_feature_dim),
                hidden_dim=int(args.hidden_dim),
                max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
                context_encoder_arch=context_encoder_arch,
                permitted_parent_target_sha256s=(
                    full_pose_hard_target_initializer_parent_hashes(
                        targets=targets,
                        targets_sha256=file_sha256_short(targets_path),
                    )
                ),
            )
            if bool(initialization.get("rgb_cost_volume_only", False)) != bool(
                args.rgb_cost_volume_only
            ):
                raise ValueError(
                    "P1 checkpoint RGB-only mode differs from the frozen replay request"
                )
        model = model.to(state.device)
        if state.enabled:
            model_for_audit: torch.nn.Module = DistributedDataParallel(
                model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
                broadcast_buffers=False,
            )
        else:
            model_for_audit = model
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        prediction_transforms = {}
        for component in score_components:
            prediction_transforms[component] = (
                None
                if component == "combined"
                else lambda prediction, name=component: candidate_pose_rgb_spatial_score_component_prediction(
                    prediction=prediction, component=name
                )
            )
        component_metrics = _evaluate_inner_validation_components(
            model=model_for_audit,
            groups=groups,
            complete_runtime=complete_runtime,
            query_ids=inner_validation,
            image_ids=image_ids,
            image_root=Path(args.image_root),
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=search_radius + float(args.context_radius_px),
            step_px=float(args.step_px),
            cache=cache,
            state=state,
            max_points_per_query=int(args.max_points_per_query),
            seed=int(args.seed),
            pose_margin=float(args.pose_margin),
            missing_edge_log_likelihood_ratio=float(args.missing_edge_log_likelihood_ratio),
            max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
            amp_enabled=state.device.type == "cuda" and not bool(args.no_amp),
            permutation_control_shift=int(args.permutation_control_shift),
            prediction_transforms=prediction_transforms,
            context_only=bool(context_only_checkpoint),
            rgb_cost_volume_only=bool(args.rgb_cost_volume_only),
        )
        edge_evidence = (
            _evaluate_inner_validation_edge_evidence_layers(
                model=model_for_audit,
                groups=groups,
                complete_runtime=complete_runtime,
                query_ids=inner_validation,
                image_ids=image_ids,
                image_root=Path(args.image_root),
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=search_radius + float(args.context_radius_px),
                step_px=float(args.step_px),
                cache=cache,
                state=state,
                max_points_per_query=int(args.max_points_per_query),
                seed=int(args.seed),
                missing_edge_log_likelihood_ratio=float(args.missing_edge_log_likelihood_ratio),
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                amp_enabled=state.device.type == "cuda" and not bool(args.no_amp),
                prediction_transform=prediction_transforms[gate_component],
                rgb_cost_volume_only=bool(args.rgb_cost_volume_only),
                registered_identity_masks_by_source=registered_identity_masks,
            )
            if bool(args.edge_evidence_layers)
            else None
        )
        target_free_selector_sweep = (
            _evaluate_inner_validation_target_free_selector_sweep(
                model=model_for_audit,
                layout=layout,
                groups=groups,
                complete_runtime=complete_runtime,
                query_ids=inner_validation,
                image_ids=image_ids,
                image_root=Path(args.image_root),
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=search_radius + float(args.context_radius_px),
                step_px=float(args.step_px),
                cache=cache,
                state=state,
                policies=selector_policies,
                point_budgets=selector_budgets,
                grid_rows=int(args.target_free_selector_grid_rows),
                grid_columns=int(args.target_free_selector_grid_columns),
                pose_margin=float(args.pose_margin),
                missing_edge_log_likelihood_ratio=float(args.missing_edge_log_likelihood_ratio),
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                amp_enabled=state.device.type == "cuda" and not bool(args.no_amp),
                permutation_control_shift=int(args.permutation_control_shift),
                prediction_transform=prediction_transforms[gate_component],
                context_only=bool(context_only_checkpoint),
                rgb_cost_volume_only=bool(args.rgb_cost_volume_only),
                minimum_win_fraction=float(args.minimum_win_fraction),
                minimum_normal_gap=float(args.minimum_normal_gap),
                minimum_visual_gap_delta=float(args.minimum_visual_gap_delta),
            )
            if bool(args.target_free_selector_sweep)
            else None
        )
        metrics = component_metrics[gate_component]
        visual_gap_delta = float(
            metrics["normal_mean_correct_minus_hardest_wrong"]
            - metrics["permuted_mean_correct_minus_hardest_wrong"]
        )
        gate = training_gate_decision(
            metrics,
            minimum_win_fraction=float(args.minimum_win_fraction),
            minimum_normal_gap=float(args.minimum_normal_gap),
            minimum_visual_gap_delta=float(args.minimum_visual_gap_delta),
        )
        if state.rank == 0:
            result = {
                "format": AUDIT_FORMAT,
                "stage": "audit_candidate_pose_rgb_spatial_checkpoint_p1",
                "checkpoint": initialization,
                "metrics": {**metrics, "visual_gap_delta": visual_gap_delta},
                "component_metrics": {
                    component: {
                        **values,
                        "visual_gap_delta": float(
                            values["normal_mean_correct_minus_hardest_wrong"]
                            - values["permuted_mean_correct_minus_hardest_wrong"]
                        ),
                    }
                    for component, values in component_metrics.items()
                },
                "zero_shot_transfer_gate": gate,
                "inputs": {
                    "rgb_spatial_layout": {
                        "path": str(layout_path),
                        "sha256": file_sha256_short(layout_path),
                    },
                    "training_targets": {
                        "path": str(targets_path),
                        "sha256": file_sha256_short(targets_path),
                    },
                    "registered_identity_targets": (
                        None
                        if registered_identity_targets_path is None
                        else {
                            "path": str(registered_identity_targets_path),
                            "sha256": file_sha256_short(
                                registered_identity_targets_path
                            ),
                        }
                    ),
                },
                "config": {
                    "search_radius_px": search_radius,
                    "context_radius_px": float(args.context_radius_px),
                    "step_px": float(args.step_px),
                    "max_points_per_query": int(args.max_points_per_query),
                    "context_windows": context_windows,
                    "context_encoder_arch": context_encoder_arch,
                    "score_components": list(score_components),
                    "gate_component": gate_component,
                    "inner_validation_fold_count": int(args.inner_validation_fold_count),
                    "inner_validation_fold_index": int(args.inner_validation_fold_index),
                    "inner_validation_query_count": int(len(inner_validation)),
                    "world_size": int(state.world_size),
                    "allow_ineligible_diagnostic_checkpoint": bool(
                        args.allow_ineligible_diagnostic_checkpoint
                    ),
                    "checkpoint_kind": checkpoint_kind,
                    "context_only_checkpoint": bool(context_only_checkpoint),
                    "rgb_cost_volume_only": bool(args.rgb_cost_volume_only),
                    "rgb_cache_dtype": str(args.rgb_cache_dtype),
                    "p1_point_selection": point_selection_contract,
                    "target_free_selector_sweep": bool(args.target_free_selector_sweep),
                    "target_free_selector_policies": list(selector_policies),
                    "target_free_selector_point_budgets": list(selector_budgets),
                    "target_free_selector_grid_rows": int(args.target_free_selector_grid_rows),
                    "target_free_selector_grid_columns": int(args.target_free_selector_grid_columns),
                },
                "protocol": {
                    "frozen_checkpoint": True,
                    "model_weights_updated": False,
                    "runtime_layout_target_free": True,
                    "pose_targets_joined_only_after_network_prediction": True,
                    "train_query_only": True,
                    "external_validation_or_test_used": False,
                    "pnp_or_pose_estimation_run": False,
                    "no_render": True,
                    "no_image_retrieval_or_submap": True,
                    "radio_alike_context_bypassed": bool(args.rgb_cost_volume_only),
                    "context_only_model_forward": bool(context_only_checkpoint),
                    "diagnostic_ineligible_checkpoint_override": bool(
                        args.allow_ineligible_diagnostic_checkpoint
                    ),
                    "registered_observation_oracle": (
                        "unavailable_without_explicit_registered_identity_targets_v1"
                        if registered_identity_targets_path is None
                        else "explicit_registered_exact_identity_targets_joined_after_visual_forward_v1"
                    ),
                    "checkpoint_remains_ineligible_for_p1_or_pnp": bool(
                        initialization.get("eligible_for_p1_finetune") is not True
                    ),
                    "historic_component_metrics_use_train_target_observation_preserving_sampler": not bool(
                        point_selection_contract["runtime_eligible"]
                    ),
                    "historic_component_metrics_are_not_a_runtime_selector_gate": not bool(
                        point_selection_contract["runtime_eligible"]
                    ),
                    "frozen_target_free_selector_rows_replayed_without_subsampling": bool(
                        point_selection_contract["runtime_eligible"]
                    ),
                },
                "rgb_cache_rank0": cache.summary(),
            }
            if edge_evidence is not None:
                result["edge_evidence_layers"] = edge_evidence
            if target_free_selector_sweep is not None:
                result["target_free_selector_sweep"] = target_free_selector_sweep
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if state.enabled:
            distributed.barrier()
        return {"output": str(output_path), "rank": int(state.rank)}
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_candidate_pose_rgb_spatial_hard_pose_pretrain_p1(args)
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
