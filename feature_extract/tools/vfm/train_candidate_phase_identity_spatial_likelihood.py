"""Train a strict RADIO-final identity plus real-RGB spatial pose likelihood.

The frozen RADIO-final phase expert is admitted only after its own
registered-exact, support-appearance re-audit passes.  This trainer then fits
a fresh high-resolution RGB *spatial* density from real images.  The runtime
composition is deliberately narrow:

* RADIO-final scores a fixed query-point/candidate/support-view identity.
* RGB scores a candidate-specific local spatial mode at a pose projection.
* Both factors are used only where both visual sources are available and the
  RGB projection is in-window.
* Fixed top-L candidate and explicit-null mass are marginalized without
  reselecting candidates or support views.

The RGB branch also receives train-only RGB-only support-derangement controls,
registered-observation contrast, and exact-identity static hard repeats from
the phase inner-train partition.  Those labels are joined only after the
target-free visual forwards; the phase held-out partition remains gate-only.

The phase expert is frozen.  Train-only correct/wrong pose targets, exact
hard-repeat pairs, and all labels are joined only after both visual forwards.
The final inner-fold gate is a diagnostic gate: even a pass allows only a
frozen held-out pose-rank audit, never PnP integration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.audit_candidate_multiscale_phase_identity_llr import (
    AUDIT_FORMAT,
    _checkpoint_partition,
    _load_checkpoint as _load_phase_checkpoint,
    _load_model as _load_phase_model,
    _validate_checkpoint as _validate_phase_checkpoint,
    promotable_phase_identity_sources,
)
from feature_extract.tools.vfm.train_candidate_multiscale_phase_identity_llr import (
    _source_table,
    build_exact_identity_query_targets,
    filter_static_hard_repeat_groups_to_registered_exact_identity,
    validate_external_frozen_current_hard_targets,
)
from feature_extract.tools.vfm.train_candidate_highres_rgb_multiscale_likelihood import (
    FIXED_FINAL_EPOCH_SELECTION_POLICY,
    appearance_control_margin_loss,
    balanced_ddp_query_schedules,
    configure_trainable_parameters,
    ddp_owner_cost_balance_metrics,
    pose_margin_terms,
    registered_observation_appearance_control_terms,
    target_free_query_owner_costs,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatBatch,
    HardRepeatQueryTargets,
    TrainQueryGroup,
    _DistributedState,
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _finalize_distributed,
    _hard_repeat_batch_from_group,
    _initialize_distributed,
    _query_batch_from_group,
    _select_group_points,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    validate_rgb_coordinate_bridge,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
    CandidateHighresRGBMultiscaleLikelihood,
    CandidateHighresRGBMultiscalePrediction,
    highres_rgb_spatial_density_nll,
)
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_LLR_FORMAT,
    CandidateMultiscalePhaseIdentityLLR,
    CandidateMultiscalePhaseIdentityPrediction,
)
from feature_extract.vfm.localization.candidate_phase_identity_spatial_likelihood import (
    CandidatePhaseIdentitySpatialPoseScore,
    permute_support_patches_with_phase_identity_point_blocks,
    resolve_phase_spatial_weights,
    score_candidate_phase_identity_spatial_batch,
    selected_candidate_phase_identity_spatial_log_likelihood_ratios,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


CHECKPOINT_FORMAT = "candidate_phase_identity_spatial_likelihood_checkpoint_v1"
HYBRID_SOURCE_NAME = "radio_final_plus_rgb_fine_strict_common_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--geometry-training-targets", required=True)
    parser.add_argument("--registered-identity-targets", required=True)
    parser.add_argument("--current-hard-repeat-targets", required=True)
    parser.add_argument("--current-hard-mining-checkpoint", required=True)
    parser.add_argument("--static-hard-repeat-targets", required=True)
    parser.add_argument("--phase-checkpoint", required=True)
    parser.add_argument("--phase-reaudit", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine-search-radius-px", type=float, default=8.0)
    parser.add_argument("--fine-context-radius-px", type=float, default=12.0)
    parser.add_argument("--texture-feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--edge-chunk-size", type=int, default=128)
    parser.add_argument("--rgb-temperature", type=float, default=10.0)
    parser.add_argument("--max-abs-edge-log-ratio", type=float, default=3.0)
    parser.add_argument("--max-abs-pose-log-ratio", type=float, default=6.0)
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--spatial-weight", type=float, default=1.0)
    parser.add_argument("--max-points-per-query", type=int, default=64)
    parser.add_argument("--max-current-hard-edges-per-query", type=int, default=64)
    parser.add_argument("--max-static-train-hard-edges-per-query", type=int, default=64)
    parser.add_argument("--max-static-hard-edges-per-query", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--density-loss-weight", type=float, default=0.25)
    parser.add_argument("--dustbin-loss-weight", type=float, default=0.50)
    parser.add_argument("--pose-loss-weight", type=float, default=1.0)
    parser.add_argument("--soft-hard-loss-weight", type=float, default=0.50)
    parser.add_argument("--soft-hard-temperature", type=float, default=0.35)
    parser.add_argument("--pose-margin", type=float, default=0.20)
    parser.add_argument("--hard-repeat-loss-weight", type=float, default=1.0)
    parser.add_argument("--static-hard-repeat-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-repeat-margin", type=float, default=0.20)
    parser.add_argument("--pose-appearance-control-loss-weight", type=float, default=0.25)
    parser.add_argument("--hard-repeat-appearance-control-loss-weight", type=float, default=0.25)
    parser.add_argument("--rgb-only-pose-control-loss-weight", type=float, default=0.50)
    parser.add_argument("--rgb-only-hard-repeat-control-loss-weight", type=float, default=0.25)
    parser.add_argument("--static-hard-repeat-rgb-control-loss-weight", type=float, default=0.25)
    parser.add_argument("--registered-observation-rgb-control-loss-weight", type=float, default=0.50)
    parser.add_argument("--registered-observation-rgb-control-margin", type=float, default=0.05)
    parser.add_argument("--appearance-control-margin", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp-init-scale", type=float, default=4096.0)
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument("--rgb-cache-dtype", choices=("uint8", "float16"), default="uint8")
    parser.add_argument("--rgb-cache-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--support-permutation-shift", type=int, default=1)
    parser.add_argument("--minimum-pose-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-pose-gap", type=float, default=0.05)
    parser.add_argument("--minimum-both-visual-pose-delta", type=float, default=0.05)
    parser.add_argument("--minimum-rgb-visual-pose-delta", type=float, default=0.005)
    parser.add_argument("--minimum-hard-eligible-query-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-hard-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-hard-gap", type=float, default=0.05)
    parser.add_argument("--minimum-both-visual-hard-delta", type=float, default=0.05)
    parser.add_argument("--minimum-rgb-visual-hard-delta", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if (
        int(args.epochs) <= 0
        or int(args.max_points_per_query) < 4
        or int(args.max_current_hard_edges_per_query) < 0
        or int(args.max_static_train_hard_edges_per_query) < 0
        or int(args.max_static_hard_edges_per_query) < 0
        or int(args.edge_chunk_size) <= 0
        or int(args.texture_feature_dim) < 4
        or int(args.hidden_dim) < 4
        or int(args.support_permutation_shift) == 0
        or float(args.learning_rate) <= 0.0
        or float(args.weight_decay) < 0.0
        or float(args.rgb_cache_gb) <= 0.0
    ):
        raise ValueError("phase/spatial hybrid training arguments are invalid")
    finite = (
        args.fine_search_radius_px,
        args.fine_context_radius_px,
        args.rgb_temperature,
        args.max_abs_edge_log_ratio,
        args.max_abs_pose_log_ratio,
        args.identity_weight,
        args.spatial_weight,
        args.density_loss_weight,
        args.dustbin_loss_weight,
        args.pose_loss_weight,
        args.soft_hard_loss_weight,
        args.soft_hard_temperature,
        args.pose_margin,
        args.hard_repeat_loss_weight,
        args.static_hard_repeat_loss_weight,
        args.hard_repeat_margin,
        args.pose_appearance_control_loss_weight,
        args.hard_repeat_appearance_control_loss_weight,
        args.rgb_only_pose_control_loss_weight,
        args.rgb_only_hard_repeat_control_loss_weight,
        args.static_hard_repeat_rgb_control_loss_weight,
        args.registered_observation_rgb_control_loss_weight,
        args.registered_observation_rgb_control_margin,
        args.appearance_control_margin,
        args.gradient_clip_norm,
        args.amp_init_scale,
        args.minimum_pose_win_fraction,
        args.minimum_pose_gap,
        args.minimum_both_visual_pose_delta,
        args.minimum_rgb_visual_pose_delta,
        args.minimum_hard_eligible_query_fraction,
        args.minimum_hard_win_fraction,
        args.minimum_hard_gap,
        args.minimum_both_visual_hard_delta,
        args.minimum_rgb_visual_hard_delta,
    )
    if (
        not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in finite)
        or float(args.max_abs_edge_log_ratio) <= 0.0
        or float(args.max_abs_pose_log_ratio) <= 0.0
        or float(args.rgb_temperature) <= 0.0
        or float(args.soft_hard_temperature) <= 0.0
        or not 0.0 < float(args.minimum_pose_win_fraction) <= 1.0
        or not 0.0 < float(args.minimum_hard_eligible_query_fraction) <= 1.0
        or not 0.0 <= float(args.minimum_hard_win_fraction) <= 1.0
    ):
        raise ValueError("phase/spatial hybrid floating-point arguments are invalid")
    resolve_phase_spatial_weights(
        identity_weight=float(args.identity_weight), spatial_weight=float(args.spatial_weight)
    )


def hybrid_inner_gate_decision(*, metrics: Mapping[str, float], args: argparse.Namespace) -> dict[str, bool]:
    """Require independent pose, exact-repeat, and RGB-control evidence."""

    required = (
        "normal_pose_win_fraction",
        "normal_pose_gap",
        "normal_minus_both_deranged_pose_gap",
        "normal_minus_rgb_deranged_pose_gap",
        "hard_eligible_query_fraction",
        "hard_win_fraction",
        "hard_gap",
        "hard_normal_minus_both_deranged_gap",
        "hard_normal_minus_rgb_deranged_gap",
    )
    if any(name not in metrics or not math.isfinite(float(metrics[name])) for name in required):
        raise ValueError("phase/spatial hybrid inner gate metrics are incomplete")
    checks = {
        "pose_win": float(metrics["normal_pose_win_fraction"]) >= float(args.minimum_pose_win_fraction),
        "pose_gap": float(metrics["normal_pose_gap"]) >= float(args.minimum_pose_gap),
        "both_visual_pose": float(metrics["normal_minus_both_deranged_pose_gap"])
        >= float(args.minimum_both_visual_pose_delta),
        "rgb_visual_pose": float(metrics["normal_minus_rgb_deranged_pose_gap"])
        >= float(args.minimum_rgb_visual_pose_delta),
        "hard_coverage": float(metrics["hard_eligible_query_fraction"])
        >= float(args.minimum_hard_eligible_query_fraction),
        "hard_win": float(metrics["hard_win_fraction"]) >= float(args.minimum_hard_win_fraction),
        "hard_gap": float(metrics["hard_gap"]) >= float(args.minimum_hard_gap),
        "both_visual_hard": float(metrics["hard_normal_minus_both_deranged_gap"])
        >= float(args.minimum_both_visual_hard_delta),
        "rgb_visual_hard": float(metrics["hard_normal_minus_rgb_deranged_gap"])
        >= float(args.minimum_rgb_visual_hard_delta),
    }
    return {**checks, "passed": bool(all(checks.values()))}


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("phase identity re-audit is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError("phase identity re-audit is invalid")
    return value


def _validate_phase_reaudit(
    *,
    re_audit: Mapping[str, object],
    re_audit_path: Path,
    phase_checkpoint: Path,
    paths: Mapping[str, Path],
) -> None:
    if re_audit.get("format") != AUDIT_FORMAT or not Path(re_audit_path).is_file():
        raise ValueError("phase identity re-audit has the wrong format")
    if str(re_audit.get("checkpoint_sha256", "")) != file_sha256_short(phase_checkpoint):
        raise ValueError("phase identity re-audit refers to another checkpoint")
    promotable = promotable_phase_identity_sources(re_audit.get("gate", {}))
    promotion = re_audit.get("promotion")
    inputs = re_audit.get("inputs")
    if (
        "radio_final" not in promotable
        or not isinstance(promotion, Mapping)
        or promotion.get("allowed") is not True
        or promotion.get("pnp_integration_allowed") is not False
        or not isinstance(inputs, Mapping)
    ):
        raise ValueError("phase identity re-audit does not admit RADIO-final to hybrid training")
    expected = {
        "layout": paths["layout"],
        "geometry": paths["geometry"],
        "identity": paths["identity"],
        "static_hard": paths["static_hard"],
    }
    for name, path in expected.items():
        item = inputs.get(name)
        if not isinstance(item, Mapping) or str(item.get("sha256", "")) != file_sha256_short(path):
            raise ValueError("phase identity re-audit input lineage differs from hybrid training")


def _phase_control_common_availability(
    *,
    phase_normal: CandidateMultiscalePhaseIdentityPrediction,
    phase_deranged: CandidateMultiscalePhaseIdentityPrediction,
    rgb_normal: CandidateHighresRGBMultiscalePrediction,
    rgb_deranged: CandidateHighresRGBMultiscalePrediction,
) -> torch.Tensor:
    """Keep all paired variants on an identical source-availability denominator."""

    normal_phase = phase_normal.source_edge_usable["radio_final"]
    deranged_phase = phase_deranged.source_edge_usable["radio_final"].to(normal_phase.device)
    normal_rgb = rgb_normal.sources["fine"].edge_usable.to(normal_phase.device)
    deranged_rgb = rgb_deranged.sources["fine"].edge_usable.to(normal_phase.device)
    if (
        not (
            normal_phase.shape
            == deranged_phase.shape
            == normal_rgb.shape
            == deranged_rgb.shape
        )
        or not torch.equal(normal_rgb, deranged_rgb)
    ):
        raise ValueError("hybrid appearance control changed RGB source availability")
    return normal_phase & deranged_phase & normal_rgb & deranged_rgb


def _hybrid_pose_scores(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    phase_prediction: CandidateMultiscalePhaseIdentityPrediction,
    rgb_prediction: CandidateHighresRGBMultiscalePrediction,
    availability: torch.Tensor,
    correct_offsets_xy: torch.Tensor,
    correct_valid: torch.Tensor,
    wrong_offsets_xy: torch.Tensor,
    wrong_valid: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[CandidatePhaseIdentitySpatialPoseScore, CandidatePhaseIdentitySpatialPoseScore]:
    common = {
        "runtime": runtime,
        "phase_prediction": phase_prediction,
        "spatial_prediction": rgb_prediction,
        "phase_source_name": "radio_final",
        "spatial_source_name": "fine",
        "identity_weight": float(args.identity_weight),
        "spatial_weight": float(args.spatial_weight),
        "edge_availability_override": availability,
        "max_abs_spatial_log_likelihood_ratio": float(args.max_abs_pose_log_ratio),
    }
    correct = score_candidate_phase_identity_spatial_batch(
        candidate_projection_offsets_xy=correct_offsets_xy.unsqueeze(0),
        candidate_projection_valid=correct_valid.unsqueeze(0),
        **common,
    )
    wrong = score_candidate_phase_identity_spatial_batch(
        candidate_projection_offsets_xy=wrong_offsets_xy,
        candidate_projection_valid=wrong_valid,
        **common,
    )
    if correct.pose_log_likelihood_ratios.shape != (1,) or wrong.pose_log_likelihood_ratios.ndim != 1:
        raise ValueError("hybrid correct/wrong pose score shapes are invalid")
    return correct, wrong


def _hybrid_hard_gaps(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    phase_prediction: CandidateMultiscalePhaseIdentityPrediction,
    rgb_prediction: CandidateHighresRGBMultiscalePrediction,
    availability: torch.Tensor,
    hard_batch: HardRepeatBatch | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if hard_batch is None:
        return None, None
    common = {
        "runtime": runtime,
        "phase_prediction": phase_prediction,
        "spatial_prediction": rgb_prediction,
        "phase_source_name": "radio_final",
        "spatial_source_name": "fine",
        "identity_weight": float(args.identity_weight),
        "spatial_weight": float(args.spatial_weight),
        "edge_availability_override": availability,
        "max_abs_spatial_log_likelihood_ratio": float(args.max_abs_pose_log_ratio),
    }
    positive, positive_usable = selected_candidate_phase_identity_spatial_log_likelihood_ratios(
        point_indices=hard_batch.point_indices,
        candidate_indices=hard_batch.positive_candidate_indices,
        offsets_xy=hard_batch.positive_offsets_xy,
        **common,
    )
    negative, negative_usable = selected_candidate_phase_identity_spatial_log_likelihood_ratios(
        point_indices=hard_batch.point_indices,
        candidate_indices=hard_batch.negative_candidate_indices,
        offsets_xy=hard_batch.negative_offsets_xy,
        **common,
    )
    return positive - negative, positive_usable & negative_usable


def _required_hard_source_ids(
    *hard_targets: HardRepeatQueryTargets | None,
) -> np.ndarray | None:
    """Return the train-only source-point union needed by multiple hard families."""

    values = [
        np.asarray(target.source_point_ids, dtype=np.int64).reshape(-1)
        for target in hard_targets
        if target is not None
    ]
    if not values:
        return None
    merged = np.unique(np.concatenate(values, axis=0))
    if len(merged) == 0:
        return None
    return merged.astype(np.int64, copy=False)


def _strict_hard_repeat_objective_terms(
    *,
    normal_values: torch.Tensor | None,
    normal_usable: torch.Tensor | None,
    both_deranged_values: torch.Tensor | None,
    both_deranged_usable: torch.Tensor | None,
    rgb_deranged_values: torch.Tensor | None,
    rgb_deranged_usable: torch.Tensor | None,
    hard_margin: float,
    appearance_control_margin: float,
    anchor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Score one exact hard family on a common normal/RGB/phase denominator.

    The hard margin uses the normal visual pairing.  The two control terms
    independently require that normal evidence exceeds a fully deranged
    pairing and an RGB-only derangement.  The latter prevents a frozen phase
    expert from satisfying the appearance objective by itself.
    """

    zero = torch.as_tensor(anchor).sum() * 0.0
    entries = (
        normal_values,
        normal_usable,
        both_deranged_values,
        both_deranged_usable,
        rgb_deranged_values,
        rgb_deranged_usable,
    )
    if any(value is None for value in entries):
        return zero, zero, zero, zero, 0.0
    assert normal_values is not None
    assert normal_usable is not None
    assert both_deranged_values is not None
    assert both_deranged_usable is not None
    assert rgb_deranged_values is not None
    assert rgb_deranged_usable is not None
    if not (
        normal_values.shape
        == both_deranged_values.shape
        == rgb_deranged_values.shape
        == normal_usable.shape
        == both_deranged_usable.shape
        == rgb_deranged_usable.shape
    ):
        raise ValueError("hybrid hard-repeat paired scores have incompatible layouts")
    common = normal_usable & both_deranged_usable & rgb_deranged_usable
    if not bool(common.any()):
        return zero, zero, zero, zero, 0.0
    normal = normal_values[common]
    both = both_deranged_values[common]
    rgb = rgb_deranged_values[common]
    hard_loss = F.softplus(float(hard_margin) - normal).mean()
    both_control, _ = appearance_control_margin_loss(
        normal_gaps=normal,
        permuted_gaps=both,
        margin=float(appearance_control_margin),
    )
    rgb_control, _ = appearance_control_margin_loss(
        normal_gaps=normal,
        permuted_gaps=rgb,
        margin=float(appearance_control_margin),
    )
    return hard_loss, both_control, rgb_control, normal.mean(), float(common.sum().item())


def _reduce(state: _DistributedState, values: torch.Tensor) -> torch.Tensor:
    result = values.detach().clone()
    if state.enabled:
        distributed.all_reduce(result, op=distributed.ReduceOp.SUM)
    return result


def _save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    core = model.module if isinstance(model, DistributedDataParallel) else model
    return {name: value.detach().cpu().clone() for name, value in core.state_dict().items()}


@torch.no_grad()
def evaluate_hybrid_inner_gate(
    *,
    rgb_model: nn.Module,
    phase_model: CandidateMultiscalePhaseIdentityLLR,
    groups: Mapping[str, TrainQueryGroup],
    static_hard_groups: Mapping[str, HardRepeatQueryTargets],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    query_ids: Sequence[str],
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    cache: TensorImageLRUCache,
    cache_device: torch.device,
    state: _DistributedState,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Run the all-P1, query-disjoint hybrid gate with paired ablations."""

    if not query_ids:
        raise ValueError("hybrid inner gate has no held-out queries")
    rgb_core = rgb_model.module if isinstance(rgb_model, DistributedDataParallel) else rgb_model
    if not isinstance(rgb_core, CandidateHighresRGBMultiscaleLikelihood):
        raise ValueError("hybrid inner gate RGB model is invalid")
    rgb_core.eval()
    phase_model.eval()
    # normal, phase-only deranged, RGB-only deranged, both deranged pose
    # gaps/wins; query count; then the same four direct hard-pair gaps/wins,
    # hard edge count, and queries with a common hard edge.
    totals = torch.zeros((19,), dtype=torch.float64, device=state.device)
    for position, query_id in enumerate(query_ids):
        if position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        hard = static_hard_groups.get(str(query_id))
        if group is None or hard is None:
            raise ValueError("hybrid inner gate query is unresolved")
        points = np.arange(group.point_count, dtype=np.int64)
        batch = _query_batch_from_group(
            group=group, complete_runtime=complete_runtime, point_positions=points, device=state.device
        )
        query_patches, support_patches = _crop_runtime_rgb_patches(
            runtime=batch.runtime,
            image_ids=image_ids,
            image_root=image_root,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=float(rgb_core.full_patch_radius_px),
            step_px=1.0,
            cache=cache,
            device=state.device,
            cache_device=cache_device,
        )
        rgb_deranged_patches = permute_support_patches_with_phase_identity_point_blocks(
            runtime=batch.runtime,
            support_patches=support_patches,
            shift=int(args.support_permutation_shift),
        )
        with torch.autocast(device_type=state.device.type, enabled=state.device.type == "cuda"):
            rgb_normal = rgb_core(
                runtime=batch.runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                active_sources=("fine",),
            )
            rgb_deranged = rgb_core(
                runtime=batch.runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=rgb_deranged_patches,
                active_sources=("fine",),
            )
        phase_normal = phase_model(runtime=batch.runtime)
        phase_deranged = phase_model(
            runtime=batch.runtime, support_permutation_shift=int(args.support_permutation_shift)
        )
        availability = _phase_control_common_availability(
            phase_normal=phase_normal,
            phase_deranged=phase_deranged,
            rgb_normal=rgb_normal,
            rgb_deranged=rgb_deranged,
        )
        variants = {
            "normal": (phase_normal, rgb_normal),
            "phase_deranged": (phase_deranged, rgb_normal),
            "rgb_deranged": (phase_normal, rgb_deranged),
            "both_deranged": (phase_deranged, rgb_deranged),
        }
        pose_gaps: dict[str, torch.Tensor] = {}
        for name, (phase_prediction, rgb_prediction) in variants.items():
            correct, wrong = _hybrid_pose_scores(
                runtime=batch.runtime,
                phase_prediction=phase_prediction,
                rgb_prediction=rgb_prediction,
                availability=availability,
                correct_offsets_xy=batch.correct_projection_offsets_xy,
                correct_valid=batch.correct_projection_valid,
                wrong_offsets_xy=batch.wrong_projection_offsets_xy,
                wrong_valid=batch.wrong_projection_valid,
                args=args,
            )
            _, _, pose_gaps[name] = pose_margin_terms(
                correct_scores=correct.pose_log_likelihood_ratios,
                wrong_scores=wrong.pose_log_likelihood_ratios,
                margin=float(args.pose_margin),
                temperature=float(args.soft_hard_temperature),
            )
        totals[:9] += torch.tensor(
            [
                float(pose_gaps["normal"].item()),
                float((pose_gaps["normal"] > 0.0).float().item()),
                float(pose_gaps["phase_deranged"].item()),
                float(pose_gaps["rgb_deranged"].item()),
                float(pose_gaps["both_deranged"].item()),
                1.0,
                float((pose_gaps["normal"] - pose_gaps["both_deranged"]).item()),
                float((pose_gaps["normal"] - pose_gaps["rgb_deranged"]).item()),
                float((pose_gaps["normal"] - pose_gaps["phase_deranged"]).item()),
            ],
            dtype=torch.float64,
            device=state.device,
        )
        hard_batch = _hard_repeat_batch_from_group(
            hard_targets=hard,
            group=group,
            point_positions=points,
            device=state.device,
            max_edges=int(args.max_static_hard_edges_per_query),
            seed=int(args.seed),
        )
        gaps: dict[str, torch.Tensor | None] = {}
        usable: dict[str, torch.Tensor | None] = {}
        for name, (phase_prediction, rgb_prediction) in variants.items():
            gaps[name], usable[name] = _hybrid_hard_gaps(
                runtime=batch.runtime,
                phase_prediction=phase_prediction,
                rgb_prediction=rgb_prediction,
                availability=availability,
                hard_batch=hard_batch,
                args=args,
            )
        common = usable["normal"]
        for name in ("phase_deranged", "rgb_deranged", "both_deranged"):
            if common is not None and usable[name] is not None:
                common = common & usable[name]  # type: ignore[operator]
        if common is not None and bool(common.any()):
            assert all(gaps[name] is not None for name in variants)
            normal_hard = gaps["normal"][common]  # type: ignore[index]
            phase_hard = gaps["phase_deranged"][common]  # type: ignore[index]
            rgb_hard = gaps["rgb_deranged"][common]  # type: ignore[index]
            both_hard = gaps["both_deranged"][common]  # type: ignore[index]
            totals[9:] += torch.tensor(
                [
                    float(normal_hard.sum().item()),
                    float((normal_hard > 0.0).float().sum().item()),
                    float(phase_hard.sum().item()),
                    float(rgb_hard.sum().item()),
                    float(both_hard.sum().item()),
                    float(common.sum().item()),
                    1.0,
                    float((normal_hard - both_hard).sum().item()),
                    float((normal_hard - rgb_hard).sum().item()),
                    float((normal_hard - phase_hard).sum().item()),
                ],
                dtype=torch.float64,
                device=state.device,
            )
    totals = _reduce(state, totals)
    query_count = float(totals[5].item())
    hard_count = float(totals[14].item())
    if query_count <= 0.0 or hard_count <= 0.0:
        raise RuntimeError("hybrid inner gate has no common held-out evidence")
    return {
        "query_count": query_count,
        "normal_pose_gap": float(totals[0].item() / query_count),
        "normal_pose_win_fraction": float(totals[1].item() / query_count),
        "phase_deranged_pose_gap": float(totals[2].item() / query_count),
        "rgb_deranged_pose_gap": float(totals[3].item() / query_count),
        "both_deranged_pose_gap": float(totals[4].item() / query_count),
        "normal_minus_both_deranged_pose_gap": float(totals[6].item() / query_count),
        "normal_minus_rgb_deranged_pose_gap": float(totals[7].item() / query_count),
        "normal_minus_phase_deranged_pose_gap": float(totals[8].item() / query_count),
        "hard_active_edge_count": hard_count,
        "hard_eligible_query_fraction": float(totals[15].item() / query_count),
        "hard_gap": float(totals[9].item() / hard_count),
        "hard_win_fraction": float(totals[10].item() / hard_count),
        "hard_phase_deranged_gap": float(totals[11].item() / hard_count),
        "hard_rgb_deranged_gap": float(totals[12].item() / hard_count),
        "hard_both_deranged_gap": float(totals[13].item() / hard_count),
        "hard_normal_minus_both_deranged_gap": float(totals[16].item() / hard_count),
        "hard_normal_minus_rgb_deranged_gap": float(totals[17].item() / hard_count),
        "hard_normal_minus_phase_deranged_gap": float(totals[18].item() / hard_count),
    }


def train_candidate_phase_identity_spatial_likelihood(args: argparse.Namespace) -> dict[str, object]:
    """Fit a fresh RGB spatial residual against the admitted phase expert."""

    _validate_args(args)
    state = _initialize_distributed(str(args.device))
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.manual_seed_all(int(args.seed) + state.rank)
        random.seed(int(args.seed) + state.rank)
        np.random.seed(int(args.seed) + state.rank)
        torch.manual_seed(int(args.seed) + state.rank)
        start = time.time()
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_phase_identity_spatial_likelihood.pt"
        summary_path = output_dir / "summary.json"
        history_path = output_dir / "history.json"
        if state.rank == 0 and any(path.exists() for path in (checkpoint_path, summary_path, history_path)) and not bool(args.force):
            raise FileExistsError("refusing to overwrite phase/spatial hybrid output")
        if state.enabled:
            conflict = torch.tensor(
                [int(state.rank == 0 and checkpoint_path.exists() and not bool(args.force))],
                dtype=torch.int64,
                device=state.device,
            )
            distributed.broadcast(conflict, src=0)
            if bool(int(conflict.item())):
                raise FileExistsError("phase/spatial hybrid output already exists")

        paths = {
            "layout": Path(args.rgb_spatial_layout),
            "geometry": Path(args.geometry_training_targets),
            "identity": Path(args.registered_identity_targets),
            "current_hard": Path(args.current_hard_repeat_targets),
            "static_hard": Path(args.static_hard_repeat_targets),
            "phase_checkpoint": Path(args.phase_checkpoint),
            "phase_reaudit": Path(args.phase_reaudit),
        }
        source_paths = {
            "radio_final": Path(args.radio_final_context_cache),
            "radio_intermediate": Path(args.radio_intermediate_context_cache),
            "alike": Path(args.alike_spatial_context_cache),
        }
        if not all(path.is_file() for path in (*paths.values(), *source_paths.values())):
            raise FileNotFoundError("phase/spatial hybrid input is absent")
        re_audit = _read_json(paths["phase_reaudit"])
        _validate_phase_reaudit(
            re_audit=re_audit,
            re_audit_path=paths["phase_reaudit"],
            phase_checkpoint=paths["phase_checkpoint"],
            paths=paths,
        )

        layout = load_candidate_pose_rgb_spatial_layout(paths["layout"])
        geometry_targets = load_candidate_pose_rgb_spatial_training_targets(paths["geometry"])
        identity_targets = load_candidate_pose_rgb_spatial_training_targets(paths["identity"])
        layout_sha = file_sha256_short(paths["layout"])
        geometry_sha = file_sha256_short(paths["geometry"])
        identity_sha = file_sha256_short(paths["identity"])
        validate_training_layout_and_targets(
            layout=layout, targets=geometry_targets, layout_sha256=layout_sha
        )
        validate_training_layout_and_targets(
            layout=layout, targets=identity_targets, layout_sha256=layout_sha
        )
        groups = build_train_query_groups(layout=layout, targets=geometry_targets)
        exact_by_query = build_exact_identity_query_targets(
            groups=groups, identity_targets=identity_targets
        )
        del exact_by_query  # Its strict semantic check is the only hybrid use.

        phase_checkpoint = _load_phase_checkpoint(paths["phase_checkpoint"])
        phase_partition, heldout_query_ids = _checkpoint_partition(
            phase_checkpoint, all_train_query_ids=tuple(sorted(groups))
        )
        current_hard_targets = load_candidate_pose_rgb_spatial_hard_repeat_targets(paths["current_hard"])
        current_hard_groups = build_hard_repeat_query_targets(
            layout=layout,
            targets=geometry_targets,
            hard_repeat_targets=current_hard_targets,
            layout_sha256=layout_sha,
            targets_sha256=geometry_sha,
        )
        current_hard_provenance = validate_external_frozen_current_hard_targets(
            mined_targets=current_hard_targets,
            mined_groups=current_hard_groups,
            registered_identity_targets=identity_targets,
            registered_identity_targets_sha256=identity_sha,
            expected_partition=phase_partition,
            mining_checkpoint_path=Path(args.current_hard_mining_checkpoint),
        )
        static_hard_targets = load_candidate_pose_rgb_spatial_hard_repeat_targets(paths["static_hard"])
        static_geometry_groups = build_hard_repeat_query_targets(
            layout=layout,
            targets=geometry_targets,
            hard_repeat_targets=static_hard_targets,
            layout_sha256=layout_sha,
            targets_sha256=geometry_sha,
        )
        static_hard_groups, static_hard_filter = filter_static_hard_repeat_groups_to_registered_exact_identity(
            static_groups=static_geometry_groups, registered_identity_targets=identity_targets
        )
        if not set(heldout_query_ids).issubset(static_hard_groups):
            raise ValueError("hybrid static exact hard targets miss a held-out phase fold query")
        train_query_ids = tuple(
            str(value)
            for value in phase_partition["inner_train"]["query_ids"]  # type: ignore[index]
        )
        if not train_query_ids or set(train_query_ids).intersection(heldout_query_ids):
            raise ValueError("hybrid phase fold partition is invalid")

        sources = load_context_attention_sources(
            radio_final_context_cache=source_paths["radio_final"],
            radio_intermediate_context_cache=source_paths["radio_intermediate"],
            alike_spatial_context_cache=source_paths["alike"],
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        image_ids, image_sizes, source_grids = _source_table(sources)
        source_manifest = str(sources[0].metadata.get("source_image_manifest_sha256", ""))
        if not source_manifest:
            raise ValueError("hybrid phase source manifest is absent")
        _validate_phase_checkpoint(
            checkpoint=phase_checkpoint,
            checkpoint_path=paths["phase_checkpoint"],
            layout_path=paths["layout"],
            layout_metadata=layout.metadata,
            source_paths=source_paths,
            source_manifest_sha256=source_manifest,
        )
        phase_model = _load_phase_model(
            checkpoint=phase_checkpoint,
            source_grids=source_grids,
            image_sizes=image_sizes,
            device=state.device,
        )
        for parameter in phase_model.parameters():
            parameter.requires_grad_(False)
        phase_model.eval()

        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("hybrid RGB training requires one processed coordinate size")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=sources[0].metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        rgb_model = CandidateHighresRGBMultiscaleLikelihood(
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            fine_search_radius_px=float(args.fine_search_radius_px),
            fine_context_radius_px=float(args.fine_context_radius_px),
            broad_search_radius_px=float(args.fine_search_radius_px),
            broad_context_radius_px=float(args.fine_context_radius_px),
            texture_feature_dim=int(args.texture_feature_dim),
            hidden_dim=int(args.hidden_dim),
            edge_chunk_size=int(args.edge_chunk_size),
            rgb_temperature=float(args.rgb_temperature),
            max_abs_edge_log_ratio=float(args.max_abs_edge_log_ratio),
        ).to(state.device)
        trainable_names = configure_trainable_parameters(model=rgb_model, source="fine")
        rgb_for_train: nn.Module = rgb_model
        if state.enabled:
            rgb_for_train = DistributedDataParallel(
                rgb_model,
                device_ids=[state.local_rank] if state.device.type == "cuda" else None,
                output_device=state.local_rank if state.device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in rgb_model.parameters() if parameter.requires_grad],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        amp_enabled = state.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled, init_scale=float(args.amp_init_scale))
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * (1024**3)),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        cache_device = torch.device("cpu") if str(args.rgb_cache_device) == "cpu" else state.device
        owner_costs = target_free_query_owner_costs(layout=layout, groups=groups)
        history: list[dict[str, object]] = []

        for epoch in range(int(args.epochs)):
            rgb_for_train.train()
            schedule = balanced_ddp_query_schedules(
                query_ids=train_query_ids,
                owner_costs=owner_costs,
                world_size=state.world_size,
                seed=int(args.seed) + epoch,
            )
            local_queries = schedule[state.rank]
            balance = ddp_owner_cost_balance_metrics(schedules=schedule, owner_costs=owner_costs)
            totals = torch.zeros((23,), dtype=torch.float64, device=state.device)
            epoch_start = time.time()
            for step, query_id in enumerate(local_queries):
                group = groups[str(query_id)]
                current_hard = current_hard_groups.get(str(query_id))
                static_train_hard = static_hard_groups.get(str(query_id))
                positions = _select_group_points(
                    group=group,
                    max_points=int(args.max_points_per_query),
                    seed=int(args.seed) + epoch * 100003 + step,
                    required_source_point_ids=_required_hard_source_ids(
                        current_hard, static_train_hard
                    ),
                )
                batch = _query_batch_from_group(
                    group=group, complete_runtime=runtime, point_positions=positions, device=state.device
                )
                query_patches, support_patches = _crop_runtime_rgb_patches(
                    runtime=batch.runtime,
                    image_ids=image_ids,
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    radius_px=float(rgb_model.full_patch_radius_px),
                    step_px=1.0,
                    cache=cache,
                    device=state.device,
                    cache_device=cache_device,
                )
                rgb_deranged_patches = permute_support_patches_with_phase_identity_point_blocks(
                    runtime=batch.runtime,
                    support_patches=support_patches,
                    shift=int(args.support_permutation_shift),
                )
                current_hard_batch = (
                    None
                    if current_hard is None
                    else _hard_repeat_batch_from_group(
                        hard_targets=current_hard,
                        group=group,
                        point_positions=positions,
                        device=state.device,
                        max_edges=int(args.max_current_hard_edges_per_query),
                        seed=int(args.seed) + epoch * 100003 + step,
                    )
                )
                static_hard_batch = (
                    None
                    if static_train_hard is None
                    else _hard_repeat_batch_from_group(
                        hard_targets=static_train_hard,
                        group=group,
                        point_positions=positions,
                        device=state.device,
                        max_edges=int(args.max_static_train_hard_edges_per_query),
                        seed=int(args.seed) + epoch * 100003 + step + 7919,
                    )
                )
                with torch.no_grad():
                    phase_normal = phase_model(runtime=batch.runtime)
                    phase_deranged = phase_model(
                        runtime=batch.runtime,
                        support_permutation_shift=int(args.support_permutation_shift),
                    )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=state.device.type, enabled=amp_enabled):
                    rgb_normal = rgb_for_train(
                        runtime=batch.runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=support_patches,
                        active_sources=("fine",),
                    )
                    rgb_deranged = rgb_for_train(
                        runtime=batch.runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=rgb_deranged_patches,
                        active_sources=("fine",),
                    )
                    availability = _phase_control_common_availability(
                        phase_normal=phase_normal,
                        phase_deranged=phase_deranged,
                        rgb_normal=rgb_normal,
                        rgb_deranged=rgb_deranged,
                    )
                    density_loss, _ = highres_rgb_spatial_density_nll(
                        scale_prediction=rgb_normal.sources["fine"],
                        target_offsets_xy=batch.spatial_target_offsets_xy,
                        target_dustbin=batch.spatial_target_dustbin,
                        target_supervised=batch.spatial_target_supervised,
                        dustbin_weight=float(args.dustbin_loss_weight),
                        balance_observed_and_dustbin=True,
                    )
                    normal_correct, normal_wrong = _hybrid_pose_scores(
                        runtime=batch.runtime,
                        phase_prediction=phase_normal,
                        rgb_prediction=rgb_normal,
                        availability=availability,
                        correct_offsets_xy=batch.correct_projection_offsets_xy,
                        correct_valid=batch.correct_projection_valid,
                        wrong_offsets_xy=batch.wrong_projection_offsets_xy,
                        wrong_valid=batch.wrong_projection_valid,
                        args=args,
                    )
                    both_deranged_correct, both_deranged_wrong = _hybrid_pose_scores(
                        runtime=batch.runtime,
                        phase_prediction=phase_deranged,
                        rgb_prediction=rgb_deranged,
                        availability=availability,
                        correct_offsets_xy=batch.correct_projection_offsets_xy,
                        correct_valid=batch.correct_projection_valid,
                        wrong_offsets_xy=batch.wrong_projection_offsets_xy,
                        wrong_valid=batch.wrong_projection_valid,
                        args=args,
                    )
                    rgb_deranged_correct, rgb_deranged_wrong = _hybrid_pose_scores(
                        runtime=batch.runtime,
                        phase_prediction=phase_normal,
                        rgb_prediction=rgb_deranged,
                        availability=availability,
                        correct_offsets_xy=batch.correct_projection_offsets_xy,
                        correct_valid=batch.correct_projection_valid,
                        wrong_offsets_xy=batch.wrong_projection_offsets_xy,
                        wrong_valid=batch.wrong_projection_valid,
                        args=args,
                    )
                    pose_loss, soft_loss, normal_gap = pose_margin_terms(
                        correct_scores=normal_correct.pose_log_likelihood_ratios,
                        wrong_scores=normal_wrong.pose_log_likelihood_ratios,
                        margin=float(args.pose_margin),
                        temperature=float(args.soft_hard_temperature),
                    )
                    _, _, both_deranged_gap = pose_margin_terms(
                        correct_scores=both_deranged_correct.pose_log_likelihood_ratios,
                        wrong_scores=both_deranged_wrong.pose_log_likelihood_ratios,
                        margin=float(args.pose_margin),
                        temperature=float(args.soft_hard_temperature),
                    )
                    _, _, rgb_deranged_gap = pose_margin_terms(
                        correct_scores=rgb_deranged_correct.pose_log_likelihood_ratios,
                        wrong_scores=rgb_deranged_wrong.pose_log_likelihood_ratios,
                        margin=float(args.pose_margin),
                        temperature=float(args.soft_hard_temperature),
                    )
                    pose_control, _ = appearance_control_margin_loss(
                        normal_gaps=normal_gap,
                        permuted_gaps=both_deranged_gap,
                        margin=float(args.appearance_control_margin),
                    )
                    rgb_pose_control, _ = appearance_control_margin_loss(
                        normal_gaps=normal_gap,
                        permuted_gaps=rgb_deranged_gap,
                        margin=float(args.appearance_control_margin),
                    )
                    normal_hard, normal_hard_usable = _hybrid_hard_gaps(
                        runtime=batch.runtime,
                        phase_prediction=phase_normal,
                        rgb_prediction=rgb_normal,
                        availability=availability,
                        hard_batch=current_hard_batch,
                        args=args,
                    )
                    both_deranged_hard, both_deranged_hard_usable = _hybrid_hard_gaps(
                        runtime=batch.runtime,
                        phase_prediction=phase_deranged,
                        rgb_prediction=rgb_deranged,
                        availability=availability,
                        hard_batch=current_hard_batch,
                        args=args,
                    )
                    rgb_deranged_hard, rgb_deranged_hard_usable = _hybrid_hard_gaps(
                        runtime=batch.runtime,
                        phase_prediction=phase_normal,
                        rgb_prediction=rgb_deranged,
                        availability=availability,
                        hard_batch=current_hard_batch,
                        args=args,
                    )
                    (
                        hard_loss,
                        hard_control,
                        hard_rgb_control,
                        hard_gap,
                        hard_active,
                    ) = _strict_hard_repeat_objective_terms(
                        normal_values=normal_hard,
                        normal_usable=normal_hard_usable,
                        both_deranged_values=both_deranged_hard,
                        both_deranged_usable=both_deranged_hard_usable,
                        rgb_deranged_values=rgb_deranged_hard,
                        rgb_deranged_usable=rgb_deranged_hard_usable,
                        hard_margin=float(args.hard_repeat_margin),
                        appearance_control_margin=float(args.appearance_control_margin),
                        anchor=density_loss,
                    )
                    static_normal_hard, static_normal_hard_usable = _hybrid_hard_gaps(
                        runtime=batch.runtime,
                        phase_prediction=phase_normal,
                        rgb_prediction=rgb_normal,
                        availability=availability,
                        hard_batch=static_hard_batch,
                        args=args,
                    )
                    static_both_hard, static_both_hard_usable = _hybrid_hard_gaps(
                        runtime=batch.runtime,
                        phase_prediction=phase_deranged,
                        rgb_prediction=rgb_deranged,
                        availability=availability,
                        hard_batch=static_hard_batch,
                        args=args,
                    )
                    static_rgb_hard, static_rgb_hard_usable = _hybrid_hard_gaps(
                        runtime=batch.runtime,
                        phase_prediction=phase_normal,
                        rgb_prediction=rgb_deranged,
                        availability=availability,
                        hard_batch=static_hard_batch,
                        args=args,
                    )
                    (
                        static_hard_loss,
                        _static_hard_both_control,
                        static_hard_rgb_control,
                        static_hard_gap,
                        static_hard_active,
                    ) = _strict_hard_repeat_objective_terms(
                        normal_values=static_normal_hard,
                        normal_usable=static_normal_hard_usable,
                        both_deranged_values=static_both_hard,
                        both_deranged_usable=static_both_hard_usable,
                        rgb_deranged_values=static_rgb_hard,
                        rgb_deranged_usable=static_rgb_hard_usable,
                        hard_margin=float(args.hard_repeat_margin),
                        appearance_control_margin=float(args.appearance_control_margin),
                        anchor=density_loss,
                    )
                    registered_rgb_control = density_loss * 0.0
                    registered_rgb_metrics = {
                        "normal_minus_permuted": 0.0,
                        "usable_candidate_count": 0.0,
                    }
                    if float(args.registered_observation_rgb_control_loss_weight) > 0.0:
                        registered_rgb_control, registered_rgb_metrics = (
                            registered_observation_appearance_control_terms(
                                runtime=batch.runtime,
                                normal_prediction=rgb_normal,
                                permuted_prediction=rgb_deranged,
                                target_offsets_xy=batch.spatial_target_offsets_xy,
                                target_observed=batch.spatial_target_observed,
                                target_supervised=batch.spatial_target_supervised,
                                source="fine",
                                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                                margin=float(args.registered_observation_rgb_control_margin),
                            )
                        )
                    total_loss = (
                        float(args.density_loss_weight) * density_loss
                        + float(args.pose_loss_weight) * pose_loss
                        + float(args.soft_hard_loss_weight) * soft_loss
                        + float(args.hard_repeat_loss_weight) * hard_loss
                        + float(args.static_hard_repeat_loss_weight) * static_hard_loss
                        + float(args.pose_appearance_control_loss_weight) * pose_control
                        + float(args.hard_repeat_appearance_control_loss_weight) * hard_control
                        + float(args.rgb_only_pose_control_loss_weight) * rgb_pose_control
                        + float(args.rgb_only_hard_repeat_control_loss_weight) * hard_rgb_control
                        + float(args.static_hard_repeat_rgb_control_loss_weight)
                        * static_hard_rgb_control
                        + float(args.registered_observation_rgb_control_loss_weight)
                        * registered_rgb_control
                    )
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                local_finite = all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in rgb_model.parameters()
                    if parameter.requires_grad
                )
                finite = torch.tensor([int(local_finite)], dtype=torch.int64, device=state.device)
                if state.enabled:
                    distributed.all_reduce(finite, op=distributed.ReduceOp.MIN)
                if bool(int(finite.item())):
                    if float(args.gradient_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in rgb_model.parameters() if parameter.requires_grad],
                            float(args.gradient_clip_norm),
                        )
                    scaler.step(optimizer)
                else:
                    optimizer.zero_grad(set_to_none=True)
                scaler.update()
                totals += torch.tensor(
                    [
                        float(total_loss.detach().item()),
                        float(density_loss.detach().item()),
                        float(pose_loss.detach().item()),
                        float(soft_loss.detach().item()),
                        float(normal_gap.detach().item()),
                        float((normal_gap - both_deranged_gap).detach().item()),
                        float((normal_gap - rgb_deranged_gap).detach().item()),
                        float(pose_control.detach().item()),
                        float(rgb_pose_control.detach().item()),
                        float(hard_loss.detach().item()),
                        float(hard_gap.detach().item()),
                        float(hard_active),
                        float(hard_control.detach().item()),
                        float(hard_rgb_control.detach().item()),
                        float(static_hard_loss.detach().item()),
                        float(static_hard_gap.detach().item()),
                        float(static_hard_active),
                        float(static_hard_rgb_control.detach().item()),
                        float(registered_rgb_control.detach().item()),
                        float(registered_rgb_metrics["normal_minus_permuted"]),
                        float(registered_rgb_metrics["usable_candidate_count"]),
                        1.0 - float(int(finite.item())),
                        1.0,
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            totals = _reduce(state, totals)
            if state.rank == 0:
                steps = float(len(local_queries) * state.world_size)
                record = {
                    "epoch": int(epoch + 1),
                    "global_query_steps": int(steps),
                    "train_total_loss": float(totals[0].item() / steps),
                    "train_density_loss": float(totals[1].item() / steps),
                    "train_pose_loss": float(totals[2].item() / steps),
                    "train_soft_hard_loss": float(totals[3].item() / steps),
                    "train_normal_pose_gap": float(totals[4].item() / steps),
                    "train_normal_minus_both_deranged_pose_gap": float(totals[5].item() / steps),
                    "train_normal_minus_rgb_deranged_pose_gap": float(totals[6].item() / steps),
                    "train_pose_control_loss": float(totals[7].item() / steps),
                    "train_rgb_pose_control_loss": float(totals[8].item() / steps),
                    "train_hard_repeat_loss": float(totals[9].item() / steps),
                    "train_hard_repeat_gap": float(totals[10].item() / steps),
                    "train_hard_repeat_active_edges": float(totals[11].item() / steps),
                    "train_hard_repeat_control_loss": float(totals[12].item() / steps),
                    "train_hard_repeat_rgb_control_loss": float(totals[13].item() / steps),
                    "train_static_hard_repeat_loss": float(totals[14].item() / steps),
                    "train_static_hard_repeat_gap": float(totals[15].item() / steps),
                    "train_static_hard_repeat_active_edges": float(totals[16].item() / steps),
                    "train_static_hard_repeat_rgb_control_loss": float(totals[17].item() / steps),
                    "train_registered_rgb_control_loss": float(totals[18].item() / steps),
                    "train_registered_rgb_control_delta": float(totals[19].item() / steps),
                    "train_registered_rgb_control_active_candidates": float(totals[20].item() / steps),
                    "train_nonfinite_update_fraction": float(totals[21].item() / steps),
                    "ddp_owner_cost_mean_abs_difference": float(balance["mean_abs_owner_cost_difference"]),
                    "ddp_owner_cost_max_abs_difference": float(balance["max_abs_owner_cost_difference"]),
                    "epoch_seconds": float(time.time() - epoch_start),
                }
                history.append(record)
                print(json.dumps({"phase_spatial_train": record}, sort_keys=True), flush=True)
            if state.enabled:
                distributed.barrier()

        final_metrics = evaluate_hybrid_inner_gate(
            rgb_model=rgb_for_train,
            phase_model=phase_model,
            groups=groups,
            static_hard_groups=static_hard_groups,
            complete_runtime=runtime,
            query_ids=heldout_query_ids,
            image_ids=image_ids,
            image_root=Path(args.image_root),
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            cache=cache,
            cache_device=cache_device,
            state=state,
            args=args,
        )
        final_gate = hybrid_inner_gate_decision(metrics=final_metrics, args=args)
        if state.enabled:
            distributed.barrier()
        if state.rank != 0:
            return {}
        metadata: dict[str, object] = {
            "format": CHECKPOINT_FORMAT,
            "model_format": CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
            "hybrid_format": HYBRID_SOURCE_NAME,
            "contains_target_fields": False,
            "checkpoint_contains_train_targets": False,
            "runtime_layout_is_target_free": True,
            "diagnostic_only": True,
            "promotion_allowed": False,
            "pnp_integration_allowed": False,
            "raw_scores_must_not_feed_pnp": True,
            "heldout_evaluation_allowed": bool(final_gate["passed"]),
            "train_only_inner_gate_passed": bool(final_gate["passed"]),
            "fixed_global_topl": True,
            "fixed_candidate_top_k": int(layout.candidate_count),
            "fixed_support_view_count": int(layout.support_view_count),
            "explicit_null": True,
            "candidate_reselection_per_pose": False,
            "support_reselection_per_pose": False,
            "projection_after_network_only": True,
            "out_of_window_projection_semantics": "fixed_neutral_missing_edge_not_learned_dustbin_v1",
            "render": False,
            "image_retrieval_or_submap_used": False,
            "phase_source_name": "radio_final",
            "spatial_source_name": "fine",
            "phase_expert_frozen": True,
            "strict_common_phase_rgb_availability": True,
            "appearance_control": "shared_distant_point_block_support_appearance_derangement_v1",
            "encoder_excludes": [
                "pose_matrix",
                "projection_offset",
                "reprojection_residual",
                "ground_truth_label",
                "track_id",
                "candidate_rank",
                "coarse_score",
            ],
            "config": {
                "fine_search_radius_px": float(args.fine_search_radius_px),
                "fine_context_radius_px": float(args.fine_context_radius_px),
                "texture_feature_dim": int(args.texture_feature_dim),
                "hidden_dim": int(args.hidden_dim),
                "edge_chunk_size": int(args.edge_chunk_size),
                "rgb_temperature": float(args.rgb_temperature),
                "max_abs_edge_log_ratio": float(args.max_abs_edge_log_ratio),
                "max_abs_pose_log_ratio": float(args.max_abs_pose_log_ratio),
                "identity_weight": float(args.identity_weight),
                "spatial_weight": float(args.spatial_weight),
                "max_current_hard_edges_per_query": int(args.max_current_hard_edges_per_query),
                "max_static_train_hard_edges_per_query": int(
                    args.max_static_train_hard_edges_per_query
                ),
                "max_static_gate_hard_edges_per_query": int(args.max_static_hard_edges_per_query),
                "loss_weights": {
                    "density": float(args.density_loss_weight),
                    "pose": float(args.pose_loss_weight),
                    "soft_hard": float(args.soft_hard_loss_weight),
                    "current_hard_repeat": float(args.hard_repeat_loss_weight),
                    "static_exact_hard_repeat": float(args.static_hard_repeat_loss_weight),
                    "both_visual_pose_control": float(args.pose_appearance_control_loss_weight),
                    "both_visual_current_hard_control": float(
                        args.hard_repeat_appearance_control_loss_weight
                    ),
                    "rgb_only_pose_control": float(args.rgb_only_pose_control_loss_weight),
                    "rgb_only_current_hard_control": float(
                        args.rgb_only_hard_repeat_control_loss_weight
                    ),
                    "rgb_only_static_hard_control": float(
                        args.static_hard_repeat_rgb_control_loss_weight
                    ),
                    "registered_observation_rgb_control": float(
                        args.registered_observation_rgb_control_loss_weight
                    ),
                },
                "trainable_parameter_names": list(trainable_names),
            },
            "phase_expert": {
                "checkpoint": str(paths["phase_checkpoint"]),
                "checkpoint_sha256": file_sha256_short(paths["phase_checkpoint"]),
                "reaudit": str(paths["phase_reaudit"]),
                "reaudit_sha256": file_sha256_short(paths["phase_reaudit"]),
                "promotable_source_name": "radio_final",
            },
            "lineage": {
                "layout_sha256": layout_sha,
                "geometry_training_targets_sha256": geometry_sha,
                "registered_identity_targets_sha256": identity_sha,
                "current_hard_repeat_targets_sha256": file_sha256_short(paths["current_hard"]),
                "static_hard_repeat_targets_sha256": file_sha256_short(paths["static_hard"]),
                "static_hard_registered_exact_filter": static_hard_filter,
                "source_image_manifest_sha256": source_manifest,
                "descriptor_space_id": str(layout.metadata.get("descriptor_space_id", "")),
                "projection_space_id": str(layout.metadata.get("projection_space_id", "")),
                "rgb_coordinate_bridge": rgb_bridge,
            },
            "training": {
                "objective": "frozen_radio_final_exact_identity_plus_fresh_real_rgb_candidate_spatial_density_rgb_only_counterfactual_registered_observation_and_exact_static_hard_pose_margin_v2",
                "epochs": int(args.epochs),
                "world_size": int(state.world_size),
                "phase_train_query_partition": phase_partition,
                "current_hard_provenance": current_hard_provenance,
                "static_hard_training": {
                    "selection_scope": "phase_inner_train_only_excluding_heldout_gate_queries_v1",
                    "semantics": "registered_exact_identity_filter_of_static_coherent_wrong_edges_v1",
                    "gate_queries_never_enter_optimizer": True,
                },
                "final_inner_validation_metrics": final_metrics,
                "final_inner_gate": final_gate,
                "checkpoint_selection_policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
                "inner_validation_used_for_model_selection": False,
            },
        }
        _save_checkpoint(
            checkpoint_path,
            {"format": CHECKPOINT_FORMAT, "state_dict": _model_state(rgb_for_train), "metadata": metadata},
        )
        history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
        summary = {
            "stage": "train_candidate_phase_identity_spatial_likelihood",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": file_sha256_short(checkpoint_path),
            "final_inner_validation": final_metrics,
            "final_inner_gate": final_gate,
            "history": history,
            "elapsed_seconds": float(time.time() - start),
            "next_step": (
                "frozen_global_topl_pose_rank_audit"
                if bool(final_gate["passed"])
                else "diagnostic_only_redesign_rgb_spatial_evidence_before_heldout_pose_rank_audit"
            ),
            "protocol": {
                "target_free_runtime": True,
                "phase_expert_frozen_after_registered_exact_gate": True,
                "train_only_targets_joined_after_visual_forward": True,
                "fixed_global_topl_and_explicit_null": True,
                "no_render": True,
                "no_image_retrieval_or_submap": True,
                "no_pnp": True,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"phase_spatial_final_gate": final_gate, "output": str(output_dir)}, sort_keys=True))
        return summary
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    train_candidate_phase_identity_spatial_likelihood(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover - command entry point
    main()
