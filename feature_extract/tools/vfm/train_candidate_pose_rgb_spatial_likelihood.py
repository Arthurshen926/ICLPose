"""Train target-free high-resolution RGB spatial likelihood on train-only poses.

The runtime model receives only fixed P1 query/support image coordinates,
frozen RADIO/ALIKE contexts, and real RGB crops.  Correct and coherent-wrong
pose projections are read exclusively from the train-only target artifact
after the model has emitted its local density.  The resulting checkpoint is a
diagnostic candidate-pose likelihood and must pass an independent held-out
gate before it can influence downstream pose ranking.

Typical two-GPU invocation::

    torchrun --standalone --nproc_per_node=2 \
      feature_extract/tools/vfm/train_candidate_pose_rgb_spatial_likelihood.py \
      --rgb-spatial-layout .../s1351.../layout.npz \
      --training-targets .../s1355.../targets.npz \
      --registered-identity-targets .../registered_identity.../targets.npz \
      --radio-final-context-cache ... --radio-intermediate-context-cache ... \
      --alike-spatial-context-cache ... --image-root .../processed \
      --output-dir ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from PIL import Image
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


# ``torchrun path/to/script.py`` does not consistently place the repository
# root on ``sys.path``.  Keep the documented two-GPU command self-contained.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_llr import (
    query_grouped_pose_margin_loss,
    query_grouped_pose_permutation_soft_hard_margin_loss,
    query_grouped_pose_soft_hard_margin_loss,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_CHECKPOINT_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_NULL_MASS,
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_OBJECTIVE,
    CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_IDENTITY_PRETRAIN_OBJECTIVE,
    CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
    CONTEXT_IDENTITY_HEAD_FINAL_WEIGHT_STD,
    CandidatePoseRGBSpatialEdgePrediction,
    CandidatePoseRGBSpatialLikelihood,
    CandidatePoseRGBSpatialRuntime,
    context_identity_cross_entropy_loss,
    context_identity_support_permutation_margin_loss,
    permute_runtime_support_image_appearance_only,
    resolve_candidate_pose_rgb_spatial_context_encoder_arch,
    resolve_candidate_pose_rgb_spatial_context_windows,
    runtime_from_target_free_layout,
    selected_candidate_view_context_log_likelihood_ratio,
    score_candidate_pose_rgb_spatial_batch,
    selected_candidate_view_log_likelihood_ratio_at_offsets,
    spatial_density_nll,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    CandidatePoseRGBSpatialHardRepeatTargets,
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_system_hard import (
    CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT,
    CURRENT_SYSTEM_HARD_TARGET_FREE_RANKED_MODES_FORMAT,
    SUPPORTED_CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MINING_FORMATS,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
    CandidatePoseRGBSpatialTrainingTargets,
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_selector import (
    TARGET_FREE_SELECTOR_POLICIES,
    select_target_free_spatial_quota,
    selector_input_from_target_free_layout,
    target_free_selector_scores,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    ContextAttentionSource,
    load_context_attention_source_headers,
    load_context_attention_sources,
)


_CONTEXT_OBSERVATION_FIXED_FINAL_EPOCH_SELECTION_POLICY = (
    "fixed_final_epoch_without_inner_validation_model_selection_v1"
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    _crop_cached_rgb_windows_grouped,
    _load_query_rgb,
    _load_query_rgb_uint8,
    resolve_rgb_image_cache_storage_dtype,
)


CHECKPOINT_FORMAT = "candidate_pose_rgb_spatial_likelihood_checkpoint_v2"
FIXED_FINAL_EPOCH_SELECTION_POLICY = (
    "fixed_final_epoch_without_inner_validation_model_selection_v1"
)
INNER_GATE_EVALUATOR_MANIFEST_FORMAT = (
    "candidate_pose_rgb_spatial_inner_gate_evaluator_manifest_v1"
)
_INNER_GATE_EVALUATOR_SEMANTICS = {
    "format": INNER_GATE_EVALUATOR_MANIFEST_FORMAT,
    "version": "target_free_static_selector_normal_deranged_pose_and_static_repeat_v2",
    "query_selection": "frozen_layout_coarse_margin_spatial_quota_before_target_join_v1",
    "visual_control": "geometry_fixed_support_image_derangement_recrop_v2",
    "pose_diagnostic": "full_coherent_wrong_pool_post_forward_query_margin_v1",
    "repeat_diagnostic": "common_normal_deranged_availability_post_forward_v1",
    "checkpoint_selection": FIXED_FINAL_EPOCH_SELECTION_POLICY,
}
IDENTITY_LLR_TEXTURE_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT = (
    "candidate_pose_rgb_spatial_identity_llr_observation_pretrain_v2"
)
IDENTITY_LLR_TEXTURE_OBSERVATION_PRETRAIN_GATE_VERSION = (
    "combined_visual_controls_plus_conditional_source_visual_ablation_and_hard_pose_v5"
)
SOURCE_SAFE_EDGE_AVAILABILITY = {
    "format": "independent_rgb_context_union_v1",
    "rgb_cost_volume": "in_bounds_real_rgb_query_and_support_windows_v1",
    "context_identity": "all_full_map_radio_alike_crops_valid_v1",
    "learned_spatial_residual_and_dustbin": "rgb_and_context_intersection_only_v1",
    "combined_fallback": "rgb_raw_cost_volume_or_context_scalar_with_neutral_missing_peer_v1",
}
_TARGET_FREE_STATIC_SELECTOR_POLICIES = tuple(
    policy for policy in TARGET_FREE_SELECTOR_POLICIES if "rgb" not in policy
)


def current_inner_gate_evaluator_manifest() -> dict[str, object]:
    """Return the immutable semantic contract for a P1 inner-gate result.

    A historical scalar gate is not comparable once the selector, visual
    control, or post-forward pose aggregation changes.  Checkpoints therefore
    carry this manifest and continuation/mining/scoring reject legacy values
    instead of silently treating an old pass as a current one.
    """

    semantics = dict(_INNER_GATE_EVALUATOR_SEMANTICS)
    encoded = json.dumps(semantics, sort_keys=True, separators=(",", ":"))
    return {
        **semantics,
        "semantic_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16],
    }


def require_current_inner_gate_evaluator_manifest(
    manifest: object, *, subject: str
) -> dict[str, object]:
    """Reject a checkpoint/artifact whose gate was measured by another evaluator."""

    expected = current_inner_gate_evaluator_manifest()
    if not isinstance(manifest, Mapping) or dict(manifest) != expected:
        raise ValueError(f"{subject} inner-gate evaluator manifest is stale or missing")
    return expected


@dataclass(frozen=True)
class _DistributedState:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    enabled: bool


@dataclass(frozen=True)
class TrainQueryGroup:
    """One train query with all of its coherent-wrong modes aligned to P1 rows."""

    query_id: str
    source_point_ids: np.ndarray
    layout_rows: np.ndarray
    target_rows: np.ndarray
    spatial_target_offsets_xy: np.ndarray
    spatial_target_observed: np.ndarray
    spatial_target_supervised: np.ndarray
    spatial_target_dustbin: np.ndarray
    correct_projection_offsets_xy: np.ndarray
    correct_projection_valid: np.ndarray
    wrong_pair_ids: np.ndarray
    wrong_projection_offsets_xy: np.ndarray
    wrong_projection_valid: np.ndarray

    def __post_init__(self) -> None:
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        layout_rows = np.asarray(self.layout_rows, dtype=np.int64).reshape(-1)
        target_rows = np.asarray(self.target_rows, dtype=np.int64).reshape(-1)
        spatial_offsets = np.asarray(self.spatial_target_offsets_xy, dtype=np.float32)
        spatial_observed = np.asarray(self.spatial_target_observed, dtype=bool)
        spatial_supervised = np.asarray(self.spatial_target_supervised, dtype=bool)
        spatial_dustbin = np.asarray(self.spatial_target_dustbin, dtype=bool)
        correct_offsets = np.asarray(self.correct_projection_offsets_xy, dtype=np.float32)
        correct_valid = np.asarray(self.correct_projection_valid, dtype=bool)
        wrong_pair_ids = np.asarray(self.wrong_pair_ids, dtype=np.int64).reshape(-1)
        wrong_offsets = np.asarray(self.wrong_projection_offsets_xy, dtype=np.float32)
        wrong_valid = np.asarray(self.wrong_projection_valid, dtype=bool)
        point_count = len(source_ids)
        if (
            not str(self.query_id)
            or point_count == 0
            or len(np.unique(source_ids)) != point_count
            or layout_rows.shape != (point_count,)
            or target_rows.shape != (point_count,)
            or np.any(layout_rows < 0)
            or np.any(target_rows < 0)
            or spatial_offsets.ndim != 3
            or spatial_offsets.shape[0] != point_count
            or spatial_offsets.shape[2] != 2
            or spatial_observed.shape != spatial_offsets.shape[:2]
            or spatial_supervised.shape != spatial_offsets.shape[:2]
            or spatial_dustbin.shape != spatial_offsets.shape[:2]
            or np.any(spatial_observed & ~spatial_supervised)
            or np.any(spatial_observed & spatial_dustbin)
            or np.any(spatial_dustbin & ~spatial_supervised)
            or correct_offsets.shape != spatial_offsets.shape
            or correct_valid.shape != spatial_offsets.shape[:2]
            or wrong_offsets.ndim != 4
            or wrong_offsets.shape[0] == 0
            or wrong_pair_ids.shape != (wrong_offsets.shape[0],)
            or len(np.unique(wrong_pair_ids)) != len(wrong_pair_ids)
            or wrong_offsets.shape[1:] != spatial_offsets.shape
            or wrong_valid.shape != wrong_offsets.shape[:3]
            or not np.isfinite(spatial_offsets).all()
            or not np.isfinite(correct_offsets).all()
            or not np.isfinite(wrong_offsets).all()
        ):
            raise ValueError("train query group arrays are invalid")
        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "layout_rows", layout_rows)
        object.__setattr__(self, "target_rows", target_rows)
        object.__setattr__(self, "spatial_target_offsets_xy", spatial_offsets)
        object.__setattr__(self, "spatial_target_observed", spatial_observed)
        object.__setattr__(self, "spatial_target_supervised", spatial_supervised)
        object.__setattr__(self, "spatial_target_dustbin", spatial_dustbin)
        object.__setattr__(self, "correct_projection_offsets_xy", correct_offsets)
        object.__setattr__(self, "correct_projection_valid", correct_valid)
        object.__setattr__(self, "wrong_pair_ids", wrong_pair_ids)
        object.__setattr__(self, "wrong_projection_offsets_xy", wrong_offsets)
        object.__setattr__(self, "wrong_projection_valid", wrong_valid)

    @property
    def point_count(self) -> int:
        return int(len(self.source_point_ids))

    @property
    def wrong_mode_count(self) -> int:
        return int(self.wrong_projection_offsets_xy.shape[0])


@dataclass(frozen=True)
class HardRepeatQueryTargets:
    """Train-only coherent-repeat candidate edges for one frozen query group."""

    query_id: str
    source_point_ids: np.ndarray
    pair_ids: np.ndarray
    positive_candidate_indices: np.ndarray
    negative_candidate_indices: np.ndarray
    positive_offsets_xy: np.ndarray
    negative_offsets_xy: np.ndarray

    def __post_init__(self) -> None:
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        pair_ids = np.asarray(self.pair_ids, dtype=np.int64).reshape(-1)
        positive = np.asarray(self.positive_candidate_indices, dtype=np.int64).reshape(-1)
        negative = np.asarray(self.negative_candidate_indices, dtype=np.int64).reshape(-1)
        positive_offsets = np.asarray(self.positive_offsets_xy, dtype=np.float32)
        negative_offsets = np.asarray(self.negative_offsets_xy, dtype=np.float32)
        count = int(len(source_ids))
        edge_keys = (
            np.stack([pair_ids, source_ids, positive, negative], axis=1)
            if count
            else np.zeros((0, 4), dtype=np.int64)
        )
        if (
            not str(self.query_id)
            or count == 0
            or pair_ids.shape != (count,)
            or positive.shape != (count,)
            or negative.shape != (count,)
            or positive_offsets.shape != (count, 2)
            or negative_offsets.shape != (count, 2)
            or np.any(positive < 0)
            or np.any(negative < 0)
            or np.any(positive == negative)
            or len(np.unique(edge_keys, axis=0)) != count
            or not np.isfinite(positive_offsets).all()
            or not np.isfinite(negative_offsets).all()
        ):
            raise ValueError("hard-repeat query target arrays are invalid")
        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "pair_ids", pair_ids)
        object.__setattr__(self, "positive_candidate_indices", positive)
        object.__setattr__(self, "negative_candidate_indices", negative)
        object.__setattr__(self, "positive_offsets_xy", positive_offsets)
        object.__setattr__(self, "negative_offsets_xy", negative_offsets)


@dataclass(frozen=True)
class HardRepeatBatch:
    """Hard-repeat target rows remapped to a selected runtime point batch."""

    point_indices: torch.Tensor
    positive_candidate_indices: torch.Tensor
    negative_candidate_indices: torch.Tensor
    positive_offsets_xy: torch.Tensor
    negative_offsets_xy: torch.Tensor
    # Pair IDs are train-only coherent-wrong pose-mode labels.  They are
    # optional for the legacy per-edge losses, but retained when a caller
    # needs to aggregate mutually compatible edges into one pose group.
    pair_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        points = torch.as_tensor(self.point_indices, dtype=torch.long).reshape(-1)
        positive = torch.as_tensor(self.positive_candidate_indices, dtype=torch.long).reshape(-1)
        negative = torch.as_tensor(self.negative_candidate_indices, dtype=torch.long).reshape(-1)
        positive_offsets = torch.as_tensor(self.positive_offsets_xy, dtype=torch.float32)
        negative_offsets = torch.as_tensor(self.negative_offsets_xy, dtype=torch.float32)
        pair_ids = (
            None
            if self.pair_ids is None
            else torch.as_tensor(self.pair_ids, dtype=torch.long).reshape(-1)
        )
        if (
            len(points) == 0
            or positive.shape != points.shape
            or negative.shape != points.shape
            or positive_offsets.shape != (len(points), 2)
            or negative_offsets.shape != (len(points), 2)
            or (pair_ids is not None and pair_ids.shape != points.shape)
            or torch.any(points < 0)
            or torch.any(positive < 0)
            or torch.any(negative < 0)
            or torch.any(positive == negative)
            or (pair_ids is not None and torch.any(pair_ids < 0))
            or not torch.isfinite(positive_offsets).all()
            or not torch.isfinite(negative_offsets).all()
        ):
            raise ValueError("hard-repeat batch arrays are invalid")
        object.__setattr__(self, "point_indices", points)
        object.__setattr__(self, "positive_candidate_indices", positive)
        object.__setattr__(self, "negative_candidate_indices", negative)
        object.__setattr__(self, "positive_offsets_xy", positive_offsets)
        object.__setattr__(self, "negative_offsets_xy", negative_offsets)
        object.__setattr__(self, "pair_ids", pair_ids)


@dataclass(frozen=True)
class _QueryBatch:
    runtime: CandidatePoseRGBSpatialRuntime
    spatial_target_offsets_xy: torch.Tensor
    spatial_target_observed: torch.Tensor
    spatial_target_supervised: torch.Tensor
    spatial_target_dustbin: torch.Tensor
    correct_projection_offsets_xy: torch.Tensor
    correct_projection_valid: torch.Tensor
    wrong_projection_offsets_xy: torch.Tensor
    wrong_projection_valid: torch.Tensor


@dataclass(frozen=True)
class RegisteredIdentityBatch:
    """Registered exact-track labels joined after a target-free forward pass.

    The full-pose geometry target and the registered-identity target deliberately
    have different semantics and may use different local support radii.  This
    small train-only container prevents the latter from being accidentally
    substituted for density or pose-margin supervision.
    """

    target_offsets_xy: torch.Tensor
    target_observed: torch.Tensor

    def __post_init__(self) -> None:
        offsets = torch.as_tensor(self.target_offsets_xy, dtype=torch.float32)
        observed = torch.as_tensor(self.target_observed, dtype=torch.bool)
        if (
            offsets.ndim != 3
            or offsets.shape[2] != 2
            or observed.shape != offsets.shape[:2]
            or torch.any(observed.sum(dim=1) > 1)
            or not bool(torch.isfinite(offsets).all())
        ):
            raise ValueError("registered identity batch arrays are invalid")
        object.__setattr__(self, "target_offsets_xy", offsets)
        object.__setattr__(self, "target_observed", observed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument(
        "--registered-identity-targets",
        default="",
        help=(
            "Optional registered exact-track train-only target artifact used together "
            "with a geometry/full-pose --training-targets artifact. It supervises "
            "only context identity and fixed-offset support-appearance objectives."
        ),
    )
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional compatible target-free RGB likelihood checkpoint for controlled fine-tuning.",
    )
    parser.add_argument(
        "--allow-registered-identity-sidecar-introduction-from-identity-free-init",
        action="store_true",
        help=(
            "Explicitly permit a separate registered-identity sidecar to be introduced "
            "when the --init-checkpoint is proven identity-free. Existing checkpoints "
            "that used a different identity sidecar remain incompatible."
        ),
    )
    parser.add_argument(
        "--observation-pretrain-checkpoint",
        default="",
        help=(
            "Gate-approved broad train-observation pretrain checkpoint. It is an "
            "initializer only and must exactly match the context-source lineage."
        ),
    )
    parser.add_argument(
        "--hard-pose-pretrain-checkpoint",
        default="",
        help=(
            "Gate-approved correct-vs-coherent-wrong broad pretrain checkpoint. "
            "It is an initializer only and must exactly match the context-source lineage."
        ),
    )
    parser.add_argument(
        "--texture-observation-pretrain-checkpoint",
        default="",
        help=(
            "Gate-approved RGB cost-volume observation pretrain used only to initialize "
            "TexturePatchEncoder in a component-wise combined likelihood run."
        ),
    )
    parser.add_argument(
        "--rgb-hard-repeat-texture-checkpoint",
        default="",
        help=(
            "Current-P1 RGB-only checkpoint that passed the direct hard-repeat visual "
            "gate. It transfers TexturePatchEncoder only; its failed full-pose gate "
            "cannot initialize density, dustbin, context, or fusion heads."
        ),
    )
    parser.add_argument(
        "--identity-llr-texture-pretrain-checkpoint",
        default="",
        help=(
            "Gate-approved V5 candidate-identity checkpoint used only to initialize "
            "TexturePatchEncoder. Its RGB crop geometry may differ, so the transfer is "
            "recorded as a train-only cross-geometry ablation and still requires a "
            "separate spatial-likelihood gate."
        ),
    )
    parser.add_argument(
        "--context-observation-pretrain-checkpoint",
        default="",
        help=(
            "Gate-approved RADIO/ALIKE context observation pretrain used only to initialize "
            "context encoders and the context identity head in a component-wise combined run."
        ),
    )
    parser.add_argument(
        "--context-observation-pairs",
        default="",
        help=(
            "Source observation-pair artifact for a context initializer. The "
            "inner train/validation split must exactly match this P1 fit."
        ),
    )
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
            "Train and evaluate only the real-RGB query/support FPN cost volume. "
            "This bypasses RADIO/ALIKE context plus learned residual/dustbin heads."
        ),
    )
    parser.add_argument("--radio-final-context-window", type=int, default=15)
    parser.add_argument("--radio-intermediate-context-window", type=int, default=15)
    parser.add_argument("--alike-context-window", type=int, default=13)
    parser.add_argument(
        "--context-encoder-arch",
        default="conv_v1",
        help="Explicit context encoder architecture serialized into checkpoint lineage.",
    )
    parser.add_argument("--max-points-per-query", type=int, default=128)
    parser.add_argument(
        "--validation-selector-policy",
        choices=_TARGET_FREE_STATIC_SELECTOR_POLICIES,
        default="coarse_margin",
        help=(
            "Strict target-free static point policy for checkpoint selection; training "
            "sampling may still retain train-only supervised rows."
        ),
    )
    parser.add_argument("--validation-selector-point-budget", type=int, default=64)
    parser.add_argument("--validation-selector-grid-rows", type=int, default=4)
    parser.add_argument("--validation-selector-grid-columns", type=int, default=4)
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pose-margin", type=float, default=0.25)
    parser.add_argument("--pose-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--pose-pool-soft-hard-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Train-only normalized soft-hard margin over every full-pose wrong mode. "
            "It is restricted to the explicit full-pose-hard target format."
        ),
    )
    parser.add_argument(
        "--pose-pool-soft-hard-temperature",
        type=float,
        default=0.35,
        help="Temperature for normalized log-mean-exp full-pose negative aggregation.",
    )
    parser.add_argument(
        "--pose-pool-permutation-soft-hard-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Train-only listwise support-appearance control over every full-pose wrong "
            "mode; restricted to the explicit full-pose-hard target format."
        ),
    )
    parser.add_argument(
        "--pose-pool-permutation-soft-hard-margin",
        type=float,
        default=0.05,
        help="Required real-support minus deranged-support full-pool soft-hard gap.",
    )
    parser.add_argument(
        "--pose-pool-permutation-soft-hard-temperature",
        type=float,
        default=0.35,
        help="Temperature for listwise real-versus-deranged full-pose aggregation.",
    )
    parser.add_argument("--density-loss-weight", type=float, default=0.25)
    parser.add_argument("--dustbin-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--hard-repeat-targets",
        default="",
        help="Train-only coherent-repeat candidate-edge artifact; never loaded by runtime scoring.",
    )
    parser.add_argument("--hard-repeat-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-repeat-margin", type=float, default=0.25)
    parser.add_argument(
        "--mined-hard-repeat-targets",
        default="",
        help=(
            "Current-model mined coherent-repeat edges for the inner-train loss only. "
            "The fixed --hard-repeat-targets artifact remains the sole validation gate."
        ),
    )
    parser.add_argument(
        "--mined-registered-identity-targets",
        default="",
        help=(
            "Registered exact-track target artifact used only to audit mined positive/negative "
            "edge labels before the inner-train loss is enabled."
        ),
    )
    parser.add_argument(
        "--mined-hard-repeat-loss-weight",
        type=float,
        default=0.0,
        help="Additional inner-train-only margin weight for current-model mined edges.",
    )
    parser.add_argument(
        "--mined-hard-repeat-margin",
        type=float,
        default=0.25,
        help="Required positive-minus-current-wrong edge margin for mined edges.",
    )
    parser.add_argument(
        "--mined-hard-repeat-context-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Additional inner-train-only RADIO/ALIKE context margin for current-model "
            "mined edges. It cannot use the RGB spatial-density branch."
        ),
    )
    parser.add_argument(
        "--mined-hard-repeat-context-margin",
        type=float,
        default=0.25,
        help="Required context-only true-minus-current-wrong candidate LLR.",
    )
    parser.add_argument(
        "--mined-hard-pose-pool-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Additional inner-train-only normalized soft-hard pose loss over the "
            "frozen target-free top-H current-system wrong modes."
        ),
    )
    parser.add_argument(
        "--mined-hard-pose-pool-margin",
        type=float,
        default=0.25,
        help="Required correct-pose margin over the frozen current-system top-H mode pool.",
    )
    parser.add_argument(
        "--mined-hard-pose-pool-temperature",
        type=float,
        default=0.35,
        help="Log-mean-exp temperature for the frozen current-system top-H mode pool.",
    )
    parser.add_argument(
        "--mined-hard-pose-pool-warmup-epochs",
        type=int,
        default=0,
        help="Epoch-only warmup for the current-system top-H pose-pool loss.",
    )
    parser.add_argument(
        "--max-mined-hard-repeat-edges-per-query",
        type=int,
        default=64,
        help="Maximum current-model mined coherent-repeat edges per inner-train query.",
    )
    parser.add_argument(
        "--mined-hard-repeat-warmup-epochs",
        type=int,
        default=0,
        help="Epoch-only warmup for the extra mined hard-repeat loss.",
    )
    parser.add_argument(
        "--hard-repeat-context-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Train-only RADIO/ALIKE context-only margin on exact-track versus "
            "coherent-repeat candidate pairs."
        ),
    )
    parser.add_argument(
        "--hard-repeat-context-margin",
        type=float,
        default=0.25,
        help="Required context-only true-minus-coherent-repeat candidate LLR.",
    )
    parser.add_argument("--max-hard-repeat-edges-per-query", type=int, default=64)
    parser.add_argument(
        "--minimum-hard-repeat-eligible-query-fraction",
        type=float,
        default=0.90,
        help=(
            "When a hard-repeat artifact is supplied, require this fraction of the "
            "target-free inner-fold queries to retain a normal/permuted common pair."
        ),
    )
    parser.add_argument(
        "--minimum-hard-repeat-win-fraction",
        type=float,
        default=0.55,
        help="Required correct-versus-coherent-wrong win fraction on common hard-repeat edges.",
    )
    parser.add_argument(
        "--minimum-hard-repeat-gap",
        type=float,
        default=0.05,
        help="Required normal correct-minus-coherent-wrong gap on common hard-repeat edges.",
    )
    parser.add_argument(
        "--minimum-hard-repeat-visual-gap-delta",
        type=float,
        default=0.05,
        help=(
            "Required normal-minus-support-deranged hard-repeat gap. This rejects a "
            "geometry-only or fixed-slot shortcut."
        ),
    )
    parser.add_argument(
        "--hard-repeat-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Linearly ramp the coherent hard-repeat loss from zero over this many "
            "epochs; zero applies its configured weight immediately."
        ),
    )
    parser.add_argument(
        "--permutation-contrastive-loss-weight",
        type=float,
        default=0.0,
        help="Train-only penalty that requires normal support pairing to beat a deranged pairing.",
    )
    parser.add_argument(
        "--permutation-contrastive-margin",
        type=float,
        default=0.05,
        help="Required normal-minus-deranged hardest-pose gap during train-only fitting.",
    )
    parser.add_argument(
        "--identity-support-contrastive-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Train-only exact-identity edge penalty requiring its normal support appearance "
            "to beat a deranged support appearance at the registered local offset."
        ),
    )
    parser.add_argument(
        "--identity-support-contrastive-margin",
        type=float,
        default=0.05,
        help="Required exact-identity normal-minus-deranged edge LLR during train-only fitting.",
    )
    parser.add_argument(
        "--context-identity-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Train-only candidate-level cross entropy on the full RADIO/ALIKE context "
            "mixture for registered exact identities."
        ),
    )
    parser.add_argument(
        "--context-identity-support-permutation-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Train-only context-only support-appearance derangement penalty. It keeps "
            "query anchors, candidate priors, and support-view weights fixed."
        ),
    )
    parser.add_argument(
        "--context-identity-support-permutation-margin",
        type=float,
        default=0.25,
        help="Required normal-minus-deranged registered-candidate context logit gap.",
    )
    parser.add_argument(
        "--permutation-control-shift",
        type=int,
        default=1,
        help="Reserved derangement shift for the train-only inner gate; never used by its training loss.",
    )
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=0)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-normal-gap", type=float, default=0.05)
    parser.add_argument("--minimum-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--missing-edge-log-likelihood-ratio", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--rgb-cache-gb", type=float, default=4.0)
    parser.add_argument(
        "--rgb-cache-dtype",
        choices=("float16", "uint8"),
        default="float16",
        help="Full-image RGB cache storage; uint8 keeps all source images compact on GPU.",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _initialize_distributed(device_name: str) -> _DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP RGB spatial likelihood training requires CUDA")
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RGB spatial likelihood training requested CUDA but it is unavailable")
    return _DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        enabled=enabled,
    )


def _finalize_distributed(state: _DistributedState) -> None:
    if state.enabled and distributed.is_initialized():
        distributed.barrier()
        distributed.destroy_process_group()


def validate_training_layout_and_targets(
    *,
    layout: CandidatePoseRGBSpatialLayout,
    targets: CandidatePoseRGBSpatialTrainingTargets,
    layout_sha256: str,
) -> None:
    """Reject stale, mixed-split, or semantically incompatible supervision."""

    if not isinstance(layout, CandidatePoseRGBSpatialLayout) or not isinstance(
        targets, CandidatePoseRGBSpatialTrainingTargets
    ):
        raise ValueError("RGB spatial training requires validated layout and targets")
    if str(targets.metadata.get("rgb_spatial_layout_sha256", "")) != str(layout_sha256):
        raise ValueError("RGB spatial targets do not match the frozen layout hash")
    if (
        str(targets.metadata.get("projection_space_id", ""))
        != str(layout.metadata.get("projection_space_id", ""))
        or str(targets.metadata.get("descriptor_space_id", ""))
        != str(layout.metadata.get("descriptor_space_id", ""))
        or targets.candidate_count != layout.candidate_count
    ):
        raise ValueError("RGB spatial target descriptor/projection lineage differs from layout")
    layout_row_by_source = {
        int(source_id): row
        for row, source_id in enumerate(np.asarray(layout.source_point_ids, dtype=np.int64).tolist())
    }
    for source_id, query_id in zip(targets.source_point_ids.tolist(), targets.query_ids.tolist()):
        row = layout_row_by_source.get(int(source_id))
        if row is None:
            raise ValueError("RGB spatial target source point is absent from layout")
        if str(layout.split_names[row]) != "train":
            raise ValueError("RGB spatial target source point is not train split")
        if str(layout.query_ids[row]) != str(query_id):
            raise ValueError("RGB spatial target query does not match frozen layout")


def is_registered_exact_identity_targets(
    targets: CandidatePoseRGBSpatialTrainingTargets,
) -> bool:
    """Return whether a train-only artifact has the strict registered-track contract."""

    if not isinstance(targets, CandidatePoseRGBSpatialTrainingTargets):
        return False
    metadata = targets.metadata
    return (
        str(metadata.get("format", ""))
        == CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT
        and str(metadata.get("spatial_supervision_mode", ""))
        == "registered_exact_identity"
        and str(metadata.get("spatial_target_semantics", ""))
        == "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
    )


def validate_registered_identity_targets_for_geometry(
    *,
    layout: CandidatePoseRGBSpatialLayout,
    geometry_targets: CandidatePoseRGBSpatialTrainingTargets,
    registered_identity_targets: CandidatePoseRGBSpatialTrainingTargets,
    layout_sha256: str,
) -> None:
    """Require an exact source universe before combining two train-only targets.

    Geometry targets provide correct/wrong pose projections over the larger
    model search window.  Registered identity targets provide sparse exact-track
    labels in their own (possibly smaller) local window.  Matching only tensor
    shapes would silently mix rows when either builder is regenerated, so this
    validator checks their immutable point/query and map lineage explicitly.
    """

    validate_training_layout_and_targets(
        layout=layout,
        targets=geometry_targets,
        layout_sha256=layout_sha256,
    )
    validate_training_layout_and_targets(
        layout=layout,
        targets=registered_identity_targets,
        layout_sha256=layout_sha256,
    )
    if not is_registered_exact_identity_targets(registered_identity_targets):
        raise ValueError("registered identity target lacks the exact-track supervision contract")
    if geometry_targets.candidate_count != registered_identity_targets.candidate_count:
        raise ValueError("registered identity target candidate count differs from geometry targets")

    geometry_by_source = {
        int(source_id): str(query_id)
        for source_id, query_id in zip(
            geometry_targets.source_point_ids.tolist(), geometry_targets.query_ids.tolist()
        )
    }
    identity_by_source = {
        int(source_id): str(query_id)
        for source_id, query_id in zip(
            registered_identity_targets.source_point_ids.tolist(),
            registered_identity_targets.query_ids.tolist(),
        )
    }
    if geometry_by_source.keys() != identity_by_source.keys():
        raise ValueError("registered identity target source-point universe differs from geometry targets")
    if any(
        geometry_by_source[source_id] != identity_by_source[source_id]
        for source_id in geometry_by_source
    ):
        raise ValueError("registered identity target query/source ownership differs from geometry targets")

    for field in (
        "train_pairs_sha256",
        "support_geometry_index_sha256",
        "projected_landmark_bank_sha256",
        "projection_space_id",
        "descriptor_space_id",
    ):
        if str(geometry_targets.metadata.get(field, "")) != str(
            registered_identity_targets.metadata.get(field, "")
        ):
            raise ValueError(f"registered identity target {field} differs from geometry targets")
    try:
        geometry_radius = float(geometry_targets.metadata["spatial_search_radius_px"])
        identity_radius = float(registered_identity_targets.metadata["registered_identity_radius_px"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("registered identity target radius lineage is invalid") from error
    if (
        not math.isfinite(geometry_radius)
        or not math.isfinite(identity_radius)
        or identity_radius <= 0.0
        or identity_radius > geometry_radius + 1e-5
    ):
        raise ValueError("registered identity target radius is incompatible with geometry support")
    if not bool(registered_identity_targets.spatial_target_observed.any()):
        raise ValueError("registered identity target has no exact observed tracks")


def full_pose_hard_target_initializer_parent_hashes(
    *, targets: CandidatePoseRGBSpatialTrainingTargets, targets_sha256: str
) -> tuple[str, ...]:
    """Permit one declared train-only parent target for full-pose continuation.

    This is intentionally narrower than a generic stale-checkpoint override.
    The full-pose target must preserve the fixed runtime layout and declare the
    immediately preceding target hash; every other target change stays on the
    exact-target-hash path.
    """

    if not isinstance(targets, CandidatePoseRGBSpatialTrainingTargets):
        raise ValueError("initializer target continuation requires validated targets")
    current = str(targets_sha256).strip()
    metadata = targets.metadata
    if (
        str(metadata.get("full_pose_hard_target_format", ""))
        != "candidate_pose_rgb_spatial_full_pool_hard_modes_v1"
    ):
        return ()
    parent = str(metadata.get("base_training_targets_sha256", "")).strip()
    if (
        not current
        or not parent
        or parent == current
        or metadata.get("full_pose_hard_mode_source")
        != "frozen_global_hypotheses_then_train_only_target_join_v1"
        or metadata.get("serialized_pose_or_residual") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("contains_validation_or_test_targets") is not False
    ):
        raise ValueError("full-pose hard target continuation lineage is invalid")
    return (parent,)


def _stable_query_hash(query_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(str(query_id).encode("utf-8")).digest()[:8], byteorder="big"
    )


def train_query_partition_manifest(
    *,
    all_query_ids: Sequence[str],
    inner_train_query_ids: Sequence[str],
    inner_validation_query_ids: Sequence[str],
    fold_count: int,
    fold_index: int,
) -> dict[str, object]:
    """Serialize the exact train-only query partition for later OOF checks.

    The checkpoint does not contain target arrays, but a later calibration must
    still be able to prove that a scored train query was excluded from the P1
    fit that produced the checkpoint.  Counts alone cannot establish that
    property, so retain the deterministic query manifests beside the fold
    configuration.
    """

    def _normalized(values: Sequence[str], *, name: str) -> tuple[str, ...]:
        output = tuple(sorted(set(str(value) for value in values)))
        if not output or any(not value for value in output):
            raise ValueError(f"RGB spatial {name} query manifest is invalid")
        return output

    all_ids = _normalized(all_query_ids, name="all")
    train_ids = _normalized(inner_train_query_ids, name="inner-train")
    validation_ids = _normalized(
        inner_validation_query_ids, name="inner-validation"
    )
    if (
        set(train_ids).intersection(validation_ids)
        or set(train_ids).union(validation_ids) != set(all_ids)
        or int(fold_count) < 2
        or not 0 <= int(fold_index) < int(fold_count)
    ):
        raise ValueError("RGB spatial train query partition is inconsistent")

    def _digest(values: Sequence[str]) -> str:
        payload = "\n".join(values) + "\n"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _entry(values: tuple[str, ...]) -> dict[str, object]:
        return {
            "query_count": int(len(values)),
            "query_ids": list(values),
            "query_ids_sha256": _digest(values),
        }

    return {
        "format": "candidate_pose_rgb_spatial_train_query_partition_v1",
        "assignment": "sorted_unique_query_index_modulo_fold_count_v1",
        "fold_count": int(fold_count),
        "fold_index": int(fold_index),
        "all_train": _entry(all_ids),
        "inner_train": _entry(train_ids),
        "inner_validation": _entry(validation_ids),
    }


def _partition_train_queries_for_inner_validation(
    *, query_ids: Sequence[str], fold_count: int, fold_index: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split only train query IDs, with deterministic nonempty query folds."""

    ids = tuple(sorted(set(str(query_id) for query_id in query_ids)))
    count = int(fold_count)
    index = int(fold_index)
    if len(ids) < 2 or count < 2 or count > len(ids) or not 0 <= index < count:
        raise ValueError("RGB spatial likelihood inner validation fold is invalid")
    validation = tuple(query_id for position, query_id in enumerate(ids) if position % count == index)
    train = tuple(query_id for query_id in ids if query_id not in set(validation))
    if not train or not validation:
        raise ValueError("RGB spatial likelihood inner validation fold is empty")
    return train, validation


def build_train_query_groups(
    *, layout: CandidatePoseRGBSpatialLayout, targets: CandidatePoseRGBSpatialTrainingTargets
) -> dict[str, TrainQueryGroup]:
    """Align all train-only wrong modes to the frozen layout's point order."""

    layout_row_by_source = {
        int(source_id): row
        for row, source_id in enumerate(np.asarray(layout.source_point_ids, dtype=np.int64).tolist())
    }
    target_row_by_source = {
        int(source_id): row
        for row, source_id in enumerate(np.asarray(targets.source_point_ids, dtype=np.int64).tolist())
    }
    target_rows_by_query: dict[str, list[int]] = {}
    for target_row, query_id in enumerate(np.asarray(targets.query_ids).astype(str).tolist()):
        target_rows_by_query.setdefault(str(query_id), []).append(int(target_row))
    pair_indices_by_query: dict[str, list[int]] = {}
    for pair_index, query_id in enumerate(np.asarray(targets.pair_query_ids).astype(str).tolist()):
        pair_indices_by_query.setdefault(str(query_id), []).append(int(pair_index))
    if set(target_rows_by_query) != set(pair_indices_by_query):
        raise ValueError("RGB spatial targets do not provide wrong modes for every train query")

    groups: dict[str, TrainQueryGroup] = {}
    for query_id in sorted(target_rows_by_query):
        target_rows_unsorted = np.asarray(target_rows_by_query[query_id], dtype=np.int64)
        source_set = set(int(targets.source_point_ids[row]) for row in target_rows_unsorted.tolist())
        layout_rows = np.asarray(
            [
                row
                for row, (source_id, source_query, split) in enumerate(
                    zip(
                        layout.source_point_ids.tolist(),
                        layout.query_ids.astype(str).tolist(),
                        layout.split_names.astype(str).tolist(),
                    )
                )
                if int(source_id) in source_set
                and str(source_query) == query_id
                and str(split) == "train"
            ],
            dtype=np.int64,
        )
        if len(layout_rows) != len(source_set):
            raise ValueError("RGB spatial train query layout/source coverage differs")
        source_ids = np.asarray(layout.source_point_ids[layout_rows], dtype=np.int64)
        target_rows = np.asarray(
            [target_row_by_source[int(source_id)] for source_id in source_ids.tolist()], dtype=np.int64
        )
        if set(target_rows.tolist()) != set(target_rows_unsorted.tolist()):
            raise ValueError("RGB spatial target rows cannot align to frozen layout order")
        correct_offsets: np.ndarray | None = None
        correct_valid: np.ndarray | None = None
        wrong_offsets: list[np.ndarray] = []
        wrong_valid: list[np.ndarray] = []
        wrong_pair_ids: list[int] = []
        for pair_index in pair_indices_by_query[query_id]:
            start = int(targets.pair_point_offsets[pair_index])
            stop = int(targets.pair_point_offsets[pair_index + 1])
            pair_source_ids = np.asarray(targets.pair_source_point_ids[start:stop], dtype=np.int64)
            if len(pair_source_ids) != len(source_ids) or set(pair_source_ids.tolist()) != set(source_ids.tolist()):
                raise ValueError("RGB spatial wrong mode does not cover the frozen query points")
            position_by_source = {
                int(source_id): position for position, source_id in enumerate(pair_source_ids.tolist())
            }
            reorder = np.asarray(
                [position_by_source[int(source_id)] for source_id in source_ids.tolist()], dtype=np.int64
            )
            pair_correct_offsets = np.asarray(
                targets.correct_projection_offsets_xy[start:stop][reorder], dtype=np.float32
            )
            pair_correct_valid = np.asarray(
                targets.correct_projection_valid[start:stop][reorder], dtype=bool
            )
            if correct_offsets is None:
                correct_offsets = pair_correct_offsets
                correct_valid = pair_correct_valid
            elif not (
                np.allclose(correct_offsets, pair_correct_offsets, atol=1e-5, rtol=0.0)
                and np.array_equal(correct_valid, pair_correct_valid)
            ):
                raise ValueError("RGB spatial wrong modes disagree on correct projection")
            wrong_offsets.append(
                np.asarray(
                    targets.coherent_wrong_projection_offsets_xy[start:stop][reorder],
                    dtype=np.float32,
                )
            )
            wrong_valid.append(
                np.asarray(
                    targets.coherent_wrong_projection_valid[start:stop][reorder], dtype=bool
                )
            )
            wrong_pair_ids.append(int(targets.pair_ids[pair_index]))
        if correct_offsets is None or correct_valid is None or not wrong_offsets:
            raise RuntimeError("RGB spatial query group lost its correct/wrong targets")
        groups[query_id] = TrainQueryGroup(
            query_id=query_id,
            source_point_ids=source_ids,
            layout_rows=layout_rows,
            target_rows=target_rows,
            spatial_target_offsets_xy=np.asarray(
                targets.spatial_target_offsets_xy[target_rows], dtype=np.float32
            ),
            spatial_target_observed=np.asarray(
                targets.spatial_target_observed[target_rows], dtype=bool
            ),
            spatial_target_supervised=np.asarray(
                targets.spatial_target_supervised[target_rows], dtype=bool
            ),
            spatial_target_dustbin=np.asarray(
                targets.spatial_target_dustbin[target_rows], dtype=bool
            ),
            correct_projection_offsets_xy=correct_offsets,
            correct_projection_valid=correct_valid,
            wrong_pair_ids=np.asarray(wrong_pair_ids, dtype=np.int64),
            wrong_projection_offsets_xy=np.stack(wrong_offsets, axis=0),
            wrong_projection_valid=np.stack(wrong_valid, axis=0),
        )
    if not groups:
        raise ValueError("RGB spatial targets have no train query groups")
    return groups


def build_hard_repeat_query_targets(
    *,
    layout: CandidatePoseRGBSpatialLayout,
    targets: CandidatePoseRGBSpatialTrainingTargets,
    hard_repeat_targets: CandidatePoseRGBSpatialHardRepeatTargets,
    layout_sha256: str,
    targets_sha256: str,
) -> dict[str, HardRepeatQueryTargets]:
    """Validate and group train-only coherent-repeat candidate edges."""

    if not isinstance(hard_repeat_targets, CandidatePoseRGBSpatialHardRepeatTargets):
        raise ValueError("hard-repeat supervision must be a validated train-only artifact")
    metadata = hard_repeat_targets.metadata
    if (
        str(metadata.get("rgb_spatial_layout_sha256", "")) != str(layout_sha256)
        or str(metadata.get("rgb_spatial_targets_sha256", "")) != str(targets_sha256)
        or str(metadata.get("projection_space_id", ""))
        != str(layout.metadata.get("projection_space_id", ""))
        or str(metadata.get("descriptor_space_id", ""))
        != str(layout.metadata.get("descriptor_space_id", ""))
        or int(metadata.get("candidate_count", 0)) != int(layout.candidate_count)
    ):
        raise ValueError("hard-repeat target lineage differs from the frozen RGB layout")
    def lookup_unique_int64_rows(
        *, values: np.ndarray, queries: np.ndarray, name: str
    ) -> np.ndarray:
        """Resolve a unique integer key table without per-edge Python dict work."""

        table = np.asarray(values, dtype=np.int64).reshape(-1)
        requested = np.asarray(queries, dtype=np.int64).reshape(-1)
        order = np.argsort(table, kind="stable")
        sorted_table = table[order]
        if len(sorted_table) == 0 or np.any(sorted_table[1:] == sorted_table[:-1]):
            raise ValueError(f"hard-repeat {name} key table is not unique")
        positions = np.searchsorted(sorted_table, requested)
        clipped = np.clip(positions, 0, len(sorted_table) - 1)
        if np.any(positions >= len(sorted_table)) or np.any(
            sorted_table[clipped] != requested
        ):
            raise ValueError(f"hard-repeat row references an unknown {name}")
        return order[positions]

    def lookup_pair_source_rows(
        *, pair_ids: np.ndarray, source_ids: np.ndarray
    ) -> np.ndarray:
        """Resolve ``(pair_id, source_id)`` rows in the train-only projection table."""

        pair_offsets = np.asarray(targets.pair_point_offsets, dtype=np.int64)
        pair_lengths = np.diff(pair_offsets)
        source_table = np.asarray(targets.pair_source_point_ids, dtype=np.int64)
        if (
            pair_offsets.ndim != 1
            or pair_offsets.shape != (len(targets.pair_ids) + 1,)
            or np.any(pair_lengths < 0)
            or int(pair_offsets[-1]) != len(source_table)
        ):
            raise ValueError("hard-repeat pair/source projection table is invalid")
        pair_table = np.repeat(
            np.asarray(targets.pair_ids, dtype=np.int64), pair_lengths.astype(np.int64)
        )
        dtype = np.dtype([("pair", np.int64), ("source", np.int64)])
        table_keys = np.empty((len(source_table),), dtype=dtype)
        table_keys["pair"] = pair_table
        table_keys["source"] = source_table
        order = np.argsort(table_keys, kind="stable")
        sorted_keys = table_keys[order]
        if len(sorted_keys) == 0 or np.any(sorted_keys[1:] == sorted_keys[:-1]):
            raise ValueError("hard-repeat pair/source projection table is not unique")
        requested = np.empty((len(pair_ids),), dtype=dtype)
        requested["pair"] = np.asarray(pair_ids, dtype=np.int64)
        requested["source"] = np.asarray(source_ids, dtype=np.int64)
        positions = np.searchsorted(sorted_keys, requested)
        clipped = np.clip(positions, 0, len(sorted_keys) - 1)
        if np.any(positions >= len(sorted_keys)) or np.any(
            sorted_keys[clipped] != requested
        ):
            raise ValueError("hard-repeat row cannot join a train-only pair/source projection")
        return order[positions]

    hard_source_ids = np.asarray(hard_repeat_targets.source_point_ids, dtype=np.int64)
    hard_query_ids = np.asarray(hard_repeat_targets.query_ids).astype(str)
    hard_pair_ids = np.asarray(hard_repeat_targets.pair_ids, dtype=np.int64)
    hard_positive = np.asarray(hard_repeat_targets.positive_candidate_indices, dtype=np.int64)
    hard_negative = np.asarray(hard_repeat_targets.negative_candidate_indices, dtype=np.int64)
    edge_count = int(len(hard_source_ids))
    if (
        edge_count == 0
        or hard_query_ids.shape != (edge_count,)
        or hard_pair_ids.shape != (edge_count,)
        or hard_positive.shape != (edge_count,)
        or hard_negative.shape != (edge_count,)
    ):
        raise ValueError("hard-repeat target arrays are inconsistent")
    layout_rows = lookup_unique_int64_rows(
        values=np.asarray(layout.source_point_ids, dtype=np.int64),
        queries=hard_source_ids,
        name="layout source",
    )
    target_rows = lookup_unique_int64_rows(
        values=np.asarray(targets.source_point_ids, dtype=np.int64),
        queries=hard_source_ids,
        name="training-target source",
    )
    pair_indices = lookup_unique_int64_rows(
        values=np.asarray(targets.pair_ids, dtype=np.int64),
        queries=hard_pair_ids,
        name="pair",
    )
    pair_positions = lookup_pair_source_rows(
        pair_ids=hard_pair_ids,
        source_ids=hard_source_ids,
    )
    exact_identity_mode = (
        str(targets.metadata.get("spatial_supervision_mode", ""))
        == "registered_exact_identity"
    )
    candidate_count = int(layout.candidate_count)
    valid_candidate_indices = (
        (hard_positive >= 0)
        & (hard_positive < candidate_count)
        & (hard_negative >= 0)
        & (hard_negative < candidate_count)
    )
    if not bool(np.all(valid_candidate_indices)):
        raise ValueError("hard-repeat row cannot join the frozen train-only layout")
    layout_query_ids = np.asarray(layout.query_ids[layout_rows]).astype(str)
    layout_splits = np.asarray(layout.split_names[layout_rows]).astype(str)
    pair_query_ids = np.asarray(targets.pair_query_ids[pair_indices]).astype(str)
    layout_candidate_tracks = np.asarray(layout.candidate_track_ids, dtype=np.int64)
    layout_support_valid = np.asarray(layout.support_view_valid, dtype=bool)
    valid_layout_rows = (
        (layout_splits == "train")
        & (layout_query_ids == hard_query_ids)
        & (pair_query_ids == hard_query_ids)
        & (layout_candidate_tracks[layout_rows, hard_positive] >= 0)
        & (layout_candidate_tracks[layout_rows, hard_negative] >= 0)
        & np.any(layout_support_valid[layout_rows, hard_positive], axis=1)
        & np.any(layout_support_valid[layout_rows, hard_negative], axis=1)
    )
    if not bool(np.all(valid_layout_rows)):
        raise ValueError("hard-repeat row cannot join the frozen train-only layout")
    expected_positive = np.asarray(targets.correct_projection_offsets_xy, dtype=np.float32)[
        pair_positions, hard_positive
    ]
    expected_negative = np.asarray(targets.coherent_wrong_projection_offsets_xy, dtype=np.float32)[
        pair_positions, hard_negative
    ]
    valid_offsets = (
        np.asarray(targets.correct_projection_valid, dtype=bool)[
            pair_positions, hard_positive
        ]
        & np.asarray(targets.coherent_wrong_projection_valid, dtype=bool)[
            pair_positions, hard_negative
        ]
        & np.all(
            np.isclose(
                np.asarray(hard_repeat_targets.positive_offsets_xy, dtype=np.float32),
                expected_positive,
                atol=1e-5,
                rtol=0.0,
            ),
            axis=1,
        )
        & np.all(
            np.isclose(
                np.asarray(hard_repeat_targets.negative_offsets_xy, dtype=np.float32),
                expected_negative,
                atol=1e-5,
                rtol=0.0,
            ),
            axis=1,
        )
    )
    if not bool(np.all(valid_offsets)):
        raise ValueError("hard-repeat offsets differ from their train-only pose projections")
    if exact_identity_mode:
        valid_polarity = (
            np.asarray(targets.spatial_target_supervised, dtype=bool)[
                target_rows, hard_positive
            ]
            & np.asarray(targets.spatial_target_observed, dtype=bool)[
                target_rows, hard_positive
            ]
            & np.asarray(targets.spatial_target_dustbin, dtype=bool)[
                target_rows, hard_negative
            ]
        )
        if not bool(np.all(valid_polarity)):
            raise ValueError("exact-identity hard-repeat row does not preserve target polarity")

    query_order = np.argsort(hard_query_ids, kind="stable")
    sorted_query_ids = hard_query_ids[query_order]
    group_starts = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.flatnonzero(sorted_query_ids[1:] != sorted_query_ids[:-1]) + 1]
    )
    group_stops = np.concatenate([group_starts[1:], np.asarray([edge_count], dtype=np.int64)])
    if len(group_starts) == 0:
        raise ValueError("hard-repeat artifact has no train query targets")
    output: dict[str, HardRepeatQueryTargets] = {}
    for start, stop in zip(group_starts.tolist(), group_stops.tolist()):
        selected = query_order[int(start) : int(stop)]
        query_id = str(hard_query_ids[int(selected[0])])
        output[query_id] = HardRepeatQueryTargets(
            query_id=query_id,
            source_point_ids=hard_repeat_targets.source_point_ids[selected],
            pair_ids=hard_repeat_targets.pair_ids[selected],
            positive_candidate_indices=hard_repeat_targets.positive_candidate_indices[selected],
            negative_candidate_indices=hard_repeat_targets.negative_candidate_indices[selected],
            positive_offsets_xy=hard_repeat_targets.positive_offsets_xy[selected],
            negative_offsets_xy=hard_repeat_targets.negative_offsets_xy[selected],
        )
    return output


def validate_current_system_mined_hard_repeat_targets(
    *,
    mined_targets: CandidatePoseRGBSpatialHardRepeatTargets,
    mined_groups: Mapping[str, HardRepeatQueryTargets],
    registered_identity_targets: CandidatePoseRGBSpatialTrainingTargets,
    registered_identity_targets_sha256: str,
    expected_partition: Mapping[str, object],
    initialization_checkpoint_path: Path | None,
) -> dict[str, object]:
    """Verify that current-system hard targets cannot influence the gate fold.

    The mined rows are selected from a frozen, gate-approved model's own
    errors, so they are useful only when the continuation starts from exactly
    that checkpoint and consumes only the checkpoint's inner-train queries.
    This is intentionally stricter than the generic hard-repeat loader:
    static hard targets are allowed in the gate, while mined targets are not.
    """

    if not isinstance(mined_targets, CandidatePoseRGBSpatialHardRepeatTargets):
        raise ValueError("mined hard-repeat supervision must be a validated train-only artifact")
    if not isinstance(registered_identity_targets, CandidatePoseRGBSpatialTrainingTargets):
        raise ValueError("mined hard-repeat identity supervision is invalid")
    if initialization_checkpoint_path is None:
        raise ValueError("mined hard-repeat targets require an exact initialization checkpoint")
    metadata = mined_targets.metadata
    mining_format = str(metadata.get("mining_format", ""))
    if (
        mining_format not in SUPPORTED_CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MINING_FORMATS
        or str(metadata.get("score_component", "")) != "combined"
        or str(metadata.get("registered_identity_targets_sha256", ""))
        != str(registered_identity_targets_sha256)
    ):
        raise ValueError("mined hard-repeat targets were not selected by the full current system")
    if mining_format == CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT:
        wrong_mode_selection = metadata.get("wrong_mode_selection")
        ranked_modes = metadata.get("target_free_ranked_wrong_modes")
        try:
            valid_multi_mode_contract = (
                isinstance(wrong_mode_selection, Mapping)
                and wrong_mode_selection.get("policy")
                == "target_free_combined_pose_llr_descending_pair_id_tiebreak_top_h_v1"
                and int(wrong_mode_selection.get("max_wrong_modes_per_query", 0)) > 1
                and wrong_mode_selection.get("target_join_after_mode_ranking") is True
                and wrong_mode_selection.get("label_based_mode_backfill") is False
                and isinstance(ranked_modes, Mapping)
                and ranked_modes.get("format")
                == CURRENT_SYSTEM_HARD_TARGET_FREE_RANKED_MODES_FORMAT
                and int(ranked_modes.get("query_count", 0)) > 0
                and isinstance(ranked_modes.get("ranked_modes_by_query"), Mapping)
                and ranked_modes.get("selection_before_train_only_target_join") is True
            )
        except (TypeError, ValueError):
            valid_multi_mode_contract = False
        if not valid_multi_mode_contract:
            raise ValueError("multi-mode mined hard-repeat target contract is invalid")
    identity_metadata = registered_identity_targets.metadata
    if (
        str(identity_metadata.get("format", ""))
        != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT
        or str(identity_metadata.get("spatial_supervision_mode", ""))
        != "registered_exact_identity"
        or str(identity_metadata.get("spatial_target_semantics", ""))
        != "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
    ):
        raise ValueError("mined hard-repeat identity targets do not provide registered tracks")
    frozen_checkpoint = metadata.get("frozen_checkpoint")
    mined_partition = metadata.get("train_query_partition")
    mining_config = metadata.get("mining_checkpoint_config")
    if (
        not isinstance(frozen_checkpoint, Mapping)
        or frozen_checkpoint.get("train_only_inner_gate_passed") is not True
        or not isinstance(mined_partition, Mapping)
        or not isinstance(mining_config, Mapping)
    ):
        raise ValueError("mined hard-repeat target lineage is incomplete")
    require_current_inner_gate_evaluator_manifest(
        metadata.get("inner_gate_evaluator_manifest"),
        subject="mined hard-repeat target",
    )
    require_current_inner_gate_evaluator_manifest(
        frozen_checkpoint.get("inner_gate_evaluator_manifest"),
        subject="mined hard-repeat frozen checkpoint",
    )
    try:
        reconstructed_partition = train_query_partition_manifest(
            all_query_ids=mined_partition["all_train"]["query_ids"],  # type: ignore[index]
            inner_train_query_ids=mined_partition["inner_train"]["query_ids"],  # type: ignore[index]
            inner_validation_query_ids=mined_partition["inner_validation"]["query_ids"],  # type: ignore[index]
            fold_count=int(mined_partition["fold_count"]),
            fold_index=int(mined_partition["fold_index"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("mined hard-repeat target query partition is malformed") from error
    if dict(mined_partition) != reconstructed_partition:
        raise ValueError("mined hard-repeat target query partition digest is stale")
    if dict(reconstructed_partition) != dict(expected_partition):
        raise ValueError("mined hard-repeat target partition differs from this training fold")
    try:
        config_payload = json.dumps(
            dict(mining_config), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("mined hard-repeat checkpoint config is invalid") from error
    config_sha256 = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()[:16]
    if str(metadata.get("mining_checkpoint_config_sha256", "")) != config_sha256:
        raise ValueError("mined hard-repeat checkpoint config digest is stale")
    initialization_sha256 = file_sha256_short(initialization_checkpoint_path)
    if (
        str(metadata.get("mining_checkpoint_sha256", "")) != initialization_sha256
        or str(frozen_checkpoint.get("sha256", "")) != initialization_sha256
    ):
        raise ValueError("mined hard-repeat targets require their exact mining checkpoint")
    inner_train = set(
        str(value) for value in reconstructed_partition["inner_train"]["query_ids"]  # type: ignore[index]
    )
    inner_validation = set(
        str(value)
        for value in reconstructed_partition["inner_validation"]["query_ids"]  # type: ignore[index]
    )
    target_query_ids = set(np.asarray(mined_targets.query_ids).astype(str).tolist())
    if (
        not target_query_ids
        or not target_query_ids.issubset(inner_train)
        or bool(target_query_ids.intersection(inner_validation))
        or target_query_ids != set(str(query_id) for query_id in mined_groups)
    ):
        raise ValueError("mined hard-repeat targets contain a non-inner-train query")
    identity_rows_by_source = {
        int(source_id): row
        for row, source_id in enumerate(
            np.asarray(registered_identity_targets.source_point_ids, dtype=np.int64).tolist()
        )
    }
    try:
        identity_rows = np.asarray(
            [
                identity_rows_by_source[int(source_id)]
                for source_id in np.asarray(mined_targets.source_point_ids, dtype=np.int64)
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("mined hard-repeat edge is absent from registered identity targets") from error
    identity_query_ids = np.asarray(registered_identity_targets.query_ids).astype(str)
    identity_mask = np.asarray(registered_identity_targets.spatial_target_observed, dtype=bool)
    positive = np.asarray(mined_targets.positive_candidate_indices, dtype=np.int64)
    negative = np.asarray(mined_targets.negative_candidate_indices, dtype=np.int64)
    mined_query_ids = np.asarray(mined_targets.query_ids).astype(str)
    if (
        identity_mask.shape[1] <= int(max(positive.max(), negative.max()))
        or np.any(identity_mask.sum(axis=1) > 1)
        or np.any(identity_query_ids[identity_rows] != mined_query_ids)
        or not bool(np.all(identity_mask[identity_rows, positive]))
        or bool(np.any(identity_mask[identity_rows, negative]))
    ):
        raise ValueError("mined hard-repeat edges do not preserve registered identity polarity")
    return {
        "selection_scope": "current_model_inner_train_only_v1",
        "mining_checkpoint_sha256": initialization_sha256,
        "mining_checkpoint_config_sha256": config_sha256,
        "partition_validation": "exact_checkpoint_and_training_fold_match_v1",
        "mined_query_count": int(len(target_query_ids)),
        "mined_edge_count": int(mined_targets.count),
        "registered_identity_targets_sha256": str(registered_identity_targets_sha256),
        "registered_identity_polarity_validation": "exact_positive_and_distinct_negative_v1",
    }


def resolve_current_system_mined_top_h_mode_positions(
    *,
    mined_targets: CandidatePoseRGBSpatialHardRepeatTargets,
    mined_groups: Mapping[str, HardRepeatQueryTargets],
    geometry_groups: Mapping[str, TrainQueryGroup],
) -> dict[str, np.ndarray]:
    """Resolve the frozen target-free top-H wrong-mode pool for train loss.

    A current-system hard artifact carries local exact-track edges, but those
    edges are only a materialization of a prior target-free pose ranking. This
    helper recovers the ordered pair IDs from that immutable manifest and joins
    them to the geometry target *after* the visual forward. It rejects the
    tempting but invalid fallback of deriving a mode set from whichever edges
    survived target-side eligibility.
    """

    if not isinstance(mined_targets, CandidatePoseRGBSpatialHardRepeatTargets):
        raise ValueError("current-system top-H mode target is invalid")
    metadata = mined_targets.metadata
    if (
        str(metadata.get("mining_format", ""))
        != CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT
    ):
        raise ValueError("current-system top-H pose-pool loss requires a multi-mode artifact")
    wrong_mode_selection = metadata.get("wrong_mode_selection")
    ranked_manifest = metadata.get("target_free_ranked_wrong_modes")
    if not isinstance(wrong_mode_selection, Mapping) or not isinstance(ranked_manifest, Mapping):
        raise ValueError("current-system top-H mode manifest is missing")
    try:
        max_modes = int(wrong_mode_selection["max_wrong_modes_per_query"])
        query_count = int(ranked_manifest["query_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("current-system top-H mode manifest is malformed") from error
    ranked_by_query = ranked_manifest.get("ranked_modes_by_query")
    if (
        max_modes <= 1
        or wrong_mode_selection.get("policy")
        != "target_free_combined_pose_llr_descending_pair_id_tiebreak_top_h_v1"
        or wrong_mode_selection.get("target_join_after_mode_ranking") is not True
        or wrong_mode_selection.get("label_based_mode_backfill") is not False
        or ranked_manifest.get("format")
        != CURRENT_SYSTEM_HARD_TARGET_FREE_RANKED_MODES_FORMAT
        or ranked_manifest.get("selection_before_train_only_target_join") is not True
        or not isinstance(ranked_by_query, Mapping)
        or query_count != len(ranked_by_query)
        or not set(str(query_id) for query_id in mined_groups).issubset(
            set(str(query_id) for query_id in ranked_by_query)
        )
    ):
        raise ValueError("current-system top-H mode manifest violates its target-free contract")

    output: dict[str, np.ndarray] = {}
    for raw_query_id, raw_modes in ranked_by_query.items():
        query_id = str(raw_query_id)
        group = geometry_groups.get(query_id)
        if group is None or not isinstance(raw_modes, list):
            raise ValueError("current-system top-H mode query coverage is inconsistent")
        expected_count = min(int(max_modes), int(group.wrong_mode_count))
        if len(raw_modes) != expected_count or expected_count <= 0:
            raise ValueError("current-system top-H mode count differs from frozen geometry")
        positions: list[int] = []
        pair_ids: list[int] = []
        for rank, raw_mode in enumerate(raw_modes):
            if not isinstance(raw_mode, Mapping):
                raise ValueError("current-system top-H mode row is invalid")
            try:
                mode_rank = int(raw_mode["mode_rank"])
                mode_index = int(raw_mode["mode_index"])
                pair_id = int(raw_mode["pair_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("current-system top-H mode row is malformed") from error
            if (
                mode_rank != rank
                or mode_index < 0
                or mode_index >= group.wrong_mode_count
                or int(group.wrong_pair_ids[mode_index]) != pair_id
            ):
                raise ValueError("current-system top-H mode row differs from frozen geometry")
            positions.append(mode_index)
            pair_ids.append(pair_id)
        if len(set(positions)) != len(positions) or len(set(pair_ids)) != len(pair_ids):
            raise ValueError("current-system top-H mode manifest repeats a pose mode")
        hard_targets = mined_groups.get(query_id)
        if hard_targets is not None:
            materialized_pairs = set(
                int(pair_id)
                for pair_id in np.asarray(hard_targets.pair_ids, dtype=np.int64).tolist()
            )
            if not materialized_pairs or not materialized_pairs.issubset(set(pair_ids)):
                raise ValueError(
                    "current-system hard edges are not contained in the frozen top-H mode pool"
                )
        output[query_id] = np.asarray(positions, dtype=np.int64)
    if not output:
        raise ValueError("current-system top-H mode pool is empty")
    return output


def _hard_repeat_batch_from_group(
    *,
    hard_targets: HardRepeatQueryTargets,
    group: TrainQueryGroup,
    point_positions: np.ndarray,
    device: torch.device,
    max_edges: int,
    seed: int,
) -> HardRepeatBatch | None:
    """Restrict hard target rows to the current sampled P1 points."""

    if str(hard_targets.query_id) != str(group.query_id):
        raise ValueError("hard-repeat query targets do not match the train query group")
    positions = np.asarray(point_positions, dtype=np.int64).reshape(-1)
    if len(positions) == 0 or np.any(positions < 0) or np.any(positions >= group.point_count):
        raise ValueError("hard-repeat point selection is invalid")
    limit = int(max_edges)
    if limit < 0:
        raise ValueError("hard-repeat edge limit must be non-negative")
    point_position_by_source = {
        int(source_id): position
        for position, source_id in enumerate(group.source_point_ids[positions].tolist())
    }
    mode_by_pair_id = {
        int(pair_id): mode for mode, pair_id in enumerate(group.wrong_pair_ids.tolist())
    }
    selected_rows = np.asarray(
        [
            row
            for row, (source_id, pair_id) in enumerate(
                zip(hard_targets.source_point_ids.tolist(), hard_targets.pair_ids.tolist())
            )
            if int(source_id) in point_position_by_source and int(pair_id) in mode_by_pair_id
        ],
        dtype=np.int64,
    )
    if not len(selected_rows):
        return None
    if limit > 0 and len(selected_rows) > limit:
        # A flat random draw can be dominated by the most prolific coherent
        # wrong pose mode. Keep the source-point batch fixed, but draw edge
        # rows round-robin across the train-only wrong-pose groups so every
        # current hypothesis family contributes before any receives a second
        # edge. Within a group the order remains deterministic-random.
        generator = np.random.default_rng(int(seed) ^ _stable_query_hash(group.query_id))
        pair_values = hard_targets.pair_ids[selected_rows]
        pair_to_rows = {
            int(pair_id): generator.permutation(selected_rows[pair_values == pair_id]).tolist()
            for pair_id in np.unique(pair_values).tolist()
        }
        pair_order = generator.permutation(np.asarray(sorted(pair_to_rows), dtype=np.int64)).tolist()
        chosen: list[int] = []
        while len(chosen) < limit:
            progressed = False
            for pair_id in pair_order:
                rows = pair_to_rows[int(pair_id)]
                if not rows:
                    continue
                chosen.append(int(rows.pop()))
                progressed = True
                if len(chosen) == limit:
                    break
            if not progressed:
                break
        if len(chosen) != limit:
            raise RuntimeError("hard-repeat balanced edge sampling exhausted early")
        selected_rows = np.sort(np.asarray(chosen, dtype=np.int64))
    relative_points = np.asarray(
        [point_position_by_source[int(source_id)] for source_id in hard_targets.source_point_ids[selected_rows]],
        dtype=np.int64,
    )
    return HardRepeatBatch(
        point_indices=torch.from_numpy(relative_points).to(device=device),
        positive_candidate_indices=torch.from_numpy(
            hard_targets.positive_candidate_indices[selected_rows]
        ).to(device=device),
        negative_candidate_indices=torch.from_numpy(
            hard_targets.negative_candidate_indices[selected_rows]
        ).to(device=device),
        positive_offsets_xy=torch.from_numpy(hard_targets.positive_offsets_xy[selected_rows]).to(
            device=device
        ),
        negative_offsets_xy=torch.from_numpy(hard_targets.negative_offsets_xy[selected_rows]).to(
            device=device
        ),
        pair_ids=torch.from_numpy(hard_targets.pair_ids[selected_rows]).to(device=device),
    )


def coherent_hard_repeat_edge_margin_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    hard_batch: HardRepeatBatch | None,
    margin: float,
    missing_edge_log_likelihood_ratio: float,
    max_abs_pose_log_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Directly rank true candidate evidence over coherent-repeat evidence."""

    target_margin = float(margin)
    if not math.isfinite(target_margin) or target_margin < 0.0:
        raise ValueError("hard-repeat margin is invalid")
    if hard_batch is None:
        zero = prediction.spatial_logits.sum() * 0.0
        return zero, {
            "hard_repeat_active_edges": 0.0,
            "hard_repeat_correct_win_fraction": 0.0,
            "hard_repeat_mean_positive_minus_negative": 0.0,
            "hard_repeat_margin_loss": 0.0,
        }
    positive_scores, positive_usable = selected_candidate_view_log_likelihood_ratio_at_offsets(
        runtime=runtime,
        prediction=prediction,
        point_indices=hard_batch.point_indices,
        candidate_indices=hard_batch.positive_candidate_indices,
        offsets_xy=hard_batch.positive_offsets_xy,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    )
    negative_scores, negative_usable = selected_candidate_view_log_likelihood_ratio_at_offsets(
        runtime=runtime,
        prediction=prediction,
        point_indices=hard_batch.point_indices,
        candidate_indices=hard_batch.negative_candidate_indices,
        offsets_xy=hard_batch.negative_offsets_xy,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    )
    active = positive_usable & negative_usable
    if not bool(active.any()):
        zero = prediction.spatial_logits.sum() * 0.0
        return zero, {
            "hard_repeat_active_edges": 0.0,
            "hard_repeat_correct_win_fraction": 0.0,
            "hard_repeat_mean_positive_minus_negative": 0.0,
            "hard_repeat_margin_loss": 0.0,
        }
    gaps = positive_scores[active] - negative_scores[active]
    loss = F.softplus(
        torch.as_tensor(target_margin, dtype=gaps.dtype, device=gaps.device) - gaps
    ).mean()
    return loss, {
        "hard_repeat_active_edges": float(active.sum().item()),
        "hard_repeat_correct_win_fraction": float((gaps.detach() > 0.0).float().mean().item()),
        "hard_repeat_mean_positive_minus_negative": float(gaps.detach().mean().item()),
        "hard_repeat_margin_loss": float(loss.detach().item()),
    }


def coherent_hard_repeat_context_margin_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    hard_batch: HardRepeatBatch | None,
    margin: float,
    missing_edge_log_likelihood_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Directly train the RADIO/ALIKE context branch on coherent repeats.

    The standard hard-repeat objective scores context and RGB density jointly.
    This complementary objective removes local RGB density entirely, so the
    full 2D RADIO final/intermediate and ALIKE context branch must itself rank
    the registered exact track above the candidate that explains the same
    anchor under a mined coherent-wrong pose.
    """

    target_margin = float(margin)
    if not math.isfinite(target_margin) or target_margin < 0.0:
        raise ValueError("hard-repeat context margin is invalid")
    if hard_batch is None:
        zero = prediction.context_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "hard_repeat_context_active_edges": 0.0,
            "hard_repeat_context_correct_win_fraction": 0.0,
            "hard_repeat_context_mean_positive_minus_negative": 0.0,
            "hard_repeat_context_margin_loss": 0.0,
        }
    positive_scores, positive_usable = selected_candidate_view_context_log_likelihood_ratio(
        runtime=runtime,
        prediction=prediction,
        point_indices=hard_batch.point_indices,
        candidate_indices=hard_batch.positive_candidate_indices,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
    )
    negative_scores, negative_usable = selected_candidate_view_context_log_likelihood_ratio(
        runtime=runtime,
        prediction=prediction,
        point_indices=hard_batch.point_indices,
        candidate_indices=hard_batch.negative_candidate_indices,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
    )
    active = positive_usable & negative_usable
    if not bool(active.any()):
        zero = prediction.context_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "hard_repeat_context_active_edges": 0.0,
            "hard_repeat_context_correct_win_fraction": 0.0,
            "hard_repeat_context_mean_positive_minus_negative": 0.0,
            "hard_repeat_context_margin_loss": 0.0,
        }
    gaps = positive_scores[active] - negative_scores[active]
    loss = F.softplus(
        torch.as_tensor(target_margin, dtype=gaps.dtype, device=gaps.device) - gaps
    ).mean()
    return loss, {
        "hard_repeat_context_active_edges": float(active.sum().item()),
        "hard_repeat_context_correct_win_fraction": float(
            (gaps.detach() > 0.0).float().mean().item()
        ),
        "hard_repeat_context_mean_positive_minus_negative": float(gaps.detach().mean().item()),
        "hard_repeat_context_margin_loss": float(loss.detach().item()),
    }


def training_gate_decision(
    metrics: Mapping[str, float],
    *,
    minimum_win_fraction: float,
    minimum_normal_gap: float,
    minimum_visual_gap_delta: float,
) -> dict[str, float | bool]:
    """Apply the train-only normal-versus-permuted visual control gate."""

    try:
        normal_win = float(metrics["normal_correct_win_fraction"])
        normal_gap = float(metrics["normal_mean_correct_minus_hardest_wrong"])
        permuted_gap = float(metrics["permuted_mean_correct_minus_hardest_wrong"])
        min_win = float(minimum_win_fraction)
        min_gap = float(minimum_normal_gap)
        min_delta = float(minimum_visual_gap_delta)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("RGB spatial training gate metrics are incomplete") from error
    if not all(math.isfinite(value) for value in (normal_win, normal_gap, permuted_gap, min_win, min_gap, min_delta)):
        raise ValueError("RGB spatial training gate metrics are not finite")
    if not 0.0 <= min_win <= 1.0 or min_gap < 0.0 or min_delta < 0.0:
        raise ValueError("RGB spatial training gate thresholds are invalid")
    visual_delta = normal_gap - permuted_gap
    passed = normal_win >= min_win and normal_gap >= min_gap and visual_delta >= min_delta
    return {
        "passed": bool(passed),
        "normal_minus_permuted_gap": float(visual_delta),
        "minimum_win_fraction": float(min_win),
        "minimum_normal_gap": float(min_gap),
        "minimum_visual_gap_delta": float(min_delta),
    }


def hard_repeat_training_gate_decision(
    metrics: Mapping[str, float],
    *,
    minimum_win_fraction: float,
    minimum_normal_gap: float,
    minimum_visual_gap_delta: float,
    require_hard_repeat: bool,
    minimum_hard_repeat_eligible_query_fraction: float,
    minimum_hard_repeat_win_fraction: float,
    minimum_hard_repeat_gap: float,
    minimum_hard_repeat_visual_gap_delta: float,
) -> dict[str, float | bool]:
    """Gate a checkpoint on generic pose ranking *and* current hard repeats.

    Generic coherent-wrong modes can be too easy: a local likelihood may pass
    them while still assigning a repeated window to the wrong 3D track.  When
    an explicit current-system hard-repeat artifact is present, its pair labels
    are joined only after the frozen target-free forward and must pass an
    independent common-availability support-derangement control.
    """

    decision = training_gate_decision(
        metrics,
        minimum_win_fraction=float(minimum_win_fraction),
        minimum_normal_gap=float(minimum_normal_gap),
        minimum_visual_gap_delta=float(minimum_visual_gap_delta),
    )
    if not bool(require_hard_repeat):
        decision["hard_repeat_required"] = False
        return decision
    try:
        coverage = float(metrics["hard_repeat_eligible_query_fraction"])
        win = float(metrics["hard_repeat_correct_win_fraction"])
        gap = float(metrics["hard_repeat_mean_correct_minus_coherent_wrong"])
        permuted_gap = float(
            metrics["hard_repeat_permuted_mean_correct_minus_coherent_wrong"]
        )
        minimum_coverage = float(minimum_hard_repeat_eligible_query_fraction)
        minimum_win = float(minimum_hard_repeat_win_fraction)
        minimum_gap = float(minimum_hard_repeat_gap)
        minimum_delta = float(minimum_hard_repeat_visual_gap_delta)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("hard-repeat training gate metrics are incomplete") from error
    values = (
        coverage,
        win,
        gap,
        permuted_gap,
        minimum_coverage,
        minimum_win,
        minimum_gap,
        minimum_delta,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or not 0.0 < minimum_coverage <= 1.0
        or not 0.0 <= minimum_win <= 1.0
        or minimum_gap < 0.0
        or minimum_delta < 0.0
    ):
        raise ValueError("hard-repeat training gate thresholds are invalid")
    visual_delta = gap - permuted_gap
    hard_passed = (
        coverage >= minimum_coverage
        and win >= minimum_win
        and gap >= minimum_gap
        and visual_delta >= minimum_delta
    )
    decision.update(
        {
            "hard_repeat_required": True,
            "hard_repeat_eligible_query_fraction": coverage,
            "minimum_hard_repeat_eligible_query_fraction": minimum_coverage,
            "hard_repeat_correct_win_fraction": win,
            "minimum_hard_repeat_win_fraction": minimum_win,
            "hard_repeat_mean_correct_minus_coherent_wrong": gap,
            "minimum_hard_repeat_gap": minimum_gap,
            "hard_repeat_permuted_mean_correct_minus_coherent_wrong": permuted_gap,
            "hard_repeat_minus_permuted_gap": visual_delta,
            "minimum_hard_repeat_visual_gap_delta": minimum_delta,
            "hard_repeat_passed": bool(hard_passed),
            "passed": bool(decision["passed"] and hard_passed),
        }
    )
    return decision


def support_permutation_contrastive_loss(
    *,
    normal_correct_scores: torch.Tensor,
    normal_wrong_scores: torch.Tensor,
    permuted_correct_scores: torch.Tensor,
    permuted_wrong_scores: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require true support pairing to improve the hardest-pose margin.

    The normal and deranged branches receive the same frozen point/candidate
    priors and the same train-only correct/wrong projections.  Their
    difference therefore removes a generic center/dustbin shortcut: only
    evidence that depends on the candidate-specific support appearance can
    satisfy this loss.
    """

    target_margin = float(margin)
    if not math.isfinite(target_margin) or target_margin < 0.0:
        raise ValueError("support permutation contrastive margin is invalid")

    def gap(correct: torch.Tensor, wrong: torch.Tensor, *, name: str) -> torch.Tensor:
        correct_values = torch.as_tensor(correct)
        wrong_values = torch.as_tensor(wrong, device=correct_values.device)
        if correct_values.ndim != 1:
            raise ValueError(f"{name} correct scores must have shape (query,)")
        if wrong_values.ndim == 1:
            wrong_values = wrong_values.unsqueeze(0)
        if (
            wrong_values.ndim != 2
            or wrong_values.shape[0] != correct_values.shape[0]
            or wrong_values.shape[1] == 0
            or not torch.isfinite(correct_values).all()
            or not torch.isfinite(wrong_values).all()
        ):
            raise ValueError(f"{name} pose scores are invalid")
        return correct_values - torch.amax(wrong_values, dim=1)

    normal_gap = gap(normal_correct_scores, normal_wrong_scores, name="normal")
    permuted_gap = gap(permuted_correct_scores, permuted_wrong_scores, name="permuted")
    visual_delta = normal_gap - permuted_gap
    loss = F.relu(float(target_margin) - visual_delta).mean()
    metrics = {
        "support_permutation_normal_gap": float(normal_gap.detach().mean().item()),
        "support_permutation_permuted_gap": float(permuted_gap.detach().mean().item()),
        "support_permutation_gap_delta": float(visual_delta.detach().mean().item()),
        "support_permutation_margin_loss": float(loss.detach().item()),
    }
    return loss, metrics


def exact_identity_support_appearance_contrastive_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    permuted_runtime: CandidatePoseRGBSpatialRuntime,
    permuted_prediction: CandidatePoseRGBSpatialEdgePrediction,
    target_offsets_xy: torch.Tensor,
    target_observed: torch.Tensor,
    margin: float,
    missing_edge_log_likelihood_ratio: float,
    max_abs_pose_log_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require exact registered identity edges to depend on support appearance.

    This is deliberately narrower than the query-level pose permutation loss:
    only an exact train observation's candidate track is selected, and its
    normal and deranged branches are evaluated at the same registered local
    offset.  Candidate priors, view weights, query pixels, offsets, and all
    pose-derived inputs are otherwise identical.  A positive gap can therefore
    only be explained by the normal candidate-specific support appearance.

    ``target_observed`` is train-only and must come from the strict
    registered-exact-identity target contract.  Runtime scoring never calls
    this function or loads these targets.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime) or not isinstance(
        permuted_runtime, CandidatePoseRGBSpatialRuntime
    ):
        raise ValueError("identity support contrastive loss requires target-free runtimes")
    target_margin = float(margin)
    if not math.isfinite(target_margin) or target_margin < 0.0:
        raise ValueError("identity support contrastive margin is invalid")
    device = prediction.joint_log_probabilities.device
    active = runtime.to(device)
    permuted = permuted_runtime.to(device)
    offsets = torch.as_tensor(target_offsets_xy, dtype=torch.float32, device=device)
    observed = torch.as_tensor(target_observed, dtype=torch.bool, device=device)
    expected = (active.point_count, active.candidate_count)
    if (
        offsets.shape != (*expected, 2)
        or observed.shape != expected
        or prediction.joint_log_probabilities.shape[:2] != expected
        or permuted_prediction.joint_log_probabilities.shape[:2] != expected
        or permuted.support_image_indices.shape != active.support_image_indices.shape
        or not torch.equal(permuted.query_image_indices, active.query_image_indices)
        or not torch.equal(permuted.query_xy, active.query_xy)
        or not torch.equal(permuted.support_view_valid, active.support_view_valid)
        or not torch.equal(permuted.candidate_view_weights, active.candidate_view_weights)
        or not torch.equal(permuted.candidate_probabilities, active.candidate_probabilities)
        or not torch.equal(permuted.null_probabilities, active.null_probabilities)
        or (
            torch.equal(permuted.support_image_indices, active.support_image_indices)
            and torch.equal(permuted.support_xy, active.support_xy)
        )
        or not torch.isfinite(offsets).all()
        or torch.any(observed.sum(dim=1) > 1)
    ):
        raise ValueError("identity support contrastive targets do not match the runtime")
    point_indices, candidate_indices = torch.nonzero(observed, as_tuple=True)
    if len(point_indices) == 0:
        zero = prediction.spatial_logits.sum() * 0.0 + permuted_prediction.spatial_logits.sum() * 0.0
        return zero, {
            "identity_support_active_edges": 0.0,
            "identity_support_normal_mean_llr": 0.0,
            "identity_support_permuted_mean_llr": 0.0,
            "identity_support_mean_gap": 0.0,
            "identity_support_normal_win_fraction": 0.0,
            "identity_support_margin_loss": 0.0,
        }
    selected_offsets = offsets[point_indices, candidate_indices]
    normal_scores, normal_usable = selected_candidate_view_log_likelihood_ratio_at_offsets(
        runtime=active,
        prediction=prediction,
        point_indices=point_indices,
        candidate_indices=candidate_indices,
        offsets_xy=selected_offsets,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    )
    permuted_scores, permuted_usable = selected_candidate_view_log_likelihood_ratio_at_offsets(
        runtime=permuted,
        prediction=permuted_prediction,
        point_indices=point_indices,
        candidate_indices=candidate_indices,
        offsets_xy=selected_offsets,
        missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    )
    usable = normal_usable & permuted_usable
    if not bool(usable.any()):
        zero = prediction.spatial_logits.sum() * 0.0 + permuted_prediction.spatial_logits.sum() * 0.0
        return zero, {
            "identity_support_active_edges": 0.0,
            "identity_support_normal_mean_llr": 0.0,
            "identity_support_permuted_mean_llr": 0.0,
            "identity_support_mean_gap": 0.0,
            "identity_support_normal_win_fraction": 0.0,
            "identity_support_margin_loss": 0.0,
        }
    gaps = normal_scores[usable] - permuted_scores[usable]
    loss = F.softplus(
        torch.as_tensor(target_margin, dtype=gaps.dtype, device=gaps.device) - gaps
    ).mean()
    return loss, {
        "identity_support_active_edges": float(usable.sum().item()),
        "identity_support_normal_mean_llr": float(normal_scores[usable].detach().mean().item()),
        "identity_support_permuted_mean_llr": float(
            permuted_scores[usable].detach().mean().item()
        ),
        "identity_support_mean_gap": float(gaps.detach().mean().item()),
        "identity_support_normal_win_fraction": float(
            (gaps.detach() > 0.0).float().mean().item()
        ),
        "identity_support_margin_loss": float(loss.detach().item()),
    }


def train_support_permutation_shift(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    query_id: str,
    epoch: int,
    reserved_control_shift: int,
) -> int:
    """Choose a deterministic nonidentity derangement not used by the gate."""

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime) or not str(query_id):
        raise ValueError("support permutation shift requires a valid runtime and query")
    control = int(reserved_control_shift)
    if control <= 0:
        raise ValueError("support permutation control shift must be positive")
    counts = sorted(
        {
            int(value)
            for value in runtime.support_view_valid.sum(dim=(1, 2)).detach().cpu().tolist()
        }
    )
    if not counts or min(counts) < 2:
        raise ValueError("support permutation contrastive loss requires at least two views per point")
    if any(control % count == 0 for count in counts):
        raise ValueError("support permutation control shift leaves an input unchanged")
    candidates = [
        shift
        for shift in range(1, max(counts) * 2 + 1)
        if all(shift % count != 0 and shift % count != control % count for count in counts)
    ]
    if not candidates:
        raise ValueError("no train-only support derangement remains after reserving the gate")
    position = (int(_stable_query_hash(query_id)) + int(epoch) * 2654435761) % len(candidates)
    return int(candidates[position])


def _slice_runtime(
    runtime: CandidatePoseRGBSpatialRuntime, rows: np.ndarray
) -> CandidatePoseRGBSpatialRuntime:
    indices = torch.from_numpy(np.asarray(rows, dtype=np.int64).reshape(-1))
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=runtime.query_image_indices.index_select(0, indices),
        query_xy=runtime.query_xy.index_select(0, indices),
        support_image_indices=runtime.support_image_indices.index_select(0, indices),
        support_xy=runtime.support_xy.index_select(0, indices),
        support_view_valid=runtime.support_view_valid.index_select(0, indices),
        candidate_view_weights=runtime.candidate_view_weights.index_select(0, indices),
        candidate_probabilities=runtime.candidate_probabilities.index_select(0, indices),
        null_probabilities=runtime.null_probabilities.index_select(0, indices),
    )


def _select_group_points(
    *,
    group: TrainQueryGroup,
    max_points: int,
    seed: int,
    required_source_point_ids: np.ndarray | Sequence[int] | None = None,
) -> np.ndarray:
    """Keep scarce observations and subsample dense hard-repeat sources safely.

    Exact registered observations are sparse: in the current P1 contract most
    rows have a dustbin target for nearly every candidate.  Sampling merely by
    ``spatial_target_supervised`` therefore looks balanced while randomly
    discarding the only candidate-specific positive rows.  The selection keeps
    every observed source point, plus any explicitly requested hard-repeat
    source, before using deterministic filler rows for pose coverage. Expanded
    observation pools can contain hundreds of valid hard-repeat sources per
    query, though, so requiring all of them under a fixed 64-point budget would
    silently make the intended multi-negative protocol untrainable. In that
    dense case we sample only from the required source pool, deterministically
    per query/epoch; the later edge sampler still draws the actual coherent
    wrong candidate identities after this source-level selection.
    """

    limit = int(max_points)
    if limit <= 0 or limit >= group.point_count:
        return np.arange(group.point_count, dtype=np.int64)
    if limit < 4:
        raise ValueError("RGB spatial likelihood requires at least four query points per group")
    generator = np.random.default_rng(int(seed) ^ _stable_query_hash(group.query_id))
    observed = np.flatnonzero(
        np.any(group.spatial_target_observed & group.spatial_target_supervised, axis=1)
    )
    required = np.asarray(
        [] if required_source_point_ids is None else required_source_point_ids,
        dtype=np.int64,
    ).reshape(-1)
    # A source point can participate in several mined coherent-wrong pose
    # pairs.  Selection is per point, so deduplicate that normal pair-level
    # repetition before applying the fixed point budget.
    required = np.unique(required)
    source_position = {
        int(source_id): position
        for position, source_id in enumerate(group.source_point_ids.tolist())
    }
    try:
        required_positions = np.asarray(
            [source_position[int(source_id)] for source_id in required.tolist()], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("RGB spatial required hard-repeat source is absent from query group") from error
    if len(required_positions) > limit:
        return np.sort(
            generator.choice(required_positions, size=limit, replace=False)
        ).astype(np.int64)
    priority = np.unique(np.concatenate([observed, required_positions])).astype(np.int64)
    if len(priority) > limit:
        optional_observed = np.setdiff1d(observed, required_positions, assume_unique=False)
        filler_count = limit - len(required_positions)
        priority = np.sort(
            np.concatenate(
                [
                    required_positions,
                    generator.choice(optional_observed, size=filler_count, replace=False),
                ]
            )
        ).astype(np.int64)
    if len(priority) >= limit:
        return np.sort(priority).astype(np.int64)
    remaining = np.setdiff1d(
        np.arange(group.point_count, dtype=np.int64), priority, assume_unique=True
    )
    additional = generator.choice(remaining, size=limit - len(priority), replace=False)
    return np.sort(np.concatenate([priority, additional])).astype(np.int64)


def configure_rgb_cost_volume_only_trainable_parameters(
    model: CandidatePoseRGBSpatialLikelihood,
) -> tuple[str, ...]:
    """Freeze all non-RGB paths before DDP registers trainable parameters."""

    if not isinstance(model, CandidatePoseRGBSpatialLikelihood):
        raise ValueError("RGB-only trainable scope requires the candidate likelihood model")
    selected: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith("texture_encoder.")
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(name)
    if not selected:
        raise RuntimeError("RGB-only trainable scope found no texture parameters")
    return tuple(selected)


def _query_batch_from_group(
    *,
    group: TrainQueryGroup,
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    point_positions: np.ndarray,
    device: torch.device,
) -> _QueryBatch:
    positions = np.asarray(point_positions, dtype=np.int64).reshape(-1)
    if len(positions) == 0 or np.any(positions < 0) or np.any(positions >= group.point_count):
        raise ValueError("RGB spatial query point selection is invalid")
    runtime = _slice_runtime(complete_runtime, group.layout_rows[positions])
    return _QueryBatch(
        runtime=runtime,
        spatial_target_offsets_xy=torch.from_numpy(
            group.spatial_target_offsets_xy[positions]
        ).to(device=device),
        spatial_target_observed=torch.from_numpy(
            group.spatial_target_observed[positions]
        ).to(device=device),
        spatial_target_supervised=torch.from_numpy(
            group.spatial_target_supervised[positions]
        ).to(device=device),
        spatial_target_dustbin=torch.from_numpy(
            group.spatial_target_dustbin[positions]
        ).to(device=device),
        correct_projection_offsets_xy=torch.from_numpy(
            group.correct_projection_offsets_xy[positions]
        ).to(device=device),
        correct_projection_valid=torch.from_numpy(
            group.correct_projection_valid[positions]
        ).to(device=device),
        wrong_projection_offsets_xy=torch.from_numpy(
            group.wrong_projection_offsets_xy[:, positions]
        ).to(device=device),
        wrong_projection_valid=torch.from_numpy(
            group.wrong_projection_valid[:, positions]
        ).to(device=device),
    )


def registered_identity_observed_source_point_ids(
    group: TrainQueryGroup,
) -> np.ndarray:
    """Return sparse registered-positive source IDs for train-only batch retention."""

    if not isinstance(group, TrainQueryGroup):
        raise ValueError("registered identity source selection requires a train query group")
    observed = np.any(
        group.spatial_target_observed & group.spatial_target_supervised,
        axis=1,
    )
    return np.asarray(group.source_point_ids[observed], dtype=np.int64)


def registered_identity_batch_for_geometry_selection(
    *,
    geometry_group: TrainQueryGroup,
    registered_identity_group: TrainQueryGroup,
    point_positions: np.ndarray,
    device: torch.device,
) -> RegisteredIdentityBatch:
    """Join registered labels to selected geometry rows by immutable source ID.

    This function is intentionally called only after the normal target-free
    visual prediction is emitted. It permits a regenerated identity artifact
    to use a different row order, but rejects every source/query/candidate
    mismatch rather than relying on coincident tensor dimensions.
    """

    if (
        not isinstance(geometry_group, TrainQueryGroup)
        or not isinstance(registered_identity_group, TrainQueryGroup)
        or str(geometry_group.query_id) != str(registered_identity_group.query_id)
    ):
        raise ValueError("registered identity query group does not match geometry group")
    if (
        geometry_group.spatial_target_offsets_xy.shape[1]
        != registered_identity_group.spatial_target_offsets_xy.shape[1]
    ):
        raise ValueError("registered identity group candidate count differs from geometry group")
    if set(geometry_group.source_point_ids.tolist()) != set(
        registered_identity_group.source_point_ids.tolist()
    ):
        raise ValueError("registered identity group source-point universe differs from geometry group")
    positions = np.asarray(point_positions, dtype=np.int64).reshape(-1)
    if (
        len(positions) == 0
        or np.any(positions < 0)
        or np.any(positions >= geometry_group.point_count)
    ):
        raise ValueError("registered identity selection positions are invalid")
    identity_position_by_source = {
        int(source_id): position
        for position, source_id in enumerate(registered_identity_group.source_point_ids.tolist())
    }
    selected_source_ids = geometry_group.source_point_ids[positions]
    try:
        identity_positions = np.asarray(
            [identity_position_by_source[int(source_id)] for source_id in selected_source_ids.tolist()],
            dtype=np.int64,
        )
    except KeyError as error:  # Defensive: the full-universe check above should catch this.
        raise ValueError("registered identity selection source is absent") from error
    target_observed = np.asarray(
        registered_identity_group.spatial_target_observed[identity_positions], dtype=bool
    )
    target_supervised = np.asarray(
        registered_identity_group.spatial_target_supervised[identity_positions], dtype=bool
    )
    if np.any(target_observed & ~target_supervised):
        raise ValueError("registered identity observed target is not supervised")
    return RegisteredIdentityBatch(
        target_offsets_xy=torch.from_numpy(
            np.asarray(
                registered_identity_group.spatial_target_offsets_xy[identity_positions],
                dtype=np.float32,
            )
        ).to(device=device),
        target_observed=torch.from_numpy(target_observed).to(device=device),
    )


def _source_table(sources: Sequence[ContextAttentionSource]) -> tuple[np.ndarray, np.ndarray, dict[str, torch.Tensor]]:
    by_name = {str(source.name): source for source in sources}
    if set(by_name) != {"radio_final", "radio_intermediate", "alike"}:
        raise ValueError("RGB spatial likelihood context sources are incomplete")
    reference = by_name["radio_final"]
    image_ids = np.asarray(reference.image_ids).astype(str)
    image_sizes = np.asarray(reference.image_sizes, dtype=np.int64)
    for source in by_name.values():
        if not np.array_equal(np.asarray(source.image_ids).astype(str), image_ids) or not np.array_equal(
            np.asarray(source.image_sizes, dtype=np.int64), image_sizes
        ):
            raise ValueError("RGB spatial likelihood context source ownership differs")
    if len(image_ids) == 0 or image_sizes.shape != (len(image_ids), 2):
        raise ValueError("RGB spatial likelihood context image table is invalid")
    return image_ids, image_sizes, {
        name: torch.from_numpy(np.asarray(source.grid, dtype=np.float32))
        for name, source in by_name.items()
    }


def _rgb_loader_for_image(
    *,
    image_root: Path,
    image_id: str,
    expected_size: tuple[int, int],
    cache_storage_dtype: torch.dtype | None,
) -> Any:
    path = Path(image_root) / str(image_id)
    if not path.is_file():
        raise FileNotFoundError(f"RGB image is absent: {path}")

    def load() -> torch.Tensor:
        image = (
            _load_query_rgb_uint8(path)
            if cache_storage_dtype == torch.uint8
            else _load_query_rgb(path)
        )
        if image.ndim != 3 or image.shape[0] != 3 or (int(image.shape[2]), int(image.shape[1])) != expected_size:
            raise ValueError(f"RGB image size differs from frozen context source: {path}")
        return image

    return load


def rgb_coordinate_scale(
    *, coordinate_image_size: tuple[int, int], rgb_image_size: tuple[int, int]
) -> float:
    """Return the explicit isotropic raw-RGB-to-SfM coordinate scale.

    P1 anchors, candidate projections, and context grids use the SfM coordinate
    frame.  When real RGB retains a uniformly larger processed resolution,
    centers, search radius, and sampling step must all be scaled together.  The
    density still emits offsets in the original SfM-pixel unit.
    """

    coordinate_width, coordinate_height = (int(value) for value in coordinate_image_size)
    rgb_width, rgb_height = (int(value) for value in rgb_image_size)
    if min(coordinate_width, coordinate_height, rgb_width, rgb_height) <= 1:
        raise ValueError("RGB coordinate bridge image sizes are invalid")
    scale_x = float(rgb_width) / float(coordinate_width)
    scale_y = float(rgb_height) / float(coordinate_height)
    if not math.isclose(scale_x, scale_y, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("RGB and context coordinate frames are not isotropically related")
    return scale_x


def validate_rgb_coordinate_bridge(
    *,
    source_metadata: Mapping[str, object],
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
) -> dict[str, object]:
    """Require the frozen context cache to explain any RGB resolution change."""

    observed_scale = rgb_coordinate_scale(
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
    )
    bridge = source_metadata.get("coordinate_bridge") if isinstance(source_metadata, Mapping) else None
    if bridge is None:
        if coordinate_image_size != rgb_image_size:
            raise ValueError("frozen context cache lacks an RGB coordinate bridge")
        return {
            "format": "identity_rgb_coordinate_bridge_v1",
            "aligned_coordinate_size": list(coordinate_image_size),
            "raw_rgb_size": list(rgb_image_size),
            "raw_pixels_per_aligned_pixel": 1.0,
        }
    if not isinstance(bridge, Mapping) or bridge.get("format") != "normalized_adaptive_grid_coordinate_bridge_v1":
        raise ValueError("frozen context cache RGB coordinate bridge is invalid")
    try:
        bridge_coordinate_size = tuple(int(value) for value in bridge["aligned_coordinate_size"])
        bridge_rgb_size = tuple(int(value) for value in bridge["raw_rgb_size"])
        bridge_scale = float(bridge["raw_pixels_per_aligned_pixel"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frozen context cache RGB coordinate bridge is invalid") from error
    if (
        bridge_coordinate_size != tuple(coordinate_image_size)
        or bridge_rgb_size != tuple(rgb_image_size)
        or not math.isfinite(bridge_scale)
        or not math.isclose(bridge_scale, observed_scale, rel_tol=1e-6, abs_tol=1e-6)
    ):
        raise ValueError("real RGB root differs from frozen context coordinate bridge")
    return dict(bridge)


def _discover_rgb_image_size(*, image_root: Path, image_id: str) -> tuple[int, int]:
    path = Path(image_root) / str(image_id)
    if not path.is_file():
        raise FileNotFoundError(f"RGB image is absent: {path}")
    with Image.open(path) as image:
        width, height = image.size
    if int(width) <= 1 or int(height) <= 1:
        raise ValueError(f"RGB image dimensions are invalid: {path}")
    return int(width), int(height)


def _crop_runtime_rgb_patches(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    radius_px: float,
    step_px: float,
    cache: TensorImageLRUCache,
    device: torch.device,
    cache_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load/crop query once and all support edges grouped by RGB owner image.

    ``cache_device=None`` retains the historical GPU-resident cache behavior.
    A CPU cache keeps decoded uint8 source images out of activation memory;
    each grouped owner batch is still copied to ``device`` only for cropping.
    """

    active = runtime.to("cpu")
    ids = np.asarray(image_ids).astype(str)
    coordinate_scale = rgb_coordinate_scale(
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
    )
    width, height = (int(rgb_image_size[0]), int(rgb_image_size[1]))
    storage_device = device if cache_device is None else torch.device(cache_device)
    if (
        active.query_image_indices.numel() == 0
        or torch.any(active.query_image_indices >= len(ids))
        or torch.any(active.support_image_indices >= len(ids))
    ):
        raise ValueError("RGB crop runtime image indices are invalid")
    loader_by_id: dict[str, Any] = {}

    def spec_for_index(index: int) -> tuple[str, Any]:
        image_id = str(ids[int(index)])
        loader = loader_by_id.get(image_id)
        if loader is None:
            loader = _rgb_loader_for_image(
                image_root=Path(image_root),
                image_id=image_id,
                expected_size=(width, height),
                cache_storage_dtype=cache.storage_dtype,
            )
            loader_by_id[image_id] = loader
        return image_id, loader

    query_indices = active.query_image_indices.detach().cpu().numpy().astype(np.int64)
    query_specs = [spec_for_index(int(index)) for index in query_indices.tolist()]
    query_centers = (
        active.query_xy.detach().cpu().numpy().astype(np.float32) * coordinate_scale
    ).tolist()
    query_patches = _crop_cached_rgb_windows_grouped(
        query_specs,
        query_centers,
        cache=cache,
        cache_device=storage_device,
        crop_device=device,
        radius_px=float(radius_px) * coordinate_scale,
        step_px=float(step_px) * coordinate_scale,
        image_width=width,
        image_height=height,
    )
    point_count, candidate_count, view_count = active.support_image_indices.shape
    support_indices = active.support_image_indices.detach().cpu().numpy().astype(np.int64)
    support_valid = active.support_view_valid.detach().cpu().numpy().astype(bool)
    support_centers = active.support_xy.detach().cpu().numpy().astype(np.float32)
    support_specs: list[tuple[str, Any]] = []
    support_center_rows: list[list[float]] = []
    for point_index in range(point_count):
        fallback = int(query_indices[point_index])
        for candidate_index in range(candidate_count):
            for view_index in range(view_count):
                index = int(support_indices[point_index, candidate_index, view_index])
                if not bool(support_valid[point_index, candidate_index, view_index]):
                    index = fallback
                support_specs.append(spec_for_index(index))
                support_center_rows.append(
                    (
                        support_centers[point_index, candidate_index, view_index]
                        * coordinate_scale
                    ).tolist()
                )
    flat_support = _crop_cached_rgb_windows_grouped(
        support_specs,
        support_center_rows,
        cache=cache,
        cache_device=storage_device,
        crop_device=device,
        radius_px=float(radius_px) * coordinate_scale,
        step_px=float(step_px) * coordinate_scale,
        image_width=width,
        image_height=height,
    )
    support_patches = flat_support.reshape(
        point_count, candidate_count, view_count, *flat_support.shape[1:]
    )
    invalid = ~active.support_view_valid.to(device=support_patches.device)
    support_patches = torch.where(
        invalid[:, :, :, None, None, None],
        torch.zeros_like(support_patches),
        support_patches,
    )
    return query_patches, support_patches


def _crop_geometry_fixed_permuted_support_patches(
    *,
    normal_query_patches: torch.Tensor,
    permuted_runtime: CandidatePoseRGBSpatialRuntime,
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    radius_px: float,
    step_px: float,
    cache: TensorImageLRUCache,
    device: torch.device,
) -> torch.Tensor:
    """Recrop a geometry-fixed support-image derangement from real RGB.

    Reindexing an already cropped support tensor is valid only when the
    support image *and* support coordinate move together.  The strict control
    keeps coordinates fixed, so its RGB support patches must be sampled again
    from the deranged image IDs at the original coordinates.  The query crop
    is an invariant and is checked explicitly.
    """

    deranged_query, deranged_support = _crop_runtime_rgb_patches(
        runtime=permuted_runtime,
        image_ids=image_ids,
        image_root=image_root,
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
        radius_px=float(radius_px),
        step_px=float(step_px),
        cache=cache,
        device=device,
    )
    if not torch.equal(torch.as_tensor(normal_query_patches), deranged_query):
        raise RuntimeError("geometry-fixed support control changed the query RGB crop")
    return deranged_support


def _reduce_tensor(state: _DistributedState, values: torch.Tensor) -> torch.Tensor:
    output = values.detach().clone()
    if state.enabled:
        distributed.all_reduce(output, op=distributed.ReduceOp.SUM)
    return output


def fixed_final_epoch_checkpoint_selection(*, epochs: int) -> dict[str, object]:
    """Declare that inner validation is a gate, not a checkpoint selector."""

    if int(epochs) <= 0:
        raise ValueError("RGB spatial fixed-final-epoch selection requires positive epochs")
    return {
        "policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
        "selected_epoch": int(epochs),
        "inner_validation_used_for_model_selection": False,
    }


def _model_state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _assert_geometry_fixed_support_image_control(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    permuted_runtime: CandidatePoseRGBSpatialRuntime,
    normal_prediction: CandidatePoseRGBSpatialEdgePrediction,
    permuted_prediction: CandidatePoseRGBSpatialEdgePrediction,
) -> None:
    """Reject a visual control that changes geometry, mass, or availability."""

    invariants = (
        (runtime.query_image_indices, permuted_runtime.query_image_indices),
        (runtime.query_xy, permuted_runtime.query_xy),
        (runtime.support_xy, permuted_runtime.support_xy),
        (runtime.support_view_valid, permuted_runtime.support_view_valid),
        (runtime.candidate_view_weights, permuted_runtime.candidate_view_weights),
        (runtime.candidate_probabilities, permuted_runtime.candidate_probabilities),
        (runtime.null_probabilities, permuted_runtime.null_probabilities),
    )
    if any(not torch.equal(left, right) for left, right in invariants):
        raise RuntimeError("RGB spatial support control changed geometry or mixture mass")
    valid_count = runtime.support_view_valid.reshape(runtime.point_count, -1).sum(dim=1)
    if bool(torch.any(valid_count > 1)) and torch.equal(
        runtime.support_image_indices, permuted_runtime.support_image_indices
    ):
        raise RuntimeError("RGB spatial support control did not derange image content")
    for name in ("edge_usable", "rgb_edge_usable", "context_edge_usable"):
        normal = getattr(normal_prediction, name, None)
        permuted = getattr(permuted_prediction, name, None)
        if normal is None or permuted is None or not torch.equal(normal, permuted):
            raise RuntimeError(
                "RGB spatial support control changed source crop availability"
            )


def _is_better_inner_validation_epoch(
    *,
    candidate: Mapping[str, float],
    incumbent: Mapping[str, float] | None,
    candidate_gate_passed: bool = False,
    incumbent_gate_passed: bool = False,
) -> bool:
    loss = float(candidate["normal_query_grouped_loss"])
    wins = float(candidate["normal_correct_win_fraction"])
    if not math.isfinite(loss) or not math.isfinite(wins):
        raise ValueError("RGB spatial likelihood inner validation metrics are invalid")
    if incumbent is None:
        return True
    if bool(candidate_gate_passed) != bool(incumbent_gate_passed):
        return bool(candidate_gate_passed)
    hard_keys = (
        "hard_repeat_mean_correct_minus_coherent_wrong",
        "hard_repeat_correct_win_fraction",
        "hard_repeat_eligible_query_fraction",
        "hard_repeat_permuted_mean_correct_minus_coherent_wrong",
    )
    if all(name in candidate for name in hard_keys) and all(name in incumbent for name in hard_keys):
        candidate_hard_gap = float(candidate[hard_keys[0]])
        candidate_hard_win = float(candidate[hard_keys[1]])
        candidate_hard_coverage = float(candidate[hard_keys[2]])
        candidate_hard_visual = candidate_hard_gap - float(candidate[hard_keys[3]])
        incumbent_hard_gap = float(incumbent[hard_keys[0]])
        incumbent_hard_win = float(incumbent[hard_keys[1]])
        incumbent_hard_coverage = float(incumbent[hard_keys[2]])
        incumbent_hard_visual = incumbent_hard_gap - float(incumbent[hard_keys[3]])
        candidate_key = (
            candidate_hard_gap,
            candidate_hard_win,
            candidate_hard_visual,
            candidate_hard_coverage,
        )
        incumbent_key = (
            incumbent_hard_gap,
            incumbent_hard_win,
            incumbent_hard_visual,
            incumbent_hard_coverage,
        )
        if not all(math.isfinite(value) for value in (*candidate_key, *incumbent_key)):
            raise ValueError("RGB spatial hard-repeat inner validation metrics are invalid")
        if candidate_key != incumbent_key:
            return candidate_key > incumbent_key
    old_loss = float(incumbent["normal_query_grouped_loss"])
    old_wins = float(incumbent["normal_correct_win_fraction"])
    return loss < old_loss - 1e-12 or (abs(loss - old_loss) <= 1e-12 and wins > old_wins)


@torch.no_grad()
def _evaluate_inner_validation_components(
    *,
    model: torch.nn.Module,
    groups: Mapping[str, TrainQueryGroup],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
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
    pose_margin: float,
    missing_edge_log_likelihood_ratio: float,
    max_abs_pose_log_ratio: float,
    amp_enabled: bool,
    permutation_control_shift: int,
    prediction_transforms: Mapping[
        str,
        Callable[[CandidatePoseRGBSpatialEdgePrediction], CandidatePoseRGBSpatialEdgePrediction]
        | None,
    ],
    context_only: bool = False,
    rgb_cost_volume_only: bool = False,
    context_appearance_mode: str = "visual",
) -> dict[str, dict[str, float]]:
    """Evaluate several score decompositions from one visual forward pass.

    Every component sees byte-identical normal/permuted RGB and frozen runtime
    inputs.  Only the post-encoder target-free density/context tensors are
    neutralized differently, so a component audit does not multiply expensive
    RGB or cross-attention inference by the number of reported decompositions.
    """

    appearance_mode = str(context_appearance_mode).strip().lower()
    if (
        not query_ids
        or appearance_mode not in {"visual", "position_only"}
        or (appearance_mode != "visual" and not bool(context_only))
        or (bool(context_only) and bool(rgb_cost_volume_only))
    ):
        raise ValueError("RGB spatial inner validation has no query groups")
    component_names = tuple(str(name) for name in prediction_transforms)
    if (
        not component_names
        or len(set(component_names)) != len(component_names)
        or any(not name for name in component_names)
    ):
        raise ValueError("RGB spatial component evaluation transforms are invalid")
    model.eval()
    totals = torch.zeros((len(component_names), 7), dtype=torch.float64, device=state.device)
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        if group is None:
            raise ValueError("RGB spatial inner validation query group is unresolved")
        positions = _select_group_points(
            group=group, max_points=max_points_per_query, seed=int(seed)
        )
        batch = _query_batch_from_group(
            group=group, complete_runtime=complete_runtime, point_positions=positions, device=state.device
        )
        if bool(context_only):
            query_patches = None
            support_patches = None
        else:
            query_patches, support_patches = _crop_runtime_rgb_patches(
                runtime=batch.runtime,
                image_ids=image_ids,
                image_root=image_root,
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=radius_px,
                step_px=step_px,
                cache=cache,
                device=state.device,
            )
        permuted_runtime = permute_runtime_support_image_appearance_only(
            batch.runtime, shift=int(permutation_control_shift)
        )
        if torch.equal(permuted_runtime.support_image_indices, batch.runtime.support_image_indices):
            raise ValueError("support permutation control did not change the visual inputs")
        if bool(context_only):
            permuted_support_patches = None
        else:
            if support_patches is None:
                raise RuntimeError("RGB validation support patches are unexpectedly absent")
            permuted_support_patches = _crop_geometry_fixed_permuted_support_patches(
                normal_query_patches=query_patches,
                permuted_runtime=permuted_runtime,
                image_ids=image_ids,
                image_root=image_root,
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=radius_px,
                step_px=step_px,
                cache=cache,
                device=state.device,
            )
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            normal_prediction_base = model(
                runtime=batch.runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                context_only=bool(context_only),
                rgb_cost_volume_only=bool(rgb_cost_volume_only),
                context_appearance_mode=appearance_mode,
            )
            permuted_prediction_base = model(
                runtime=permuted_runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=permuted_support_patches,
                context_only=bool(context_only),
                rgb_cost_volume_only=bool(rgb_cost_volume_only),
                context_appearance_mode=appearance_mode,
            )
            _assert_geometry_fixed_support_image_control(
                runtime=batch.runtime,
                permuted_runtime=permuted_runtime,
                normal_prediction=normal_prediction_base,
                permuted_prediction=permuted_prediction_base,
            )
            for component_position, component_name in enumerate(component_names):
                transform = prediction_transforms[component_name]
                normal_prediction = (
                    normal_prediction_base
                    if transform is None
                    else transform(normal_prediction_base)
                )
                permuted_prediction = (
                    permuted_prediction_base
                    if transform is None
                    else transform(permuted_prediction_base)
                )
                normal_correct = score_candidate_pose_rgb_spatial_batch(
                    runtime=batch.runtime,
                    prediction=normal_prediction,
                    candidate_projection_offsets_xy=batch.correct_projection_offsets_xy.unsqueeze(0),
                    candidate_projection_valid=batch.correct_projection_valid.unsqueeze(0),
                    missing_edge_log_likelihood_ratio=missing_edge_log_likelihood_ratio,
                    max_abs_log_likelihood_ratio=max_abs_pose_log_ratio,
                ).pose_log_likelihood_ratios
                normal_wrong = score_candidate_pose_rgb_spatial_batch(
                    runtime=batch.runtime,
                    prediction=normal_prediction,
                    candidate_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                    candidate_projection_valid=batch.wrong_projection_valid,
                    missing_edge_log_likelihood_ratio=missing_edge_log_likelihood_ratio,
                    max_abs_log_likelihood_ratio=max_abs_pose_log_ratio,
                ).pose_log_likelihood_ratios
                normal_loss, normal_metrics = query_grouped_pose_margin_loss(
                    correct_scores=normal_correct,
                    coherent_wrong_scores=normal_wrong.reshape(1, -1),
                    margin=pose_margin,
                )
                permuted_correct = score_candidate_pose_rgb_spatial_batch(
                    runtime=permuted_runtime,
                    prediction=permuted_prediction,
                    candidate_projection_offsets_xy=batch.correct_projection_offsets_xy.unsqueeze(0),
                    candidate_projection_valid=batch.correct_projection_valid.unsqueeze(0),
                    missing_edge_log_likelihood_ratio=missing_edge_log_likelihood_ratio,
                    max_abs_log_likelihood_ratio=max_abs_pose_log_ratio,
                ).pose_log_likelihood_ratios
                permuted_wrong = score_candidate_pose_rgb_spatial_batch(
                    runtime=permuted_runtime,
                    prediction=permuted_prediction,
                    candidate_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                    candidate_projection_valid=batch.wrong_projection_valid,
                    missing_edge_log_likelihood_ratio=missing_edge_log_likelihood_ratio,
                    max_abs_log_likelihood_ratio=max_abs_pose_log_ratio,
                ).pose_log_likelihood_ratios
                permuted_loss, permuted_metrics = query_grouped_pose_margin_loss(
                    correct_scores=permuted_correct,
                    coherent_wrong_scores=permuted_wrong.reshape(1, -1),
                    margin=pose_margin,
                )
                totals[component_position] += torch.tensor(
                    [
                        float(normal_loss.item()),
                        float(normal_metrics["query_mean_correct_minus_hardest_wrong"]),
                        float(normal_metrics["query_correct_win_fraction"]),
                        float(permuted_loss.item()),
                        float(permuted_metrics["query_mean_correct_minus_hardest_wrong"]),
                        float(permuted_metrics["query_correct_win_fraction"]),
                        1.0,
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
    totals = _reduce_tensor(state, totals)
    if bool(torch.any(totals[:, -1] <= 0.0)):
        raise RuntimeError("RGB spatial inner validation did not evaluate a query")
    return {
        component_name: {
            "normal_query_grouped_loss": float(
                (totals[position, 0] / totals[position, -1]).item()
            ),
            "normal_mean_correct_minus_hardest_wrong": float(
                (totals[position, 1] / totals[position, -1]).item()
            ),
            "normal_correct_win_fraction": float(
                (totals[position, 2] / totals[position, -1]).item()
            ),
            "permuted_query_grouped_loss": float(
                (totals[position, 3] / totals[position, -1]).item()
            ),
            "permuted_mean_correct_minus_hardest_wrong": float(
                (totals[position, 4] / totals[position, -1]).item()
            ),
            "permuted_correct_win_fraction": float(
                (totals[position, 5] / totals[position, -1]).item()
            ),
            "query_count": float(totals[position, -1].item()),
        }
        for position, component_name in enumerate(component_names)
    }


@torch.no_grad()
def _evaluate_inner_validation(
    *,
    model: torch.nn.Module,
    groups: Mapping[str, TrainQueryGroup],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
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
    pose_margin: float,
    missing_edge_log_likelihood_ratio: float,
    max_abs_pose_log_ratio: float,
    amp_enabled: bool,
    permutation_control_shift: int,
    prediction_transform: Callable[
        [CandidatePoseRGBSpatialEdgePrediction], CandidatePoseRGBSpatialEdgePrediction
    ]
    | None = None,
    rgb_cost_volume_only: bool = False,
) -> dict[str, float]:
    """Backward-compatible one-component wrapper for training checkpoints."""

    return _evaluate_inner_validation_components(
        model=model,
        groups=groups,
        complete_runtime=complete_runtime,
        query_ids=query_ids,
        image_ids=image_ids,
        image_root=image_root,
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
        radius_px=radius_px,
        step_px=step_px,
        cache=cache,
        state=state,
        max_points_per_query=max_points_per_query,
        seed=seed,
        pose_margin=pose_margin,
        missing_edge_log_likelihood_ratio=missing_edge_log_likelihood_ratio,
        max_abs_pose_log_ratio=max_abs_pose_log_ratio,
        amp_enabled=amp_enabled,
        permutation_control_shift=permutation_control_shift,
        prediction_transforms={"single": prediction_transform},
        rgb_cost_volume_only=bool(rgb_cost_volume_only),
    )["single"]


@torch.no_grad()
def _evaluate_inner_validation_target_free_static_selector(
    *,
    model: torch.nn.Module,
    layout: CandidatePoseRGBSpatialLayout,
    groups: Mapping[str, TrainQueryGroup],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    query_ids: Sequence[str],
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    radius_px: float,
    step_px: float,
    cache: TensorImageLRUCache,
    state: _DistributedState,
    selector_policy: str,
    selector_point_budget: int,
    selector_grid_rows: int,
    selector_grid_columns: int,
    pose_margin: float,
    missing_edge_log_likelihood_ratio: float,
    max_abs_pose_log_ratio: float,
    amp_enabled: bool,
    permutation_control_shift: int,
    seed: int,
    rgb_cost_volume_only: bool,
    hard_repeat_by_query: Mapping[str, HardRepeatQueryTargets] | None = None,
    registered_identity_groups: Mapping[str, TrainQueryGroup] | None = None,
    max_hard_repeat_edges_per_query: int = 0,
    include_context_identity_diagnostics: bool = False,
) -> dict[str, float]:
    """Run the checkpoint gate with target-free static selection.

    The training sampler may retain scarce registered observations and mined
    hard-repeat rows to preserve supervision. The gate cannot use that
    privilege: it selects from frozen layout fields, runs the visual scorer,
    and only then joins correct/coherent-wrong projections for diagnostics.
    """

    if (
        not query_ids
        or str(selector_policy) not in _TARGET_FREE_STATIC_SELECTOR_POLICIES
        or int(selector_point_budget) < 4
        or int(selector_grid_rows) <= 0
        or int(selector_grid_columns) <= 0
        or int(max_hard_repeat_edges_per_query) < 0
        or int(seed) < 0
        or (bool(rgb_cost_volume_only) and bool(include_context_identity_diagnostics))
    ):
        raise ValueError("RGB spatial target-free validation arguments are invalid")
    model.eval()
    hard_repeat_enabled = hard_repeat_by_query is not None
    hard_repeat_metric_count = 6 if hard_repeat_enabled else 0
    context_metric_offset = 7 + hard_repeat_metric_count
    totals = torch.zeros(
        (
            context_metric_offset
            + (11 if bool(include_context_identity_diagnostics) else 0),
        ),
        dtype=torch.float64,
        device=state.device,
    )
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        if group is None:
            raise ValueError("RGB spatial target-free validation query is unresolved")
        selector_input = selector_input_from_target_free_layout(
            layout=layout, rows=group.layout_rows
        )
        if int(selector_point_budget) > selector_input.point_count:
            raise ValueError("RGB spatial validation selector budget exceeds frozen P1 pool")
        selector_scores = target_free_selector_scores(
            selector_input=selector_input,
            policy=str(selector_policy),
        )
        positions = select_target_free_spatial_quota(
            selector_input=selector_input,
            quality_scores=selector_scores,
            point_budget=int(selector_point_budget),
            grid_rows=int(selector_grid_rows),
            grid_columns=int(selector_grid_columns),
            image_size=coordinate_image_size,
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
        permuted_runtime = permute_runtime_support_image_appearance_only(
            batch.runtime, shift=int(permutation_control_shift)
        )
        if torch.equal(permuted_runtime.support_image_indices, batch.runtime.support_image_indices):
            raise RuntimeError("RGB spatial target-free gate permutation did not change support appearances")
        permuted_support_patches = _crop_geometry_fixed_permuted_support_patches(
            normal_query_patches=query_patches,
            permuted_runtime=permuted_runtime,
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
            normal_prediction = model(
                runtime=batch.runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                rgb_cost_volume_only=bool(rgb_cost_volume_only),
            )
            permuted_prediction = model(
                runtime=permuted_runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=permuted_support_patches,
                rgb_cost_volume_only=bool(rgb_cost_volume_only),
            )
            _assert_geometry_fixed_support_image_control(
                runtime=batch.runtime,
                permuted_runtime=permuted_runtime,
                normal_prediction=normal_prediction,
                permuted_prediction=permuted_prediction,
            )
            normal_correct = score_candidate_pose_rgb_spatial_batch(
                runtime=batch.runtime,
                prediction=normal_prediction,
                candidate_projection_offsets_xy=batch.correct_projection_offsets_xy.unsqueeze(0),
                candidate_projection_valid=batch.correct_projection_valid.unsqueeze(0),
                missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
            ).pose_log_likelihood_ratios
            normal_wrong = score_candidate_pose_rgb_spatial_batch(
                runtime=batch.runtime,
                prediction=normal_prediction,
                candidate_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                candidate_projection_valid=batch.wrong_projection_valid,
                missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
            ).pose_log_likelihood_ratios
            normal_loss, normal_metrics = query_grouped_pose_margin_loss(
                correct_scores=normal_correct,
                coherent_wrong_scores=normal_wrong.reshape(1, -1),
                margin=float(pose_margin),
            )
            permuted_correct = score_candidate_pose_rgb_spatial_batch(
                runtime=permuted_runtime,
                prediction=permuted_prediction,
                candidate_projection_offsets_xy=batch.correct_projection_offsets_xy.unsqueeze(0),
                candidate_projection_valid=batch.correct_projection_valid.unsqueeze(0),
                missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
            ).pose_log_likelihood_ratios
            permuted_wrong = score_candidate_pose_rgb_spatial_batch(
                runtime=permuted_runtime,
                prediction=permuted_prediction,
                candidate_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                candidate_projection_valid=batch.wrong_projection_valid,
                missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
            ).pose_log_likelihood_ratios
            permuted_loss, permuted_metrics = query_grouped_pose_margin_loss(
                correct_scores=permuted_correct,
                coherent_wrong_scores=permuted_wrong.reshape(1, -1),
                margin=float(pose_margin),
            )
            hard_repeat_metrics = {
                "active_edges": 0.0,
                "normal_gap_sum": 0.0,
                "normal_win_sum": 0.0,
                "permuted_gap_sum": 0.0,
                "permuted_win_sum": 0.0,
                "eligible_query": 0.0,
            }
            if hard_repeat_by_query is not None:
                hard_targets = hard_repeat_by_query.get(str(query_id))
                if hard_targets is not None:
                    # The frozen visual forwards and generic pose scores above
                    # are complete before the train-only repeat-pair targets
                    # are joined. The exact same selected P1 rows and candidate
                    # slots are then scored under normal and deranged support
                    # appearance, with availability intersected explicitly.
                    hard_batch = _hard_repeat_batch_from_group(
                        hard_targets=hard_targets,
                        group=group,
                        point_positions=positions,
                        device=state.device,
                        max_edges=int(max_hard_repeat_edges_per_query),
                        seed=int(seed),
                    )
                    if hard_batch is not None:
                        normal_positive, normal_positive_usable = (
                            selected_candidate_view_log_likelihood_ratio_at_offsets(
                                runtime=batch.runtime,
                                prediction=normal_prediction,
                                point_indices=hard_batch.point_indices,
                                candidate_indices=hard_batch.positive_candidate_indices,
                                offsets_xy=hard_batch.positive_offsets_xy,
                                missing_edge_log_likelihood_ratio=float(
                                    missing_edge_log_likelihood_ratio
                                ),
                                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
                            )
                        )
                        normal_negative, normal_negative_usable = (
                            selected_candidate_view_log_likelihood_ratio_at_offsets(
                                runtime=batch.runtime,
                                prediction=normal_prediction,
                                point_indices=hard_batch.point_indices,
                                candidate_indices=hard_batch.negative_candidate_indices,
                                offsets_xy=hard_batch.negative_offsets_xy,
                                missing_edge_log_likelihood_ratio=float(
                                    missing_edge_log_likelihood_ratio
                                ),
                                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
                            )
                        )
                        permuted_positive, permuted_positive_usable = (
                            selected_candidate_view_log_likelihood_ratio_at_offsets(
                                runtime=permuted_runtime,
                                prediction=permuted_prediction,
                                point_indices=hard_batch.point_indices,
                                candidate_indices=hard_batch.positive_candidate_indices,
                                offsets_xy=hard_batch.positive_offsets_xy,
                                missing_edge_log_likelihood_ratio=float(
                                    missing_edge_log_likelihood_ratio
                                ),
                                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
                            )
                        )
                        permuted_negative, permuted_negative_usable = (
                            selected_candidate_view_log_likelihood_ratio_at_offsets(
                                runtime=permuted_runtime,
                                prediction=permuted_prediction,
                                point_indices=hard_batch.point_indices,
                                candidate_indices=hard_batch.negative_candidate_indices,
                                offsets_xy=hard_batch.negative_offsets_xy,
                                missing_edge_log_likelihood_ratio=float(
                                    missing_edge_log_likelihood_ratio
                                ),
                                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
                            )
                        )
                        common = (
                            normal_positive_usable
                            & normal_negative_usable
                            & permuted_positive_usable
                            & permuted_negative_usable
                        )
                        if bool(common.any()):
                            normal_gap = normal_positive[common] - normal_negative[common]
                            permuted_gap = (
                                permuted_positive[common] - permuted_negative[common]
                            )
                            hard_repeat_metrics = {
                                "active_edges": float(common.sum().item()),
                                "normal_gap_sum": float(normal_gap.sum().item()),
                                "normal_win_sum": float(
                                    (normal_gap > 0.0).to(dtype=torch.float32).sum().item()
                                ),
                                "permuted_gap_sum": float(permuted_gap.sum().item()),
                                "permuted_win_sum": float(
                                    (permuted_gap > 0.0)
                                    .to(dtype=torch.float32)
                                    .sum()
                                    .item()
                                ),
                                "eligible_query": 1.0,
                            }
            context_identity_metrics = {
                "context_identity_cross_entropy": 0.0,
                "context_identity_top1_accuracy": 0.0,
                "context_identity_mean_margin": 0.0,
                "context_identity_active_rows": 0.0,
            }
            context_identity_permutation_metrics = {
                "context_identity_permutation_margin_loss": 0.0,
                "context_identity_permutation_mean_gap": 0.0,
                "context_identity_permutation_win_fraction": 0.0,
                "context_identity_permutation_active_rows": 0.0,
            }
            permuted_context_identity_metrics = {
                "context_identity_cross_entropy": 0.0,
                "context_identity_top1_accuracy": 0.0,
                "context_identity_mean_margin": 0.0,
                "context_identity_active_rows": 0.0,
            }
            if bool(include_context_identity_diagnostics):
                # Both full forwards have already emitted their target-free
                # RADIO/ALIKE context logits. Their context branch is independent
                # of RGB density, so this joins strict train-only identities only
                # after normal and deranged visual predictions are fixed.
                identity_group = (
                    group
                    if registered_identity_groups is None
                    else registered_identity_groups.get(str(query_id))
                )
                if identity_group is None:
                    raise ValueError(
                        "registered identity target-free validation query is unresolved"
                    )
                identity_batch = registered_identity_batch_for_geometry_selection(
                    geometry_group=group,
                    registered_identity_group=identity_group,
                    point_positions=positions,
                    device=state.device,
                )
                _, context_identity_metrics = context_identity_cross_entropy_loss(
                    runtime=batch.runtime,
                    prediction=normal_prediction,
                    target_observed=identity_batch.target_observed,
                )
                _, context_identity_permutation_metrics = (
                    context_identity_support_permutation_margin_loss(
                        runtime=batch.runtime,
                        prediction=normal_prediction,
                        permuted_runtime=permuted_runtime,
                        permuted_prediction=permuted_prediction,
                        target_observed=identity_batch.target_observed,
                        margin=0.0,
                    )
                )
                _, permuted_context_identity_metrics = context_identity_cross_entropy_loss(
                    runtime=permuted_runtime,
                    prediction=permuted_prediction,
                    target_observed=identity_batch.target_observed,
                )
        values = [
            float(normal_loss.item()),
            float(normal_metrics["query_mean_correct_minus_hardest_wrong"]),
            float(normal_metrics["query_correct_win_fraction"]),
            float(permuted_loss.item()),
            float(permuted_metrics["query_mean_correct_minus_hardest_wrong"]),
            float(permuted_metrics["query_correct_win_fraction"]),
            1.0,
        ]
        if hard_repeat_enabled:
            values.extend(
                [
                    float(hard_repeat_metrics["normal_gap_sum"]),
                    float(hard_repeat_metrics["normal_win_sum"]),
                    float(hard_repeat_metrics["permuted_gap_sum"]),
                    float(hard_repeat_metrics["permuted_win_sum"]),
                    float(hard_repeat_metrics["active_edges"]),
                    float(hard_repeat_metrics["eligible_query"]),
                ]
            )
        if bool(include_context_identity_diagnostics):
            values.extend(
                [
                    float(context_identity_metrics["context_identity_cross_entropy"]),
                    float(context_identity_metrics["context_identity_top1_accuracy"]),
                    float(context_identity_metrics["context_identity_mean_margin"]),
                    float(context_identity_metrics["context_identity_active_rows"]),
                    float(
                        context_identity_permutation_metrics[
                            "context_identity_permutation_margin_loss"
                        ]
                    ),
                    float(
                        context_identity_permutation_metrics[
                            "context_identity_permutation_mean_gap"
                        ]
                    ),
                    float(
                        context_identity_permutation_metrics[
                            "context_identity_permutation_win_fraction"
                        ]
                    ),
                    float(
                        context_identity_permutation_metrics[
                            "context_identity_permutation_active_rows"
                        ]
                    ),
                    float(
                        permuted_context_identity_metrics[
                            "context_identity_cross_entropy"
                        ]
                    ),
                    float(
                        permuted_context_identity_metrics[
                            "context_identity_top1_accuracy"
                        ]
                    ),
                    float(
                        permuted_context_identity_metrics[
                            "context_identity_mean_margin"
                        ]
                    ),
                ]
            )
        totals += torch.tensor(values, dtype=torch.float64, device=state.device)
    totals = _reduce_tensor(state, totals)
    query_count = totals[6]
    if float(query_count.item()) <= 0.0:
        raise RuntimeError("RGB spatial target-free validation did not evaluate a query")
    metrics = {
        "normal_query_grouped_loss": float((totals[0] / query_count).item()),
        "normal_mean_correct_minus_hardest_wrong": float((totals[1] / query_count).item()),
        "normal_correct_win_fraction": float((totals[2] / query_count).item()),
        "permuted_query_grouped_loss": float((totals[3] / query_count).item()),
        "permuted_mean_correct_minus_hardest_wrong": float((totals[4] / query_count).item()),
        "permuted_correct_win_fraction": float((totals[5] / query_count).item()),
        "query_count": float(query_count.item()),
    }
    if hard_repeat_enabled:
        hard_offset = 7
        active_edges = totals[hard_offset + 4]
        eligible_queries = totals[hard_offset + 5]
        if float(active_edges.item()) > 0.0:
            normal_gap = float((totals[hard_offset] / active_edges).item())
            normal_win = float((totals[hard_offset + 1] / active_edges).item())
            permuted_gap = float((totals[hard_offset + 2] / active_edges).item())
            permuted_win = float((totals[hard_offset + 3] / active_edges).item())
        else:
            normal_gap = 0.0
            normal_win = 0.0
            permuted_gap = 0.0
            permuted_win = 0.0
        metrics.update(
            {
                "hard_repeat_mean_correct_minus_coherent_wrong": normal_gap,
                "hard_repeat_correct_win_fraction": normal_win,
                "hard_repeat_permuted_mean_correct_minus_coherent_wrong": permuted_gap,
                "hard_repeat_permuted_correct_win_fraction": permuted_win,
                "hard_repeat_common_active_edges": float(active_edges.item()),
                "hard_repeat_common_active_edges_per_eligible_query": float(
                    (active_edges / eligible_queries).item()
                    if float(eligible_queries.item()) > 0.0
                    else 0.0
                ),
                "hard_repeat_eligible_query_fraction": float(
                    (eligible_queries / query_count).item()
                ),
            }
        )
    if bool(include_context_identity_diagnostics):
        context_offset = context_metric_offset
        metrics.update(
            {
                "context_identity_cross_entropy": float(
                    (totals[context_offset] / query_count).item()
                ),
                "context_identity_top1_accuracy": float(
                    (totals[context_offset + 1] / query_count).item()
                ),
                "context_identity_mean_margin": float(
                    (totals[context_offset + 2] / query_count).item()
                ),
                "context_identity_active_rows_per_query": float(
                    (totals[context_offset + 3] / query_count).item()
                ),
                "context_identity_permutation_margin_loss": float(
                    (totals[context_offset + 4] / query_count).item()
                ),
                "context_identity_permutation_mean_gap": float(
                    (totals[context_offset + 5] / query_count).item()
                ),
                "context_identity_permutation_win_fraction": float(
                    (totals[context_offset + 6] / query_count).item()
                ),
                "context_identity_permutation_active_rows_per_query": float(
                    (totals[context_offset + 7] / query_count).item()
                ),
                "permuted_context_identity_cross_entropy": float(
                    (totals[context_offset + 8] / query_count).item()
                ),
                "permuted_context_identity_top1_accuracy": float(
                    (totals[context_offset + 9] / query_count).item()
                ),
                "permuted_context_identity_mean_margin": float(
                    (totals[context_offset + 10] / query_count).item()
                ),
            }
        )
    return metrics


def _source_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": file_sha256_short(path)}
        for name, path in paths.items()
    }


def _validate_training_args(args: argparse.Namespace) -> dict[str, int]:
    values = (
        float(args.context_radius_px),
        float(args.step_px),
        float(args.learning_rate),
        float(args.weight_decay),
        float(args.pose_margin),
        float(args.pose_loss_weight),
        float(args.pose_pool_soft_hard_loss_weight),
        float(args.pose_pool_soft_hard_temperature),
        float(args.pose_pool_permutation_soft_hard_loss_weight),
        float(args.pose_pool_permutation_soft_hard_margin),
        float(args.pose_pool_permutation_soft_hard_temperature),
        float(args.density_loss_weight),
        float(args.dustbin_loss_weight),
        float(args.hard_repeat_loss_weight),
        float(args.hard_repeat_margin),
        float(args.hard_repeat_context_loss_weight),
        float(args.hard_repeat_context_margin),
        float(args.mined_hard_repeat_loss_weight),
        float(args.mined_hard_repeat_margin),
        float(args.mined_hard_repeat_context_loss_weight),
        float(args.mined_hard_repeat_context_margin),
        float(args.mined_hard_pose_pool_loss_weight),
        float(args.mined_hard_pose_pool_margin),
        float(args.mined_hard_pose_pool_temperature),
        float(args.minimum_hard_repeat_eligible_query_fraction),
        float(args.minimum_hard_repeat_win_fraction),
        float(args.minimum_hard_repeat_gap),
        float(args.minimum_hard_repeat_visual_gap_delta),
        float(args.permutation_contrastive_loss_weight),
        float(args.permutation_contrastive_margin),
        float(args.identity_support_contrastive_loss_weight),
        float(args.identity_support_contrastive_margin),
        float(args.context_identity_loss_weight),
        float(args.context_identity_support_permutation_loss_weight),
        float(args.context_identity_support_permutation_margin),
        float(args.max_abs_context_log_ratio),
        float(args.max_abs_pose_log_ratio),
        float(args.missing_edge_log_likelihood_ratio),
        float(args.rgb_cache_gb),
    )
    if (
        int(args.epochs) <= 0
        or int(args.texture_feature_dim) <= 0
        or int(args.hidden_dim) < 4
        or int(args.edge_chunk_size) <= 0
        or int(args.max_points_per_query) == 1
        or int(args.validation_selector_point_budget) < 4
        or int(args.validation_selector_grid_rows) <= 0
        or int(args.validation_selector_grid_columns) <= 0
        or int(args.max_train_queries) < 0
        or int(args.max_hard_repeat_edges_per_query) < 0
        or int(args.max_mined_hard_repeat_edges_per_query) < 0
        or int(args.inner_validation_fold_count) < 2
        or int(args.permutation_control_shift) <= 0
        or not all(math.isfinite(value) for value in values)
        or float(args.context_radius_px) <= 0.0
        or float(args.step_px) <= 0.0
        or float(args.learning_rate) <= 0.0
        or float(args.weight_decay) < 0.0
        or float(args.pose_margin) < 0.0
        or float(args.pose_loss_weight) < 0.0
        or float(args.pose_pool_soft_hard_loss_weight) < 0.0
        or float(args.pose_pool_soft_hard_temperature) <= 0.0
        or float(args.pose_pool_permutation_soft_hard_loss_weight) < 0.0
        or float(args.pose_pool_permutation_soft_hard_margin) < 0.0
        or float(args.pose_pool_permutation_soft_hard_temperature) <= 0.0
        or float(args.density_loss_weight) <= 0.0
        or float(args.dustbin_loss_weight) < 0.0
        or float(args.hard_repeat_loss_weight) < 0.0
        or float(args.hard_repeat_margin) < 0.0
        or float(args.hard_repeat_context_loss_weight) < 0.0
        or float(args.hard_repeat_context_margin) < 0.0
        or float(args.mined_hard_repeat_loss_weight) < 0.0
        or float(args.mined_hard_repeat_margin) < 0.0
        or float(args.mined_hard_repeat_context_loss_weight) < 0.0
        or float(args.mined_hard_repeat_context_margin) < 0.0
        or float(args.mined_hard_pose_pool_loss_weight) < 0.0
        or float(args.mined_hard_pose_pool_margin) < 0.0
        or float(args.mined_hard_pose_pool_temperature) <= 0.0
        or not 0.0 < float(args.minimum_hard_repeat_eligible_query_fraction) <= 1.0
        or not 0.0 <= float(args.minimum_hard_repeat_win_fraction) <= 1.0
        or float(args.minimum_hard_repeat_gap) < 0.0
        or float(args.minimum_hard_repeat_visual_gap_delta) < 0.0
        or int(args.hard_repeat_warmup_epochs) < 0
        or int(args.mined_hard_repeat_warmup_epochs) < 0
        or int(args.mined_hard_pose_pool_warmup_epochs) < 0
        or float(args.permutation_contrastive_loss_weight) < 0.0
        or float(args.permutation_contrastive_margin) < 0.0
        or float(args.identity_support_contrastive_loss_weight) < 0.0
        or float(args.identity_support_contrastive_margin) < 0.0
        or float(args.context_identity_loss_weight) < 0.0
        or float(args.context_identity_support_permutation_loss_weight) < 0.0
        or float(args.context_identity_support_permutation_margin) < 0.0
        or float(args.max_abs_context_log_ratio) <= 0.0
        or float(args.max_abs_pose_log_ratio) <= 0.0
        or float(args.rgb_cache_gb) <= 0.0
    ):
        raise ValueError("RGB spatial likelihood training arguments are invalid")
    if (
        float(args.hard_repeat_loss_weight) > 0.0
        or float(args.hard_repeat_context_loss_weight) > 0.0
    ) and not str(args.hard_repeat_targets).strip():
        raise ValueError("hard-repeat loss requires a train-only hard-repeat target artifact")
    registered_identity_targets_requested = bool(
        str(args.registered_identity_targets).strip()
    )
    exact_identity_objective_requested = (
        float(args.identity_support_contrastive_loss_weight) > 0.0
        or float(args.context_identity_loss_weight) > 0.0
        or float(args.context_identity_support_permutation_loss_weight) > 0.0
    )
    if registered_identity_targets_requested and not exact_identity_objective_requested:
        raise ValueError(
            "registered identity targets require a positive exact-identity objective weight"
        )
    if bool(args.allow_registered_identity_sidecar_introduction_from_identity_free_init):
        if not registered_identity_targets_requested or not str(args.init_checkpoint).strip():
            raise ValueError(
                "registered-identity sidecar introduction requires both a separate "
                "--registered-identity-targets artifact and --init-checkpoint"
            )
        if not exact_identity_objective_requested:
            raise ValueError(
                "registered-identity sidecar introduction requires a positive "
                "exact-identity objective weight"
            )
    mined_targets_requested = bool(str(args.mined_hard_repeat_targets).strip())
    mined_direct_loss_enabled = float(args.mined_hard_repeat_loss_weight) > 0.0
    mined_context_loss_enabled = (
        float(args.mined_hard_repeat_context_loss_weight) > 0.0
    )
    mined_pose_pool_loss_enabled = float(args.mined_hard_pose_pool_loss_weight) > 0.0
    mined_loss_enabled = (
        mined_direct_loss_enabled
        or mined_context_loss_enabled
        or mined_pose_pool_loss_enabled
    )
    if mined_loss_enabled and not mined_targets_requested:
        raise ValueError("mined hard-repeat loss requires a current-system target artifact")
    if mined_targets_requested and not mined_loss_enabled:
        raise ValueError(
            "mined hard-repeat targets require a positive mined edge, context, or pose-pool loss weight"
        )
    if mined_targets_requested and not str(args.mined_registered_identity_targets).strip():
        raise ValueError(
            "mined hard-repeat targets require registered identity targets for positive-edge validation"
        )
    if (
        not mined_targets_requested
        and str(args.mined_registered_identity_targets).strip()
    ):
        raise ValueError("mined registered identity targets require mined hard-repeat targets")
    if mined_targets_requested and not str(args.hard_repeat_targets).strip():
        raise ValueError(
            "mined hard-repeat targets require fixed hard-repeat targets for the validation gate"
        )
    if mined_targets_requested and not str(args.init_checkpoint).strip():
        raise ValueError(
            "mined hard-repeat targets require the exact checkpoint used for mining as --init-checkpoint"
        )
    if bool(args.rgb_cost_volume_only) and (
        float(args.dustbin_loss_weight) != 0.0
        or float(args.hard_repeat_context_loss_weight) != 0.0
        or mined_loss_enabled
        or float(args.context_identity_loss_weight) != 0.0
        or float(args.context_identity_support_permutation_loss_weight) != 0.0
    ):
        raise ValueError(
            "RGB-only cost-volume fitting requires zero dustbin and context-only loss weights"
        )
    full_initialization_count = sum(
        bool(str(path).strip())
        for path in (
            args.init_checkpoint,
            args.observation_pretrain_checkpoint,
            args.hard_pose_pretrain_checkpoint,
        )
    )
    texture_component_paths = (
        args.texture_observation_pretrain_checkpoint,
        args.identity_llr_texture_pretrain_checkpoint,
    )
    texture_component_count = sum(
        bool(str(path).strip()) for path in texture_component_paths
    )
    hard_repeat_texture_component_present = bool(
        str(args.rgb_hard_repeat_texture_checkpoint).strip()
    )
    context_component_present = bool(
        str(args.context_observation_pretrain_checkpoint).strip()
    )
    context_pairs_present = bool(str(args.context_observation_pairs).strip())
    component_initialization_count = texture_component_count + int(context_component_present)
    if full_initialization_count > 1:
        raise ValueError(
            "P1 initialization accepts exactly one compatible checkpoint type"
        )
    if full_initialization_count > 0 and (
        component_initialization_count > 0 or hard_repeat_texture_component_present
    ):
        raise ValueError(
            "component-wise initialization cannot be mixed with a whole-model initializer"
        )
    if texture_component_count > 1:
        raise ValueError(
            "component-wise initialization accepts exactly one texture initializer"
        )
    if hard_repeat_texture_component_present and component_initialization_count > 0:
        raise ValueError(
            "hard-repeat RGB texture initialization must be used without other component initializers"
        )
    if texture_component_count > 0 and not context_component_present:
        raise ValueError(
            "a texture component initializer requires the paired context initializer"
        )
    if context_component_present and not context_pairs_present:
        raise ValueError(
            "context-observation initialization requires its source observation-pair artifact"
        )
    if context_pairs_present and not context_component_present:
        raise ValueError(
            "context-observation pairs are valid only with a context initializer"
        )
    if (
        component_initialization_count or hard_repeat_texture_component_present
    ) and bool(args.rgb_cost_volume_only):
        raise ValueError(
            "component-wise initialization requires the combined RGB/context likelihood"
        )
    if hard_repeat_texture_component_present and not str(args.hard_repeat_targets).strip():
        raise ValueError(
            "hard-repeat RGB texture initialization requires the matching train-only hard-repeat target artifact"
        )
    resolve_candidate_pose_rgb_spatial_context_encoder_arch(args.context_encoder_arch)
    return resolve_candidate_pose_rgb_spatial_context_windows(
        {
            "radio_final": int(args.radio_final_context_window),
            "radio_intermediate": int(args.radio_intermediate_context_window),
            "alike": int(args.alike_context_window),
        }
    )


def hard_repeat_loss_weight_for_epoch(
    *, base_weight: float, epoch_index: int, warmup_epochs: int
) -> float:
    """Return the scheduled hard-repeat weight for a zero-based training epoch.

    The direct repeat edge objective has a much sharper local gradient than the
    query-grouped pose objective.  Starting it at full strength can therefore
    destroy an already useful pose-ranking solution before the two objectives
    are balanced.  The schedule deliberately depends only on the epoch, never
    on validation metrics or target-bearing runtime inputs.
    """

    weight = float(base_weight)
    epoch = int(epoch_index)
    warmup = int(warmup_epochs)
    if not math.isfinite(weight) or weight < 0.0 or epoch < 0 or warmup < 0:
        raise ValueError("hard-repeat warmup arguments are invalid")
    if weight == 0.0 or warmup == 0:
        return weight
    return weight * min(1.0, float(epoch + 1) / float(warmup))


def load_target_free_initialization_checkpoint(
    *,
    path: Path,
    model: CandidatePoseRGBSpatialLikelihood,
    layout_sha256: str,
    targets_sha256: str,
    candidate_count: int,
    support_view_count: int,
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    texture_feature_dim: int,
    hidden_dim: int,
    max_abs_context_log_ratio: float,
    context_encoder_arch: str = "conv_v1",
    permitted_parent_target_sha256s: Sequence[str] = (),
    registered_identity_targets_sha256: str = "",
    allow_registered_identity_sidecar_introduction_from_identity_free_parent: bool = False,
) -> dict[str, object]:
    """Strictly initialize only from a compatible target-free model checkpoint."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RGB likelihood initialization checkpoint is absent: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("RGB likelihood initialization checkpoint is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("model_format") != CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("pose_or_ground_truth_used_by_runtime_scorer") is not False
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or int(metadata.get("fixed_candidate_top_k", -1)) != int(candidate_count)
        or int(metadata.get("fixed_support_view_count", -1)) != int(support_view_count)
    ):
        raise ValueError("RGB likelihood initialization checkpoint is not target-free compatible")
    if (
        metadata.get("train_only_inner_gate_passed") is not True
        or metadata.get("holdout_evaluation_allowed") is not True
    ):
        raise ValueError("RGB likelihood initialization checkpoint did not pass its inner gate")
    gate_evaluator_manifest = require_current_inner_gate_evaluator_manifest(
        metadata.get("inner_gate_evaluator_manifest"),
        subject="RGB likelihood initialization checkpoint",
    )
    lineage = metadata.get("lineage")
    config = metadata.get("config")
    if not isinstance(lineage, Mapping) or not isinstance(config, Mapping):
        raise ValueError("RGB likelihood initialization checkpoint lacks lineage/config")
    checkpoint_target_sha256 = str(lineage.get("training_targets_sha256", ""))
    parent_hashes = tuple(
        sorted(
            {
                str(value).strip()
                for value in permitted_parent_target_sha256s
                if str(value).strip()
            }
        )
    )
    if str(lineage.get("layout_sha256", "")) != str(layout_sha256):
        raise ValueError("RGB likelihood initialization checkpoint lineage is stale")
    if checkpoint_target_sha256 == str(targets_sha256):
        target_lineage = "exact_target_hash"
    elif checkpoint_target_sha256 in parent_hashes:
        target_lineage = "declared_train_only_parent_target_continuation"
    else:
        raise ValueError("RGB likelihood initialization checkpoint lineage is stale")
    expected_registered_identity_sha256 = str(registered_identity_targets_sha256).strip()
    checkpoint_registered_identity_sha256 = str(
        lineage.get("registered_identity_targets_sha256", "")
    ).strip()
    registered_identity_lineage_transition = "not_requested"
    if (
        expected_registered_identity_sha256
        and checkpoint_registered_identity_sha256 != expected_registered_identity_sha256
    ):
        if checkpoint_registered_identity_sha256:
            raise ValueError("RGB likelihood initialization registered-identity lineage is stale")
        if not bool(
            allow_registered_identity_sidecar_introduction_from_identity_free_parent
        ):
            raise ValueError("RGB likelihood initialization registered-identity lineage is stale")
        training = metadata.get("training")
        if not isinstance(training, Mapping):
            raise ValueError(
                "RGB likelihood initialization identity-free parent status is unproven"
            )
        supervision = training.get("registered_identity_supervision")
        legacy_identity_records = (
            "registered_candidate_context_identity",
            "registered_candidate_context_support_appearance_derangement",
            "exact_identity_support_appearance_contrastive",
        )
        if isinstance(supervision, Mapping):
            parent_identity_free = supervision.get("enabled") is False
        else:
            # Older identity-free checkpoints predate the sidecar metadata but
            # may still serialize disabled objective configurations. They are
            # eligible only when every present exact-identity record explicitly
            # says disabled; an unknown or enabled record cannot be treated as
            # evidence that the parent was label-free.
            parent_identity_free = True
            for name in legacy_identity_records:
                record = training.get(name)
                if record is None:
                    continue
                if not isinstance(record, Mapping) or record.get("enabled") is not False:
                    parent_identity_free = False
                    break
        if not parent_identity_free:
            raise ValueError(
                "RGB likelihood initialization identity-free parent status is unproven"
            )
        registered_identity_lineage_transition = (
            "explicit_sidecar_introduction_from_identity_free_parent_v1"
        )
    elif expected_registered_identity_sha256:
        registered_identity_lineage_transition = "exact_registered_identity_match_v1"
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
            raise ValueError("RGB likelihood initialization checkpoint config is incomplete") from error
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("RGB likelihood initialization checkpoint config differs")
    if (
        int(config.get("texture_feature_dim", -1)) != int(texture_feature_dim)
        or int(config.get("hidden_dim", -1)) != int(hidden_dim)
    ):
        raise ValueError("RGB likelihood initialization checkpoint model width differs")
    rgb_cost_volume_only = config.get("rgb_cost_volume_only", False)
    if not isinstance(rgb_cost_volume_only, bool):
        raise ValueError("RGB likelihood initialization checkpoint RGB-only mode is invalid")
    try:
        observed_context_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            config.get("context_encoder_arch", "conv_v1")
        )
        expected_context_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            context_encoder_arch
        )
    except (TypeError, ValueError) as error:
        raise ValueError("RGB likelihood initialization checkpoint context architecture is invalid") from error
    if observed_context_arch != expected_context_arch:
        raise ValueError("RGB likelihood initialization checkpoint context architecture differs")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("RGB likelihood initialization state dict is incompatible") from error
    return {
        "path": str(checkpoint_path),
        "sha256": file_sha256_short(checkpoint_path),
        "rgb_cost_volume_only": bool(rgb_cost_volume_only),
        "target_lineage": target_lineage,
        "checkpoint_training_targets_sha256": checkpoint_target_sha256,
        "checkpoint_registered_identity_targets_sha256": checkpoint_registered_identity_sha256,
        "registered_identity_lineage_transition": registered_identity_lineage_transition,
        "inner_gate_evaluator_manifest": gate_evaluator_manifest,
        "selected_epoch": int(
            metadata.get("training", {})
            .get("inner_validation", {})
            .get("selected_epoch", -1)
        )
        if isinstance(metadata.get("training"), Mapping)
        else -1,
    }


def load_observation_pretrain_initialization_checkpoint(
    *,
    path: Path,
    model: CandidatePoseRGBSpatialLikelihood,
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
    context_encoder_arch: str = "conv_v1",
) -> dict[str, object]:
    """Load only a gate-approved broad-observation initializer for P1.

    This is intentionally separate from :func:`load_target_free_initialization_checkpoint`.
    A broad SfM-observation pretrain has a different candidate layout from P1,
    so matching P1 layout/target hashes would be nonsensical.  It must instead
    prove that its train-only inner gate passed and that every visual source
    which defines descriptor semantics is identical to the P1 run.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"observation-pretrain initialization checkpoint is absent: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("observation-pretrain initialization checkpoint is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        payload.get("format")
        != CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
        or not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("format")
        != CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
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
        or metadata.get("p1_finetune_allowed") is not True
    ):
        raise ValueError("observation-pretrain initialization checkpoint is not eligible")
    excluded = metadata.get("encoder_excludes")
    required_excluded = {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
    }
    if (
        not isinstance(excluded, Sequence)
        or isinstance(excluded, (str, bytes))
        or not required_excluded.issubset({str(value) for value in excluded})
    ):
        raise ValueError("observation-pretrain encoder exclusion contract is incomplete")
    training = metadata.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("observation-pretrain checkpoint lacks training metadata")
    inner = training.get("inner_validation")
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or int(inner.get("selected_epoch", -1)) < 1
    ):
        raise ValueError("observation-pretrain inner gate did not pass")
    lineage = metadata.get("lineage")
    config = metadata.get("config")
    if not isinstance(lineage, Mapping) or not isinstance(config, Mapping):
        raise ValueError("observation-pretrain checkpoint lacks lineage/config")
    expected_cache_paths = {
        "radio_final": "radio_final_context_cache_sha256",
        "radio_intermediate": "radio_intermediate_context_cache_sha256",
        "alike": "alike_spatial_context_cache_sha256",
    }
    if set(source_cache_paths) != set(expected_cache_paths):
        raise ValueError("observation-pretrain source-cache set is incomplete")
    for source_name, lineage_name in expected_cache_paths.items():
        if str(lineage.get(lineage_name, "")) != file_sha256_short(
            Path(source_cache_paths[source_name])
        ):
            raise ValueError("observation-pretrain source cache lineage differs")
    if str(lineage.get("source_image_manifest_sha256", "")) != str(
        source_image_manifest_sha256
    ):
        raise ValueError("observation-pretrain image manifest lineage differs")
    try:
        serialized_bridge = json.dumps(
            lineage.get("rgb_coordinate_bridge"), sort_keys=True, separators=(",", ":")
        )
        expected_bridge = json.dumps(
            dict(rgb_coordinate_bridge), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("observation-pretrain RGB coordinate bridge is invalid") from error
    if serialized_bridge != expected_bridge:
        raise ValueError("observation-pretrain RGB coordinate bridge differs")
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
            raise ValueError("observation-pretrain checkpoint config is incomplete") from error
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("observation-pretrain checkpoint config differs")
    if (
        int(config.get("texture_feature_dim", -1)) != int(texture_feature_dim)
        or int(config.get("hidden_dim", -1)) != int(hidden_dim)
    ):
        raise ValueError("observation-pretrain checkpoint model width differs")
    rgb_cost_volume_only = config.get("rgb_cost_volume_only", False)
    if not isinstance(rgb_cost_volume_only, bool):
        raise ValueError("observation-pretrain checkpoint RGB-only mode is invalid")
    try:
        observed_windows = resolve_candidate_pose_rgb_spatial_context_windows(
            config.get("context_windows")
        )
        expected_windows = resolve_candidate_pose_rgb_spatial_context_windows(context_windows)
    except (TypeError, ValueError) as error:
        raise ValueError("observation-pretrain context-window config is invalid") from error
    if observed_windows != expected_windows:
        raise ValueError("observation-pretrain context-window config differs")
    try:
        observed_context_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            config.get("context_encoder_arch", "conv_v1")
        )
        expected_context_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            context_encoder_arch
        )
    except (TypeError, ValueError) as error:
        raise ValueError("observation-pretrain context architecture is invalid") from error
    if observed_context_arch != expected_context_arch:
        raise ValueError("observation-pretrain context architecture differs")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("observation-pretrain initialization state dict is incompatible") from error
    return {
        "kind": "gate_approved_observation_pretrain",
        "path": str(checkpoint_path),
        "sha256": file_sha256_short(checkpoint_path),
        "rgb_cost_volume_only": bool(rgb_cost_volume_only),
        "selected_epoch": int(inner["selected_epoch"]),
        "source_candidate_layout": "broad_sfm_observation_pairs_not_p1_runtime_layout_v1",
    }


def load_texture_observation_pretrain_component_checkpoint(
    *,
    path: Path,
    model: CandidatePoseRGBSpatialLikelihood,
    source_cache_paths: Mapping[str, Path],
    source_image_manifest_sha256: str,
    rgb_coordinate_bridge: Mapping[str, object],
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    texture_feature_dim: int,
    hidden_dim: int,
    max_abs_context_log_ratio: float,
) -> dict[str, object]:
    """Load a gated RGB-only pretrain into ``TexturePatchEncoder`` alone.

    A full RGB-cost-volume pretrain contains random context modules because
    those modules are structurally present but bypassed.  Loading its entire
    state into a combined likelihood would silently discard a separately
    validated RADIO/ALIKE context initializer.  This component loader instead
    validates the same target-free visual lineage and only transfers the FPN
    texture encoder whose receptive field and sampling geometry match exactly.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"texture-observation initialization checkpoint is absent: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("texture-observation initialization checkpoint is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        payload.get("format")
        != CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
        or not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("format")
        != CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
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
        or metadata.get("p1_finetune_allowed") is not True
    ):
        raise ValueError("texture-observation initialization checkpoint is not eligible")
    excluded = metadata.get("encoder_excludes")
    required_excluded = {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
    }
    if (
        not isinstance(excluded, Sequence)
        or isinstance(excluded, (str, bytes))
        or not required_excluded.issubset({str(value) for value in excluded})
    ):
        raise ValueError("texture-observation encoder exclusion contract is incomplete")
    training = metadata.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("texture-observation checkpoint lacks training metadata")
    inner = training.get("inner_validation")
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or int(inner.get("selected_epoch", -1)) < 1
    ):
        raise ValueError("texture-observation inner gate did not pass")
    lineage = metadata.get("lineage")
    config = metadata.get("config")
    if not isinstance(lineage, Mapping) or not isinstance(config, Mapping):
        raise ValueError("texture-observation checkpoint lacks lineage/config")
    expected_cache_paths = {
        "radio_final": "radio_final_context_cache_sha256",
        "radio_intermediate": "radio_intermediate_context_cache_sha256",
        "alike": "alike_spatial_context_cache_sha256",
    }
    if set(source_cache_paths) != set(expected_cache_paths):
        raise ValueError("texture-observation source-cache set is incomplete")
    for source_name, lineage_name in expected_cache_paths.items():
        if str(lineage.get(lineage_name, "")) != file_sha256_short(
            Path(source_cache_paths[source_name])
        ):
            raise ValueError("texture-observation source cache lineage differs")
    if str(lineage.get("source_image_manifest_sha256", "")) != str(
        source_image_manifest_sha256
    ):
        raise ValueError("texture-observation image manifest lineage differs")
    try:
        serialized_bridge = json.dumps(
            lineage.get("rgb_coordinate_bridge"), sort_keys=True, separators=(",", ":")
        )
        expected_bridge = json.dumps(
            dict(rgb_coordinate_bridge), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("texture-observation RGB coordinate bridge is invalid") from error
    if serialized_bridge != expected_bridge:
        raise ValueError("texture-observation RGB coordinate bridge differs")
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
            raise ValueError("texture-observation checkpoint config is incomplete") from error
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("texture-observation checkpoint config differs")
    if (
        config.get("rgb_cost_volume_only") is not True
        or int(config.get("texture_feature_dim", -1)) != int(texture_feature_dim)
        or int(config.get("hidden_dim", -1)) != int(hidden_dim)
    ):
        raise ValueError("texture-observation checkpoint model config differs")
    prefix = "texture_encoder."
    expected_texture_keys = {
        name for name in model.state_dict() if str(name).startswith(prefix)
    }
    source_texture_state = {
        str(name): value for name, value in state_dict.items() if str(name).startswith(prefix)
    }
    if set(source_texture_state) != expected_texture_keys:
        raise ValueError("texture-observation initialization state dict is incompatible")
    try:
        incompatible = model.load_state_dict(source_texture_state, strict=False)
    except RuntimeError as error:
        raise ValueError("texture-observation initialization state dict is incompatible") from error
    if incompatible.unexpected_keys or any(
        str(name).startswith(prefix) for name in incompatible.missing_keys
    ):
        raise ValueError("texture-observation initializer did not load every texture parameter")
    return {
        "kind": "gate_approved_rgb_cost_volume_observation_pretrain_texture_only",
        "path": str(checkpoint_path),
        "sha256": file_sha256_short(checkpoint_path),
        "selected_epoch": int(inner["selected_epoch"]),
        "loaded_parameter_scope": "texture_encoder_only",
        "source_candidate_layout": "broad_sfm_observation_pairs_not_p1_runtime_layout_v1",
    }


def load_hard_repeat_gated_rgb_texture_initialization_checkpoint(
    *,
    path: Path,
    model: CandidatePoseRGBSpatialLikelihood,
    layout_sha256: str,
    targets_sha256: str,
    hard_repeat_targets_sha256: str,
    candidate_count: int,
    support_view_count: int,
    source_cache_paths: Mapping[str, Path],
    source_image_manifest_sha256: str,
    rgb_coordinate_bridge: Mapping[str, object],
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    texture_feature_dim: int,
    hidden_dim: int,
    max_abs_context_log_ratio: float,
) -> dict[str, object]:
    """Transfer only a source-safe RGB FPN that passed the hard-repeat gate.

    The source P1 RGB-only checkpoint deliberately remains ineligible for
    held-out scoring because its aggregate full-pose visual permutation gate
    failed.  Its direct, common-availability hard-repeat audit nevertheless
    establishes a narrow fact: the high-resolution RGB texture encoder can
    separate the current correct local candidate from coherent repeat errors.
    This loader preserves that boundary by importing only ``texture_encoder``
    after exact P1 RGB lineage checks.  All density, dustbin, RADIO/ALIKE
    context, and fusion parameters are fresh and must pass the new combined
    gate.  In particular, the target may use a newer context architecture:
    those parameters are deliberately not transferred.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"hard-repeat RGB texture initialization checkpoint is absent: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("hard-repeat RGB texture initialization checkpoint is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("format") != CHECKPOINT_FORMAT
        or metadata.get("model_format") != CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT
        or metadata.get("architecture")
        != "real_rgb_fpn_candidate_specific_high_resolution_cost_volume_only_v1"
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("pose_or_ground_truth_used_by_runtime_scorer") is not False
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("fixed_global_topl") is not True
        or int(metadata.get("fixed_candidate_top_k", -1)) != int(candidate_count)
        or int(metadata.get("fixed_support_view_count", -1)) != int(support_view_count)
        or metadata.get("explicit_null") is not True
        or metadata.get("projection_after_network_only") is not True
        or metadata.get("out_of_window_projection_semantics")
        != "fixed_neutral_missing_edge_not_learned_dustbin_v1"
        or metadata.get("edge_source_availability") != SOURCE_SAFE_EDGE_AVAILABILITY
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("raw_scores_must_not_feed_pnp") is not True
        or metadata.get("train_only_inner_gate_passed") is not False
        or metadata.get("holdout_evaluation_allowed") is not False
    ):
        raise ValueError("hard-repeat RGB texture initialization checkpoint is not eligible")
    excluded = metadata.get("encoder_excludes")
    required_excluded = {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
    }
    if (
        not isinstance(excluded, Sequence)
        or isinstance(excluded, (str, bytes))
        or not required_excluded.issubset({str(value) for value in excluded})
    ):
        raise ValueError("hard-repeat RGB texture encoder exclusion contract is incomplete")
    training = metadata.get("training")
    inner = training.get("inner_validation") if isinstance(training, Mapping) else None
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or gate.get("hard_repeat_required") is not True
        or gate.get("hard_repeat_passed") is not True
        or gate.get("passed") is not False
        or int(inner.get("selected_epoch", -1)) < 1
    ):
        raise ValueError("hard-repeat RGB texture checkpoint did not pass its direct hard-repeat gate")
    lineage = metadata.get("lineage")
    config = metadata.get("config")
    inputs = metadata.get("inputs")
    if (
        not isinstance(lineage, Mapping)
        or not isinstance(config, Mapping)
        or not isinstance(inputs, Mapping)
    ):
        raise ValueError("hard-repeat RGB texture checkpoint lacks lineage/config")
    if (
        str(lineage.get("layout_sha256", "")) != str(layout_sha256)
        or str(lineage.get("training_targets_sha256", "")) != str(targets_sha256)
        or str(lineage.get("hard_repeat_targets_sha256", ""))
        != str(hard_repeat_targets_sha256)
    ):
        raise ValueError("hard-repeat RGB texture checkpoint lineage is stale")
    if str(lineage.get("source_image_manifest_sha256", "")) != str(
        source_image_manifest_sha256
    ):
        raise ValueError("hard-repeat RGB texture image manifest lineage differs")
    try:
        serialized_bridge = json.dumps(
            lineage.get("rgb_coordinate_bridge"), sort_keys=True, separators=(",", ":")
        )
        expected_bridge = json.dumps(
            dict(rgb_coordinate_bridge), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("hard-repeat RGB texture RGB coordinate bridge is invalid") from error
    if serialized_bridge != expected_bridge:
        raise ValueError("hard-repeat RGB texture RGB coordinate bridge differs")
    expected_cache_inputs = {
        "radio_final": "radio_final_context_cache",
        "radio_intermediate": "radio_intermediate_context_cache",
        "alike": "alike_spatial_context_cache",
    }
    if set(source_cache_paths) != set(expected_cache_inputs):
        raise ValueError("hard-repeat RGB texture source-cache set is incomplete")
    expected_input_hashes = {
        "rgb_spatial_layout": str(layout_sha256),
        "training_targets": str(targets_sha256),
        "hard_repeat_targets": str(hard_repeat_targets_sha256),
    }
    expected_input_hashes.update(
        {
            input_name: file_sha256_short(Path(source_cache_paths[source_name]))
            for source_name, input_name in expected_cache_inputs.items()
        }
    )
    for input_name, expected_hash in expected_input_hashes.items():
        observed = inputs.get(input_name)
        if (
            not isinstance(observed, Mapping)
            or str(observed.get("sha256", "")) != str(expected_hash)
        ):
            raise ValueError("hard-repeat RGB texture input lineage differs")
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
            raise ValueError("hard-repeat RGB texture checkpoint config is incomplete") from error
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("hard-repeat RGB texture checkpoint config differs")
    if (
        config.get("rgb_cost_volume_only") is not True
        or config.get("trainable_parameter_scope") != "texture_encoder_only"
        or int(config.get("texture_feature_dim", -1)) != int(texture_feature_dim)
        or int(config.get("hidden_dim", -1)) != int(hidden_dim)
    ):
        raise ValueError("hard-repeat RGB texture checkpoint model config differs")
    hard_repeat_config = config.get("hard_repeat_inner_gate")
    if (
        not isinstance(hard_repeat_config, Mapping)
        or hard_repeat_config.get("required_when_targets_loaded") is not True
        or hard_repeat_config.get("common_normal_permuted_availability_only") is not True
    ):
        raise ValueError("hard-repeat RGB texture checkpoint gate protocol is incomplete")
    prefix = "texture_encoder."
    expected_texture_keys = {
        name for name in model.state_dict() if str(name).startswith(prefix)
    }
    source_texture_state = {
        str(name): value for name, value in state_dict.items() if str(name).startswith(prefix)
    }
    if set(source_texture_state) != expected_texture_keys:
        raise ValueError("hard-repeat RGB texture initialization state dict is incompatible")
    try:
        incompatible = model.load_state_dict(source_texture_state, strict=False)
    except RuntimeError as error:
        raise ValueError("hard-repeat RGB texture initialization state dict is incompatible") from error
    if incompatible.unexpected_keys or any(
        str(name).startswith(prefix) for name in incompatible.missing_keys
    ):
        raise ValueError("hard-repeat RGB texture initializer did not load every texture parameter")
    return {
        "kind": "hard_repeat_gated_rgb_texture_component_initializer_v1",
        "path": str(checkpoint_path),
        "sha256": file_sha256_short(checkpoint_path),
        "selected_epoch": int(inner["selected_epoch"]),
        "loaded_parameter_scope": "texture_encoder_only",
        "source_checkpoint_overall_gate_passed": False,
        "source_checkpoint_holdout_evaluation_allowed": False,
        "source_hard_repeat_gate": {
            "correct_win_fraction": float(gate["hard_repeat_correct_win_fraction"]),
            "mean_correct_minus_coherent_wrong": float(
                gate["hard_repeat_mean_correct_minus_coherent_wrong"]
            ),
            "visual_gap_delta": float(gate["hard_repeat_minus_permuted_gap"]),
        },
        "source_unused_context_configuration": {
            "context_windows": config.get("context_windows"),
            "context_encoder_arch": config.get("context_encoder_arch"),
        },
        "transfer_contract": (
            "exact_current_p1_lineage_hard_repeat_gate_texture_encoder_only_v1; "
            "source_full_pose_gate_failed_and_no_source_density_dustbin_context_or_fusion_"
            "parameter_is_imported_target_context_architecture_may_differ"
        ),
    }


def load_identity_llr_texture_pretrain_initialization_checkpoint(
    *,
    path: Path,
    model: CandidatePoseRGBSpatialLikelihood,
    candidate_count: int,
    source_cache_paths: Mapping[str, Path],
    source_image_manifest_sha256: str,
    rgb_coordinate_bridge: Mapping[str, object],
    texture_feature_dim: int,
    hidden_dim: int,
) -> dict[str, object]:
    """Transfer only V5's independently gated RGB FPN into the spatial branch.

    The broad V5 identity pretrain established that its RGB expert carries
    target-free visual information under source-masked and support-permutation
    controls.  Its crop geometry is intentionally larger than this local
    density model's geometry, however.  The transfer is therefore restricted
    to the geometry-agnostic ``TexturePatchEncoder`` parameters, recorded as
    an ablation, and cannot bypass this trainer's own spatial source controls.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"identity-LLR texture initialization checkpoint is absent: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("identity-LLR texture initialization checkpoint is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        payload.get("format") != IDENTITY_LLR_TEXTURE_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
        or not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("format") != IDENTITY_LLR_TEXTURE_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
        or metadata.get("model_format") != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("pose_or_ground_truth_used_by_runtime_scorer") is not False
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("raw_scores_must_not_feed_pnp") is not True
        or metadata.get("candidate_slot_permutation_equivariant") is not True
        or metadata.get("p1_initialization_allowed") is not True
        or metadata.get("visual_evidence_gate_version")
        != IDENTITY_LLR_TEXTURE_OBSERVATION_PRETRAIN_GATE_VERSION
        or int(metadata.get("fixed_candidate_count", -1)) != int(candidate_count)
    ):
        raise ValueError("identity-LLR texture initialization checkpoint is not eligible")
    excluded = metadata.get("encoder_excludes")
    required_excluded = {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
    }
    if (
        not isinstance(excluded, Sequence)
        or isinstance(excluded, (str, bytes))
        or not required_excluded.issubset({str(value) for value in excluded})
    ):
        raise ValueError("identity-LLR texture encoder exclusion contract is incomplete")
    training = metadata.get("training")
    inner = training.get("inner_validation") if isinstance(training, Mapping) else None
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or int(inner.get("selected_epoch", -1)) < 1
    ):
        raise ValueError("identity-LLR texture inner gate did not pass")
    lineage = metadata.get("lineage")
    config = metadata.get("config")
    if not isinstance(lineage, Mapping) or not isinstance(config, Mapping):
        raise ValueError("identity-LLR texture checkpoint lacks lineage/config")
    expected_cache_paths = {
        "radio_final": "radio_final_context_cache_sha256",
        "radio_intermediate": "radio_intermediate_context_cache_sha256",
        "alike": "alike_spatial_context_cache_sha256",
    }
    if set(source_cache_paths) != set(expected_cache_paths):
        raise ValueError("identity-LLR texture source-cache set is incomplete")
    for source_name, lineage_name in expected_cache_paths.items():
        if str(lineage.get(lineage_name, "")) != file_sha256_short(
            Path(source_cache_paths[source_name])
        ):
            raise ValueError("identity-LLR texture source cache lineage differs")
    if str(lineage.get("source_image_manifest_sha256", "")) != str(
        source_image_manifest_sha256
    ):
        raise ValueError("identity-LLR texture image manifest lineage differs")
    try:
        serialized_bridge = json.dumps(
            lineage.get("rgb_coordinate_bridge"), sort_keys=True, separators=(",", ":")
        )
        expected_bridge = json.dumps(
            dict(rgb_coordinate_bridge), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("identity-LLR texture RGB coordinate bridge is invalid") from error
    if serialized_bridge != expected_bridge:
        raise ValueError("identity-LLR texture RGB coordinate bridge differs")
    try:
        source_radius = float(config["rgb_context_radius_px"])
        source_step = float(config["rgb_step_px"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("identity-LLR texture checkpoint config is incomplete") from error
    if (
        not math.isfinite(source_radius)
        or not math.isfinite(source_step)
        or source_radius <= 0.0
        or source_step <= 0.0
        or int(config.get("texture_feature_dim", -1)) != int(texture_feature_dim)
        or int(config.get("hidden_dim", -1)) != int(hidden_dim)
    ):
        raise ValueError("identity-LLR texture checkpoint model config differs")
    prefix = "texture_encoder."
    expected_texture_keys = {
        name for name in model.state_dict() if str(name).startswith(prefix)
    }
    source_texture_state = {
        str(name): value for name, value in state_dict.items() if str(name).startswith(prefix)
    }
    if set(source_texture_state) != expected_texture_keys:
        raise ValueError("identity-LLR texture initialization state dict is incompatible")
    try:
        incompatible = model.load_state_dict(source_texture_state, strict=False)
    except RuntimeError as error:
        raise ValueError(
            "identity-LLR texture initialization state dict is incompatible"
        ) from error
    if incompatible.unexpected_keys or any(
        str(name).startswith(prefix) for name in incompatible.missing_keys
    ):
        raise ValueError("identity-LLR texture initializer did not load every texture parameter")
    return {
        "kind": "gate_approved_identity_llr_texture_only_cross_patch_geometry_transfer",
        "path": str(checkpoint_path),
        "sha256": file_sha256_short(checkpoint_path),
        "selected_epoch": int(inner["selected_epoch"]),
        "loaded_parameter_scope": "texture_encoder_only",
        "source_candidate_layout": "broad_sfm_observation_pairs_not_p1_runtime_layout_v1",
        "source_visual_evidence_gate_version": IDENTITY_LLR_TEXTURE_OBSERVATION_PRETRAIN_GATE_VERSION,
        "source_rgb_patch_geometry": {
            "context_radius_px": source_radius,
            "step_px": source_step,
        },
        "transfer_contract": (
            "texture_encoder_architecture_only_cross_patch_geometry_ablation_v1; "
            "target_spatial_likelihood_requires_its_own_train_only_source_controls"
        ),
    }


def load_context_observation_pretrain_initialization_checkpoint(
    *,
    path: Path,
    model: CandidatePoseRGBSpatialLikelihood,
    source_cache_paths: Mapping[str, Path],
    source_image_manifest_sha256: str,
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    texture_feature_dim: int,
    hidden_dim: int,
    max_abs_context_log_ratio: float,
    context_windows: Mapping[str, int],
    context_encoder_arch: str,
    allow_rgb_step_mismatch: bool = False,
    allow_spatial_search_radius_mismatch: bool = False,
    observation_pairs_path: Path | None = None,
    expected_pretrain_train_query_ids: Sequence[str] | None = None,
    expected_pretrain_validation_query_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Load only a gated absolute-context L0 initializer into a P1 model.

    The observation L0 checkpoint intentionally never trains the RGB texture,
    local-density, or dustbin heads.  Loading its full state into a P1 model
    would silently replace a useful spatial branch with random frozen weights.
    This loader therefore admits only the context encoders and the context LLR
    head, after checking the frozen descriptor-source lineage exactly.  The
    optional RGB-step exception is valid only for a component-wise merge: RGB
    sampling never enters a context crop or its attention weights, while every
    RADIO/ALIKE source, crop window, architecture, and width remains exact.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"context-observation initialization checkpoint is absent: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("context-observation initialization checkpoint is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        payload.get("format")
        != CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
        or not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("format")
        != CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
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
        or metadata.get("p1_context_transfer_allowed") is not True
        or metadata.get("appearance_control_geometry_fixed") is not True
        or metadata.get("checkpoint_selection_policy")
        != _CONTEXT_OBSERVATION_FIXED_FINAL_EPOCH_SELECTION_POLICY
        or metadata.get("inner_validation_used_for_model_selection") is not False
    ):
        raise ValueError("context-observation initialization checkpoint is not eligible")
    excluded = metadata.get("encoder_excludes")
    required_excluded = {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
        "rgb_patch",
    }
    if (
        not isinstance(excluded, Sequence)
        or isinstance(excluded, (str, bytes))
        or not required_excluded.issubset({str(value) for value in excluded})
    ):
        raise ValueError("context-observation encoder exclusion contract is incomplete")
    training = metadata.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("context-observation checkpoint lacks training metadata")
    inner = training.get("inner_validation")
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    selection = inner.get("checkpoint_selection") if isinstance(inner, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or not isinstance(selection, Mapping)
        or selection.get("policy")
        != _CONTEXT_OBSERVATION_FIXED_FINAL_EPOCH_SELECTION_POLICY
        or selection.get("inner_validation_used_for_model_selection") is not False
        or int(selection.get("selected_epoch", -1)) != int(inner.get("selected_epoch", -1))
        or int(inner.get("selected_epoch", -1)) < 1
    ):
        raise ValueError("context-observation inner gate did not pass")
    lineage = metadata.get("lineage")
    config = metadata.get("config")
    if not isinstance(lineage, Mapping) or not isinstance(config, Mapping):
        raise ValueError("context-observation checkpoint lacks lineage/config")
    expected_cache_paths = {
        "radio_final": "radio_final_context_cache_sha256",
        "radio_intermediate": "radio_intermediate_context_cache_sha256",
        "alike": "alike_spatial_context_cache_sha256",
    }
    if set(source_cache_paths) != set(expected_cache_paths):
        raise ValueError("context-observation source-cache set is incomplete")
    for source_name, lineage_name in expected_cache_paths.items():
        if str(lineage.get(lineage_name, "")) != file_sha256_short(
            Path(source_cache_paths[source_name])
        ):
            raise ValueError("context-observation source cache lineage differs")
    if str(lineage.get("source_image_manifest_sha256", "")) != str(
        source_image_manifest_sha256
    ):
        raise ValueError("context-observation image manifest lineage differs")
    expected_floats = {
        "context_radius_px": float(context_radius_px),
        "max_abs_context_log_ratio": float(max_abs_context_log_ratio),
    }
    if not bool(allow_spatial_search_radius_mismatch):
        expected_floats["search_radius_px"] = float(search_radius_px)
    for name, expected in expected_floats.items():
        try:
            observed = float(config[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("context-observation checkpoint config is incomplete") from error
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("context-observation checkpoint config differs")
    try:
        observed_step_px = float(config["step_px"])
        observed_search_radius_px = float(config["search_radius_px"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("context-observation checkpoint config is incomplete") from error
    if (
        not math.isfinite(observed_step_px)
        or observed_step_px <= 0.0
        or not math.isfinite(observed_search_radius_px)
        or observed_search_radius_px <= 0.0
        or (
            not bool(allow_rgb_step_mismatch)
            and not math.isclose(observed_step_px, float(step_px), rel_tol=1e-6, abs_tol=1e-6)
        )
    ):
        raise ValueError("context-observation checkpoint config differs")
    if (
        config.get("context_only") is not True
        or int(config.get("texture_feature_dim", -1)) != int(texture_feature_dim)
        or int(config.get("hidden_dim", -1)) != int(hidden_dim)
    ):
        raise ValueError("context-observation checkpoint model config differs")
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
        raise ValueError("context-observation context configuration is invalid") from error
    if (
        observed_windows != expected_windows
        or observed_context_arch != expected_context_arch
        or observed_context_arch != "absolute_cross_attention_v3"
    ):
        raise ValueError("context-observation context configuration differs")
    query_split_lineage: dict[str, object] | None = None
    expected_train = (
        None
        if expected_pretrain_train_query_ids is None
        else tuple(sorted(set(str(value) for value in expected_pretrain_train_query_ids)))
    )
    expected_validation = (
        None
        if expected_pretrain_validation_query_ids is None
        else tuple(sorted(set(str(value) for value in expected_pretrain_validation_query_ids)))
    )
    if (expected_train is None) != (expected_validation is None):
        raise ValueError("context-observation query split expectations are incomplete")
    if expected_train is not None:
        pairs_path = None if observation_pairs_path is None else Path(observation_pairs_path)
        if pairs_path is None or not pairs_path.is_file():
            raise ValueError("context-observation initializer requires its source pair artifact")
        if str(lineage.get("observation_pairs_sha256", "")) != file_sha256_short(pairs_path):
            raise ValueError("context-observation pair artifact lineage differs")
        with np.load(pairs_path, allow_pickle=False) as pairs:
            required_pair_fields = {"query_image_ids", "split_names"}
            if not required_pair_fields.issubset(set(pairs.files)):
                raise ValueError("context-observation pair artifact lacks query split fields")
            pair_query_ids = np.asarray(pairs["query_image_ids"]).astype(str)
            pair_splits = np.asarray(pairs["split_names"]).astype(str)
        if pair_query_ids.shape != pair_splits.shape or len(pair_query_ids) == 0:
            raise ValueError("context-observation pair artifact query split fields are invalid")
        observed_train = tuple(sorted(set(pair_query_ids[pair_splits == "inner_train"].tolist())))
        observed_validation = tuple(
            sorted(set(pair_query_ids[pair_splits == "inner_validation"].tolist()))
        )
        if (
            not observed_train
            or not observed_validation
            or set(observed_train) & set(observed_validation)
            or observed_train != expected_train
            or observed_validation != expected_validation
        ):
            raise ValueError("context-observation pretrain split does not match the P1 fold")
        query_split_lineage = {
            "observation_pairs_path": str(pairs_path),
            "observation_pairs_sha256": file_sha256_short(pairs_path),
            "inner_train_query_count": len(observed_train),
            "inner_validation_query_count": len(observed_validation),
            "exact_p1_fold_match": True,
        }
    prefixes = ("context_encoders.", "context_identity_head.")
    expected_context_keys = {
        name for name in model.state_dict() if name.startswith(prefixes)
    }
    source_context_state = {
        str(name): value
        for name, value in state_dict.items()
        if str(name).startswith(prefixes)
    }
    if set(source_context_state) != expected_context_keys:
        raise ValueError("context-observation context state dict is incompatible")
    try:
        incompatible = model.load_state_dict(source_context_state, strict=False)
    except RuntimeError as error:
        raise ValueError("context-observation initialization state dict is incompatible") from error
    if incompatible.unexpected_keys or any(
        name.startswith(prefixes) for name in incompatible.missing_keys
    ):
        raise ValueError("context-observation initializer did not load every context parameter")
    return {
        "kind": "gate_approved_context_observation_pretrain_context_only",
        "path": str(checkpoint_path),
        "sha256": file_sha256_short(checkpoint_path),
        "selected_epoch": int(inner["selected_epoch"]),
        "loaded_parameter_scope": "context_encoders_and_context_identity_head_only",
        "rgb_step_px_component_invariant_mismatch": bool(
            not math.isclose(observed_step_px, float(step_px), rel_tol=1e-6, abs_tol=1e-6)
        ),
        "spatial_search_radius_component_invariant_mismatch": bool(
            not math.isclose(
                observed_search_radius_px,
                float(search_radius_px),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ),
        "source_candidate_layout": "broad_sfm_observation_pairs_not_p1_runtime_layout_v1",
        "query_split_lineage": query_split_lineage,
    }


def load_hard_pose_pretrain_initialization_checkpoint(
    *,
    path: Path,
    model: CandidatePoseRGBSpatialLikelihood,
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
    context_encoder_arch: str = "conv_v1",
    allow_diagnostic_ineligible: bool = False,
) -> dict[str, object]:
    """Load a hard-pose initializer while preserving promotion eligibility.

    The broad hard-pose pretrain deliberately has a four-way uniform candidate
    mixture rather than the P1 top-L posterior.  Candidate priors never enter
    the encoder, so this is a valid weight initializer only after the visual
    source lineage, receptive fields, target-free boundary, and hard-pose
    train-only gate all match exactly.  ``allow_diagnostic_ineligible`` is
    deliberately limited to train-only audit callers: it still validates every
    target-free, source-lineage, geometry, and architecture contract, but
    records the checkpoint as ineligible and must never be used to initialize
    a P1 training or production scoring path.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"hard-pose-pretrain initialization checkpoint is absent: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("hard-pose-pretrain initialization checkpoint is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        payload.get("format")
        != CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_CHECKPOINT_FORMAT
        or not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("format")
        != CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_CHECKPOINT_FORMAT
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
    ):
        raise ValueError("hard-pose-pretrain initialization checkpoint is not eligible")
    p1_finetune_allowed = metadata.get("p1_finetune_allowed") is True
    if not bool(allow_diagnostic_ineligible) and not p1_finetune_allowed:
        raise ValueError("hard-pose-pretrain initialization checkpoint is not eligible")
    excluded = metadata.get("encoder_excludes")
    required_excluded = {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
    }
    if (
        not isinstance(excluded, Sequence)
        or isinstance(excluded, (str, bytes))
        or not required_excluded.issubset({str(value) for value in excluded})
    ):
        raise ValueError("hard-pose-pretrain encoder exclusion contract is incomplete")
    training = metadata.get("training")
    allowed_objectives = {CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_OBJECTIVE}
    if bool(allow_diagnostic_ineligible):
        allowed_objectives.add(CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_IDENTITY_PRETRAIN_OBJECTIVE)
    if not isinstance(training, Mapping) or training.get("objective") not in allowed_objectives:
        raise ValueError("hard-pose-pretrain checkpoint objective differs")
    inner = training.get("inner_validation")
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or int(inner.get("selected_epoch", -1)) < 1
    ):
        raise ValueError("hard-pose-pretrain inner gate did not pass")
    if not bool(allow_diagnostic_ineligible) and gate.get("passed") is not True:
        raise ValueError("hard-pose-pretrain inner gate did not pass")
    lineage = metadata.get("lineage")
    config = metadata.get("config")
    if not isinstance(lineage, Mapping) or not isinstance(config, Mapping):
        raise ValueError("hard-pose-pretrain checkpoint lacks lineage/config")
    if (
        training.get("objective") == CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_IDENTITY_PRETRAIN_OBJECTIVE
        and config.get("context_only") is not True
    ):
        raise ValueError("context identity checkpoint lacks its L0 configuration")
    expected_cache_paths = {
        "radio_final": "radio_final_context_cache_sha256",
        "radio_intermediate": "radio_intermediate_context_cache_sha256",
        "alike": "alike_spatial_context_cache_sha256",
    }
    if set(source_cache_paths) != set(expected_cache_paths):
        raise ValueError("hard-pose-pretrain source-cache set is incomplete")
    for source_name, lineage_name in expected_cache_paths.items():
        if str(lineage.get(lineage_name, "")) != file_sha256_short(
            Path(source_cache_paths[source_name])
        ):
            raise ValueError("hard-pose-pretrain source cache lineage differs")
    if str(lineage.get("source_image_manifest_sha256", "")) != str(
        source_image_manifest_sha256
    ):
        raise ValueError("hard-pose-pretrain image manifest lineage differs")
    try:
        serialized_bridge = json.dumps(
            lineage.get("rgb_coordinate_bridge"), sort_keys=True, separators=(",", ":")
        )
        expected_bridge = json.dumps(
            dict(rgb_coordinate_bridge), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("hard-pose-pretrain RGB coordinate bridge is invalid") from error
    if serialized_bridge != expected_bridge:
        raise ValueError("hard-pose-pretrain RGB coordinate bridge differs")
    expected_floats = {
        "search_radius_px": float(search_radius_px),
        "context_radius_px": float(context_radius_px),
        "step_px": float(step_px),
        "max_abs_context_log_ratio": float(max_abs_context_log_ratio),
        "fixed_candidate_null_mass": float(
            CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_NULL_MASS
        ),
    }
    for name, expected in expected_floats.items():
        try:
            observed = float(config[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("hard-pose-pretrain checkpoint config is incomplete") from error
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("hard-pose-pretrain checkpoint config differs")
    if (
        int(config.get("texture_feature_dim", -1)) != int(texture_feature_dim)
        or int(config.get("hidden_dim", -1)) != int(hidden_dim)
    ):
        raise ValueError("hard-pose-pretrain checkpoint model width differs")
    try:
        observed_windows = resolve_candidate_pose_rgb_spatial_context_windows(
            config.get("context_windows")
        )
        expected_windows = resolve_candidate_pose_rgb_spatial_context_windows(context_windows)
    except (TypeError, ValueError) as error:
        raise ValueError("hard-pose-pretrain context-window config is invalid") from error
    if observed_windows != expected_windows:
        raise ValueError("hard-pose-pretrain context-window config differs")
    try:
        observed_context_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            config.get("context_encoder_arch", "conv_v1")
        )
        expected_context_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            context_encoder_arch
        )
    except (TypeError, ValueError) as error:
        raise ValueError("hard-pose-pretrain context architecture is invalid") from error
    if observed_context_arch != expected_context_arch:
        raise ValueError("hard-pose-pretrain context architecture differs")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("hard-pose-pretrain initialization state dict is incompatible") from error
    return {
        "kind": (
            "gate_approved_hard_pose_pretrain"
            if p1_finetune_allowed
            else "diagnostic_ineligible_hard_pose_pretrain"
        ),
        "path": str(checkpoint_path),
        "sha256": file_sha256_short(checkpoint_path),
        "selected_epoch": int(inner["selected_epoch"]),
        "eligible_for_p1_finetune": bool(p1_finetune_allowed),
        "diagnostic_override": bool(
            bool(allow_diagnostic_ineligible) and not p1_finetune_allowed
        ),
        "source_candidate_layout": (
            "broad_same_track_mapping_support_plus_radio_pca_ann_hard_negatives_v1"
        ),
    }


def train_candidate_pose_rgb_spatial_likelihood(args: argparse.Namespace) -> dict[str, object]:
    """Fit the train-only density/margin objective with DDP-safe query groups."""

    context_windows = _validate_training_args(args)
    context_encoder_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
        args.context_encoder_arch
    )
    state = _initialize_distributed(str(args.device))
    try:
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_pose_rgb_spatial_likelihood.pt"
        history_path = output_dir / "history.json"
        summary_path = output_dir / "summary.json"
        if state.rank == 0 and (
            checkpoint_path.exists() or history_path.exists() or summary_path.exists()
        ) and not bool(args.force):
            raise FileExistsError("refusing to overwrite RGB spatial likelihood output")
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
        target_path = Path(args.training_targets)
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        targets = load_candidate_pose_rgb_spatial_training_targets(target_path)
        validate_training_layout_and_targets(
            layout=layout, targets=targets, layout_sha256=file_sha256_short(layout_path)
        )
        target_radius = float(targets.metadata["spatial_search_radius_px"])
        spatial_class_balanced = (
            str(targets.metadata.get("spatial_class_balance", ""))
            == "per_batch_observed_dustbin_mean_v1"
        )
        primary_targets_are_registered_exact_identity = is_registered_exact_identity_targets(
            targets
        )
        full_pose_hard_target = (
            str(targets.metadata.get("full_pose_hard_target_format", ""))
            == "candidate_pose_rgb_spatial_full_pool_hard_modes_v1"
        )
        pose_pool_soft_hard_enabled = float(args.pose_pool_soft_hard_loss_weight) > 0.0
        pose_pool_permutation_soft_hard_enabled = (
            float(args.pose_pool_permutation_soft_hard_loss_weight) > 0.0
        )
        if (
            pose_pool_soft_hard_enabled or pose_pool_permutation_soft_hard_enabled
        ) and not full_pose_hard_target:
            raise ValueError(
                "full-pose soft-hard objectives require an explicit full-pose-hard target artifact"
            )
        identity_support_contrastive_enabled = (
            float(args.identity_support_contrastive_loss_weight) > 0.0
        )
        context_identity_enabled = float(args.context_identity_loss_weight) > 0.0
        context_identity_support_permutation_enabled = (
            float(args.context_identity_support_permutation_loss_weight) > 0.0
        )
        exact_identity_objective_enabled = (
            identity_support_contrastive_enabled
            or context_identity_enabled
            or context_identity_support_permutation_enabled
        )
        search_radius = target_radius if args.search_radius_px is None else float(args.search_radius_px)
        if not math.isclose(search_radius, target_radius, abs_tol=1e-5, rel_tol=1e-5):
            raise ValueError("RGB spatial search radius differs from train-only target support")
        groups = build_train_query_groups(layout=layout, targets=targets)
        registered_identity_targets_path = (
            Path(str(args.registered_identity_targets))
            if str(args.registered_identity_targets).strip()
            else None
        )
        registered_identity_targets: CandidatePoseRGBSpatialTrainingTargets | None = None
        registered_identity_groups: dict[str, TrainQueryGroup] = {}
        registered_identity_target_source = ""
        if exact_identity_objective_enabled:
            if primary_targets_are_registered_exact_identity:
                if registered_identity_targets_path is not None:
                    raise ValueError(
                        "registered identity targets must not duplicate primary registered exact-identity targets"
                    )
                registered_identity_targets = targets
                registered_identity_groups = groups
                registered_identity_target_source = "primary_training_targets"
            elif registered_identity_targets_path is not None:
                registered_identity_targets = load_candidate_pose_rgb_spatial_training_targets(
                    registered_identity_targets_path
                )
                validate_registered_identity_targets_for_geometry(
                    layout=layout,
                    geometry_targets=targets,
                    registered_identity_targets=registered_identity_targets,
                    layout_sha256=file_sha256_short(layout_path),
                )
                registered_identity_groups = build_train_query_groups(
                    layout=layout,
                    targets=registered_identity_targets,
                )
                if set(registered_identity_groups) != set(groups):
                    raise ValueError(
                        "registered identity target query universe differs from geometry targets"
                    )
                registered_identity_target_source = "separate_registered_identity_targets"
            else:
                raise ValueError(
                    "exact-identity context/support objectives require --registered-identity-targets "
                    "when --training-targets is a geometry/full-pose target artifact"
                )
            if not bool(registered_identity_targets.spatial_target_observed.any()):
                raise ValueError("exact-identity context/support objective has no exact observed target")
        registered_identity_target_mode_verified = registered_identity_targets is not None
        hard_repeat_path = (
            Path(str(args.hard_repeat_targets))
            if str(args.hard_repeat_targets).strip()
            else None
        )
        mined_hard_repeat_path = (
            Path(str(args.mined_hard_repeat_targets))
            if str(args.mined_hard_repeat_targets).strip()
            else None
        )
        mined_registered_identity_targets_path = (
            Path(str(args.mined_registered_identity_targets))
            if str(args.mined_registered_identity_targets).strip()
            else None
        )
        hard_repeat_groups: dict[str, HardRepeatQueryTargets] = {}
        if hard_repeat_path is not None:
            hard_repeat_groups = build_hard_repeat_query_targets(
                layout=layout,
                targets=targets,
                hard_repeat_targets=load_candidate_pose_rgb_spatial_hard_repeat_targets(
                    hard_repeat_path
                ),
                layout_sha256=file_sha256_short(layout_path),
                targets_sha256=file_sha256_short(target_path),
            )
        mined_hard_repeat_targets: CandidatePoseRGBSpatialHardRepeatTargets | None = None
        mined_hard_repeat_groups: dict[str, HardRepeatQueryTargets] = {}
        mined_registered_identity_targets: CandidatePoseRGBSpatialTrainingTargets | None = None
        if mined_hard_repeat_path is not None:
            mined_hard_repeat_targets = load_candidate_pose_rgb_spatial_hard_repeat_targets(
                mined_hard_repeat_path
            )
            mined_hard_repeat_groups = build_hard_repeat_query_targets(
                layout=layout,
                targets=targets,
                hard_repeat_targets=mined_hard_repeat_targets,
                layout_sha256=file_sha256_short(layout_path),
                targets_sha256=file_sha256_short(target_path),
            )
            if mined_registered_identity_targets_path is None:
                raise AssertionError("argument validation lost mined registered identity targets")
            mined_registered_identity_targets = (
                load_candidate_pose_rgb_spatial_training_targets(
                    mined_registered_identity_targets_path
                )
            )
            validate_training_layout_and_targets(
                layout=layout,
                targets=mined_registered_identity_targets,
                layout_sha256=file_sha256_short(layout_path),
            )
        hard_repeat_direct_enabled = float(args.hard_repeat_loss_weight) > 0.0
        hard_repeat_context_enabled = float(args.hard_repeat_context_loss_weight) > 0.0
        hard_repeat_enabled = hard_repeat_direct_enabled or hard_repeat_context_enabled
        mined_hard_repeat_direct_enabled = (
            float(args.mined_hard_repeat_loss_weight) > 0.0
        )
        mined_hard_repeat_context_enabled = (
            float(args.mined_hard_repeat_context_loss_weight) > 0.0
        )
        mined_hard_pose_pool_enabled = (
            float(args.mined_hard_pose_pool_loss_weight) > 0.0
        )
        mined_hard_repeat_enabled = (
            mined_hard_repeat_direct_enabled
            or mined_hard_repeat_context_enabled
            or mined_hard_pose_pool_enabled
        )
        if hard_repeat_enabled and not hard_repeat_groups:
            raise ValueError("hard-repeat targets contain no train query group")
        if mined_hard_repeat_enabled and not mined_hard_repeat_groups:
            raise ValueError("mined hard-repeat targets contain no inner-train query group")
        all_query_ids = tuple(sorted(groups))
        if int(args.max_train_queries) > 0:
            count = int(args.max_train_queries)
            if count < 2:
                raise ValueError("RGB spatial max train queries must be zero or at least two")
            all_query_ids = tuple(sorted(all_query_ids, key=_stable_query_hash)[:count])
        inner_train_ids, inner_validation_ids = _partition_train_queries_for_inner_validation(
            query_ids=all_query_ids,
            fold_count=int(args.inner_validation_fold_count),
            fold_index=int(args.inner_validation_fold_index),
        )
        mined_hard_repeat_lineage: dict[str, object] | None = None
        mined_hard_pose_pool_mode_positions: dict[str, np.ndarray] = {}
        if mined_hard_repeat_targets is not None:
            if mined_registered_identity_targets is None or mined_registered_identity_targets_path is None:
                raise AssertionError("mined identity target loading is incomplete")
            mined_hard_repeat_lineage = validate_current_system_mined_hard_repeat_targets(
                mined_targets=mined_hard_repeat_targets,
                mined_groups=mined_hard_repeat_groups,
                registered_identity_targets=mined_registered_identity_targets,
                registered_identity_targets_sha256=file_sha256_short(
                    mined_registered_identity_targets_path
                ),
                expected_partition=train_query_partition_manifest(
                    all_query_ids=all_query_ids,
                    inner_train_query_ids=inner_train_ids,
                    inner_validation_query_ids=inner_validation_ids,
                    fold_count=int(args.inner_validation_fold_count),
                    fold_index=int(args.inner_validation_fold_index),
                ),
                initialization_checkpoint_path=Path(str(args.init_checkpoint)),
            )
            if mined_hard_pose_pool_enabled:
                mined_hard_pose_pool_mode_positions = (
                    resolve_current_system_mined_top_h_mode_positions(
                        mined_targets=mined_hard_repeat_targets,
                        mined_groups=mined_hard_repeat_groups,
                        geometry_groups=groups,
                    )
                )
                if set(mined_hard_pose_pool_mode_positions) != set(inner_train_ids):
                    raise ValueError(
                        "current-system top-H pose-pool modes do not cover exactly the inner-train fold"
                    )
        rgb_cost_volume_only = bool(args.rgb_cost_volume_only)
        if rgb_cost_volume_only:
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
            raise ValueError("RGB spatial likelihood currently requires a common processed RGB size")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        # Fail before allocating the model if the real RGB root cannot be
        # related to the frozen SfM/context coordinate frame by the lineage
        # bridge declared by the RADIO-final source cache.
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
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        initialization_checkpoint_path = (
            Path(str(args.init_checkpoint)) if str(args.init_checkpoint).strip() else None
        )
        observation_pretrain_checkpoint_path = (
            Path(str(args.observation_pretrain_checkpoint))
            if str(args.observation_pretrain_checkpoint).strip()
            else None
        )
        hard_pose_pretrain_checkpoint_path = (
            Path(str(args.hard_pose_pretrain_checkpoint))
            if str(args.hard_pose_pretrain_checkpoint).strip()
            else None
        )
        texture_observation_pretrain_checkpoint_path = (
            Path(str(args.texture_observation_pretrain_checkpoint))
            if str(args.texture_observation_pretrain_checkpoint).strip()
            else None
        )
        identity_llr_texture_pretrain_checkpoint_path = (
            Path(str(args.identity_llr_texture_pretrain_checkpoint))
            if str(args.identity_llr_texture_pretrain_checkpoint).strip()
            else None
        )
        hard_repeat_rgb_texture_checkpoint_path = (
            Path(str(args.rgb_hard_repeat_texture_checkpoint))
            if str(args.rgb_hard_repeat_texture_checkpoint).strip()
            else None
        )
        context_observation_pretrain_checkpoint_path = (
            Path(str(args.context_observation_pretrain_checkpoint))
            if str(args.context_observation_pretrain_checkpoint).strip()
            else None
        )
        context_observation_pairs_path = (
            Path(str(args.context_observation_pairs))
            if str(args.context_observation_pairs).strip()
            else None
        )
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
            activation_checkpointing=True,
            context_windows=context_windows,
            context_encoder_arch=context_encoder_arch,
        )
        initialization = None
        if initialization_checkpoint_path is not None:
            initialization = load_target_free_initialization_checkpoint(
                path=initialization_checkpoint_path,
                model=model,
                layout_sha256=file_sha256_short(layout_path),
                targets_sha256=file_sha256_short(target_path),
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
                        targets_sha256=file_sha256_short(target_path),
                    )
                ),
                registered_identity_targets_sha256=(
                    ""
                    if registered_identity_targets is None
                    else (
                        file_sha256_short(target_path)
                        if registered_identity_targets_path is None
                        else file_sha256_short(registered_identity_targets_path)
                    )
                ),
                allow_registered_identity_sidecar_introduction_from_identity_free_parent=bool(
                    args.allow_registered_identity_sidecar_introduction_from_identity_free_init
                ),
            )
        elif observation_pretrain_checkpoint_path is not None:
            initialization = load_observation_pretrain_initialization_checkpoint(
                path=observation_pretrain_checkpoint_path,
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
        elif hard_pose_pretrain_checkpoint_path is not None:
            initialization = load_hard_pose_pretrain_initialization_checkpoint(
                path=hard_pose_pretrain_checkpoint_path,
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
        elif hard_repeat_rgb_texture_checkpoint_path is not None:
            if hard_repeat_path is None:
                raise AssertionError(
                    "argument validation must require hard-repeat targets for RGB texture initialization"
                )
            initialization = load_hard_repeat_gated_rgb_texture_initialization_checkpoint(
                path=hard_repeat_rgb_texture_checkpoint_path,
                model=model,
                layout_sha256=file_sha256_short(layout_path),
                targets_sha256=file_sha256_short(target_path),
                hard_repeat_targets_sha256=file_sha256_short(hard_repeat_path),
                candidate_count=int(layout.candidate_count),
                support_view_count=int(layout.support_view_count),
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
            )
        elif context_observation_pretrain_checkpoint_path is not None:
            if context_observation_pairs_path is None:
                raise AssertionError(
                    "argument validation must require context observation pairs"
                )
            texture_initialization: dict[str, object] | None = None
            if texture_observation_pretrain_checkpoint_path is not None:
                texture_initialization = load_texture_observation_pretrain_component_checkpoint(
                    path=texture_observation_pretrain_checkpoint_path,
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
                )
            elif identity_llr_texture_pretrain_checkpoint_path is not None:
                texture_initialization = (
                    load_identity_llr_texture_pretrain_initialization_checkpoint(
                        path=identity_llr_texture_pretrain_checkpoint_path,
                        model=model,
                        candidate_count=int(layout.candidate_count),
                        source_cache_paths=context_cache_paths,
                        source_image_manifest_sha256=str(
                            source_metadata.get("source_image_manifest_sha256", "")
                        ),
                        rgb_coordinate_bridge=rgb_bridge,
                        texture_feature_dim=int(args.texture_feature_dim),
                        hidden_dim=int(args.hidden_dim),
                    )
                )
            context_initialization = load_context_observation_pretrain_initialization_checkpoint(
                path=context_observation_pretrain_checkpoint_path,
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
                allow_rgb_step_mismatch=True,
                allow_spatial_search_radius_mismatch=True,
                observation_pairs_path=context_observation_pairs_path,
                expected_pretrain_train_query_ids=inner_train_ids,
                expected_pretrain_validation_query_ids=inner_validation_ids,
            )
            if texture_initialization is None:
                initialization = {
                    "kind": "gate_approved_context_observation_pretrain_only",
                    "context": context_initialization,
                    "merge_contract": (
                        "context_only_initializer_leaves_rgb_texture_density_and_dustbin_"
                        "parameters_random_v1"
                    ),
                }
            else:
                initialization = {
                    "kind": "component_wise_texture_and_context_observation_pretrains",
                    "texture": texture_initialization,
                    "context": context_initialization,
                    "merge_contract": (
                        "texture_pretrain_is_exact_rgb_sampling_geometry_or_explicit_cross_geometry_"
                        "architecture_only_ablation; "
                        "context_pretrain_allows_only_rgb_step_px_and_spatial_search_radius_"
                        "mismatch because both are outside_frozen_radio_alike_context_encoder_inputs_v2"
                    ),
                }
        if (
            initialization is not None
            and "rgb_cost_volume_only" in initialization
            and bool(initialization["rgb_cost_volume_only"]) != rgb_cost_volume_only
        ):
            raise ValueError("RGB-only P1 mode differs from its initializer checkpoint")
        if rgb_cost_volume_only:
            trainable_parameter_names = configure_rgb_cost_volume_only_trainable_parameters(
                model
            )
        else:
            trainable_parameter_names = tuple(
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            )
        model = model.to(state.device)
        if state.enabled:
            model_for_train: torch.nn.Module = DistributedDataParallel(
                model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
                broadcast_buffers=False,
            )
        else:
            model_for_train = model
        core_model = model_for_train.module if state.enabled else model_for_train
        assert isinstance(core_model, CandidatePoseRGBSpatialLikelihood)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model_for_train.parameters() if parameter.requires_grad],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        cache_bytes = int(float(args.rgb_cache_gb) * 1024**3)
        image_cache = TensorImageLRUCache(
            max_bytes=cache_bytes,
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        permutation_contrastive_enabled = float(args.permutation_contrastive_loss_weight) > 0.0
        full_support_permutation_training_enabled = (
            permutation_contrastive_enabled
            or pose_pool_permutation_soft_hard_enabled
            or identity_support_contrastive_enabled
        )
        support_permutation_training_enabled = (
            full_support_permutation_training_enabled
            or context_identity_support_permutation_enabled
        )
        steps_per_rank = int(math.ceil(len(inner_train_ids) / state.world_size))
        final_metrics: dict[str, float] | None = None
        final_state_dict: dict[str, torch.Tensor] | None = None
        final_epoch = -1
        history: list[dict[str, object]] = []
        start_time = time.time()
        for epoch in range(int(args.epochs)):
            model_for_train.train()
            effective_hard_repeat_loss_weight = hard_repeat_loss_weight_for_epoch(
                base_weight=float(args.hard_repeat_loss_weight),
                epoch_index=int(epoch),
                warmup_epochs=int(args.hard_repeat_warmup_epochs),
            )
            effective_hard_repeat_context_loss_weight = hard_repeat_loss_weight_for_epoch(
                base_weight=float(args.hard_repeat_context_loss_weight),
                epoch_index=int(epoch),
                warmup_epochs=int(args.hard_repeat_warmup_epochs),
            )
            effective_mined_hard_repeat_loss_weight = hard_repeat_loss_weight_for_epoch(
                base_weight=float(args.mined_hard_repeat_loss_weight),
                epoch_index=int(epoch),
                warmup_epochs=int(args.mined_hard_repeat_warmup_epochs),
            )
            effective_mined_hard_repeat_context_loss_weight = hard_repeat_loss_weight_for_epoch(
                base_weight=float(args.mined_hard_repeat_context_loss_weight),
                epoch_index=int(epoch),
                warmup_epochs=int(args.mined_hard_repeat_warmup_epochs),
            )
            effective_mined_hard_pose_pool_loss_weight = hard_repeat_loss_weight_for_epoch(
                base_weight=float(args.mined_hard_pose_pool_loss_weight),
                epoch_index=int(epoch),
                warmup_epochs=int(args.mined_hard_pose_pool_warmup_epochs),
            )
            order = np.random.default_rng(int(args.seed) + epoch).permutation(len(inner_train_ids))
            local_ids = tuple(
                inner_train_ids[int(order[(state.rank + step * state.world_size) % len(order)])]
                for step in range(steps_per_rank)
            )
            totals = torch.zeros((61,), dtype=torch.float64, device=state.device)
            epoch_start = time.time()
            for local_step, query_id in enumerate(local_ids):
                group = groups[str(query_id)]
                registered_identity_group: TrainQueryGroup | None = None
                required_source_lists: list[np.ndarray] = []
                if exact_identity_objective_enabled:
                    try:
                        registered_identity_group = registered_identity_groups[str(query_id)]
                    except KeyError as error:
                        raise RuntimeError(
                            "registered identity target lost a geometry training query"
                        ) from error
                    required_source_lists.append(
                        registered_identity_observed_source_point_ids(
                            registered_identity_group
                        )
                    )
                if hard_repeat_enabled and str(query_id) in hard_repeat_groups:
                    required_source_lists.append(
                        hard_repeat_groups[str(query_id)].source_point_ids
                    )
                if mined_hard_repeat_enabled and str(query_id) in mined_hard_repeat_groups:
                    required_source_lists.append(
                        mined_hard_repeat_groups[str(query_id)].source_point_ids
                    )
                required_source_point_ids = (
                    None
                    if not required_source_lists
                    else np.unique(np.concatenate(required_source_lists)).astype(np.int64)
                )
                positions = _select_group_points(
                    group=group,
                    max_points=int(args.max_points_per_query),
                    seed=int(args.seed) + epoch * 100003 + local_step,
                    required_source_point_ids=required_source_point_ids,
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
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    radius_px=search_radius + float(args.context_radius_px),
                    step_px=float(args.step_px),
                    cache=image_cache,
                    device=state.device,
                )
                permuted_runtime: CandidatePoseRGBSpatialRuntime | None = None
                permuted_support_patches: torch.Tensor | None = None
                if support_permutation_training_enabled:
                    shift = train_support_permutation_shift(
                        runtime=batch.runtime,
                        query_id=str(query_id),
                        epoch=int(epoch),
                        reserved_control_shift=int(args.permutation_control_shift),
                    )
                    permuted_runtime = permute_runtime_support_image_appearance_only(
                        batch.runtime, shift=shift
                    )
                    if full_support_permutation_training_enabled:
                        permuted_support_patches = (
                            _crop_geometry_fixed_permuted_support_patches(
                                normal_query_patches=query_patches,
                                permuted_runtime=permuted_runtime,
                                image_ids=image_ids,
                                image_root=Path(args.image_root),
                                coordinate_image_size=coordinate_image_size,
                                rgb_image_size=rgb_image_size,
                                radius_px=search_radius + float(args.context_radius_px),
                                step_px=float(args.step_px),
                                cache=image_cache,
                                device=state.device,
                            )
                        )
                hard_repeat_batch = (
                    _hard_repeat_batch_from_group(
                        hard_targets=hard_repeat_groups[str(query_id)],
                        group=group,
                        point_positions=positions,
                        device=state.device,
                        max_edges=int(args.max_hard_repeat_edges_per_query),
                        seed=int(args.seed) + epoch * 100003 + local_step,
                    )
                    if hard_repeat_enabled and str(query_id) in hard_repeat_groups
                    else None
                )
                mined_hard_repeat_batch = (
                    _hard_repeat_batch_from_group(
                        hard_targets=mined_hard_repeat_groups[str(query_id)],
                        group=group,
                        point_positions=positions,
                        device=state.device,
                        max_edges=int(args.max_mined_hard_repeat_edges_per_query),
                        seed=int(args.seed) + epoch * 100003 + local_step + 7919,
                    )
                    if mined_hard_repeat_enabled
                    and str(query_id) in mined_hard_repeat_groups
                    else None
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    prediction = model_for_train(
                        runtime=batch.runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=support_patches,
                        rgb_cost_volume_only=rgb_cost_volume_only,
                    )
                    registered_identity_batch: RegisteredIdentityBatch | None = None
                    if exact_identity_objective_enabled:
                        if registered_identity_group is None:
                            raise RuntimeError(
                                "registered identity target group was not initialized"
                            )
                        # The visual forward above receives only frozen layout,
                        # context crops, and real RGB. Exact identity labels join
                        # only after it has emitted target-free edge evidence.
                        registered_identity_batch = (
                            registered_identity_batch_for_geometry_selection(
                                geometry_group=group,
                                registered_identity_group=registered_identity_group,
                                point_positions=positions,
                                device=state.device,
                            )
                        )
                    density_target_dustbin = (
                        torch.zeros_like(batch.spatial_target_dustbin)
                        if rgb_cost_volume_only
                        else batch.spatial_target_dustbin
                    )
                    density_target_supervised = (
                        batch.spatial_target_supervised & batch.spatial_target_observed
                        if rgb_cost_volume_only
                        else batch.spatial_target_supervised
                    )
                    density_loss, density_metrics = spatial_density_nll(
                        prediction=prediction,
                        target_offsets_xy=batch.spatial_target_offsets_xy,
                        target_dustbin=density_target_dustbin,
                        target_supervised=density_target_supervised,
                        dustbin_weight=float(args.dustbin_loss_weight),
                        balance_observed_and_dustbin=spatial_class_balanced,
                    )
                    correct_scores = score_candidate_pose_rgb_spatial_batch(
                        runtime=batch.runtime,
                        prediction=prediction,
                        candidate_projection_offsets_xy=batch.correct_projection_offsets_xy.unsqueeze(0),
                        candidate_projection_valid=batch.correct_projection_valid.unsqueeze(0),
                        missing_edge_log_likelihood_ratio=float(args.missing_edge_log_likelihood_ratio),
                        max_abs_log_likelihood_ratio=float(args.max_abs_pose_log_ratio),
                    ).pose_log_likelihood_ratios
                    wrong_scores = score_candidate_pose_rgb_spatial_batch(
                        runtime=batch.runtime,
                        prediction=prediction,
                        candidate_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                        candidate_projection_valid=batch.wrong_projection_valid,
                        missing_edge_log_likelihood_ratio=float(args.missing_edge_log_likelihood_ratio),
                        max_abs_log_likelihood_ratio=float(args.max_abs_pose_log_ratio),
                    ).pose_log_likelihood_ratios
                    pose_loss, pose_metrics = query_grouped_pose_margin_loss(
                        correct_scores=correct_scores,
                        coherent_wrong_scores=wrong_scores.reshape(1, -1),
                        margin=float(args.pose_margin),
                    )
                    pose_pool_soft_hard_loss = pose_loss * 0.0
                    pose_pool_soft_hard_metrics = {
                        "query_mean_correct_minus_soft_hard_wrong": 0.0,
                        "query_soft_hard_correct_win_fraction": 0.0,
                        "query_soft_hard_effective_wrong_mode_count": 0.0,
                    }
                    if pose_pool_soft_hard_enabled:
                        (
                            pose_pool_soft_hard_loss,
                            pose_pool_soft_hard_metrics,
                        ) = query_grouped_pose_soft_hard_margin_loss(
                            correct_scores=correct_scores,
                            coherent_wrong_scores=wrong_scores.reshape(1, -1),
                            margin=float(args.pose_margin),
                            temperature=float(args.pose_pool_soft_hard_temperature),
                        )
                    mined_hard_pose_pool_loss = pose_loss * 0.0
                    mined_hard_pose_pool_metrics = {
                        "query_mean_correct_minus_soft_hard_wrong": 0.0,
                        "query_soft_hard_correct_win_fraction": 0.0,
                        "query_soft_hard_effective_wrong_mode_count": 0.0,
                    }
                    if mined_hard_pose_pool_enabled:
                        try:
                            mined_mode_positions = mined_hard_pose_pool_mode_positions[
                                str(query_id)
                            ]
                        except KeyError as error:
                            raise RuntimeError(
                                "current-system top-H pose-pool mode is absent from an inner-train query"
                            ) from error
                        mode_indices = torch.from_numpy(
                            np.asarray(mined_mode_positions, dtype=np.int64)
                        ).to(device=state.device, dtype=torch.long)
                        if (
                            mode_indices.ndim != 1
                            or len(mode_indices) == 0
                            or torch.any(mode_indices < 0)
                            or torch.any(mode_indices >= len(wrong_scores))
                        ):
                            raise RuntimeError("current-system top-H pose-pool mode indices are invalid")
                        (
                            mined_hard_pose_pool_loss,
                            mined_hard_pose_pool_metrics,
                        ) = query_grouped_pose_soft_hard_margin_loss(
                            correct_scores=correct_scores,
                            coherent_wrong_scores=wrong_scores.reshape(1, -1).index_select(
                                1, mode_indices
                            ),
                            margin=float(args.mined_hard_pose_pool_margin),
                            temperature=float(args.mined_hard_pose_pool_temperature),
                        )
                    context_identity_loss = (
                        prediction.context_log_likelihood_ratios.sum() * 0.0
                    )
                    context_identity_metrics = {
                        "context_identity_active_rows": 0.0,
                        "context_identity_top1_accuracy": 0.0,
                        "context_identity_mean_margin": 0.0,
                        "context_identity_cross_entropy": 0.0,
                    }
                    if context_identity_enabled:
                        if registered_identity_batch is None:
                            raise RuntimeError("registered identity labels were not joined")
                        # The target-free full edge prediction is emitted before the
                        # separate train-only registered-track labels are joined here.
                        context_identity_loss, context_identity_metrics = (
                            context_identity_cross_entropy_loss(
                                runtime=batch.runtime,
                                prediction=prediction,
                                target_observed=registered_identity_batch.target_observed,
                            )
                        )
                    hard_repeat_loss = torch.zeros((), dtype=pose_loss.dtype, device=state.device)
                    hard_repeat_metrics = {
                        "hard_repeat_active_edges": 0.0,
                        "hard_repeat_correct_win_fraction": 0.0,
                        "hard_repeat_mean_positive_minus_negative": 0.0,
                        "hard_repeat_margin_loss": 0.0,
                    }
                    if hard_repeat_direct_enabled:
                        hard_repeat_loss, hard_repeat_metrics = coherent_hard_repeat_edge_margin_loss(
                            runtime=batch.runtime,
                            prediction=prediction,
                            hard_batch=hard_repeat_batch,
                            margin=float(args.hard_repeat_margin),
                            missing_edge_log_likelihood_ratio=float(
                                args.missing_edge_log_likelihood_ratio
                            ),
                            max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                        )
                    mined_hard_repeat_loss = torch.zeros(
                        (), dtype=pose_loss.dtype, device=state.device
                    )
                    mined_hard_repeat_metrics = {
                        "mined_hard_repeat_active_edges": 0.0,
                        "mined_hard_repeat_correct_win_fraction": 0.0,
                        "mined_hard_repeat_mean_positive_minus_negative": 0.0,
                        "mined_hard_repeat_margin_loss": 0.0,
                    }
                    if mined_hard_repeat_direct_enabled:
                        mined_hard_repeat_loss, raw_mined_hard_repeat_metrics = (
                            coherent_hard_repeat_edge_margin_loss(
                                runtime=batch.runtime,
                                prediction=prediction,
                                hard_batch=mined_hard_repeat_batch,
                                margin=float(args.mined_hard_repeat_margin),
                                missing_edge_log_likelihood_ratio=float(
                                    args.missing_edge_log_likelihood_ratio
                                ),
                                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                            )
                        )
                        mined_hard_repeat_metrics = {
                            "mined_hard_repeat_active_edges": float(
                                raw_mined_hard_repeat_metrics["hard_repeat_active_edges"]
                            ),
                            "mined_hard_repeat_correct_win_fraction": float(
                                raw_mined_hard_repeat_metrics[
                                    "hard_repeat_correct_win_fraction"
                                ]
                            ),
                            "mined_hard_repeat_mean_positive_minus_negative": float(
                                raw_mined_hard_repeat_metrics[
                                    "hard_repeat_mean_positive_minus_negative"
                                ]
                            ),
                            "mined_hard_repeat_margin_loss": float(
                                raw_mined_hard_repeat_metrics["hard_repeat_margin_loss"]
                            ),
                        }
                    mined_hard_repeat_context_loss = torch.zeros(
                        (), dtype=pose_loss.dtype, device=state.device
                    )
                    mined_hard_repeat_context_metrics = {
                        "mined_hard_repeat_context_active_edges": 0.0,
                        "mined_hard_repeat_context_correct_win_fraction": 0.0,
                        "mined_hard_repeat_context_mean_positive_minus_negative": 0.0,
                        "mined_hard_repeat_context_margin_loss": 0.0,
                    }
                    if mined_hard_repeat_context_enabled:
                        (
                            mined_hard_repeat_context_loss,
                            raw_mined_hard_repeat_context_metrics,
                        ) = coherent_hard_repeat_context_margin_loss(
                            runtime=batch.runtime,
                            prediction=prediction,
                            hard_batch=mined_hard_repeat_batch,
                            margin=float(args.mined_hard_repeat_context_margin),
                            missing_edge_log_likelihood_ratio=float(
                                args.missing_edge_log_likelihood_ratio
                            ),
                        )
                        mined_hard_repeat_context_metrics = {
                            "mined_hard_repeat_context_active_edges": float(
                                raw_mined_hard_repeat_context_metrics[
                                    "hard_repeat_context_active_edges"
                                ]
                            ),
                            "mined_hard_repeat_context_correct_win_fraction": float(
                                raw_mined_hard_repeat_context_metrics[
                                    "hard_repeat_context_correct_win_fraction"
                                ]
                            ),
                            "mined_hard_repeat_context_mean_positive_minus_negative": float(
                                raw_mined_hard_repeat_context_metrics[
                                    "hard_repeat_context_mean_positive_minus_negative"
                                ]
                            ),
                            "mined_hard_repeat_context_margin_loss": float(
                                raw_mined_hard_repeat_context_metrics[
                                    "hard_repeat_context_margin_loss"
                                ]
                            ),
                        }
                    hard_repeat_context_loss = torch.zeros(
                        (), dtype=pose_loss.dtype, device=state.device
                    )
                    hard_repeat_context_metrics = {
                        "hard_repeat_context_active_edges": 0.0,
                        "hard_repeat_context_correct_win_fraction": 0.0,
                        "hard_repeat_context_mean_positive_minus_negative": 0.0,
                        "hard_repeat_context_margin_loss": 0.0,
                    }
                    if hard_repeat_context_enabled:
                        hard_repeat_context_loss, hard_repeat_context_metrics = (
                            coherent_hard_repeat_context_margin_loss(
                                runtime=batch.runtime,
                                prediction=prediction,
                                hard_batch=hard_repeat_batch,
                                margin=float(args.hard_repeat_context_margin),
                                missing_edge_log_likelihood_ratio=float(
                                    args.missing_edge_log_likelihood_ratio
                                ),
                            )
                        )
                    permutation_loss = torch.zeros((), dtype=pose_loss.dtype, device=state.device)
                    permutation_metrics = {
                        "support_permutation_normal_gap": 0.0,
                        "support_permutation_permuted_gap": 0.0,
                        "support_permutation_gap_delta": 0.0,
                        "support_permutation_margin_loss": 0.0,
                    }
                    pose_pool_permutation_soft_hard_loss = torch.zeros(
                        (), dtype=pose_loss.dtype, device=state.device
                    )
                    pose_pool_permutation_soft_hard_metrics = {
                        "normal_mean_correct_minus_soft_hard_wrong": 0.0,
                        "permuted_mean_correct_minus_soft_hard_wrong": 0.0,
                        "normal_minus_permuted_soft_hard_gap": 0.0,
                        "normal_soft_hard_effective_wrong_mode_count": 0.0,
                        "permuted_soft_hard_effective_wrong_mode_count": 0.0,
                        "permutation_soft_hard_margin_loss": 0.0,
                    }
                    identity_support_loss = torch.zeros(
                        (), dtype=pose_loss.dtype, device=state.device
                    )
                    identity_support_metrics = {
                        "identity_support_active_edges": 0.0,
                        "identity_support_normal_mean_llr": 0.0,
                        "identity_support_permuted_mean_llr": 0.0,
                        "identity_support_mean_gap": 0.0,
                        "identity_support_normal_win_fraction": 0.0,
                        "identity_support_margin_loss": 0.0,
                    }
                    if permuted_runtime is not None and permuted_support_patches is not None:
                        permuted_prediction = model_for_train(
                            runtime=permuted_runtime,
                            query_rgb_patches=query_patches,
                            support_rgb_patches=permuted_support_patches,
                            rgb_cost_volume_only=rgb_cost_volume_only,
                        )
                        _assert_geometry_fixed_support_image_control(
                            runtime=batch.runtime,
                            permuted_runtime=permuted_runtime,
                            normal_prediction=prediction,
                            permuted_prediction=permuted_prediction,
                        )
                        if (
                            permutation_contrastive_enabled
                            or pose_pool_permutation_soft_hard_enabled
                        ):
                            permuted_correct_scores = score_candidate_pose_rgb_spatial_batch(
                                runtime=permuted_runtime,
                                prediction=permuted_prediction,
                                candidate_projection_offsets_xy=batch.correct_projection_offsets_xy.unsqueeze(0),
                                candidate_projection_valid=batch.correct_projection_valid.unsqueeze(0),
                                missing_edge_log_likelihood_ratio=float(args.missing_edge_log_likelihood_ratio),
                                max_abs_log_likelihood_ratio=float(args.max_abs_pose_log_ratio),
                            ).pose_log_likelihood_ratios
                            permuted_wrong_scores = score_candidate_pose_rgb_spatial_batch(
                                runtime=permuted_runtime,
                                prediction=permuted_prediction,
                                candidate_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                                candidate_projection_valid=batch.wrong_projection_valid,
                                missing_edge_log_likelihood_ratio=float(args.missing_edge_log_likelihood_ratio),
                                max_abs_log_likelihood_ratio=float(args.max_abs_pose_log_ratio),
                            ).pose_log_likelihood_ratios
                            if permutation_contrastive_enabled:
                                permutation_loss, permutation_metrics = (
                                    support_permutation_contrastive_loss(
                                        normal_correct_scores=correct_scores,
                                        normal_wrong_scores=wrong_scores.reshape(1, -1),
                                        permuted_correct_scores=permuted_correct_scores,
                                        permuted_wrong_scores=permuted_wrong_scores.reshape(1, -1),
                                        margin=float(args.permutation_contrastive_margin),
                                    )
                                )
                            if pose_pool_permutation_soft_hard_enabled:
                                (
                                    pose_pool_permutation_soft_hard_loss,
                                    pose_pool_permutation_soft_hard_metrics,
                                ) = query_grouped_pose_permutation_soft_hard_margin_loss(
                                    normal_correct_scores=correct_scores,
                                    normal_wrong_scores=wrong_scores.reshape(1, -1),
                                    permuted_correct_scores=permuted_correct_scores,
                                    permuted_wrong_scores=permuted_wrong_scores.reshape(1, -1),
                                    margin=float(
                                        args.pose_pool_permutation_soft_hard_margin
                                    ),
                                    temperature=float(
                                        args.pose_pool_permutation_soft_hard_temperature
                                    ),
                                )
                        if identity_support_contrastive_enabled:
                            if registered_identity_batch is None:
                                raise RuntimeError("registered identity labels were not joined")
                            identity_support_loss, identity_support_metrics = (
                                exact_identity_support_appearance_contrastive_loss(
                                    runtime=batch.runtime,
                                    prediction=prediction,
                                    permuted_runtime=permuted_runtime,
                                    permuted_prediction=permuted_prediction,
                                    target_offsets_xy=registered_identity_batch.target_offsets_xy,
                                    target_observed=registered_identity_batch.target_observed,
                                    margin=float(args.identity_support_contrastive_margin),
                                    missing_edge_log_likelihood_ratio=float(
                                        args.missing_edge_log_likelihood_ratio
                                    ),
                                    max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                                )
                            )
                    context_identity_support_permutation_loss = (
                        prediction.context_log_likelihood_ratios.sum() * 0.0
                    )
                    context_identity_support_permutation_metrics = {
                        "context_identity_permutation_active_rows": 0.0,
                        "context_identity_permutation_mean_gap": 0.0,
                        "context_identity_permutation_win_fraction": 0.0,
                        "context_identity_permutation_margin_loss": 0.0,
                    }
                    if context_identity_support_permutation_enabled:
                        if permuted_runtime is None or registered_identity_batch is None:
                            raise RuntimeError(
                                "context identity support permutation or labels were not initialized"
                            )
                        # This independent L0 forward deliberately omits RGB so a
                        # support derangement cannot be satisfied by the local
                        # density branch.  Geometry, prior, and candidate slots are
                        # invariant across the two target-free predictions.
                        normal_context_prediction = model_for_train(
                            runtime=batch.runtime,
                            context_only=True,
                        )
                        permuted_context_prediction = model_for_train(
                            runtime=permuted_runtime,
                            context_only=True,
                        )
                        _assert_geometry_fixed_support_image_control(
                            runtime=batch.runtime,
                            permuted_runtime=permuted_runtime,
                            normal_prediction=normal_context_prediction,
                            permuted_prediction=permuted_context_prediction,
                        )
                        (
                            context_identity_support_permutation_loss,
                            context_identity_support_permutation_metrics,
                        ) = context_identity_support_permutation_margin_loss(
                            runtime=batch.runtime,
                            prediction=normal_context_prediction,
                            permuted_runtime=permuted_runtime,
                            permuted_prediction=permuted_context_prediction,
                            target_observed=registered_identity_batch.target_observed,
                            margin=float(args.context_identity_support_permutation_margin),
                        )
                    loss = (
                        float(args.pose_loss_weight) * pose_loss
                        + float(args.pose_pool_soft_hard_loss_weight)
                        * pose_pool_soft_hard_loss
                        + float(args.pose_pool_permutation_soft_hard_loss_weight)
                        * pose_pool_permutation_soft_hard_loss
                        + float(args.density_loss_weight) * density_loss
                        + effective_hard_repeat_loss_weight * hard_repeat_loss
                        + effective_mined_hard_repeat_loss_weight * mined_hard_repeat_loss
                        + effective_mined_hard_repeat_context_loss_weight
                        * mined_hard_repeat_context_loss
                        + effective_mined_hard_pose_pool_loss_weight
                        * mined_hard_pose_pool_loss
                        + effective_hard_repeat_context_loss_weight * hard_repeat_context_loss
                        + float(args.permutation_contrastive_loss_weight) * permutation_loss
                        + float(args.identity_support_contrastive_loss_weight)
                        * identity_support_loss
                        + float(args.context_identity_loss_weight) * context_identity_loss
                        + float(args.context_identity_support_permutation_loss_weight)
                        * context_identity_support_permutation_loss
                    )
                scaler.scale(loss).backward()
                if float(args.gradient_clip_norm) > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model_for_train.parameters(), float(args.gradient_clip_norm)
                    )
                scaler.step(optimizer)
                scaler.update()
                totals += torch.tensor(
                    [
                        float(loss.detach().item()),
                        float(pose_loss.detach().item()),
                        float(density_loss.detach().item()),
                        float(pose_metrics["query_mean_correct_minus_hardest_wrong"]),
                        float(pose_metrics["query_correct_win_fraction"]),
                        float(density_metrics["spatial_density_active_edges"]),
                        float(permutation_metrics["support_permutation_margin_loss"]),
                        float(permutation_metrics["support_permutation_permuted_gap"]),
                        float(permutation_metrics["support_permutation_gap_delta"]),
                        1.0 if permutation_contrastive_enabled else 0.0,
                        float(hard_repeat_metrics["hard_repeat_margin_loss"]),
                        float(hard_repeat_metrics["hard_repeat_mean_positive_minus_negative"]),
                        float(hard_repeat_metrics["hard_repeat_correct_win_fraction"]),
                        float(hard_repeat_metrics["hard_repeat_active_edges"]),
                        float(effective_hard_repeat_loss_weight),
                        float(identity_support_metrics["identity_support_margin_loss"]),
                        float(identity_support_metrics["identity_support_normal_mean_llr"]),
                        float(identity_support_metrics["identity_support_permuted_mean_llr"]),
                        float(identity_support_metrics["identity_support_mean_gap"]),
                        float(identity_support_metrics["identity_support_normal_win_fraction"]),
                        float(identity_support_metrics["identity_support_active_edges"]),
                        1.0 if identity_support_contrastive_enabled else 0.0,
                        float(hard_repeat_context_metrics["hard_repeat_context_margin_loss"]),
                        float(
                            hard_repeat_context_metrics[
                                "hard_repeat_context_mean_positive_minus_negative"
                            ]
                        ),
                        float(
                            hard_repeat_context_metrics[
                                "hard_repeat_context_correct_win_fraction"
                            ]
                        ),
                        float(hard_repeat_context_metrics["hard_repeat_context_active_edges"]),
                        float(effective_hard_repeat_context_loss_weight),
                        1.0 if hard_repeat_context_enabled else 0.0,
                        float(context_identity_metrics["context_identity_cross_entropy"]),
                        float(context_identity_metrics["context_identity_top1_accuracy"]),
                        float(context_identity_metrics["context_identity_mean_margin"]),
                        float(context_identity_metrics["context_identity_active_rows"]),
                        float(
                            context_identity_support_permutation_metrics[
                                "context_identity_permutation_margin_loss"
                            ]
                        ),
                        float(
                            context_identity_support_permutation_metrics[
                                "context_identity_permutation_mean_gap"
                            ]
                        ),
                        float(
                            context_identity_support_permutation_metrics[
                                "context_identity_permutation_win_fraction"
                            ]
                        ),
                        float(
                            context_identity_support_permutation_metrics[
                                "context_identity_permutation_active_rows"
                            ]
                        ),
                        float(pose_pool_soft_hard_loss.detach().item()),
                        float(
                            pose_pool_soft_hard_metrics[
                                "query_mean_correct_minus_soft_hard_wrong"
                            ]
                        ),
                        float(
                            pose_pool_soft_hard_metrics[
                                "query_soft_hard_effective_wrong_mode_count"
                            ]
                        ),
                        float(pose_pool_permutation_soft_hard_loss.detach().item()),
                        float(
                            pose_pool_permutation_soft_hard_metrics[
                                "normal_mean_correct_minus_soft_hard_wrong"
                            ]
                        ),
                        float(
                            pose_pool_permutation_soft_hard_metrics[
                                "permuted_mean_correct_minus_soft_hard_wrong"
                            ]
                        ),
                        float(
                            pose_pool_permutation_soft_hard_metrics[
                                "normal_minus_permuted_soft_hard_gap"
                            ]
                        ),
                        float(
                            pose_pool_permutation_soft_hard_metrics[
                                "normal_soft_hard_effective_wrong_mode_count"
                            ]
                        ),
                        float(
                            pose_pool_permutation_soft_hard_metrics[
                                "permuted_soft_hard_effective_wrong_mode_count"
                            ]
                        ),
                        1.0 if pose_pool_permutation_soft_hard_enabled else 0.0,
                        float(
                            mined_hard_repeat_metrics["mined_hard_repeat_margin_loss"]
                        ),
                        float(
                            mined_hard_repeat_metrics[
                                "mined_hard_repeat_mean_positive_minus_negative"
                            ]
                        ),
                        float(
                            mined_hard_repeat_metrics[
                                "mined_hard_repeat_correct_win_fraction"
                            ]
                        ),
                        float(
                            mined_hard_repeat_metrics["mined_hard_repeat_active_edges"]
                        ),
                        float(effective_mined_hard_repeat_loss_weight),
                        float(
                            mined_hard_repeat_context_metrics[
                                "mined_hard_repeat_context_margin_loss"
                            ]
                        ),
                        float(
                            mined_hard_repeat_context_metrics[
                                "mined_hard_repeat_context_mean_positive_minus_negative"
                            ]
                        ),
                        float(
                            mined_hard_repeat_context_metrics[
                                "mined_hard_repeat_context_correct_win_fraction"
                            ]
                        ),
                        float(
                            mined_hard_repeat_context_metrics[
                                "mined_hard_repeat_context_active_edges"
                            ]
                        ),
                        float(effective_mined_hard_repeat_context_loss_weight),
                        1.0 if mined_hard_repeat_context_enabled else 0.0,
                        float(mined_hard_pose_pool_loss.detach().item()),
                        float(
                            mined_hard_pose_pool_metrics[
                                "query_mean_correct_minus_soft_hard_wrong"
                            ]
                        ),
                        float(
                            mined_hard_pose_pool_metrics[
                                "query_soft_hard_effective_wrong_mode_count"
                            ]
                        ),
                        float(effective_mined_hard_pose_pool_loss_weight),
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            totals = _reduce_tensor(state, totals)
            global_steps = float(steps_per_rank * state.world_size)
            epoch_metrics: dict[str, object] = {
                "epoch": int(epoch + 1),
                "train_total_loss": float((totals[0] / global_steps).item()),
                "train_pose_margin_loss": float((totals[1] / global_steps).item()),
                "train_spatial_density_loss": float((totals[2] / global_steps).item()),
                "train_mean_correct_minus_hardest_wrong": float((totals[3] / global_steps).item()),
                "train_correct_win_fraction": float((totals[4] / global_steps).item()),
                "train_spatial_active_edges_per_step": float((totals[5] / global_steps).item()),
                "train_support_permutation_margin_loss": float((totals[6] / global_steps).item()),
                "train_support_permutation_permuted_gap": float((totals[7] / global_steps).item()),
                "train_support_permutation_gap_delta": float((totals[8] / global_steps).item()),
                "train_support_permutation_enabled": bool(float(totals[9].item()) > 0.0),
                "train_hard_repeat_margin_loss": float((totals[10] / global_steps).item()),
                "train_hard_repeat_mean_positive_minus_negative": float(
                    (totals[11] / global_steps).item()
                ),
                "train_hard_repeat_correct_win_fraction": float((totals[12] / global_steps).item()),
                "train_hard_repeat_active_edges_per_step": float(
                    (totals[13] / global_steps).item()
                ),
                "train_hard_repeat_loss_weight": float((totals[14] / global_steps).item()),
                "train_hard_repeat_enabled": bool(hard_repeat_direct_enabled),
                "train_identity_support_margin_loss": float(
                    (totals[15] / global_steps).item()
                ),
                "train_identity_support_normal_mean_llr": float(
                    (totals[16] / global_steps).item()
                ),
                "train_identity_support_permuted_mean_llr": float(
                    (totals[17] / global_steps).item()
                ),
                "train_identity_support_mean_gap": float(
                    (totals[18] / global_steps).item()
                ),
                "train_identity_support_normal_win_fraction": float(
                    (totals[19] / global_steps).item()
                ),
                "train_identity_support_active_edges_per_step": float(
                    (totals[20] / global_steps).item()
                ),
                "train_identity_support_enabled": bool(float(totals[21].item()) > 0.0),
                "train_hard_repeat_context_margin_loss": float(
                    (totals[22] / global_steps).item()
                ),
                "train_hard_repeat_context_mean_positive_minus_negative": float(
                    (totals[23] / global_steps).item()
                ),
                "train_hard_repeat_context_correct_win_fraction": float(
                    (totals[24] / global_steps).item()
                ),
                "train_hard_repeat_context_active_edges_per_step": float(
                    (totals[25] / global_steps).item()
                ),
                "train_hard_repeat_context_loss_weight": float(
                    (totals[26] / global_steps).item()
                ),
                "train_hard_repeat_context_enabled": bool(float(totals[27].item()) > 0.0),
                "train_context_identity_cross_entropy": float(
                    (totals[28] / global_steps).item()
                ),
                "train_context_identity_top1_accuracy": float(
                    (totals[29] / global_steps).item()
                ),
                "train_context_identity_mean_margin": float(
                    (totals[30] / global_steps).item()
                ),
                "train_context_identity_active_rows_per_step": float(
                    (totals[31] / global_steps).item()
                ),
                "train_context_identity_enabled": bool(context_identity_enabled),
                "train_context_identity_support_permutation_margin_loss": float(
                    (totals[32] / global_steps).item()
                ),
                "train_context_identity_support_permutation_mean_gap": float(
                    (totals[33] / global_steps).item()
                ),
                "train_context_identity_support_permutation_win_fraction": float(
                    (totals[34] / global_steps).item()
                ),
                "train_context_identity_support_permutation_active_rows_per_step": float(
                    (totals[35] / global_steps).item()
                ),
                "train_context_identity_support_permutation_enabled": bool(
                    context_identity_support_permutation_enabled
                ),
                "train_pose_pool_soft_hard_loss": float(
                    (totals[36] / global_steps).item()
                ),
                "train_mean_correct_minus_soft_hard_wrong": float(
                    (totals[37] / global_steps).item()
                ),
                "train_pose_pool_soft_hard_effective_wrong_mode_count": float(
                    (totals[38] / global_steps).item()
                ),
                "train_pose_pool_soft_hard_enabled": bool(pose_pool_soft_hard_enabled),
                "train_pose_pool_permutation_soft_hard_loss": float(
                    (totals[39] / global_steps).item()
                ),
                "train_pose_pool_permutation_soft_hard_normal_gap": float(
                    (totals[40] / global_steps).item()
                ),
                "train_pose_pool_permutation_soft_hard_permuted_gap": float(
                    (totals[41] / global_steps).item()
                ),
                "train_pose_pool_permutation_soft_hard_gap_delta": float(
                    (totals[42] / global_steps).item()
                ),
                "train_pose_pool_permutation_soft_hard_normal_effective_wrong_mode_count": float(
                    (totals[43] / global_steps).item()
                ),
                "train_pose_pool_permutation_soft_hard_permuted_effective_wrong_mode_count": float(
                    (totals[44] / global_steps).item()
                ),
                "train_pose_pool_permutation_soft_hard_enabled": bool(
                    float(totals[45].item()) > 0.0
                ),
                "train_mined_hard_repeat_margin_loss": float(
                    (totals[46] / global_steps).item()
                ),
                "train_mined_hard_repeat_mean_positive_minus_negative": float(
                    (totals[47] / global_steps).item()
                ),
                "train_mined_hard_repeat_correct_win_fraction": float(
                    (totals[48] / global_steps).item()
                ),
                "train_mined_hard_repeat_active_edges_per_step": float(
                    (totals[49] / global_steps).item()
                ),
                "train_mined_hard_repeat_loss_weight": float(
                    (totals[50] / global_steps).item()
                ),
                "train_mined_hard_repeat_enabled": bool(mined_hard_repeat_enabled),
                "train_mined_hard_repeat_context_margin_loss": float(
                    (totals[51] / global_steps).item()
                ),
                "train_mined_hard_repeat_context_mean_positive_minus_negative": float(
                    (totals[52] / global_steps).item()
                ),
                "train_mined_hard_repeat_context_correct_win_fraction": float(
                    (totals[53] / global_steps).item()
                ),
                "train_mined_hard_repeat_context_active_edges_per_step": float(
                    (totals[54] / global_steps).item()
                ),
                "train_mined_hard_repeat_context_loss_weight": float(
                    (totals[55] / global_steps).item()
                ),
                "train_mined_hard_repeat_context_enabled": bool(
                    mined_hard_repeat_context_enabled
                ),
                "train_mined_hard_pose_pool_loss": float(
                    (totals[57] / global_steps).item()
                ),
                "train_mined_hard_pose_pool_mean_correct_minus_soft_hard_wrong": float(
                    (totals[58] / global_steps).item()
                ),
                "train_mined_hard_pose_pool_effective_wrong_mode_count": float(
                    (totals[59] / global_steps).item()
                ),
                "train_mined_hard_pose_pool_loss_weight": float(
                    (totals[60] / global_steps).item()
                ),
                "train_mined_hard_pose_pool_enabled": bool(
                    mined_hard_pose_pool_enabled
                ),
                "global_query_steps": int(global_steps),
                "epoch_seconds": float(time.time() - epoch_start),
            }
            inner_metrics = _evaluate_inner_validation_target_free_static_selector(
                model=model_for_train,
                layout=layout,
                groups=groups,
                complete_runtime=complete_runtime,
                query_ids=inner_validation_ids,
                image_ids=image_ids,
                image_root=Path(args.image_root),
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=search_radius + float(args.context_radius_px),
                step_px=float(args.step_px),
                cache=image_cache,
                state=state,
                selector_policy=str(args.validation_selector_policy),
                selector_point_budget=int(args.validation_selector_point_budget),
                selector_grid_rows=int(args.validation_selector_grid_rows),
                selector_grid_columns=int(args.validation_selector_grid_columns),
                pose_margin=float(args.pose_margin),
                missing_edge_log_likelihood_ratio=float(args.missing_edge_log_likelihood_ratio),
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                amp_enabled=amp_enabled,
                permutation_control_shift=int(args.permutation_control_shift),
                seed=int(args.seed),
                rgb_cost_volume_only=rgb_cost_volume_only,
                hard_repeat_by_query=(
                    hard_repeat_groups if hard_repeat_path is not None else None
                ),
                registered_identity_groups=(
                    registered_identity_groups
                    if exact_identity_objective_enabled
                    else None
                ),
                max_hard_repeat_edges_per_query=int(args.max_hard_repeat_edges_per_query),
                include_context_identity_diagnostics=bool(
                    context_identity_enabled
                    or context_identity_support_permutation_enabled
                ),
            )
            epoch_metrics.update({f"inner_{key}": value for key, value in inner_metrics.items()})
            if state.rank == 0:
                candidate_gate = hard_repeat_training_gate_decision(
                    inner_metrics,
                    minimum_win_fraction=float(args.minimum_win_fraction),
                    minimum_normal_gap=float(args.minimum_normal_gap),
                    minimum_visual_gap_delta=float(args.minimum_visual_gap_delta),
                    require_hard_repeat=hard_repeat_path is not None,
                    minimum_hard_repeat_eligible_query_fraction=float(
                        args.minimum_hard_repeat_eligible_query_fraction
                    ),
                    minimum_hard_repeat_win_fraction=float(
                        args.minimum_hard_repeat_win_fraction
                    ),
                    minimum_hard_repeat_gap=float(args.minimum_hard_repeat_gap),
                    minimum_hard_repeat_visual_gap_delta=float(
                        args.minimum_hard_repeat_visual_gap_delta
                    ),
                )
                epoch_metrics["inner_gate_passed"] = bool(candidate_gate["passed"])
                # The inner fold is a gate only. Selecting an epoch from it
                # would turn this train-only audit into implicit model tuning.
                final_metrics = dict(inner_metrics)
                final_epoch = int(epoch + 1)
                final_state_dict = _model_state_cpu(core_model)
                history.append(epoch_metrics)
                print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
            if state.enabled:
                distributed.barrier()
        if state.rank == 0:
            if (
                final_state_dict is None
                or final_metrics is None
                or final_epoch != int(args.epochs)
            ):
                raise RuntimeError("RGB spatial likelihood did not retain its final epoch")
            gate = hard_repeat_training_gate_decision(
                final_metrics,
                minimum_win_fraction=float(args.minimum_win_fraction),
                minimum_normal_gap=float(args.minimum_normal_gap),
                minimum_visual_gap_delta=float(args.minimum_visual_gap_delta),
                require_hard_repeat=hard_repeat_path is not None,
                minimum_hard_repeat_eligible_query_fraction=float(
                    args.minimum_hard_repeat_eligible_query_fraction
                ),
                minimum_hard_repeat_win_fraction=float(
                    args.minimum_hard_repeat_win_fraction),
                minimum_hard_repeat_gap=float(args.minimum_hard_repeat_gap),
                minimum_hard_repeat_visual_gap_delta=float(
                    args.minimum_hard_repeat_visual_gap_delta
                ),
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            input_paths = {
                "rgb_spatial_layout": layout_path,
                "training_targets": target_path,
                "radio_final_context_cache": Path(args.radio_final_context_cache),
                "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
                "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
            }
            if registered_identity_targets_path is not None:
                input_paths["registered_identity_targets"] = registered_identity_targets_path
            if hard_repeat_path is not None:
                input_paths["hard_repeat_targets"] = hard_repeat_path
            if mined_hard_repeat_path is not None:
                input_paths["mined_hard_repeat_targets"] = mined_hard_repeat_path
            if mined_registered_identity_targets_path is not None:
                input_paths[
                    "mined_registered_identity_targets"
                ] = mined_registered_identity_targets_path
            if initialization_checkpoint_path is not None:
                input_paths["initialization_checkpoint"] = initialization_checkpoint_path
            if observation_pretrain_checkpoint_path is not None:
                input_paths["observation_pretrain_checkpoint"] = observation_pretrain_checkpoint_path
            if hard_pose_pretrain_checkpoint_path is not None:
                input_paths["hard_pose_pretrain_checkpoint"] = hard_pose_pretrain_checkpoint_path
            if texture_observation_pretrain_checkpoint_path is not None:
                input_paths[
                    "texture_observation_pretrain_checkpoint"
                ] = texture_observation_pretrain_checkpoint_path
            if identity_llr_texture_pretrain_checkpoint_path is not None:
                input_paths[
                    "identity_llr_texture_pretrain_checkpoint"
                ] = identity_llr_texture_pretrain_checkpoint_path
            if hard_repeat_rgb_texture_checkpoint_path is not None:
                input_paths[
                    "rgb_hard_repeat_texture_checkpoint"
                ] = hard_repeat_rgb_texture_checkpoint_path
            if context_observation_pretrain_checkpoint_path is not None:
                input_paths[
                    "context_observation_pretrain_checkpoint"
                ] = context_observation_pretrain_checkpoint_path
            if context_observation_pairs_path is not None:
                input_paths["context_observation_pairs"] = context_observation_pairs_path
            metadata = {
                "format": CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
                "architecture": (
                    "real_rgb_fpn_candidate_specific_high_resolution_cost_volume_only_v1"
                    if rgb_cost_volume_only
                    else "full_2d_radio_final_intermediate_and_alike_context_plus_high_resolution_real_rgb_texture_cost_volume_v1"
                ),
                "contains_target_fields": False,
                "checkpoint_contains_train_targets": False,
                "runtime_layout_is_target_free": True,
                "pose_or_ground_truth_used_by_runtime_scorer": False,
                "render": False,
                "image_retrieval_or_submap_used": False,
                "fixed_global_topl": True,
                "fixed_candidate_top_k": int(layout.candidate_count),
                "fixed_support_view_count": int(layout.support_view_count),
                "explicit_null": True,
                "projection_after_network_only": True,
                "out_of_window_projection_semantics": "fixed_neutral_missing_edge_not_learned_dustbin_v1",
                "edge_source_availability": dict(SOURCE_SAFE_EDGE_AVAILABILITY),
                "candidate_reselection_per_pose": False,
                "support_reselection_per_pose": False,
                "diagnostic_only": True,
                "promotion_allowed": False,
                "raw_scores_must_not_feed_pnp": True,
                "appearance_control_geometry_fixed": True,
                "checkpoint_selection_policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
                "inner_validation_used_for_model_selection": False,
                "inner_gate_evaluator_manifest": current_inner_gate_evaluator_manifest(),
                "train_only_inner_gate_passed": bool(gate["passed"]),
                "holdout_evaluation_allowed": bool(gate["passed"]),
                "encoder_inputs": (
                    [
                        "frozen_query_anchor_xy",
                        "fixed_support_observation_xy",
                        "real_rgb_query_and_fixed_support_patches",
                    ]
                    if rgb_cost_volume_only
                    else [
                        "frozen_query_anchor_xy",
                        "fixed_support_observation_xy",
                        "full_2d_radio_final_context_crop",
                        "full_2d_radio_intermediate_context_crop",
                        "full_2d_alike_context_crop",
                        "real_rgb_query_and_fixed_support_patches",
                    ]
                ),
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
                    "search_radius_px": float(search_radius),
                    "context_radius_px": float(args.context_radius_px),
                    "step_px": float(args.step_px),
                    "texture_feature_dim": int(args.texture_feature_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "max_abs_context_log_ratio": float(args.max_abs_context_log_ratio),
                    "max_abs_pose_log_ratio": float(args.max_abs_pose_log_ratio),
                    "edge_chunk_size": int(args.edge_chunk_size),
                    "rgb_cache_dtype": str(args.rgb_cache_dtype),
                    "permutation_control_shift": int(args.permutation_control_shift),
                    "hard_repeat_max_edges_per_query": int(
                        args.max_hard_repeat_edges_per_query
                    ),
                    "hard_repeat_warmup_epochs": int(args.hard_repeat_warmup_epochs),
                    "hard_repeat_context_margin": float(args.hard_repeat_context_margin),
                    "mined_hard_repeat_max_edges_per_query": int(
                        args.max_mined_hard_repeat_edges_per_query
                    ),
                    "mined_hard_repeat_warmup_epochs": int(
                        args.mined_hard_repeat_warmup_epochs
                    ),
                    "mined_hard_repeat_margin": float(args.mined_hard_repeat_margin),
                    "mined_hard_repeat_context_margin": float(
                        args.mined_hard_repeat_context_margin
                    ),
                    "mined_hard_pose_pool_warmup_epochs": int(
                        args.mined_hard_pose_pool_warmup_epochs
                    ),
                    "mined_hard_pose_pool_margin": float(
                        args.mined_hard_pose_pool_margin
                    ),
                    "mined_hard_pose_pool_temperature": float(
                        args.mined_hard_pose_pool_temperature
                    ),
                    "hard_repeat_inner_gate": {
                        "required_when_targets_loaded": hard_repeat_path is not None,
                        "common_normal_permuted_availability_only": True,
                        "minimum_eligible_query_fraction": float(
                            args.minimum_hard_repeat_eligible_query_fraction
                        ),
                        "minimum_win_fraction": float(args.minimum_hard_repeat_win_fraction),
                        "minimum_gap": float(args.minimum_hard_repeat_gap),
                        "minimum_visual_gap_delta": float(
                            args.minimum_hard_repeat_visual_gap_delta
                        ),
                    },
                    "pose_pool_soft_hard_temperature": float(
                        args.pose_pool_soft_hard_temperature
                    ),
                    "pose_pool_permutation_soft_hard_margin": float(
                        args.pose_pool_permutation_soft_hard_margin
                    ),
                    "pose_pool_permutation_soft_hard_temperature": float(
                        args.pose_pool_permutation_soft_hard_temperature
                    ),
                    "identity_support_contrastive_margin": float(
                        args.identity_support_contrastive_margin
                    ),
                    "context_identity_support_permutation_margin": float(
                        args.context_identity_support_permutation_margin
                    ),
                    "rgb_cost_volume_only": rgb_cost_volume_only,
                    "trainable_parameter_scope": (
                        "texture_encoder_only" if rgb_cost_volume_only else "all_candidate_likelihood_parameters"
                    ),
                    "trainable_parameter_count": int(len(trainable_parameter_names)),
                    "context_windows": dict(context_windows),
                    "context_encoder_arch": context_encoder_arch,
                    "context_identity_head_final_initialization": {
                        "weight": "normal_zero_mean_v1",
                        "weight_std": float(CONTEXT_IDENTITY_HEAD_FINAL_WEIGHT_STD),
                        "bias": "zero",
                        "reason": (
                            "near_neutral_context_llr_with_nonzero_first_step_encoder_"
                            "gradient_v1"
                        ),
                    },
                    "validation_selector": {
                        "policy": str(args.validation_selector_policy),
                        "point_budget": int(args.validation_selector_point_budget),
                        "grid_rows": int(args.validation_selector_grid_rows),
                        "grid_columns": int(args.validation_selector_grid_columns),
                        "target_free_static_only": True,
                    },
                },
                "training": {
                    "objective": (
                        (
                            "observed_rgb_spatial_density_nll_plus_query_grouped_correct_vs_hardest_coherent_wrong_margin_plus_direct_repeat_and_support_appearance_controls_rgb_cost_volume_only_v1"
                            if rgb_cost_volume_only
                            else "correct_pose_spatial_density_nll_plus_query_grouped_hardest_coherent_wrong_margin_plus_optional_direct_and_context_only_coherent_hard_repeat_margins_plus_exact_registered_candidate_context_cross_entropy_and_independent_context_support_appearance_derangement_v6"
                        )
                        + (
                            "_plus_normalized_full_pose_pool_soft_hard_margin_v1"
                            if pose_pool_soft_hard_enabled
                            else ""
                        )
                        + (
                            "_plus_real_vs_deranged_full_pose_pool_soft_hard_margin_v1"
                            if pose_pool_permutation_soft_hard_enabled
                            else ""
                        )
                        + (
                            "_plus_current_system_mined_inner_train_hard_repeat_margin_v1"
                            if mined_hard_repeat_direct_enabled
                            else ""
                        )
                        + (
                            "_plus_current_system_mined_inner_train_context_only_hard_repeat_margin_v1"
                            if mined_hard_repeat_context_enabled
                            else ""
                        )
                        + (
                            "_plus_current_system_mined_target_free_top_h_pose_pool_soft_hard_margin_v1"
                            if mined_hard_pose_pool_enabled
                            else ""
                        )
                    ),
                    "epochs": int(args.epochs),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "pose_margin": float(args.pose_margin),
                    "pose_loss_weight": float(args.pose_loss_weight),
                    "full_pose_pool_soft_hard": {
                        "enabled": bool(pose_pool_soft_hard_enabled),
                        "loss_weight": float(args.pose_pool_soft_hard_loss_weight),
                        "margin": float(args.pose_margin),
                        "temperature": float(args.pose_pool_soft_hard_temperature),
                        "aggregation": "normalized_logmeanexp_all_train_only_wrong_modes_v1",
                        "requires_full_pose_hard_target": True,
                    },
                    "full_pose_pool_support_permutation_soft_hard": {
                        "enabled": bool(pose_pool_permutation_soft_hard_enabled),
                        "loss_weight": float(
                            args.pose_pool_permutation_soft_hard_loss_weight
                        ),
                        "margin": float(args.pose_pool_permutation_soft_hard_margin),
                        "temperature": float(
                            args.pose_pool_permutation_soft_hard_temperature
                        ),
                        "aggregation": "normalized_logmeanexp_all_train_only_wrong_modes_v1",
                        "normal_and_deranged_compare": (
                            "same_frozen_query_candidates_priors_and_train_only_pose_"
                            "projections_real_vs_deranged_support_appearance_v1"
                        ),
                        "requires_full_pose_hard_target": True,
                    },
                    "density_loss_weight": float(args.density_loss_weight),
                    "dustbin_loss_weight": float(args.dustbin_loss_weight),
                    "spatial_supervision": {
                        "mode": str(
                            targets.metadata.get("spatial_supervision_mode", "geometry_projected")
                        ),
                        "semantics": str(targets.metadata.get("spatial_target_semantics", "")),
                        "class_balanced": bool(spatial_class_balanced),
                        "supervised_candidate_edge_count": int(
                            np.count_nonzero(targets.spatial_target_supervised)
                        ),
                        "observed_candidate_edge_count": int(
                            np.count_nonzero(targets.spatial_target_observed)
                        ),
                        "dustbin_targets_ignored_by_rgb_cost_volume_only": rgb_cost_volume_only,
                    },
                    "registered_identity_supervision": {
                        "enabled": bool(exact_identity_objective_enabled),
                        "target_source": registered_identity_target_source,
                        "target_mode_verified": bool(registered_identity_target_mode_verified),
                        "join_key": "query_id_plus_source_point_id_v1",
                        "joined_after_normal_target_free_visual_forward": True,
                        "used_for": [
                            "context_candidate_cross_entropy",
                            "fixed_offset_support_appearance_derangement",
                        ],
                        "excluded_from": [
                            "spatial_density_nll",
                            "correct_vs_coherent_wrong_pose_margin",
                        ],
                        "registered_identity_radius_px": (
                            None
                            if registered_identity_targets is None
                            else float(
                                registered_identity_targets.metadata[
                                    "registered_identity_radius_px"
                                ]
                            )
                        ),
                        "observed_candidate_edge_count": (
                            0
                            if registered_identity_targets is None
                            else int(
                                np.count_nonzero(
                                    registered_identity_targets.spatial_target_observed
                                )
                            )
                        ),
                        "initialization_parent_identity_lineage": (
                            "no_rgb_likelihood_parent"
                            if initialization is None
                            else str(
                                initialization.get(
                                    "registered_identity_lineage_transition",
                                    "not_applicable",
                                )
                            )
                        ),
                    },
                    "coherent_hard_repeat_edge": {
                        "enabled": bool(hard_repeat_direct_enabled),
                        "loss_weight": float(args.hard_repeat_loss_weight),
                        "warmup_epochs": int(args.hard_repeat_warmup_epochs),
                        "margin": float(args.hard_repeat_margin),
                        "max_edges_per_query": int(args.max_hard_repeat_edges_per_query),
                        "targets_loaded_train_only": hard_repeat_path is not None,
                        "selection": "correct_local_candidate_vs_distinct_coherent_wrong_local_candidate_v1",
                    },
                    "coherent_hard_repeat_context": {
                        "enabled": bool(hard_repeat_context_enabled),
                        "loss_weight": float(args.hard_repeat_context_loss_weight),
                        "warmup_epochs": int(args.hard_repeat_warmup_epochs),
                        "margin": float(args.hard_repeat_context_margin),
                        "targets_loaded_train_only": hard_repeat_path is not None,
                        "encoder_evidence": (
                            "full_2d_radio_final_intermediate_and_alike_context_only_no_rgb_density_v1"
                        ),
                        "selection": "same_registered_exact_track_vs_distinct_coherent_wrong_candidate_v1",
                    },
                    "coherent_current_system_mined_hard_repeat_edge": {
                        "enabled": bool(mined_hard_repeat_direct_enabled),
                        "loss_weight": float(args.mined_hard_repeat_loss_weight),
                        "warmup_epochs": int(args.mined_hard_repeat_warmup_epochs),
                        "margin": float(args.mined_hard_repeat_margin),
                        "max_edges_per_query": int(
                            args.max_mined_hard_repeat_edges_per_query
                        ),
                        "targets_loaded_train_only": mined_hard_repeat_path is not None,
                        "selection_scope": "current_model_inner_train_only_v1",
                        "validation_usage": "forbidden_static_hard_repeat_gate_only_v1",
                        "lineage": mined_hard_repeat_lineage,
                    },
                    "coherent_current_system_mined_hard_repeat_context": {
                        "enabled": bool(mined_hard_repeat_context_enabled),
                        "loss_weight": float(
                            args.mined_hard_repeat_context_loss_weight
                        ),
                        "warmup_epochs": int(args.mined_hard_repeat_warmup_epochs),
                        "margin": float(args.mined_hard_repeat_context_margin),
                        "max_edges_per_query": int(
                            args.max_mined_hard_repeat_edges_per_query
                        ),
                        "targets_loaded_train_only": mined_hard_repeat_path is not None,
                        "encoder_evidence": (
                            "full_2d_radio_final_intermediate_and_alike_context_only_no_rgb_density_v1"
                        ),
                        "selection_scope": "current_model_inner_train_only_v1",
                        "validation_usage": "forbidden_static_hard_repeat_gate_only_v1",
                        "lineage": mined_hard_repeat_lineage,
                    },
                    "coherent_current_system_mined_top_h_pose_pool": {
                        "enabled": bool(mined_hard_pose_pool_enabled),
                        "loss_weight": float(args.mined_hard_pose_pool_loss_weight),
                        "warmup_epochs": int(args.mined_hard_pose_pool_warmup_epochs),
                        "margin": float(args.mined_hard_pose_pool_margin),
                        "temperature": float(args.mined_hard_pose_pool_temperature),
                        "aggregation": (
                            "normalized_logmeanexp_frozen_target_free_top_h_"
                            "current_wrong_modes_v1"
                        ),
                        "frozen_mode_count": (
                            0
                            if not mined_hard_pose_pool_mode_positions
                            else int(
                                len(next(iter(mined_hard_pose_pool_mode_positions.values())))
                            )
                        ),
                        "targets_loaded_train_only": mined_hard_repeat_path is not None,
                        "selection_scope": "current_model_inner_train_only_v1",
                        "target_free_mode_selection": (
                            "frozen_metadata_pair_id_mode_index_manifest_v1"
                        ),
                        "selection_before_target_join": True,
                        "validation_usage": "forbidden_static_hard_repeat_gate_only_v1",
                        "lineage": mined_hard_repeat_lineage,
                    },
                    "support_permutation_contrastive": {
                        "enabled": bool(permutation_contrastive_enabled),
                        "loss_weight": float(args.permutation_contrastive_loss_weight),
                        "margin": float(args.permutation_contrastive_margin),
                        "train_derangement": "query_epoch_deterministic_nonidentity_excluding_reserved_gate_shift_v1",
                        "rgb_patch_derangement": (
                            "recrop_deranged_support_image_at_fixed_support_coordinate_v2"
                        ),
                        "reserved_inner_gate_shift": int(args.permutation_control_shift),
                    },
                    "exact_identity_support_appearance_contrastive": {
                        "enabled": bool(identity_support_contrastive_enabled),
                        "loss_weight": float(args.identity_support_contrastive_loss_weight),
                        "margin": float(args.identity_support_contrastive_margin),
                        "requires_registered_exact_identity_targets": True,
                        "target_mode_verified": bool(registered_identity_target_mode_verified),
                        "target_source": registered_identity_target_source,
                        "normal_and_deranged_compare": (
                            "same_registered_exact_track_same_local_offset_fixed_candidate_view_mixture_v1"
                        ),
                        "train_derangement": (
                            "query_epoch_deterministic_nonidentity_excluding_reserved_gate_shift_v1"
                        ),
                        "reserved_inner_gate_shift": int(args.permutation_control_shift),
                    },
                    "registered_candidate_context_identity": {
                        "enabled": bool(context_identity_enabled),
                        "loss_weight": float(args.context_identity_loss_weight),
                        "requires_registered_exact_identity_targets": True,
                        "target_mode_verified": bool(registered_identity_target_mode_verified),
                        "target_source": registered_identity_target_source,
                        "target_join": (
                            "full_target_free_rgb_context_edge_prediction_then_train_only_"
                            "registered_identity_source_id_join_v2"
                        ),
                        "candidate_mixture": (
                            "fixed_global_topl_candidates_and_fixed_maplet_support_view_"
                            "weights_v1"
                        ),
                    },
                    "registered_candidate_context_support_appearance_derangement": {
                        "enabled": bool(context_identity_support_permutation_enabled),
                        "loss_weight": float(
                            args.context_identity_support_permutation_loss_weight
                        ),
                        "margin": float(args.context_identity_support_permutation_margin),
                        "requires_registered_exact_identity_targets": True,
                        "target_mode_verified": bool(registered_identity_target_mode_verified),
                        "target_source": registered_identity_target_source,
                        "normal_prediction": (
                            "independent_context_only_target_free_forward_before_"
                            "train_only_identity_join_v2"
                        ),
                        "deranged_prediction": (
                            "independent_context_only_target_free_forward_no_rgb_density_"
                            "fixed_query_candidate_prior_and_support_weight_v1"
                        ),
                        "train_derangement": (
                            "query_epoch_deterministic_image_only_nonidentity_excluding_"
                            "reserved_gate_shift_v2"
                        ),
                        "reserved_inner_gate_shift": int(args.permutation_control_shift),
                    },
                    "world_size": int(state.world_size),
                    "seed": int(args.seed),
                    "inner_validation": {
                        "split": "train_query_only",
                        "gate_evaluator_manifest": current_inner_gate_evaluator_manifest(),
                        "fold_count": int(args.inner_validation_fold_count),
                        "fold_index": int(args.inner_validation_fold_index),
                        "query_partition": train_query_partition_manifest(
                            all_query_ids=all_query_ids,
                            inner_train_query_ids=inner_train_ids,
                            inner_validation_query_ids=inner_validation_ids,
                            fold_count=int(args.inner_validation_fold_count),
                            fold_index=int(args.inner_validation_fold_index),
                        ),
                        "selected_epoch": int(final_epoch),
                        "selected_metrics": final_metrics,
                        "checkpoint_selection": fixed_final_epoch_checkpoint_selection(
                            epochs=int(args.epochs)
                        ),
                        "support_permutation_control": {
                            "enabled": True,
                            "reserved_shift": int(args.permutation_control_shift),
                            "geometry_fixed_image_only": True,
                            "rgb_support_patch": (
                                "recrop_deranged_image_at_fixed_coordinate_v2"
                            ),
                            "not_used_by_train_contrastive_loss": bool(
                                support_permutation_training_enabled
                            ),
                        },
                        "selector": {
                            "policy": str(args.validation_selector_policy),
                            "point_budget": int(args.validation_selector_point_budget),
                            "grid_rows": int(args.validation_selector_grid_rows),
                            "grid_columns": int(args.validation_selector_grid_columns),
                            "target_free_static_only": True,
                        },
                        "selection_before_target_join": True,
                        "observed_target_preserving_sampler_used": bool(
                            exact_identity_objective_enabled
                        ),
                        "current_system_mined_targets_loaded": False,
                        "current_system_mined_targets_excluded_from_gate": True,
                        "current_system_mined_top_h_pose_pool_excluded_from_gate": True,
                        "gate": gate,
                    },
                },
                "lineage": {
                    "layout_sha256": file_sha256_short(layout_path),
                    "training_targets_sha256": file_sha256_short(target_path),
                    "registered_identity_targets_sha256": (
                        ""
                        if registered_identity_targets is None
                        else (
                            file_sha256_short(target_path)
                            if registered_identity_targets_path is None
                            else file_sha256_short(registered_identity_targets_path)
                        )
                    ),
                    "registered_identity_target_source": registered_identity_target_source,
                    "hard_repeat_targets_sha256": (
                        "" if hard_repeat_path is None else file_sha256_short(hard_repeat_path)
                    ),
                    "mined_hard_repeat_targets_sha256": (
                        ""
                        if mined_hard_repeat_path is None
                        else file_sha256_short(mined_hard_repeat_path)
                    ),
                    "mined_registered_identity_targets_sha256": (
                        ""
                        if mined_registered_identity_targets_path is None
                        else file_sha256_short(mined_registered_identity_targets_path)
                    ),
                    "mined_hard_repeat_lineage": mined_hard_repeat_lineage,
                    "initialization_checkpoint": initialization,
                    "source_image_manifest_sha256": str(
                        source_metadata.get("source_image_manifest_sha256", "")
                    ),
                    "projection_space_id": str(layout.metadata["projection_space_id"]),
                    "descriptor_space_id": str(layout.metadata["descriptor_space_id"]),
                    "rgb_coordinate_bridge": rgb_bridge,
                },
                "inputs": _source_manifest(input_paths),
                "train_query_count": int(len(all_query_ids)),
                "inner_train_query_count": int(len(inner_train_ids)),
                "inner_validation_query_count": int(len(inner_validation_ids)),
            }
            torch.save(
                {"format": CHECKPOINT_FORMAT, "state_dict": final_state_dict, "metadata": metadata},
                checkpoint_path,
            )
            summary = {
                "stage": "train_candidate_pose_rgb_spatial_likelihood",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256_short(checkpoint_path),
                "elapsed_seconds": float(time.time() - start_time),
                "history": history,
                "checkpoint_selection": {
                    **fixed_final_epoch_checkpoint_selection(epochs=int(args.epochs)),
                    "inner_validation": final_metrics,
                    "gate": gate,
                },
                "rgb_cache_rank0": image_cache.summary(),
                "protocol": {
                    "train_only_target_artifact": True,
                    "runtime_layout_remains_target_free": True,
                    "pose_or_ground_truth_not_available_to_runtime_encoder": True,
                    "training_sampler_may_use_train_targets": True,
                    "inner_validation_selector_target_free": True,
                    "inner_validation_observed_target_sampler_used": False,
                    "current_system_mined_targets_are_inner_train_loss_only": True,
                    "current_system_mined_targets_excluded_from_inner_validation": True,
                    "current_system_mined_top_h_pose_pool_excluded_from_inner_validation": True,
                    "inner_gate_evaluator_manifest": current_inner_gate_evaluator_manifest(),
                    "support_appearance_control_geometry_fixed": True,
                    "checkpoint_selected_from_final_epoch_only": True,
                    "no_render": True,
                    "no_image_retrieval_or_submap": True,
                    "heldout_validation_or_test_not_run": True,
                },
            }
            history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if state.enabled:
            distributed.barrier()
        return {"checkpoint": str(checkpoint_path), "rank": int(state.rank)}
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = train_candidate_pose_rgb_spatial_likelihood(args)
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
