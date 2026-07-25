"""Broad real-image pretraining for candidate-set identity LLR.

The P1 layout has only a few hundred exact observation identities.  This
trainer first learns the target-free candidate/view appearance likelihood from
many train-only SfM observation pairs:

* one fixed same-track support observation;
* nineteen distinct RADIO-PCA global hard-negative tracks;
* a candidate-slot permutation applied before the target is joined.

The runtime encoder sees only RGB patches, frozen 2-D RADIO/ALIKE feature maps,
and fixed query/support coordinates.  It never receives a pose, residual,
track ID, candidate rank, coarse score, or label.  A checkpoint that fails the
inner normal/support-deranged/position-only gate is diagnostic-only and must
not initialize P1 or PnP.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys
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

from feature_extract.tools.vfm.pretrain_candidate_pose_context_identity_l0 import (
    candidate_slot_permutations_from_anchor_ids,
    positive_targets_after_candidate_slot_permutation,
)
from feature_extract.tools.vfm.pretrain_candidate_pose_rgb_spatial_likelihood import (
    _DistributedState,
    _crop_pair_rgb_patches,
    _discover_rgb_image_size,
    _finalize_distributed,
    _initialize_distributed,
    _rank_batch_rows,
    _runtime_from_pair_rows,
    _source_table,
    _validate_rgb_coordinate_bridge,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_identity_llr import (
    _component_edge_only_prediction,
    _identity_llr_expert_update_l2,
    _identity_llr_head_statistics,
    _identity_llr_optimizer_parameter_groups,
    _identity_llr_scalar_head_parameters,
    _write_json_atomically,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_V2_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES,
    CandidatePoseRGBSpatialIdentityLLR,
    CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    candidate_pose_rgb_spatial_identity_component_edge_usable,
    marginalize_candidate_pose_rgb_spatial_identity_llr,
    resolve_candidate_pose_rgb_spatial_identity_context_windows,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_pose_identity import (
    CandidatePoseRGBSpatialHardPoseIdentityTargets,
    load_candidate_pose_rgb_spatial_hard_pose_identity_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    permute_runtime_candidate_slots,
    permute_runtime_support_appearance,
    permute_support_patch_appearance,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_observation_pairs import (
    CandidatePoseRGBSpatialObservationPairs,
    load_candidate_pose_rgb_spatial_observation_pairs,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


CHECKPOINT_FORMAT = "candidate_pose_rgb_spatial_identity_llr_observation_pretrain_v2"
OBSERVATION_IDENTITY_GATE_VERSION = (
    "combined_visual_controls_plus_conditional_source_visual_ablation_and_hard_pose_v5"
)
_REPRESENTATION_INITIALIZER_FORMATS = frozenset(
    {
        "candidate_pose_rgb_spatial_identity_llr_observation_pretrain_v1",
        CHECKPOINT_FORMAT,
    }
)
_REPRESENTATION_INITIALIZER_PREFIXES = (
    "context_encoders.",
    "global_projections.",
    "texture_encoder.",
    "context_projector.",
    "rgb_projector.",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-pairs", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help=(
            "Optional gate-approved target-free identity checkpoint. It may initialize "
            "a new train-only hard-pose mining curriculum only when visual-source "
            "lineage and model architecture match exactly."
        ),
    )
    parser.add_argument(
        "--representation-init-checkpoint",
        default="",
        help=(
            "Optional train-only visual representation initializer. It may load only "
            "the RGB/RADIO/ALIKE encoders and projectors from a compatible older "
            "broad checkpoint; no scalar edge, view, or null head is transferred."
        ),
    )
    parser.add_argument(
        "--hard-pose-identity-targets",
        default="",
        help=(
            "Optional train-only sidecar identifying distinct candidates that are "
            "locally plausible under current coherent-wrong poses."
        ),
    )
    parser.add_argument("--rgb-context-radius-px", type=float, default=48.0)
    parser.add_argument("--rgb-step-px", type=float, default=1.0)
    parser.add_argument("--texture-feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-abs-log-ratio", type=float, default=4.0)
    parser.add_argument("--edge-chunk-size", type=int, default=128)
    parser.add_argument("--radio-final-context-window", type=int, default=15)
    parser.add_argument("--radio-intermediate-context-window", type=int, default=15)
    parser.add_argument("--alike-context-window", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--projector-learning-rate", type=float, default=2e-4)
    parser.add_argument("--context-learning-rate", type=float, default=1e-4)
    parser.add_argument("--texture-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--rgb-identity-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional RGB-only exact-candidate regularizer. V5 uses paired conditional "
            "visual-ablation supervision by default instead of requiring RGB alone to "
            "solve the full candidate set."
        ),
    )
    parser.add_argument(
        "--context-coherence-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional context-only exact-candidate regularizer. V5 uses paired "
            "conditional visual-ablation supervision by default instead."
        ),
    )
    parser.add_argument("--hard-pose-identity-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--rgb-identity-hard-pose-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Direct correct-versus-coherent-wrong margin for the RGB-only "
            "identity expert; requires the train-only hard-pose sidecar."
        ),
    )
    parser.add_argument(
        "--context-coherence-hard-pose-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Direct correct-versus-coherent-wrong margin for the RADIO/ALIKE-only "
            "context expert; requires the train-only hard-pose sidecar."
        ),
    )
    parser.add_argument("--hard-pose-identity-margin", type=float, default=0.25)
    parser.add_argument("--support-permutation-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--rgb-identity-support-permutation-loss-weight",
        type=float,
        default=0.25,
        help=(
            "Support-appearance derangement margin for the RGB-only identity "
            "expert, evaluated on common available edges."
        ),
    )
    parser.add_argument(
        "--context-coherence-support-permutation-loss-weight",
        type=float,
        default=0.25,
        help=(
            "Support-appearance derangement margin for the RADIO/ALIKE-only "
            "context expert, evaluated on common available edges."
        ),
    )
    parser.add_argument(
        "--rgb-identity-position-only-loss-weight",
        type=float,
        default=0.25,
        help=(
            "Require the RGB-only correct-candidate score to exceed its "
            "coordinate-only ablation during broad pretraining."
        ),
    )
    parser.add_argument(
        "--context-coherence-position-only-loss-weight",
        type=float,
        default=0.25,
        help=(
            "Require the RADIO/ALIKE-only correct-candidate score to exceed its "
            "coordinate-only ablation during broad pretraining."
        ),
    )
    parser.add_argument(
        "--rgb-identity-conditional-visual-ablation-loss-weight",
        type=float,
        default=0.5,
        help=(
            "Require the joint candidate margin to drop when RGB content alone is "
            "ablated while fixed geometry and RADIO/ALIKE content remain unchanged."
        ),
    )
    parser.add_argument(
        "--context-coherence-conditional-visual-ablation-loss-weight",
        type=float,
        default=0.5,
        help=(
            "Require the joint candidate margin to drop when all RADIO/ALIKE content "
            "is ablated while fixed geometry and RGB content remain unchanged."
        ),
    )
    parser.add_argument(
        "--rgb-identity-conditional-visual-ablation-hard-pose-loss-weight",
        type=float,
        default=0.25,
        help="Train-only coherent-wrong counterpart of RGB conditional visual ablation.",
    )
    parser.add_argument(
        "--context-coherence-conditional-visual-ablation-hard-pose-loss-weight",
        type=float,
        default=0.25,
        help="Train-only coherent-wrong counterpart of context conditional visual ablation.",
    )
    parser.add_argument("--support-permutation-margin", type=float, default=0.25)
    parser.add_argument("--position-only-margin", type=float, default=0.25)
    parser.add_argument("--conditional-visual-ablation-margin", type=float, default=0.05)
    parser.add_argument(
        "--support-permutation-every-n-steps",
        type=int,
        default=1,
        help=(
            "Apply the expensive train-time support derangement every N local steps. "
            "The validation audit always evaluates every row under all controls."
        ),
    )
    parser.add_argument(
        "--position-only-every-n-steps",
        type=int,
        default=1,
        help=(
            "Apply the target-free visual-content ablation margin every N local "
            "steps; validation always evaluates it on every row."
        ),
    )
    parser.add_argument(
        "--conditional-visual-ablation-every-n-steps",
        type=int,
        default=4,
        help=(
            "Apply the two paired single-source visual ablations every N local steps. "
            "Validation always audits them on every row."
        ),
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-normal-margin", type=float, default=0.05)
    parser.add_argument("--minimum-support-visual-gap", type=float, default=0.05)
    parser.add_argument("--minimum-position-visual-gap", type=float, default=0.05)
    parser.add_argument("--minimum-support-correct-score-gap", type=float, default=0.05)
    parser.add_argument("--minimum-position-correct-score-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-pose-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-hard-pose-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-pose-visual-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-pose-eligible-query-fraction", type=float, default=0.9)
    parser.add_argument("--minimum-conditional-visual-ablation-margin", type=float, default=0.05)
    parser.add_argument(
        "--minimum-conditional-visual-ablation-win-fraction", type=float, default=0.55
    )
    parser.add_argument(
        "--minimum-conditional-visual-ablation-top1-delta", type=float, default=0.0
    )
    parser.add_argument(
        "--minimum-conditional-visual-ablation-hard-pose-gap", type=float, default=0.05
    )
    parser.add_argument(
        "--minimum-conditional-visual-ablation-hard-pose-win-fraction",
        type=float,
        default=0.55,
    )
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument("--rgb-cache-dtype", choices=("float16", "uint8"), default="uint8")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> dict[str, int]:
    values = (
        float(args.rgb_context_radius_px),
        float(args.rgb_step_px),
        float(args.max_abs_log_ratio),
        float(args.learning_rate),
        float(args.head_learning_rate),
        float(args.projector_learning_rate),
        float(args.context_learning_rate),
        float(args.texture_learning_rate),
        float(args.weight_decay),
        float(args.identity_loss_weight),
        float(args.rgb_identity_loss_weight),
        float(args.context_coherence_loss_weight),
        float(args.hard_pose_identity_loss_weight),
        float(args.rgb_identity_hard_pose_loss_weight),
        float(args.context_coherence_hard_pose_loss_weight),
        float(args.hard_pose_identity_margin),
        float(args.support_permutation_loss_weight),
        float(args.rgb_identity_support_permutation_loss_weight),
        float(args.context_coherence_support_permutation_loss_weight),
        float(args.rgb_identity_position_only_loss_weight),
        float(args.context_coherence_position_only_loss_weight),
        float(args.rgb_identity_conditional_visual_ablation_loss_weight),
        float(args.context_coherence_conditional_visual_ablation_loss_weight),
        float(args.rgb_identity_conditional_visual_ablation_hard_pose_loss_weight),
        float(args.context_coherence_conditional_visual_ablation_hard_pose_loss_weight),
        float(args.support_permutation_margin),
        float(args.position_only_margin),
        float(args.conditional_visual_ablation_margin),
        float(args.gradient_clip_norm),
        float(args.minimum_win_fraction),
        float(args.minimum_normal_margin),
        float(args.minimum_support_visual_gap),
        float(args.minimum_position_visual_gap),
        float(args.minimum_support_correct_score_gap),
        float(args.minimum_position_correct_score_gap),
        float(args.minimum_hard_pose_win_fraction),
        float(args.minimum_hard_pose_gap),
        float(args.minimum_hard_pose_visual_gap),
        float(args.minimum_hard_pose_eligible_query_fraction),
        float(args.minimum_conditional_visual_ablation_margin),
        float(args.minimum_conditional_visual_ablation_win_fraction),
        float(args.minimum_conditional_visual_ablation_top1_delta),
        float(args.minimum_conditional_visual_ablation_hard_pose_gap),
        float(args.minimum_conditional_visual_ablation_hard_pose_win_fraction),
        float(args.rgb_cache_gb),
    )
    if (
        int(args.texture_feature_dim) <= 0
        or int(args.hidden_dim) < 4
        or int(args.edge_chunk_size) <= 0
        or int(args.batch_size) < 2
        or int(args.epochs) <= 0
        or int(args.max_train_rows) < 0
        or int(args.max_validation_rows) < 0
        or int(args.support_permutation_every_n_steps) <= 0
        or int(args.position_only_every_n_steps) <= 0
        or int(args.conditional_visual_ablation_every_n_steps) <= 0
        or not all(math.isfinite(value) for value in values)
        or float(args.rgb_context_radius_px) <= 0.0
        or float(args.rgb_step_px) <= 0.0
        or float(args.max_abs_log_ratio) <= 0.0
        or float(args.learning_rate) <= 0.0
        or min(
            float(args.head_learning_rate),
            float(args.projector_learning_rate),
            float(args.context_learning_rate),
            float(args.texture_learning_rate),
        ) < 0.0
        or float(args.weight_decay) < 0.0
        or float(args.identity_loss_weight) <= 0.0
        or min(
            float(args.rgb_identity_loss_weight),
            float(args.context_coherence_loss_weight),
        ) < 0.0
        or min(
            float(args.hard_pose_identity_loss_weight),
            float(args.rgb_identity_hard_pose_loss_weight),
            float(args.context_coherence_hard_pose_loss_weight),
            float(args.rgb_identity_conditional_visual_ablation_hard_pose_loss_weight),
            float(args.context_coherence_conditional_visual_ablation_hard_pose_loss_weight),
        ) < 0.0
        or float(args.hard_pose_identity_margin) < 0.0
        or min(
            float(args.support_permutation_loss_weight),
            float(args.rgb_identity_support_permutation_loss_weight),
            float(args.context_coherence_support_permutation_loss_weight),
            float(args.rgb_identity_position_only_loss_weight),
            float(args.context_coherence_position_only_loss_weight),
            float(args.rgb_identity_conditional_visual_ablation_loss_weight),
            float(args.context_coherence_conditional_visual_ablation_loss_weight),
        ) < 0.0
        or min(
            float(args.support_permutation_margin),
            float(args.position_only_margin),
            float(args.conditional_visual_ablation_margin),
        ) < 0.0
        or float(args.gradient_clip_norm) <= 0.0
        or not 0.0 <= float(args.minimum_win_fraction) <= 1.0
        or min(
            float(args.minimum_normal_margin),
            float(args.minimum_support_visual_gap),
            float(args.minimum_position_visual_gap),
            float(args.minimum_support_correct_score_gap),
            float(args.minimum_position_correct_score_gap),
        ) < 0.0
        or not 0.0 <= float(args.minimum_hard_pose_win_fraction) <= 1.0
        or float(args.minimum_hard_pose_gap) < 0.0
        or float(args.minimum_hard_pose_visual_gap) < 0.0
        or not 0.0 < float(args.minimum_hard_pose_eligible_query_fraction) <= 1.0
        or float(args.minimum_conditional_visual_ablation_margin) < 0.0
        or not 0.0 <= float(args.minimum_conditional_visual_ablation_win_fraction) <= 1.0
        or float(args.minimum_conditional_visual_ablation_top1_delta) < 0.0
        or float(args.minimum_conditional_visual_ablation_hard_pose_gap) < 0.0
        or not 0.0 <= float(args.minimum_conditional_visual_ablation_hard_pose_win_fraction) <= 1.0
        or float(args.rgb_cache_gb) <= 0.0
    ):
        raise ValueError("identity observation pretraining arguments are invalid")
    if max(
        float(args.hard_pose_identity_loss_weight),
        float(args.rgb_identity_hard_pose_loss_weight),
        float(args.context_coherence_hard_pose_loss_weight),
        float(args.rgb_identity_conditional_visual_ablation_hard_pose_loss_weight),
        float(args.context_coherence_conditional_visual_ablation_hard_pose_loss_weight),
    ) > 0.0 and not str(args.hard_pose_identity_targets).strip():
        raise ValueError("hard-pose identity loss requires a train-only target sidecar")
    if str(args.init_checkpoint).strip() and str(args.representation_init_checkpoint).strip():
        raise ValueError("full and representation-only identity initializers are mutually exclusive")
    return resolve_candidate_pose_rgb_spatial_identity_context_windows(
        {
            "radio_final": int(args.radio_final_context_window),
            "radio_intermediate": int(args.radio_intermediate_context_window),
            "alike": int(args.alike_context_window),
        }
    )


def _limited_rows(rows: np.ndarray, *, limit: int, seed: int) -> np.ndarray:
    values = np.asarray(rows, dtype=np.int64).reshape(-1)
    if len(values) == 0 or int(limit) < 0:
        raise ValueError("identity observation row cap is invalid")
    if int(limit) == 0 or len(values) <= int(limit):
        return values
    selected = np.random.default_rng(int(seed)).choice(values, size=int(limit), replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def _load_identity_observation_initializer(
    *,
    model: CandidatePoseRGBSpatialIdentityLLR,
    path: Path,
    candidate_count: int,
    source_lineage: Mapping[str, object],
    windows: Mapping[str, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Load only a gate-approved target-free visual initializer.

    The observation-pair target artifact is intentionally allowed to change:
    that is the point of hard-pose curriculum training.  Image/feature source
    lineage, architecture, and the broad visual gate must remain identical.
    """

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"identity initializer does not exist: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("identity initializer cannot be loaded") from error
    if not isinstance(payload, Mapping):
        raise ValueError("identity initializer payload is invalid")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("identity initializer lacks metadata or state")
    training = metadata.get("training")
    inner = training.get("inner_validation") if isinstance(training, Mapping) else None
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    config = metadata.get("config")
    lineage = metadata.get("lineage")
    expected_config = {
        "rgb_context_radius_px": float(args.rgb_context_radius_px),
        "rgb_step_px": float(args.rgb_step_px),
        "texture_feature_dim": int(args.texture_feature_dim),
        "hidden_dim": int(args.hidden_dim),
        "max_abs_log_ratio": float(args.max_abs_log_ratio),
    }
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or metadata.get("format") != CHECKPOINT_FORMAT
        or metadata.get("model_format") != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("p1_initialization_allowed") is not True
        or metadata.get("visual_evidence_gate_version") != OBSERVATION_IDENTITY_GATE_VERSION
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or int(metadata.get("fixed_candidate_count", -1)) != int(candidate_count)
        or not isinstance(config, Mapping)
        or not isinstance(lineage, Mapping)
    ):
        raise ValueError("identity initializer is not a gate-approved target-free checkpoint")
    for name, value in expected_config.items():
        candidate = config.get(name)
        if isinstance(value, float):
            if not isinstance(candidate, (float, int)) or not math.isclose(
                float(candidate), value, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError("identity initializer configuration differs")
        elif candidate != value:
            raise ValueError("identity initializer configuration differs")
    if dict(config.get("context_windows", {})) != dict(windows):
        raise ValueError("identity initializer context windows differ")
    for name, value in source_lineage.items():
        if lineage.get(name) != value:
            raise ValueError("identity initializer visual-source lineage differs")
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise ValueError("identity initializer state is incompatible") from error
    return {
        "kind": "gate_approved_target_free_identity_initializer",
        "path": str(checkpoint),
        "sha256": file_sha256_short(checkpoint),
        "selected_epoch": int(inner.get("selected_epoch", -1)),
        "source_observation_pairs_sha256": str(lineage.get("observation_pairs_sha256", "")),
    }


def _load_visual_representation_initializer(
    *,
    model: CandidatePoseRGBSpatialIdentityLLR,
    path: Path,
    candidate_count: int,
    source_lineage: Mapping[str, object],
    windows: Mapping[str, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Load only shared visual representations from an older broad checkpoint.

    A v2 checkpoint can contain useful real-RGB and frozen-feature encoders,
    but its fused scalar edge head is not valid evidence for v3's independent
    RGB-identity and RADIO/ALIKE-context experts.  This loader therefore uses
    the checkpoint strictly as a representation initializer: it copies only
    the common visual encoders and projectors, then leaves every scalar head
    at the current model's explicit initialization.
    """

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"representation initializer does not exist: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("representation initializer cannot be loaded") from error
    if not isinstance(payload, Mapping):
        raise ValueError("representation initializer payload is invalid")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("representation initializer lacks metadata or state")
    training = metadata.get("training")
    inner = training.get("inner_validation") if isinstance(training, Mapping) else None
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    config = metadata.get("config")
    lineage = metadata.get("lineage")
    expected_config = {
        "rgb_context_radius_px": float(args.rgb_context_radius_px),
        "rgb_step_px": float(args.rgb_step_px),
        "texture_feature_dim": int(args.texture_feature_dim),
        "hidden_dim": int(args.hidden_dim),
        "max_abs_log_ratio": float(args.max_abs_log_ratio),
    }
    if (
        payload.get("format") not in _REPRESENTATION_INITIALIZER_FORMATS
        or metadata.get("format") != payload.get("format")
        or metadata.get("model_format")
        not in {
            CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_V2_FORMAT,
            CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
        }
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("pose_or_ground_truth_used_by_runtime_scorer") is not False
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("candidate_slot_permutation_equivariant") is not True
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or int(metadata.get("fixed_candidate_count", -1)) != int(candidate_count)
        or not isinstance(config, Mapping)
        or not isinstance(lineage, Mapping)
    ):
        raise ValueError("representation initializer is not a gate-approved target-free checkpoint")
    for name, value in expected_config.items():
        candidate = config.get(name)
        if isinstance(value, float):
            if not isinstance(candidate, (float, int)) or not math.isclose(
                float(candidate), value, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError("representation initializer configuration differs")
        elif candidate != value:
            raise ValueError("representation initializer configuration differs")
    if dict(config.get("context_windows", {})) != dict(windows):
        raise ValueError("representation initializer context windows differ")
    for name, value in source_lineage.items():
        if lineage.get(name) != value:
            raise ValueError("representation initializer visual-source lineage differs")

    destination_state = model.state_dict()
    source_keys = {
        str(name)
        for name in state_dict
        if str(name).startswith(_REPRESENTATION_INITIALIZER_PREFIXES)
    }
    destination_keys = {
        str(name)
        for name in destination_state
        if str(name).startswith(_REPRESENTATION_INITIALIZER_PREFIXES)
    }
    if not source_keys or source_keys != destination_keys:
        raise ValueError("representation initializer visual state layout is incompatible")
    merged_state = dict(destination_state)
    loaded_parameter_count = 0
    for name in sorted(destination_keys):
        source_value = state_dict[name]
        destination_value = destination_state[name]
        if (
            not isinstance(source_value, torch.Tensor)
            or source_value.shape != destination_value.shape
            or source_value.dtype != destination_value.dtype
        ):
            raise ValueError("representation initializer visual state is incompatible")
        merged_state[name] = source_value
        loaded_parameter_count += int(source_value.numel())
    try:
        model.load_state_dict(merged_state, strict=True)
    except RuntimeError as error:
        raise ValueError("representation initializer state is incompatible") from error
    return {
        "kind": "representation_only_visual_initializer",
        "path": str(checkpoint),
        "sha256": file_sha256_short(checkpoint),
        "source_checkpoint_format": str(payload.get("format", "")),
        "source_model_format": str(metadata.get("model_format", "")),
        "selected_epoch": int(inner.get("selected_epoch", -1)),
        "loaded_prefixes": list(_REPRESENTATION_INITIALIZER_PREFIXES),
        "loaded_parameter_count": int(loaded_parameter_count),
        "scalar_heads_transferred": False,
        "source_observation_pairs_sha256": str(lineage.get("observation_pairs_sha256", "")),
    }


def _hard_pose_identity_mask_lookup(
    *,
    pairs: CandidatePoseRGBSpatialObservationPairs,
    targets: CandidatePoseRGBSpatialHardPoseIdentityTargets,
    pair_path: Path,
) -> dict[int, np.ndarray]:
    """Validate direct hard targets against the fixed visual pair layout."""

    metadata = targets.metadata
    if (
        str(metadata.get("hard_pose_identity_pairs_sha256", "")) != file_sha256_short(pair_path)
        or int(metadata.get("candidate_count", -1)) != int(pairs.negative_count + 1)
        or metadata.get("positive_candidate_is_original_slot_zero") is not True
        or metadata.get("pose_projection_or_residual_serialized") is not False
    ):
        raise ValueError("hard-pose identity sidecar lineage differs from observation pairs")
    pair_row_by_anchor = {
        int(anchor): row for row, anchor in enumerate(np.asarray(pairs.anchor_ids, dtype=np.int64))
    }
    if len(pair_row_by_anchor) != pairs.row_count:
        raise ValueError("observation-pair anchors are not unique")
    lookup: dict[int, np.ndarray] = {}
    for anchor, query_id, mask in zip(
        targets.anchor_ids.tolist(),
        targets.query_image_ids.tolist(),
        targets.hard_negative_candidate_mask,
    ):
        pair_row = pair_row_by_anchor.get(int(anchor))
        if pair_row is None or str(pairs.query_image_ids[pair_row]) != str(query_id):
            raise ValueError("hard-pose identity sidecar does not join the observation layout")
        if int(anchor) in lookup:
            raise ValueError("hard-pose identity sidecar repeats an anchor")
        lookup[int(anchor)] = np.asarray(mask, dtype=bool).copy()
    if not lookup:
        raise ValueError("hard-pose identity sidecar has no joinable targets")
    return lookup


def _permuted_hard_pose_negative_mask(
    *,
    pairs: CandidatePoseRGBSpatialObservationPairs,
    rows: np.ndarray,
    permutations: torch.Tensor,
    lookup: Mapping[int, np.ndarray] | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Join target-only hard candidates after the target-free slot permutation."""

    if lookup is None:
        return None
    selected = np.asarray(rows, dtype=np.int64).reshape(-1)
    order = torch.as_tensor(permutations, dtype=torch.long, device=device)
    candidate_count = int(pairs.negative_count + 1)
    if order.shape != (len(selected), candidate_count):
        raise ValueError("hard-pose identity target permutation is invalid")
    original = np.zeros((len(selected), candidate_count), dtype=bool)
    for position, anchor in enumerate(pairs.anchor_ids[selected].tolist()):
        mask = lookup.get(int(anchor))
        if mask is not None:
            original[position] = mask
    original_mask = torch.from_numpy(original).to(device=device)
    return original_mask.gather(1, order)


def _hard_pose_identity_statistics(
    *,
    candidate_values: torch.Tensor,
    candidate_usable: torch.Tensor,
    positive_targets: torch.Tensor,
    hard_negative_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return direct positive-vs-hardest-coherent-wrong gaps after scoring."""

    values = torch.as_tensor(candidate_values)
    usable = torch.as_tensor(candidate_usable, dtype=torch.bool, device=values.device)
    positive = torch.as_tensor(positive_targets, dtype=torch.bool, device=values.device)
    hard = (
        None
        if hard_negative_mask is None
        else torch.as_tensor(hard_negative_mask, dtype=torch.bool, device=values.device)
    )
    if (
        values.ndim != 2
        or usable.shape != values.shape
        or positive.shape != values.shape
        or torch.any(positive.sum(dim=1) != 1)
        or (hard is not None and hard.shape != values.shape)
    ):
        raise ValueError("hard-pose identity statistic inputs are invalid")
    gaps = torch.zeros((len(values),), dtype=values.dtype, device=values.device)
    wins = torch.zeros((len(values),), dtype=torch.bool, device=values.device)
    active = torch.zeros((len(values),), dtype=torch.bool, device=values.device)
    if hard is None:
        return gaps, wins, active
    labels = torch.argmax(positive.to(dtype=torch.long), dim=1)
    valid_hard = hard & usable
    active = usable.gather(1, labels[:, None]).squeeze(1) & torch.any(valid_hard, dim=1)
    if bool(active.any()):
        rows = torch.nonzero(active, as_tuple=False).reshape(-1)
        positive_values = values[rows, labels[rows]]
        negative_values = values[rows].masked_fill(~valid_hard[rows], -torch.inf).amax(dim=1)
        gaps[rows] = positive_values - negative_values
        wins[rows] = gaps[rows] > 0.0
    return gaps, wins, active


def _hard_pose_identity_margin_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    positive_targets: torch.Tensor,
    hard_negative_mask: torch.Tensor | None,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train the exact candidate above the hardest coherent wrong identity."""

    values, usable = _candidate_values(runtime=runtime, prediction=prediction)
    gaps, wins, active = _hard_pose_identity_statistics(
        candidate_values=values,
        candidate_usable=usable,
        positive_targets=positive_targets,
        hard_negative_mask=hard_negative_mask,
    )
    if not bool(active.any()):
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {"active_rows": 0.0, "mean_gap": 0.0, "win_fraction": 0.0}
    loss = F.softplus(torch.as_tensor(float(margin), device=gaps.device) - gaps[active]).mean()
    return loss, {
        "active_rows": float(active.sum().item()),
        "mean_gap": float(gaps[active].detach().mean().item()),
        "win_fraction": float(wins[active].float().detach().mean().item()),
    }


def _runtime_and_targets(
    *,
    pairs: CandidatePoseRGBSpatialObservationPairs,
    rows: np.ndarray,
    image_index_by_id: Mapping[str, int],
    permutation_seed: int,
) -> tuple[CandidatePoseRGBSpatialRuntime, torch.Tensor, torch.Tensor]:
    """Randomize complete candidate slots before labels are materialized."""

    selected = np.asarray(rows, dtype=np.int64).reshape(-1)
    runtime = _runtime_from_pair_rows(
        pairs=pairs, rows=selected, image_index_by_id=image_index_by_id
    )
    order = candidate_slot_permutations_from_anchor_ids(
        anchor_ids=pairs.anchor_ids[selected],
        candidate_count=runtime.candidate_count,
        seed=int(permutation_seed),
    )
    return (
        permute_runtime_candidate_slots(runtime, permutations=order),
        positive_targets_after_candidate_slot_permutation(order),
        order,
    )


def _candidate_values(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = marginalize_candidate_pose_rgb_spatial_identity_llr(
        prediction=prediction,
        runtime=runtime,
        missing_edge_log_likelihood_ratio=0.0,
    )
    usable = torch.any(prediction.edge_usable, dim=2)
    if values.shape != usable.shape:
        raise RuntimeError("identity observation candidate marginalization drifted")
    return values, usable


def _row_identity_statistics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-row CE, correct-minus-hardest margin, top-1 and active mask."""

    values, usable = _candidate_values(runtime=runtime, prediction=prediction)
    observed = torch.as_tensor(targets, dtype=torch.bool, device=values.device)
    if observed.shape != values.shape or torch.any(observed.sum(dim=1) != 1):
        raise ValueError("identity observation labels are incompatible with candidate outputs")
    labels = torch.argmax(observed.to(dtype=torch.long), dim=1)
    active = usable.gather(1, labels[:, None]).squeeze(1) & (usable.sum(dim=1) >= 2)
    masked = values.masked_fill(~usable, -torch.inf)
    ce = torch.zeros((len(masked),), dtype=masked.dtype, device=masked.device)
    if bool(active.any()):
        ce[active] = F.cross_entropy(masked[active], labels[active], reduction="none")
    target = masked.gather(1, labels[:, None]).squeeze(1)
    competitors = masked.clone()
    competitors.scatter_(1, labels[:, None], -torch.inf)
    margin = target - competitors.amax(dim=1)
    top1 = torch.argmax(masked, dim=1) == labels
    return ce, margin, top1, active


def _identity_cross_entropy_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce, margin, top1, active = _row_identity_statistics(
        runtime=runtime, prediction=prediction, targets=targets
    )
    if not bool(active.any()):
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "active_rows": 0.0,
            "cross_entropy": 0.0,
            "mean_margin": 0.0,
            "top1_accuracy": 0.0,
        }
    return ce[active].mean(), {
        "active_rows": float(active.sum().item()),
        "cross_entropy": float(ce[active].detach().mean().item()),
        "mean_margin": float(margin[active].detach().mean().item()),
        "top1_accuracy": float(top1[active].float().detach().mean().item()),
    }


def _support_derangement_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    deranged_runtime: CandidatePoseRGBSpatialRuntime,
    deranged_prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    targets: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    normal, normal_usable = _candidate_values(runtime=runtime, prediction=prediction)
    deranged, deranged_usable = _candidate_values(
        runtime=deranged_runtime, prediction=deranged_prediction
    )
    observed = torch.as_tensor(targets, dtype=torch.bool, device=normal.device)
    if observed.shape != normal.shape or deranged.shape != normal.shape:
        raise ValueError("identity support-derangement inputs are incompatible")
    labels = torch.argmax(observed.to(dtype=torch.long), dim=1)
    active = normal_usable.gather(1, labels[:, None]).squeeze(1) & deranged_usable.gather(
        1, labels[:, None]
    ).squeeze(1)
    if not bool(active.any()):
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {"active_rows": 0.0, "mean_gap": 0.0, "win_fraction": 0.0}
    rows = torch.nonzero(active, as_tuple=False).reshape(-1)
    gaps = normal[rows, labels[rows]] - deranged[rows, labels[rows]]
    loss = F.softplus(torch.as_tensor(float(margin), device=gaps.device) - gaps).mean()
    return loss, {
        "active_rows": float(len(rows)),
        "mean_gap": float(gaps.detach().mean().item()),
        "win_fraction": float((gaps.detach() > 0.0).float().mean().item()),
    }


def _visual_content_ablation_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    position_only_prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    targets: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Make a target-free visual score beat its own coordinate-only ablation.

    This has the same target timing as support derangement: the model scores a
    normal visual input and a zero-content input first, then the train-only
    correct candidate is joined to their difference.  The identical runtime on
    both sides keeps candidate ownership, view mass, and coordinates fixed.
    """

    return _support_derangement_loss(
        runtime=runtime,
        prediction=prediction,
        deranged_runtime=runtime,
        deranged_prediction=position_only_prediction,
        targets=targets,
        margin=float(margin),
    )


def _query_grouped_means(
    *, query_ids: Sequence[str], values: np.ndarray, active: np.ndarray
) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for query_id, value, enabled in zip(query_ids, values.tolist(), active.tolist()):
        if bool(enabled):
            grouped[str(query_id)].append(float(value))
    return np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)


def observation_identity_gate(
    metrics: Mapping[str, float],
    *,
    minimum_win_fraction: float,
    minimum_normal_margin: float,
    minimum_support_visual_gap: float,
    minimum_position_visual_gap: float,
    minimum_support_correct_score_gap: float,
    minimum_position_correct_score_gap: float,
) -> dict[str, object]:
    normal = float(metrics["normal_mean_margin"])
    support = float(metrics["support_permuted_mean_margin"])
    position = float(metrics["position_only_mean_margin"])
    checks = {
        "normal_margin": normal >= float(minimum_normal_margin),
        "normal_win_fraction": float(metrics["normal_win_fraction"])
        >= float(minimum_win_fraction),
        "support_appearance_gap": normal - support >= float(minimum_support_visual_gap),
        "visual_over_position_gap": normal - position >= float(minimum_position_visual_gap),
        "support_correct_score_gap": float(
            metrics["normal_minus_support_permuted_correct_candidate_score"]
        )
        >= float(minimum_support_correct_score_gap),
        "position_correct_score_gap": float(
            metrics["normal_minus_position_only_correct_candidate_score"]
        )
        >= float(minimum_position_correct_score_gap),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "normal_minus_support_permuted_margin": float(normal - support),
        "normal_minus_position_only_margin": float(normal - position),
        "normal_minus_support_permuted_correct_candidate_score": float(
            metrics["normal_minus_support_permuted_correct_candidate_score"]
        ),
        "normal_minus_position_only_correct_candidate_score": float(
            metrics["normal_minus_position_only_correct_candidate_score"]
        ),
    }


def _component_observation_identity_metrics(
    *, metrics: Mapping[str, float], component: str
) -> dict[str, float]:
    """Extract one independent expert's train-only visual-control metrics.

    The two source-masked heads must each beat the same appearance controls as
    the combined scorer.  Keeping this extraction explicit prevents a future
    checkpoint from being promoted because the fused sum happens to work while
    one alleged evidence factor is neutral or a coordinate shortcut.
    """

    prefix = f"{str(component)}_"
    fields = (
        "normal_mean_margin",
        "normal_win_fraction",
        "support_permuted_mean_margin",
        "position_only_mean_margin",
        "normal_minus_support_permuted_correct_candidate_score",
        "normal_minus_position_only_correct_candidate_score",
    )
    output: dict[str, float] = {}
    for field in fields:
        name = f"{prefix}{field}"
        if name not in metrics:
            raise ValueError(f"independent {component} observation metrics are incomplete")
        output[field] = float(metrics[name])
    return output


def independent_source_masked_observation_gate(
    metrics: Mapping[str, float],
    *,
    minimum_win_fraction: float,
    minimum_normal_margin: float,
    minimum_support_visual_gap: float,
    minimum_position_visual_gap: float,
    minimum_support_correct_score_gap: float,
    minimum_position_correct_score_gap: float,
) -> dict[str, object]:
    """Require the combined scorer and both independently masked experts.

    The combined score is allowed to use the sum of independent log-likelihood
    factors, but it cannot substitute for either factor's own evidence gate.
    """

    kwargs = {
        "minimum_win_fraction": float(minimum_win_fraction),
        "minimum_normal_margin": float(minimum_normal_margin),
        "minimum_support_visual_gap": float(minimum_support_visual_gap),
        "minimum_position_visual_gap": float(minimum_position_visual_gap),
        "minimum_support_correct_score_gap": float(minimum_support_correct_score_gap),
        "minimum_position_correct_score_gap": float(minimum_position_correct_score_gap),
    }
    combined = observation_identity_gate(metrics, **kwargs)
    rgb = observation_identity_gate(
        _component_observation_identity_metrics(metrics=metrics, component="rgb_identity"), **kwargs
    )
    context = observation_identity_gate(
        _component_observation_identity_metrics(
            metrics=metrics, component="context_coherence"
        ),
        **kwargs,
    )
    checks = dict(combined["checks"])
    checks.update(
        {f"rgb_identity_{name}": bool(value) for name, value in rgb["checks"].items()}
    )
    checks.update(
        {
            f"context_coherence_{name}": bool(value)
            for name, value in context["checks"].items()
        }
    )
    return {
        **combined,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "rgb_identity": rgb,
        "context_coherence": context,
    }


def conditional_source_visual_ablation_gate(
    metrics: Mapping[str, float],
    *,
    minimum_win_fraction: float,
    minimum_normal_margin: float,
    minimum_support_visual_gap: float,
    minimum_position_visual_gap: float,
    minimum_support_correct_score_gap: float,
    minimum_position_correct_score_gap: float,
    minimum_source_margin_gain: float,
    minimum_source_margin_win_fraction: float,
    minimum_source_top1_delta: float,
) -> dict[str, object]:
    """Gate an additive likelihood by paired source-content ablations.

    A source factor need not independently solve every fixed top-20 identity
    decision.  It must instead make the full candidate ranking better than an
    otherwise identical runtime where that source's visual content alone has
    been removed.  The combined scorer still has to pass the original strict
    normal/support/position controls.
    """

    combined = observation_identity_gate(
        metrics,
        minimum_win_fraction=float(minimum_win_fraction),
        minimum_normal_margin=float(minimum_normal_margin),
        minimum_support_visual_gap=float(minimum_support_visual_gap),
        minimum_position_visual_gap=float(minimum_position_visual_gap),
        minimum_support_correct_score_gap=float(minimum_support_correct_score_gap),
        minimum_position_correct_score_gap=float(minimum_position_correct_score_gap),
    )
    checks = dict(combined["checks"])
    components: dict[str, dict[str, object]] = {}
    for component in ("rgb_identity", "context_coherence"):
        required = {
            "mean_margin": (
                f"{component}_conditional_visual_content_minus_ablated_mean_margin"
            ),
            "margin_win_fraction": (
                f"{component}_conditional_visual_content_margin_win_fraction"
            ),
            "top1_delta": (
                f"{component}_conditional_visual_content_minus_ablated_top1_delta"
            ),
        }
        if any(name not in metrics for name in required.values()):
            raise ValueError(f"{component} conditional visual-ablation metrics are incomplete")
        values = {name: float(metrics[field]) for name, field in required.items()}
        component_checks = {
            "visual_margin_gain": values["mean_margin"] >= float(minimum_source_margin_gain),
            "visual_margin_win_fraction": values["margin_win_fraction"]
            >= float(minimum_source_margin_win_fraction),
            "visual_top1_delta": values["top1_delta"] >= float(minimum_source_top1_delta),
        }
        checks.update({f"{component}_{name}": value for name, value in component_checks.items()})
        components[component] = {"checks": component_checks, **values}
    return {
        **combined,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "conditional_source_visual_ablation": components,
    }


def _source_masked_control_predictions(
    *,
    normal_prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    deranged_prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    position_prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    component: str,
) -> tuple[
    CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    CandidatePoseRGBSpatialIdentityLLREdgePrediction,
]:
    """Expose one expert under one fixed availability mask for all controls.

    Support crop boundaries can differ after a support derangement.  The
    independent visual control must therefore score exactly the intersection of
    normal, deranged, and position-only available edges; otherwise availability
    itself can masquerade as visual evidence.
    """

    common = candidate_pose_rgb_spatial_identity_component_edge_usable(
        prediction=normal_prediction, component=str(component)
    )
    for prediction in (deranged_prediction, position_prediction):
        common &= candidate_pose_rgb_spatial_identity_component_edge_usable(
            prediction=prediction, component=str(component)
        ).to(device=common.device)
    return (
        _component_edge_only_prediction(
            prediction=normal_prediction,
            component=component,
            edge_usable_override=common,
        ),
        _component_edge_only_prediction(
            prediction=deranged_prediction,
            component=component,
            edge_usable_override=common,
        ),
        _component_edge_only_prediction(
            prediction=position_prediction,
            component=component,
            edge_usable_override=common,
        ),
    )


def _fixed_view_prediction(
    *,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    edge_usable_override: torch.Tensor | None = None,
) -> CandidatePoseRGBSpatialIdentityLLREdgePrediction:
    """Expose one combined edge score under immutable runtime view mass.

    Broad observation pairs supervise candidate identity only.  The learned
    support-view and null heads are frozen there, but this explicit adapter
    keeps a future refactor from accidentally comparing a combined score with
    learned view routing against a source expert evaluated with fixed routing.
    It also lets conditional source-gain diagnostics use the exact same
    normal/deranged/position availability intersection as both experts.
    """

    values = torch.as_tensor(prediction.edge_log_likelihood_ratios)
    usable = torch.as_tensor(prediction.edge_usable, dtype=torch.bool, device=values.device)
    if edge_usable_override is not None:
        override = torch.as_tensor(
            edge_usable_override,
            dtype=torch.bool,
            device=values.device,
        )
        if override.shape != usable.shape:
            raise ValueError("identity LLR fixed-view availability override is incompatible")
        usable = usable & override
    if values.shape != usable.shape:
        raise ValueError("identity LLR fixed-view edge shape is invalid")
    return CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=values,
        edge_usable=usable,
        support_view_logits=torch.zeros_like(values),
        point_null_log_likelihood_ratios=torch.zeros(
            (len(values),), dtype=values.dtype, device=values.device
        ),
        rgb_edge_usable=prediction.rgb_edge_usable & usable,
        context_edge_usable=prediction.context_edge_usable & usable,
    )


def _detached_edge_prediction(
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
) -> CandidatePoseRGBSpatialIdentityLLREdgePrediction:
    """Keep a target-free normal-score reference without retaining its graph.

    Paired source ablation is a regularizer on the ablated branch: normal
    candidate discrimination is already optimized by the primary identity and
    coherent-wrong objectives.  Detaching the normal half before the extra
    source forward prevents five high-resolution autograd graphs from
    accumulating in one backward pass on 24 GB GPUs.
    """

    values = prediction.edge_log_likelihood_ratios.detach()
    usable = prediction.edge_usable.detach()
    return CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=values,
        edge_usable=usable,
        support_view_logits=torch.zeros_like(values),
        point_null_log_likelihood_ratios=torch.zeros(
            (len(values),), dtype=values.dtype, device=values.device
        ),
        rgb_edge_usable=prediction.rgb_edge_usable.detach(),
        context_edge_usable=prediction.context_edge_usable.detach(),
    )


def _combined_component_gain_statistics(
    *,
    combined_margin: torch.Tensor,
    combined_top1: torch.Tensor,
    combined_active: torch.Tensor,
    component_margin: torch.Tensor,
    component_top1: torch.Tensor,
    component_active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure whether the joint likelihood improves over one source factor.

    The value is joined to training labels only after both target-free scores
    exist.  It is a diagnostic rather than a claim that either individual
    source must solve the entire fixed 20-way candidate set on its own.
    """

    combined_margin = torch.as_tensor(combined_margin)
    component_margin = torch.as_tensor(
        component_margin,
        dtype=combined_margin.dtype,
        device=combined_margin.device,
    )
    combined_top1 = torch.as_tensor(
        combined_top1,
        dtype=torch.bool,
        device=combined_margin.device,
    )
    component_top1 = torch.as_tensor(
        component_top1,
        dtype=torch.bool,
        device=combined_margin.device,
    )
    combined_active = torch.as_tensor(
        combined_active,
        dtype=torch.bool,
        device=combined_margin.device,
    )
    component_active = torch.as_tensor(
        component_active,
        dtype=torch.bool,
        device=combined_margin.device,
    )
    if (
        combined_margin.shape != component_margin.shape
        or combined_top1.shape != combined_margin.shape
        or component_top1.shape != combined_margin.shape
        or combined_active.shape != combined_margin.shape
        or component_active.shape != combined_margin.shape
    ):
        raise ValueError("identity LLR conditional source-gain tensors are incompatible")
    active = combined_active & component_active
    margin_gain = combined_margin - component_margin
    top1_gain = combined_top1.to(dtype=combined_margin.dtype) - component_top1.to(
        dtype=combined_margin.dtype
    )
    combined_margin_wins = margin_gain > 0.0
    return margin_gain, top1_gain, combined_margin_wins, active


def _component_visual_ablation_scales(component: str) -> dict[str, float]:
    """Remove one source's appearance while retaining all fixed geometry.

    RGB is one source factor.  RADIO-final, RADIO-intermediate, and ALIKE are
    the single multiscale context factor, so they are ablated together.  The
    returned mapping is consumed only by the target-free visual forward path.
    """

    names = tuple(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES)
    if str(component) == "rgb_identity":
        return {name: float(name != "rgb") for name in names}
    if str(component) == "context_coherence":
        return {name: float(name == "rgb") for name in names}
    raise ValueError("identity LLR visual-ablation component is invalid")


def _conditional_visual_ablation_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    normal_prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    source_ablated_prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    targets: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train a source to add conditional visual candidate discrimination.

    Both predictions see identical point/candidate/view coordinates.  Only one
    source's visual content is zeroed in the paired prediction.  The loss is
    therefore insensitive to a source-specific constant score shift and only
    rewards an increase in correct-versus-hardest-wrong candidate margin.
    """

    common = (
        torch.as_tensor(normal_prediction.edge_usable, dtype=torch.bool)
        & torch.as_tensor(source_ablated_prediction.edge_usable, dtype=torch.bool)
    )
    normal_fixed = _fixed_view_prediction(
        prediction=normal_prediction,
        edge_usable_override=common,
    )
    ablated_fixed = _fixed_view_prediction(
        prediction=source_ablated_prediction,
        edge_usable_override=common,
    )
    _unused, normal_margin, normal_top1, normal_active = _row_identity_statistics(
        runtime=runtime,
        prediction=normal_fixed,
        targets=targets,
    )
    _unused, ablated_margin, ablated_top1, ablated_active = _row_identity_statistics(
        runtime=runtime,
        prediction=ablated_fixed,
        targets=targets,
    )
    gain, top1_gain, _wins, active = _combined_component_gain_statistics(
        combined_margin=normal_margin,
        combined_top1=normal_top1,
        combined_active=normal_active,
        component_margin=ablated_margin,
        component_top1=ablated_top1,
        component_active=ablated_active,
    )
    if not bool(active.any()):
        zero = normal_prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "active_rows": 0.0,
            "mean_gap": 0.0,
            "win_fraction": 0.0,
            "top1_delta": 0.0,
        }
    active_gain = gain[active]
    loss = F.softplus(torch.as_tensor(float(margin), device=gain.device) - active_gain).mean()
    return loss, {
        "active_rows": float(active.sum().item()),
        "mean_gap": float(active_gain.detach().mean().item()),
        "win_fraction": float((active_gain.detach() > 0.0).float().mean().item()),
        "top1_delta": float(top1_gain[active].detach().mean().item()),
    }


def _conditional_visual_ablation_hard_pose_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    normal_prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    source_ablated_prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    targets: torch.Tensor,
    hard_negative_mask: torch.Tensor | None,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train the paired source factor on correct-versus-coherent-wrong gaps."""

    if hard_negative_mask is None:
        zero = normal_prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {"active_rows": 0.0, "mean_gap": 0.0, "win_fraction": 0.0}
    common = (
        torch.as_tensor(normal_prediction.edge_usable, dtype=torch.bool)
        & torch.as_tensor(source_ablated_prediction.edge_usable, dtype=torch.bool)
    )
    normal_fixed = _fixed_view_prediction(
        prediction=normal_prediction,
        edge_usable_override=common,
    )
    ablated_fixed = _fixed_view_prediction(
        prediction=source_ablated_prediction,
        edge_usable_override=common,
    )
    normal_values, normal_usable = _candidate_values(runtime=runtime, prediction=normal_fixed)
    ablated_values, ablated_usable = _candidate_values(runtime=runtime, prediction=ablated_fixed)
    normal_gap, normal_win, normal_active = _hard_pose_identity_statistics(
        candidate_values=normal_values,
        candidate_usable=normal_usable,
        positive_targets=targets,
        hard_negative_mask=hard_negative_mask,
    )
    ablated_gap, ablated_win, ablated_active = _hard_pose_identity_statistics(
        candidate_values=ablated_values,
        candidate_usable=ablated_usable,
        positive_targets=targets,
        hard_negative_mask=hard_negative_mask,
    )
    gain, _top1_gain, _wins, active = _combined_component_gain_statistics(
        combined_margin=normal_gap,
        combined_top1=normal_win,
        combined_active=normal_active,
        component_margin=ablated_gap,
        component_top1=ablated_win,
        component_active=ablated_active,
    )
    if not bool(active.any()):
        zero = normal_prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {"active_rows": 0.0, "mean_gap": 0.0, "win_fraction": 0.0}
    active_gain = gain[active]
    loss = F.softplus(torch.as_tensor(float(margin), device=gain.device) - active_gain).mean()
    return loss, {
        "active_rows": float(active.sum().item()),
        "mean_gap": float(active_gain.detach().mean().item()),
        "win_fraction": float((active_gain.detach() > 0.0).float().mean().item()),
    }


def _identity_pretrain_gate(
    *, metrics: Mapping[str, float], args: argparse.Namespace, hard_pose_enabled: bool
) -> dict[str, object]:
    """Gate the joint likelihood and paired source-content contributions."""

    result = conditional_source_visual_ablation_gate(
        metrics,
        minimum_win_fraction=float(args.minimum_win_fraction),
        minimum_normal_margin=float(args.minimum_normal_margin),
        minimum_support_visual_gap=float(args.minimum_support_visual_gap),
        minimum_position_visual_gap=float(args.minimum_position_visual_gap),
        minimum_support_correct_score_gap=float(args.minimum_support_correct_score_gap),
        minimum_position_correct_score_gap=float(args.minimum_position_correct_score_gap),
        minimum_source_margin_gain=float(args.minimum_conditional_visual_ablation_margin),
        minimum_source_margin_win_fraction=float(
            args.minimum_conditional_visual_ablation_win_fraction
        ),
        minimum_source_top1_delta=float(args.minimum_conditional_visual_ablation_top1_delta),
    )
    if not hard_pose_enabled:
        return result
    hard_gap = float(metrics["hard_pose_mean_gap"])
    hard_support = float(metrics["hard_pose_support_permuted_mean_gap"])
    hard_position = float(metrics["hard_pose_position_only_mean_gap"])
    checks = dict(result["checks"])
    checks.update(
        {
            "hard_pose_coverage": float(metrics["hard_pose_eligible_query_fraction"])
            >= float(args.minimum_hard_pose_eligible_query_fraction),
            "hard_pose_margin": hard_gap >= float(args.minimum_hard_pose_gap),
            "hard_pose_win_fraction": float(metrics["hard_pose_win_fraction"])
            >= float(args.minimum_hard_pose_win_fraction),
            "hard_pose_support_visual_gap": hard_gap - hard_support
            >= float(args.minimum_hard_pose_visual_gap),
            "hard_pose_position_visual_gap": hard_gap - hard_position
            >= float(args.minimum_hard_pose_visual_gap),
        }
    )
    source_hard: dict[str, dict[str, float | bool]] = {}
    for component in ("rgb_identity", "context_coherence"):
        gap_name = (
            f"{component}_conditional_visual_content_minus_ablated_hard_pose_mean_gap"
        )
        win_name = f"{component}_conditional_visual_content_hard_pose_gap_win_fraction"
        if gap_name not in metrics or win_name not in metrics:
            raise ValueError(f"{component} conditional visual-ablation hard-pose metrics are incomplete")
        source_gap = float(metrics[gap_name])
        source_win = float(metrics[win_name])
        source_checks = {
            "hard_pose_gap": source_gap
            >= float(args.minimum_conditional_visual_ablation_hard_pose_gap),
            "hard_pose_win_fraction": source_win
            >= float(args.minimum_conditional_visual_ablation_hard_pose_win_fraction),
        }
        checks.update(
            {f"{component}_conditional_{name}": value for name, value in source_checks.items()}
        )
        source_hard[component] = {
            "mean_gap": source_gap,
            "win_fraction": source_win,
            **source_checks,
        }
    return {
        **result,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "hard_pose_minus_support_permuted_gap": float(hard_gap - hard_support),
        "hard_pose_minus_position_only_gap": float(hard_gap - hard_position),
        "conditional_source_visual_ablation_hard_pose": source_hard,
    }


def _is_better(
    *,
    candidate: Mapping[str, float],
    incumbent: Mapping[str, float] | None,
    candidate_passed: bool,
    incumbent_passed: bool,
    hard_pose_enabled: bool,
) -> bool:
    if incumbent is None:
        return True

    def expert_key(metrics: Mapping[str, float], component: str) -> tuple[float, ...]:
        return (
            float(metrics[f"{component}_conditional_visual_content_minus_ablated_mean_margin"]),
            float(metrics[f"{component}_conditional_visual_content_margin_win_fraction"]),
            float(metrics[f"{component}_conditional_visual_content_minus_ablated_top1_delta"]),
            float(metrics[f"{component}_conditional_combined_minus_component_mean_margin"]),
            float(metrics[f"{component}_conditional_combined_margin_win_fraction"]),
        )

    standard_key = (
        int(bool(candidate_passed)),
        float(candidate["normal_minus_position_only_correct_candidate_score"]),
        float(candidate["normal_minus_support_permuted_correct_candidate_score"]),
        float(candidate["normal_mean_margin"]) - float(candidate["position_only_mean_margin"]),
        float(candidate["normal_mean_margin"]) - float(candidate["support_permuted_mean_margin"]),
        float(candidate["normal_mean_margin"]),
        float(candidate["normal_win_fraction"]),
        *expert_key(candidate, "rgb_identity"),
        *expert_key(candidate, "context_coherence"),
    )
    standard_prior = (
        int(bool(incumbent_passed)),
        float(incumbent["normal_minus_position_only_correct_candidate_score"]),
        float(incumbent["normal_minus_support_permuted_correct_candidate_score"]),
        float(incumbent["normal_mean_margin"]) - float(incumbent["position_only_mean_margin"]),
        float(incumbent["normal_mean_margin"]) - float(incumbent["support_permuted_mean_margin"]),
        float(incumbent["normal_mean_margin"]),
        float(incumbent["normal_win_fraction"]),
        *expert_key(incumbent, "rgb_identity"),
        *expert_key(incumbent, "context_coherence"),
    )
    if not hard_pose_enabled:
        return standard_key > standard_prior
    key = (
        *standard_key,
        float(candidate["hard_pose_mean_gap"]),
        float(candidate["hard_pose_win_fraction"]),
        float(candidate["hard_pose_mean_gap"])
        - float(candidate["hard_pose_position_only_mean_gap"]),
        float(candidate["rgb_identity_conditional_visual_content_minus_ablated_hard_pose_mean_gap"]),
        float(candidate["rgb_identity_conditional_visual_content_hard_pose_gap_win_fraction"]),
        float(candidate["context_coherence_conditional_visual_content_minus_ablated_hard_pose_mean_gap"]),
        float(candidate["context_coherence_conditional_visual_content_hard_pose_gap_win_fraction"]),
    )
    prior = (
        *standard_prior,
        float(incumbent["hard_pose_mean_gap"]),
        float(incumbent["hard_pose_win_fraction"]),
        float(incumbent["hard_pose_mean_gap"])
        - float(incumbent["hard_pose_position_only_mean_gap"]),
        float(incumbent["rgb_identity_conditional_visual_content_minus_ablated_hard_pose_mean_gap"]),
        float(incumbent["rgb_identity_conditional_visual_content_hard_pose_gap_win_fraction"]),
        float(incumbent["context_coherence_conditional_visual_content_minus_ablated_hard_pose_mean_gap"]),
        float(incumbent["context_coherence_conditional_visual_content_hard_pose_gap_win_fraction"]),
    )
    return key > prior


@torch.no_grad()
def _evaluate_inner_validation(
    *,
    model: nn.Module,
    pairs: CandidatePoseRGBSpatialObservationPairs,
    rows: np.ndarray,
    image_index_by_id: Mapping[str, int],
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    radius_px: float,
    step_px: float,
    batch_size: int,
    seed: int,
    cache: TensorImageLRUCache,
    device: torch.device,
    amp_enabled: bool,
    hard_pose_lookup: Mapping[int, np.ndarray] | None = None,
) -> dict[str, float]:
    """Audit normal, support-deranged and coordinate-only candidate ranking."""

    model.eval()
    query_ids: list[str] = []
    normal_ce: list[float] = []
    normal_margin: list[float] = []
    normal_win: list[float] = []
    support_margin: list[float] = []
    position_margin: list[float] = []
    support_correct_score_gap: list[float] = []
    position_correct_score_gap: list[float] = []
    active_values: list[bool] = []
    hard_normal_gap: list[float] = []
    hard_normal_win: list[float] = []
    hard_support_gap: list[float] = []
    hard_position_gap: list[float] = []
    hard_active_values: list[bool] = []
    component_values: dict[str, dict[str, list[float] | list[bool]]] = {
        component: {
            "cross_entropy": [],
            "normal_margin": [],
            "normal_win": [],
            "support_margin": [],
            "position_margin": [],
            "support_correct_score_gap": [],
            "position_correct_score_gap": [],
            "active": [],
            "hard_normal_gap": [],
            "hard_normal_win": [],
            "hard_support_gap": [],
            "hard_position_gap": [],
            "hard_active": [],
            "conditional_margin_gain": [],
            "conditional_top1_gain": [],
            "conditional_margin_win": [],
            "conditional_active": [],
            "conditional_hard_gap_gain": [],
            "conditional_hard_win_gain": [],
            "conditional_hard_gap_win": [],
            "conditional_hard_active": [],
            "visual_ablation_margin_gain": [],
            "visual_ablation_top1_gain": [],
            "visual_ablation_margin_win": [],
            "visual_ablation_active": [],
            "visual_ablation_hard_gap_gain": [],
            "visual_ablation_hard_win_gain": [],
            "visual_ablation_hard_gap_win": [],
            "visual_ablation_hard_active": [],
        }
        for component in ("rgb_identity", "context_coherence")
    }
    for begin in range(0, len(rows), int(batch_size)):
        batch_rows = np.asarray(rows[begin : begin + int(batch_size)], dtype=np.int64)
        runtime, targets, permutations = _runtime_and_targets(
            pairs=pairs,
            rows=batch_rows,
            image_index_by_id=image_index_by_id,
            permutation_seed=int(seed) + 7919,
        )
        hard_negative_mask = _permuted_hard_pose_negative_mask(
            pairs=pairs,
            rows=batch_rows,
            permutations=permutations,
            lookup=hard_pose_lookup,
            device=device,
        )
        query_patches, support_patches = _crop_pair_rgb_patches(
            runtime=runtime,
            image_ids=image_ids,
            image_root=image_root,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=float(radius_px),
            step_px=float(step_px),
            cache=cache,
            device=device,
        )
        deranged_runtime = permute_runtime_support_appearance(runtime, shift=1)
        deranged_patches = permute_support_patch_appearance(
            runtime=deranged_runtime, support_patches=support_patches, shift=1
        )
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            normal_prediction = model(
                runtime=runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
            )
            deranged_prediction = model(
                runtime=deranged_runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=deranged_patches,
            )
            position_prediction = model(
                runtime=runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                visual_content_scale=0.0,
            )
            visual_ablated_predictions = {
                component: model(
                    runtime=runtime,
                    query_rgb_patches=query_patches,
                    support_rgb_patches=support_patches,
                    visual_source_scales=_component_visual_ablation_scales(component),
                )
                for component in ("rgb_identity", "context_coherence")
            }
            ce, normal, top1, normal_active = _row_identity_statistics(
                runtime=runtime, prediction=normal_prediction, targets=targets
            )
            _unused, support, _unused_top1, support_active = _row_identity_statistics(
                runtime=deranged_runtime, prediction=deranged_prediction, targets=targets
            )
            _unused, position, _unused_top1, position_active = _row_identity_statistics(
                runtime=runtime, prediction=position_prediction, targets=targets
            )
            labels = torch.argmax(targets.to(dtype=torch.long, device=device), dim=1)
            normal_values, normal_usable = _candidate_values(
                runtime=runtime, prediction=normal_prediction
            )
            support_values, support_usable = _candidate_values(
                runtime=deranged_runtime, prediction=deranged_prediction
            )
            position_values, position_usable = _candidate_values(
                runtime=runtime, prediction=position_prediction
            )
            row_indices = torch.arange(len(labels), device=device)
            normal_correct = normal_values[row_indices, labels]
            support_correct = support_values[row_indices, labels]
            position_correct = position_values[row_indices, labels]
            normal_hard_gap, normal_hard_win, normal_hard_active = (
                _hard_pose_identity_statistics(
                    candidate_values=normal_values,
                    candidate_usable=normal_usable,
                    positive_targets=targets,
                    hard_negative_mask=hard_negative_mask,
                )
            )
            support_hard_gap, _unused_hard_win, support_hard_active = (
                _hard_pose_identity_statistics(
                    candidate_values=support_values,
                    candidate_usable=support_usable,
                    positive_targets=targets,
                    hard_negative_mask=hard_negative_mask,
                )
            )
            position_hard_gap, _unused_hard_win, position_hard_active = (
                _hard_pose_identity_statistics(
                    candidate_values=position_values,
                    candidate_usable=position_usable,
                    positive_targets=targets,
                    hard_negative_mask=hard_negative_mask,
                )
            )
            common_control_usable = (
                torch.as_tensor(normal_prediction.edge_usable, dtype=torch.bool)
                & torch.as_tensor(deranged_prediction.edge_usable, dtype=torch.bool)
                & torch.as_tensor(position_prediction.edge_usable, dtype=torch.bool)
            )
            combined_common_prediction = _fixed_view_prediction(
                prediction=normal_prediction,
                edge_usable_override=common_control_usable,
            )
            (
                _unused_combined_common_ce,
                combined_common_margin,
                combined_common_top1,
                combined_common_active,
            ) = _row_identity_statistics(
                runtime=runtime,
                prediction=combined_common_prediction,
                targets=targets,
            )
            combined_common_values, combined_common_usable = _candidate_values(
                runtime=runtime,
                prediction=combined_common_prediction,
            )
            (
                combined_common_hard_gap,
                combined_common_hard_win,
                combined_common_hard_active,
            ) = _hard_pose_identity_statistics(
                candidate_values=combined_common_values,
                candidate_usable=combined_common_usable,
                positive_targets=targets,
                hard_negative_mask=hard_negative_mask,
            )
            component_statistics: dict[str, dict[str, torch.Tensor]] = {}
            for component in ("rgb_identity", "context_coherence"):
                component_normal, component_deranged, component_position = (
                    _source_masked_control_predictions(
                        normal_prediction=normal_prediction,
                        deranged_prediction=deranged_prediction,
                        position_prediction=position_prediction,
                        component=component,
                    )
                )
                component_ce, component_normal_margin, component_top1, component_normal_active = (
                    _row_identity_statistics(
                        runtime=runtime,
                        prediction=component_normal,
                        targets=targets,
                    )
                )
                (
                    _unused_component_ce,
                    component_support_margin,
                    _unused_component_top1,
                    component_support_active,
                ) = _row_identity_statistics(
                    runtime=deranged_runtime,
                    prediction=component_deranged,
                    targets=targets,
                )
                (
                    _unused_component_position_ce,
                    component_position_margin,
                    _unused_component_position_top1,
                    component_position_active,
                ) = _row_identity_statistics(
                    runtime=runtime,
                    prediction=component_position,
                    targets=targets,
                )
                component_normal_values, component_normal_usable = _candidate_values(
                    runtime=runtime,
                    prediction=component_normal,
                )
                component_support_values, component_support_usable = _candidate_values(
                    runtime=deranged_runtime,
                    prediction=component_deranged,
                )
                component_position_values, component_position_usable = _candidate_values(
                    runtime=runtime,
                    prediction=component_position,
                )
                component_active = (
                    component_normal_active
                    & component_support_active
                    & component_position_active
                    & component_normal_usable[row_indices, labels]
                    & component_support_usable[row_indices, labels]
                    & component_position_usable[row_indices, labels]
                )
                (
                    component_hard_normal_gap,
                    component_hard_normal_win,
                    component_hard_normal_active,
                ) = _hard_pose_identity_statistics(
                    candidate_values=component_normal_values,
                    candidate_usable=component_normal_usable,
                    positive_targets=targets,
                    hard_negative_mask=hard_negative_mask,
                )
                (
                    component_hard_support_gap,
                    _unused_component_hard_support_win,
                    component_hard_support_active,
                ) = _hard_pose_identity_statistics(
                    candidate_values=component_support_values,
                    candidate_usable=component_support_usable,
                    positive_targets=targets,
                    hard_negative_mask=hard_negative_mask,
                )
                (
                    component_hard_position_gap,
                    _unused_component_hard_position_win,
                    component_hard_position_active,
                ) = _hard_pose_identity_statistics(
                    candidate_values=component_position_values,
                    candidate_usable=component_position_usable,
                    positive_targets=targets,
                    hard_negative_mask=hard_negative_mask,
                )
                (
                    conditional_margin_gain,
                    conditional_top1_gain,
                    conditional_margin_win,
                    conditional_active,
                ) = _combined_component_gain_statistics(
                    combined_margin=combined_common_margin,
                    combined_top1=combined_common_top1,
                    combined_active=combined_common_active,
                    component_margin=component_normal_margin,
                    component_top1=component_top1,
                    component_active=component_normal_active,
                )
                (
                    conditional_hard_gap_gain,
                    conditional_hard_win_gain,
                    conditional_hard_gap_win,
                    conditional_hard_active,
                ) = _combined_component_gain_statistics(
                    combined_margin=combined_common_hard_gap,
                    combined_top1=combined_common_hard_win,
                    combined_active=combined_common_hard_active,
                    component_margin=component_hard_normal_gap,
                    component_top1=component_hard_normal_win,
                    component_active=component_hard_normal_active,
                )
                source_ablated_prediction = visual_ablated_predictions[component]
                visual_ablation_common_usable = (
                    torch.as_tensor(normal_prediction.edge_usable, dtype=torch.bool)
                    & torch.as_tensor(source_ablated_prediction.edge_usable, dtype=torch.bool)
                )
                visual_ablation_normal_prediction = _fixed_view_prediction(
                    prediction=normal_prediction,
                    edge_usable_override=visual_ablation_common_usable,
                )
                visual_ablation_prediction = _fixed_view_prediction(
                    prediction=source_ablated_prediction,
                    edge_usable_override=visual_ablation_common_usable,
                )
                (
                    _unused_visual_ablation_normal_ce,
                    visual_ablation_normal_margin,
                    visual_ablation_normal_top1,
                    visual_ablation_normal_active,
                ) = _row_identity_statistics(
                    runtime=runtime,
                    prediction=visual_ablation_normal_prediction,
                    targets=targets,
                )
                (
                    _unused_visual_ablation_ce,
                    visual_ablation_margin,
                    visual_ablation_top1,
                    visual_ablation_active,
                ) = _row_identity_statistics(
                    runtime=runtime,
                    prediction=visual_ablation_prediction,
                    targets=targets,
                )
                visual_ablation_normal_values, visual_ablation_normal_usable = _candidate_values(
                    runtime=runtime,
                    prediction=visual_ablation_normal_prediction,
                )
                visual_ablation_values, visual_ablation_usable = _candidate_values(
                    runtime=runtime,
                    prediction=visual_ablation_prediction,
                )
                (
                    visual_ablation_normal_hard_gap,
                    visual_ablation_normal_hard_win,
                    visual_ablation_normal_hard_active,
                ) = _hard_pose_identity_statistics(
                    candidate_values=visual_ablation_normal_values,
                    candidate_usable=visual_ablation_normal_usable,
                    positive_targets=targets,
                    hard_negative_mask=hard_negative_mask,
                )
                (
                    visual_ablation_hard_gap,
                    visual_ablation_hard_win,
                    visual_ablation_hard_active,
                ) = _hard_pose_identity_statistics(
                    candidate_values=visual_ablation_values,
                    candidate_usable=visual_ablation_usable,
                    positive_targets=targets,
                    hard_negative_mask=hard_negative_mask,
                )
                (
                    visual_ablation_margin_gain,
                    visual_ablation_top1_gain,
                    visual_ablation_margin_win,
                    visual_ablation_active_common,
                ) = _combined_component_gain_statistics(
                    combined_margin=visual_ablation_normal_margin,
                    combined_top1=visual_ablation_normal_top1,
                    combined_active=visual_ablation_normal_active,
                    component_margin=visual_ablation_margin,
                    component_top1=visual_ablation_top1,
                    component_active=visual_ablation_active,
                )
                (
                    visual_ablation_hard_gap_gain,
                    visual_ablation_hard_win_gain,
                    visual_ablation_hard_gap_win,
                    visual_ablation_hard_active_common,
                ) = _combined_component_gain_statistics(
                    combined_margin=visual_ablation_normal_hard_gap,
                    combined_top1=visual_ablation_normal_hard_win,
                    combined_active=visual_ablation_normal_hard_active,
                    component_margin=visual_ablation_hard_gap,
                    component_top1=visual_ablation_hard_win,
                    component_active=visual_ablation_hard_active,
                )
                component_statistics[component] = {
                    "cross_entropy": component_ce,
                    "normal_margin": component_normal_margin,
                    "normal_win": component_top1,
                    "support_margin": component_support_margin,
                    "position_margin": component_position_margin,
                    "support_correct_score_gap": component_normal_values[row_indices, labels]
                    - component_support_values[row_indices, labels],
                    "position_correct_score_gap": component_normal_values[row_indices, labels]
                    - component_position_values[row_indices, labels],
                    "active": component_active,
                    "hard_normal_gap": component_hard_normal_gap,
                    "hard_normal_win": component_hard_normal_win,
                    "hard_support_gap": component_hard_support_gap,
                    "hard_position_gap": component_hard_position_gap,
                    "hard_active": (
                        component_hard_normal_active
                        & component_hard_support_active
                        & component_hard_position_active
                    ),
                    "conditional_margin_gain": conditional_margin_gain,
                    "conditional_top1_gain": conditional_top1_gain,
                    "conditional_margin_win": conditional_margin_win,
                    "conditional_active": conditional_active,
                    "conditional_hard_gap_gain": conditional_hard_gap_gain,
                    "conditional_hard_win_gain": conditional_hard_win_gain,
                    "conditional_hard_gap_win": conditional_hard_gap_win,
                    "conditional_hard_active": conditional_hard_active,
                    "visual_ablation_margin_gain": visual_ablation_margin_gain,
                    "visual_ablation_top1_gain": visual_ablation_top1_gain,
                    "visual_ablation_margin_win": visual_ablation_margin_win,
                    "visual_ablation_active": visual_ablation_active_common,
                    "visual_ablation_hard_gap_gain": visual_ablation_hard_gap_gain,
                    "visual_ablation_hard_win_gain": visual_ablation_hard_win_gain,
                    "visual_ablation_hard_gap_win": visual_ablation_hard_gap_win,
                    "visual_ablation_hard_active": visual_ablation_hard_active_common,
                }
        active = normal_active & support_active & position_active
        active &= (
            normal_usable[row_indices, labels]
            & support_usable[row_indices, labels]
            & position_usable[row_indices, labels]
        )
        query_ids.extend(pairs.query_image_ids[batch_rows].tolist())
        normal_ce.extend(ce.detach().float().cpu().numpy().tolist())
        normal_margin.extend(normal.detach().float().cpu().numpy().tolist())
        normal_win.extend(top1.detach().float().cpu().numpy().tolist())
        support_margin.extend(support.detach().float().cpu().numpy().tolist())
        position_margin.extend(position.detach().float().cpu().numpy().tolist())
        support_correct_score_gap.extend(
            (normal_correct - support_correct).detach().float().cpu().numpy().tolist()
        )
        position_correct_score_gap.extend(
            (normal_correct - position_correct).detach().float().cpu().numpy().tolist()
        )
        active_values.extend(active.detach().cpu().numpy().astype(bool).tolist())
        hard_active = normal_hard_active & support_hard_active & position_hard_active
        hard_normal_gap.extend(normal_hard_gap.detach().float().cpu().numpy().tolist())
        hard_normal_win.extend(normal_hard_win.detach().float().cpu().numpy().tolist())
        hard_support_gap.extend(support_hard_gap.detach().float().cpu().numpy().tolist())
        hard_position_gap.extend(position_hard_gap.detach().float().cpu().numpy().tolist())
        hard_active_values.extend(hard_active.detach().cpu().numpy().astype(bool).tolist())
        for component, statistics in component_statistics.items():
            component_values[component]["cross_entropy"].extend(
                statistics["cross_entropy"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["normal_margin"].extend(
                statistics["normal_margin"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["normal_win"].extend(
                statistics["normal_win"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["support_margin"].extend(
                statistics["support_margin"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["position_margin"].extend(
                statistics["position_margin"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["support_correct_score_gap"].extend(
                statistics["support_correct_score_gap"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["position_correct_score_gap"].extend(
                statistics["position_correct_score_gap"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["active"].extend(
                statistics["active"].detach().cpu().numpy().astype(bool).tolist()
            )
            component_values[component]["hard_normal_gap"].extend(
                statistics["hard_normal_gap"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["hard_normal_win"].extend(
                statistics["hard_normal_win"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["hard_support_gap"].extend(
                statistics["hard_support_gap"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["hard_position_gap"].extend(
                statistics["hard_position_gap"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["hard_active"].extend(
                statistics["hard_active"].detach().cpu().numpy().astype(bool).tolist()
            )
            component_values[component]["conditional_margin_gain"].extend(
                statistics["conditional_margin_gain"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["conditional_top1_gain"].extend(
                statistics["conditional_top1_gain"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["conditional_margin_win"].extend(
                statistics["conditional_margin_win"].detach().cpu().numpy().astype(bool).tolist()
            )
            component_values[component]["conditional_active"].extend(
                statistics["conditional_active"].detach().cpu().numpy().astype(bool).tolist()
            )
            component_values[component]["conditional_hard_gap_gain"].extend(
                statistics["conditional_hard_gap_gain"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["conditional_hard_win_gain"].extend(
                statistics["conditional_hard_win_gain"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["conditional_hard_gap_win"].extend(
                statistics["conditional_hard_gap_win"].detach().cpu().numpy().astype(bool).tolist()
            )
            component_values[component]["conditional_hard_active"].extend(
                statistics["conditional_hard_active"].detach().cpu().numpy().astype(bool).tolist()
            )
            component_values[component]["visual_ablation_margin_gain"].extend(
                statistics["visual_ablation_margin_gain"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["visual_ablation_top1_gain"].extend(
                statistics["visual_ablation_top1_gain"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["visual_ablation_margin_win"].extend(
                statistics["visual_ablation_margin_win"].detach().cpu().numpy().astype(bool).tolist()
            )
            component_values[component]["visual_ablation_active"].extend(
                statistics["visual_ablation_active"].detach().cpu().numpy().astype(bool).tolist()
            )
            component_values[component]["visual_ablation_hard_gap_gain"].extend(
                statistics["visual_ablation_hard_gap_gain"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["visual_ablation_hard_win_gain"].extend(
                statistics["visual_ablation_hard_win_gain"].detach().float().cpu().numpy().tolist()
            )
            component_values[component]["visual_ablation_hard_gap_win"].extend(
                statistics["visual_ablation_hard_gap_win"].detach().cpu().numpy().astype(bool).tolist()
            )
            component_values[component]["visual_ablation_hard_active"].extend(
                statistics["visual_ablation_hard_active"].detach().cpu().numpy().astype(bool).tolist()
            )
    active = np.asarray(active_values, dtype=bool)
    if len(active) == 0 or not np.any(active):
        raise RuntimeError("identity observation inner validation has no usable rows")
    grouped = [
        _query_grouped_means(query_ids=query_ids, values=np.asarray(values), active=active)
        for values in (
            normal_ce,
            normal_margin,
            normal_win,
            support_margin,
            position_margin,
            support_correct_score_gap,
            position_correct_score_gap,
        )
    ]
    if not all(len(values) == len(grouped[0]) and len(values) > 0 for values in grouped):
        raise RuntimeError("identity observation query-grouped validation is incomplete")
    component_metrics: dict[str, float] = {}
    for component, values in component_values.items():
        component_active = np.asarray(values["active"], dtype=bool)
        if len(component_active) != len(query_ids) or not np.any(component_active):
            raise RuntimeError(f"{component} observation inner validation has no usable rows")
        component_grouped = [
            _query_grouped_means(
                query_ids=query_ids,
                values=np.asarray(values[field]),
                active=component_active,
            )
            for field in (
                "cross_entropy",
                "normal_margin",
                "normal_win",
                "support_margin",
                "position_margin",
                "support_correct_score_gap",
                "position_correct_score_gap",
            )
        ]
        if not all(
            len(group) == len(component_grouped[0]) and len(group) > 0
            for group in component_grouped
        ):
            raise RuntimeError(f"{component} query-grouped validation is incomplete")
        component_metrics.update(
            {
                f"{component}_query_count": float(len(component_grouped[0])),
                f"{component}_active_row_count": float(component_active.sum()),
                f"{component}_normal_cross_entropy": float(np.mean(component_grouped[0])),
                f"{component}_normal_mean_margin": float(np.mean(component_grouped[1])),
                f"{component}_normal_win_fraction": float(np.mean(component_grouped[2])),
                f"{component}_support_permuted_mean_margin": float(
                    np.mean(component_grouped[3])
                ),
                f"{component}_position_only_mean_margin": float(
                    np.mean(component_grouped[4])
                ),
                f"{component}_normal_minus_support_permuted_correct_candidate_score": float(
                    np.mean(component_grouped[5])
                ),
                f"{component}_normal_minus_position_only_correct_candidate_score": float(
                    np.mean(component_grouped[6])
                ),
            }
        )
        visual_ablation_active = np.asarray(values["visual_ablation_active"], dtype=bool)
        if len(visual_ablation_active) != len(query_ids) or not np.any(visual_ablation_active):
            raise RuntimeError(f"{component} visual-ablation audit has no usable rows")
        visual_ablation_grouped = [
            _query_grouped_means(
                query_ids=query_ids,
                values=np.asarray(values[field]),
                active=visual_ablation_active,
            )
            for field in (
                "visual_ablation_margin_gain",
                "visual_ablation_top1_gain",
                "visual_ablation_margin_win",
            )
        ]
        if not all(
            len(group) == len(visual_ablation_grouped[0]) and len(group) > 0
            for group in visual_ablation_grouped
        ):
            raise RuntimeError(f"{component} visual-ablation validation is incomplete")
        component_metrics.update(
            {
                f"{component}_conditional_visual_ablation_query_count": float(
                    len(visual_ablation_grouped[0])
                ),
                f"{component}_conditional_visual_ablation_active_row_count": float(
                    visual_ablation_active.sum()
                ),
                f"{component}_conditional_visual_content_minus_ablated_mean_margin": float(
                    np.mean(visual_ablation_grouped[0])
                ),
                f"{component}_conditional_visual_content_minus_ablated_top1_delta": float(
                    np.mean(visual_ablation_grouped[1])
                ),
                f"{component}_conditional_visual_content_margin_win_fraction": float(
                    np.mean(visual_ablation_grouped[2])
                ),
            }
        )
        conditional_active = np.asarray(values["conditional_active"], dtype=bool)
        if len(conditional_active) != len(query_ids) or not np.any(conditional_active):
            raise RuntimeError(f"{component} conditional source-gain audit has no usable rows")
        conditional_grouped = [
            _query_grouped_means(
                query_ids=query_ids,
                values=np.asarray(values[field]),
                active=conditional_active,
            )
            for field in (
                "conditional_margin_gain",
                "conditional_top1_gain",
                "conditional_margin_win",
            )
        ]
        if not all(
            len(group) == len(conditional_grouped[0]) and len(group) > 0
            for group in conditional_grouped
        ):
            raise RuntimeError(f"{component} conditional source-gain validation is incomplete")
        component_metrics.update(
            {
                f"{component}_conditional_combined_query_count": float(
                    len(conditional_grouped[0])
                ),
                f"{component}_conditional_combined_active_row_count": float(
                    conditional_active.sum()
                ),
                f"{component}_conditional_combined_minus_component_mean_margin": float(
                    np.mean(conditional_grouped[0])
                ),
                f"{component}_conditional_combined_minus_component_top1_delta": float(
                    np.mean(conditional_grouped[1])
                ),
                f"{component}_conditional_combined_margin_win_fraction": float(
                    np.mean(conditional_grouped[2])
                ),
            }
        )
    result = {
        "query_count": float(len(grouped[0])),
        "active_row_count": float(active.sum()),
        "normal_cross_entropy": float(np.mean(grouped[0])),
        "normal_mean_margin": float(np.mean(grouped[1])),
        "normal_win_fraction": float(np.mean(grouped[2])),
        "support_permuted_mean_margin": float(np.mean(grouped[3])),
        "position_only_mean_margin": float(np.mean(grouped[4])),
        "normal_minus_support_permuted_correct_candidate_score": float(np.mean(grouped[5])),
        "normal_minus_position_only_correct_candidate_score": float(np.mean(grouped[6])),
        **component_metrics,
    }
    component_hard_metrics: dict[str, float] = {}
    for component, values in component_values.items():
        component_hard_active = np.asarray(values["hard_active"], dtype=bool)
        if hard_pose_lookup is None:
            component_hard_metrics.update(
                {
                    f"{component}_hard_pose_query_count": 0.0,
                    f"{component}_hard_pose_eligible_query_fraction": 0.0,
                    f"{component}_hard_pose_active_row_count": 0.0,
                    f"{component}_hard_pose_mean_gap": 0.0,
                    f"{component}_hard_pose_win_fraction": 0.0,
                    f"{component}_hard_pose_support_permuted_mean_gap": 0.0,
                    f"{component}_hard_pose_position_only_mean_gap": 0.0,
                    f"{component}_conditional_combined_hard_pose_query_count": 0.0,
                    f"{component}_conditional_combined_hard_pose_active_row_count": 0.0,
                    f"{component}_conditional_combined_minus_component_hard_pose_mean_gap": 0.0,
                    f"{component}_conditional_combined_minus_component_hard_pose_win_delta": 0.0,
                    f"{component}_conditional_combined_hard_pose_gap_win_fraction": 0.0,
                    f"{component}_conditional_visual_ablation_hard_pose_query_count": 0.0,
                    f"{component}_conditional_visual_ablation_hard_pose_active_row_count": 0.0,
                    f"{component}_conditional_visual_content_minus_ablated_hard_pose_mean_gap": 0.0,
                    f"{component}_conditional_visual_content_minus_ablated_hard_pose_win_delta": 0.0,
                    f"{component}_conditional_visual_content_hard_pose_gap_win_fraction": 0.0,
                }
            )
            continue
        if len(component_hard_active) != len(query_ids) or not np.any(component_hard_active):
            raise RuntimeError(f"{component} hard-pose identity inner validation has no usable rows")
        component_hard_grouped = [
            _query_grouped_means(
                query_ids=query_ids,
                values=np.asarray(values[field]),
                active=component_hard_active,
            )
            for field in (
                "hard_normal_gap",
                "hard_normal_win",
                "hard_support_gap",
                "hard_position_gap",
            )
        ]
        if not all(
            len(group) == len(component_hard_grouped[0]) and len(group) > 0
            for group in component_hard_grouped
        ):
            raise RuntimeError(f"{component} hard-pose query-grouped validation is incomplete")
        component_hard_metrics.update(
            {
                f"{component}_hard_pose_query_count": float(len(component_hard_grouped[0])),
                f"{component}_hard_pose_eligible_query_fraction": float(
                    len(component_hard_grouped[0]) / len(grouped[0])
                ),
                f"{component}_hard_pose_active_row_count": float(component_hard_active.sum()),
                f"{component}_hard_pose_mean_gap": float(np.mean(component_hard_grouped[0])),
                f"{component}_hard_pose_win_fraction": float(
                    np.mean(component_hard_grouped[1])
                ),
                f"{component}_hard_pose_support_permuted_mean_gap": float(
                    np.mean(component_hard_grouped[2])
                ),
                f"{component}_hard_pose_position_only_mean_gap": float(
                    np.mean(component_hard_grouped[3])
                ),
            }
        )
        visual_ablation_hard_active = np.asarray(
            values["visual_ablation_hard_active"], dtype=bool
        )
        if len(visual_ablation_hard_active) != len(query_ids) or not np.any(
            visual_ablation_hard_active
        ):
            raise RuntimeError(f"{component} visual-ablation hard-pose audit has no usable rows")
        visual_ablation_hard_grouped = [
            _query_grouped_means(
                query_ids=query_ids,
                values=np.asarray(values[field]),
                active=visual_ablation_hard_active,
            )
            for field in (
                "visual_ablation_hard_gap_gain",
                "visual_ablation_hard_win_gain",
                "visual_ablation_hard_gap_win",
            )
        ]
        if not all(
            len(group) == len(visual_ablation_hard_grouped[0]) and len(group) > 0
            for group in visual_ablation_hard_grouped
        ):
            raise RuntimeError(f"{component} visual-ablation hard-pose validation is incomplete")
        component_hard_metrics.update(
            {
                f"{component}_conditional_visual_ablation_hard_pose_query_count": float(
                    len(visual_ablation_hard_grouped[0])
                ),
                f"{component}_conditional_visual_ablation_hard_pose_active_row_count": float(
                    visual_ablation_hard_active.sum()
                ),
                f"{component}_conditional_visual_content_minus_ablated_hard_pose_mean_gap": float(
                    np.mean(visual_ablation_hard_grouped[0])
                ),
                f"{component}_conditional_visual_content_minus_ablated_hard_pose_win_delta": float(
                    np.mean(visual_ablation_hard_grouped[1])
                ),
                f"{component}_conditional_visual_content_hard_pose_gap_win_fraction": float(
                    np.mean(visual_ablation_hard_grouped[2])
                ),
            }
        )
        conditional_hard_active = np.asarray(values["conditional_hard_active"], dtype=bool)
        if len(conditional_hard_active) != len(query_ids) or not np.any(conditional_hard_active):
            raise RuntimeError(f"{component} conditional hard-pose audit has no usable rows")
        conditional_hard_grouped = [
            _query_grouped_means(
                query_ids=query_ids,
                values=np.asarray(values[field]),
                active=conditional_hard_active,
            )
            for field in (
                "conditional_hard_gap_gain",
                "conditional_hard_win_gain",
                "conditional_hard_gap_win",
            )
        ]
        if not all(
            len(group) == len(conditional_hard_grouped[0]) and len(group) > 0
            for group in conditional_hard_grouped
        ):
            raise RuntimeError(f"{component} conditional hard-pose validation is incomplete")
        component_hard_metrics.update(
            {
                f"{component}_conditional_combined_hard_pose_query_count": float(
                    len(conditional_hard_grouped[0])
                ),
                f"{component}_conditional_combined_hard_pose_active_row_count": float(
                    conditional_hard_active.sum()
                ),
                f"{component}_conditional_combined_minus_component_hard_pose_mean_gap": float(
                    np.mean(conditional_hard_grouped[0])
                ),
                f"{component}_conditional_combined_minus_component_hard_pose_win_delta": float(
                    np.mean(conditional_hard_grouped[1])
                ),
                f"{component}_conditional_combined_hard_pose_gap_win_fraction": float(
                    np.mean(conditional_hard_grouped[2])
                ),
            }
        )
    hard_active = np.asarray(hard_active_values, dtype=bool)
    if hard_pose_lookup is None:
        return {
            **result,
            "hard_pose_query_count": 0.0,
            "hard_pose_eligible_query_fraction": 0.0,
            "hard_pose_active_row_count": 0.0,
            "hard_pose_mean_gap": 0.0,
            "hard_pose_win_fraction": 0.0,
            "hard_pose_support_permuted_mean_gap": 0.0,
            "hard_pose_position_only_mean_gap": 0.0,
            **component_hard_metrics,
        }
    if len(hard_active) != len(query_ids) or not np.any(hard_active):
        raise RuntimeError("hard-pose identity inner validation has no usable rows")
    hard_grouped = [
        _query_grouped_means(query_ids=query_ids, values=np.asarray(values), active=hard_active)
        for values in (
            hard_normal_gap,
            hard_normal_win,
            hard_support_gap,
            hard_position_gap,
        )
    ]
    if not all(len(values) == len(hard_grouped[0]) and len(values) > 0 for values in hard_grouped):
        raise RuntimeError("hard-pose identity query-grouped validation is incomplete")
    return {
        **result,
        "hard_pose_query_count": float(len(hard_grouped[0])),
        "hard_pose_eligible_query_fraction": float(len(hard_grouped[0]) / len(grouped[0])),
        "hard_pose_active_row_count": float(hard_active.sum()),
        "hard_pose_mean_gap": float(np.mean(hard_grouped[0])),
        "hard_pose_win_fraction": float(np.mean(hard_grouped[1])),
        "hard_pose_support_permuted_mean_gap": float(np.mean(hard_grouped[2])),
        "hard_pose_position_only_mean_gap": float(np.mean(hard_grouped[3])),
        **component_hard_metrics,
    }


def pretrain_candidate_pose_rgb_spatial_identity_llr(args: argparse.Namespace) -> dict[str, object]:
    """Fit broad RGB/context candidate identity before sparse P1 fine-tuning."""

    windows = _validate_args(args)
    state = _initialize_distributed(str(args.device))
    try:
        output_dir = Path(args.output_dir)
        if state.rank == 0:
            if output_dir.exists() and not bool(args.force):
                raise FileExistsError(f"refusing to overwrite identity observation output: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomically(
                output_dir / "progress.json",
                {"stage": "pretrain_candidate_pose_rgb_spatial_identity_llr", "status": "running", "completed_epochs": 0},
            )
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
        except AttributeError:  # pragma: no cover
            pass

        pair_path = Path(args.observation_pairs)
        pairs = load_candidate_pose_rgb_spatial_observation_pairs(pair_path)
        hard_pose_target_path = (
            Path(str(args.hard_pose_identity_targets))
            if str(args.hard_pose_identity_targets).strip()
            else None
        )
        hard_pose_lookup: dict[int, np.ndarray] | None = None
        if hard_pose_target_path is not None:
            hard_pose_lookup = _hard_pose_identity_mask_lookup(
                pairs=pairs,
                targets=load_candidate_pose_rgb_spatial_hard_pose_identity_targets(
                    hard_pose_target_path
                ),
                pair_path=pair_path,
            )
        train_rows = _limited_rows(
            np.flatnonzero(pairs.split_names == "inner_train"),
            limit=int(args.max_train_rows),
            seed=int(args.seed) + 17,
        )
        validation_rows = _limited_rows(
            np.flatnonzero(pairs.split_names == "inner_validation"),
            limit=int(args.max_validation_rows),
            seed=int(args.seed) + 29,
        )
        if len(train_rows) == 0 or len(validation_rows) == 0:
            raise ValueError("identity observation pairs lack inner train/validation rows")
        if hard_pose_lookup is not None:
            train_hard_count = sum(
                int(anchor) in hard_pose_lookup for anchor in pairs.anchor_ids[train_rows].tolist()
            )
            validation_hard_count = sum(
                int(anchor) in hard_pose_lookup
                for anchor in pairs.anchor_ids[validation_rows].tolist()
            )
            if train_hard_count == 0 or validation_hard_count == 0:
                raise ValueError("hard-pose identity sidecar has no train or validation targets")
        sources = load_context_attention_sources(
            radio_final_context_cache=Path(args.radio_final_context_cache),
            radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
            alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        image_ids, image_sizes, source_tensors = _source_table(sources)
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("identity observation pretraining requires common image dimensions")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        rgb_bridge = _validate_rgb_coordinate_bridge(
            source_metadata=sources[0].metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        image_index_by_id = {str(image_id): index for index, image_id in enumerate(image_ids.tolist())}
        pair_images = set(pairs.query_image_ids.tolist())
        pair_images.update(pairs.positive_support_image_ids.tolist())
        pair_images.update(pairs.negative_support_image_ids.reshape(-1).tolist())
        if not pair_images.issubset(image_index_by_id):
            raise ValueError("identity observation pairs reference an image absent from context sources")
        source_lineage = {
            "radio_final_context_cache_sha256": file_sha256_short(
                Path(args.radio_final_context_cache)
            ),
            "radio_intermediate_context_cache_sha256": file_sha256_short(
                Path(args.radio_intermediate_context_cache)
            ),
            "alike_spatial_context_cache_sha256": file_sha256_short(
                Path(args.alike_spatial_context_cache)
            ),
            "source_image_manifest_sha256": str(
                sources[0].metadata.get("source_image_manifest_sha256", "")
            ),
            "rgb_coordinate_bridge": rgb_bridge,
        }

        model = CandidatePoseRGBSpatialIdentityLLR(
            sources=source_tensors,
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            rgb_context_radius_px=float(args.rgb_context_radius_px),
            rgb_step_px=float(args.rgb_step_px),
            texture_feature_dim=int(args.texture_feature_dim),
            hidden_dim=int(args.hidden_dim),
            max_abs_log_ratio=float(args.max_abs_log_ratio),
            edge_chunk_size=int(args.edge_chunk_size),
            activation_checkpointing=True,
            context_windows=windows,
        )
        initialization: dict[str, object] = {"kind": "random_initialization"}
        if str(args.representation_init_checkpoint).strip():
            initialization = _load_visual_representation_initializer(
                model=model,
                path=Path(args.representation_init_checkpoint),
                candidate_count=int(pairs.negative_count + 1),
                source_lineage=source_lineage,
                windows=windows,
                args=args,
            )
        elif str(args.init_checkpoint).strip():
            initialization = _load_identity_observation_initializer(
                model=model,
                path=Path(args.init_checkpoint),
                candidate_count=int(pairs.negative_count + 1),
                source_lineage=source_lineage,
                windows=windows,
                args=args,
            )
        # Broad observation pairs supervise candidate identity only.  They do
        # not contain an independent target for learned support-view routing or
        # an explicit null posterior, so leaving those heads trainable would
        # create unused DDP parameters or a later shortcut.  P1 owns their
        # separate calibration after both visual experts pass this broad gate.
        for head_name in ("support_view_head", "null_head"):
            for parameter in getattr(model, head_name).parameters():
                parameter.requires_grad_(False)
        model = model.to(state.device)
        model_for_train: nn.Module = model
        if state.enabled:
            model_for_train = DistributedDataParallel(
                model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
                broadcast_buffers=False,
            )
        optimizer = torch.optim.AdamW(
            _identity_llr_optimizer_parameter_groups(model=model, args=args),
            weight_decay=float(args.weight_decay),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        best_metrics: dict[str, float] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = -1
        history: list[dict[str, object]] = []
        started = time.time()
        for epoch in range(int(args.epochs)):
            model_for_train.train()
            expert_before = {
                name: parameter.detach().clone()
                for name, parameter in _identity_llr_scalar_head_parameters(model_for_train).items()
                if name.startswith(("rgb_identity_head.", "context_coherence_head."))
            }
            if not expert_before:
                raise RuntimeError("independent observation experts are not trainable")
            batches = _rank_batch_rows(
                rows=train_rows,
                batch_size=int(args.batch_size),
                rank=state.rank,
                world_size=state.world_size,
                seed=int(args.seed),
                epoch=int(epoch),
            )
            totals = torch.zeros((65,), dtype=torch.float64, device=state.device)
            epoch_started = time.time()
            for step, batch_rows in enumerate(batches):
                runtime, targets, permutations = _runtime_and_targets(
                    pairs=pairs,
                    rows=batch_rows,
                    image_index_by_id=image_index_by_id,
                    permutation_seed=int(args.seed) + int(epoch) * 1000003 + int(step),
                )
                hard_negative_mask = _permuted_hard_pose_negative_mask(
                    pairs=pairs,
                    rows=batch_rows,
                    permutations=permutations,
                    lookup=hard_pose_lookup,
                    device=state.device,
                )
                query_patches, support_patches = _crop_pair_rgb_patches(
                    runtime=runtime,
                    image_ids=image_ids,
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    radius_px=float(args.rgb_context_radius_px),
                    step_px=float(args.rgb_step_px),
                    cache=cache,
                    device=state.device,
                )
                deranged_runtime = permute_runtime_support_appearance(runtime, shift=1)
                deranged_patches = permute_support_patch_appearance(
                    runtime=deranged_runtime, support_patches=support_patches, shift=1
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    prediction = model_for_train(
                        runtime=runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=support_patches,
                    )
                    rgb_prediction = _component_edge_only_prediction(
                        prediction=prediction,
                        component="rgb_identity",
                    )
                    context_prediction = _component_edge_only_prediction(
                        prediction=prediction,
                        component="context_coherence",
                    )
                    identity_loss, identity = _identity_cross_entropy_loss(
                        runtime=runtime, prediction=prediction, targets=targets
                    )
                    rgb_identity_loss, rgb_identity = _identity_cross_entropy_loss(
                        runtime=runtime,
                        prediction=rgb_prediction,
                        targets=targets,
                    )
                    context_identity_loss, context_identity = _identity_cross_entropy_loss(
                        runtime=runtime,
                        prediction=context_prediction,
                        targets=targets,
                    )
                    hard_pose_loss, hard_pose = _hard_pose_identity_margin_loss(
                        runtime=runtime,
                        prediction=prediction,
                        positive_targets=targets,
                        hard_negative_mask=hard_negative_mask,
                        margin=float(args.hard_pose_identity_margin),
                    )
                    rgb_hard_pose_loss, rgb_hard_pose = _hard_pose_identity_margin_loss(
                        runtime=runtime,
                        prediction=rgb_prediction,
                        positive_targets=targets,
                        hard_negative_mask=hard_negative_mask,
                        margin=float(args.hard_pose_identity_margin),
                    )
                    context_hard_pose_loss, context_hard_pose = _hard_pose_identity_margin_loss(
                        runtime=runtime,
                        prediction=context_prediction,
                        positive_targets=targets,
                        hard_negative_mask=hard_negative_mask,
                        margin=float(args.hard_pose_identity_margin),
                    )
                    permutation_loss = prediction.edge_log_likelihood_ratios.sum() * 0.0
                    permutation = {"active_rows": 0.0, "mean_gap": 0.0, "win_fraction": 0.0}
                    rgb_permutation_loss = prediction.edge_log_likelihood_ratios.sum() * 0.0
                    rgb_permutation = {
                        "active_rows": 0.0,
                        "mean_gap": 0.0,
                        "win_fraction": 0.0,
                    }
                    context_permutation_loss = prediction.edge_log_likelihood_ratios.sum() * 0.0
                    context_permutation = {
                        "active_rows": 0.0,
                        "mean_gap": 0.0,
                        "win_fraction": 0.0,
                    }
                    rgb_position_only_loss = prediction.edge_log_likelihood_ratios.sum() * 0.0
                    rgb_position_only = {
                        "active_rows": 0.0,
                        "mean_gap": 0.0,
                        "win_fraction": 0.0,
                    }
                    context_position_only_loss = prediction.edge_log_likelihood_ratios.sum() * 0.0
                    context_position_only = {
                        "active_rows": 0.0,
                        "mean_gap": 0.0,
                        "win_fraction": 0.0,
                    }
                    rgb_conditional_visual_ablation_loss = (
                        prediction.edge_log_likelihood_ratios.sum() * 0.0
                    )
                    rgb_conditional_visual_ablation = {
                        "active_rows": 0.0,
                        "mean_gap": 0.0,
                        "win_fraction": 0.0,
                        "top1_delta": 0.0,
                    }
                    context_conditional_visual_ablation_loss = (
                        prediction.edge_log_likelihood_ratios.sum() * 0.0
                    )
                    context_conditional_visual_ablation = {
                        "active_rows": 0.0,
                        "mean_gap": 0.0,
                        "win_fraction": 0.0,
                        "top1_delta": 0.0,
                    }
                    rgb_conditional_visual_ablation_hard_pose_loss = (
                        prediction.edge_log_likelihood_ratios.sum() * 0.0
                    )
                    rgb_conditional_visual_ablation_hard_pose = {
                        "active_rows": 0.0,
                        "mean_gap": 0.0,
                        "win_fraction": 0.0,
                    }
                    context_conditional_visual_ablation_hard_pose_loss = (
                        prediction.edge_log_likelihood_ratios.sum() * 0.0
                    )
                    context_conditional_visual_ablation_hard_pose = {
                        "active_rows": 0.0,
                        "mean_gap": 0.0,
                        "win_fraction": 0.0,
                    }
                    conditional_visual_ablation_scheduled = 0.0
                    if int(step) % int(args.support_permutation_every_n_steps) == 0:
                        deranged_prediction = model_for_train(
                            runtime=deranged_runtime,
                            query_rgb_patches=query_patches,
                            support_rgb_patches=deranged_patches,
                        )
                        permutation_loss, permutation = _support_derangement_loss(
                            runtime=runtime,
                            prediction=prediction,
                            deranged_runtime=deranged_runtime,
                            deranged_prediction=deranged_prediction,
                            targets=targets,
                            margin=float(args.support_permutation_margin),
                        )
                        rgb_common_expert_usable = (
                            candidate_pose_rgb_spatial_identity_component_edge_usable(
                                prediction=prediction, component="rgb_identity"
                            )
                            & candidate_pose_rgb_spatial_identity_component_edge_usable(
                                prediction=deranged_prediction, component="rgb_identity"
                            )
                        )
                        context_common_expert_usable = (
                            candidate_pose_rgb_spatial_identity_component_edge_usable(
                                prediction=prediction, component="context_coherence"
                            )
                            & candidate_pose_rgb_spatial_identity_component_edge_usable(
                                prediction=deranged_prediction,
                                component="context_coherence",
                            )
                        )
                        rgb_normal_permutation_prediction = _component_edge_only_prediction(
                            prediction=prediction,
                            component="rgb_identity",
                            edge_usable_override=rgb_common_expert_usable,
                        )
                        rgb_deranged_prediction = _component_edge_only_prediction(
                            prediction=deranged_prediction,
                            component="rgb_identity",
                            edge_usable_override=rgb_common_expert_usable,
                        )
                        context_normal_permutation_prediction = _component_edge_only_prediction(
                            prediction=prediction,
                            component="context_coherence",
                            edge_usable_override=context_common_expert_usable,
                        )
                        context_deranged_prediction = _component_edge_only_prediction(
                            prediction=deranged_prediction,
                            component="context_coherence",
                            edge_usable_override=context_common_expert_usable,
                        )
                        rgb_permutation_loss, rgb_permutation = _support_derangement_loss(
                            runtime=runtime,
                            prediction=rgb_normal_permutation_prediction,
                            deranged_runtime=deranged_runtime,
                            deranged_prediction=rgb_deranged_prediction,
                            targets=targets,
                            margin=float(args.support_permutation_margin),
                        )
                        context_permutation_loss, context_permutation = (
                            _support_derangement_loss(
                                runtime=runtime,
                                prediction=context_normal_permutation_prediction,
                                deranged_runtime=deranged_runtime,
                                deranged_prediction=context_deranged_prediction,
                                targets=targets,
                                margin=float(args.support_permutation_margin),
                            )
                        )
                    if int(step) % int(args.position_only_every_n_steps) == 0:
                        position_prediction = model_for_train(
                            runtime=runtime,
                            query_rgb_patches=query_patches,
                            support_rgb_patches=support_patches,
                            visual_content_scale=0.0,
                        )
                        rgb_common_position_usable = (
                            candidate_pose_rgb_spatial_identity_component_edge_usable(
                                prediction=prediction, component="rgb_identity"
                            )
                            & candidate_pose_rgb_spatial_identity_component_edge_usable(
                                prediction=position_prediction, component="rgb_identity"
                            )
                        )
                        context_common_position_usable = (
                            candidate_pose_rgb_spatial_identity_component_edge_usable(
                                prediction=prediction, component="context_coherence"
                            )
                            & candidate_pose_rgb_spatial_identity_component_edge_usable(
                                prediction=position_prediction,
                                component="context_coherence",
                            )
                        )
                        rgb_normal_position_prediction = _component_edge_only_prediction(
                            prediction=prediction,
                            component="rgb_identity",
                            edge_usable_override=rgb_common_position_usable,
                        )
                        rgb_position_prediction = _component_edge_only_prediction(
                            prediction=position_prediction,
                            component="rgb_identity",
                            edge_usable_override=rgb_common_position_usable,
                        )
                        context_normal_position_prediction = _component_edge_only_prediction(
                            prediction=prediction,
                            component="context_coherence",
                            edge_usable_override=context_common_position_usable,
                        )
                        context_position_prediction = _component_edge_only_prediction(
                            prediction=position_prediction,
                            component="context_coherence",
                            edge_usable_override=context_common_position_usable,
                        )
                        rgb_position_only_loss, rgb_position_only = _visual_content_ablation_loss(
                            runtime=runtime,
                            prediction=rgb_normal_position_prediction,
                            position_only_prediction=rgb_position_prediction,
                            targets=targets,
                            margin=float(args.position_only_margin),
                        )
                        context_position_only_loss, context_position_only = (
                            _visual_content_ablation_loss(
                                runtime=runtime,
                                prediction=context_normal_position_prediction,
                                position_only_prediction=context_position_prediction,
                                targets=targets,
                                margin=float(args.position_only_margin),
                            )
                        )
                    base_loss = (
                        float(args.identity_loss_weight) * identity_loss
                        + float(args.rgb_identity_loss_weight) * rgb_identity_loss
                        + float(args.context_coherence_loss_weight) * context_identity_loss
                        + float(args.hard_pose_identity_loss_weight) * hard_pose_loss
                        + float(args.rgb_identity_hard_pose_loss_weight) * rgb_hard_pose_loss
                        + float(args.context_coherence_hard_pose_loss_weight)
                        * context_hard_pose_loss
                        + float(args.support_permutation_loss_weight) * permutation_loss
                        + float(args.rgb_identity_support_permutation_loss_weight)
                        * rgb_permutation_loss
                        + float(args.context_coherence_support_permutation_loss_weight)
                        * context_permutation_loss
                        + float(args.rgb_identity_position_only_loss_weight)
                        * rgb_position_only_loss
                        + float(args.context_coherence_position_only_loss_weight)
                        * context_position_only_loss
                    )
                    normal_reference = _detached_edge_prediction(prediction)
                # Release normal/deranged/position graphs before constructing
                # either high-resolution source-ablation graph.  The paired
                # regularizer uses a detached normal reference; the primary
                # identity/hard-pose objectives already optimize that branch.
                scaler.scale(base_loss).backward()
                conditional_weighted_loss = base_loss.detach() * 0.0
                conditional_enabled = max(
                    float(args.rgb_identity_conditional_visual_ablation_loss_weight),
                    float(args.context_coherence_conditional_visual_ablation_loss_weight),
                    float(args.rgb_identity_conditional_visual_ablation_hard_pose_loss_weight),
                    float(args.context_coherence_conditional_visual_ablation_hard_pose_loss_weight),
                ) > 0.0
                if (
                    conditional_enabled
                    and int(step) % int(args.conditional_visual_ablation_every_n_steps) == 0
                ):
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        rgb_source_ablated_prediction = model_for_train(
                            runtime=runtime,
                            query_rgb_patches=query_patches,
                            support_rgb_patches=support_patches,
                            visual_source_scales=_component_visual_ablation_scales(
                                "rgb_identity"
                            ),
                        )
                        (
                            rgb_conditional_visual_ablation_loss,
                            rgb_conditional_visual_ablation,
                        ) = _conditional_visual_ablation_loss(
                            runtime=runtime,
                            normal_prediction=normal_reference,
                            source_ablated_prediction=rgb_source_ablated_prediction,
                            targets=targets,
                            margin=float(args.conditional_visual_ablation_margin),
                        )
                        (
                            rgb_conditional_visual_ablation_hard_pose_loss,
                            rgb_conditional_visual_ablation_hard_pose,
                        ) = _conditional_visual_ablation_hard_pose_loss(
                            runtime=runtime,
                            normal_prediction=normal_reference,
                            source_ablated_prediction=rgb_source_ablated_prediction,
                            targets=targets,
                            hard_negative_mask=hard_negative_mask,
                            margin=float(args.conditional_visual_ablation_margin),
                        )
                        rgb_conditional_weighted_loss = (
                            float(args.rgb_identity_conditional_visual_ablation_loss_weight)
                            * rgb_conditional_visual_ablation_loss
                            + float(
                                args.rgb_identity_conditional_visual_ablation_hard_pose_loss_weight
                            )
                            * rgb_conditional_visual_ablation_hard_pose_loss
                        )
                    scaler.scale(rgb_conditional_weighted_loss).backward()
                    conditional_weighted_loss = (
                        conditional_weighted_loss + rgb_conditional_weighted_loss.detach()
                    )
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        context_source_ablated_prediction = model_for_train(
                            runtime=runtime,
                            query_rgb_patches=query_patches,
                            support_rgb_patches=support_patches,
                            visual_source_scales=_component_visual_ablation_scales(
                                "context_coherence"
                            ),
                        )
                        (
                            context_conditional_visual_ablation_loss,
                            context_conditional_visual_ablation,
                        ) = _conditional_visual_ablation_loss(
                            runtime=runtime,
                            normal_prediction=normal_reference,
                            source_ablated_prediction=context_source_ablated_prediction,
                            targets=targets,
                            margin=float(args.conditional_visual_ablation_margin),
                        )
                        (
                            context_conditional_visual_ablation_hard_pose_loss,
                            context_conditional_visual_ablation_hard_pose,
                        ) = _conditional_visual_ablation_hard_pose_loss(
                            runtime=runtime,
                            normal_prediction=normal_reference,
                            source_ablated_prediction=context_source_ablated_prediction,
                            targets=targets,
                            hard_negative_mask=hard_negative_mask,
                            margin=float(args.conditional_visual_ablation_margin),
                        )
                        context_conditional_weighted_loss = (
                            float(args.context_coherence_conditional_visual_ablation_loss_weight)
                            * context_conditional_visual_ablation_loss
                            + float(
                                args.context_coherence_conditional_visual_ablation_hard_pose_loss_weight
                            )
                            * context_conditional_visual_ablation_hard_pose_loss
                        )
                    scaler.scale(context_conditional_weighted_loss).backward()
                    conditional_weighted_loss = (
                        conditional_weighted_loss + context_conditional_weighted_loss.detach()
                    )
                    conditional_visual_ablation_scheduled = 1.0
                loss = base_loss.detach() + conditional_weighted_loss
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_for_train.parameters(), float(args.gradient_clip_norm))
                scaler.step(optimizer)
                scaler.update()
                totals += torch.tensor(
                    [
                        float(loss.detach().item()),
                        float(identity_loss.detach().item()),
                        float(identity["active_rows"]),
                        float(identity["cross_entropy"]),
                        float(identity["mean_margin"]),
                        float(identity["top1_accuracy"]),
                        float(permutation_loss.detach().item()),
                        float(permutation["mean_gap"]),
                        float(permutation["win_fraction"]),
                        float(hard_pose_loss.detach().item()),
                        float(hard_pose["active_rows"]),
                        float(hard_pose["mean_gap"]),
                        float(hard_pose["win_fraction"]),
                        float(rgb_identity_loss.detach().item()),
                        float(rgb_identity["active_rows"]),
                        float(rgb_identity["cross_entropy"]),
                        float(rgb_identity["mean_margin"]),
                        float(rgb_identity["top1_accuracy"]),
                        float(context_identity_loss.detach().item()),
                        float(context_identity["active_rows"]),
                        float(context_identity["cross_entropy"]),
                        float(context_identity["mean_margin"]),
                        float(context_identity["top1_accuracy"]),
                        float(rgb_permutation_loss.detach().item()),
                        float(rgb_permutation["active_rows"]),
                        float(rgb_permutation["mean_gap"]),
                        float(rgb_permutation["win_fraction"]),
                        float(context_permutation_loss.detach().item()),
                        float(context_permutation["active_rows"]),
                        float(context_permutation["mean_gap"]),
                        float(context_permutation["win_fraction"]),
                        float(rgb_hard_pose_loss.detach().item()),
                        float(rgb_hard_pose["active_rows"]),
                        float(rgb_hard_pose["mean_gap"]),
                        float(rgb_hard_pose["win_fraction"]),
                        float(context_hard_pose_loss.detach().item()),
                        float(context_hard_pose["active_rows"]),
                        float(context_hard_pose["mean_gap"]),
                        float(context_hard_pose["win_fraction"]),
                        float(rgb_position_only_loss.detach().item()),
                        float(rgb_position_only["active_rows"]),
                        float(rgb_position_only["mean_gap"]),
                        float(rgb_position_only["win_fraction"]),
                        float(context_position_only_loss.detach().item()),
                        float(context_position_only["active_rows"]),
                        float(context_position_only["mean_gap"]),
                        float(context_position_only["win_fraction"]),
                        float(rgb_conditional_visual_ablation_loss.detach().item()),
                        float(rgb_conditional_visual_ablation["active_rows"]),
                        float(rgb_conditional_visual_ablation["mean_gap"]),
                        float(rgb_conditional_visual_ablation["win_fraction"]),
                        float(context_conditional_visual_ablation_loss.detach().item()),
                        float(context_conditional_visual_ablation["active_rows"]),
                        float(context_conditional_visual_ablation["mean_gap"]),
                        float(context_conditional_visual_ablation["win_fraction"]),
                        float(rgb_conditional_visual_ablation_hard_pose_loss.detach().item()),
                        float(rgb_conditional_visual_ablation_hard_pose["active_rows"]),
                        float(rgb_conditional_visual_ablation_hard_pose["mean_gap"]),
                        float(rgb_conditional_visual_ablation_hard_pose["win_fraction"]),
                        float(
                            context_conditional_visual_ablation_hard_pose_loss.detach().item()
                        ),
                        float(context_conditional_visual_ablation_hard_pose["active_rows"]),
                        float(context_conditional_visual_ablation_hard_pose["mean_gap"]),
                        float(context_conditional_visual_ablation_hard_pose["win_fraction"]),
                        float(conditional_visual_ablation_scheduled),
                        1.0,
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            if state.enabled:
                distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
            global_steps = float(totals[64].item())
            if global_steps <= 0.0:
                raise RuntimeError("identity observation pretraining has no global steps")
            if state.rank == 0:
                core = model_for_train.module if isinstance(model_for_train, DistributedDataParallel) else model_for_train
                rgb_identity_update_l2 = _identity_llr_expert_update_l2(
                    model=model_for_train,
                    before=expert_before,
                    expert="rgb_identity",
                )
                context_coherence_update_l2 = _identity_llr_expert_update_l2(
                    model=model_for_train,
                    before=expert_before,
                    expert="context_coherence",
                )
                if rgb_identity_update_l2 <= 0.0 or context_coherence_update_l2 <= 0.0:
                    raise RuntimeError("independently supervised broad observation expert did not update")
                validation = _evaluate_inner_validation(
                    model=core,
                    pairs=pairs,
                    rows=validation_rows,
                    image_index_by_id=image_index_by_id,
                    image_ids=image_ids,
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    radius_px=float(args.rgb_context_radius_px),
                    step_px=float(args.rgb_step_px),
                    batch_size=int(args.batch_size),
                    seed=int(args.seed),
                    cache=cache,
                    device=state.device,
                    amp_enabled=amp_enabled,
                    hard_pose_lookup=hard_pose_lookup,
                )
                gate = _identity_pretrain_gate(
                    metrics=validation,
                    args=args,
                    hard_pose_enabled=hard_pose_lookup is not None,
                )
                incumbent_gate = None if best_metrics is None else _identity_pretrain_gate(
                    metrics=best_metrics,
                    args=args,
                    hard_pose_enabled=hard_pose_lookup is not None,
                )
                record: dict[str, object] = {
                    "epoch": int(epoch + 1),
                    "epoch_seconds": float(time.time() - epoch_started),
                    "global_steps": int(global_steps),
                    "train_total_loss": float((totals[0] / global_steps).item()),
                    "train_identity_loss": float((totals[1] / global_steps).item()),
                    "train_identity_active_rows_per_step": float((totals[2] / global_steps).item()),
                    "train_identity_cross_entropy": float((totals[3] / global_steps).item()),
                    "train_identity_mean_margin": float((totals[4] / global_steps).item()),
                    "train_identity_top1_accuracy": float((totals[5] / global_steps).item()),
                    "train_support_permutation_loss": float((totals[6] / global_steps).item()),
                    "train_support_permutation_mean_gap": float((totals[7] / global_steps).item()),
                    "train_support_permutation_win_fraction": float((totals[8] / global_steps).item()),
                    "train_hard_pose_identity_loss": float((totals[9] / global_steps).item()),
                    "train_hard_pose_identity_active_rows_per_step": float(
                        (totals[10] / global_steps).item()
                    ),
                    "train_hard_pose_identity_mean_gap": float((totals[11] / global_steps).item()),
                    "train_hard_pose_identity_win_fraction": float(
                        (totals[12] / global_steps).item()
                    ),
                    "train_rgb_identity_loss": float((totals[13] / global_steps).item()),
                    "train_rgb_identity_active_rows_per_step": float(
                        (totals[14] / global_steps).item()
                    ),
                    "train_rgb_identity_cross_entropy": float(
                        (totals[15] / global_steps).item()
                    ),
                    "train_rgb_identity_mean_margin": float(
                        (totals[16] / global_steps).item()
                    ),
                    "train_rgb_identity_top1_accuracy": float(
                        (totals[17] / global_steps).item()
                    ),
                    "train_context_coherence_loss": float((totals[18] / global_steps).item()),
                    "train_context_coherence_active_rows_per_step": float(
                        (totals[19] / global_steps).item()
                    ),
                    "train_context_coherence_cross_entropy": float(
                        (totals[20] / global_steps).item()
                    ),
                    "train_context_coherence_mean_margin": float(
                        (totals[21] / global_steps).item()
                    ),
                    "train_context_coherence_top1_accuracy": float(
                        (totals[22] / global_steps).item()
                    ),
                    "train_rgb_identity_support_permutation_loss": float(
                        (totals[23] / global_steps).item()
                    ),
                    "train_rgb_identity_support_permutation_active_rows_per_step": float(
                        (totals[24] / global_steps).item()
                    ),
                    "train_rgb_identity_support_permutation_mean_gap": float(
                        (totals[25] / global_steps).item()
                    ),
                    "train_rgb_identity_support_permutation_win_fraction": float(
                        (totals[26] / global_steps).item()
                    ),
                    "train_context_coherence_support_permutation_loss": float(
                        (totals[27] / global_steps).item()
                    ),
                    "train_context_coherence_support_permutation_active_rows_per_step": float(
                        (totals[28] / global_steps).item()
                    ),
                    "train_context_coherence_support_permutation_mean_gap": float(
                        (totals[29] / global_steps).item()
                    ),
                    "train_context_coherence_support_permutation_win_fraction": float(
                        (totals[30] / global_steps).item()
                    ),
                    "train_rgb_identity_hard_pose_loss": float(
                        (totals[31] / global_steps).item()
                    ),
                    "train_rgb_identity_hard_pose_active_rows_per_step": float(
                        (totals[32] / global_steps).item()
                    ),
                    "train_rgb_identity_hard_pose_mean_gap": float(
                        (totals[33] / global_steps).item()
                    ),
                    "train_rgb_identity_hard_pose_win_fraction": float(
                        (totals[34] / global_steps).item()
                    ),
                    "train_context_coherence_hard_pose_loss": float(
                        (totals[35] / global_steps).item()
                    ),
                    "train_context_coherence_hard_pose_active_rows_per_step": float(
                        (totals[36] / global_steps).item()
                    ),
                    "train_context_coherence_hard_pose_mean_gap": float(
                        (totals[37] / global_steps).item()
                    ),
                    "train_context_coherence_hard_pose_win_fraction": float(
                        (totals[38] / global_steps).item()
                    ),
                    "train_rgb_identity_position_only_loss": float(
                        (totals[39] / global_steps).item()
                    ),
                    "train_rgb_identity_position_only_active_rows_per_step": float(
                        (totals[40] / global_steps).item()
                    ),
                    "train_rgb_identity_position_only_mean_gap": float(
                        (totals[41] / global_steps).item()
                    ),
                    "train_rgb_identity_position_only_win_fraction": float(
                        (totals[42] / global_steps).item()
                    ),
                    "train_context_coherence_position_only_loss": float(
                        (totals[43] / global_steps).item()
                    ),
                    "train_context_coherence_position_only_active_rows_per_step": float(
                        (totals[44] / global_steps).item()
                    ),
                    "train_context_coherence_position_only_mean_gap": float(
                        (totals[45] / global_steps).item()
                    ),
                    "train_context_coherence_position_only_win_fraction": float(
                        (totals[46] / global_steps).item()
                    ),
                    "train_conditional_visual_ablation_scheduled_steps": float(
                        totals[63].item()
                    ),
                    "train_rgb_identity_conditional_visual_ablation_loss": float(
                        (totals[47] / global_steps).item()
                    ),
                    "train_rgb_identity_conditional_visual_ablation_active_rows_per_step": float(
                        (totals[48] / global_steps).item()
                    ),
                    "train_rgb_identity_conditional_visual_ablation_mean_gap": float(
                        (totals[49] / global_steps).item()
                    ),
                    "train_rgb_identity_conditional_visual_ablation_win_fraction": float(
                        (totals[50] / global_steps).item()
                    ),
                    "train_context_coherence_conditional_visual_ablation_loss": float(
                        (totals[51] / global_steps).item()
                    ),
                    "train_context_coherence_conditional_visual_ablation_active_rows_per_step": float(
                        (totals[52] / global_steps).item()
                    ),
                    "train_context_coherence_conditional_visual_ablation_mean_gap": float(
                        (totals[53] / global_steps).item()
                    ),
                    "train_context_coherence_conditional_visual_ablation_win_fraction": float(
                        (totals[54] / global_steps).item()
                    ),
                    "train_rgb_identity_conditional_visual_ablation_hard_pose_loss": float(
                        (totals[55] / global_steps).item()
                    ),
                    "train_rgb_identity_conditional_visual_ablation_hard_pose_active_rows_per_step": float(
                        (totals[56] / global_steps).item()
                    ),
                    "train_rgb_identity_conditional_visual_ablation_hard_pose_mean_gap": float(
                        (totals[57] / global_steps).item()
                    ),
                    "train_rgb_identity_conditional_visual_ablation_hard_pose_win_fraction": float(
                        (totals[58] / global_steps).item()
                    ),
                    "train_context_coherence_conditional_visual_ablation_hard_pose_loss": float(
                        (totals[59] / global_steps).item()
                    ),
                    "train_context_coherence_conditional_visual_ablation_hard_pose_active_rows_per_step": float(
                        (totals[60] / global_steps).item()
                    ),
                    "train_context_coherence_conditional_visual_ablation_hard_pose_mean_gap": float(
                        (totals[61] / global_steps).item()
                    ),
                    "train_context_coherence_conditional_visual_ablation_hard_pose_win_fraction": float(
                        (totals[62] / global_steps).item()
                    ),
                    "train_rgb_identity_head_update_l2": float(rgb_identity_update_l2),
                    "train_context_coherence_head_update_l2": float(context_coherence_update_l2),
                    **_identity_llr_head_statistics(model_for_train),
                    **{f"inner_{name}": value for name, value in validation.items()},
                    "inner_gate_passed": bool(gate["passed"]),
                }
                if _is_better(
                    candidate=validation,
                    incumbent=best_metrics,
                    candidate_passed=bool(gate["passed"]),
                    incumbent_passed=bool(False if incumbent_gate is None else incumbent_gate["passed"]),
                    hard_pose_enabled=hard_pose_lookup is not None,
                ):
                    best_metrics = dict(validation)
                    best_epoch = int(epoch + 1)
                    best_state = {
                        name: value.detach().cpu().clone() for name, value in core.state_dict().items()
                    }
                history.append(record)
                _write_json_atomically(output_dir / "history.partial.json", history)
                _write_json_atomically(
                    output_dir / "progress.json",
                    {"stage": "pretrain_candidate_pose_rgb_spatial_identity_llr", "status": "running", "completed_epochs": int(epoch + 1), "last_epoch": record},
                )
                print(json.dumps(record, sort_keys=True), flush=True)
            if state.enabled:
                distributed.barrier()

        if state.rank != 0:
            return {"rank": int(state.rank)}
        if best_state is None or best_metrics is None or best_epoch < 1:
            raise RuntimeError("identity observation pretraining did not select a checkpoint")
        gate = _identity_pretrain_gate(
            metrics=best_metrics,
            args=args,
            hard_pose_enabled=hard_pose_lookup is not None,
        )
        checkpoint_path = output_dir / "candidate_pose_rgb_spatial_identity_observation_pretrain.pt"
        metadata = {
            "format": CHECKPOINT_FORMAT,
            "model_format": CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
            "architecture": "candidate_set_conditioned_additive_rgb_identity_plus_radio_alike_context_coherence_source_specific_availability_conditional_visual_ablation_observation_pretrain_v6",
            "contains_target_fields": False,
            "checkpoint_contains_train_targets": False,
            "runtime_layout_is_target_free": True,
            "pose_or_ground_truth_used_by_runtime_scorer": False,
            "render": False,
            "image_retrieval_or_submap_used": False,
            "fixed_candidate_count": int(pairs.negative_count + 1),
            "observation_pair_inner_validation": {
                "fold_count": int(pairs.metadata["inner_validation_fold_count"]),
                "fold_index": int(pairs.metadata["inner_validation_fold_index"]),
            },
            "candidate_slot_permutation_equivariant": True,
            "candidate_slots_randomized_before_target_join": True,
            "visual_evidence_gate_version": OBSERVATION_IDENTITY_GATE_VERSION,
            "position_only_control": "zero_rgb_and_frozen_descriptor_values_keep_coordinate_geometry_v1",
            "source_masked_experts": {
                "rgb_identity": "rgb_embedding_half_only_with_fixed_runtime_view_mass_v1",
                "context_coherence": "radio_alike_embedding_half_only_with_fixed_runtime_view_mass_v1",
                "combined": "sum_of_raw_independent_log_likelihood_ratios_then_bound_once_v1",
            },
            "edge_source_availability": {
                "rgb_identity": "rgb_crop_usable_only_else_neutral_llr_v1",
                "context_coherence": "full_map_radio_alike_context_usable_only_else_neutral_llr_v1",
                "combined": "union_of_independent_source_masks_no_cross_source_invalidation_v1",
            },
            "conditional_source_visual_ablation": {
                "rgb_identity": "zero_rgb_content_keep_radio_alike_and_all_fixed_geometry_v1",
                "context_coherence": "zero_radio_alike_content_keep_rgb_and_all_fixed_geometry_v1",
                "candidate_margin": "combined_correct_minus_hardest_wrong_vs_paired_source_ablated_v1",
                "hard_pose_margin": "combined_correct_minus_coherent_wrong_vs_paired_source_ablated_v1",
            },
            "broad_pretrain_frozen_auxiliary_heads": ["edge_head", "support_view_head", "null_head"],
            "diagnostic_only": True,
            "promotion_allowed": False,
            "p1_initialization_allowed": bool(gate["passed"]),
            "raw_scores_must_not_feed_pnp": True,
            "initialization": initialization,
            "direct_coherent_wrong_identity_gate_enabled": hard_pose_lookup is not None,
            "encoder_inputs": [
                "frozen_query_anchor_xy",
                "fixed_support_observation_xy",
                "full_2d_radio_final_absolute_context_crop",
                "full_2d_radio_intermediate_absolute_context_crop",
                "full_2d_alike_phase_context_crop",
                "real_rgb_query_and_support_region_patches",
            ],
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
                "rgb_context_radius_px": float(args.rgb_context_radius_px),
                "rgb_step_px": float(args.rgb_step_px),
                "texture_feature_dim": int(args.texture_feature_dim),
                "hidden_dim": int(args.hidden_dim),
                "max_abs_log_ratio": float(args.max_abs_log_ratio),
                "edge_chunk_size": int(args.edge_chunk_size),
                "context_windows": dict(windows),
            },
            "training": {
                "objective": (
                    "joint_same_track_vs_19_global_radio_pca_hard_negative_cross_entropy_"
                    "plus_combined_support_derangement_plus_per_source_score_position_"
                    "controls_plus_paired_single_source_visual_ablation_candidate_and_"
                    "coherent_wrong_margin_v5"
                    if hard_pose_lookup is not None
                    else "joint_same_track_vs_19_global_radio_pca_hard_negative_cross_entropy_"
                    "plus_combined_support_derangement_plus_per_source_score_position_"
                    "controls_plus_paired_single_source_visual_ablation_candidate_margin_v5"
                ),
                "epochs": int(args.epochs),
                "learning_rate_groups": {
                    str(group["group_name"]): float(group["lr"])
                    for group in optimizer.param_groups
                },
                "weight_decay": float(args.weight_decay),
                "identity_loss_weight": float(args.identity_loss_weight),
                "rgb_identity_loss_weight": float(args.rgb_identity_loss_weight),
                "context_coherence_loss_weight": float(args.context_coherence_loss_weight),
                "hard_pose_identity_loss_weight": float(args.hard_pose_identity_loss_weight),
                "rgb_identity_hard_pose_loss_weight": float(
                    args.rgb_identity_hard_pose_loss_weight
                ),
                "context_coherence_hard_pose_loss_weight": float(
                    args.context_coherence_hard_pose_loss_weight
                ),
                "hard_pose_identity_margin": float(args.hard_pose_identity_margin),
                "support_permutation_loss_weight": float(args.support_permutation_loss_weight),
                "rgb_identity_support_permutation_loss_weight": float(
                    args.rgb_identity_support_permutation_loss_weight
                ),
                "context_coherence_support_permutation_loss_weight": float(
                    args.context_coherence_support_permutation_loss_weight
                ),
                "rgb_identity_position_only_loss_weight": float(
                    args.rgb_identity_position_only_loss_weight
                ),
                "context_coherence_position_only_loss_weight": float(
                    args.context_coherence_position_only_loss_weight
                ),
                "rgb_identity_conditional_visual_ablation_loss_weight": float(
                    args.rgb_identity_conditional_visual_ablation_loss_weight
                ),
                "context_coherence_conditional_visual_ablation_loss_weight": float(
                    args.context_coherence_conditional_visual_ablation_loss_weight
                ),
                "rgb_identity_conditional_visual_ablation_hard_pose_loss_weight": float(
                    args.rgb_identity_conditional_visual_ablation_hard_pose_loss_weight
                ),
                "context_coherence_conditional_visual_ablation_hard_pose_loss_weight": float(
                    args.context_coherence_conditional_visual_ablation_hard_pose_loss_weight
                ),
                "support_permutation_margin": float(args.support_permutation_margin),
                "support_permutation_every_n_steps": int(args.support_permutation_every_n_steps),
                "position_only_margin": float(args.position_only_margin),
                "position_only_every_n_steps": int(args.position_only_every_n_steps),
                "conditional_visual_ablation_margin": float(
                    args.conditional_visual_ablation_margin
                ),
                "conditional_visual_ablation_every_n_steps": int(
                    args.conditional_visual_ablation_every_n_steps
                ),
                "inner_gate_thresholds": {
                    "minimum_win_fraction": float(args.minimum_win_fraction),
                    "minimum_normal_margin": float(args.minimum_normal_margin),
                    "minimum_support_visual_gap": float(args.minimum_support_visual_gap),
                    "minimum_position_visual_gap": float(args.minimum_position_visual_gap),
                    "minimum_support_correct_score_gap": float(
                        args.minimum_support_correct_score_gap
                    ),
                    "minimum_position_correct_score_gap": float(
                        args.minimum_position_correct_score_gap
                    ),
                    "minimum_hard_pose_win_fraction": float(args.minimum_hard_pose_win_fraction),
                    "minimum_hard_pose_gap": float(args.minimum_hard_pose_gap),
                    "minimum_hard_pose_visual_gap": float(args.minimum_hard_pose_visual_gap),
                    "minimum_hard_pose_eligible_query_fraction": float(
                        args.minimum_hard_pose_eligible_query_fraction
                    ),
                    "minimum_conditional_visual_ablation_margin": float(
                        args.minimum_conditional_visual_ablation_margin
                    ),
                    "minimum_conditional_visual_ablation_win_fraction": float(
                        args.minimum_conditional_visual_ablation_win_fraction
                    ),
                    "minimum_conditional_visual_ablation_top1_delta": float(
                        args.minimum_conditional_visual_ablation_top1_delta
                    ),
                    "minimum_conditional_visual_ablation_hard_pose_gap": float(
                        args.minimum_conditional_visual_ablation_hard_pose_gap
                    ),
                    "minimum_conditional_visual_ablation_hard_pose_win_fraction": float(
                        args.minimum_conditional_visual_ablation_hard_pose_win_fraction
                    ),
                },
                "inner_validation": {"selected_epoch": int(best_epoch), "metrics": best_metrics, "gate": gate},
            },
            "lineage": {
                "observation_pairs_sha256": file_sha256_short(pair_path),
                "hard_pose_identity_targets_sha256": (
                    ""
                    if hard_pose_target_path is None
                    else file_sha256_short(hard_pose_target_path)
                ),
                **source_lineage,
            },
        }
        torch.save({"format": CHECKPOINT_FORMAT, "state_dict": best_state, "metadata": metadata}, checkpoint_path)
        summary = {
            "stage": "pretrain_candidate_pose_rgb_spatial_identity_llr",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256_short(checkpoint_path),
            "initialization": initialization,
            "elapsed_seconds": float(time.time() - started),
            "history": history,
            "checkpoint_selection": {"selected_epoch": int(best_epoch), "inner_validation": best_metrics, "gate": gate},
            "rgb_cache_rank0": cache.summary(),
            "protocol": {
                "train_only_observation_targets": True,
                "runtime_checkpoint_target_free": True,
                "candidate_slots_randomized_before_target_join": True,
                "support_derangement_control": True,
                "source_masked_expert_controls": True,
                "common_availability_for_source_masked_controls": True,
                "position_only_control": True,
                "position_only_control_is_train_time_loss": True,
                "direct_coherent_wrong_identity_margin": hard_pose_lookup is not None,
                "direct_coherent_wrong_source_masked_margins": bool(
                    float(args.rgb_identity_hard_pose_loss_weight) > 0.0
                    and float(args.context_coherence_hard_pose_loss_weight) > 0.0
                ),
                "paired_single_source_visual_ablation": True,
                "paired_single_source_visual_ablation_train_time_loss": True,
                # The normal branch is optimized by the base objectives first;
                # its detached score then anchors each separately recomputed
                # source-ablation branch.  This preserves the paired objective
                # while avoiding five simultaneous autograd graphs under DDP.
                "source_ablation_normal_reference_detached_for_memory": True,
                "validation_or_test_labels_used_by_fit": False,
                "pnp_or_heldout_pose_not_run": True,
                "no_render": True,
                "no_image_retrieval_or_submap": True,
            },
        }
        _write_json_atomically(output_dir / "history.json", history)
        _write_json_atomically(output_dir / "summary.json", summary)
        _write_json_atomically(
            output_dir / "progress.json",
            {"stage": "pretrain_candidate_pose_rgb_spatial_identity_llr", "status": "complete", "completed_epochs": int(args.epochs), "selected_epoch": int(best_epoch), "inner_gate_passed": bool(gate["passed"])},
        )
        return {"checkpoint": str(checkpoint_path), "rank": int(state.rank)}
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    result = pretrain_candidate_pose_rgb_spatial_identity_llr(parse_args(argv))
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
