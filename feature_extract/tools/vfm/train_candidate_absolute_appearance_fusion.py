"""Train a target-free RADIO-final phase plus fine-RGB pose likelihood.

This trainer is intentionally a new protocol rather than a continuation of
the historical RGB spatial branch.  The runtime visual forward receives only
the frozen P1 query/support layout, full two-dimensional RADIO-final context,
and real RGB crops.  Correct and coherent-wrong pose projections, exact-track
labels, and hard-repeat identities are joined only after that forward.

The output is diagnostic-only until a query-disjoint train-only gate passes.
The gate uses fixed top-L/null mass and a frozen target-free point selector;
it reports RADIO-only, RGB-only, and conservative fused evidence under normal,
support-image-deranged, and structural-zero controls.  No held-out pose/PnP
evaluation is performed by this command.
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

from feature_extract.tools.vfm.train_candidate_highres_rgb_multiscale_likelihood import (
    appearance_control_margin_loss,
    balanced_ddp_query_schedules,
    ddp_owner_cost_balance_metrics,
    fixed_final_epoch_checkpoint_selection,
    pose_margin_terms,
    target_free_query_owner_costs,
)
from feature_extract.tools.vfm.train_candidate_multiscale_phase_identity_llr import (
    build_exact_identity_query_targets,
    filter_static_hard_repeat_groups_to_registered_exact_identity,
    validate_external_frozen_current_hard_targets,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatBatch,
    HardRepeatQueryTargets,
    TrainQueryGroup,
    _DistributedState,
    _crop_geometry_fixed_permuted_support_patches,
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _finalize_distributed,
    _hard_repeat_batch_from_group,
    _initialize_distributed,
    _partition_train_queries_for_inner_validation,
    _query_batch_from_group,
    _select_group_points,
    _stable_query_hash,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    permute_runtime_support_image_appearance_only,
    train_query_partition_manifest,
    validate_registered_identity_targets_for_geometry,
    validate_rgb_coordinate_bridge,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_absolute_appearance_fusion import (
    CANDIDATE_ABSOLUTE_APPEARANCE_FUSION_FORMAT,
    CandidateAbsoluteAppearanceEvidence,
    CandidateAbsoluteAppearanceFusion,
    phase_conditioned_candidate_posterior_margin_loss,
    phase_conditioned_candidate_posterior_nll,
)
from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
    CandidateHighresRGBMultiscaleLikelihood,
    CandidateHighresRGBMultiscalePrediction,
    highres_rgb_spatial_density_nll,
)
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_LLR_FORMAT,
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    CandidateMultiscalePhaseIdentityLLR,
    CandidateMultiscalePhaseIdentityPrediction,
    canonical_registered_identity_or_null_targets,
    current_hard_repeat_identity_margin_loss,
    exact_identity_or_null_cross_entropy,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_selector import (
    select_target_free_spatial_quota,
    selector_input_from_target_free_layout,
    target_free_selector_scores,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CandidatePoseRGBSpatialTrainingTargets,
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_source_headers,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


CHECKPOINT_FORMAT = "candidate_absolute_appearance_fusion_checkpoint_v2"
EXPERT_INITIALIZATION_CHECKPOINT_FORMATS = frozenset(
    {
        "candidate_absolute_appearance_fusion_checkpoint_v1",
        CHECKPOINT_FORMAT,
    }
)
FUSION_INNER_GATE_MANIFEST_FORMAT = "candidate_absolute_appearance_fusion_inner_gate_manifest_v2"
FUSION_INNER_GATE_MANIFEST = {
    "format": FUSION_INNER_GATE_MANIFEST_FORMAT,
    "version": "radiofinal_identity_prior_fine_rgb_fixed_topl_target_free_gate_v2",
    "query_selection": "frozen_layout_coarse_margin_spatial_quota_before_target_join_v1",
    "visual_controls": (
        "normal_real_support_image_v1",
        "geometry_fixed_support_image_derangement_recrop_v2",
        "structural_zero_visual_evidence_v1",
    ),
    "pose_diagnostic": "correct_vs_full_coherent_wrong_pool_post_forward_v1",
    "phase_conditioning_diagnostic": "paired_fused_minus_rgb_only_pose_gap_same_hypotheses_v1",
    "repeat_diagnostic": "registered_exact_static_repeat_full_target_free_p1_pool_chunked_common_visual_availability_v2",
    "fusion": "radiofinal_phase_conditions_candidate_null_prior_rgb_only_pose_projected_likelihood_v2",
    "checkpoint_selection": "fixed_final_epoch_without_inner_validation_model_selection_v1",
}


def current_fusion_inner_gate_manifest() -> dict[str, object]:
    """Return the immutable semantics used for a promotable fusion checkpoint."""

    payload = json.dumps(FUSION_INNER_GATE_MANIFEST, sort_keys=True, separators=(",", ":"))
    return {
        **FUSION_INNER_GATE_MANIFEST,
        "semantic_sha256": __import__("hashlib").sha256(payload.encode("utf-8")).hexdigest()[:16],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--geometry-training-targets", required=True)
    parser.add_argument("--registered-identity-targets", required=True)
    parser.add_argument("--current-hard-repeat-targets", required=True)
    parser.add_argument("--current-hard-mining-checkpoint", required=True)
    parser.add_argument("--static-hard-repeat-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-points-per-query", type=int, default=64)
    parser.add_argument("--max-hard-repeat-edges-per-query", type=int, default=256)
    parser.add_argument("--validation-selector-policy", choices=("coarse_margin",), default="coarse_margin")
    parser.add_argument("--validation-selector-point-budget", type=int, default=64)
    parser.add_argument("--validation-selector-grid-rows", type=int, default=4)
    parser.add_argument("--validation-selector-grid-columns", type=int, default=4)
    parser.add_argument("--fine-search-radius-px", type=float, default=12.0)
    parser.add_argument("--fine-context-radius-px", type=float, default=12.0)
    parser.add_argument("--texture-feature-dim", type=int, default=32)
    parser.add_argument("--rgb-hidden-dim", type=int, default=32)
    parser.add_argument("--rgb-edge-chunk-size", type=int, default=128)
    parser.add_argument("--rgb-temperature", type=float, default=10.0)
    parser.add_argument("--rgb-max-abs-edge-log-ratio", type=float, default=3.0)
    parser.add_argument("--phase-hidden-dim", type=int, default=96)
    parser.add_argument("--phase-max-abs-log-ratio", type=float, default=4.0)
    parser.add_argument("--phase-source-storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--phase-learning-rate", type=float, default=1e-3)
    parser.add_argument("--rgb-learning-rate", type=float, default=2e-4)
    parser.add_argument("--fusion-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--initial-radio-final-phase-prior-strength", type=float, default=1.0)
    parser.add_argument("--max-radio-final-phase-prior-strength", type=float, default=2.0)
    parser.add_argument("--density-loss-weight", type=float, default=0.25)
    parser.add_argument("--dustbin-loss-weight", type=float, default=0.50)
    parser.add_argument("--identity-loss-weight", type=float, default=0.50)
    parser.add_argument("--phase-posterior-identity-loss-weight", type=float, default=1.0)
    parser.add_argument("--current-hard-phase-loss-weight", type=float, default=0.50)
    parser.add_argument("--current-hard-posterior-loss-weight", type=float, default=1.0)
    parser.add_argument("--current-hard-fusion-loss-weight", type=float, default=0.50)
    parser.add_argument("--static-hard-fusion-loss-weight", type=float, default=0.25)
    parser.add_argument("--pose-loss-weight", type=float, default=1.0)
    parser.add_argument("--soft-hard-loss-weight", type=float, default=0.50)
    parser.add_argument("--appearance-control-loss-weight", type=float, default=0.25)
    parser.add_argument("--pose-margin", type=float, default=0.25)
    parser.add_argument("--hard-repeat-margin", type=float, default=0.25)
    parser.add_argument("--appearance-control-margin", type=float, default=0.05)
    parser.add_argument("--soft-hard-temperature", type=float, default=0.35)
    parser.add_argument("--max-abs-pose-log-ratio", type=float, default=6.0)
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=1)
    parser.add_argument("--minimum-pose-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-pose-gap", type=float, default=0.05)
    parser.add_argument("--minimum-pose-permutation-delta", type=float, default=0.05)
    parser.add_argument("--minimum-phase-conditioned-pose-lift", type=float, default=0.005)
    parser.add_argument("--minimum-hard-repeat-eligible-query-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-hard-repeat-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-hard-repeat-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-repeat-permutation-delta", type=float, default=0.05)
    parser.add_argument(
        "--gate-every-epoch",
        action="store_true",
        help="Diagnostic telemetry only; the fixed final epoch remains the selected checkpoint.",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp-init-scale", type=float, default=4096.0)
    parser.add_argument("--rgb-cache-gb", type=float, default=6.0)
    parser.add_argument("--rgb-cache-dtype", choices=("float16", "uint8"), default="uint8")
    parser.add_argument("--rgb-cache-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--expert-initialization-checkpoint",
        default="",
        help=(
            "Strictly lineage-checked phase/RGB expert initializer. The prior "
            "calibrator is always initialized under this v2 protocol."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "epochs",
        "max_points_per_query",
        "max_hard_repeat_edges_per_query",
        "validation_selector_point_budget",
        "validation_selector_grid_rows",
        "validation_selector_grid_columns",
        "texture_feature_dim",
        "rgb_hidden_dim",
        "rgb_edge_chunk_size",
        "phase_hidden_dim",
        "inner_validation_fold_count",
    )
    if any(int(getattr(args, name)) <= 0 for name in positive_ints) or int(args.max_points_per_query) < 4:
        raise ValueError("absolute appearance fusion integer arguments are invalid")
    if not 0 <= int(args.inner_validation_fold_index) < int(args.inner_validation_fold_count):
        raise ValueError("absolute appearance fusion fold is invalid")
    positive_floats = (
        "fine_search_radius_px",
        "fine_context_radius_px",
        "rgb_temperature",
        "rgb_max_abs_edge_log_ratio",
        "phase_max_abs_log_ratio",
        "phase_learning_rate",
        "rgb_learning_rate",
        "fusion_learning_rate",
        "rgb_cache_gb",
        "amp_init_scale",
        "soft_hard_temperature",
        "max_abs_pose_log_ratio",
    )
    nonnegative_floats = (
        "weight_decay",
        "density_loss_weight",
        "dustbin_loss_weight",
        "identity_loss_weight",
        "phase_posterior_identity_loss_weight",
        "current_hard_phase_loss_weight",
        "current_hard_posterior_loss_weight",
        "current_hard_fusion_loss_weight",
        "static_hard_fusion_loss_weight",
        "pose_loss_weight",
        "soft_hard_loss_weight",
        "appearance_control_loss_weight",
        "pose_margin",
        "hard_repeat_margin",
        "appearance_control_margin",
        "gradient_clip_norm",
        "minimum_phase_conditioned_pose_lift",
    )
    if (
        any(not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0.0 for name in positive_floats)
        or any(not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) < 0.0 for name in nonnegative_floats)
        or not 0.0 < float(args.initial_radio_final_phase_prior_strength)
        < float(args.max_radio_final_phase_prior_strength)
        or not math.isfinite(float(args.max_radio_final_phase_prior_strength))
        or float(args.max_radio_final_phase_prior_strength) <= 0.0
        or not 0.0 <= float(args.minimum_pose_win_fraction) <= 1.0
        or not 0.0 <= float(args.minimum_hard_repeat_win_fraction) <= 1.0
        or not 0.0 < float(args.minimum_hard_repeat_eligible_query_fraction) <= 1.0
    ):
        raise ValueError("absolute appearance fusion floating-point arguments are invalid")


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _model_state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in _unwrap(model).state_dict().items()}


def _load_expert_initialization(
    *,
    checkpoint_path: Path | None,
    phase_model: CandidateMultiscalePhaseIdentityLLR,
    rgb_model: CandidateHighresRGBMultiscaleLikelihood,
    expected_lineage: Mapping[str, object],
) -> dict[str, object]:
    """Load only compatible visual experts, never an old fusion calibrator."""

    if checkpoint_path is None:
        return {"enabled": False}
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError("expert initialization checkpoint does not exist")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or str(payload.get("format", "")) not in EXPERT_INITIALIZATION_CHECKPOINT_FORMATS:
        raise ValueError("expert initialization checkpoint format is incompatible")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("expert initialization checkpoint lacks metadata")
    if (
        not bool(metadata.get("runtime_layout_is_target_free", False))
        or bool(metadata.get("render", True))
        or bool(metadata.get("image_retrieval_or_submap", True))
        or bool(metadata.get("checkpoint_contains_train_targets", True))
    ):
        raise ValueError("expert initialization checkpoint violates the target-free runtime contract")
    lineage = metadata.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("expert initialization checkpoint lacks lineage")
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            raise ValueError(f"expert initialization lineage mismatch: {key}")
    phase_state = payload.get("phase_state_dict")
    rgb_state = payload.get("rgb_state_dict")
    if not isinstance(phase_state, Mapping) or not isinstance(rgb_state, Mapping):
        raise ValueError("expert initialization checkpoint lacks visual expert states")
    phase_model.load_state_dict(dict(phase_state), strict=True)
    rgb_model.load_state_dict(dict(rgb_state), strict=True)
    return {
        "enabled": True,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha256_short(path),
        "checkpoint_format": str(payload["format"]),
        "loaded_components": ["radio_final_phase", "rgb_fine"],
        "old_fusion_calibrator_loaded": False,
    }


def _output_conflict(*, state: _DistributedState, paths: Sequence[Path]) -> bool:
    conflict = bool(any(path.exists() for path in paths)) if state.rank == 0 else False
    if state.enabled:
        value = torch.tensor([int(conflict)], dtype=torch.int64, device=state.device)
        distributed.broadcast(value, src=0)
        conflict = bool(value.item())
    return conflict


def _reduce(state: _DistributedState, values: torch.Tensor) -> torch.Tensor:
    out = values.detach().clone()
    if state.enabled:
        distributed.all_reduce(out, op=distributed.ReduceOp.SUM)
    return out


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", mode="w", encoding="utf-8", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def configure_radio_final_phase_trainable_parameters(
    model: CandidateMultiscalePhaseIdentityLLR,
) -> tuple[str, ...]:
    """Train only the independently audited RADIO-final phase expert."""

    names: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith("source_heads.radio_final.")
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    if not names:
        raise RuntimeError("RADIO-final phase branch has no trainable parameters")
    return tuple(names)


def configure_fine_rgb_trainable_parameters(
    model: CandidateHighresRGBMultiscaleLikelihood,
) -> tuple[str, ...]:
    """Train the shared RGB FPN and fine one-pixel calibration head only."""

    names: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith("texture_encoder.") or name.startswith("calibrators.fine.")
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    if not names:
        raise RuntimeError("fine RGB branch has no trainable parameters")
    return tuple(names)


def source_only_mode(source: str) -> str:
    if source == "radio_final_phase":
        return "phase_identity_only"
    if source == "rgb_fine":
        return "rgb_only"
    raise ValueError("absolute appearance diagnostic source is invalid")


def _load_radio_final_grid16_only(
    *, path: Path, headers: object
) -> tuple[np.ndarray, torch.Tensor]:
    """Load only the independently enabled RADIO-final descriptor grid.

    ``load_context_attention_source_headers`` has already verified the
    RADIO-intermediate and ALIKE cache lineage, image order, and dimensions.
    Their values are intentionally not decompressed when their fixed phase
    mass is zero.  This avoids converting an ablation into a hidden multi-GiB
    input pipeline and keeps the actual visual source set honest.
    """

    image_ids = np.asarray(headers.image_ids).astype(str)
    final_dimension = int(headers.descriptor_dimensions["radio_final"])
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {"image_ids", "grid16_descriptors"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError("RADIO-final phase cache misses grid16 descriptors")
        cached_ids = np.asarray(payload["image_ids"]).astype(str).reshape(-1)
        grid = np.asarray(payload["grid16_descriptors"])
    if not np.array_equal(cached_ids, image_ids):
        raise ValueError("RADIO-final phase grid image order differs from validated header")
    if grid.ndim == 3:
        if grid.shape[:2] != (len(image_ids), 16 * 16):
            raise ValueError("RADIO-final phase grid geometry is invalid")
        grid = grid.reshape(len(image_ids), 16, 16, grid.shape[2])
    if (
        grid.ndim != 4
        or grid.shape[:3] != (len(image_ids), 16, 16)
        or grid.shape[3] != final_dimension
        or grid.dtype not in {np.dtype(np.float16), np.dtype(np.float32)}
    ):
        raise ValueError("RADIO-final phase grid descriptor contract is invalid")
    return image_ids, torch.from_numpy(np.asarray(grid))


def _selected_identity_tensors(
    *, identity_by_query: Mapping[str, object], query_id: str, positions: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    item = identity_by_query.get(str(query_id))
    if item is None:
        raise ValueError("registered identity query is missing")
    selected = np.asarray(positions, dtype=np.int64)
    observed = torch.from_numpy(np.asarray(item.observed_candidate_mask[selected], dtype=bool)).to(device)
    dustbin = torch.from_numpy(np.asarray(item.candidate_dustbin_mask[selected], dtype=bool)).to(device)
    supervised = torch.from_numpy(np.asarray(item.candidate_supervised_mask[selected], dtype=bool)).to(device)
    return canonical_registered_identity_or_null_targets(
        observed_candidate_mask=observed,
        candidate_dustbin_mask=dustbin,
        candidate_supervised_mask=supervised,
    )


def _required_training_sources(
    *,
    query_id: str,
    identity_by_query: Mapping[str, object],
    current_hard: Mapping[str, HardRepeatQueryTargets],
    static_hard: Mapping[str, HardRepeatQueryTargets],
) -> np.ndarray:
    ids: list[np.ndarray] = []
    identity = identity_by_query.get(str(query_id))
    if identity is None:
        raise ValueError("registered identity query is missing")
    observed = np.asarray(identity.observed_candidate_mask, dtype=bool).any(axis=1)
    ids.append(np.asarray(identity.source_point_ids, dtype=np.int64)[observed])
    for targets in (current_hard.get(str(query_id)), static_hard.get(str(query_id))):
        if targets is not None:
            ids.append(np.asarray(targets.source_point_ids, dtype=np.int64))
    return np.unique(np.concatenate(ids)) if ids else np.zeros((0,), dtype=np.int64)


def _phase_identity_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateMultiscalePhaseIdentityPrediction,
    observed: torch.Tensor,
    dustbin: torch.Tensor,
    supervised: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    return exact_identity_or_null_cross_entropy(
        runtime=runtime,
        prediction=prediction,
        observed_candidate_mask=observed,
        target_dustbin=dustbin,
        target_supervised=supervised,
        candidate_prior_logit_weight=0.0,
        source_name="radio_final",
        balance_observed_and_null=True,
    )


def _phase_posterior_identity_loss(
    *,
    fusion: CandidateAbsoluteAppearanceFusion,
    runtime: CandidatePoseRGBSpatialRuntime,
    evidence: CandidateAbsoluteAppearanceEvidence,
    observed: torch.Tensor,
    dustbin: torch.Tensor,
    supervised: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train the v2 candidate/null posterior, not only raw phase edge logits."""

    return phase_conditioned_candidate_posterior_nll(
        fusion=fusion,
        runtime=runtime,
        evidence=evidence,
        observed_candidate_mask=observed,
        target_dustbin=dustbin,
        target_supervised=supervised,
        balance_observed_and_null=True,
    )


def _phase_posterior_hard_repeat_loss(
    *,
    fusion: CandidateAbsoluteAppearanceFusion,
    runtime: CandidatePoseRGBSpatialRuntime,
    evidence: CandidateAbsoluteAppearanceEvidence,
    hard_batch: HardRepeatBatch | None,
    margin: float,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if hard_batch is None:
        return reference.sum() * 0.0, {
            "posterior_hard_repeat_active": 0.0,
            "posterior_hard_repeat_gap": 0.0,
            "posterior_hard_repeat_win": 0.0,
        }
    return phase_conditioned_candidate_posterior_margin_loss(
        fusion=fusion,
        runtime=runtime,
        evidence=evidence,
        point_indices=hard_batch.point_indices,
        positive_candidate_indices=hard_batch.positive_candidate_indices,
        negative_candidate_indices=hard_batch.negative_candidate_indices,
        margin=float(margin),
    )


@torch.no_grad()
def _phase_posterior_identity_audit(
    *,
    fusion: CandidateAbsoluteAppearanceFusion,
    runtime: CandidatePoseRGBSpatialRuntime,
    evidence: CandidateAbsoluteAppearanceEvidence,
    observed: torch.Tensor,
    dustbin: torch.Tensor,
    supervised: torch.Tensor,
) -> torch.Tensor:
    """Return target-joined ranks for the posterior actually used at runtime.

    The candidate set and visual forward are frozen before this function sees
    labels.  A positive rank lift here is necessary, though not sufficient,
    for a RADIO-conditioned prior to improve pose ranking.
    """

    candidates, null, _, usable, _ = fusion.phase_conditioned_candidate_probabilities(
        runtime=runtime,
        evidence=evidence,
    )
    observed = torch.as_tensor(observed, dtype=torch.bool, device=candidates.device)
    dustbin = torch.as_tensor(dustbin, dtype=torch.bool, device=candidates.device).reshape(-1)
    supervised = torch.as_tensor(supervised, dtype=torch.bool, device=candidates.device).reshape(-1)
    if (
        observed.shape != candidates.shape
        or dustbin.shape != (len(candidates),)
        or supervised.shape != dustbin.shape
        or torch.any(observed.sum(dim=1) > 1)
        or torch.any(observed & ~supervised[:, None])
        or torch.any(dustbin & ~supervised)
    ):
        raise ValueError("phase posterior identity audit targets are invalid")
    labels = observed.to(dtype=torch.long).argmax(dim=1)
    observed_rows = observed.any(dim=1)
    target_usable = usable.gather(1, labels[:, None]).squeeze(1)
    active = supervised & observed_rows & target_usable & (usable.sum(dim=1) >= 2)
    totals = torch.zeros((7,), dtype=torch.float64, device=candidates.device)
    if not bool(active.any()):
        return totals
    active_labels = labels[active]
    base = torch.cat((runtime.to(candidates.device).candidate_probabilities, runtime.to(candidates.device).null_probabilities[:, None]), dim=1)
    conditioned = torch.cat((candidates, null[:, None]), dim=1)
    base_order = base.argsort(dim=1, descending=True)
    conditioned_order = conditioned.argsort(dim=1, descending=True)
    base_rank = (base_order[active] == active_labels[:, None]).to(dtype=torch.long).argmax(dim=1) + 1
    conditioned_rank = (
        (conditioned_order[active] == active_labels[:, None]).to(dtype=torch.long).argmax(dim=1) + 1
    )
    totals[:] = torch.tensor(
        [
            float(active.sum().item()),
            float((base.argmax(dim=1)[active] == active_labels).sum().item()),
            float((conditioned.argmax(dim=1)[active] == active_labels).sum().item()),
            float(base_rank.sum().item()),
            float(conditioned_rank.sum().item()),
            float(base[active, active_labels].sum().item()),
            float(conditioned[active, active_labels].sum().item()),
        ],
        dtype=torch.float64,
        device=candidates.device,
    )
    return totals


def _fine_density_loss(
    *,
    prediction: CandidateHighresRGBMultiscalePrediction,
    target_offsets_xy: torch.Tensor,
    target_dustbin: torch.Tensor,
    target_supervised: torch.Tensor,
    dustbin_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    return highres_rgb_spatial_density_nll(
        scale_prediction=prediction.sources["fine"],
        target_offsets_xy=target_offsets_xy,
        target_dustbin=target_dustbin,
        target_supervised=target_supervised,
        dustbin_weight=float(dustbin_weight),
        balance_observed_and_dustbin=True,
    )


def fusion_pose_scores(
    *,
    fusion: CandidateAbsoluteAppearanceFusion,
    runtime: CandidatePoseRGBSpatialRuntime,
    evidence: CandidateAbsoluteAppearanceEvidence,
    correct_projection_offsets_xy: torch.Tensor,
    correct_projection_valid: torch.Tensor,
    wrong_projection_offsets_xy: torch.Tensor,
    wrong_projection_valid: torch.Tensor,
    mode: str = "fused",
    max_abs_pose_log_ratio: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    correct = fusion.score(
        runtime=runtime,
        evidence=evidence,
        candidate_projection_offsets_xy=correct_projection_offsets_xy.unsqueeze(0),
        candidate_projection_valid=correct_projection_valid.unsqueeze(0),
        mode=mode,
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    ).pose_log_likelihood_ratios
    wrong = fusion.score(
        runtime=runtime,
        evidence=evidence,
        candidate_projection_offsets_xy=wrong_projection_offsets_xy,
        candidate_projection_valid=wrong_projection_valid,
        mode=mode,
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    ).pose_log_likelihood_ratios
    if correct.shape != (1,) or wrong.ndim != 1 or len(wrong) == 0:
        raise ValueError("absolute appearance pose score shapes are invalid")
    return correct, wrong


def fusion_hard_repeat_gaps(
    *,
    fusion: CandidateAbsoluteAppearanceFusion,
    runtime: CandidatePoseRGBSpatialRuntime,
    evidence: CandidateAbsoluteAppearanceEvidence,
    hard_batch: HardRepeatBatch | None,
    mode: str = "fused",
    max_abs_pose_log_ratio: float = 6.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Score exact positive/negative candidate edges after visual inference."""

    if hard_batch is None:
        return None, None
    count = len(hard_batch.point_indices)
    if count == 0:
        return None, None
    device = hard_batch.point_indices.device
    offsets = torch.zeros(
        (1, runtime.point_count, runtime.candidate_count, 2), dtype=torch.float32, device=device
    )
    valid = torch.zeros(
        (1, runtime.point_count, runtime.candidate_count), dtype=torch.bool, device=device
    )
    points = hard_batch.point_indices.to(device=device, dtype=torch.long)
    positive = hard_batch.positive_candidate_indices.to(device=device, dtype=torch.long)
    negative = hard_batch.negative_candidate_indices.to(device=device, dtype=torch.long)
    offsets[0, points, positive] = hard_batch.positive_offsets_xy.to(device=device)
    offsets[0, points, negative] = hard_batch.negative_offsets_xy.to(device=device)
    valid[0, points, positive] = True
    valid[0, points, negative] = True
    candidate, usable = fusion.candidate_identity_log_likelihood_ratios(
        runtime=runtime,
        evidence=evidence,
        candidate_projection_offsets_xy=offsets,
        candidate_projection_valid=valid,
        mode=mode,
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    )
    candidate = candidate[0]
    usable = usable[0]
    positive_value = candidate[points, positive]
    negative_value = candidate[points, negative]
    common = usable[points, positive] & usable[points, negative]
    return positive_value - negative_value, common


def _hard_margin_loss(
    gaps: torch.Tensor | None, usable: torch.Tensor | None, margin: float, reference: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    if gaps is None or usable is None or not bool(usable.any()):
        return reference.sum() * 0.0, {"active": 0.0, "gap": 0.0, "win": 0.0}
    active = gaps[usable]
    loss = F.softplus(float(margin) - active).mean()
    return loss, {
        "active": float(len(active)),
        "gap": float(active.detach().mean().item()),
        "win": float((active.detach() > 0.0).float().mean().item()),
    }


def fusion_inner_gate_decision(*, metrics: Mapping[str, float], args: argparse.Namespace) -> dict[str, bool]:
    """Require fused pose evidence and independent RADIO repeat evidence."""

    required = (
        "fused_pose_win_fraction",
        "fused_pose_gap",
        "fused_pose_normal_minus_permuted_gap",
        "rgb_only_pose_gap",
        "fused_static_repeat_eligible_query_fraction",
        "fused_static_repeat_win_fraction",
        "fused_static_repeat_gap",
        "fused_static_repeat_normal_minus_permuted_gap",
        "phase_static_repeat_eligible_query_fraction",
        "phase_static_repeat_win_fraction",
        "phase_static_repeat_gap",
        "phase_static_repeat_normal_minus_permuted_gap",
    )
    if any(name not in metrics or not math.isfinite(float(metrics[name])) for name in required):
        raise ValueError("absolute appearance fusion inner gate metrics are incomplete")
    checks = {
        "fused_pose_win": float(metrics["fused_pose_win_fraction"]) >= float(args.minimum_pose_win_fraction),
        "fused_pose_gap": float(metrics["fused_pose_gap"]) >= float(args.minimum_pose_gap),
        "fused_pose_permutation": float(metrics["fused_pose_normal_minus_permuted_gap"])
        >= float(args.minimum_pose_permutation_delta),
        "fused_pose_phase_conditioning_lift": (
            float(metrics["fused_pose_gap"]) - float(metrics["rgb_only_pose_gap"])
        ) >= float(args.minimum_phase_conditioned_pose_lift),
        "fused_repeat_coverage": float(metrics["fused_static_repeat_eligible_query_fraction"])
        >= float(args.minimum_hard_repeat_eligible_query_fraction),
        "fused_repeat_win": float(metrics["fused_static_repeat_win_fraction"])
        >= float(args.minimum_hard_repeat_win_fraction),
        "fused_repeat_gap": float(metrics["fused_static_repeat_gap"])
        >= float(args.minimum_hard_repeat_gap),
        "fused_repeat_permutation": float(metrics["fused_static_repeat_normal_minus_permuted_gap"])
        >= float(args.minimum_hard_repeat_permutation_delta),
        "phase_repeat_coverage": float(metrics["phase_static_repeat_eligible_query_fraction"])
        >= float(args.minimum_hard_repeat_eligible_query_fraction),
        "phase_repeat_win": float(metrics["phase_static_repeat_win_fraction"])
        >= float(args.minimum_hard_repeat_win_fraction),
        "phase_repeat_gap": float(metrics["phase_static_repeat_gap"])
        >= float(args.minimum_hard_repeat_gap),
        "phase_repeat_permutation": float(metrics["phase_static_repeat_normal_minus_permuted_gap"])
        >= float(args.minimum_hard_repeat_permutation_delta),
    }
    return {**checks, "passed": bool(all(checks.values()))}


def _gate_positions(
    *,
    layout: CandidatePoseRGBSpatialLayout,
    group: TrainQueryGroup,
    coordinate_image_size: tuple[int, int],
    args: argparse.Namespace,
) -> np.ndarray:
    selector = selector_input_from_target_free_layout(layout=layout, rows=group.layout_rows)
    scores = target_free_selector_scores(selector_input=selector, policy=str(args.validation_selector_policy))
    return select_target_free_spatial_quota(
        selector_input=selector,
        quality_scores=scores,
        point_budget=int(args.validation_selector_point_budget),
        grid_rows=int(args.validation_selector_grid_rows),
        grid_columns=int(args.validation_selector_grid_columns),
        image_size=coordinate_image_size,
    )


@torch.no_grad()
def evaluate_static_repeat_full_pool(
    *,
    phase_model: nn.Module,
    rgb_model: nn.Module,
    fusion: CandidateAbsoluteAppearanceFusion,
    groups: Mapping[str, TrainQueryGroup],
    static_hard: Mapping[str, HardRepeatQueryTargets],
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
    """Audit static hard-repeat edges over every frozen P1 point.

    Pose scoring deliberately uses the deployment-sized 64-point target-free
    selector.  Static-repeat coverage is a different question: a target-free
    selector that retains only one third of a query cannot establish whether a
    visual relation rejects repeat structures across that query.  This audit
    therefore processes the complete P1 pool in fixed contiguous chunks.  The
    chunking is memory-only; no target field affects membership or ordering.
    """

    phase = _unwrap(phase_model).eval()
    rgb = _unwrap(rgb_model).eval()
    fusion.eval()
    # Fused normal sum/win/permuted sum/common count; phase equivalent; then
    # query eligibility counts for fused and phase respectively.
    totals = torch.zeros((10,), dtype=torch.float64, device=state.device)
    chunk_size = int(args.validation_selector_point_budget)
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        hard_targets = static_hard.get(str(query_id))
        if group is None or hard_targets is None:
            raise ValueError("full static-repeat audit query is unresolved")
        fused_query_eligible = False
        phase_query_eligible = False
        for start in range(0, group.point_count, chunk_size):
            positions = np.arange(start, min(start + chunk_size, group.point_count), dtype=np.int64)
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
                radius_px=float(rgb.full_patch_radius_px),
                step_px=1.0,
                cache=cache,
                device=state.device,
                cache_device=cache_device,
            )
            permuted_runtime = permute_runtime_support_image_appearance_only(batch.runtime, shift=1)
            permuted_support_patches = _crop_geometry_fixed_permuted_support_patches(
                normal_query_patches=query_patches,
                permuted_runtime=permuted_runtime,
                image_ids=image_ids,
                image_root=image_root,
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=float(rgb.full_patch_radius_px),
                step_px=1.0,
                cache=cache,
                device=state.device,
            )
            with torch.cuda.amp.autocast(enabled=state.device.type == "cuda" and not bool(args.no_amp)):
                normal_evidence = fusion(
                    runtime=batch.runtime,
                    phase_prediction=phase(runtime=batch.runtime),
                    rgb_prediction=rgb(
                        runtime=batch.runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=support_patches,
                        active_sources=("fine",),
                    ),
                )
                permuted_evidence = fusion(
                    runtime=permuted_runtime,
                    phase_prediction=phase(runtime=permuted_runtime),
                    rgb_prediction=rgb(
                        runtime=permuted_runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=permuted_support_patches,
                        active_sources=("fine",),
                    ),
                )
            hard_batch = _hard_repeat_batch_from_group(
                hard_targets=hard_targets,
                group=group,
                point_positions=positions,
                device=state.device,
                max_edges=0,
                seed=int(args.seed),
            )
            normal_fused, normal_fused_usable = fusion_hard_repeat_gaps(
                fusion=fusion,
                runtime=batch.runtime,
                evidence=normal_evidence,
                hard_batch=hard_batch,
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
            )
            permuted_fused, permuted_fused_usable = fusion_hard_repeat_gaps(
                fusion=fusion,
                runtime=permuted_runtime,
                evidence=permuted_evidence,
                hard_batch=hard_batch,
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
            )
            normal_phase, normal_phase_usable = fusion_hard_repeat_gaps(
                fusion=fusion,
                runtime=batch.runtime,
                evidence=normal_evidence,
                hard_batch=hard_batch,
                mode=source_only_mode("radio_final_phase"),
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
            )
            permuted_phase, permuted_phase_usable = fusion_hard_repeat_gaps(
                fusion=fusion,
                runtime=permuted_runtime,
                evidence=permuted_evidence,
                hard_batch=hard_batch,
                mode=source_only_mode("radio_final_phase"),
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
            )
            if (
                normal_fused is not None
                and normal_fused_usable is not None
                and permuted_fused is not None
                and permuted_fused_usable is not None
            ):
                common = normal_fused_usable & permuted_fused_usable
                if bool(common.any()):
                    values = normal_fused[common]
                    permuted_values = permuted_fused[common]
                    totals[:4] += torch.tensor(
                        [
                            float(values.sum().item()),
                            float((values > 0.0).float().sum().item()),
                            float(permuted_values.sum().item()),
                            float(common.sum().item()),
                        ],
                        dtype=torch.float64,
                        device=state.device,
                    )
                    fused_query_eligible = True
            if (
                normal_phase is not None
                and normal_phase_usable is not None
                and permuted_phase is not None
                and permuted_phase_usable is not None
            ):
                common = normal_phase_usable & permuted_phase_usable
                if bool(common.any()):
                    values = normal_phase[common]
                    permuted_values = permuted_phase[common]
                    totals[4:8] += torch.tensor(
                        [
                            float(values.sum().item()),
                            float((values > 0.0).float().sum().item()),
                            float(permuted_values.sum().item()),
                            float(common.sum().item()),
                        ],
                        dtype=torch.float64,
                        device=state.device,
                    )
                    phase_query_eligible = True
        totals[8] += float(fused_query_eligible)
        totals[9] += float(phase_query_eligible)
    totals = _reduce(state, totals)
    fused_count = float(totals[3].item())
    phase_count = float(totals[7].item())
    query_count = float(len(query_ids))
    if query_count <= 0.0:
        raise RuntimeError("full static-repeat audit evaluated no query")
    return {
        "fused_static_repeat_gap": float((totals[0] / fused_count).item()) if fused_count else 0.0,
        "fused_static_repeat_win_fraction": float((totals[1] / fused_count).item()) if fused_count else 0.0,
        "fused_static_repeat_permuted_gap": float((totals[2] / fused_count).item()) if fused_count else 0.0,
        "fused_static_repeat_normal_minus_permuted_gap": (
            float(((totals[0] - totals[2]) / fused_count).item()) if fused_count else 0.0
        ),
        "fused_static_repeat_common_active_edges": fused_count,
        "fused_static_repeat_eligible_query_fraction": float(totals[8].item() / query_count),
        "phase_static_repeat_gap": float((totals[4] / phase_count).item()) if phase_count else 0.0,
        "phase_static_repeat_win_fraction": float((totals[5] / phase_count).item()) if phase_count else 0.0,
        "phase_static_repeat_permuted_gap": float((totals[6] / phase_count).item()) if phase_count else 0.0,
        "phase_static_repeat_normal_minus_permuted_gap": (
            float(((totals[4] - totals[6]) / phase_count).item()) if phase_count else 0.0
        ),
        "phase_static_repeat_common_active_edges": phase_count,
        "phase_static_repeat_eligible_query_fraction": float(totals[9].item() / query_count),
    }


@torch.no_grad()
def evaluate_inner_gate(
    *,
    phase_model: nn.Module,
    rgb_model: nn.Module,
    fusion: CandidateAbsoluteAppearanceFusion,
    layout: CandidatePoseRGBSpatialLayout,
    groups: Mapping[str, TrainQueryGroup],
    identity_by_query: Mapping[str, object],
    static_hard: Mapping[str, HardRepeatQueryTargets],
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
    """Evaluate a target-free selected query fold before joining diagnostics."""

    phase = _unwrap(phase_model).eval()
    rgb = _unwrap(rgb_model).eval()
    fusion.eval()
    # Fused normal/perm gap + win, phase/rgb normal gap, query count;
    # then fused and phase direct static-repeat normal/perm gap + wins + edge/query counts.
    totals = torch.zeros((27,), dtype=torch.float64, device=state.device)
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        if group is None:
            raise ValueError("absolute appearance fusion gate query is unresolved")
        positions = _gate_positions(
            layout=layout, group=group, coordinate_image_size=coordinate_image_size, args=args
        )
        batch = _query_batch_from_group(
            group=group, complete_runtime=complete_runtime, point_positions=positions, device=state.device
        )
        query_patches, support_patches = _crop_runtime_rgb_patches(
            runtime=batch.runtime,
            image_ids=image_ids,
            image_root=image_root,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=float(rgb.full_patch_radius_px),
            step_px=1.0,
            cache=cache,
            device=state.device,
            cache_device=cache_device,
        )
        permuted_runtime = permute_runtime_support_image_appearance_only(batch.runtime, shift=1)
        if torch.equal(permuted_runtime.support_image_indices, batch.runtime.support_image_indices):
            raise RuntimeError("absolute appearance fusion gate permutation is a no-op")
        permuted_support_patches = _crop_geometry_fixed_permuted_support_patches(
            normal_query_patches=query_patches,
            permuted_runtime=permuted_runtime,
            image_ids=image_ids,
            image_root=image_root,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=float(rgb.full_patch_radius_px),
            step_px=1.0,
            cache=cache,
            device=state.device,
        )
        with torch.cuda.amp.autocast(enabled=state.device.type == "cuda" and not bool(args.no_amp)):
            normal_evidence = fusion(
                runtime=batch.runtime,
                phase_prediction=phase(runtime=batch.runtime),
                rgb_prediction=rgb(
                    runtime=batch.runtime,
                    query_rgb_patches=query_patches,
                    support_rgb_patches=support_patches,
                    active_sources=("fine",),
                ),
            )
            permuted_evidence = fusion(
                runtime=permuted_runtime,
                phase_prediction=phase(runtime=permuted_runtime),
                rgb_prediction=rgb(
                    runtime=permuted_runtime,
                    query_rgb_patches=query_patches,
                    support_rgb_patches=permuted_support_patches,
                    active_sources=("fine",),
                ),
            )
            normal_correct, normal_wrong = fusion_pose_scores(
                fusion=fusion,
                runtime=batch.runtime,
                evidence=normal_evidence,
                correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                correct_projection_valid=batch.correct_projection_valid,
                wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                wrong_projection_valid=batch.wrong_projection_valid,
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
            )
            permuted_correct, permuted_wrong = fusion_pose_scores(
                fusion=fusion,
                runtime=permuted_runtime,
                evidence=permuted_evidence,
                correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                correct_projection_valid=batch.correct_projection_valid,
                wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                wrong_projection_valid=batch.wrong_projection_valid,
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
            )
            _, _, normal_gap = pose_margin_terms(
                correct_scores=normal_correct,
                wrong_scores=normal_wrong,
                margin=float(args.pose_margin),
                temperature=float(args.soft_hard_temperature),
            )
            _, _, permuted_gap = pose_margin_terms(
                correct_scores=permuted_correct,
                wrong_scores=permuted_wrong,
                margin=float(args.pose_margin),
                temperature=float(args.soft_hard_temperature),
            )
            phase_correct, phase_wrong = fusion_pose_scores(
                fusion=fusion,
                runtime=batch.runtime,
                evidence=normal_evidence,
                correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                correct_projection_valid=batch.correct_projection_valid,
                wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                wrong_projection_valid=batch.wrong_projection_valid,
                mode=source_only_mode("radio_final_phase"),
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
            )
            rgb_correct, rgb_wrong = fusion_pose_scores(
                fusion=fusion,
                runtime=batch.runtime,
                evidence=normal_evidence,
                correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                correct_projection_valid=batch.correct_projection_valid,
                wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                wrong_projection_valid=batch.wrong_projection_valid,
                mode=source_only_mode("rgb_fine"),
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
            )
            _, _, phase_gap = pose_margin_terms(
                correct_scores=phase_correct,
                wrong_scores=phase_wrong,
                margin=float(args.pose_margin),
                temperature=float(args.soft_hard_temperature),
            )
            _, _, rgb_gap = pose_margin_terms(
                correct_scores=rgb_correct,
                wrong_scores=rgb_wrong,
                margin=float(args.pose_margin),
                temperature=float(args.soft_hard_temperature),
            )
        totals[:7] += torch.tensor(
            [
                float(normal_gap.item()),
                float((normal_gap > 0.0).float().item()),
                float(permuted_gap.item()),
                float(phase_gap.item()),
                float(rgb_gap.item()),
                1.0,
                0.0,
            ],
            dtype=torch.float64,
            device=state.device,
        )
        observed, dustbin, supervised = _selected_identity_tensors(
            identity_by_query=identity_by_query,
            query_id=str(query_id),
            positions=positions,
            device=state.device,
        )
        totals[20:27] += _phase_posterior_identity_audit(
            fusion=fusion,
            runtime=batch.runtime,
            evidence=normal_evidence,
            observed=observed,
            dustbin=dustbin,
            supervised=supervised,
        )
        static = static_hard.get(str(query_id))
        if static is None:
            continue
        hard_batch = _hard_repeat_batch_from_group(
            hard_targets=static,
            group=group,
            point_positions=positions,
            device=state.device,
            max_edges=int(args.max_hard_repeat_edges_per_query),
            seed=int(args.seed),
        )
        normal_fused, normal_fused_usable = fusion_hard_repeat_gaps(
            fusion=fusion,
            runtime=batch.runtime,
            evidence=normal_evidence,
            hard_batch=hard_batch,
            max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
        )
        permuted_fused, permuted_fused_usable = fusion_hard_repeat_gaps(
            fusion=fusion,
            runtime=permuted_runtime,
            evidence=permuted_evidence,
            hard_batch=hard_batch,
            max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
        )
        normal_phase, normal_phase_usable = fusion_hard_repeat_gaps(
            fusion=fusion,
            runtime=batch.runtime,
            evidence=normal_evidence,
            hard_batch=hard_batch,
            mode=source_only_mode("radio_final_phase"),
            max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
        )
        permuted_phase, permuted_phase_usable = fusion_hard_repeat_gaps(
            fusion=fusion,
            runtime=permuted_runtime,
            evidence=permuted_evidence,
            hard_batch=hard_batch,
            mode=source_only_mode("radio_final_phase"),
            max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
        )
        if (
            normal_fused is not None
            and normal_fused_usable is not None
            and permuted_fused is not None
            and permuted_fused_usable is not None
        ):
            common = normal_fused_usable & permuted_fused_usable
            if bool(common.any()):
                values = normal_fused[common]
                permuted_values = permuted_fused[common]
                totals[7:13] += torch.tensor(
                    [
                        float(values.sum().item()),
                        float((values > 0.0).float().sum().item()),
                        float(permuted_values.sum().item()),
                        float(common.sum().item()),
                        1.0,
                        0.0,
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
        if (
            normal_phase is not None
            and normal_phase_usable is not None
            and permuted_phase is not None
            and permuted_phase_usable is not None
        ):
            common = normal_phase_usable & permuted_phase_usable
            if bool(common.any()):
                values = normal_phase[common]
                permuted_values = permuted_phase[common]
                totals[13:18] += torch.tensor(
                    [
                        float(values.sum().item()),
                        float((values > 0.0).float().sum().item()),
                        float(permuted_values.sum().item()),
                        float(common.sum().item()),
                        1.0,
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
    totals = _reduce(state, totals)
    query_count = float(totals[5].item())
    if query_count <= 0.0:
        raise RuntimeError("absolute appearance fusion gate evaluated no query")
    fused_edges = float(totals[10].item())
    phase_edges = float(totals[16].item())
    fused_queries = float(totals[11].item())
    phase_queries = float(totals[17].item())
    fused_gap = float((totals[0] / query_count).item())
    fused_permuted = float((totals[2] / query_count).item())
    posterior_identity_count = float(totals[20].item())
    full_static_metrics = evaluate_static_repeat_full_pool(
        phase_model=phase_model,
        rgb_model=rgb_model,
        fusion=fusion,
        groups=groups,
        static_hard=static_hard,
        complete_runtime=complete_runtime,
        query_ids=query_ids,
        image_ids=image_ids,
        image_root=image_root,
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
        cache=cache,
        cache_device=cache_device,
        state=state,
        args=args,
    )
    return {
        "fused_pose_gap": fused_gap,
        "fused_pose_win_fraction": float((totals[1] / query_count).item()),
        "fused_pose_permuted_gap": fused_permuted,
        "fused_pose_normal_minus_permuted_gap": fused_gap - fused_permuted,
        "fused_pose_minus_rgb_only_gap": fused_gap - float((totals[4] / query_count).item()),
        "phase_only_pose_gap": float((totals[3] / query_count).item()),
        "rgb_only_pose_gap": float((totals[4] / query_count).item()),
        "query_count": query_count,
        "fused_static_repeat_gap": float((totals[7] / fused_edges).item()) if fused_edges else 0.0,
        "fused_static_repeat_win_fraction": float((totals[8] / fused_edges).item()) if fused_edges else 0.0,
        "fused_static_repeat_permuted_gap": float((totals[9] / fused_edges).item()) if fused_edges else 0.0,
        "fused_static_repeat_normal_minus_permuted_gap": (
            float(((totals[7] - totals[9]) / fused_edges).item()) if fused_edges else 0.0
        ),
        "fused_static_repeat_common_active_edges": fused_edges,
        "fused_static_repeat_eligible_query_fraction": fused_queries / query_count,
        "phase_static_repeat_gap": float((totals[13] / phase_edges).item()) if phase_edges else 0.0,
        "phase_static_repeat_win_fraction": float((totals[14] / phase_edges).item()) if phase_edges else 0.0,
        "phase_static_repeat_permuted_gap": float((totals[15] / phase_edges).item()) if phase_edges else 0.0,
        "phase_static_repeat_normal_minus_permuted_gap": (
            float(((totals[13] - totals[15]) / phase_edges).item()) if phase_edges else 0.0
        ),
        "phase_static_repeat_common_active_edges": phase_edges,
        "phase_static_repeat_eligible_query_fraction": phase_queries / query_count,
        "base_prior_identity_observed_count": posterior_identity_count,
        "base_prior_identity_top1": (
            float((totals[21] / posterior_identity_count).item())
            if posterior_identity_count
            else 0.0
        ),
        "phase_posterior_identity_top1": (
            float((totals[22] / posterior_identity_count).item())
            if posterior_identity_count
            else 0.0
        ),
        "base_prior_identity_mean_rank": (
            float((totals[23] / posterior_identity_count).item())
            if posterior_identity_count
            else 0.0
        ),
        "phase_posterior_identity_mean_rank": (
            float((totals[24] / posterior_identity_count).item())
            if posterior_identity_count
            else 0.0
        ),
        "base_prior_identity_correct_probability": (
            float((totals[25] / posterior_identity_count).item())
            if posterior_identity_count
            else 0.0
        ),
        "phase_posterior_identity_correct_probability": (
            float((totals[26] / posterior_identity_count).item())
            if posterior_identity_count
            else 0.0
        ),
        **full_static_metrics,
    }


def _finite_trainable(parameters: Sequence[torch.nn.Parameter]) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)


def _sync_fusion_gradient(*, fusion: CandidateAbsoluteAppearanceFusion, state: _DistributedState) -> None:
    """Synchronize the standalone global prior calibrator after DDP gradients."""

    if state.enabled and fusion.phase_prior_logit.grad is not None:
        distributed.all_reduce(fusion.phase_prior_logit.grad, op=distributed.ReduceOp.SUM)
        fusion.phase_prior_logit.grad.div_(float(state.world_size))


def train_candidate_absolute_appearance_fusion(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    state = _initialize_distributed(str(args.device))
    try:
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_absolute_appearance_fusion.pt"
        history_path = output_dir / "history.json"
        summary_path = output_dir / "summary.json"
        if _output_conflict(state=state, paths=(checkpoint_path, history_path, summary_path)) and not bool(args.force):
            raise FileExistsError("refusing to overwrite absolute appearance fusion output")
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

        layout_path = Path(args.rgb_spatial_layout)
        geometry_path = Path(args.geometry_training_targets)
        identity_path = Path(args.registered_identity_targets)
        current_hard_path = Path(args.current_hard_repeat_targets)
        static_hard_path = Path(args.static_hard_repeat_targets)
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        geometry_targets = load_candidate_pose_rgb_spatial_training_targets(geometry_path)
        identity_targets = load_candidate_pose_rgb_spatial_training_targets(identity_path)
        layout_sha = file_sha256_short(layout_path)
        geometry_sha = file_sha256_short(geometry_path)
        identity_sha = file_sha256_short(identity_path)
        validate_registered_identity_targets_for_geometry(
            layout=layout,
            geometry_targets=geometry_targets,
            registered_identity_targets=identity_targets,
            layout_sha256=layout_sha,
        )
        try:
            target_radius = float(geometry_targets.metadata["spatial_search_radius_px"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("geometry target lacks a usable spatial search radius") from error
        if not math.isclose(
            target_radius,
            float(args.fine_search_radius_px),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("fine RGB search radius must match the frozen geometry target")
        groups = build_train_query_groups(layout=layout, targets=geometry_targets)
        identity_by_query = build_exact_identity_query_targets(
            groups=groups, identity_targets=identity_targets
        )
        all_query_ids = tuple(sorted(groups))
        inner_train_ids, inner_validation_ids = _partition_train_queries_for_inner_validation(
            query_ids=all_query_ids,
            fold_count=int(args.inner_validation_fold_count),
            fold_index=int(args.inner_validation_fold_index),
        )
        partition = train_query_partition_manifest(
            all_query_ids=all_query_ids,
            inner_train_query_ids=inner_train_ids,
            inner_validation_query_ids=inner_validation_ids,
            fold_count=int(args.inner_validation_fold_count),
            fold_index=int(args.inner_validation_fold_index),
        )
        current_hard_payload = load_candidate_pose_rgb_spatial_hard_repeat_targets(current_hard_path)
        current_hard = build_hard_repeat_query_targets(
            layout=layout,
            targets=geometry_targets,
            hard_repeat_targets=current_hard_payload,
            layout_sha256=layout_sha,
            targets_sha256=geometry_sha,
        )
        current_hard_provenance = validate_external_frozen_current_hard_targets(
            mined_targets=current_hard_payload,
            mined_groups=current_hard,
            registered_identity_targets=identity_targets,
            registered_identity_targets_sha256=identity_sha,
            expected_partition=partition,
            mining_checkpoint_path=Path(args.current_hard_mining_checkpoint),
        )
        static_hard_payload = load_candidate_pose_rgb_spatial_hard_repeat_targets(static_hard_path)
        static_hard_geometry = build_hard_repeat_query_targets(
            layout=layout,
            targets=geometry_targets,
            hard_repeat_targets=static_hard_payload,
            layout_sha256=layout_sha,
            targets_sha256=geometry_sha,
        )
        static_hard, static_hard_filter = filter_static_hard_repeat_groups_to_registered_exact_identity(
            static_groups=static_hard_geometry,
            registered_identity_targets=identity_targets,
        )
        if not set(inner_validation_ids).issubset(static_hard):
            raise ValueError("static hard-repeat target misses an inner-validation query")
        owner_costs = target_free_query_owner_costs(layout=layout, groups=groups)

        headers = load_context_attention_source_headers(
            radio_final_context_cache=Path(args.radio_final_context_cache),
            radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
            alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
            expected_radio_checkpoint="",
        )
        image_ids, radio_final_grid = _load_radio_final_grid16_only(
            path=Path(args.radio_final_context_cache), headers=headers
        )
        image_sizes = np.asarray(headers.image_sizes, dtype=np.int64)
        source_grids = {"radio_final": radio_final_grid}
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("absolute appearance fusion requires one processed coordinate size")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=headers.metadata_by_name["radio_final"],
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)

        storage_dtype = torch.float16 if str(args.phase_source_storage_dtype) == "float16" else torch.float32
        phase_model = CandidateMultiscalePhaseIdentityLLR(
            sources=source_grids,
            image_sizes=torch.from_numpy(image_sizes),
            source_weights={"radio_final": 1.0, "radio_intermediate": 0.0, "alike": 0.0},
            hidden_dim=int(args.phase_hidden_dim),
            max_abs_log_ratio=float(args.phase_max_abs_log_ratio),
            source_storage_dtype=storage_dtype,
        ).to(state.device)
        rgb_model = CandidateHighresRGBMultiscaleLikelihood(
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            fine_search_radius_px=float(args.fine_search_radius_px),
            fine_context_radius_px=float(args.fine_context_radius_px),
            broad_search_radius_px=float(args.fine_search_radius_px),
            broad_context_radius_px=float(args.fine_context_radius_px),
            texture_feature_dim=int(args.texture_feature_dim),
            hidden_dim=int(args.rgb_hidden_dim),
            edge_chunk_size=int(args.rgb_edge_chunk_size),
            rgb_temperature=float(args.rgb_temperature),
            max_abs_edge_log_ratio=float(args.rgb_max_abs_edge_log_ratio),
        ).to(state.device)
        fusion = CandidateAbsoluteAppearanceFusion(
            initial_phase_prior_strength=float(args.initial_radio_final_phase_prior_strength),
            max_phase_prior_strength=float(args.max_radio_final_phase_prior_strength),
        ).to(state.device)
        expert_initialization = _load_expert_initialization(
            checkpoint_path=(
                Path(str(args.expert_initialization_checkpoint))
                if str(args.expert_initialization_checkpoint).strip()
                else None
            ),
            phase_model=phase_model,
            rgb_model=rgb_model,
            expected_lineage={
                "layout_sha256": layout_sha,
                "geometry_training_targets_sha256": geometry_sha,
                "registered_identity_targets_sha256": identity_sha,
                "current_hard_repeat_targets_sha256": file_sha256_short(current_hard_path),
                "static_hard_repeat_targets_sha256": file_sha256_short(static_hard_path),
                "source_cache_sha256": {
                    "radio_final": file_sha256_short(Path(args.radio_final_context_cache)),
                    "radio_intermediate": file_sha256_short(Path(args.radio_intermediate_context_cache)),
                    "alike": file_sha256_short(Path(args.alike_spatial_context_cache)),
                },
                "projection_space_id": str(layout.metadata.get("projection_space_id", "")),
                "descriptor_space_id": str(layout.metadata.get("descriptor_space_id", "")),
            },
        )
        phase_names = configure_radio_final_phase_trainable_parameters(phase_model)
        rgb_names = configure_fine_rgb_trainable_parameters(rgb_model)
        phase_for_train: nn.Module = phase_model
        rgb_for_train: nn.Module = rgb_model
        if state.enabled:
            phase_for_train = DistributedDataParallel(
                phase_model,
                device_ids=[state.local_rank] if state.device.type == "cuda" else None,
                output_device=state.local_rank if state.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            rgb_for_train = DistributedDataParallel(
                rgb_model,
                device_ids=[state.local_rank] if state.device.type == "cuda" else None,
                output_device=state.local_rank if state.device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in phase_model.parameters() if p.requires_grad], "lr": float(args.phase_learning_rate)},
                {"params": [p for p in rgb_model.parameters() if p.requires_grad], "lr": float(args.rgb_learning_rate)},
                {"params": list(fusion.parameters()), "lr": float(args.fusion_learning_rate)},
            ],
            weight_decay=float(args.weight_decay),
        )
        trainable = [
            *[p for p in phase_model.parameters() if p.requires_grad],
            *[p for p in rgb_model.parameters() if p.requires_grad],
            *list(fusion.parameters()),
        ]
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled, init_scale=float(args.amp_init_scale))
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        cache_device = torch.device("cpu") if str(args.rgb_cache_device) == "cpu" else state.device
        history: list[dict[str, object]] = []
        start_time = time.time()
        baseline = evaluate_inner_gate(
            phase_model=phase_for_train,
            rgb_model=rgb_for_train,
            fusion=fusion,
            layout=layout,
            groups=groups,
            identity_by_query=identity_by_query,
            static_hard=static_hard,
            complete_runtime=complete_runtime,
            query_ids=inner_validation_ids,
            image_ids=image_ids,
            image_root=Path(args.image_root),
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            cache=cache,
            cache_device=cache_device,
            state=state,
            args=args,
        )
        final_metrics = dict(baseline)
        final_gate = fusion_inner_gate_decision(metrics=baseline, args=args)
        steps_per_rank = int(math.ceil(len(inner_train_ids) / state.world_size))
        for epoch in range(1, int(args.epochs) + 1):
            phase_for_train.train()
            rgb_for_train.train()
            fusion.train()
            schedules = balanced_ddp_query_schedules(
                query_ids=inner_train_ids,
                owner_costs=owner_costs,
                world_size=state.world_size,
                seed=int(args.seed) + epoch,
            )
            local_query_ids = schedules[state.rank]
            if len(local_query_ids) != steps_per_rank:
                raise RuntimeError("absolute appearance fusion DDP schedule drifted")
            balance = ddp_owner_cost_balance_metrics(schedules=schedules, owner_costs=owner_costs)
            totals = torch.zeros((15,), dtype=torch.float64, device=state.device)
            epoch_start = time.time()
            for step, query_id in enumerate(local_query_ids):
                group = groups[str(query_id)]
                required = _required_training_sources(
                    query_id=str(query_id),
                    identity_by_query=identity_by_query,
                    current_hard=current_hard,
                    static_hard=static_hard,
                )
                positions = _select_group_points(
                    group=group,
                    max_points=int(args.max_points_per_query),
                    seed=int(args.seed) + epoch * 100003 + step,
                    required_source_point_ids=required,
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
                    radius_px=float(rgb_model.full_patch_radius_px),
                    step_px=1.0,
                    cache=cache,
                    device=state.device,
                    cache_device=cache_device,
                )
                permuted_runtime = permute_runtime_support_image_appearance_only(batch.runtime, shift=1)
                permuted_support_patches = _crop_geometry_fixed_permuted_support_patches(
                    normal_query_patches=query_patches,
                    permuted_runtime=permuted_runtime,
                    image_ids=image_ids,
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    radius_px=float(rgb_model.full_patch_radius_px),
                    step_px=1.0,
                    cache=cache,
                    device=state.device,
                )
                current_batch = _hard_repeat_batch_from_group(
                    hard_targets=current_hard.get(str(query_id)),
                    group=group,
                    point_positions=positions,
                    device=state.device,
                    max_edges=int(args.max_hard_repeat_edges_per_query),
                    seed=int(args.seed) + epoch * 104729 + step,
                ) if current_hard.get(str(query_id)) is not None else None
                static_batch = _hard_repeat_batch_from_group(
                    hard_targets=static_hard.get(str(query_id)),
                    group=group,
                    point_positions=positions,
                    device=state.device,
                    max_edges=int(args.max_hard_repeat_edges_per_query),
                    seed=int(args.seed) + epoch * 130363 + step,
                ) if static_hard.get(str(query_id)) is not None else None
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    normal_phase = phase_for_train(runtime=batch.runtime)
                    normal_rgb = rgb_for_train(
                        runtime=batch.runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=support_patches,
                        active_sources=("fine",),
                    )
                    normal_evidence = fusion(
                        runtime=batch.runtime,
                        phase_prediction=normal_phase,
                        rgb_prediction=normal_rgb,
                    )
                    observed, dustbin, supervised = _selected_identity_tensors(
                        identity_by_query=identity_by_query,
                        query_id=str(query_id),
                        positions=positions,
                        device=state.device,
                    )
                    density_loss, _ = _fine_density_loss(
                        prediction=normal_rgb,
                        target_offsets_xy=batch.spatial_target_offsets_xy,
                        target_dustbin=batch.spatial_target_dustbin,
                        target_supervised=batch.spatial_target_supervised,
                        dustbin_weight=float(args.dustbin_loss_weight),
                    )
                    identity_loss, _ = _phase_identity_loss(
                        runtime=batch.runtime,
                        prediction=normal_phase,
                        observed=observed,
                        dustbin=dustbin,
                        supervised=supervised,
                    )
                    posterior_identity_loss, _ = _phase_posterior_identity_loss(
                        fusion=fusion,
                        runtime=batch.runtime,
                        evidence=normal_evidence,
                        observed=observed,
                        dustbin=dustbin,
                        supervised=supervised,
                    )
                    current_phase_loss, _ = current_hard_repeat_identity_margin_loss(
                        runtime=batch.runtime,
                        prediction=normal_phase,
                        point_indices=current_batch.point_indices,
                        positive_candidate_indices=current_batch.positive_candidate_indices,
                        negative_candidate_indices=current_batch.negative_candidate_indices,
                        margin=float(args.hard_repeat_margin),
                        source_name="radio_final",
                    ) if current_batch is not None else (density_loss * 0.0, {"hard_repeat_active": 0.0})
                    current_posterior_loss, _ = _phase_posterior_hard_repeat_loss(
                        fusion=fusion,
                        runtime=batch.runtime,
                        evidence=normal_evidence,
                        hard_batch=current_batch,
                        margin=float(args.hard_repeat_margin),
                        reference=density_loss,
                    )
                    current_gaps, current_usable = fusion_hard_repeat_gaps(
                        fusion=fusion,
                        runtime=batch.runtime,
                        evidence=normal_evidence,
                        hard_batch=current_batch,
                        max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                    )
                    current_fusion_loss, current_metrics = _hard_margin_loss(
                        current_gaps, current_usable, float(args.hard_repeat_margin), density_loss
                    )
                    static_gaps, static_usable = fusion_hard_repeat_gaps(
                        fusion=fusion,
                        runtime=batch.runtime,
                        evidence=normal_evidence,
                        hard_batch=static_batch,
                        max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                    )
                    static_fusion_loss, static_metrics = _hard_margin_loss(
                        static_gaps, static_usable, float(args.hard_repeat_margin), density_loss
                    )
                    normal_correct, normal_wrong = fusion_pose_scores(
                        fusion=fusion,
                        runtime=batch.runtime,
                        evidence=normal_evidence,
                        correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                        correct_projection_valid=batch.correct_projection_valid,
                        wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                        wrong_projection_valid=batch.wrong_projection_valid,
                        max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                    )
                    pose_loss, soft_hard_loss, normal_gap = pose_margin_terms(
                        correct_scores=normal_correct,
                        wrong_scores=normal_wrong,
                        margin=float(args.pose_margin),
                        temperature=float(args.soft_hard_temperature),
                    )
                    # The deranged branch is a fixed target-free appearance
                    # control.  Keeping its FPN graph would retain a second
                    # 64x20x2 high-resolution patch activation set beside the
                    # normal objective.  Its score is intentionally detached:
                    # gradients require genuine pairing to beat this frozen
                    # visual baseline, while all gate diagnostics still run
                    # both branches symmetrically without this memory shortcut.
                    with torch.no_grad():
                        permuted_phase = _unwrap(phase_for_train)(runtime=permuted_runtime)
                        permuted_rgb = _unwrap(rgb_for_train)(
                            runtime=permuted_runtime,
                            query_rgb_patches=query_patches,
                            support_rgb_patches=permuted_support_patches,
                            active_sources=("fine",),
                        )
                        permuted_evidence = fusion(
                            runtime=permuted_runtime,
                            phase_prediction=permuted_phase,
                            rgb_prediction=permuted_rgb,
                        )
                        permuted_correct, permuted_wrong = fusion_pose_scores(
                            fusion=fusion,
                            runtime=permuted_runtime,
                            evidence=permuted_evidence,
                            correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                            correct_projection_valid=batch.correct_projection_valid,
                            wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                            wrong_projection_valid=batch.wrong_projection_valid,
                            max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                        )
                        _, _, permuted_gap = pose_margin_terms(
                            correct_scores=permuted_correct,
                            wrong_scores=permuted_wrong,
                            margin=float(args.pose_margin),
                            temperature=float(args.soft_hard_temperature),
                        )
                    control_loss, control_metrics = appearance_control_margin_loss(
                        normal_gaps=normal_gap,
                        permuted_gaps=permuted_gap,
                        margin=float(args.appearance_control_margin),
                    )
                    total_loss = (
                        float(args.density_loss_weight) * density_loss
                        + float(args.identity_loss_weight) * identity_loss
                        + float(args.phase_posterior_identity_loss_weight) * posterior_identity_loss
                        + float(args.current_hard_phase_loss_weight) * current_phase_loss
                        + float(args.current_hard_posterior_loss_weight) * current_posterior_loss
                        + float(args.current_hard_fusion_loss_weight) * current_fusion_loss
                        + float(args.static_hard_fusion_loss_weight) * static_fusion_loss
                        + float(args.pose_loss_weight) * pose_loss
                        + float(args.soft_hard_loss_weight) * soft_hard_loss
                        + float(args.appearance_control_loss_weight) * control_loss
                    )
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                _sync_fusion_gradient(fusion=fusion, state=state)
                finite = _finite_trainable(trainable)
                finite_tensor = torch.tensor([int(finite)], dtype=torch.int64, device=state.device)
                if state.enabled:
                    distributed.all_reduce(finite_tensor, op=distributed.ReduceOp.MIN)
                if bool(finite_tensor.item()):
                    if float(args.gradient_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(trainable, float(args.gradient_clip_norm))
                    scaler.step(optimizer)
                else:
                    optimizer.zero_grad(set_to_none=True)
                scaler.update()
                totals += torch.tensor(
                    [
                        float(total_loss.detach().item()),
                        float(density_loss.detach().item()),
                        float(identity_loss.detach().item()),
                        float(current_phase_loss.detach().item()),
                        float(current_fusion_loss.detach().item()),
                        float(static_fusion_loss.detach().item()),
                        float(pose_loss.detach().item()),
                        float(soft_hard_loss.detach().item()),
                        float(normal_gap.detach().item()),
                        float(control_metrics["normal_minus_permuted"]),
                        float(current_metrics["gap"]),
                        float(static_metrics["gap"]),
                        1.0 - float(finite_tensor.item()),
                        float(posterior_identity_loss.detach().item()),
                        float(current_posterior_loss.detach().item()),
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            totals = _reduce(state, totals)
            run_gate = bool(args.gate_every_epoch) or epoch == int(args.epochs)
            epoch_metrics: dict[str, float] | None = None
            epoch_gate: dict[str, bool] | None = None
            if run_gate:
                epoch_metrics = evaluate_inner_gate(
                    phase_model=phase_for_train,
                    rgb_model=rgb_for_train,
                    fusion=fusion,
                    layout=layout,
                    groups=groups,
                    identity_by_query=identity_by_query,
                    static_hard=static_hard,
                    complete_runtime=complete_runtime,
                    query_ids=inner_validation_ids,
                    image_ids=image_ids,
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    cache=cache,
                    cache_device=cache_device,
                    state=state,
                    args=args,
                )
                epoch_gate = fusion_inner_gate_decision(metrics=epoch_metrics, args=args)
                final_metrics = dict(epoch_metrics)
                final_gate = dict(epoch_gate)
            if state.rank == 0:
                train_steps = float(steps_per_rank * state.world_size)
                record = {
                    "epoch": int(epoch),
                    "train_total_loss": float((totals[0] / train_steps).item()),
                    "train_density_loss": float((totals[1] / train_steps).item()),
                    "train_identity_loss": float((totals[2] / train_steps).item()),
                    "train_current_hard_phase_loss": float((totals[3] / train_steps).item()),
                    "train_current_hard_fusion_loss": float((totals[4] / train_steps).item()),
                    "train_static_hard_fusion_loss": float((totals[5] / train_steps).item()),
                    "train_pose_loss": float((totals[6] / train_steps).item()),
                    "train_soft_hard_loss": float((totals[7] / train_steps).item()),
                    "train_pose_gap": float((totals[8] / train_steps).item()),
                    "train_pose_appearance_delta": float((totals[9] / train_steps).item()),
                    "train_current_hard_gap": float((totals[10] / train_steps).item()),
                    "train_static_hard_gap": float((totals[11] / train_steps).item()),
                    "train_nonfinite_update_fraction": float((totals[12] / train_steps).item()),
                    "train_phase_posterior_identity_loss": float((totals[13] / train_steps).item()),
                    "train_current_hard_posterior_loss": float((totals[14] / train_steps).item()),
                    "fusion_phase_prior_strength": float(
                        fusion.learned_phase_prior_strength().detach().cpu().item()
                    ),
                    "ddp_owner_cost_mean_abs_difference": float(balance["mean_abs_owner_cost_difference"]),
                    "ddp_owner_cost_max_abs_difference": float(balance["max_abs_owner_cost_difference"]),
                    "epoch_seconds": float(time.time() - epoch_start),
                    "inner_gate_evaluated": bool(run_gate),
                    **(
                        {f"inner_{key}": value for key, value in epoch_metrics.items()}
                        if epoch_metrics is not None
                        else {}
                    ),
                    "inner_gate_passed": (
                        bool(epoch_gate["passed"]) if epoch_gate is not None else None
                    ),
                }
                history.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            if state.enabled:
                distributed.barrier()

        if state.rank == 0:
            selection = fixed_final_epoch_checkpoint_selection(epochs=int(args.epochs))
            final_phase_prior_strength = float(
                fusion.learned_phase_prior_strength().detach().cpu().item()
            )
            metadata: dict[str, object] = {
                "format": CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_ABSOLUTE_APPEARANCE_FUSION_FORMAT,
                "phase_model_format": CANDIDATE_MULTISCALE_PHASE_IDENTITY_LLR_FORMAT,
                "rgb_model_format": CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
                "architecture": "radio_final_full_2d_phase_conditions_fixed_candidate_null_prior_plus_real_rgb_fine_1px_pose_projected_likelihood_v2",
                "runtime_layout_is_target_free": True,
                "contains_target_fields": False,
                "checkpoint_contains_train_targets": False,
                "pose_or_ground_truth_used_by_runtime_scorer": False,
                "render": False,
                "image_retrieval_or_submap": False,
                "fixed_global_topl": True,
                "fixed_candidate_top_k": int(layout.candidate_count),
                "fixed_support_view_count": int(layout.support_view_count),
                "explicit_null": True,
                "projection_after_network_only": True,
                "out_of_window_projection_semantics": "fixed_neutral_missing_edge_not_learned_dustbin_v1",
                "radio_final_only": True,
                "inactive_phase_sources": ["radio_intermediate", "alike"],
                "source_combination": "radiofinal_identity_prior_conditioning_plus_rgb_only_pose_projected_likelihood_v2",
                "checkpoint_selection_policy": selection["policy"],
                "inner_validation_used_for_model_selection": False,
                "train_only_inner_gate_passed": bool(final_gate["passed"]),
                "heldout_evaluation_allowed": bool(final_gate["passed"]),
                "pnp_integration_allowed": False,
                "diagnostic_only": not bool(final_gate["passed"]),
                "inner_gate_evaluator_manifest": current_fusion_inner_gate_manifest(),
                "config": {
                    "fine_search_radius_px": float(args.fine_search_radius_px),
                    "fine_context_radius_px": float(args.fine_context_radius_px),
                    "texture_feature_dim": int(args.texture_feature_dim),
                    "rgb_hidden_dim": int(args.rgb_hidden_dim),
                    "rgb_edge_chunk_size": int(args.rgb_edge_chunk_size),
                    "phase_hidden_dim": int(args.phase_hidden_dim),
                    "phase_max_abs_log_ratio": float(args.phase_max_abs_log_ratio),
                    "phase_posterior_identity_loss_weight": float(
                        args.phase_posterior_identity_loss_weight
                    ),
                    "current_hard_posterior_loss_weight": float(
                        args.current_hard_posterior_loss_weight
                    ),
                    "phase_source_storage_dtype": str(args.phase_source_storage_dtype),
                    "final_radio_final_phase_prior_strength": final_phase_prior_strength,
                    "max_radio_final_phase_prior_strength": float(args.max_radio_final_phase_prior_strength),
                    "phase_trainable_parameter_names": list(phase_names),
                    "rgb_trainable_parameter_names": list(rgb_names),
                    "expert_initialization": expert_initialization,
                },
                "lineage": {
                    "layout_sha256": layout_sha,
                    "geometry_training_targets_sha256": geometry_sha,
                    "registered_identity_targets_sha256": identity_sha,
                    "current_hard_repeat_targets_sha256": file_sha256_short(current_hard_path),
                    "static_hard_repeat_targets_sha256": file_sha256_short(static_hard_path),
                    "current_hard_negative_provenance": current_hard_provenance,
                    "static_hard_registered_exact_filter": static_hard_filter,
                    "source_cache_sha256": {
                        "radio_final": file_sha256_short(Path(args.radio_final_context_cache)),
                        "radio_intermediate": file_sha256_short(Path(args.radio_intermediate_context_cache)),
                        "alike": file_sha256_short(Path(args.alike_spatial_context_cache)),
                    },
                    "source_image_manifest_sha256": str(
                        headers.metadata_by_name["radio_final"].get("source_image_manifest_sha256", "")
                    ),
                    "projection_space_id": str(layout.metadata.get("projection_space_id", "")),
                    "descriptor_space_id": str(layout.metadata.get("descriptor_space_id", "")),
                    "rgb_coordinate_bridge": rgb_bridge,
                },
                "training": {
                    "objective": "target_free_radiofinal_raw_and_conditioned_exact_identity_plus_conditioned_current_hard_repeat_plus_fine_rgb_density_plus_correct_vs_full_coherent_wrong_pose_margin_v4",
                    "epochs": int(args.epochs),
                    "world_size": int(state.world_size),
                    "inner_train_query_count": len(inner_train_ids),
                    "inner_validation_query_count": len(inner_validation_ids),
                    "checkpoint_selection": selection,
                    "gate_every_epoch": bool(args.gate_every_epoch),
                    "epoch_zero_inner_validation_metrics": baseline,
                    "final_epoch_inner_validation_metrics": final_metrics,
                    "final_epoch_inner_gate": final_gate,
                    "expert_initialization": expert_initialization,
                },
            }
            _atomic_torch_save(
                {
                    "format": CHECKPOINT_FORMAT,
                    "phase_state_dict": _model_state_cpu(phase_for_train),
                    "rgb_state_dict": _model_state_cpu(rgb_for_train),
                    "fusion_state_dict": _model_state_cpu(fusion),
                    "metadata": metadata,
                },
                checkpoint_path,
            )
            summary = {
                "stage": "train_candidate_absolute_appearance_fusion",
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": file_sha256_short(checkpoint_path),
                "checkpoint_selection": {
                    **selection,
                    "baseline_inner_validation": baseline,
                    "final_inner_validation": final_metrics,
                    "final_gate": final_gate,
                },
                "elapsed_seconds": float(time.time() - start_time),
                "history": history,
                "protocol": {
                    "runtime_layout_remains_target_free": True,
                    "pose_or_ground_truth_not_available_to_visual_encoder": True,
                    "inner_validation_query_disjoint": True,
                    "inner_gate_target_free_static_selector": True,
                    "heldout_validation_or_test_not_run": True,
                    "no_render": True,
                    "no_image_retrieval_or_submap": True,
                },
                "rgb_cache_device": str(cache_device),
                "rgb_cache_rank0": cache.summary(),
            }
            _atomic_json_save(history, history_path)
            _atomic_json_save(summary, summary_path)
            return summary
        return {}
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    train_candidate_absolute_appearance_fusion(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
