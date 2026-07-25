"""Audit high-resolution RGB as a target-free top-L identity prior.

The existing RGB cost-volume branch was trained primarily as a local spatial
density evaluated *after* a candidate pose is projected.  Before using it in
another pose experiment, this audit asks the stricter question that matters
for identity ambiguity: without a pose projection, can its fixed-support-view
appearance quality move the correct track up inside the frozen global top-L?

The visual forward consumes only the target-free P1 layout and real RGB
patches.  Registered-track labels are joined after normal and support-patch
derangement forwards have completed.  This is train-only diagnostic evidence;
a passing result may justify a new identity-specific training objective, but
never authorizes PnP integration by itself.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.train_candidate_highres_rgb_multiscale_likelihood import (
    CHECKPOINT_FORMAT,
    FIXED_FINAL_EPOCH_SELECTION_POLICY,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _partition_train_queries_for_inner_validation,
    _slice_runtime,
    build_train_query_groups,
    validate_rgb_coordinate_bridge,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
    CandidateHighresRGBMultiscaleLikelihood,
    highres_rgb_candidate_identity_plus_null_logits,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    permute_support_patch_appearance,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_source_headers,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)
from feature_extract.tools.vfm.train_candidate_multiscale_phase_identity_llr import (
    build_exact_identity_query_targets,
)


AUDIT_FORMAT = "candidate_highres_rgb_identity_prior_audit_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--geometry-training-targets", required=True)
    parser.add_argument("--registered-identity-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", choices=("fine", "broad"), default="fine")
    parser.add_argument("--points-per-forward", type=int, default=64)
    parser.add_argument("--inner-fold-count", type=int, default=5)
    parser.add_argument("--inner-fold-index", type=int, default=1)
    parser.add_argument("--support-permutation-shift", type=int, default=1)
    parser.add_argument("--rgb-cache-gb", type=float, default=6.0)
    parser.add_argument("--rgb-cache-dtype", choices=("uint8", "float16"), default="uint8")
    parser.add_argument("--rgb-cache-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--minimum-eligible-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-top1-lift", type=float, default=0.02)
    parser.add_argument("--minimum-mean-rank-reduction", type=float, default=0.25)
    parser.add_argument("--minimum-correct-probability-lift", type=float, default=0.01)
    parser.add_argument("--minimum-control-top1-delta", type=float, default=0.01)
    parser.add_argument("--minimum-control-rank-reduction-delta", type=float, default=0.10)
    parser.add_argument("--minimum-control-correct-probability-delta", type=float, default=0.005)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    values = (
        args.rgb_cache_gb,
        args.minimum_eligible_fraction,
        args.minimum_top1_lift,
        args.minimum_mean_rank_reduction,
        args.minimum_correct_probability_lift,
        args.minimum_control_top1_delta,
        args.minimum_control_rank_reduction_delta,
        args.minimum_control_correct_probability_delta,
    )
    if (
        int(args.points_per_forward) <= 0
        or int(args.inner_fold_count) < 2
        or not 0 <= int(args.inner_fold_index) < int(args.inner_fold_count)
        or int(args.support_permutation_shift) <= 0
        or not all(math.isfinite(float(value)) for value in values)
        or float(args.rgb_cache_gb) <= 0.0
        or not 0.0 < float(args.minimum_eligible_fraction) <= 1.0
        or any(float(value) < 0.0 for value in values[2:])
    ):
        raise ValueError("high-resolution RGB identity audit arguments are invalid")


def _atomic_json_write(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_checkpoint(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("high-resolution RGB identity checkpoint is invalid")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("high-resolution RGB identity checkpoint is incomplete")
    required = {
        "format": CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": 20,
        "fixed_support_view_count": 2,
        "explicit_null": True,
        "projection_after_network_only": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "pnp_integration_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "checkpoint_selection_policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
        "inner_validation_used_for_model_selection": False,
    }
    if any(metadata.get(name) != value for name, value in required.items()):
        raise ValueError("high-resolution RGB identity checkpoint violates the target-free contract")
    excluded = set(str(value) for value in metadata.get("encoder_excludes", ()))
    if not {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
    }.issubset(excluded):
        raise ValueError("high-resolution RGB identity checkpoint encoder exclusions are incomplete")
    config = metadata.get("config")
    lineage = metadata.get("lineage")
    training = metadata.get("training")
    if not isinstance(config, Mapping) or not isinstance(lineage, Mapping) or not isinstance(training, Mapping):
        raise ValueError("high-resolution RGB identity checkpoint metadata is incomplete")
    if (
        training.get("support_permutation_control")
        != "fixed_runtime_geometry_and_validity_with_rgb_patch_derangement_only_v2"
        or not isinstance(training.get("checkpoint_selection"), Mapping)
        or training["checkpoint_selection"].get("policy") != FIXED_FINAL_EPOCH_SELECTION_POLICY
        or training["checkpoint_selection"].get("inner_validation_used_for_model_selection") is not False
    ):
        raise ValueError("high-resolution RGB identity checkpoint selection contract is invalid")
    required_config = (
        "source",
        "fine_search_radius_px",
        "fine_context_radius_px",
        "fine_step_px",
        "broad_search_radius_px",
        "broad_context_radius_px",
        "broad_feature_step_px",
        "broad_output_step_px",
        "texture_feature_dim",
        "hidden_dim",
        "edge_chunk_size",
        "rgb_temperature",
        "max_abs_edge_log_ratio",
    )
    if any(name not in config for name in required_config):
        raise ValueError("high-resolution RGB identity checkpoint configuration is incomplete")
    return dict(metadata), dict(state_dict)


def _load_model(
    *,
    metadata: Mapping[str, object],
    state_dict: Mapping[str, object],
    image_sizes: np.ndarray,
    device: torch.device,
    edge_chunk_size: int,
) -> CandidateHighresRGBMultiscaleLikelihood:
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("high-resolution RGB identity model configuration is invalid")
    model = CandidateHighresRGBMultiscaleLikelihood(
        image_sizes=torch.from_numpy(np.asarray(image_sizes, dtype=np.float32)),
        fine_search_radius_px=float(config["fine_search_radius_px"]),
        fine_context_radius_px=float(config["fine_context_radius_px"]),
        fine_step_px=float(config["fine_step_px"]),
        broad_search_radius_px=float(config["broad_search_radius_px"]),
        broad_context_radius_px=float(config["broad_context_radius_px"]),
        broad_feature_step_px=float(config["broad_feature_step_px"]),
        broad_output_step_px=float(config["broad_output_step_px"]),
        texture_feature_dim=int(config["texture_feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        edge_chunk_size=int(edge_chunk_size),
        rgb_temperature=float(config["rgb_temperature"]),
        max_abs_edge_log_ratio=float(config["max_abs_edge_log_ratio"]),
    ).to(device)
    model.load_state_dict(dict(state_dict), strict=True)
    return model.eval()


def _validate_lineage(
    *,
    metadata: Mapping[str, object],
    layout_path: Path,
    targets_path: Path,
    layout_metadata: Mapping[str, object],
    source_image_manifest_sha256: str,
    source: str,
) -> dict[str, object]:
    config = metadata.get("config")
    lineage = metadata.get("lineage")
    if not isinstance(config, Mapping) or not isinstance(lineage, Mapping):
        raise ValueError("high-resolution RGB identity checkpoint lineage is invalid")
    if (
        str(config.get("source", "")) != str(source)
        or str(lineage.get("layout_sha256", "")) != file_sha256_short(layout_path)
        or str(lineage.get("training_targets_sha256", "")) != file_sha256_short(targets_path)
        or str(lineage.get("projection_space_id", ""))
        != str(layout_metadata.get("projection_space_id", ""))
        or str(lineage.get("descriptor_space_id", ""))
        != str(layout_metadata.get("descriptor_space_id", ""))
        or str(lineage.get("source_image_manifest_sha256", ""))
        != str(source_image_manifest_sha256)
    ):
        raise ValueError("high-resolution RGB identity audit inputs do not match checkpoint lineage")
    bridge = lineage.get("rgb_coordinate_bridge")
    if not isinstance(bridge, Mapping):
        raise ValueError("high-resolution RGB identity checkpoint lacks an RGB coordinate bridge")
    return dict(bridge)


def stable_candidate_ranks(
    *,
    candidate_logits: torch.Tensor,
    target_candidate_indices: torch.Tensor,
    candidate_supported: torch.Tensor,
) -> torch.Tensor:
    """Return deterministic 1-based candidate ranks without using labels as inputs.

    Candidate slots are only a stable tie break in this metric.  They never
    enter the RGB encoder or alter the posterior itself.
    """

    logits = torch.as_tensor(candidate_logits, dtype=torch.float32)
    target = torch.as_tensor(target_candidate_indices, dtype=torch.long, device=logits.device).reshape(-1)
    supported = torch.as_tensor(candidate_supported, dtype=torch.bool, device=logits.device)
    if (
        logits.ndim != 2
        or logits.shape[0] == 0
        or logits.shape[1] < 2
        or target.shape != (len(logits),)
        or supported.shape != logits.shape
        or torch.any(target < 0)
        or torch.any(target >= logits.shape[1])
        or not torch.isfinite(logits).all()
        or not bool(torch.all(supported.gather(1, target[:, None])))
    ):
        raise ValueError("candidate identity rank inputs are invalid")
    target_logits = logits.gather(1, target[:, None])
    slots = torch.arange(logits.shape[1], device=logits.device)[None, :]
    earlier_tie = (logits == target_logits) & (slots < target[:, None])
    preceding = supported & ((logits > target_logits) | earlier_tie)
    return 1 + preceding.sum(dim=1)


def candidate_identity_rank_metrics(
    *,
    candidate_null_logits: torch.Tensor,
    target_candidate_indices: torch.Tensor,
    candidate_supported: torch.Tensor,
) -> dict[str, float]:
    """Summarize exact-track rank under an explicit candidate/null posterior."""

    logits = torch.as_tensor(candidate_null_logits, dtype=torch.float32)
    if logits.ndim != 2 or logits.shape[1] < 3 or not torch.isfinite(logits).all():
        raise ValueError("candidate/null identity logits are invalid")
    candidate_logits = logits[:, :-1]
    ranks = stable_candidate_ranks(
        candidate_logits=candidate_logits,
        target_candidate_indices=target_candidate_indices,
        candidate_supported=candidate_supported,
    ).to(dtype=torch.float32)
    target = torch.as_tensor(target_candidate_indices, dtype=torch.long, device=logits.device).reshape(-1)
    probabilities = torch.softmax(logits, dim=1)
    correct_probability = probabilities[:, :-1].gather(1, target[:, None]).squeeze(1)
    return {
        "observed_count": float(len(ranks)),
        "top1": float((ranks == 1.0).to(dtype=torch.float32).mean().item()),
        "recall_at_5": float((ranks <= 5.0).to(dtype=torch.float32).mean().item()),
        "recall_at_10": float((ranks <= 10.0).to(dtype=torch.float32).mean().item()),
        "recall_at_20": float((ranks <= 20.0).to(dtype=torch.float32).mean().item()),
        "mean_rank": float(ranks.mean().item()),
        "median_rank": float(ranks.median().item()),
        "correct_probability": float(correct_probability.mean().item()),
    }


def candidate_identity_rank_gate(
    *,
    base: Mapping[str, float],
    normal: Mapping[str, float],
    permuted: Mapping[str, float],
    eligible_fraction: float,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    """Require true top-L ordering gain and a support-appearance control gap."""

    required_metrics = ("top1", "mean_rank", "correct_probability", "observed_count")
    required_thresholds = (
        "minimum_eligible_fraction",
        "minimum_top1_lift",
        "minimum_mean_rank_reduction",
        "minimum_correct_probability_lift",
        "minimum_control_top1_delta",
        "minimum_control_rank_reduction_delta",
        "minimum_control_correct_probability_delta",
    )
    if (
        any(name not in base or name not in normal or name not in permuted for name in required_metrics)
        or any(name not in thresholds for name in required_thresholds)
        or not all(math.isfinite(float(value)) for mapping in (base, normal, permuted) for value in mapping.values())
        or not math.isfinite(float(eligible_fraction))
        or not 0.0 <= float(eligible_fraction) <= 1.0
        or float(normal["observed_count"]) <= 0.0
        or float(permuted["observed_count"]) != float(normal["observed_count"])
        or float(base["observed_count"]) != float(normal["observed_count"])
    ):
        raise ValueError("candidate identity rank gate inputs are invalid")
    normal_top1_lift = float(normal["top1"]) - float(base["top1"])
    normal_rank_reduction = float(base["mean_rank"]) - float(normal["mean_rank"])
    normal_probability_lift = float(normal["correct_probability"]) - float(base["correct_probability"])
    permuted_top1_lift = float(permuted["top1"]) - float(base["top1"])
    permuted_rank_reduction = float(base["mean_rank"]) - float(permuted["mean_rank"])
    permuted_probability_lift = float(permuted["correct_probability"]) - float(base["correct_probability"])
    control_top1_delta = normal_top1_lift - permuted_top1_lift
    control_rank_reduction_delta = normal_rank_reduction - permuted_rank_reduction
    control_probability_delta = normal_probability_lift - permuted_probability_lift
    checks = {
        "coverage": float(eligible_fraction) >= float(thresholds["minimum_eligible_fraction"]),
        "top1_lift": normal_top1_lift >= float(thresholds["minimum_top1_lift"]),
        "mean_rank_reduction": normal_rank_reduction
        >= float(thresholds["minimum_mean_rank_reduction"]),
        "correct_probability_lift": normal_probability_lift
        >= float(thresholds["minimum_correct_probability_lift"]),
        "control_top1": control_top1_delta >= float(thresholds["minimum_control_top1_delta"]),
        "control_mean_rank": control_rank_reduction_delta
        >= float(thresholds["minimum_control_rank_reduction_delta"]),
        "control_correct_probability": control_probability_delta
        >= float(thresholds["minimum_control_correct_probability_delta"]),
    }
    return {
        "thresholds": {name: float(thresholds[name]) for name in required_thresholds},
        "eligible_fraction": float(eligible_fraction),
        "normal_top1_lift": normal_top1_lift,
        "normal_mean_rank_reduction": normal_rank_reduction,
        "normal_correct_probability_lift": normal_probability_lift,
        "permuted_top1_lift": permuted_top1_lift,
        "permuted_mean_rank_reduction": permuted_rank_reduction,
        "permuted_correct_probability_lift": permuted_probability_lift,
        "normal_minus_permuted_top1_lift": control_top1_delta,
        "normal_minus_permuted_mean_rank_reduction": control_rank_reduction_delta,
        "normal_minus_permuted_correct_probability_lift": control_probability_delta,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _fixed_prior_logits(runtime: CandidatePoseRGBSpatialRuntime, *, device: torch.device) -> torch.Tensor:
    active = runtime.to(device)
    candidate = active.candidate_probabilities.to(dtype=torch.float32)
    null = active.null_probabilities.to(dtype=torch.float32)
    if (
        torch.any(candidate < 0.0)
        or torch.any(null < 0.0)
        or torch.any(torch.abs(candidate.sum(dim=1) + null - 1.0) > 1e-4)
    ):
        raise ValueError("fixed candidate/null prior is invalid")
    return torch.cat(
        (
            torch.log(candidate.clamp_min(torch.finfo(candidate.dtype).tiny)),
            torch.log(null.clamp_min(torch.finfo(null.dtype).tiny))[:, None],
        ),
        dim=1,
    )


def _observed_identity_rows(
    *,
    observed_candidate_mask: np.ndarray,
    candidate_supervised_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed_candidate_mask, dtype=bool)
    supervised = np.asarray(candidate_supervised_mask, dtype=bool)
    if observed.ndim != 2 or supervised.shape != observed.shape:
        raise ValueError("registered identity audit labels are invalid")
    count = observed.sum(axis=1)
    rows = (count == 1) & supervised.all(axis=1)
    if np.any(observed & ~supervised):
        raise ValueError("registered identity audit observed label is not supervised")
    targets = np.argmax(observed, axis=1).astype(np.int64)
    return rows, targets


def _metric_difference(
    *, base: Mapping[str, float], value: Mapping[str, float]
) -> dict[str, float]:
    return {
        "top1_lift": float(value["top1"]) - float(base["top1"]),
        "mean_rank_reduction": float(base["mean_rank"]) - float(value["mean_rank"]),
        "correct_probability_lift": float(value["correct_probability"])
        - float(base["correct_probability"]),
    }


@torch.no_grad()
def audit_candidate_highres_rgb_identity(args: argparse.Namespace) -> dict[str, object]:
    """Run a train-only query-grouped top-L identity audit."""

    _validate_args(args)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite high-resolution RGB identity audit")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("high-resolution RGB identity audit requested CUDA but CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    paths = {
        "layout": Path(args.rgb_spatial_layout),
        "geometry_targets": Path(args.geometry_training_targets),
        "registered_identity_targets": Path(args.registered_identity_targets),
        "checkpoint": Path(args.checkpoint),
        "radio_final_context_cache": Path(args.radio_final_context_cache),
        "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
        "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("high-resolution RGB identity audit input is missing")
    image_root = Path(args.image_root)
    if not image_root.is_dir():
        raise FileNotFoundError("high-resolution RGB identity image root is missing")

    layout = load_candidate_pose_rgb_spatial_layout(paths["layout"])
    geometry_targets = load_candidate_pose_rgb_spatial_training_targets(paths["geometry_targets"])
    registered_targets = load_candidate_pose_rgb_spatial_training_targets(
        paths["registered_identity_targets"]
    )
    layout_sha = file_sha256_short(paths["layout"])
    validate_training_layout_and_targets(
        layout=layout, targets=geometry_targets, layout_sha256=layout_sha
    )
    validate_training_layout_and_targets(
        layout=layout, targets=registered_targets, layout_sha256=layout_sha
    )
    groups = build_train_query_groups(layout=layout, targets=geometry_targets)
    exact_by_query = build_exact_identity_query_targets(
        groups=groups, identity_targets=registered_targets
    )
    inner_train_ids, heldout_query_ids = _partition_train_queries_for_inner_validation(
        query_ids=tuple(sorted(groups)),
        fold_count=int(args.inner_fold_count),
        fold_index=int(args.inner_fold_index),
    )
    if set(heldout_query_ids) - set(exact_by_query):
        raise ValueError("high-resolution RGB identity audit held-out query labels are incomplete")

    headers = load_context_attention_source_headers(
        radio_final_context_cache=paths["radio_final_context_cache"],
        radio_intermediate_context_cache=paths["radio_intermediate_context_cache"],
        alike_spatial_context_cache=paths["alike_spatial_context_cache"],
        expected_radio_checkpoint="",
    )
    image_ids = np.asarray(headers.image_ids).astype(str)
    image_sizes = np.asarray(headers.image_sizes, dtype=np.int64)
    unique_sizes = np.unique(image_sizes, axis=0)
    if unique_sizes.shape != (1, 2):
        raise ValueError("high-resolution RGB identity audit requires one coordinate image size")
    coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
    rgb_image_size = _discover_rgb_image_size(image_root=image_root, image_id=str(image_ids[0]))
    rgb_bridge = validate_rgb_coordinate_bridge(
        source_metadata=headers.metadata_by_name["radio_final"],
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
    )
    metadata, state_dict = _load_checkpoint(paths["checkpoint"])
    checkpoint_bridge = _validate_lineage(
        metadata=metadata,
        layout_path=paths["layout"],
        targets_path=paths["geometry_targets"],
        layout_metadata=layout.metadata,
        source_image_manifest_sha256=str(
            headers.metadata_by_name["radio_final"].get("source_image_manifest_sha256", "")
        ),
        source=str(args.source),
    )
    if checkpoint_bridge != rgb_bridge:
        raise ValueError("high-resolution RGB identity checkpoint coordinate bridge differs")
    complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
    model = _load_model(
        metadata=metadata,
        state_dict=state_dict,
        image_sizes=image_sizes,
        device=device,
        edge_chunk_size=int(dict(metadata["config"])["edge_chunk_size"]),
    )
    cache = TensorImageLRUCache(
        max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
        storage_dtype=resolve_rgb_image_cache_storage_dtype(str(args.rgb_cache_dtype)),
    )
    cache_device = torch.device(str(args.rgb_cache_device))
    base_logits_parts: list[torch.Tensor] = []
    normal_logits_parts: list[torch.Tensor] = []
    permuted_logits_parts: list[torch.Tensor] = []
    supported_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    normal_target_usable_parts: list[torch.Tensor] = []
    permuted_target_usable_parts: list[torch.Tensor] = []
    per_query: dict[str, dict[str, float]] = {}
    for query_id in heldout_query_ids:
        group = groups[str(query_id)]
        exact = exact_by_query[str(query_id)]
        if not np.array_equal(group.source_point_ids, exact.source_point_ids):
            raise ValueError("high-resolution RGB identity audit source-point alignment differs")
        observed_rows, target_indices = _observed_identity_rows(
            observed_candidate_mask=exact.observed_candidate_mask,
            candidate_supervised_mask=exact.candidate_supervised_mask,
        )
        query_base: list[torch.Tensor] = []
        query_normal: list[torch.Tensor] = []
        query_permuted: list[torch.Tensor] = []
        query_supported: list[torch.Tensor] = []
        query_normal_usable: list[torch.Tensor] = []
        query_permuted_usable: list[torch.Tensor] = []
        query_target: list[np.ndarray] = []
        for begin in range(0, group.point_count, int(args.points_per_forward)):
            stop = min(begin + int(args.points_per_forward), group.point_count)
            positions = np.arange(begin, stop, dtype=np.int64)
            # The target-free runtime is fully constructed before either RGB
            # forward.  No label, pose, residual, or track ID reaches model().
            runtime = _slice_runtime(complete_runtime, group.layout_rows[positions])
            query_patches, support_patches = _crop_runtime_rgb_patches(
                runtime=runtime,
                image_ids=image_ids,
                image_root=image_root,
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=float(model.full_patch_radius_px),
                step_px=1.0,
                cache=cache,
                device=device,
                cache_device=cache_device,
            )
            permuted_support_patches = permute_support_patch_appearance(
                runtime=runtime,
                support_patches=support_patches,
                shift=int(args.support_permutation_shift),
            )
            if torch.equal(support_patches, permuted_support_patches):
                raise ValueError("high-resolution RGB identity support control did not alter patches")
            with (
                torch.autocast(device_type=device.type, enabled=True)
                if device.type == "cuda" and not bool(args.no_amp)
                else nullcontext()
            ):
                normal_prediction = model(
                    runtime=runtime,
                    query_rgb_patches=query_patches,
                    support_rgb_patches=support_patches,
                    active_sources=(str(args.source),),
                )
                permuted_prediction = model(
                    runtime=runtime,
                    query_rgb_patches=query_patches,
                    support_rgb_patches=permuted_support_patches,
                    active_sources=(str(args.source),),
                )
            normal_logits, normal_usable, _ = highres_rgb_candidate_identity_plus_null_logits(
                runtime=runtime,
                prediction=normal_prediction,
                source=str(args.source),
            )
            permuted_logits, permuted_usable, _ = highres_rgb_candidate_identity_plus_null_logits(
                runtime=runtime,
                prediction=permuted_prediction,
                source=str(args.source),
            )
            if not torch.equal(normal_usable, permuted_usable):
                raise RuntimeError("support appearance control changed visual availability")

            # Join registered identity labels only after both target-free
            # visual predictions have been emitted and their availability has
            # been checked.  Observed rows are retained solely for metrics.
            local_observed = observed_rows[positions]
            if not bool(np.any(local_observed)):
                continue
            local_targets = torch.from_numpy(target_indices[positions][local_observed]).to(
                device=device, dtype=torch.long
            )
            selected = torch.from_numpy(np.flatnonzero(local_observed)).to(device=device, dtype=torch.long)
            active_runtime = runtime.to(device)
            query_base.append(_fixed_prior_logits(runtime, device=device).index_select(0, selected).cpu())
            query_normal.append(normal_logits.index_select(0, selected).cpu())
            query_permuted.append(permuted_logits.index_select(0, selected).cpu())
            query_supported.append(
                (active_runtime.candidate_probabilities > 0.0).index_select(0, selected).cpu()
            )
            query_normal_usable.append(
                normal_usable.gather(1, local_targets[:, None]).squeeze(1).cpu()
            )
            query_permuted_usable.append(
                permuted_usable.gather(1, local_targets[:, None]).squeeze(1).cpu()
            )
            query_target.append(local_targets.cpu().numpy())
        if not query_target:
            per_query[str(query_id)] = {"observed_count": 0.0, "common_visual_observed_count": 0.0}
            continue
        query_target_tensor = torch.from_numpy(np.concatenate(query_target, axis=0)).long()
        query_base_tensor = torch.cat(query_base, dim=0)
        query_normal_tensor = torch.cat(query_normal, dim=0)
        query_permuted_tensor = torch.cat(query_permuted, dim=0)
        query_supported_tensor = torch.cat(query_supported, dim=0)
        normal_target_usable = torch.cat(query_normal_usable, dim=0)
        permuted_target_usable = torch.cat(query_permuted_usable, dim=0)
        common = normal_target_usable & permuted_target_usable
        values: dict[str, float] = {
            "observed_count": float(len(query_target_tensor)),
            "common_visual_observed_count": float(common.sum().item()),
        }
        if bool(common.any()):
            values["base_top1"] = candidate_identity_rank_metrics(
                candidate_null_logits=query_base_tensor[common],
                target_candidate_indices=query_target_tensor[common],
                candidate_supported=query_supported_tensor[common],
            )["top1"]
            values["normal_top1"] = candidate_identity_rank_metrics(
                candidate_null_logits=query_normal_tensor[common],
                target_candidate_indices=query_target_tensor[common],
                candidate_supported=query_supported_tensor[common],
            )["top1"]
            values["permuted_top1"] = candidate_identity_rank_metrics(
                candidate_null_logits=query_permuted_tensor[common],
                target_candidate_indices=query_target_tensor[common],
                candidate_supported=query_supported_tensor[common],
            )["top1"]
        per_query[str(query_id)] = values
        base_logits_parts.append(query_base_tensor)
        normal_logits_parts.append(query_normal_tensor)
        permuted_logits_parts.append(query_permuted_tensor)
        supported_parts.append(query_supported_tensor)
        target_parts.append(query_target_tensor)
        normal_target_usable_parts.append(normal_target_usable)
        permuted_target_usable_parts.append(permuted_target_usable)

    if not base_logits_parts:
        raise RuntimeError("high-resolution RGB identity audit found no registered observed rows")
    base_logits = torch.cat(base_logits_parts, dim=0)
    normal_logits = torch.cat(normal_logits_parts, dim=0)
    permuted_logits = torch.cat(permuted_logits_parts, dim=0)
    supported = torch.cat(supported_parts, dim=0)
    target_indices = torch.cat(target_parts, dim=0)
    normal_target_usable = torch.cat(normal_target_usable_parts, dim=0)
    permuted_target_usable = torch.cat(permuted_target_usable_parts, dim=0)
    common_visual = normal_target_usable & permuted_target_usable
    if not bool(common_visual.any()):
        raise RuntimeError("high-resolution RGB identity audit has no common visual observed rows")
    all_observed = {
        "base": candidate_identity_rank_metrics(
            candidate_null_logits=base_logits,
            target_candidate_indices=target_indices,
            candidate_supported=supported,
        ),
        "normal": candidate_identity_rank_metrics(
            candidate_null_logits=normal_logits,
            target_candidate_indices=target_indices,
            candidate_supported=supported,
        ),
        "support_permuted": candidate_identity_rank_metrics(
            candidate_null_logits=permuted_logits,
            target_candidate_indices=target_indices,
            candidate_supported=supported,
        ),
    }
    common_observed = {
        "base": candidate_identity_rank_metrics(
            candidate_null_logits=base_logits[common_visual],
            target_candidate_indices=target_indices[common_visual],
            candidate_supported=supported[common_visual],
        ),
        "normal": candidate_identity_rank_metrics(
            candidate_null_logits=normal_logits[common_visual],
            target_candidate_indices=target_indices[common_visual],
            candidate_supported=supported[common_visual],
        ),
        "support_permuted": candidate_identity_rank_metrics(
            candidate_null_logits=permuted_logits[common_visual],
            target_candidate_indices=target_indices[common_visual],
            candidate_supported=supported[common_visual],
        ),
    }
    thresholds = {
        "minimum_eligible_fraction": float(args.minimum_eligible_fraction),
        "minimum_top1_lift": float(args.minimum_top1_lift),
        "minimum_mean_rank_reduction": float(args.minimum_mean_rank_reduction),
        "minimum_correct_probability_lift": float(args.minimum_correct_probability_lift),
        "minimum_control_top1_delta": float(args.minimum_control_top1_delta),
        "minimum_control_rank_reduction_delta": float(
            args.minimum_control_rank_reduction_delta
        ),
        "minimum_control_correct_probability_delta": float(
            args.minimum_control_correct_probability_delta
        ),
    }
    gate = candidate_identity_rank_gate(
        base=common_observed["base"],
        normal=common_observed["normal"],
        permuted=common_observed["support_permuted"],
        eligible_fraction=float(common_visual.to(dtype=torch.float32).mean().item()),
        thresholds=thresholds,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result: dict[str, object] = {
        "format": AUDIT_FORMAT,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256_short(paths["checkpoint"]),
        "checkpoint_model_config": dict(metadata["config"]),
        "heldout_train_query_ids": list(heldout_query_ids),
        "heldout_train_query_count": len(heldout_query_ids),
        "inner_train_query_count": len(inner_train_ids),
        "all_registered_observed": {
            **all_observed,
            "normal_minus_base": _metric_difference(
                base=all_observed["base"], value=all_observed["normal"]
            ),
            "normal_minus_support_permuted": _metric_difference(
                base=all_observed["support_permuted"], value=all_observed["normal"]
            ),
        },
        "common_visual_registered_observed": {
            **common_observed,
            "normal_minus_base": _metric_difference(
                base=common_observed["base"], value=common_observed["normal"]
            ),
            "normal_minus_support_permuted": _metric_difference(
                base=common_observed["support_permuted"], value=common_observed["normal"]
            ),
            "eligible_fraction_of_registered_observed": float(
                common_visual.to(dtype=torch.float32).mean().item()
            ),
        },
        "gate": gate,
        "per_query": per_query,
        "cache": cache.summary(),
        "coordinate_bridge": rgb_bridge,
        "protocol": {
            "train_only_query_disjoint_inner_fold": True,
            "target_free_runtime_forward": True,
            "registered_identity_labels_joined_after_normal_and_control_forwards": True,
            "candidate_identity_encoder_inputs": [
                "query_rgb_anchor_patch",
                "fixed_candidate_support_rgb_patches",
            ],
            "excluded_from_encoder": [
                "pose",
                "projection_offset",
                "residual",
                "track_id",
                "candidate_rank",
                "coarse_score",
                "registered_identity_label",
            ],
            "fixed_global_topl": True,
            "fixed_support_views": 2,
            "explicit_null": True,
            "support_appearance_control": "support_patch_derangement_geometry_and_candidate_prior_fixed_v1",
            "render": False,
            "image_retrieval_or_submap": False,
            "pnp_integration_allowed": False,
            "passing_next_stage": "identity_specific_highres_rgb_training_design_only",
        },
        "inputs": {
            "layout": {"path": str(paths["layout"]), "sha256": layout_sha},
            "geometry_targets": {
                "path": str(paths["geometry_targets"]),
                "sha256": file_sha256_short(paths["geometry_targets"]),
            },
            "registered_identity_targets": {
                "path": str(paths["registered_identity_targets"]),
                "sha256": file_sha256_short(paths["registered_identity_targets"]),
            },
            "context_source_manifest_sha256": str(
                headers.metadata_by_name["radio_final"].get("source_image_manifest_sha256", "")
            ),
            "context_caches": {
                name: str(paths[name])
                for name in (
                    "radio_final_context_cache",
                    "radio_intermediate_context_cache",
                    "alike_spatial_context_cache",
                )
            },
            "image_root": str(image_root),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(summary_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    result = audit_candidate_highres_rgb_identity(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
