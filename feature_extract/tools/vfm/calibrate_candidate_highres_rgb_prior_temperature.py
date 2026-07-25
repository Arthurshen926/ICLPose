"""Calibrate a target-free top-L prior temperature for high-resolution RGB.

The RGB encoder is frozen before this command starts.  The command uses only
train-query targets to choose one scalar that tempers the *fixed* coarse top-L
prior, then evaluates that scalar on a query-disjoint inner fold with the
normal/support-permuted/zero-visual and hard-repeat gate.  It never writes pose
or target arrays into the runtime calibration artifact, and passing this gate
only permits a separate held-out audit; it never permits direct PnP use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.train_candidate_highres_rgb_multiscale_likelihood import (
    CHECKPOINT_FORMAT,
    FIXED_FINAL_EPOCH_SELECTION_POLICY,
    _DistributedState,
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _finalize_distributed,
    _initialize_distributed,
    _output_conflict,
    _partition_train_queries_for_inner_validation,
    _pose_scores,
    _query_batch_from_group,
    active_source_names,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    evaluate_inner_gate,
    inner_gate_decision,
    target_free_query_owner_costs,
    validate_rgb_coordinate_bridge,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CandidateHighresRGBMultiscaleLikelihood,
    temper_candidate_prior_probabilities,
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
    permute_support_patch_appearance,
    runtime_from_target_free_layout,
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


CALIBRATION_FORMAT = "candidate_highres_rgb_prior_temperature_calibration_v2"


def parse_temperature_grid(value: str) -> tuple[float, ...]:
    """Parse a deterministic, duplicate-free positive temperature grid."""

    try:
        values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError("candidate prior-temperature grid is invalid") from error
    if (
        not values
        or any(not math.isfinite(item) or item <= 0.0 for item in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("candidate prior-temperature grid is invalid")
    return tuple(sorted(values))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument("--hard-repeat-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=0)
    parser.add_argument(
        "--candidate-prior-temperature-grid",
        default="0.35,0.4,0.45,0.5,0.75,1.0",
    )
    parser.add_argument("--minimum-effective-candidate-count-median", type=float, default=2.5)
    parser.add_argument("--minimum-effective-candidate-count-p10", type=float, default=1.1)
    parser.add_argument("--max-points-per-query", type=int, default=32)
    parser.add_argument("--max-hard-repeat-edges-per-query", type=int, default=256)
    parser.add_argument("--minimum-pose-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-pose-gap", type=float, default=0.05)
    parser.add_argument("--minimum-pose-permutation-delta", type=float, default=0.05)
    parser.add_argument("--minimum-pose-zero-visual-delta", type=float, default=0.05)
    parser.add_argument("--minimum-hard-repeat-eligible-query-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-hard-repeat-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-hard-repeat-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-repeat-permutation-delta", type=float, default=0.05)
    parser.add_argument("--minimum-hard-repeat-zero-visual-delta", type=float, default=0.05)
    parser.add_argument("--support-permutation-shift", type=int, default=1)
    parser.add_argument("--rgb-cache-gb", type=float, default=6.0)
    parser.add_argument("--rgb-cache-dtype", choices=("float16", "uint8"), default="uint8")
    parser.add_argument("--rgb-cache-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[float, ...]:
    grid = parse_temperature_grid(str(args.candidate_prior_temperature_grid))
    finite_positive = (
        "minimum_effective_candidate_count_median",
        "minimum_effective_candidate_count_p10",
        "max_points_per_query",
        "max_hard_repeat_edges_per_query",
        "minimum_pose_win_fraction",
        "minimum_pose_gap",
        "minimum_pose_permutation_delta",
        "minimum_pose_zero_visual_delta",
        "minimum_hard_repeat_eligible_query_fraction",
        "minimum_hard_repeat_win_fraction",
        "minimum_hard_repeat_gap",
        "minimum_hard_repeat_permutation_delta",
        "minimum_hard_repeat_zero_visual_delta",
        "rgb_cache_gb",
    )
    if (
        any(not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0.0 for name in finite_positive)
        or int(args.inner_validation_fold_count) < 2
        or int(args.inner_validation_fold_index) < 0
        or int(args.inner_validation_fold_index) >= int(args.inner_validation_fold_count)
        or int(args.support_permutation_shift) <= 0
    ):
        raise ValueError("high-resolution RGB prior-temperature calibration arguments are invalid")
    return grid


def _save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _query_ids_hash(query_ids: Sequence[str]) -> str:
    payload = json.dumps(tuple(sorted(str(query_id) for query_id in query_ids)), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_checkpoint_model(
    *, path: Path, image_sizes: np.ndarray, device: torch.device
) -> tuple[CandidateHighresRGBMultiscaleLikelihood, dict[str, object]]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("high-resolution RGB checkpoint format is invalid")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("pnp_integration_allowed") is not False
        or metadata.get("appearance_control_geometry_fixed") is not True
        or metadata.get("checkpoint_selection_policy") != FIXED_FINAL_EPOCH_SELECTION_POLICY
        or metadata.get("inner_validation_used_for_model_selection") is not False
        or not isinstance(metadata.get("config"), Mapping)
    ):
        raise ValueError("high-resolution RGB checkpoint contract is invalid")
    training = metadata.get("training")
    if (
        not isinstance(training, Mapping)
        or training.get("support_permutation_control")
        != "fixed_runtime_geometry_and_validity_with_rgb_patch_derangement_only_v2"
        or not isinstance(training.get("checkpoint_selection"), Mapping)
        or training["checkpoint_selection"].get("policy") != FIXED_FINAL_EPOCH_SELECTION_POLICY
        or training["checkpoint_selection"].get("inner_validation_used_for_model_selection") is not False
        or int(training["checkpoint_selection"].get("selected_epoch", -1))
        != int(training.get("epochs", -2))
    ):
        raise ValueError("high-resolution RGB checkpoint appearance-control or selection contract is invalid")
    config = dict(metadata["config"])
    required = (
        "fine_search_radius_px",
        "fine_context_radius_px",
        "broad_search_radius_px",
        "broad_context_radius_px",
        "texture_feature_dim",
        "hidden_dim",
        "edge_chunk_size",
        "rgb_temperature",
        "max_abs_edge_log_ratio",
    )
    if any(name not in config for name in required):
        raise ValueError("high-resolution RGB checkpoint configuration is incomplete")
    model = CandidateHighresRGBMultiscaleLikelihood(
        image_sizes=torch.from_numpy(np.asarray(image_sizes, dtype=np.float32)),
        fine_search_radius_px=float(config["fine_search_radius_px"]),
        fine_context_radius_px=float(config["fine_context_radius_px"]),
        broad_search_radius_px=float(config["broad_search_radius_px"]),
        broad_context_radius_px=float(config["broad_context_radius_px"]),
        texture_feature_dim=int(config["texture_feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        edge_chunk_size=int(config["edge_chunk_size"]),
        rgb_temperature=float(config["rgb_temperature"]),
        max_abs_edge_log_ratio=float(config["max_abs_edge_log_ratio"]),
    ).to(device)
    model.load_state_dict(dict(state_dict), strict=True)
    model.eval()
    return model, dict(metadata)


def effective_candidate_count_metrics(
    *, candidate_probabilities: np.ndarray, null_probabilities: np.ndarray, temperature: float
) -> dict[str, float]:
    """Return ESS diagnostics for the conditional, non-null top-L posterior."""

    candidates = torch.from_numpy(np.asarray(candidate_probabilities, dtype=np.float32))
    null = torch.from_numpy(np.asarray(null_probabilities, dtype=np.float32))
    tempered = temper_candidate_prior_probabilities(
        candidate_probabilities=candidates,
        null_probabilities=null,
        temperature=float(temperature),
    )
    conditional = tempered / (1.0 - null).clamp_min(1e-8).unsqueeze(1)
    ess = 1.0 / conditional.square().sum(dim=1).clamp_min(1e-12)
    return {
        "effective_candidate_count_median": float(torch.median(ess).item()),
        "effective_candidate_count_p10": float(torch.quantile(ess, 0.10).item()),
        "top1_probability_median": float(torch.median(conditional.max(dim=1).values).item()),
    }


def select_temperature(
    *,
    temperatures: Sequence[float],
    train_metrics: Mapping[float, Mapping[str, float]],
    effective_counts: Mapping[float, Mapping[str, float]],
    args: argparse.Namespace,
) -> float:
    """Select solely from train-query metrics plus label-free ESS safeguards."""

    eligible: list[float] = []
    for raw_temperature in temperatures:
        temperature = float(raw_temperature)
        metrics = train_metrics[temperature]
        ess = effective_counts[temperature]
        if (
            float(metrics["normal_pose_gap"]) >= float(args.minimum_pose_gap)
            and float(metrics["normal_pose_win_fraction"])
            >= float(args.minimum_pose_win_fraction)
            and float(ess["effective_candidate_count_median"])
            >= float(args.minimum_effective_candidate_count_median)
            and float(ess["effective_candidate_count_p10"])
            >= float(args.minimum_effective_candidate_count_p10)
        ):
            eligible.append(temperature)
    if not eligible:
        raise ValueError("no prior temperature passes train-only pose and ESS eligibility")
    return max(
        eligible,
        key=lambda temperature: (
            float(train_metrics[temperature]["normal_minus_permuted_pose_gap"]),
            float(train_metrics[temperature]["normal_pose_gap"]),
            float(train_metrics[temperature]["normal_pose_win_fraction"]),
            -float(temperature),
        ),
    )


@torch.no_grad()
def collect_temperature_pose_metrics(
    *,
    model: CandidateHighresRGBMultiscaleLikelihood,
    groups: Mapping[str, object],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    query_ids: Sequence[str],
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    cache: TensorImageLRUCache,
    cache_device: torch.device,
    state: _DistributedState,
    source: str,
    temperatures: Sequence[float],
    max_points_per_query: int,
    max_abs_pose_log_ratio: float,
    support_permutation_shift: int,
    amp_enabled: bool,
) -> dict[float, dict[str, float]]:
    """Score all calibration temperatures from one visual forward per branch."""

    if not query_ids or not temperatures:
        raise ValueError("prior-temperature metric collection inputs are invalid")
    totals = torch.zeros((len(temperatures), 5), dtype=torch.float64, device=state.device)
    for position, query_id in enumerate(query_ids):
        if position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        if group is None or int(group.point_count) > int(max_points_per_query):
            raise ValueError("prior-temperature calibration query group is invalid")
        batch = _query_batch_from_group(
            group=group,
            complete_runtime=complete_runtime,
            point_positions=np.arange(group.point_count, dtype=np.int64),
            device=state.device,
        )
        query_patches, support_patches = _crop_runtime_rgb_patches(
            runtime=batch.runtime,
            image_ids=image_ids,
            image_root=image_root,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=float(model.full_patch_radius_px),
            step_px=1.0,
            cache=cache,
            device=state.device,
            cache_device=cache_device,
        )
        permuted_patches = permute_support_patch_appearance(
            runtime=batch.runtime,
            support_patches=support_patches,
            shift=int(support_permutation_shift),
        )
        if torch.equal(permuted_patches, support_patches):
            raise ValueError("high-resolution RGB support permutation did not alter appearance")
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            normal_prediction = model(
                runtime=batch.runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                active_sources=active_source_names(source),
            )
            permuted_prediction = model(
                runtime=batch.runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=permuted_patches,
                active_sources=active_source_names(source),
            )
            for index, temperature in enumerate(temperatures):
                normal_correct, normal_wrong = _pose_scores(
                    runtime=batch.runtime,
                    prediction=normal_prediction,
                    correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                    correct_projection_valid=batch.correct_projection_valid,
                    wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                    wrong_projection_valid=batch.wrong_projection_valid,
                    source=source,
                    max_abs_pose_log_ratio=float(max_abs_pose_log_ratio),
                    candidate_prior_temperature=float(temperature),
                )
                permuted_correct, permuted_wrong = _pose_scores(
                    runtime=batch.runtime,
                    prediction=permuted_prediction,
                    correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                    correct_projection_valid=batch.correct_projection_valid,
                    wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                    wrong_projection_valid=batch.wrong_projection_valid,
                    source=source,
                    max_abs_pose_log_ratio=float(max_abs_pose_log_ratio),
                    candidate_prior_temperature=float(temperature),
                )
                normal_gap = normal_correct - normal_wrong.max()
                permuted_gap = permuted_correct - permuted_wrong.max()
                # ``_pose_scores`` intentionally keeps the correct score as
                # shape ``[1]`` for the training loss.  Calibration aggregates
                # one scalar per query, so normalize both gaps before stacking
                # them with scalar counters.
                normal_gap = normal_gap.reshape(())
                permuted_gap = permuted_gap.reshape(())
                totals[index] += torch.stack(
                    [
                        normal_gap.to(torch.float64),
                        (normal_gap > 0.0).to(torch.float64),
                        permuted_gap.to(torch.float64),
                        (permuted_gap > 0.0).to(torch.float64),
                        torch.ones((), dtype=torch.float64, device=state.device),
                    ]
                )
    if state.enabled:
        distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
    values: dict[float, dict[str, float]] = {}
    for index, temperature in enumerate(temperatures):
        count = float(totals[index, 4].item())
        if count <= 0.0:
            raise RuntimeError("prior-temperature calibration evaluated no query")
        normal_gap = float((totals[index, 0] / count).item())
        permuted_gap = float((totals[index, 2] / count).item())
        values[float(temperature)] = {
            "query_count": count,
            "normal_pose_gap": normal_gap,
            "normal_pose_win_fraction": float((totals[index, 1] / count).item()),
            "permuted_pose_gap": permuted_gap,
            "permuted_pose_win_fraction": float((totals[index, 3] / count).item()),
            "normal_minus_permuted_pose_gap": normal_gap - permuted_gap,
            "normal_minus_zero_visual_pose_gap": normal_gap,
        }
    return values


def calibrate_candidate_highres_rgb_prior_temperature(args: argparse.Namespace) -> dict[str, object]:
    temperatures = _validate_args(args)
    state = _initialize_distributed(str(args.device))
    try:
        output_dir = Path(args.output_dir)
        output_path = output_dir / "prior_temperature_calibration.json"
        if _output_conflict(state=state, paths=(output_path,)) and not bool(args.force):
            raise FileExistsError("refusing to overwrite high-resolution RGB prior-temperature calibration")
        layout_path = Path(args.rgb_spatial_layout)
        targets_path = Path(args.training_targets)
        hard_repeat_path = Path(args.hard_repeat_targets)
        checkpoint_path = Path(args.checkpoint)
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        targets = load_candidate_pose_rgb_spatial_training_targets(targets_path)
        validate_training_layout_and_targets(
            layout=layout,
            targets=targets,
            layout_sha256=file_sha256_short(layout_path),
        )
        groups = build_train_query_groups(layout=layout, targets=targets)
        hard_repeat_groups = build_hard_repeat_query_targets(
            layout=layout,
            targets=targets,
            hard_repeat_targets=load_candidate_pose_rgb_spatial_hard_repeat_targets(hard_repeat_path),
            layout_sha256=file_sha256_short(layout_path),
            targets_sha256=file_sha256_short(targets_path),
        )
        train_ids, validation_ids = _partition_train_queries_for_inner_validation(
            query_ids=tuple(sorted(groups)),
            fold_count=int(args.inner_validation_fold_count),
            fold_index=int(args.inner_validation_fold_index),
        )
        headers = load_context_attention_source_headers(
            radio_final_context_cache=Path(args.radio_final_context_cache),
            radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
            alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
            expected_radio_checkpoint="",
        )
        image_ids = np.asarray(headers.image_ids).astype(str)
        image_sizes = np.asarray(headers.image_sizes, dtype=np.int64)
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("prior-temperature calibration requires one processed image size")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=headers.metadata_by_name["radio_final"],
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        model, checkpoint_metadata = _load_checkpoint_model(
            path=checkpoint_path, image_sizes=image_sizes, device=state.device
        )
        lineage = checkpoint_metadata.get("lineage")
        if (
            not isinstance(lineage, Mapping)
            or str(lineage.get("layout_sha256", "")) != file_sha256_short(layout_path)
            or str(lineage.get("training_targets_sha256", "")) != file_sha256_short(targets_path)
            or str(lineage.get("hard_repeat_targets_sha256", "")) != file_sha256_short(hard_repeat_path)
        ):
            raise ValueError("prior-temperature checkpoint lineage differs from calibration inputs")
        config = checkpoint_metadata["config"]
        source = str(config.get("source", ""))
        if source not in {"fine", "broad", "combined"}:
            raise ValueError("prior-temperature checkpoint source is invalid")
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        owner_costs = target_free_query_owner_costs(layout=layout, groups=groups)
        train_rows = np.concatenate([groups[query_id].layout_rows for query_id in train_ids])
        effective_counts = {
            float(temperature): effective_candidate_count_metrics(
                candidate_probabilities=np.asarray(layout.candidate_prior_probabilities)[train_rows],
                null_probabilities=np.asarray(layout.null_probabilities)[train_rows],
                temperature=float(temperature),
            )
            for temperature in temperatures
        }
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        cache_device = torch.device("cpu") if str(args.rgb_cache_device) == "cpu" else state.device
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        train_metrics = collect_temperature_pose_metrics(
            model=model,
            groups=groups,
            complete_runtime=complete_runtime,
            query_ids=train_ids,
            image_ids=image_ids,
            image_root=Path(args.image_root),
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            cache=cache,
            cache_device=cache_device,
            state=state,
            source=source,
            temperatures=temperatures,
            max_points_per_query=int(args.max_points_per_query),
            max_abs_pose_log_ratio=float(config["max_abs_pose_log_ratio"]),
            support_permutation_shift=int(args.support_permutation_shift),
            amp_enabled=amp_enabled,
        )
        chosen = select_temperature(
            temperatures=temperatures,
            train_metrics=train_metrics,
            effective_counts=effective_counts,
            args=args,
        )
        selected_tensor = torch.tensor([float(chosen)], dtype=torch.float64, device=state.device)
        if state.enabled:
            distributed.broadcast(selected_tensor, src=0)
        chosen = float(selected_tensor.item())
        validation_metrics = evaluate_inner_gate(
            model=model,
            groups=groups,
            hard_repeat_groups=hard_repeat_groups,
            complete_runtime=complete_runtime,
            query_ids=validation_ids,
            image_ids=image_ids,
            image_root=Path(args.image_root),
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            patch_radius_px=float(model.full_patch_radius_px),
            cache=cache,
            cache_device=cache_device,
            state=state,
            source=source,
            max_points_per_query=int(args.max_points_per_query),
            max_hard_repeat_edges_per_query=int(args.max_hard_repeat_edges_per_query),
            pose_margin=0.0,
            max_abs_pose_log_ratio=float(config["max_abs_pose_log_ratio"]),
            support_permutation_shift=int(args.support_permutation_shift),
            amp_enabled=amp_enabled,
            seed=0,
            candidate_prior_temperature=chosen,
        )
        gate = inner_gate_decision(metrics=validation_metrics, args=args)
        if state.rank != 0:
            return {}
        payload: dict[str, object] = {
            "format": CALIBRATION_FORMAT,
            "runtime_safe": True,
            "contains_target_fields": False,
            "calibration_uses_train_only_targets": True,
            "runtime_inputs": [
                "frozen_topl_candidate_probabilities",
                "explicit_null_probability",
                "candidate_specific_highres_rgb_likelihood",
            ],
            "runtime_excludes": [
                "pose_matrix",
                "projection_offset",
                "reprojection_residual",
                "ground_truth_label",
                "track_id",
                "candidate_rank",
                "coarse_score_as_encoder_input",
            ],
            "mixture_semantics": "tempered_fixed_topl_marginal_with_explicit_null_v1",
            "candidate_identity_semantics": "all_positive_topl_candidates_remain_in_mixture_v1",
            "appearance_control_semantics": "fixed_runtime_geometry_and_validity_with_rgb_patch_derangement_only_v2",
            "checkpoint_selection_policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
            "checkpoint_inner_validation_used_for_model_selection": False,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": file_sha256_short(checkpoint_path),
            "source": source,
            "candidate_prior_temperature": chosen,
            "temperature_grid": list(temperatures),
            "effective_candidate_counts": {str(key): value for key, value in effective_counts.items()},
            "train_selection_metrics": {str(key): value for key, value in train_metrics.items()},
            "selection": {
                "split": "train_only",
                "query_count": len(train_ids),
                "query_ids_sha256": _query_ids_hash(train_ids),
                "rule": "max_train_normal_minus_permuted_pose_gap_subject_to_train_pose_and_label_free_ess_guards_v1",
            },
            "inner_validation": {
                "query_count": len(validation_ids),
                "query_ids_sha256": _query_ids_hash(validation_ids),
                "metrics": validation_metrics,
                "gate": gate,
            },
            "lineage": {
                "layout_sha256": file_sha256_short(layout_path),
                "training_targets_sha256": file_sha256_short(targets_path),
                "hard_repeat_targets_sha256": file_sha256_short(hard_repeat_path),
                "source_image_manifest_sha256": str(
                    headers.metadata_by_name["radio_final"].get("source_image_manifest_sha256", "")
                ),
                "projection_space_id": str(layout.metadata.get("projection_space_id", "")),
                "descriptor_space_id": str(layout.metadata.get("descriptor_space_id", "")),
                "rgb_coordinate_bridge": rgb_bridge,
            },
            "safety": {
                "minimum_effective_candidate_count_median": float(
                    args.minimum_effective_candidate_count_median
                ),
                "minimum_effective_candidate_count_p10": float(
                    args.minimum_effective_candidate_count_p10
                ),
                "prior_temperature_not_an_argmax": True,
                "heldout_evaluation_allowed": bool(gate["passed"]),
                "pnp_integration_allowed": False,
                "promotion_allowed": False,
            },
            "protocol": {
                "no_render": True,
                "no_image_retrieval_or_submap": True,
                "query_disjoint_inner_validation": True,
                "checkpoint_fixed_final_epoch_without_inner_validation_model_selection": True,
                "owner_cost_range": {"min": int(min(owner_costs.values())), "max": int(max(owner_costs.values()))},
            },
            "rgb_cache_rank0": cache.summary(),
        }
        _save_json(output_path, payload)
        return payload
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    calibrate_candidate_highres_rgb_prior_temperature(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
