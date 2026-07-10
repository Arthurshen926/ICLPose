"""Train real-image RADIO selector + coarse + measurement jointly from full joint caches."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.build_real_radio_joint_cache import (
    REFERENCED_MANIFEST_FORMAT,
    RealRadioReferencedJointSampleProvider,
    SfMTrackObservationIndex,
    load_track_observation_index,
)
from feature_extract.vfm.matcha_coarse_fine_adapter import save_matcha_coarse_fine_adapter
from feature_extract.vfm.matcha_joint_training import (
    MatchaJointTrainingConfig,
    MatchaJointTrainingSet,
    joint_run_as_coarse_fine_adapter_run,
    load_matcha_joint_model,
    load_matcha_joint_training_set_npz,
    save_matcha_joint_model,
    train_matcha_joint_model,
    train_matcha_joint_model_from_manifest,
    train_matcha_joint_model_from_sample_provider,
)


def _initialize_distributed_runtime(args: argparse.Namespace) -> dict[str, int | bool | str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if int(world_size) <= 1:
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "backend": "",
        }
    if not torch.cuda.is_available():
        raise RuntimeError("distributed real RADIO training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    if local_rank < 0 or local_rank >= int(torch.cuda.device_count()):
        raise ValueError(f"LOCAL_RANK={local_rank} is invalid for {torch.cuda.device_count()} CUDA devices")
    torch.cuda.set_device(int(local_rank))
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    rank = int(torch.distributed.get_rank())
    args.device = f"cuda:{local_rank}"
    return {
        "enabled": True,
        "rank": int(rank),
        "local_rank": int(local_rank),
        "world_size": int(torch.distributed.get_world_size()),
        "backend": str(torch.distributed.get_backend()),
    }


def _finish_distributed_runtime(runtime: dict[str, int | bool | str]) -> None:
    if not bool(runtime.get("enabled", False)):
        return
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _validate_joint_localization_training_set(
    samples: MatchaJointTrainingSet,
    *,
    source: str,
    require_measurement_supervision: bool = False,
    require_landmark_retrieval_supervision: bool = False,
) -> dict[str, object]:
    missing: list[str] = []
    if samples.query_feature_maps is None or samples.render_feature_maps is None:
        missing.append("full-map query/reference feature maps")
    if samples.sample_pair_indices is None or samples.query_cell_indices is None or samples.render_cell_indices is None:
        missing.append("full-map correspondence pair/cell indices")
    if samples.query_rgb_images is None or samples.render_rgb_images is None:
        missing.append("RGB measurement images")
    if bool(require_measurement_supervision):
        fine_required = (
            samples.fine_sample_pair_indices,
            samples.fine_query_cell_indices,
            samples.fine_render_cell_indices,
            samples.fine_query_offset_labels,
            samples.fine_render_offset_labels,
        )
        if any(value is None for value in fine_required):
            missing.append("fine correspondence supervision for measurement loss")
    track_ids = None if samples.sample_track_ids is None else np.asarray(samples.sample_track_ids, dtype=np.int64)
    positive_mask = np.ones((samples.coarse_fine_samples.sample_count,), dtype=bool)
    if samples.sample_no_match_labels is not None:
        positive_mask &= np.asarray(samples.sample_no_match_labels, dtype=np.int64) == 0
    if samples.sample_ignore_mask is not None:
        positive_mask &= ~np.asarray(samples.sample_ignore_mask, dtype=bool)
    valid_positive_track_ids = (
        track_ids is not None
        and bool(np.any(positive_mask))
        and bool(np.all(track_ids[positive_mask] >= 0))
    )
    landmark_track_ids = (
        None
        if samples.landmark_track_ids is None
        else np.asarray(samples.landmark_track_ids, dtype=np.int64).reshape(-1)
    )
    valid_landmark_retrieval = (
        landmark_track_ids is not None
        and landmark_track_ids.size > 0
        and bool(np.all(landmark_track_ids >= 0))
        and samples.landmark_query_xy is not None
        and samples.landmark_reference_xy is not None
        and samples.pair_query_image_sizes is not None
        and samples.pair_reference_image_sizes is not None
    )
    if bool(require_landmark_retrieval_supervision) and not valid_landmark_retrieval:
        missing.append("continuous SfM observation supervision for query-to-landmark retrieval loss")
    if missing:
        raise ValueError(f"{source} joint cache is not a full-map real localization cache; missing {', '.join(missing)}")
    query_maps = np.asarray(samples.query_feature_maps, dtype=np.float32)
    reference_maps = np.asarray(samples.render_feature_maps, dtype=np.float32)
    query_rgb = np.asarray(samples.query_rgb_images, dtype=np.float32)
    reference_rgb = np.asarray(samples.render_rgb_images, dtype=np.float32)
    if int(query_maps.shape[0]) != int(reference_maps.shape[0]):
        raise ValueError(f"{source} query/reference feature maps must contain the same pair count")
    if int(query_maps.shape[1]) != int(reference_maps.shape[1]):
        raise ValueError(f"{source} query/reference feature-map channels must match")
    if int(query_maps.shape[1]) != int(samples.coarse_fine_samples.input_dim):
        raise ValueError(f"{source} feature-map channels must match coarse_fine_samples input_dim")
    if int(query_rgb.shape[0]) != int(query_maps.shape[0]) or int(reference_rgb.shape[0]) != int(reference_maps.shape[0]):
        raise ValueError(f"{source} RGB image count must match feature-map pair count")
    return {
        "source": str(source),
        "sample_count": int(samples.coarse_fine_samples.sample_count),
        "pair_count": int(query_maps.shape[0]),
        "input_dim": int(samples.coarse_fine_samples.input_dim),
        "query_feature_map_shape": [int(value) for value in query_maps.shape],
        "reference_feature_map_shape": [int(value) for value in reference_maps.shape],
        "query_rgb_shape": [int(value) for value in query_rgb.shape],
        "reference_rgb_shape": [int(value) for value in reference_rgb.shape],
        "has_measurement_fine_supervision": bool(samples.fine_sample_pair_indices is not None),
        "has_landmark_track_supervision": bool(valid_positive_track_ids),
        "landmark_track_supervision_count": int(0 if landmark_track_ids is None else landmark_track_ids.size),
        "landmark_xyz_supervision_count": int(
            0
            if samples.landmark_track_xyz is None
            else np.count_nonzero(np.isfinite(np.asarray(samples.landmark_track_xyz)).all(axis=1))
        ),
    }


def _load_first_manifest_shard(path: Path) -> tuple[MatchaJointTrainingSet, dict[str, object]]:
    manifest_path = Path(path)
    metadata = json.loads(manifest_path.read_text())
    shards = list(metadata.get("shards", []))
    if not shards:
        raise ValueError(f"{manifest_path} contains no joint-cache shards")
    shard_path = Path(str(shards[0]["path"]))
    if not shard_path.is_absolute():
        shard_path = manifest_path.parent / shard_path
    samples, _sample_metadata = load_matcha_joint_training_set_npz(shard_path)
    output_metadata = dict(metadata)
    output_metadata["audit_shard"] = str(shard_path)
    return samples, output_metadata


def _manifest_format(path: Path) -> str:
    return str(json.loads(Path(path).read_text()).get("format", ""))


def _is_referenced_manifest(path: Path) -> bool:
    return _manifest_format(Path(path)) == REFERENCED_MANIFEST_FORMAT


def _compact_referenced_metadata(metadata: dict[str, object]) -> dict[str, object]:
    output = dict(metadata)
    records = output.pop("records", None)
    if isinstance(records, list):
        output.setdefault("record_count", int(len(records)))
        output["records_preview"] = [
            {
                "query_id": str(item.get("query_id", "")),
                "reference_image_id": str(item.get("reference_image_id", "")),
                "row_count": int(item.get("row_count", len(item.get("row_indices", [])))),
            }
            for item in records[:3]
            if isinstance(item, dict)
        ]
    return output


def _load_referenced_joint_provider(
    path: Path,
    *,
    source: str,
    require_measurement_supervision: bool = False,
    require_landmark_retrieval_supervision: bool = False,
    feature_cache_size: int = 4,
    rgb_cache_size: int = 8,
    track_xyz_by_id: dict[int, np.ndarray] | None = None,
    track_observation_index: SfMTrackObservationIndex | None = None,
) -> tuple[RealRadioReferencedJointSampleProvider, MatchaJointTrainingSet, dict[str, object], dict[str, object]]:
    provider_kwargs: dict[str, object] = {
        "feature_cache_size": int(feature_cache_size),
        "rgb_cache_size": int(rgb_cache_size),
    }
    if track_xyz_by_id is not None:
        provider_kwargs["track_xyz_by_id"] = track_xyz_by_id
    if track_observation_index is not None:
        provider_kwargs["track_observation_index"] = track_observation_index
    provider = RealRadioReferencedJointSampleProvider(Path(path), **provider_kwargs)
    samples = provider.get(0)
    metadata = _compact_referenced_metadata(dict(getattr(provider, "metadata", json.loads(Path(path).read_text()))))
    metadata["audit_record_index"] = 0
    metadata["provider_sample_count"] = int(len(provider))
    audit = _validate_joint_localization_training_set(
        samples,
        source=str(source),
        require_measurement_supervision=bool(require_measurement_supervision),
        require_landmark_retrieval_supervision=bool(require_landmark_retrieval_supervision),
    )
    return provider, samples, metadata, audit


def _load_joint_set(
    path: Path,
    *,
    is_manifest: bool,
    source: str,
    require_measurement_supervision: bool = False,
    require_landmark_retrieval_supervision: bool = False,
) -> tuple[MatchaJointTrainingSet, dict[str, object], dict[str, object]]:
    if bool(is_manifest) and _is_referenced_manifest(Path(path)):
        _provider, samples, metadata, audit = _load_referenced_joint_provider(
            Path(path),
            source=str(source),
            require_measurement_supervision=bool(require_measurement_supervision),
            require_landmark_retrieval_supervision=bool(require_landmark_retrieval_supervision),
        )
        return samples, metadata, audit
    samples, metadata = (
        _load_first_manifest_shard(Path(path))
        if bool(is_manifest)
        else load_matcha_joint_training_set_npz(Path(path))
    )
    audit = _validate_joint_localization_training_set(
        samples,
        source=str(source),
        require_measurement_supervision=bool(require_measurement_supervision),
        require_landmark_retrieval_supervision=bool(require_landmark_retrieval_supervision),
    )
    return samples, dict(metadata), audit


def _resolve_radio_dual_dims(samples: MatchaJointTrainingSet, *, fine_input_dim: int, coarse_input_dim: int) -> tuple[int, int]:
    input_dim = int(samples.coarse_fine_samples.input_dim)
    fine = int(fine_input_dim)
    coarse = int(coarse_input_dim)
    if fine <= 0 and coarse <= 0:
        if input_dim % 2 != 0:
            raise ValueError("fine_input_dim/coarse_input_dim are required when input_dim is odd")
        return input_dim // 2, input_dim // 2
    if fine <= 0:
        fine = input_dim - coarse
    if coarse <= 0:
        coarse = input_dim - fine
    if fine <= 0 or coarse <= 0 or fine + coarse != input_dim:
        raise ValueError("fine_input_dim + coarse_input_dim must equal joint cache input_dim")
    return fine, coarse


def _build_config(args: argparse.Namespace, samples: MatchaJointTrainingSet) -> MatchaJointTrainingConfig:
    fine_dim, coarse_dim = (int(args.fine_input_dim), int(args.coarse_input_dim))
    if str(args.model_type) == "radio_dual_attention":
        fine_dim, coarse_dim = _resolve_radio_dual_dims(
            samples,
            fine_input_dim=int(args.fine_input_dim),
            coarse_input_dim=int(args.coarse_input_dim),
        )
    return MatchaJointTrainingConfig(
        model_type=str(args.model_type),
        output_dim=int(args.output_dim),
        residual_hidden_dim=int(args.residual_hidden_dim),
        fine_input_dim=int(fine_dim),
        coarse_input_dim=int(coarse_dim),
        attention_hidden_dim=int(args.attention_hidden_dim),
        attention_depth=int(args.attention_depth),
        attention_heads=int(args.attention_heads),
        attention_patch_size=int(args.attention_patch_size),
        attention_upsample_mode=str(args.attention_upsample_mode),
        attention_fusion_mode=str(args.attention_fusion_mode),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        temperature=float(args.temperature),
        dual_softmax_weight=float(args.dual_softmax_weight),
        offset_loss_weight=float(args.offset_loss_weight),
        pair_fine_loss_weight=float(args.pair_fine_loss_weight),
        query_pair_fine_loss_weight=float(args.query_pair_fine_loss_weight),
        fine_continuous_loss_weight=float(args.fine_continuous_loss_weight),
        fine_loss_mode=str(args.fine_loss_mode),
        fine_uncertainty_loss_weight=float(args.fine_uncertainty_loss_weight),
        pair_confidence_loss_weight=float(args.pair_confidence_loss_weight),
        dense_heatmap_loss_weight=float(args.dense_heatmap_loss_weight),
        rgb_keypoint_loss_weight=float(args.rgb_keypoint_loss_weight),
        rgb_keypoint_position_loss_weight=float(args.rgb_keypoint_position_loss_weight),
        repeatability_loss_weight=float(args.repeatability_loss_weight),
        local_fine_transformer_loss_weight=float(args.local_fine_transformer_loss_weight),
        local_window_fine_loss_weight=float(args.local_window_fine_loss_weight),
        local_window_fine_mode=str(args.local_window_fine_mode),
        patch_correlation_loss_weight=float(args.patch_correlation_loss_weight),
        patch_correlation_window_size=int(args.patch_correlation_window_size),
        hard_negative_weight=float(args.hard_negative_weight),
        hard_negative_margin=float(args.hard_negative_margin),
        coarse_candidate_rank_loss_weight=float(args.coarse_candidate_rank_loss_weight),
        coarse_candidate_rank_margin=float(args.coarse_candidate_rank_margin),
        hard_false_match_weight=float(args.hard_false_match_weight),
        hard_false_match_margin=float(args.hard_false_match_margin),
        landmark_retrieval_loss_weight=float(args.landmark_retrieval_loss_weight),
        landmark_retrieval_temperature=float(args.landmark_retrieval_temperature),
        landmark_prototype_history_mix=float(args.landmark_prototype_history_mix),
        landmark_memory_capacity=int(args.landmark_memory_capacity),
        landmark_memory_momentum=float(args.landmark_memory_momentum),
        landmark_memory_candidate_pool_size=int(args.landmark_memory_candidate_pool_size),
        landmark_semantic_hard_negatives_per_query=int(args.landmark_semantic_hard_negatives_per_query),
        landmark_geometry_hard_negatives_per_track=int(args.landmark_geometry_hard_negatives_per_track),
        landmark_random_negatives=int(args.landmark_random_negatives),
        landmark_max_memory_negatives=int(args.landmark_max_memory_negatives),
        landmark_dustbin_logit=float(args.landmark_dustbin_logit),
        group_size=int(args.group_size),
        input_norm_mode=str(args.input_norm_mode),
        gate_mode=str(args.gate_mode),
        residual_gate_scale=float(args.residual_gate_scale),
        map_pair_batch_size=int(args.map_pair_batch_size),
        measurement_patch_loss_weight=float(args.measurement_patch_loss_weight),
        measurement_patch_direct_loss_weight=float(args.measurement_patch_direct_loss_weight),
        measurement_patch_epe_weight=float(args.measurement_patch_epe_weight),
        measurement_patch_dustbin_bce_weight=float(args.measurement_patch_dustbin_bce_weight),
        measurement_patch_batch_size=int(args.measurement_patch_batch_size),
        measurement_patch_max_samples_per_pair=int(args.measurement_patch_max_samples_per_pair),
        measurement_patch_search_radius_px=float(args.measurement_patch_search_radius_px),
        measurement_patch_context_radius_px=float(args.measurement_patch_context_radius_px),
        measurement_patch_step_px=float(args.measurement_patch_step_px),
        measurement_patch_coarse_search_radius_px=float(args.measurement_patch_coarse_search_radius_px),
        measurement_patch_coarse_step_px=float(args.measurement_patch_coarse_step_px),
        measurement_patch_feature_dim=int(args.measurement_patch_feature_dim),
        measurement_patch_hidden_dim=int(args.measurement_patch_hidden_dim),
        measurement_patch_target_heatmap_sigma_px=float(args.measurement_patch_target_heatmap_sigma_px),
        measurement_patch_dustbin_positive_weight=float(args.measurement_patch_dustbin_positive_weight),
        measurement_patch_encoder_arch=str(args.measurement_patch_encoder_arch),
        measurement_patch_input_mode=str(args.measurement_patch_input_mode),
        device=str(args.device),
        seed=int(args.seed),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--joint_cache", default="")
    source.add_argument("--joint_cache_manifest", default="")
    parser.add_argument("--validation_joint_cache", default="")
    parser.add_argument("--validation_joint_cache_manifest", default="")
    parser.add_argument("--warm_start_joint_checkpoint", default="")
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_joint_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--manifest_shard_cache_size", type=int, default=1)
    parser.add_argument("--manifest_steps_per_shard", type=int, default=1)
    parser.add_argument("--referenced_feature_cache_size", type=int, default=4)
    parser.add_argument("--referenced_rgb_cache_size", type=int, default=8)
    parser.add_argument("--landmark_track_observations", default="")
    parser.add_argument("--allow_sparse_landmark_csv_fallback", action="store_true")
    parser.add_argument("--provider_prefetch_workers", type=int, default=0)
    parser.add_argument("--provider_prefetch_depth", type=int, default=0)
    parser.add_argument("--provider_gradient_accumulation_pairs", type=int, default=1)
    parser.add_argument("--provider_pair_batch_size", type=int, default=1)
    parser.add_argument("--provider_progress_interval_steps", type=int, default=0)
    parser.add_argument("--provider_empty_cuda_cache_interval_steps", type=int, default=0)
    parser.add_argument("--model_type", choices=("radio_dual_attention", "residual_adapter"), default="residual_adapter")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--residual_hidden_dim", type=int, default=256)
    parser.add_argument("--fine_input_dim", type=int, default=0)
    parser.add_argument("--coarse_input_dim", type=int, default=0)
    parser.add_argument("--attention_hidden_dim", type=int, default=128)
    parser.add_argument("--attention_depth", type=int, default=1)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--attention_patch_size", type=int, default=4)
    parser.add_argument("--attention_upsample_mode", choices=("bilinear", "pixel_shuffle"), default="pixel_shuffle")
    parser.add_argument("--attention_fusion_mode", choices=("legacy", "matcha_original"), default="matcha_original")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dual_softmax_weight", type=float, default=1.0)
    parser.add_argument("--offset_loss_weight", type=float, default=0.25)
    parser.add_argument("--pair_fine_loss_weight", type=float, default=0.0)
    parser.add_argument("--query_pair_fine_loss_weight", type=float, default=0.0)
    parser.add_argument("--fine_continuous_loss_weight", type=float, default=0.25)
    parser.add_argument("--fine_loss_mode", choices=("ce", "ce_plus_continuous", "continuous"), default="ce_plus_continuous")
    parser.add_argument("--fine_uncertainty_loss_weight", type=float, default=0.05)
    parser.add_argument("--pair_confidence_loss_weight", type=float, default=0.1)
    parser.add_argument("--dense_heatmap_loss_weight", type=float, default=0.25)
    parser.add_argument("--rgb_keypoint_loss_weight", type=float, default=0.0)
    parser.add_argument("--rgb_keypoint_position_loss_weight", type=float, default=0.0)
    parser.add_argument("--repeatability_loss_weight", type=float, default=0.0)
    parser.add_argument("--local_fine_transformer_loss_weight", type=float, default=0.0)
    parser.add_argument("--local_window_fine_loss_weight", type=float, default=0.5)
    parser.add_argument("--local_window_fine_mode", choices=("mlp", "correlation"), default="correlation")
    parser.add_argument("--patch_correlation_loss_weight", type=float, default=0.0)
    parser.add_argument("--patch_correlation_window_size", type=int, default=3)
    parser.add_argument("--hard_negative_weight", type=float, default=0.1)
    parser.add_argument("--hard_negative_margin", type=float, default=0.2)
    parser.add_argument("--coarse_candidate_rank_loss_weight", type=float, default=0.1)
    parser.add_argument("--coarse_candidate_rank_margin", type=float, default=0.2)
    parser.add_argument("--hard_false_match_weight", type=float, default=0.0)
    parser.add_argument("--hard_false_match_margin", type=float, default=0.2)
    parser.add_argument("--landmark_retrieval_loss_weight", type=float, default=0.25)
    parser.add_argument("--landmark_retrieval_temperature", type=float, default=0.07)
    parser.add_argument("--landmark_prototype_history_mix", type=float, default=0.5)
    parser.add_argument("--landmark_memory_capacity", type=int, default=65536)
    parser.add_argument("--landmark_memory_momentum", type=float, default=0.9)
    parser.add_argument("--landmark_memory_candidate_pool_size", type=int, default=4096)
    parser.add_argument("--landmark_semantic_hard_negatives_per_query", type=int, default=16)
    parser.add_argument("--landmark_geometry_hard_negatives_per_track", type=int, default=8)
    parser.add_argument("--landmark_random_negatives", type=int, default=128)
    parser.add_argument("--landmark_max_memory_negatives", type=int, default=2048)
    parser.add_argument("--landmark_dustbin_logit", type=float, default=0.0)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--input_norm_mode", choices=("identity", "layernorm"), default="identity")
    parser.add_argument("--gate_mode", choices=("residual", "sigmoid"), default="residual")
    parser.add_argument("--residual_gate_scale", type=float, default=0.1)
    parser.add_argument("--map_pair_batch_size", type=int, default=8)
    parser.add_argument("--measurement_patch_loss_weight", type=float, default=1.0)
    parser.add_argument("--measurement_patch_direct_loss_weight", type=float, default=0.25)
    parser.add_argument("--measurement_patch_epe_weight", type=float, default=0.05)
    parser.add_argument("--measurement_patch_dustbin_bce_weight", type=float, default=0.25)
    parser.add_argument("--measurement_patch_batch_size", type=int, default=128)
    parser.add_argument("--measurement_patch_max_samples_per_pair", type=int, default=512)
    parser.add_argument("--measurement_patch_search_radius_px", type=float, default=8.0)
    parser.add_argument("--measurement_patch_context_radius_px", type=float, default=8.0)
    parser.add_argument("--measurement_patch_step_px", type=float, default=1.0)
    parser.add_argument("--measurement_patch_coarse_search_radius_px", type=float, default=0.0)
    parser.add_argument("--measurement_patch_coarse_step_px", type=float, default=0.0)
    parser.add_argument("--measurement_patch_feature_dim", type=int, default=32)
    parser.add_argument("--measurement_patch_hidden_dim", type=int, default=64)
    parser.add_argument("--measurement_patch_target_heatmap_sigma_px", type=float, default=0.5)
    parser.add_argument("--measurement_patch_dustbin_positive_weight", type=float, default=1.0)
    parser.add_argument("--measurement_patch_encoder_arch", default="simple")
    parser.add_argument("--measurement_patch_input_mode", default="rgb")
    parser.add_argument("--validation_interval", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    distributed_runtime = _initialize_distributed_runtime(args)
    started = time.perf_counter()
    train_path = Path(args.joint_cache_manifest or args.joint_cache)
    require_measurement = float(args.measurement_patch_loss_weight) > 0.0
    require_landmark_retrieval = float(args.landmark_retrieval_loss_weight) > 0.0
    track_observation_index = (
        load_track_observation_index(Path(args.landmark_track_observations))
        if str(args.landmark_track_observations)
        else None
    )
    track_xyz_by_id = (
        None if track_observation_index is None else track_observation_index.track_xyz_by_id
    )
    train_provider = None
    train_is_referenced_manifest = bool(args.joint_cache_manifest) and _is_referenced_manifest(train_path)
    if train_is_referenced_manifest:
        train_provider, train_samples, train_metadata, train_audit = _load_referenced_joint_provider(
            train_path,
            source="train",
            require_measurement_supervision=bool(require_measurement),
            require_landmark_retrieval_supervision=bool(require_landmark_retrieval),
            feature_cache_size=int(args.referenced_feature_cache_size),
            rgb_cache_size=int(args.referenced_rgb_cache_size),
            track_xyz_by_id=track_xyz_by_id,
            track_observation_index=track_observation_index,
        )
    else:
        train_samples, train_metadata, train_audit = _load_joint_set(
            train_path,
            is_manifest=bool(args.joint_cache_manifest),
            source="train",
            require_measurement_supervision=bool(require_measurement),
            require_landmark_retrieval_supervision=bool(require_landmark_retrieval),
        )
    if train_provider is not None and bool(require_landmark_retrieval):
        dataset_audit = train_provider.landmark_retrieval_audit()
        if (
            str(dataset_audit.get("supervision_source", "")) != "sfm_common_track_observations"
            and not bool(args.allow_sparse_landmark_csv_fallback)
        ):
            raise ValueError(
                "landmark retrieval requires full SfM common-track observations; pass "
                "--landmark_track_observations or explicitly allow the sparse CSV diagnostic fallback"
            )
        dataset_audit["memory_capacity"] = int(args.landmark_memory_capacity)
        dataset_audit["memory_capacity_covers_unique_tracks"] = bool(
            int(args.landmark_memory_capacity) >= int(dataset_audit["unique_track_count"])
        )
        train_audit["landmark_retrieval_dataset"] = dataset_audit
    validation_samples = None
    validation_provider = None
    validation_manifest_path = None
    validation_metadata: dict[str, object] = {}
    validation_audit: dict[str, object] = {}
    validation_path = Path(args.validation_joint_cache_manifest or args.validation_joint_cache) if (args.validation_joint_cache_manifest or args.validation_joint_cache) else None
    if validation_path is not None:
        validation_is_referenced = bool(args.validation_joint_cache_manifest) and _is_referenced_manifest(validation_path)
        if validation_is_referenced:
            validation_provider, validation_samples, validation_metadata, validation_audit = _load_referenced_joint_provider(
                validation_path,
                source="validation",
                require_measurement_supervision=bool(require_measurement),
                require_landmark_retrieval_supervision=bool(require_landmark_retrieval),
                feature_cache_size=int(args.referenced_feature_cache_size),
                rgb_cache_size=int(args.referenced_rgb_cache_size),
                track_xyz_by_id=track_xyz_by_id,
                track_observation_index=track_observation_index,
            )
        else:
            validation_samples, validation_metadata, validation_audit = _load_joint_set(
                validation_path,
                is_manifest=bool(args.validation_joint_cache_manifest),
                source="validation",
                require_measurement_supervision=bool(require_measurement),
                require_landmark_retrieval_supervision=bool(require_landmark_retrieval),
            )
        if bool(args.validation_joint_cache_manifest) and not validation_is_referenced:
            validation_manifest_path = validation_path
            validation_samples = None
    cfg = _build_config(args, train_samples)
    warm_start_model = None
    if str(args.warm_start_joint_checkpoint):
        warm_start_model = load_matcha_joint_model(Path(args.warm_start_joint_checkpoint), device=str(args.device)).model
    if train_provider is not None:
        run = train_matcha_joint_model_from_sample_provider(
            int(len(train_provider)),
            train_provider.get,
            cfg,
            validation_sample_count=0 if validation_provider is None else int(len(validation_provider)),
            get_validation_sample=None if validation_provider is None else validation_provider.get,
            validation_interval=int(args.validation_interval),
            steps_per_sample=int(args.manifest_steps_per_shard),
            provider_gradient_accumulation_pairs=int(args.provider_gradient_accumulation_pairs),
            provider_pair_batch_size=int(args.provider_pair_batch_size),
            provider_prefetch_workers=int(args.provider_prefetch_workers),
            provider_prefetch_depth=int(args.provider_prefetch_depth),
            provider_progress_interval_steps=int(args.provider_progress_interval_steps),
            provider_empty_cuda_cache_interval_steps=int(args.provider_empty_cuda_cache_interval_steps),
            warm_start_model=warm_start_model,
            provider_name="real_radio_referenced_manifest",
        )
    elif bool(args.joint_cache_manifest):
        if bool(distributed_runtime["enabled"]):
            raise ValueError("distributed training is currently supported only for image-referenced lazy manifests")
        run = train_matcha_joint_model_from_manifest(
            train_path,
            cfg,
            validation_samples=validation_samples,
            validation_manifest_path=validation_manifest_path,
            validation_interval=int(args.validation_interval),
            shard_cache_size=int(args.manifest_shard_cache_size),
            steps_per_shard=int(args.manifest_steps_per_shard),
            warm_start_model=warm_start_model,
        )
    else:
        if bool(distributed_runtime["enabled"]):
            raise ValueError("distributed training is currently supported only for image-referenced lazy manifests")
        run = train_matcha_joint_model(
            train_samples,
            cfg,
            validation_samples=validation_samples,
            validation_interval=int(args.validation_interval),
            warm_start_model=warm_start_model,
        )
    if int(distributed_runtime["rank"]) != 0:
        _finish_distributed_runtime(distributed_runtime)
        return
    adapter_run = joint_run_as_coarse_fine_adapter_run(run)
    save_matcha_coarse_fine_adapter(adapter_run, Path(args.output_model))
    save_matcha_joint_model(run, Path(args.output_joint_model))
    summary = {
        "stage": "real_radio_joint_localization_training",
        "elapsed_sec": float(time.perf_counter() - started),
        "joint_cache": {
            "path": str(train_path),
            "is_manifest": bool(args.joint_cache_manifest),
            "metadata": train_metadata,
            "audit": train_audit,
        },
        "validation_joint_cache": {
            "path": "" if validation_path is None else str(validation_path),
            "is_manifest": bool(args.validation_joint_cache_manifest),
            "metadata": validation_metadata,
            "audit": validation_audit,
        },
        "joint_training_contract": {
            "requires_full_feature_maps": True,
            "requires_full_map_correspondence_indices": True,
            "requires_fine_measurement_supervision": bool(require_measurement),
            "requires_sfm_track_identity": bool(require_landmark_retrieval),
            "landmark_retrieval_target": "query_full_map_to_multi_observation_track_prototype",
            "landmark_hard_negative_sources": ["same_pair_covisible", "global_semantic_memory", "nearby_3d_memory"],
            "landmark_track_observations": str(args.landmark_track_observations),
            "requires_rgb_measurement_images": True,
            "rejects_row_only_sample_cache": True,
            "reference_source": "real_image",
            "default_feature_source": "radio_final",
        },
        "distributed_runtime": dict(distributed_runtime),
        "config": asdict(cfg),
        "training": dict(run.summary),
        "outputs": {
            "adapter_model": str(args.output_model),
            "joint_model": str(args.output_joint_model),
            "summary": str(args.summary_json),
        },
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    _finish_distributed_runtime(distributed_runtime)


if __name__ == "__main__":  # pragma: no cover
    main()
