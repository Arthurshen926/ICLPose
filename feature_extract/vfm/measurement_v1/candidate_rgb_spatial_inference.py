"""Target-free RGB spatial-likelihood inference for frozen candidate groups."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    POSE_VIEW_MIXTURE_SEMANTICS,
    normalized_spatial_log_probabilities_with_dustbin,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_inference import (
    CandidateRGBInferenceData,
    prepare_candidate_rgb_inference_batch,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_spatial_export import (
    NORMALIZED_SPATIAL_DUSTBIN_SEMANTICS,
    _architecture_signature,
    _load_model,
    factorize_pose_view_ensemble,
)
from feature_extract.vfm.measurement_v1.rgb_data_contract import (
    require_inference_compatible_contracts,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import TensorImageLRUCache


SPATIAL_INFERENCE_FORMAT = "candidate_spatial_likelihood_v7"


def _text(rows: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    return np.asarray([str(row.get(key, "")) for row in rows], dtype=np.str_)


def _integer(
    rows: Sequence[Mapping[str, object]], key: str, *, default: int = -1
) -> np.ndarray:
    return np.asarray(
        [int(str(row.get(key, default) or default)) for row in rows], dtype=np.int64
    )


def _floating(
    rows: Sequence[Mapping[str, object]], key: str, *, default: float = np.nan
) -> np.ndarray:
    values = []
    for row in rows:
        text = str(row.get(key, "")).strip()
        values.append(float(default) if not text else float(text))
    return np.asarray(values, dtype=np.float32)


def _forward(
    model: torch.nn.Module,
    batch: Mapping[str, object],
    *,
    device: torch.device,
    use_amp: bool,
) -> Any:
    with torch.autocast(
        device_type=device.type,
        dtype=(torch.float16 if device.type == "cuda" else torch.bfloat16),
        enabled=bool(use_amp and device.type == "cuda"),
    ):
        return model(
            query_patches_by_group=batch["query_patches_by_group"].to(device),
            support_patches=batch["support_patches"].to(device),
            pair_group_indices=batch["pair_group_indices"].to(device),
            pair_candidate_indices=batch["pair_candidate_indices"].to(device),
            pair_view_slots=batch["pair_view_slots"].to(device),
            pair_view_probabilities=batch["pair_view_probabilities"].to(device),
            candidate_valid=batch["candidate_valid"].to(device),
        )


def export_candidate_rgb_spatial_inference(
    *,
    checkpoints: Sequence[Path],
    inference_evidence: Path,
    train_rows_csv: Path,
    validation_rows_csv: Path,
    test_rows_csv: Path,
    split_name: str,
    image_root: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    batch_size: int = 128,
    device: str = "cuda",
    image_cache_max_gb: float = 9.0,
    image_cache_dtype: str = "float16",
    use_amp: bool = True,
    query_shard_count: int = 1,
    query_shard_index: int = 0,
) -> dict[str, Any]:
    split = str(split_name)
    if split not in {"train", "validation", "test"}:
        raise ValueError("RGB spatial inference supports train/validation/test")
    if not checkpoints:
        raise ValueError("RGB spatial inference requires at least one checkpoint")
    if int(batch_size) <= 0:
        raise ValueError("RGB spatial inference batch size must be positive")
    if int(query_shard_count) <= 0 or not 0 <= int(query_shard_index) < int(
        query_shard_count
    ):
        raise ValueError("RGB inference query shard is invalid")
    torch_device = torch.device(
        device
        if torch.cuda.is_available() or not str(device).startswith("cuda")
        else "cpu"
    )
    if torch_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    models = [_load_model(Path(path), torch_device) for path in checkpoints]
    signature = _architecture_signature(models[0])
    if any(_architecture_signature(model) != signature for model in models[1:]):
        raise ValueError("independent RGB ensemble checkpoints differ")
    for model in models:
        if (
            str(model._checkpoint_format) != "independent_rgb_candidate_verifier_v5"
            or str(model._dustbin_probability_semantics)
            != NORMALIZED_SPATIAL_DUSTBIN_SEMANTICS
            or not bool(model.pose_view_mixture_enabled)
        ):
            raise ValueError(
                "strict RGB inference requires normalized v5 pose-view checkpoints"
            )

    data = CandidateRGBInferenceData(
        inference_evidence=Path(inference_evidence),
        rows_by_split={
            "train": Path(train_rows_csv),
            "validation": Path(validation_rows_csv),
            "test": Path(test_rows_csv),
        },
        max_views=int(models[0].max_views),
    )
    runtime_contract = data.runtime_contract(
        image_root=Path(image_root),
        image_width=int(image_width),
        image_height=int(image_height),
    )
    for checkpoint_path, model in zip(checkpoints, models):
        require_inference_compatible_contracts(
            model._data_contract,
            runtime_contract,
            context=f"strict RGB inference checkpoint {checkpoint_path}",
        )

    cache_dtype = {"float16": torch.float16, "float32": torch.float32}.get(
        str(image_cache_dtype).lower()
    )
    if cache_dtype is None:
        raise ValueError("image_cache_dtype must be float16 or float32")
    cache_max_bytes = (
        None
        if float(image_cache_max_gb) <= 0.0
        else int(float(image_cache_max_gb) * 1024**3)
    )
    image_cache = TensorImageLRUCache(
        max_bytes=cache_max_bytes,
        storage_dtype=(cache_dtype if torch_device.type == "cuda" else torch.float32),
    )
    indices = data.indices_by_split[split]
    indices = indices[data.has_any_rgb[indices]]
    indices = indices[int(query_shard_index) :: int(query_shard_count)]
    if len(indices) == 0:
        raise ValueError(f"{split} has no RGB-supported candidate groups")

    rows_out: list[dict[str, str]] = []
    log_probability_blocks: list[np.ndarray] = []
    dustbin_blocks: list[np.ndarray] = []
    entropy_blocks: list[np.ndarray] = []
    covariance_trace_blocks: list[np.ndarray] = []
    support_view_probability_blocks: list[np.ndarray] = []
    offsets_xy: np.ndarray | None = None
    amp_enabled = bool(use_amp and torch_device.type == "cuda")
    started = time.perf_counter()
    total_batches = int(math.ceil(len(indices) / int(batch_size)))
    with torch.inference_mode():
        for batch_number, start in enumerate(
            range(0, len(indices), int(batch_size)), start=1
        ):
            batch_indices = indices[start : start + int(batch_size)].tolist()
            batch = prepare_candidate_rgb_inference_batch(
                data,
                batch_indices,
                image_root=Path(image_root),
                image_width=int(image_width),
                image_height=int(image_height),
                image_cache=image_cache,
                image_cache_device=torch_device,
                crop_radius_px=float(models[0].crop_radius_px),
                step_px=float(models[0].step_px),
            )
            predictions = [
                _forward(model, batch, device=torch_device, use_amp=amp_enabled)
                for model in models
            ]
            model_joint = torch.stack(
                [
                    normalized_spatial_log_probabilities_with_dustbin(
                        prediction.view_spatial_logits.float(),
                        prediction.view_measurement_validity_logits.float(),
                    )
                    for prediction in predictions
                ],
                dim=0,
            )
            model_view = torch.stack(
                [
                    prediction.view_pose_mixture_probabilities.float()
                    for prediction in predictions
                ],
                dim=0,
            )
            ensemble_joint, ensemble_view = factorize_pose_view_ensemble(
                model_joint, model_view
            )
            local_mass = torch.logsumexp(ensemble_joint[:, :-1], dim=1)
            local_log_probability = ensemble_joint[:, :-1] - local_mass[:, None]
            validity_probability = torch.exp(local_mass)
            probability = torch.exp(local_log_probability)
            local_offsets = predictions[0].view_spatial_offsets_xy.detach().float()
            if any(
                not torch.equal(
                    prediction.view_spatial_offsets_xy,
                    predictions[0].view_spatial_offsets_xy,
                )
                for prediction in predictions[1:]
            ):
                raise RuntimeError("RGB ensemble spatial supports differ")
            local_offsets_np = local_offsets.cpu().numpy().astype(np.float32)
            if offsets_xy is None:
                offsets_xy = local_offsets_np
            elif not np.array_equal(offsets_xy, local_offsets_np):
                raise RuntimeError("RGB spatial support changed between batches")
            local_offsets_device = local_offsets.to(device=probability.device)
            mean_offset = probability @ local_offsets_device
            second_moment = torch.sum(
                probability
                * torch.sum(local_offsets_device.square(), dim=1)[None],
                dim=1,
            )
            covariance_trace = second_moment - torch.sum(mean_offset.square(), dim=1)
            entropy = -torch.sum(probability * local_log_probability, dim=1)
            flat_rows = batch["flat_rows"]
            if not isinstance(flat_rows, list) or len(flat_rows) != len(
                local_log_probability
            ):
                raise RuntimeError("RGB inference rows and predictions are misaligned")
            rows_out.extend(flat_rows)
            log_probability_blocks.append(
                local_log_probability.cpu().numpy().astype(np.float16)
            )
            dustbin_blocks.append(
                (1.0 - validity_probability).cpu().numpy().astype(np.float32)
            )
            entropy_blocks.append(entropy.cpu().numpy().astype(np.float32))
            covariance_trace_blocks.append(
                covariance_trace.clamp_min(0.0).cpu().numpy().astype(np.float32)
            )
            support_view_probability_blocks.append(
                ensemble_view.cpu().numpy().astype(np.float32)
            )
            if batch_number == 1 or batch_number % 5 == 0 or batch_number == total_batches:
                elapsed = time.perf_counter() - started
                print(
                    f"[{split}] RGB inference batch {batch_number}/{total_batches} "
                    f"elapsed={elapsed:.1f}s cache_items={len(image_cache)}",
                    flush=True,
                )

    if offsets_xy is None or not rows_out:
        raise RuntimeError("RGB spatial inference produced no rows")
    local_log_probabilities = np.concatenate(log_probability_blocks, axis=0)
    dustbin_probabilities = np.concatenate(dustbin_blocks, axis=0)
    likelihood_entropy = np.concatenate(entropy_blocks, axis=0)
    covariance_trace = np.concatenate(covariance_trace_blocks, axis=0)
    support_view_probabilities = np.concatenate(
        support_view_probability_blocks, axis=0
    )
    if len(local_log_probabilities) != len(rows_out):
        raise RuntimeError("RGB spatial inference arrays have different row counts")
    conditional_mass = np.sum(
        np.exp(local_log_probabilities.astype(np.float64)), axis=1
    )
    joint_mass = (1.0 - dustbin_probabilities) * conditional_mass + dustbin_probabilities
    if (
        np.max(np.abs(conditional_mass - 1.0), initial=0.0) > 2e-3
        or np.max(np.abs(joint_mass - 1.0), initial=0.0) > 2e-3
    ):
        raise RuntimeError("RGB spatial inference probability mass is invalid")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / f"{SPATIAL_INFERENCE_FORMAT}.npz"
    checkpoint_manifest = [
        {"path": str(path), "sha256": file_sha256_short(Path(path))}
        for path in checkpoints
    ]
    rows_path = data.rows_paths[split]
    metadata = {
        "format": SPATIAL_INFERENCE_FORMAT,
        "format_version": 7,
        "rows_csv": str(rows_path),
        "rows_csv_sha256": file_sha256_short(rows_path),
        "candidate_evidence_sha256": data.source_candidate_evidence_sha256,
        "candidate_inference_evidence": str(inference_evidence),
        "candidate_inference_evidence_sha256": file_sha256_short(
            Path(inference_evidence)
        ),
        "measurement_checkpoint_sha256": "+".join(
            item["sha256"] for item in checkpoint_manifest
        ),
        "measurement_checkpoints": checkpoint_manifest,
        "data_contract": runtime_contract,
        "checkpoint_data_contract_compatibility": (
            "rgb_inference_coordinate_and_real_image_source_v1;"
            "candidate_inputs_are_runtime_target_free_artifacts"
        ),
        "coordinate_space_id": runtime_contract["coordinate_space"][
            "coordinate_space_id"
        ],
        "image_source_manifest_sha256": runtime_contract["image_source"][
            "sampled_content_manifest_sha256"
        ],
        "query_source": "real_pair",
        "support_patch_warp": "none",
        "image_width": int(image_width),
        "image_height": int(image_height),
        "search_radius_px": float(models[0].search_radius_px),
        "context_radius_px": float(models[0].context_radius_px),
        "step_px": float(models[0].step_px),
        "spatial_probability_semantics": "ensemble_mixture_normalized_k_plus_dustbin",
        "dustbin_probability_semantics": NORMALIZED_SPATIAL_DUSTBIN_SEMANTICS,
        "joint_probability_mass_normalized": True,
        "spatial_target_sigma_px": float(models[0]._spatial_target_sigma_px),
        "support_views_unmarginalized": True,
        "support_view_probability_semantics": POSE_VIEW_MIXTURE_SEMANTICS,
        "support_view_probability_mass": "one_over_all_measured_views_per_candidate",
        "pose_view_ensemble_factorization": (
            "exact_mean_model_sum_view_weight_times_joint_k_plus_dustbin"
        ),
        "missing_support_view_likelihood_ratio": 1.0,
        "pose_or_ground_truth_used_for_inference": False,
        "ground_truth_loaded_by_inference_process": False,
        "contains_ground_truth_arrays": False,
        "input_schema_allowlisted": True,
        "prediction_frozen_before_target_join": True,
        "query_shard_count": int(query_shard_count),
        "query_shard_index": int(query_shard_index),
        "render": False,
        "image_retrieval": False,
        "submap": False,
    }
    np.savez_compressed(
        artifact,
        source_row_indices=np.arange(len(rows_out), dtype=np.int64),
        supervision_source_row_indices=_integer(
            rows_out, "supervision_source_row_index"
        ),
        query_ids=_text(rows_out, "query_id"),
        source_query_rows=_integer(rows_out, "source_query_row"),
        candidate_identity_keys=_text(rows_out, "candidate_identity_key"),
        candidate_measurement_cache_keys=np.full(
            (len(rows_out),), "", dtype=np.str_
        ),
        candidate_measurement_ranks=_integer(
            rows_out, "candidate_measurement_rank"
        ),
        candidate_track_ids=_integer(rows_out, "track_id"),
        candidate_prototype_ids=_integer(rows_out, "candidate_prototype_id"),
        support_image_ids=_text(rows_out, "support_image_id"),
        support_view_ranks=_integer(rows_out, "support_view_rank"),
        support_view_probabilities=support_view_probabilities,
        candidate_prior_probabilities=_floating(
            rows_out, "candidate_assignment_probability"
        ),
        center_xy=np.stack(
            [_floating(rows_out, "center_x"), _floating(rows_out, "center_y")],
            axis=1,
        ),
        offsets_xy=offsets_xy,
        local_log_probabilities=local_log_probabilities,
        dustbin_probabilities=dustbin_probabilities,
        likelihood_entropy=likelihood_entropy,
        likelihood_covariance_trace_px2=covariance_trace,
        measurement_geometry_probabilities=1.0 - dustbin_probabilities,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    elapsed = time.perf_counter() - started
    summary = {
        "stage": "independent_rgb_candidate_spatial_target_free_inference",
        "split": split,
        "protocol": {
            "query_source": "real_pair",
            "render": False,
            "ground_truth_loaded": False,
            "pose_loaded": False,
            "target_free_input_schema_enforced": True,
            "prediction_frozen_before_external_audit": True,
            "per_view_spatial_likelihood_retained": True,
            "support_view_posterior_retained": True,
            "query_sharding_changes_only_execution": True,
        },
        "counts": {
            "candidate_group_count": int(len(indices)),
            "prediction_row_count": int(len(rows_out)),
            "offset_bin_count": int(len(offsets_xy)),
        },
        "runtime": {
            "elapsed_seconds": float(elapsed),
            "prediction_rows_per_second": float(len(rows_out) / max(elapsed, 1e-9)),
            "batch_size_groups": int(batch_size),
            "device": str(torch_device),
            "amp": bool(amp_enabled),
            "query_shard_count": int(query_shard_count),
            "query_shard_index": int(query_shard_index),
        },
        "prediction_distribution": {
            "dustbin_probability_mean": float(np.mean(dustbin_probabilities)),
            "dustbin_probability_median": float(np.median(dustbin_probabilities)),
            "conditional_entropy_mean": float(np.mean(likelihood_entropy)),
            "covariance_trace_px2_mean": float(np.mean(covariance_trace)),
        },
        "image_cache": image_cache.summary(),
        "inputs": {
            "candidate_inference_evidence": str(inference_evidence),
            "candidate_inference_evidence_sha256": file_sha256_short(
                Path(inference_evidence)
            ),
            "source_candidate_evidence_sha256": data.source_candidate_evidence_sha256,
            "rows_csv": str(rows_path),
            "rows_csv_sha256": file_sha256_short(rows_path),
            "checkpoints": checkpoint_manifest,
        },
        "outputs": {
            "spatial_likelihood": str(artifact),
            "spatial_likelihood_sha256": file_sha256_short(artifact),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
