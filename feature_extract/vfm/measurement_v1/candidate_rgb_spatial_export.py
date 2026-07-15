"""Export independent-RGB per-view spatial likelihoods for latent pose inference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_training import (
    CandidateRGBTrainingData,
    _bool_text,
    _float_text,
    _forward_batch,
    _prepare_batch,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    GT_POSE_SPATIAL_DENSITY_SEMANTICS,
    IndependentRGBCandidateVerifier,
    MEASUREMENT_MODE_SUCCESS_SEMANTICS,
    POSE_VIEW_MIXTURE_SEMANTICS,
    SUPPORT_VIEW_MIXTURE_CONTRACT,
    measurement_mode_residual_and_success,
    normalized_spatial_log_probabilities_with_dustbin,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import TensorImageLRUCache
from feature_extract.vfm.measurement_v1.rgb_data_contract import (
    require_inference_compatible_contracts,
)


MEASUREMENT_VALIDITY_SEMANTICS = MEASUREMENT_MODE_SUCCESS_SEMANTICS
MEASUREMENT_DUSTBIN_SEMANTICS = (
    "one_minus_probability_predicted_spatial_mode_gt_pose_projection_residual_le_2px"
)
NORMALIZED_SPATIAL_DUSTBIN_SEMANTICS = (
    "normalized_k_plus_dustbin_gt_pose_projected_offset_outside_local_support"
)
LEGACY_IDENTITY_DUSTBIN_SEMANTICS = (
    "one_minus_actual_observation_identity_probability_DIAGNOSTIC_ONLY"
)


def factorize_pose_view_ensemble(
    model_joint_log_probabilities: torch.Tensor,
    model_view_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor an ensemble mixture without introducing model/view cross terms.

    Returns an effective per-view K+1 density and the mean view posterior such
    that their view mixture is exactly ``mean_m sum_v b_mv p_mv``.
    """

    joint = model_joint_log_probabilities.float()
    view = model_view_probabilities.to(device=joint.device, dtype=torch.float32)
    if joint.ndim != 3 or view.shape != joint.shape[:2]:
        raise ValueError(
            "ensemble joint probabilities must be (M,N,K+1) and views (M,N)"
        )
    if int(joint.shape[0]) <= 0 or bool(torch.any(~torch.isfinite(joint))):
        raise ValueError("ensemble K+1 log probabilities are invalid")
    if bool(torch.any(~torch.isfinite(view))) or bool(torch.any(view <= 0.0)):
        raise ValueError("ensemble pose-view probabilities must be positive")
    if not bool(
        torch.allclose(
            torch.logsumexp(joint, dim=2),
            torch.zeros_like(view),
            atol=2e-5,
            rtol=2e-5,
        )
    ):
        raise ValueError("ensemble member K+1 probability mass must equal one")
    log_view = torch.log(view)
    effective_joint = torch.logsumexp(
        log_view[..., None] + joint, dim=0
    ) - torch.logsumexp(log_view, dim=0)[..., None]
    return effective_joint, torch.mean(view, dim=0)


def _load_model(
    path: Path,
    device: torch.device,
    *,
    allow_legacy_identity_dustbin: bool = False,
) -> IndependentRGBCandidateVerifier:
    payload = torch.load(Path(path), map_location="cpu")
    checkpoint_format = str(payload.get("format", ""))
    if checkpoint_format not in {
        "independent_rgb_candidate_verifier_v2",
        "independent_rgb_candidate_verifier_v3",
        "independent_rgb_candidate_verifier_v4",
        "independent_rgb_candidate_verifier_v5",
    }:
        raise ValueError(
            "production RGB spatial export requires verifier v3, v4, or v5 with "
            "target-only GT-pose supervision"
        )
    if (
        checkpoint_format == "independent_rgb_candidate_verifier_v2"
        and not bool(allow_legacy_identity_dustbin)
    ):
        raise ValueError(
            "verifier v2 uses identity probability as dustbin; it is diagnostic-only"
        )
    config = dict(payload["config"])
    if config.get("support_view_mixture") != SUPPORT_VIEW_MIXTURE_CONTRACT:
        raise ValueError(
            "RGB checkpoint support-view mixture contract is missing or incompatible"
        )
    training = dict(payload.get("training", {}))
    spatial_target_sigma_px = float(
        training.get("spatial_target_sigma_px", float("nan"))
    )
    model = IndependentRGBCandidateVerifier(
        search_radius_px=float(config["search_radius_px"]),
        context_radius_px=float(config["context_radius_px"]),
        step_px=float(config["step_px"]),
        feature_dim=int(config["feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        input_mode=str(config["input_mode"]),
        encoder_arch=str(config["encoder_arch"]),
        template_scale_factors=tuple(config["template_scale_factors"]),
        max_views=int(config["max_views"]),
        measurement_validity_semantics=str(
            config.get(
                "measurement_validity_semantics",
                MEASUREMENT_MODE_SUCCESS_SEMANTICS,
            )
        ),
        pose_view_mixture_enabled=bool(
            config.get("pose_view_mixture_enabled", False)
        ),
    )
    if checkpoint_format in {
        "independent_rgb_candidate_verifier_v4",
        "independent_rgb_candidate_verifier_v5",
    }:
        if (
            config.get("measurement_validity_semantics")
            != GT_POSE_SPATIAL_DENSITY_SEMANTICS
        ):
            raise ValueError(
                "RGB verifier normalized spatial-density contract is missing "
                "or incompatible"
            )
        if checkpoint_format == "independent_rgb_candidate_verifier_v5":
            if not bool(config.get("pose_view_mixture_enabled")) or config.get(
                "pose_view_mixture_semantics"
            ) != POSE_VIEW_MIXTURE_SEMANTICS:
                raise ValueError(
                    "RGB verifier v5 lacks its pose-view mixture contract"
                )
            model.load_state_dict(payload["model"], strict=True)
        else:
            incompatible = model.load_state_dict(
                payload["model"], strict=False
            )
            allowed_missing = {
                "view_pose_mixture_head.weight",
                "view_pose_mixture_head.bias",
            }
            if (
                set(incompatible.missing_keys) - allowed_missing
                or incompatible.unexpected_keys
            ):
                raise ValueError("RGB verifier v4 state is structurally incompatible")
        dustbin_semantics = NORMALIZED_SPATIAL_DUSTBIN_SEMANTICS
        success_threshold_px = float("nan")
        if not math.isfinite(spatial_target_sigma_px) or spatial_target_sigma_px <= 0.0:
            raise ValueError(
                "RGB verifier v4 lacks its positive spatial target sigma contract"
            )
        data_contract = dict(payload.get("data_contract", {}))
        if not data_contract:
            raise ValueError("production RGB verifier v4 lacks its RGB data contract")
    elif checkpoint_format == "independent_rgb_candidate_verifier_v3":
        if config.get("measurement_validity_semantics") != MEASUREMENT_VALIDITY_SEMANTICS:
            raise ValueError(
                "RGB verifier measurement-validity contract is missing or incompatible"
            )
        incompatible = model.load_state_dict(payload["model"], strict=False)
        allowed_missing = {
            "view_pose_mixture_head.weight",
            "view_pose_mixture_head.bias",
        }
        if (
            set(incompatible.missing_keys) - allowed_missing
            or incompatible.unexpected_keys
        ):
            raise ValueError("RGB verifier v3 state is structurally incompatible")
        dustbin_semantics = MEASUREMENT_DUSTBIN_SEMANTICS
        success_threshold_px = float(
            dict(payload.get("training", {})).get(
                "measurement_success_threshold_px", 2.0
            )
        )
        if not math.isclose(success_threshold_px, 2.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("production RGB verifier must use the 2px validity contract")
        data_contract = dict(payload.get("data_contract", {}))
        if not data_contract:
            raise ValueError("production RGB verifier v3 lacks its RGB data contract")
    else:
        incompatible = model.load_state_dict(payload["model"], strict=False)
        allowed_missing = {
            "view_measurement_validity_head.weight",
            "view_measurement_validity_head.bias",
            "view_pose_mixture_head.weight",
            "view_pose_mixture_head.bias",
        }
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise ValueError("legacy RGB verifier state is structurally incompatible")
        dustbin_semantics = LEGACY_IDENTITY_DUSTBIN_SEMANTICS
        success_threshold_px = 2.0
        data_contract = {}
    model._checkpoint_format = checkpoint_format
    model._dustbin_probability_semantics = dustbin_semantics
    model._measurement_success_threshold_px = success_threshold_px
    model._spatial_target_sigma_px = spatial_target_sigma_px
    model._data_contract = data_contract
    return model.to(device).eval()


def _architecture_signature(model: IndependentRGBCandidateVerifier) -> dict[str, object]:
    config = model.config()
    signature = {
        key: config[key]
        for key in (
            "search_radius_px",
            "context_radius_px",
            "step_px",
            "feature_dim",
            "hidden_dim",
            "input_mode",
            "encoder_arch",
            "template_scale_factors",
            "max_views",
            "pose_view_mixture_enabled",
            "pose_view_mixture_semantics",
        )
    }
    signature.update({
        "checkpoint_format": str(model._checkpoint_format),
        "dustbin_probability_semantics": str(
            model._dustbin_probability_semantics
        ),
        "measurement_success_threshold_px": (
            float(model._measurement_success_threshold_px)
            if math.isfinite(float(model._measurement_success_threshold_px))
            else None
        ),
        "spatial_target_sigma_px": (
            float(model._spatial_target_sigma_px)
            if math.isfinite(float(model._spatial_target_sigma_px))
            else None
        ),
        "measurement_validity_semantics": str(
            model.measurement_validity_semantics
        ),
    })
    return signature


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
    return np.asarray(
        [_float_text(row.get(key), default=float(default)) for row in rows],
        dtype=np.float32,
    )


def export_candidate_rgb_spatial_likelihood(
    *,
    checkpoints: Sequence[Path],
    candidate_evidence: Path,
    availability_evidence: Path,
    train_rows_csv: Path,
    validation_rows_csv: Path,
    test_rows_csv: Path,
    split_name: str,
    image_root: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    batch_size: int = 32,
    device: str = "cuda",
    image_cache_max_gb: float = 9.0,
    image_cache_dtype: str = "float16",
    use_amp: bool = True,
    allow_legacy_identity_dustbin: bool = False,
) -> dict[str, Any]:
    split = str(split_name)
    if split not in {"validation", "test"}:
        raise ValueError("spatial likelihood export supports validation or test")
    if not checkpoints:
        raise ValueError("spatial likelihood export requires at least one checkpoint")
    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    models = [
        _load_model(
            Path(path),
            torch_device,
            allow_legacy_identity_dustbin=bool(allow_legacy_identity_dustbin),
        )
        for path in checkpoints
    ]
    signature = _architecture_signature(models[0])
    if any(_architecture_signature(model) != signature for model in models[1:]):
        raise ValueError("independent RGB ensemble checkpoints have different architectures")
    data = CandidateRGBTrainingData(
        candidate_evidence=Path(candidate_evidence),
        availability_evidence=Path(availability_evidence),
        train_rows_csv=Path(train_rows_csv),
        validation_rows_csv=Path(validation_rows_csv),
        test_rows_csv=Path(test_rows_csv),
        max_views=int(models[0].max_views),
    )
    runtime_data_contract = data.runtime_contract(
        image_root=Path(image_root),
        image_width=int(image_width),
        image_height=int(image_height),
    )
    for checkpoint_path, model in zip(checkpoints, models):
        if str(model._checkpoint_format) in {
            "independent_rgb_candidate_verifier_v3",
            "independent_rgb_candidate_verifier_v4",
            "independent_rgb_candidate_verifier_v5",
        }:
            require_inference_compatible_contracts(
                model._data_contract,
                runtime_data_contract,
                context=f"RGB spatial export checkpoint {checkpoint_path}",
            )
    cache_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(str(image_cache_dtype).lower())
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
    if len(indices) == 0:
        raise ValueError(f"{split} has no RGB-supported candidate groups")

    rows_out: list[dict[str, str]] = []
    log_probability_blocks: list[np.ndarray] = []
    dustbin_blocks: list[np.ndarray] = []
    entropy_blocks: list[np.ndarray] = []
    covariance_trace_blocks: list[np.ndarray] = []
    target_offset_blocks: list[np.ndarray] = []
    actual_observation_offset_blocks: list[np.ndarray] = []
    actual_spatial_valid_blocks: list[np.ndarray] = []
    measurement_success_blocks: list[np.ndarray] = []
    measurement_mode_residual_blocks: list[np.ndarray] = []
    measurement_supervision_weight_blocks: list[np.ndarray] = []
    supervised_dustbin_target_blocks: list[np.ndarray] = []
    supervised_dustbin_probability_blocks: list[np.ndarray] = []
    support_view_probability_blocks: list[np.ndarray] = []
    offsets_xy: np.ndarray | None = None
    amp_enabled = bool(use_amp and torch_device.type == "cuda")

    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)].tolist()
            batch = _prepare_batch(
                data,
                batch_indices,
                image_root=Path(image_root),
                image_width=int(image_width),
                image_height=int(image_height),
                image_cache=image_cache,
                image_cache_device=torch_device,
                crop_radius_px=models[0].crop_radius_px,
                step_px=models[0].step_px,
                identity_threshold_px=2.0,
                identity_negative_threshold_px=5.0,
                spatial_radius_px=models[0].search_radius_px,
            )
            predictions = [
                _forward_batch(model, batch, device=torch_device, use_amp=amp_enabled)
                for model in models
            ]
            normalized_density = (
                str(models[0]._dustbin_probability_semantics)
                == NORMALIZED_SPATIAL_DUSTBIN_SEMANTICS
            )
            learned_pose_view_mixture = bool(
                models[0].pose_view_mixture_enabled
            )
            if normalized_density:
                model_joint_log_probabilities = torch.stack(
                    [
                        normalized_spatial_log_probabilities_with_dustbin(
                            prediction.view_spatial_logits.float(),
                            prediction.view_measurement_validity_logits.float(),
                        )
                        for prediction in predictions
                    ],
                    dim=0,
                )
                if learned_pose_view_mixture:
                    model_view_probabilities = torch.stack(
                        [
                            prediction.view_pose_mixture_probabilities.float()
                            for prediction in predictions
                        ],
                        dim=0,
                    )
                    # This factorization exactly preserves the ensemble
                    # candidate density: mean_m sum_v b_mv p_mv.
                    (
                        ensemble_joint_log_probability,
                        ensemble_pose_view_probability,
                    ) = factorize_pose_view_ensemble(
                        model_joint_log_probabilities,
                        model_view_probabilities,
                    )
                else:
                    ensemble_joint_log_probability = torch.logsumexp(
                        model_joint_log_probabilities, dim=0
                    ) - math.log(float(len(models)))
                    ensemble_pose_view_probability = batch[
                        "pair_view_probabilities"
                    ].to(
                        device=ensemble_joint_log_probability.device,
                        dtype=torch.float32,
                    )
                local_log_mass = torch.logsumexp(
                    ensemble_joint_log_probability[:, :-1], dim=1
                )
                ensemble_log_probability = (
                    ensemble_joint_log_probability[:, :-1]
                    - local_log_mass[:, None]
                )
                ensemble_validity_probability = torch.exp(local_log_mass)
            else:
                model_log_probabilities = torch.stack(
                    [
                        F.log_softmax(
                            prediction.view_spatial_logits.float(), dim=1
                        )
                        for prediction in predictions
                    ],
                    dim=0,
                )
                ensemble_log_probability = torch.logsumexp(
                    model_log_probabilities, dim=0
                ) - math.log(float(len(models)))
            if normalized_density:
                pass
            elif (
                str(models[0]._dustbin_probability_semantics)
                == MEASUREMENT_DUSTBIN_SEMANTICS
            ):
                ensemble_validity_probability = torch.mean(
                    torch.stack(
                        [
                            prediction.view_measurement_validity_probabilities.float()
                            for prediction in predictions
                        ],
                        dim=0,
                    ),
                    dim=0,
                )
            else:
                ensemble_validity_probability = torch.mean(
                    torch.stack(
                        [
                            torch.sigmoid(prediction.view_identity_logits.float())
                            for prediction in predictions
                        ],
                        dim=0,
                    ),
                    dim=0,
                )
            local_offsets = predictions[0].view_spatial_offsets_xy.detach().cpu().numpy()
            if any(
                not torch.equal(
                    prediction.view_spatial_offsets_xy,
                    predictions[0].view_spatial_offsets_xy,
                )
                for prediction in predictions[1:]
            ):
                raise RuntimeError("RGB ensemble spatial supports differ")
            if offsets_xy is None:
                offsets_xy = np.asarray(local_offsets, dtype=np.float32)
            elif not np.array_equal(offsets_xy, local_offsets):
                raise RuntimeError("RGB spatial support changed between batches")

            probability = torch.exp(ensemble_log_probability)
            spatial_offsets_device = predictions[0].view_spatial_offsets_xy.to(
                device=probability.device, dtype=torch.float32
            )
            mean_offset = probability @ spatial_offsets_device
            second_moment = torch.sum(
                probability
                * torch.sum(
                    spatial_offsets_device.square(), dim=1
                )[None],
                dim=1,
            )
            covariance_trace = second_moment - torch.sum(mean_offset.square(), dim=1)
            entropy = -torch.sum(
                probability * ensemble_log_probability, dim=1
            )
            if normalized_density:
                calibration_target_offset = batch[
                    "target_gt_projected_offset_xy"
                ].float()
                supervised = batch["target_gt_projection_evaluable"].bool()
                supervision_weight = batch[
                    "measurement_validity_supervision_weight"
                ].float()
                physical = batch[
                    "target_gt_projection_physical_valid"
                ].bool()
                finite_target = torch.all(
                    torch.isfinite(calibration_target_offset), dim=1
                )
                offset_minimum = torch.min(
                    spatial_offsets_device, dim=0
                ).values.cpu()
                offset_maximum = torch.max(
                    spatial_offsets_device, dim=0
                ).values.cpu()
                measurement_success = (
                    supervised
                    & physical
                    & finite_target
                    & torch.all(
                        calibration_target_offset >= offset_minimum[None],
                        dim=1,
                    )
                    & torch.all(
                        calibration_target_offset <= offset_maximum[None],
                        dim=1,
                    )
                )
                target_is_dustbin = ~measurement_success
                mode_xy_device = spatial_offsets_device[
                    torch.argmax(ensemble_log_probability, dim=1)
                ]
                mode_residual = torch.linalg.norm(
                    mode_xy_device
                    - calibration_target_offset.to(mode_xy_device.device),
                    dim=1,
                )
                mode_residual = torch.where(
                    physical.to(mode_residual.device),
                    mode_residual,
                    torch.full_like(mode_residual, torch.inf),
                )
            elif (
                str(models[0]._dustbin_probability_semantics)
                == MEASUREMENT_DUSTBIN_SEMANTICS
            ):
                mode_residual, measurement_success = (
                    measurement_mode_residual_and_success(
                        ensemble_log_probability,
                        predictions[0].view_spatial_offsets_xy,
                        batch["target_gt_projected_offset_xy"],
                        batch["target_gt_projection_physical_valid"],
                        success_threshold_px=float(
                            models[0]._measurement_success_threshold_px
                        ),
                    )
                )
                supervised = batch["target_gt_projection_evaluable"].bool()
                supervision_weight = batch[
                    "measurement_validity_supervision_weight"
                ].float()
                target_is_dustbin = ~measurement_success
                calibration_target_offset = batch[
                    "target_gt_projected_offset_xy"
                ].float()
            else:
                pair_groups = batch["pair_group_indices"].to(dtype=torch.long)
                pair_candidates = batch["pair_candidate_indices"].to(
                    dtype=torch.long
                )
                labels = batch["candidate_labels"][
                    pair_groups, pair_candidates
                ].bool()
                supervised = batch["identity_supervision_valid"][
                    pair_groups, pair_candidates
                ].bool()
                supervision_weight = supervised.float()
                target_is_dustbin = ~labels
                calibration_target_offset = batch["spatial_target_xy"].float()
                mode_residual = torch.linalg.norm(
                    spatial_offsets_device[
                        torch.argmax(ensemble_log_probability, dim=1)
                    ]
                    - calibration_target_offset.to(spatial_offsets_device.device),
                    dim=1,
                )
                measurement_success = ~target_is_dustbin

            flat_rows = batch["flat_rows"]
            if not isinstance(flat_rows, list) or len(flat_rows) != len(ensemble_log_probability):
                raise RuntimeError("RGB spatial rows and predictions are misaligned")
            rows_out.extend(flat_rows)
            log_probability_blocks.append(
                ensemble_log_probability.cpu().numpy().astype(np.float16)
            )
            dustbin_blocks.append(
                (1.0 - ensemble_validity_probability).cpu().numpy().astype(np.float32)
            )
            entropy_blocks.append(entropy.cpu().numpy().astype(np.float32))
            covariance_trace_blocks.append(
                covariance_trace.clamp_min(0.0).cpu().numpy().astype(np.float32)
            )
            target_offset_blocks.append(
                calibration_target_offset.numpy().astype(np.float32)
            )
            actual_observation_offset_blocks.append(
                batch["spatial_target_xy"].numpy().astype(np.float32)
            )
            actual_spatial_valid_blocks.append(
                batch["spatial_valid"].numpy().astype(bool)
            )
            measurement_success_blocks.append(
                measurement_success.cpu().numpy().astype(bool)
            )
            measurement_mode_residual_blocks.append(
                mode_residual.cpu().numpy().astype(np.float32)
            )
            measurement_supervision_weight_blocks.append(
                supervision_weight.numpy().astype(np.float32)
            )
            supervised_dustbin_target_blocks.append(
                target_is_dustbin.cpu()[supervised].numpy()
            )
            supervised_dustbin_probability_blocks.append(
                (1.0 - ensemble_validity_probability.cpu())[supervised].numpy()
            )
            support_view_probability_blocks.append(
                ensemble_pose_view_probability.cpu().numpy().astype(np.float32)
            )

    if offsets_xy is None or not rows_out:
        raise RuntimeError("RGB spatial export produced no rows")
    local_log_probabilities = np.concatenate(log_probability_blocks, axis=0)
    dustbin_probabilities = np.concatenate(dustbin_blocks, axis=0)
    likelihood_entropy = np.concatenate(entropy_blocks, axis=0)
    covariance_trace = np.concatenate(covariance_trace_blocks, axis=0)
    target_offset_xy = np.concatenate(target_offset_blocks, axis=0)
    actual_observation_offset_xy = np.concatenate(
        actual_observation_offset_blocks, axis=0
    )
    actual_spatial_valid = np.concatenate(actual_spatial_valid_blocks, axis=0)
    measurement_success = np.concatenate(measurement_success_blocks).astype(bool)
    measurement_mode_residual = np.concatenate(
        measurement_mode_residual_blocks
    ).astype(np.float32)
    measurement_supervision_weight = np.concatenate(
        measurement_supervision_weight_blocks
    ).astype(np.float32)
    if len(local_log_probabilities) != len(rows_out):
        raise RuntimeError("RGB spatial output arrays have different row counts")
    mode_xy = offsets_xy[np.argmax(local_log_probabilities, axis=1)]
    target_inside_spatial_grid = (
        np.all(np.isfinite(target_offset_xy), axis=1)
        & (target_offset_xy[:, 0] >= float(np.min(offsets_xy[:, 0])))
        & (target_offset_xy[:, 0] <= float(np.max(offsets_xy[:, 0])))
        & (target_offset_xy[:, 1] >= float(np.min(offsets_xy[:, 1])))
        & (target_offset_xy[:, 1] <= float(np.max(offsets_xy[:, 1])))
    )
    if normalized_density:
        target_inside_spatial_grid &= measurement_success
    spatial_calibration_supervision_weight = (
        measurement_supervision_weight * target_inside_spatial_grid.astype(np.float32)
    )
    actual_observation_epe = np.linalg.norm(
        mode_xy - actual_observation_offset_xy, axis=1
    )
    gt_projection_epe = np.linalg.norm(mode_xy - target_offset_xy, axis=1)
    supervised_target = np.concatenate(supervised_dustbin_target_blocks).astype(bool)
    supervised_probability = np.concatenate(
        supervised_dustbin_probability_blocks
    ).astype(np.float64)
    support_view_probabilities = np.concatenate(
        support_view_probability_blocks
    ).astype(np.float32)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized_density = (
        str(models[0]._dustbin_probability_semantics)
        == NORMALIZED_SPATIAL_DUSTBIN_SEMANTICS
    )
    production_validity = str(models[0]._dustbin_probability_semantics) in {
        MEASUREMENT_DUSTBIN_SEMANTICS,
        NORMALIZED_SPATIAL_DUSTBIN_SEMANTICS,
    }
    learned_pose_view_mixture = bool(models[0].pose_view_mixture_enabled)
    artifact_format = (
        "candidate_spatial_likelihood_v6"
        if normalized_density and learned_pose_view_mixture
        else "candidate_spatial_likelihood_v5"
        if normalized_density
        else (
            "candidate_spatial_likelihood_v4"
            if production_validity
            else "candidate_spatial_likelihood_v3"
        )
    )
    artifact = output / f"{artifact_format}.npz"
    checkpoint_manifest = [
        {"path": str(path), "sha256": file_sha256_short(Path(path))}
        for path in checkpoints
    ]
    metadata = {
        "format": artifact_format,
        "format_version": (
            6
            if normalized_density and learned_pose_view_mixture
            else 5
            if normalized_density
            else 4
            if production_validity
            else 3
        ),
        "rows_csv": str(data.rows_paths[split]),
        "rows_csv_sha256": file_sha256_short(data.rows_paths[split]),
        "measurement_checkpoint_sha256": "+".join(
            item["sha256"] for item in checkpoint_manifest
        ),
        "measurement_checkpoints": checkpoint_manifest,
        "candidate_evidence_sha256": file_sha256_short(Path(candidate_evidence)),
        "data_contract": runtime_data_contract,
        "checkpoint_data_contract_compatibility": (
            "rgb_inference_coordinate_and_real_image_source_v1;"
            "candidate_evidence_availability_and_rows_are_runtime_inputs"
        ),
        "coordinate_space_id": runtime_data_contract["coordinate_space"][
            "coordinate_space_id"
        ],
        "image_source_manifest_sha256": runtime_data_contract["image_source"][
            "sampled_content_manifest_sha256"
        ],
        "query_source": "real_pair",
        "support_patch_warp": "none",
        "image_width": int(image_width),
        "image_height": int(image_height),
        "search_radius_px": float(models[0].search_radius_px),
        "context_radius_px": float(models[0].context_radius_px),
        "step_px": float(models[0].step_px),
        "spatial_probability_semantics": (
            "ensemble_mixture_normalized_k_plus_dustbin"
            if normalized_density
            else "ensemble_mixture_conditional_local_offset"
        ),
        "spatial_target_semantics": (
            "gt_pose_projected_offset_every_evaluable_candidate_view_target_only"
            if normalized_density
            else "gt_pose_projected_offset_target_only"
            if production_validity
            else "actual_observation_offset_DIAGNOSTIC_ONLY"
        ),
        "dustbin_probability_semantics": str(
            models[0]._dustbin_probability_semantics
        ),
        "target_dustbin_semantics": (
            "gt_pose_projection_outside_local_support_or_not_physical"
            if normalized_density
            else "predicted_ensemble_spatial_mode_gt_pose_projection_residual_gt_2px"
            if production_validity
            else "legacy_actual_observation_identity_negative_DIAGNOSTIC_ONLY"
        ),
        "measurement_success_threshold_px": (
            None
            if normalized_density
            else float(models[0]._measurement_success_threshold_px)
        ),
        "spatial_target_sigma_px": (
            float(models[0]._spatial_target_sigma_px)
            if normalized_density
            else None
        ),
        "joint_probability_mass_normalized": bool(normalized_density),
        "posthoc_calibration_required": bool(
            production_validity and not normalized_density
        ),
        "training_probability_objective": (
            "normalized_k_plus_dustbin_natural_distribution_mle"
            if normalized_density
            else "legacy_separate_conditional_map_and_validity_head"
        ),
        "measurement_validity_changes_identity_prior": False,
        "measurement_validity_used_as_independent_identity_likelihood": False,
        "support_views_unmarginalized": True,
        "support_view_probability_semantics": (
            POSE_VIEW_MIXTURE_SEMANTICS
            if learned_pose_view_mixture
            else "frozen_candidate_maplet_view_posterior"
        ),
        "support_view_probability_mass": (
            "one_over_all_measured_views_per_candidate"
            if learned_pose_view_mixture
            else "may_include_neutral_missing_mass"
        ),
        "pose_view_ensemble_factorization": (
            "exact_mean_model_sum_view_weight_times_joint_k_plus_dustbin"
            if learned_pose_view_mixture
            else "not_applicable"
        ),
        "missing_support_view_likelihood_ratio": 1.0,
        "pose_or_ground_truth_used_for_inference": False,
        "ground_truth_arrays_target_only": [
            "target_offset_xy",
            "target_is_dustbin",
            "target_gt_projected_xy",
            "target_gt_projected_residual_px",
            "target_geometry_correct_1px",
            "target_geometry_correct_2px",
            "target_geometry_correct_5px",
            "target_measurement_success_2px",
            "target_measurement_mode_gt_residual_px",
            "measurement_validity_supervision_weight",
            "spatial_calibration_supervision_weight",
        ],
        "render": False,
    }
    target_gt_projected_xy = np.stack(
        [
            _floating(rows_out, "target_gt_projected_x"),
            _floating(rows_out, "target_gt_projected_y"),
        ],
        axis=1,
    )
    np.savez_compressed(
        artifact,
        source_row_indices=_integer(rows_out, "__source_row_index__"),
        query_ids=_text(rows_out, "query_id"),
        source_query_rows=_integer(rows_out, "source_query_row"),
        candidate_identity_keys=_text(rows_out, "candidate_identity_key"),
        candidate_measurement_cache_keys=np.full(
            (len(rows_out),), "", dtype=np.str_
        ),
        candidate_measurement_ranks=_integer(rows_out, "candidate_measurement_rank"),
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
        target_offset_xy=target_offset_xy,
        actual_observation_target_offset_xy=actual_observation_offset_xy,
        target_is_dustbin=~measurement_success,
        target_measurement_success_2px=measurement_success,
        target_measurement_mode_gt_residual_px=measurement_mode_residual,
        measurement_validity_supervision_weight=(
            measurement_supervision_weight
        ),
        spatial_calibration_supervision_weight=(
            spatial_calibration_supervision_weight
        ),
        dustbin_supervision_weight=measurement_supervision_weight,
        target_gt_projected_xy=target_gt_projected_xy,
        target_gt_projected_residual_px=_floating(
            rows_out, "target_gt_projected_residual_px"
        ),
        target_geometry_correct_1px=np.asarray(
            [_bool_text(row.get("target_geometry_correct_1px")) for row in rows_out],
            dtype=bool,
        ),
        target_geometry_correct_2px=np.asarray(
            [_bool_text(row.get("target_geometry_correct_2px")) for row in rows_out],
            dtype=bool,
        ),
        target_geometry_correct_5px=np.asarray(
            [_bool_text(row.get("target_geometry_correct_5px")) for row in rows_out],
            dtype=bool,
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    summary = {
        "stage": "independent_rgb_candidate_spatial_export",
        "split": split,
        "protocol": {
            "query_source": "real_pair",
            "render": False,
            "pose_or_ground_truth_used_for_inference": False,
            "per_view_spatial_likelihood_retained": True,
            "support_view_posterior_retained": True,
            "missing_view_is_neutral": True,
            "measurement_validity_source": (
                "learned_normalized_gt_pose_spatial_density"
                if normalized_density
                else "learned_true_pose_residual_head"
                if production_validity
                else "legacy_identity_head_DIAGNOSTIC_ONLY"
            ),
            "identity_probability_used_as_dustbin": not production_validity,
        },
        "metrics": {
            "actual_observation_spatial": {
                "sample_count": int(np.count_nonzero(actual_spatial_valid)),
                "mode_epe_median_px": float(
                    np.median(actual_observation_epe[actual_spatial_valid])
                ),
                "mode_epe_p90_px": float(
                    np.quantile(actual_observation_epe[actual_spatial_valid], 0.9)
                ),
            },
            (
                "gt_pose_spatial_non_dustbin_TARGET_ONLY"
                if normalized_density
                else "measurement_geometric_validity_TARGET_ONLY"
            ): confidence_metrics(
                ~supervised_target, 1.0 - supervised_probability
            ),
            "measurement_dustbin_TARGET_ONLY": confidence_metrics(
                supervised_target, supervised_probability
            ),
            "gt_projection_mode_epe_TARGET_ONLY": {
                "sample_count": int(
                    np.count_nonzero(spatial_calibration_supervision_weight > 0.0)
                ),
                "median_px": float(
                    np.median(
                        gt_projection_epe[
                            spatial_calibration_supervision_weight > 0.0
                        ]
                    )
                ),
                "p90_px": float(
                    np.quantile(
                        gt_projection_epe[
                            spatial_calibration_supervision_weight > 0.0
                        ],
                        0.9,
                    )
                ),
            },
        },
        "image_cache": image_cache.summary(),
        "inputs": {
            "candidate_evidence": str(candidate_evidence),
            "candidate_evidence_sha256": file_sha256_short(Path(candidate_evidence)),
            "rows_csv": str(data.rows_paths[split]),
            "rows_csv_sha256": file_sha256_short(data.rows_paths[split]),
            "checkpoints": checkpoint_manifest,
        },
        "outputs": {
            "candidate_spatial_likelihood": str(artifact),
            "candidate_spatial_likelihood_sha256": file_sha256_short(artifact),
        },
    }
    summary_path = output / "summary.json"
    summary["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
