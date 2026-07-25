"""Score frozen held-out poses with the calibrated high-resolution RGB branch.

This scorer deliberately sits after the target-free RGB network boundary.  It
uses a frozen P1 query/support layout and fixed global top-L candidates to emit
one visual prediction, then evaluates that prediction at externally supplied
frozen hypothesis projections.  It neither reads pose targets nor permits the
raw likelihood to enter PnP.  A train-only calibrated prior temperature is
mandatory and is validated before any held-out image is read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.calibrate_candidate_highres_rgb_prior_temperature import (
    CALIBRATION_FORMAT,
)
from feature_extract.tools.vfm.score_candidate_pose_rgb_spatial_likelihood import (
    _assert_layout_matches_mixed_points,
    _canonical_hash,
    _frozen_query_evidence_sha,
    _input_manifest,
    _project_candidate_offsets,
    _query_geometry,
    _query_rows,
    _slice_layout,
    _validate_baseline_reference_hypothesis_equivalence,
    _validate_query_layout_against_points,
)
from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _load_bank_xyz,
    _load_exact_hypotheses,
)
from feature_extract.tools.vfm.train_candidate_highres_rgb_multiscale_likelihood import (
    CHECKPOINT_FORMAT,
    FIXED_FINAL_EPOCH_SELECTION_POLICY,
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    active_source_names,
    validate_rgb_coordinate_bridge,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
    CandidateHighresRGBMultiscaleLikelihood,
    CandidateHighresRGBMultiscalePrediction,
    score_candidate_highres_rgb_multiscale_batch,
    temper_candidate_prior_probabilities,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    CANDIDATE_POSE_LLR_SCORE_FORMAT,
    validate_target_free_pose_llr_score_metadata,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    permute_support_patch_appearance,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_source_headers,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    load_mixed_verification_points,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


SCORE_VERSION = "candidate_highres_rgb_multiscale_prior_calibrated_scores_v1"
_BASELINE_REFERENCE_FIELDS = ("poses_w2c", "chosen_for_optional_pose")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis-artifact", required=True)
    parser.add_argument("--baseline-score-artifact", required=True)
    parser.add_argument("--baseline-reference-hypothesis-artifact", default="")
    parser.add_argument("--detector-query-cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--fixed-candidate-prior-overlay", required=True)
    parser.add_argument("--mixed-verification-points-artifact", required=True)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prior-temperature-calibration", required=True)
    parser.add_argument("--hypothesis-batch-size", type=int, default=8)
    parser.add_argument("--edge-chunk-size", type=int, default=0)
    parser.add_argument("--rgb-cache-gb", type=float, default=6.0)
    parser.add_argument("--rgb-cache-dtype", choices=("uint8", "float16"), default="uint8")
    parser.add_argument("--rgb-cache-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--evidence-variant",
        choices=("visual", "support_descriptor_permutation_control"),
        default="visual",
    )
    parser.add_argument(
        "--hypothesis-limit",
        type=int,
        default=0,
        help="development-only frozen hypothesis prefix; zero scores all rows",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("high-resolution RGB calibration artifact is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("high-resolution RGB calibration artifact is invalid")
    return payload


def _validate_calibration_for_scoring(
    *,
    calibration: Mapping[str, object],
    calibration_path: Path,
    checkpoint_path: Path,
    layout_path: Path,
    layout: CandidatePoseRGBSpatialLayout,
    source_image_manifest_sha256: str,
) -> tuple[float, str]:
    """Accept only a passed train-only calibration for this frozen scorer."""

    safety = calibration.get("safety")
    lineage = calibration.get("lineage")
    inner = calibration.get("inner_validation")
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    if (
        calibration.get("format") != CALIBRATION_FORMAT
        or calibration.get("runtime_safe") is not True
        or calibration.get("contains_target_fields") is not False
        or calibration.get("calibration_uses_train_only_targets") is not True
        or not isinstance(safety, Mapping)
        or safety.get("heldout_evaluation_allowed") is not True
        or safety.get("pnp_integration_allowed") is not False
        or safety.get("promotion_allowed") is not False
        or calibration.get("appearance_control_semantics")
        != "fixed_runtime_geometry_and_validity_with_rgb_patch_derangement_only_v2"
        or calibration.get("checkpoint_selection_policy") != FIXED_FINAL_EPOCH_SELECTION_POLICY
        or calibration.get("checkpoint_inner_validation_used_for_model_selection") is not False
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or not isinstance(lineage, Mapping)
        or str(calibration.get("checkpoint_sha256", ""))
        != file_sha256_short(Path(checkpoint_path))
        or str(lineage.get("layout_sha256", "")) != file_sha256_short(Path(layout_path))
        or str(lineage.get("projection_space_id", ""))
        != str(layout.metadata.get("projection_space_id", ""))
        or str(lineage.get("descriptor_space_id", ""))
        != str(layout.metadata.get("descriptor_space_id", ""))
        or str(lineage.get("source_image_manifest_sha256", ""))
        != str(source_image_manifest_sha256)
    ):
        raise ValueError("high-resolution RGB prior-temperature calibration is not eligible")
    temperature = float(calibration.get("candidate_prior_temperature", float("nan")))
    source = str(calibration.get("source", "")).strip().lower()
    if (
        not math.isfinite(temperature)
        or temperature <= 0.0
        or source not in {"fine", "broad"}
        or not Path(calibration_path).is_file()
    ):
        raise ValueError("high-resolution RGB prior-temperature calibration is malformed")
    return temperature, source


def _load_checkpoint_model(
    *, path: Path, image_sizes: np.ndarray, device: torch.device
) -> tuple[CandidateHighresRGBMultiscaleLikelihood, dict[str, object]]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        payload = torch.load(Path(path), map_location="cpu")
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    state_dict = payload.get("state_dict") if isinstance(payload, Mapping) else None
    required = {
        "format": CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "pnp_integration_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": 20,
        "fixed_support_view_count": 2,
        "explicit_null": True,
        "projection_after_network_only": True,
        "appearance_control_geometry_fixed": True,
        "checkpoint_selection_policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
        "inner_validation_used_for_model_selection": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or any(metadata.get(name) != value for name, value in required.items())
        or not isinstance(metadata.get("config"), Mapping)
        or not isinstance(metadata.get("lineage"), Mapping)
    ):
        raise ValueError("high-resolution RGB checkpoint violates the target-free contract")
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
        raise ValueError("high-resolution RGB checkpoint encoder contract is incomplete")
    config = dict(metadata["config"])
    required_config = (
        "fine_search_radius_px",
        "fine_context_radius_px",
        "broad_search_radius_px",
        "broad_context_radius_px",
        "texture_feature_dim",
        "hidden_dim",
        "edge_chunk_size",
        "rgb_temperature",
        "max_abs_edge_log_ratio",
        "max_abs_pose_log_ratio",
        "source",
    )
    if any(name not in config for name in required_config):
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
    return model.eval(), dict(metadata)


def _score_hypotheses(
    *,
    model: CandidateHighresRGBMultiscaleLikelihood,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateHighresRGBMultiscalePrediction,
    geometry: object,
    poses_w2c: np.ndarray,
    source: str,
    candidate_prior_temperature: float,
    hypothesis_batch_size: int,
    max_abs_pose_log_ratio: float,
) -> dict[str, np.ndarray]:
    """Score all frozen poses after exactly one target-free RGB forward."""

    poses = np.asarray(poses_w2c, dtype=np.float64)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or len(poses) == 0
        or int(hypothesis_batch_size) <= 0
        or not math.isfinite(float(max_abs_pose_log_ratio))
        or float(max_abs_pose_log_ratio) <= 0.0
    ):
        raise ValueError("high-resolution RGB hypothesis scoring inputs are invalid")
    active = runtime.to(model.device)
    tempered = temper_candidate_prior_probabilities(
        candidate_probabilities=active.candidate_probabilities,
        null_probabilities=active.null_probabilities,
        temperature=float(candidate_prior_temperature),
    )
    pose_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    candidate_count_parts: list[np.ndarray] = []
    view_mass_parts: list[np.ndarray] = []
    source_log_parts: list[np.ndarray] = []
    source_count_parts: list[np.ndarray] = []
    for begin in range(0, len(poses), int(hypothesis_batch_size)):
        end = min(begin + int(hypothesis_batch_size), len(poses))
        offsets, valid = _project_candidate_offsets(
            geometry=geometry, runtime=runtime, poses_w2c=poses[begin:end], device=model.device
        )
        with torch.autocast(device_type=model.device.type, enabled=model.device.type == "cuda"):
            score = score_candidate_highres_rgb_multiscale_batch(
                runtime=runtime,
                prediction=prediction,
                candidate_projection_offsets_xy=offsets,
                candidate_projection_valid=valid,
                source=source,
                candidate_prior_temperature=float(candidate_prior_temperature),
                missing_edge_log_likelihood_ratio=0.0,
                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
            )
        usable_candidate = score.edge_usable.any(dim=3) & (tempered.unsqueeze(0) > 0.0)
        effective_mass = (
            score.edge_usable.to(dtype=torch.float32)
            * tempered.unsqueeze(0).unsqueeze(3)
            * active.candidate_view_weights.unsqueeze(0)
        ).sum(dim=(2, 3))
        point = score.point_log_likelihood_ratios.detach().cpu().numpy().astype(np.float32)
        effective = usable_candidate.sum(dim=2).detach().cpu().numpy().astype(np.int16)
        pose_parts.append(score.pose_log_likelihood_ratios.detach().cpu().numpy().astype(np.float32))
        point_parts.append(point)
        candidate_count_parts.append(effective)
        view_mass_parts.append(effective_mass.detach().cpu().numpy().astype(np.float32))
        source_log_parts.append(point.mean(axis=1, keepdims=True).astype(np.float32))
        source_count_parts.append((effective > 0).sum(axis=1, keepdims=True).astype(np.int16))
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    return {
        "pose_log_likelihood_ratios": np.concatenate(pose_parts, axis=0),
        "point_log_likelihood_ratios": np.concatenate(point_parts, axis=0),
        "point_effective_candidate_counts": np.concatenate(candidate_count_parts, axis=0),
        "point_effective_view_masses": np.concatenate(view_mass_parts, axis=0),
        "source_names": np.asarray([source], dtype=np.str_),
        "source_log_likelihood_means": np.concatenate(source_log_parts, axis=0),
        "source_effective_point_counts": np.concatenate(source_count_parts, axis=0),
    }


def score_candidate_highres_rgb_multiscale_likelihood(args: argparse.Namespace) -> dict[str, object]:
    if (
        int(args.hypothesis_batch_size) <= 0
        or int(args.edge_chunk_size) < 0
        or int(args.hypothesis_limit) < 0
        or not math.isfinite(float(args.rgb_cache_gb))
        or float(args.rgb_cache_gb) <= 0.0
    ):
        raise ValueError("high-resolution RGB scorer arguments are invalid")
    output_path = Path(args.output)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite high-resolution RGB score artifact")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("high-resolution RGB scorer requested CUDA but CUDA is unavailable")
    paths = {
        "hypothesis_artifact": Path(args.hypothesis_artifact),
        "baseline_score_artifact": Path(args.baseline_score_artifact),
        "detector_query_cache": Path(args.detector_query_cache),
        "proposals": Path(args.proposals),
        "candidate_artifact": Path(args.candidate_artifact),
        "fixed_candidate_prior_overlay": Path(args.fixed_candidate_prior_overlay),
        "mixed_verification_points_artifact": Path(args.mixed_verification_points_artifact),
        "rgb_spatial_layout": Path(args.rgb_spatial_layout),
        "projected_landmark_bank": Path(args.projected_landmark_bank),
        "radio_final_context_cache": Path(args.radio_final_context_cache),
        "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
        "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
        "checkpoint": Path(args.checkpoint),
        "prior_temperature_calibration": Path(args.prior_temperature_calibration),
        "colmap_cameras_bin": Path(args.colmap_model_dir) / "cameras.bin",
        "colmap_images_bin": Path(args.colmap_model_dir) / "images.bin",
    }
    reference_path = (
        None
        if not str(args.baseline_reference_hypothesis_artifact).strip()
        else Path(args.baseline_reference_hypothesis_artifact)
    )
    if reference_path is not None:
        paths["baseline_reference_hypothesis_artifact"] = reference_path
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"high-resolution RGB scorer input is missing: {name} ({path})")
    if not Path(args.image_root).is_dir():
        raise FileNotFoundError("high-resolution RGB image root is missing")

    start = time.time()
    baseline_hypothesis_path = paths["hypothesis_artifact"] if reference_path is None else reference_path
    exact, baseline_hypothesis_metadata, baseline_metadata = _load_exact_hypotheses(
        hypothesis_path=baseline_hypothesis_path,
        baseline_path=paths["baseline_score_artifact"],
        detector_path=paths["detector_query_cache"],
        proposals_path=paths["proposals"],
        candidate_path=paths["candidate_artifact"],
        prior_path=paths["fixed_candidate_prior_overlay"],
        fixed_candidate_top_k=20,
    )
    baseline_reference_equivalence = None
    if reference_path is None:
        hypothesis_metadata = baseline_hypothesis_metadata
    else:
        baseline_reference_equivalence = _validate_baseline_reference_hypothesis_equivalence(
            current_hypothesis_path=paths["hypothesis_artifact"],
            reference_hypothesis_path=reference_path,
        )
        from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import load_inference_artifact_fields

        _unused, hypothesis_metadata = load_inference_artifact_fields(
            paths["hypothesis_artifact"], _BASELINE_REFERENCE_FIELDS
        )
    if int(args.hypothesis_limit) > 0:
        exact = {name: np.asarray(value)[: int(args.hypothesis_limit)] for name, value in exact.items()}
    query_ids = np.unique(np.asarray(exact["query_ids"]).astype(str))
    split_names = np.unique(np.asarray(exact["split_names"]).astype(str))
    if len(query_ids) != 1 or len(split_names) != 1 or str(split_names[0]) not in {"validation", "test"}:
        raise ValueError("high-resolution RGB scorer accepts one held-out query shard")
    query_id, split_name = str(query_ids[0]), str(split_names[0])

    layout = load_candidate_pose_rgb_spatial_layout(paths["rgb_spatial_layout"])
    points = load_mixed_verification_points(paths["mixed_verification_points_artifact"])
    layout_point_rows, layout_point_lineage = _assert_layout_matches_mixed_points(
        layout=layout, points=points, points_path=paths["mixed_verification_points_artifact"]
    )
    layout_rows = _query_rows(layout=layout, query_id=query_id, split_name=split_name)
    point_rows = np.asarray(layout_point_rows[layout_rows], dtype=np.int64)
    query_layout = _slice_layout(layout, layout_rows)
    _validate_query_layout_against_points(
        query_layout=query_layout, points=points, points_rows=point_rows
    )
    if query_layout.row_count != 32:
        raise ValueError("high-resolution RGB scorer requires the frozen 32-point P1 denominator")

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
        raise ValueError("high-resolution RGB scorer requires one processed coordinate size")
    coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
    source_metadata = headers.metadata_by_name["radio_final"]
    source_manifest = str(source_metadata.get("source_image_manifest_sha256", ""))
    rgb_image_size = _discover_rgb_image_size(
        image_root=Path(args.image_root), image_id=str(image_ids[0])
    )
    rgb_bridge = validate_rgb_coordinate_bridge(
        source_metadata=source_metadata,
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
    )
    calibration = _load_json(paths["prior_temperature_calibration"])
    prior_temperature, calibration_source = _validate_calibration_for_scoring(
        calibration=calibration,
        calibration_path=paths["prior_temperature_calibration"],
        checkpoint_path=paths["checkpoint"],
        layout_path=paths["rgb_spatial_layout"],
        layout=layout,
        source_image_manifest_sha256=source_manifest,
    )
    model, checkpoint_metadata = _load_checkpoint_model(
        path=paths["checkpoint"], image_sizes=image_sizes, device=device
    )
    checkpoint_source = str(checkpoint_metadata["config"].get("source", "")).strip().lower()
    if checkpoint_source != calibration_source:
        raise ValueError("high-resolution RGB checkpoint/calibration source differs")
    requested_chunk = int(args.edge_chunk_size)
    if requested_chunk:
        model.edge_chunk_size = requested_chunk

    runtime = runtime_from_target_free_layout(query_layout, image_ids=image_ids)
    bank_track_ids, bank_xyz, bank_metadata = _load_bank_xyz(paths["projected_landmark_bank"])
    if str(bank_metadata.get("descriptor_space_manifest", {}).get("projection_space_id", "")) != str(
        query_layout.metadata.get("projection_space_id", "")
    ):
        raise ValueError("projected landmark bank projection space differs from RGB layout")
    geometry = _query_geometry(
        query_id=query_id,
        query_layout=query_layout,
        bank_track_ids=bank_track_ids,
        bank_xyz=bank_xyz,
        colmap_model_dir=Path(args.colmap_model_dir),
    )
    cache = TensorImageLRUCache(
        max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
        storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
    )
    cache_device = torch.device("cpu") if str(args.rgb_cache_device) == "cpu" else device
    with torch.no_grad():
        query_patches, support_patches = _crop_runtime_rgb_patches(
            runtime=runtime,
            image_ids=image_ids,
            image_root=Path(args.image_root),
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=float(model.full_patch_radius_px),
            step_px=1.0,
            cache=cache,
            device=device,
            cache_device=cache_device,
        )
        active_runtime = runtime
        active_support_patches = support_patches
        if str(args.evidence_variant) == "support_descriptor_permutation_control":
            active_support_patches = permute_support_patch_appearance(
                runtime=runtime, support_patches=support_patches, shift=1
            )
            if torch.equal(active_support_patches, support_patches):
                raise ValueError("high-resolution RGB support permutation did not alter appearance")
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and not bool(args.no_amp)):
            prediction = model(
                runtime=active_runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=active_support_patches,
                active_sources=active_source_names(calibration_source),
            )
        statistics = _score_hypotheses(
            model=model,
            runtime=active_runtime,
            prediction=prediction,
            geometry=geometry,
            poses_w2c=np.asarray(exact["poses_w2c"], dtype=np.float64),
            source=calibration_source,
            candidate_prior_temperature=prior_temperature,
            hypothesis_batch_size=int(args.hypothesis_batch_size),
            max_abs_pose_log_ratio=float(checkpoint_metadata["config"]["max_abs_pose_log_ratio"]),
        )
    arrays: dict[str, np.ndarray] = {
        "query_ids": np.asarray(exact["query_ids"]).astype(str),
        "split_names": np.asarray(exact["split_names"]).astype(str),
        "evaluation_labels": np.asarray(exact["evaluation_labels"]).astype(str),
        "hypothesis_indices": np.asarray(exact["hypothesis_indices"], dtype=np.int64),
        "source_chosen_for_optional_pose": np.asarray(exact["source_chosen_for_optional_pose"], dtype=bool),
        "baseline_score_top1": np.asarray(exact["independent_score_top1"], dtype=bool),
        "baseline_selection_scores": np.asarray(exact["independent_selection_scores"], dtype=np.float64),
        **statistics,
        "verification_source_point_ids": query_layout.source_point_ids,
        "verification_point_sources": query_layout.point_sources,
        "verification_source_detector_rows": points.source_detector_rows[point_rows],
        "verification_xy": query_layout.xy,
        "candidate_track_ids": query_layout.candidate_track_ids,
        "candidate_probabilities": query_layout.candidate_prior_probabilities,
        "null_probabilities": query_layout.null_probabilities,
        "candidate_view_weights": runtime.candidate_view_weights.numpy(),
        "candidate_support_image_ids": query_layout.support_image_ids,
    }
    row_count = len(arrays["query_ids"])
    row_fields = (
        "query_ids", "split_names", "evaluation_labels", "hypothesis_indices",
        "source_chosen_for_optional_pose", "baseline_score_top1", "baseline_selection_scores",
        "pose_log_likelihood_ratios", "point_log_likelihood_ratios",
        "point_effective_candidate_counts", "point_effective_view_masses",
        "source_log_likelihood_means", "source_effective_point_counts",
    )
    if any(np.asarray(arrays[name]).shape[0] != row_count for name in row_fields):
        raise RuntimeError("high-resolution RGB score row arrays are not aligned")
    metadata: dict[str, Any] = {
        "format": CANDIDATE_POSE_LLR_SCORE_FORMAT,
        "version": SCORE_VERSION,
        "model_format": CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "row_count": int(row_count),
        "query_count": 1,
        "query_id": query_id,
        "split_name": split_name,
        "evidence_variant": str(args.evidence_variant),
        "prior_temperature": prior_temperature,
        "prior_temperature_calibration": {
            "path": str(paths["prior_temperature_calibration"]),
            "sha256": file_sha256_short(paths["prior_temperature_calibration"]),
            "format": calibration.get("format"),
            "train_only_gate_passed": True,
        },
        "model_checkpoint": {"path": str(paths["checkpoint"]), "sha256": file_sha256_short(paths["checkpoint"])},
        "model_checkpoint_contract": {
            "architecture": checkpoint_metadata.get("architecture"),
            "config": checkpoint_metadata.get("config"),
            "heldout_evaluation_allowed": True,
            "train_only_inner_gate_passed": True,
        },
        "strict_candidate_pose_llr_contract": {
            "heldout_query_rows": True,
            "formal_p1_mixed_multiscale_points": True,
            "fixed_global_topl": True,
            "fixed_candidate_top_k": 20,
            "candidate_identity_fixed_across_hypotheses": True,
            "candidate_reselection_per_pose": False,
            "support_reselection_per_pose": False,
            "fixed_support_view_count": 2,
            "explicit_null": True,
            "candidate_projection_is_only_pose_dependent_encoder_input": True,
            "candidate_pose_matrix_excluded_from_encoder": True,
            "residual_and_target_excluded_from_encoder": True,
            "support_descriptor_permutation_control": str(args.evidence_variant)
            == "support_descriptor_permutation_control",
            "no_pnp": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
        "hypothesis_scope": {
            "all_frozen_hypotheses": int(args.hypothesis_limit) == 0,
            "development_prefix_limit": int(args.hypothesis_limit),
            "scored_hypothesis_count": int(row_count),
        },
        "raw_score_semantics": (
            "frozen_candidate_null_mixture_of_high_resolution_real_rgb_local_density_v1;"
            f"source={calibration_source};prior_temperature={prior_temperature:.6g}"
        ),
        "raw_score_is_calibrated_independent_pose_likelihood": False,
        "frozen_query_evidence_sha256": _frozen_query_evidence_sha(
            layout=query_layout, runtime=runtime
        ),
        "verification_point_selection": {
            "source": str(layout_point_lineage["mode"]),
            "point_count": int(query_layout.row_count),
            "parent_point_count": int(layout_point_lineage["parent_point_count"]),
            "frozen_layout_selected_point_count": int(layout_point_lineage["selected_point_count"]),
            "frozen_rgb_selector": layout_point_lineage["frozen_rgb_selector"],
            "source_point_ids_sha256": _canonical_hash(
                {"source_point_ids": query_layout.source_point_ids.tolist()}
            ),
            "target_free_static_only": True,
        },
        "support_descriptor_derangement": (
            None if str(args.evidence_variant) == "visual" else "fixed_valid_support_edge_cyclic_shift_v1"
        ),
        "baseline_reference_hypothesis_equivalence": baseline_reference_equivalence,
        "rgb_coordinate_bridge": rgb_bridge,
        "inputs": _input_manifest(paths),
        "source_metadata_hashes": {
            "hypothesis": _canonical_hash(hypothesis_metadata),
            "baseline_s0": _canonical_hash(baseline_metadata),
            "mixed_verification_points": _canonical_hash(points.metadata),
            "rgb_spatial_layout": _canonical_hash(query_layout.metadata),
            "projected_landmark_bank": _canonical_hash(bank_metadata),
        },
        "rgb_cache": {"device": str(cache_device), **cache.summary()},
        "elapsed_seconds": float(time.time() - start),
    }
    validate_target_free_pose_llr_score_metadata(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
    os.replace(temporary, output_path)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps({"stage": "score_candidate_highres_rgb_multiscale_likelihood", "output": str(output_path), "metadata": metadata}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "metadata": metadata}, sort_keys=True))
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    score_candidate_highres_rgb_multiscale_likelihood(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
