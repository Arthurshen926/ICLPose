"""Score frozen held-out pose hypotheses with a trained candidate-specific LLR.

This program deliberately has no target artifact argument.  Candidate pose
matrices are used only to project fixed 3-D candidates into the held-out query
image.  The learned encoder receives real-image feature crops and fixed support
views, then emits a bounded visual log-likelihood ratio.  The artifact remains
diagnostic-only until a separate target-side audit passes its gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _load_exact_hypotheses,
    _load_npz_allowlist,
)
from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    load_inference_artifact_fields,
)
from feature_extract.tools.vfm.score_v5_dynamic_absolute_context_pose_evidence import (
    _formal_p1_mixed_evidence_for_query,
)
from feature_extract.tools.vfm.train_candidate_pose_llr import (
    CHECKPOINT_FORMAT,
    _QueryRuntime,
    _prepare_query_runtimes,
    _project_candidate_positions,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_llr import (
    CANDIDATE_POSE_LLR_FORMAT,
    CANDIDATE_POSE_LLR_SCORE_FORMAT,
    CandidatePoseLLRRuntime,
    CandidateSpecificPoseLLR,
    grouped_hypothesis_semantic_manifest,
    score_candidate_pose_batch,
    validate_grouped_hypothesis_semantic_match,
    validate_target_free_pose_llr_score_metadata,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.localization.frozen_pose_conditioned_maplet_appearance import (
    deterministic_support_descriptor_derangement,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    load_mixed_verification_points,
)


SCORE_VERSION = "candidate_specific_correct_vs_coherent_wrong_pose_llr_target_free_v3"
_CHECKPOINT_INPUT_KEYS = (
    "verification_points",
    "maplet_support_index",
    "support_geometry_index",
    "projected_landmark_bank",
    "radio_final_context_cache",
    "radio_intermediate_context_cache",
    "alike_spatial_context_cache",
    "colmap_cameras_bin",
    "colmap_images_bin",
)
_BASELINE_REFERENCE_FIELDS = ("poses_w2c", "chosen_for_optional_pose")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis-artifact", required=True)
    parser.add_argument("--baseline-score-artifact", required=True)
    parser.add_argument(
        "--baseline-reference-hypothesis-artifact",
        default="",
        help=(
            "optional target-free source artifact referenced by the baseline; "
            "it is accepted only after exact row/pose/selector equivalence "
            "with --hypothesis-artifact"
        ),
    )
    parser.add_argument("--detector-query-cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--fixed-candidate-prior-overlay", required=True)
    parser.add_argument("--mixed-verification-points-artifact", required=True)
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support-view-count", type=int, default=2)
    parser.add_argument("--hypothesis-batch-size", type=int, default=4)
    parser.add_argument(
        "--edge-chunk-size",
        type=int,
        default=0,
        help="edge scoring chunk override; zero preserves the checkpoint default",
    )
    parser.add_argument("--missing-edge-log-likelihood-ratio", type=float, default=0.0)
    parser.add_argument(
        "--evidence-variant",
        choices=("visual", "support_descriptor_permutation_control"),
        default="visual",
    )
    parser.add_argument(
        "--hypothesis-limit",
        type=int,
        default=0,
        help="development-only frozen hypothesis prefix; zero scores every input row",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def resolve_edge_chunk_size(
    *, checkpoint_edge_chunk_size: int, requested_edge_chunk_size: int
) -> int:
    """Use the checkpoint setting unless a positive scoring override is requested."""

    checkpoint_size = int(checkpoint_edge_chunk_size)
    requested_size = int(requested_edge_chunk_size)
    if checkpoint_size <= 0:
        raise ValueError("checkpoint edge chunk size must be positive")
    if requested_size < 0:
        raise ValueError("requested edge chunk size must be zero or positive")
    return checkpoint_size if requested_size == 0 else requested_size


def _input_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": file_sha256_short(path)}
        for name, path in paths.items()
    }


def _checkpoint_input_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        "verification_points": {
            "sha256": file_sha256_short(paths["mixed_verification_points_artifact"])
        },
        "maplet_support_index": {"sha256": file_sha256_short(paths["maplet_support_index"])},
        "support_geometry_index": {
            "sha256": file_sha256_short(paths["support_geometry_index"])
        },
        "projected_landmark_bank": {
            "sha256": file_sha256_short(paths["projected_landmark_bank"])
        },
        "radio_final_context_cache": {
            "sha256": file_sha256_short(paths["radio_final_context_cache"])
        },
        "radio_intermediate_context_cache": {
            "sha256": file_sha256_short(paths["radio_intermediate_context_cache"])
        },
        "alike_spatial_context_cache": {
            "sha256": file_sha256_short(paths["alike_spatial_context_cache"])
        },
        "colmap_cameras_bin": {"sha256": file_sha256_short(paths["colmap_cameras_bin"])},
        "colmap_images_bin": {"sha256": file_sha256_short(paths["colmap_images_bin"])},
    }


def validate_checkpoint_for_target_free_scoring(
    metadata: Mapping[str, object], *, expected_inputs: Mapping[str, Mapping[str, object]]
) -> None:
    """Reject stale or promotable checkpoints before held-out scoring."""

    required = {
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": 20,
        "fixed_support_view_count": 2,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "explicit_null": True,
    }
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("format") != CHECKPOINT_FORMAT
        or metadata.get("model_format") != CANDIDATE_POSE_LLR_FORMAT
        or any(metadata.get(key) != value for key, value in required.items())
    ):
        raise ValueError("candidate pose-LLR checkpoint violates the diagnostic contract")
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping) or set(expected_inputs) != set(_CHECKPOINT_INPUT_KEYS):
        raise ValueError("candidate pose-LLR checkpoint inputs are invalid")
    for name in _CHECKPOINT_INPUT_KEYS:
        observed = inputs.get(name)
        expected = expected_inputs[name]
        if (
            not isinstance(observed, Mapping)
            or not isinstance(expected, Mapping)
            or str(observed.get("sha256", "")) != str(expected.get("sha256", ""))
        ):
            raise ValueError(f"candidate pose-LLR checkpoint is stale for {name}")


def validate_checkpoint_hypothesis_semantic_lineage(
    *, checkpoint_metadata: Mapping[str, object], hypothesis_metadata: Mapping[str, object]
) -> None:
    """Keep a train-fitted LLR out of a different grouped-PnP distribution."""

    expected = (
        checkpoint_metadata.get("hypothesis_semantic_lineage")
        if isinstance(checkpoint_metadata, Mapping)
        else None
    )
    if not isinstance(expected, Mapping):
        raise ValueError("candidate pose-LLR checkpoint lacks hypothesis semantic lineage")
    observed = grouped_hypothesis_semantic_manifest(hypothesis_metadata)
    validate_grouped_hypothesis_semantic_match(expected=expected, observed=observed)


def validate_baseline_reference_hypothesis_equivalence(
    *, current_hypothesis_path: Path, reference_hypothesis_path: Path
) -> dict[str, object]:
    """Prove a baseline source can be reused only for an exact frozen-row copy.

    Some artifact revisions add an explicit evidence-version contract without
    changing any frozen hypotheses.  A baseline score may bridge such a
    revision only when the target-free input manifest and every row's key,
    pose, and source selector state are identical.  This is deliberately
    stronger than matching descriptor dimensionality or semantic labels.
    """

    current, current_metadata = load_inference_artifact_fields(
        Path(current_hypothesis_path), _BASELINE_REFERENCE_FIELDS
    )
    reference, reference_metadata = load_inference_artifact_fields(
        Path(reference_hypothesis_path), _BASELINE_REFERENCE_FIELDS
    )
    if (
        current_metadata.get("format") != "grouped_pose_hypotheses_inference_only_v1"
        or reference_metadata.get("format")
        != "grouped_pose_hypotheses_inference_only_v1"
        or current_metadata.get("contains_target_fields") is not False
        or reference_metadata.get("contains_target_fields") is not False
        or current_metadata.get("pose_or_ground_truth_used_for_generation") is not False
        or reference_metadata.get("pose_or_ground_truth_used_for_generation")
        is not False
        or current_metadata.get("inputs") != reference_metadata.get("inputs")
    ):
        raise ValueError("baseline reference artifact has different target-free inputs")

    def rows_by_key(
        arrays: Mapping[str, np.ndarray], *, source: str
    ) -> dict[tuple[str, str, str, int], int]:
        keys = list(
            zip(
                np.asarray(arrays["query_ids"]).astype(str).tolist(),
                np.asarray(arrays["split_names"]).astype(str).tolist(),
                np.asarray(arrays["evaluation_labels"]).astype(str).tolist(),
                np.asarray(arrays["hypothesis_indices"], dtype=np.int64).tolist(),
            )
        )
        if len(keys) == 0 or len(keys) != len(set(keys)):
            raise ValueError(f"baseline reference {source} has invalid row identities")
        return {key: index for index, key in enumerate(keys)}

    current_rows = rows_by_key(current, source="current artifact")
    reference_rows = rows_by_key(reference, source="reference artifact")
    if set(current_rows) != set(reference_rows):
        raise ValueError("baseline reference artifact has different frozen row identities")
    for key in current_rows:
        current_index = current_rows[key]
        reference_index = reference_rows[key]
        if not np.array_equal(
            np.asarray(current["poses_w2c"])[current_index],
            np.asarray(reference["poses_w2c"])[reference_index],
        ):
            raise ValueError("baseline reference artifact has a different frozen pose")
        if bool(np.asarray(current["chosen_for_optional_pose"])[current_index]) != bool(
            np.asarray(reference["chosen_for_optional_pose"])[reference_index]
        ):
            raise ValueError("baseline reference artifact has a different source selector")
    return {
        "rule": "exact_target_free_row_pose_selector_equivalence_v1",
        "current_hypothesis_artifact": {
            "path": str(current_hypothesis_path),
            "sha256": file_sha256_short(Path(current_hypothesis_path)),
        },
        "reference_hypothesis_artifact": {
            "path": str(reference_hypothesis_path),
            "sha256": file_sha256_short(Path(reference_hypothesis_path)),
        },
        "validated_row_count": int(len(current_rows)),
    }


def remap_support_image_indices(
    *,
    runtime: CandidatePoseLLRRuntime,
    image_ids: Sequence[str] | np.ndarray,
    descriptor_derangement: Mapping[str, str],
) -> CandidatePoseLLRRuntime:
    """Replace only support descriptor image addresses for the paired control."""

    ids = np.asarray(image_ids).astype(str).reshape(-1)
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise ValueError("candidate pose-LLR image IDs are invalid")
    positions = {image_id: index for index, image_id in enumerate(ids.tolist())}
    if set(descriptor_derangement) != set(positions):
        raise ValueError("candidate pose-LLR descriptor control does not cover every image")
    mapped = np.asarray(
        [
            positions.get(str(descriptor_derangement.get(ids[int(index)], "")), -1)
            for index in runtime.support_image_indices.reshape(-1).tolist()
        ],
        dtype=np.int64,
    ).reshape(tuple(runtime.support_image_indices.shape))
    if np.any(mapped < 0):
        raise ValueError("candidate pose-LLR descriptor control refers to an unknown image")
    return CandidatePoseLLRRuntime(
        query_image_indices=runtime.query_image_indices,
        support_image_indices=torch.from_numpy(mapped),
        support_xy=runtime.support_xy,
        support_view_valid=runtime.support_view_valid,
        candidate_view_weights=runtime.candidate_view_weights,
        candidate_probabilities=runtime.candidate_probabilities,
        null_probabilities=runtime.null_probabilities,
    )


def _load_checkpoint_model(
    *,
    path: Path,
    expected_inputs: Mapping[str, Mapping[str, object]],
    sources: Sequence[object],
    device: torch.device,
) -> tuple[CandidateSpecificPoseLLR, dict[str, object]]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch < 2.0
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
        raise ValueError("candidate pose-LLR checkpoint is malformed")
    metadata = dict(payload["metadata"])
    validate_checkpoint_for_target_free_scoring(metadata, expected_inputs=expected_inputs)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("candidate pose-LLR checkpoint has no state dict")
    source_by_name = {str(source.name): source for source in sources}
    if set(source_by_name) != {"radio_final", "radio_intermediate", "alike"}:
        raise ValueError("candidate pose-LLR scoring sources are incomplete")
    reference = source_by_name["radio_final"]
    for source in source_by_name.values():
        if not np.array_equal(source.image_ids, reference.image_ids) or not np.array_equal(
            source.image_sizes, reference.image_sizes
        ):
            raise ValueError("candidate pose-LLR scoring cache ownership differs across scales")
    model = CandidateSpecificPoseLLR(
        sources={
            name: torch.from_numpy(np.asarray(source.grid))
            for name, source in source_by_name.items()
        },
        image_sizes=torch.from_numpy(np.asarray(reference.image_sizes, dtype=np.float32)),
        hidden_dim=int(metadata.get("hidden_dim", 0)),
        max_abs_log_ratio=float(metadata.get("max_abs_log_ratio", 0.0)),
        edge_chunk_size=int(metadata.get("edge_chunk_size", 0)),
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), metadata


def _assert_runtime_matches_formal_points(
    *, query: _QueryRuntime, formal: Mapping[str, np.ndarray]
) -> None:
    runtime = query.runtime
    if (
        runtime.candidate_probabilities.shape != torch.Size(formal["candidate_probabilities"].shape)
        or tuple(query.candidate_xyz.shape[:2])
        != tuple(np.asarray(formal["candidate_track_ids"]).shape)
        or not np.allclose(
            runtime.candidate_probabilities.numpy(),
            np.asarray(formal["candidate_probabilities"], dtype=np.float32),
            atol=1e-6,
        )
        or not np.allclose(
            runtime.null_probabilities.numpy(),
            np.asarray(formal["null_probabilities"], dtype=np.float32),
            atol=1e-6,
        )
    ):
        raise ValueError("candidate pose-LLR runtime differs from formal held-out points")


def _score_hypotheses(
    *,
    model: CandidateSpecificPoseLLR,
    query: _QueryRuntime,
    poses_w2c: np.ndarray,
    point_sources: np.ndarray,
    hypothesis_batch_size: int,
    missing_edge_log_likelihood_ratio: float,
) -> dict[str, np.ndarray]:
    poses = np.asarray(poses_w2c, dtype=np.float64)
    sources = np.asarray(point_sources).astype(str).reshape(-1)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or len(poses) == 0
        or len(sources) != len(query.runtime.query_image_indices)
        or int(hypothesis_batch_size) <= 0
        or not np.isfinite(poses).all()
    ):
        raise ValueError("candidate pose-LLR hypothesis score inputs are invalid")
    source_names = np.asarray(sorted(set(sources.tolist())), dtype=np.str_)
    outputs: dict[str, list[np.ndarray]] = {
        "pose_log_likelihood_ratios": [],
        "point_log_likelihood_ratios": [],
        "point_effective_candidate_counts": [],
        "point_effective_view_masses": [],
        "source_log_likelihood_means": [],
        "source_effective_point_counts": [],
    }
    candidate_probability = query.runtime.candidate_probabilities.to(
        device=model.device, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(3)
    view_weights = query.runtime.candidate_view_weights.to(
        device=model.device, dtype=torch.float32
    ).unsqueeze(0)
    amp_enabled = model.device.type == "cuda"
    with torch.no_grad():
        for begin in range(0, len(poses), int(hypothesis_batch_size)):
            end = min(begin + int(hypothesis_batch_size), len(poses))
            projected_xy, projected_valid = _project_candidate_positions(
                query=query,
                poses_w2c=torch.from_numpy(poses[begin:end]),
                device=model.device,
            )
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                score = score_candidate_pose_batch(
                    model=model,
                    runtime=query.runtime,
                    candidate_query_xy=projected_xy,
                    candidate_projection_valid=projected_valid,
                    missing_edge_log_likelihood_ratio=float(missing_edge_log_likelihood_ratio),
                )
            usable_candidate = score.edge_usable.any(dim=3) & (candidate_probability[..., 0] > 0.0)
            effective_mass = (
                score.edge_usable.to(dtype=torch.float32)
                * candidate_probability
                * view_weights
            ).sum(dim=(2, 3))
            point = score.point_log_likelihood_ratios.detach().cpu().numpy().astype(np.float32)
            effective = usable_candidate.sum(dim=2).detach().cpu().numpy().astype(np.int16)
            mass = effective_mass.detach().cpu().numpy().astype(np.float32)
            outputs["pose_log_likelihood_ratios"].append(
                score.pose_log_likelihood_ratios.detach().cpu().numpy().astype(np.float32)
            )
            outputs["point_log_likelihood_ratios"].append(point)
            outputs["point_effective_candidate_counts"].append(effective)
            outputs["point_effective_view_masses"].append(mass)
            outputs["source_log_likelihood_means"].append(
                np.stack(
                    [point[:, sources == name].mean(axis=1) for name in source_names], axis=1
                ).astype(np.float32)
            )
            outputs["source_effective_point_counts"].append(
                np.stack(
                    [
                        (effective[:, sources == name] > 0).sum(axis=1)
                        for name in source_names
                    ],
                    axis=1,
                ).astype(np.int16)
            )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    result = {name: np.concatenate(values, axis=0) for name, values in outputs.items()}
    result["source_names"] = source_names
    return result


def _source_static_digest(
    *, formal: Mapping[str, np.ndarray], runtime: CandidatePoseLLRRuntime
) -> str:
    digest = hashlib.sha256()
    arrays = {
        "source_point_ids": np.asarray(formal["source_point_ids"]),
        "xy": np.asarray(formal["xy"]),
        "point_sources": np.asarray(formal["point_sources"]),
        "candidate_track_ids": np.asarray(formal["candidate_track_ids"]),
        "candidate_probabilities": np.asarray(formal["candidate_probabilities"]),
        "null_probabilities": np.asarray(formal["null_probabilities"]),
        "support_image_indices": runtime.support_image_indices.numpy(),
        "support_xy": runtime.support_xy.numpy(),
        "support_view_valid": runtime.support_view_valid.numpy(),
        "candidate_view_weights": runtime.candidate_view_weights.numpy(),
    }
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()[:16]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.time()
    if (
        int(args.support_view_count) != 2
        or int(args.hypothesis_batch_size) <= 0
        or int(args.hypothesis_limit) < 0
        or not np.isfinite(float(args.missing_edge_log_likelihood_ratio))
    ):
        raise ValueError("candidate pose-LLR scorer requires two support views and positive batches")
    output_path = Path(args.output)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"output already exists: {output_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("candidate pose-LLR scorer requested CUDA but CUDA is unavailable")
    paths = {
        "hypothesis_artifact": Path(args.hypothesis_artifact),
        "baseline_score_artifact": Path(args.baseline_score_artifact),
        "detector_query_cache": Path(args.detector_query_cache),
        "proposals": Path(args.proposals),
        "candidate_artifact": Path(args.candidate_artifact),
        "fixed_candidate_prior_overlay": Path(args.fixed_candidate_prior_overlay),
        "mixed_verification_points_artifact": Path(args.mixed_verification_points_artifact),
        "maplet_support_index": Path(args.maplet_support_index),
        "support_geometry_index": Path(args.support_geometry_index),
        "projected_landmark_bank": Path(args.projected_landmark_bank),
        "radio_final_context_cache": Path(args.radio_final_context_cache),
        "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
        "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
        "checkpoint": Path(args.checkpoint),
        "colmap_cameras_bin": Path(args.colmap_model_dir) / "cameras.bin",
        "colmap_images_bin": Path(args.colmap_model_dir) / "images.bin",
    }
    baseline_reference_path = (
        None
        if not str(args.baseline_reference_hypothesis_artifact).strip()
        else Path(args.baseline_reference_hypothesis_artifact)
    )
    if baseline_reference_path is not None:
        paths["baseline_reference_hypothesis_artifact"] = baseline_reference_path
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"candidate pose-LLR input is missing: {name} ({path})")
    baseline_hypothesis_path = (
        paths["hypothesis_artifact"]
        if baseline_reference_path is None
        else baseline_reference_path
    )
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
    if baseline_reference_path is None:
        hypothesis_metadata = baseline_hypothesis_metadata
    else:
        baseline_reference_equivalence = validate_baseline_reference_hypothesis_equivalence(
            current_hypothesis_path=paths["hypothesis_artifact"],
            reference_hypothesis_path=baseline_reference_path,
        )
        _current_fields, hypothesis_metadata = load_inference_artifact_fields(
            paths["hypothesis_artifact"], _BASELINE_REFERENCE_FIELDS
        )
    if int(args.hypothesis_limit) > 0:
        exact = {
            name: np.asarray(value)[: int(args.hypothesis_limit)]
            for name, value in exact.items()
        }
    query_id = str(np.asarray(exact["query_ids"]).astype(str)[0])
    query_split = str(np.asarray(exact["split_names"]).astype(str)[0])
    if query_split not in {"validation", "test"}:
        raise ValueError("candidate pose-LLR scorer accepts only held-out query shards")

    detector, detector_metadata, _ = _load_npz_allowlist(
        paths["detector_query_cache"], ("image_ids", "offsets", "xy", "detector_scores")
    )
    proposals, proposal_metadata, _ = _load_npz_allowlist(
        paths["proposals"], ("query_ids", "candidate_track_ids", "coarse_scores"), metadata_required=False
    )
    candidate, candidate_metadata, _ = _load_npz_allowlist(
        paths["candidate_artifact"], ("selected_rows",)
    )
    if (
        detector_metadata.get("format") != "alike_detector_mapped_radio_query_cache_v1"
        or candidate_metadata.get("contains_ground_truth") is not False
        or candidate_metadata.get("contains_pose_derived_selection") is not False
        or proposal_metadata.get("format") not in {None, "detector_support_reranked_proposals_v1"}
    ):
        raise ValueError("candidate pose-LLR source artifacts violate the target-free contract")
    formal, point_selection, mixed_metadata = _formal_p1_mixed_evidence_for_query(
        points_path=paths["mixed_verification_points_artifact"],
        query_id=query_id,
        query_split=query_split,
        detector_path=paths["detector_query_cache"],
        candidate_path=paths["candidate_artifact"],
        landmark_bank_path=paths["projected_landmark_bank"],
        detector=detector,
        proposals=proposals,
        selected_rows=np.asarray(candidate["selected_rows"], dtype=np.int64),
        point_count=192,
    )
    if mixed_metadata.get("format") != MIXED_VERIFICATION_POINTS_FORMAT:
        raise ValueError("candidate pose-LLR mixed points have an invalid format")
    points = load_mixed_verification_points(paths["mixed_verification_points_artifact"])
    source_rows = points.rows_for_query(query_id)
    formal_fields = {
        "xy": points.xy,
        "candidate_track_ids": points.candidate_track_ids,
        "candidate_probabilities": points.candidate_prior_probabilities,
        "null_probabilities": points.null_probabilities,
    }
    formal_names = {
        "xy": "xy",
        "candidate_track_ids": "candidate_track_ids",
        "candidate_probabilities": "candidate_probabilities",
        "null_probabilities": "null_probabilities",
    }
    if any(
        not np.allclose(
            np.asarray(values)[source_rows],
            np.asarray(formal[formal_names[name]]),
            atol=1e-6,
        )
        for name, values in formal_fields.items()
    ):
        raise ValueError("candidate pose-LLR formal points do not preserve source order")
    sources = load_context_attention_sources(
        radio_final_context_cache=paths["radio_final_context_cache"],
        radio_intermediate_context_cache=paths["radio_intermediate_context_cache"],
        alike_spatial_context_cache=paths["alike_spatial_context_cache"],
        expected_radio_checkpoint="",
        require_equal_descriptor_dimensions=False,
    )
    runtimes = _prepare_query_runtimes(
        points=points,
        sources=sources,
        maplet_support_index=paths["maplet_support_index"],
        support_geometry_index=paths["support_geometry_index"],
        projected_landmark_bank=paths["projected_landmark_bank"],
        colmap_model_dir=Path(args.colmap_model_dir),
        support_view_count=int(args.support_view_count),
        required_split=query_split,
    )
    query = runtimes.get(query_id)
    if query is None:
        raise ValueError("candidate pose-LLR query has no held-out runtime")
    _assert_runtime_matches_formal_points(query=query, formal=formal)
    reference_source = next(source for source in sources if str(source.name) == "radio_final")
    active_runtime = query.runtime
    descriptor_derangement: Mapping[str, str] | None = None
    if str(args.evidence_variant) == "support_descriptor_permutation_control":
        descriptor_derangement = deterministic_support_descriptor_derangement(
            image_ids=np.asarray(reference_source.image_ids).astype(str),
            image_sizes=np.asarray(reference_source.image_sizes, dtype=np.int64),
        )
        active_runtime = remap_support_image_indices(
            runtime=active_runtime,
            image_ids=np.asarray(reference_source.image_ids).astype(str),
            descriptor_derangement=descriptor_derangement,
        )
        query = _QueryRuntime(
            runtime=active_runtime,
            candidate_xyz=query.candidate_xyz,
            focal_length=query.focal_length,
            principal_x=query.principal_x,
            principal_y=query.principal_y,
            radial_k=query.radial_k,
            image_width=query.image_width,
            image_height=query.image_height,
        )
    checkpoint_inputs = _checkpoint_input_manifest(paths)
    model, checkpoint_metadata = _load_checkpoint_model(
        path=paths["checkpoint"],
        expected_inputs=checkpoint_inputs,
        sources=sources,
        device=device,
    )
    validate_checkpoint_hypothesis_semantic_lineage(
        checkpoint_metadata=checkpoint_metadata,
        hypothesis_metadata=hypothesis_metadata,
    )
    inference_edge_chunk_size = resolve_edge_chunk_size(
        checkpoint_edge_chunk_size=int(checkpoint_metadata.get("edge_chunk_size", 0)),
        requested_edge_chunk_size=int(args.edge_chunk_size),
    )
    model.edge_chunk_size = int(inference_edge_chunk_size)
    statistics = _score_hypotheses(
        model=model,
        query=query,
        poses_w2c=np.asarray(exact["poses_w2c"], dtype=np.float64),
        point_sources=np.asarray(formal["point_sources"]).astype(str),
        hypothesis_batch_size=int(args.hypothesis_batch_size),
        missing_edge_log_likelihood_ratio=float(args.missing_edge_log_likelihood_ratio),
    )
    static_digest = _source_static_digest(formal=formal, runtime=active_runtime if descriptor_derangement is None else runtimes[query_id].runtime)
    support_image_ids = np.asarray(reference_source.image_ids).astype(str)[
        runtimes[query_id].runtime.support_image_indices.numpy()
    ]
    arrays: dict[str, np.ndarray] = {
        "query_ids": np.asarray(exact["query_ids"]).astype(str),
        "split_names": np.asarray(exact["split_names"]).astype(str),
        "evaluation_labels": np.asarray(exact["evaluation_labels"]).astype(str),
        "hypothesis_indices": np.asarray(exact["hypothesis_indices"], dtype=np.int64),
        "source_chosen_for_optional_pose": np.asarray(exact["source_chosen_for_optional_pose"], dtype=bool),
        "baseline_score_top1": np.asarray(exact["independent_score_top1"], dtype=bool),
        "baseline_selection_scores": np.asarray(exact["independent_selection_scores"], dtype=np.float64),
        "pose_log_likelihood_ratios": statistics["pose_log_likelihood_ratios"],
        "point_log_likelihood_ratios": statistics["point_log_likelihood_ratios"],
        "point_effective_candidate_counts": statistics["point_effective_candidate_counts"],
        "point_effective_view_masses": statistics["point_effective_view_masses"],
        "source_names": statistics["source_names"],
        "source_log_likelihood_means": statistics["source_log_likelihood_means"],
        "source_effective_point_counts": statistics["source_effective_point_counts"],
        "verification_source_point_ids": np.asarray(formal["source_point_ids"], dtype=np.int64),
        "verification_point_sources": np.asarray(formal["point_sources"]).astype(str),
        "verification_source_detector_rows": np.asarray(formal["source_detector_rows"], dtype=np.int64),
        "verification_xy": np.asarray(formal["xy"], dtype=np.float32),
        "candidate_track_ids": np.asarray(formal["candidate_track_ids"], dtype=np.int64),
        "candidate_probabilities": np.asarray(formal["candidate_probabilities"], dtype=np.float32),
        "null_probabilities": np.asarray(formal["null_probabilities"], dtype=np.float32),
        "candidate_view_weights": runtimes[query_id].runtime.candidate_view_weights.numpy(),
        "candidate_support_image_ids": support_image_ids,
    }
    metadata: dict[str, Any] = {
        "format": CANDIDATE_POSE_LLR_SCORE_FORMAT,
        "version": SCORE_VERSION,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "row_count": int(len(arrays["query_ids"])),
        "query_count": 1,
        "query_id": query_id,
        "split_name": query_split,
        "evidence_variant": str(args.evidence_variant),
        "model_checkpoint": {"path": str(paths["checkpoint"]), "sha256": file_sha256_short(paths["checkpoint"])},
        "model_checkpoint_contract": {
            key: checkpoint_metadata.get(key)
            for key in ("architecture", "hidden_dim", "max_abs_log_ratio", "edge_chunk_size")
        },
        "inference_edge_chunk_size": int(inference_edge_chunk_size),
        "requested_edge_chunk_size": int(args.edge_chunk_size),
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
            "scored_hypothesis_count": int(len(arrays["query_ids"])),
        },
        "raw_score_semantics": "train_only_correct_vs_coherent_wrong_bounded_visual_llr_under_fixed_candidate_null_mixture",
        "raw_score_is_calibrated_independent_pose_likelihood": False,
        "frozen_query_evidence_sha256": static_digest,
        "verification_point_selection": point_selection,
        "support_descriptor_derangement": (
            None
            if descriptor_derangement is None
            else _canonical_hash(dict(sorted(descriptor_derangement.items())))
        ),
        "baseline_reference_hypothesis_equivalence": baseline_reference_equivalence,
        "inputs": _input_manifest(paths),
        "source_metadata_hashes": {
            "hypothesis": _canonical_hash(hypothesis_metadata),
            "baseline_s0": _canonical_hash(baseline_metadata),
            "detector": _canonical_hash(detector_metadata),
            "proposals": _canonical_hash(proposal_metadata),
            "candidate": _canonical_hash(candidate_metadata),
            "mixed_verification_points": _canonical_hash(mixed_metadata),
        },
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
    summary_path.write_text(
        json.dumps(
            {
                "stage": "score_candidate_pose_llr",
                "output": str(output_path),
                "metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"output": str(output_path), "metadata": metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
