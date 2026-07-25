"""Score frozen held-out hypotheses with latent identity and pose alignment.

This program has no pose-target or SfM-observation-target argument.  It first
evaluates a soft candidate identity posterior at fixed observed query tokens,
then freezes that posterior while candidate-specific real-image alignment
evidence scores each frozen pose.  Visual and support-descriptor-permutation
control variants are emitted together from the identical frozen layout.
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

from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    load_inference_artifact_fields,
)
from feature_extract.tools.vfm.score_candidate_pose_llr import (
    remap_support_image_indices,
    resolve_edge_chunk_size,
)
from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _load_npz_allowlist,
)
from feature_extract.tools.vfm.score_v5_dynamic_absolute_context_pose_evidence import (
    _formal_p1_mixed_evidence_for_query,
)
from feature_extract.tools.vfm.train_candidate_pose_latent_evidence import (
    CHECKPOINT_FORMAT,
)
from feature_extract.tools.vfm.train_candidate_pose_llr import (
    _QueryRuntime,
    _prepare_query_runtimes,
    _project_candidate_positions,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_latent_evidence import (
    CANDIDATE_POSE_LATENT_EVIDENCE_FORMAT,
    CandidatePoseLatentEvidence,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    CandidatePoseLLRRuntime,
    grouped_hypothesis_semantic_manifest,
    validate_grouped_hypothesis_semantic_match,
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
    validate_heldout_scoring_point_cache,
)


SCORE_FORMAT = "candidate_pose_latent_evidence_scores_v1"
SCORE_VERSION = "latent_identity_soft_topl_candidate_alignment_paired_control_ragged_v2"
EVIDENCE_VARIANTS = ("visual", "support_descriptor_permutation_control")
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
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis-artifact", required=True)
    parser.add_argument(
        "--baseline-score-artifact",
        default="",
        help=(
            "deprecated for current v10 grouped hypotheses; leave empty because "
            "the immutable baseline is stored in the frozen hypothesis artifact"
        ),
    )
    parser.add_argument(
        "--baseline-reference-hypothesis-artifact",
        default="",
        help="deprecated for current v10 grouped hypotheses; leave empty",
    )
    parser.add_argument("--detector-query-cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate-artifact", required=True)
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
    parser.add_argument("--hypothesis-batch-size", type=int, default=8)
    parser.add_argument("--edge-chunk-size", type=int, default=0)
    parser.add_argument(
        "--evidence-variants",
        default="visual,support_descriptor_permutation_control",
        help="must include the paired visual and support-descriptor control variants",
    )
    parser.add_argument(
        "--score-splits",
        default="validation,test",
        help="comma-separated held-out source splits to score",
    )
    parser.add_argument("--hypothesis-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def parse_evidence_variants(value: str) -> tuple[str, ...]:
    """Require a visual/control pair rather than an unpaired appearance run."""

    variants = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if (
        len(variants) != len(set(variants))
        or set(variants) != set(EVIDENCE_VARIANTS)
    ):
        raise ValueError("scorer requires the paired visual/control evidence variants")
    return variants


def parse_score_splits(value: str) -> tuple[str, ...]:
    """Parse the held-out source split subset without admitting train rows."""

    splits = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if (
        not splits
        or len(splits) != len(set(splits))
        or set(splits) - {"validation", "test"}
    ):
        raise ValueError("score splits must be a unique non-empty validation/test subset")
    return splits


def select_heldout_query_groups(
    *, query_ids: Sequence[str] | np.ndarray, split_names: Sequence[str] | np.ndarray
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """Keep whole validation/test query groups from a mixed frozen shard."""

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    splits = np.asarray(split_names).astype(str).reshape(-1)
    if len(ids) == 0 or splits.shape != ids.shape:
        raise ValueError("frozen query IDs and split names are invalid")
    groups: list[tuple[str, str, tuple[int, ...]]] = []
    for query_id in sorted(set(ids.tolist())):
        rows = np.flatnonzero(ids == query_id)
        query_splits = tuple(sorted(set(splits[rows].tolist())))
        if len(query_splits) != 1:
            raise ValueError("a frozen query appears under multiple splits")
        split = query_splits[0]
        if split in {"validation", "test"}:
            groups.append((str(query_id), split, tuple(int(row) for row in rows.tolist())))
        elif split != "train":
            raise ValueError("frozen query has an unsupported split")
    if not groups:
        raise ValueError("frozen shard has no held-out query groups")
    return tuple(groups)


def select_scored_query_groups(
    *,
    query_ids: Sequence[str] | np.ndarray,
    split_names: Sequence[str] | np.ndarray,
    score_splits: Sequence[str],
    hypothesis_limit: int,
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """Select whole held-out query groups with an optional per-group prefix.

    A mixed frozen shard is not a single query artifact.  Applying a raw row
    prefix to it can silently keep train rows or truncate an unrelated query.
    The development limit therefore applies independently inside each selected
    query group.
    """

    requested = tuple(str(split) for split in score_splits)
    if (
        not requested
        or len(requested) != len(set(requested))
        or set(requested) - {"validation", "test"}
        or int(hypothesis_limit) < 0
    ):
        raise ValueError("scored query splits or hypothesis limit are invalid")
    groups = select_heldout_query_groups(query_ids=query_ids, split_names=split_names)
    selected: list[tuple[str, str, tuple[int, ...]]] = []
    for query_id, split_name, rows in groups:
        if split_name not in requested:
            continue
        kept = rows if int(hypothesis_limit) == 0 else rows[: int(hypothesis_limit)]
        if not kept:
            raise RuntimeError("held-out query group lost every frozen hypothesis")
        selected.append((query_id, split_name, kept))
    if not selected:
        raise ValueError("frozen shard has no requested held-out query groups")
    return tuple(selected)


def native_frozen_hypothesis_baseline(
    *,
    query_ids: np.ndarray,
    split_names: np.ndarray,
    evaluation_labels: np.ndarray,
    hypothesis_indices: np.ndarray,
    verification_log_likelihood_means: np.ndarray,
    chosen_for_optional_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a v10 hypothesis-native verifier score into an auditable baseline.

    V10 stores a score only for hypotheses admitted to verification, leaving
    the rest as NaN.  Those NaNs must rank below every verified hypothesis, but
    cannot be passed into downstream rank/fusion code.  This function assigns
    each unverified row a finite value immediately below that query group's
    finite range and proves that its deterministic top-1 remains the frozen
    ``chosen_for_optional_pose`` row.
    """

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    splits = np.asarray(split_names).astype(str).reshape(-1)
    labels = np.asarray(evaluation_labels).astype(str).reshape(-1)
    indices = np.asarray(hypothesis_indices, dtype=np.int64).reshape(-1)
    raw = np.asarray(verification_log_likelihood_means, dtype=np.float64).reshape(-1)
    chosen = np.asarray(chosen_for_optional_pose, dtype=bool).reshape(-1)
    count = len(ids)
    if (
        count == 0
        or not (
            splits.shape
            == labels.shape
            == indices.shape
            == raw.shape
            == chosen.shape
            == (count,)
        )
        or np.any(ids == "")
        or np.any(splits == "")
        or np.any(labels == "")
        or len(set(zip(splits.tolist(), labels.tolist(), ids.tolist(), indices.tolist())))
        != count
        or np.any(np.isinf(raw))
    ):
        raise ValueError("native frozen baseline rows are invalid")

    scores = raw.copy()
    for split_name, label, query_id in sorted(
        set(zip(splits.tolist(), labels.tolist(), ids.tolist()))
    ):
        rows = np.flatnonzero(
            (splits == split_name) & (labels == label) & (ids == query_id)
        )
        finite = rows[np.isfinite(raw[rows])]
        chosen_rows = rows[chosen[rows]]
        if len(finite) == 0 or len(chosen_rows) != 1:
            raise ValueError("native frozen baseline lacks a unique verified selection")
        ranking = np.lexsort((indices[finite], -raw[finite]))
        winner = int(finite[int(ranking[0])])
        if int(chosen_rows[0]) != winner:
            raise ValueError("native frozen baseline does not reproduce chosen_for_optional_pose")
        floor = np.nextafter(float(np.min(raw[finite])), -np.inf)
        if not np.isfinite(floor):
            raise ValueError("native frozen baseline cannot create a finite unverified floor")
        scores[rows[~np.isfinite(raw[rows])]] = floor
    if not np.isfinite(scores).all():
        raise RuntimeError("native frozen baseline retains a non-finite score")
    return scores, chosen.copy()


def baseline_top1_for_scored_rows(
    *,
    query_ids: np.ndarray,
    split_names: np.ndarray,
    evaluation_labels: np.ndarray,
    hypothesis_indices: np.ndarray,
    baseline_selection_scores: np.ndarray,
) -> np.ndarray:
    """Select a deterministic baseline top-1 inside the rows actually scored."""

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    splits = np.asarray(split_names).astype(str).reshape(-1)
    labels = np.asarray(evaluation_labels).astype(str).reshape(-1)
    indices = np.asarray(hypothesis_indices, dtype=np.int64).reshape(-1)
    scores = np.asarray(baseline_selection_scores, dtype=np.float64).reshape(-1)
    count = len(ids)
    if (
        count == 0
        or not (splits.shape == labels.shape == indices.shape == scores.shape == (count,))
        or not np.isfinite(scores).all()
        or len(set(zip(splits.tolist(), labels.tolist(), ids.tolist(), indices.tolist())))
        != count
    ):
        raise ValueError("scored baseline rows are invalid")
    top1 = np.zeros((count,), dtype=bool)
    for split_name, label, query_id in sorted(
        set(zip(splits.tolist(), labels.tolist(), ids.tolist()))
    ):
        rows = np.flatnonzero(
            (splits == split_name) & (labels == label) & (ids == query_id)
        )
        ranking = np.lexsort((indices[rows], -scores[rows]))
        top1[int(rows[int(ranking[0])])] = True
    return top1


def ragged_offsets(lengths: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return canonical int64 offsets for non-negative ragged segment lengths."""

    values = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if np.any(values < 0):
        raise ValueError("ragged segment lengths must be non-negative")
    return np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(values, dtype=np.int64))
    )


def materialize_support_image_ids(
    *,
    support_image_indices: np.ndarray,
    support_view_valid: np.ndarray,
    cache_image_ids: np.ndarray,
) -> np.ndarray:
    """Materialize support provenance without indexing invalid ``-1`` entries."""

    indices = np.asarray(support_image_indices, dtype=np.int64)
    valid = np.asarray(support_view_valid, dtype=bool)
    image_ids = np.asarray(cache_image_ids).astype(str).reshape(-1)
    if (
        indices.ndim != 3
        or valid.shape != indices.shape
        or len(image_ids) == 0
        or len(set(image_ids.tolist())) != len(image_ids)
        or np.any(image_ids == "")
        or np.any((indices[valid] < 0) | (indices[valid] >= len(image_ids)))
    ):
        raise ValueError("support image provenance inputs are invalid")
    width = max(1, max(len(value) for value in image_ids.tolist()))
    output = np.full(indices.shape, "", dtype=f"<U{width}")
    output[valid] = image_ids[indices[valid]]
    return output


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _input_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": file_sha256_short(path)}
        for name, path in paths.items()
    }


def _checkpoint_input_manifest(
    paths: Mapping[str, Path],
    *,
    verification_points_scoring_compatibility: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {
        "verification_points": {
            "sha256": file_sha256_short(paths["mixed_verification_points_artifact"]),
            "scoring_compatibility": dict(verification_points_scoring_compatibility),
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
    return manifest


def validate_checkpoint_for_target_free_scoring(
    metadata: Mapping[str, object], *, expected_inputs: Mapping[str, Mapping[str, object]]
) -> None:
    """Reject target-bearing, stale, or semantically incomplete checkpoints."""

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
        "identity_candidate_mixture": "soft_observed_coordinate_posterior_detached_before_pose",
        "alignment_token_weights": "identity_max_conditional_posterior_detached_before_pose",
    }
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("format") != CHECKPOINT_FORMAT
        or metadata.get("model_format") != CANDIDATE_POSE_LATENT_EVIDENCE_FORMAT
        or any(metadata.get(key) != value for key, value in required.items())
    ):
        raise ValueError("latent evidence checkpoint violates the diagnostic contract")
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping) or set(expected_inputs) != set(_CHECKPOINT_INPUT_KEYS):
        raise ValueError("latent evidence checkpoint inputs are invalid")
    for name in _CHECKPOINT_INPUT_KEYS:
        observed = inputs.get(name)
        expected = expected_inputs[name]
        if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
            raise ValueError(f"latent evidence checkpoint is stale for {name}")
        if str(observed.get("sha256", "")) == str(expected.get("sha256", "")):
            continue
        if name != "verification_points":
            raise ValueError(f"latent evidence checkpoint is stale for {name}")
        checkpoint_compatibility = metadata.get(
            "verification_points_scoring_compatibility"
        )
        runtime_compatibility = expected.get("scoring_compatibility")
        if (
            not isinstance(checkpoint_compatibility, Mapping)
            or not isinstance(runtime_compatibility, Mapping)
            or dict(checkpoint_compatibility) != dict(runtime_compatibility)
        ):
            raise ValueError(
                "latent evidence checkpoint is stale for verification_points semantic compatibility"
            )


def _validate_checkpoint_hypothesis_semantic_lineage(
    *, checkpoint_metadata: Mapping[str, object], hypothesis_metadata: Mapping[str, object]
) -> None:
    expected = checkpoint_metadata.get("hypothesis_semantic_lineage")
    if not isinstance(expected, Mapping):
        raise ValueError("latent evidence checkpoint lacks hypothesis semantic lineage")
    validate_grouped_hypothesis_semantic_match(
        expected=expected, observed=grouped_hypothesis_semantic_manifest(hypothesis_metadata)
    )


def _load_checkpoint_model(
    *,
    path: Path,
    expected_inputs: Mapping[str, Mapping[str, object]],
    sources: Sequence[object],
    device: torch.device,
) -> tuple[CandidatePoseLatentEvidence, dict[str, object]]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch before weights_only.
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
        raise ValueError("latent evidence checkpoint is malformed")
    metadata = dict(payload["metadata"])
    validate_checkpoint_for_target_free_scoring(metadata, expected_inputs=expected_inputs)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("latent evidence checkpoint has no state dictionary")
    source_by_name = {str(source.name): source for source in sources}
    if set(source_by_name) != {"radio_final", "radio_intermediate", "alike"}:
        raise ValueError("latent evidence scoring sources are incomplete")
    reference = source_by_name["radio_final"]
    for source in source_by_name.values():
        if not np.array_equal(source.image_ids, reference.image_ids) or not np.array_equal(
            source.image_sizes, reference.image_sizes
        ):
            raise ValueError("latent evidence scoring cache ownership differs across scales")
    model = CandidatePoseLatentEvidence(
        sources={
            name: torch.from_numpy(np.asarray(source.grid, dtype=np.float32))
            for name, source in source_by_name.items()
        },
        image_sizes=torch.from_numpy(np.asarray(reference.image_sizes, dtype=np.float32)),
        hidden_dim=int(metadata.get("hidden_dim", 0)),
        max_abs_identity_residual=float(metadata.get("max_abs_identity_residual", 0.0)),
        max_abs_alignment_log_ratio=float(
            metadata.get("max_abs_alignment_log_ratio", 0.0)
        ),
        edge_chunk_size=int(metadata.get("edge_chunk_size", 0)),
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), metadata


def _assert_runtime_matches_formal_points(
    *, query: _QueryRuntime, formal: Mapping[str, np.ndarray]
) -> None:
    runtime = query.runtime
    if (
        runtime.candidate_probabilities.shape
        != torch.Size(np.asarray(formal["candidate_probabilities"]).shape)
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
        raise ValueError("latent evidence runtime differs from formal held-out points")


def _score_variant(
    *,
    model: CandidatePoseLatentEvidence,
    query: _QueryRuntime,
    observed_xy: np.ndarray,
    poses_w2c: np.ndarray,
    point_sources: np.ndarray,
    hypothesis_batch_size: int,
) -> dict[str, np.ndarray]:
    """Score one appearance variant with a pose-independent identity posterior."""

    poses = np.asarray(poses_w2c, dtype=np.float64)
    sources = np.asarray(point_sources).astype(str).reshape(-1)
    observed = np.asarray(observed_xy, dtype=np.float32)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or len(poses) == 0
        or observed.shape != (len(query.runtime.query_image_indices), 2)
        or len(sources) != len(observed)
        or int(hypothesis_batch_size) <= 0
        or not np.isfinite(poses).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("latent evidence hypothesis score inputs are invalid")
    source_names = np.asarray(sorted(set(sources.tolist())), dtype=np.str_)
    outputs: dict[str, list[np.ndarray]] = {
        "pose_log_likelihood_ratios": [],
        "point_log_likelihood_ratios": [],
        "point_geometric_candidate_counts": [],
        "point_geometric_view_masses": [],
        "source_log_likelihood_means": [],
        "source_effective_point_counts": [],
    }
    amp_enabled = model.device.type == "cuda"
    with torch.no_grad():
        identity = model.identity_posterior(
            runtime=query.runtime,
            observed_xy=torch.from_numpy(observed).to(device=model.device),
        )
        static_candidate = identity.candidate_probabilities.detach()
        static_null = identity.null_probabilities.detach()
        selector = identity.selector_weights.detach()
        raw_candidate = query.runtime.candidate_probabilities.to(
            device=model.device, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(3)
        view_weights = query.runtime.candidate_view_weights.to(
            device=model.device, dtype=torch.float32
        ).unsqueeze(0)
        for begin in range(0, len(poses), int(hypothesis_batch_size)):
            end = min(begin + int(hypothesis_batch_size), len(poses))
            projected_xy, projected_valid = _project_candidate_positions(
                query=query,
                poses_w2c=torch.from_numpy(poses[begin:end]),
                device=model.device,
            )
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                score = model.pose_log_likelihood_ratios(
                    runtime=query.runtime,
                    candidate_query_xy=projected_xy,
                    candidate_projection_valid=projected_valid,
                    selector_weights=selector,
                    static_candidate_probabilities=static_candidate,
                    static_null_probabilities=static_null,
                )
            usable_candidate = score.edge_usable.any(dim=3) & (raw_candidate[..., 0] > 0.0)
            geometric_mass = (
                score.edge_usable.to(dtype=torch.float32) * raw_candidate * view_weights
            ).sum(dim=(2, 3))
            point = score.point_log_likelihood_ratios.detach().cpu().numpy().astype(np.float32)
            effective = usable_candidate.sum(dim=2).detach().cpu().numpy().astype(np.int16)
            outputs["pose_log_likelihood_ratios"].append(
                score.pose_log_likelihood_ratios.detach().cpu().numpy().astype(np.float32)
            )
            outputs["point_log_likelihood_ratios"].append(point)
            outputs["point_geometric_candidate_counts"].append(effective)
            outputs["point_geometric_view_masses"].append(
                geometric_mass.detach().cpu().numpy().astype(np.float32)
            )
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
    result.update(
        {
            "source_names": source_names,
            "identity_candidate_probabilities": identity.candidate_probabilities.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "identity_conditional_probabilities": identity.conditional_probabilities.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "identity_candidate_residual": identity.candidate_residual.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "identity_selector_weights": identity.selector_weights.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "identity_edge_usable": identity.edge_usable.detach().cpu().numpy().astype(bool),
        }
    )
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


def validate_target_free_score_metadata(metadata: Mapping[str, object]) -> None:
    required = {
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "paired_visual_control": True,
    }
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("format") != SCORE_FORMAT
        or any(metadata.get(key) is not value for key, value in required.items())
    ):
        raise ValueError("latent evidence score metadata violates the target-free contract")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.time()
    variants = parse_evidence_variants(str(args.evidence_variants))
    score_splits = parse_score_splits(str(args.score_splits))
    if (
        int(args.support_view_count) != 2
        or int(args.hypothesis_batch_size) <= 0
        or int(args.hypothesis_limit) < 0
        or int(args.edge_chunk_size) < 0
    ):
        raise ValueError("latent evidence scorer requires two support views and positive batches")
    if (
        str(args.baseline_score_artifact).strip()
        or str(args.baseline_reference_hypothesis_artifact).strip()
    ):
        raise ValueError(
            "current v10 grouped hypotheses carry their immutable baseline; "
            "do not supply legacy baseline-score arguments"
        )
    output_path = Path(args.output)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"output already exists: {output_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("latent evidence scorer requested CUDA but it is unavailable")
    paths: dict[str, Path] = {
        "hypothesis_artifact": Path(args.hypothesis_artifact),
        "detector_query_cache": Path(args.detector_query_cache),
        "proposals": Path(args.proposals),
        "candidate_artifact": Path(args.candidate_artifact),
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
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"latent evidence input is missing: {name} ({path})")
    exact, hypothesis_metadata = load_inference_artifact_fields(
        paths["hypothesis_artifact"], ("poses_w2c",)
    )
    native_baseline_scores, native_baseline_top1 = native_frozen_hypothesis_baseline(
        query_ids=np.asarray(exact["query_ids"]),
        split_names=np.asarray(exact["split_names"]),
        evaluation_labels=np.asarray(exact["evaluation_labels"]),
        hypothesis_indices=np.asarray(exact["hypothesis_indices"], dtype=np.int64),
        verification_log_likelihood_means=np.asarray(
            exact["verification_log_likelihood_means"], dtype=np.float64
        ),
        chosen_for_optional_pose=np.asarray(
            exact["chosen_for_optional_pose"], dtype=bool
        ),
    )
    query_groups = select_scored_query_groups(
        query_ids=np.asarray(exact["query_ids"]),
        split_names=np.asarray(exact["split_names"]),
        score_splits=score_splits,
        hypothesis_limit=int(args.hypothesis_limit),
    )

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
        raise ValueError("latent evidence source artifacts violate the target-free contract")
    points = load_mixed_verification_points(paths["mixed_verification_points_artifact"])
    verification_points_scoring_compatibility = validate_heldout_scoring_point_cache(
        metadata=points.metadata,
        point_split_names=points.split_names,
        requested_splits=score_splits,
    )
    sources = load_context_attention_sources(
        radio_final_context_cache=paths["radio_final_context_cache"],
        radio_intermediate_context_cache=paths["radio_intermediate_context_cache"],
        alike_spatial_context_cache=paths["alike_spatial_context_cache"],
        expected_radio_checkpoint="",
        require_equal_descriptor_dimensions=False,
    )
    runtimes_by_split = {
        split_name: _prepare_query_runtimes(
            points=points,
            sources=sources,
            maplet_support_index=paths["maplet_support_index"],
            support_geometry_index=paths["support_geometry_index"],
            projected_landmark_bank=paths["projected_landmark_bank"],
            colmap_model_dir=Path(args.colmap_model_dir),
            support_view_count=int(args.support_view_count),
            required_split=split_name,
        )
        for split_name in sorted({split_name for _query_id, split_name, _rows in query_groups})
    }
    checkpoint_inputs = _checkpoint_input_manifest(
        paths,
        verification_points_scoring_compatibility=verification_points_scoring_compatibility,
    )
    model, checkpoint_metadata = _load_checkpoint_model(
        path=paths["checkpoint"],
        expected_inputs=checkpoint_inputs,
        sources=sources,
        device=device,
    )
    _validate_checkpoint_hypothesis_semantic_lineage(
        checkpoint_metadata=checkpoint_metadata,
        hypothesis_metadata=hypothesis_metadata,
    )
    model.edge_chunk_size = resolve_edge_chunk_size(
        checkpoint_edge_chunk_size=int(checkpoint_metadata.get("edge_chunk_size", 0)),
        requested_edge_chunk_size=int(args.edge_chunk_size),
    )
    reference_source = next(source for source in sources if str(source.name) == "radio_final")
    descriptor_derangement = deterministic_support_descriptor_derangement(
        image_ids=np.asarray(reference_source.image_ids).astype(str),
        image_sizes=np.asarray(reference_source.image_sizes, dtype=np.int64),
    )
    row_parts: dict[str, list[np.ndarray]] = {
        "query_ids": [],
        "split_names": [],
        "evaluation_labels": [],
        "hypothesis_indices": [],
        "source_chosen_for_optional_pose": [],
        "baseline_selection_scores": [],
    }
    static_parts: dict[str, list[np.ndarray]] = {
        "verification_source_point_ids": [],
        "verification_point_sources": [],
        "verification_source_detector_rows": [],
        "verification_xy": [],
        "candidate_track_ids": [],
        "candidate_prior_probabilities": [],
        "null_probabilities": [],
        "candidate_view_weights": [],
        "candidate_support_image_ids": [],
    }
    row_statistic_fields = (
        "pose_log_likelihood_ratios",
        "source_log_likelihood_means",
        "source_effective_point_counts",
    )
    point_statistic_fields = (
        "point_log_likelihood_ratios",
        "point_geometric_candidate_counts",
        "point_geometric_view_masses",
    )
    identity_static_fields = (
        "identity_candidate_probabilities",
        "identity_conditional_probabilities",
        "identity_candidate_residual",
        "identity_selector_weights",
        "identity_edge_usable",
    )
    statistic_parts = {
        prefix: {
            field: []
            for field in (*row_statistic_fields, *point_statistic_fields, *identity_static_fields)
        }
        for prefix in ("visual", "control")
    }
    verification_query_ids: list[str] = []
    verification_split_names: list[str] = []
    verification_point_lengths: list[int] = []
    hypothesis_point_lengths: list[int] = []
    verification_static_digests: list[str] = []
    verification_point_selection: list[dict[str, object]] = []
    source_names: np.ndarray | None = None
    for query_id, query_split, row_indices in query_groups:
        rows = np.asarray(row_indices, dtype=np.int64)
        if (
            len(rows) == 0
            or np.any(np.asarray(exact["query_ids"]).astype(str)[rows] != query_id)
            or np.any(np.asarray(exact["split_names"]).astype(str)[rows] != query_split)
        ):
            raise RuntimeError("scored frozen query group has inconsistent row ownership")
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
            raise ValueError("latent evidence mixed points have an invalid format")
        source_rows = points.rows_for_query(query_id)
        formal_fields = {
            "xy": points.xy,
            "candidate_track_ids": points.candidate_track_ids,
            "candidate_probabilities": points.candidate_prior_probabilities,
            "null_probabilities": points.null_probabilities,
        }
        if any(
            not np.allclose(
                np.asarray(values)[source_rows], np.asarray(formal[name]), atol=1e-6
            )
            for name, values in formal_fields.items()
        ):
            raise ValueError("latent evidence formal points do not preserve source order")
        query = runtimes_by_split[query_split].get(query_id)
        if query is None:
            raise ValueError("latent evidence query has no held-out runtime")
        _assert_runtime_matches_formal_points(query=query, formal=formal)
        variant_statistics: dict[str, dict[str, np.ndarray]] = {}
        for variant in variants:
            active_query = query
            if variant == "support_descriptor_permutation_control":
                remapped_runtime = remap_support_image_indices(
                    runtime=query.runtime,
                    image_ids=np.asarray(reference_source.image_ids).astype(str),
                    descriptor_derangement=descriptor_derangement,
                )
                active_query = _QueryRuntime(
                    runtime=remapped_runtime,
                    candidate_xyz=query.candidate_xyz,
                    focal_length=query.focal_length,
                    principal_x=query.principal_x,
                    principal_y=query.principal_y,
                    radial_k=query.radial_k,
                    image_width=query.image_width,
                    image_height=query.image_height,
                )
            variant_statistics[variant] = _score_variant(
                model=model,
                query=active_query,
                observed_xy=np.asarray(formal["xy"], dtype=np.float32),
                poses_w2c=np.asarray(exact["poses_w2c"], dtype=np.float64)[rows],
                point_sources=np.asarray(formal["point_sources"]).astype(str),
                hypothesis_batch_size=int(args.hypothesis_batch_size),
            )
        visual = variant_statistics["visual"]
        control = variant_statistics["support_descriptor_permutation_control"]
        for field in (
            "point_geometric_candidate_counts",
            "point_geometric_view_masses",
            "identity_edge_usable",
        ):
            if not np.array_equal(np.asarray(visual[field]), np.asarray(control[field])):
                raise RuntimeError(f"latent visual/control geometry changed for {field}")
        point_count = int(len(np.asarray(formal["source_point_ids"])))
        hypothesis_count = int(len(rows))
        if point_count <= 0 or hypothesis_count <= 0:
            raise RuntimeError("latent evidence query group is empty")
        current_source_names = np.asarray(visual["source_names"]).astype(str)
        if source_names is None:
            source_names = current_source_names
        elif not np.array_equal(source_names, current_source_names):
            raise ValueError("latent evidence query groups expose different source names")
        if not np.array_equal(current_source_names, np.asarray(control["source_names"]).astype(str)):
            raise RuntimeError("latent visual/control source names differ")
        support_image_ids = materialize_support_image_ids(
            support_image_indices=query.runtime.support_image_indices.numpy(),
            support_view_valid=query.runtime.support_view_valid.numpy(),
            cache_image_ids=np.asarray(reference_source.image_ids),
        )
        row_parts["query_ids"].append(np.asarray(exact["query_ids"])[rows].astype(str))
        row_parts["split_names"].append(np.asarray(exact["split_names"])[rows].astype(str))
        row_parts["evaluation_labels"].append(
            np.asarray(exact["evaluation_labels"])[rows].astype(str)
        )
        row_parts["hypothesis_indices"].append(
            np.asarray(exact["hypothesis_indices"], dtype=np.int64)[rows]
        )
        row_parts["source_chosen_for_optional_pose"].append(
            np.asarray(exact["chosen_for_optional_pose"], dtype=bool)[rows]
        )
        row_parts["baseline_selection_scores"].append(native_baseline_scores[rows])
        static_parts["verification_source_point_ids"].append(
            np.asarray(formal["source_point_ids"], dtype=np.int64)
        )
        static_parts["verification_point_sources"].append(
            np.asarray(formal["point_sources"]).astype(str)
        )
        static_parts["verification_source_detector_rows"].append(
            np.asarray(formal["source_detector_rows"], dtype=np.int64)
        )
        static_parts["verification_xy"].append(np.asarray(formal["xy"], dtype=np.float32))
        static_parts["candidate_track_ids"].append(
            np.asarray(formal["candidate_track_ids"], dtype=np.int64)
        )
        static_parts["candidate_prior_probabilities"].append(
            np.asarray(formal["candidate_probabilities"], dtype=np.float32)
        )
        static_parts["null_probabilities"].append(
            np.asarray(formal["null_probabilities"], dtype=np.float32)
        )
        static_parts["candidate_view_weights"].append(
            query.runtime.candidate_view_weights.numpy()
        )
        static_parts["candidate_support_image_ids"].append(support_image_ids)
        for variant, prefix in (
            ("visual", "visual"),
            ("support_descriptor_permutation_control", "control"),
        ):
            statistics = variant_statistics[variant]
            for field in row_statistic_fields:
                value = np.asarray(statistics[field])
                if value.shape[0] != hypothesis_count:
                    raise RuntimeError(f"latent row statistic has invalid query extent: {field}")
                statistic_parts[prefix][field].append(value)
            for field in point_statistic_fields:
                value = np.asarray(statistics[field])
                if value.shape != (hypothesis_count, point_count):
                    raise RuntimeError(f"latent point statistic has invalid query extent: {field}")
                statistic_parts[prefix][field].append(value.reshape(-1))
            for field in identity_static_fields:
                value = np.asarray(statistics[field])
                if value.shape[0] != point_count:
                    raise RuntimeError(f"latent identity statistic has invalid query extent: {field}")
                statistic_parts[prefix][field].append(value)
        verification_query_ids.append(query_id)
        verification_split_names.append(query_split)
        verification_point_lengths.append(point_count)
        hypothesis_point_lengths.extend([point_count] * hypothesis_count)
        verification_static_digests.append(_source_static_digest(formal=formal, runtime=query.runtime))
        verification_point_selection.append(
            {
                "query_id": query_id,
                "split_name": query_split,
                **point_selection,
            }
        )
    if source_names is None:
        raise RuntimeError("latent evidence scorer did not materialize a source layout")
    arrays: dict[str, np.ndarray] = {
        name: np.concatenate(parts, axis=0) for name, parts in row_parts.items()
    }
    arrays["baseline_score_top1"] = baseline_top1_for_scored_rows(
        query_ids=arrays["query_ids"],
        split_names=arrays["split_names"],
        evaluation_labels=arrays["evaluation_labels"],
        hypothesis_indices=arrays["hypothesis_indices"],
        baseline_selection_scores=arrays["baseline_selection_scores"],
    )
    if int(args.hypothesis_limit) == 0 and not np.array_equal(
        arrays["baseline_score_top1"], arrays["source_chosen_for_optional_pose"]
    ):
        raise RuntimeError("full frozen baseline top-1 differs from source chosen rows")
    arrays["source_names"] = source_names
    arrays["verification_query_ids"] = np.asarray(verification_query_ids, dtype=np.str_)
    arrays["verification_split_names"] = np.asarray(verification_split_names, dtype=np.str_)
    arrays["verification_offsets"] = ragged_offsets(verification_point_lengths)
    arrays["hypothesis_verification_offsets"] = ragged_offsets(hypothesis_point_lengths)
    arrays["verification_static_digests"] = np.asarray(
        verification_static_digests, dtype=np.str_
    )
    for name, parts in static_parts.items():
        arrays[name] = np.concatenate(parts, axis=0)
    prefixes = {
        "visual": "visual",
        "support_descriptor_permutation_control": "control",
    }
    for variant, prefix in prefixes.items():
        for field, parts in statistic_parts[prefix].items():
            arrays[f"{prefix}_{field}"] = np.concatenate(parts, axis=0)
    metadata: dict[str, Any] = {
        "format": SCORE_FORMAT,
        "version": SCORE_VERSION,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "paired_visual_control": True,
        "evidence_variants": list(variants),
        "row_count": int(len(arrays["query_ids"])),
        "query_count": int(len(query_groups)),
        "score_splits": list(score_splits),
        "scored_query_groups": [
            {
                "query_id": query_id,
                "split_name": split_name,
                "hypothesis_count": int(len(rows)),
            }
            for query_id, split_name, rows in query_groups
        ],
        "model_checkpoint": {
            "path": str(paths["checkpoint"]),
            "sha256": file_sha256_short(paths["checkpoint"]),
        },
        "model_checkpoint_contract": {
            key: checkpoint_metadata.get(key)
            for key in (
                "architecture",
                "hidden_dim",
                "max_abs_identity_residual",
                "max_abs_alignment_log_ratio",
                "edge_chunk_size",
            )
        },
        "inference_edge_chunk_size": int(model.edge_chunk_size),
        "strict_candidate_pose_latent_evidence_contract": {
            "heldout_query_rows": True,
            "formal_p1_mixed_multiscale_points": True,
            "fixed_global_topl": True,
            "fixed_candidate_top_k": 20,
            "identity_candidate_posterior_soft_topl": True,
            "identity_evaluated_before_pose": True,
            "identity_posterior_fixed_across_hypotheses": True,
            "candidate_reselection_per_pose": False,
            "support_reselection_per_pose": False,
            "fixed_support_view_count": 2,
            "explicit_null": True,
            "candidate_projection_is_only_pose_dependent_encoder_input": True,
            "candidate_pose_matrix_excluded_from_encoder": True,
            "residual_and_target_excluded_from_encoder": True,
            "support_descriptor_permutation_control": True,
            "ragged_query_point_layout": True,
            "no_pnp": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
        "hypothesis_scope": {
            "all_frozen_hypotheses": int(args.hypothesis_limit) == 0,
            "development_prefix_limit_per_query": int(args.hypothesis_limit),
            "scored_hypothesis_count": int(len(arrays["query_ids"])),
        },
        "raw_score_semantics": "train_only_identity_then_correct_vs_coherent_wrong_visual_alignment_under_static_soft_candidate_null_mixture",
        "raw_score_is_calibrated_independent_pose_likelihood": False,
        "baseline_source": "native_v10_verification_log_likelihood_with_unverified_floor_v1",
        "baseline_top1_source_equivalent": bool(int(args.hypothesis_limit) == 0),
        "verification_layout": "query_and_hypothesis_ragged_offsets_v1",
        "verification_point_selection": verification_point_selection,
        "support_descriptor_derangement": _canonical_hash(
            dict(sorted(descriptor_derangement.items()))
        ),
        "inputs": _input_manifest(paths),
        "source_metadata_hashes": {
            "hypothesis": _canonical_hash(hypothesis_metadata),
            "detector": _canonical_hash(detector_metadata),
            "proposals": _canonical_hash(proposal_metadata),
            "candidate": _canonical_hash(candidate_metadata),
            "mixed_verification_points": _canonical_hash(mixed_metadata),
        },
        "verification_points_scoring_compatibility": verification_points_scoring_compatibility,
        "elapsed_seconds": float(time.time() - start),
    }
    validate_target_free_score_metadata(metadata)
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
                "stage": "score_candidate_pose_latent_evidence",
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
