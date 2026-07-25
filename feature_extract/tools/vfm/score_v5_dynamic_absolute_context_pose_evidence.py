"""Probe frozen pose hypotheses with V5 candidate-specific query projections.

This diagnostic is deliberately narrower than a production pose selector.  It
keeps the global top-20 candidates, their explicit null mass, the first two
maplet support observations, and a fixed held-out point set fixed.  The point
set is either an ALIKE-only P1.5 diagnostic or the formal P1 mixed set of 64
ALIKE, 64 RADIO-intermediate, and 64 RADIO-final points.  A hypothesis only
changes the query image position at which each fixed candidate is scored.

Neither V5 output head was trained as a correct-pose versus
coherent-wrong-pose likelihood ratio.  Consequently this script writes
*uncalibrated raw-score diagnostics* only.  It cannot load pose targets,
change a PnP overlay, or be promoted without a later calibration and separate
target-side gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    FixedCandidateViews,
    _canonical_hash,
    _fixed_candidate_views,
    _input_manifest,
    _load_bank_xyz,
    _load_exact_hypotheses,
    _load_maplet_support_fields,
    _load_npz_allowlist,
    _resolve_rows,
    _select_s0_verification_rows,
    _spatial_2x2_statistic,
)
from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _load_candidate_prior_overlay,
    _mask_fixed_candidate_posterior_topk,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES,
    CandidateBidirectionalAbsoluteContextLikelihood,
    build_fixed_candidate_context_runtime,
    load_context_attention_sources,
)
from feature_extract.vfm.localization.frozen_multiscale_pose_evidence import (
    fixed_candidate_point_log_ratios,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    POINT_SOURCE_ALIKE,
    POINT_SOURCE_RADIO_FINAL,
    POINT_SOURCE_RADIO_INTERMEDIATE,
    MixedVerificationPoints,
    load_mixed_verification_points,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    project_simple_radial_torch,
)


SCORE_FORMAT = "v5_dynamic_absolute_context_pose_scores_v1"
SCORE_VERSION = "p1_5_v5_geometry_raw_score_with_candidate_projected_query_crops_v1"
P1_MIXED_SCORE_VERSION = "p1_v5_geometry_raw_score_with_candidate_projected_query_crops_v1"
STRICT_IDENTITY_SCORE_VERSION = (
    "p1_5_v5_strict_identity_raw_score_with_candidate_projected_query_crops_v1"
)
P1_MIXED_STRICT_IDENTITY_SCORE_VERSION = (
    "p1_v5_strict_identity_raw_score_with_candidate_projected_query_crops_v1"
)
MODEL_FORMAT = "multiscale_context_attention_candidate_probe_v1"
V5_ARCHITECTURE = "bidirectional_absolute_dual_head_raw_layout_v5"
FIXED_SUPPORT_VIEW_COUNT = 2
SCORE_NAMES = ("all_scales", "radio_final", "radio_intermediate", "alike")
EVIDENCE_BRANCH_GEOMETRY = "geometry"
EVIDENCE_BRANCH_STRICT_IDENTITY = "strict_identity"
EVIDENCE_BRANCHES = (
    EVIDENCE_BRANCH_GEOMETRY,
    EVIDENCE_BRANCH_STRICT_IDENTITY,
)
_FORMAL_P1_POINT_COUNTS = {
    POINT_SOURCE_ALIKE: 64,
    POINT_SOURCE_RADIO_INTERMEDIATE: 64,
    POINT_SOURCE_RADIO_FINAL: 64,
}


def _array_digest(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(np.asarray(values[name]))
        digest.update(str(name).encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()[:16]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis_artifact", required=True)
    parser.add_argument("--baseline_score_artifact", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--fixed_candidate_prior_overlay", required=True)
    parser.add_argument(
        "--mixed_verification_points_artifact",
        default=None,
        help=(
            "formal P1 target-free 64 ALIKE + 64 RADIO-intermediate + 64 RADIO-final "
            "held-out point artifact; omitting it runs the ALIKE-only P1.5 diagnostic"
        ),
    )
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--radio_intermediate_context_cache", required=True)
    parser.add_argument("--alike_spatial_context_cache", required=True)
    parser.add_argument("--context_contract", required=True)
    parser.add_argument("--context_checkpoint", required=True)
    parser.add_argument(
        "--family",
        choices=(
            "bidirectional_absolute_dual_head_raw_layout_visual_v5",
            "bidirectional_absolute_dual_head_raw_layout_position_control_v5",
        ),
        required=True,
    )
    parser.add_argument(
        "--evidence_branch",
        choices=EVIDENCE_BRANCHES,
        default=EVIDENCE_BRANCH_GEOMETRY,
        help=(
            "which separately trained V5 raw head to probe; branches are never mixed "
            "before train-only likelihood calibration"
        ),
    )
    parser.add_argument("--fixed_candidate_top_k", type=int, default=20)
    parser.add_argument("--verification_point_count", type=int, default=192)
    parser.add_argument("--detector_log_merit_weight", type=float, default=0.01)
    parser.add_argument("--hypothesis_batch_size", type=int, default=4)
    parser.add_argument("--query_point_batch_size", type=int, default=8)
    parser.add_argument(
        "--dynamic_scoring_mode",
        choices=("discrete_lookup", "direct"),
        default="discrete_lookup",
        help="exact feature-grid lookup is the scalable default; direct is a parity diagnostic",
    )
    parser.add_argument("--lookup_point_batch_size", type=int, default=32)
    parser.add_argument("--lookup_anchor_batch_size", type=int, default=2)
    parser.add_argument(
        "--hypothesis_limit",
        type=int,
        default=0,
        help="development-only stable hypothesis prefix; zero evaluates every frozen row",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_contract(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "multiscale_context_attention_candidate_probe_contract_v5"
        or payload.get("architecture") != V5_ARCHITECTURE
        or payload.get("contains_ground_truth") is not False
        or payload.get("contains_target_errors") is not False
        or payload.get("pose_or_ground_truth_used") is not False
        or payload.get("image_retrieval_or_submap_used") is not False
        or payload.get("render") is not False
        or int(payload.get("candidate_top_k", -1)) != 20
        or int(payload.get("support_view_count", -1)) != FIXED_SUPPORT_VIEW_COUNT
    ):
        raise ValueError("context contract is not the immutable target-free V5 contract")
    families = tuple(str(value) for value in payload.get("families", ()))
    if set(families) != {
        "bidirectional_absolute_dual_head_raw_layout_visual_v5",
        "bidirectional_absolute_dual_head_raw_layout_position_control_v5",
    }:
        raise ValueError("context contract has an unexpected V5 family profile")
    return payload


def _validate_sources_against_contract(
    *, contract: Mapping[str, object], sources: Sequence[object]
) -> None:
    declared = {
        str(item.get("name")): item
        for item in contract.get("source_scales", [])
        if isinstance(item, Mapping)
    }
    if set(declared) != {scale.name for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES}:
        raise ValueError("context contract does not declare the complete source profile")
    for source in sources:
        item = declared.get(str(source.name))
        if not isinstance(item, Mapping):
            raise ValueError("context source is absent from the contract")
        if (
            file_sha256_short(source.path) != str(item.get("sha256", ""))
            or int(source.spatial_grid_size) != int(item.get("grid_size", -1))
            or int(source.descriptor_dim) != int(item.get("descriptor_dim", -1))
            or source.metadata.get("source_image_manifest_sha256")
            != item.get("source_image_manifest_sha256")
        ):
            raise ValueError("context source differs from its V5 contract lineage")


def _load_v5_model(
    *,
    checkpoint_path: Path,
    contract_path: Path,
    contract: Mapping[str, object],
    family: str,
    sources: Sequence[object],
    runtime: object,
    query_xy: np.ndarray,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    device: torch.device,
) -> tuple[CandidateBidirectionalAbsoluteContextLikelihood, dict[str, object]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("context checkpoint is malformed")
    metadata = checkpoint.get("metadata")
    state = checkpoint.get("state_dict")
    if (
        checkpoint.get("format") != MODEL_FORMAT
        or str(checkpoint.get("family")) != str(family)
        or not isinstance(metadata, Mapping)
        or not isinstance(state, Mapping)
        or str(metadata.get("architecture_id")) != V5_ARCHITECTURE
        or str(metadata.get("contract_sha256")) != file_sha256_short(contract_path)
        or str(metadata.get("family_profile")) != V5_ARCHITECTURE
        or metadata.get("candidate_log_likelihood_ratio_is_independent_pose_likelihood")
        is not False
        or metadata.get("identity_candidate_probability_allowed_for_pnp_overlay") is not False
    ):
        raise ValueError("context checkpoint cannot be used as the declared V5 diagnostic")
    source_tensors = {source.name: torch.from_numpy(np.asarray(source.grid)) for source in sources}
    model = CandidateBidirectionalAbsoluteContextLikelihood(
        family=str(family),
        sources=source_tensors,
        image_sizes=torch.from_numpy(np.asarray(sources[0].image_sizes, dtype=np.float32)),
        runtime=runtime,
        query_xy=np.asarray(query_xy, dtype=np.float32),
        base_candidate_probabilities=np.asarray(candidate_probabilities, dtype=np.float32),
        base_null_probabilities=np.asarray(null_probabilities, dtype=np.float32),
        hidden_dim=int(metadata.get("hidden_dim", 0)),
        heads=int(metadata.get("heads", 0)),
        dropout=float(metadata.get("dropout", 0.0)),
    )
    missing, unexpected = model.load_state_dict(dict(state), strict=False)
    if missing or unexpected:
        raise ValueError(
            "V5 checkpoint model state differs from its declared architecture: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device).eval()
    return model, dict(metadata)


def _first_fixed_support_views(
    views: FixedCandidateViews, *, count: int
) -> FixedCandidateViews:
    """Preserve the V5 train-time maplet ordering, never average support8."""

    if int(count) <= 0 or int(count) > int(views.valid.shape[2]):
        raise ValueError("requested fixed support-view prefix is invalid")
    image_ids = np.asarray(views.support_image_ids)[:, :, : int(count)].copy()
    valid = np.asarray(views.valid, dtype=bool)[:, :, : int(count)].copy()
    weights = np.asarray(views.weights, dtype=np.float32)[:, :, : int(count)].copy()
    normalizer = weights.sum(axis=2, keepdims=True)
    weights = np.divide(weights, normalizer, out=np.zeros_like(weights), where=normalizer > 0.0)
    required = np.asarray(views.weights, dtype=np.float32).sum(axis=2) > 0.0
    if np.any(required & ~np.any(valid, axis=2)):
        raise ValueError("the first fixed support views omit a positive-mass candidate")
    return FixedCandidateViews(support_image_ids=image_ids, valid=valid, weights=weights)


def _formal_p1_mixed_evidence_for_query(
    *,
    points_path: Path,
    query_id: str,
    query_split: str,
    detector_path: Path,
    candidate_path: Path,
    landmark_bank_path: Path,
    detector: Mapping[str, np.ndarray],
    proposals: Mapping[str, np.ndarray],
    selected_rows: np.ndarray,
    point_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    """Load one formal P1 point set and prove its held-out provenance.

    The mixed artifact owns both the query points and their global top-20
    candidates.  It is intentionally independent from S0's fit candidates,
    except for the explicit check that its ALIKE detector rows were not used
    to generate frozen hypotheses.  Dense RADIO lattice points have no
    detector-row identity and are instead checked through their fixed source
    labels and target-free artifact lineage.
    """

    if int(point_count) != sum(_FORMAL_P1_POINT_COUNTS.values()):
        raise ValueError("formal P1 requires exactly 64 points from each of three sources")
    points: MixedVerificationPoints = load_mixed_verification_points(Path(points_path))
    metadata = dict(points.metadata)
    if (
        metadata.get("format") != MIXED_VERIFICATION_POINTS_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("render") is not False
        or metadata.get("candidate_reselection") is not False
        or metadata.get("candidate_set")
        != "fixed_full_global_faiss_top_l_unique_tracks"
        or metadata.get("global_landmark_ann_scope") != "full_projected_landmark_bank_only"
        or int(metadata.get("candidate_top_k", -1)) != 20
        or str(metadata.get("candidate_fit_artifact_sha256", ""))
        != file_sha256_short(Path(candidate_path))
        or str(metadata.get("detector_query_cache_sha256", ""))
        != file_sha256_short(Path(detector_path))
        or str(metadata.get("projected_landmark_bank_sha256", ""))
        != file_sha256_short(Path(landmark_bank_path))
    ):
        raise ValueError("formal P1 mixed verification artifact has incompatible target-free lineage")
    # Development validation must not even materialize test-image point rows.
    # Although those rows contain no targets, retaining them in the same cache
    # makes it too easy to turn a source-only test leak into accidental model
    # selection.  A later frozen test run may use a separate test-only cache.
    exported_splits = {str(value) for value in metadata.get("exported_splits", ())}
    if str(query_split) == "validation" and (
        "test" in exported_splits
        or metadata.get("test_source_points_materialized") is not False
    ):
        raise ValueError(
            "formal P1 validation requires a mixed-point cache with no materialized test sources"
        )
    rows = points.rows_for_query(str(query_id))
    if len(rows) != int(point_count) or not np.all(points.split_names[rows] == str(query_split)):
        raise ValueError("formal P1 mixed point rows differ from the frozen query/split")
    point_sources = np.asarray(points.point_sources, dtype=np.str_)[rows]
    observed_counts = {
        source: int(np.count_nonzero(point_sources == source))
        for source in sorted(set(point_sources.tolist()))
    }
    if observed_counts != _FORMAL_P1_POINT_COUNTS:
        raise ValueError("formal P1 mixed point sources are not the required 64/64/64 layout")
    source_point_ids = np.asarray(points.source_point_ids, dtype=np.int64)[rows]
    if len(np.unique(source_point_ids)) != len(source_point_ids):
        raise ValueError("formal P1 mixed verifier repeats source point IDs for one query")

    detector_ids = np.asarray(detector["image_ids"]).astype(str).reshape(-1)
    offsets = np.asarray(detector["offsets"], dtype=np.int64).reshape(-1)
    detector_position = np.flatnonzero(detector_ids == str(query_id))
    if len(detector_position) != 1 or offsets.shape != (len(detector_ids) + 1,):
        raise ValueError("formal P1 detector ownership is invalid")
    position = int(detector_position[0])
    query_detector_rows = np.arange(int(offsets[position]), int(offsets[position + 1]), dtype=np.int64)
    source_detector_rows = np.asarray(points.source_detector_rows, dtype=np.int64)[rows]
    alike = point_sources == POINT_SOURCE_ALIKE
    if (
        np.any((source_detector_rows >= 0) != alike)
        or np.any(~np.isin(source_detector_rows[alike], query_detector_rows))
        or len(np.unique(source_detector_rows[alike])) != int(np.sum(alike))
    ):
        raise ValueError("formal P1 ALIKE rows do not have valid per-query detector provenance")
    proposal_queries = np.asarray(proposals["query_ids"]).astype(str).reshape(-1)
    if np.any(proposal_queries[source_detector_rows[alike]] != str(query_id)):
        raise ValueError("formal P1 ALIKE rows disagree with proposal query ownership")
    fit_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    fit_rows = fit_rows[proposal_queries[fit_rows] == str(query_id)]
    if np.any(np.isin(source_detector_rows[alike], fit_rows)):
        raise ValueError("formal P1 ALIKE verifier reuses a frozen-hypothesis fit row")

    arrays = {
        "source_point_ids": source_point_ids,
        "xy": np.asarray(points.xy, dtype=np.float32)[rows],
        "point_sources": point_sources,
        "source_detector_rows": source_detector_rows,
        "candidate_bank_rows": np.asarray(points.candidate_bank_rows, dtype=np.int64)[rows],
        "candidate_track_ids": np.asarray(points.candidate_track_ids, dtype=np.int64)[rows],
        "candidate_probabilities": np.asarray(
            points.candidate_prior_probabilities, dtype=np.float32
        )[rows],
        "null_probabilities": np.asarray(points.null_probabilities, dtype=np.float32)[rows],
    }
    selection = {
        "source": "formal_p1_mixed_multiscale_verification_points_v1",
        "point_count": int(len(rows)),
        "point_sources": observed_counts,
        "source_point_ids_sha256": _array_digest({"source_point_ids": source_point_ids}),
        "candidate_fit_rows_excluded_from_alike": True,
        "fit_query_point_count": int(len(fit_rows)),
        "available_unused_query_point_count": int(
            len(np.setdiff1d(query_detector_rows, fit_rows, assume_unique=True))
        ),
        "alike_detector_point_count": int(np.sum(alike)),
        "radio_intermediate_lattice_point_count": int(
            np.count_nonzero(point_sources == POINT_SOURCE_RADIO_INTERMEDIATE)
        ),
        "radio_final_lattice_point_count": int(
            np.count_nonzero(point_sources == POINT_SOURCE_RADIO_FINAL)
        ),
        "dense_point_detector_rows_are_sentinel_minus_one": bool(
            np.all(source_detector_rows[~alike] == -1)
        ),
        "development_test_source_points_excluded": bool(
            str(query_split) != "validation"
            or ("test" not in exported_splits and metadata.get("test_source_points_materialized") is False)
        ),
    }
    return arrays, selection, metadata


def _raw_score_statistics(
    point_scores: torch.Tensor,
    *,
    query_xy: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, np.ndarray]:
    """Summarize [hypothesis, fixed-point, score-family] without target input."""

    values = torch.as_tensor(point_scores)
    if values.ndim != 3 or not bool(torch.isfinite(values).all()):
        raise ValueError("raw dynamic point scores are invalid")
    count = int(values.shape[1])
    worst_count = max(1, int(np.ceil(float(count) * 0.25)))
    sorted_values = torch.sort(values, dim=1).values
    spatial = torch.stack(
        [
            _spatial_2x2_statistic(
                values[:, :, index],
                query_xy=np.asarray(query_xy, dtype=np.float32),
                image_width=int(image_width),
                image_height=int(image_height),
            )
            for index in range(values.shape[2])
        ],
        dim=1,
    )
    return {
        "means": values.mean(dim=1).detach().cpu().numpy().astype(np.float64, copy=False),
        "medians": values.median(dim=1).values.detach().cpu().numpy().astype(
            np.float64, copy=False
        ),
        "worst_quartile_means": sorted_values[:, :worst_count].mean(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False),
        "spatial_median_of_means_2x2": spatial.detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False),
    }


def _pairwise_raw_branch_scores(
    *,
    model: CandidateBidirectionalAbsoluteContextLikelihood,
    rows: torch.Tensor,
    query_xy_by_candidate: torch.Tensor,
    evidence_branch: str,
    scale_names: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate exactly one uncalibrated V5 evidence head at dynamic crops.

    The geometry and strict-identity heads intentionally remain separate.
    This helper only normalizes their tensor names so the frozen-denominator
    diagnostic can audit either one under identical candidate/support inputs;
    it never turns either raw residual into a posterior or combines them.
    """

    branch = str(evidence_branch)
    if branch == EVIDENCE_BRANCH_GEOMETRY:
        dynamic = model.forward_pairwise_raw_layout_at_query_xy(
            rows, query_xy_by_candidate, scale_names=scale_names
        )
        view_key = "view_raw_scores"
        per_scale_key = "per_scale_view_raw_scores"
    elif branch == EVIDENCE_BRANCH_STRICT_IDENTITY:
        dynamic = model.forward_pairwise_identity_raw_layout_at_query_xy(
            rows, query_xy_by_candidate, scale_names=scale_names
        )
        view_key = "identity_view_raw_scores"
        per_scale_key = "identity_per_scale_view_raw_scores"
    else:
        raise ValueError(f"unsupported V5 evidence branch: {branch}")
    return (
        dynamic[view_key],
        dynamic[per_scale_key],
        dynamic["support_view_available"],
        dynamic["query_projection_in_image"],
    )


def _score_frozen_hypotheses(
    *,
    model: CandidateBidirectionalAbsoluteContextLikelihood,
    poses_w2c: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    candidate_view_weights: np.ndarray,
    query_xy: np.ndarray,
    camera: object,
    image_width: int,
    image_height: int,
    hypothesis_batch_size: int,
    query_point_batch_size: int,
    evidence_branch: str,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Score only candidate-projected held-out crops under immutable inputs."""

    if int(getattr(camera, "model_id")) != 2:
        raise ValueError("dynamic V5 scorer requires COLMAP SIMPLE_RADIAL cameras")
    params = tuple(float(value) for value in getattr(camera, "params"))
    if (
        len(params) != 4
        or int(hypothesis_batch_size) <= 0
        or int(query_point_batch_size) <= 0
    ):
        raise ValueError("dynamic V5 score batching configuration is invalid")
    xyz = np.asarray(candidate_xyz, dtype=np.float32)
    probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
    weights = np.asarray(candidate_view_weights, dtype=np.float32)
    point_count, candidate_count, view_count = weights.shape
    if (
        xyz.shape != (point_count, candidate_count, 3)
        or probabilities.shape != (point_count, candidate_count)
        or null.shape != (point_count,)
        or np.asarray(query_xy).shape != (point_count, 2)
        or int(model.row_count) != point_count
        or int(model._support_image_indices.shape[1]) != candidate_count
        or int(model._support_image_indices.shape[2]) != view_count
        or not np.isfinite(xyz).all()
    ):
        raise ValueError("dynamic V5 candidate geometry is incompatible")
    xyz_tensor = torch.as_tensor(xyz, dtype=torch.float32, device=device)
    probability_tensor = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
    null_tensor = torch.as_tensor(null, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    all_statistics: dict[str, list[np.ndarray]] = {
        "means": [],
        "medians": [],
        "worst_quartile_means": [],
        "spatial_median_of_means_2x2": [],
        "effective_point_counts": [],
        "effective_view_masses": [],
    }
    with torch.inference_mode():
        for begin in range(0, len(poses_w2c), int(hypothesis_batch_size)):
            end = min(begin + int(hypothesis_batch_size), len(poses_w2c))
            batch_size = end - begin
            pose_tensor = torch.as_tensor(poses_w2c[begin:end], dtype=torch.float32, device=device)
            point_scores = torch.empty(
                (batch_size, point_count, len(SCORE_NAMES)), dtype=torch.float32, device=device
            )
            effective_points = torch.empty(
                (batch_size, point_count), dtype=torch.bool, device=device
            )
            effective_mass = torch.empty(
                (batch_size, point_count), dtype=torch.float32, device=device
            )
            for point_begin in range(0, point_count, int(query_point_batch_size)):
                point_end = min(point_begin + int(query_point_batch_size), point_count)
                local_count = point_end - point_begin
                projected, projection_valid = project_simple_radial_torch(
                    xyz_tensor[point_begin:point_end].reshape(-1, 3),
                    pose_tensor,
                    focal_length=params[0],
                    principal_x=params[1],
                    principal_y=params[2],
                    radial_k=params[3],
                    image_width=int(image_width),
                    image_height=int(image_height),
                )
                projected = projected.reshape(batch_size, local_count, candidate_count, 2)
                projection_valid = projection_valid.reshape(batch_size, local_count, candidate_count)
                safe_projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
                row_indices = torch.arange(
                    point_begin, point_end, dtype=torch.long, device=device
                ).repeat(batch_size)
                raw_all, raw_per_scale, support_available, projection_in_image = (
                    _pairwise_raw_branch_scores(
                        model=model,
                        rows=row_indices,
                        query_xy_by_candidate=safe_projected.reshape(
                            batch_size * local_count, candidate_count, 2
                        ),
                        evidence_branch=evidence_branch,
                    )
                )
                raw_all = raw_all.reshape(
                    batch_size, local_count, candidate_count, view_count
                )
                raw_per_scale = raw_per_scale.reshape(
                    batch_size,
                    local_count,
                    candidate_count,
                    view_count,
                    len(BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES),
                )
                raw_families = torch.cat([raw_all[..., None], raw_per_scale], dim=-1)
                support_available = support_available.reshape(
                    batch_size, local_count, candidate_count, view_count
                )
                projection_in_image = projection_in_image.reshape(
                    batch_size, local_count, candidate_count, view_count
                )
                in_window = projection_in_image & projection_valid[..., None]
                first_contributed: torch.Tensor | None = None
                for score_index in range(len(SCORE_NAMES)):
                    logs, _ratios, contributed = fixed_candidate_point_log_ratios(
                        view_log_ratios=raw_families[..., score_index],
                        view_available=support_available,
                        view_geometric_in_window=in_window,
                        candidate_view_weights=weight_tensor[point_begin:point_end],
                        candidate_probabilities=probability_tensor[point_begin:point_end],
                        null_probabilities=null_tensor[point_begin:point_end],
                    )
                    point_scores[:, point_begin:point_end, score_index] = logs
                    if first_contributed is None:
                        first_contributed = contributed
                if first_contributed is None:
                    raise RuntimeError("dynamic V5 scorer did not produce any raw score family")
                effective_points[:, point_begin:point_end] = (
                    first_contributed.sum(dim=2) > 0.0
                )
                effective_mass[:, point_begin:point_end] = first_contributed.sum(dim=2)
            statistics = _raw_score_statistics(
                point_scores,
                query_xy=np.asarray(query_xy, dtype=np.float32),
                image_width=int(image_width),
                image_height=int(image_height),
            )
            for key in (
                "means",
                "medians",
                "worst_quartile_means",
                "spatial_median_of_means_2x2",
            ):
                all_statistics[key].append(statistics[key])
            all_statistics["effective_point_counts"].append(
                effective_points.sum(dim=1).detach().cpu().numpy().astype(np.int64, copy=False)
            )
            all_statistics["effective_view_masses"].append(
                effective_mass.sum(dim=1).detach().cpu().numpy().astype(np.float64, copy=False)
            )
    return {
        key: np.concatenate(values, axis=0)
        for key, values in all_statistics.items()
    }


def _grid_anchor_coordinates(
    *, image_width: int, image_height: int, grid_size: int, device: torch.device
) -> torch.Tensor:
    """One in-image pixel coordinate per floor-based feature-grid cell."""

    if int(image_width) <= 1 or int(image_height) <= 1 or int(grid_size) <= 0:
        raise ValueError("feature-grid anchor geometry is invalid")
    rows = torch.arange(int(grid_size), dtype=torch.float32, device=device)
    columns = torch.arange(int(grid_size), dtype=torch.float32, device=device)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack(
        [
            (column_grid.reshape(-1) + 0.5) / float(grid_size) * float(image_width),
            (row_grid.reshape(-1) + 0.5) / float(grid_size) * float(image_height),
        ],
        dim=1,
    )


def _precompute_scale_grid_lookup(
    *,
    model: CandidateBidirectionalAbsoluteContextLikelihood,
    scale_name: str,
    grid_size: int,
    image_width: int,
    image_height: int,
    point_batch_size: int,
    anchor_batch_size: int,
    evidence_branch: str,
    device: torch.device,
) -> torch.Tensor:
    """Materialize exact V5 raw scores for every discrete query crop anchor.

    The raw V5 crop extractor uses a floor mapping to a descriptor-grid cell.
    For fixed query images and support views, every dynamic projected crop is
    therefore one of this finite set.  Precomputing it removes the expensive
    cost volume from the hypothesis loop without changing a score value.
    """

    if int(point_batch_size) <= 0 or int(anchor_batch_size) <= 0:
        raise ValueError("V5 lookup precompute batches must be positive")
    point_count = int(model.row_count)
    candidate_count = int(model._support_image_indices.shape[1])
    view_count = int(model._support_image_indices.shape[2])
    anchors = _grid_anchor_coordinates(
        image_width=int(image_width),
        image_height=int(image_height),
        grid_size=int(grid_size),
        device=device,
    )
    lookup = torch.empty(
        (point_count, len(anchors), candidate_count, view_count),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        for point_begin in range(0, point_count, int(point_batch_size)):
            point_end = min(point_begin + int(point_batch_size), point_count)
            base_rows = torch.arange(point_begin, point_end, dtype=torch.long, device=device)
            for anchor_begin in range(0, len(anchors), int(anchor_batch_size)):
                anchor_end = min(anchor_begin + int(anchor_batch_size), len(anchors))
                local_anchors = anchors[anchor_begin:anchor_end]
                row_indices = base_rows[:, None].expand(-1, len(local_anchors)).reshape(-1)
                coordinates = (
                    local_anchors[None]
                    .expand(len(base_rows), -1, -1)
                    .reshape(-1, 2)[:, None]
                    .expand(-1, candidate_count, -1)
                )
                raw, _per_scale, _available, _in_image = _pairwise_raw_branch_scores(
                    model=model,
                    rows=row_indices,
                    query_xy_by_candidate=coordinates,
                    evidence_branch=evidence_branch,
                    scale_names=(str(scale_name),),
                )
                values = raw.reshape(
                    len(base_rows), len(local_anchors), candidate_count, view_count
                )
                lookup[point_begin:point_end, anchor_begin:anchor_end] = values
    return lookup


def _lookup_raw_scores(
    *,
    lookup: torch.Tensor,
    projected_xy: torch.Tensor,
    image_width: int,
    image_height: int,
    grid_size: int,
) -> torch.Tensor:
    """Read candidate-specific score cells using the crop extractor's floor rule."""

    coordinates = torch.as_tensor(projected_xy, dtype=torch.float32, device=lookup.device)
    if coordinates.ndim != 4 or coordinates.shape[2] != lookup.shape[2] or coordinates.shape[3] != 2:
        raise ValueError("dynamic V5 lookup projections are incompatible")
    hypothesis_count, point_count, candidate_count, _ = coordinates.shape
    if point_count != lookup.shape[0]:
        raise ValueError("dynamic V5 lookup point count differs from projections")
    safe = torch.nan_to_num(coordinates, nan=0.0, posinf=0.0, neginf=0.0)
    columns = torch.floor(safe[..., 0] / float(image_width) * float(grid_size)).to(torch.long)
    rows = torch.floor(safe[..., 1] / float(image_height) * float(grid_size)).to(torch.long)
    anchors = rows.clamp(0, int(grid_size) - 1) * int(grid_size) + columns.clamp(
        0, int(grid_size) - 1
    )
    point_indices = torch.arange(point_count, device=lookup.device)[None, :, None].expand(
        hypothesis_count, -1, candidate_count
    )
    candidate_indices = torch.arange(candidate_count, device=lookup.device)[None, None, :].expand(
        hypothesis_count, point_count, -1
    )
    return lookup[point_indices, anchors, candidate_indices]


def _score_frozen_hypotheses_from_grid_lookups(
    *,
    lookups: Mapping[str, torch.Tensor],
    poses_w2c: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    candidate_view_weights: np.ndarray,
    query_xy: np.ndarray,
    camera: object,
    image_width: int,
    image_height: int,
    hypothesis_batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Evaluate frozen hypotheses with precomputed exact discrete crop scores."""

    if int(getattr(camera, "model_id")) != 2:
        raise ValueError("dynamic V5 scorer requires COLMAP SIMPLE_RADIAL cameras")
    params = tuple(float(value) for value in getattr(camera, "params"))
    if len(params) != 4 or int(hypothesis_batch_size) <= 0:
        raise ValueError("dynamic V5 lookup score batching configuration is invalid")
    scale_names = tuple(scale.name for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES)
    if set(lookups) != set(scale_names):
        raise ValueError("dynamic V5 lookup source set is incomplete")
    xyz = np.asarray(candidate_xyz, dtype=np.float32)
    probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
    weights = np.asarray(candidate_view_weights, dtype=np.float32)
    point_count, candidate_count, view_count = weights.shape
    if (
        xyz.shape != (point_count, candidate_count, 3)
        or probabilities.shape != (point_count, candidate_count)
        or null.shape != (point_count,)
        or np.asarray(query_xy).shape != (point_count, 2)
        or not np.isfinite(xyz).all()
        or any(
            lookup.shape != (point_count, int(scale.grid_size) ** 2, candidate_count, view_count)
            for scale, lookup in (
                (scale, lookups[scale.name]) for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES
            )
        )
    ):
        raise ValueError("dynamic V5 lookup geometry is incompatible")
    xyz_tensor = torch.as_tensor(xyz.reshape(-1, 3), dtype=torch.float32, device=device)
    probability_tensor = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
    null_tensor = torch.as_tensor(null, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    support_available = weight_tensor[None] > 0.0
    all_statistics: dict[str, list[np.ndarray]] = {
        "means": [],
        "medians": [],
        "worst_quartile_means": [],
        "spatial_median_of_means_2x2": [],
        "effective_point_counts": [],
        "effective_view_masses": [],
    }
    with torch.inference_mode():
        for begin in range(0, len(poses_w2c), int(hypothesis_batch_size)):
            end = min(begin + int(hypothesis_batch_size), len(poses_w2c))
            pose_tensor = torch.as_tensor(poses_w2c[begin:end], dtype=torch.float32, device=device)
            projected, projection_valid = project_simple_radial_torch(
                xyz_tensor,
                pose_tensor,
                focal_length=params[0],
                principal_x=params[1],
                principal_y=params[2],
                radial_k=params[3],
                image_width=int(image_width),
                image_height=int(image_height),
            )
            projected = projected.reshape(end - begin, point_count, candidate_count, 2)
            projection_valid = projection_valid.reshape(end - begin, point_count, candidate_count)
            per_scale = [
                _lookup_raw_scores(
                    lookup=lookups[scale.name],
                    projected_xy=projected,
                    image_width=int(image_width),
                    image_height=int(image_height),
                    grid_size=int(scale.grid_size),
                )
                for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES
            ]
            raw_families = torch.stack(
                [torch.stack(per_scale, dim=-1).sum(dim=-1), *per_scale], dim=-1
            )
            in_window = projection_valid[..., None].expand(-1, -1, -1, view_count)
            available = support_available.expand_as(in_window)
            point_scores: list[torch.Tensor] = []
            first_contributed: torch.Tensor | None = None
            for score_index in range(len(SCORE_NAMES)):
                logs, _ratios, contributed = fixed_candidate_point_log_ratios(
                    view_log_ratios=raw_families[..., score_index],
                    view_available=available,
                    view_geometric_in_window=in_window,
                    candidate_view_weights=weight_tensor,
                    candidate_probabilities=probability_tensor,
                    null_probabilities=null_tensor,
                )
                point_scores.append(logs)
                if first_contributed is None:
                    first_contributed = contributed
            if first_contributed is None:
                raise RuntimeError("dynamic V5 lookup scorer did not produce any score family")
            statistics = _raw_score_statistics(
                torch.stack(point_scores, dim=-1),
                query_xy=np.asarray(query_xy, dtype=np.float32),
                image_width=int(image_width),
                image_height=int(image_height),
            )
            for key in (
                "means",
                "medians",
                "worst_quartile_means",
                "spatial_median_of_means_2x2",
            ):
                all_statistics[key].append(statistics[key])
            all_statistics["effective_point_counts"].append(
                (first_contributed.sum(dim=2) > 0.0)
                .sum(dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            all_statistics["effective_view_masses"].append(
                first_contributed.sum(dim=(1, 2))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
    return {key: np.concatenate(values, axis=0) for key, values in all_statistics.items()}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.time()
    output_path = Path(args.output)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"output already exists: {output_path}")
    if (
        int(args.fixed_candidate_top_k) != 20
        or int(args.verification_point_count) <= 0
        or int(args.hypothesis_batch_size) <= 0
        or int(args.query_point_batch_size) <= 0
        or int(args.lookup_point_batch_size) <= 0
        or int(args.lookup_anchor_batch_size) <= 0
        or int(args.hypothesis_limit) < 0
        or str(args.evidence_branch) not in EVIDENCE_BRANCHES
    ):
        raise ValueError("dynamic V5 probe accepts only positive batches and frozen top-20")
    if (
        args.mixed_verification_points_artifact is not None
        and int(args.verification_point_count) != sum(_FORMAL_P1_POINT_COUNTS.values())
    ):
        raise ValueError("formal mixed P1 scoring requires verification_point_count=192")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
    paths = {
        "hypothesis_artifact": Path(args.hypothesis_artifact),
        "baseline_score_artifact": Path(args.baseline_score_artifact),
        "detector_query_cache": Path(args.detector_query_cache),
        "proposals": Path(args.proposals),
        "candidate_artifact": Path(args.candidate_artifact),
        "fixed_candidate_prior_overlay": Path(args.fixed_candidate_prior_overlay),
        "maplet_support_index": Path(args.maplet_support_index),
        "support_geometry_index": Path(args.support_geometry_index),
        "projected_landmark_bank": Path(args.projected_landmark_bank),
        "radio_final_context_cache": Path(args.radio_final_context_cache),
        "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
        "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
        "context_contract": Path(args.context_contract),
        "context_checkpoint": Path(args.context_checkpoint),
        "colmap_cameras_bin": Path(args.colmap_model_dir) / "cameras.bin",
        "colmap_images_bin_camera_ownership_only": Path(args.colmap_model_dir) / "images.bin",
    }
    if args.mixed_verification_points_artifact is not None:
        paths["mixed_verification_points_artifact"] = Path(
            args.mixed_verification_points_artifact
        )
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    contract = _load_contract(paths["context_contract"])
    exact, hypothesis_metadata, baseline_metadata = _load_exact_hypotheses(
        hypothesis_path=paths["hypothesis_artifact"],
        baseline_path=paths["baseline_score_artifact"],
        detector_path=paths["detector_query_cache"],
        proposals_path=paths["proposals"],
        candidate_path=paths["candidate_artifact"],
        prior_path=paths["fixed_candidate_prior_overlay"],
        fixed_candidate_top_k=int(args.fixed_candidate_top_k),
    )
    if int(args.hypothesis_limit) > 0:
        count = min(int(args.hypothesis_limit), len(exact["query_ids"]))
        exact = {key: np.asarray(value)[:count] for key, value in exact.items()}
    query_id = str(np.asarray(exact["query_ids"]).astype(str)[0])
    query_split = str(np.asarray(exact["split_names"]).astype(str)[0])
    if query_split not in {"validation", "test"}:
        raise ValueError("dynamic V5 probe accepts only held-out validation/test query shards")

    detector, detector_metadata, _detector_names = _load_npz_allowlist(
        paths["detector_query_cache"], ("image_ids", "offsets", "xy", "detector_scores")
    )
    proposals, proposal_metadata, _proposal_names = _load_npz_allowlist(
        paths["proposals"],
        ("query_ids", "candidate_track_ids", "coarse_scores"),
        metadata_required=False,
    )
    candidate_artifact, candidate_metadata, _candidate_names = _load_npz_allowlist(
        paths["candidate_artifact"], ("selected_rows",)
    )
    if (
        detector_metadata.get("format") != "alike_detector_mapped_radio_query_cache_v1"
        or candidate_metadata.get("contains_ground_truth") is not False
        or candidate_metadata.get("contains_pose_derived_selection") is not False
        or proposal_metadata.get("format") not in {None, "detector_support_reranked_proposals_v1"}
    ):
        raise ValueError("dynamic V5 inputs violate the target-free frozen-row contract")
    prior_overlay, prior_metadata = _load_candidate_prior_overlay(
        paths["fixed_candidate_prior_overlay"], proposals_path=paths["proposals"], proposals=proposals
    )
    mixed_metadata: dict[str, object] | None = None
    if args.mixed_verification_points_artifact is None:
        probabilities_all, null_all, _retained = _mask_fixed_candidate_posterior_topk(
            candidate_track_ids=prior_overlay["candidate_track_ids"],
            candidate_probabilities=prior_overlay["candidate_probabilities"],
            null_probabilities=prior_overlay["null_probabilities"],
            top_k=int(args.fixed_candidate_top_k),
        )
        kept_rows, selection_audit = _select_s0_verification_rows(
            query_id,
            detector=detector,
            proposals=proposals,
            selected_rows=np.asarray(candidate_artifact["selected_rows"], dtype=np.int64),
            point_count=int(args.verification_point_count),
            detector_log_merit_weight=float(args.detector_log_merit_weight),
        )
        candidate_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)[kept_rows]
        candidate_probabilities = probabilities_all[kept_rows]
        null_probabilities = null_all[kept_rows]
        query_xy = np.asarray(detector["xy"], dtype=np.float32)[kept_rows]
        candidate_bank_rows_from_points: np.ndarray | None = None
        verification_source_row_indices = np.asarray(kept_rows, dtype=np.int64)
        verification_point_sources = np.full(
            (len(kept_rows),), POINT_SOURCE_ALIKE, dtype=np.str_
        )
        verification_source_detector_rows = np.asarray(kept_rows, dtype=np.int64)
        selection_audit = {
            "source": "s0_detector_rows_unused_by_candidate_artifact",
            "point_count": int(args.verification_point_count),
            "detector_log_merit_weight": float(args.detector_log_merit_weight),
            "point_sources": "ALike_detector_only_not_final_p1_mixed_point_set",
            **selection_audit,
        }
        if (
            candidate_tracks.shape[1] != 20
            or not np.array_equal(candidate_tracks, prior_overlay["candidate_track_ids"][kept_rows])
        ):
            raise ValueError("dynamic V5 fixed candidate layout differs from the immutable overlay")
    else:
        mixed, selection_audit, mixed_metadata = _formal_p1_mixed_evidence_for_query(
            points_path=paths["mixed_verification_points_artifact"],
            query_id=query_id,
            query_split=query_split,
            detector_path=paths["detector_query_cache"],
            candidate_path=paths["candidate_artifact"],
            landmark_bank_path=paths["projected_landmark_bank"],
            detector=detector,
            proposals=proposals,
            selected_rows=np.asarray(candidate_artifact["selected_rows"], dtype=np.int64),
            point_count=int(args.verification_point_count),
        )
        candidate_tracks = np.asarray(mixed["candidate_track_ids"], dtype=np.int64)
        candidate_probabilities = np.asarray(mixed["candidate_probabilities"], dtype=np.float32)
        null_probabilities = np.asarray(mixed["null_probabilities"], dtype=np.float32)
        query_xy = np.asarray(mixed["xy"], dtype=np.float32)
        candidate_bank_rows_from_points = np.asarray(mixed["candidate_bank_rows"], dtype=np.int64)
        verification_source_row_indices = np.asarray(mixed["source_point_ids"], dtype=np.int64)
        verification_point_sources = np.asarray(mixed["point_sources"], dtype=np.str_)
        verification_source_detector_rows = np.asarray(
            mixed["source_detector_rows"], dtype=np.int64
        )
        if candidate_tracks.shape != candidate_probabilities.shape or candidate_tracks.shape[1] != 20:
            raise ValueError("formal P1 mixed candidate layout is invalid")

    bank_tracks, bank_xyz, bank_metadata = _load_bank_xyz(paths["projected_landmark_bank"])
    maplet_tracks, maplet_image_ids, maplet_image_indices, maplet_coverage, maplet_metadata = (
        _load_maplet_support_fields(paths["maplet_support_index"])
    )
    if str(maplet_metadata.get("source_landmark_index_sha256", "")) != file_sha256_short(
        paths["projected_landmark_bank"]
    ):
        raise ValueError("maplet support index was built from another landmark bank")
    candidate_bank_rows = (
        _resolve_rows(candidate_tracks, canonical_track_ids=bank_tracks)
        if candidate_bank_rows_from_points is None
        else np.asarray(candidate_bank_rows_from_points, dtype=np.int64)
    )
    if (
        candidate_bank_rows.shape != candidate_tracks.shape
        or np.any(candidate_bank_rows[candidate_tracks >= 0] < 0)
        or np.any(candidate_bank_rows[candidate_tracks >= 0] >= len(bank_tracks))
        or not np.array_equal(
            bank_tracks[candidate_bank_rows[candidate_tracks >= 0]],
            candidate_tracks[candidate_tracks >= 0],
        )
    ):
        raise ValueError("V5 candidate bank rows do not identify their declared physical tracks")
    candidate_xyz = np.zeros((*candidate_tracks.shape, 3), dtype=np.float32)
    valid_tracks = candidate_bank_rows >= 0
    candidate_xyz[valid_tracks] = bank_xyz[candidate_bank_rows[valid_tracks]]
    if np.any((candidate_probabilities > 0.0) & ~valid_tracks):
        raise ValueError("a positive-mass V5 candidate lacks landmark-bank geometry")
    all_views = _fixed_candidate_views(
        candidate_track_ids=candidate_tracks,
        candidate_probabilities=candidate_probabilities,
        maplet_track_ids=maplet_tracks,
        support_image_ids=maplet_image_ids,
        support_image_indices=maplet_image_indices,
        support_coverage_counts=maplet_coverage,
    )
    views = _first_fixed_support_views(all_views, count=FIXED_SUPPORT_VIEW_COUNT)
    support_geometry, support_geometry_metadata = load_support_observation_geometry_index_npz(
        paths["support_geometry_index"]
    )
    sources = load_context_attention_sources(
        radio_final_context_cache=paths["radio_final_context_cache"],
        radio_intermediate_context_cache=paths["radio_intermediate_context_cache"],
        alike_spatial_context_cache=paths["alike_spatial_context_cache"],
        expected_radio_checkpoint=str(contract.get("radio_checkpoint_sha256", "")),
    )
    _validate_sources_against_contract(contract=contract, sources=sources)
    reference_source = sources[0]
    runtime = build_fixed_candidate_context_runtime(
        query_ids=np.full((len(verification_source_row_indices),), query_id),
        query_xy=query_xy,
        candidate_track_ids=candidate_tracks,
        candidate_support_image_ids=views.support_image_ids,
        candidate_view_valid=views.valid,
        cache_image_ids=reference_source.image_ids,
        support_geometry=support_geometry,
    )
    query_image_positions = np.flatnonzero(
        np.asarray(reference_source.image_ids).astype(str) == query_id
    )
    if len(query_image_positions) != 1:
        raise ValueError("query image is absent or duplicated in the V5 context cache")
    image_width, image_height = (
        int(value) for value in np.asarray(reference_source.image_sizes)[int(query_image_positions[0])]
    )
    if not np.all((query_xy[:, 0] >= 0.0) & (query_xy[:, 0] <= image_width - 1.0)) or not np.all(
        (query_xy[:, 1] >= 0.0) & (query_xy[:, 1] <= image_height - 1.0)
    ):
        raise ValueError("held-out verification coordinates exceed the query image")
    cameras = read_colmap_cameras_binary(paths["colmap_cameras_bin"])
    camera_ids = read_colmap_image_camera_ids_binary(paths["colmap_images_bin_camera_ownership_only"])
    camera_id = camera_ids.get(query_id)
    if camera_id is None or int(camera_id) not in cameras:
        raise ValueError("query image has no declared COLMAP camera")
    camera = cameras[int(camera_id)]
    if (int(camera.width), int(camera.height)) != (image_width, image_height):
        raise ValueError("query camera geometry differs from the context cache")
    model, checkpoint_metadata = _load_v5_model(
        checkpoint_path=paths["context_checkpoint"],
        contract_path=paths["context_contract"],
        contract=contract,
        family=str(args.family),
        sources=sources,
        runtime=runtime,
        query_xy=query_xy,
        candidate_probabilities=candidate_probabilities,
        null_probabilities=null_probabilities,
        device=device,
    )
    if str(args.dynamic_scoring_mode) == "direct":
        statistics = _score_frozen_hypotheses(
            model=model,
            poses_w2c=np.asarray(exact["poses_w2c"], dtype=np.float64),
            candidate_xyz=candidate_xyz,
            candidate_probabilities=candidate_probabilities,
            null_probabilities=null_probabilities,
            candidate_view_weights=views.weights,
            query_xy=query_xy,
            camera=camera,
            image_width=image_width,
            image_height=image_height,
            hypothesis_batch_size=int(args.hypothesis_batch_size),
            query_point_batch_size=int(args.query_point_batch_size),
            evidence_branch=str(args.evidence_branch),
            device=device,
        )
    else:
        lookups = {
            scale.name: _precompute_scale_grid_lookup(
                model=model,
                scale_name=scale.name,
                grid_size=int(scale.grid_size),
                image_width=image_width,
                image_height=image_height,
                point_batch_size=int(args.lookup_point_batch_size),
                anchor_batch_size=int(args.lookup_anchor_batch_size),
                evidence_branch=str(args.evidence_branch),
                device=device,
            )
            for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES
        }
        statistics = _score_frozen_hypotheses_from_grid_lookups(
            lookups=lookups,
            poses_w2c=np.asarray(exact["poses_w2c"], dtype=np.float64),
            candidate_xyz=candidate_xyz,
            candidate_probabilities=candidate_probabilities,
            null_probabilities=null_probabilities,
            candidate_view_weights=views.weights,
            query_xy=query_xy,
            camera=camera,
            image_width=image_width,
            image_height=image_height,
            hypothesis_batch_size=int(args.hypothesis_batch_size),
            device=device,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    strict_identity_branch = str(args.evidence_branch) == EVIDENCE_BRANCH_STRICT_IDENTITY
    arrays: dict[str, np.ndarray] = {
        "query_ids": np.asarray(exact["query_ids"]).astype(str),
        "split_names": np.asarray(exact["split_names"]).astype(str),
        "evaluation_labels": np.asarray(exact["evaluation_labels"]).astype(str),
        "hypothesis_indices": np.asarray(exact["hypothesis_indices"], dtype=np.int64),
        "source_chosen_for_optional_pose": np.asarray(
            exact["source_chosen_for_optional_pose"], dtype=bool
        ),
        "baseline_score_top1": np.asarray(exact["independent_score_top1"], dtype=bool),
        "baseline_selection_scores": np.asarray(
            exact["independent_selection_scores"], dtype=np.float64
        ),
        "score_names": np.asarray(SCORE_NAMES, dtype=np.str_),
        "raw_score_means": statistics["means"],
        "raw_score_medians": statistics["medians"],
        "raw_score_worst_quartile_means": statistics["worst_quartile_means"],
        "raw_score_spatial_median_of_means_2x2": statistics[
            "spatial_median_of_means_2x2"
        ],
        "effective_point_counts": statistics["effective_point_counts"],
        "effective_view_masses": statistics["effective_view_masses"],
        "verification_query_ids": np.full((len(verification_source_row_indices),), query_id),
        "verification_source_row_indices": verification_source_row_indices,
        "verification_point_sources": verification_point_sources,
        "verification_source_detector_rows": verification_source_detector_rows,
        "verification_xy": query_xy,
        "candidate_track_ids": candidate_tracks,
        "candidate_probabilities": candidate_probabilities,
        "null_probabilities": null_probabilities,
        "candidate_view_weights": views.weights,
        "candidate_support_image_ids": views.support_image_ids,
    }
    metadata: dict[str, Any] = {
        "format": SCORE_FORMAT,
        "version": (
            (
                P1_MIXED_STRICT_IDENTITY_SCORE_VERSION
                if strict_identity_branch
                else P1_MIXED_SCORE_VERSION
            )
            if args.mixed_verification_points_artifact is not None
            else (
                STRICT_IDENTITY_SCORE_VERSION if strict_identity_branch else SCORE_VERSION
            )
        ),
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "row_count": int(len(arrays["query_ids"])),
        "query_count": 1,
        "query_id": query_id,
        "split_name": query_split,
        "model_family": str(args.family),
        "evidence_branch": str(args.evidence_branch),
        "model_checkpoint_contract": {
            key: checkpoint_metadata.get(key)
            for key in (
                "contract_sha256",
                "architecture",
                "architecture_id",
                "family_profile",
                "hidden_dim",
                "heads",
                "dropout",
                "position_only_control",
                "candidate_log_likelihood_ratio_is_independent_pose_likelihood",
                "identity_candidate_probability_allowed_for_pnp_overlay",
            )
        },
        "raw_score_semantics": (
            "strict_exact_track_identity_raw_candidate_residual_mixed_under_"
            "fixed_candidate_null_denominator"
            if strict_identity_branch
            else "geometry_head_raw_candidate_reranking_residual_mixed_under_"
            "fixed_candidate_null_denominator"
        ),
        "raw_score_is_calibrated_independent_pose_likelihood": False,
        "identity_head_used": strict_identity_branch,
        "identity_head_allowed_for_pnp_overlay": False,
        "dynamic_scoring_backend": str(args.dynamic_scoring_mode),
        "dynamic_crop_lookup": {
            "enabled": str(args.dynamic_scoring_mode) == "discrete_lookup",
            "equivalence_contract": (
                "query_crop_and_absolute_position_use_the_same_floor_feature_grid_cell_v1"
            ),
            "lookup_point_batch_size": int(args.lookup_point_batch_size),
            "lookup_anchor_batch_size": int(args.lookup_anchor_batch_size),
        },
        "strict_frozen_evidence_contract": {
            "heldout_query_rows": True,
            "verification_point_selector_fixed_across_hypotheses": True,
            "fixed_global_topl": True,
            "fixed_candidate_top_k": 20,
            "explicit_null_mass": True,
            "candidate_identity_fixed_across_hypotheses": True,
            "candidate_reselection_per_pose": False,
            "support_reselection_per_pose": False,
            "support_view_descriptor_averaging": False,
            "support_views": "fixed_maplet_rank_prefix_two_matching_v5_train_layout_v1",
            "candidate_specific_query_projection": True,
            "query_center_gaussian_fallback": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "geometry_head_only": not strict_identity_branch,
            "identity_head_not_used": not strict_identity_branch,
            "strict_identity_raw_residual_only": strict_identity_branch,
            "raw_scores_calibrated_as_independent_pose_likelihood": False,
            "formal_p1_mixed_multiscale_points": args.mixed_verification_points_artifact
            is not None,
        },
        "score_names": list(SCORE_NAMES),
        "frozen_query_evidence_sha256": _array_digest(
            {
                "verification_source_row_indices": verification_source_row_indices,
                "verification_point_sources": verification_point_sources,
                "verification_source_detector_rows": verification_source_detector_rows,
                "verification_xy": query_xy,
                "candidate_track_ids": candidate_tracks,
                "candidate_probabilities": candidate_probabilities,
                "null_probabilities": null_probabilities,
                "candidate_view_weights": views.weights,
                "candidate_support_image_ids": views.support_image_ids,
            }
        ),
        "verification_point_selection": selection_audit,
        "hypothesis_scope": {
            "all_frozen_hypotheses": int(args.hypothesis_limit) == 0,
            "development_prefix_limit": int(args.hypothesis_limit),
            "scored_hypothesis_count": int(len(np.asarray(exact["query_ids"]))),
        },
        "camera": {
            "camera_id": int(camera.camera_id),
            "model_id": int(camera.model_id),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
            "ownership_parser": "image_name_to_camera_id_pose_discarded_v1",
        },
        "source_cache_lineage": {
            source.name: {
                "source_image_manifest_sha256": source.metadata.get(
                    "source_image_manifest_sha256"
                ),
                "pca_fit_scope": source.metadata.get("pca_fit_scope"),
                "radio_checkpoint_sha256": source.metadata.get("radio_checkpoint_sha256"),
                "alike_checkpoint_sha256": source.metadata.get("alike_checkpoint_sha256"),
            }
            for source in sources
        },
        "source_metadata_hashes": {
            "contract": _canonical_hash(contract),
            "hypothesis": _canonical_hash(hypothesis_metadata),
            "baseline_s0": _canonical_hash(baseline_metadata),
            "detector": _canonical_hash(detector_metadata),
            "proposals": _canonical_hash(proposal_metadata),
            "candidate": _canonical_hash(candidate_metadata),
            "candidate_prior": _canonical_hash(prior_metadata),
            "maplet": _canonical_hash(maplet_metadata),
            "support_geometry": _canonical_hash(support_geometry_metadata),
            "landmark_bank": _canonical_hash(bank_metadata),
            "mixed_verification_points": (
                None if mixed_metadata is None else _canonical_hash(mixed_metadata)
            ),
        },
        "inputs": _input_manifest(paths),
        "implementation": {
            "script_sha256": file_sha256_short(Path(__file__)),
            "context_probe_module_sha256": file_sha256_short(
                Path("feature_extract/vfm/localization/context_attention_candidate_probe.py")
            ),
        },
        "elapsed_seconds": float(time.time() - start),
    }
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
                "stage": "v5_dynamic_absolute_context_pose_evidence",
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
