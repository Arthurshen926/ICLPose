"""Export target-free, candidate-specific absolute appearance evidence.

The exporter starts from the exact frozen S0 top-20 verifier rows.  For every
``(query anchor, candidate track, fixed maplet support view)`` it compares real
query and support image feature patches at their own SfM observation anchors.
Unlike the previous S1 spatial scorer, no candidate 3-D point is projected by
a hypothesised pose here.  The output is therefore a pose-free appearance
artifact that can answer a narrower question first: do C-RADIO/ALIKE features
contain enough absolute, candidate-specific information to rank the right
track above repeated alternatives?

Targets are deliberately absent from the command line and every NPZ input is
read through the S0 inference-safe allowlist.  A separate audit script may
join the frozen artifact with SfM targets afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    FixedCandidateViews,
    _as_image_grid_sources,
    _fixed_candidate_views,
    _load_bank_xyz,
    _load_maplet_support_fields,
    _load_npz_allowlist,
    _resolve_rows,
    _select_s0_verification_rows,
    _validate_baseline_score_metadata,
)
from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _load_candidate_prior_overlay,
    _mask_fixed_candidate_posterior_topk,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    build_fixed_candidate_context_runtime,
)
from feature_extract.vfm.localization.frozen_multiscale_candidate_appearance import (
    FROZEN_MULTISCALE_CANDIDATE_APPEARANCE_VERSION,
    aligned_patch_ncc,
    coverage_weighted_view_appearance,
)
from feature_extract.vfm.localization.frozen_multiscale_candidate_region_layout import (
    FROZEN_MULTISCALE_CANDIDATE_REGION_LAYOUT_VERSION,
    REGION_LAYOUT_STATISTIC_NAMES,
    region_layout_similarity,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    ImageGridFeatureSource,
)


ARTIFACT_FORMAT = "frozen_multiscale_candidate_absolute_appearance_v1"
ARTIFACT_VERSION = FROZEN_MULTISCALE_CANDIDATE_APPEARANCE_VERSION
REGION_LAYOUT_ARTIFACT_FORMAT = "frozen_multiscale_candidate_region_layout_v1"


@dataclass(frozen=True)
class AppearanceProfile:
    """One fixed image feature space and aligned context window."""

    name: str
    source_name: str
    window_size: int

    def __post_init__(self) -> None:
        if (
            not str(self.name)
            or not str(self.source_name)
            or int(self.window_size) <= 0
            or int(self.window_size) % 2 != 1
        ):
            raise ValueError("absolute appearance profile is invalid")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis_artifact", required=True)
    parser.add_argument(
        "--baseline_score_artifact",
        default=None,
        help=(
            "optional strict S0 score-lineage check; query identity can instead "
            "come from the inference-only hypothesis manifest"
        ),
    )
    parser.add_argument(
        "--query_id",
        default=None,
        help=(
            "select one query from a multi-query train score shard; omitted "
            "only when the score artifact contains exactly one query"
        ),
    )
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--fixed_candidate_prior_overlay", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--radio_intermediate_context_cache", required=True)
    parser.add_argument("--alike_spatial_context_cache", required=True)
    parser.add_argument(
        "--profiles",
        default=(
            "radio_final_center:radio_final:1;"
            "radio_final_context3:radio_final:3;"
            "radio_final_context5:radio_final:5;"
            "radio_intermediate_center:radio_intermediate:1;"
            "radio_intermediate_context5:radio_intermediate:5;"
            "radio_intermediate_context9:radio_intermediate:9;"
            "radio_intermediate_context13:radio_intermediate:13;"
            "alike_center:alike:1;"
            "alike_context3:alike:3;"
            "alike_context5:alike:5;"
            "alike_context9:alike:9"
        ),
        help="semicolon-separated name:source:window profiles",
    )
    parser.add_argument("--fixed_candidate_top_k", type=int, default=20)
    parser.add_argument("--verification_point_count", type=int, default=192)
    parser.add_argument("--detector_log_merit_weight", type=float, default=0.01)
    parser.add_argument("--template_batch_size", type=int, default=2048)
    parser.add_argument("--minimum_support_fraction", type=float, default=0.75)
    parser.add_argument("--minimum_overlap_fraction", type=float, default=0.75)
    parser.add_argument(
        "--appearance_mode",
        choices=("aligned_ncc", "region_layout"),
        default="aligned_ncc",
        help="target-free local aligned NCC or landmark-centred large-region layout",
    )
    parser.add_argument(
        "--region_grid_size",
        type=int,
        default=3,
        help="fixed region partition for --appearance_mode region_layout",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def parse_profiles(
    value: str, *, source_grid_sizes: Mapping[str, int]
) -> tuple[AppearanceProfile, ...]:
    profiles: list[AppearanceProfile] = []
    for raw in str(value).split(";"):
        item = raw.strip()
        if not item:
            continue
        parts = tuple(part.strip() for part in item.split(":"))
        if len(parts) != 3:
            raise ValueError("each appearance profile must be name:source:window")
        try:
            window = int(parts[2])
        except ValueError as error:
            raise ValueError("appearance profile window must be an integer") from error
        profile = AppearanceProfile(
            name=parts[0], source_name=parts[1], window_size=window
        )
        grid_size = source_grid_sizes.get(profile.source_name)
        if grid_size is None:
            raise ValueError(f"appearance profile uses unknown source {profile.source_name!r}")
        if profile.window_size > int(grid_size):
            raise ValueError(
                f"appearance profile {profile.name} exceeds {profile.source_name} grid"
            )
        profiles.append(profile)
    if not profiles or len({profile.name for profile in profiles}) != len(profiles):
        raise ValueError("appearance profiles must be non-empty with unique names")
    return tuple(profiles)


def _load_baseline_query_identity(
    path: Path, *, requested_query_id: str | None = None
) -> tuple[str, str, dict[str, object]]:
    """Resolve one target-free S0 query identity from a score artifact.

    Validation/test score shards carry one query each, whereas the frozen
    training shards intentionally pack several train queries together.  The
    latter need an explicit name so the exporter cannot accidentally combine
    their independent verification rows or their supervision roles.
    """

    arrays, metadata, _names = _load_npz_allowlist(path, ("query_ids", "split_names"))
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    if len(query_ids) == 0 or len(query_ids) != len(split_names):
        raise ValueError("baseline score query identity fields are invalid")
    unique_queries = tuple(dict.fromkeys(query_ids.tolist()))
    if requested_query_id is None:
        if len(unique_queries) != 1:
            raise ValueError(
                "a multi-query baseline score artifact requires --query_id"
            )
        query_id = str(unique_queries[0])
    else:
        query_id = str(requested_query_id)
        if not query_id or query_id not in set(unique_queries):
            raise ValueError("--query_id is absent from the baseline score artifact")
    selected_splits = tuple(dict.fromkeys(split_names[query_ids == query_id].tolist()))
    if len(selected_splits) != 1 or selected_splits[0] not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("baseline score query has an invalid split assignment")
    return query_id, str(selected_splits[0]), metadata


def _load_hypothesis_query_identity(
    path: Path, *, requested_query_id: str | None
) -> tuple[str, str, dict[str, object]]:
    """Resolve an S0 query only from inference-safe hypothesis manifest fields."""

    arrays, metadata, _names = _load_npz_allowlist(path, ("query_ids", "split_names"))
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    if len(query_ids) == 0 or len(query_ids) != len(split_names):
        raise ValueError("hypothesis manifest query identity fields are invalid")
    if (
        metadata.get("format") != "grouped_pose_hypotheses_inference_only_v1"
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_generation") is not False
    ):
        raise ValueError("hypothesis manifest violates the inference-only contract")
    unique_queries = tuple(dict.fromkeys(query_ids.tolist()))
    if requested_query_id is None:
        if len(unique_queries) != 1:
            raise ValueError("a multi-query hypothesis artifact requires --query_id")
        query_id = str(unique_queries[0])
    else:
        query_id = str(requested_query_id)
        if not query_id or query_id not in set(unique_queries):
            raise ValueError("--query_id is absent from the hypothesis artifact")
    selected_splits = tuple(dict.fromkeys(split_names[query_ids == query_id].tolist()))
    if len(selected_splits) != 1 or selected_splits[0] not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("hypothesis manifest query has an invalid split assignment")
    return query_id, str(selected_splits[0]), metadata


def _validate_hypothesis_input_lineage(
    metadata: Mapping[str, object],
    *,
    candidate_path: Path,
    proposals_path: Path,
) -> None:
    """Bind a score-free query manifest to the frozen inference inputs."""

    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("hypothesis manifest lacks its input lineage")
    if (
        str(inputs.get("candidate_artifact_sha256", ""))
        != file_sha256_short(candidate_path)
        or str(inputs.get("proposals_sha256", "")) != file_sha256_short(proposals_path)
    ):
        raise ValueError("hypothesis manifest input lineage differs from S0 appearance inputs")


def _score_profile_views(
    *,
    profile: AppearanceProfile,
    source: ImageGridFeatureSource,
    query_id: str,
    query_xy: np.ndarray,
    views: FixedCandidateViews,
    runtime_support_xy: np.ndarray,
    template_batch_size: int,
    minimum_support_fraction: float,
    minimum_overlap_fraction: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw per-view aligned appearance values for one frozen profile."""

    if int(template_batch_size) <= 0:
        raise ValueError("appearance template batch size must be positive")
    support_ids = np.asarray(views.support_image_ids).astype(str)
    support_valid = np.asarray(views.valid, dtype=bool)
    support_xy = np.asarray(runtime_support_xy, dtype=np.float32)
    coordinates = np.asarray(query_xy, dtype=np.float32)
    if (
        support_ids.shape != support_valid.shape
        or support_xy.shape != (*support_valid.shape, 2)
        or coordinates.shape != (support_valid.shape[0], 2)
    ):
        raise ValueError("appearance profile support layout is incompatible")
    point_indices, candidate_indices, view_indices = np.nonzero(support_valid)
    if len(point_indices) == 0:
        raise ValueError("positive-mass candidates have no fixed support views")
    shape = support_valid.shape
    scores = np.full(shape, np.nan, dtype=np.float32)
    overlap = np.zeros(shape, dtype=np.float32)
    support_fraction = np.zeros(shape, dtype=np.float32)
    usable = np.zeros(shape, dtype=bool)
    source_grid_cache: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        query_patches, query_valid = source.context_patches_torch(
            np.full((len(coordinates),), str(query_id)),
            coordinates,
            window_size=profile.window_size,
            device=device,
            source_grid_cache=source_grid_cache,
        )
        flat_support_ids = support_ids[point_indices, candidate_indices, view_indices]
        flat_support_xy = support_xy[point_indices, candidate_indices, view_indices]
        for begin in range(0, len(flat_support_ids), int(template_batch_size)):
            end = min(begin + int(template_batch_size), len(flat_support_ids))
            selected_points = torch.as_tensor(
                point_indices[begin:end], dtype=torch.long, device=device
            )
            support_patches, support_mask = source.context_patches_torch(
                flat_support_ids[begin:end],
                flat_support_xy[begin:end],
                window_size=profile.window_size,
                device=device,
                source_grid_cache=source_grid_cache,
            )
            evidence = aligned_patch_ncc(
                query_patches=query_patches.index_select(0, selected_points),
                support_patches=support_patches,
                query_valid=query_valid.index_select(0, selected_points),
                support_valid=support_mask,
                minimum_support_fraction=float(minimum_support_fraction),
                minimum_overlap_fraction=float(minimum_overlap_fraction),
            )
            rows = point_indices[begin:end]
            columns = candidate_indices[begin:end]
            slots = view_indices[begin:end]
            scores[rows, columns, slots] = evidence.score.detach().cpu().numpy()
            overlap[rows, columns, slots] = (
                evidence.overlap_fraction.detach().cpu().numpy()
            )
            support_fraction[rows, columns, slots] = (
                evidence.support_fraction.detach().cpu().numpy()
            )
            usable[rows, columns, slots] = evidence.usable.detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if np.any(usable & ~np.isfinite(scores)):
        raise RuntimeError("usable direct appearance view has no finite score")
    if np.any(~support_valid & usable):
        raise RuntimeError("invalid fixed support view unexpectedly has direct evidence")
    return scores, overlap, support_fraction, usable


def _score_region_layout_profile_views(
    *,
    profile: AppearanceProfile,
    source: ImageGridFeatureSource,
    query_id: str,
    query_xy: np.ndarray,
    views: FixedCandidateViews,
    runtime_support_xy: np.ndarray,
    template_batch_size: int,
    region_grid_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Export pose-free per-view large-context region-layout similarities."""

    if int(template_batch_size) <= 0 or int(region_grid_size) <= 0:
        raise ValueError("region-layout profile configuration is invalid")
    support_ids = np.asarray(views.support_image_ids).astype(str)
    support_valid = np.asarray(views.valid, dtype=bool)
    support_xy = np.asarray(runtime_support_xy, dtype=np.float32)
    coordinates = np.asarray(query_xy, dtype=np.float32)
    if (
        support_ids.shape != support_valid.shape
        or support_xy.shape != (*support_valid.shape, 2)
        or coordinates.shape != (support_valid.shape[0], 2)
        or int(profile.window_size) < int(region_grid_size)
    ):
        raise ValueError("region-layout support layout is incompatible")
    point_indices, candidate_indices, view_indices = np.nonzero(support_valid)
    if len(point_indices) == 0:
        raise ValueError("positive-mass candidates have no fixed support views")
    statistic_count = len(REGION_LAYOUT_STATISTIC_NAMES)
    shape = (*support_valid.shape, statistic_count)
    scores = np.full(shape, np.nan, dtype=np.float32)
    usable = np.zeros(shape, dtype=bool)
    region_pairs = np.zeros(support_valid.shape, dtype=np.int16)
    source_grid_cache: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        query_patches, query_valid = source.context_patches_torch(
            np.full((len(coordinates),), str(query_id)),
            coordinates,
            window_size=profile.window_size,
            device=device,
            source_grid_cache=source_grid_cache,
        )
        flat_support_ids = support_ids[point_indices, candidate_indices, view_indices]
        flat_support_xy = support_xy[point_indices, candidate_indices, view_indices]
        for begin in range(0, len(flat_support_ids), int(template_batch_size)):
            end = min(begin + int(template_batch_size), len(flat_support_ids))
            selected_points = torch.as_tensor(
                point_indices[begin:end], dtype=torch.long, device=device
            )
            support_patches, support_mask = source.context_patches_torch(
                flat_support_ids[begin:end],
                flat_support_xy[begin:end],
                window_size=profile.window_size,
                device=device,
                source_grid_cache=source_grid_cache,
            )
            evidence = region_layout_similarity(
                query_patches=query_patches.index_select(0, selected_points),
                support_patches=support_patches,
                query_valid=query_valid.index_select(0, selected_points),
                support_valid=support_mask,
                region_grid_size=int(region_grid_size),
            )
            rows = point_indices[begin:end]
            columns = candidate_indices[begin:end]
            slots = view_indices[begin:end]
            scores[rows, columns, slots] = evidence.scores.detach().cpu().numpy()
            usable[rows, columns, slots] = evidence.usable.detach().cpu().numpy()
            region_pairs[rows, columns, slots] = (
                evidence.region_pair_count.detach().cpu().numpy().astype(np.int16)
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if np.any(usable & ~np.isfinite(scores)):
        raise RuntimeError("usable region-layout view has no finite score")
    if np.any(~support_valid[..., None] & usable):
        raise RuntimeError("invalid fixed support view unexpectedly has region evidence")
    return scores, usable, region_pairs


def _aggregate_profile_views(
    *,
    view_scores: np.ndarray,
    view_usable: np.ndarray,
    view_weights: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coverage-marginalize view scores while retaining raw per-view output."""

    values = np.asarray(view_scores, dtype=np.float32)
    usable_values = np.asarray(view_usable, dtype=bool)
    weights_values = np.asarray(view_weights, dtype=np.float32)
    if (
        values.ndim != 3
        or usable_values.shape != values.shape
        or weights_values.shape != values.shape
        or values.shape[2] == 0
    ):
        raise ValueError("profile view aggregation layout is invalid")
    point_count, candidate_count, view_count = values.shape
    with torch.inference_mode():
        scores = torch.as_tensor(
            values.reshape(-1, view_count, 1), dtype=torch.float32, device=device
        )
        usable = torch.as_tensor(
            usable_values.reshape(-1, view_count, 1), dtype=torch.bool, device=device
        )
        weights = torch.as_tensor(
            weights_values.reshape(-1, view_count), dtype=torch.float32, device=device
        )
        mean, maximum, mass = coverage_weighted_view_appearance(
            view_scores=scores,
            view_usable=usable,
            view_weights=weights,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (
        mean[..., 0]
        .detach()
        .cpu()
        .numpy()
        .reshape(point_count, candidate_count)
        .astype(np.float32, copy=False),
        maximum[..., 0]
        .detach()
        .cpu()
        .numpy()
        .reshape(point_count, candidate_count)
        .astype(np.float32, copy=False),
        mass[..., 0]
        .detach()
        .cpu()
        .numpy()
        .reshape(point_count, candidate_count)
        .astype(np.float32, copy=False),
    )


def _input_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": file_sha256_short(path)}
        for name, path in paths.items()
    }


def build_frozen_multiscale_candidate_appearance(
    *,
    hypothesis_artifact: Path,
    baseline_score_artifact: Path | None,
    detector_query_cache: Path,
    proposals: Path,
    candidate_artifact: Path,
    fixed_candidate_prior_overlay: Path,
    maplet_support_index: Path,
    support_geometry_index: Path,
    projected_landmark_bank: Path,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    output: Path,
    summary_json: Path,
    profiles_value: str,
    fixed_candidate_top_k: int,
    verification_point_count: int,
    detector_log_merit_weight: float,
    template_batch_size: int,
    minimum_support_fraction: float,
    minimum_overlap_fraction: float,
    appearance_mode: str,
    region_grid_size: int,
    query_id: str | None,
    device_name: str,
    force: bool,
) -> dict[str, object]:
    """Build one target-free, exact-S0 query shard of raw appearance evidence."""

    output = Path(output)
    summary_json = Path(summary_json)
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite frozen appearance output")
    if (
        int(fixed_candidate_top_k) <= 0
        or int(verification_point_count) <= 0
        or int(template_batch_size) <= 0
        or not (0.0 < float(minimum_support_fraction) <= 1.0)
        or not (0.0 < float(minimum_overlap_fraction) <= 1.0)
        or str(appearance_mode) not in {"aligned_ncc", "region_layout"}
        or int(region_grid_size) <= 0
    ):
        raise ValueError("frozen appearance export arguments are invalid")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
    paths = {
        "hypothesis_artifact": Path(hypothesis_artifact),
        "detector_query_cache": Path(detector_query_cache),
        "proposals": Path(proposals),
        "candidate_artifact": Path(candidate_artifact),
        "fixed_candidate_prior_overlay": Path(fixed_candidate_prior_overlay),
        "maplet_support_index": Path(maplet_support_index),
        "support_geometry_index": Path(support_geometry_index),
        "projected_landmark_bank": Path(projected_landmark_bank),
        "radio_final_context_cache": Path(radio_final_context_cache),
        "radio_intermediate_context_cache": Path(radio_intermediate_context_cache),
        "alike_spatial_context_cache": Path(alike_spatial_context_cache),
    }
    if baseline_score_artifact is not None:
        paths["baseline_score_artifact"] = Path(baseline_score_artifact)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    started = time.time()
    query_identity_source: str
    baseline_metadata: dict[str, object] | None
    if "baseline_score_artifact" in paths:
        query_id, split_name, baseline_metadata = _load_baseline_query_identity(
            paths["baseline_score_artifact"], requested_query_id=query_id
        )
        _validate_baseline_score_metadata(
            baseline_metadata,
            hypothesis_path=paths["hypothesis_artifact"],
            detector_path=paths["detector_query_cache"],
            proposals_path=paths["proposals"],
            candidate_path=paths["candidate_artifact"],
            prior_path=paths["fixed_candidate_prior_overlay"],
            fixed_candidate_top_k=int(fixed_candidate_top_k),
        )
        query_identity_source = "strict_s0_score_artifact_v1"
    else:
        query_id, split_name, hypothesis_metadata = _load_hypothesis_query_identity(
            paths["hypothesis_artifact"], requested_query_id=query_id
        )
        _validate_hypothesis_input_lineage(
            hypothesis_metadata,
            candidate_path=paths["candidate_artifact"],
            proposals_path=paths["proposals"],
        )
        baseline_metadata = None
        query_identity_source = "inference_only_hypothesis_manifest_v1"
    detector, detector_metadata, _detector_names = _load_npz_allowlist(
        paths["detector_query_cache"], ("image_ids", "offsets", "xy", "detector_scores")
    )
    proposal_arrays, proposal_metadata, _proposal_names = _load_npz_allowlist(
        paths["proposals"], ("query_ids", "candidate_track_ids", "coarse_scores"), metadata_required=False
    )
    candidate_arrays, candidate_metadata, _candidate_names = _load_npz_allowlist(
        paths["candidate_artifact"], ("selected_rows",)
    )
    if (
        detector_metadata.get("format") != "alike_detector_mapped_radio_query_cache_v1"
        or candidate_metadata.get("contains_ground_truth") is not False
        or candidate_metadata.get("contains_pose_derived_selection") is not False
        or proposal_metadata.get("format") not in {None, "detector_support_reranked_proposals_v1"}
    ):
        raise ValueError("appearance source artifacts violate the target-free S0 contract")
    if (
        np.asarray(proposal_arrays["query_ids"]).shape != np.asarray(detector["xy"]).shape[:1]
        or np.asarray(proposal_arrays["candidate_track_ids"]).shape[0]
        != len(proposal_arrays["query_ids"])
        or np.asarray(proposal_arrays["coarse_scores"]).shape
        != np.asarray(proposal_arrays["candidate_track_ids"]).shape
    ):
        raise ValueError("detector and proposal inference arrays are incompatible")
    prior_overlay, prior_metadata = _load_candidate_prior_overlay(
        paths["fixed_candidate_prior_overlay"],
        proposals_path=paths["proposals"],
        proposals=proposal_arrays,
    )
    probabilities_all, null_all, _retained = _mask_fixed_candidate_posterior_topk(
        candidate_track_ids=prior_overlay["candidate_track_ids"],
        candidate_probabilities=prior_overlay["candidate_probabilities"],
        null_probabilities=prior_overlay["null_probabilities"],
        top_k=int(fixed_candidate_top_k),
    )
    kept_rows, selection_audit = _select_s0_verification_rows(
        query_id,
        detector=detector,
        proposals=proposal_arrays,
        selected_rows=np.asarray(candidate_arrays["selected_rows"], dtype=np.int64),
        point_count=int(verification_point_count),
        detector_log_merit_weight=float(detector_log_merit_weight),
    )
    candidate_tracks = np.asarray(proposal_arrays["candidate_track_ids"], dtype=np.int64)[
        kept_rows
    ]
    candidate_probabilities = np.asarray(probabilities_all, dtype=np.float32)[kept_rows]
    null_probabilities = np.asarray(null_all, dtype=np.float32)[kept_rows]
    query_xy = np.asarray(detector["xy"], dtype=np.float32)[kept_rows]
    if candidate_tracks.shape != (
        len(kept_rows),
        int(fixed_candidate_top_k),
    ) or not np.array_equal(
        candidate_tracks, np.asarray(prior_overlay["candidate_track_ids"], dtype=np.int64)[kept_rows]
    ):
        raise ValueError("absolute appearance builder requires exact frozen S0 top-20 tracks")
    bank_tracks, _bank_xyz, bank_metadata = _load_bank_xyz(paths["projected_landmark_bank"])
    bank_rows = _resolve_rows(candidate_tracks, canonical_track_ids=bank_tracks)
    if np.any((candidate_probabilities > 0.0) & (bank_rows < 0)):
        raise ValueError("positive-mass S0 candidate is absent from the projected landmark bank")
    maplet_tracks, maplet_image_ids, maplet_image_indices, maplet_coverage, maplet_metadata = (
        _load_maplet_support_fields(paths["maplet_support_index"])
    )
    if str(maplet_metadata.get("source_landmark_index_sha256", "")) != file_sha256_short(
        paths["projected_landmark_bank"]
    ):
        raise ValueError("maplet support index was built from a different landmark bank")
    views = _fixed_candidate_views(
        candidate_track_ids=candidate_tracks,
        candidate_probabilities=candidate_probabilities,
        maplet_track_ids=maplet_tracks,
        support_image_ids=maplet_image_ids,
        support_image_indices=maplet_image_indices,
        support_coverage_counts=maplet_coverage,
    )
    support_geometry, support_geometry_metadata = load_support_observation_geometry_index_npz(
        paths["support_geometry_index"]
    )
    sources = _as_image_grid_sources(
        radio_final_context_cache=paths["radio_final_context_cache"],
        radio_intermediate_context_cache=paths["radio_intermediate_context_cache"],
        alike_spatial_context_cache=paths["alike_spatial_context_cache"],
    )
    source_grid_sizes = {name: source.grid_size for name, source in sources.items()}
    profiles = parse_profiles(profiles_value, source_grid_sizes=source_grid_sizes)
    reference = sources["radio_final"]
    runtime = build_fixed_candidate_context_runtime(
        query_ids=np.full((len(kept_rows),), query_id),
        query_xy=query_xy,
        candidate_track_ids=candidate_tracks,
        candidate_support_image_ids=views.support_image_ids,
        candidate_view_valid=views.valid,
        cache_image_ids=reference.image_ids,
        support_geometry=support_geometry,
    )
    reference_grid, reference_size = reference.image_grid(query_id)
    del reference_grid
    image_width, image_height = (int(value) for value in reference_size)
    if (
        not np.all((query_xy[:, 0] >= 0.0) & (query_xy[:, 0] <= image_width - 1))
        or not np.all((query_xy[:, 1] >= 0.0) & (query_xy[:, 1] <= image_height - 1))
    ):
        raise ValueError("held-out S0 detector coordinates exceed the image cache bounds")

    view_scores: list[np.ndarray] = []
    view_usable: list[np.ndarray] = []
    candidate_means: list[np.ndarray] = []
    candidate_maxima: list[np.ndarray] = []
    candidate_usable_masses: list[np.ndarray] = []
    family_names: list[str] = []
    view_overlap: list[np.ndarray] = []
    view_support_fraction: list[np.ndarray] = []
    view_region_pairs: list[np.ndarray] = []
    for profile in profiles:
        source = sources[profile.source_name]
        if not np.array_equal(source.image_ids, reference.image_ids) or not np.array_equal(
            source.image_sizes, reference.image_sizes
        ):
            raise ValueError("appearance feature sources do not share image geometry")
        if str(appearance_mode) == "aligned_ncc":
            score, overlap, support_fraction, usable = _score_profile_views(
                profile=profile,
                source=source,
                query_id=query_id,
                query_xy=query_xy,
                views=views,
                runtime_support_xy=runtime.support_xy,
                template_batch_size=int(template_batch_size),
                minimum_support_fraction=float(minimum_support_fraction),
                minimum_overlap_fraction=float(minimum_overlap_fraction),
                device=device,
            )
            mean, maximum, mass = _aggregate_profile_views(
                view_scores=score,
                view_usable=usable,
                view_weights=views.weights,
                device=device,
            )
            family_names.append(profile.name)
            view_scores.append(score)
            view_overlap.append(overlap)
            view_support_fraction.append(support_fraction)
            view_usable.append(usable)
            candidate_means.append(mean)
            candidate_maxima.append(maximum)
            candidate_usable_masses.append(mass)
        else:
            score_values, usable_values, region_pairs = _score_region_layout_profile_views(
                profile=profile,
                source=source,
                query_id=query_id,
                query_xy=query_xy,
                views=views,
                runtime_support_xy=runtime.support_xy,
                template_batch_size=int(template_batch_size),
                region_grid_size=int(region_grid_size),
                device=device,
            )
            for statistic_index, statistic_name in enumerate(REGION_LAYOUT_STATISTIC_NAMES):
                score = score_values[..., statistic_index]
                usable = usable_values[..., statistic_index]
                mean, maximum, mass = _aggregate_profile_views(
                    view_scores=score,
                    view_usable=usable,
                    view_weights=views.weights,
                    device=device,
                )
                family_names.append(f"{profile.name}__{statistic_name}")
                view_scores.append(score)
                view_usable.append(usable)
                view_region_pairs.append(region_pairs)
                candidate_means.append(mean)
                candidate_maxima.append(maximum)
                candidate_usable_masses.append(mass)
    if not view_scores:
        raise RuntimeError("no frozen appearance profile was materialized")
    score_tensor = np.stack(view_scores, axis=3)
    usable_tensor = np.stack(view_usable, axis=3)
    mean_tensor = np.stack(candidate_means, axis=2)
    maximum_tensor = np.stack(candidate_maxima, axis=2)
    mass_tensor = np.stack(candidate_usable_masses, axis=2)
    if (
        score_tensor.shape != (*views.valid.shape, len(family_names))
        or usable_tensor.shape != score_tensor.shape
        or mean_tensor.shape != (*candidate_tracks.shape, len(family_names))
        or maximum_tensor.shape != mean_tensor.shape
        or mass_tensor.shape != mean_tensor.shape
        or np.any(~np.isfinite(score_tensor[usable_tensor]))
        or np.any(~np.isfinite(mean_tensor[mass_tensor > 0.0]))
        or np.any(~np.isfinite(maximum_tensor[mass_tensor > 0.0]))
        or np.any(mass_tensor < 0.0)
        or np.any(mass_tensor > 1.0001)
    ):
        raise RuntimeError("frozen appearance output tensors are invalid")

    arrays: dict[str, np.ndarray] = {
        "verification_query_ids": np.full((len(kept_rows),), query_id),
        "split_names": np.full((len(kept_rows),), split_name),
        "verification_source_row_indices": kept_rows.astype(np.int64, copy=False),
        "verification_xy": query_xy.astype(np.float32, copy=False),
        "candidate_track_ids": candidate_tracks.astype(np.int64, copy=False),
        "candidate_probabilities": candidate_probabilities.astype(np.float32, copy=False),
        "null_probabilities": null_probabilities.astype(np.float32, copy=False),
        "candidate_view_weights": np.asarray(views.weights, dtype=np.float32),
        "family_names": np.asarray(family_names, dtype=np.str_),
        "candidate_view_usable": usable_tensor.astype(bool, copy=False),
        "candidate_usable_view_weight_mass": mass_tensor.astype(np.float32, copy=False),
    }
    if str(appearance_mode) == "aligned_ncc":
        overlap_tensor = np.stack(view_overlap, axis=3)
        support_fraction_tensor = np.stack(view_support_fraction, axis=3)
        arrays.update(
            {
                "candidate_view_aligned_ncc": score_tensor.astype(np.float32, copy=False),
                "candidate_view_overlap_fraction": overlap_tensor.astype(
                    np.float32, copy=False
                ),
                "candidate_view_support_fraction": support_fraction_tensor.astype(
                    np.float32, copy=False
                ),
                "candidate_coverage_weighted_ncc": mean_tensor.astype(
                    np.float32, copy=False
                ),
                "candidate_max_view_ncc": maximum_tensor.astype(np.float32, copy=False),
            }
        )
    else:
        arrays.update(
            {
                "candidate_view_region_similarity": score_tensor.astype(
                    np.float32, copy=False
                ),
                "candidate_view_region_pair_count": np.stack(
                    view_region_pairs, axis=3
                ).astype(np.int16, copy=False),
                "candidate_coverage_weighted_region_similarity": mean_tensor.astype(
                    np.float32, copy=False
                ),
                "candidate_max_view_region_similarity": maximum_tensor.astype(
                    np.float32, copy=False
                ),
            }
        )
    implementation = {
        "script_sha256": file_sha256_short(Path(__file__)),
        "appearance_module_sha256": file_sha256_short(
            Path(
                "feature_extract/vfm/localization/frozen_multiscale_candidate_appearance.py"
                if str(appearance_mode) == "aligned_ncc"
                else "feature_extract/vfm/localization/frozen_multiscale_candidate_region_layout.py"
            )
        ),
        "shared_s0_handoff_script_sha256": file_sha256_short(
            Path("feature_extract/tools/vfm/score_frozen_multiscale_candidate_pose_evidence.py")
        ),
    }
    metadata: dict[str, Any] = {
        "format": (
            ARTIFACT_FORMAT
            if str(appearance_mode) == "aligned_ncc"
            else REGION_LAYOUT_ARTIFACT_FORMAT
        ),
        "version": (
            ARTIFACT_VERSION
            if str(appearance_mode) == "aligned_ncc"
            else FROZEN_MULTISCALE_CANDIDATE_REGION_LAYOUT_VERSION
        ),
        "contains_target_fields": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "query_id": query_id,
        "split_name": split_name,
        "fit_split_only": bool(split_name == "train"),
        "query_identity_source": query_identity_source,
        "row_count": int(len(kept_rows)),
        "query_count": 1,
        "source_field_allowlists": {
            "baseline_score_artifact": (
                None
                if "baseline_score_artifact" not in paths
                else ["query_ids", "split_names"]
            ),
            "hypothesis_artifact": ["query_ids", "split_names"],
            "detector_query_cache": ["image_ids", "offsets", "xy", "detector_scores"],
            "proposals": ["query_ids", "candidate_track_ids", "coarse_scores"],
            "candidate_artifact": ["selected_rows"],
            "projected_landmark_bank": ["track_ids", "xyz"],
            "maplet_support_index": [
                "anchor_track_ids",
                "support_image_ids",
                "support_image_indices",
                "support_coverage_counts",
            ],
        },
        "strict_frozen_appearance_contract": {
            "heldout_s0_verification_rows": True,
            "verification_row_selector": "s0_detector_merit_after_target_free_fit_rows_v1",
            "fixed_global_topl": True,
            "fixed_candidate_top_k": int(fixed_candidate_top_k),
            "candidate_posterior": "s480_explicit_null_no_renormalization_v1",
            "candidate_identity_fixed": True,
            "support_views": "maplet_track_support_views_fixed_before_appearance_v1",
            "support_view_weighting": "fixed_coverage_weighted_marginalization_v1",
            "support_view_descriptor_averaging": False,
            "query_support_patch_comparison": (
                "aligned_masked_per_cell_ncc_at_fixed_observation_anchors_v1"
                if str(appearance_mode) == "aligned_ncc"
                else "masked_landmark_centred_global_and_3x3_region_layout_cosine_v1"
            ),
            "candidate_3d_projection_or_pose_used": False,
            "missing_view_semantics": "explicit_nan_score_and_zero_usable_weight_mass_v1",
            "candidate_reselection": False,
            "support_reselection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
        "profiles": [
            {
                "name": profile.name,
                "source": profile.source_name,
                "window_size": int(profile.window_size),
            }
            for profile in profiles
        ],
        "appearance_config": {
            "appearance_mode": str(appearance_mode),
            "template_batch_size": int(template_batch_size),
            "minimum_support_fraction": float(minimum_support_fraction),
            "minimum_overlap_fraction": float(minimum_overlap_fraction),
            "region_grid_size": (
                None if str(appearance_mode) == "aligned_ncc" else int(region_grid_size)
            ),
            "descriptor_sampling": "bilinear_align_corners_true_real_image_grid_v1",
            "aggregation": "fixed_coverage_weighted_mean_and_max_raw_features_v1",
        },
        "selection_audit": selection_audit,
        "implementation": implementation,
        "implementation_hash": _canonical_hash(implementation),
        "inputs": _input_manifest(paths),
        "prior_overlay_metadata": {
            "format": prior_metadata.get("format"),
            "candidate_probability_semantics": prior_metadata.get(
                "candidate_probability_semantics"
            ),
        },
        "baseline_score_metadata": (
            None
            if baseline_metadata is None
            else {
                "format": baseline_metadata.get("format"),
                "fixed_candidate_topk_ablation": baseline_metadata.get(
                    "fixed_candidate_topk_ablation"
                ),
            }
        ),
        "projected_landmark_bank_metadata": {
            "descriptor_space_id": bank_metadata.get("descriptor_space_id"),
            "descriptor_space_manifest": bank_metadata.get("descriptor_space_manifest"),
        },
        "support_geometry_metadata": {
            "format": support_geometry_metadata.get("format"),
            "source_landmark_index_sha256": support_geometry_metadata.get(
                "source_landmark_index_sha256"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(output)
    summary = {
        "stage": (
            "build_frozen_multiscale_candidate_absolute_appearance"
            if str(appearance_mode) == "aligned_ncc"
            else "build_frozen_multiscale_candidate_region_layout"
        ),
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "format": metadata["format"],
        "query_id": query_id,
        "split_name": split_name,
        "verification_point_count": int(len(kept_rows)),
        "candidate_top_k": int(fixed_candidate_top_k),
        "family_count": int(len(family_names)),
        "usable_candidate_view_count_by_family": {
            name: int(np.sum(usable_tensor[..., index]))
            for index, name in enumerate(family_names)
        },
        "candidate_without_usable_view_count_by_family": {
            name: int(np.sum(mass_tensor[..., index] <= 0.0))
            for index, name in enumerate(family_names)
        },
        "protocol": metadata["strict_frozen_appearance_contract"],
        "elapsed_seconds": float(time.time() - started),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_frozen_multiscale_candidate_appearance(
        hypothesis_artifact=Path(args.hypothesis_artifact),
        baseline_score_artifact=(
            None
            if args.baseline_score_artifact is None
            else Path(args.baseline_score_artifact)
        ),
        detector_query_cache=Path(args.detector_query_cache),
        proposals=Path(args.proposals),
        candidate_artifact=Path(args.candidate_artifact),
        fixed_candidate_prior_overlay=Path(args.fixed_candidate_prior_overlay),
        maplet_support_index=Path(args.maplet_support_index),
        support_geometry_index=Path(args.support_geometry_index),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        profiles_value=str(args.profiles),
        fixed_candidate_top_k=int(args.fixed_candidate_top_k),
        verification_point_count=int(args.verification_point_count),
        detector_log_merit_weight=float(args.detector_log_merit_weight),
        template_batch_size=int(args.template_batch_size),
        minimum_support_fraction=float(args.minimum_support_fraction),
        minimum_overlap_fraction=float(args.minimum_overlap_fraction),
        appearance_mode=str(args.appearance_mode),
        region_grid_size=int(args.region_grid_size),
        query_id=(None if args.query_id is None else str(args.query_id)),
        device_name=str(args.device),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
