"""Score frozen S0 hypotheses with candidate-specific multiscale RGB evidence.

This is an inference-only S1 diagnostic.  It keeps the S0 top-20 landmark
posterior and held-out verifier rows immutable, obtains each candidate's fixed
SfM support observations, and evaluates a hypothesised pose only by where its
candidate 3-D tracks land in a *local*, anchor-centred RGB likelihood map.

The command deliberately has no target-artifact argument.  It loads every NPZ
through a field allowlist, so proposal-side GT audit arrays cannot silently
enter the scoring process.  A separate command must join its output with pose
targets for any rank or pose-quality conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    load_inference_artifact_fields,
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
    build_fixed_candidate_context_runtime,
    load_context_attention_sources,
)
from feature_extract.vfm.localization.frozen_multiscale_pose_evidence import (
    FROZEN_MULTISCALE_POSE_EVIDENCE_VERSION,
    LocalContextModeRatios,
    fixed_candidate_point_log_ratios,
    local_context_mode_ratios,
    materialize_dense_local_context_mode_maps,
    sample_dense_local_context_mode_ratios,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    ContextPositionLikelihoodMaps,
    ImageGridFeatureSource,
    build_context_position_likelihood_maps,
    project_simple_radial_torch,
)


SCORE_FORMAT = "frozen_multiscale_candidate_pose_scores_v1"
SCORE_VERSION = "s1_candidate_specific_local_rgb_v1"
_TARGET_FIELD_MARKERS = (
    "ground_truth",
    "gt_pose",
    "translation_error",
    "rotation_error",
    "target",
    "candidate_gt",
)


@dataclass(frozen=True)
class Profile:
    """A frozen source/template/local-search specification."""

    name: str
    source_name: str
    context_window_size: int
    local_window_size: int

    def __post_init__(self) -> None:
        if (
            not str(self.name)
            or not str(self.source_name)
            or int(self.context_window_size) <= 0
            or int(self.local_window_size) <= 0
            or int(self.context_window_size) % 2 != 1
            or int(self.local_window_size) % 2 != 1
        ):
            raise ValueError("multiscale profile is invalid")


@dataclass(frozen=True)
class FixedCandidateViews:
    """Candidate-specific support observations resolved from the maplet index."""

    support_image_ids: np.ndarray
    valid: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        image_ids = np.asarray(self.support_image_ids).astype(str)
        valid = np.asarray(self.valid, dtype=bool)
        weights = np.asarray(self.weights, dtype=np.float32)
        if (
            image_ids.ndim != 3
            or valid.shape != image_ids.shape
            or weights.shape != image_ids.shape
            or np.any(weights < 0.0)
            or np.any((~valid) & (weights != 0.0))
            or np.any(valid & (image_ids == ""))
        ):
            raise ValueError("fixed candidate support views are invalid")
        object.__setattr__(self, "support_image_ids", image_ids)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "weights", weights)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis_artifact", required=True)
    parser.add_argument("--baseline_score_artifact", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--fixed_candidate_prior_overlay", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--radio_intermediate_context_cache", required=True)
    parser.add_argument("--alike_spatial_context_cache", required=True)
    parser.add_argument(
        "--profiles",
        default=(
            "radio_final_center:radio_final:1:9;"
            "radio_final_context3:radio_final:3:9;"
            "radio_final_context5:radio_final:5:9;"
            "radio_intermediate_center:radio_intermediate:1:13;"
            "radio_intermediate_context5:radio_intermediate:5:13;"
            "radio_intermediate_context9:radio_intermediate:9:13;"
            "radio_intermediate_context13:radio_intermediate:13:13;"
            "alike_center:alike:1:13;"
            "alike_context3:alike:3:13;"
            "alike_context5:alike:5:13;"
            "alike_context9:alike:9:13"
        ),
        help="semicolon-separated name:source:context-window:local-window profiles",
    )
    parser.add_argument("--fixed_candidate_top_k", type=int, default=20)
    parser.add_argument("--verification_point_count", type=int, default=192)
    parser.add_argument("--detector_log_merit_weight", type=float, default=0.01)
    parser.add_argument("--context_temperature", type=float, default=0.10)
    parser.add_argument("--context_template_batch_size", type=int, default=96)
    parser.add_argument("--context_minimum_support_fraction", type=float, default=0.75)
    parser.add_argument("--context_minimum_query_overlap_fraction", type=float, default=0.75)
    parser.add_argument("--hypothesis_batch_size", type=int, default=128)
    parser.add_argument(
        "--hypothesis_limit",
        type=int,
        default=0,
        help="development-only stable prefix limit; zero scores every frozen S0 row",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _load_npz_allowlist(
    path: Path,
    fields: Sequence[str],
    *,
    metadata_required: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object], tuple[str, ...]]:
    """Read only explicitly inference-safe fields from an NPZ container."""

    requested = tuple(str(field) for field in fields)
    if len(requested) != len(set(requested)):
        raise ValueError("NPZ field allowlist contains duplicates")
    with np.load(Path(path), allow_pickle=False) as payload:
        names = tuple(payload.files)
        missing = sorted(set(requested).difference(names))
        if missing:
            raise ValueError(f"{path}: missing required fields {missing}")
        unsafe_requested = [
            field
            for field in requested
            if any(marker in field.lower() for marker in _TARGET_FIELD_MARKERS)
        ]
        if unsafe_requested:
            raise ValueError(f"target-bearing fields were requested: {unsafe_requested}")
        arrays = {field: np.asarray(payload[field]).copy() for field in requested}
        if "metadata_json" not in names and metadata_required:
            raise ValueError(f"{path}: metadata_json is required")
        metadata = (
            {}
            if "metadata_json" not in names
            else json.loads(str(np.asarray(payload["metadata_json"]).item()))
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata_json must decode to an object")
    return arrays, metadata, names


def _parse_profiles(value: str, *, source_grid_sizes: Mapping[str, int]) -> tuple[Profile, ...]:
    profiles: list[Profile] = []
    for raw in (item.strip() for item in str(value).split(";")):
        if not raw:
            continue
        pieces = tuple(item.strip() for item in raw.split(":"))
        if len(pieces) != 4:
            raise ValueError("each profile must be name:source:context-window:local-window")
        try:
            profile = Profile(
                name=pieces[0],
                source_name=pieces[1],
                context_window_size=int(pieces[2]),
                local_window_size=int(pieces[3]),
            )
        except ValueError as error:
            raise ValueError(f"invalid profile {raw!r}") from error
        grid_size = source_grid_sizes.get(profile.source_name)
        if grid_size is None:
            raise ValueError(f"profile {profile.name} references an unknown source")
        if (
            profile.context_window_size > int(grid_size)
            or profile.local_window_size > int(grid_size)
        ):
            raise ValueError(f"profile {profile.name} exceeds {profile.source_name} grid")
        profiles.append(profile)
    if not profiles or len({profile.name for profile in profiles}) != len(profiles):
        raise ValueError("profiles must be non-empty with unique names")
    return tuple(profiles)


def _validate_baseline_score_metadata(
    metadata: Mapping[str, object],
    *,
    hypothesis_path: Path,
    detector_path: Path,
    proposals_path: Path,
    candidate_path: Path,
    prior_path: Path,
    fixed_candidate_top_k: int,
) -> None:
    """Require the exact S0 frozen-row and posterior lineage, not just dimensions."""

    if (
        metadata.get("format") != "independent_landmark_hypothesis_scores_v1"
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_scoring") is not False
        or metadata.get("supervision_arrays_loaded") is not False
    ):
        raise ValueError("baseline score artifact is not an inference-only S0 score")
    contract = metadata.get("strict_absolute_evidence_contract")
    if not isinstance(contract, Mapping) or any(
        contract.get(key) is not True
        for key in (
            "heldout_query_rows",
            "fixed_global_topl",
            "explicit_null_mass",
            "identity_prior_fixed_across_hypotheses",
            "verification_point_selector_fixed_across_hypotheses",
        )
    ):
        raise ValueError("baseline score artifact does not use frozen S0 evidence")
    topk = contract.get("fixed_candidate_topk_ablation")
    if (
        not isinstance(topk, Mapping)
        or topk.get("applied") is not True
        or int(topk.get("top_k", -1)) != int(fixed_candidate_top_k)
        or topk.get("removed_candidate_mass_transferred_to_null") is not True
    ):
        raise ValueError("baseline score artifact uses a different fixed candidate top-K")
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("baseline score artifact lacks its input manifest")
    expected = {
        "hypothesis_artifact_sha256": [file_sha256_short(hypothesis_path)],
        "detector_query_cache_sha256": file_sha256_short(detector_path),
        "proposals_sha256": file_sha256_short(proposals_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "fixed_candidate_prior_overlay_sha256": file_sha256_short(prior_path),
    }
    for key, value in expected.items():
        if inputs.get(key) != value:
            raise ValueError(f"baseline score lineage mismatch for {key}")


def _select_s0_verification_rows(
    query_id: str,
    *,
    detector: Mapping[str, np.ndarray],
    proposals: Mapping[str, np.ndarray],
    selected_rows: np.ndarray,
    point_count: int,
    detector_log_merit_weight: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Reproduce S0's held-out selector without loading an unused descriptor."""

    image_ids = np.asarray(detector["image_ids"]).astype(str).reshape(-1)
    offsets = np.asarray(detector["offsets"], dtype=np.int64).reshape(-1)
    matches = np.flatnonzero(image_ids == str(query_id))
    if len(matches) != 1 or offsets.shape != (len(image_ids) + 1,):
        raise ValueError("detector cache does not uniquely contain the query")
    image_index = int(matches[0])
    all_rows = np.arange(int(offsets[image_index]), int(offsets[image_index + 1]))
    proposal_query_ids = np.asarray(proposals["query_ids"]).astype(str).reshape(-1)
    if proposal_query_ids.shape != detector["xy"].shape[:1] or not np.all(
        proposal_query_ids[all_rows] == str(query_id)
    ):
        raise ValueError("proposal and detector row ownership differs")
    fit_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    fit_rows = fit_rows[proposal_query_ids[fit_rows] == str(query_id)]
    if len(np.unique(fit_rows)) != len(fit_rows):
        raise ValueError("candidate artifact repeats fit rows")
    unused_rows = np.setdiff1d(all_rows, fit_rows, assume_unique=True)
    if np.intersect1d(unused_rows, fit_rows).size:
        raise RuntimeError("fit and verification query rows overlap")
    coarse = np.asarray(proposals["coarse_scores"], dtype=np.float64)[unused_rows]
    detector_scores = np.asarray(detector["detector_scores"], dtype=np.float64)[unused_rows]
    merit = np.max(coarse, axis=1) + float(detector_log_merit_weight) * np.log(
        np.maximum(detector_scores, 1e-12)
    )
    keep = min(int(point_count), int(len(unused_rows)))
    kept = unused_rows[np.argsort(-merit, kind="mergesort")[:keep]]
    if len(kept) == 0 or len(np.unique(kept)) != len(kept):
        raise ValueError("S0 verification selector produced no unique rows")
    return kept.astype(np.int64, copy=False), {
        "fit_query_point_count": int(len(fit_rows)),
        "available_unused_query_point_count": int(len(unused_rows)),
        "selected_verification_point_count": int(len(kept)),
    }


def _load_bank_xyz(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    arrays, metadata, _names = _load_npz_allowlist(path, ("track_ids", "xyz"))
    tracks = np.asarray(arrays["track_ids"], dtype=np.int64).reshape(-1)
    xyz = np.asarray(arrays["xyz"], dtype=np.float32).reshape(-1, 3)
    if (
        metadata.get("format") != "landmark_map_index_npz"
        or len(tracks) == 0
        or len(np.unique(tracks)) != len(tracks)
        or xyz.shape != (len(tracks), 3)
        or np.any(~np.isfinite(xyz))
    ):
        raise ValueError("projected landmark bank has invalid track geometry")
    return tracks, xyz, metadata


def _load_maplet_support_fields(
    path: Path,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, dict[str, object]]:
    arrays, metadata, _names = _load_npz_allowlist(
        path,
        (
            "anchor_track_ids",
            "support_image_ids",
            "support_image_indices",
            "support_coverage_counts",
        ),
    )
    tracks = np.asarray(arrays["anchor_track_ids"], dtype=np.int64).reshape(-1)
    image_ids = tuple(str(value) for value in np.asarray(arrays["support_image_ids"]).tolist())
    indices = np.asarray(arrays["support_image_indices"], dtype=np.int64)
    coverage = np.asarray(arrays["support_coverage_counts"], dtype=np.int64)
    if (
        metadata.get("format") != "local_maplet_support_index_npz"
        or len(tracks) == 0
        or len(np.unique(tracks)) != len(tracks)
        or len(set(image_ids)) != len(image_ids)
        or indices.ndim != 2
        or indices.shape[0] != len(tracks)
        or coverage.shape != indices.shape
        or np.any((indices < -1) | (indices >= len(image_ids)))
        or np.any(coverage < 0)
        or np.any((indices < 0) & (coverage != 0))
    ):
        raise ValueError("maplet support index fields are invalid")
    return tracks, image_ids, indices, coverage, metadata


def _resolve_rows(
    requested_track_ids: np.ndarray,
    *,
    canonical_track_ids: np.ndarray,
) -> np.ndarray:
    """Resolve fixed physical tracks without accepting an accidental nearest row."""

    requested = np.asarray(requested_track_ids, dtype=np.int64)
    canonical = np.asarray(canonical_track_ids, dtype=np.int64).reshape(-1)
    order = np.argsort(canonical, kind="stable")
    sorted_tracks = canonical[order]
    output = np.full(requested.shape, -1, dtype=np.int64)
    valid = requested >= 0
    if not np.any(valid):
        return output
    positions = np.searchsorted(sorted_tracks, requested[valid])
    clipped = np.minimum(positions, len(sorted_tracks) - 1)
    if np.any(positions >= len(sorted_tracks)) or not np.array_equal(
        sorted_tracks[clipped], requested[valid]
    ):
        raise ValueError("a frozen candidate track is absent from the declared bank")
    output[valid] = order[positions]
    return output


def _fixed_candidate_views(
    *,
    candidate_track_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    maplet_track_ids: np.ndarray,
    support_image_ids: Sequence[str],
    support_image_indices: np.ndarray,
    support_coverage_counts: np.ndarray,
) -> FixedCandidateViews:
    """Attach the predeclared per-track support observations to every point."""

    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    if tracks.ndim != 2 or probabilities.shape != tracks.shape:
        raise ValueError("candidate tracks and posterior probabilities are incompatible")
    maplet_rows = _resolve_rows(tracks, canonical_track_ids=maplet_track_ids)
    view_count = int(support_image_indices.shape[1])
    view_indices = np.full((*tracks.shape, view_count), -1, dtype=np.int64)
    coverage = np.zeros((*tracks.shape, view_count), dtype=np.int64)
    valid_track = maplet_rows >= 0
    if np.any(valid_track):
        view_indices[valid_track] = support_image_indices[maplet_rows[valid_track]]
        coverage[valid_track] = support_coverage_counts[maplet_rows[valid_track]]
    valid = (view_indices >= 0) & (coverage > 0)
    source_ids = np.asarray(tuple(str(item) for item in support_image_ids), dtype=np.str_)
    output_ids = np.full(view_indices.shape, "", dtype=source_ids.dtype)
    output_ids[view_indices >= 0] = source_ids[view_indices[view_indices >= 0]]
    required = probabilities > 0.0
    if np.any(required & ~np.any(valid, axis=2)):
        raise ValueError("a positive-mass candidate has no fixed usable support observation")
    weights = np.where(valid, coverage, 0.0).astype(np.float32)
    normalizer = np.sum(weights, axis=2, keepdims=True)
    weights = np.divide(weights, normalizer, out=np.zeros_like(weights), where=normalizer > 0.0)
    weights = np.where(required[..., None], weights, 0.0).astype(np.float32, copy=False)
    if np.any(np.abs(weights.sum(axis=2) - required.astype(np.float32)) > 2e-5):
        raise RuntimeError("fixed support-view weights do not preserve candidate mass")
    return FixedCandidateViews(
        support_image_ids=output_ids,
        valid=valid,
        weights=weights,
    )


def _as_image_grid_sources(
    *,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    allow_mismatched_descriptor_dimensions: bool = False,
) -> dict[str, ImageGridFeatureSource]:
    """Load aligned C-RADIO/ALIKE grids with their mapping-only PCA lineage.

    By default, preserve the equal-width requirement used by attention-based
    multiscale paths.  The full-track raw-NCC diagnostic may opt into separate
    descriptor widths because it never fuses vectors across source spaces.
    """

    sources = load_context_attention_sources(
        radio_final_context_cache=radio_final_context_cache,
        radio_intermediate_context_cache=radio_intermediate_context_cache,
        alike_spatial_context_cache=alike_spatial_context_cache,
        # This branch is intentionally C-RADIO/ALIKE evidence, not mapper
        # descriptors. The loader still checks final/intermediate cache lineage.
        expected_radio_checkpoint="",
        require_equal_descriptor_dimensions=not bool(allow_mismatched_descriptor_dimensions),
    )
    output: dict[str, ImageGridFeatureSource] = {}
    for source in sources:
        output[source.name] = ImageGridFeatureSource(
            name=source.name,
            image_ids=source.image_ids,
            image_sizes=source.image_sizes,
            grid_size=source.spatial_grid_size,
            descriptors=np.asarray(source.grid, dtype=np.float32).reshape(
                len(source.image_ids), -1, source.descriptor_dim
            ),
            metadata=source.metadata,
        )
    if set(output) != {"radio_final", "radio_intermediate", "alike"}:
        raise RuntimeError("the multiscale source set is incomplete")
    return output


def _concat_modes(parts: Sequence[LocalContextModeRatios]) -> LocalContextModeRatios:
    if not parts:
        raise ValueError("at least one local context mode chunk is required")
    return LocalContextModeRatios(
        log_ratios=torch.cat([part.log_ratios for part in parts], dim=0),
        valid_cells=torch.cat([part.valid_cells for part in parts], dim=0),
        grid_rows=torch.cat([part.grid_rows for part in parts], dim=0),
        grid_columns=torch.cat([part.grid_columns for part in parts], dim=0),
        template_usable=torch.cat([part.template_usable for part in parts], dim=0),
    )


def _build_profile_maps(
    *,
    profile: Profile,
    source: ImageGridFeatureSource,
    query_id: str,
    query_xy: np.ndarray,
    runtime_support_image_ids: np.ndarray,
    runtime_support_xy: np.ndarray,
    support_valid: np.ndarray,
    context_temperature: float,
    context_template_batch_size: int,
    context_minimum_support_fraction: float,
    context_minimum_query_overlap_fraction: float,
    device: torch.device,
) -> tuple[
    object,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Build one query's fixed per-view local likelihood maps for a profile."""

    if int(context_template_batch_size) <= 0:
        raise ValueError("context template batch size must be positive")
    valid = np.asarray(support_valid, dtype=bool)
    support_ids = np.asarray(runtime_support_image_ids).astype(str)
    support_xy = np.asarray(runtime_support_xy, dtype=np.float32)
    if support_ids.shape != valid.shape or support_xy.shape != (*valid.shape, 2):
        raise ValueError("fixed support arrays are incompatible")
    point_indices, candidate_indices, view_indices = np.nonzero(valid)
    if len(point_indices) == 0:
        raise ValueError("profile has no positive-mass fixed support views")
    query_grid_np, query_size = source.image_grid(str(query_id))
    width, height = (int(value) for value in np.asarray(query_size, dtype=np.int64))
    if width <= 1 or height <= 1:
        raise ValueError("query image has invalid cache geometry")
    query_grid = torch.as_tensor(query_grid_np, dtype=torch.float32, device=device)
    chunks: list[LocalContextModeRatios] = []
    source_grid_cache: dict[str, torch.Tensor] = {}
    flat_ids = support_ids[point_indices, candidate_indices, view_indices]
    flat_xy = support_xy[point_indices, candidate_indices, view_indices]
    anchor_xy = np.asarray(query_xy, dtype=np.float32)[point_indices]
    for begin in range(0, len(flat_ids), int(context_template_batch_size)):
        end = min(begin + int(context_template_batch_size), len(flat_ids))
        patches, patch_valid = source.context_patches_torch(
            flat_ids[begin:end],
            flat_xy[begin:end],
            window_size=profile.context_window_size,
            device=device,
            source_grid_cache=source_grid_cache,
        )
        maps = build_context_position_likelihood_maps(
            query_grid=query_grid,
            support_patches=patches,
            support_patch_valid=patch_valid,
            temperature=float(context_temperature),
            minimum_support_fraction=float(context_minimum_support_fraction),
            minimum_query_overlap_fraction=float(context_minimum_query_overlap_fraction),
        )
        chunks.append(
            local_context_mode_ratios(
                maps=maps,
                anchor_xy=torch.as_tensor(anchor_xy[begin:end], dtype=torch.float32, device=device),
                image_width=width,
                image_height=height,
                local_window_size=profile.local_window_size,
            )
        )
    modes = _concat_modes(chunks)
    dense = materialize_dense_local_context_mode_maps(
        modes=modes, grid_size=source.grid_size
    )
    peak = torch.where(
        modes.valid_cells,
        modes.log_ratios,
        torch.full_like(modes.log_ratios, -torch.inf),
    ).max(dim=1).values
    peak = torch.where(modes.template_usable, peak, torch.zeros_like(peak))
    return (
        dense,
        (point_indices.astype(np.int64), candidate_indices.astype(np.int64), view_indices.astype(np.int64)),
        peak.detach().cpu().numpy().astype(np.float32, copy=False),
        modes.template_usable.detach().cpu().numpy().astype(bool, copy=False),
    )


def _spatial_2x2_statistic(
    values: torch.Tensor,
    *,
    query_xy: np.ndarray,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Median of fixed 2x2 point-cell means, matching S0's robust summary idea."""

    coordinates = np.asarray(query_xy, dtype=np.float32)
    if values.ndim != 2 or coordinates.shape != (values.shape[1], 2):
        raise ValueError("spatial summary inputs are incompatible")
    columns = np.minimum(1, np.floor(coordinates[:, 0] / float(image_width) * 2.0).astype(np.int64))
    rows = np.minimum(1, np.floor(coordinates[:, 1] / float(image_height) * 2.0).astype(np.int64))
    means: list[torch.Tensor] = []
    for cell in range(4):
        mask = rows * 2 + columns == cell
        if np.any(mask):
            indices = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=values.device)
            means.append(values.index_select(1, indices).mean(dim=1))
    if not means:
        raise RuntimeError("fixed verification points have no spatial cells")
    return torch.stack(means, dim=1).median(dim=1).values


def _score_profile_hypotheses(
    *,
    dense_modes: object,
    template_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    template_peak_log_ratios: np.ndarray,
    template_usable: np.ndarray,
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
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Project each frozen pose into maps whose support and denominator are fixed."""

    if int(getattr(camera, "model_id")) != 2:
        raise ValueError("S1 local RGB scorer requires COLMAP SIMPLE_RADIAL cameras")
    params = tuple(float(value) for value in getattr(camera, "params"))
    if len(params) != 4 or int(hypothesis_batch_size) <= 0:
        raise ValueError("camera or hypothesis batching configuration is invalid")
    xyz = np.asarray(candidate_xyz, dtype=np.float32)
    probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
    weights = np.asarray(candidate_view_weights, dtype=np.float32)
    if (
        xyz.ndim != 3
        or xyz.shape[2] != 3
        or probabilities.shape != xyz.shape[:2]
        or null.shape != (xyz.shape[0],)
        or weights.shape[:2] != xyz.shape[:2]
        or np.any(~np.isfinite(xyz))
    ):
        raise ValueError("frozen candidate geometry is incompatible")
    point_count, candidate_count, view_count = weights.shape
    if candidate_xyz.shape[:2] != (point_count, candidate_count):
        raise ValueError("candidate geometry and support weights disagree")
    point_indices, candidate_indices, view_indices = template_indices
    template_count = len(point_indices)
    if not (
        len(candidate_indices) == template_count
        and len(view_indices) == template_count
        and template_peak_log_ratios.shape == (template_count,)
        and template_usable.shape == (template_count,)
    ):
        raise ValueError("profile templates are not aligned")
    if not np.array_equal(
        np.asarray(dense_modes.template_usable.detach().cpu().numpy(), dtype=bool),
        template_usable,
    ):
        raise ValueError("profile template usability differs from the materialized maps")
    point_tensor = torch.as_tensor(point_indices, dtype=torch.long, device=device)
    candidate_tensor = torch.as_tensor(candidate_indices, dtype=torch.long, device=device)
    view_tensor = torch.as_tensor(view_indices, dtype=torch.long, device=device)
    xyz_tensor = torch.as_tensor(xyz.reshape(-1, 3), dtype=torch.float32, device=device)
    probability_tensor = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
    null_tensor = torch.as_tensor(null, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    outputs: dict[str, list[np.ndarray]] = {
        "means": [],
        "medians": [],
        "worst_quartile_means": [],
        "spatial_median_of_means_2x2": [],
        "effective_point_counts": [],
        "effective_view_masses": [],
    }
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
        template_projected = projected[:, point_tensor, candidate_tensor]
        template_visible = projection_valid[:, point_tensor, candidate_tensor]
        sampled_logs, sampled_available, sampled_in_window = sample_dense_local_context_mode_ratios(
            maps=dense_modes,
            projected_xy=template_projected,
            projection_valid=template_visible,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        batch_shape = (end - begin, point_count, candidate_count, view_count)
        view_logs = torch.zeros(batch_shape, dtype=torch.float32, device=device)
        view_available = torch.zeros(batch_shape, dtype=torch.bool, device=device)
        view_in_window = torch.zeros(batch_shape, dtype=torch.bool, device=device)
        view_logs[:, point_tensor, candidate_tensor, view_tensor] = sampled_logs
        view_available[:, point_tensor, candidate_tensor, view_tensor] = sampled_available
        view_in_window[:, point_tensor, candidate_tensor, view_tensor] = sampled_in_window
        point_logs, _candidate_ratios, contributed = fixed_candidate_point_log_ratios(
            view_log_ratios=view_logs,
            view_available=view_available,
            view_geometric_in_window=view_in_window,
            candidate_view_weights=weight_tensor,
            candidate_probabilities=probability_tensor,
            null_probabilities=null_tensor,
        )
        sorted_logs = torch.sort(point_logs, dim=1).values
        quartile_count = max(1, int(np.ceil(point_count * 0.25)))
        summaries = {
            "means": point_logs.mean(dim=1),
            "medians": point_logs.median(dim=1).values,
            "worst_quartile_means": sorted_logs[:, :quartile_count].mean(dim=1),
            "spatial_median_of_means_2x2": _spatial_2x2_statistic(
                point_logs,
                query_xy=query_xy,
                image_width=int(image_width),
                image_height=int(image_height),
            ),
            "effective_point_counts": (contributed.sum(dim=2) > 0.0).sum(dim=1),
            "effective_view_masses": contributed.sum(dim=(1, 2)),
        }
        for name, value in summaries.items():
            outputs[name].append(value.detach().cpu().numpy())
    result = {
        name: np.concatenate(values, axis=0).astype(
            np.int64 if name == "effective_point_counts" else np.float64,
            copy=False,
        )
        for name, values in outputs.items()
    }
    static_peak = np.zeros((point_count, candidate_count, view_count), dtype=np.float32)
    static_usable = np.zeros((point_count, candidate_count, view_count), dtype=bool)
    static_peak[point_indices, candidate_indices, view_indices] = template_peak_log_ratios
    static_usable[point_indices, candidate_indices, view_indices] = template_usable
    return result, static_peak, static_usable


def _load_exact_hypotheses(
    *,
    hypothesis_path: Path,
    baseline_path: Path,
    detector_path: Path,
    proposals_path: Path,
    candidate_path: Path,
    prior_path: Path,
    fixed_candidate_top_k: int,
    query_id: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    """Join S0 rows to source poses by exact immutable row identity.

    Normal scoring artifacts contain one query.  The optional ``query_id`` is
    reserved for train-only OOF scoring, where an otherwise immutable shard
    can contain several query groups.  Selection happens before the exact
    pose join so no prefix truncation can silently cross query boundaries.
    """

    baseline, baseline_metadata, _names = _load_npz_allowlist(
        baseline_path,
        (
            "query_ids",
            "split_names",
            "evaluation_labels",
            "hypothesis_indices",
            "source_chosen_for_optional_pose",
            "independent_score_top1",
            "independent_selection_scores",
        ),
    )
    _validate_baseline_score_metadata(
        baseline_metadata,
        hypothesis_path=hypothesis_path,
        detector_path=detector_path,
        proposals_path=proposals_path,
        candidate_path=candidate_path,
        prior_path=prior_path,
        fixed_candidate_top_k=fixed_candidate_top_k,
    )
    if query_id is not None:
        requested_query_id = str(query_id).strip()
        if not requested_query_id:
            raise ValueError("requested exact hypothesis query ID is empty")
        baseline_query_ids = np.asarray(baseline["query_ids"]).astype(str).reshape(-1)
        rows = np.flatnonzero(baseline_query_ids == requested_query_id).astype(np.int64)
        if len(rows) == 0:
            raise ValueError("requested query is absent from baseline score artifact")
        baseline = {
            name: np.asarray(value)[rows].copy()
            for name, value in baseline.items()
        }
    hypotheses, hypothesis_metadata = load_inference_artifact_fields(
        hypothesis_path, ("poses_w2c",)
    )
    source_keys = list(
        zip(
            np.asarray(hypotheses["query_ids"]).astype(str).tolist(),
            np.asarray(hypotheses["split_names"]).astype(str).tolist(),
            np.asarray(hypotheses["evaluation_labels"]).astype(str).tolist(),
            np.asarray(hypotheses["hypothesis_indices"], dtype=np.int64).tolist(),
        )
    )
    source_rows = {key: row for row, key in enumerate(source_keys)}
    requested_keys = list(
        zip(
            np.asarray(baseline["query_ids"]).astype(str).tolist(),
            np.asarray(baseline["split_names"]).astype(str).tolist(),
            np.asarray(baseline["evaluation_labels"]).astype(str).tolist(),
            np.asarray(baseline["hypothesis_indices"], dtype=np.int64).tolist(),
        )
    )
    if len(requested_keys) != len(set(requested_keys)):
        raise ValueError("baseline score artifact repeats exact hypothesis keys")
    try:
        rows = np.asarray([source_rows[key] for key in requested_keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError("baseline score row is absent from its hypothesis artifact") from error
    output = {
        key: np.asarray(value).copy()
        for key, value in baseline.items()
    }
    output["poses_w2c"] = np.asarray(hypotheses["poses_w2c"], dtype=np.float64)[rows]
    if not np.array_equal(
        np.asarray(hypotheses["split_names"]).astype(str)[rows],
        np.asarray(output["split_names"]).astype(str),
    ):
        raise ValueError("baseline and source hypothesis split identities differ")
    query_ids = np.unique(np.asarray(output["query_ids"]).astype(str))
    if len(query_ids) != 1:
        raise ValueError("one S1 scoring shard must contain exactly one query image")
    if query_id is not None and str(query_ids[0]) != str(query_id):
        raise RuntimeError("exact hypothesis query selection is inconsistent")
    if len(output["poses_w2c"]) == 0:
        raise ValueError("S1 source shard contains no frozen S0 hypotheses")
    return output, hypothesis_metadata, baseline_metadata


def _input_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": file_sha256_short(path)}
        for name, path in paths.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.time()
    output_path = Path(args.output)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"output already exists: {output_path}")
    if int(args.fixed_candidate_top_k) <= 0 or int(args.verification_point_count) <= 0:
        raise ValueError("fixed candidate top-K and verification point count must be positive")
    if int(args.hypothesis_limit) < 0:
        raise ValueError("hypothesis limit must be non-negative")
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
        "colmap_cameras_bin": Path(args.colmap_model_dir) / "cameras.bin",
        "colmap_images_bin_camera_ownership_only": Path(args.colmap_model_dir) / "images.bin",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

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
        raise ValueError("S1 scorer accepts only held-out validation/test query shards")

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
        raise ValueError("S1 source artifacts violate the target-free frozen-row contract")
    if (
        np.asarray(proposals["query_ids"]).shape != np.asarray(detector["xy"]).shape[:1]
        or np.asarray(proposals["candidate_track_ids"]).shape[0] != len(proposals["query_ids"])
        or np.asarray(proposals["coarse_scores"]).shape != np.asarray(proposals["candidate_track_ids"]).shape
    ):
        raise ValueError("detector/proposal inference arrays are incompatible")

    prior_overlay, prior_metadata = _load_candidate_prior_overlay(
        paths["fixed_candidate_prior_overlay"],
        proposals_path=paths["proposals"],
        proposals=proposals,
    )
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
    if candidate_tracks.shape[1] != int(args.fixed_candidate_top_k):
        raise ValueError("S1 requires the complete fixed top-20 candidate layout")
    if not np.array_equal(candidate_tracks, prior_overlay["candidate_track_ids"][kept_rows]):
        raise ValueError("fixed candidate overlay tracks differ at held-out rows")

    bank_tracks, bank_xyz, bank_metadata = _load_bank_xyz(paths["projected_landmark_bank"])
    maplet_tracks, maplet_image_ids, maplet_image_indices, maplet_coverage, maplet_metadata = _load_maplet_support_fields(
        paths["maplet_support_index"]
    )
    if (
        str(maplet_metadata.get("source_landmark_index_sha256", ""))
        != file_sha256_short(paths["projected_landmark_bank"])
    ):
        raise ValueError("maplet support index was built from a different landmark bank")
    candidate_bank_rows = _resolve_rows(candidate_tracks, canonical_track_ids=bank_tracks)
    candidate_xyz = np.zeros((*candidate_tracks.shape, 3), dtype=np.float32)
    valid_tracks = candidate_bank_rows >= 0
    candidate_xyz[valid_tracks] = bank_xyz[candidate_bank_rows[valid_tracks]]
    if np.any((candidate_probabilities > 0.0) & ~valid_tracks):
        raise ValueError("positive-mass candidate lacks a landmark-bank 3-D point")
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
    profiles = _parse_profiles(args.profiles, source_grid_sizes=source_grid_sizes)
    reference_source = sources["radio_final"]
    runtime = build_fixed_candidate_context_runtime(
        query_ids=np.full((len(kept_rows),), query_id),
        query_xy=query_xy,
        candidate_track_ids=candidate_tracks,
        candidate_support_image_ids=views.support_image_ids,
        candidate_view_valid=views.valid,
        cache_image_ids=reference_source.image_ids,
        support_geometry=support_geometry,
    )
    reference_query_grid, reference_size = reference_source.image_grid(query_id)
    del reference_query_grid
    image_width, image_height = (int(value) for value in reference_size)
    if not np.all((query_xy[:, 0] >= 0.0) & (query_xy[:, 0] <= float(image_width - 1))) or not np.all(
        (query_xy[:, 1] >= 0.0) & (query_xy[:, 1] <= float(image_height - 1))
    ):
        raise ValueError("held-out query detector coordinates exceed cache image bounds")
    cameras = read_colmap_cameras_binary(paths["colmap_cameras_bin"])
    image_camera_ids = read_colmap_image_camera_ids_binary(
        paths["colmap_images_bin_camera_ownership_only"]
    )
    camera_id = image_camera_ids.get(query_id)
    if camera_id is None or int(camera_id) not in cameras:
        raise ValueError("query image has no declared COLMAP camera ownership")
    camera = cameras[int(camera_id)]
    if (int(camera.width), int(camera.height)) != (image_width, image_height):
        raise ValueError("query cache geometry differs from its COLMAP camera")

    family_stats: list[dict[str, np.ndarray]] = []
    family_peaks: list[np.ndarray] = []
    family_usable: list[np.ndarray] = []
    for profile in profiles:
        source = sources[profile.source_name]
        if not np.array_equal(source.image_ids, reference_source.image_ids) or not np.array_equal(
            source.image_sizes, reference_source.image_sizes
        ):
            raise ValueError("multiscale sources do not share image ownership and geometry")
        dense, template_indices, peak, usable = _build_profile_maps(
            profile=profile,
            source=source,
            query_id=query_id,
            query_xy=query_xy,
            runtime_support_image_ids=views.support_image_ids,
            runtime_support_xy=runtime.support_xy,
            support_valid=views.valid,
            context_temperature=float(args.context_temperature),
            context_template_batch_size=int(args.context_template_batch_size),
            context_minimum_support_fraction=float(args.context_minimum_support_fraction),
            context_minimum_query_overlap_fraction=float(args.context_minimum_query_overlap_fraction),
            device=device,
        )
        stats, static_peak, static_usable = _score_profile_hypotheses(
            dense_modes=dense,
            template_indices=template_indices,
            template_peak_log_ratios=peak,
            template_usable=usable,
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
        family_stats.append(stats)
        family_peaks.append(static_peak)
        family_usable.append(static_usable)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

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
        "family_names": np.asarray([profile.name for profile in profiles], dtype=np.str_),
        "family_log_likelihood_means": np.stack(
            [item["means"] for item in family_stats], axis=1
        ),
        "family_log_likelihood_medians": np.stack(
            [item["medians"] for item in family_stats], axis=1
        ),
        "family_log_likelihood_worst_quartile_means": np.stack(
            [item["worst_quartile_means"] for item in family_stats], axis=1
        ),
        "family_spatial_median_of_means_2x2": np.stack(
            [item["spatial_median_of_means_2x2"] for item in family_stats], axis=1
        ),
        "family_effective_point_counts": np.stack(
            [item["effective_point_counts"] for item in family_stats], axis=1
        ),
        "family_effective_view_masses": np.stack(
            [item["effective_view_masses"] for item in family_stats], axis=1
        ),
        "verification_query_ids": np.full((len(kept_rows),), query_id),
        "verification_source_row_indices": kept_rows,
        "verification_xy": query_xy,
        "candidate_track_ids": candidate_tracks,
        "candidate_probabilities": candidate_probabilities,
        "null_probabilities": null_probabilities,
        "candidate_view_weights": views.weights,
        "candidate_view_template_available": np.stack(family_usable, axis=3),
        "candidate_view_peak_log_ratios": np.stack(family_peaks, axis=3),
    }
    implementation = {
        "script_sha256": file_sha256_short(Path(__file__)),
        "local_evidence_module_sha256": file_sha256_short(
            Path("feature_extract/vfm/localization/frozen_multiscale_pose_evidence.py")
        ),
        "support_alignment_module_sha256": file_sha256_short(
            Path("feature_extract/vfm/localization/pose_conditioned_support_alignment.py")
        ),
    }
    metadata: dict[str, Any] = {
        "format": SCORE_FORMAT,
        "version": SCORE_VERSION,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "row_count": int(len(arrays["query_ids"])),
        "query_count": 1,
        "query_id": query_id,
        "split_name": query_split,
        "source_field_allowlists": {
            "baseline_score_artifact": [
                "query_ids",
                "split_names",
                "evaluation_labels",
                "hypothesis_indices",
                "source_chosen_for_optional_pose",
                "independent_score_top1",
                "independent_selection_scores",
            ],
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
        "strict_frozen_evidence_contract": {
            "heldout_query_rows": True,
            "verification_point_selector_fixed_across_hypotheses": True,
            "fixed_global_topl": True,
            "fixed_candidate_top_k": int(args.fixed_candidate_top_k),
            "candidate_posterior": "s480_explicit_null_no_renormalization_v1",
            "candidate_identity_fixed_across_hypotheses": True,
            "support_views": "maplet_track_support_views_fixed_before_pose_scoring_v1",
            "support_view_weighting": "fixed_coverage_weighted_probability_mixture_v1",
            "support_view_descriptor_averaging": False,
            "query_anchor": "fixed_heldout_detector_xy_v1",
            "local_density": "discrete_softmax_over_fixed_anchor_window_relative_to_uniform_local_null_v1",
            "local_density_pose_dependent_denominator": False,
            "usable_view_outside_image_or_local_window": "zero_spatial_likelihood",
            "missing_or_unusable_view": "pose_independent_neutral_likelihood",
            "query_center_gaussian_fallback": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "candidate_reselection_per_pose": False,
            "support_reselection_per_pose": False,
        },
        "profiles": [
            {
                "name": profile.name,
                "source": profile.source_name,
                "context_window_size": int(profile.context_window_size),
                "local_window_size": int(profile.local_window_size),
            }
            for profile in profiles
        ],
        "context_config": {
            "temperature": float(args.context_temperature),
            "template_batch_size": int(args.context_template_batch_size),
            "minimum_support_fraction": float(args.context_minimum_support_fraction),
            "minimum_query_overlap_fraction": float(args.context_minimum_query_overlap_fraction),
            "sampling": "fixed_floor_descriptor_cell_v1",
        },
        "verification_point_selection": {
            "source": "s0_detector_rows_unused_by_candidate_artifact",
            "point_count": int(args.verification_point_count),
            "detector_log_merit_weight": float(args.detector_log_merit_weight),
            **selection_audit,
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
            name: {
                "source_image_manifest_sha256": source.metadata.get("source_image_manifest_sha256"),
                "pca_fit_scope": source.metadata.get("pca_fit_scope"),
                "radio_checkpoint_sha256": source.metadata.get("radio_checkpoint_sha256"),
                "alike_checkpoint_sha256": source.metadata.get("alike_checkpoint_sha256"),
            }
            for name, source in sources.items()
        },
        "source_metadata_hashes": {
            "hypothesis": _canonical_hash(hypothesis_metadata),
            "baseline_s0": _canonical_hash(baseline_metadata),
            "detector": _canonical_hash(detector_metadata),
            "proposals": _canonical_hash(proposal_metadata),
            "candidate": _canonical_hash(candidate_metadata),
            "candidate_prior": _canonical_hash(prior_metadata),
            "maplet": _canonical_hash(maplet_metadata),
            "support_geometry": _canonical_hash(support_geometry_metadata),
            "landmark_bank": _canonical_hash(bank_metadata),
        },
        "inputs": _input_manifest(paths),
        "implementation": implementation,
        "frozen_multiscale_pose_evidence_version": FROZEN_MULTISCALE_POSE_EVIDENCE_VERSION,
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
                "stage": "frozen_multiscale_candidate_pose_evidence",
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
