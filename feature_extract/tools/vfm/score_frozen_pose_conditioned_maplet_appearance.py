"""Score fixed hypotheses with held-out candidate-specific 3-D maplet evidence.

Unlike the earlier support-centred crop probes, this P1 scorer keeps each
candidate's support-maplet topology fixed and projects its *neighbouring* SfM
tracks through every frozen pose.  It samples only real query/support feature
maps, excludes PnP-fit image neighbourhoods, preserves top-20/null and
support-view mixtures, and emits raw diagnostic likelihoods only.

It deliberately has no pose-target input and never updates PnP.  A separate
paired target-side audit is required before any calibration experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.tools.vfm.build_frozen_fulltrack_per_view_multiscale_translation_mode import (
    _TranslationSource,
)
from feature_extract.tools.vfm.build_frozen_fulltrack_per_view_sparse_maplet_transport import (
    load_sparse_maplet_neighbor_topology_cache,
)
from feature_extract.tools.vfm.score_frozen_lifted_loftr_map_to_query_pose_evidence import (
    _load_candidate_support_view_overlay,
)
from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _input_manifest,
    _load_bank_xyz,
    _load_exact_hypotheses,
    _load_npz_allowlist,
)
from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _load_candidate_prior_overlay,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.localization.frozen_lifted_loftr_map_to_query import (
    deterministic_track_xyz_permutation,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.localization.frozen_pose_conditioned_maplet_appearance import (
    FROZEN_MAPLET_APPEARANCE_PROFILES,
    FROZEN_POSE_CONDITIONED_MAPLET_APPEARANCE_VERSION,
    FrozenCandidateMapletEvidenceLayout,
    FrozenCandidateMapletProfileLayout,
    FrozenMapletAppearanceProfile,
    build_frozen_candidate_maplet_profile_layout,
    deterministic_support_descriptor_derangement,
    fixed_candidate_maplet_group_log_ratios,
    pool_maplet_neighbor_log_ratios,
    select_maplet_slots,
    summarize_frozen_group_log_ratios,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    ImageGridFeatureSource,
    project_simple_radial_torch,
)


SCORE_FORMAT = "frozen_pose_conditioned_maplet_appearance_scores_v1"
SCORE_VERSION = FROZEN_POSE_CONDITIONED_MAPLET_APPEARANCE_VERSION
FIXED_CANDIDATE_TOP_K = 20
EVIDENCE_VARIANTS = (
    "visual",
    "support_descriptor_permutation_control",
    "xyz_permutation_control",
)
_TOPOLOGY_PROFILE_NAMES = {
    "radio_final_near": "radio_final_sparse_maplet_near",
    "radio_final_wide": "radio_final_sparse_maplet_wide",
    "radio_intermediate_near": "radio_intermediate_pca256_sparse_maplet_near",
    "radio_intermediate_wide": "radio_intermediate_pca256_sparse_maplet_wide",
    "alike_near": "alike_fpn_sparse_maplet_near",
    "alike_wide": "alike_fpn_sparse_maplet_wide",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis-artifact", required=True)
    parser.add_argument("--baseline-score-artifact", required=True)
    parser.add_argument("--detector-query-cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--fixed-candidate-prior-overlay", required=True)
    parser.add_argument("--fixed-candidate-support-view-overlay", required=True)
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--neighbor-topology-cache", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument(
        "--profiles",
        default="radio_final_wide,radio_intermediate_wide,alike_wide",
        help="comma-separated fixed maplet appearance profiles",
    )
    parser.add_argument("--verification-point-count", type=int, default=64)
    parser.add_argument("--verification-grid-size", type=int, default=8)
    parser.add_argument("--detector-log-merit-weight", type=float, default=0.01)
    parser.add_argument("--fit-exclusion-radius-px", type=float, default=16.0)
    parser.add_argument("--neighbors-per-quadrant", type=int, default=1)
    parser.add_argument("--minimum-active-neighbors-per-quadrant", type=int, default=1)
    parser.add_argument("--minimum-active-quadrants", type=int, default=3)
    parser.add_argument(
        "--quadrant-reductions",
        default="mean,median",
        help="comma-separated fixed quadrant reductions: mean,median",
    )
    parser.add_argument(
        "--missing-view-ratios",
        default="0.5",
        help="comma-separated predeclared missing-view likelihood ratios in (0,1]",
    )
    parser.add_argument(
        "--source-temperatures",
        default="radio_final=0.10,radio_intermediate=0.10,alike=0.07",
        help="comma-separated source=positive_temperature pairs",
    )
    parser.add_argument("--max-log-ratio", type=float, default=12.0)
    parser.add_argument("--hypothesis-batch-size", type=int, default=32)
    parser.add_argument("--template-chunk-size", type=int, default=2048)
    parser.add_argument("--hypothesis-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evidence-variant", choices=EVIDENCE_VARIANTS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _parse_profiles(value: str) -> tuple[FrozenMapletAppearanceProfile, ...]:
    requested = tuple(item.strip() for item in str(value).split(",") if item.strip())
    by_name = {profile.name: profile for profile in FROZEN_MAPLET_APPEARANCE_PROFILES}
    if not requested or len(set(requested)) != len(requested) or any(item not in by_name for item in requested):
        raise ValueError("maplet appearance profiles are unknown or duplicated")
    return tuple(by_name[item] for item in requested)


def _parse_reductions(value: str) -> tuple[str, ...]:
    reductions = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not reductions or len(set(reductions)) != len(reductions) or any(
        item not in {"mean", "median"} for item in reductions
    ):
        raise ValueError("maplet quadrant reductions must be unique mean/median values")
    return reductions


def _parse_ratios(value: str) -> tuple[float, ...]:
    try:
        ratios = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError("missing-view ratios are malformed") from error
    if (
        not ratios
        or len(set(ratios)) != len(ratios)
        or any(not np.isfinite(item) or not 0.0 < item <= 1.0 for item in ratios)
    ):
        raise ValueError("missing-view ratios must be unique values in (0,1]")
    return ratios


def _parse_temperatures(value: str) -> dict[str, float]:
    output: dict[str, float] = {}
    try:
        entries = tuple(item.strip() for item in str(value).split(",") if item.strip())
        for entry in entries:
            name, raw = (part.strip() for part in entry.split("=", 1))
            if name in output:
                raise ValueError
            output[name] = float(raw)
    except (ValueError, IndexError) as error:
        raise ValueError("source temperatures must be source=positive_float pairs") from error
    required = {"radio_final", "radio_intermediate", "alike"}
    if set(output) != required or any(not np.isfinite(item) or item <= 0.0 for item in output.values()):
        raise ValueError("source temperatures are incomplete or invalid")
    return output


def _ratio_name(value: float) -> str:
    return f"{float(value):.4g}".replace(".", "p").replace("-", "m")


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()[:16]


def _select_spatially_diverse_heldout_rows(
    *,
    query_id: str,
    detector: Mapping[str, np.ndarray],
    proposals: Mapping[str, np.ndarray],
    selected_rows: np.ndarray,
    image_width: int,
    image_height: int,
    point_count: int,
    grid_size: int,
    detector_log_merit_weight: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Select held-out rows with deterministic image-plane quota before scoring."""

    ids = np.asarray(detector["image_ids"]).astype(str).reshape(-1)
    offsets = np.asarray(detector["offsets"], dtype=np.int64).reshape(-1)
    xy = np.asarray(detector["xy"], dtype=np.float32)
    detector_scores = np.asarray(detector["detector_scores"], dtype=np.float64).reshape(-1)
    proposal_ids = np.asarray(proposals["query_ids"]).astype(str).reshape(-1)
    coarse = np.asarray(proposals["coarse_scores"], dtype=np.float64)
    count = int(point_count)
    grid = int(grid_size)
    if (
        count <= 0
        or grid <= 0
        or int(image_width) <= 1
        or int(image_height) <= 1
        or offsets.shape != (len(ids) + 1,)
        or xy.shape != (len(proposal_ids), 2)
        or detector_scores.shape != (len(proposal_ids),)
        or coarse.shape[0] != len(proposal_ids)
    ):
        raise ValueError("held-out spatial selector inputs are incompatible")
    image_matches = np.flatnonzero(ids == str(query_id))
    if len(image_matches) != 1:
        raise ValueError("detector cache does not uniquely contain the query image")
    image_index = int(image_matches[0])
    all_rows = np.arange(int(offsets[image_index]), int(offsets[image_index + 1]), dtype=np.int64)
    if len(all_rows) == 0 or not np.all(proposal_ids[all_rows] == str(query_id)):
        raise ValueError("query detector/proposal ownership differs")
    fit = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    fit = fit[proposal_ids[fit] == str(query_id)]
    if len(np.unique(fit)) != len(fit):
        raise ValueError("PnP fit rows repeat within the query")
    unused = np.setdiff1d(all_rows, fit, assume_unique=True)
    if len(unused) == 0:
        raise ValueError("query has no held-out detector rows")
    merit = np.max(coarse[unused], axis=1) + float(detector_log_merit_weight) * np.log(
        np.maximum(detector_scores[unused], 1e-12)
    )
    positions = xy[unused]
    columns = np.clip((positions[:, 0] * grid / float(image_width)).astype(np.int64), 0, grid - 1)
    rows = np.clip((positions[:, 1] * grid / float(image_height)).astype(np.int64), 0, grid - 1)
    cells = rows * grid + columns
    # First pass preserves one high-merit point per populated cell.  A second
    # pass fills the requested count globally without a pose-dependent choice.
    chosen: list[int] = []
    chosen_set: set[int] = set()
    for cell in range(grid * grid):
        local = np.flatnonzero(cells == cell)
        if len(local) == 0:
            continue
        ordered = sorted(local.tolist(), key=lambda index: (-float(merit[index]), int(unused[index])))
        index = int(unused[ordered[0]])
        chosen.append(index)
        chosen_set.add(index)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        ordered = sorted(range(len(unused)), key=lambda index: (-float(merit[index]), int(unused[index])))
        for local in ordered:
            index = int(unused[local])
            if index in chosen_set:
                continue
            chosen.append(index)
            chosen_set.add(index)
            if len(chosen) >= count:
                break
    selected = np.asarray(sorted(chosen), dtype=np.int64)
    return selected, {
        "fit_query_point_count": int(len(fit)),
        "available_unused_query_point_count": int(len(unused)),
        "selected_verification_point_count": int(len(selected)),
        "verification_grid_size": grid,
        "populated_grid_cell_count": int(len(set(cells.tolist()))),
    }


def _build_fit_exclusion_mask(
    *, image_width: int, image_height: int, fit_xy: np.ndarray, radius_px: float
) -> np.ndarray:
    """Create one fixed full-resolution PnP-content exclusion mask."""

    width = int(image_width)
    height = int(image_height)
    points = np.asarray(fit_xy, dtype=np.float32).reshape(-1, 2)
    radius = float(radius_px)
    if (
        width <= 1
        or height <= 1
        or len(points) == 0
        or np.any(~np.isfinite(points))
        or not np.isfinite(radius)
        or radius <= 0.0
    ):
        raise ValueError("fit-exclusion mask inputs are invalid")
    mask = np.zeros((height, width), dtype=bool)
    radius_squared = radius * radius
    for x, y in points.tolist():
        left = max(0, int(math.floor(float(x) - radius)))
        right = min(width - 1, int(math.ceil(float(x) + radius)))
        top = max(0, int(math.floor(float(y) - radius)))
        bottom = min(height - 1, int(math.ceil(float(y) + radius)))
        if left > right or top > bottom:
            continue
        yy, xx = np.ogrid[top : bottom + 1, left : right + 1]
        mask[top : bottom + 1, left : right + 1] |= (
            (xx.astype(np.float32) - float(x)) ** 2
            + (yy.astype(np.float32) - float(y)) ** 2
            <= radius_squared
        )
    return mask


def _sample_binary_mask(
    *, mask: torch.Tensor, projected_xy: torch.Tensor, image_width: int, image_height: int
) -> torch.Tensor:
    """Nearest-neighbour sample of a static full-resolution exclusion mask."""

    positions = torch.as_tensor(projected_xy)
    if (
        mask.shape != (1, 1, int(image_height), int(image_width))
        or positions.ndim != 3
        or positions.shape[2] != 2
        or not bool(torch.isfinite(torch.nan_to_num(positions)).all())
    ):
        raise ValueError("fit-exclusion mask sampling inputs are invalid")
    batch, edge_count = positions.shape[:2]
    safe = torch.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
    coordinates = torch.stack(
        (
            2.0 * safe[..., 0] / float(int(image_width) - 1) - 1.0,
            2.0 * safe[..., 1] / float(int(image_height) - 1) - 1.0,
        ),
        dim=-1,
    ).reshape(batch, edge_count, 1, 2)
    sampled = F.grid_sample(
        mask.expand(batch, -1, -1, -1),
        coordinates,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[:, 0, :, 0] > 0.5


def _build_template_normalizers(
    *, query_grid: torch.Tensor, support_descriptors: torch.Tensor, temperature: float, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute the fixed global denominator for each support descriptor."""

    grid = torch.as_tensor(query_grid, dtype=torch.float32)
    support = torch.as_tensor(support_descriptors, dtype=torch.float32, device=grid.device)
    chunk = int(chunk_size)
    if (
        grid.ndim != 3
        or support.ndim != 2
        or support.shape[1] != grid.shape[2]
        or support.shape[0] == 0
        or chunk <= 0
        or not np.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("maplet template-normalizer inputs are invalid")
    normalized_grid = F.normalize(grid.reshape(-1, grid.shape[2]), p=2, dim=1)
    normalized_support = F.normalize(support, p=2, dim=1)
    terms: list[torch.Tensor] = []
    for begin in range(0, len(normalized_support), chunk):
        part = normalized_support[begin : begin + chunk]
        values = part @ normalized_grid.T
        terms.append(
            torch.logsumexp(values / float(temperature), dim=1)
            - math.log(float(normalized_grid.shape[0]))
        )
    return normalized_grid, normalized_support, torch.cat(terms, dim=0)


def _sample_template_log_ratios(
    *,
    query_grid: torch.Tensor,
    normalized_support: torch.Tensor,
    normalizers: torch.Tensor,
    projected_xy: torch.Tensor,
    projection_valid: torch.Tensor,
    fit_exclusion_mask: torch.Tensor,
    image_width: int,
    image_height: int,
    temperature: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample fixed global template likelihoods at pose-projected maplet tracks."""

    grid = torch.as_tensor(query_grid, dtype=torch.float32)
    support = torch.as_tensor(normalized_support, dtype=torch.float32, device=grid.device)
    denominator = torch.as_tensor(normalizers, dtype=torch.float32, device=grid.device)
    positions = torch.as_tensor(projected_xy, dtype=torch.float32, device=grid.device)
    valid = torch.as_tensor(projection_valid, dtype=torch.bool, device=grid.device)
    chunk = int(chunk_size)
    if (
        grid.ndim != 3
        or support.ndim != 2
        or denominator.shape != (len(support),)
        or positions.ndim != 3
        or positions.shape[1:] != (len(support), 2)
        or valid.shape != positions.shape[:2]
        or chunk <= 0
        or float(temperature) <= 0.0
    ):
        raise ValueError("maplet template likelihood sampler inputs are invalid")
    map_tensor = grid.permute(2, 0, 1)[None]
    batch = int(positions.shape[0])
    logs = torch.zeros((batch, len(support)), dtype=torch.float32, device=grid.device)
    active = torch.zeros((batch, len(support)), dtype=torch.bool, device=grid.device)
    blocked = torch.zeros((batch, len(support)), dtype=torch.bool, device=grid.device)
    for begin in range(0, len(support), chunk):
        end = min(begin + chunk, len(support))
        local = positions[:, begin:end]
        safe = torch.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)
        coordinates = torch.stack(
            (
                2.0 * safe[..., 0] / float(int(image_width) - 1) - 1.0,
                2.0 * safe[..., 1] / float(int(image_height) - 1) - 1.0,
            ),
            dim=-1,
        ).reshape(batch, end - begin, 1, 2)
        sampled = F.grid_sample(
            map_tensor.expand(batch, -1, -1, -1),
            coordinates,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, :, :, 0].transpose(1, 2)
        sampled = F.normalize(sampled, p=2, dim=2)
        local_blocked = _sample_binary_mask(
            mask=fit_exclusion_mask,
            projected_xy=local,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        local_active = valid[:, begin:end] & ~local_blocked
        local_log = (
            torch.sum(sampled * support[begin:end][None], dim=2) / float(temperature)
            - denominator[None, begin:end]
        )
        logs[:, begin:end] = torch.where(local_active, local_log, torch.zeros_like(local_log))
        active[:, begin:end] = local_active
        blocked[:, begin:end] = valid[:, begin:end] & local_blocked
    return logs, active, blocked


def _profile_source_name(profile: FrozenMapletAppearanceProfile) -> str:
    return str(profile.source_name)


def _source_from_translation_cache(
    *, sources: Sequence[object]
) -> dict[str, ImageGridFeatureSource]:
    aliases = {
        "radio_final": "radio_final",
        "radio_intermediate_pca256": "radio_intermediate",
        "alike_fpn": "alike",
    }
    output: dict[str, ImageGridFeatureSource] = {}
    for raw in sources:
        raw_name = str(getattr(raw, "name"))
        name = aliases.get(raw_name)
        if name is None:
            raise ValueError("translation source name is unknown")
        grids = np.asarray(getattr(raw, "grids"), dtype=np.float32)
        output[name] = ImageGridFeatureSource(
            name=name,
            image_ids=np.asarray(getattr(raw, "image_ids")).astype(str),
            image_sizes=np.asarray(getattr(raw, "image_sizes"), dtype=np.int64),
            grid_size=int(getattr(raw, "grid_size")),
            descriptors=grids.reshape(len(grids), -1, grids.shape[-1]),
            metadata=dict(getattr(raw, "metadata")),
        )
    if set(output) != {"radio_final", "radio_intermediate", "alike"}:
        raise ValueError("multiscale maplet image sources are incomplete")
    reference = output["radio_final"]
    for source in output.values():
        if not np.array_equal(source.image_ids, reference.image_ids) or not np.array_equal(
            source.image_sizes, reference.image_sizes
        ):
            raise ValueError("maplet feature sources do not share image ownership")
    return output


def _load_topology_lineaged_maplet_sources(
    *,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
) -> tuple[tuple[_TranslationSource, ...], dict[str, ImageGridFeatureSource]]:
    """Load real-image grids while retaining the topology cache's exact lineage.

    The existing topology cache predates the newer image-source-contract
    schema.  It still carries exact cache hashes, source-image manifest hashes,
    PCA/checkpoint metadata, and geometry hash.  The context loader verifies
    those real-image cache invariants; ``load_sparse_maplet_neighbor_topology_cache``
    then independently requires the original cache paths and hashes.  This is
    intentionally not a permissive legacy fallback.
    """

    loaded = load_context_attention_sources(
        radio_final_context_cache=Path(radio_final_context_cache),
        radio_intermediate_context_cache=Path(radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(alike_spatial_context_cache),
        expected_radio_checkpoint="",
        require_equal_descriptor_dimensions=False,
    )
    expected = {
        "radio_final": ("radio_final", Path(radio_final_context_cache)),
        "radio_intermediate": (
            "radio_intermediate_pca256",
            Path(radio_intermediate_context_cache),
        ),
        "alike": ("alike_fpn", Path(alike_spatial_context_cache)),
    }
    translation: list[_TranslationSource] = []
    for source in loaded:
        target = expected.get(str(source.name))
        if target is None:
            raise RuntimeError("context loader returned an unknown maplet source")
        translation.append(
            _TranslationSource(
                name=target[0],
                grid_size=int(source.spatial_grid_size),
                image_ids=source.image_ids,
                image_sizes=source.image_sizes,
                grids=np.asarray(source.grid, dtype=np.float32),
                metadata=source.metadata,
                cache_path=target[1],
            )
        )
    by_name = {source.name: source for source in translation}
    if set(by_name) != {"radio_final", "radio_intermediate_pca256", "alike_fpn"}:
        raise RuntimeError("topology-lineaged maplet source aliases drifted")
    return tuple(translation), _source_from_translation_cache(sources=tuple(translation))


def _replace_xyz_with_deterministic_control(
    *, layout: FrozenCandidateMapletProfileLayout, canonical_track_ids: np.ndarray, canonical_xyz: np.ndarray, query_id: str
) -> FrozenCandidateMapletProfileLayout:
    """Break support-descriptor-to-3D association while preserving every layout axis."""

    tracks = np.asarray(canonical_track_ids, dtype=np.int64).reshape(-1)
    xyz = np.asarray(canonical_xyz, dtype=np.float32).reshape(-1, 3)
    permutation = deterministic_track_xyz_permutation(tracks, query_id=str(query_id))
    order = np.argsort(tracks, kind="stable")
    sorted_tracks = tracks[order]
    neighbors = layout.neighbor_track_ids
    valid = layout.neighbor_valid
    positions = np.searchsorted(sorted_tracks, np.maximum(neighbors, 0))
    safe_positions = np.minimum(positions, len(sorted_tracks) - 1)
    found = (positions < len(sorted_tracks)) & (sorted_tracks[safe_positions] == np.maximum(neighbors, 0))
    if np.any(valid & ~found):
        raise RuntimeError("XYZ control could not resolve a frozen neighbour track")
    bank_rows = order[safe_positions]
    controlled_xyz = np.zeros_like(layout.neighbor_xyz)
    controlled_xyz[valid] = xyz[permutation[bank_rows[valid]]]
    return FrozenCandidateMapletProfileLayout(
        profile=layout.profile,
        anchor_geometry_rows=layout.anchor_geometry_rows,
        neighbor_track_ids=layout.neighbor_track_ids,
        neighbor_xyz=controlled_xyz,
        neighbor_support_xy=layout.neighbor_support_xy,
        neighbor_valid=layout.neighbor_valid,
    )


def _flatten_profile_edges(
    layout: FrozenCandidateMapletProfileLayout,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    indices = tuple(np.asarray(value, dtype=np.int64) for value in np.nonzero(layout.neighbor_valid))
    if len(indices) != 5 or len(indices[0]) == 0:
        raise ValueError("frozen maplet profile has no real support neighbours")
    shape = layout.neighbor_valid.shape
    flat = np.ravel_multi_index(indices, shape)
    return indices, flat.astype(np.int64, copy=False)


def _score_one_profile(
    *,
    profile_layout: FrozenCandidateMapletProfileLayout,
    evidence: FrozenCandidateMapletEvidenceLayout,
    source: ImageGridFeatureSource,
    query_id: str,
    poses_w2c: np.ndarray,
    camera: object,
    image_width: int,
    image_height: int,
    fit_exclusion_mask: np.ndarray,
    source_temperature: float,
    quadrant_reductions: Sequence[str],
    missing_view_ratios: Sequence[float],
    minimum_active_neighbors_per_quadrant: int,
    minimum_active_quadrants: int,
    max_log_ratio: float,
    hypothesis_batch_size: int,
    template_chunk_size: int,
    device: torch.device,
    evidence_variant: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, object]]:
    """Score one source/profile under all immutable hypothesis rows."""

    if int(getattr(camera, "model_id")) != 2:
        raise ValueError("maplet appearance P1 requires COLMAP SIMPLE_RADIAL cameras")
    params = tuple(float(value) for value in getattr(camera, "params"))
    if len(params) != 4:
        raise ValueError("maplet appearance camera parameters are invalid")
    indices, flat_indices = _flatten_profile_edges(profile_layout)
    point_index, candidate_index, view_index, _quadrant_index, _slot_index = indices
    support_ids = evidence.support_image_ids[point_index, candidate_index, view_index]
    if evidence_variant == "support_descriptor_permutation_control":
        derangement = deterministic_support_descriptor_derangement(
            image_ids=source.image_ids, image_sizes=source.image_sizes
        )
        sample_ids = np.asarray([derangement[str(image_id)] for image_id in support_ids], dtype=np.str_)
    else:
        derangement = None
        sample_ids = support_ids
    support_xy = profile_layout.neighbor_support_xy[indices]
    support_descriptors = source.sample_numpy(sample_ids, support_xy)
    query_grid_np, query_size = source.image_grid(str(query_id))
    if tuple(int(value) for value in query_size) != (int(image_width), int(image_height)):
        raise ValueError("query image feature and COLMAP geometry differ")
    query_grid = torch.as_tensor(query_grid_np, dtype=torch.float32, device=device)
    support_tensor = torch.as_tensor(support_descriptors, dtype=torch.float32, device=device)
    _normalized_grid, normalized_support, normalizers = _build_template_normalizers(
        query_grid=query_grid,
        support_descriptors=support_tensor,
        temperature=float(source_temperature),
        chunk_size=int(template_chunk_size),
    )
    del _normalized_grid, support_tensor
    neighbor_xyz = torch.as_tensor(
        profile_layout.neighbor_xyz[indices], dtype=torch.float32, device=device
    )
    present = torch.as_tensor(profile_layout.neighbor_valid, dtype=torch.bool, device=device)
    view_weights = torch.as_tensor(
        evidence.support_view_probabilities, dtype=torch.float32, device=device
    )
    candidate_probabilities = torch.as_tensor(
        evidence.candidate_probabilities, dtype=torch.float32, device=device
    )
    null_probabilities = torch.as_tensor(
        evidence.null_probabilities, dtype=torch.float32, device=device
    )
    exclusion = torch.as_tensor(
        fit_exclusion_mask.astype(np.float32)[None, None], dtype=torch.float32, device=device
    )
    shape = profile_layout.neighbor_valid.shape
    point_count, candidate_count, view_count, quadrant_count, slots = shape
    if candidate_count != FIXED_CANDIDATE_TOP_K or quadrant_count != 4:
        raise RuntimeError("maplet evidence changed fixed top-L/quadrant layout")
    family_values: dict[str, dict[str, list[np.ndarray]]] = {}
    family_active_points: dict[str, list[np.ndarray]] = {}
    family_active_view_mass: dict[str, list[np.ndarray]] = {}
    family_visible_fraction: dict[str, list[np.ndarray]] = {}
    family_masked_fraction: dict[str, list[np.ndarray]] = {}
    for reduction in quadrant_reductions:
        for missing in missing_view_ratios:
            family = f"{profile_layout.profile.name}__q{reduction}__miss{_ratio_name(missing)}"
            family_values[family] = {name: [] for name in ("mean", "median", "worst_quartile_mean", "spatial_median_of_means_2x2")}
            family_active_points[family] = []
            family_active_view_mass[family] = []
            family_visible_fraction[family] = []
            family_masked_fraction[family] = []
    for begin in range(0, len(poses_w2c), int(hypothesis_batch_size)):
        end = min(begin + int(hypothesis_batch_size), len(poses_w2c))
        pose_tensor = torch.as_tensor(poses_w2c[begin:end], dtype=torch.float32, device=device)
        projected, visible = project_simple_radial_torch(
            neighbor_xyz,
            pose_tensor,
            focal_length=params[0],
            principal_x=params[1],
            principal_y=params[2],
            radial_k=params[3],
            image_width=int(image_width),
            image_height=int(image_height),
        )
        edge_logs, edge_active, edge_masked = _sample_template_log_ratios(
            query_grid=query_grid,
            normalized_support=normalized_support,
            normalizers=normalizers,
            projected_xy=projected,
            projection_valid=visible,
            fit_exclusion_mask=exclusion,
            image_width=int(image_width),
            image_height=int(image_height),
            temperature=float(source_temperature),
            chunk_size=int(template_chunk_size),
        )
        batch = end - begin
        dense_logs = torch.zeros((batch, int(np.prod(shape))), dtype=torch.float32, device=device)
        dense_active = torch.zeros((batch, int(np.prod(shape))), dtype=torch.bool, device=device)
        dense_logs[:, torch.as_tensor(flat_indices, dtype=torch.long, device=device)] = edge_logs
        dense_active[:, torch.as_tensor(flat_indices, dtype=torch.long, device=device)] = edge_active
        dense_logs = dense_logs.reshape(batch, *shape)
        dense_active = dense_active.reshape(batch, *shape)
        visible_fraction = visible.to(dtype=torch.float32).mean(dim=1)
        masked_fraction = edge_masked.to(dtype=torch.float32).sum(dim=1) / visible.to(dtype=torch.float32).sum(dim=1).clamp_min(1.0)
        for reduction in quadrant_reductions:
            view_logs, view_usable, active_quadrants = pool_maplet_neighbor_log_ratios(
                neighbor_log_ratios=dense_logs,
                neighbor_active=dense_active,
                neighbor_present=present,
                minimum_active_neighbors_per_quadrant=int(minimum_active_neighbors_per_quadrant),
                minimum_active_quadrants=int(minimum_active_quadrants),
                quadrant_reduction=str(reduction),
            )
            for missing in missing_view_ratios:
                family = f"{profile_layout.profile.name}__q{reduction}__miss{_ratio_name(missing)}"
                group_logs, _candidate_ratio = fixed_candidate_maplet_group_log_ratios(
                    view_log_ratios=view_logs,
                    view_usable=view_usable,
                    support_view_probabilities=view_weights,
                    candidate_probabilities=candidate_probabilities,
                    null_probabilities=null_probabilities,
                    missing_view_ratio=float(missing),
                    max_log_ratio=float(max_log_ratio),
                )
                summaries = summarize_frozen_group_log_ratios(
                    group_log_ratios=group_logs,
                    query_xy=evidence.verification_xy,
                    image_width=int(image_width),
                    image_height=int(image_height),
                )
                for statistic, value in summaries.items():
                    family_values[family][statistic].append(value.detach().cpu().numpy())
                family_active_points[family].append(
                    view_usable.any(dim=3).any(dim=2).sum(dim=1).detach().cpu().numpy()
                )
                family_active_view_mass[family].append(
                    torch.sum(
                        view_usable.to(dtype=torch.float32) * view_weights[None], dim=(1, 2, 3)
                    ).detach().cpu().numpy()
                )
                family_visible_fraction[family].append(visible_fraction.detach().cpu().numpy())
                family_masked_fraction[family].append(masked_fraction.detach().cpu().numpy())
    output = {
        family: {
            statistic: np.concatenate(values, axis=0).astype(np.float64, copy=False)
            for statistic, values in by_statistic.items()
        }
        for family, by_statistic in family_values.items()
    }
    diagnostics = {
        "active_point_counts": {
            family: np.concatenate(values, axis=0).astype(np.int64, copy=False)
            for family, values in family_active_points.items()
        },
        "active_view_masses": {
            family: np.concatenate(values, axis=0).astype(np.float64, copy=False)
            for family, values in family_active_view_mass.items()
        },
        "projection_visible_fractions": {
            family: np.concatenate(values, axis=0).astype(np.float64, copy=False)
            for family, values in family_visible_fraction.items()
        },
        "fit_masked_visible_fractions": {
            family: np.concatenate(values, axis=0).astype(np.float64, copy=False)
            for family, values in family_masked_fraction.items()
        },
    }
    static = {
        "profile": profile_layout.profile.name,
        "static_neighbor_count": int(profile_layout.neighbor_valid.sum()),
        "static_candidate_view_with_neighbor_count": int(
            np.any(profile_layout.neighbor_valid, axis=(3, 4)).sum()
        ),
        "neighbor_track_ids_sha256": _array_digest(profile_layout.neighbor_track_ids),
        "neighbor_xyz_sha256": _array_digest(profile_layout.neighbor_xyz),
        "neighbor_support_xy_sha256": _array_digest(profile_layout.neighbor_support_xy),
        "neighbor_valid_sha256": _array_digest(profile_layout.neighbor_valid),
        "support_descriptor_derangement": None
        if derangement is None
        else hashlib.sha256(json.dumps(derangement, sort_keys=True).encode()).hexdigest()[:16],
    }
    return output, diagnostics, static


def _combine_equal_log_ratio_families(
    *, families: dict[str, dict[str, np.ndarray]], profile_names: Sequence[str]
) -> None:
    """Append predeclared equal-weight multi-scale combinations when available."""

    requested = set(profile_names)
    combinations = {
        "final_intermediate_wide_equal": ("radio_final_wide", "radio_intermediate_wide"),
        "final_intermediate_alike_wide_equal": (
            "radio_final_wide",
            "radio_intermediate_wide",
            "alike_wide",
        ),
    }
    original = list(families.items())
    for label, members in combinations.items():
        if not set(members) <= requested:
            continue
        suffixes = sorted(
            {
                name.split("__", 1)[1]
                for name, _scores in original
                if name.split("__", 1)[0] in members
            }
        )
        for suffix in suffixes:
            names = [f"{member}__{suffix}" for member in members]
            if not all(name in families for name in names):
                continue
            combined = {
                statistic: np.mean(
                    np.stack([families[name][statistic] for name in names], axis=1), axis=1
                )
                for statistic in families[names[0]]
            }
            families[f"{label}__{suffix}"] = combined


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if (
        int(args.verification_point_count) <= 0
        or int(args.verification_grid_size) <= 0
        or int(args.neighbors_per_quadrant) <= 0
        or int(args.minimum_active_neighbors_per_quadrant) <= 0
        or int(args.minimum_active_quadrants) <= 0
        or int(args.hypothesis_batch_size) <= 0
        or int(args.template_chunk_size) <= 0
        or int(args.hypothesis_limit) < 0
        or float(args.fit_exclusion_radius_px) <= 0.0
        or float(args.max_log_ratio) <= 0.0
    ):
        raise ValueError("maplet P1 numeric arguments are invalid")
    profiles = _parse_profiles(args.profiles)
    reductions = _parse_reductions(args.quadrant_reductions)
    missing_ratios = _parse_ratios(args.missing_view_ratios)
    temperatures = _parse_temperatures(args.source_temperatures)
    if int(args.minimum_active_quadrants) > 4:
        raise ValueError("minimum active maplet quadrants exceeds four")
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
        "fixed_candidate_support_view_overlay": Path(args.fixed_candidate_support_view_overlay),
        "maplet_support_index": Path(args.maplet_support_index),
        "support_geometry_index": Path(args.support_geometry_index),
        "projected_landmark_bank": Path(args.projected_landmark_bank),
        "neighbor_topology_cache": Path(args.neighbor_topology_cache),
        "radio_final_context_cache": Path(args.radio_final_context_cache),
        "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
        "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
        "colmap_cameras_bin": Path(args.colmap_model_dir) / "cameras.bin",
        "colmap_images_bin_camera_ownership_only": Path(args.colmap_model_dir) / "images.bin",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    started = time.time()
    exact, hypothesis_metadata, baseline_metadata = _load_exact_hypotheses(
        hypothesis_path=paths["hypothesis_artifact"],
        baseline_path=paths["baseline_score_artifact"],
        detector_path=paths["detector_query_cache"],
        proposals_path=paths["proposals"],
        candidate_path=paths["candidate_artifact"],
        prior_path=paths["fixed_candidate_prior_overlay"],
        fixed_candidate_top_k=FIXED_CANDIDATE_TOP_K,
    )
    if int(args.hypothesis_limit) > 0:
        count = min(int(args.hypothesis_limit), len(exact["query_ids"]))
        exact = {key: np.asarray(value)[:count] for key, value in exact.items()}
    query_id = str(np.asarray(exact["query_ids"]).astype(str)[0])
    split_name = str(np.asarray(exact["split_names"]).astype(str)[0])
    if split_name not in {"validation", "test"}:
        raise ValueError("maplet P1 only accepts frozen validation/test query rows")

    detector, detector_metadata, _detector_names = _load_npz_allowlist(
        paths["detector_query_cache"], ("image_ids", "offsets", "xy", "detector_scores")
    )
    proposals, proposal_metadata, _proposal_names = _load_npz_allowlist(
        paths["proposals"], ("query_ids", "candidate_track_ids", "coarse_scores"), metadata_required=False
    )
    candidate_artifact, candidate_metadata, _candidate_names = _load_npz_allowlist(
        paths["candidate_artifact"], ("selected_rows",)
    )
    if (
        detector_metadata.get("format") != "alike_detector_mapped_radio_query_cache_v1"
        or candidate_metadata.get("contains_ground_truth") is not False
        or candidate_metadata.get("contains_pose_derived_selection") is not False
        or proposal_metadata.get("format") not in {None, "detector_support_reranked_proposals_v1"}
        or np.asarray(proposals["candidate_track_ids"]).shape[1] != FIXED_CANDIDATE_TOP_K
    ):
        raise ValueError("maplet P1 source artifacts violate the frozen top-20 protocol")
    prior, prior_metadata = _load_candidate_prior_overlay(
        paths["fixed_candidate_prior_overlay"],
        proposals_path=paths["proposals"],
        proposals=proposals,
    )
    view_overlay, view_metadata = _load_candidate_support_view_overlay(
        path=paths["fixed_candidate_support_view_overlay"],
        proposals_path=paths["proposals"],
        proposals=proposals,
        candidate_prior_path=paths["fixed_candidate_prior_overlay"],
        maplet_support_index=paths["maplet_support_index"],
    )
    if not np.array_equal(prior["candidate_track_ids"], view_overlay["candidate_track_ids"]):
        raise ValueError("fixed candidate/support-view overlays disagree")

    translation_sources, sources = _load_topology_lineaged_maplet_sources(
        radio_final_context_cache=paths["radio_final_context_cache"],
        radio_intermediate_context_cache=paths["radio_intermediate_context_cache"],
        alike_spatial_context_cache=paths["alike_spatial_context_cache"],
    )
    query_grid, query_size = sources["radio_final"].image_grid(query_id)
    del query_grid
    image_width, image_height = (int(value) for value in query_size)
    cameras = read_colmap_cameras_binary(paths["colmap_cameras_bin"])
    image_camera_ids = read_colmap_image_camera_ids_binary(paths["colmap_images_bin_camera_ownership_only"])
    camera_id = image_camera_ids.get(query_id)
    if camera_id is None or int(camera_id) not in cameras:
        raise ValueError("query image has no COLMAP camera ownership")
    camera = cameras[int(camera_id)]
    if (int(camera.width), int(camera.height)) != (image_width, image_height):
        raise ValueError("query camera dimensions differ from image feature cache")
    verification_rows, selector_metadata = _select_spatially_diverse_heldout_rows(
        query_id=query_id,
        detector=detector,
        proposals=proposals,
        selected_rows=np.asarray(candidate_artifact["selected_rows"], dtype=np.int64),
        image_width=image_width,
        image_height=image_height,
        point_count=int(args.verification_point_count),
        grid_size=int(args.verification_grid_size),
        detector_log_merit_weight=float(args.detector_log_merit_weight),
    )
    evidence = FrozenCandidateMapletEvidenceLayout(
        verification_source_rows=verification_rows,
        verification_xy=np.asarray(detector["xy"], dtype=np.float32)[verification_rows],
        candidate_track_ids=np.asarray(proposals["candidate_track_ids"], dtype=np.int64)[verification_rows],
        candidate_probabilities=np.asarray(prior["candidate_probabilities"], dtype=np.float32)[verification_rows],
        null_probabilities=np.asarray(prior["null_probabilities"], dtype=np.float32)[verification_rows],
        support_view_probabilities=np.asarray(view_overlay["support_view_probabilities"], dtype=np.float32)[verification_rows],
        support_image_ids=np.asarray(view_overlay["candidate_support_image_ids"]).astype(str)[verification_rows],
        support_view_valid=np.asarray(view_overlay["candidate_support_view_valid"], dtype=bool)[verification_rows],
    )
    if set(evidence.support_image_ids.reshape(-1).tolist()) & {query_id}:
        raise ValueError("query image leaked into fixed support-view evidence")
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        paths["support_geometry_index"]
    )
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("maplet P1 requires real SfM support observation coordinates")
    topology = load_sparse_maplet_neighbor_topology_cache(
        path=paths["neighbor_topology_cache"],
        geometry_path=paths["support_geometry_index"],
        geometry_metadata=geometry_metadata,
        geometry_row_count=len(geometry),
        sources=translation_sources,
    )
    bank_tracks, bank_xyz, bank_metadata = _load_bank_xyz(paths["projected_landmark_bank"])
    if len(np.unique(bank_tracks)) != len(bank_tracks):
        raise ValueError("maplet P1 needs one canonical 3-D row per physical track")
    fit_rows = np.asarray(candidate_artifact["selected_rows"], dtype=np.int64)
    proposal_ids = np.asarray(proposals["query_ids"]).astype(str)
    fit_xy = np.asarray(detector["xy"], dtype=np.float32)[fit_rows[proposal_ids[fit_rows] == query_id]]
    if len(fit_xy) == 0:
        raise ValueError("query lacks immutable PnP fit rows for cross-fit exclusion")
    fit_mask = _build_fit_exclusion_mask(
        image_width=image_width,
        image_height=image_height,
        fit_xy=fit_xy,
        radius_px=float(args.fit_exclusion_radius_px),
    )

    family_scores: dict[str, dict[str, np.ndarray]] = {}
    family_diagnostics: dict[str, dict[str, np.ndarray]] = {}
    profile_static: dict[str, object] = {}
    for profile in profiles:
        topology_name = _TOPOLOGY_PROFILE_NAMES[profile.name]
        if topology_name not in topology:
            raise ValueError(f"strict topology cache lacks {topology_name}")
        layout = build_frozen_candidate_maplet_profile_layout(
            evidence=evidence,
            profile=profile,
            neighbor_topology=topology[topology_name],
            support_geometry=geometry,
            canonical_track_ids=bank_tracks,
            canonical_xyz=bank_xyz,
        )
        layout = select_maplet_slots(layout, slots_per_quadrant=int(args.neighbors_per_quadrant))
        if args.evidence_variant == "xyz_permutation_control":
            layout = _replace_xyz_with_deterministic_control(
                layout=layout,
                canonical_track_ids=bank_tracks,
                canonical_xyz=bank_xyz,
                query_id=query_id,
            )
        scores, diagnostics, static = _score_one_profile(
            profile_layout=layout,
            evidence=evidence,
            source=sources[_profile_source_name(profile)],
            query_id=query_id,
            poses_w2c=np.asarray(exact["poses_w2c"], dtype=np.float64),
            camera=camera,
            image_width=image_width,
            image_height=image_height,
            fit_exclusion_mask=fit_mask,
            source_temperature=float(temperatures[_profile_source_name(profile)]),
            quadrant_reductions=reductions,
            missing_view_ratios=missing_ratios,
            minimum_active_neighbors_per_quadrant=int(args.minimum_active_neighbors_per_quadrant),
            minimum_active_quadrants=int(args.minimum_active_quadrants),
            max_log_ratio=float(args.max_log_ratio),
            hypothesis_batch_size=int(args.hypothesis_batch_size),
            template_chunk_size=int(args.template_chunk_size),
            device=device,
            evidence_variant=str(args.evidence_variant),
        )
        if set(family_scores) & set(scores):
            raise RuntimeError("maplet score family names collided")
        family_scores.update(scores)
        family_diagnostics.update(
            {
                family: {
                    key: np.asarray(value[family])
                    for key, value in diagnostics.items()
                }
                for family in scores
            }
        )
        profile_static[profile.name] = static
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _combine_equal_log_ratio_families(
        families=family_scores, profile_names=[profile.name for profile in profiles]
    )
    # Multiscale combinations retain score statistics but have no standalone
    # view-coverage semantics.  Their diagnostics are the conservative minimum
    # across their fixed component families, recorded explicitly below.
    for family in family_scores:
        if family in family_diagnostics:
            continue
        label, suffix = family.split("__", 1)
        components = (
            ("radio_final_wide", "radio_intermediate_wide")
            if label == "final_intermediate_wide_equal"
            else ("radio_final_wide", "radio_intermediate_wide", "alike_wide")
        )
        names = [f"{name}__{suffix}" for name in components]
        family_diagnostics[family] = {
            key: np.min(np.stack([family_diagnostics[name][key] for name in names], axis=1), axis=1)
            for key in family_diagnostics[names[0]]
        }
    family_names = tuple(sorted(family_scores))
    row_count = len(exact["query_ids"])
    if not family_names or any(
        scores[statistic].shape != (row_count,)
        for scores in family_scores.values()
        for statistic in ("mean", "median", "worst_quartile_mean", "spatial_median_of_means_2x2")
    ):
        raise RuntimeError("maplet P1 score output shape drifted")
    arrays = {
        "query_ids": np.asarray(exact["query_ids"]).astype(str),
        "split_names": np.asarray(exact["split_names"]).astype(str),
        "evaluation_labels": np.asarray(exact["evaluation_labels"]).astype(str),
        "hypothesis_indices": np.asarray(exact["hypothesis_indices"], dtype=np.int64),
        "source_chosen_for_optional_pose": np.asarray(exact["source_chosen_for_optional_pose"], dtype=bool),
        "baseline_score_top1": np.asarray(exact["independent_score_top1"], dtype=bool),
        "baseline_selection_scores": np.asarray(exact["independent_selection_scores"], dtype=np.float64),
        "family_names": np.asarray(family_names, dtype=np.str_),
        "family_log_likelihood_means": np.stack([family_scores[name]["mean"] for name in family_names], axis=1),
        "family_log_likelihood_medians": np.stack([family_scores[name]["median"] for name in family_names], axis=1),
        "family_log_likelihood_worst_quartile_means": np.stack([family_scores[name]["worst_quartile_mean"] for name in family_names], axis=1),
        "family_spatial_median_of_means_2x2": np.stack([family_scores[name]["spatial_median_of_means_2x2"] for name in family_names], axis=1),
        "family_effective_point_counts": np.stack([family_diagnostics[name]["active_point_counts"] for name in family_names], axis=1),
        "family_effective_view_masses": np.stack([family_diagnostics[name]["active_view_masses"] for name in family_names], axis=1),
        "family_projection_visible_fractions": np.stack([family_diagnostics[name]["projection_visible_fractions"] for name in family_names], axis=1),
        "family_fit_masked_visible_fractions": np.stack([family_diagnostics[name]["fit_masked_visible_fractions"] for name in family_names], axis=1),
        "verification_source_row_indices": evidence.verification_source_rows,
        "verification_xy": evidence.verification_xy,
        "fit_query_xy": fit_xy,
        "candidate_track_ids": evidence.candidate_track_ids,
        "candidate_probabilities": evidence.candidate_probabilities,
        "null_probabilities": evidence.null_probabilities,
        "support_view_probabilities": evidence.support_view_probabilities,
        "support_image_ids": evidence.support_image_ids,
    }
    strict_contract = {
        "heldout_query_rows": True,
        "heldout_query_image_content_excludes_pnp_fit_neighborhoods": True,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": FIXED_CANDIDATE_TOP_K,
        "candidate_identity_fixed_across_hypotheses": True,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "candidate_support_view_posterior_fixed_before_pose_scoring": True,
        "candidate_group_latent_identity_marginalized": True,
        "candidate_group_topl_denominator_fixed": True,
        "candidate_group_explicit_null": True,
        "support_maplet_center_excluded": True,
        "support_maplet_topology_fixed": True,
        "maplet_neighbor_identity_fixed": True,
        "pose_dependent_correspondence_selection": False,
        "fit_neighborhood_missing_evidence_penalized": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "raw_scores_calibrated_or_promoted": False,
        "raw_scores_must_not_feed_pnp": True,
        "support_descriptor_permutation_control": args.evidence_variant == "support_descriptor_permutation_control",
        "xyz_permutation_control": args.evidence_variant == "xyz_permutation_control",
    }
    metadata: dict[str, Any] = {
        "format": SCORE_FORMAT,
        "version": SCORE_VERSION,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "evidence_variant": str(args.evidence_variant),
        "row_count": int(row_count),
        "query_count": 1,
        "query_id": query_id,
        "split_name": split_name,
        "strict_frozen_maplet_appearance_contract": strict_contract,
        "verification_selector": selector_metadata,
        "maplet_config": {
            "profiles": [
                {
                    "name": profile.name,
                    "source_name": profile.source_name,
                    "topology_key": profile.topology_key,
                    "grid_size": profile.grid_size,
                    "radius_cells": profile.radius_cells,
                }
                for profile in profiles
            ],
            "neighbors_per_quadrant": int(args.neighbors_per_quadrant),
            "minimum_active_neighbors_per_quadrant": int(args.minimum_active_neighbors_per_quadrant),
            "minimum_active_quadrants": int(args.minimum_active_quadrants),
            "quadrant_reductions": list(reductions),
            "missing_view_ratios": list(missing_ratios),
            "source_temperatures": temperatures,
            "fit_exclusion_radius_px": float(args.fit_exclusion_radius_px),
            "fit_exclusion_mask_fraction": float(fit_mask.mean()),
            "max_log_ratio": float(args.max_log_ratio),
        },
        "profile_static_layout": profile_static,
        "evidence_layout_digest": {
            "verification_source_rows_sha256": _array_digest(evidence.verification_source_rows),
            "verification_xy_sha256": _array_digest(evidence.verification_xy),
            "fit_query_xy_sha256": _array_digest(fit_xy),
            "candidate_track_ids_sha256": _array_digest(evidence.candidate_track_ids),
            "candidate_probabilities_sha256": _array_digest(evidence.candidate_probabilities),
            "null_probabilities_sha256": _array_digest(evidence.null_probabilities),
            "support_view_probabilities_sha256": _array_digest(evidence.support_view_probabilities),
            "support_image_ids_sha256": _array_digest(evidence.support_image_ids),
            "fit_exclusion_mask_sha256": _array_digest(fit_mask),
        },
        "inputs": _input_manifest(paths),
        "input_metadata": {
            "hypothesis_format": hypothesis_metadata.get("format"),
            "baseline_format": baseline_metadata.get("format"),
            "candidate_prior_format": prior_metadata.get("format"),
            "support_view_overlay_format": view_metadata.get("format"),
            "projected_landmark_bank_format": bank_metadata.get("format"),
            "support_geometry_format": geometry_metadata.get("format"),
        },
        "runtime": {
            "hypothesis_limit": int(args.hypothesis_limit),
            "all_frozen_hypotheses": bool(int(args.hypothesis_limit) == 0),
            "device": str(device),
            "elapsed_seconds": float(time.time() - started),
        },
        "implementation": {
            "script_sha256": file_sha256_short(Path(__file__)),
            "maplet_core_sha256": file_sha256_short(
                Path("feature_extract/vfm/localization/frozen_pose_conditioned_maplet_appearance.py")
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(
        json.dumps(
            {
                "stage": "score_frozen_pose_conditioned_maplet_appearance",
                "output": str(output_path),
                "output_sha256": file_sha256_short(output_path),
                "query_id": query_id,
                "row_count": int(row_count),
                "family_count": len(family_names),
                "evidence_variant": str(args.evidence_variant),
                "elapsed_seconds": float(time.time() - started),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
