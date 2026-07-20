"""Audit target-free coverage of frozen spatial-pyramid shift evidence.

The audit proves which frozen global-top20 candidate-rank buckets retain a
usable centre anchor for their candidate/support-view context crop.  Visual
crop borders are reflected from real feature maps; original-crop overlap is
audited through a separate mask-control artifact.  The audit intentionally
does not load identity labels, reprojection residuals, or poses, so coverage
cannot be confused with separability before the OOF identity probe is allowed
to run.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_hard_pairs import (
    _validate_projected_bank_lineage,
)
from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_region_context_coverage import (
    EXPECTED_QUERY_COUNTS,
    EXPECTED_ROWS_PER_QUERY,
    RANK_BUCKETS,
    frozen_candidate_ranks,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_FAMILIES,
    FULLTRACK_PER_VIEW_FEATURE_GRANULARITY_BY_EDGE_SEMANTICS,
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS,
    FrozenFulltrackPerViewAppearanceFeatures,
    _compatibility,
    profile_indices_for_fulltrack_per_view_family,
)
from feature_extract.vfm.localization.frozen_fulltrack_spatial_pyramid_shift import (
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_APPEARANCE_FORMAT,
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_APPEARANCE_FORMAT,
    SPATIAL_PYRAMID_SHIFT_FEATURE_NAMES,
    SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_NAMES,
)


ARTIFACT_FORMAT = "frozen_fulltrack_per_view_spatial_pyramid_shift_coverage_audit_v1"


@dataclass(frozen=True)
class _StreamingArtifactHeader:
    """Small immutable CSR state retained while edge-validity is streamed."""

    path: Path
    metadata: Mapping[str, Any]
    compatibility: Mapping[str, Any]
    profile_names: tuple[str, ...]
    query_ids: np.ndarray
    split_names: np.ndarray
    source_row_indices: np.ndarray
    candidate_track_ids: np.ndarray
    candidate_probabilities: np.ndarray
    candidate_support_observation_counts: np.ndarray
    edge_candidate_offsets: np.ndarray
    edge_count: int

    def __post_init__(self) -> None:
        rows = len(self.query_ids)
        candidate_shape = self.candidate_track_ids.shape
        if (
            rows <= 0
            or self.split_names.shape != (rows,)
            or self.source_row_indices.shape != (rows,)
            or self.candidate_probabilities.shape != candidate_shape
            or self.candidate_support_observation_counts.shape != candidate_shape
            or candidate_shape[0] != rows
            or self.edge_candidate_offsets.shape != (candidate_shape[0] * candidate_shape[1] + 1,)
            or int(self.edge_candidate_offsets[0]) != 0
            or int(self.edge_candidate_offsets[-1]) != int(self.edge_count)
        ):
            raise ValueError("streaming spatial-pyramid artifact header is invalid")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifacts", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--families",
        default="all",
        help="comma-separated spatial-pyramid families; all selects every compatible family",
    )
    parser.add_argument(
        "--artifact-role",
        choices=("visual", "mask_control"),
        default="visual",
        help="audit descriptor correlations or the separate original-crop mask control",
    )
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("spatial-pyramid artifact paths must be non-empty and unique")
    return paths


def _semantic_for_role(artifact_role: str) -> str:
    if str(artifact_role) == "visual":
        return FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS
    if str(artifact_role) == "mask_control":
        return FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS
    raise ValueError("spatial-pyramid artifact role is invalid")


def _families(value: str, *, artifact_role: str = "visual") -> tuple[str, ...]:
    expected_semantics = _semantic_for_role(artifact_role)
    if str(value).strip() == "all":
        names = tuple(
            name
            for name, spec in FULLTRACK_PER_VIEW_FAMILIES.items()
            if spec.edge_feature_semantics == expected_semantics
        )
    else:
        names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if (
        not names
        or len(set(names)) != len(names)
        or set(names).difference(FULLTRACK_PER_VIEW_FAMILIES)
        or any(
            FULLTRACK_PER_VIEW_FAMILIES[name].edge_feature_semantics
            != expected_semantics
            for name in names
        )
    ):
        raise ValueError("spatial-pyramid audit families are invalid or incompatible")
    return names


def _load_streaming_header(path: Path) -> _StreamingArtifactHeader:
    """Read only the compact CSR header, never the 666-D score tensor.

    ``np.load`` on an ``.npz`` member materializes that member in memory.  The
    target-free coverage audit only needs validity masks, so keeping scores out
    of this header is what makes the complete 84-query audit bounded-memory.
    """

    source = Path(path)
    required = {
        "verification_query_ids",
        "split_names",
        "verification_source_row_indices",
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_support_observation_counts",
        "profile_names",
        "edge_candidate_offsets",
        "edge_geometry_rows",
        "edge_profile_scores",
        "edge_profile_valid",
        "metadata_json",
    }
    with np.load(source, allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{source}: spatial-pyramid artifact lacks {missing}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{source}: spatial-pyramid metadata is invalid")
        query_ids = np.asarray(payload["verification_query_ids"]).astype(str)
        split_names = np.asarray(payload["split_names"]).astype(str)
        source_rows = np.asarray(payload["verification_source_row_indices"], dtype=np.int64)
        tracks = np.asarray(payload["candidate_track_ids"], dtype=np.int64)
        candidate = np.asarray(payload["candidate_probabilities"], dtype=np.float32)
        null = np.asarray(payload["null_probabilities"], dtype=np.float32)
        counts = np.asarray(payload["candidate_support_observation_counts"], dtype=np.int64)
        offsets = np.asarray(payload["edge_candidate_offsets"], dtype=np.int64)
        profile_names = tuple(np.asarray(payload["profile_names"]).astype(str).tolist())
        edge_count = int(np.asarray(payload["edge_geometry_rows"]).shape[0])
    if (
        len(query_ids) != EXPECTED_ROWS_PER_QUERY
        or len(set(query_ids.tolist())) != 1
        or len(set(split_names.tolist())) != 1
        or str(split_names[0]) not in EXPECTED_QUERY_COUNTS
        or len(np.unique(source_rows)) != EXPECTED_ROWS_PER_QUERY
        or tracks.shape != candidate.shape
        or tracks.shape != counts.shape
        or tracks.shape != (len(query_ids), 20)
        or null.shape != (len(query_ids),)
        or len(profile_names) == 0
        or len(set(profile_names)) != len(profile_names)
        or offsets.shape != (counts.size + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != edge_count
        or not np.array_equal(np.diff(offsets), counts.reshape(-1))
        or np.any(counts < 0)
        or np.any(~np.isfinite(candidate))
        or np.any(~np.isfinite(null))
        or np.any(candidate < 0.0)
        or np.any(null < 0.0)
        or np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 2e-5
    ):
        raise ValueError(f"{source}: spatial-pyramid CSR header is invalid")
    return _StreamingArtifactHeader(
        path=source,
        metadata=metadata,
        compatibility=_compatibility(metadata),
        profile_names=profile_names,
        query_ids=query_ids,
        split_names=split_names,
        source_row_indices=source_rows,
        candidate_track_ids=tracks,
        candidate_probabilities=candidate,
        candidate_support_observation_counts=counts,
        edge_candidate_offsets=offsets,
        edge_count=edge_count,
    )


def _load_streaming_headers(paths: Sequence[Path]) -> tuple[_StreamingArtifactHeader, ...]:
    """Validate that the complete set shares one immutable CSR contract."""

    headers = tuple(_load_streaming_header(Path(path)) for path in paths)
    if not headers:
        raise ValueError("spatial-pyramid artifacts must be non-empty")
    compatibility = headers[0].compatibility
    profile_names = headers[0].profile_names
    seen_rows: set[tuple[str, int]] = set()
    split_queries: dict[str, set[str]] = {split: set() for split in EXPECTED_QUERY_COUNTS}
    for header in headers:
        if header.compatibility != compatibility or header.profile_names != profile_names:
            raise ValueError("spatial-pyramid artifacts have incompatible frozen contracts")
        query_id = str(header.query_ids[0])
        split = str(header.split_names[0])
        split_queries[split].add(query_id)
        keys = {(str(value), int(row)) for value, row in zip(header.query_ids, header.source_row_indices)}
        if len(keys) != EXPECTED_ROWS_PER_QUERY or seen_rows.intersection(keys):
            raise ValueError("spatial-pyramid artifacts overlap query/source rows")
        seen_rows.update(keys)
    for split, expected_queries in EXPECTED_QUERY_COUNTS.items():
        if len(split_queries[split]) != expected_queries:
            raise ValueError("spatial-pyramid artifacts have incomplete query coverage")
    return headers


def _validate_complete_frozen_set(
    features: FrozenFulltrackPerViewAppearanceFeatures,
) -> None:
    if set(features.split_names.tolist()).difference(EXPECTED_QUERY_COUNTS):
        raise ValueError("spatial-pyramid artifacts have an unsupported split")
    for split, expected_queries in EXPECTED_QUERY_COUNTS.items():
        rows = np.flatnonzero(features.split_names == split)
        query_ids = features.query_ids[rows]
        unique = tuple(dict.fromkeys(query_ids.tolist()))
        if len(unique) != expected_queries:
            raise ValueError("spatial-pyramid artifacts have incomplete query coverage")
        for query_id in unique:
            query_rows = rows[query_ids == query_id]
            if (
                len(query_rows) != EXPECTED_ROWS_PER_QUERY
                or len(np.unique(features.source_row_indices[query_rows]))
                != EXPECTED_ROWS_PER_QUERY
            ):
                raise ValueError("spatial-pyramid artifact has an incomplete query shard")


def _validate_spatial_pyramid_metadata_contract(
    *,
    paths: Sequence[Path],
    metadata_rows: Sequence[Mapping[str, Any]],
    profile_names: Sequence[str],
    artifact_role: str,
) -> None:
    is_visual = str(artifact_role) == "visual"
    expected_names = (
        SPATIAL_PYRAMID_SHIFT_FEATURE_NAMES
        if is_visual
        else SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_NAMES
    )
    expected_format = (
        FULLTRACK_SPATIAL_PYRAMID_SHIFT_APPEARANCE_FORMAT
        if is_visual
        else FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_APPEARANCE_FORMAT
    )
    expected_role = (
        "visual_descriptor_correlation"
        if is_visual
        else "original_crop_mask_overlap_control"
    )
    if tuple(profile_names) != tuple(expected_names):
        raise ValueError("spatial-pyramid profile order differs from its frozen contract")
    if len(paths) != len(metadata_rows) or not paths:
        raise ValueError("spatial-pyramid metadata rows are incomplete")
    for path, metadata in zip(paths, metadata_rows):
        strict = metadata.get("strict_fulltrack_appearance_contract")
        contract = metadata.get("spatial_pyramid_shift_contract")
        if (
            metadata.get("format") != expected_format
            or not isinstance(strict, Mapping)
            or strict.get("candidate_identity_fixed") is not True
            or strict.get("candidate_posterior_preserved") is not True
            or strict.get("candidate_reselection") is not False
            or strict.get("support_reselection") is not False
            or strict.get("all_real_sfm_track_observations_enumerated") is not True
            or strict.get("candidate_3d_projection_or_pose_used") is not False
            or strict.get("image_retrieval_or_submap_used") is not False
            or strict.get("render") is not False
            or strict.get("incomplete_center_anchor_is_unknown_not_visual_value")
            is not True
            or strict.get("visual_descriptor_values_included") is not is_visual
            or strict.get("original_crop_mask_values_included") is not (not is_visual)
            or strict.get("paired_mask_control_artifact_required") is not True
            or metadata.get("artifact_role") != expected_role
            or not isinstance(contract, Mapping)
            or contract.get("mode")
            != "candidate_specific_multiscale_spatial_pyramid_shift_correlation_v1"
            or contract.get("support_coordinate_source") != "sfm_observation_xy"
            or contract.get("view_aggregation")
            != "none_before_learned_logsumexp_mixture_v1"
            or contract.get("missing_evidence")
            != "invalid_center_anchor_edge_omitted_neutral_v1"
            or contract.get("visual_border_padding")
            != "reflection_from_real_feature_map_v1"
            or contract.get("original_crop_mask_control_exported_separately") is not True
            or contract.get("explicit_availability_or_neighbor_count_feature")
            is not False
            or contract.get("full_crop_required") is not False
            or contract.get("center_anchor_required") is not True
            or not isinstance(contract.get("profiles"), list)
            or not contract["profiles"]
        ):
            raise ValueError(f"{path}: spatial-pyramid contract is invalid")
        if is_visual:
            appearance = metadata.get("appearance_config")
            if (
                not isinstance(appearance, Mapping)
                or appearance.get("original_crop_mask_values_included") is not False
                # The immutable strict contract is authoritative for these
                # two flags.  The original visual exporter predates the
                # optional duplicated appearance-config fields, so omission
                # means the visual defaults rather than control semantics.
                or appearance.get("visual_descriptor_values_included", True) is not True
                or appearance.get("control_only", False) is not False
            ):
                raise ValueError(f"{path}: visual artifact contains control semantics")
        else:
            appearance = metadata.get("appearance_config")
            if (
                not isinstance(appearance, Mapping)
                or appearance.get("original_crop_mask_values_included") is not True
                or appearance.get("visual_descriptor_values_included") is not False
                or appearance.get("control_only") is not True
                or not isinstance(contract.get("mask_control_profiles"), list)
                or not contract["mask_control_profiles"]
            ):
                raise ValueError(f"{path}: mask-control artifact lacks its separation contract")


def _validate_spatial_pyramid_contract(
    features: FrozenFulltrackPerViewAppearanceFeatures,
    *,
    artifact_role: str,
) -> None:
    """Keep the in-memory helper for small unit fixtures and direct callers."""

    _validate_spatial_pyramid_metadata_contract(
        paths=features.paths,
        metadata_rows=features.artifact_metadata,
        profile_names=features.profile_names,
        artifact_role=artifact_role,
    )


def _edge_candidate_indices(
    features: FrozenFulltrackPerViewAppearanceFeatures,
) -> np.ndarray:
    counts = np.asarray(features.candidate_support_observation_counts, dtype=np.int64)
    return np.repeat(np.arange(counts.size, dtype=np.int64), counts.reshape(-1))


def summarize_spatial_pyramid_shift_coverage(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    families: Sequence[str],
    artifact_role: str = "visual",
) -> list[dict[str, Any]]:
    """Summarize actual visual-edge coverage by immutable candidate rank."""

    family_names = _families(
        ",".join(str(name) for name in families), artifact_role=artifact_role
    )
    ranks = frozen_candidate_ranks(features)
    candidate = np.asarray(features.candidate_probabilities, dtype=np.float32)
    counts = np.asarray(features.candidate_support_observation_counts, dtype=np.int64)
    edge_candidates = _edge_candidate_indices(features)
    if len(edge_candidates) != len(features.edge_geometry_rows):
        raise ValueError("spatial-pyramid CSR edge count is inconsistent")
    edge_rows = edge_candidates // features.candidate_count
    edge_columns = edge_candidates % features.candidate_count
    edge_ranks = ranks[edge_rows, edge_columns]
    output: list[dict[str, Any]] = []
    for family in family_names:
        profile_indices = profile_indices_for_fulltrack_per_view_family(
            family, profile_names=features.profile_names
        )
        edge_valid = np.all(features.edge_profile_valid[:, profile_indices], axis=1)
        candidate_usable_views = np.zeros((candidate.size,), dtype=np.int64)
        np.add.at(candidate_usable_views, edge_candidates, edge_valid.astype(np.int64))
        for split in EXPECTED_QUERY_COUNTS:
            row_mask = features.split_names == split
            for bucket_name, minimum_rank, maximum_rank in RANK_BUCKETS:
                candidate_mask = (
                    row_mask[:, None]
                    & (ranks >= int(minimum_rank))
                    & (ranks <= int(maximum_rank))
                )
                flat = np.flatnonzero(candidate_mask.reshape(-1))
                masses = candidate.reshape(-1)[flat]
                support_counts = counts.reshape(-1)[flat]
                usable_views = candidate_usable_views[flat]
                usable_candidate = usable_views > 0
                candidate_mass = float(np.sum(masses, dtype=np.float64))
                covered_mass = float(np.sum(masses[usable_candidate], dtype=np.float64))
                edge_mask = row_mask[edge_rows] & (edge_ranks >= int(minimum_rank)) & (
                    edge_ranks <= int(maximum_rank)
                )
                edge_count = int(np.sum(edge_mask))
                usable_edge_count = int(np.sum(edge_valid[edge_mask]))
                output.append(
                    {
                        "family": family,
                        "profile_feature_count": int(len(profile_indices)),
                        "split": split,
                        "rank_bucket": bucket_name,
                        "candidate_count": int(len(flat)),
                        "candidate_prior_mass": candidate_mass,
                        "candidate_with_usable_spatial_pyramid_edge_count": int(
                            np.sum(usable_candidate)
                        ),
                        "candidate_coverage_rate": (
                            float(np.mean(usable_candidate)) if len(flat) else None
                        ),
                        "candidate_prior_mass_covered": covered_mass,
                        "candidate_prior_mass_coverage_rate": (
                            covered_mass / candidate_mass if candidate_mass > 0.0 else None
                        ),
                        "mean_real_support_view_count": (
                            float(np.mean(support_counts)) if len(flat) else None
                        ),
                        "mean_usable_spatial_pyramid_view_count": (
                            float(np.mean(usable_views)) if len(flat) else None
                        ),
                        "edge_count": edge_count,
                        "usable_spatial_pyramid_edge_count": usable_edge_count,
                        "joint_edge_coverage_rate": (
                            usable_edge_count / edge_count if edge_count else None
                        ),
                        "missing_edge_rate": (
                            1.0 - usable_edge_count / edge_count if edge_count else None
                        ),
                    }
                )
    return output


def _frozen_candidate_ranks(
    candidate_track_ids: np.ndarray, candidate_probabilities: np.ndarray
) -> np.ndarray:
    """Return fixed posterior ranks without materializing a feature tensor."""

    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float32)
    if tracks.shape != candidate.shape or tracks.ndim != 2:
        raise ValueError("streaming candidate ranks need a rectangular top-L matrix")
    ranks = np.full(tracks.shape, -1, dtype=np.int64)
    for row in range(len(tracks)):
        valid = (tracks[row] >= 0) & (candidate[row] > 0.0)
        columns = np.flatnonzero(valid)
        order = columns[np.argsort(-candidate[row, columns], kind="stable")]
        ranks[row, order] = np.arange(1, len(order) + 1, dtype=np.int64)
    return ranks


def _empty_coverage_totals(*, profile_feature_count: int) -> dict[str, float | int]:
    return {
        "profile_feature_count": int(profile_feature_count),
        "candidate_count": 0,
        "candidate_prior_mass": 0.0,
        "candidate_with_usable_spatial_pyramid_edge_count": 0,
        "candidate_prior_mass_covered": 0.0,
        "real_support_view_sum": 0.0,
        "usable_spatial_pyramid_view_sum": 0.0,
        "edge_count": 0,
        "usable_spatial_pyramid_edge_count": 0,
    }


def _stream_spatial_pyramid_shift_coverage(
    *,
    headers: Sequence[_StreamingArtifactHeader],
    families: Sequence[str],
    artifact_role: str,
) -> list[dict[str, Any]]:
    """Aggregate coverage one `.npz` shard at a time.

    This must never touch ``edge_profile_scores``: they are not evidence for a
    coverage audit, and loading all of them would transiently allocate tens of
    GiB after the float16-to-float32 conversion used by the training loader.
    """

    if not headers:
        raise ValueError("streaming coverage needs at least one artifact header")
    family_names = _families(
        ",".join(str(name) for name in families), artifact_role=artifact_role
    )
    profile_indices = {
        family: profile_indices_for_fulltrack_per_view_family(
            family, profile_names=headers[0].profile_names
        )
        for family in family_names
    }
    totals: dict[tuple[str, str, str], dict[str, float | int]] = {
        (family, split, bucket): _empty_coverage_totals(
            profile_feature_count=len(profile_indices[family])
        )
        for family in family_names
        for split in EXPECTED_QUERY_COUNTS
        for bucket, _minimum, _maximum in RANK_BUCKETS
    }
    for header in headers:
        ranks = _frozen_candidate_ranks(
            header.candidate_track_ids, header.candidate_probabilities
        )
        candidate_count = int(header.candidate_track_ids.shape[1])
        counts = np.asarray(header.candidate_support_observation_counts, dtype=np.int64)
        edge_candidates = np.repeat(
            np.arange(counts.size, dtype=np.int64), counts.reshape(-1)
        )
        if len(edge_candidates) != int(header.edge_count):
            raise ValueError(f"{header.path}: streaming CSR edge count is inconsistent")
        candidate_rows = np.repeat(
            np.arange(len(header.query_ids), dtype=np.int64), candidate_count
        )
        edge_rows = edge_candidates // candidate_count
        edge_ranks = ranks.reshape(-1)[edge_candidates]
        candidate_ranks = ranks.reshape(-1)
        candidate_mass = np.asarray(header.candidate_probabilities, dtype=np.float32).reshape(-1)
        support_counts = counts.reshape(-1)
        with np.load(header.path, allow_pickle=False) as payload:
            edge_profile_valid = np.asarray(payload["edge_profile_valid"], dtype=bool)
        if edge_profile_valid.shape != (int(header.edge_count), len(header.profile_names)):
            raise ValueError(f"{header.path}: streaming edge-validity shape is invalid")
        for family in family_names:
            edge_valid = np.all(edge_profile_valid[:, profile_indices[family]], axis=1)
            candidate_usable_views = np.zeros((counts.size,), dtype=np.int64)
            np.add.at(candidate_usable_views, edge_candidates, edge_valid.astype(np.int64))
            for split in EXPECTED_QUERY_COUNTS:
                row_mask = header.split_names == split
                candidate_split_mask = row_mask[candidate_rows]
                edge_split_mask = row_mask[edge_rows]
                for bucket, minimum_rank, maximum_rank in RANK_BUCKETS:
                    key = (family, split, bucket)
                    total = totals[key]
                    candidate_mask = (
                        candidate_split_mask
                        & (candidate_ranks >= int(minimum_rank))
                        & (candidate_ranks <= int(maximum_rank))
                    )
                    flat = np.flatnonzero(candidate_mask)
                    usable_candidate = candidate_usable_views[flat] > 0
                    masses = candidate_mass[flat]
                    total["candidate_count"] = int(total["candidate_count"]) + int(len(flat))
                    total["candidate_prior_mass"] = float(total["candidate_prior_mass"]) + float(
                        np.sum(masses, dtype=np.float64)
                    )
                    total["candidate_with_usable_spatial_pyramid_edge_count"] = int(
                        total["candidate_with_usable_spatial_pyramid_edge_count"]
                    ) + int(np.sum(usable_candidate))
                    total["candidate_prior_mass_covered"] = float(
                        total["candidate_prior_mass_covered"]
                    ) + float(np.sum(masses[usable_candidate], dtype=np.float64))
                    total["real_support_view_sum"] = float(total["real_support_view_sum"]) + float(
                        np.sum(support_counts[flat], dtype=np.float64)
                    )
                    total["usable_spatial_pyramid_view_sum"] = float(
                        total["usable_spatial_pyramid_view_sum"]
                    ) + float(
                        np.sum(candidate_usable_views[flat], dtype=np.float64)
                    )
                    edge_mask = (
                        edge_split_mask
                        & (edge_ranks >= int(minimum_rank))
                        & (edge_ranks <= int(maximum_rank))
                    )
                    total["edge_count"] = int(total["edge_count"]) + int(np.sum(edge_mask))
                    total["usable_spatial_pyramid_edge_count"] = int(
                        total["usable_spatial_pyramid_edge_count"]
                    ) + int(np.sum(edge_valid[edge_mask]))
        del edge_profile_valid
    output: list[dict[str, Any]] = []
    for family in family_names:
        for split in EXPECTED_QUERY_COUNTS:
            for bucket, _minimum_rank, _maximum_rank in RANK_BUCKETS:
                total = totals[(family, split, bucket)]
                candidate_total = int(total["candidate_count"])
                edge_total = int(total["edge_count"])
                candidate_mass_total = float(total["candidate_prior_mass"])
                usable_candidate_total = int(
                    total["candidate_with_usable_spatial_pyramid_edge_count"]
                )
                usable_edge_total = int(total["usable_spatial_pyramid_edge_count"])
                output.append(
                    {
                        "family": family,
                        "profile_feature_count": int(total["profile_feature_count"]),
                        "split": split,
                        "rank_bucket": bucket,
                        "candidate_count": candidate_total,
                        "candidate_prior_mass": candidate_mass_total,
                        "candidate_with_usable_spatial_pyramid_edge_count": usable_candidate_total,
                        "candidate_coverage_rate": (
                            usable_candidate_total / candidate_total
                            if candidate_total
                            else None
                        ),
                        "candidate_prior_mass_covered": float(
                            total["candidate_prior_mass_covered"]
                        ),
                        "candidate_prior_mass_coverage_rate": (
                            float(total["candidate_prior_mass_covered"])
                            / candidate_mass_total
                            if candidate_mass_total > 0.0
                            else None
                        ),
                        "mean_real_support_view_count": (
                            float(total["real_support_view_sum"]) / candidate_total
                            if candidate_total
                            else None
                        ),
                        "mean_usable_spatial_pyramid_view_count": (
                            float(total["usable_spatial_pyramid_view_sum"]) / candidate_total
                            if candidate_total
                            else None
                        ),
                        "edge_count": edge_total,
                        "usable_spatial_pyramid_edge_count": usable_edge_total,
                        "joint_edge_coverage_rate": (
                            usable_edge_total / edge_total if edge_total else None
                        ),
                        "missing_edge_rate": (
                            1.0 - usable_edge_total / edge_total if edge_total else None
                        ),
                    }
                )
    return output


def audit_frozen_fulltrack_per_view_spatial_pyramid_shift_coverage(
    *,
    appearance_artifacts: Sequence[Path],
    projected_landmark_bank: Path,
    output_dir: Path,
    families: Sequence[str],
    artifact_role: str = "visual",
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite spatial-pyramid coverage audit")
    paths = tuple(Path(path) for path in appearance_artifacts)
    headers = _load_streaming_headers(paths)
    expected_semantics = _semantic_for_role(artifact_role)
    if headers[0].compatibility.get("per_view_edge_feature_semantics") != expected_semantics:
        raise ValueError("coverage audit needs full-CSR spatial-pyramid artifacts")
    _validate_projected_bank_lineage(
        tuple(header.metadata for header in headers),
        projected_landmark_bank=Path(projected_landmark_bank),
    )
    _validate_spatial_pyramid_metadata_contract(
        paths=paths,
        metadata_rows=tuple(header.metadata for header in headers),
        profile_names=headers[0].profile_names,
        artifact_role=artifact_role,
    )
    rows = _stream_spatial_pyramid_shift_coverage(
        headers=headers, families=families, artifact_role=artifact_role
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "rank_bucket_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": "audit_frozen_fulltrack_per_view_spatial_pyramid_shift_coverage",
        "format": ARTIFACT_FORMAT,
        "appearance_artifact_count": int(len(paths)),
        "appearance_artifact_sha256": [file_sha256_short(path) for path in paths],
        "projected_landmark_bank": {
            "path": str(projected_landmark_bank),
            "sha256": file_sha256_short(Path(projected_landmark_bank)),
        },
        "families": list(families),
        "artifact_role": str(artifact_role),
        "feature_granularity": FULLTRACK_PER_VIEW_FEATURE_GRANULARITY_BY_EDGE_SEMANTICS[
            expected_semantics
        ],
        "row_count": int(sum(len(header.query_ids) for header in headers)),
        "edge_count": int(sum(header.edge_count for header in headers)),
        "rank_buckets": [name for name, _minimum, _maximum in RANK_BUCKETS],
        "coverage_rows": rows,
        "protocol": {
            "target_free": True,
            "identity_or_pose_targets_loaded": False,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_support_observations_retained": True,
            "support_view_features_averaged_before_inference": False,
            "incomplete_crop_is_unknown_not_identity_feature": True,
            "coverage_aggregation": "one_npz_shard_at_a_time_edge_validity_only_v1",
            "edge_profile_scores_loaded": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = audit_frozen_fulltrack_per_view_spatial_pyramid_shift_coverage(
        appearance_artifacts=_paths(str(args.appearance_artifacts)),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        output_dir=Path(args.output_dir),
        families=_families(str(args.families), artifact_role=str(args.artifact_role)),
        artifact_role=str(args.artifact_role),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
