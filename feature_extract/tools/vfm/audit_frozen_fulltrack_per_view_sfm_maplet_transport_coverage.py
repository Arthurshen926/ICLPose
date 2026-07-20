"""Audit target-free coverage and lineage of SfM-maplet transport evidence.

This audit deliberately reads no identity labels, targets, poses, or
hypotheses.  It verifies that every frozen global top-20 candidate rank bucket
retains usable center-excluded maplet evidence and that maplet availability is
only a diagnostic, never a learned identity input.
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
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES,
    FrozenFulltrackPerViewAppearanceFeatures,
    fulltrack_per_view_feature_granularity,
    load_frozen_fulltrack_per_view_appearance_features,
    profile_indices_for_fulltrack_per_view_family,
)
from feature_extract.vfm.localization.frozen_fulltrack_sfm_maplet_transport import (
    SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE,
    SFM_MAPLET_TRANSPORT_PROFILES,
)


ARTIFACT_FORMAT = "frozen_fulltrack_per_view_sfm_maplet_transport_coverage_audit_v1"


@dataclass(frozen=True)
class MapletEdgeDiagnostics:
    """Audit-only support-maplet availability aligned to merged CSR edges."""

    profile_usable: np.ndarray
    support_quadrant_counts: np.ndarray

    def __post_init__(self) -> None:
        usable = np.asarray(self.profile_usable, dtype=bool)
        counts = np.asarray(self.support_quadrant_counts, dtype=np.int64)
        if (
            usable.ndim != 2
            or usable.shape[1] != len(SFM_MAPLET_TRANSPORT_PROFILES)
            or counts.shape != (*usable.shape, 4)
            or np.any(counts < 0)
        ):
            raise ValueError("SfM-maplet audit diagnostics are invalid")
        object.__setattr__(self, "profile_usable", usable)
        object.__setattr__(self, "support_quadrant_counts", counts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifacts", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--families",
        default="all",
        help="comma-separated SfM-maplet family names; all selects the frozen S1 sweep",
    )
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("SfM-maplet artifact paths must be non-empty and unique")
    return paths


def _families(value: str) -> tuple[str, ...]:
    if str(value).strip() == "all":
        names = tuple(
            name
            for name, spec in FULLTRACK_PER_VIEW_FAMILIES.items()
            if spec.edge_feature_semantics
            == FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        )
    else:
        names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if (
        not names
        or len(set(names)) != len(names)
        or set(names).difference(FULLTRACK_PER_VIEW_FAMILIES)
        or any(
            FULLTRACK_PER_VIEW_FAMILIES[name].edge_feature_semantics
            != FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
            for name in names
        )
    ):
        raise ValueError("SfM-maplet audit families are invalid or incompatible")
    return names


def _validate_complete_frozen_set(features: FrozenFulltrackPerViewAppearanceFeatures) -> None:
    if set(features.split_names.tolist()).difference(EXPECTED_QUERY_COUNTS):
        raise ValueError("SfM-maplet artifacts have an unsupported split")
    for split, expected_queries in EXPECTED_QUERY_COUNTS.items():
        rows = np.flatnonzero(features.split_names == split)
        query_ids = features.query_ids[rows]
        unique = tuple(dict.fromkeys(query_ids.tolist()))
        if len(unique) != expected_queries:
            raise ValueError("SfM-maplet artifacts have incomplete query coverage")
        for query_id in unique:
            query_rows = rows[query_ids == query_id]
            if (
                len(query_rows) != EXPECTED_ROWS_PER_QUERY
                or len(np.unique(features.source_row_indices[query_rows]))
                != EXPECTED_ROWS_PER_QUERY
            ):
                raise ValueError("SfM-maplet artifact has an incomplete query shard")


def _edge_candidate_indices(features: FrozenFulltrackPerViewAppearanceFeatures) -> np.ndarray:
    counts = np.asarray(features.candidate_support_observation_counts, dtype=np.int64)
    return np.repeat(np.arange(counts.size, dtype=np.int64), counts.reshape(-1))


def _maplet_profile_feature_indices(
    profile_names: Sequence[str],
) -> dict[str, np.ndarray]:
    names = tuple(str(name) for name in profile_names)
    if names != tuple(FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES):
        raise ValueError("SfM-maplet profile feature order differs from the frozen contract")
    return {
        profile.name: np.asarray(
            [names.index(name) for name in SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name]],
            dtype=np.int64,
        )
        for profile in SFM_MAPLET_TRANSPORT_PROFILES
    }


def _family_maplet_profile_indices(
    *, family: str, profile_names: Sequence[str]
) -> np.ndarray:
    family_indices = set(
        profile_indices_for_fulltrack_per_view_family(
            family, profile_names=profile_names
        ).tolist()
    )
    groups = _maplet_profile_feature_indices(profile_names)
    selected = [
        index
        for index, profile in enumerate(SFM_MAPLET_TRANSPORT_PROFILES)
        if set(groups[profile.name].tolist()).issubset(family_indices)
    ]
    if not selected or set().union(
        *(set(groups[SFM_MAPLET_TRANSPORT_PROFILES[index].name].tolist()) for index in selected)
    ) != family_indices:
        raise ValueError("SfM-maplet family does not select complete profile groups")
    return np.asarray(selected, dtype=np.int64)


def _load_maplet_diagnostics(
    *,
    paths: Sequence[Path],
    features: FrozenFulltrackPerViewAppearanceFeatures,
) -> MapletEdgeDiagnostics:
    groups = _maplet_profile_feature_indices(features.profile_names)
    usable_blocks: list[np.ndarray] = []
    count_blocks: list[np.ndarray] = []
    for path, metadata in zip(paths, features.artifact_metadata):
        strict = metadata.get("strict_fulltrack_appearance_contract")
        contract = metadata.get("sfm_maplet_transport_contract")
        if (
            not isinstance(strict, Mapping)
            or strict.get("candidate_center_descriptor_excluded") is not True
            or strict.get("missing_or_partial_maplet_is_unknown") is not True
            or strict.get("availability_or_neighbor_count_is_not_a_learned_feature")
            is not True
            or not isinstance(contract, Mapping)
            or contract.get("availability_is_a_learned_feature") is not False
        ):
            raise ValueError("SfM-maplet artifact violates the center-excluded contract")
        with np.load(Path(path), allow_pickle=False) as payload:
            required = {
                "profile_names",
                "edge_profile_valid",
                "edge_maplet_profile_usable",
                "edge_maplet_support_quadrant_counts",
            }
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(f"{path}: SfM-maplet diagnostics lack {missing}")
            names = tuple(np.asarray(payload["profile_names"]).astype(str).tolist())
            valid = np.asarray(payload["edge_profile_valid"], dtype=bool)
            usable = np.asarray(payload["edge_maplet_profile_usable"], dtype=bool)
            counts = np.asarray(payload["edge_maplet_support_quadrant_counts"], dtype=np.int64)
        if names != tuple(features.profile_names):
            raise ValueError(f"{path}: SfM-maplet feature profile order differs")
        diagnostics = MapletEdgeDiagnostics(
            profile_usable=usable, support_quadrant_counts=counts
        )
        if valid.shape != (len(usable), len(features.profile_names)):
            raise ValueError(f"{path}: SfM-maplet edge validity shape differs")
        for profile_index, profile in enumerate(SFM_MAPLET_TRANSPORT_PROFILES):
            expected = np.all(valid[:, groups[profile.name]], axis=1)
            if not np.array_equal(expected, usable[:, profile_index]):
                raise ValueError(
                    f"{path}: SfM-maplet profile validity is not an all-quadrant unknown mask"
                )
            if np.any(usable[:, profile_index] & np.any(counts[:, profile_index] < 1, axis=1)):
                raise ValueError(f"{path}: usable SfM-maplet profile lacks a real quadrant")
        usable_blocks.append(diagnostics.profile_usable)
        count_blocks.append(diagnostics.support_quadrant_counts)
    diagnostics = MapletEdgeDiagnostics(
        profile_usable=np.concatenate(usable_blocks, axis=0),
        support_quadrant_counts=np.concatenate(count_blocks, axis=0),
    )
    if len(diagnostics.profile_usable) != len(features.edge_geometry_rows):
        raise ValueError("merged SfM-maplet diagnostic CSR edge count differs")
    return diagnostics


def summarize_sfm_maplet_transport_coverage(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    diagnostics: MapletEdgeDiagnostics,
    families: Sequence[str],
) -> list[dict[str, Any]]:
    """Summarize frozen-rank coverage without consuming correctness labels."""

    family_names = _families(",".join(str(name) for name in families))
    if len(diagnostics.profile_usable) != len(features.edge_geometry_rows):
        raise ValueError("SfM-maplet coverage inputs have inconsistent CSR edges")
    ranks = frozen_candidate_ranks(features)
    tracks = np.asarray(features.candidate_track_ids, dtype=np.int64)
    candidate = np.asarray(features.candidate_probabilities, dtype=np.float32)
    counts = np.asarray(features.candidate_support_observation_counts, dtype=np.int64)
    edge_candidates = _edge_candidate_indices(features)
    edge_rows = edge_candidates // features.candidate_count
    edge_columns = edge_candidates % features.candidate_count
    edge_ranks = ranks[edge_rows, edge_columns]
    output: list[dict[str, Any]] = []
    for family in family_names:
        feature_indices = profile_indices_for_fulltrack_per_view_family(
            family, profile_names=features.profile_names
        )
        maplet_indices = _family_maplet_profile_indices(
            family=family, profile_names=features.profile_names
        )
        edge_usable = np.all(diagnostics.profile_usable[:, maplet_indices], axis=1)
        expected_usable = np.all(features.edge_profile_valid[:, feature_indices], axis=1)
        if not np.array_equal(edge_usable, expected_usable):
            raise ValueError("SfM-maplet feature validity and audit availability differ")
        candidate_usable_views = np.zeros((tracks.size,), dtype=np.int64)
        np.add.at(candidate_usable_views, edge_candidates, edge_usable.astype(np.int64))
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
                usable_views = candidate_usable_views[flat]
                usable_candidate = usable_views > 0
                candidate_mass = float(np.sum(masses, dtype=np.float64))
                covered_mass = float(np.sum(masses[usable_candidate], dtype=np.float64))
                edge_mask = row_mask[edge_rows] & (edge_ranks >= int(minimum_rank)) & (
                    edge_ranks <= int(maximum_rank)
                )
                selected_counts = diagnostics.support_quadrant_counts[
                    edge_mask][:, maplet_indices, :
                ]
                if len(selected_counts):
                    per_profile_minimum = np.min(selected_counts, axis=2)
                    complete_quadrants = np.all(selected_counts >= 1, axis=(1, 2))
                    mean_minimum_neighbors = float(np.mean(per_profile_minimum))
                else:
                    complete_quadrants = np.zeros((0,), dtype=bool)
                    mean_minimum_neighbors = None
                edge_count = int(np.sum(edge_mask))
                usable_edge_count = int(np.sum(edge_usable[edge_mask]))
                output.append(
                    {
                        "family": family,
                        "maplet_profile_count": int(len(maplet_indices)),
                        "split": split,
                        "rank_bucket": bucket_name,
                        "candidate_count": int(len(flat)),
                        "candidate_prior_mass": candidate_mass,
                        "candidate_with_usable_maplet_edge_count": int(np.sum(usable_candidate)),
                        "candidate_coverage_rate": (
                            float(np.mean(usable_candidate)) if len(flat) else None
                        ),
                        "candidate_prior_mass_covered": covered_mass,
                        "candidate_prior_mass_coverage_rate": (
                            covered_mass / candidate_mass if candidate_mass > 0.0 else None
                        ),
                        "mean_real_support_view_count": (
                            float(np.mean(counts.reshape(-1)[flat])) if len(flat) else None
                        ),
                        "mean_usable_maplet_view_count": (
                            float(np.mean(usable_views)) if len(flat) else None
                        ),
                        "edge_count": edge_count,
                        "usable_maplet_edge_count": usable_edge_count,
                        "joint_edge_coverage_rate": (
                            usable_edge_count / edge_count if edge_count else None
                        ),
                        "all_support_quadrants_present_edge_count": int(np.sum(complete_quadrants)),
                        "all_support_quadrants_present_rate": (
                            float(np.mean(complete_quadrants)) if edge_count else None
                        ),
                        "mean_minimum_support_neighbors_per_quadrant": mean_minimum_neighbors,
                        "missing_edge_rate": (
                            1.0 - usable_edge_count / edge_count if edge_count else None
                        ),
                    }
                )
    return output


def audit_frozen_fulltrack_per_view_sfm_maplet_transport_coverage(
    *,
    appearance_artifacts: Sequence[Path],
    projected_landmark_bank: Path,
    output_dir: Path,
    families: Sequence[str],
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite SfM-maplet coverage audit")
    paths = tuple(Path(path) for path in appearance_artifacts)
    features = load_frozen_fulltrack_per_view_appearance_features(paths)
    if (
        features.compatibility.get("per_view_edge_feature_semantics")
        != FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
    ):
        raise ValueError("coverage audit needs full-CSR SfM-maplet artifacts")
    _validate_complete_frozen_set(features)
    _validate_projected_bank_lineage(
        features.artifact_metadata, projected_landmark_bank=Path(projected_landmark_bank)
    )
    diagnostics = _load_maplet_diagnostics(paths=paths, features=features)
    rows = summarize_sfm_maplet_transport_coverage(
        features=features, diagnostics=diagnostics, families=families
    )
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "rank_bucket_coverage.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": "audit_frozen_fulltrack_per_view_sfm_maplet_transport_coverage",
        "format": ARTIFACT_FORMAT,
        "appearance_artifact_count": int(len(paths)),
        "appearance_artifact_sha256": [file_sha256_short(path) for path in paths],
        "projected_landmark_bank": {
            "path": str(projected_landmark_bank),
            "sha256": file_sha256_short(Path(projected_landmark_bank)),
        },
        "families": list(families),
        "feature_granularity": fulltrack_per_view_feature_granularity(features),
        "row_count": int(len(features.query_ids)),
        "edge_count": int(len(features.edge_geometry_rows)),
        "rank_buckets": [name for name, _minimum, _maximum in RANK_BUCKETS],
        "coverage_rows": rows,
        "protocol": {
            "target_free": True,
            "identity_or_pose_targets_loaded": False,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_support_observations_retained": True,
            "candidate_center_descriptor_excluded": True,
            "availability_or_neighbor_count_is_not_a_learned_feature": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "raw_csr_and_s0_projected_bank_lineage_verified": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_frozen_fulltrack_per_view_sfm_maplet_transport_coverage(
        appearance_artifacts=_paths(args.appearance_artifacts),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        output_dir=Path(args.output_dir),
        families=_families(args.families),
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"appearance_artifact_sha256", "coverage_rows"}
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
