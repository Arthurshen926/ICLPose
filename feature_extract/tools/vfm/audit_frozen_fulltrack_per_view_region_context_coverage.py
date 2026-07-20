"""Audit target-free coverage of full-CSR multiscale region-context evidence.

The audit does not load identities, targets, poses, or hypotheses.  It checks
whether the fixed global top-20 candidates at ranks 1--5, 6--10, and 11--20
actually retain usable per-observation context evidence before a train-only
identity probe is allowed to consume the artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_hard_pairs import (
    _validate_projected_bank_lineage,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_FAMILIES,
    FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS,
    FrozenFulltrackPerViewAppearanceFeatures,
    fulltrack_per_view_feature_granularity,
    load_frozen_fulltrack_per_view_appearance_features,
    profile_indices_for_fulltrack_per_view_family,
)


ARTIFACT_FORMAT = "frozen_fulltrack_per_view_region_context_coverage_audit_v1"
RANK_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("rank_1_5", 1, 5),
    ("rank_6_10", 6, 10),
    ("rank_11_20", 11, 20),
)
EXPECTED_QUERY_COUNTS = {"train": 63, "validation": 21}
EXPECTED_ROWS_PER_QUERY = 192


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifacts", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--families",
        default="all",
        help="comma-separated region family names; all selects the frozen S1 sweep",
    )
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("region-context artifact paths must be non-empty and unique")
    return paths


def _families(value: str) -> tuple[str, ...]:
    if str(value).strip() == "all":
        names = tuple(
            name
            for name, spec in FULLTRACK_PER_VIEW_FAMILIES.items()
            if spec.edge_feature_semantics
            == FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
        )
    else:
        names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if (
        not names
        or len(set(names)) != len(names)
        or set(names).difference(FULLTRACK_PER_VIEW_FAMILIES)
        or any(
            FULLTRACK_PER_VIEW_FAMILIES[name].edge_feature_semantics
            != FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
            for name in names
        )
    ):
        raise ValueError("region-context audit families are invalid or incompatible")
    return names


def _validate_complete_frozen_set(features: FrozenFulltrackPerViewAppearanceFeatures) -> None:
    if set(features.split_names.tolist()).difference(EXPECTED_QUERY_COUNTS):
        raise ValueError("region-context artifacts have an unsupported split")
    for split, expected_queries in EXPECTED_QUERY_COUNTS.items():
        rows = np.flatnonzero(features.split_names == split)
        query_ids = features.query_ids[rows]
        unique = tuple(dict.fromkeys(query_ids.tolist()))
        if len(unique) != expected_queries:
            raise ValueError("region-context artifacts have incomplete query coverage")
        for query_id in unique:
            query_rows = rows[query_ids == query_id]
            if (
                len(query_rows) != EXPECTED_ROWS_PER_QUERY
                or len(np.unique(features.source_row_indices[query_rows]))
                != EXPECTED_ROWS_PER_QUERY
            ):
                raise ValueError("region-context artifact has an incomplete query shard")


def frozen_candidate_ranks(features: FrozenFulltrackPerViewAppearanceFeatures) -> np.ndarray:
    """Return immutable posterior ranks without assuming stored column order."""

    tracks = np.asarray(features.candidate_track_ids, dtype=np.int64)
    candidate = np.asarray(features.candidate_probabilities, dtype=np.float32)
    ranks = np.full(tracks.shape, -1, dtype=np.int64)
    for row in range(len(tracks)):
        valid = (tracks[row] >= 0) & (candidate[row] > 0.0)
        columns = np.flatnonzero(valid)
        order = columns[np.argsort(-candidate[row, columns], kind="stable")]
        ranks[row, order] = np.arange(1, len(order) + 1, dtype=np.int64)
    return ranks


def _edge_candidate_indices(features: FrozenFulltrackPerViewAppearanceFeatures) -> np.ndarray:
    counts = np.asarray(features.candidate_support_observation_counts, dtype=np.int64)
    return np.repeat(
        np.arange(counts.size, dtype=np.int64), counts.reshape(-1)
    )


def summarize_region_context_coverage(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    families: Sequence[str],
) -> list[dict[str, Any]]:
    """Summarize candidate and edge coverage by frozen posterior rank bucket."""

    family_names = _families(",".join(str(name) for name in families))
    ranks = frozen_candidate_ranks(features)
    tracks = np.asarray(features.candidate_track_ids, dtype=np.int64)
    candidate = np.asarray(features.candidate_probabilities, dtype=np.float32)
    counts = np.asarray(features.candidate_support_observation_counts, dtype=np.int64)
    edge_candidates = _edge_candidate_indices(features)
    expected_edge_count = len(np.asarray(features.edge_geometry_rows, dtype=np.int64))
    if len(edge_candidates) != expected_edge_count:
        raise ValueError("region-context CSR edge count is inconsistent")
    edge_rows = edge_candidates // features.candidate_count
    edge_columns = edge_candidates % features.candidate_count
    edge_ranks = ranks[edge_rows, edge_columns]
    summary: list[dict[str, Any]] = []
    for family in family_names:
        profile_indices = profile_indices_for_fulltrack_per_view_family(
            family, profile_names=features.profile_names
        )
        edge_valid = np.all(features.edge_profile_valid[:, profile_indices], axis=1)
        candidate_usable_views = np.zeros((tracks.size,), dtype=np.int64)
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
                if len(flat):
                    masses = candidate.reshape(-1)[flat]
                    support_counts = counts.reshape(-1)[flat]
                    usable_views = candidate_usable_views[flat]
                    usable_candidate = usable_views > 0
                    candidate_mass = float(np.sum(masses, dtype=np.float64))
                    covered_mass = float(
                        np.sum(masses[usable_candidate], dtype=np.float64)
                    )
                else:
                    support_counts = np.zeros((0,), dtype=np.int64)
                    usable_views = np.zeros((0,), dtype=np.int64)
                    usable_candidate = np.zeros((0,), dtype=bool)
                    candidate_mass = 0.0
                    covered_mass = 0.0
                edge_mask = row_mask[edge_rows] & (edge_ranks >= int(minimum_rank)) & (
                    edge_ranks <= int(maximum_rank)
                )
                edge_count = int(np.sum(edge_mask))
                usable_edge_count = int(np.sum(edge_valid[edge_mask]))
                summary.append(
                    {
                        "family": family,
                        "profile_count": int(len(profile_indices)),
                        "split": split,
                        "rank_bucket": bucket_name,
                        "candidate_count": int(len(flat)),
                        "candidate_prior_mass": candidate_mass,
                        "candidate_with_usable_edge_count": int(
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
                        "mean_usable_context_view_count": (
                            float(np.mean(usable_views)) if len(flat) else None
                        ),
                        "edge_count": edge_count,
                        "usable_edge_count": usable_edge_count,
                        "joint_edge_coverage_rate": (
                            usable_edge_count / edge_count if edge_count else None
                        ),
                        "missing_edge_rate": (
                            1.0 - usable_edge_count / edge_count if edge_count else None
                        ),
                    }
                )
    return summary


def audit_frozen_fulltrack_per_view_region_context_coverage(
    *,
    appearance_artifacts: Sequence[Path],
    projected_landmark_bank: Path,
    output_dir: Path,
    families: Sequence[str],
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite region-context coverage audit")
    paths = tuple(Path(path) for path in appearance_artifacts)
    features = load_frozen_fulltrack_per_view_appearance_features(paths)
    if (
        features.compatibility.get("per_view_edge_feature_semantics")
        != FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
    ):
        raise ValueError("coverage audit needs full-CSR multisource region artifacts")
    _validate_complete_frozen_set(features)
    _validate_projected_bank_lineage(
        features.artifact_metadata, projected_landmark_bank=Path(projected_landmark_bank)
    )
    rows = summarize_region_context_coverage(features=features, families=families)
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "rank_bucket_coverage.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": "audit_frozen_fulltrack_per_view_region_context_coverage",
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
            "image_retrieval_or_submap_used": False,
            "render": False,
            "raw_csr_and_s0_projected_bank_lineage_verified": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_frozen_fulltrack_per_view_region_context_coverage(
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
