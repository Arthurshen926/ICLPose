"""Audit target-free coverage and lineage of dense translation-mode evidence.

The audit reads neither identity labels, target residuals, poses, nor
hypotheses.  It establishes that every frozen global top-20 candidate rank
bucket retains the intended per-real-observation dense visual evidence before
the train-only identity probe is allowed to consume it.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from feature_extract.vfm.localization.frozen_fulltrack_multiscale_translation_mode import (
    MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES_BY_PROFILE,
    MULTISCALE_TRANSLATION_MODE_PROFILES,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_FAMILIES,
    FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES,
    FrozenFulltrackPerViewAppearanceFeatures,
    fulltrack_per_view_feature_granularity,
    load_frozen_fulltrack_per_view_appearance_features,
    profile_indices_for_fulltrack_per_view_family,
)


ARTIFACT_FORMAT = (
    "frozen_fulltrack_per_view_multiscale_translation_mode_coverage_audit_v1"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifacts", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--families",
        default="all",
        help=(
            "comma-separated translation-mode family names; all selects the "
            "frozen S1 sweep"
        ),
    )
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("translation-mode artifact paths must be non-empty and unique")
    return paths


def _families(value: str) -> tuple[str, ...]:
    if str(value).strip() == "all":
        names = tuple(
            name
            for name, spec in FULLTRACK_PER_VIEW_FAMILIES.items()
            if spec.edge_feature_semantics
            == FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
        )
    else:
        names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if (
        not names
        or len(set(names)) != len(names)
        or set(names).difference(FULLTRACK_PER_VIEW_FAMILIES)
        or any(
            FULLTRACK_PER_VIEW_FAMILIES[name].edge_feature_semantics
            != FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
            for name in names
        )
    ):
        raise ValueError("translation-mode audit families are invalid or incompatible")
    return names


def _validate_complete_frozen_set(features: FrozenFulltrackPerViewAppearanceFeatures) -> None:
    if set(features.split_names.tolist()).difference(EXPECTED_QUERY_COUNTS):
        raise ValueError("translation-mode artifacts have an unsupported split")
    for split, expected_queries in EXPECTED_QUERY_COUNTS.items():
        rows = np.flatnonzero(features.split_names == split)
        query_ids = features.query_ids[rows]
        unique = tuple(dict.fromkeys(query_ids.tolist()))
        if len(unique) != expected_queries:
            raise ValueError("translation-mode artifacts have incomplete query coverage")
        for query_id in unique:
            query_rows = rows[query_ids == query_id]
            if (
                len(query_rows) != EXPECTED_ROWS_PER_QUERY
                or len(np.unique(features.source_row_indices[query_rows]))
                != EXPECTED_ROWS_PER_QUERY
            ):
                raise ValueError("translation-mode artifact has an incomplete query shard")


def _edge_candidate_indices(features: FrozenFulltrackPerViewAppearanceFeatures) -> np.ndarray:
    counts = np.asarray(features.candidate_support_observation_counts, dtype=np.int64)
    return np.repeat(np.arange(counts.size, dtype=np.int64), counts.reshape(-1))


def _expected_profiles() -> list[dict[str, Any]]:
    return [
        {
            "name": profile.name,
            "source": profile.source_name,
            "grid_size": int(profile.grid_size),
            "window_size": int(profile.window_size),
            "maximum_shift": int(profile.maximum_shift),
            "feature_names": list(
                MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES_BY_PROFILE[profile.name]
            ),
        }
        for profile in MULTISCALE_TRANSLATION_MODE_PROFILES
    ]


def _validate_translation_mode_contract(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
) -> None:
    if tuple(features.profile_names) != tuple(
        FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES
    ):
        raise ValueError("translation-mode feature profile order differs from contract")
    expected_profiles = _expected_profiles()
    for path, metadata in zip(features.paths, features.artifact_metadata):
        strict = metadata.get("strict_fulltrack_appearance_contract")
        contract = metadata.get("translation_mode_contract")
        appearance = metadata.get("appearance_config")
        if (
            not isinstance(strict, Mapping)
            or strict.get("source_fulltrack_csr_edges_preserved") is not True
            or strict.get("support_view_features_averaged_before_inference")
            is not False
            or strict.get("explicit_availability_or_neighbor_count_is_not_a_learned_feature")
            is not True
            or not isinstance(contract, Mapping)
            or contract.get("mode")
            != "candidate_specific_multiscale_dense_translation_mode_v1"
            or contract.get("support_coordinate_source") != "sfm_observation_xy"
            or contract.get("view_aggregation")
            != "none_before_learned_logsumexp_mixture_v1"
            or contract.get("explicit_availability_or_neighbor_count_feature")
            is not False
            or contract.get("all_profiles_require_only_center_sample_validity")
            is not True
            or contract.get("profiles") != expected_profiles
            or not isinstance(appearance, Mapping)
            or appearance.get("candidate_specific") is not True
            or appearance.get("per_view") is not True
            or appearance.get("support_view_marginalization")
            != "not_aggregated_export_per_view_v1"
            or appearance.get("whole_image_retrieval_or_candidate_reselection")
            is not False
        ):
            raise ValueError(f"{path}: translation-mode contract is invalid")


def summarize_multiscale_translation_mode_coverage(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    families: Sequence[str],
) -> list[dict[str, Any]]:
    """Summarize frozen-rank dense visual coverage without correctness labels."""

    family_names = _families(",".join(str(name) for name in families))
    ranks = frozen_candidate_ranks(features)
    tracks = np.asarray(features.candidate_track_ids, dtype=np.int64)
    candidate = np.asarray(features.candidate_probabilities, dtype=np.float32)
    counts = np.asarray(features.candidate_support_observation_counts, dtype=np.int64)
    edge_candidates = _edge_candidate_indices(features)
    expected_edge_count = len(np.asarray(features.edge_geometry_rows, dtype=np.int64))
    if len(edge_candidates) != expected_edge_count:
        raise ValueError("translation-mode CSR edge count is inconsistent")
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
                        "profile_feature_count": int(len(profile_indices)),
                        "split": split,
                        "rank_bucket": bucket_name,
                        "candidate_count": int(len(flat)),
                        "candidate_prior_mass": candidate_mass,
                        "candidate_with_usable_dense_mode_edge_count": int(
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
                        "mean_usable_dense_mode_view_count": (
                            float(np.mean(usable_views)) if len(flat) else None
                        ),
                        "edge_count": edge_count,
                        "usable_dense_mode_edge_count": usable_edge_count,
                        "joint_edge_coverage_rate": (
                            usable_edge_count / edge_count if edge_count else None
                        ),
                        "missing_edge_rate": (
                            1.0 - usable_edge_count / edge_count if edge_count else None
                        ),
                    }
                )
    return summary


def audit_frozen_fulltrack_per_view_multiscale_translation_mode_coverage(
    *,
    appearance_artifacts: Sequence[Path],
    projected_landmark_bank: Path,
    output_dir: Path,
    families: Sequence[str],
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite translation-mode coverage audit")
    paths = tuple(Path(path) for path in appearance_artifacts)
    features = load_frozen_fulltrack_per_view_appearance_features(paths)
    if (
        features.compatibility.get("per_view_edge_feature_semantics")
        != FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
    ):
        raise ValueError("coverage audit needs full-CSR translation-mode artifacts")
    _validate_complete_frozen_set(features)
    _validate_projected_bank_lineage(
        features.artifact_metadata, projected_landmark_bank=Path(projected_landmark_bank)
    )
    _validate_translation_mode_contract(features=features)
    rows = summarize_multiscale_translation_mode_coverage(
        features=features, families=families
    )
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "rank_bucket_coverage.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": "audit_frozen_fulltrack_per_view_multiscale_translation_mode_coverage",
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
            "support_view_features_averaged_before_inference": False,
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
    result = audit_frozen_fulltrack_per_view_multiscale_translation_mode_coverage(
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
