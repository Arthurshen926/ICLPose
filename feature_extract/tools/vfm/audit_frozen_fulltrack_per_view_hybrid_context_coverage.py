"""Audit target-free rank-bucket coverage for virtual hybrid context evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_region_context_coverage import (
    EXPECTED_QUERY_COUNTS,
    EXPECTED_ROWS_PER_QUERY,
    RANK_BUCKETS,
    frozen_candidate_ranks,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_hybrid_context import (
    FULLTRACK_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS,
    HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES,
    HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_FAMILIES,
    FrozenFulltrackPerViewAppearanceFeatures,
    load_frozen_fulltrack_per_view_appearance_features,
    profile_indices_for_fulltrack_per_view_family,
)


ARTIFACT_FORMAT = "frozen_fulltrack_candidate_per_view_hybrid_context_coverage_audit_v1"
HYBRID_FAMILY = "fixedprior_fulltrack_perview_hybrid_translation_absolute_intermediate_mixture"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _validate_complete_set(features: FrozenFulltrackPerViewAppearanceFeatures) -> None:
    if set(features.split_names.tolist()).difference(EXPECTED_QUERY_COUNTS):
        raise ValueError("hybrid context has an unsupported split")
    for split, expected_queries in EXPECTED_QUERY_COUNTS.items():
        rows = np.flatnonzero(features.split_names == split)
        query_ids = features.query_ids[rows]
        unique = tuple(dict.fromkeys(query_ids.tolist()))
        if len(unique) != expected_queries:
            raise ValueError("hybrid context has incomplete query coverage")
        for query_id in unique:
            query_rows = rows[query_ids == query_id]
            if (
                len(query_rows) != EXPECTED_ROWS_PER_QUERY
                or len(np.unique(features.source_row_indices[query_rows]))
                != EXPECTED_ROWS_PER_QUERY
            ):
                raise ValueError("hybrid context query shard is incomplete")


def _edge_candidate_indices(features: FrozenFulltrackPerViewAppearanceFeatures) -> np.ndarray:
    return np.repeat(
        np.arange(features.candidate_track_ids.size, dtype=np.int64),
        features.candidate_support_observation_counts.reshape(-1),
    )


def _coverage_for_profile_indices(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    profile_indices: np.ndarray,
    edge_candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edge_valid = np.all(features.edge_profile_valid[:, profile_indices], axis=1)
    candidate_usable_views = np.zeros(
        (features.candidate_track_ids.size,), dtype=np.int64
    )
    np.add.at(candidate_usable_views, edge_candidates, edge_valid.astype(np.int64))
    return edge_valid, candidate_usable_views


def summarize_hybrid_context_coverage(
    *, features: FrozenFulltrackPerViewAppearanceFeatures
) -> list[dict[str, Any]]:
    spec = FULLTRACK_PER_VIEW_FAMILIES[HYBRID_FAMILY]
    if spec.edge_feature_semantics != FULLTRACK_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS:
        raise RuntimeError("hybrid family semantic contract drifted")
    ranks = frozen_candidate_ranks(features)
    candidate = np.asarray(features.candidate_probabilities, dtype=np.float32)
    counts = np.asarray(features.candidate_support_observation_counts, dtype=np.int64)
    edge_candidates = _edge_candidate_indices(features)
    candidate_count = features.candidate_count
    edge_rows = edge_candidates // candidate_count
    edge_columns = edge_candidates % candidate_count
    edge_ranks = ranks[edge_rows, edge_columns]
    all_indices = profile_indices_for_fulltrack_per_view_family(
        HYBRID_FAMILY, profile_names=features.profile_names
    )
    name_to_index = {name: index for index, name in enumerate(features.profile_names)}
    translation_indices = np.asarray(
        [name_to_index[name] for name in HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES],
        dtype=np.int64,
    )
    absolute_indices = np.asarray(
        [
            name_to_index[name]
            for name in HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES
        ],
        dtype=np.int64,
    )
    all_valid, all_views = _coverage_for_profile_indices(
        features=features, profile_indices=all_indices, edge_candidates=edge_candidates
    )
    translation_valid, translation_views = _coverage_for_profile_indices(
        features=features,
        profile_indices=translation_indices,
        edge_candidates=edge_candidates,
    )
    absolute_valid, absolute_views = _coverage_for_profile_indices(
        features=features,
        profile_indices=absolute_indices,
        edge_candidates=edge_candidates,
    )
    result: list[dict[str, Any]] = []
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
            joint_candidate = all_views[flat] > 0
            translation_candidate = translation_views[flat] > 0
            absolute_candidate = absolute_views[flat] > 0
            candidate_mass = float(np.sum(masses, dtype=np.float64))
            edge_mask = row_mask[edge_rows] & (edge_ranks >= int(minimum_rank)) & (
                edge_ranks <= int(maximum_rank)
            )
            result.append(
                {
                    "family": HYBRID_FAMILY,
                    "split": split,
                    "rank_bucket": bucket_name,
                    "candidate_count": int(len(flat)),
                    "candidate_prior_mass": candidate_mass,
                    "hybrid_candidate_coverage_rate": (
                        float(np.mean(joint_candidate)) if len(flat) else None
                    ),
                    "translation_candidate_coverage_rate": (
                        float(np.mean(translation_candidate)) if len(flat) else None
                    ),
                    "absolute_candidate_coverage_rate": (
                        float(np.mean(absolute_candidate)) if len(flat) else None
                    ),
                    "hybrid_candidate_prior_mass_coverage_rate": (
                        float(np.sum(masses[joint_candidate], dtype=np.float64))
                        / candidate_mass
                        if candidate_mass > 0.0
                        else None
                    ),
                    "translation_candidate_prior_mass_coverage_rate": (
                        float(np.sum(masses[translation_candidate], dtype=np.float64))
                        / candidate_mass
                        if candidate_mass > 0.0
                        else None
                    ),
                    "absolute_candidate_prior_mass_coverage_rate": (
                        float(np.sum(masses[absolute_candidate], dtype=np.float64))
                        / candidate_mass
                        if candidate_mass > 0.0
                        else None
                    ),
                    "edge_count": int(np.sum(edge_mask)),
                    "hybrid_joint_edge_coverage_rate": (
                        float(np.mean(all_valid[edge_mask])) if np.any(edge_mask) else None
                    ),
                    "translation_joint_edge_coverage_rate": (
                        float(np.mean(translation_valid[edge_mask]))
                        if np.any(edge_mask)
                        else None
                    ),
                    "absolute_joint_edge_coverage_rate": (
                        float(np.mean(absolute_valid[edge_mask]))
                        if np.any(edge_mask)
                        else None
                    ),
                    "mean_real_support_view_count": (
                        float(np.mean(counts.reshape(-1)[flat])) if len(flat) else None
                    ),
                    "mean_hybrid_usable_view_count": (
                        float(np.mean(all_views[flat])) if len(flat) else None
                    ),
                }
            )
    return result


def audit_hybrid_context_coverage(*, hybrid_manifest: Path, output_dir: Path) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite hybrid context coverage audit")
    manifest = Path(hybrid_manifest)
    features = load_frozen_fulltrack_per_view_appearance_features((manifest,))
    if (
        features.compatibility.get("per_view_edge_feature_semantics")
        != FULLTRACK_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS
    ):
        raise ValueError("hybrid coverage audit needs a hybrid manifest")
    _validate_complete_set(features)
    rows = summarize_hybrid_context_coverage(features=features)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "rank_bucket_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "stage": "audit_frozen_fulltrack_per_view_hybrid_context_coverage",
        "format": ARTIFACT_FORMAT,
        "hybrid_manifest": {"path": str(manifest), "sha256": file_sha256_short(manifest)},
        "row_count": int(len(features.query_ids)),
        "edge_count": int(len(features.edge_geometry_rows)),
        "profile_count": int(len(features.profile_names)),
        "coverage_rows": rows,
        "protocol": {
            "target_free": True,
            "identity_or_pose_targets_loaded": False,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_support_observations_retained": True,
            "translation_and_absolute_evidence_must_both_be_present": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_hybrid_context_coverage(
        hybrid_manifest=Path(args.hybrid_manifest), output_dir=Path(args.output_dir)
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "coverage_rows"
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
