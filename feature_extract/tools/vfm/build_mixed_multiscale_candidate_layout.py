"""Freeze mixed verification-point candidates into a maplet-view layout.

The mixed verification point artifact has already fixed full-bank FAISS
candidates before this builder runs.  This command only attaches the existing
coverage-ranked SfM support views for each candidate track, so later visual
feature export cannot retrieve, re-rank, or use any target pose/residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.tools.vfm.build_global_context_support8_candidate_probe_features import (
    _fixed_support8_layout,
    _load_maplet_support_index,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    load_mixed_verification_points,
)


ARTIFACT_FORMAT = "mixed_multiscale_verification_frozen_candidate_layout_v1"
_FEATURE_NAMES = ("radio_final_anchor_cosine",)
_ALLOWED_SPLITS = frozenset({"train", "validation"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification_points", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def build_mixed_multiscale_candidate_layout(
    *,
    verification_points: Path,
    maplet_support_index: Path,
    output: Path,
    summary_json: Path,
) -> dict[str, Any]:
    output = Path(output)
    summary_path = Path(summary_json)
    if output.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite mixed frozen-layout outputs")
    points = load_mixed_verification_points(Path(verification_points))
    if str(points.metadata.get("format", "")) != MIXED_VERIFICATION_POINTS_FORMAT:
        raise ValueError("mixed verification points have an unsupported format")
    split_names = np.asarray(points.split_names).astype(str)
    if not len(split_names) or set(split_names.tolist()) - _ALLOWED_SPLITS:
        raise ValueError("mixed frozen layout may contain train/validation points only")
    if bool(points.metadata.get("pose_or_ground_truth_used", True)) or bool(
        points.metadata.get("image_retrieval_or_submap_used", True)
    ) or bool(points.metadata.get("render", True)):
        raise ValueError("mixed verification points violate the no-target/no-retrieval protocol")

    maplet, maplet_metadata = _load_maplet_support_index(Path(maplet_support_index))
    candidate_tracks = np.asarray(points.candidate_track_ids, dtype=np.int64)
    canonical_rows = np.asarray(points.candidate_bank_rows, dtype=np.int64)
    candidate_valid = candidate_tracks >= 0
    if np.any(candidate_valid & (canonical_rows < 0)) or np.any(
        ~candidate_valid & (canonical_rows >= 0)
    ):
        raise ValueError("mixed verification candidate bank rows are invalid")
    support_ids, view_valid, coverage = _fixed_support8_layout(
        layout={
            "candidate_track_ids": candidate_tracks,
            "candidate_canonical_rows": canonical_rows,
        },
        maplet=maplet,
    )
    anchor = np.asarray(points.candidate_coarse_similarities, dtype=np.float32)
    if anchor.shape != candidate_tracks.shape or np.any(~np.isfinite(anchor[candidate_valid])):
        raise ValueError("mixed verification coarse candidate similarities are invalid")
    features = np.full((*view_valid.shape, len(_FEATURE_NAMES)), np.nan, dtype=np.float32)
    repeated_anchor = np.broadcast_to(anchor[..., None], view_valid.shape)
    features[..., 0][view_valid] = repeated_anchor[view_valid]
    if np.any(~np.isfinite(features[view_valid])):
        raise RuntimeError("mixed frozen layout emitted invalid support features")

    points_sha = file_sha256_short(Path(verification_points))
    metadata: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "render": False,
        "candidate_set": "mixed_verification_fixed_full_global_faiss_top_l_tracks",
        "candidate_top_k": int(candidate_tracks.shape[1]),
        "candidate_reselection": False,
        "feature_definition": "mixed_verification_fixed_coarse_similarity_repeated_per_maplet_view_v1",
        "support_view_selection": "fixed_maplet_coverage_rank_all_available_v1",
        "support_view_marginalization": "per_view_features_preserved_for_later_log_mixture_v1",
        # Downstream visual exporters read only this immutable point/candidate/
        # support-view contract.  Mark it complete so they cannot accidentally
        # accept a diagnostic partial layout with the same array names.
        "is_complete_frozen_layout": True,
        "verification_points": str(Path(verification_points)),
        "verification_points_sha256": points_sha,
        "verification_points_format": points.metadata.get("format"),
        "verification_point_protocol": {
            "hypothesis_detector_rows_excluded_from_alike": points.metadata.get(
                "hypothesis_detector_rows_excluded_from_alike"
            ),
            "fixed_full_global_faiss_top_l": points.metadata.get("fixed_full_global_faiss_top_l"),
            "candidate_prior_pose_calibrated": points.metadata.get(
                "candidate_prior_pose_calibrated"
            ),
        },
        # The existing multi-source exporter needs a stable provenance field.
        # It is deliberately the verification-point artifact rather than a
        # proposal artifact from a fit-set detector pipeline.
        "proposals_sha256": points_sha,
        "proposal_provenance": "mixed_verification_points_full_global_faiss_top_l_v1",
        "descriptor_space_id": points.metadata.get("descriptor_space_id"),
        "mapper_checkpoint_sha256": points.metadata.get("matcha_joint_checkpoint_sha256"),
        "projected_landmark_bank_sha256": points.metadata.get("projected_landmark_bank_sha256"),
        "maplet_support_index": str(Path(maplet_support_index)),
        "maplet_support_index_sha256": file_sha256_short(Path(maplet_support_index)),
        "maplet_support_index_format": maplet_metadata.get("format"),
        "exported_splits": sorted(set(split_names.tolist())),
        "source_test_rows_materialized": False,
        "source_target_arrays_read": False,
        "source_target_arrays_excluded": ["pose", "residual", "registered_track_identity"],
        "row_counts": {
            split: int(np.count_nonzero(split_names == split))
            for split in sorted(set(split_names.tolist()))
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_row_indices=np.asarray(points.source_point_ids, dtype=np.int64),
            query_ids=np.asarray(points.query_ids).astype(str),
            split_names=split_names,
            xy=np.asarray(points.xy, dtype=np.float32),
            point_sources=np.asarray(points.point_sources).astype(str),
            source_detector_rows=np.asarray(points.source_detector_rows, dtype=np.int64),
            candidate_track_ids=candidate_tracks,
            candidate_canonical_rows=canonical_rows,
            candidate_features=features,
            candidate_view_valid=view_valid,
            candidate_support_image_ids=support_ids,
            candidate_support_coverage_counts=coverage,
            feature_names=np.asarray(_FEATURE_NAMES, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output)
    summary = {
        "stage": "build_mixed_multiscale_target_free_fixed_candidate_layout",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "query_point_count": int(len(points.source_point_ids)),
        "valid_candidate_view_count": int(np.sum(view_valid)),
        "support_image_count": int(len(set(support_ids[view_valid].tolist()))),
        "point_count_by_source": {
            source: int(np.count_nonzero(points.point_sources == source))
            for source in sorted(set(np.asarray(points.point_sources).astype(str).tolist()))
        },
        "protocol": {
            "train_validation_only": True,
            "test_rows_materialized": False,
            "target_residuals_read": False,
            "candidate_reselection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            build_mixed_multiscale_candidate_layout(
                verification_points=Path(args.verification_points),
                maplet_support_index=Path(args.maplet_support_index),
                output=Path(args.output),
                summary_json=Path(args.summary_json),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
