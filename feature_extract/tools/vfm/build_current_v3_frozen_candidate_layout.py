"""Freeze current V3 top-L candidates into a target-free maplet-view layout.

This builder deliberately consumes only inference-time fields from the current
candidate-evidence artifact.  It emits train and validation rows only; target
residuals and every test row are excluded before any visual context feature is
computed.  Later exporters may attach real-image evidence to these fixed
candidate/support-view pairs, but may not retrieve or reselect landmarks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.build_global_context_support8_candidate_probe_features import (
    _load_maplet_support_index,
)
from feature_extract.vfm.artifacts import file_sha256_short


ARTIFACT_FORMAT = "current_v3_frozen_candidate_layout_v1"
_FEATURE_NAMES = ("radio_final_anchor_cosine",)
_REQUIRED_EVIDENCE_ARRAYS = {
    "selected_rows",
    "query_ids",
    "query_xy",
    "split_names",
    "candidate_valid",
    "candidate_track_ids",
    "candidate_bank_rows",
    "candidate_coarse_similarities",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument(
        "--splits",
        default="train,validation",
        help="comma-separated non-test splits exported into the frozen layout",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} lacks metadata_json")
    value = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("splits must be unique non-empty names")
    unsupported = set(splits) - {"train", "validation"}
    if unsupported:
        raise ValueError(
            "current V3 frozen layout may only export train/validation rows; "
            f"got {sorted(unsupported)}"
        )
    return splits


def _load_current_v3_evidence(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as data:
        missing = _REQUIRED_EVIDENCE_ARRAYS - set(data.files)
        if missing:
            raise ValueError(f"candidate evidence lacks {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in _REQUIRED_EVIDENCE_ARRAYS}
        metadata = _metadata(data, context="candidate evidence")
    if metadata.get("format") != "candidate_evidence_v3":
        raise ValueError("current frozen layout requires candidate_evidence_v3")
    if metadata.get("candidate_probability_semantics") != (
        "factorized_top_l_availability_times_conditional_identity"
    ):
        raise ValueError("candidate evidence has incompatible identity semantics")
    if bool(metadata.get("pose_used_for_selection", True)) or bool(
        metadata.get("image_retrieval", True)
    ) or bool(metadata.get("render", True)):
        raise ValueError("candidate evidence violates the fixed global no-render protocol")
    return arrays, metadata


def _fixed_maplet_views(
    *,
    candidate_valid: np.ndarray,
    candidate_tracks: np.ndarray,
    candidate_canonical_rows: np.ndarray,
    maplet: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(candidate_valid, dtype=bool)
    tracks = np.asarray(candidate_tracks, dtype=np.int64)
    canonical = np.asarray(candidate_canonical_rows, dtype=np.int64)
    if valid.shape != tracks.shape or canonical.shape != tracks.shape:
        raise ValueError("candidate evidence identity arrays are incompatible")
    if np.any(valid & ((tracks < 0) | (canonical < 0))):
        raise ValueError("a valid V3 candidate has no track or canonical bank row")
    if np.any(~valid & ((tracks >= 0) | (canonical >= 0))):
        raise ValueError("an invalid V3 candidate retains a track or canonical bank row")
    maplet_tracks = np.asarray(maplet["anchor_track_ids"], dtype=np.int64)
    if np.any(canonical[valid] >= len(maplet_tracks)):
        raise ValueError("a V3 candidate canonical row is absent from the maplet index")
    safe_canonical = np.maximum(canonical, 0)
    if np.any(maplet_tracks[safe_canonical][valid] != tracks[valid]):
        raise ValueError("V3 candidate tracks differ from the maplet support index")
    source_indices = np.asarray(maplet["support_image_indices"], dtype=np.int64)[
        safe_canonical
    ]
    coverage = np.asarray(maplet["support_coverage_counts"], dtype=np.int32)[
        safe_canonical
    ]
    view_valid = valid[..., None] & (source_indices >= 0)
    image_ids = np.asarray(maplet["support_image_ids"]).astype(str)
    support_ids = np.full(source_indices.shape, "", dtype=image_ids.dtype)
    support_ids[view_valid] = image_ids[source_indices[view_valid]]
    coverage = np.where(view_valid, coverage, 0).astype(np.int32, copy=False)
    if np.any(valid & ~np.any(view_valid, axis=2)):
        raise ValueError("a valid V3 candidate has no fixed maplet support view")
    return support_ids, view_valid, coverage


def build_current_v3_frozen_candidate_layout(
    *,
    candidate_evidence: Path,
    maplet_support_index: Path,
    output: Path,
    summary_json: Path,
    splits: Sequence[str] = ("train", "validation"),
    force: bool = False,
) -> dict[str, Any]:
    selected_splits = _parse_splits(",".join(str(value) for value in splits))
    output = Path(output)
    summary_json = Path(summary_json)
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite current V3 frozen layout outputs")
    evidence, evidence_metadata = _load_current_v3_evidence(Path(candidate_evidence))
    maplet, maplet_metadata = _load_maplet_support_index(Path(maplet_support_index))
    expected_maplet_sha = str(evidence_metadata.get("maplet_support_index_sha256", ""))
    actual_maplet_sha = file_sha256_short(Path(maplet_support_index))
    if not expected_maplet_sha or expected_maplet_sha != actual_maplet_sha:
        raise ValueError("candidate evidence and maplet support index provenance differs")

    source_rows = np.asarray(evidence["selected_rows"], dtype=np.int64).reshape(-1)
    query_ids = np.asarray(evidence["query_ids"]).astype(str).reshape(-1)
    query_xy = np.asarray(evidence["query_xy"], dtype=np.float32)
    split_names = np.asarray(evidence["split_names"]).astype(str).reshape(-1)
    candidate_valid = np.asarray(evidence["candidate_valid"], dtype=bool)
    candidate_tracks = np.asarray(evidence["candidate_track_ids"], dtype=np.int64)
    candidate_rows = np.asarray(evidence["candidate_bank_rows"], dtype=np.int64)
    candidate_anchor = np.asarray(
        evidence["candidate_coarse_similarities"], dtype=np.float32
    )
    if (
        len(source_rows) == 0
        or len(set(source_rows.tolist())) != len(source_rows)
        or not (query_ids.shape == split_names.shape == (len(source_rows),))
        or query_xy.shape != (len(source_rows), 2)
        or candidate_valid.shape != candidate_tracks.shape
        or candidate_rows.shape != candidate_tracks.shape
        or candidate_anchor.shape != candidate_tracks.shape
    ):
        raise ValueError("candidate evidence arrays are not aligned")
    if set(split_names.tolist()) - {"train", "validation", "test"}:
        raise ValueError("candidate evidence has an unknown split")
    selected = np.isin(split_names, selected_splits)
    if not np.any(selected):
        raise ValueError("candidate evidence has no requested frozen-layout rows")
    if np.any(split_names[selected] == "test"):
        raise RuntimeError("test rows entered the current V3 frozen layout")

    source_rows = source_rows[selected]
    query_ids = query_ids[selected]
    query_xy = query_xy[selected]
    split_names = split_names[selected]
    candidate_valid = candidate_valid[selected]
    candidate_tracks = candidate_tracks[selected]
    candidate_rows = candidate_rows[selected]
    candidate_anchor = candidate_anchor[selected]
    support_ids, view_valid, coverage = _fixed_maplet_views(
        candidate_valid=candidate_valid,
        candidate_tracks=candidate_tracks,
        candidate_canonical_rows=candidate_rows,
        maplet=maplet,
    )
    if np.any(~np.isfinite(candidate_anchor[candidate_valid])):
        raise ValueError("a valid V3 candidate has non-finite RADIO-final coarse similarity")
    features = np.full(
        (*candidate_tracks.shape, view_valid.shape[2], len(_FEATURE_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    repeated_anchor = np.broadcast_to(candidate_anchor[..., None], view_valid.shape)
    features[..., 0][view_valid] = repeated_anchor[view_valid]
    if np.any(~np.isfinite(features[view_valid])):
        raise RuntimeError("current V3 frozen layout emitted invalid coarse anchors")
    canonical = np.where(candidate_valid, candidate_rows, -1).astype(np.int64)
    tracks = np.where(candidate_valid, candidate_tracks, -1).astype(np.int64)
    if np.any(~candidate_valid & np.any(view_valid, axis=2)):
        raise RuntimeError("invalid V3 candidates received maplet support views")

    metadata: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "render": False,
        "candidate_set": "current_v3_fixed_global_top_l_tracks",
        "candidate_top_k": int(tracks.shape[1]),
        "feature_definition": (
            "current_v3_radio_final_coarse_similarity_repeated_per_fixed_maplet_view_v1"
        ),
        "support_view_selection": "fixed_maplet_coverage_rank_all_available_v1",
        "support_view_marginalization": "per_view_features_preserved_for_later_log_mixture_v1",
        "candidate_evidence": str(Path(candidate_evidence)),
        "candidate_evidence_sha256": file_sha256_short(Path(candidate_evidence)),
        "candidate_evidence_format": evidence_metadata.get("format"),
        "proposals_sha256": evidence_metadata.get("proposals_sha256"),
        "descriptor_space_id": evidence_metadata.get("descriptor_space_id"),
        # This is the trained full-map descriptor mapper that produced the
        # frozen top-L candidates.  It is deliberately distinct from the
        # frozen C-RADIO backbone used by independent appearance caches.
        "mapper_checkpoint_sha256": evidence_metadata.get("descriptor_space_manifest", {}).get(
            "checkpoint_sha256"
        ),
        "maplet_support_index": str(Path(maplet_support_index)),
        "maplet_support_index_sha256": actual_maplet_sha,
        "maplet_support_index_format": maplet_metadata.get("format"),
        "exported_splits": list(selected_splits),
        "source_test_rows_materialized": False,
        "source_target_arrays_read": False,
        "source_target_arrays_excluded": ["candidate_target_gt_residuals_px"],
        "row_counts": {
            split: int(np.count_nonzero(split_names == split))
            for split in selected_splits
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_row_indices=source_rows,
            query_ids=query_ids,
            split_names=split_names,
            xy=query_xy,
            candidate_track_ids=tracks,
            candidate_canonical_rows=canonical,
            candidate_features=features,
            candidate_view_valid=view_valid,
            candidate_support_image_ids=support_ids,
            candidate_support_coverage_counts=coverage,
            feature_names=np.asarray(_FEATURE_NAMES, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output)
    summary = {
        "stage": "build_current_v3_target_free_fixed_candidate_layout",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "metadata": metadata,
        "query_row_count": int(len(source_rows)),
        "valid_candidate_view_count": int(np.sum(view_valid)),
        "support_image_count": int(len(set(support_ids[view_valid].tolist()))),
        "protocol": {
            "train_validation_only": True,
            "test_rows_materialized": False,
            "target_residuals_read": False,
            "candidate_reselection": False,
            "image_retrieval": False,
            "render": False,
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_current_v3_frozen_candidate_layout(
        candidate_evidence=Path(args.candidate_evidence),
        maplet_support_index=Path(args.maplet_support_index),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        splits=_parse_splits(args.splits),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
