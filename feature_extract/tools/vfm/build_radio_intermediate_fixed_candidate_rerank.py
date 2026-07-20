"""Attach raw RADIO-intermediate evidence to an immutable final top-L pool.

The candidate tracks remain the existing final-space global top-20 tracks.
This command only samples an independently constructed RADIO-intermediate PCA
descriptor at each frozen detector point and scores those same physical tracks
in the matching projected-observation intermediate bank.  It never performs a
new nearest-neighbour search, reads targets, or scores a pose.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.build_radio_intermediate_image_context_pca_cache import (
    PCA_FORMAT,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.context_observation_landmark_bank import (
    canonical_track_rows,
    sample_spatial_context_descriptors,
    spatial_context_boundary_audit,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.spatial_image_context import (
    load_spatial_image_context_cache,
)


ARTIFACT_FORMAT = "radio_intermediate_fixed_final_topl_rerank_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-cache", required=True)
    parser.add_argument("--context-landmark-bank", required=True)
    parser.add_argument("--detector-query-cache", required=True)
    parser.add_argument("--selected-point-artifact", required=True)
    parser.add_argument("--candidate-prior-overlay", required=True)
    parser.add_argument("--query-split", required=True)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args(argv)


def _metadata(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{context} lacks metadata_json")
    value = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata must be an object")
    return value


def _load_detector_query_cache(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    required = {"image_ids", "offsets", "xy", "metadata_json"}
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"detector query cache lacks {sorted(missing)}")
        image_ids = np.asarray(payload["image_ids"]).astype(str).reshape(-1)
        offsets = np.asarray(payload["offsets"], dtype=np.int64).reshape(-1)
        xy = np.asarray(payload["xy"], dtype=np.float32)
        metadata = _metadata(payload, context="detector query cache")
    if (
        metadata.get("format") != "alike_detector_mapped_radio_query_cache_v1"
        or len(image_ids) == 0
        or len(set(image_ids.tolist())) != len(image_ids)
        or offsets.shape != (len(image_ids) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(xy)
        or np.any(offsets[1:] < offsets[:-1])
        or xy.shape != (len(xy), 2)
        or np.any(~np.isfinite(xy))
        or int(metadata.get("detector_point_count", -1)) != len(xy)
    ):
        raise ValueError("detector query cache violates the frozen point contract")
    return image_ids, offsets, xy, metadata


def _load_selected_point_rows(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    required = {"selected_rows", "selected_columns", "metadata_json"}
    with np.load(Path(path), allow_pickle=False) as payload:
        if "labels" in payload.files:
            raise ValueError("selected-point artifact unexpectedly contains labels")
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"selected-point artifact lacks {sorted(missing)}")
        rows = np.asarray(payload["selected_rows"], dtype=np.int64).reshape(-1)
        columns = np.asarray(payload["selected_columns"], dtype=np.int64)
        metadata = _metadata(payload, context="selected-point artifact")
    if (
        metadata.get("format") != "detector_maplet_geometry_features_v1"
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_pose_derived_selection") is not False
        or metadata.get("supervision_mode") != "none_inference_only"
        or len(rows) == 0
        or np.unique(rows).size != len(rows)
        or columns.ndim != 2
        or columns.shape[0] != len(rows)
        or int(metadata.get("candidate_top_k", -1)) != columns.shape[1]
        or not np.all(np.sort(columns, axis=1) == np.arange(columns.shape[1]))
    ):
        raise ValueError("selected-point artifact violates the frozen candidate contract")
    return rows, columns, metadata


def _load_candidate_prior_overlay(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    required = {
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"candidate prior overlay lacks {sorted(missing)}")
        tracks = np.asarray(payload["candidate_track_ids"], dtype=np.int64)
        candidate = np.asarray(payload["candidate_probabilities"], dtype=np.float32)
        null = np.asarray(payload["null_probabilities"], dtype=np.float32).reshape(-1)
        metadata = _metadata(payload, context="candidate prior overlay")
    if (
        metadata.get("format") != "candidate_maplet_prior_overlay_v1"
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("probability_semantics")
        != "candidate_identity_probability_plus_explicit_null_equals_one"
        or tracks.ndim != 2
        or candidate.shape != tracks.shape
        or null.shape != (len(tracks),)
        or np.any(candidate < 0.0)
        or np.any(null <= 0.0)
        or np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 3e-4
    ):
        raise ValueError("candidate prior overlay violates the explicit-null contract")
    return tracks, candidate, null, metadata


def _split_by_query_id(path: Path, *, detector_image_ids: np.ndarray) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    labels: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        values = payload.get(split)
        if not isinstance(values, list):
            raise ValueError("query split lacks one required image list")
        for query_id in values:
            text = str(query_id)
            if not text or text in labels:
                raise ValueError("query split has duplicate or invalid image IDs")
            labels[text] = split
    image_ids = np.asarray(detector_image_ids).astype(str).reshape(-1)
    if set(labels) != set(image_ids.tolist()):
        raise ValueError("query split and detector query image IDs differ")
    return np.asarray([labels[image_id] for image_id in image_ids], dtype=np.str_)


def _query_rows_to_images(
    *, selected_rows: np.ndarray, image_ids: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    if np.any(rows < 0) or np.any(rows >= int(offsets[-1])):
        raise ValueError("selected detector rows are outside the query cache")
    positions = np.searchsorted(offsets[1:], rows, side="right")
    if np.any(positions < 0) or np.any(positions >= len(image_ids)):
        raise RuntimeError("selected detector rows do not resolve to images")
    return image_ids[positions]


def build_radio_intermediate_fixed_candidate_rerank(
    *,
    context_cache: Path,
    context_landmark_bank: Path,
    detector_query_cache: Path,
    selected_point_artifact: Path,
    candidate_prior_overlay: Path,
    query_split: Path,
    grid_size: int,
    output: Path,
    summary_json: Path,
) -> dict[str, Any]:
    """Export independent raw intermediate scores for fixed final candidates."""

    output_path = Path(output)
    summary_path = Path(summary_json)
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite intermediate rerank outputs")
    context_path = Path(context_cache)
    cache = load_spatial_image_context_cache(context_path, expected_format=PCA_FORMAT)
    cache_metadata = dict(cache.metadata)
    if (
        bool(cache_metadata.get("pose_or_ground_truth_used", True))
        or bool(cache_metadata.get("image_retrieval_or_submap_used", True))
        or bool(cache_metadata.get("render", True))
        or int(grid_size) not in cache.grids
    ):
        raise ValueError("intermediate context cache violates the no-retrieval protocol")
    bank_path = Path(context_landmark_bank)
    bank, bank_metadata = load_landmark_index_npz(bank_path)
    descriptor_manifest = bank_metadata.get("descriptor_space_manifest")
    if (
        bank_metadata.get("stage")
        != "radio_intermediate_pca_projected_observation_landmark_bank_v1"
        or not isinstance(descriptor_manifest, dict)
        or descriptor_manifest.get("projection_source")
        != "raw_radio_intermediate_pca_projected_observation_full_map"
        or descriptor_manifest.get("context_cache_sha256")
        != file_sha256_short(context_path)
        or int(descriptor_manifest.get("grid_size", -1)) != int(grid_size)
        or int(bank.feature_dim) != int(cache.descriptor_dim)
    ):
        raise ValueError("intermediate landmark bank is not descriptor-compatible")
    image_ids, offsets, detector_xy, detector_metadata = _load_detector_query_cache(
        Path(detector_query_cache)
    )
    selected_rows, selected_columns, selected_metadata = _load_selected_point_rows(
        Path(selected_point_artifact)
    )
    if str(selected_metadata.get("detector_query_cache_sha256", "")) != str(
        file_sha256_short(Path(detector_query_cache))
    ):
        raise ValueError("selected points reference a different detector cache")
    all_tracks, all_candidate_prior, all_null_prior, overlay_metadata = (
        _load_candidate_prior_overlay(Path(candidate_prior_overlay))
    )
    overlay_manifest = overlay_metadata.get("inference_data_manifest")
    if not isinstance(overlay_manifest, dict):
        raise ValueError("candidate prior overlay lacks its frozen inference manifest")
    if (
        all_tracks.shape[0] != len(detector_xy)
        or all_tracks.shape[1] != selected_columns.shape[1]
        or str(selected_metadata.get("proposals_sha256", ""))
        != str(overlay_metadata.get("proposals_sha256", ""))
        or str(overlay_manifest.get("detector_query_cache_sha256", ""))
        != str(file_sha256_short(Path(detector_query_cache)))
    ):
        raise ValueError("candidate prior overlay does not match the frozen selected points")
    selected_query_ids = _query_rows_to_images(
        selected_rows=selected_rows,
        image_ids=image_ids,
        offsets=offsets,
    )
    split_by_image = _split_by_query_id(Path(query_split), detector_image_ids=image_ids)
    image_positions = {image_id: row for row, image_id in enumerate(image_ids.tolist())}
    selected_splits = np.asarray(
        [split_by_image[image_positions[query_id]] for query_id in selected_query_ids],
        dtype=np.str_,
    )
    selected_xy = detector_xy[selected_rows]
    clamped_count, maximum_excursion = spatial_context_boundary_audit(
        cache,
        image_ids=selected_query_ids,
        xy=selected_xy,
        grid_size=int(grid_size),
    )
    if clamped_count:
        raise ValueError("detector query points fall outside the frozen image coordinates")
    started = time.monotonic()
    query_features = sample_spatial_context_descriptors(
        cache,
        image_ids=selected_query_ids,
        xy=selected_xy,
        grid_size=int(grid_size),
    )
    selected_tracks = np.take_along_axis(
        all_tracks[selected_rows], selected_columns, axis=1
    )
    selected_prior = np.take_along_axis(
        all_candidate_prior[selected_rows], selected_columns, axis=1
    )
    candidate_valid = selected_tracks >= 0
    candidate_rows = np.full(selected_tracks.shape, -1, dtype=np.int64)
    candidate_rows[candidate_valid] = canonical_track_rows(
        selected_tracks[candidate_valid], bank.track_ids
    )
    if np.any(candidate_valid & (candidate_rows < 0)):
        raise ValueError("fixed final candidate is absent from the intermediate track bank")
    safe_rows = np.maximum(candidate_rows, 0)
    scores = np.full(selected_tracks.shape, np.nan, dtype=np.float32)
    scores[candidate_valid] = np.einsum(
        "nd,nkd->nk",
        query_features,
        bank.features[safe_rows],
        optimize=True,
    )[candidate_valid]
    if np.any(~np.isfinite(scores[candidate_valid])):
        raise RuntimeError("intermediate candidate cosine score is non-finite")
    metadata: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "candidate_set": "existing_final_space_fixed_global_top_l_tracks",
        "candidate_reselection": False,
        "candidate_top_k": int(selected_tracks.shape[1]),
        "query_sampling": "real_image_radio_intermediate_pca_full_grid_bilinear_v1",
        "query_boundary_clamped_observation_count": int(clamped_count),
        "query_maximum_boundary_excursion_px": float(maximum_excursion),
        "descriptor_space_id": bank_metadata.get("descriptor_space_id"),
        "descriptor_space_manifest": descriptor_manifest,
        "context_cache": str(context_path),
        "context_cache_sha256": file_sha256_short(context_path),
        "context_landmark_bank": str(bank_path),
        "context_landmark_bank_sha256": file_sha256_short(bank_path),
        "detector_query_cache": str(Path(detector_query_cache)),
        "detector_query_cache_sha256": file_sha256_short(Path(detector_query_cache)),
        "detector_selection_version": detector_metadata.get("detector_selection_version"),
        "selected_point_artifact": str(Path(selected_point_artifact)),
        "selected_point_artifact_sha256": file_sha256_short(Path(selected_point_artifact)),
        "selected_point_role": "target_free_row_and_candidate_column_selection_only",
        "candidate_prior_overlay": str(Path(candidate_prior_overlay)),
        "candidate_prior_overlay_sha256": file_sha256_short(Path(candidate_prior_overlay)),
        "query_split": str(Path(query_split)),
        "query_split_sha256": file_sha256_short(Path(query_split)),
        "source_final_descriptor_space_id": detector_metadata.get("descriptor_space_id"),
        "source_final_prior_probability_semantics": overlay_metadata.get(
            "probability_semantics"
        ),
        "source_final_proposals_sha256": overlay_metadata.get("proposals_sha256"),
        "row_count": int(len(selected_rows)),
        "query_count": int(len(set(selected_query_ids.tolist()))),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_row_indices=selected_rows,
            query_ids=selected_query_ids,
            split_names=selected_splits,
            xy=selected_xy,
            selected_candidate_columns=selected_columns,
            candidate_track_ids=selected_tracks,
            candidate_bank_rows=candidate_rows,
            candidate_valid=candidate_valid,
            candidate_prior_probabilities=selected_prior,
            null_prior_probabilities=all_null_prior[selected_rows],
            radio_intermediate_cosine=scores,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    result: dict[str, Any] = {
        "stage": "build_radio_intermediate_fixed_candidate_rerank",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "row_count": int(len(selected_rows)),
        "query_count": int(len(set(selected_query_ids.tolist()))),
        "candidate_top_k": int(selected_tracks.shape[1]),
        "score_range": {
            "minimum": float(np.min(scores[candidate_valid])),
            "maximum": float(np.max(scores[candidate_valid])),
            "mean": float(np.mean(scores[candidate_valid])),
        },
        "elapsed_seconds": float(time.monotonic() - started),
        "protocol": {
            "fixed_final_top_l": True,
            "new_ann_search": False,
            "candidate_reselection": False,
            "target_free": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_radio_intermediate_fixed_candidate_rerank(
        context_cache=Path(args.context_cache),
        context_landmark_bank=Path(args.context_landmark_bank),
        detector_query_cache=Path(args.detector_query_cache),
        selected_point_artifact=Path(args.selected_point_artifact),
        candidate_prior_overlay=Path(args.candidate_prior_overlay),
        query_split=Path(args.query_split),
        grid_size=int(args.grid_size),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
