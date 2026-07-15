"""Materialize a target-free spatial subset of candidate-maplet features.

The source feature artifact must already be inference-only. Query points are
selected exclusively from a target-free detector cache using detector score,
deterministic spatial quota, and optional NMS. Pose-derived proposal fields are
never loaded and are never copied to the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.real_image_observation_features import (
    SPATIAL_DETECTION_SELECTION_VERSION,
    spatially_diverse_detection_indices,
)


SUBSET_SELECTION_POLICY = "detector_score_soft_grid_quota_target_free_v1"


def _metadata(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{context} has no metadata_json")
    return json.loads(str(np.asarray(payload["metadata_json"]).item()))


def _expanded_detector_rows(
    image_ids: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    ids = np.asarray(image_ids).astype(str).reshape(-1)
    boundaries = np.asarray(offsets, dtype=np.int64).reshape(-1)
    if boundaries.shape != (len(ids) + 1,) or boundaries[0] != 0:
        raise ValueError("detector cache offsets are invalid")
    counts = np.diff(boundaries)
    if np.any(counts < 0):
        raise ValueError("detector cache offsets are not monotonic")
    return np.repeat(ids, counts)


def materialize_candidate_maplet_inference_subset(
    *,
    proposals_path: Path,
    detector_query_cache_path: Path,
    inference_feature_artifact_path: Path,
    output_path: Path,
    query_points_per_image: int,
    image_width: int,
    image_height: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    nms_radius_px: float = 0.0,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if int(query_points_per_image) <= 0:
        raise ValueError("query_points_per_image must be positive")
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image dimensions must be positive")

    # Deliberately load only target-free proposal fields. In particular, this
    # process never reads pose_keep_mask or any GT residual array.
    with np.load(Path(proposals_path), allow_pickle=False) as payload:
        proposal_query_ids = np.asarray(payload["query_ids"]).astype(str)
        proposal_xy = np.asarray(payload["xy"], dtype=np.float32)
    with np.load(Path(detector_query_cache_path), allow_pickle=False) as payload:
        detector_image_ids = np.asarray(payload["image_ids"]).astype(str)
        detector_offsets = np.asarray(payload["offsets"], dtype=np.int64)
        detector_xy = np.asarray(payload["xy"], dtype=np.float32)
        detector_scores = np.asarray(payload["detector_scores"], dtype=np.float32)
        detector_metadata = _metadata(payload, context="detector query cache")
    expanded_query_ids = _expanded_detector_rows(
        detector_image_ids, detector_offsets
    )
    if (
        proposal_xy.shape != detector_xy.shape
        or detector_scores.shape != proposal_query_ids.shape
        or not np.array_equal(proposal_query_ids, expanded_query_ids)
        or not np.allclose(proposal_xy, detector_xy, rtol=0.0, atol=1e-5)
    ):
        raise ValueError("detector cache rows do not align with proposal rows")

    with np.load(Path(inference_feature_artifact_path), allow_pickle=False) as payload:
        forbidden = {"labels", "selected_from_pose_keep"} & set(payload.files)
        if forbidden:
            raise ValueError(
                "source inference feature artifact contains supervision fields: "
                f"{sorted(forbidden)}"
            )
        source_rows = np.asarray(payload["selected_rows"], dtype=np.int64)
        source_columns = np.asarray(payload["selected_columns"], dtype=np.int64)
        source_features = np.asarray(payload["features"], dtype=np.float32)
        source_valid = np.asarray(payload["valid_edges"], dtype=bool)
        feature_names = np.asarray(payload["feature_names"])
        source_metadata = _metadata(payload, context="inference feature artifact")
    if str(source_metadata.get("supervision_mode")) != "none_inference_only":
        raise ValueError("source feature artifact is not inference-only")
    if str(source_metadata.get("proposals_sha256")) != str(
        file_sha256_short(Path(proposals_path))
    ):
        raise ValueError("source feature artifact references different proposals")
    if (
        source_columns.shape != source_valid.shape
        or source_features.shape[:2] != source_columns.shape
        or source_rows.shape != (source_columns.shape[0],)
        or len(np.unique(source_rows)) != len(source_rows)
    ):
        raise ValueError("source feature arrays have incompatible shapes")
    if np.any((source_rows < 0) | (source_rows >= len(proposal_query_ids))):
        raise ValueError("source feature rows are outside the proposal artifact")

    selected_rows: list[int] = []
    per_image_counts: dict[str, int] = {}
    for image_index, query_id in enumerate(detector_image_ids.tolist()):
        start = int(detector_offsets[image_index])
        end = int(detector_offsets[image_index + 1])
        local_count = end - start
        if local_count < int(query_points_per_image):
            raise ValueError(
                f"query {query_id} has only {local_count} detector rows"
            )
        local = spatially_diverse_detection_indices(
            detector_xy[start:end],
            detector_scores[start:end],
            top_k=int(query_points_per_image),
            nms_radius_px=float(nms_radius_px),
            image_width=int(image_width),
            image_height=int(image_height),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
        )
        if len(local) != int(query_points_per_image):
            raise ValueError(f"query {query_id} did not produce the requested subset")
        selected_rows.extend((local + start).astype(np.int64).tolist())
        per_image_counts[str(query_id)] = int(len(local))
    selected = np.asarray(selected_rows, dtype=np.int64)
    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("target-free query subset contains duplicate rows")

    source_position = np.full((len(proposal_query_ids),), -1, dtype=np.int64)
    source_position[source_rows] = np.arange(len(source_rows), dtype=np.int64)
    positions = source_position[selected]
    if np.any(positions < 0):
        raise ValueError("source inference features do not cover the selected rows")
    output_metadata = dict(source_metadata)
    output_metadata.update(
        {
            "contains_ground_truth": False,
            "contains_pose_derived_selection": False,
            "detector_query_cache_sha256": file_sha256_short(
                Path(detector_query_cache_path)
            ),
            "detector_selection_version": SPATIAL_DETECTION_SELECTION_VERSION,
            "query_point_selection": SUBSET_SELECTION_POLICY,
            "query_points_per_image": int(query_points_per_image),
            "selected_row_count": int(len(selected)),
            "selection_grid_rows": int(grid_rows),
            "selection_grid_cols": int(grid_cols),
            "selection_nms_radius_px": float(nms_radius_px),
            "selection_image_width": int(image_width),
            "selection_image_height": int(image_height),
            "source_inference_feature_artifact_sha256": file_sha256_short(
                Path(inference_feature_artifact_path)
            ),
            "source_query_point_selection": source_metadata.get(
                "query_point_selection"
            ),
            "supervision_mode": "none_inference_only",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        selected_rows=selected,
        selected_columns=source_columns[positions],
        features=source_features[positions],
        valid_edges=source_valid[positions],
        feature_names=feature_names,
        metadata_json=np.asarray(json.dumps(output_metadata, sort_keys=True)),
    )
    return {
        "stage": "candidate_maplet_target_free_spatial_subset",
        "protocol": {
            "ground_truth_loaded": False,
            "pose_keep_loaded": False,
            "selection_inputs": ["detector_xy", "detector_scores"],
            "selection_policy": SUBSET_SELECTION_POLICY,
        },
        "inputs": {
            "proposals": str(proposals_path),
            "proposals_sha256": file_sha256_short(Path(proposals_path)),
            "detector_query_cache": str(detector_query_cache_path),
            "detector_query_cache_sha256": file_sha256_short(
                Path(detector_query_cache_path)
            ),
            "inference_feature_artifact": str(inference_feature_artifact_path),
            "inference_feature_artifact_sha256": file_sha256_short(
                Path(inference_feature_artifact_path)
            ),
        },
        "selection": {
            "query_count": int(len(detector_image_ids)),
            "query_points_per_image": int(query_points_per_image),
            "selected_row_count": int(len(selected)),
            "per_image_counts": per_image_counts,
            "grid_rows": int(grid_rows),
            "grid_cols": int(grid_cols),
            "nms_radius_px": float(nms_radius_px),
            "detector_cache_selection_version": detector_metadata.get(
                "detector_selection_version"
            ),
        },
        "outputs": {
            "feature_artifact": str(output),
            "feature_artifact_sha256": file_sha256_short(output),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--inference_feature_artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query_points_per_image", type=int, default=128)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--grid_cols", type=int, default=4)
    parser.add_argument("--nms_radius_px", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = materialize_candidate_maplet_inference_subset(
        proposals_path=Path(args.proposals),
        detector_query_cache_path=Path(args.detector_query_cache),
        inference_feature_artifact_path=Path(args.inference_feature_artifact),
        output_path=Path(args.output),
        query_points_per_image=int(args.query_points_per_image),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        grid_rows=int(args.grid_rows),
        grid_cols=int(args.grid_cols),
        nms_radius_px=float(args.nms_radius_px),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
