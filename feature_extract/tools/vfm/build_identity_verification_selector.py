"""Freeze a spatially diverse, identity-confidence verification point set.

The selector is deliberately an inference-only artifact.  It scores only the
already frozen candidate posterior, detector coordinates, and the candidate
fit-row partition.  It never reads poses, residuals, visibility, or labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


ARTIFACT_FORMAT = "identity_posterior_verification_selector_v1"
SELECTION_STRATEGY = "candidate_max_probability_spatial_quota_v1"
EXACT_IDENTITY_SEMANTICS = (
    "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--identity_prior_overlay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--point_count", type=int, default=64)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--grid_columns", type=int, default=4)
    parser.add_argument("--points_per_cell", type=int, default=4)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--image_height", type=int, default=576)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _metadata(
    data: Mapping[str, np.ndarray], *, context: str, required: bool = True
) -> dict[str, object]:
    if "metadata_json" not in data:
        if required:
            raise ValueError(f"{context} lacks metadata_json")
        return {}
    value = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def _load_fields(
    path: Path, keys: Sequence[str], *, context: str, require_metadata: bool = True
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        missing = set(keys) - set(data.files)
        if missing:
            raise ValueError(f"{context} lacks fields: {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in keys}
        metadata = _metadata(data, context=context, required=require_metadata)
    return arrays, metadata


def _sorted_by_score(rows: np.ndarray, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.int64).reshape(-1)
    merit = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape != merit.shape or np.any(~np.isfinite(merit)):
        raise ValueError("selector rows and scores are incompatible")
    return values[np.lexsort((values, -merit))]


def select_identity_confidence_spatial_quota(
    *,
    source_rows: np.ndarray,
    xy: np.ndarray,
    candidate_probabilities: np.ndarray,
    point_count: int,
    grid_rows: int,
    grid_columns: int,
    points_per_cell: int,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select fixed held-out rows by posterior confidence plus spatial quotas."""

    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    coordinates = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    posterior = np.asarray(candidate_probabilities, dtype=np.float64)
    if (
        len(rows) == 0
        or coordinates.shape != (len(rows), 2)
        or posterior.ndim != 2
        or posterior.shape[0] != len(rows)
        or int(point_count) <= 0
        or int(point_count) > len(rows)
        or int(grid_rows) <= 0
        or int(grid_columns) <= 0
        or int(points_per_cell) <= 0
        or int(image_width) <= 0
        or int(image_height) <= 0
    ):
        raise ValueError("identity verification selector inputs are invalid")
    if len(np.unique(rows)) != len(rows) or np.any(~np.isfinite(coordinates)):
        raise ValueError("identity verification selector rows are invalid")
    if np.any(~np.isfinite(posterior)) or np.any(posterior < 0.0):
        raise ValueError("identity posterior probabilities are invalid")
    confidence = np.max(posterior, axis=1)
    if np.any(coordinates[:, 0] < 0.0) or np.any(coordinates[:, 1] < 0.0):
        raise ValueError("identity verification coordinates must be non-negative")
    columns = np.minimum(
        (coordinates[:, 0] * int(grid_columns) / float(image_width)).astype(np.int64),
        int(grid_columns) - 1,
    )
    grid_rows_index = np.minimum(
        (coordinates[:, 1] * int(grid_rows) / float(image_height)).astype(np.int64),
        int(grid_rows) - 1,
    )
    columns = np.maximum(columns, 0)
    grid_rows_index = np.maximum(grid_rows_index, 0)
    selected: list[int] = []
    selected_set: set[int] = set()
    for row_index in range(int(grid_rows)):
        for column_index in range(int(grid_columns)):
            mask = (grid_rows_index == row_index) & (columns == column_index)
            local = _sorted_by_score(rows[mask], confidence[mask])
            for value in local[: int(points_per_cell)].tolist():
                selected.append(int(value))
                selected_set.add(int(value))
                if len(selected) == int(point_count):
                    break
            if len(selected) == int(point_count):
                break
        if len(selected) == int(point_count):
            break
    if len(selected) < int(point_count):
        for value in _sorted_by_score(rows, confidence).tolist():
            if int(value) in selected_set:
                continue
            selected.append(int(value))
            selected_set.add(int(value))
            if len(selected) == int(point_count):
                break
    chosen = np.asarray(selected, dtype=np.int64)
    if chosen.shape != (int(point_count),) or len(np.unique(chosen)) != len(chosen):
        raise RuntimeError("identity verification selector failed to fill point budget")
    lookup = {int(row): index for index, row in enumerate(rows.tolist())}
    return chosen, confidence[[lookup[int(row)] for row in chosen.tolist()]].astype(
        np.float32, copy=False
    )


def build_identity_verification_selector(
    *,
    detector_path: Path,
    proposals_path: Path,
    candidate_path: Path,
    identity_overlay_path: Path,
    output_path: Path,
    summary_path: Path,
    point_count: int,
    grid_rows: int,
    grid_columns: int,
    points_per_cell: int,
    image_width: int,
    image_height: int,
    force: bool = False,
) -> dict[str, object]:
    if output_path.exists() and not bool(force):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if summary_path.exists() and not bool(force):
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    detector, detector_metadata = _load_fields(
        detector_path, ("image_ids", "offsets", "xy"), context="detector cache"
    )
    proposals, proposal_metadata = _load_fields(
        proposals_path,
        ("query_ids", "candidate_track_ids"),
        context="proposals",
        require_metadata=False,
    )
    candidate, candidate_metadata = _load_fields(
        candidate_path, ("selected_rows",), context="candidate artifact"
    )
    overlay, overlay_metadata = _load_fields(
        identity_overlay_path,
        ("candidate_track_ids", "candidate_probabilities", "null_probabilities"),
        context="identity overlay",
    )
    if bool(candidate_metadata.get("contains_ground_truth", True)) or str(
        candidate_metadata.get("supervision_mode", "")
    ) != "none_inference_only":
        raise ValueError("candidate artifact must be inference-only")
    expected_detector_hash = file_sha256_short(detector_path)
    expected_proposals_hash = file_sha256_short(proposals_path)
    if str(candidate_metadata.get("detector_query_cache_sha256", "")) != str(
        expected_detector_hash
    ):
        raise ValueError("candidate artifact references different detector cache")
    if str(candidate_metadata.get("proposals_sha256", "")) != str(
        expected_proposals_hash
    ):
        raise ValueError("candidate artifact references different proposals")
    if bool(overlay_metadata.get("contains_ground_truth", True)) or bool(
        overlay_metadata.get("contains_target_errors", True)
    ):
        raise ValueError("identity overlay must be target-free")
    if str(overlay_metadata.get("probability_semantics", "")) != EXACT_IDENTITY_SEMANTICS:
        raise ValueError("identity overlay must have exact-track probability semantics")
    if str(overlay_metadata.get("proposals_sha256", "")) != str(expected_proposals_hash):
        raise ValueError("identity overlay references different proposals")
    image_ids = np.asarray(detector["image_ids"]).astype(str).reshape(-1)
    offsets = np.asarray(detector["offsets"], dtype=np.int64).reshape(-1)
    xy = np.asarray(detector["xy"], dtype=np.float32).reshape(-1, 2)
    proposal_queries = np.asarray(proposals["query_ids"]).astype(str).reshape(-1)
    proposal_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    selected_rows = np.asarray(candidate["selected_rows"], dtype=np.int64).reshape(-1)
    overlay_tracks = np.asarray(overlay["candidate_track_ids"], dtype=np.int64)
    posterior = np.asarray(overlay["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(overlay["null_probabilities"], dtype=np.float32).reshape(-1)
    if (
        offsets.shape != (len(image_ids) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(xy)
        or np.any(offsets[1:] < offsets[:-1])
        or proposal_queries.shape != (len(xy),)
        or proposal_tracks.shape != overlay_tracks.shape != posterior.shape
        or proposal_tracks.shape[0] != len(xy)
        or null.shape != (len(xy),)
        or not np.array_equal(proposal_tracks, overlay_tracks)
        or np.any((selected_rows < 0) | (selected_rows >= len(xy)))
        or len(np.unique(selected_rows)) != len(selected_rows)
    ):
        raise ValueError("identity selector inputs are not aligned")
    valid = proposal_tracks >= 0
    mass = np.sum(np.where(valid, posterior, 0.0), axis=1, dtype=np.float64) + null
    if (
        np.any(~np.isfinite(posterior))
        or np.any(~np.isfinite(null))
        or np.any(posterior < 0.0)
        or np.any(null < 0.0)
        or np.any(np.abs(posterior[~valid]) > 1e-6)
        or np.max(np.abs(mass - 1.0)) > 1e-4
    ):
        raise ValueError("identity overlay posterior mass is invalid")
    output_queries: list[str] = []
    output_offsets = [0]
    output_rows: list[np.ndarray] = []
    output_scores: list[np.ndarray] = []
    selected_set = set(selected_rows.tolist())
    for image_index, query_id in enumerate(image_ids.tolist()):
        begin, end = int(offsets[image_index]), int(offsets[image_index + 1])
        rows = np.arange(begin, end, dtype=np.int64)
        if not np.all(proposal_queries[rows] == str(query_id)):
            raise ValueError(f"proposal ownership differs for {query_id}")
        unused = np.asarray(
            [int(row) for row in rows.tolist() if int(row) not in selected_set],
            dtype=np.int64,
        )
        chosen, confidence = select_identity_confidence_spatial_quota(
            source_rows=unused,
            xy=xy[unused],
            candidate_probabilities=posterior[unused],
            point_count=int(point_count),
            grid_rows=int(grid_rows),
            grid_columns=int(grid_columns),
            points_per_cell=int(points_per_cell),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        output_queries.append(str(query_id))
        output_rows.append(chosen)
        output_scores.append(confidence)
        output_offsets.append(output_offsets[-1] + len(chosen))
    rows = np.concatenate(output_rows)
    scores = np.concatenate(output_scores)
    metadata = {
        "format": ARTIFACT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "selection_strategy": SELECTION_STRATEGY,
        "selection": {
            "point_count": int(point_count),
            "grid_rows": int(grid_rows),
            "grid_columns": int(grid_columns),
            "points_per_cell": int(points_per_cell),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "fallback": "global_identity_confidence_after_grid_quota_v1",
        },
        "source_row_selection": "detector_rows_excluding_frozen_candidate_fit_rows_v1",
        "identity_overlay_probability_semantics": EXACT_IDENTITY_SEMANTICS,
        "detector_query_cache_sha256": expected_detector_hash,
        "proposals_sha256": expected_proposals_hash,
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "identity_prior_overlay_sha256": file_sha256_short(identity_overlay_path),
        "query_count": int(len(output_queries)),
        "selected_row_count": int(len(rows)),
        "detector_coordinate_space_id": detector_metadata.get("coordinate_space_id"),
        "proposals_format": proposal_metadata.get("format"),
        "proposals_metadata_present": bool(proposal_metadata),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        query_ids=np.asarray(output_queries, dtype=np.str_),
        offsets=np.asarray(output_offsets, dtype=np.int64),
        source_row_indices=rows.astype(np.int64, copy=False),
        identity_confidence=scores.astype(np.float32, copy=False),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "stage": "build_identity_verification_selector",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "protocol": {
            "target_free": True,
            "pose_or_ground_truth_used": False,
            "fixed_before_hypothesis_scoring": True,
            "candidate_fit_rows_disjoint": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
        "metadata": metadata,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_identity_verification_selector(
        detector_path=Path(args.detector_query_cache),
        proposals_path=Path(args.proposals),
        candidate_path=Path(args.candidate_artifact),
        identity_overlay_path=Path(args.identity_prior_overlay),
        output_path=Path(args.output),
        summary_path=Path(args.summary_json),
        point_count=int(args.point_count),
        grid_rows=int(args.grid_rows),
        grid_columns=int(args.grid_columns),
        points_per_cell=int(args.points_per_cell),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
