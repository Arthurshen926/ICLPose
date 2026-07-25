"""Export fixed candidate-conditioned support-view posteriors with lineage.

``CandidateMapletMatcher`` predicts support-view probabilities in compact
candidate order.  The proposal bank, pose hypotheses, and all P1 verifiers use
the original top-L proposal order instead.  This exporter performs that one
permutation explicitly and binds every posterior slot to the deterministic
coverage-ranked SfM support observation used by the matcher.

The artifact is inference-only.  It contains neither residuals nor poses and
is deliberately separate from the candidate identity/null overlay: downstream
code must combine their two fixed probability distributions only as distinct
latent variables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


ARTIFACT_FORMAT = "candidate_maplet_support_view_overlay_v1"
_IDENTITY_OVERLAY_FORMAT = "candidate_maplet_prior_overlay_v1"
_POSTERIOR_SEMANTICS = (
    "identity_conditioned_support_view_probability_normalized_per_candidate_v1"
)
_SUPPORT_LAYOUT = "maplet_coverage_prefix_cycle_fixed_slots_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--feature-artifact", required=True)
    parser.add_argument("--inference-scores", required=True)
    parser.add_argument("--inference-summary", required=True)
    parser.add_argument("--candidate-prior-overlay", required=True)
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--score-prefix", default="ensemble")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _metadata(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{context} has no metadata_json")
    try:
        value = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} metadata is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata must be an object")
    return dict(value)


def _support_probability_keys(
    score_payload: Mapping[str, np.ndarray], *, prefix: str
) -> tuple[str, ...]:
    prefix_text = f"{str(prefix)}__support_view_probability_"
    parsed: list[tuple[int, str]] = []
    for key in score_payload:
        if not str(key).startswith(prefix_text):
            continue
        suffix = str(key)[len(prefix_text) :]
        if not suffix.isdigit():
            continue
        parsed.append((int(suffix), str(key)))
    parsed.sort()
    if not parsed or [index for index, _key in parsed] != list(range(len(parsed))):
        raise ValueError("support-view posterior keys must be a contiguous zero-based set")
    return tuple(key for _index, key in parsed)


def remap_compact_support_view_probabilities(
    *,
    selected_rows: np.ndarray,
    selected_columns: np.ndarray,
    compact_probabilities: np.ndarray,
    proposal_row_count: int,
    candidate_count: int,
) -> np.ndarray:
    """Map compact matcher columns back to the immutable proposal columns."""

    rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    columns = np.asarray(selected_columns, dtype=np.int64)
    probabilities = np.asarray(compact_probabilities, dtype=np.float32)
    if (
        int(proposal_row_count) <= 0
        or int(candidate_count) <= 0
        or rows.shape != (int(proposal_row_count),)
        or not np.array_equal(rows, np.arange(int(proposal_row_count), dtype=np.int64))
        or columns.shape != probabilities.shape[:2]
        or columns.shape != (int(proposal_row_count), int(candidate_count))
        or probabilities.ndim != 3
        or probabilities.shape[2] <= 0
        or np.any((columns < 0) | (columns >= int(candidate_count)))
        or not np.all(np.sort(columns, axis=1) == np.arange(int(candidate_count)))
        or np.any(~np.isfinite(probabilities))
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
        or np.any(
            np.abs(probabilities.sum(axis=2, dtype=np.float64) - 1.0) > 1e-5
        )
    ):
        raise ValueError("compact support-view posterior inputs are invalid")
    output = np.zeros(
        (int(proposal_row_count), int(candidate_count), probabilities.shape[2]),
        dtype=np.float32,
    )
    output[rows[:, None], columns] = probabilities
    if np.any(
        np.abs(output.sum(axis=2, dtype=np.float64) - 1.0) > 1e-5
    ):
        raise RuntimeError("support-view posterior remapping lost probability mass")
    return output


def coverage_prefix_support_slots(
    *,
    candidate_track_ids: np.ndarray,
    maplet_track_ids: np.ndarray,
    support_image_ids: np.ndarray,
    support_image_indices: np.ndarray,
    support_coverage_counts: np.ndarray,
    view_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the matcher\'s fixed coverage-prefix view slots exactly.

    ``CandidateMapletEpisodeStore`` takes the first ``view_count`` coverage
    views and cycles them only if a track has fewer valid views.  Repeating a
    slot is intentional: the model emitted a posterior over those slots, not
    over a deduplicated set of support images.
    """

    candidates = np.asarray(candidate_track_ids, dtype=np.int64)
    tracks = np.asarray(maplet_track_ids, dtype=np.int64).reshape(-1)
    image_ids = np.asarray(support_image_ids).astype(str).reshape(-1)
    indices = np.asarray(support_image_indices, dtype=np.int64)
    coverage = np.asarray(support_coverage_counts, dtype=np.int32)
    if (
        candidates.ndim != 2
        or candidates.shape[0] == 0
        or np.any(candidates < 0)
        or len(tracks) == 0
        or len(np.unique(tracks)) != len(tracks)
        or indices.ndim != 2
        or indices.shape != coverage.shape
        or indices.shape[0] != len(tracks)
        or int(view_count) <= 0
        or int(view_count) > indices.shape[1]
        or np.any((indices < -1) | (indices >= len(image_ids)))
        or np.any(coverage < 0)
        or np.any((indices < 0) & (coverage != 0))
    ):
        raise ValueError("coverage-prefix support-view inputs are invalid")
    lookup = {int(track_id): row for row, track_id in enumerate(tracks.tolist())}
    slot_ids = np.full((*candidates.shape, int(view_count)), "", dtype=image_ids.dtype)
    valid = np.zeros(slot_ids.shape, dtype=bool)
    for row, candidate_row in enumerate(candidates.tolist()):
        for column, track_id in enumerate(candidate_row):
            maplet_row = lookup.get(int(track_id))
            if maplet_row is None:
                raise ValueError("proposal candidate is absent from maplet support index")
            prefix = indices[maplet_row, : int(view_count)]
            prefix_coverage = coverage[maplet_row, : int(view_count)]
            supported = [
                int(image_index)
                for image_index, count in zip(prefix.tolist(), prefix_coverage.tolist())
                if int(image_index) >= 0 and int(count) > 0
            ]
            if not supported:
                raise ValueError("fixed candidate has no valid coverage-prefix support view")
            for view in range(int(view_count)):
                slot_ids[row, column, view] = image_ids[
                    supported[view % len(supported)]
                ]
                valid[row, column, view] = True
    if np.any(slot_ids[valid] == ""):
        raise RuntimeError("coverage-prefix support slots are incomplete")
    return slot_ids, valid


def _load_allowlist(path: Path, fields: Sequence[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    requested = tuple(str(field) for field in fields)
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = set(requested).difference(payload.files)
        if missing:
            raise ValueError(f"{path} lacks required fields: {sorted(missing)}")
        arrays = {field: np.asarray(payload[field]).copy() for field in requested}
        metadata = _metadata(payload, context=str(path)) if "metadata_json" in payload.files else {}
    return arrays, metadata


def build_candidate_maplet_support_view_overlay(
    *,
    proposals: Path,
    feature_artifact: Path,
    inference_scores: Path,
    inference_summary: Path,
    candidate_prior_overlay: Path,
    maplet_support_index: Path,
    score_prefix: str,
    output: Path,
) -> dict[str, Any]:
    """Build a proposal-aligned fixed support-view posterior artifact."""

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    proposal_path = Path(proposals)
    feature_path = Path(feature_artifact)
    score_path = Path(inference_scores)
    summary_path = Path(inference_summary)
    prior_path = Path(candidate_prior_overlay)
    maplet_path = Path(maplet_support_index)
    proposal_arrays, _proposal_metadata = _load_allowlist(
        proposal_path, ("query_ids", "candidate_track_ids")
    )
    feature_arrays, feature_metadata = _load_allowlist(
        feature_path, ("selected_rows", "selected_columns", "valid_edges")
    )
    with np.load(feature_path, allow_pickle=False) as feature_payload:
        if "labels" in feature_payload.files:
            raise ValueError("support-view posterior source must be inference-only")
    summary = json.loads(summary_path.read_text())
    protocol = summary.get("protocol")
    data_manifest = summary.get("data_manifest")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("inference_only") is not True
        or protocol.get("supervision_arrays_loaded") is not False
        or not isinstance(data_manifest, Mapping)
        or str(data_manifest.get("proposals_sha256", ""))
        != file_sha256_short(proposal_path)
        or str(data_manifest.get("feature_artifact_sha256", ""))
        != file_sha256_short(feature_path)
    ):
        raise ValueError("support-view posterior inference lineage is invalid")
    if (
        feature_metadata.get("format") != "detector_maplet_geometry_features_v1"
        or feature_metadata.get("supervision_mode") != "none_inference_only"
        or str(feature_metadata.get("proposals_sha256", ""))
        != file_sha256_short(proposal_path)
        or str(feature_metadata.get("maplet_support_index_sha256", ""))
        != file_sha256_short(maplet_path)
        or str(feature_metadata.get("support_view_selection", "")) != "coverage"
    ):
        raise ValueError("support-view posterior feature source is incompatible")
    score_hash = file_sha256_short(score_path)
    if str(dict(summary.get("outputs") or {}).get("scores_sha256", "")) != score_hash:
        raise ValueError("support-view posterior score hash differs from its summary")
    with np.load(score_path, allow_pickle=False) as payload:
        score_payload = {key: np.asarray(payload[key]).copy() for key in payload.files}
    keys = _support_probability_keys(score_payload, prefix=str(score_prefix))
    compact_views = np.stack(
        [np.asarray(score_payload[key], dtype=np.float32) for key in keys], axis=2
    )
    candidate_tracks = np.asarray(proposal_arrays["candidate_track_ids"], dtype=np.int64)
    query_ids = np.asarray(proposal_arrays["query_ids"]).astype(str).reshape(-1)
    rows = np.asarray(feature_arrays["selected_rows"], dtype=np.int64).reshape(-1)
    columns = np.asarray(feature_arrays["selected_columns"], dtype=np.int64)
    valid_edges = np.asarray(feature_arrays["valid_edges"], dtype=bool)
    if (
        candidate_tracks.ndim != 2
        or query_ids.shape != (len(candidate_tracks),)
        or columns.shape != candidate_tracks.shape
        or valid_edges.shape != candidate_tracks.shape
        or compact_views.shape[:2] != candidate_tracks.shape
        or not np.all(valid_edges)
        or int(feature_metadata.get("support_view_count", -1)) != len(keys)
        or int(feature_metadata.get("support_view_candidate_count", -1)) < len(keys)
    ):
        raise ValueError("support-view posterior source arrays are incompatible")
    probabilities = remap_compact_support_view_probabilities(
        selected_rows=rows,
        selected_columns=columns,
        compact_probabilities=compact_views,
        proposal_row_count=len(candidate_tracks),
        candidate_count=candidate_tracks.shape[1],
    )
    prior_arrays, prior_metadata = _load_allowlist(
        prior_path,
        ("candidate_track_ids", "candidate_probabilities", "null_probabilities"),
    )
    if (
        prior_metadata.get("format") != _IDENTITY_OVERLAY_FORMAT
        or prior_metadata.get("contains_ground_truth") is not False
        or prior_metadata.get("contains_target_errors") is not False
        or str(prior_metadata.get("proposals_sha256", ""))
        != file_sha256_short(proposal_path)
        or str(prior_metadata.get("feature_artifact_sha256", ""))
        != file_sha256_short(feature_path)
        or str(prior_metadata.get("inference_scores_sha256", "")) != score_hash
        or not np.array_equal(
            np.asarray(prior_arrays["candidate_track_ids"], dtype=np.int64), candidate_tracks
        )
    ):
        raise ValueError("identity/null and support-view posteriors are not co-lineaged")
    maplet_arrays, maplet_metadata = _load_allowlist(
        maplet_path,
        (
            "anchor_track_ids",
            "support_image_ids",
            "support_image_indices",
            "support_coverage_counts",
        ),
    )
    if maplet_metadata.get("format") != "local_maplet_support_index_npz":
        raise ValueError("support-view posterior maplet index has an unsupported format")
    support_ids, support_valid = coverage_prefix_support_slots(
        candidate_track_ids=candidate_tracks,
        maplet_track_ids=np.asarray(maplet_arrays["anchor_track_ids"], dtype=np.int64),
        support_image_ids=np.asarray(maplet_arrays["support_image_ids"]).astype(str),
        support_image_indices=np.asarray(maplet_arrays["support_image_indices"], dtype=np.int64),
        support_coverage_counts=np.asarray(maplet_arrays["support_coverage_counts"], dtype=np.int32),
        view_count=len(keys),
    )
    if (
        not np.all(support_valid)
        or np.any(support_ids == "")
        or np.any(support_ids == query_ids[:, None, None])
    ):
        raise ValueError("fixed support-view layout is invalid or leaks query images")
    metadata: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "version": 1,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "candidate_reselection": False,
        "support_reselection": False,
        "candidate_count": int(candidate_tracks.shape[1]),
        "proposal_row_count": int(len(candidate_tracks)),
        "support_view_count": int(len(keys)),
        "probability_semantics": _POSTERIOR_SEMANTICS,
        "support_view_layout": _SUPPORT_LAYOUT,
        "source_score_prefix": str(score_prefix),
        "source_support_view_probability_keys": list(keys),
        "source_support_view_selection": "coverage",
        "source_support_view_candidate_count": int(
            feature_metadata["support_view_candidate_count"]
        ),
        "proposals_sha256": file_sha256_short(proposal_path),
        "feature_artifact_sha256": file_sha256_short(feature_path),
        "inference_scores_sha256": score_hash,
        "inference_summary_sha256": file_sha256_short(summary_path),
        "candidate_prior_overlay_sha256": file_sha256_short(prior_path),
        "maplet_support_index_sha256": file_sha256_short(maplet_path),
        "source_inference_data_manifest": dict(data_manifest),
        "audit": {
            "minimum_probability_mass": float(np.min(probabilities.sum(axis=2))),
            "maximum_probability_mass": float(np.max(probabilities.sum(axis=2))),
            "query_support_image_overlap_count": 0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        candidate_track_ids=candidate_tracks,
        support_view_probabilities=probabilities.astype(np.float32),
        candidate_support_image_ids=support_ids.astype(np.str_),
        candidate_support_view_valid=support_valid.astype(bool),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "row_count": int(len(candidate_tracks)),
        "candidate_count": int(candidate_tracks.shape[1]),
        "support_view_count": int(len(keys)),
        "support_image_count": int(len(set(support_ids.reshape(-1).tolist()))),
        "metadata": metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_candidate_maplet_support_view_overlay(
        proposals=Path(args.proposals),
        feature_artifact=Path(args.feature_artifact),
        inference_scores=Path(args.inference_scores),
        inference_summary=Path(args.inference_summary),
        candidate_prior_overlay=Path(args.candidate_prior_overlay),
        maplet_support_index=Path(args.maplet_support_index),
        score_prefix=str(args.score_prefix),
        output=Path(args.output),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
