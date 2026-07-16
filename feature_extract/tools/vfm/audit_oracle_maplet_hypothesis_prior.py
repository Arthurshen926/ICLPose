"""Audit whether an oracle maplet prior can disambiguate frozen pose hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis_targets", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--maplet_index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--positive_threshold_px", type=float, default=2.0)
    return parser.parse_args(argv)


def _pose_metrics(translation: np.ndarray, rotation: np.ndarray) -> dict[str, float]:
    t = np.asarray(translation, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    return {
        "query_count": int(len(t)),
        "median_translation_m": float(np.median(t)),
        "p90_translation_m": float(np.quantile(t, 0.9)),
        "median_rotation_deg": float(np.median(r)),
        "p90_rotation_deg": float(np.quantile(r, 0.9)),
        "recall_10cm_5deg": float(np.mean((t <= 0.1) & (r <= 5.0))),
        "recall_5cm_5deg": float(np.mean((t <= 0.05) & (r <= 5.0))),
    }


def _query_maplet_members(
    proposals: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    neighbor_by_track: dict[int, np.ndarray],
    *,
    threshold_px: float,
) -> dict[str, set[int]]:
    output: dict[str, set[int]] = {}
    for row in np.asarray(selected_rows, dtype=np.int64):
        query_id = str(proposals["query_ids"][row])
        tracks = proposals["candidate_track_ids"][row]
        residuals = proposals["candidate_gt_residuals_px"][row]
        correct = tracks[(tracks >= 0) & np.isfinite(residuals) & (residuals <= threshold_px)]
        members = output.setdefault(query_id, set())
        for track_id in correct.tolist():
            members.add(int(track_id))
            neighbors = neighbor_by_track.get(int(track_id))
            if neighbors is not None:
                members.update(int(value) for value in neighbors if int(value) >= 0)
    return output


def _maplet_coverage(sample_track_ids: np.ndarray, members: set[int]) -> float:
    tracks = np.asarray(sample_track_ids, dtype=np.int64)
    valid = tracks >= 0
    if not np.any(valid) or not members:
        return 0.0
    return float(np.mean([int(track) in members for track in tracks[valid]]))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target_path = Path(args.hypothesis_targets)
    with np.load(target_path, allow_pickle=False) as data:
        targets = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
        target_metadata = json.loads(str(data["metadata_json"].item()))
    source_paths = [Path(value) for value in target_metadata["inference_artifacts"]]
    source_tracks: list[np.ndarray] = []
    source_scores: list[np.ndarray] = []
    source_chosen: list[np.ndarray] = []
    for source_path in source_paths:
        with np.load(source_path, allow_pickle=False) as data:
            source_tracks.append(np.asarray(data["sample_track_ids"], dtype=np.int64))
            verification = np.asarray(data["verification_log_likelihood_means"], dtype=np.float64)
            shortlist = np.asarray(data["shortlist_log_likelihood_means"], dtype=np.float64)
            source_scores.append(np.where(np.isfinite(verification), verification, shortlist))
            source_chosen.append(np.asarray(data["chosen_for_optional_pose"], dtype=bool))
    artifact_index = targets["source_artifact_indices"].astype(np.int64)
    source_row = targets["source_row_indices"].astype(np.int64)
    sample_tracks = np.stack(
        [source_tracks[int(a)][int(r)] for a, r in zip(artifact_index, source_row)]
    )
    baseline_scores = np.asarray(
        [source_scores[int(a)][int(r)] for a, r in zip(artifact_index, source_row)],
        dtype=np.float64,
    )
    chosen = np.asarray(
        [source_chosen[int(a)][int(r)] for a, r in zip(artifact_index, source_row)],
        dtype=bool,
    )

    proposal_path = Path(args.proposals)
    with np.load(proposal_path, allow_pickle=False) as data:
        proposals = {
            "query_ids": np.asarray(data["query_ids"]).astype(str),
            "candidate_track_ids": np.asarray(data["candidate_track_ids"], dtype=np.int64),
            "candidate_gt_residuals_px": np.asarray(
                data["candidate_gt_residuals_px"], dtype=np.float32
            ),
        }
    with np.load(Path(args.candidate_artifact), allow_pickle=False) as data:
        selected_rows = np.asarray(data["selected_rows"], dtype=np.int64)
    maplet_path = Path(args.maplet_index)
    with np.load(maplet_path, allow_pickle=False) as data:
        anchors = np.asarray(data["anchor_track_ids"], dtype=np.int64)
        neighbors = np.asarray(data["neighbor_track_ids"], dtype=np.int64)
    neighbor_by_track = {int(track): neighbors[row] for row, track in enumerate(anchors)}
    members_by_query = _query_maplet_members(
        proposals,
        selected_rows,
        neighbor_by_track,
        threshold_px=float(args.positive_threshold_px),
    )
    coverage = np.asarray(
        [
            _maplet_coverage(sample_tracks[row], members_by_query.get(str(query_id), set()))
            for row, query_id in enumerate(targets["query_ids"].astype(str))
        ],
        dtype=np.float32,
    )

    query_ids = targets["query_ids"].astype(str)
    translations = targets["translation_errors_m"].astype(np.float64)
    rotations = targets["rotation_errors_deg"].astype(np.float64)
    splits = targets["split_names"].astype(str)
    rows = []
    for query_id in np.unique(query_ids):
        indices = np.flatnonzero(query_ids == query_id)
        finite_score = np.where(np.isfinite(baseline_scores[indices]), baseline_scores[indices], -np.inf)
        chosen_rows = indices[chosen[indices]]
        baseline_row = int(chosen_rows[0]) if len(chosen_rows) else int(indices[np.argmax(finite_score)])
        order = np.lexsort((-finite_score, -coverage[indices]))
        oracle_row = int(indices[order[0]])
        correct = (translations[indices] <= 0.1) & (rotations[indices] <= 5.0)
        correct_rank = None
        if np.any(correct):
            correct_rank = int(np.flatnonzero(correct[order])[0] + 1)
        max_coverage = float(np.max(coverage[indices]))
        max_coverage_rows = indices[np.isclose(coverage[indices], max_coverage)]
        rows.append(
            {
                "query_id": str(query_id),
                "split": str(splits[indices[0]]),
                "has_oracle_maplet": bool(members_by_query.get(str(query_id))),
                "maximum_maplet_coverage": max_coverage,
                "best_10cm_rank_under_oracle_maplet": correct_rank,
                "baseline_translation_m": float(translations[baseline_row]),
                "baseline_rotation_deg": float(rotations[baseline_row]),
                "oracle_maplet_translation_m": float(translations[oracle_row]),
                "oracle_maplet_rotation_deg": float(rotations[oracle_row]),
                "maplet_conditioned_oracle_translation_m": float(
                    np.min(translations[max_coverage_rows])
                ),
            }
        )

    metrics = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        metrics[split] = {
            "maplet_available_fraction": float(
                np.mean([row["has_oracle_maplet"] for row in selected])
            ),
            "baseline": _pose_metrics(
                [row["baseline_translation_m"] for row in selected],
                [row["baseline_rotation_deg"] for row in selected],
            ),
            "oracle_maplet_selected": _pose_metrics(
                [row["oracle_maplet_translation_m"] for row in selected],
                [row["oracle_maplet_rotation_deg"] for row in selected],
            ),
            "median_best_10cm_rank_under_oracle_maplet": float(
                np.median(
                    [
                        row["best_10cm_rank_under_oracle_maplet"]
                        for row in selected
                        if row["best_10cm_rank_under_oracle_maplet"] is not None
                    ]
                )
            ),
            "maplet_conditioned_oracle_median_translation_m": float(
                np.median(
                    [row["maplet_conditioned_oracle_translation_m"] for row in selected]
                )
            ),
        }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": "oracle_maplet_hypothesis_prior_audit_v1",
        "contains_target_fields": True,
        "evaluation_only_not_inference_input": True,
        "definition": "GT-near candidate track plus its frozen SfM maplet neighbors",
        "positive_threshold_px": float(args.positive_threshold_px),
        "inputs": {
            "hypothesis_targets_sha256": file_sha256_short(target_path),
            "proposals_sha256": file_sha256_short(proposal_path),
            "candidate_artifact_sha256": file_sha256_short(Path(args.candidate_artifact)),
            "maplet_index_sha256": file_sha256_short(maplet_path),
        },
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "per_query.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
