"""Target-only oracle audit for a soft maplet prior on frozen hypotheses.

The audit deliberately consumes GT-near proposal tracks only after all pose
hypotheses and their independent scores are frozen.  It is an upper-bound
diagnostic for maplet context, never an inference input or a trainable prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


AUDIT_FORMAT = "oracle_maplet_hypothesis_prior_audit_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypothesis_artifacts",
        required=True,
        help="comma-separated frozen grouped-hypothesis NPZ shards",
    )
    parser.add_argument(
        "--score_artifacts",
        required=True,
        help="comma-separated frozen independent-score NPZ shards",
    )
    parser.add_argument("--hypothesis_targets", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--maplet_index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--positive_threshold_px", type=float, default=2.0)
    parser.add_argument("--success_translation_m", type=float, default=0.10)
    parser.add_argument("--success_rotation_deg", type=float, default=5.0)
    parser.add_argument(
        "--soft_prior_weights",
        default="0,0.125,0.25,0.5,1,2,4,8",
        help=(
            "comma-separated weights for score_z + weight * maplet_coverage; "
            "target-side oracle sweep only"
        ),
    )
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths:
        raise ValueError("artifact list is empty")
    return paths


def _weights(value: str) -> tuple[float, ...]:
    weights = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not weights or any(not np.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("soft maplet prior weights must be finite and non-negative")
    if len(set(weights)) != len(weights):
        raise ValueError("soft maplet prior weights must be unique")
    return weights


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError(f"{path}: metadata_json is required")
        arrays = {
            key: np.asarray(data[key]).copy()
            for key in data.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(data["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata_json must contain an object")
    return arrays, metadata


def _keys(arrays: Mapping[str, np.ndarray]) -> list[tuple[str, str, int]]:
    required = ("query_ids", "evaluation_labels", "hypothesis_indices")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"hypothesis identity fields are missing: {missing}")
    return list(
        zip(
            np.asarray(arrays["query_ids"]).astype(str).tolist(),
            np.asarray(arrays["evaluation_labels"]).astype(str).tolist(),
            np.asarray(arrays["hypothesis_indices"], dtype=np.int64).tolist(),
        )
    )


def _pose_metrics(
    translation: Sequence[float],
    rotation: Sequence[float],
    *,
    success_translation_m: float,
    success_rotation_deg: float,
) -> dict[str, float | int | None]:
    values_t = np.asarray(translation, dtype=np.float64)
    values_r = np.asarray(rotation, dtype=np.float64)
    if values_t.shape != values_r.shape:
        raise ValueError("translation and rotation arrays are not aligned")
    if len(values_t) == 0:
        return {
            "query_count": 0,
            "median_translation_m": None,
            "p90_translation_m": None,
            "median_rotation_deg": None,
            "p90_rotation_deg": None,
            "recall_success": None,
        }
    return {
        "query_count": int(len(values_t)),
        "median_translation_m": float(np.median(values_t)),
        "p90_translation_m": float(np.quantile(values_t, 0.9)),
        "median_rotation_deg": float(np.median(values_r)),
        "p90_rotation_deg": float(np.quantile(values_r, 0.9)),
        "recall_success": float(
            np.mean(
                (values_t <= float(success_translation_m))
                & (values_r <= float(success_rotation_deg))
            )
        ),
    }


def _distribution(values: Sequence[float | int | None]) -> dict[str, float | int | None]:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"sample_count": 0, "mean": None, "median": None, "p90": None}
    return {
        "sample_count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _query_maplet_members(
    proposals: Mapping[str, np.ndarray],
    selected_rows: np.ndarray,
    neighbor_by_track: Mapping[int, np.ndarray],
    *,
    threshold_px: float,
) -> dict[str, set[int]]:
    """Build a target-only query maplet from GT-near candidate tracks."""

    output: dict[str, set[int]] = {}
    for row in np.asarray(selected_rows, dtype=np.int64):
        query_id = str(proposals["query_ids"][row])
        tracks = proposals["candidate_track_ids"][row]
        residuals = proposals["candidate_gt_residuals_px"][row]
        correct = tracks[
            (tracks >= 0) & np.isfinite(residuals) & (residuals <= float(threshold_px))
        ]
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


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(scores) == 0 or np.any(~np.isfinite(scores)):
        raise ValueError("maplet audit scores must be finite and non-empty")
    center = float(np.median(scores))
    mad = float(np.median(np.abs(scores - center)))
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = max(float(np.std(scores)), 1.0)
    return (scores - center) / scale


def _soft_maplet_scores(
    baseline_scores: np.ndarray,
    coverage: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    """Scale-free diagnostic soft prior, normalized independently per query."""

    score = np.asarray(baseline_scores, dtype=np.float64).reshape(-1)
    value = np.asarray(coverage, dtype=np.float64).reshape(-1)
    if score.shape != value.shape or np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("maplet soft-prior scores are invalid")
    if not np.isfinite(float(weight)) or float(weight) < 0.0:
        raise ValueError("maplet soft-prior weight must be non-negative")
    return _robust_zscore(score) + float(weight) * value


def _first_success_rank(
    order: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    *,
    success_translation_m: float,
    success_rotation_deg: float,
) -> int | None:
    success = (
        np.asarray(translation, dtype=np.float64)[order] <= float(success_translation_m)
    ) & (
        np.asarray(rotation, dtype=np.float64)[order] <= float(success_rotation_deg)
    )
    positions = np.flatnonzero(success)
    return None if len(positions) == 0 else int(positions[0] + 1)


def _load_aligned_hypotheses(
    *,
    hypothesis_paths: Sequence[Path],
    score_paths: Sequence[Path],
    target_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if len(hypothesis_paths) != len(score_paths):
        raise ValueError("hypothesis and score shard counts differ")
    targets, target_metadata = _load_npz(Path(target_path))
    required_targets = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "translation_errors_m",
        "rotation_errors_deg",
    }
    missing = sorted(required_targets.difference(targets))
    if missing:
        raise ValueError(f"target artifact lacks fields: {missing}")
    if target_metadata.get("contains_target_fields") is not True:
        raise ValueError("maplet audit targets must be explicitly target-side")

    target_count = len(targets["query_ids"])
    for key in required_targets:
        if np.asarray(targets[key]).shape[0] != target_count:
            raise ValueError("target arrays are not row-aligned")
    target_hypothesis_hashes = target_metadata.get("hypothesis_artifact_sha256")
    target_score_hashes = target_metadata.get("score_artifact_sha256")
    if not isinstance(target_hypothesis_hashes, list) or not isinstance(target_score_hashes, list):
        raise ValueError("target artifact lacks source hash manifests")
    if len(target_hypothesis_hashes) != len(hypothesis_paths) or len(target_score_hashes) != len(score_paths):
        raise ValueError("target source shard count differs from audit inputs")

    output: dict[str, list[np.ndarray]] = {
        "query_ids": [],
        "split_names": [],
        "evaluation_labels": [],
        "hypothesis_indices": [],
        "independent_scores": [],
        "source_selected": [],
        "sample_track_ids": [],
        "translation_errors_m": [],
        "rotation_errors_deg": [],
    }
    offset = 0
    for shard, (hypothesis_path, score_path) in enumerate(zip(hypothesis_paths, score_paths)):
        if str(target_hypothesis_hashes[shard]) != file_sha256_short(hypothesis_path):
            raise ValueError(f"target hypothesis hash differs for shard {shard}")
        if str(target_score_hashes[shard]) != file_sha256_short(score_path):
            raise ValueError(f"target score hash differs for shard {shard}")
        hypothesis, hypothesis_metadata = _load_npz(hypothesis_path)
        score, score_metadata = _load_npz(score_path)
        if hypothesis_metadata.get("contains_target_fields") is not False:
            raise ValueError(f"{hypothesis_path}: hypothesis artifact is not inference-only")
        if score_metadata.get("contains_target_fields") is not False or score_metadata.get(
            "pose_or_ground_truth_used_for_scoring"
        ) is not False:
            raise ValueError(f"{score_path}: score artifact is not target-free")
        required_score = {
            "query_ids",
            "split_names",
            "evaluation_labels",
            "hypothesis_indices",
            "independent_selection_scores",
            "independent_score_top1",
        }
        required_hypothesis = {
            "query_ids",
            "split_names",
            "evaluation_labels",
            "hypothesis_indices",
            "sample_track_ids",
        }
        missing_score = sorted(required_score.difference(score))
        missing_hypothesis = sorted(required_hypothesis.difference(hypothesis))
        if missing_score or missing_hypothesis:
            raise ValueError(
                f"shard {shard}: missing score={missing_score} hypothesis={missing_hypothesis}"
            )
        score_count = len(score["query_ids"])
        if any(np.asarray(score[key]).shape[0] != score_count for key in required_score):
            raise ValueError(f"{score_path}: score arrays are not row-aligned")
        if np.asarray(hypothesis["sample_track_ids"]).shape[0] != len(
            hypothesis["query_ids"]
        ):
            raise ValueError(f"{hypothesis_path}: sample tracks are not row-aligned")
        target_slice = slice(offset, offset + score_count)
        if target_slice.stop > target_count:
            raise ValueError("score rows exceed target rows")
        for key in ("query_ids", "split_names", "evaluation_labels", "hypothesis_indices"):
            score_values = np.asarray(score[key])
            target_values = np.asarray(targets[key])[target_slice]
            if key != "hypothesis_indices":
                score_values = score_values.astype(str)
                target_values = target_values.astype(str)
            else:
                score_values = score_values.astype(np.int64)
                target_values = target_values.astype(np.int64)
            if not np.array_equal(score_values, target_values):
                raise ValueError(f"shard {shard}: target and score {key} ordering differs")
        score_keys = _keys(score)
        hypothesis_keys = _keys(hypothesis)
        if len(score_keys) != len(set(score_keys)) or len(hypothesis_keys) != len(set(hypothesis_keys)):
            raise ValueError(f"shard {shard}: hypothesis identity keys are not unique")
        hypothesis_position = {key: row for row, key in enumerate(hypothesis_keys)}
        if set(score_keys).difference(hypothesis_position):
            raise ValueError(
                f"shard {shard}: score contains hypotheses absent from the frozen source shard"
            )
        aligned_hypothesis_rows = np.asarray(
            [hypothesis_position[key] for key in score_keys], dtype=np.int64
        )
        independent_scores = np.asarray(
            score["independent_selection_scores"], dtype=np.float64
        )
        if np.any(~np.isfinite(independent_scores)):
            raise ValueError(f"{score_path}: independent score contains non-finite rows")
        output["query_ids"].append(np.asarray(score["query_ids"]).astype(str))
        output["split_names"].append(np.asarray(score["split_names"]).astype(str))
        output["evaluation_labels"].append(
            np.asarray(score["evaluation_labels"]).astype(str)
        )
        output["hypothesis_indices"].append(
            np.asarray(score["hypothesis_indices"], dtype=np.int64)
        )
        output["independent_scores"].append(independent_scores)
        output["source_selected"].append(
            np.asarray(score["independent_score_top1"], dtype=bool)
        )
        output["sample_track_ids"].append(
            np.asarray(hypothesis["sample_track_ids"], dtype=np.int64)[
                aligned_hypothesis_rows
            ]
        )
        output["translation_errors_m"].append(
            np.asarray(targets["translation_errors_m"], dtype=np.float64)[target_slice]
        )
        output["rotation_errors_deg"].append(
            np.asarray(targets["rotation_errors_deg"], dtype=np.float64)[target_slice]
        )
        offset += score_count
    if offset != target_count:
        raise ValueError("target rows remain after consuming score shards")
    return {key: np.concatenate(value, axis=0) for key, value in output.items()}, target_metadata


def _method_summary(
    rows: Sequence[Mapping[str, object]],
    method: str,
    *,
    success_translation_m: float,
    success_rotation_deg: float,
) -> dict[str, object]:
    translation = [float(row[f"{method}_translation_m"]) for row in rows]
    rotation = [float(row[f"{method}_rotation_deg"]) for row in rows]
    ranks = [row.get(f"{method}_first_success_rank") for row in rows]
    return {
        "pose": _pose_metrics(
            translation,
            rotation,
            success_translation_m=success_translation_m,
            success_rotation_deg=success_rotation_deg,
        ),
        "first_success_rank": _distribution(ranks),
    }


def _paired_summary(rows: Sequence[Mapping[str, object]], method: str) -> dict[str, object]:
    baseline = np.asarray([float(row["baseline_translation_m"]) for row in rows])
    probe = np.asarray([float(row[f"{method}_translation_m"]) for row in rows])
    delta = probe - baseline
    tolerance = 1e-9
    return {
        "translation_delta_m": _distribution(delta.tolist()),
        "wins": int(np.count_nonzero(delta < -tolerance)),
        "losses": int(np.count_nonzero(delta > tolerance)),
        "ties": int(np.count_nonzero(np.abs(delta) <= tolerance)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if float(args.positive_threshold_px) <= 0.0:
        raise ValueError("positive threshold must be positive")
    if float(args.success_translation_m) <= 0.0 or float(args.success_rotation_deg) <= 0.0:
        raise ValueError("success thresholds must be positive")
    hypothesis_paths = _paths(args.hypothesis_artifacts)
    score_paths = _paths(args.score_artifacts)
    soft_weights = _weights(args.soft_prior_weights)
    target_path = Path(args.hypothesis_targets)
    proposal_path = Path(args.proposals)
    candidate_path = Path(args.candidate_artifact)
    maplet_path = Path(args.maplet_index)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")

    aligned, target_metadata = _load_aligned_hypotheses(
        hypothesis_paths=hypothesis_paths,
        score_paths=score_paths,
        target_path=target_path,
    )
    with np.load(proposal_path, allow_pickle=False) as data:
        required = {"query_ids", "candidate_track_ids", "candidate_gt_residuals_px"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"proposal artifact lacks fields: {missing}")
        proposals = {
            "query_ids": np.asarray(data["query_ids"]).astype(str),
            "candidate_track_ids": np.asarray(data["candidate_track_ids"], dtype=np.int64),
            "candidate_gt_residuals_px": np.asarray(
                data["candidate_gt_residuals_px"], dtype=np.float64
            ),
        }
    with np.load(candidate_path, allow_pickle=False) as data:
        if "selected_rows" not in data.files or "metadata_json" not in data.files:
            raise ValueError("candidate artifact lacks selected_rows or metadata")
        candidate_metadata = json.loads(str(data["metadata_json"].item()))
        selected_rows = np.asarray(data["selected_rows"], dtype=np.int64)
    if candidate_metadata.get("contains_ground_truth") is not False:
        raise ValueError("candidate artifact must be inference-only")
    if str(candidate_metadata.get("proposals_sha256")) != file_sha256_short(proposal_path):
        raise ValueError("candidate artifact proposal manifest differs")
    if np.any((selected_rows < 0) | (selected_rows >= len(proposals["query_ids"]))) or len(
        np.unique(selected_rows)
    ) != len(selected_rows):
        raise ValueError("candidate selected rows are invalid")
    with np.load(maplet_path, allow_pickle=False) as data:
        required = {"anchor_track_ids", "neighbor_track_ids"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"maplet index lacks fields: {missing}")
        anchors = np.asarray(data["anchor_track_ids"], dtype=np.int64)
        neighbors = np.asarray(data["neighbor_track_ids"], dtype=np.int64)
    if neighbors.ndim != 2 or anchors.shape != (neighbors.shape[0],):
        raise ValueError("maplet index arrays are not aligned")
    expected_maplet_hash = candidate_metadata.get("maplet_support_index_sha256")
    if expected_maplet_hash is not None and str(expected_maplet_hash) != file_sha256_short(
        maplet_path
    ):
        raise ValueError("candidate artifact maplet manifest differs")
    neighbor_by_track = {int(track): neighbors[row] for row, track in enumerate(anchors)}
    members_by_query = _query_maplet_members(
        proposals,
        selected_rows,
        neighbor_by_track,
        threshold_px=float(args.positive_threshold_px),
    )

    query_ids = np.asarray(aligned["query_ids"]).astype(str)
    splits = np.asarray(aligned["split_names"]).astype(str)
    scores = np.asarray(aligned["independent_scores"], dtype=np.float64)
    chosen = np.asarray(aligned["source_selected"], dtype=bool)
    tracks = np.asarray(aligned["sample_track_ids"], dtype=np.int64)
    translation = np.asarray(aligned["translation_errors_m"], dtype=np.float64)
    rotation = np.asarray(aligned["rotation_errors_deg"], dtype=np.float64)
    coverage = np.asarray(
        [
            _maplet_coverage(tracks[row], members_by_query.get(str(query_id), set()))
            for row, query_id in enumerate(query_ids)
        ],
        dtype=np.float64,
    )

    per_query: list[dict[str, object]] = []
    method_names = ["baseline", "hard_oracle_coverage"] + [
        f"soft_oracle_weight_{weight:g}" for weight in soft_weights
    ]
    for query_id in np.unique(query_ids):
        indices = np.flatnonzero(query_ids == query_id)
        if len(np.unique(splits[indices])) != 1:
            raise ValueError(f"{query_id}: query appears in multiple splits")
        selected = indices[chosen[indices]]
        if len(selected) != 1:
            raise ValueError(f"{query_id}: frozen baseline must select exactly one pose")
        baseline_row = int(selected[0])
        local_scores = scores[indices]
        local_coverage = coverage[indices]
        baseline_order = np.argsort(-local_scores, kind="stable")
        hard_order = np.lexsort((-local_scores, -local_coverage))
        choices: dict[str, tuple[int, np.ndarray]] = {
            "baseline": (baseline_row, baseline_order),
            "hard_oracle_coverage": (int(indices[hard_order[0]]), hard_order),
        }
        for weight in soft_weights:
            method = f"soft_oracle_weight_{weight:g}"
            local_soft = _soft_maplet_scores(
                local_scores, local_coverage, weight=float(weight)
            )
            order = np.argsort(-local_soft, kind="stable")
            choices[method] = (int(indices[order[0]]), order)
        record: dict[str, object] = {
            "query_id": str(query_id),
            "split": str(splits[indices[0]]),
            "has_oracle_maplet": bool(members_by_query.get(str(query_id))),
            "maximum_maplet_coverage": float(np.max(local_coverage)),
            "baseline_selected_hypothesis_index": int(
                aligned["hypothesis_indices"][baseline_row]
            ),
        }
        for method, (global_row, local_order) in choices.items():
            record[f"{method}_translation_m"] = float(translation[global_row])
            record[f"{method}_rotation_deg"] = float(rotation[global_row])
            record[f"{method}_hypothesis_index"] = int(
                aligned["hypothesis_indices"][global_row]
            )
            record[f"{method}_maplet_coverage"] = float(coverage[global_row])
            record[f"{method}_first_success_rank"] = _first_success_rank(
                local_order,
                translation[indices],
                rotation[indices],
                success_translation_m=float(args.success_translation_m),
                success_rotation_deg=float(args.success_rotation_deg),
            )
        per_query.append(record)

    metrics: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in per_query if row["split"] == split]
        if not rows:
            continue
        metrics[split] = {
            "maplet_available_fraction": float(
                np.mean([bool(row["has_oracle_maplet"]) for row in rows])
            ),
            "baseline": _method_summary(
                rows,
                "baseline",
                success_translation_m=float(args.success_translation_m),
                success_rotation_deg=float(args.success_rotation_deg),
            ),
            "hard_oracle_coverage_TARGET_ONLY": _method_summary(
                rows,
                "hard_oracle_coverage",
                success_translation_m=float(args.success_translation_m),
                success_rotation_deg=float(args.success_rotation_deg),
            ),
            "hard_oracle_coverage_paired_TARGET_ONLY": _paired_summary(
                rows, "hard_oracle_coverage"
            ),
            "soft_oracle_prior_TARGET_ONLY": {
                f"{weight:g}": {
                    "selection": _method_summary(
                        rows,
                        f"soft_oracle_weight_{weight:g}",
                        success_translation_m=float(args.success_translation_m),
                        success_rotation_deg=float(args.success_rotation_deg),
                    ),
                    "paired_to_baseline": _paired_summary(
                        rows, f"soft_oracle_weight_{weight:g}"
                    ),
                }
                for weight in soft_weights
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": AUDIT_FORMAT,
        "contains_target_fields": True,
        "evaluation_only_not_inference_input": True,
        "definition": (
            "GT-near fixed-topL tracks plus frozen SfM maplet neighbors; coverage "
            "is target-side oracle evidence, not a production prior"
        ),
        "soft_prior": {
            "formula": "robust_zscore(independent_score_per_query) + weight * maplet_coverage",
            "weights": list(soft_weights),
            "selection_is_target_only": True,
        },
        "success_threshold": {
            "translation_m": float(args.success_translation_m),
            "rotation_deg": float(args.success_rotation_deg),
        },
        "positive_threshold_px": float(args.positive_threshold_px),
        "inputs": {
            "hypothesis_targets_sha256": file_sha256_short(target_path),
            "hypothesis_artifacts_sha256": [
                file_sha256_short(path) for path in hypothesis_paths
            ],
            "score_artifacts_sha256": [file_sha256_short(path) for path in score_paths],
            "proposals_sha256": file_sha256_short(proposal_path),
            "candidate_artifact_sha256": file_sha256_short(candidate_path),
            "maplet_index_sha256": file_sha256_short(maplet_path),
            "target_score_compatibility_sha256": target_metadata.get(
                "score_compatibility_sha256"
            ),
        },
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "per_query.json").write_text(
        json.dumps(per_query, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
