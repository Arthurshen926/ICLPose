"""Target-side audit for a frozen S0 plus RGB rank-percentile fusion policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.apply_candidate_pose_rgb_spatial_rank_fusion import (
    FUSED_SCORE_FORMAT,
)
from feature_extract.tools.vfm.fit_candidate_pose_rgb_spatial_rank_fusion import (
    _EPSILON,
    _load_targets,
    _metrics,
    _row_keys,
    _tail_safe_gate,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rank_fusion import selected_position


_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "baseline_score_top1",
    "baseline_selection_scores",
    "fused_rank_percentile_scores",
)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("fused score artifact paths must be non-empty and unique")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-score-artifacts", type=_paths, required=True)
    parser.add_argument("--target-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_fused(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = set(_FIELDS).difference(payload.files)
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: fused score is incomplete ({sorted(missing)})")
        arrays = {field: np.asarray(payload[field]).copy() for field in _FIELDS}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != FUSED_SCORE_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_scoring") is not False
        or metadata.get("promotion_allowed") is not True
    ):
        raise ValueError(f"{path}: not a promotion-approved target-free fused score")
    count = int(metadata.get("row_count", -1))
    if count <= 0 or any(np.asarray(arrays[field]).shape[0] != count for field in _FIELDS):
        raise ValueError(f"{path}: fused score rows are invalid")
    if not np.isfinite(np.asarray(arrays["fused_rank_percentile_scores"], dtype=np.float64)).all():
        raise ValueError(f"{path}: fused scores are non-finite")
    if len(_row_keys(arrays)) != len(set(_row_keys(arrays))):
        raise ValueError(f"{path}: fused score repeats frozen rows")
    return arrays, metadata


def _lineage(metadata: Sequence[Mapping[str, object]], target_metadata: Mapping[str, object]) -> None:
    expected = set(str(value) for value in target_metadata.get("inference_artifact_sha256", []))
    if not expected:
        raise ValueError("fused score target lacks frozen hypothesis lineage")
    for item in metadata:
        inputs = item.get("inputs")
        if not isinstance(inputs, Mapping) or str(
            inputs.get("hypothesis_artifact", {}).get("sha256", "")
        ) not in expected:
            raise ValueError("fused score and target use different frozen hypotheses")


def _per_query_rows(
    *, arrays: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]
) -> list[dict[str, object]]:
    keys = _row_keys(arrays)
    target_rows_by_key = {key: row for row, key in enumerate(_row_keys(targets))}
    try:
        target_rows = np.asarray([target_rows_by_key[key] for key in keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError("fused score row is absent from target artifact") from error
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for row, (split, label, query_id, _hypothesis) in enumerate(keys):
        grouped.setdefault((split, label, query_id), []).append(row)
    output: list[dict[str, object]] = []
    for (split, label, query_id), group_rows in sorted(grouped.items()):
        rows = np.asarray(group_rows, dtype=np.int64)
        tie = np.asarray(arrays["hypothesis_indices"], dtype=np.int64)[rows]
        baseline = np.asarray(arrays["baseline_selection_scores"], dtype=np.float64)[rows]
        fused = np.asarray(arrays["fused_rank_percentile_scores"], dtype=np.float64)[rows]
        source = np.asarray(arrays["baseline_score_top1"], dtype=bool)[rows]
        if int(np.count_nonzero(source)) != 1:
            raise ValueError("fused score group lacks one immutable baseline top-1")
        baseline_position = selected_position(baseline, tie)
        if baseline_position != int(np.flatnonzero(source)[0]):
            raise ValueError("fused score baseline ordering differs from immutable S0")
        selected = selected_position(fused, tie)
        target = target_rows[rows]
        translation = np.asarray(targets["translation_errors_m"], dtype=np.float64)[target]
        rotation = np.asarray(targets["rotation_errors_deg"], dtype=np.float64)[target]
        output.append(
            {
                "split_name": str(split),
                "evaluation_label": str(label),
                "query_id": str(query_id),
                "hypothesis_count": int(len(rows)),
                "selected_hypothesis_index": int(tie[selected]),
                "baseline_hypothesis_index": int(tie[baseline_position]),
                "selected_translation_error_m_TARGET_ONLY": float(translation[selected]),
                "selected_rotation_error_deg_TARGET_ONLY": float(rotation[selected]),
                "baseline_translation_error_m_TARGET_ONLY": float(translation[baseline_position]),
                "baseline_rotation_error_deg_TARGET_ONLY": float(rotation[baseline_position]),
                "selection_changed_from_baseline": bool(selected != baseline_position),
                "new_catastrophic_TARGET_ONLY": bool(
                    translation[baseline_position] <= 1.0 and translation[selected] > 1.0
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty fused audit rows")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite fused audit: {summary_path}")
    loaded = [_load_fused(path) for path in args.fused_score_artifacts]
    arrays = {
        field: np.concatenate([np.asarray(item[0][field]) for item in loaded], axis=0)
        for field in _FIELDS
    }
    if len(_row_keys(arrays)) != len(set(_row_keys(arrays))):
        raise ValueError("fused score artifacts repeat frozen rows")
    targets, target_metadata = _load_targets(Path(args.target_artifact))
    _lineage([item[1] for item in loaded], target_metadata)
    rows = _per_query_rows(arrays=arrays, targets=targets)
    metrics = _metrics(rows)
    gate = _tail_safe_gate(metrics, alpha=1.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_query_TARGET_ONLY.csv", rows)
    summary = {
        "stage": "audit_candidate_pose_rgb_spatial_rank_fusion",
        "format": "candidate_pose_rgb_spatial_rank_percentile_fusion_audit_v1",
        "metrics_TARGET_ONLY": metrics,
        "promotion_gate": gate,
        "protocol": {
            "target_join_after_fused_target_free_scoring": True,
            "no_candidate_or_pose_regeneration": True,
            "no_pnp": True,
            "no_render": True,
            "no_image_retrieval_or_submap": True,
        },
        "inputs": {
            "fused_score_artifacts": [str(path) for path in args.fused_score_artifacts],
            "fused_score_artifact_sha256": [
                file_sha256_short(path) for path in args.fused_score_artifacts
            ],
            "target_artifact": str(args.target_artifact),
            "target_artifact_sha256": file_sha256_short(Path(args.target_artifact)),
        },
        "outputs": {"per_query": str(output_dir / "per_query_TARGET_ONLY.csv")},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
