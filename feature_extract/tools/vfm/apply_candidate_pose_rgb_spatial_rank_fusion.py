"""Apply a frozen target-free rank-percentile fusion policy to visual scores.

No target artifact is accepted.  The result is a target-free pose-hypothesis
score artifact that can later be audited on a separate target-side command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.fit_candidate_pose_rgb_spatial_rank_fusion import (
    CALIBRATION_FORMAT,
    _canonical_hash,
    _contract_fingerprint,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_llr import (
    CANDIDATE_POSE_LLR_SCORE_FORMAT,
    validate_target_free_pose_llr_score_metadata,
)
from feature_extract.vfm.localization.candidate_pose_rank_fusion import (
    RANK_PERCENTILE_FUSION_POLICY,
    fuse_rank_percentiles,
    selected_position,
)


FUSED_SCORE_FORMAT = "candidate_pose_rgb_spatial_rank_percentile_fused_scores_v1"
_ROW_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "baseline_score_top1",
    "baseline_selection_scores",
    "pose_log_likelihood_ratios",
)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("visual score artifact paths must be non-empty and unique")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-score-artifacts", type=_paths, required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _row_keys(arrays: Mapping[str, np.ndarray]) -> tuple[tuple[str, str, str, int], ...]:
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    splits = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    labels = np.asarray(arrays["evaluation_labels"]).astype(str).reshape(-1)
    hypotheses = np.asarray(arrays["hypothesis_indices"], dtype=np.int64).reshape(-1)
    if len({len(query_ids), len(splits), len(labels), len(hypotheses)}) != 1:
        raise ValueError("rank fusion score rows are misaligned")
    return tuple(
        (str(split), str(label), str(query_id), int(hypothesis))
        for split, label, query_id, hypothesis in zip(splits, labels, query_ids, hypotheses)
    )


def _load_calibration(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("rank fusion calibration is unreadable") from error
    if (
        not isinstance(payload, dict)
        or payload.get("format") != CALIBRATION_FORMAT
        or payload.get("policy") != RANK_PERCENTILE_FUSION_POLICY
        or payload.get("promotion_allowed") is not True
        or not isinstance(payload.get("score_contract_sha256"), str)
    ):
        raise ValueError("rank fusion calibration is not promotion-approved")
    alpha = payload.get("frozen_alpha")
    if not isinstance(alpha, (int, float)) or not np.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("rank fusion calibration has no positive frozen alpha")
    return payload


def _load_visual_score(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = set(_ROW_FIELDS).difference(payload.files)
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: visual score is incomplete ({sorted(missing)})")
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files if key != "metadata_json"}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: visual score metadata is invalid")
    validate_target_free_pose_llr_score_metadata(metadata)
    if (
        metadata.get("format") != CANDIDATE_POSE_LLR_SCORE_FORMAT
        or metadata.get("evidence_variant") != "visual"
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_scoring") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or metadata.get("promotion_allowed") is not False
    ):
        raise ValueError(f"{path}: score is not an immutable target-free visual score")
    count = int(metadata.get("row_count", -1))
    if count <= 0 or any(np.asarray(arrays[field]).shape[0] != count for field in _ROW_FIELDS):
        raise ValueError(f"{path}: visual score rows are invalid")
    if len(_row_keys(arrays)) != len(set(_row_keys(arrays))):
        raise ValueError(f"{path}: visual score repeats frozen hypothesis rows")
    return arrays, metadata


def _fused_scores(arrays: Mapping[str, np.ndarray], *, alpha: float) -> np.ndarray:
    keys = _row_keys(arrays)
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for row, (split, label, query_id, _hypothesis) in enumerate(keys):
        grouped.setdefault((split, label, query_id), []).append(row)
    output = np.empty((len(keys),), dtype=np.float64)
    baseline_top1 = np.asarray(arrays["baseline_score_top1"], dtype=bool)
    for group_rows in grouped.values():
        rows = np.asarray(group_rows, dtype=np.int64)
        ties = np.asarray(arrays["hypothesis_indices"], dtype=np.int64)[rows]
        baseline = np.asarray(arrays["baseline_selection_scores"], dtype=np.float64)[rows]
        visual = np.asarray(arrays["pose_log_likelihood_ratios"], dtype=np.float64)[rows]
        source = np.asarray(baseline_top1[rows], dtype=bool)
        if int(np.count_nonzero(source)) != 1:
            raise ValueError("visual score group lacks one immutable baseline top-1")
        if selected_position(baseline, ties) != int(np.flatnonzero(source)[0]):
            raise ValueError("visual score baseline ordering differs from immutable S0")
        output[rows] = fuse_rank_percentiles(
            baseline_scores=baseline,
            visual_scores=visual,
            tie_break_orders=ties,
            alpha=float(alpha),
        )
    if not np.isfinite(output).all():
        raise RuntimeError("rank fusion produced non-finite scores")
    return output.astype(np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    calibration_path = Path(args.calibration)
    calibration = _load_calibration(calibration_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []
    for score_path in args.visual_score_artifacts:
        arrays, metadata = _load_visual_score(score_path)
        if _contract_fingerprint([metadata]) != str(calibration["score_contract_sha256"]):
            raise ValueError("visual score contract differs from frozen rank fusion calibration")
        output_path = output_dir / f"{score_path.stem}.rank_fused.npz"
        if output_path.exists() and not bool(args.force):
            raise FileExistsError(f"refusing to overwrite fused score: {output_path}")
        output_arrays = dict(arrays)
        output_arrays["fused_rank_percentile_scores"] = _fused_scores(
            arrays, alpha=float(calibration["frozen_alpha"])
        )
        fused_metadata = {
            "format": FUSED_SCORE_FORMAT,
            "contains_target_fields": False,
            "pose_or_ground_truth_used_for_scoring": False,
            "supervision_arrays_loaded": False,
            "render": False,
            "image_retrieval_or_submap_used": False,
            "fixed_global_topl": True,
            "candidate_reselection_per_pose": False,
            "support_reselection_per_pose": False,
            "diagnostic_only": False,
            "promotion_allowed": True,
            "selection_requires_separate_target_side_audit": True,
            "policy": RANK_PERCENTILE_FUSION_POLICY,
            "frozen_alpha": float(calibration["frozen_alpha"]),
            "row_count": int(len(np.asarray(arrays["query_ids"]))),
            "query_count": int(len(set(np.asarray(arrays["query_ids"]).astype(str).tolist()))),
            "score_contract_sha256": str(calibration["score_contract_sha256"]),
            "parent_visual_score": {
                "path": str(score_path),
                "sha256": file_sha256_short(score_path),
                "metadata_sha256": _canonical_hash(metadata),
            },
            "calibration": {
                "path": str(calibration_path),
                "sha256": file_sha256_short(calibration_path),
            },
            "parent_strict_candidate_pose_llr_contract": metadata.get(
                "strict_candidate_pose_llr_contract"
            ),
            "inputs": metadata.get("inputs"),
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **output_arrays,
                metadata_json=np.asarray(json.dumps(fused_metadata, sort_keys=True), dtype=np.str_),
            )
        temporary.replace(output_path)
        outputs.append(
            {
                "input": str(score_path),
                "output": str(output_path),
                "output_sha256": file_sha256_short(output_path),
            }
        )
    summary = {
        "stage": "apply_candidate_pose_rgb_spatial_rank_fusion",
        "calibration": str(calibration_path),
        "outputs": outputs,
        "target_or_pose_loaded": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
