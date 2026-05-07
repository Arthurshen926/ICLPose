from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


REQUIRED_LIST_COLUMNS = (
    "hyp_final_trans_err_mm",
    "hyp_final_rot_err_deg",
    "hyp_loftr_render_scores",
    "hyp_loftr_render_inliers",
    "hyp_loftr_render_matches",
    "hyp_loftr_render_mean_conf",
    "hyp_loftr_render_consistency_m",
)
OPTIONAL_LIST_COLUMNS = (
    "hyp_candidate_quality_scores",
    "hyp_consensus_support",
)
ALL_LIST_COLUMNS = REQUIRED_LIST_COLUMNS + OPTIONAL_LIST_COLUMNS


@dataclass(frozen=True)
class Sample:
    rot_err_deg: np.ndarray
    trans_err_mm: np.ndarray
    loftr_scores: np.ndarray
    loftr_inliers: np.ndarray
    loftr_matches: np.ndarray
    loftr_mean_conf: np.ndarray
    loftr_consistency_m: np.ndarray
    quality_scores: np.ndarray
    consensus_support: np.ndarray


@dataclass(frozen=True)
class SelectorSummary:
    rule: str
    num_samples: int
    num_selected: int
    median_rot_deg: float
    median_trans_mm: float
    mean_rot_deg: float
    mean_trans_mm: float
    success_pose_pct: float
    success_trans_pct: float
    selected_indices: List[int]


def _invalid_float(invalid_value: float) -> float:
    return float(invalid_value)


def _coerce_float(value: object, invalid_value: float) -> float:
    if value is None:
        return _invalid_float(invalid_value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"none", "null", "nan"}:
            return _invalid_float(invalid_value)
        value = stripped
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _invalid_float(invalid_value)
    if not math.isfinite(number):
        return _invalid_float(invalid_value)
    return number


def parse_list_cell(cell: object, *, invalid_value: float = float("-inf")) -> List[float]:
    """Parse evaluate CSV list cells written as JSON or Python list strings."""
    if cell is None:
        return []
    if isinstance(cell, np.ndarray):
        return [_coerce_float(v, invalid_value) for v in cell.tolist()]
    if isinstance(cell, (list, tuple)):
        return [_coerce_float(v, invalid_value) for v in cell]

    text = str(cell).strip()
    if text == "":
        return []

    parsed: object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = [part.strip() for part in text.split(",") if part.strip()]

    if isinstance(parsed, (list, tuple)):
        values = parsed
    else:
        values = [parsed]
    return [_coerce_float(v, invalid_value) for v in values]


def _array_from_row(row: Dict[str, str], column: str, length: int, invalid_value: float) -> np.ndarray:
    values = parse_list_cell(row.get(column), invalid_value=invalid_value)
    if len(values) < length:
        values = values + [invalid_value] * (length - len(values))
    elif len(values) > length:
        values = values[:length]
    return np.asarray(values, dtype=np.float64)


def _read_samples(csv_path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header")
        missing = [col for col in REQUIRED_LIST_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

        for row_idx, row in enumerate(reader, start=2):
            rot = parse_list_cell(row.get("hyp_final_rot_err_deg"))
            trans = parse_list_cell(row.get("hyp_final_trans_err_mm"))
            length = min(len(rot), len(trans))
            if length <= 0:
                continue
            rot_arr = np.asarray(rot[:length], dtype=np.float64)
            trans_arr = np.asarray(trans[:length], dtype=np.float64)
            if not np.any(np.isfinite(rot_arr) & np.isfinite(trans_arr)):
                continue

            sample = Sample(
                rot_err_deg=rot_arr,
                trans_err_mm=trans_arr,
                loftr_scores=_array_from_row(row, "hyp_loftr_render_scores", length, float("-inf")),
                loftr_inliers=_array_from_row(row, "hyp_loftr_render_inliers", length, float("-inf")),
                loftr_matches=_array_from_row(row, "hyp_loftr_render_matches", length, float("-inf")),
                loftr_mean_conf=_array_from_row(row, "hyp_loftr_render_mean_conf", length, float("-inf")),
                loftr_consistency_m=_array_from_row(
                    row,
                    "hyp_loftr_render_consistency_m",
                    length,
                    float("inf"),
                ),
                quality_scores=_array_from_row(row, "hyp_candidate_quality_scores", length, float("-inf")),
                consensus_support=_array_from_row(row, "hyp_consensus_support", length, float("-inf")),
            )
            if len(sample.rot_err_deg) != length or len(sample.trans_err_mm) != length:
                raise ValueError(f"row {row_idx} has inconsistent final error list lengths")
            samples.append(sample)
    return samples


def _loftr_valid_mask(sample: Sample) -> np.ndarray:
    has_matches = np.isfinite(sample.loftr_matches) & (sample.loftr_matches > 0)
    has_inliers = np.isfinite(sample.loftr_inliers) & (sample.loftr_inliers > 0)
    return has_matches | has_inliers


def _argmax(values: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[int]:
    usable = np.isfinite(values)
    if mask is not None:
        usable &= mask
    if not np.any(usable):
        return None
    scores = np.where(usable, values, float("-inf"))
    return int(np.argmax(scores))


def _argmin(values: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[int]:
    usable = np.isfinite(values)
    if mask is not None:
        usable &= mask
    if not np.any(usable):
        return None
    scores = np.where(usable, values, float("inf"))
    return int(np.argmin(scores))


def _normalize(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float64)
    finite_values = values[finite]
    lo = float(np.min(finite_values))
    hi = float(np.max(finite_values))
    if hi <= lo:
        normalized = np.zeros_like(values, dtype=np.float64)
        normalized[finite] = 1.0
        return normalized
    normalized = np.zeros_like(values, dtype=np.float64)
    normalized[finite] = (values[finite] - lo) / (hi - lo)
    return normalized


def _score_quality_tiebreak(sample: Sample) -> Optional[int]:
    usable = np.isfinite(sample.loftr_scores)
    if not np.any(usable):
        return None
    quality = np.where(np.isfinite(sample.quality_scores), sample.quality_scores, float("-inf"))
    best_idx: Optional[int] = None
    best_key = (float("-inf"), float("-inf"))
    for idx in np.flatnonzero(usable):
        key = (float(sample.loftr_scores[idx]), float(quality[idx]))
        if key > best_key:
            best_key = key
            best_idx = int(idx)
    return best_idx


def _support_loftr_hybrid(sample: Sample) -> Optional[int]:
    valid = np.isfinite(sample.loftr_scores)
    if not np.any(valid):
        return None
    if np.any(np.isfinite(sample.consensus_support)):
        hybrid = _normalize(sample.consensus_support) + 0.25 * _normalize(sample.loftr_scores)
    else:
        hybrid = _normalize(sample.loftr_scores)
    return _argmax(hybrid, valid)


Selector = Callable[[Sample], Optional[int]]


SELECTORS: Dict[str, Selector] = {
    "current_score_max": lambda sample: _argmax(sample.loftr_scores),
    "inliers_max": lambda sample: _argmax(sample.loftr_inliers),
    "matches_max": lambda sample: _argmax(sample.loftr_matches),
    "mean_conf_max": lambda sample: _argmax(sample.loftr_mean_conf),
    "consistency_min": lambda sample: _argmin(sample.loftr_consistency_m, _loftr_valid_mask(sample)),
    "inliers_over_1plus_consistency": lambda sample: _argmax(
        sample.loftr_inliers / (1.0 + np.maximum(sample.loftr_consistency_m, 0.0)),
        np.isfinite(sample.loftr_consistency_m),
    ),
    "score_quality_tiebreak": _score_quality_tiebreak,
    "support_loftr_hybrid": _support_loftr_hybrid,
}


def _summary(rule: str, samples: Sequence[Sample], selector: Selector) -> SelectorSummary:
    selected_indices: List[int] = []
    selected_rot: List[float] = []
    selected_trans: List[float] = []

    for sample in samples:
        idx = selector(sample)
        if idx is None:
            selected_indices.append(-1)
            continue
        if idx < 0 or idx >= len(sample.rot_err_deg):
            selected_indices.append(-1)
            continue
        rot = float(sample.rot_err_deg[idx])
        trans = float(sample.trans_err_mm[idx])
        if not (math.isfinite(rot) and math.isfinite(trans)):
            selected_indices.append(-1)
            continue
        selected_indices.append(int(idx))
        selected_rot.append(rot)
        selected_trans.append(trans)

    if not selected_rot:
        nan = float("nan")
        return SelectorSummary(rule, len(samples), 0, nan, nan, nan, nan, nan, nan, selected_indices)

    rot_arr = np.asarray(selected_rot, dtype=np.float64)
    trans_arr = np.asarray(selected_trans, dtype=np.float64)
    success_pose = (rot_arr < 0.5) & (trans_arr < 100.0)
    success_trans = trans_arr < 100.0
    return SelectorSummary(
        rule=rule,
        num_samples=len(samples),
        num_selected=len(selected_rot),
        median_rot_deg=float(np.median(rot_arr)),
        median_trans_mm=float(np.median(trans_arr)),
        mean_rot_deg=float(np.mean(rot_arr)),
        mean_trans_mm=float(np.mean(trans_arr)),
        success_pose_pct=float(np.mean(success_pose) * 100.0),
        success_trans_pct=float(np.mean(success_trans) * 100.0),
        selected_indices=selected_indices,
    )


def sweep_selectors(csv_path: str | Path, selectors: Optional[Iterable[str]] = None) -> List[SelectorSummary]:
    samples = _read_samples(Path(csv_path))
    selector_names = list(selectors) if selectors is not None else list(SELECTORS.keys())
    unknown = [name for name in selector_names if name not in SELECTORS]
    if unknown:
        raise ValueError(f"unknown selector(s): {', '.join(unknown)}")
    return [_summary(name, samples, SELECTORS[name]) for name in selector_names]


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def write_summary_csv(rows: Sequence[SelectorSummary], stream) -> None:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "rule",
            "num_samples",
            "num_selected",
            "median_rot_deg",
            "median_trans_mm",
            "mean_rot_deg",
            "mean_trans_mm",
            "success_pose_pct",
            "success_trans_pct",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.rule,
                row.num_samples,
                row.num_selected,
                _format_float(row.median_rot_deg),
                _format_float(row.median_trans_mm),
                _format_float(row.mean_rot_deg),
                _format_float(row.mean_trans_mm),
                _format_float(row.success_pose_pct),
                _format_float(row.success_trans_pct),
            ]
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep offline LoFTR-render candidate selectors from feature_retrieval sample_metrics.csv."
    )
    parser.add_argument("--csv", required=True, type=Path, help="Path to evaluate sample_metrics.csv")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for selector summary CSV")
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=None,
        help="Optional subset of selector names to run",
    )
    args = parser.parse_args(argv)

    rows = sweep_selectors(args.csv, selectors=args.selectors)
    if args.output is None:
        write_summary_csv(rows, sys.stdout)
    else:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            write_summary_csv(rows, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
