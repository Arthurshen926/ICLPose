#!/usr/bin/env python3
"""Select oracle plus score-hard wrong candidates from a scored candidate cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


if not hasattr(argparse, "BooleanOptionalAction"):
    class _BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            options = []
            for option in option_strings:
                options.append(option)
                if option.startswith("--"):
                    options.append("--no-" + option[2:])
            super().__init__(option_strings=options, dest=dest, nargs=0, default=default, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not str(option_string).startswith("--no-"))

    argparse.BooleanOptionalAction = _BooleanOptionalAction


def _row_id(row: dict) -> int:
    return int(row.get("global_row", row.get("row", 0)))


def _valid_row(row: dict) -> bool:
    return bool(row.get("valid", True))


def _group_rows(rows: Iterable[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_row_id(row), []).append(row)
    return grouped


def _select_indices_for_row(
    rows: list[dict],
    *,
    hard_count: int,
    score_fill_count: int,
    cost_gap_m: float,
    include_oracle: bool,
) -> tuple[list[int], list[bool]]:
    valid_rows = [row for row in rows if _valid_row(row)]
    if not valid_rows:
        return [0] * (int(include_oracle) + int(hard_count) + int(score_fill_count)), [False] * (
            int(include_oracle) + int(hard_count) + int(score_fill_count)
        )
    oracle = min(valid_rows, key=lambda row: float(row.get("pose_cost_m", float("inf"))))
    oracle_cost = float(oracle.get("pose_cost_m", float("inf")))
    selected: list[int] = []
    selected_valid: list[bool] = []
    used: set[int] = set()
    if include_oracle:
        idx = int(oracle["candidate_idx"])
        selected.append(idx)
        selected_valid.append(True)
        used.add(idx)

    hard_rows = [
        row
        for row in valid_rows
        if int(row["candidate_idx"]) not in used and float(row.get("pose_cost_m", float("inf"))) >= oracle_cost + float(cost_gap_m)
    ]
    hard_rows.sort(key=lambda row: float(row.get("score", -float("inf"))), reverse=True)
    for row in hard_rows[: int(hard_count)]:
        idx = int(row["candidate_idx"])
        selected.append(idx)
        selected_valid.append(True)
        used.add(idx)

    fill_rows = [row for row in valid_rows if int(row["candidate_idx"]) not in used]
    fill_rows.sort(key=lambda row: float(row.get("score", -float("inf"))), reverse=True)
    target = int(include_oracle) + int(hard_count) + int(score_fill_count)
    for row in fill_rows[: max(0, target - len(selected))]:
        idx = int(row["candidate_idx"])
        selected.append(idx)
        selected_valid.append(True)
        used.add(idx)

    while len(selected) < target:
        selected.append(int(oracle["candidate_idx"]))
        selected_valid.append(False)
    return selected, selected_valid


def select_score_hard_candidate_arrays(
    cache: dict[str, np.ndarray],
    candidate_rows: Iterable[dict],
    *,
    hard_count: int = 1,
    score_fill_count: int = 0,
    cost_gap_m: float = 0.12,
    include_oracle: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """Filter a pose-candidate cache to oracle plus score-high wrong candidates."""
    if "pose_init_candidates" not in cache:
        raise KeyError("cache must contain pose_init_candidates")
    candidates = np.asarray(cache["pose_init_candidates"])
    if candidates.ndim < 3:
        raise ValueError("pose_init_candidates must have shape (B,K,...)")
    cache_bsz, num_candidates = int(candidates.shape[0]), int(candidates.shape[1])
    grouped = _group_rows(candidate_rows)
    bsz = min(cache_bsz, max(grouped.keys()) + 1) if grouped else cache_bsz
    selected_indices = []
    selected_valid = []
    for row_idx in range(bsz):
        indices, valid = _select_indices_for_row(
            grouped.get(row_idx, []),
            hard_count=int(hard_count),
            score_fill_count=int(score_fill_count),
            cost_gap_m=float(cost_gap_m),
            include_oracle=bool(include_oracle),
        )
        selected_indices.append(indices)
        selected_valid.append(valid)
    selected_idx = np.asarray(selected_indices, dtype=np.int64)
    selected_valid_arr = np.asarray(selected_valid, dtype=bool)
    if selected_idx.size and (selected_idx.min() < 0 or selected_idx.max() >= num_candidates):
        raise ValueError("candidate table contains candidate_idx outside cache range")

    row_index = np.arange(bsz)[:, None]
    selected_cache: dict[str, np.ndarray] = {}
    for key, value in cache.items():
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.shape[0] == cache_bsz and arr.shape[1] == num_candidates:
            selected_cache[key] = arr[row_index, selected_idx]
        elif arr.ndim >= 1 and arr.shape[0] == cache_bsz:
            selected_cache[key] = arr[:bsz]
        else:
            selected_cache[key] = arr
    selected_cache["candidate_valid_mask"] = selected_valid_arr
    metadata = {
        "score_hard_selected": True,
        "num_rows": int(bsz),
        "num_input_rows": int(cache_bsz),
        "num_input_candidates": int(num_candidates),
        "num_output_candidates": int(selected_idx.shape[1]) if selected_idx.ndim == 2 else 0,
        "include_oracle": bool(include_oracle),
        "hard_count": int(hard_count),
        "score_fill_count": int(score_fill_count),
        "cost_gap_m": float(cost_gap_m),
        "selected_indices": selected_idx.tolist(),
        "selected_valid_fraction": float(selected_valid_arr.mean()) if selected_valid_arr.size else 0.0,
    }
    selected_cache["stats"] = np.array([metadata], dtype=object)
    return selected_cache, metadata


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hard-count", type=int, default=1)
    parser.add_argument("--score-fill-count", type=int, default=0)
    parser.add_argument("--cost-gap-m", type=float, default=0.12)
    parser.add_argument("--include-oracle", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=True) as data:
        cache = {key: data[key] for key in data.files}
    rows = _load_jsonl(args.candidate_table)
    selected, metadata = select_score_hard_candidate_arrays(
        cache,
        rows,
        hard_count=int(args.hard_count),
        score_fill_count=int(args.score_fill_count),
        cost_gap_m=float(args.cost_gap_m),
        include_oracle=bool(args.include_oracle),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **selected)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**metadata, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
