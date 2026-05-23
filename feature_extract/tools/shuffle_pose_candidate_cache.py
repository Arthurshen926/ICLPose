#!/usr/bin/env python3
"""Shuffle per-query pose candidate order in an exported candidate cache.

This is a protocol utility for detecting fixed-lattice artifacts.  It preserves
the candidate set and all per-candidate teacher fields, but applies a
deterministic row-wise permutation to every array whose second dimension is the
candidate dimension.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _is_candidate_axis_array(array: np.ndarray, num_rows: int, num_candidates: int) -> bool:
    return array.ndim >= 2 and array.shape[0] == num_rows and array.shape[1] == num_candidates


def _balanced_row_permutations(num_rows: int, num_candidates: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    base_order = rng.permutation(num_candidates)
    shifts = np.arange(num_rows, dtype=np.int64) % int(num_candidates)
    rng.shuffle(shifts)
    return np.stack([np.roll(base_order, int(shift)) for shift in shifts], axis=0).astype(np.int64)


def permute_pose_candidate_cache_arrays(
    cache: dict[str, np.ndarray],
    *,
    permutations: np.ndarray | None = None,
    seed: int = 20260515,
) -> tuple[dict[str, np.ndarray], dict]:
    """Apply row-wise candidate permutations to every per-candidate cache field."""

    arrays = {key: np.asarray(value) for key, value in cache.items()}
    if "pose_init_candidates" not in arrays:
        raise KeyError("cache must contain pose_init_candidates")
    num_rows, num_candidates = arrays["pose_init_candidates"].shape[:2]
    if permutations is None:
        permutations = _balanced_row_permutations(num_rows, num_candidates, seed=int(seed))
    else:
        permutations = np.asarray(permutations, dtype=np.int64)
    if permutations.shape != (num_rows, num_candidates):
        raise ValueError("permutations must have shape (num_rows,num_candidates)")
    if permutations.size and (permutations.min() < 0 or permutations.max() >= num_candidates):
        raise ValueError("permutations contain candidate indices outside cache range")
    expected = np.arange(num_candidates, dtype=np.int64)
    for row in permutations:
        if not np.array_equal(np.sort(row), expected):
            raise ValueError("each permutation row must contain every candidate index exactly once")

    shuffled = {}
    for key, array in arrays.items():
        if _is_candidate_axis_array(array, num_rows, num_candidates):
            index = permutations.reshape(num_rows, num_candidates, *([1] * (array.ndim - 2)))
            shuffled[key] = np.take_along_axis(array, index, axis=1)
        else:
            shuffled[key] = array

    stats = list(shuffled.get("stats", np.asarray([], dtype=object)).tolist())
    if stats and isinstance(stats[0], dict):
        stats[0] = dict(stats[0])
        stats[0]["candidate_order_shuffled"] = True
        stats[0]["candidate_shuffle_seed"] = int(seed)
        stats[0]["candidate_order_shuffle_mode"] = "balanced_cyclic_random_base"
        stats[0]["candidate_order_input_candidates"] = int(num_candidates)
        stats[0]["candidate_order_num_queries"] = int(num_rows)
        stats[0]["candidate_order_permutation_field"] = "candidate_permutation"
    else:
        stats = [
            {
                "candidate_order_shuffled": True,
                "candidate_shuffle_seed": int(seed),
                "candidate_order_shuffle_mode": "balanced_cyclic_random_base",
                "candidate_order_input_candidates": int(num_candidates),
                "candidate_order_num_queries": int(num_rows),
                "candidate_order_permutation_field": "candidate_permutation",
            }
        ]
    shuffled["stats"] = np.asarray(stats, dtype=object)
    shuffled["candidate_order_permutation"] = permutations
    if "candidate_permutation" not in arrays:
        shuffled["candidate_permutation"] = permutations

    metadata = {
        "candidate_order_shuffled": True,
        "candidate_shuffle_seed": int(seed),
        "seed": int(seed),
        "shuffle_mode": "balanced_cyclic_random_base",
        "num_rows": int(num_rows),
        "num_candidates": int(num_candidates),
    }
    return shuffled, metadata


def shuffle_cache(input_path: str, output_path: str, *, seed: int = 20260515) -> dict:
    src = np.load(input_path, allow_pickle=True)
    arrays = {key: src[key] for key in src.files}
    shuffled, metadata = permute_pose_candidate_cache_arrays(arrays, seed=int(seed))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **shuffled)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {**metadata, "output": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()
    print(shuffle_cache(args.input, args.output, seed=args.seed))


if __name__ == "__main__":
    main()
