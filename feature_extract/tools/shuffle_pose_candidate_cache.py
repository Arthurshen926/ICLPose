#!/usr/bin/env python3
"""Shuffle per-query pose candidate order in an exported candidate cache.

This is a protocol utility for detecting fixed-lattice artifacts.  It preserves
the candidate set and all per-candidate teacher fields, but applies a
deterministic row-wise permutation to every array whose second dimension is the
candidate dimension.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _is_candidate_axis_array(array: np.ndarray, num_rows: int, num_candidates: int) -> bool:
    return array.ndim >= 2 and array.shape[0] == num_rows and array.shape[1] == num_candidates


def shuffle_cache(input_path: str, output_path: str, *, seed: int = 20260515) -> dict:
    src = np.load(input_path, allow_pickle=True)
    arrays = {key: src[key] for key in src.files}
    if "pose_init_candidates" not in arrays:
        raise KeyError("cache must contain pose_init_candidates")
    num_rows, num_candidates = arrays["pose_init_candidates"].shape[:2]
    rng = np.random.default_rng(int(seed))
    permutations = np.stack([rng.permutation(num_candidates) for _ in range(num_rows)], axis=0).astype(np.int64)

    shuffled = {}
    for key, array in arrays.items():
        if _is_candidate_axis_array(array, num_rows, num_candidates):
            shuffled[key] = np.take_along_axis(array, permutations.reshape(num_rows, num_candidates, *([1] * (array.ndim - 2))), axis=1)
        else:
            shuffled[key] = array

    stats = list(shuffled.get("stats", np.asarray([], dtype=object)).tolist())
    if stats and isinstance(stats[0], dict):
        stats[0] = dict(stats[0])
        stats[0]["candidate_order_shuffled"] = True
        stats[0]["candidate_shuffle_seed"] = int(seed)
    else:
        stats = [{"candidate_order_shuffled": True, "candidate_shuffle_seed": int(seed)}]
    shuffled["stats"] = np.asarray(stats, dtype=object)
    shuffled["candidate_permutation"] = permutations

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **shuffled)
    return {"num_rows": int(num_rows), "num_candidates": int(num_candidates), "output": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()
    print(shuffle_cache(args.input, args.output, seed=args.seed))


if __name__ == "__main__":
    main()
