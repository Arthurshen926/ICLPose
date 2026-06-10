"""Merge MATCHA coarse-fine adapter sample caches."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.matcha_coarse_fine_adapter import (
    load_matcha_coarse_fine_training_set_npz,
    merge_matcha_coarse_fine_training_sets,
    save_matcha_coarse_fine_training_set_npz,
)


def _sha(path: str) -> str:
    value = Path(path)
    return file_sha256_short(value) if value.exists() and value.is_file() else ""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    samples = []
    input_summaries = []
    for path_str in args.inputs:
        sample, metadata = load_matcha_coarse_fine_training_set_npz(Path(path_str))
        samples.append(sample)
        input_summaries.append(
            {
                "path": str(path_str),
                "sha256": _sha(path_str),
                "sample_count": int(sample.sample_count),
                "metadata": metadata,
            }
        )
    merged = merge_matcha_coarse_fine_training_sets(samples, max_samples=int(args.max_samples), seed=int(args.seed))
    save_matcha_coarse_fine_training_set_npz(merged, Path(args.output))
    summary = {
        "stage": "merge_matcha_coarse_fine_adapter_samples",
        "elapsed_sec": float(time.perf_counter() - started),
        "inputs": input_summaries,
        "config": {"max_samples": int(args.max_samples), "seed": int(args.seed)},
        "outputs": {
            "sample_cache": str(args.output),
            "sample_count": int(merged.sample_count),
            "metadata": dict(merged.metadata or {}),
        },
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
