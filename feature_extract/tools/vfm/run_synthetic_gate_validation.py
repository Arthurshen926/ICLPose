"""Run a synthetic positive-control gate validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.synthetic import run_synthetic_feature_utility_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic VFM gate validation")
    parser.add_argument("--query_count", type=int, default=64)
    parser.add_argument("--candidates_per_query", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = run_synthetic_feature_utility_validation(
        query_count=args.query_count,
        candidates_per_query=args.candidates_per_query,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
