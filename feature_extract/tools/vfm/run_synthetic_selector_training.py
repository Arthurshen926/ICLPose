"""Run a synthetic selector-training positive control."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from feature_extract.vfm.training import (
    SyntheticSelectorTrainingConfig,
    run_synthetic_selector_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic VFM selector training")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_dim", type=int, default=16)
    parser.add_argument("--output_dim", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--candidates_per_query", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = SyntheticSelectorTrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        group_size=args.group_size,
        candidates_per_query=args.candidates_per_query,
        seed=args.seed,
        device=args.device,
    )
    result = run_synthetic_selector_training(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "result": asdict(result),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
