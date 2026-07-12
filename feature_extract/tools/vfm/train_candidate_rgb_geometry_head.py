"""Train the isolated RGB candidate-geometry head on frozen top-M rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.measurement_v1.candidate_rgb_geometry_training import (
    train_candidate_rgb_geometry_head,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_rows_csv", required=True)
    parser.add_argument("--validation_rows_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--target_key", default="target_geometry_correct_5px")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--max_eval_rows", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--positive_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_cache_max_gb", type=float, default=18.0)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train_candidate_rgb_geometry_head(
                train_rows_csv=Path(args.train_rows_csv),
                validation_rows_csv=Path(args.validation_rows_csv),
                image_root=Path(args.image_root),
                init_checkpoint=Path(args.init_checkpoint),
                output_dir=Path(args.output_dir),
                image_width=int(args.image_width),
                image_height=int(args.image_height),
                target_key=str(args.target_key),
                steps=int(args.steps),
                batch_size=int(args.batch_size),
                eval_batch_size=int(args.eval_batch_size),
                max_eval_rows=int(args.max_eval_rows),
                learning_rate=float(args.learning_rate),
                positive_fraction=float(args.positive_fraction),
                seed=int(args.seed),
                device=str(args.device),
                image_cache_max_gb=float(args.image_cache_max_gb),
                use_amp=not bool(args.no_amp),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
