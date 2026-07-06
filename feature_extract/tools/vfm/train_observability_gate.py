"""Train a measurement observability gate from audit rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.observability_gate import DEFAULT_OBSERVABILITY_GATE_FEATURES, train_observability_gate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_rows_csv", required=True)
    parser.add_argument("--val_rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_names", nargs="*", default=list(DEFAULT_OBSERVABILITY_GATE_FEATURES))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--target_precision", type=float, default=0.85)
    parser.add_argument("--minimum_threshold", type=float, default=0.5)
    parser.add_argument("--threshold_split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train_observability_gate(
        train_rows_csv=Path(args.train_rows_csv),
        val_rows_csv=Path(args.val_rows_csv),
        output_dir=Path(args.output_dir),
        feature_names=list(args.feature_names),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        target_precision=float(args.target_precision),
        minimum_threshold=float(args.minimum_threshold),
        threshold_split=str(args.threshold_split),
        device=str(args.device),
        seed=int(args.seed),
        max_train_rows=int(args.max_train_rows) if int(args.max_train_rows) > 0 else None,
        max_val_rows=int(args.max_val_rows) if int(args.max_val_rows) > 0 else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
