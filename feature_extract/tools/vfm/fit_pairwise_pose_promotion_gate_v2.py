"""Fit a tail-safe pairwise pose promotion gate from explicit row artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization.pairwise_pose_promotion_v2 import (
    fit_pairwise_pose_promotion_gate_v2,
)


def _row_spec(value: str) -> tuple[Path, tuple[str, ...]]:
    parts = str(value).split("::", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("row spec must be PATH::dotted.json.key")
    return Path(parts[0]), tuple(item for item in parts[1].split(".") if item)


def _load_rows(spec: tuple[Path, tuple[str, ...]]) -> list[dict[str, object]]:
    path, keys = spec
    value = json.loads(path.read_text())
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"row spec key is absent: {path}::{'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"row spec does not resolve to a row list: {path}")
    return [dict(row) for row in value]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--baseline_{split}", type=_row_spec, required=True)
        parser.add_argument(f"--optional_{split}", type=_row_spec, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--c_value", type=float, default=0.1)
    parser.add_argument("--fold_count", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    specs = [
        args.baseline_train,
        args.optional_train,
        args.baseline_validation,
        args.optional_validation,
        args.baseline_test,
        args.optional_test,
    ]
    summary = fit_pairwise_pose_promotion_gate_v2(
        train_baseline_rows=_load_rows(args.baseline_train),
        train_optional_rows=_load_rows(args.optional_train),
        validation_baseline_rows=_load_rows(args.baseline_validation),
        validation_optional_rows=_load_rows(args.optional_validation),
        test_baseline_rows=_load_rows(args.baseline_test),
        test_optional_rows=_load_rows(args.optional_test),
        output_dir=Path(args.output_dir),
        input_paths=[spec[0] for spec in specs],
        c_value=float(args.c_value),
        fold_count=int(args.fold_count),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
