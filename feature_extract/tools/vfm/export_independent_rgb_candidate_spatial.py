"""Export per-view spatial likelihoods from independent real-RGB verifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_rgb_spatial_export import (
    export_candidate_rgb_spatial_likelihood,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True, help="comma-separated checkpoints")
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument("--availability_evidence", required=True)
    parser.add_argument("--train_rows_csv", required=True)
    parser.add_argument("--validation_rows_csv", required=True)
    parser.add_argument("--test_rows_csv", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_cache_max_gb", type=float, default=9.0)
    parser.add_argument(
        "--image_cache_dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--allow_legacy_identity_dustbin",
        action="store_true",
        help="diagnostic only: allow verifier v2 identity probability as dustbin",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoints = tuple(
        Path(value.strip())
        for value in str(args.checkpoints).split(",")
        if value.strip()
    )
    summary = export_candidate_rgb_spatial_likelihood(
        checkpoints=checkpoints,
        candidate_evidence=Path(args.candidate_evidence),
        availability_evidence=Path(args.availability_evidence),
        train_rows_csv=Path(args.train_rows_csv),
        validation_rows_csv=Path(args.validation_rows_csv),
        test_rows_csv=Path(args.test_rows_csv),
        split_name=str(args.split),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        batch_size=int(args.batch_size),
        device=str(args.device),
        image_cache_max_gb=float(args.image_cache_max_gb),
        image_cache_dtype=str(args.image_cache_dtype),
        use_amp=not bool(args.no_amp),
        allow_legacy_identity_dustbin=bool(
            args.allow_legacy_identity_dustbin
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
