"""Build a validation-only pose overlay from summary-top-four predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization.frozen_fulltrack_candidate_appearance_residual_probe import (
    load_frozen_fulltrack_appearance_features,
)
from feature_extract.vfm.localization.frozen_fulltrack_summary_pose_overlay import (
    build_frozen_fulltrack_summary_top4_validation_pose_overlay,
)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise argparse.ArgumentTypeError(
            "appearance artifacts must be non-empty and unique"
        )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifacts", required=True, type=_paths)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--base-prior-overlay", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_frozen_fulltrack_summary_top4_validation_pose_overlay(
        features=load_frozen_fulltrack_appearance_features(args.appearance_artifacts),
        predictions_path=Path(args.predictions),
        base_prior_overlay_path=Path(args.base_prior_overlay),
        proposals_path=Path(args.proposals),
        family=str(args.family),
        output_path=Path(args.output),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
