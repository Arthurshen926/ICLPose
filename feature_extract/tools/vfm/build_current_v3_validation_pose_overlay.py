"""Build a validation-only V3 probe posterior overlay for frozen pose scoring.

The tool never reads residual labels.  It copies the full proposal posterior,
then replaces only validation rows whose target-free visual predictions were
frozen after train-only fitting.  Train and test proposal rows remain exactly
at the supplied base posterior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization.current_v3_candidate_probe import (
    build_current_v3_validation_pose_overlay,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--base_prior_overlay", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_current_v3_validation_pose_overlay(
        features_path=Path(args.features),
        candidate_evidence_path=Path(args.candidate_evidence),
        predictions_path=Path(args.predictions),
        base_prior_overlay_path=Path(args.base_prior_overlay),
        family=str(args.family),
        output_path=Path(args.output),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
