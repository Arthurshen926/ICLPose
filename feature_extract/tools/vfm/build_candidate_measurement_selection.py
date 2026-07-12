"""Build a frozen candidate-specific top-M pool for RGB measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_measurement_selection import (
    build_candidate_measurement_selection,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--score_artifact", required=True)
    parser.add_argument("--policy_artifact", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate_score_key", default="member_1__set_candidate_probability")
    parser.add_argument("--candidates_per_token", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_candidate_measurement_selection(
        proposals_path=Path(args.proposals),
        candidate_artifact_path=Path(args.candidate_artifact),
        score_artifact_path=Path(args.score_artifact),
        policy_artifact_path=Path(args.policy_artifact),
        split_json_path=Path(args.split_json),
        output_path=Path(args.output),
        candidate_score_key=str(args.candidate_score_key),
        candidates_per_token=int(args.candidates_per_token),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

