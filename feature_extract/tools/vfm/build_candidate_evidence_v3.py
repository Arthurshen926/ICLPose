"""Build a provenance-locked top-M candidate evidence V3 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_evidence_v3 import (
    build_candidate_evidence_v3,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--score_artifact", required=True)
    parser.add_argument("--score_summary", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score_prefix", default="ensemble")
    parser.add_argument("--candidate_score_key", default="")
    parser.add_argument("--dustbin_score_key", default="")
    parser.add_argument("--candidates_per_token", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_candidate_evidence_v3(
        proposals_path=Path(args.proposals),
        candidate_artifact_path=Path(args.candidate_artifact),
        score_artifact_path=Path(args.score_artifact),
        score_summary_path=Path(args.score_summary),
        split_json_path=Path(args.split_json),
        colmap_model_dir=Path(args.colmap_model_dir),
        projected_landmark_bank_path=Path(args.projected_landmark_bank),
        maplet_support_index_path=Path(args.maplet_support_index),
        support_geometry_index_path=Path(args.support_geometry_index),
        output_path=Path(args.output),
        score_prefix=str(args.score_prefix),
        candidate_score_key=(
            None if not str(args.candidate_score_key) else str(args.candidate_score_key)
        ),
        dustbin_score_key=(
            None if not str(args.dustbin_score_key) else str(args.dustbin_score_key)
        ),
        candidates_per_token=int(args.candidates_per_token),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
