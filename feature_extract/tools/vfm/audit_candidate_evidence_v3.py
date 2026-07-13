"""Marginalize RGB support views and run the P20 pre-PnP evidence audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_evidence_v3_audit import (
    build_and_audit_candidate_evidence_v3,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument("--spatial_likelihood", required=True)
    parser.add_argument("--split_name", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--validity_source",
        default="inverse_dustbin",
        choices=("inverse_dustbin", "geometry_head"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_and_audit_candidate_evidence_v3(
        candidate_evidence_path=Path(args.candidate_evidence),
        spatial_likelihood_path=Path(args.spatial_likelihood),
        split_name=str(args.split_name),
        output_path=Path(args.output),
        validity_source=str(args.validity_source),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
