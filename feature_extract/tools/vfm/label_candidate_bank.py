"""Join score-table labels onto a deployable candidate bank."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.candidate_labeling import label_candidate_bank_from_score_table
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Label a candidate bank from a score-table bank")
    parser.add_argument("--unlabeled_bank", required=True)
    parser.add_argument("--score_table_bank", required=True)
    parser.add_argument("--protocol_name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    labeled = label_candidate_bank_from_score_table(
        unlabeled_bank=CandidateHypothesisBank.from_jsonl(Path(args.unlabeled_bank)),
        score_table_bank=CandidateHypothesisBank.from_jsonl(Path(args.score_table_bank)),
        protocol_name=args.protocol_name,
    )
    labeled.to_jsonl(Path(args.output))


if __name__ == "__main__":
    main()
