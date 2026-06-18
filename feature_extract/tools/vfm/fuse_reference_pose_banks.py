"""Fuse multiple reference-pose retrieval candidate banks with RRF reranking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.retrieval_fusion import fuse_reference_pose_bank_paths


def _parse_weights(value: str) -> list[float] | None:
    if not str(value).strip():
        return None
    return [float(item) for item in str(value).split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_banks", nargs="+", required=True)
    parser.add_argument("--protocol_name", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--bank_weights", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    fused = fuse_reference_pose_bank_paths(
        [Path(path) for path in args.candidate_banks],
        protocol_name=str(args.protocol_name),
        top_k=int(args.top_k),
        rrf_k=float(args.rrf_k),
        bank_weights=_parse_weights(str(args.bank_weights)),
    )
    fused.to_jsonl(Path(args.output))


if __name__ == "__main__":
    main()
