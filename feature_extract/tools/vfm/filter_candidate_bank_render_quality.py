"""Filter a candidate bank with 3DGS render quality metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.render_quality_filter import (
    filter_candidate_bank_by_render_quality,
    load_render_quality_sidecar,
)


def _optional_float(value: float) -> float | None:
    return None if float(value) < 0.0 else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_bank", required=True)
    parser.add_argument("--quality_sidecar", required=True)
    parser.add_argument("--min_alpha_coverage", type=float, default=-1.0)
    parser.add_argument("--min_visible_ratio", type=float, default=-1.0)
    parser.add_argument("--min_overlap_score", type=float, default=-1.0)
    parser.add_argument("--max_view_angle_deg", type=float, default=-1.0)
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--allow_invalid_render", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    filtered = filter_candidate_bank_by_render_quality(
        CandidateHypothesisBank.from_jsonl(Path(args.candidate_bank)),
        load_render_quality_sidecar(Path(args.quality_sidecar)),
        min_alpha_coverage=_optional_float(float(args.min_alpha_coverage)),
        min_visible_ratio=_optional_float(float(args.min_visible_ratio)),
        min_overlap_score=_optional_float(float(args.min_overlap_score)),
        max_view_angle_deg=_optional_float(float(args.max_view_angle_deg)),
        require_render_valid=not bool(args.allow_invalid_render),
        drop_missing=not bool(args.allow_missing),
    )
    filtered.to_jsonl(Path(args.output))


if __name__ == "__main__":
    main()
