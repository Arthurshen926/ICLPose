"""Build query-level hard-case split files from a labeled candidate bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.hard_cases import HardCaseCandidate, build_hard_case_splits
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank


def _metadata_float(candidate, field: str) -> float | None:
    value = candidate.metadata.get(field)
    if value is None:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VFM hard-case splits")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--accept_threshold", type=float, default=0.5)
    parser.add_argument("--translation_threshold_m", type=float, default=0.25)
    parser.add_argument("--rotation_threshold_deg", type=float, default=5.0)
    parser.add_argument("--pnp_score_field", default="pnp_inliers")
    parser.add_argument("--pnp_score_threshold", type=float, default=100.0)
    parser.add_argument("--near_identity_threshold_m", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bank = CandidateHypothesisBank.from_jsonl(Path(args.bank))
    rows = []
    for candidate in bank.candidates:
        if candidate.query_id is None or candidate.pose_error is None:
            continue
        rows.append(
            HardCaseCandidate(
                query_id=candidate.query_id,
                candidate_id=candidate.candidate_id,
                cost_m=float(candidate.pose_error.translation_m),
                basin_label=candidate.basin_label(
                    args.translation_threshold_m,
                    args.rotation_threshold_deg,
                ),
                retrieval_rank=None
                if candidate.metadata.get("retrieval_rank") is None
                else int(candidate.metadata["retrieval_rank"]),
                verifier_score=None if candidate.prior_score is None else float(candidate.prior_score),
                pnp_score=_metadata_float(candidate, args.pnp_score_field),
                identity_delta_m=_metadata_float(candidate, "identity_delta_m"),
            )
        )
    splits = build_hard_case_splits(
        rows,
        accept_threshold=args.accept_threshold,
        pnp_score_threshold=args.pnp_score_threshold,
        near_identity_threshold_m=args.near_identity_threshold_m,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = splits.to_dict()
    payload["counts"] = {key: len(value) for key, value in payload.items()}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
