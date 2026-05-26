"""Build a serialized candidate hypothesis bank."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_extract.vfm.candidate_adapters import load_candidate_records
from feature_extract.vfm.configs import load_protocol_config
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a VFM candidate hypothesis bank")
    parser.add_argument("--protocol_name", required=True)
    parser.add_argument("--protocol_kind", required=True, choices=[kind.value for kind in ProtocolKind])
    parser.add_argument("--protocol_config", default=None, help="Optional VFM protocol YAML for fingerprinting")
    parser.add_argument("--candidates", required=True, help="Normalized JSON/CSV candidate table")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    protocol_fingerprint = ""
    if args.protocol_config is not None:
        protocol_fingerprint = load_protocol_config(args.protocol_config).fingerprint()

    bank = CandidateHypothesisBank.from_candidates(
        protocol_name=args.protocol_name,
        protocol_kind=ProtocolKind(args.protocol_kind),
        candidates=load_candidate_records(Path(args.candidates)),
        protocol_fingerprint=protocol_fingerprint,
    )
    bank.to_jsonl(Path(args.output))


if __name__ == "__main__":
    main()
