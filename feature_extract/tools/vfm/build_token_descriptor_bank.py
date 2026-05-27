"""Build a lightweight descriptor bank from a dense token manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.token_descriptor_bank import build_token_descriptor_bank
from feature_extract.vfm.tokens import TokenBankManifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pooled token descriptors")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--pooling", default="mean", choices=["mean", "gem"])
    parser.add_argument("--gem_power", type=float, default=3.0)
    parser.add_argument("--normalize_tokens", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--metadata_json", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metadata = {}
    if args.metadata_json:
        metadata = json.loads(Path(args.metadata_json).read_text())
    metadata = {
        **metadata,
        "token_manifest": str(args.manifest),
        "token_manifest_sha256": file_sha256_short(Path(args.manifest)),
    }
    bank = build_token_descriptor_bank(
        manifest=TokenBankManifest.from_json(Path(args.manifest)),
        layer_name=args.layer_name,
        pooling=args.pooling,
        gem_power=args.gem_power,
        normalize_tokens=bool(args.normalize_tokens),
        normalize=not args.no_normalize,
        metadata=metadata,
    )
    bank.to_npz(Path(args.output))


if __name__ == "__main__":
    main()
