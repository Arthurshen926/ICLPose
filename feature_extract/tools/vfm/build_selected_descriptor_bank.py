"""Build cached selected-descriptor banks from dense token manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.selected_descriptor_bank import build_selected_descriptor_bank
from feature_extract.vfm.selector_descriptor_scoring import load_selector_from_checkpoint
from feature_extract.vfm.tokens import TokenBankManifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache selector-projected dense-token descriptors")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--utility_weighted_pooling", action="store_true")
    parser.add_argument("--utility_spatial_mask", choices=("none", "high", "low"), default="none")
    parser.add_argument("--utility_spatial_mask_fraction", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selector = load_selector_from_checkpoint(Path(args.selector_checkpoint), device=args.device)
    bank = build_selected_descriptor_bank(
        manifest=TokenBankManifest.from_json(Path(args.manifest)),
        selector=selector,
        layer_name=args.layer_name,
        device=args.device,
        batch_size=args.batch_size,
        utility_weighted_pooling=args.utility_weighted_pooling,
        utility_spatial_mask=args.utility_spatial_mask,
        utility_spatial_mask_fraction=args.utility_spatial_mask_fraction,
        metadata={
            "token_manifest": str(args.manifest),
            "token_manifest_sha256": file_sha256_short(Path(args.manifest)),
            "selector_checkpoint": str(args.selector_checkpoint),
            "selector_checkpoint_sha256": file_sha256_short(Path(args.selector_checkpoint)),
        },
    )
    bank.to_npz(Path(args.output))


if __name__ == "__main__":
    main()
