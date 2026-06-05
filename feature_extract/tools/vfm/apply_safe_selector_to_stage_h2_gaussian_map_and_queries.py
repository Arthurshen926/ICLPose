"""Apply a Stage C2 safe selector checkpoint to Stage H2 Gaussian anchors and queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_c0_compressed_features import _dir_bytes
from feature_extract.tools.vfm.train_stage_c2_safe_selector import _write_safe_query_manifest
from feature_extract.vfm.gaussian_raw_landmarks import project_gaussian_anchor_map_features
from feature_extract.vfm.patch_selector_training import load_safe_patch_selector_checkpoint
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap
from feature_extract.vfm.tokens import TokenBankManifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply a safe selector to Stage H2 Gaussian anchors and query tokens")
    parser.add_argument("--input_anchor_npz", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--output_layer_name", default="")
    parser.add_argument("--output_anchor_npz", required=True)
    parser.add_argument("--output_query_dir", required=True)
    parser.add_argument("--output_query_manifest", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_rows", type=int, default=65536)
    parser.add_argument("--batch_tokens", type=int, default=65536)
    args = parser.parse_args(argv)

    run = load_safe_patch_selector_checkpoint(Path(args.selector_checkpoint), device=args.device)
    anchor_map = SemiDenseAnchorMap.load_npz(Path(args.input_anchor_npz))
    projected = project_gaussian_anchor_map_features(
        anchor_map,
        run,
        output_dim=int(run.summary.output_dim),
        device=args.device,
        batch_size=int(args.batch_rows),
    )
    projected.save_npz(Path(args.output_anchor_npz))

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    output_layer_name = args.output_layer_name or args.layer_name
    encoded_manifest, query_count = _write_safe_query_manifest(
        manifest,
        run,
        layer_name=args.layer_name,
        output_query_dir=Path(args.output_query_dir),
        output_layer_name=output_layer_name,
        device=args.device,
        batch_tokens=int(args.batch_tokens),
    )
    encoded_manifest.to_json(Path(args.output_query_manifest))
    summary = {
        "stage": "stage_h2_apply_safe_selector_to_gaussian_map_and_queries",
        "input_anchor_count": int(len(anchor_map)),
        "input_feature_dim": int(anchor_map.feature_dim),
        "output_anchor_count": int(len(projected)),
        "output_feature_dim": int(projected.feature_dim),
        "query_record_count": int(query_count),
        "selector": {
            "path": args.selector_checkpoint,
            "output_dim": int(run.summary.output_dim),
            "active_group_count": int(run.summary.active_group_count),
            "group_count": int(run.summary.group_count),
        },
        "storage_bytes": {
            "anchor_map": _dir_bytes(Path(args.output_anchor_npz)),
            "query_tokens": _dir_bytes(Path(args.output_query_dir)),
        },
        "outputs": {
            "anchor_map": args.output_anchor_npz,
            "query_manifest": args.output_query_manifest,
            "query_dir": args.output_query_dir,
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
