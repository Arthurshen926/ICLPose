#!/usr/bin/env python3
"""Build POFD-FS reference-pose candidate banks from HLoc pair files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import colmap_to_w2c, read_colmap_images  # noqa: E402
from feature_extract.localizability.reference_pose_bank import (  # noqa: E402
    build_reference_pose_bank,
    parse_hloc_pairs_file,
    save_reference_pose_bank,
)


def _load_colmap_pose_by_name(colmap_dir: str | Path) -> dict[str, object]:
    images = read_colmap_images(str(Path(colmap_dir) / "images.bin"))
    return {meta.name: colmap_to_w2c(meta.qvec, meta.tvec).astype("float32") for meta in images.values()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--colmap-dir", required=True)
    parser.add_argument("--pairs-file", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rot-cost-weight", type=float, default=0.1)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_by_name = _load_colmap_pose_by_name(args.colmap_dir)
    query_to_refs = parse_hloc_pairs_file(args.pairs_file)
    bank = build_reference_pose_bank(
        query_poses=pose_by_name,
        reference_poses=pose_by_name,
        query_to_refs=query_to_refs,
        topk=args.topk,
        scene=args.scene,
        rot_cost_weight=args.rot_cost_weight,
    )
    save_reference_pose_bank(bank, args.save_path)
    valid = bank.valid_mask
    summary = {
        "scene": args.scene,
        "pairs_file": str(args.pairs_file),
        "save_path": str(args.save_path),
        "num_queries": len(bank.sample_names),
        "topk": int(args.topk),
        "valid_fraction": float(valid.float().mean().item()) if valid is not None else 1.0,
        "median_top1_trans_m": float(bank.trans_err_m[:, 0].median().item()),
        "median_oracle_trans_m": float(bank.trans_err_m.masked_fill(~valid, float("inf")).min(dim=1).values.median().item())
        if valid is not None
        else float(bank.trans_err_m.min(dim=1).values.median().item()),
    }
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
