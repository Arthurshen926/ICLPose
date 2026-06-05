"""Project a raw Stage H2 Gaussian anchor map with a trained selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.gaussian_raw_landmarks import project_gaussian_anchor_map_features
from feature_extract.vfm.patch_selector_training import load_safe_patch_selector_checkpoint
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Project Stage H2 raw Gaussian anchor map with a selector")
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    raw_map = SemiDenseAnchorMap.load_npz(Path(args.input_npz))
    selector_run = load_safe_patch_selector_checkpoint(Path(args.selector_checkpoint), device=args.device)
    projected = project_gaussian_anchor_map_features(
        raw_map,
        selector_run,
        output_dim=int(selector_run.summary.output_dim),
        device=args.device,
        batch_size=int(args.batch_size),
    )
    projected.save_npz(Path(args.output_npz))
    summary = {
        "stage": "stage_h2_selector_projected_gaussian_anchor_map",
        "input_anchor_count": int(len(raw_map)),
        "input_feature_dim": int(raw_map.feature_dim),
        "output_anchor_count": int(len(projected)),
        "output_feature_dim": int(projected.feature_dim),
        "selector_output_dim": int(selector_run.summary.output_dim),
        "active_group_count": int(selector_run.summary.active_group_count),
        "inputs": {
            "input_npz": args.input_npz,
            "selector_checkpoint": args.selector_checkpoint,
        },
        "outputs": {"anchor_map": args.output_npz},
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
