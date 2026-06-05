"""Project a Stage F raw VFM landmark bank into a trained selector space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.map_lifting import load_selected_track_bank_npz, save_selected_track_bank_npz
from feature_extract.vfm.patch_selector_training import load_safe_patch_selector_checkpoint
from feature_extract.vfm.vfm_aware_landmarks import project_selected_track_bank


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Project Stage F raw landmark bank with a safe selector checkpoint")
    parser.add_argument("--input_bank", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--output_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    bank = load_selected_track_bank_npz(Path(args.input_bank))
    selector_run = load_safe_patch_selector_checkpoint(Path(args.selector_checkpoint), device=args.device)
    projected = project_selected_track_bank(
        bank,
        selector_run.model,
        output_dim=int(selector_run.summary.output_dim),
        device=args.device,
        batch_size=int(args.batch_size),
        active_group_mask=selector_run.active_group_mask,
    )
    output_bank = Path(args.output_bank)
    save_selected_track_bank_npz(projected, output_bank)
    summary = {
        "stage": "stage_f_selector_projected_landmark_bank",
        "inputs": {
            "input_bank": str(args.input_bank),
            "selector_checkpoint": str(args.selector_checkpoint),
        },
        "outputs": {
            "bank": str(output_bank),
        },
        "input_track_count": int(len(bank.tracks)),
        "input_feature_dim": int(bank.feature_dim),
        "output_track_count": int(len(projected.tracks)),
        "output_feature_dim": int(projected.feature_dim),
        "selector_output_dim": int(selector_run.summary.output_dim),
        "active_group_count": int(selector_run.summary.active_group_count),
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
