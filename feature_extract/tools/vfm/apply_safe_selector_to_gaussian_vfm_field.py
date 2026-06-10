"""Apply a safe patch selector checkpoint to a GaussianVFMField."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMField
from feature_extract.vfm.gaussian_vfm_field_projection import project_gaussian_vfm_field_features
from feature_extract.vfm.patch_selector_training import load_safe_patch_selector_checkpoint


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_field", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--output_field", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_rows", type=int, default=65536)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    field = GaussianVFMField.load_npz(Path(args.input_field))
    run = load_safe_patch_selector_checkpoint(Path(args.selector_checkpoint), device=args.device)
    projected = project_gaussian_vfm_field_features(
        field,
        run,
        device=args.device,
        batch_size=int(args.batch_rows),
        selector_path=args.selector_checkpoint,
    )
    projected.save_npz(Path(args.output_field))
    summary = {
        "stage": "apply_safe_selector_to_gaussian_vfm_field",
        "elapsed_sec": float(time.perf_counter() - started),
        "input": {
            "field": str(args.input_field),
            "feature_dim": int(field.feature_dim),
            "gaussian_count": int(len(field)),
        },
        "selector": {
            "checkpoint": str(args.selector_checkpoint),
            "input_dim": int(run.summary.input_dim),
            "output_dim": int(run.summary.output_dim),
            "active_group_count": int(run.summary.active_group_count),
            "group_count": int(run.summary.group_count),
        },
        "output": {
            "field": str(args.output_field),
            "feature_dim": int(projected.feature_dim),
            "gaussian_count": int(len(projected)),
            "feature_norm_mean": float(np.mean(np.linalg.norm(projected.features, axis=1))) if len(projected) else 0.0,
        },
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
