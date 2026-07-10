"""Run real-image RADIO selector + coarse + measurement closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from feature_extract.vfm.localization.coarse_matcher import MatchaTopKCoarseMatcher
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.measurement import RGBPatchMeasurementAdapter
from feature_extract.vfm.localization.pipeline import (
    load_real_radio_localization_pairs_csv,
    run_real_radio_localization_pairs,
)
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.measurement_v1.rgb_patch_match_table_fusion import load_rgb_patch_measurement_branch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--measurement_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_key", default="")
    parser.add_argument("--feature_path_template", default="{image_id}.npz")
    parser.add_argument("--k_per_query", type=int, default=1)
    parser.add_argument("--mutual_mode", default="annotate", choices=("none", "filter", "annotate"))
    parser.add_argument("--logit_scale", type=float, default=10.0)
    parser.add_argument("--min_similarity", type=float, default=-1.0)
    parser.add_argument("--max_matches", type=int, default=0)
    parser.add_argument("--anchor_side", default="query", choices=("query", "reference"))
    parser.add_argument("--prediction_head", default="gated", choices=("likelihood_mean", "mean", "direct", "gated"))
    parser.add_argument("--measurement_batch_size", type=int, default=128)
    parser.add_argument("--gt_reference_radius_px", type=float, default=4.0)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def resolve_runtime_device(device: str) -> torch.device:
    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    runtime_device = resolve_runtime_device(str(args.device))
    device_text = str(runtime_device)
    joint_run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=device_text)
    measurement_checkpoint = Path(args.measurement_checkpoint) if str(args.measurement_checkpoint) else Path(args.matcha_joint_checkpoint)
    measurement_branch = load_rgb_patch_measurement_branch(measurement_checkpoint, device=runtime_device)
    mutual_mode = str(args.mutual_mode)
    backend_anchor_side = "render" if str(args.anchor_side) == "reference" else "query"
    pairs = load_real_radio_localization_pairs_csv(
        Path(args.pairs_csv),
        feature_path_template=str(args.feature_path_template),
    )
    summary = run_real_radio_localization_pairs(
        pairs,
        image_root=Path(args.image_root),
        feature_root=Path(args.feature_root),
        output_dir=Path(args.output_dir),
        feature_mapper=JointFeatureMapper(joint_run.model, device=device_text),
        coarse_matcher=MatchaTopKCoarseMatcher(
            k_per_query=int(args.k_per_query),
            mutual_mode=mutual_mode,
            logit_scale=float(args.logit_scale),
            min_similarity=float(args.min_similarity),
            max_matches=int(args.max_matches) if int(args.max_matches) > 0 else None,
            anchor_side=backend_anchor_side,
        ),
        measurement_branch=RGBPatchMeasurementAdapter(
            branch=measurement_branch,
            device=device_text,
            prediction_head=str(args.prediction_head),
            batch_size=int(args.measurement_batch_size),
        ),
        feature_key=str(args.feature_key),
        max_pairs=int(args.max_pairs) if int(args.max_pairs) > 0 else None,
        gt_reference_radius_px=float(args.gt_reference_radius_px),
    )
    summary = {
        **summary,
        "pairs_csv": str(args.pairs_csv),
        "image_root": str(args.image_root),
        "feature_root": str(args.feature_root),
        "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
        "measurement_checkpoint": str(measurement_checkpoint),
        "feature_key": str(args.feature_key),
        "feature_path_template": str(args.feature_path_template),
        "k_per_query": int(args.k_per_query),
        "mutual_mode": str(mutual_mode),
        "anchor_side": str(args.anchor_side),
        "prediction_head": str(args.prediction_head),
        "measurement_batch_size": int(args.measurement_batch_size),
        "max_pairs": int(args.max_pairs) if int(args.max_pairs) > 0 else None,
    }
    Path(summary["outputs"]["summary"]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
