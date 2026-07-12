"""Audit full-map descriptor parity between landmark-bank building and query evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from feature_extract.vfm.localization.descriptor_path_audit import (
    audit_full_map_observation_projection_parity,
)
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--feature_key", default="radio_final")
    parser.add_argument("--sample_mode", default="bilinear", choices=("nearest", "bilinear"))
    parser.add_argument("--max_observations", type=int, default=64)
    parser.add_argument("--min_cosine", type=float, default=0.99999)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    requested = torch.device(str(args.device))
    device = "cpu" if requested.type == "cuda" and not torch.cuda.is_available() else str(requested)
    observations = load_colmap_track_observations_jsonl(Path(args.track_observations))
    manifest = TokenBankManifest.from_json(Path(args.token_manifest))
    manifest.validate(verify_checksums=False)
    joint_run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=device)
    metrics = audit_full_map_observation_projection_parity(
        observations,
        manifest,
        JointFeatureMapper(joint_run.model, device=device),
        feature_key=str(args.feature_key),
        sample_mode=str(args.sample_mode),
        max_observations=int(args.max_observations),
        min_cosine_threshold=float(args.min_cosine),
    )
    summary = {
        "stage": "projected_observation_descriptor_path_audit",
        "track_observations": str(args.track_observations),
        "token_manifest": str(args.token_manifest),
        "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
        "device": device,
        "metrics": metrics,
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
