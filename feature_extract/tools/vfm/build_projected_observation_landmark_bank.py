"""Build projected-observation 3D landmark descriptors from a trained joint mapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.landmark_hybrid import (
    build_projected_observation_landmark_index,
    save_landmark_index_npz,
)
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.landmark_feature_aggregation import AGGREGATION_METHODS
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--feature_key", default="radio_final")
    parser.add_argument(
        "--method",
        default="mean",
        choices=tuple(sorted(AGGREGATION_METHODS)),
    )
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--missing", default="skip", choices=("skip", "error"))
    parser.add_argument(
        "--utility_mode",
        default="inverse_reprojection",
        choices=(
            "inverse_reprojection",
            "center",
            "inverse_reprojection_center",
            "view_consistency",
            "inverse_reprojection_center_view",
        ),
    )
    parser.add_argument("--sample_mode", default="bilinear", choices=("nearest", "bilinear"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trim_fraction", type=float, default=0.2)
    parser.add_argument("--view_consistent_keep", type=int, default=4)
    parser.add_argument("--geometric_median_iterations", type=int, default=32)
    parser.add_argument("--l2_normalize_observations", action="store_true")
    parser.add_argument("--weight_floor", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_index", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def _resolve_device(device: str) -> str:
    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = _resolve_device(str(args.device))
    observations = load_colmap_track_observations_jsonl(Path(args.track_observations))
    manifest = TokenBankManifest.from_json(Path(args.token_manifest))
    joint_run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=device)
    mapper = JointFeatureMapper(joint_run.model, device=device)
    index, metadata = build_projected_observation_landmark_index(
        observations,
        manifest,
        mapper,
        feature_key=str(args.feature_key),
        aggregation_method=str(args.method),
        min_observations=int(args.min_observations),
        missing=str(args.missing),
        utility_mode=str(args.utility_mode),
        weight_floor=float(args.weight_floor),
        sample_mode=str(args.sample_mode),
        seed=int(args.seed),
        trim_fraction=float(args.trim_fraction),
        view_consistent_keep=int(args.view_consistent_keep),
        geometric_median_iterations=int(args.geometric_median_iterations),
        l2_normalize_observations=bool(args.l2_normalize_observations),
    )
    output_index = Path(args.output_index)
    output_metadata = {
        **metadata,
        "track_observations": str(args.track_observations),
        "token_manifest": str(args.token_manifest),
        "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
        "device": device,
    }
    save_landmark_index_npz(index, output_index, metadata=output_metadata)
    summary = {
        "stage": "projected_observation_3d_landmark_feature_aggregation",
        "projection_mode": "full_map_projected_observations",
        "input_files": {
            "track_observations": {
                "path": str(args.track_observations),
                "sha256": file_sha256_short(Path(args.track_observations)),
            },
            "token_manifest": {
                "path": str(args.token_manifest),
                "sha256": file_sha256_short(Path(args.token_manifest)),
            },
            "matcha_joint_checkpoint": {
                "path": str(args.matcha_joint_checkpoint),
                "sha256": file_sha256_short(Path(args.matcha_joint_checkpoint)),
            },
        },
        "output_files": {
            "projected_landmark_index": str(output_index),
        },
        "metadata": output_metadata,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
