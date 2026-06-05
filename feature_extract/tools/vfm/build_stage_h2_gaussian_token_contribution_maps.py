"""Export approximate Gaussian token-contribution maps for Stage H2 raw aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import (
    _load_camera_by_image,
    _load_feature,
    _parse_default_camera,
    _safe_image_stem,
    _select_records,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_raw_landmarks import (
    GaussianTokenContributionConfig,
    build_gaussian_token_contribution_view,
)
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView, load_gaussian_vfm_source_from_ply
from feature_extract.vfm.tokens import TokenBankManifest


def _save_contribution_npz(path: Path, contribution) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    top_alpha = contribution.top_alpha
    alpha_entropy = contribution.alpha_entropy
    arrays = {"top_contributor": np.asarray(contribution.top_contributor, dtype=np.int64)}
    if top_alpha is not None:
        arrays["top_alpha"] = np.asarray(top_alpha, dtype=np.float32)
    if alpha_entropy is not None:
        arrays["alpha_entropy"] = np.asarray(alpha_entropy, dtype=np.float32)
    np.savez_compressed(path, **arrays)
    valid = arrays["top_contributor"] >= 0
    return {
        "path": str(path),
        "valid_token_count": int(np.sum(valid)),
        "valid_token_fraction": float(np.mean(valid)),
        "mean_top_alpha": None if top_alpha is None or not np.any(valid) else float(np.mean(top_alpha[valid])),
        "mean_alpha_entropy": (
            None if alpha_entropy is None or not np.any(valid) else float(np.mean(alpha_entropy[valid]))
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Gaussian token contribution maps for Stage H2")
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--view_selection", default="uniform", choices=("prefix", "uniform"))
    parser.add_argument("--max_gaussians", type=int, default=0)
    parser.add_argument("--radius_px", type=float, default=1.5)
    parser.add_argument("--depth_epsilon", type=float, default=0.02)
    parser.add_argument("--opacity_threshold", type=float, default=0.05)
    parser.add_argument("--view_angle_power", type=float, default=0.0)
    parser.add_argument("--default_camera", default="2,1024,576,883,512,288,0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.reference_manifest))
    manifest.validate(verify_checksums=False)
    pose_by_image = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.reference_pose_file))}
    camera_by_image = _load_camera_by_image(args.camera_model_dir)
    fallback_camera = _parse_default_camera(args.default_camera)
    records = [record for record in manifest.records if record.image_id in pose_by_image]
    records = _select_records(records, int(args.max_views), args.view_selection)
    if not records:
        raise ValueError("no reference records with both tokens and poses")
    source = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply), max_gaussians=int(args.max_gaussians))
    config = GaussianTokenContributionConfig(
        radius_px=float(args.radius_px),
        depth_epsilon=float(args.depth_epsilon),
        opacity_threshold=float(args.opacity_threshold),
        view_angle_power=float(args.view_angle_power),
    )
    output_dir = Path(args.output_dir)
    rows = []
    for record in records:
        view = GaussianVFMFeatureView(
            image_id=record.image_id,
            feature_map=_load_feature(Path(record.token_path), args.layer_name),
            pose_w2c=pose_by_image[record.image_id].pose_w2c,
            camera=camera_by_image.get(record.image_id, fallback_camera),
        )
        contribution = build_gaussian_token_contribution_view(source, view, config)
        row = _save_contribution_npz(output_dir / f"{_safe_image_stem(record.image_id)}.npz", contribution)
        row["image_id"] = str(record.image_id)
        rows.append(row)
    summary = {
        "stage": "stage_h2_gaussian_token_contribution_maps",
        "source_gaussian_count": int(source.xyz.shape[0]),
        "view_count": int(len(rows)),
        "config": config.to_dict(),
        "mean_valid_token_fraction": float(np.mean([row["valid_token_fraction"] for row in rows])),
        "mean_top_alpha": float(
            np.mean([row["mean_top_alpha"] for row in rows if row["mean_top_alpha"] is not None])
        )
        if any(row["mean_top_alpha"] is not None for row in rows)
        else None,
        "mean_alpha_entropy": float(
            np.mean([row["mean_alpha_entropy"] for row in rows if row["mean_alpha_entropy"] is not None])
        )
        if any(row["mean_alpha_entropy"] is not None for row in rows)
        else None,
        "inputs": {
            "gaussian_ply": str(args.gaussian_ply),
            "reference_manifest": str(args.reference_manifest),
            "reference_pose_file": str(args.reference_pose_file),
            "camera_model_dir": str(args.camera_model_dir),
            "layer_name": str(args.layer_name),
            "view_selection": str(args.view_selection),
        },
        "outputs": {"output_dir": str(output_dir)},
        "views": rows,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
