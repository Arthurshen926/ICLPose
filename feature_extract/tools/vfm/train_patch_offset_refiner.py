#!/usr/bin/env python3
"""Train a scene-specific patch-to-pixel offset refiner."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.patch_offset_refiner import (
    HeatmapPatchOffsetRefinerConfig,
    PatchOffsetRefinerConfig,
    build_patch_offset_samples_from_rows,
    save_patch_offset_refiner_checkpoint,
    train_heatmap_patch_offset_refiner,
    train_patch_offset_refiner,
)


def _path_sha_or_empty(path: str) -> str:
    value = Path(path)
    return file_sha256_short(value) if value.exists() else ""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_jsonl", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--positive_stride", type=float, default=1.0)
    parser.add_argument("--negative_stride", type=float, default=2.0)
    parser.add_argument("--max_target_offset_stride", type=float, default=1.0)
    parser.add_argument("--require_pnp_inlier", action="store_true")
    parser.add_argument(
        "--positive_label_mode",
        default="default",
        choices=("default", "patch", "bounded", "patch_or_bounded", "patch_and_bounded"),
    )
    parser.add_argument("--bounded_positive_stride", type=float, default=0.5)
    parser.add_argument("--positive_sample_weight", type=float, default=1.0)
    parser.add_argument("--negative_sample_weight", type=float, default=1.0)
    parser.add_argument("--model_type", default="regression", choices=("regression", "heatmap"))
    parser.add_argument("--heatmap_bin_count", type=int, default=8)
    parser.add_argument("--heatmap_residual_loss_weight", type=float, default=0.25)
    parser.add_argument("--heatmap_bin_loss_weight", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_offset_stride", type=float, default=0.5)
    parser.add_argument("--confidence_loss_weight", type=float, default=0.5)
    parser.add_argument("--anchor_zero_weight", type=float, default=0.0)
    parser.add_argument("--eval_split_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    pose_by_query = {record.image_id: record.pose_w2c for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    samples, sample_summary = build_patch_offset_samples_from_rows(
        match_jsonl=Path(args.match_jsonl),
        query_manifest=Path(args.query_manifest),
        landmark_bank=Path(args.landmark_bank),
        pose_by_query=pose_by_query,
        camera=camera,
        layer_name=str(args.layer_name),
        window_size=int(args.window_size),
        max_samples=int(args.max_samples),
        positive_stride=float(args.positive_stride),
        negative_stride=float(args.negative_stride),
        max_target_offset_stride=float(args.max_target_offset_stride),
        require_pnp_inlier=bool(args.require_pnp_inlier),
        positive_label_mode=str(args.positive_label_mode),
        bounded_positive_stride=float(args.bounded_positive_stride),
        positive_sample_weight=float(args.positive_sample_weight),
        negative_sample_weight=float(args.negative_sample_weight),
        heatmap_bin_count=int(args.heatmap_bin_count),
        seed=int(args.seed),
    )
    base_kwargs = {
        "feature_dim": int(samples.feature_dim),
        "stats_dim": int(samples.stats_dim),
        "window_size": int(samples.window_size),
        "hidden_dim": int(args.hidden_dim),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "max_offset_stride": float(args.max_offset_stride),
        "confidence_loss_weight": float(args.confidence_loss_weight),
        "anchor_zero_weight": float(args.anchor_zero_weight),
        "eval_split_fraction": float(args.eval_split_fraction),
        "seed": int(args.seed),
        "device": str(args.device),
    }
    if args.model_type == "heatmap":
        run = train_heatmap_patch_offset_refiner(
            samples,
            HeatmapPatchOffsetRefinerConfig(
                **base_kwargs,
                bin_count=int(args.heatmap_bin_count),
                residual_loss_weight=float(args.heatmap_residual_loss_weight),
                bin_loss_weight=float(args.heatmap_bin_loss_weight),
            ),
        )
    else:
        run = train_patch_offset_refiner(samples, PatchOffsetRefinerConfig(**base_kwargs))
    save_patch_offset_refiner_checkpoint(run, args.output_model)
    summary = {
        "stage": "patch_offset_refiner_training",
        "elapsed_sec": float(time.perf_counter() - started),
        "inputs": {
            "match_jsonl": {"path": args.match_jsonl, "sha256": _path_sha_or_empty(args.match_jsonl)},
            "query_manifest": {"path": args.query_manifest, "sha256": _path_sha_or_empty(args.query_manifest)},
            "landmark_bank": {"path": args.landmark_bank, "sha256": _path_sha_or_empty(args.landmark_bank)},
            "query_pose_file": {"path": args.query_pose_file, "sha256": _path_sha_or_empty(args.query_pose_file)},
            "camera_source": camera_source,
        },
        "sample_summary": sample_summary,
        "training": asdict(run.summary),
        "model_type": args.model_type,
        "model": args.output_model,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.summary_json}")


if __name__ == "__main__":
    main()
