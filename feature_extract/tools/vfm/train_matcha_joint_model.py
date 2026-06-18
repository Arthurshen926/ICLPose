"""Train the MATCHA-first joint model from a coarse-fine sample cache.

The current sample cache format contains coarse/fine correspondence rows. If
additional full-map or RGB detector tensors are needed, use the Python API in
``feature_extract.vfm.matcha_joint_training`` until a full joint-cache builder
is generated.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.matcha_coarse_fine_adapter import (
    load_matcha_coarse_fine_training_set_npz,
    save_matcha_coarse_fine_adapter,
)
from feature_extract.vfm.matcha_joint_training import (
    MatchaJointTrainingConfig,
    MatchaJointTrainingSet,
    joint_run_as_coarse_fine_adapter_run,
    load_matcha_joint_model,
    load_matcha_joint_training_set_manifest,
    load_matcha_joint_training_set_npz,
    save_matcha_joint_model,
    train_matcha_joint_model,
    train_matcha_joint_model_from_manifest,
)


def _sha(path: str) -> str:
    value = Path(path)
    return file_sha256_short(value) if value.exists() and value.is_file() else ""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_cache", default="")
    parser.add_argument("--joint_cache", default="")
    parser.add_argument("--joint_cache_manifest", default="")
    parser.add_argument("--lazy_manifest_training", action="store_true")
    parser.add_argument("--manifest_shard_cache_size", type=int, default=1)
    parser.add_argument("--manifest_steps_per_shard", type=int, default=1)
    parser.add_argument("--validation_joint_cache", default="")
    parser.add_argument("--validation_joint_cache_manifest", default="")
    parser.add_argument("--validation_interval", type=int, default=0)
    parser.add_argument("--warm_start_joint_checkpoint", default="")
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_joint_model", default="")
    parser.add_argument("--output_best_joint_model", default="")
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--model_type", choices=("residual_adapter", "radio_dual_attention"), default="residual_adapter")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--residual_hidden_dim", type=int, default=256)
    parser.add_argument("--fine_input_dim", type=int, default=0)
    parser.add_argument("--coarse_input_dim", type=int, default=0)
    parser.add_argument("--attention_hidden_dim", type=int, default=256)
    parser.add_argument("--attention_depth", type=int, default=2)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--attention_patch_size", type=int, default=2)
    parser.add_argument("--attention_upsample_mode", choices=("bilinear", "pixel_shuffle"), default="bilinear")
    parser.add_argument("--attention_fusion_mode", choices=("legacy", "matcha_original"), default="legacy")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dual_softmax_weight", type=float, default=1.0)
    parser.add_argument("--offset_loss_weight", type=float, default=0.25)
    parser.add_argument("--pair_fine_loss_weight", type=float, default=0.25)
    parser.add_argument("--query_pair_fine_loss_weight", type=float, default=0.0)
    parser.add_argument("--fine_continuous_loss_weight", type=float, default=0.25)
    parser.add_argument("--fine_uncertainty_loss_weight", type=float, default=0.05)
    parser.add_argument("--pair_confidence_loss_weight", type=float, default=0.1)
    parser.add_argument("--dense_heatmap_loss_weight", type=float, default=0.0)
    parser.add_argument("--rgb_keypoint_loss_weight", type=float, default=0.0)
    parser.add_argument("--rgb_keypoint_position_loss_weight", type=float, default=0.0)
    parser.add_argument("--repeatability_loss_weight", type=float, default=0.0)
    parser.add_argument("--local_fine_transformer_loss_weight", type=float, default=0.0)
    parser.add_argument("--local_window_fine_loss_weight", type=float, default=0.0)
    parser.add_argument("--local_window_fine_mode", choices=("mlp", "correlation"), default="mlp")
    parser.add_argument("--patch_correlation_loss_weight", type=float, default=0.0)
    parser.add_argument("--patch_correlation_window_size", type=int, default=3)
    parser.add_argument("--hard_negative_weight", type=float, default=0.2)
    parser.add_argument("--hard_negative_margin", type=float, default=0.2)
    parser.add_argument("--hard_false_match_weight", type=float, default=0.0)
    parser.add_argument("--hard_false_match_margin", type=float, default=0.2)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--input_norm_mode", choices=("identity", "layernorm"), default="identity")
    parser.add_argument("--gate_mode", choices=("residual", "sigmoid"), default="residual")
    parser.add_argument("--residual_gate_scale", type=float, default=0.1)
    parser.add_argument("--map_pair_batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    if not args.sample_cache and not args.joint_cache and not args.joint_cache_manifest:
        raise SystemExit("one of --sample_cache, --joint_cache, or --joint_cache_manifest is required")
    validation_samples = None
    validation_metadata = {}
    validation_cache_path = ""
    validation_manifest_path = None
    if args.validation_joint_cache_manifest:
        validation_cache_path = str(args.validation_joint_cache_manifest)
        if bool(args.lazy_manifest_training):
            validation_manifest_path = Path(args.validation_joint_cache_manifest)
            validation_metadata = json.loads(validation_manifest_path.read_text())
        else:
            validation_samples, validation_metadata = load_matcha_joint_training_set_manifest(Path(args.validation_joint_cache_manifest))
    elif args.validation_joint_cache:
        validation_samples, validation_metadata = load_matcha_joint_training_set_npz(Path(args.validation_joint_cache))
        validation_cache_path = str(args.validation_joint_cache)

    if args.joint_cache_manifest and bool(args.lazy_manifest_training):
        joint_samples = None
        samples = None
        metadata = {"format": "vfm_matcha_joint_training_manifest_v1", "lazy_training": True}
        sample_cache_path = str(args.joint_cache_manifest)
    elif args.joint_cache_manifest:
        joint_samples, metadata = load_matcha_joint_training_set_manifest(Path(args.joint_cache_manifest))
        samples = joint_samples.coarse_fine_samples
        sample_cache_path = str(args.joint_cache_manifest)
    elif args.joint_cache:
        joint_samples, metadata = load_matcha_joint_training_set_npz(Path(args.joint_cache))
        samples = joint_samples.coarse_fine_samples
        sample_cache_path = str(args.joint_cache)
    else:
        samples, metadata = load_matcha_coarse_fine_training_set_npz(Path(args.sample_cache))
        joint_samples = MatchaJointTrainingSet(coarse_fine_samples=samples)
        sample_cache_path = str(args.sample_cache)
    config = MatchaJointTrainingConfig(
        model_type=str(args.model_type),
        output_dim=int(args.output_dim),
        residual_hidden_dim=int(args.residual_hidden_dim),
        fine_input_dim=int(args.fine_input_dim),
        coarse_input_dim=int(args.coarse_input_dim),
        attention_hidden_dim=int(args.attention_hidden_dim),
        attention_depth=int(args.attention_depth),
        attention_heads=int(args.attention_heads),
        attention_patch_size=int(args.attention_patch_size),
        attention_upsample_mode=str(args.attention_upsample_mode),
        attention_fusion_mode=str(args.attention_fusion_mode),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        temperature=float(args.temperature),
        dual_softmax_weight=float(args.dual_softmax_weight),
        offset_loss_weight=float(args.offset_loss_weight),
        pair_fine_loss_weight=float(args.pair_fine_loss_weight),
        query_pair_fine_loss_weight=float(args.query_pair_fine_loss_weight),
        fine_continuous_loss_weight=float(args.fine_continuous_loss_weight),
        fine_uncertainty_loss_weight=float(args.fine_uncertainty_loss_weight),
        pair_confidence_loss_weight=float(args.pair_confidence_loss_weight),
        dense_heatmap_loss_weight=float(args.dense_heatmap_loss_weight),
        rgb_keypoint_loss_weight=float(args.rgb_keypoint_loss_weight),
        rgb_keypoint_position_loss_weight=float(args.rgb_keypoint_position_loss_weight),
        repeatability_loss_weight=float(args.repeatability_loss_weight),
        local_fine_transformer_loss_weight=float(args.local_fine_transformer_loss_weight),
        local_window_fine_loss_weight=float(args.local_window_fine_loss_weight),
        local_window_fine_mode=str(args.local_window_fine_mode),
        patch_correlation_loss_weight=float(args.patch_correlation_loss_weight),
        patch_correlation_window_size=int(args.patch_correlation_window_size),
        hard_negative_weight=float(args.hard_negative_weight),
        hard_negative_margin=float(args.hard_negative_margin),
        hard_false_match_weight=float(args.hard_false_match_weight),
        hard_false_match_margin=float(args.hard_false_match_margin),
        group_size=int(args.group_size),
        input_norm_mode=str(args.input_norm_mode),
        gate_mode=str(args.gate_mode),
        residual_gate_scale=float(args.residual_gate_scale),
        map_pair_batch_size=int(args.map_pair_batch_size),
        device=str(args.device),
        seed=int(args.seed),
    )
    warm_start_model = None
    if str(args.warm_start_joint_checkpoint):
        warm_start_model = load_matcha_joint_model(Path(args.warm_start_joint_checkpoint), device=str(args.device)).model
    if args.joint_cache_manifest and bool(args.lazy_manifest_training):
        run = train_matcha_joint_model_from_manifest(
            Path(args.joint_cache_manifest),
            config,
            validation_samples=validation_samples,
            validation_manifest_path=validation_manifest_path,
            validation_interval=int(args.validation_interval),
            shard_cache_size=int(args.manifest_shard_cache_size),
            steps_per_shard=int(args.manifest_steps_per_shard),
            warm_start_model=warm_start_model,
        )
    else:
        run = train_matcha_joint_model(
            joint_samples,
            config,
            validation_samples=validation_samples,
            validation_interval=int(args.validation_interval),
            warm_start_model=warm_start_model,
        )
    adapter_run = joint_run_as_coarse_fine_adapter_run(run)
    save_matcha_coarse_fine_adapter(adapter_run, Path(args.output_model))
    joint_output = Path(args.output_joint_model) if args.output_joint_model else Path(args.output_model).with_name(Path(args.output_model).stem + "_joint.pt")
    save_matcha_joint_model(run, joint_output)
    best_output = Path(args.output_best_joint_model) if args.output_best_joint_model else None
    if best_output is not None:
        save_matcha_joint_model(run, best_output)
    sample_count = int(run.summary.get("sample_count", samples.sample_count if samples is not None else 0))
    summary = {
        "stage": "matcha_style_joint_model_training",
        "elapsed_sec": float(time.perf_counter() - started),
        "sample_cache": {
            "path": sample_cache_path,
            "sha256": _sha(sample_cache_path),
            "metadata": metadata,
            "sample_count": int(sample_count),
        },
        "validation_cache": {
            "path": validation_cache_path,
            "sha256": _sha(validation_cache_path),
            "metadata": validation_metadata,
            "sample_count": int(validation_samples.coarse_fine_samples.sample_count)
            if validation_samples is not None
            else int(validation_metadata.get("sample_count", 0) if isinstance(validation_metadata, dict) else 0),
        },
        "config": asdict(config),
        "training": run.summary,
        "outputs": {
            "adapter_model": str(args.output_model),
            "joint_model": str(joint_output),
            "best_joint_model": str(best_output) if best_output is not None else "",
        },
        "notes": [
            "Use --joint_cache for full-map dense heatmap and RGB-local detector losses.",
            "Use --sample_cache for row-level coarse-fine adapter training only.",
        ],
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
