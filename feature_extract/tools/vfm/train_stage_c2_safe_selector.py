"""Train Stage C2 safe localizable descriptor selector and export descriptors."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_c0_compressed_features import (
    _dir_bytes,
    _layer_spec,
    _safe_token_filename,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.patch_selector_training import (
    SafePatchSelectorTrainingConfig,
    encode_rows_with_safe_selector,
    load_patch_selector_training_set_npz,
    save_safe_patch_selector_checkpoint,
    train_safe_patch_selector,
)
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _path_sha_or_empty(path: str) -> str:
    value = Path(path)
    return file_sha256_short(value) if value.exists() else ""


def _encode_feature_map(
    feature_map: np.ndarray,
    run,
    device: str,
    batch_tokens: int,
) -> np.ndarray:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = values.shape
    rows = values.reshape(channels, height * width).T
    encoded = run.encode_rows(rows, device=device, batch_size=batch_tokens)
    return encoded.T.reshape(run.summary.output_dim, height, width).astype(np.float32, copy=False)


def _write_safe_query_manifest(
    source_manifest: TokenBankManifest,
    run,
    layer_name: str,
    output_query_dir: Path,
    output_layer_name: str,
    device: str,
    batch_tokens: int,
) -> tuple[TokenBankManifest, int]:
    records = []
    output_query_dir.mkdir(parents=True, exist_ok=True)
    for record in source_manifest.records:
        spec = _layer_spec(record, layer_name)
        with np.load(record.token_path) as data:
            if layer_name not in data:
                raise ValueError(f"layer {layer_name!r} not found in {record.token_path}")
            feature_map = np.asarray(data[layer_name], dtype=np.float32)
        encoded = _encode_feature_map(feature_map, run, device=device, batch_tokens=batch_tokens)
        output_path = output_query_dir / _safe_token_filename(record.image_id)
        np.savez_compressed(output_path, **{output_layer_name: encoded})
        records.append(
            TokenBankRecord(
                image_id=record.image_id,
                token_path=output_path,
                layers=(
                    TokenLayerSpec(
                        name=output_layer_name,
                        model=f"{spec.model}:stage_c2_safe_selector",
                        layer=spec.layer,
                        channels=int(run.summary.output_dim),
                        stride=int(spec.stride),
                    ),
                ),
                split=record.split,
                scene=record.scene,
                metadata={
                    **dict(record.metadata),
                    "stage": "stage_c2_safe_selector",
                    "source_token_path": str(record.token_path),
                    "source_layer_name": layer_name,
                    "selector_output_dim": int(run.summary.output_dim),
                    "active_group_count": int(run.summary.active_group_count),
                    "group_count": int(run.summary.group_count),
                },
            )
        )
    return TokenBankManifest(records=tuple(records)), len(records)


def _transform_track_bank_with_safe_selector(
    source_bank: Path,
    output_bank: Path,
    run,
    device: str,
    batch_rows: int,
) -> dict[str, float | int | str]:
    with np.load(Path(source_bank)) as data:
        track_ids = data["track_ids"].astype(np.int64)
        mean_features = data["mean_features"].astype(np.float32)
        observation_counts = data["observation_counts"].astype(np.int64)
        mean_utilities = data["mean_utilities"].astype(np.float32)
        observation_image_ids = (
            data["observation_image_ids"]
            if "observation_image_ids" in data
            else np.asarray([""] * int(track_ids.shape[0]), dtype=str)
        )
    encoded_means = run.encode_rows(mean_features, device=device, batch_size=batch_rows)
    encoded_variances = np.zeros_like(encoded_means, dtype=np.float32)
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_bank,
        track_ids=track_ids,
        mean_features=encoded_means.astype(np.float32, copy=False),
        variances=encoded_variances.astype(np.float32, copy=False),
        observation_counts=observation_counts,
        mean_utilities=mean_utilities,
        observation_image_ids=observation_image_ids,
        feature_dim=np.asarray(run.summary.output_dim, dtype=np.int64),
    )
    return {
        "track_count": int(track_ids.shape[0]),
        "feature_dim": int(run.summary.output_dim),
        "mean_observation_count": float(np.mean(observation_counts)) if observation_counts.size else 0.0,
        "mean_track_variance": float(np.mean(encoded_variances)) if encoded_variances.size else 0.0,
        "mean_utility": float(np.mean(mean_utilities)) if mean_utilities.size else 0.0,
        "variance_policy": "zero_for_nonlinear_descriptor_export",
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train Stage C2 safe localizable descriptor selector")
    parser.add_argument("--sample_cache", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--output_layer_name", default="")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--residual_hidden_dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--inlier_loss_weight", type=float, default=0.2)
    parser.add_argument("--group_lasso_weight", type=float, default=0.0)
    parser.add_argument("--eval_split_fraction", type=float, default=0.1)
    parser.add_argument("--center_inputs", action="store_true")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--hard_gate_keep_fraction", type=float, default=1.0)
    parser.add_argument("--hard_gate_min_groups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_query_dir", default="")
    parser.add_argument("--output_query_manifest", default="")
    parser.add_argument("--output_landmark_bank", default="")
    parser.add_argument("--export_device", default="")
    parser.add_argument("--batch_rows", type=int, default=65536)
    parser.add_argument("--batch_tokens", type=int, default=65536)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    samples, sample_metadata = load_patch_selector_training_set_npz(Path(args.sample_cache))
    run = train_safe_patch_selector(
        samples,
        SafePatchSelectorTrainingConfig(
            output_dim=int(args.output_dim),
            residual_hidden_dim=int(args.residual_hidden_dim),
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            temperature=float(args.temperature),
            inlier_loss_weight=float(args.inlier_loss_weight),
            group_lasso_weight=float(args.group_lasso_weight),
            seed=int(args.seed),
            device=str(args.device),
            eval_split_fraction=float(args.eval_split_fraction),
            center_inputs=bool(args.center_inputs),
            group_size=int(args.group_size),
            hard_gate_keep_fraction=float(args.hard_gate_keep_fraction),
            hard_gate_min_groups=int(args.hard_gate_min_groups),
        ),
    )
    save_safe_patch_selector_checkpoint(run, Path(args.output_model))

    export_summary: dict[str, object] = {}
    export_args = [args.output_query_dir, args.output_query_manifest, args.output_landmark_bank]
    if any(export_args):
        if not all(export_args):
            raise ValueError("--output_query_dir, --output_query_manifest and --output_landmark_bank must be provided together")
        manifest = TokenBankManifest.from_json(Path(args.query_manifest))
        manifest.validate(verify_checksums=False)
        export_device = args.export_device or args.device
        mapability = _transform_track_bank_with_safe_selector(
            Path(args.landmark_bank),
            Path(args.output_landmark_bank),
            run,
            device=export_device,
            batch_rows=int(args.batch_rows),
        )
        output_layer_name = args.output_layer_name or args.layer_name
        compressed_manifest, query_count = _write_safe_query_manifest(
            manifest,
            run,
            layer_name=args.layer_name,
            output_query_dir=Path(args.output_query_dir),
            output_layer_name=output_layer_name,
            device=export_device,
            batch_tokens=int(args.batch_tokens),
        )
        compressed_manifest.to_json(Path(args.output_query_manifest))
        export_summary = {
            "query_record_count": int(query_count),
            "mapability": mapability,
            "device": export_device,
            "storage_bytes": {
                "query_tokens": _dir_bytes(Path(args.output_query_dir)),
                "landmark_bank": _dir_bytes(Path(args.output_landmark_bank)),
                "model": _dir_bytes(Path(args.output_model)),
            },
            "outputs": {
                "query_manifest": str(args.output_query_manifest),
                "query_dir": str(args.output_query_dir),
                "landmark_bank": str(args.output_landmark_bank),
                "model": str(args.output_model),
            },
        }

    summary = {
        "stage": "stage_c2_safe_localizable_descriptor_selection",
        "elapsed_sec": float(time.perf_counter() - started),
        "sample_summary": {
            "sample_cache": str(args.sample_cache),
            "sample_metadata": sample_metadata,
            "sample_count": int(samples.sample_count),
        },
        "training": asdict(run.summary),
        "active_group_mask": [float(value) for value in run.active_group_mask.tolist()],
        "inputs": {
            "sample_cache": {
                "path": str(args.sample_cache),
                "sha256": _path_sha_or_empty(args.sample_cache),
            },
            "query_manifest": {
                "path": str(args.query_manifest),
                "sha256": _path_sha_or_empty(args.query_manifest),
            },
            "landmark_bank": {
                "path": str(args.landmark_bank),
                "sha256": _path_sha_or_empty(args.landmark_bank),
            },
        },
        "export": export_summary,
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
