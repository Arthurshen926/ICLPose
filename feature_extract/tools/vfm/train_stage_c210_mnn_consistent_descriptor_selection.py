#!/usr/bin/env python3
"""Train Stage C2.10 MNN-consistent pose-aware descriptor selection."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_c0_compressed_features import _dir_bytes, _layer_spec, _safe_token_filename
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import CalibratedLogisticConfidence
from feature_extract.vfm.mnn_consistent_descriptor_selection import (
    MNNConsistentDescriptorSelectionConfig,
    build_mnn_consistent_descriptor_samples,
    save_mnn_consistent_selector_checkpoint,
    train_mnn_consistent_descriptor_selector,
)
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _path_sha_or_empty(path: str) -> str:
    value = Path(path)
    return file_sha256_short(value) if value.exists() else ""


def _encode_feature_map(feature_map: np.ndarray, run, device: str, batch_tokens: int) -> np.ndarray:
    values = np.asarray(feature_map, dtype=np.float32)
    channels, height, width = values.shape
    rows = values.reshape(channels, height * width).T
    encoded = run.encode_rows(rows, device=device, batch_size=batch_tokens)
    return encoded.T.reshape(run.summary.output_dim, height, width).astype(np.float32, copy=False)


def _write_query_manifest(
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
                        model=f"{spec.model}:stage_c210_mnn_consistent_selector",
                        layer=spec.layer,
                        channels=int(run.summary.output_dim),
                        stride=int(spec.stride),
                    ),
                ),
                split=record.split,
                scene=record.scene,
                metadata={
                    **dict(record.metadata),
                    "stage": "stage_c210_mnn_consistent_pose_aware_descriptor_selection",
                    "source_token_path": str(record.token_path),
                    "source_layer_name": layer_name,
                    "output_dim": int(run.summary.output_dim),
                },
            )
        )
    return TokenBankManifest(records=tuple(records)), len(records)


def _transform_track_bank(source_bank: Path, output_bank: Path, run, device: str, batch_rows: int) -> dict[str, object]:
    with np.load(source_bank) as data:
        track_ids = data["track_ids"].astype(np.int64)
        mean_features = data["mean_features"].astype(np.float32)
        variances = data["variances"].astype(np.float32) if "variances" in data else np.zeros_like(mean_features, dtype=np.float32)
        observation_counts = data["observation_counts"].astype(np.int64)
        mean_utilities = data["mean_utilities"].astype(np.float32)
        observation_image_ids = (
            data["observation_image_ids"]
            if "observation_image_ids" in data
            else np.asarray([""] * int(track_ids.shape[0]), dtype=str)
        )
    encoded = run.encode_rows(mean_features, device=device, batch_size=batch_rows)
    if encoded.shape[1] == variances.shape[1]:
        encoded_variances = variances.astype(np.float32, copy=True)
        variance_policy = "copied_input_variance_same_dim_proxy"
    else:
        mean_variance = np.mean(variances, axis=1, keepdims=True) if variances.size else np.zeros((encoded.shape[0], 1), dtype=np.float32)
        encoded_variances = np.repeat(mean_variance.astype(np.float32), encoded.shape[1], axis=1)
        variance_policy = "mean_input_variance_repeated_proxy"
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_bank,
        track_ids=track_ids,
        mean_features=encoded.astype(np.float32, copy=False),
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
        "variance_policy": variance_policy,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_jsonl", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--output_layer_name", default="")
    parser.add_argument("--teacher_model", default="")
    parser.add_argument("--teacher_feature_set", default="full")
    parser.add_argument("--max_tokens_per_query", type=int, default=64)
    parser.add_argument("--max_landmarks_per_query", type=int, default=256)
    parser.add_argument("--max_candidates_per_token", type=int, default=16)
    parser.add_argument("--max_query_groups", type=int, default=0)
    parser.add_argument("--positive_stride", type=float, default=1.0)
    parser.add_argument("--negative_stride", type=float, default=2.0)
    parser.add_argument("--soft_label_sigma_stride", type=float, default=1.0)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dual_loss_weight", type=float, default=1.0)
    parser.add_argument("--soft_reproj_loss_weight", type=float, default=0.5)
    parser.add_argument("--wrong_pose_loss_weight", type=float, default=0.1)
    parser.add_argument("--teacher_distill_loss_weight", type=float, default=0.2)
    parser.add_argument("--anchor_loss_weight", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--teacher_temperature", type=float, default=0.2)
    parser.add_argument("--eval_split_fraction", type=float, default=0.1)
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
    teacher = CalibratedLogisticConfidence.load_json(args.teacher_model) if args.teacher_model else None
    samples, sample_summary = build_mnn_consistent_descriptor_samples(
        match_jsonl=Path(args.match_jsonl),
        query_manifest=Path(args.query_manifest),
        landmark_bank=Path(args.landmark_bank),
        layer_name=str(args.layer_name),
        max_tokens_per_query=int(args.max_tokens_per_query),
        max_landmarks_per_query=int(args.max_landmarks_per_query),
        max_candidates_per_token=int(args.max_candidates_per_token),
        max_query_groups=int(args.max_query_groups),
        positive_stride=float(args.positive_stride),
        negative_stride=float(args.negative_stride),
        soft_label_sigma_stride=float(args.soft_label_sigma_stride),
        teacher_model=teacher,
        teacher_feature_set=str(args.teacher_feature_set),
        seed=int(args.seed),
    )
    run = train_mnn_consistent_descriptor_selector(
        samples,
        MNNConsistentDescriptorSelectionConfig(
            output_dim=int(args.output_dim),
            hidden_dim=int(args.hidden_dim),
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            temperature=float(args.temperature),
            dual_loss_weight=float(args.dual_loss_weight),
            soft_reproj_loss_weight=float(args.soft_reproj_loss_weight),
            wrong_pose_loss_weight=float(args.wrong_pose_loss_weight),
            teacher_distill_loss_weight=float(args.teacher_distill_loss_weight),
            anchor_loss_weight=float(args.anchor_loss_weight),
            margin=float(args.margin),
            teacher_temperature=float(args.teacher_temperature),
            eval_split_fraction=float(args.eval_split_fraction),
            seed=int(args.seed),
            device=str(args.device),
        ),
    )
    save_mnn_consistent_selector_checkpoint(run, args.output_model)

    export_summary: dict[str, object] = {}
    export_args = [args.output_query_dir, args.output_query_manifest, args.output_landmark_bank]
    if any(export_args):
        if not all(export_args):
            raise ValueError("--output_query_dir, --output_query_manifest and --output_landmark_bank must be provided together")
        manifest = TokenBankManifest.from_json(Path(args.query_manifest))
        manifest.validate(verify_checksums=False)
        export_device = args.export_device or args.device
        output_layer = args.output_layer_name or args.layer_name
        mapability = _transform_track_bank(Path(args.landmark_bank), Path(args.output_landmark_bank), run, export_device, int(args.batch_rows))
        output_manifest, query_count = _write_query_manifest(
            manifest,
            run,
            layer_name=args.layer_name,
            output_query_dir=Path(args.output_query_dir),
            output_layer_name=output_layer,
            device=export_device,
            batch_tokens=int(args.batch_tokens),
        )
        output_manifest.to_json(Path(args.output_query_manifest))
        export_summary = {
            "query_record_count": int(query_count),
            "mapability": mapability,
            "outputs": {
                "query_dir": args.output_query_dir,
                "query_manifest": args.output_query_manifest,
                "landmark_bank": args.output_landmark_bank,
            },
            "storage_bytes": {
                "query_tokens": _dir_bytes(Path(args.output_query_dir)),
                "query_manifest": Path(args.output_query_manifest).stat().st_size,
                "landmark_bank": Path(args.output_landmark_bank).stat().st_size,
            },
        }

    summary = {
        "stage": "stage_c210_mnn_consistent_pose_aware_descriptor_selection",
        "elapsed_sec": float(time.perf_counter() - started),
        "inputs": {
            "match_jsonl": {"path": args.match_jsonl, "sha256": _path_sha_or_empty(args.match_jsonl)},
            "query_manifest": {"path": args.query_manifest, "sha256": _path_sha_or_empty(args.query_manifest)},
            "landmark_bank": {"path": args.landmark_bank, "sha256": _path_sha_or_empty(args.landmark_bank)},
            "teacher_model": {"path": args.teacher_model, "sha256": _path_sha_or_empty(args.teacher_model)} if args.teacher_model else None,
        },
        "sample_summary": sample_summary,
        "training": asdict(run.summary),
        "model": args.output_model,
        "export": export_summary,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.summary_json}")


if __name__ == "__main__":
    main()
