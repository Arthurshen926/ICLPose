"""Build Stage C0 non-learned compressed query and 3D landmark features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.feature_compression import (
    FeatureCompressionTransform,
    fit_feature_compression,
    slice_feature_compression,
)
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _layer_spec(record: TokenBankRecord, layer_name: str) -> TokenLayerSpec:
    for layer in record.layers:
        if layer.name == layer_name:
            return layer
    raise ValueError(f"layer {layer_name!r} not found in manifest record {record.image_id}")


def _safe_token_filename(image_id: str) -> str:
    return str(Path(str(image_id)).with_suffix("")).replace("/", "__").replace("\\", "__") + ".npz"


def _load_bank_feature_matrix(bank_path: Path) -> np.ndarray:
    with np.load(Path(bank_path)) as data:
        return np.asarray(data["mean_features"], dtype=np.float32)


def _apply_rows_batched(
    rows: np.ndarray,
    transform: FeatureCompressionTransform,
    device: str,
    batch_size: int,
    variances: bool = False,
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != int(transform.input_dim):
        raise ValueError("rows must have shape (N, input_dim)")
    if device == "cpu" or transform.selected_channels is not None:
        return transform.apply_variances(values) if variances else transform.apply_rows(values)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch_device = torch.device(device)
    matrix_np = np.square(transform.matrix) if variances else transform.matrix
    if matrix_np is None:
        return values.copy()
    matrix = torch.from_numpy(matrix_np.astype(np.float32, copy=False)).to(torch_device)
    mean = None if variances else torch.from_numpy(transform.mean.astype(np.float32, copy=False)).to(torch_device)
    chunks = []
    with torch.no_grad():
        for start in range(0, values.shape[0], int(batch_size)):
            batch = torch.from_numpy(values[start : start + int(batch_size)]).to(torch_device)
            if mean is not None:
                batch = batch - mean.view(1, -1)
            projected = batch @ matrix
            if transform.l2_normalize and not variances:
                projected = torch.nn.functional.normalize(projected, dim=1, eps=1e-8)
            chunks.append(projected.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, transform.output_dim), dtype=np.float32)


def _apply_channel_first_batched(
    feature_map: np.ndarray,
    transform: FeatureCompressionTransform,
    device: str,
    batch_tokens: int,
) -> np.ndarray:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != int(transform.input_dim):
        raise ValueError("feature_map must have shape (input_dim, H, W)")
    channels, height, width = values.shape
    rows = values.reshape(channels, height * width).T
    transformed = _apply_rows_batched(rows, transform, device=device, batch_size=batch_tokens, variances=False)
    return transformed.T.reshape(transform.output_dim, height, width).astype(np.float32, copy=False)


def _transform_track_bank_npz(
    source_bank: Path,
    output_bank: Path,
    transform: FeatureCompressionTransform,
    device: str,
    batch_rows: int,
) -> dict[str, float | int]:
    with np.load(Path(source_bank)) as data:
        track_ids = data["track_ids"].astype(np.int64)
        mean_features = data["mean_features"].astype(np.float32)
        variances = data["variances"].astype(np.float32)
        observation_counts = data["observation_counts"].astype(np.int64)
        mean_utilities = data["mean_utilities"].astype(np.float32)
        observation_image_ids = (
            data["observation_image_ids"]
            if "observation_image_ids" in data
            else np.asarray([""] * int(track_ids.shape[0]), dtype=str)
        )
    compressed_means = _apply_rows_batched(
        mean_features,
        transform,
        device=device,
        batch_size=batch_rows,
        variances=False,
    )
    compressed_variances = _apply_rows_batched(
        variances,
        transform,
        device=device,
        batch_size=batch_rows,
        variances=True,
    )
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_bank,
        track_ids=track_ids,
        mean_features=compressed_means.astype(np.float32, copy=False),
        variances=compressed_variances.astype(np.float32, copy=False),
        observation_counts=observation_counts,
        mean_utilities=mean_utilities,
        observation_image_ids=observation_image_ids,
        feature_dim=np.asarray(transform.output_dim, dtype=np.int64),
    )
    return {
        "track_count": int(track_ids.shape[0]),
        "feature_dim": int(transform.output_dim),
        "mean_observation_count": float(np.mean(observation_counts)) if observation_counts.size else 0.0,
        "mean_track_variance": float(np.mean(compressed_variances)) if compressed_variances.size else 0.0,
        "mean_utility": float(np.mean(mean_utilities)) if mean_utilities.size else 0.0,
    }


def _write_compressed_query_manifest(
    source_manifest: TokenBankManifest,
    transform: FeatureCompressionTransform,
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
        compressed = _apply_channel_first_batched(feature_map, transform, device=device, batch_tokens=batch_tokens)
        output_path = output_query_dir / _safe_token_filename(record.image_id)
        np.savez_compressed(output_path, **{output_layer_name: compressed})
        records.append(
            TokenBankRecord(
                image_id=record.image_id,
                token_path=output_path,
                layers=(
                    TokenLayerSpec(
                        name=output_layer_name,
                        model=f"{spec.model}:stage_c0_{transform.method}",
                        layer=spec.layer,
                        channels=int(transform.output_dim),
                        stride=int(spec.stride),
                    ),
                ),
                split=record.split,
                scene=record.scene,
                metadata={
                    **dict(record.metadata),
                    "stage": "stage_c0_compression",
                    "source_token_path": str(record.token_path),
                    "source_layer_name": layer_name,
                    "compression_method": transform.method,
                    "compression_output_dim": int(transform.output_dim),
                },
            )
        )
    return TokenBankManifest(records=tuple(records)), len(records)


def _dir_bytes(path: Path) -> int:
    if not Path(path).exists():
        return 0
    if Path(path).is_file():
        return int(Path(path).stat().st_size)
    return int(sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file()))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build Stage C0 compressed VFM query and landmark features")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--output_layer_name", default="")
    parser.add_argument(
        "--method",
        required=True,
        choices=("identity", "pca", "random", "first_channels", "channel_variance", "idf", "fisher"),
    )
    parser.add_argument("--output_dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_fit_samples", type=int, default=30000)
    parser.add_argument("--idf_threshold", type=float, default=0.0)
    parser.add_argument("--fit_labels_npy", default="")
    parser.add_argument("--l2_normalize", action="store_true")
    parser.add_argument("--input_transform", default="")
    parser.add_argument("--output_query_dir", required=True)
    parser.add_argument("--output_query_manifest", required=True)
    parser.add_argument("--output_landmark_bank", required=True)
    parser.add_argument("--output_transform", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_rows", type=int, default=65536)
    parser.add_argument("--batch_tokens", type=int, default=65536)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    landmark_features = _load_bank_feature_matrix(Path(args.landmark_bank))
    if landmark_features.ndim != 2:
        raise ValueError("landmark bank mean_features must have shape (N, C)")
    output_dim = int(args.output_dim or landmark_features.shape[1])
    if args.input_transform:
        transform = FeatureCompressionTransform.from_npz(Path(args.input_transform))
        if args.output_dim and int(args.output_dim) != int(transform.output_dim):
            transform = slice_feature_compression(transform, int(args.output_dim))
    else:
        labels = None
        if args.fit_labels_npy:
            labels = np.load(Path(args.fit_labels_npy))
        transform = fit_feature_compression(
            landmark_features,
            method=args.method,
            output_dim=output_dim,
            seed=args.seed,
            max_fit_samples=args.max_fit_samples,
            labels=labels,
            idf_threshold=args.idf_threshold,
            l2_normalize=bool(args.l2_normalize),
        )
    transform.to_npz(
        Path(args.output_transform),
        metadata={
            "source_landmark_bank": str(args.landmark_bank),
            "fit_source": "landmark_bank_mean_features",
            "max_fit_samples": int(args.max_fit_samples),
            "seed": int(args.seed),
            "fit_labels_npy": args.fit_labels_npy,
        },
    )

    mapability = _transform_track_bank_npz(
        Path(args.landmark_bank),
        Path(args.output_landmark_bank),
        transform,
        device=args.device,
        batch_rows=int(args.batch_rows),
    )

    output_layer_name = args.output_layer_name or args.layer_name
    compressed_manifest, query_count = _write_compressed_query_manifest(
        manifest,
        transform,
        layer_name=args.layer_name,
        output_query_dir=Path(args.output_query_dir),
        output_layer_name=output_layer_name,
        device=args.device,
        batch_tokens=int(args.batch_tokens),
    )
    compressed_manifest.to_json(Path(args.output_query_manifest))

    elapsed = float(time.perf_counter() - started)
    summary = dict(mapability)
    summary.update(
        {
            "stage": "stage_c0_nonlearned_feature_compression",
            "method": transform.method,
            "input_dim": int(transform.input_dim),
            "output_dim": int(transform.output_dim),
            "seed": int(args.seed),
            "max_fit_samples": int(args.max_fit_samples),
            "idf_threshold": float(args.idf_threshold),
            "l2_normalize": bool(transform.l2_normalize),
            "device": args.device,
            "batch_rows": int(args.batch_rows),
            "batch_tokens": int(args.batch_tokens),
            "query_record_count": int(query_count),
            "elapsed_sec": elapsed,
            "storage_bytes": {
                "query_tokens": _dir_bytes(Path(args.output_query_dir)),
                "landmark_bank": _dir_bytes(Path(args.output_landmark_bank)),
                "transform": _dir_bytes(Path(args.output_transform)),
            },
            "inputs": {
                "query_manifest": {
                    "path": str(args.query_manifest),
                    "sha256": file_sha256_short(Path(args.query_manifest)),
                },
                "landmark_bank": {
                    "path": str(args.landmark_bank),
                    "sha256": file_sha256_short(Path(args.landmark_bank)),
                },
                "layer_name": args.layer_name,
                "fit_labels_npy": args.fit_labels_npy,
            },
            "outputs": {
                "query_manifest": str(args.output_query_manifest),
                "query_dir": str(args.output_query_dir),
                "landmark_bank": str(args.output_landmark_bank),
                "transform": str(args.output_transform),
            },
        }
    )
    if transform.selected_channels is not None:
        summary["selected_channels_head"] = [int(v) for v in transform.selected_channels[: min(32, transform.output_dim)]]
    if transform.channel_scores is not None:
        summary["channel_score_mean"] = float(np.mean(transform.channel_scores))
        summary["channel_score_max"] = float(np.max(transform.channel_scores))
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
