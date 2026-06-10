"""Train residual QKV attention on frozen MATCHA coarse-fine descriptors."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.matcha_coarse_fine_adapter import (
    load_matcha_coarse_fine_adapter,
    load_matcha_coarse_fine_training_set_npz,
)
from feature_extract.vfm.matcha_qkv_attention import (
    QKVAttentionTrainingConfig,
    save_qkv_attention_checkpoint,
    train_qkv_attention,
)


def _sha(path: str) -> str:
    value = Path(path)
    return file_sha256_short(value) if value.exists() and value.is_file() else ""


def _encode_array(model, features: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape (N, C)")
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    model = model.to(torch_device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, values.shape[0], int(batch_size)):
            tensor = torch.as_tensor(values[start : start + int(batch_size)], dtype=torch.float32, device=torch_device)
            outputs.append(model.encode(tensor).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, int(model.output_dim)), dtype=np.float32)


def _cap_samples(
    query: np.ndarray,
    render: np.ndarray,
    negatives: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(max_samples) <= 0 or query.shape[0] <= int(max_samples):
        return query, render, negatives
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(query.shape[0], size=int(max_samples), replace=False))
    return query[keep], render[keep], negatives[keep]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_cache", required=True)
    parser.add_argument("--base_adapter_checkpoint", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--max_samples", type=int, default=30000)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--logit_scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--eval_split_fraction", type=float, default=0.1)
    parser.add_argument("--encode_batch_size", type=int, default=32768)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    samples, metadata = load_matcha_coarse_fine_training_set_npz(Path(args.sample_cache))
    base_run = load_matcha_coarse_fine_adapter(Path(args.base_adapter_checkpoint), device=args.device)
    query = _encode_array(base_run.model, samples.query_features, device=str(args.device), batch_size=int(args.encode_batch_size))
    render = _encode_array(base_run.model, samples.render_features, device=str(args.device), batch_size=int(args.encode_batch_size))
    neg_flat = samples.negative_render_features.reshape(-1, samples.negative_render_features.shape[-1])
    neg_encoded = _encode_array(base_run.model, neg_flat, device=str(args.device), batch_size=int(args.encode_batch_size))
    negatives = neg_encoded.reshape(samples.negative_render_features.shape[0], samples.negative_render_features.shape[1], -1)
    query, render, negatives = _cap_samples(
        query,
        render,
        negatives,
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    config = QKVAttentionTrainingConfig(
        attention_dim=int(args.attention_dim),
        alpha=float(args.alpha),
        logit_scale=float(args.logit_scale),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        temperature=float(args.temperature),
        eval_split_fraction=float(args.eval_split_fraction),
        device=str(args.device),
        seed=int(args.seed),
    )
    run = train_qkv_attention(query, render, negatives, config)
    save_qkv_attention_checkpoint(run.model, Path(args.output_model), summary=run.summary)
    summary = {
        "stage": "matcha_qkv_attention_training",
        "elapsed_sec": float(time.perf_counter() - started),
        "sample_cache": {
            "path": str(args.sample_cache),
            "sha256": _sha(args.sample_cache),
            "metadata": metadata,
            "sample_count": int(samples.sample_count),
            "used_sample_count": int(query.shape[0]),
        },
        "base_adapter_checkpoint": {
            "path": str(args.base_adapter_checkpoint),
            "sha256": _sha(args.base_adapter_checkpoint),
        },
        "config": asdict(config),
        "training": run.summary,
        "outputs": {"model": str(args.output_model)},
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
