"""Train only the candidate geometry head on real top-L RGB pairs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics
from feature_extract.vfm.measurement_v1.rgb_patch_diagnostics import _load_model
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    _read_csv,
    _stack_patch_batch,
)


def _bool_text(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _labels(rows: Sequence[Mapping[str, object]], target_key: str) -> torch.Tensor:
    values = []
    for row in rows:
        text = str(row.get(str(target_key), "")).strip()
        if not text:
            raise ValueError(f"candidate geometry row lacks {target_key}")
        values.append(_bool_text(text))
    return torch.tensor(values, dtype=torch.float32)


def _sample_balanced_batch(
    positive_rows: Sequence[dict[str, str]],
    negative_rows: Sequence[dict[str, str]],
    *,
    batch_size: int,
    positive_fraction: float,
    rng: random.Random,
) -> list[dict[str, str]]:
    positive_count = max(1, min(int(batch_size) - 1, int(round(batch_size * positive_fraction))))
    negative_count = int(batch_size) - positive_count
    batch = [positive_rows[rng.randrange(len(positive_rows))] for _ in range(positive_count)]
    batch.extend(negative_rows[rng.randrange(len(negative_rows))] for _ in range(negative_count))
    rng.shuffle(batch)
    return batch


def _sample_eval_rows(
    rows: Sequence[dict[str, str]], *, count: int, seed: int
) -> list[dict[str, str]]:
    if int(count) >= len(rows):
        return list(rows)
    indices = random.Random(int(seed)).sample(range(len(rows)), int(count))
    return [rows[index] for index in sorted(indices)]


def _forward_geometry(
    model: torch.nn.Module,
    rows: Sequence[dict[str, str]],
    *,
    image_root: Path,
    image_width: int,
    image_height: int,
    image_cache: TensorImageLRUCache,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    query_patch, support_patch, _target, _baseline, _dustbin = _stack_patch_batch(
        rows,
        image_root=Path(image_root),
        render_cache_by_query={},
        image_width=int(image_width),
        image_height=int(image_height),
        crop_radius_px=float(model.crop_radius_px),
        step_px=float(model.step_px),
        query_cache=image_cache,
        render_cache=image_cache,
        query_source="real_pair",
        render_patch_augmentation="none",
        support_patch_warp="none",
        image_cache_device=device,
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(use_amp and device.type == "cuda"),
    ):
        prediction = model.forward_from_patches(
            query_patch.to(device), support_patch.to(device)
        )
    if prediction.geometry_logit is None:
        raise RuntimeError("RGB measurement model has no candidate geometry head")
    return prediction.geometry_logit.float()


def _evaluate(
    model: torch.nn.Module,
    rows: Sequence[dict[str, str]],
    *,
    target_key: str,
    image_root: Path,
    image_width: int,
    image_height: int,
    image_cache: TensorImageLRUCache,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> dict[str, object]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            batch = rows[start : start + int(batch_size)]
            logits = _forward_geometry(
                model,
                batch,
                image_root=image_root,
                image_width=int(image_width),
                image_height=int(image_height),
                image_cache=image_cache,
                device=device,
                use_amp=bool(use_amp),
            )
            labels.append(_labels(batch, str(target_key)).numpy().astype(bool))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    label_values = np.concatenate(labels)
    probability_values = np.concatenate(probabilities)
    return {
        **confidence_metrics(label_values, probability_values),
        "sample_count": int(len(label_values)),
        "positive_count": int(np.sum(label_values)),
    }


def train_candidate_rgb_geometry_head(
    *,
    train_rows_csv: Path,
    validation_rows_csv: Path,
    image_root: Path,
    init_checkpoint: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    target_key: str = "target_geometry_correct_5px",
    steps: int = 500,
    batch_size: int = 256,
    eval_batch_size: int = 256,
    max_eval_rows: int = 8192,
    learning_rate: float = 1e-3,
    positive_fraction: float = 0.5,
    seed: int = 0,
    device: str = "cuda",
    image_cache_max_gb: float = 18.0,
    use_amp: bool = True,
) -> dict[str, Any]:
    train_rows = _read_csv(Path(train_rows_csv))
    validation_rows = _read_csv(Path(validation_rows_csv))
    positives = [row for row in train_rows if _bool_text(row.get(str(target_key), ""))]
    negatives = [row for row in train_rows if not _bool_text(row.get(str(target_key), ""))]
    if not positives or not negatives:
        raise ValueError("candidate geometry training requires both target classes")
    if not 0.0 < float(positive_fraction) < 1.0:
        raise ValueError("positive_fraction must be in (0, 1)")
    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    initial_payload = torch.load(Path(init_checkpoint), map_location="cpu")
    initial_state = (
        initial_payload["model"]
        if isinstance(initial_payload, dict) and "model" in initial_payload
        else initial_payload
    )
    initialized_from_inverse_dustbin = not any(
        str(key).startswith("geometry_head.") for key in initial_state
    )
    model = _load_model(Path(init_checkpoint), device=torch_device)
    if initialized_from_inverse_dustbin:
        model.geometry_head.load_state_dict(model.dustbin_head.state_dict())
        final_layer = model.geometry_head[-1]
        with torch.no_grad():
            final_layer.weight.mul_(-1.0)
            final_layer.bias.mul_(-1.0)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.geometry_head.parameters():
        parameter.requires_grad_(True)
    model.train()
    optimizer = torch.optim.AdamW(model.geometry_head.parameters(), lr=float(learning_rate))
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(use_amp and torch_device.type == "cuda")
    )
    cache_bytes = (
        None
        if float(image_cache_max_gb) <= 0.0
        else int(float(image_cache_max_gb) * (1024**3))
    )
    image_cache = TensorImageLRUCache(max_bytes=cache_bytes)
    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))
    final_loss = float("nan")
    for _step in range(int(steps)):
        batch = _sample_balanced_batch(
            positives,
            negatives,
            batch_size=int(batch_size),
            positive_fraction=float(positive_fraction),
            rng=rng,
        )
        labels = _labels(batch, str(target_key)).to(torch_device)
        logits = _forward_geometry(
            model,
            batch,
            image_root=Path(image_root),
            image_width=int(image_width),
            image_height=int(image_height),
            image_cache=image_cache,
            device=torch_device,
            use_amp=bool(use_amp),
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        final_loss = float(loss.detach().cpu().item())

    validation_eval_rows = _sample_eval_rows(
        validation_rows, count=int(max_eval_rows), seed=int(seed) + 1
    )
    train_eval_rows = _sample_eval_rows(
        train_rows, count=int(max_eval_rows), seed=int(seed) + 2
    )
    train_metrics = _evaluate(
        model,
        train_eval_rows,
        target_key=str(target_key),
        image_root=Path(image_root),
        image_width=int(image_width),
        image_height=int(image_height),
        image_cache=image_cache,
        device=torch_device,
        batch_size=int(eval_batch_size),
        use_amp=bool(use_amp),
    )
    validation_metrics = _evaluate(
        model,
        validation_eval_rows,
        target_key=str(target_key),
        image_root=Path(image_root),
        image_width=int(image_width),
        image_height=int(image_height),
        image_cache=image_cache,
        device=torch_device,
        batch_size=int(eval_batch_size),
        use_amp=bool(use_amp),
    )
    config = dict(initial_payload.get("config", {}) if isinstance(initial_payload, dict) else {})
    config["candidate_geometry_head_target"] = str(target_key)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "rgb_patch_measurement_branch.pt"
    torch.save({"model": model.state_dict(), "config": config}, checkpoint)
    summary = {
        "stage": "candidate_rgb_geometry_head_train",
        "protocol": {
            "trainable_scope": "geometry_head_only",
            "target": str(target_key),
            "target_source": "GT_pose_projection_residual_TARGET_ONLY",
            "pose_features": False,
            "update_head_trainable": False,
            "dustbin_head_trainable": False,
            "offset_heads_trainable": False,
            "geometry_initialization": (
                "inverse_frozen_dustbin"
                if initialized_from_inverse_dustbin
                else "checkpoint_geometry_head"
            ),
            "render": False,
            "query_source": "real_pair",
        },
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "positive_fraction": float(positive_fraction),
        "final_loss": final_loss,
        "train_positive_count": int(len(positives)),
        "train_negative_count": int(len(negatives)),
        "train_metrics_sampled": train_metrics,
        "validation_metrics_sampled": validation_metrics,
        "max_eval_rows": int(max_eval_rows),
        "use_amp": bool(use_amp),
        "image_cache": image_cache.summary(),
        "inputs": {
            "train_rows_csv": str(train_rows_csv),
            "train_rows_sha256": file_sha256_short(Path(train_rows_csv)),
            "validation_rows_csv": str(validation_rows_csv),
            "validation_rows_sha256": file_sha256_short(Path(validation_rows_csv)),
            "init_checkpoint": str(init_checkpoint),
            "init_checkpoint_sha256": file_sha256_short(Path(init_checkpoint)),
        },
        "outputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256_short(checkpoint),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
