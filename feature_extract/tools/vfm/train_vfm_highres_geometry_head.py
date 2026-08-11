"""Train/evaluate high-resolution RADIO geometry head on 2DGS labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from feature_extract.vfm.vfm_depth_head import aggregate_depth_metrics, compute_depth_metrics
from feature_extract.vfm.vfm_highres_geometry_head import (
    RadioHighResGeometryHead,
    masked_geometry_loss,
    masked_log_depth_l1,
    masked_log_depth_gradient_loss,
    masked_normal_cosine_loss,
    masked_scale_invariant_log_depth_loss,
)


class HighResGeometryDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        max_records: int = 0,
        layer_name: str = "radio_final",
        cache_in_memory: bool = False,
    ) -> None:
        payload = json.loads(Path(manifest_path).read_text())
        records = list(payload["records"])
        if int(max_records) > 0:
            records = records[: int(max_records)]
        self.records = records
        self.layer_name = str(layer_name)
        # RADIO token NPZs are deliberately compressed and expensive to decode.
        # Repeating that work every epoch starves the GPU, while each St Mary's
        # fold comfortably fits in host RAM.  Keep this opt-in so larger scenes
        # retain the streaming behaviour.
        self._cache = (
            tuple(self._load_item(index) for index in range(len(self.records)))
            if bool(cache_in_memory) else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def _load_item(self, index: int) -> dict[str, object]:
        record = self.records[int(index)]
        with np.load(record["token_path"]) as token_data:
            if self.layer_name not in token_data:
                available = ", ".join(token_data.files)
                raise KeyError(
                    f"Token layer {self.layer_name!r} is missing from {record['token_path']}; "
                    f"available layers: {available}"
                )
            token = np.asarray(token_data[self.layer_name], dtype=np.float32)
        with np.load(record["geometry_path"]) as geometry_data:
            depth = np.asarray(geometry_data["depth"], dtype=np.float32)
            normal = np.asarray(geometry_data["normal_cam"], dtype=np.float32).transpose(2, 0, 1)
            valid = np.asarray(geometry_data["valid"], dtype=bool)
        return {
            "image_id": str(record["image_id"]),
            "token": torch.from_numpy(token),
            "depth": torch.from_numpy(depth),
            "normal": torch.from_numpy(normal),
            "valid": torch.from_numpy(valid),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        if self._cache is not None:
            return self._cache[int(index)]
        return self._load_item(int(index))


def _collate(batch: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "image_id": [str(item["image_id"]) for item in batch],
        "token": torch.stack([item["token"] for item in batch], dim=0),
        "depth": torch.stack([item["depth"] for item in batch], dim=0),
        "normal": torch.stack([item["normal"] for item in batch], dim=0),
        "valid": torch.stack([item["valid"] for item in batch], dim=0),
    }


def _train_depth_median(dataset: HighResGeometryDataset) -> float:
    values = []
    for record in dataset.records:
        with np.load(record["geometry_path"]) as data:
            depth = np.asarray(data["depth"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)
        if np.any(valid):
            values.append(depth[valid])
    if not values:
        return 1.0
    return float(np.median(np.concatenate(values, axis=0)))


def _normal_metrics(pred_normal: np.ndarray, target_normal: np.ndarray, valid_mask: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred_normal, dtype=np.float64)
    target = np.asarray(target_normal, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[0] != 3:
        raise ValueError("normal arrays must have shape (3, H, W)")
    if valid.shape != pred.shape[1:]:
        raise ValueError("valid mask must have shape (H, W)")
    if not np.any(valid):
        return {"valid_count": 0, "normal_cos": float("nan"), "normal_angle_deg": float("nan")}
    p = pred.transpose(1, 2, 0)[valid]
    t = target.transpose(1, 2, 0)[valid]
    p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-8)
    cos = np.clip(np.sum(p * t, axis=1), -1.0, 1.0)
    return {
        "valid_count": int(cos.size),
        "normal_cos": float(np.mean(cos)),
        "normal_angle_deg": float(np.mean(np.degrees(np.arccos(cos)))),
    }


def _aggregate_normal_metrics(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    valid_rows = [row for row in rows if int(row.get("valid_count", 0)) > 0]
    if not valid_rows:
        return {"image_count": 0, "valid_count": 0}
    total = float(sum(int(row["valid_count"]) for row in valid_rows))
    return {
        "image_count": int(len(valid_rows)),
        "valid_count": int(total),
        "normal_cos": float(sum(float(row["normal_cos"]) * int(row["valid_count"]) for row in valid_rows) / max(total, 1.0)),
        "normal_angle_deg": float(sum(float(row["normal_angle_deg"]) * int(row["valid_count"]) for row in valid_rows) / max(total, 1.0)),
    }


def _colorize(values: np.ndarray, valid: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for geometry visualization") from exc
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if vmin is None:
        vmin = float(np.percentile(values[valid], 5.0)) if np.any(valid) else 0.0
    if vmax is None:
        vmax = float(np.percentile(values[valid], 95.0)) if np.any(valid) else 1.0
    if float(vmax) <= float(vmin):
        vmax = float(vmin) + 1.0
    norm = np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    norm[valid] = np.asarray(scaled[valid] * 255.0, dtype=np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _normal_rgb(normal_chw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal_chw, dtype=np.float32).transpose(1, 2, 0)
    rgb = np.zeros((*valid.shape, 3), dtype=np.uint8)
    mapped = np.clip((normal + 1.0) * 0.5 * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb[np.asarray(valid, dtype=bool)] = mapped[np.asarray(valid, dtype=bool)]
    return rgb


def _label_panel(image: np.ndarray, title: str) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for geometry visualization") from exc
    image = np.asarray(image, dtype=np.uint8)
    h, w = image.shape[:2]
    canvas = np.zeros((h + 34, w, 3), dtype=np.uint8)
    canvas[34:] = image
    cv2.putText(canvas, title, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _write_visualizations(
    output_dir: Path,
    image_ids: Sequence[str],
    pred_depth: np.ndarray,
    pred_normal: np.ndarray,
    target_depth: np.ndarray,
    target_normal: np.ndarray,
    valid: np.ndarray,
    limit: int,
    start_index: int,
) -> None:
    if int(limit) <= 0:
        return
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for geometry visualization") from exc
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    count = min(max(int(limit) - int(start_index), 0), int(pred_depth.shape[0]))
    for idx in range(count):
        mask = valid[idx]
        if np.any(mask):
            vmin = float(np.percentile(target_depth[idx][mask], 5.0))
            vmax = float(np.percentile(target_depth[idx][mask], 95.0))
        else:
            vmin, vmax = 0.0, 1.0
        err = np.zeros_like(target_depth[idx], dtype=np.float32)
        err[mask] = np.abs(pred_depth[idx][mask] - target_depth[idx][mask])
        err_max = float(np.percentile(err[mask], 95.0)) if np.any(mask) else 1.0
        panel = np.concatenate(
            [
                _label_panel(_colorize(target_depth[idx], mask, vmin, vmax), "target depth"),
                _label_panel(_colorize(pred_depth[idx], mask, vmin, vmax), "pred depth"),
                _label_panel(_colorize(err, mask, 0.0, max(err_max, 1e-3)), "abs depth error"),
                _label_panel(_normal_rgb(target_normal[idx], mask), "target normal"),
                _label_panel(_normal_rgb(pred_normal[idx], mask), "pred normal"),
            ],
            axis=1,
        )
        safe_id = str(image_ids[idx]).replace("/", "__").replace("\\", "__")
        cv2.imwrite(str(vis_dir / f"{int(start_index) + idx:03d}_{safe_id}_geometry_pred.png"), panel)


def evaluate(
    model: RadioHighResGeometryHead,
    loader: DataLoader,
    device: torch.device,
    baseline_depth: float,
    output_dir: Path | None = None,
    visualize_limit: int = 0,
) -> dict[str, object]:
    model.eval()
    depth_rows = []
    baseline_rows = []
    scale_aligned_rows = []
    normal_rows = []
    confidence_brier_sum = 0.0
    confidence_valid_sum = 0.0
    confidence_invalid_sum = 0.0
    confidence_count = 0
    confidence_valid_count = 0
    confidence_invalid_count = 0
    vis_written = 0
    with torch.no_grad():
        for batch in loader:
            token = batch["token"].to(device=device, dtype=torch.float32)
            target_depth = batch["depth"].to(device=device, dtype=torch.float32)
            target_normal = batch["normal"].to(device=device, dtype=torch.float32)
            valid = batch["valid"].to(device=device, dtype=torch.bool)
            output = model(token, output_size=tuple(target_depth.shape[-2:]))
            pred_depth = output.depth.detach().cpu().numpy()
            pred_normal = output.normal.detach().cpu().numpy()
            pred_confidence = output.confidence.detach().cpu().numpy()
            target_depth_np = target_depth.detach().cpu().numpy()
            target_normal_np = target_normal.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            target_confidence = valid_np.astype(np.float32)
            confidence_brier_sum += float(np.sum(np.square(pred_confidence - target_confidence)))
            confidence_count += int(pred_confidence.size)
            confidence_valid_sum += float(np.sum(pred_confidence[valid_np]))
            confidence_valid_count += int(np.sum(valid_np))
            confidence_invalid_sum += float(np.sum(pred_confidence[~valid_np]))
            confidence_invalid_count += int(np.sum(~valid_np))
            baseline_np = np.full_like(target_depth_np, float(baseline_depth), dtype=np.float32)
            for idx in range(pred_depth.shape[0]):
                depth_rows.append(compute_depth_metrics(pred_depth[idx], target_depth_np[idx], valid_np[idx]))
                baseline_rows.append(compute_depth_metrics(baseline_np[idx], target_depth_np[idx], valid_np[idx]))
                if np.any(valid_np[idx]):
                    ratio = target_depth_np[idx][valid_np[idx]] / np.maximum(
                        pred_depth[idx][valid_np[idx]], 1e-6,
                    )
                    scale = float(np.median(ratio))
                else:
                    scale = 1.0
                scale_aligned_rows.append(compute_depth_metrics(
                    pred_depth[idx] * scale, target_depth_np[idx], valid_np[idx],
                ))
                normal_rows.append(_normal_metrics(pred_normal[idx], target_normal_np[idx], valid_np[idx]))
            if output_dir is not None and vis_written < int(visualize_limit):
                _write_visualizations(
                    output_dir,
                    batch["image_id"],
                    pred_depth,
                    pred_normal,
                    target_depth_np,
                    target_normal_np,
                    valid_np,
                    int(visualize_limit),
                    vis_written,
                )
                vis_written = min(int(visualize_limit), vis_written + int(pred_depth.shape[0]))
    return {
        "depth": aggregate_depth_metrics(depth_rows),
        "depth_scale_aligned": aggregate_depth_metrics(scale_aligned_rows),
        "median_depth_baseline": aggregate_depth_metrics(baseline_rows),
        "normal": _aggregate_normal_metrics(normal_rows),
        "confidence": {
            "brier": float(confidence_brier_sum / max(confidence_count, 1)),
            "valid_mean": float(confidence_valid_sum / max(confidence_valid_count, 1)),
            "invalid_mean": float(confidence_invalid_sum / max(confidence_invalid_count, 1)),
            "pixel_count": int(confidence_count),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_geometry_manifest", required=True)
    parser.add_argument("--eval_geometry_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--max_train_records", type=int, default=0)
    parser.add_argument("--max_eval_records", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--architecture", default="shared", choices=("shared", "separate_decoders"))
    parser.add_argument("--task", default="multitask", choices=("multitask", "depth_only", "normal_only"))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--normal_weight", type=float, default=0.25)
    parser.add_argument("--confidence_weight", type=float, default=0.10)
    parser.add_argument("--absolute_depth_weight", type=float, default=1.0)
    parser.add_argument("--scale_invariant_depth_weight", type=float, default=0.0)
    parser.add_argument("--depth_gradient_weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_in_memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--visualize", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = HighResGeometryDataset(
        Path(args.train_geometry_manifest), int(args.max_train_records), args.layer_name,
        cache_in_memory=bool(args.cache_in_memory),
    )
    eval_dataset = HighResGeometryDataset(
        Path(args.eval_geometry_manifest), int(args.max_eval_records), args.layer_name,
        cache_in_memory=bool(args.cache_in_memory),
    )
    if len(train_dataset) == 0:
        raise ValueError("empty training geometry manifest")
    if len(eval_dataset) == 0:
        raise ValueError("empty eval geometry manifest")
    first = train_dataset[0]
    in_channels = int(first["token"].shape[0])
    device_name = str(args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model = RadioHighResGeometryHead(
        in_channels=in_channels,
        hidden_channels=int(args.hidden_channels),
        architecture=str(args.architecture),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    amp_enabled = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    baseline_depth = _train_depth_median(train_dataset)
    history = []
    for epoch in range(int(args.epochs)):
        model.train()
        losses = []
        valid_counts = []
        for batch in train_loader:
            token = batch["token"].to(device=device, dtype=torch.float32)
            target_depth = batch["depth"].to(device=device, dtype=torch.float32)
            target_normal = batch["normal"].to(device=device, dtype=torch.float32)
            valid = batch["valid"].to(device=device, dtype=torch.bool)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(token, output_size=tuple(target_depth.shape[-2:]))
                if str(args.task) == "depth_only":
                    loss = masked_log_depth_l1(output.depth, target_depth, valid)
                elif str(args.task) == "normal_only":
                    loss = masked_normal_cosine_loss(output.normal, F.normalize(target_normal, dim=1, eps=1e-6), valid)
                else:
                    loss = masked_geometry_loss(
                        output.depth,
                        output.normal,
                        target_depth,
                        F.normalize(target_normal, dim=1, eps=1e-6),
                        valid,
                        normal_weight=float(args.normal_weight),
                        pred_confidence=output.confidence,
                        confidence_weight=float(args.confidence_weight),
                        absolute_depth_weight=float(args.absolute_depth_weight),
                        scale_invariant_depth_weight=float(args.scale_invariant_depth_weight),
                        depth_gradient_weight=float(args.depth_gradient_weight),
                    )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            valid_counts.append(int(torch.sum(valid).detach().cpu()))
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": None if not losses else float(np.mean(losses)),
                "train_valid_pixels": int(np.sum(valid_counts)),
            }
        )
    metrics = evaluate(model, eval_loader, device, baseline_depth, output_dir=output_dir, visualize_limit=int(args.visualize))
    checkpoint_path = output_dir / "radio_highres_geometry_head.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": int(in_channels),
            "hidden_channels": int(args.hidden_channels),
            "architecture": str(args.architecture),
            "task": str(args.task),
            "baseline_depth": float(baseline_depth),
            "args": vars(args),
            "history": history,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    summary = {
        "stage": "radio_highres_geometry_head",
        "elapsed_sec": float(time.perf_counter() - started),
        "train_count": int(len(train_dataset)),
        "eval_count": int(len(eval_dataset)),
        "device": str(device),
        "architecture": str(args.architecture),
        "task": str(args.task),
        "seed": int(args.seed),
        "baseline_depth_m": float(baseline_depth),
        "history": history,
        "metrics": metrics,
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "summary": str(output_dir / "geometry_head_summary.json"),
            "visualizations": str(output_dir / "visualizations"),
        },
    }
    (output_dir / "geometry_head_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
