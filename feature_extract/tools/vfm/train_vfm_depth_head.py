"""Train and evaluate a lightweight RADIO token-grid depth head."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from feature_extract.vfm.vfm_depth_head import (
    RADIO_TOKEN_DEPTH_INVALID,
    RadioTokenDepthHead,
    aggregate_depth_metrics,
    compute_depth_metrics,
    masked_log_depth_l1,
    scale_invariant_log_loss,
)


class TokenDepthDataset(Dataset):
    def __init__(self, manifest_path: Path, max_records: int = 0) -> None:
        payload = json.loads(Path(manifest_path).read_text())
        records = list(payload["records"])
        if int(max_records) > 0:
            records = records[: int(max_records)]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[int(index)]
        with np.load(record["token_path"]) as token_data:
            if "radio_final" in token_data:
                token = np.asarray(token_data["radio_final"], dtype=np.float32)
            else:
                first = token_data.files[0]
                token = np.asarray(token_data[first], dtype=np.float32)
        with np.load(record["depth_path"]) as depth_data:
            depth = np.asarray(depth_data["depth"], dtype=np.float32)
            valid = np.asarray(depth_data["valid"], dtype=bool)
        return {
            "image_id": str(record["image_id"]),
            "token": torch.from_numpy(token),
            "depth": torch.from_numpy(depth),
            "valid": torch.from_numpy(valid),
        }


def _collate(batch: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "image_id": [str(item["image_id"]) for item in batch],
        "token": torch.stack([item["token"] for item in batch], dim=0),
        "depth": torch.stack([item["depth"] for item in batch], dim=0),
        "valid": torch.stack([item["valid"] for item in batch], dim=0),
    }


def _train_depth_median(dataset: TokenDepthDataset) -> float:
    values = []
    for record in dataset.records:
        with np.load(record["depth_path"]) as depth_data:
            depth = np.asarray(depth_data["depth"], dtype=np.float32)
            valid = np.asarray(depth_data["valid"], dtype=bool)
        if np.any(valid):
            values.append(depth[valid])
    if not values:
        return 1.0
    return float(np.median(np.concatenate(values, axis=0)))


def _colorize_depth(depth: np.ndarray, valid: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for depth visualization") from exc
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if vmax <= vmin:
        vmax = vmin + 1.0
    clipped = np.clip((depth - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    normalized[valid] = np.asarray(clipped[valid] * 255.0, dtype=np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _label_panel(image: np.ndarray, title: str) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for depth visualization") from exc
    image = np.asarray(image, dtype=np.uint8)
    h0, w0 = image.shape[:2]
    scale = max(1, int(np.ceil(320.0 / max(float(w0), 1.0))), int(np.ceil(180.0 / max(float(h0), 1.0))))
    if scale > 1:
        image = cv2.resize(image, (w0 * scale, h0 * scale), interpolation=cv2.INTER_NEAREST)
    h, w = image.shape[:2]
    canvas = np.zeros((h + 34, w, 3), dtype=np.uint8)
    canvas[34:] = image
    cv2.putText(canvas, title, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _write_visualizations(
    output_dir: Path,
    image_ids: Sequence[str],
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    limit: int,
    start_index: int = 0,
) -> None:
    if int(limit) <= 0:
        return
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for depth visualization") from exc
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    count = min(max(int(limit) - int(start_index), 0), int(pred.shape[0]))
    for idx in range(count):
        mask = valid[idx]
        if np.any(mask):
            vmin = float(np.percentile(target[idx][mask], 5.0))
            vmax = float(np.percentile(target[idx][mask], 95.0))
        else:
            vmin, vmax = 0.0, 1.0
        target_color = _colorize_depth(target[idx], mask, vmin, vmax)
        pred_color = _colorize_depth(pred[idx], mask, vmin, vmax)
        abs_err = np.zeros_like(target[idx], dtype=np.float32)
        abs_err[mask] = np.abs(pred[idx][mask] - target[idx][mask])
        err_valid = mask & np.isfinite(abs_err)
        err_max = float(np.percentile(abs_err[err_valid], 95.0)) if np.any(err_valid) else 1.0
        err_color = _colorize_depth(abs_err, err_valid, 0.0, max(err_max, 1e-3))
        panel = np.concatenate(
            [
                _label_panel(target_color, "target depth: blue near, red far"),
                _label_panel(pred_color, "pred depth: same scale as target"),
                _label_panel(err_color, "abs error: blue low, red high"),
            ],
            axis=1,
        )
        safe_id = str(image_ids[idx]).replace("/", "__").replace("\\", "__")
        cv2.imwrite(str(vis_dir / f"{int(start_index) + idx:03d}_{safe_id}_target_pred_error.png"), panel)


def evaluate(
    model: RadioTokenDepthHead,
    loader: DataLoader,
    device: torch.device,
    baseline_depth: float,
    output_dir: Path | None = None,
    visualize_limit: int = 0,
) -> dict[str, object]:
    model.eval()
    model_rows = []
    baseline_rows = []
    vis_written = 0
    with torch.no_grad():
        for batch in loader:
            token = batch["token"].to(device=device, dtype=torch.float32)
            target = batch["depth"].to(device=device, dtype=torch.float32)
            valid = batch["valid"].to(device=device, dtype=torch.bool)
            pred = model(token)
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            baseline_np = np.full_like(target_np, float(baseline_depth), dtype=np.float32)
            for idx in range(pred_np.shape[0]):
                model_rows.append(compute_depth_metrics(pred_np[idx], target_np[idx], valid_np[idx]))
                baseline_rows.append(compute_depth_metrics(baseline_np[idx], target_np[idx], valid_np[idx]))
            if output_dir is not None and vis_written < int(visualize_limit):
                _write_visualizations(
                    output_dir,
                    batch["image_id"],
                    pred_np,
                    target_np,
                    valid_np,
                    int(visualize_limit),
                    start_index=vis_written,
                )
                vis_written = min(int(visualize_limit), vis_written + int(pred_np.shape[0]))
    return {"model": aggregate_depth_metrics(model_rows), "median_baseline": aggregate_depth_metrics(baseline_rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_depth_manifest", required=True)
    parser.add_argument("--eval_depth_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_records", type=int, default=0)
    parser.add_argument("--max_eval_records", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--silog_weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--visualize", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = TokenDepthDataset(Path(args.train_depth_manifest), max_records=int(args.max_train_records))
    eval_dataset = TokenDepthDataset(Path(args.eval_depth_manifest), max_records=int(args.max_eval_records))
    if len(train_dataset) == 0:
        raise ValueError("empty training depth manifest")
    if len(eval_dataset) == 0:
        raise ValueError("empty eval depth manifest")
    first = train_dataset[0]
    in_channels = int(first["token"].shape[0])
    requested = str(args.device)
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)
    model = RadioTokenDepthHead(in_channels=in_channels, hidden_channels=int(args.hidden_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
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
            target = batch["depth"].to(device=device, dtype=torch.float32)
            valid = batch["valid"].to(device=device, dtype=torch.bool)
            pred = model(token)
            loss = masked_log_depth_l1(pred, target, valid)
            if float(args.silog_weight) > 0.0:
                loss = loss + float(args.silog_weight) * scale_invariant_log_loss(pred, target, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            valid_counts.append(int(torch.sum(valid).detach().cpu()))
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": None if not losses else float(np.mean(losses)),
                "train_valid_tokens": int(np.sum(valid_counts)),
            }
        )
    metrics = evaluate(
        model,
        eval_loader,
        device,
        baseline_depth=baseline_depth,
        output_dir=output_dir,
        visualize_limit=int(args.visualize),
    )
    checkpoint_path = output_dir / "radio_token_depth_head.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": in_channels,
            "hidden_channels": int(args.hidden_channels),
            "baseline_depth": float(baseline_depth),
            "args": vars(args),
            "history": history,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    summary = {
        "stage": "radio_token_depth_head",
        "elapsed_sec": float(time.perf_counter() - started),
        "train_count": int(len(train_dataset)),
        "eval_count": int(len(eval_dataset)),
        "device": str(device),
        "baseline_depth_m": float(baseline_depth),
        "history": history,
        "metrics": metrics,
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "visualizations": str(output_dir / "visualizations"),
            "summary": str(output_dir / "depth_head_summary.json"),
        },
    }
    (output_dir / "depth_head_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
