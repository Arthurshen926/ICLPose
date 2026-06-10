"""Train a MATCHA-style RGB-local 65-bin keypoint detector by ALIKE distillation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import _safe_image_stem
from feature_extract.vfm.matcha_keypoint_distillation import AlikeKeypointExtractor, build_keypoint_label_map
from feature_extract.vfm.matcha_rgb_keypoint_detector import MatchaRgbKeypointDetector, matcha_alike_distillation_loss
from feature_extract.vfm.tokens import TokenBankManifest


def _select_records(records: Sequence[object], max_queries: int, mode: str) -> list[object]:
    values = list(records)
    if int(max_queries) <= 0 or int(max_queries) >= len(values):
        return values
    if str(mode) == "prefix":
        return values[: int(max_queries)]
    indices = np.linspace(0, len(values) - 1, int(max_queries), dtype=np.int64)
    return [values[int(idx)] for idx in indices]


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_rgb(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return np.asarray(image, dtype=np.uint8)
    width = max(8, (width // 8) * 8)
    height = max(8, (height // 8) * 8)
    return cv2.resize(np.asarray(image, dtype=np.uint8), (width, height), interpolation=cv2.INTER_AREA)


def _load_render_rgb(cache_dir: Path, image_id: str, width: int, height: int) -> np.ndarray | None:
    path = cache_dir / f"{_safe_image_stem(image_id)}_gt_{int(width)}x{int(height)}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    if "rgb" not in data:
        return None
    return np.asarray(data["rgb"], dtype=np.uint8)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--render_rgb_depth_cache_dir", default="")
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--include_render_rgb", action="store_true")
    parser.add_argument("--max_queries", type=int, default=32)
    parser.add_argument("--view_selection", default="uniform", choices=("prefix", "uniform"))
    parser.add_argument("--train_width", type=int, default=512)
    parser.add_argument("--train_height", type=int, default=288)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--non_keypoint_divisor", type=int, default=32)
    parser.add_argument("--alike_repo", default="/root/matcha")
    parser.add_argument("--alike_model", default="alike-t")
    parser.add_argument("--alike_top_k", type=int, default=2048)
    parser.add_argument("--alike_scores_th", type=float, default=0.1)
    parser.add_argument("--alike_n_limit", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = _select_records(manifest.records, int(args.max_queries), str(args.view_selection))
    extractor = AlikeKeypointExtractor(
        matcha_repo=str(args.alike_repo),
        model_name=str(args.alike_model),
        top_k=int(args.alike_top_k),
        scores_th=float(args.alike_scores_th),
        n_limit=int(args.alike_n_limit),
        device=str(device),
    )
    images: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []

    def add_image(rgb: np.ndarray, *, image_id: str, source: str) -> None:
        resized = _resize_rgb(rgb, width=int(args.train_width), height=int(args.train_height))
        keypoints, scores = extractor(resized)
        label_map, stats = build_keypoint_label_map(
            keypoints,
            image_width=int(resized.shape[1]),
            image_height=int(resized.shape[0]),
            grid_width=int(resized.shape[1]) // 8,
            grid_height=int(resized.shape[0]) // 8,
            scores=scores,
        )
        images.append(torch.as_tensor(resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0)
        labels.append(torch.as_tensor(label_map, dtype=torch.long))
        rows.append(
            {
                "image_id": image_id,
                "source": source,
                "width": int(resized.shape[1]),
                "height": int(resized.shape[0]),
                **stats,
            }
        )

    render_cache_dir = Path(args.render_rgb_depth_cache_dir) if str(args.render_rgb_depth_cache_dir) else None
    for record in records:
        add_image(_read_rgb(Path(args.image_root) / record.image_id), image_id=record.image_id, source="query")
        if bool(args.include_render_rgb) and render_cache_dir is not None:
            render_rgb = _load_render_rgb(render_cache_dir, record.image_id, int(args.render_width), int(args.render_height))
            if render_rgb is not None:
                add_image(render_rgb, image_id=record.image_id, source="render")

    if not images:
        raise ValueError("no training images were loaded")
    image_tensor = torch.stack(images, dim=0).to(device)
    label_tensor = torch.stack(labels, dim=0).to(device)
    model = MatchaRgbKeypointDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(args.seed))
    initial_metrics = None
    final_metrics = None
    for step in range(int(args.steps)):
        index = int(rng.integers(0, image_tensor.shape[0]))
        logits = model(image_tensor[index : index + 1])
        loss, metrics = matcha_alike_distillation_loss(
            logits,
            label_tensor[index : index + 1],
            non_keypoint_divisor=int(args.non_keypoint_divisor),
            seed=int(args.seed) + int(step),
        )
        if step == 0:
            initial_metrics = dict(metrics)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_metrics = dict(metrics)
    with torch.no_grad():
        eval_losses = []
        eval_acc = []
        eval_pos = []
        eval_non = []
        for index in range(image_tensor.shape[0]):
            loss, metrics = matcha_alike_distillation_loss(
                model(image_tensor[index : index + 1]),
                label_tensor[index : index + 1],
                non_keypoint_divisor=int(args.non_keypoint_divisor),
                seed=int(args.seed) + 10000 + int(index),
            )
            eval_losses.append(float(loss.detach().cpu().item()))
            eval_acc.append(float(metrics["acc"]))
            eval_pos.append(float(metrics["positive_acc"]))
            eval_non.append(float(metrics["non_keypoint_acc"]))
    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.cpu().eval().state_dict(),
            "config": vars(args),
            "initial": initial_metrics,
            "final": final_metrics,
        },
        output_model,
    )
    summary = {
        "stage": "matcha_rgb_keypoint_detector_training",
        "elapsed_sec": float(time.perf_counter() - started),
        "config": vars(args),
        "image_count": int(len(images)),
        "rows": rows,
        "training": {
            "initial": initial_metrics,
            "final": final_metrics,
            "eval_loss": float(np.mean(eval_losses)),
            "eval_acc": float(np.mean(eval_acc)),
            "eval_positive_acc": float(np.mean(eval_pos)),
            "eval_non_keypoint_acc": float(np.mean(eval_non)),
        },
        "outputs": {"model": str(output_model)},
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
