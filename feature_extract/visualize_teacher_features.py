#!/usr/bin/env python3
"""Visualize raw and cached dual-scale RADIO teacher features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from feature_extract.extract_radio_dual_features import DualScaleRADIOExtractor
from feature_extract.joint_radio import discover_images, safe_torch_load


IMAGE_PATTERNS = [
    "seq*/*.png",
    "seq*/*.jpg",
    "images/*.png",
    "images/*.jpg",
    "*.png",
    "*.jpg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dual-scale RADIO teacher features")
    parser.add_argument("--source_dir", required=True, help="Dataset directory")
    parser.add_argument("--feature_dir", default=None, help="Optional cached fine_geo/coarse_sem directory")
    parser.add_argument(
        "--output_dir",
        default="feature_extract/output/teacher_visuals",
        help="Output directory",
    )
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--sample_indices", type=int, nargs="*", default=None)
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--shallow_block", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _choose_indices(total: int, num_samples: int, sample_indices: list[int] | None) -> list[int]:
    if total <= 0:
        return []
    if sample_indices:
        chosen = [idx for idx in sample_indices if 0 <= idx < total]
        if chosen:
            return chosen
    if total <= num_samples:
        return list(range(total))
    return np.linspace(0, total - 1, num_samples, dtype=int).tolist()


def _fit_pca_stats(
    features: Iterable[torch.Tensor],
    n_components: int = 3,
    max_samples_per_feature: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_chunks = []
    for feature in features:
        feat = feature.float()
        pixels = feat.reshape(feat.shape[0], -1).T
        if pixels.shape[0] > max_samples_per_feature:
            sample_idx = torch.randperm(pixels.shape[0])[:max_samples_per_feature]
            pixels = pixels[sample_idx]
        flat_chunks.append(pixels)
    pixels = torch.cat(flat_chunks, dim=0)
    mean = pixels.mean(dim=0)
    centered = pixels - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:n_components]
    projected = centered @ basis.T
    value_min = projected.min(dim=0).values
    value_max = projected.max(dim=0).values
    return mean, basis, value_min, value_max


def _feature_to_rgb(
    feature: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    value_min: torch.Tensor,
    value_max: torch.Tensor,
) -> np.ndarray:
    feat = feature.float()
    height, width = feat.shape[-2:]
    pixels = feat.reshape(feat.shape[0], -1).T
    projected = (pixels - mean.to(pixels)) @ basis.T.to(pixels)
    projected = (projected - value_min.to(projected)) / (value_max.to(projected) - value_min.to(projected) + 1e-6)
    rgb = projected.clamp(0.0, 1.0).reshape(height, width, 3)
    return (rgb.cpu().numpy() * 255.0).astype(np.uint8)


def _annotate(image: np.ndarray, label: str) -> Image.Image:
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    text_height = 18
    canvas = Image.new("RGB", (pil_image.width, pil_image.height + text_height), color=(16, 16, 16))
    canvas.paste(pil_image, (0, text_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), label, fill=(240, 240, 240), font=font)
    return canvas


def _rgb_to_array(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image, dtype=np.uint8)


def _load_cached_feature(feature_dir: Path, scale_name: str, index: int) -> torch.Tensor:
    pattern = f"rgb_{index}_{scale_name}_*.pt"
    matches = sorted((feature_dir / scale_name).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing {scale_name} feature for index {index} under {feature_dir}")
    return safe_torch_load(matches[0]).float()


def _build_panel(rows: list[list[Image.Image]]) -> Image.Image:
    cell_w = max(tile.width for row in rows for tile in row)
    cell_h = max(tile.height for row in rows for tile in row)
    cols = max(len(row) for row in rows)
    canvas = Image.new("RGB", (cols * cell_w, len(rows) * cell_h), color=(8, 8, 8))
    for row_idx, row in enumerate(rows):
        for col_idx, tile in enumerate(row):
            resized = tile.resize((cell_w, cell_h), Image.BILINEAR)
            canvas.paste(resized, (col_idx * cell_w, row_idx * cell_h))
    return canvas


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(source_dir, IMAGE_PATTERNS)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {source_dir}")

    selected_indices = _choose_indices(len(image_paths), args.num_samples, args.sample_indices)
    selected_paths = [image_paths[idx] for idx in selected_indices]

    extractor = DualScaleRADIOExtractor(
        device=args.device,
        radio_repo=args.radio_repo,
        shallow_block=args.shallow_block,
    )
    to_tensor = transforms.ToTensor()

    raw_geo_features: list[torch.Tensor] = []
    raw_sem_features: list[torch.Tensor] = []
    cached_geo_features: list[torch.Tensor] = []
    cached_sem_features: list[torch.Tensor] = []
    records: list[dict[str, object]] = []

    feature_dir = Path(args.feature_dir) if args.feature_dir else None
    for index, image_path in zip(selected_indices, selected_paths):
        image = Image.open(image_path).convert("RGB")
        tensor = to_tensor(image).unsqueeze(0)
        result = extractor.extract(tensor)
        raw_geo = result["geo"].cpu()
        raw_sem = result["sem"].cpu()
        raw_geo_features.append(raw_geo)
        raw_sem_features.append(raw_sem)

        record = {
            "index": index,
            "image_path": str(image_path),
            "relative_path": image_path.relative_to(source_dir).as_posix(),
            "raw_geo": raw_geo,
            "raw_sem": raw_sem,
        }

        if feature_dir is not None:
            cached_geo = _load_cached_feature(feature_dir, "fine_geo", index)
            cached_sem = _load_cached_feature(feature_dir, "coarse_sem", index)
            cached_geo_features.append(cached_geo)
            cached_sem_features.append(cached_sem)
            record["cached_geo"] = cached_geo
            record["cached_sem"] = cached_sem

        records.append(record)

    raw_geo_stats = _fit_pca_stats(raw_geo_features)
    raw_sem_stats = _fit_pca_stats(raw_sem_features)
    cached_geo_stats = _fit_pca_stats(cached_geo_features) if cached_geo_features else None
    cached_sem_stats = _fit_pca_stats(cached_sem_features) if cached_sem_features else None

    contact_tiles: list[Image.Image] = []
    manifest: list[dict[str, object]] = []

    for record in records:
        rgb_tile = _annotate(_rgb_to_array(Path(record["image_path"])), f"rgb #{record['index']}")
        raw_geo_tile = _annotate(
            _feature_to_rgb(record["raw_geo"], *raw_geo_stats),
            "teacher fine raw",
        )
        raw_sem_tile = _annotate(
            _feature_to_rgb(record["raw_sem"], *raw_sem_stats),
            "teacher coarse raw",
        )

        top_row = [rgb_tile, raw_geo_tile, raw_sem_tile]
        bottom_row: list[Image.Image] = []
        if cached_geo_stats is not None and cached_sem_stats is not None:
            bottom_row.extend(
                [
                    _annotate(_feature_to_rgb(record["cached_geo"], *cached_geo_stats), "teacher fine pca64"),
                    _annotate(_feature_to_rgb(record["cached_sem"], *cached_sem_stats), "teacher coarse pca64"),
                ]
            )

        panel = _build_panel([top_row, bottom_row] if bottom_row else [top_row])
        panel_path = output_dir / f"teacher_{int(record['index']):05d}.png"
        panel.save(panel_path)
        contact_tiles.append(panel.resize((900, int(round(panel.height * (900.0 / panel.width)))), Image.BILINEAR))
        manifest.append(
            {
                "index": int(record["index"]),
                "image": str(record["relative_path"]),
                "panel": str(panel_path),
                "cached_features": feature_dir is not None,
            }
        )

    if contact_tiles:
        total_height = sum(tile.height for tile in contact_tiles)
        contact_sheet = Image.new("RGB", (900, total_height), color=(12, 12, 12))
        cursor = 0
        for tile in contact_tiles:
            contact_sheet.paste(tile, (0, cursor))
            cursor += tile.height
        contact_sheet.save(output_dir / "contact_sheet.png")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(manifest)} teacher visualization panels to {output_dir}")


if __name__ == "__main__":
    main()