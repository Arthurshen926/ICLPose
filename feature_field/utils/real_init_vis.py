from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw


def _normalize_rgb(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.0)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def load_rgb_image(path: str, target_hw: Optional[tuple] = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if target_hw is not None:
        image = image.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(image)


def feature_map_to_rgb(feature_map) -> np.ndarray:
    if hasattr(feature_map, "detach"):
        feature_map = feature_map.detach().cpu().float().numpy()
    feat = np.asarray(feature_map, dtype=np.float32)
    if feat.ndim == 4:
        feat = feat[0]
    if feat.ndim != 3:
        raise ValueError(f"Expected feature map [C,H,W], got {feat.shape}")

    channels = feat.shape[0]
    if channels >= 3:
        rgb = np.stack([feat[0], feat[channels // 2], feat[-1]], axis=-1)
    else:
        rgb = np.repeat(feat[:1].transpose(1, 2, 0), 3, axis=2)
    return _normalize_rgb(rgb)


def save_triptych(
    query_rgb: np.ndarray,
    init_rgb: np.ndarray,
    final_rgb: np.ndarray,
    save_path: str,
    caption_lines: Optional[Iterable[str]] = None,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    caption_lines = list(caption_lines) if caption_lines is not None else []
    panels = [Image.fromarray(arr.astype(np.uint8)) for arr in (query_rgb, init_rgb, final_rgb)]
    target_h = max(img.height for img in panels)
    resized = []
    for img in panels:
        if img.height != target_h:
            width = int(round(img.width * (target_h / img.height)))
            img = img.resize((width, target_h), Image.BILINEAR)
        resized.append(img)

    gap = 8
    text_h = 18 * len(caption_lines)
    canvas_w = sum(img.width for img in resized) + gap * 4
    canvas_h = target_h + gap * 2 + text_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))

    x = gap
    for img in resized:
        canvas.paste(img, (x, gap))
        x += img.width + gap

    if caption_lines:
        draw = ImageDraw.Draw(canvas)
        y = target_h + gap + 2
        for line in caption_lines:
            draw.text((gap, y), str(line), fill=(230, 230, 230))
            y += 16

    canvas.save(save_path)
