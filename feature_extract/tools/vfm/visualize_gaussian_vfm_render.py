"""Visualize a rendered Gaussian VFM feature map with PCA colors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _pca_rgb(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    channels, height, width = features.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    if not np.any(mask):
        return rgb
    pixels = features.reshape(channels, -1).T
    visible = pixels[mask.reshape(-1)]
    visible = visible - visible.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(visible, full_matrices=False)
    basis = vt[:3].T
    projected = visible @ basis
    lo = np.percentile(projected, 1.0, axis=0, keepdims=True)
    hi = np.percentile(projected, 99.0, axis=0, keepdims=True)
    projected = (projected - lo) / np.maximum(hi - lo, 1e-6)
    projected = np.clip(projected, 0.0, 1.0)
    rgb.reshape(-1, 3)[mask.reshape(-1)] = projected.astype(np.float32)
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA-color visualize a Gaussian VFM render NPZ")
    parser.add_argument("--render", required=True)
    parser.add_argument("--output_png", required=True)
    args = parser.parse_args()

    with np.load(Path(args.render)) as data:
        feature_map = np.asarray(data["feature_map"], dtype=np.float32)
        mask = np.asarray(data["visibility_mask"], dtype=bool)
    rgb = _pca_rgb(feature_map, mask)
    output = Path(args.output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb * 255.0, dtype=np.uint8)).save(output)


if __name__ == "__main__":
    main()
