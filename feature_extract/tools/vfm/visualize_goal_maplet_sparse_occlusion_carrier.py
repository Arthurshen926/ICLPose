"""Visualize base query planes against observed-only sparse-occlusion carriers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions


def _overlay(image: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    labels = cv2.resize(labels.astype(np.int32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    shown = image.copy().astype(np.float32)
    rng = np.random.default_rng(seed)
    colors = rng.integers(35, 245, size=(max(int(labels.max()) + 1, 1), 3), dtype=np.uint8)
    for row in range(len(colors)):
        mask = labels == row
        shown[mask] = 0.68 * shown[mask] + 0.32 * colors[row]
    support_boundary = cv2.morphologyEx(
        (labels >= 0).astype(np.uint8), cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    region_boundary = np.zeros(labels.shape, bool)
    horizontal = (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] >= 0) & (labels[:, :-1] >= 0)
    vertical = (labels[1:, :] != labels[:-1, :]) & (labels[1:, :] >= 0) & (labels[:-1, :] >= 0)
    region_boundary[:, 1:] |= horizontal
    region_boundary[:, :-1] |= horizontal
    region_boundary[1:, :] |= vertical
    region_boundary[:-1, :] |= vertical
    shown[support_boundary] = np.asarray([255, 230, 0], np.float32)
    shown[region_boundary] = np.asarray([0, 255, 255], np.float32)
    return np.clip(shown, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--base_query_plane_dir", type=Path, nargs="+", required=True)
    parser.add_argument("--carrier_query_plane_dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite sparse-occlusion visualization")
    if len(args.base_query_plane_dir) != len(args.carrier_query_plane_dir):
        raise ValueError("base/carrier route counts differ")
    rows = []
    for base_dir, carrier_dir in zip(args.base_query_plane_dir, args.carrier_query_plane_dir):
        for carrier_path in sorted(carrier_dir.glob("*.npz")):
            base_path = base_dir / carrier_path.name
            base, _ = QueryPlaneRegions.load_npz(base_path)
            carrier, metadata = QueryPlaneRegions.load_npz(carrier_path)
            diagnostic = metadata.get("carrier_diagnostics", {})
            if not np.array_equal(base.labels >= 0, carrier.labels >= 0):
                raise ValueError("carrier visualization found inferred pixels")
            rows.append((
                int(diagnostic.get("merged_component_count", 0)), carrier_path.name,
                base, carrier, diagnostic,
            ))
    rows = sorted(rows, key=lambda row: (-row[0], row[1]))[: int(args.count)]
    if not rows:
        raise ValueError("carrier inventory is empty")
    figure, axes = plt.subplots(len(rows), 2, figsize=(12, 4 * len(rows)), squeeze=False)
    for row_index, (merged, name, base, carrier, diagnostic) in enumerate(rows):
        image_id = name[:-4].replace("__", "/")
        bgr = cv2.imread(str(args.image_root / image_id))
        if bgr is None:
            raise FileNotFoundError(args.image_root / image_id)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        axes[row_index, 0].imshow(_overlay(image, base.labels, seed=17))
        axes[row_index, 1].imshow(_overlay(image, carrier.labels, seed=17))
        axes[row_index, 0].set_title(f"{image_id}: connected base, {len(base.normals_camera)} regions")
        fractions = [
            path["foreground_fraction"] for edge in diagnostic.get("accepted_edges", [])
            for path in edge.get("bridge_paths", []) if path.get("accepted")
        ]
        text = "" if not fractions else f", foreground median={np.median(fractions):.2f}"
        axes[row_index, 1].set_title(
            f"observed-only carrier, {len(carrier.normals_camera)} regions ({merged} merges){text}"
        )
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle(
        "Sparse-foreground carrier: identical observed pixels; only region identity crosses accepted gaps",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.992))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
