"""Visualize raw VFM 3D landmark banks as colored point clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, load_selected_track_bank_npz
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def _track_xyz_index(track_observations_path: Path) -> dict[int, np.ndarray]:
    index: dict[int, np.ndarray] = {}
    for obs in load_colmap_track_observations_jsonl(track_observations_path):
        index.setdefault(int(obs.track_id), np.asarray(obs.xyz, dtype=np.float64).reshape(3))
    return index


def _aligned_arrays(
    bank: SelectedTrackFeatureBank,
    xyz_index: dict[int, np.ndarray],
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    track_ids = [track_id for track_id in sorted(bank.tracks) if track_id in xyz_index]
    if max_points > 0 and len(track_ids) > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(track_ids), size=max_points, replace=False))
        track_ids = [track_ids[int(idx)] for idx in keep]
    if not track_ids:
        raise ValueError("no landmark tracks overlap between bank and track observations")
    xyz = np.stack([xyz_index[track_id] for track_id in track_ids], axis=0).astype(np.float64)
    features = np.stack([bank.tracks[track_id].mean_feature for track_id in track_ids], axis=0).astype(np.float32)
    variances = np.asarray([bank.tracks[track_id].mean_variance for track_id in track_ids], dtype=np.float32)
    return np.asarray(track_ids, dtype=np.int64), xyz, features, variances


def _normalize_colors(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    channels = []
    for idx in range(array.shape[1]):
        channel = array[:, idx]
        low, high = np.percentile(channel, [1.0, 99.0])
        if high - low < 1e-12:
            channels.append(np.full(channel.shape, 127.0, dtype=np.float64))
        else:
            channels.append(np.clip((channel - low) / (high - low), 0.0, 1.0) * 255.0)
    color = np.stack(channels, axis=1)
    if color.shape[1] == 1:
        color = np.concatenate([color, np.zeros_like(color), 255.0 - color], axis=1)
    while color.shape[1] < 3:
        color = np.concatenate([color, np.zeros((color.shape[0], 1), dtype=np.float64)], axis=1)
    return np.asarray(np.round(color[:, :3]), dtype=np.uint8)


def _pca_colors(features: np.ndarray) -> np.ndarray:
    centered = features.astype(np.float64) - np.mean(features.astype(np.float64), axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return np.full((centered.shape[0], 3), 127, dtype=np.uint8)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = centered @ vt[: min(3, vt.shape[0])].T
    return _normalize_colors(components)


def _variance_colors(variances: np.ndarray) -> np.ndarray:
    scalar = np.asarray(variances, dtype=np.float64).reshape(-1, 1)
    color = _normalize_colors(scalar)
    return np.stack([color[:, 0], np.zeros_like(color[:, 0]), 255 - color[:, 0]], axis=1).astype(np.uint8)


def _write_ascii_ply(path: Path, xyz: np.ndarray, colors: np.ndarray, track_ids: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property int track_id",
        "end_header",
    ]
    for point, color, track_id in zip(xyz, colors, track_ids):
        lines.append(
            f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
            f"{int(color[0])} {int(color[1])} {int(color[2])} {int(track_id)}"
        )
    output.write_text("\n".join(lines) + "\n")


def _write_projection_png(path: Path, xyz: np.ndarray, colors: np.ndarray) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    ax.scatter(xyz[:, 0], xyz[:, 2], c=colors.astype(np.float32) / 255.0, s=0.4, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Raw VFM Landmark Feature PCA Color")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return True


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Write raw VFM landmark feature visualization PLY files")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--output_pca_ply", required=True)
    parser.add_argument("--output_variance_ply", required=True)
    parser.add_argument("--output_pca_png", default="")
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    bank_path = Path(args.bank)
    tracks_path = Path(args.track_observations)
    bank = load_selected_track_bank_npz(bank_path)
    track_ids, xyz, features, variances = _aligned_arrays(
        bank,
        _track_xyz_index(tracks_path),
        max_points=args.max_points,
        seed=args.seed,
    )
    pca_colors = _pca_colors(features)
    variance_colors = _variance_colors(variances)
    _write_ascii_ply(Path(args.output_pca_ply), xyz, pca_colors, track_ids)
    _write_ascii_ply(Path(args.output_variance_ply), xyz, variance_colors, track_ids)
    png_written = False
    if args.output_pca_png:
        png_written = _write_projection_png(Path(args.output_pca_png), xyz, pca_colors)

    summary = {
        "visualized_track_count": int(track_ids.shape[0]),
        "feature_dim": int(features.shape[1]),
        "mean_variance": float(np.mean(variances)),
        "max_variance": float(np.max(variances)),
        "pca_ply": args.output_pca_ply,
        "variance_ply": args.output_variance_ply,
        "pca_png": args.output_pca_png if png_written else "",
        "input_files": {
            "bank": {"path": str(bank_path), "sha256": file_sha256_short(bank_path)},
            "track_observations": {"path": str(tracks_path), "sha256": file_sha256_short(tracks_path)},
        },
    }
    output_summary = Path(args.summary_json)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
