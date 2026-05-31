"""Evaluate patch-positive coverage for Stage E semi-dense anchor maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _load_query_feature,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def _track_stats(track_observations: Path) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    xyz_by_track: dict[int, np.ndarray] = {}
    reproj_by_track: dict[int, list[float]] = {}
    for obs in load_colmap_track_observations_jsonl(Path(track_observations)):
        xyz_by_track.setdefault(int(obs.track_id), np.asarray(obs.xyz, dtype=np.float64).reshape(3))
        reproj_by_track.setdefault(int(obs.track_id), []).append(float(obs.reprojection_error))
    return xyz_by_track, {
        int(track_id): float(np.mean(values)) for track_id, values in reproj_by_track.items() if values
    }


def _mean(rows: list[dict[str, object]], section: str, key: str) -> float:
    values = [float(row[section][key]) for row in rows if row.get(section) is not None]
    return float(np.mean(values)) if values else 0.0


def _project_landmarks_fast(index: LandmarkMapIndex, pose_w2c: np.ndarray, camera) -> tuple[np.ndarray, np.ndarray]:
    """Project landmarks for coverage diagnostics without OpenCV Jacobian overhead."""

    if len(index) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    points = np.asarray(index.xyz, dtype=np.float64).reshape(-1, 3)
    camera_points = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T
    z = camera_points[:, 2]
    visible = z > 1e-8
    safe_z = np.where(visible, z, 1.0)
    x = camera_points[:, 0] / safe_z
    y = camera_points[:, 1] / safe_z
    if int(camera.model_id) == 0:
        f, cx, cy = camera.params[:3]
        u = float(f) * x + float(cx)
        v = float(f) * y + float(cy)
    elif int(camera.model_id) == 1:
        fx, fy, cx, cy = camera.params[:4]
        u = float(fx) * x + float(cx)
        v = float(fy) * y + float(cy)
    elif int(camera.model_id) == 2:
        f, cx, cy, k = camera.params[:4]
        radial = 1.0 + float(k) * (x * x + y * y)
        u = float(f) * x * radial + float(cx)
        v = float(f) * y * radial + float(cy)
    else:
        raise ValueError(f"unsupported camera model id for Stage E fast coverage: {camera.model_id}")
    xy = np.stack([u, v], axis=1).astype(np.float64)
    visible &= (xy[:, 0] >= 0.0) & (xy[:, 0] <= float(camera.width - 1))
    visible &= (xy[:, 1] >= 0.0) & (xy[:, 1] <= float(camera.height - 1))
    return xy, visible


def _fast_patch_positive_stats(
    index: LandmarkMapIndex,
    pose_w2c: np.ndarray,
    camera,
    token_width: int,
    token_height: int,
    patch_scale: float,
) -> dict[str, float | int]:
    token_count = int(token_width) * int(token_height)
    if token_count <= 0:
        return {
            "token_count": 0,
            "visible_landmark_count": 0,
            "positive_landmark_count": 0,
            "mean_positives_per_token": 0.0,
            "median_positives_per_token": 0.0,
            "mean_positives_per_nonempty_token": 0.0,
            "max_positives_per_token": 0,
            "zero_positive_token_ratio": 1.0,
            "nonempty_patch_fraction": 0.0,
            "positive_landmark_density_per_token": 0.0,
        }
    xy, visible = _project_landmarks_fast(index, pose_w2c, camera)
    visible_xy = xy[visible]
    visible_count = int(visible_xy.shape[0])
    counts = np.zeros((token_count,), dtype=np.int64)
    if visible_count > 0:
        stride_x = float(camera.width - 1) / max(float(token_width - 1), 1.0)
        stride_y = float(camera.height - 1) / max(float(token_height - 1), 1.0)
        half_x = 0.5 * stride_x * float(patch_scale)
        half_y = 0.5 * stride_y * float(patch_scale)
        x_idx = np.rint(
            np.clip(visible_xy[:, 0] / max(float(camera.width - 1), 1.0), 0.0, 1.0) * max(token_width - 1, 0)
        ).astype(np.int64)
        y_idx = np.rint(
            np.clip(visible_xy[:, 1] / max(float(camera.height - 1), 1.0), 0.0, 1.0) * max(token_height - 1, 0)
        ).astype(np.int64)
        assigned = np.zeros((visible_count,), dtype=bool)
        search_radius = max(1, int(np.ceil(float(patch_scale))) + 1)
        for dy in range(-search_radius, search_radius + 1):
            yy = y_idx + dy
            y_valid = (yy >= 0) & (yy < int(token_height))
            if not np.any(y_valid):
                continue
            center_y = yy.astype(np.float64) * stride_y
            y_contains = y_valid & (np.abs(visible_xy[:, 1] - center_y) <= half_y)
            if not np.any(y_contains):
                continue
            for dx in range(-search_radius, search_radius + 1):
                xx = x_idx + dx
                x_valid = (xx >= 0) & (xx < int(token_width))
                if not np.any(x_valid):
                    continue
                center_x = xx.astype(np.float64) * stride_x
                mask = y_contains & x_valid & (np.abs(visible_xy[:, 0] - center_x) <= half_x)
                if not np.any(mask):
                    continue
                token_indices = yy[mask] * int(token_width) + xx[mask]
                np.add.at(counts, token_indices.astype(np.int64), 1)
                assigned[mask] = True
        positive_landmark_count = int(np.count_nonzero(assigned))
    else:
        positive_landmark_count = 0
    nonempty = counts[counts > 0]
    return {
        "token_count": int(token_count),
        "visible_landmark_count": int(visible_count),
        "positive_landmark_count": int(positive_landmark_count),
        "mean_positives_per_token": float(np.mean(counts)),
        "median_positives_per_token": float(np.median(counts)),
        "mean_positives_per_nonempty_token": 0.0 if nonempty.size == 0 else float(np.mean(nonempty)),
        "max_positives_per_token": int(np.max(counts)) if counts.size else 0,
        "zero_positive_token_ratio": float(np.mean(counts == 0)),
        "nonempty_patch_fraction": float(np.mean(counts > 0)),
        "positive_landmark_density_per_token": float(positive_landmark_count / max(token_count, 1)),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare sparse vs semi-dense patch positive coverage")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--sparse_landmark_bank", required=True)
    parser.add_argument("--semidense_npz", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    xyz_by_track, reproj_by_track = _track_stats(Path(args.track_observations))
    sparse = LandmarkMapIndex.from_track_bank(
        load_selected_track_bank_npz(Path(args.sparse_landmark_bank)),
        xyz_by_track,
        reproj_by_track,
    )
    semidense = SemiDenseAnchorMap.load_npz(Path(args.semidense_npz)).to_landmark_index()
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    poses = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]

    rows = []
    for record in records:
        pose = poses.get(record.image_id)
        if pose is None:
            continue
        feature = _load_query_feature(record.token_path, args.layer_name)
        _channels, token_height, token_width = feature.shape
        sparse_stats = _fast_patch_positive_stats(
            sparse, pose.pose_w2c, camera, token_width, token_height, float(args.patch_scale)
        )
        semidense_stats = _fast_patch_positive_stats(
            semidense, pose.pose_w2c, camera, token_width, token_height, float(args.patch_scale)
        )
        rows.append(
            {
                "query_id": record.image_id,
                "sparse": sparse_stats,
                "semidense": semidense_stats,
                "positive_density_gain": float(
                    semidense_stats["positive_landmark_density_per_token"]
                    / max(float(sparse_stats["positive_landmark_density_per_token"]), 1e-12)
                ),
                "zero_positive_delta": float(
                    float(semidense_stats["zero_positive_token_ratio"])
                    - float(sparse_stats["zero_positive_token_ratio"])
                ),
            }
        )
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))
    summary = {
        "stage": "stage_e_semidense_patch_coverage",
        "query_count": int(len(rows)),
        "camera_source": camera_source,
        "sparse_anchor_count": int(len(sparse)),
        "semidense_anchor_count": int(len(semidense)),
        "mean_sparse_visible_landmarks": _mean(rows, "sparse", "visible_landmark_count"),
        "mean_semidense_visible_landmarks": _mean(rows, "semidense", "visible_landmark_count"),
        "mean_sparse_positive_density": _mean(rows, "sparse", "positive_landmark_density_per_token"),
        "mean_semidense_positive_density": _mean(rows, "semidense", "positive_landmark_density_per_token"),
        "mean_sparse_zero_positive_ratio": _mean(rows, "sparse", "zero_positive_token_ratio"),
        "mean_semidense_zero_positive_ratio": _mean(rows, "semidense", "zero_positive_token_ratio"),
        "mean_positive_density_gain": float(np.mean([row["positive_density_gain"] for row in rows])) if rows else 0.0,
        "mean_zero_positive_delta": float(np.mean([row["zero_positive_delta"] for row in rows])) if rows else 0.0,
        "inputs": {
            "query_manifest": args.query_manifest,
            "sparse_landmark_bank": args.sparse_landmark_bank,
            "semidense_npz": args.semidense_npz,
            "track_observations": args.track_observations,
            "query_pose_file": args.query_pose_file,
        },
        "outputs": {"rows": str(output_jsonl)},
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
