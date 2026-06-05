"""Stage F0: diagnose SfM-anchor coverage of VFM-distinctive query patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _limit_submap,
    _load_camera_with_source,
    _load_reference_submaps,
    _load_track_stats,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_to_3d_matching import build_patch_positive_sets, patch_positive_set_stats
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, filter_landmarks_by_reference_images, token_grid_xy
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_aware_landmarks import (
    anchor_coverage_summary,
    flatten_feature_map,
    load_token_feature_map,
    token_distinctiveness_scores,
)


def _mean_numeric(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    summary = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and np.isfinite(float(row[key]))]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Stage F0 VFM-anchor coverage diagnostic")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--submap_top_n", type=int, default=10)
    parser.add_argument("--max_submap_landmarks", type=int, default=20000)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--top_fractions", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--distinctiveness_block_size", type=int, default=1024)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    poses = {record.image_id: record.pose_w2c for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    bank = load_selected_track_bank_npz(Path(args.landmark_bank))
    xyz_by_track, reprojection_error_by_track = _load_track_stats(Path(args.track_observations))
    landmark_index = LandmarkMapIndex.from_track_bank(bank, xyz_by_track, reprojection_error_by_track)
    reference_submaps = _load_reference_submaps(args.candidate_bank, args.submap_top_n) if args.candidate_bank else {}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))

    rows: list[dict[str, object]] = []
    for record in manifest.records:
        if args.max_queries and len(rows) >= int(args.max_queries):
            break
        pose_w2c = poses.get(record.image_id)
        if pose_w2c is None:
            continue
        feature_map = load_token_feature_map(record.token_path, args.layer_name)
        _channels, token_height, token_width = feature_map.shape
        submap = landmark_index
        references = reference_submaps.get(record.image_id)
        if references is not None:
            submap = filter_landmarks_by_reference_images(submap, references)
        submap = _limit_submap(submap, int(args.max_submap_landmarks))
        positives = build_patch_positive_sets(
            submap,
            pose_w2c,
            camera,
            token_width=token_width,
            token_height=token_height,
            patch_scale=float(args.patch_scale),
        )
        has_anchor = np.asarray([positives.by_token[idx].count > 0 for idx in range(token_width * token_height)], dtype=bool)
        saliency = token_distinctiveness_scores(
            flatten_feature_map(feature_map),
            block_size=int(args.distinctiveness_block_size),
        )
        coverage = anchor_coverage_summary(saliency, has_anchor, top_fractions=tuple(args.top_fractions))
        patch_stats = patch_positive_set_stats(positives)
        xy = token_grid_xy(token_width, token_height, camera.width, camera.height)
        high_missing = []
        order = np.argsort(-saliency, kind="mergesort")
        for token_idx in order[: min(20, len(order))]:
            if not bool(has_anchor[int(token_idx)]):
                high_missing.append(
                    {
                        "token_index": int(token_idx),
                        "xy": [float(xy[int(token_idx), 0]), float(xy[int(token_idx), 1])],
                        "saliency": float(saliency[int(token_idx)]),
                    }
                )
        row = {
            "query_id": record.image_id,
            "submap_landmark_count": int(len(submap)),
            "token_width": int(token_width),
            "token_height": int(token_height),
            **coverage,
            **{f"patch_{key}": value for key, value in patch_stats.items()},
            "top20_missing_anchor_examples": high_missing[:10],
        }
        rows.append(row)

    output = {
        "stage": "stage_f0_vfm_anchor_coverage",
        "query_count": int(len(rows)),
        "camera_source": camera_source,
        "config": {
            "patch_scale": float(args.patch_scale),
            "top_fractions": [float(item) for item in args.top_fractions],
            "submap_top_n": int(args.submap_top_n),
            "max_submap_landmarks": int(args.max_submap_landmarks),
            "distinctiveness_block_size": int(args.distinctiveness_block_size),
        },
        "inputs": {
            "query_manifest": str(args.query_manifest),
            "landmark_bank": str(args.landmark_bank),
            "track_observations": str(args.track_observations),
            "query_pose_file": str(args.query_pose_file),
            "candidate_bank": str(args.candidate_bank),
        },
        "aggregate": _mean_numeric(rows),
        "rows": rows,
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
