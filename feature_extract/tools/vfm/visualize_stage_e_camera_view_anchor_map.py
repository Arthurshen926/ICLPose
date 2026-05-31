"""Render Stage E sparse vs semi-dense anchor maps from query camera views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex
from feature_extract.vfm.semidense_anchor_map import (
    SemiDenseAnchorMap,
    write_semidense_anchor_camera_view_visualization,
)
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def _load_track_stats(track_observations: Path) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    xyz_by_track: dict[int, np.ndarray] = {}
    reproj_by_track: dict[int, list[float]] = {}
    for obs in load_colmap_track_observations_jsonl(Path(track_observations)):
        xyz_by_track.setdefault(int(obs.track_id), np.asarray(obs.xyz, dtype=np.float64).reshape(3))
        reproj_by_track.setdefault(int(obs.track_id), []).append(float(obs.reprojection_error))
    return xyz_by_track, {
        int(track_id): float(np.mean(values)) for track_id, values in reproj_by_track.items() if values
    }


def _read_image_rgb(path: Path) -> np.ndarray | None:
    if not Path(path).exists():
        return None
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for image-backed camera-view visualization") from exc
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _safe_name(image_id: str) -> str:
    return str(image_id).replace("/", "__").replace("\\", "__").replace(" ", "_")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render Stage E anchor maps in query camera view")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--sparse_landmark_bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--semidense_npz", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--query_ids", default="")
    parser.add_argument("--max_queries", type=int, default=4)
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--max_points", type=int, default=50000)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--max_output_width", type=int, default=1920)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    poses = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))

    xyz_by_track, reproj_by_track = _load_track_stats(Path(args.track_observations))
    sparse_index = LandmarkMapIndex.from_track_bank(
        load_selected_track_bank_npz(Path(args.sparse_landmark_bank)),
        xyz_by_track,
        reproj_by_track,
    )
    semidense_map = SemiDenseAnchorMap.load_npz(Path(args.semidense_npz))

    requested_ids = [item.strip() for item in str(args.query_ids).split(",") if item.strip()]
    requested = set(requested_ids)
    records = [
        record
        for record in manifest.records
        if (not requested or record.image_id in requested) and record.image_id in poses
    ]
    if not requested and int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root) if args.image_root else None
    rows = []
    for record in records:
        image_rgb = None if image_root is None else _read_image_rgb(image_root / record.image_id)
        output_png = output_dir / f"{_safe_name(record.image_id)}_sparse_vs_semidense_camera.png"
        write_semidense_anchor_camera_view_visualization(
            output_png,
            sparse_index=sparse_index,
            semidense_map=semidense_map,
            pose_w2c=poses[record.image_id].pose_w2c,
            camera=camera,
            image_rgb=image_rgb,
            max_points=int(args.max_points),
            point_radius=int(args.point_radius),
            max_output_width=int(args.max_output_width),
        )
        rows.append(
            {
                "query_id": record.image_id,
                "output_png": str(output_png),
                "image_path": None if image_root is None else str(image_root / record.image_id),
                "image_loaded": image_rgb is not None,
            }
        )

    summary = {
        "stage": "stage_e_camera_view_anchor_visualization",
        "query_count": len(rows),
        "camera_source": camera_source,
        "sparse_anchor_count": int(len(sparse_index)),
        "semidense_anchor_count": int(len(semidense_map)),
        "inputs": {
            "query_manifest": args.query_manifest,
            "query_pose_file": args.query_pose_file,
            "sparse_landmark_bank": args.sparse_landmark_bank,
            "track_observations": args.track_observations,
            "semidense_npz": args.semidense_npz,
            "image_root": args.image_root,
        },
        "outputs": rows,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
