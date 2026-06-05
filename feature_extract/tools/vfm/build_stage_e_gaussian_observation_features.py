"""Build Stage E Gaussian own-observation or hybrid semi-dense features."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.patch_selector_training import load_safe_patch_selector_checkpoint
from feature_extract.vfm.query_to_3d_matching import normalize_rows
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap, semidense_anchor_map_stats
from feature_extract.vfm.tokens import TokenBankManifest


def _project_point(xyz: np.ndarray, pose_w2c: np.ndarray, camera) -> tuple[float, float, bool]:
    point = np.asarray(xyz, dtype=np.float64).reshape(3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_point = pose[:3, :3] @ point + pose[:3, 3]
    if float(camera_point[2]) <= 1e-8:
        return 0.0, 0.0, False
    x = float(camera_point[0] / camera_point[2])
    y = float(camera_point[1] / camera_point[2])
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
        raise ValueError(f"unsupported camera model id: {camera.model_id}")
    visible = 0.0 <= u <= float(camera.width - 1) and 0.0 <= v <= float(camera.height - 1)
    return u, v, bool(visible)


def _bilinear_sample(feature_map: np.ndarray, u: float, v: float, image_width: int, image_height: int) -> np.ndarray:
    features = np.asarray(feature_map, dtype=np.float32)
    channels, height, width = features.shape
    x = float(u) / max(float(image_width - 1), 1.0) * max(width - 1, 0)
    y = float(v) / max(float(image_height - 1), 1.0) * max(height - 1, 0)
    x0 = int(np.floor(np.clip(x, 0.0, max(width - 1, 0))))
    y0 = int(np.floor(np.clip(y, 0.0, max(height - 1, 0))))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    wx = float(x - x0)
    wy = float(y - y0)
    return (
        (1.0 - wx) * (1.0 - wy) * features[:, y0, x0]
        + wx * (1.0 - wy) * features[:, y0, x1]
        + (1.0 - wx) * wy * features[:, y1, x0]
        + wx * wy * features[:, y1, x1]
    ).astype(np.float32)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate Gaussian anchor own-observation VFM features")
    parser.add_argument("--semidense_npz", required=True)
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--source_layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--feature_mode", default="hybrid", choices=("own", "hybrid"))
    parser.add_argument("--hybrid_own_weight", type=float, default=0.5)
    parser.add_argument("--max_gaussian_anchors", type=int, default=0)
    parser.add_argument("--max_views_per_anchor", type=int, default=4)
    parser.add_argument("--min_observations", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_encode_rows", type=int, default=65536)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    anchor_map = SemiDenseAnchorMap.load_npz(Path(args.semidense_npz))
    manifest = TokenBankManifest.from_json(Path(args.reference_manifest))
    manifest.validate(verify_checksums=False)
    record_by_id = {record.image_id: record for record in manifest.records}
    poses = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.reference_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.reference_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    selector = load_safe_patch_selector_checkpoint(Path(args.selector_checkpoint), device=args.device)

    output_features = np.asarray(anchor_map.features, dtype=np.float32).copy()
    output_variances = np.asarray(anchor_map.feature_variances, dtype=np.float32).copy()
    gaussian_rows = np.flatnonzero(np.asarray(anchor_map.source_types, dtype=str) != "sfm")
    if int(args.max_gaussian_anchors) > 0:
        gaussian_rows = gaussian_rows[: int(args.max_gaussian_anchors)]

    feature_cache: dict[str, np.ndarray] = {}
    replaced = 0
    skipped_no_pose = 0
    skipped_no_token = 0
    skipped_no_projection = 0
    observation_counts = []
    cosines = []
    for row in gaussian_rows.tolist():
        raw_observations = []
        image_ids = list(anchor_map.observation_image_ids[int(row)])
        if int(args.max_views_per_anchor) > 0:
            image_ids = image_ids[: int(args.max_views_per_anchor)]
        for image_id in image_ids:
            pose = poses.get(str(image_id))
            if pose is None:
                skipped_no_pose += 1
                continue
            record = record_by_id.get(str(image_id))
            if record is None:
                skipped_no_token += 1
                continue
            u, v, visible = _project_point(anchor_map.xyz[int(row)], pose.pose_w2c, camera)
            if not visible:
                skipped_no_projection += 1
                continue
            if str(record.token_path) not in feature_cache:
                with np.load(record.token_path) as data:
                    if args.source_layer_name not in data:
                        raise ValueError(f"layer {args.source_layer_name!r} not found in {record.token_path}")
                    feature_cache[str(record.token_path)] = np.asarray(data[args.source_layer_name], dtype=np.float32)
            raw_observations.append(
                _bilinear_sample(feature_cache[str(record.token_path)], u, v, int(camera.width), int(camera.height))
            )
        if len(raw_observations) < int(args.min_observations):
            continue
        raw = np.stack(raw_observations, axis=0)
        encoded = selector.encode_rows(raw, device=args.device, batch_size=int(args.batch_encode_rows))
        own = np.mean(encoded, axis=0).astype(np.float32)
        own, _valid = normalize_rows(own.reshape(1, -1))
        own = own.reshape(-1)
        inherited = output_features[int(row)]
        if args.feature_mode == "hybrid":
            mixed = float(args.hybrid_own_weight) * own + (1.0 - float(args.hybrid_own_weight)) * inherited
            mixed, _valid = normalize_rows(mixed.reshape(1, -1))
            output_features[int(row)] = mixed.reshape(-1)
        else:
            output_features[int(row)] = own
        output_variances[int(row)] = float(np.mean(np.var(encoded, axis=0))) if encoded.shape[0] > 1 else 0.0
        replaced += 1
        observation_counts.append(len(raw_observations))
        cosines.append(float(np.dot(inherited, own) / max(float(np.linalg.norm(inherited) * np.linalg.norm(own)), 1e-8)))

    output_map = replace(
        anchor_map,
        features=output_features,
        feature_variances=output_variances,
        metadata={
            **dict(anchor_map.metadata or {}),
            "stage_e_feature_mode": args.feature_mode,
            "stage_e_gaussian_own_observation_replaced": int(replaced),
        },
    )
    output_path = Path(args.output_npz)
    output_map.save_npz(output_path)
    summary = {
        "stage": "stage_e_gaussian_own_observation_feature_aggregation",
        "feature_mode": args.feature_mode,
        "hybrid_own_weight": float(args.hybrid_own_weight),
        "camera_source": camera_source,
        "candidate_gaussian_anchor_count": int(len(gaussian_rows)),
        "replaced_gaussian_anchor_count": int(replaced),
        "mean_observations_per_replaced_anchor": 0.0 if not observation_counts else float(np.mean(observation_counts)),
        "mean_inherited_own_cosine": None if not cosines else float(np.mean(cosines)),
        "median_inherited_own_cosine": None if not cosines else float(np.median(cosines)),
        "skipped_no_pose": int(skipped_no_pose),
        "skipped_no_token": int(skipped_no_token),
        "skipped_no_projection": int(skipped_no_projection),
        "output_stats": semidense_anchor_map_stats(
            output_map,
            sparse_landmark_count=int(np.sum(np.asarray(output_map.source_types, dtype=str) == "sfm")),
            source_gaussian_count=int(np.sum(np.asarray(output_map.source_types, dtype=str) != "sfm")),
        ),
        "inputs": {
            "semidense_npz": args.semidense_npz,
            "reference_manifest": args.reference_manifest,
            "reference_pose_file": args.reference_pose_file,
            "selector_checkpoint": args.selector_checkpoint,
        },
        "outputs": {"semidense_npz": str(output_path)},
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
