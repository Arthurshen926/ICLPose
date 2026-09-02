"""Pose-free MoGe-3 query geometry from a camera-only Cambridge inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _focal(row: dict[str, object]) -> float:
    model = int(row["model_id"])
    params = np.asarray(row["params"], np.float64)
    if model in (0, 2):
        return float(params[0])
    if model == 1:
        return float(0.5 * (params[0] + params[1]))
    raise ValueError("unsupported camera model for MoGe3 query inference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera_manifest", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_id", default="Ruicheng/moge-3-vitl")
    parser.add_argument("--resolution_level", type=int, default=5)
    parser.add_argument("--refine_steps", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to reuse MoGe3 query geometry directory")
    camera_payload = json.loads(args.camera_manifest.read_text())
    if (
        camera_payload.get("production_contract", {}).get("contains_camera_pose") is not False
        or camera_payload.get("production_contract", {}).get("contains_sfm_points") is not False
    ):
        raise ValueError("query MoGe3 requires a camera-only authority")
    cameras = camera_payload["cameras"]
    inventory = [
        (name, cameras[name]) for name in sorted(cameras)
        if name.split("/", 1)[0] == args.route
    ]
    if not inventory:
        raise ValueError("requested route is absent from camera manifest")

    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from moge.model.v3 import MoGeModel

    model = MoGeModel.from_pretrained(args.model_id).to(args.device).eval()
    snapshot = Path(snapshot_download(args.model_id, local_files_only=True))
    weight = snapshot / "model.pt"
    args.output_dir.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats(args.device)
    rows = []
    for index, (image_id, camera) in enumerate(inventory):
        image_path = args.image_root / image_id
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        width, height = int(camera["width"]), int(camera["height"])
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        focal = _focal(camera)
        fov = math.degrees(2.0 * math.atan(width / (2.0 * focal)))
        tensor = torch.as_tensor(rgb, dtype=torch.float32, device=args.device).permute(2, 0, 1) / 255.0
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model.infer(
                tensor, resolution_level=args.resolution_level, fov_x=fov,
                use_fp16=True, apply_mask=True, refine_steps=args.refine_steps,
            )

        def resize(value, channels: bool, mode: str = "bilinear"):
            data = value.float()
            data = data.permute(2, 0, 1)[None] if channels else data[None, None]
            data = F.interpolate(
                data, size=(144, 256), mode=mode,
                align_corners=False if mode != "nearest" else None,
            )
            return (
                data[0].permute(1, 2, 0).cpu().numpy()
                if channels else data[0, 0].cpu().numpy()
            )

        points = resize(prediction["points"], True).astype(np.float32)
        depth = resize(prediction["depth"], False).astype(np.float32)
        normal = resize(prediction["normal"], True).astype(np.float32)
        normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-8)
        valid = resize(prediction["mask"].float(), False, "nearest") > 0.5
        valid &= np.isfinite(points).all(2) & np.isfinite(depth) & (depth > 0)
        points[~valid] = np.nan
        depth[~valid] = np.nan
        normal[~valid] = 0
        metadata = {
            "artifact_type": "goal_maplet_moge_query_geometry_v2",
            "image_id": image_id,
            "source_image_file_sha256": _sha(image_path),
            "camera_manifest_file_sha256": _sha(args.camera_manifest),
            "camera_model_id": int(camera["model_id"]),
            "camera_width": width,
            "camera_height": height,
            "camera_focal_px": focal,
            "fov_x_degrees": fov,
            "model_id": args.model_id,
            "resolution_level": int(args.resolution_level),
            "refine_steps": int(args.refine_steps),
            "use_fp16": True,
            "uses_pose": False,
            "uses_ground_truth": False,
        }
        metadata["content_sha256"] = _canonical(metadata)
        output = args.output_dir / (image_id.replace("/", "__") + ".npz")
        np.savez_compressed(
            output, points_camera=points, depth_camera=depth,
            normal_camera=normal, valid=valid,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        rows.append({
            "image_id": image_id, "path": str(output.resolve()),
            "file_sha256": _sha(output), "content_sha256": metadata["content_sha256"],
            "valid_fraction": float(np.mean(valid)),
            "median_depth_m": float(np.nanmedian(depth)),
            "inference_and_resize_seconds": time.perf_counter() - started,
        })
        print(f"{index + 1}/{len(inventory)} {image_id}", flush=True)
    manifest = {
        "artifact_type": "goal_maplet_moge_query_geometry_v2_manifest",
        "route": args.route,
        "query_count": len(rows),
        "camera_manifest": str(args.camera_manifest.resolve()),
        "camera_manifest_file_sha256": _sha(args.camera_manifest),
        "image_root": str(args.image_root.resolve()),
        "image_layout": "hierarchical",
        "model_id": args.model_id,
        "model_snapshot": str(snapshot),
        "model_weight_file_sha256": _sha(weight),
        "resolution_level": int(args.resolution_level),
        "refine_steps": int(args.refine_steps),
        "use_fp16": True,
        "resize_source_to_camera_canvas": True,
        "output_height": 144,
        "output_width": 256,
        "uses_pose": False,
        "uses_ground_truth": False,
        "production_eligible": False,
        "rows": rows,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
    }
    manifest["content_sha256"] = _canonical(manifest)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
