"""Build pose-free MoGe-2 metric query geometry for a named Cambridge route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


SCHEMA = "goal_maplet_moge_query_geometry_v2"
DEFAULT_MODEL_ID = "Ruicheng/moge-2-vits-normal"
MOGE_REPOSITORY_COMMIT = "7807b5de2bc0c1e80519f5f3d1f38a606f8f9925"
MOGE3_REPOSITORY_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"
OUTPUT_WIDTH = 256
OUTPUT_HEIGHT = 144


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera_manifest", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--route", default="seq10")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution_level", type=int, default=5)
    parser.add_argument(
        "--model_id", default=DEFAULT_MODEL_ID,
        choices=(
            "Ruicheng/moge-2-vits-normal", "Ruicheng/moge-2-vitb-normal",
            "Ruicheng/moge-2-vitl-normal", "Ruicheng/moge-3-vitl",
            "Ruicheng/moge-3-vitg",
        ),
    )
    parser.add_argument("--refine_steps", type=int, default=3)
    parser.add_argument("--maximum_queries", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to reuse MoGe query geometry output directory")
    output.mkdir(parents=True)
    camera_path = Path(args.camera_manifest).resolve()
    image_root = Path(args.image_root).resolve()
    cameras = json.loads(camera_path.read_text())["cameras"]
    prefix = str(args.route).rstrip("/") + "/"
    image_ids = sorted(value for value in cameras if value.startswith(prefix))
    if int(args.maximum_queries) > 0:
        image_ids = image_ids[:int(args.maximum_queries)]
    if not image_ids:
        raise ValueError("selected route contains no camera rows")

    import torch
    import torch.nn.functional as functional
    from huggingface_hub import snapshot_download
    is_v3 = "/moge-3-" in args.model_id
    if is_v3:
        from moge.model.v3 import MoGeModel
    else:
        from moge.model.v2 import MoGeModel
    repository_commit = MOGE3_REPOSITORY_COMMIT if is_v3 else MOGE_REPOSITORY_COMMIT

    model = MoGeModel.from_pretrained(args.model_id).to(args.device).eval()
    model_snapshot = Path(snapshot_download(args.model_id, local_files_only=True)).resolve()
    model_weight = model_snapshot / "model.pt"
    if not model_weight.is_file():
        raise FileNotFoundError("MoGe snapshot misses model.pt")
    model_weight_sha256 = _sha256(model_weight)
    torch.cuda.reset_peak_memory_stats(args.device)
    rows = []
    for index, image_id in enumerate(image_ids):
        image_path = image_root / image_id.replace("/", "__")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to decode {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        camera = cameras[image_id]
        width, height = int(camera["width"]), int(camera["height"])
        if rgb.shape[:2] != (height, width):
            raise ValueError("image and frozen camera canvas differ")
        focal = float(camera["params"][0])
        fov_x = math.degrees(2.0 * math.atan(width / (2.0 * focal)))
        tensor = torch.as_tensor(rgb, dtype=torch.float32, device=args.device).permute(2, 0, 1) / 255.0
        started = time.perf_counter()
        with torch.inference_mode():
            infer_options = {
                "resolution_level": int(args.resolution_level), "fov_x": fov_x,
                "use_fp16": True, "apply_mask": True,
            }
            if is_v3:
                infer_options["refine_steps"] = int(args.refine_steps)
            prediction = model.infer(tensor, **infer_options)
        def resize(value, channels: bool, mode: str = "bilinear"):
            item = value.float()
            if channels:
                item = item.permute(2, 0, 1)[None]
            else:
                item = item[None, None]
            item = functional.interpolate(
                item, size=(OUTPUT_HEIGHT, OUTPUT_WIDTH), mode=mode,
                align_corners=False if mode != "nearest" else None,
            )
            return item[0].permute(1, 2, 0).cpu().numpy() if channels else item[0, 0].cpu().numpy()
        points = resize(prediction["points"], True).astype(np.float32)
        depth = resize(prediction["depth"], False).astype(np.float32)
        normal = resize(prediction["normal"], True).astype(np.float32)
        normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1.0e-8)
        mask = resize(prediction["mask"].float(), False, "nearest") > 0.5
        valid = mask & np.isfinite(points).all(axis=2) & np.isfinite(depth) & (depth > 0.0)
        points[~valid] = np.nan
        depth[~valid] = np.nan
        normal[~valid] = 0.0
        destination = output / f"{image_id.replace('/', '__')}.npz"
        metadata = {
            "artifact_type": SCHEMA, "image_id": image_id, "model_id": args.model_id,
            "moge_repository_commit": repository_commit,
            "resolution_level": int(args.resolution_level), "fov_x_deg": fov_x,
            "refine_steps": int(args.refine_steps) if is_v3 else 0,
            "source_image": str(image_path), "source_image_file_sha256": _sha256(image_path),
            "camera_model_id": int(camera["model_id"]), "camera_width": width,
            "camera_height": height, "camera_params": camera["params"],
            "output_width": OUTPUT_WIDTH, "output_height": OUTPUT_HEIGHT,
            "uses_pose": False, "uses_ground_truth": False,
        }
        metadata["content_sha256"] = _canonical_sha(metadata)
        np.savez_compressed(
            destination, points_camera=points, depth_camera=depth, normal_camera=normal,
            valid=valid, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        rows.append({
            "image_id": image_id, "path": str(destination),
            "file_sha256": _sha256(destination), "content_sha256": metadata["content_sha256"],
            "valid_fraction": float(np.mean(valid)),
            "median_depth_m": float(np.nanmedian(depth)),
            "inference_and_resize_seconds": float(time.perf_counter() - started),
        })
        print(f"{index + 1}/{len(image_ids)} {image_id}", flush=True)
    manifest = {
        "artifact_type": SCHEMA + "_manifest", "model_id": args.model_id,
        "moge_repository_commit": repository_commit,
        "model_snapshot": str(model_snapshot), "model_weight_file_sha256": model_weight_sha256,
        "camera_manifest": str(camera_path), "camera_manifest_file_sha256": _sha256(camera_path),
        "image_root": str(image_root), "route": str(args.route), "query_count": len(rows),
        "resolution_level": int(args.resolution_level), "output_width": OUTPUT_WIDTH,
        "output_height": OUTPUT_HEIGHT, "rows": rows,
        "refine_steps": int(args.refine_steps) if is_v3 else 0,
        "uses_pose": False, "uses_ground_truth": False, "production_eligible": False,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
    }
    manifest["content_sha256"] = _canonical_sha(manifest)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "content_sha256": manifest["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
