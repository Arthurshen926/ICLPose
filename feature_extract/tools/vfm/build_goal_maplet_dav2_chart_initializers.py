"""Build source-only DAV2 chart initializers from a sealed MASt3R scene."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tree_sha(root: Path) -> str:
    rows = []
    for path in sorted(value for value in Path(root).rglob("*") if value.is_file()):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size": int(stat.st_size),
                "sha256": _sha(path),
            }
        )
    return _canonical(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posed_colmap", type=Path, required=True)
    parser.add_argument("--mast3r_source_run", type=Path, required=True)
    parser.add_argument("--disjoint_upstream_authority", type=Path, required=True)
    parser.add_argument("--matcha_repo", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder", default="vitl")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to reuse DAV2 initializer directory")
    authority = json.loads(args.disjoint_upstream_authority.read_text())
    claimed = authority.pop("content_sha256", None)
    if claimed != _canonical(authority):
        raise ValueError("disjoint upstream authority content hash differs")
    authority["content_sha256"] = claimed
    if (
        authority.get("artifact_type")
        != "goal_maplet_disjoint_chart_upstream_authority_v2"
        or not authority.get("strict_disjoint_upstream")
        or not authority.get("physical_source_held_input_roots_disjoint")
        or Path(authority["source"]["root"]).resolve()
        != args.mast3r_source_run.resolve()
        or Path(authority["isolated_source_input"]["root"]).resolve()
        != args.posed_colmap.resolve()
        or authority["isolated_source_input"]["cameras_file_sha256"]
        != _sha(args.posed_colmap / "sparse" / "0" / "cameras.bin")
        or authority["isolated_source_input"]["images_file_sha256"]
        != _sha(args.posed_colmap / "sparse" / "0" / "images.bin")
        or authority["isolated_source_input"]["points3D_file_sha256"]
        != _sha(args.posed_colmap / "sparse" / "0" / "points3D.bin")
        or authority["isolated_source_input"]["ordered_names"]
        != authority["source"]["ordered_names"]
        or authority["source"]["cameras_file_sha256"]
        != _sha(args.mast3r_source_run / "cameras.json")
        or _tree_sha(args.mast3r_source_run) != authority["source"]["tree_sha256"]
    ):
        raise ValueError("DAV2 initializer is not bound to the source-only run")

    sys.path.insert(0, str(args.matcha_repo))
    sys.path.insert(0, str(args.matcha_repo / "Depth-Anything-V2"))
    import torch
    import torch.nn.functional as functional
    from matcha.pointmap.depthanythingv2 import (
        get_pointmap_from_mast3r_scene_with_depthanything,
    )

    checkpoint = (
        args.matcha_repo
        / "Depth-Anything-V2"
        / "checkpoints"
        / f"depth_anything_v2_{args.encoder}.pth"
    )
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("DAV2 chart initialization requires a CUDA device")
    torch.cuda.set_device(device)
    # Initialize the CUDA context before resetting device-local statistics.
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    pointmap = get_pointmap_from_mast3r_scene_with_depthanything(
        scene_source_path=str(args.posed_colmap),
        n_images_in_pointmap=None,
        image_indices=None,
        white_background=False,
        eval_split=False,
        eval_split_interval=8,
        max_img_size=1600,
        pointmap_img_size=512,
        randomize_images=False,
        max_sfm_points=None,
        sfm_confidence_threshold=-1.0,
        average_focal_distances=False,
        mast3r_scene_source_path=str(args.mast3r_source_run),
        depthanything_checkpoint_dir=str(checkpoint.parent),
        depthanything_encoder=args.encoder,
        device=str(device),
        return_sfm_data=False,
    )
    names = [Path(path).name for path in pointmap.img_paths.tolist()]
    if names != authority["source"]["ordered_names"]:
        raise ValueError("DAV2 pointmap inventory differs from source authority")
    points = pointmap.points3d.float().permute(0, 3, 1, 2)
    points = functional.interpolate(points, size=(144, 256), mode="area")
    points = points.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
    poses = pointmap.poses.detach().cpu().numpy().astype(np.float64)
    args.output_dir.mkdir(parents=True)
    rows = []
    for row, name in enumerate(names):
        camera = poses[row]
        depth = ((points[row] - camera[:3, 3]) @ camera[:3, :3])[..., 2]
        valid = np.isfinite(points[row]).all(2) & np.isfinite(depth) & (depth > 0)
        metadata = {
            "artifact_type": "goal_maplet_dav2_chart_initializer_v1",
            "source_name": name,
            "source_only_mast3r_root": str(args.mast3r_source_run.resolve()),
            "isolated_source_posed_colmap_root": str(args.posed_colmap.resolve()),
            "disjoint_upstream_authority_content_sha256": claimed,
            "depth_model": f"DepthAnythingV2-{args.encoder}",
            "depth_checkpoint_file_sha256": _sha(checkpoint),
            "points_semantics": "source_only_MASt3R_metric_fitted_DAV2_world_points",
            "uses_mapping_camera_pose": True,
            "uses_query_or_ground_truth": False,
        }
        metadata["content_sha256"] = _canonical(metadata)
        output = args.output_dir / f"{name}.npz"
        np.savez_compressed(
            output,
            points_world=points[row],
            depth_camera=depth.astype(np.float32),
            valid=valid,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        rows.append(
            {
                "name": name,
                "file_sha256": _sha(output),
                "content_sha256": metadata["content_sha256"],
                "valid_fraction": float(valid.mean()),
            }
        )
    manifest = {
        "artifact_type": "goal_maplet_dav2_chart_initializer_run_v1",
        "source_only_mast3r_root": str(args.mast3r_source_run.resolve()),
        "isolated_source_posed_colmap_root": str(args.posed_colmap.resolve()),
        "isolated_source_images_file_sha256": _sha(
            args.posed_colmap / "sparse" / "0" / "images.bin"
        ),
        "source_only_mast3r_tree_sha256": authority["source"]["tree_sha256"],
        "disjoint_upstream_authority_file_sha256": _sha(
            args.disjoint_upstream_authority
        ),
        "disjoint_upstream_authority_content_sha256": claimed,
        "chart_count": len(rows),
        "depth_checkpoint_file_sha256": _sha(checkpoint),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "uses_query_or_ground_truth": False,
    }
    manifest["content_sha256"] = _canonical(manifest)
    temporary = args.output_dir / "manifest.json.temporary"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, args.output_dir / "manifest.json")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
