"""Render a complete mapping-view depth bank from the official 2DGS surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import _camera_from_model
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.official_2dgs_renderer import (
    load_official_2dgs_source_from_ply,
    render_official_2dgs_rgb_depth,
)
from feature_extract.vfm.surface_depth_bank import (
    TWO_DGS_SURFACE_DEPTH_BANK_FORMAT,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _depth_name(index: int, image_id: str) -> str:
    digest = hashlib.sha256(str(image_id).encode("utf8")).hexdigest()[:12]
    return f"{int(index):06d}_{digest}.npy"


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest)
    if manifest_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    records = list(parse_cambridge_pose_file(Path(args.mapping_pose_file)))
    if int(args.max_views) > 0:
        records = records[: int(args.max_views)]
    if not records or len({record.image_id for record in records}) != len(records):
        raise ValueError("mapping pose file contains no unique selected views")
    camera = _camera_from_model(Path(args.camera_model_dir))
    source = load_official_2dgs_source_from_ply(Path(args.ply))
    started = time.time()
    output_records = []
    for index, record in enumerate(records):
        path = output_root / _depth_name(index, record.image_id)
        if path.exists() and not bool(args.force):
            raise FileExistsError(f"refusing to overwrite {path}")
        _rgb, depth, alpha = render_official_2dgs_rgb_depth(
            source,
            pose_w2c=record.pose_w2c,
            camera=camera,
            width=int(camera.width),
            height=int(camera.height),
            device=str(args.device),
        )
        values = np.asarray(depth, dtype=np.float32)
        opacity = np.asarray(alpha, dtype=np.float32)
        valid = (
            np.isfinite(values)
            & (values > 0.0)
            & np.isfinite(opacity)
            & (opacity > 0.2)
        )
        values = np.where(valid, values, np.nan).astype(np.float32)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        temporary.replace(path)
        output_records.append(
            {
                "image_id": str(record.image_id),
                "path": str(path.resolve()),
                "byte_size": int(path.stat().st_size),
                "sha256": file_sha256_short(path),
                "valid_fraction": float(np.mean(valid)),
                "median_depth": float(np.nanmedian(values)),
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(
                json.dumps(
                    {
                        "completed": int(index + 1),
                        "total": int(len(records)),
                        "elapsed_seconds": float(time.time() - started),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    payload = {
        "format": TWO_DGS_SURFACE_DEPTH_BANK_FORMAT,
        "width": int(camera.width),
        "height": int(camera.height),
        "records": output_records,
        "metadata": {
            "representation": "official_2dgs_expected_surface_depth",
            "renderer": "gsplat.rasterization_2dgs",
            "ply": str(Path(args.ply).resolve()),
            "ply_sha256": file_sha256_short(Path(args.ply)),
            "mapping_pose_file": str(Path(args.mapping_pose_file).resolve()),
            "mapping_pose_file_sha256": file_sha256_short(
                Path(args.mapping_pose_file)
            ),
            "camera_model_dir": str(Path(args.camera_model_dir).resolve()),
            "gaussian_count": int(source.gaussian_count),
            "sh_degree": int(source.sh_degree),
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    temporary_manifest.replace(manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "view_count": int(len(records)),
                "elapsed_seconds": float(time.time() - started),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
