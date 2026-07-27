"""Bake a trained V6 metric encoder into fixed canonical maplet atlases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView
from feature_extract.vfm.localization_v6.atlas_baking import bake_feature_atlas
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.metric_encoder import (
    load_v6_metric_encoder,
)
from feature_extract.vfm.localization_v6.primitive_contributors import (
    PrimitiveContributorBuffer,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas_geometry", required=True)
    parser.add_argument("--metric_encoder", required=True)
    parser.add_argument("--contributor_dirs", nargs="+", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_atlas", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--minimum_support", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_atlas)
    summary_path = Path(args.summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite V6 metric atlas")
    geometry = MapletFeatureAtlasBank.load_npz(Path(args.atlas_geometry))
    model, encoder_metadata = load_v6_metric_encoder(
        Path(args.metric_encoder), device=str(args.device)
    )
    model.eval()
    cache_paths = sorted(
        path
        for directory in args.contributor_dirs
        for path in Path(directory).glob("*.npz")
    )
    views = []
    buffers = []
    trajectories = []
    with torch.no_grad():
        for index, path in enumerate(cache_paths):
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                image_id = str(metadata["image_id"])
                token_path = Path(str(metadata["token_path"]))
                camera = ColmapCamera(
                    camera_id=0,
                    model_id=int(data["camera_model_id"]),
                    width=int(data["camera_width"]),
                    height=int(data["camera_height"]),
                    params=tuple(
                        np.asarray(data["camera_params"], dtype=np.float64)
                    ),
                )
                pose = np.asarray(data["pose_w2c"], dtype=np.float64)
                topk_ids = np.asarray(data["topk_ids"], dtype=np.int64)
                topk_weights = np.asarray(data["topk_weights"], dtype=np.float32)
                primitive_depth = np.asarray(
                    data["dominant_depth"], dtype=np.float32
                )
            image = Image.open(Path(args.image_root) / image_id).convert("RGB")
            image = image.resize(
                (camera.width, camera.height), Image.Resampling.BILINEAR
            )
            rgb = torch.from_numpy(
                (np.asarray(image, dtype=np.float32) / 255.0)
                .transpose(2, 0, 1)
                .copy()
            )[None].to(str(args.device))
            with np.load(token_path, allow_pickle=False) as token_data:
                radio = torch.from_numpy(
                    np.asarray(token_data["radio_final"], dtype=np.float32)
                )[None].to(str(args.device))
            fine = model(radio, rgb)["fine"][0].cpu().numpy()
            views.append(
                GaussianVFMFeatureView(
                    image_id=image_id,
                    feature_map=fine,
                    pose_w2c=pose,
                    camera=camera,
                )
            )
            buffers.append(
                PrimitiveContributorBuffer(
                    dominant_ids=topk_ids[..., 0],
                    dominant_weights=topk_weights[..., 0],
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    primitive_depth=primitive_depth,
                    metadata=metadata,
                )
            )
            trajectories.append(str(metadata["trajectory_id"]))
            print(f"[{index + 1}/{len(cache_paths)}] {image_id}", flush=True)
    atlas, report = bake_feature_atlas(
        geometry,
        views,
        buffers,
        minimum_support=int(args.minimum_support),
        trajectory_ids=trajectories,
        metadata={
            "metric_encoder_artifact": str(args.metric_encoder),
            "metric_encoder_best_step": encoder_metadata.get("best_step", -1),
            "vfm_layer": "radio_final",
            "uses_shallow_rgb_phase_stem": True,
            "uses_radio_final_conditioning": True,
        },
    )
    atlas.save_npz(output_path)
    report.update(
        {
            "output_atlas": str(output_path),
            "metric_encoder": str(args.metric_encoder),
            "trajectory_ids": sorted(set(trajectories)),
            "deployment_contract": {
                "stores_mapping_rgb": False,
                "stores_mapping_image_paths": False,
                "query_compares_to_feature_map_only": True,
            },
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
