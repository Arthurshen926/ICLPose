"""Bake a trained V6 metric encoder into fixed canonical maplet atlases."""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--trajectory_ids",
        nargs="*",
        default=[],
        help="Optional explicit mapping-trajectory subset.",
    )
    parser.add_argument("--minimum_support", type=int, default=2)
    parser.add_argument("--appearance_modes", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument(
        "--feature_level",
        choices=("fine", "middle", "coarse"),
        default="fine",
    )
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
    encoder_sha256 = hashlib.sha256(
        Path(args.metric_encoder).read_bytes()
    ).hexdigest()
    model.eval()
    cache_paths = sorted(
        path
        for directory in args.contributor_dirs
        for path in Path(directory).glob("*.npz")
    )
    views = []
    buffers = []
    trajectories = []
    occlusion_policies = []
    clean_source_index_hashes = []
    clean_geometry_source_hashes = []
    geometry_source_hashes = []
    requested_trajectories = {str(value) for value in args.trajectory_ids}
    with torch.no_grad():
        for index, path in enumerate(cache_paths):
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                trajectory_id = str(metadata["trajectory_id"])
                if (
                    requested_trajectories
                    and trajectory_id not in requested_trajectories
                ):
                    continue
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
            metric_feature = model(radio, rgb)[str(args.feature_level)][
                0
            ].cpu().numpy()
            views.append(
                GaussianVFMFeatureView(
                    image_id=image_id,
                    feature_map=metric_feature,
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
            trajectories.append(trajectory_id)
            occlusion_policies.append(
                str(
                    metadata.get(
                        "occlusion_primitive_policy",
                        (
                            "declared_clean_source_index_mask"
                            if bool(
                                metadata.get(
                                    "uses_declared_clean_2dgs_for_occlusion",
                                    False,
                                )
                            )
                            else "complete_input_2dgs"
                        ),
                    )
                )
            )
            clean_source_index_hashes.append(
                str(metadata.get("clean_source_index_sha256", ""))
            )
            clean_geometry_source_hashes.append(
                str(metadata.get("clean_geometry_source_sha256", ""))
            )
            geometry_source_hashes.append(
                str(metadata.get("geometry_source_sha256", ""))
            )
            print(f"[{index + 1}/{len(cache_paths)}] {image_id}", flush=True)
    if len(set(occlusion_policies)) != 1:
        raise ValueError("contributor caches mix incompatible occlusion priors")
    occlusion_policy = occlusion_policies[0]
    contributor_lineage = {
        "clean_source_index_sha256": set(clean_source_index_hashes),
        "clean_geometry_source_sha256": set(clean_geometry_source_hashes),
        "geometry_source_sha256": set(geometry_source_hashes),
    }
    if any(len(values) != 1 for values in contributor_lineage.values()):
        raise ValueError("contributor caches mix geometry source lineages")
    contributor_lineage = {
        key: next(iter(values)) for key, values in contributor_lineage.items()
    }
    geometry_metadata = dict(geometry.metadata or {})
    geometry_lineage_verified = all(
        bool(contributor_lineage[key])
        and contributor_lineage[key] == str(geometry_metadata.get(key, ""))
        for key in contributor_lineage
    )
    lineage_declared = any(
        bool(value) for value in contributor_lineage.values()
    ) or any(
        bool(geometry_metadata.get(key, "")) for key in contributor_lineage
    )
    if lineage_declared and not geometry_lineage_verified:
        raise ValueError(
            "contributor cache and canonical atlas geometry lineages differ"
        )
    atlas, report = bake_feature_atlas(
        geometry,
        views,
        buffers,
        minimum_support=int(args.minimum_support),
        appearance_modes=int(args.appearance_modes),
        trajectory_ids=trajectories,
        metadata={
            "metric_encoder_artifact": str(args.metric_encoder),
            "metric_encoder_best_step": encoder_metadata.get("best_step", -1),
            "metric_encoder_sha256": encoder_sha256,
            "vfm_layer": "radio_final",
            "metric_feature_level": str(args.feature_level),
            "metric_feature_stride": {
                "fine": 4,
                "middle": 8,
                "coarse": 16,
            }[str(args.feature_level)],
            "uses_shallow_rgb_phase_stem": True,
            "uses_radio_final_conditioning": True,
            "contributor_occlusion_primitive_policy": occlusion_policy,
            "contributor_geometry_lineage_verified": bool(
                geometry_lineage_verified
            ),
            **{
                f"contributor_{key}": value
                for key, value in contributor_lineage.items()
            },
        },
    )
    atlas.save_npz(output_path)
    report.update(
        {
            "output_atlas": str(output_path),
            "metric_encoder": str(args.metric_encoder),
            "metric_feature_level": str(args.feature_level),
            "metric_encoder_sha256": encoder_sha256,
            "trajectory_ids": sorted(set(trajectories)),
            "contributor_occlusion_primitive_policy": occlusion_policy,
            "contributor_geometry_lineage_verified": bool(
                geometry_lineage_verified
            ),
            **{
                f"contributor_{key}": value
                for key, value in contributor_lineage.items()
            },
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
