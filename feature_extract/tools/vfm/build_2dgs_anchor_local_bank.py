"""Attach real-image ALIKE descriptors to persistent 2DGS surface anchors."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import _load_camera_by_image
from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
)
from feature_extract.vfm.localization.surface_localization import (
    LocalFeatureFrame,
    build_anchor_local_descriptor_bank,
)
from feature_extract.vfm.surface_maplet_bank import StableSurfaceAnchorMap


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument("--output_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument("--maximum_pixel_distance", type=float, default=0.25)
    parser.add_argument("--minimum_observations", type=int, default=2)
    parser.add_argument("--maximum_prototypes", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    camera_by_image = _load_camera_by_image(str(args.camera_model_dir))
    rows_by_image: dict[str, list[int]] = {}
    for row, image_id in enumerate(anchors.observation_image_ids):
        rows_by_image.setdefault(str(image_id), []).append(int(row))
    missing_cameras = sorted(set(rows_by_image) - set(camera_by_image))
    if missing_cameras:
        raise ValueError(f"anchor support image has no camera: {missing_cameras[0]}")
    image_root = Path(args.image_root)
    missing_images = sorted(
        image_id for image_id in rows_by_image if not (image_root / image_id).is_file()
    )
    if missing_images:
        raise FileNotFoundError(f"anchor support RGB is missing: {missing_images[0]}")

    extractor = AlikeDenseObservationExtractor(
        device=str(args.device),
        matcha_repo=Path(args.matcha_repo),
        model_name=str(args.alike_model_name),
    )
    support_frames: dict[str, LocalFeatureFrame] = {}
    image_hashes: dict[str, str] = {}
    per_view: list[dict[str, object]] = []
    for image_id in sorted(rows_by_image):
        rows = np.asarray(rows_by_image[image_id], dtype=np.int64)
        xy = anchors.observation_xy[rows]
        camera = camera_by_image[image_id]
        descriptors, scores, image_hash = extractor.sample_points(
            image_root / image_id,
            xy,
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        support_frames[image_id] = LocalFeatureFrame(
            image_id=image_id,
            keypoints_xy=xy,
            descriptors=descriptors,
            scores=np.maximum(scores, 0.0),
        )
        image_hashes[image_id] = image_hash
        per_view.append(
            {
                "image_id": image_id,
                "sample_count": int(len(rows)),
                "detector_score_median": float(np.median(scores)) if len(scores) else 0.0,
            }
        )
    bank = build_anchor_local_descriptor_bank(
        anchors,
        support_frames,
        maximum_pixel_distance=float(args.maximum_pixel_distance),
        minimum_observations=int(args.minimum_observations),
        maximum_prototypes=int(args.maximum_prototypes),
    )
    extractor_metadata = dict(extractor.metadata)
    extractor_metadata.update(
        {
            "sampling_convention": "mapping_camera_pixel_endpoint_to_dense_endpoint_v1",
            "image_preprocessing": "resize_rgb_to_mapping_camera_dimensions",
        }
    )
    metadata = dict(bank.metadata or {})
    metadata.update(
        {
            "local_feature": "alike_dense_fpn",
            "extractor": extractor_metadata,
            "source": "real_mapping_rgb_at_2dgs_anchor_projection",
            "image_hashes": image_hashes,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        }
    )
    bank = replace(bank, metadata=metadata)
    bank.save_npz(Path(args.output_bank))
    descriptor_counts = np.diff(bank.descriptor_offsets)
    summary = {
        "stage": "build_2dgs_anchor_local_bank",
        "production_contract": {
            "anchor_identity": "stable_surface_anchor_id",
            "support_source": "real_mapping_rgb",
            "uses_rendered_rgb": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
        "input_anchor_count": int(len(anchors)),
        "retained_anchor_count": int(len(bank)),
        "retained_anchor_fraction": float(len(bank) / max(len(anchors), 1)),
        "descriptor_count": int(bank.descriptors.shape[0]),
        "descriptor_dim": int(bank.feature_dim),
        "descriptors_per_anchor": {
            "min": int(np.min(descriptor_counts)) if descriptor_counts.size else 0,
            "median": float(np.median(descriptor_counts)) if descriptor_counts.size else 0.0,
            "mean": float(np.mean(descriptor_counts)) if descriptor_counts.size else 0.0,
            "max": int(np.max(descriptor_counts)) if descriptor_counts.size else 0,
        },
        "per_view": per_view,
        "inputs": {
            "anchors": str(args.anchors),
            "image_root": str(args.image_root),
            "camera_model_dir": str(args.camera_model_dir),
        },
        "output_bank": str(args.output_bank),
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
