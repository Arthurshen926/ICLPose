"""Orient persistent 2DGS anchor normals toward their mapping observations."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import (
    parse_cambridge_pose_file,
)
from feature_extract.vfm.surface_maplet_bank import StableSurfaceAnchorMap


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--output_anchors", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def orient_anchor_normals(
    anchors: StableSurfaceAnchorMap,
    pose_by_image: dict[str, np.ndarray],
) -> tuple[StableSurfaceAnchorMap, dict[str, object]]:
    oriented = anchors.normals.astype(np.float64).copy()
    flipped = np.zeros((len(anchors),), dtype=bool)
    support_strength = np.zeros((len(anchors),), dtype=np.float64)
    missing_pose_count = 0
    for anchor_row in range(len(anchors)):
        start = int(anchors.observation_offsets[anchor_row])
        end = int(anchors.observation_offsets[anchor_row + 1])
        view_sum = np.zeros((3,), dtype=np.float64)
        weight_sum = 0.0
        for observation_row in range(start, end):
            pose = pose_by_image.get(
                anchors.observation_image_ids[observation_row]
            )
            if pose is None:
                missing_pose_count += 1
                continue
            camera_center = -pose[:3, :3].T @ pose[:3, 3]
            direction = camera_center - anchors.xyz[anchor_row]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            weight = max(
                float(anchors.observation_weights[observation_row]),
                1e-8,
            )
            view_sum += weight * direction / norm
            weight_sum += weight
        if weight_sum <= 0.0:
            continue
        mean_view = view_sum / weight_sum
        mean_view /= max(float(np.linalg.norm(mean_view)), 1e-12)
        signed = float(np.dot(oriented[anchor_row], mean_view))
        support_strength[anchor_row] = abs(signed)
        if signed < 0.0:
            oriented[anchor_row] *= -1.0
            flipped[anchor_row] = True
    metadata = {
        **dict(anchors.metadata or {}),
        "normal_orientation": (
            "signed_toward_weighted_mapping_observation_camera_centers"
        ),
        "normal_orientation_uses_mapping_pose_at_runtime": False,
        "normal_orientation_missing_pose_observation_count": int(
            missing_pose_count
        ),
    }
    output = replace(
        anchors,
        normals=oriented.astype(np.float32),
        metadata=metadata,
    )
    valid_strength = support_strength[support_strength > 0.0]
    summary = {
        "stage": "orient_surface_anchor_normals",
        "anchor_count": len(anchors),
        "flipped_anchor_count": int(np.sum(flipped)),
        "flipped_anchor_fraction": float(np.mean(flipped)),
        "oriented_anchor_count": int(len(valid_strength)),
        "missing_pose_observation_count": int(missing_pose_count),
        "orientation_strength": {
            "median": (
                float(np.median(valid_strength))
                if len(valid_strength)
                else None
            ),
            "p10": (
                float(np.percentile(valid_strength, 10.0))
                if len(valid_strength)
                else None
            ),
        },
        "production_contract": {
            "stores_mapping_pose": False,
            "uses_mapping_pose_at_inference": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_loftr": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
    }
    return output, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(
            Path(args.mapping_pose_file)
        )
    }
    oriented, summary = orient_anchor_normals(
        anchors,
        pose_by_image,
    )
    oriented.save_npz(Path(args.output_anchors))
    summary["outputs"] = {
        "anchors": str(args.output_anchors),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
