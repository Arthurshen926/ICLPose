"""Add pose-free support viewing directions to a 2DGS descriptor map."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import (
    parse_cambridge_pose_file,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
)
from feature_extract.vfm.surface_maplet_bank import StableSurfaceAnchorMap


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--descriptor_bank", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--output_descriptor_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def add_view_directions(
    *,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    pose_by_image: dict[str, np.ndarray],
) -> tuple[AnchorLocalDescriptorBank, dict[str, object]]:
    anchor_row_by_id = anchors.row_by_id()
    directions = np.zeros(
        (len(descriptor_bank.descriptors), 3),
        dtype=np.float32,
    )
    missing_pose = 0
    missing_anchor = 0
    for bank_row, anchor_id in enumerate(
        descriptor_bank.anchor_ids.tolist()
    ):
        anchor_row = anchor_row_by_id.get(int(anchor_id))
        start = int(descriptor_bank.descriptor_offsets[bank_row])
        end = int(descriptor_bank.descriptor_offsets[bank_row + 1])
        if anchor_row is None:
            missing_anchor += end - start
            continue
        for descriptor_row in range(start, end):
            pose = pose_by_image.get(
                descriptor_bank.support_image_ids[descriptor_row]
            )
            if pose is None:
                missing_pose += 1
                continue
            center = -pose[:3, :3].T @ pose[:3, 3]
            direction = center - anchors.xyz[anchor_row]
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            directions[descriptor_row] = direction.astype(np.float32)
    valid = np.linalg.norm(directions, axis=1) > 0.5
    metadata = {
        **dict(descriptor_bank.metadata or {}),
        "view_conditioning": (
            "per_descriptor_anchor_to_support_camera_unit_direction"
        ),
        "stores_mapping_pose": False,
        "uses_mapping_pose_at_inference": False,
    }
    output = replace(
        descriptor_bank,
        support_view_directions=directions,
        metadata=metadata,
    )
    return output, {
        "stage": "add_surface_descriptor_view_directions",
        "descriptor_count": int(len(directions)),
        "valid_view_direction_count": int(np.sum(valid)),
        "missing_pose_count": int(missing_pose),
        "missing_anchor_count": int(missing_anchor),
        "production_contract": {
            "stores_mapping_pose": False,
            "stores_mapping_rgb": False,
            "uses_mapping_pose_at_inference": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_loftr": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    descriptor_bank = AnchorLocalDescriptorBank.load_npz(
        Path(args.descriptor_bank)
    )
    poses = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(
            Path(args.mapping_pose_file)
        )
    }
    output, summary = add_view_directions(
        anchors=anchors,
        descriptor_bank=descriptor_bank,
        pose_by_image=poses,
    )
    output.save_npz(Path(args.output_descriptor_bank))
    summary["outputs"] = {
        "descriptor_bank": str(args.output_descriptor_bank)
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
