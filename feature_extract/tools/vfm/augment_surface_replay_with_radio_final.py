"""Attach query-aligned RADIO-final descriptors to frozen ALIKE replay rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import (
    _load_camera_by_image,
)
from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_raw_final,
    _sample_mapped_vfm_at_pixels,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_replay_dir", required=True)
    parser.add_argument("--radio_final_manifest", required=True)
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument("--output_replay_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--radio_final_layer", default="radio_final")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if (
        int(args.shard_count) <= 0
        or not 0 <= int(args.shard_index) < int(args.shard_count)
    ):
        raise ValueError("shard index/count are invalid")
    manifest = TokenBankManifest.from_json(
        Path(args.radio_final_manifest)
    )
    record_by_image = {record.image_id: record for record in manifest.records}
    camera_by_image = _load_camera_by_image(str(args.camera_model_dir))
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint),
        device=str(args.device),
    )
    input_paths = sorted(Path(args.input_replay_dir).glob("*.npz"))
    selected = [
        path
        for row, path in enumerate(input_paths)
        if row % int(args.shard_count) == int(args.shard_index)
    ]
    output_dir = Path(args.output_replay_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    reused = 0
    row_count = 0
    feature_dim = None
    for input_path in selected:
        output_path = output_dir / input_path.name
        if output_path.is_file() and not bool(args.force):
            reused += 1
            continue
        with np.load(input_path, allow_pickle=True) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        image_id = str(payload["image_id"].item())
        record = record_by_image.get(image_id)
        camera = camera_by_image.get(image_id)
        if record is None or camera is None:
            raise FileNotFoundError(
                f"RADIO-final replay input is absent for {image_id}"
            )
        raw = _load_raw_final(
            record.token_path, str(args.radio_final_layer)
        )
        mapped = mapper.project(raw).coarse_descriptors
        vfm_descriptors = _sample_mapped_vfm_at_pixels(
            mapped,
            np.asarray(payload["xy"], dtype=np.float32),
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        payload["vfm_descriptors"] = vfm_descriptors
        np.savez_compressed(output_path, **payload)
        feature_dim = int(vfm_descriptors.shape[1])
        row_count += int(len(vfm_descriptors))
        written += 1
    summary = {
        "stage": "augment_surface_replay_with_radio_final",
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "selected_image_count": len(selected),
        "written_image_count": written,
        "reused_image_count": reused,
        "query_row_count_written": row_count,
        "vfm_feature_dim": feature_dim,
        "mapper_metadata": {
            key: value
            for key, value in dict(mapper_metadata).items()
            if key
            in {
                "format",
                "pool_sizes",
                "pool_weights",
                "uses_radio_intermediate",
            }
        },
        "production_contract": {
            "stores_mapping_rgb": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_loftr": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
