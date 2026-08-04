"""Extract offline RADIO teacher readouts at the canonical 128 query regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from feature_extract.utils.radio_loader import load_radio_model
from feature_extract.vfm.localization.surface_maplet_mapper import (
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    encode_radio_final_regions,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--radio_repo", default="/root/.cache/torch/hub/NVlabs_RADIO_main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = sorted(Path(args.contributors).glob("*.npz"))
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid teacher extraction shard")
    paths = paths[int(args.shard_index) :: int(args.shard_count)]
    model = load_radio_model(
        version="c-radio_v4-h",
        radio_repo=str(args.radio_repo),
        adaptor_names=["dino_v3_7b", "sam3", "siglip2-g"],
        visual_only_siglip2=True,
    ).to(str(args.device)).eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = RadioFinalRegionConfig(
        pool_sizes=(1, 3, 5, 9), pool_weights=(0.4, 0.3, 0.2, 0.1)
    )
    written = 0
    for source in paths:
        with np.load(source, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        image_id = str(metadata["image_id"])
        token_path = Path(str(metadata["token_path"]))
        output_path = output_dir / (image_id.replace("/", "__") + ".npz")
        if output_path.exists() and not bool(args.force):
            continue
        with np.load(token_path, allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        _indices, token_xy = select_spatially_balanced_radio_final_regions(raw)
        image = Image.open(Path(args.image_root) / image_id).convert("RGB").resize(
            (1024, 576), Image.Resampling.BILINEAR
        )
        rgb = torch.from_numpy(
            (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1).copy()
        )[None].to(str(args.device))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outputs = model(rgb, feature_fmt="NLC")
        payload = {}
        for name in ("dino_v3_7b", "sam3", "siglip2-g"):
            value = outputs[name]
            dense = value.features[0].float().reshape(36, 64, -1).permute(2, 0, 1).cpu().numpy()
            payload[name] = encode_radio_final_regions(dense, token_xy, config).astype(np.float16)
            payload[name + "_summary"] = value.summary[0].float().cpu().numpy().astype(np.float16)
        payload.update(
            token_xy=token_xy.astype(np.float32),
            metadata_json=np.asarray(json.dumps({
                "artifact_type": "v8_offline_radio_teacher_regions",
                "image_id": image_id,
                "teacher_names": ["dino_v3_7b", "sam3", "siglip2-g"],
                "runtime_map_payload": False,
                "stores_rgb": False,
                "stores_image_path": False,
            }, sort_keys=True)),
        )
        np.savez_compressed(output_path, **payload)
        written += 1
        print(json.dumps({"image_id": image_id, "written": written}), flush=True)
    print(json.dumps({"processed": len(paths), "written": written, "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
