"""Fuse ALIKE and RADIO-final replay descriptors for stable 2DGS anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alike_descriptor_bank", required=True)
    parser.add_argument("--augmented_replay_dir", required=True)
    parser.add_argument("--output_descriptor_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(
        np.linalg.norm(array, axis=1, keepdims=True),
        1e-8,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    base = AnchorLocalDescriptorBank.load_npz(
        Path(args.alike_descriptor_bank)
    )
    replay_by_image: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for path in sorted(Path(args.augmented_replay_dir).glob("*.npz")):
        with np.load(path) as data:
            if "vfm_descriptors" not in data:
                continue
            image_id = str(np.asarray(data["image_id"]).item())
            replay_by_image[image_id] = (
                np.asarray(data["target_anchor_ids"], dtype=np.int64),
                _normalize(np.asarray(data["descriptors"], dtype=np.float32)),
                _normalize(
                    np.asarray(data["vfm_descriptors"], dtype=np.float32)
                ),
                np.asarray(data["scores"], dtype=np.float32),
            )
    anchor_ids: list[int] = []
    offsets = [0]
    descriptors: list[np.ndarray] = []
    qualities: list[float] = []
    image_ids: list[str] = []
    view_directions: list[np.ndarray] = []
    matched = 0
    total = int(len(base.descriptors))
    for bank_row, anchor_id_value in enumerate(base.anchor_ids.tolist()):
        anchor_id = int(anchor_id_value)
        start = int(base.descriptor_offsets[bank_row])
        end = int(base.descriptor_offsets[bank_row + 1])
        local_records: list[
            tuple[np.ndarray, float, str, np.ndarray]
        ] = []
        for descriptor_row in range(start, end):
            image_id = base.support_image_ids[descriptor_row]
            replay = replay_by_image.get(image_id)
            if replay is None:
                continue
            target_ids, replay_alike, replay_vfm, replay_scores = replay
            candidate_rows = np.flatnonzero(target_ids == anchor_id)
            if len(candidate_rows) == 0:
                continue
            similarities = (
                replay_alike[candidate_rows]
                @ base.descriptors[descriptor_row]
            )
            best = int(candidate_rows[int(np.argmax(similarities))])
            fused = np.concatenate(
                [
                    base.descriptors[descriptor_row],
                    replay_vfm[best],
                ]
            )
            fused /= max(float(np.linalg.norm(fused)), 1e-8)
            quality = float(
                max(base.descriptor_quality[descriptor_row], 1e-8)
                * np.sqrt(max(float(replay_scores[best]), 1e-8))
            )
            local_records.append(
                (
                    fused.astype(np.float32),
                    quality,
                    image_id,
                    base.support_view_directions[descriptor_row],
                )
            )
            matched += 1
        if not local_records:
            continue
        anchor_ids.append(anchor_id)
        for fused, quality, image_id, direction in local_records:
            descriptors.append(fused)
            qualities.append(quality)
            image_ids.append(image_id)
            view_directions.append(direction)
        offsets.append(len(descriptors))
    if not descriptors:
        raise ValueError("no ALIKE/RADIO-final anchor descriptors matched")
    output = AnchorLocalDescriptorBank(
        anchor_ids=np.asarray(anchor_ids, dtype=np.int64),
        descriptor_offsets=np.asarray(offsets, dtype=np.int64),
        descriptors=np.stack(descriptors).astype(np.float32),
        support_image_ids=tuple(image_ids),
        descriptor_quality=np.asarray(qualities, dtype=np.float32),
        support_view_directions=np.stack(view_directions).astype(np.float32),
        metadata={
            **dict(base.metadata or {}),
            "representation": (
                "feature_aligned_2dgs_anchor_alike_plus_radio_final"
            ),
            "local_feature": "alike_anchor_plus_radio_final_context",
            "alike_feature_dim": int(base.feature_dim),
            "radio_final_feature_dim": int(
                len(descriptors[0]) - int(base.feature_dim)
            ),
            "uses_mapping_rgb_at_inference": False,
            "uses_radio_intermediate": False,
        },
    )
    output.save_npz(Path(args.output_descriptor_bank))
    summary = {
        "stage": "build_alike_radio_final_anchor_bank",
        "input_descriptor_count": total,
        "matched_descriptor_count": matched,
        "matched_descriptor_fraction": float(matched / max(total, 1)),
        "input_anchor_count": len(base),
        "output_anchor_count": len(output),
        "output_descriptor_count": len(output.descriptors),
        "output_feature_dim": output.feature_dim,
        "production_contract": {
            "stores_mapping_rgb": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_loftr": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
        "outputs": {
            "descriptor_bank": str(args.output_descriptor_bank)
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
