"""Build RADIO-final sparse descriptors for ALIKE-detected 2DGS anchors.

The legacy output concatenates ALIKE and RADIO-final descriptors.  The
``radio_final`` mode uses ALIKE only to select a repeatable observation and
stores/matches RADIO-final descriptors alone.  It can also collapse all
mapping observations into one anonymous prototype per anchor.
"""

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
    parser.add_argument(
        "--output_feature",
        choices=("alike_radio_final", "radio_final"),
        default="alike_radio_final",
    )
    parser.add_argument(
        "--collapse_radio_final_prototype",
        action="store_true",
        help=(
            "Store one anonymous weighted RADIO-final prototype per anchor "
            "instead of a list of mapping-view observations."
        ),
    )
    return parser.parse_args(argv)


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(
        np.linalg.norm(array, axis=1, keepdims=True),
        1e-8,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if bool(args.collapse_radio_final_prototype) and (
        str(args.output_feature) != "radio_final"
    ):
        raise ValueError(
            "prototype collapse is defined only for RADIO-final output"
        )
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
        descriptor_rows = list(range(start, end))
        if str(args.output_feature) == "radio_final":
            # Geometry already assigns replay detections to stable anchors.
            # ALIKE descriptor similarity must not participate in identity.
            best_by_image: dict[str, int] = {}
            for descriptor_row in descriptor_rows:
                image_id = str(base.support_image_ids[descriptor_row])
                previous = best_by_image.get(image_id)
                if previous is None or float(
                    base.descriptor_quality[descriptor_row]
                ) > float(base.descriptor_quality[previous]):
                    best_by_image[image_id] = int(descriptor_row)
            descriptor_rows = [
                best_by_image[image_id] for image_id in sorted(best_by_image)
            ]
        for descriptor_row in descriptor_rows:
            image_id = str(base.support_image_ids[descriptor_row])
            replay = replay_by_image.get(image_id)
            if replay is None:
                continue
            target_ids, replay_alike, replay_vfm, replay_scores = replay
            candidate_rows = np.flatnonzero(target_ids == anchor_id)
            if len(candidate_rows) == 0:
                continue
            if str(args.output_feature) == "radio_final":
                best = int(
                    candidate_rows[
                        int(np.argmax(replay_scores[candidate_rows]))
                    ]
                )
                output_descriptor = replay_vfm[best]
            else:
                similarities = (
                    replay_alike[candidate_rows]
                    @ base.descriptors[descriptor_row]
                )
                best = int(candidate_rows[int(np.argmax(similarities))])
                output_descriptor = np.concatenate(
                    [
                        base.descriptors[descriptor_row],
                        replay_vfm[best],
                    ]
                )
            output_descriptor = output_descriptor / max(
                float(np.linalg.norm(output_descriptor)), 1e-8
            )
            quality = float(
                max(base.descriptor_quality[descriptor_row], 1e-8)
                * np.sqrt(max(float(replay_scores[best]), 1e-8))
            )
            local_records.append(
                (
                    output_descriptor.astype(np.float32),
                    quality,
                    image_id,
                    base.support_view_directions[descriptor_row],
                )
            )
            matched += 1
        if not local_records:
            continue
        if bool(args.collapse_radio_final_prototype):
            local_descriptors = np.stack(
                [record[0] for record in local_records]
            ).astype(np.float32)
            local_quality = np.asarray(
                [record[1] for record in local_records], dtype=np.float32
            )
            weights = np.maximum(local_quality, 1e-8)
            prototype = np.sum(
                local_descriptors * weights[:, None], axis=0
            )
            prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
            direction = np.sum(
                np.stack([record[3] for record in local_records])
                * weights[:, None],
                axis=0,
            )
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            local_records = [
                (
                    prototype.astype(np.float32),
                    float(np.median(local_quality)),
                    "__anonymous_radio_final_prototype__",
                    direction.astype(np.float32),
                )
            ]
        anchor_ids.append(anchor_id)
        for descriptor, quality, image_id, direction in local_records:
            descriptors.append(descriptor)
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
                "feature_aligned_2dgs_anchor_radio_final"
                if str(args.output_feature) == "radio_final"
                else "feature_aligned_2dgs_anchor_alike_plus_radio_final"
            ),
            "local_feature": (
                "radio_final_at_alike_detection"
                if str(args.output_feature) == "radio_final"
                else "alike_anchor_plus_radio_final_context"
            ),
            "alike_feature_dim": (
                0
                if str(args.output_feature) == "radio_final"
                else int(base.feature_dim)
            ),
            "radio_final_feature_dim": int(
                len(descriptors[0])
                if str(args.output_feature) == "radio_final"
                else len(descriptors[0]) - int(base.feature_dim)
            ),
            "alike_descriptor_used_for_identity": (
                str(args.output_feature) != "radio_final"
            ),
            "alike_role": (
                "detector_only"
                if str(args.output_feature) == "radio_final"
                else "detector_and_identity_descriptor"
            ),
            "prototype_count_per_anchor": (
                1 if bool(args.collapse_radio_final_prototype) else None
            ),
            "stores_mapping_image_ids": not bool(
                args.collapse_radio_final_prototype
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
        "output_feature": str(args.output_feature),
        "collapse_radio_final_prototype": bool(
            args.collapse_radio_final_prototype
        ),
        "production_contract": {
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": not bool(
                args.collapse_radio_final_prototype
            ),
            "alike_descriptor_used_for_identity": (
                str(args.output_feature) != "radio_final"
            ),
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
