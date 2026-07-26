"""Replay online ALIKE/RADIO-final inputs for surface-anchor matcher training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import (
    _load_camera_by_image,
)
from feature_extract.vfm.localization.surface_localization import (
    SurfaceMapletMatchConfig,
    match_radio_final_regions_to_maplets,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    StableSurfaceAnchorMap,
    VfmSurfaceMapletBank,
    encode_radio_final_regions,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--detection_cache_dir", required=True)
    parser.add_argument("--radio_final_manifest", required=True)
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--radio_final_layer", default="radio_final")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maplet_top_k", type=int, default=8)
    parser.add_argument("--maximum_query_nodes_per_image", type=int, default=512)
    parser.add_argument("--maximum_positive_nodes_per_image", type=int, default=384)
    parser.add_argument("--target_radius_px", type=float, default=0.75)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _cache_name(image_id: str) -> str:
    return str(image_id).replace("/", "__") + ".npz"


def _load_radio_final(path: Path, layer_name: str) -> np.ndarray:
    with np.load(path) as data:
        feature = np.asarray(data[layer_name], dtype=np.float32)
    if feature.ndim == 4 and int(feature.shape[0]) == 1:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError("RADIO-final feature must have shape (C,H,W)")
    return feature


def _target_anchor_ids(
    *,
    image_id: str,
    detection_xy: np.ndarray,
    anchors: StableSurfaceAnchorMap,
    anchor_rows_by_image: dict[str, np.ndarray],
    observation_rows_by_image: dict[str, np.ndarray],
    radius_px: float,
) -> np.ndarray:
    target = np.full((len(detection_xy),), -1, dtype=np.int64)
    observation_rows = observation_rows_by_image.get(str(image_id))
    anchor_rows = anchor_rows_by_image.get(str(image_id))
    if observation_rows is None or anchor_rows is None or len(observation_rows) == 0:
        return target
    tree = cKDTree(anchors.observation_xy[observation_rows])
    distance, columns = tree.query(detection_xy, k=1)
    valid = np.isfinite(distance) & (distance <= float(radius_px))
    if np.any(valid):
        target[valid] = anchors.anchor_ids[anchor_rows[np.asarray(columns[valid], dtype=np.int64)]]
    return target


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if int(args.shard_count) <= 0 or not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("shard index/count are invalid")
    maplets = VfmSurfaceMapletBank.load_npz(Path(args.maplets))
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint), device=str(args.device)
    )
    manifest = TokenBankManifest.from_json(Path(args.radio_final_manifest))
    record_by_image = {record.image_id: record for record in manifest.records}
    camera_by_image = _load_camera_by_image(str(args.camera_model_dir))
    observation_rows_by_image: dict[str, list[int]] = {}
    anchor_rows_by_image: dict[str, list[int]] = {}
    for anchor_row in range(len(anchors)):
        start = int(anchors.observation_offsets[anchor_row])
        end = int(anchors.observation_offsets[anchor_row + 1])
        for observation_row in range(start, end):
            image_id = str(anchors.observation_image_ids[observation_row])
            observation_rows_by_image.setdefault(image_id, []).append(
                observation_row
            )
            anchor_rows_by_image.setdefault(image_id, []).append(anchor_row)
    observation_arrays = {
        key: np.asarray(value, dtype=np.int64)
        for key, value in observation_rows_by_image.items()
    }
    anchor_arrays = {
        key: np.asarray(value, dtype=np.int64)
        for key, value in anchor_rows_by_image.items()
    }
    image_ids = sorted(observation_arrays)
    selected_images = [
        image_id
        for row, image_id in enumerate(image_ids)
        if row % int(args.shard_count) == int(args.shard_index)
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_sizes = tuple(
        int(value) for value in mapper_metadata.get("pool_sizes", (1, 3, 5, 9))
    )
    pool_weights = tuple(
        float(value)
        for value in mapper_metadata.get("pool_weights", (0.4, 0.3, 0.2, 0.1))
    )
    region_config = RadioFinalRegionConfig(
        pool_sizes=pool_sizes, pool_weights=pool_weights
    )
    match_config = SurfaceMapletMatchConfig(
        top_k=int(args.maplet_top_k),
        enable_support_layout=False,
    )
    owner_by_anchor = {
        int(anchor_id): int(owner)
        for anchor_id, owner in zip(
            anchors.anchor_ids.tolist(), anchors.owner_maplet_ids.tolist()
        )
    }
    written = 0
    reused = 0
    query_count = 0
    positive_count = 0
    owner_recall_count = 0
    for image_id in selected_images:
        output_path = output_dir / _cache_name(image_id)
        if output_path.is_file() and not bool(args.force):
            reused += 1
            continue
        detection_path = Path(args.detection_cache_dir) / _cache_name(image_id)
        record = record_by_image.get(image_id)
        camera = camera_by_image.get(image_id)
        if not detection_path.is_file() or record is None or camera is None:
            raise FileNotFoundError(f"deployment replay input is missing for {image_id}")
        with np.load(detection_path) as data:
            xy = np.asarray(data["xy"], dtype=np.float32)
            descriptors = np.asarray(data["descriptors"], dtype=np.float32)
            scores = np.asarray(data["scores"], dtype=np.float32)
        targets = _target_anchor_ids(
            image_id=image_id,
            detection_xy=xy,
            anchors=anchors,
            anchor_rows_by_image=anchor_arrays,
            observation_rows_by_image=observation_arrays,
            radius_px=float(args.target_radius_px),
        )
        positive_rows = np.flatnonzero(targets >= 0)
        positive_rows = positive_rows[
            np.argsort(-scores[positive_rows], kind="mergesort")
        ][: int(args.maximum_positive_nodes_per_image)]
        positive_set = set(positive_rows.tolist())
        negative_rows = np.asarray(
            [row for row in np.argsort(-scores, kind="mergesort").tolist() if row not in positive_set],
            dtype=np.int64,
        )
        remaining = max(
            int(args.maximum_query_nodes_per_image) - len(positive_rows), 0
        )
        selected_rows = np.concatenate([positive_rows, negative_rows[:remaining]])
        selected_rows = selected_rows[
            np.argsort(-scores[selected_rows], kind="mergesort")
        ]
        xy = xy[selected_rows]
        descriptors = descriptors[selected_rows]
        scores = scores[selected_rows]
        targets = targets[selected_rows]
        raw = _load_radio_final(record.token_path, str(args.radio_final_layer))
        mapped = mapper.project(raw).coarse_descriptors
        grid_size = (int(mapped.shape[2]), int(mapped.shape[1]))
        grid_xy = xy * np.asarray(
            [
                max(grid_size[0] - 1, 1) / max(int(camera.width) - 1, 1),
                max(grid_size[1] - 1, 1) / max(int(camera.height) - 1, 1),
            ],
            dtype=np.float32,
        )
        region_descriptors = encode_radio_final_regions(
            mapped, grid_xy, region_config
        )
        match = match_radio_final_regions_to_maplets(
            grid_xy,
            region_descriptors,
            grid_size,
            maplets,
            match_config,
        )
        target_owner = np.asarray(
            [owner_by_anchor.get(int(anchor_id), -1) for anchor_id in targets],
            dtype=np.int64,
        )
        recalled = np.any(
            match.candidate_maplet_ids == target_owner[:, None], axis=1
        ) & (target_owner >= 0)
        query_count += len(xy)
        positive_count += int(np.sum(targets >= 0))
        owner_recall_count += int(np.sum(recalled))
        np.savez_compressed(
            output_path,
            image_id=np.asarray(image_id),
            xy=xy.astype(np.float32),
            descriptors=descriptors.astype(np.float32),
            scores=scores.astype(np.float32),
            target_anchor_ids=targets.astype(np.int64),
            target_owner_maplet_ids=target_owner.astype(np.int64),
            candidate_maplet_ids=match.candidate_maplet_ids.astype(np.int64),
            candidate_probabilities=match.candidate_probabilities.astype(np.float32),
            candidate_logits=match.candidate_logits.astype(np.float32),
            null_probabilities=match.null_probabilities.astype(np.float32),
        )
        written += 1
    summary = {
        "stage": "build_surface_anchor_deployment_replay",
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "selected_image_count": len(selected_images),
        "written_image_count": int(written),
        "reused_image_count": int(reused),
        "query_node_count_written": int(query_count),
        "positive_query_node_count_written": int(positive_count),
        "positive_owner_maplet_recall_at_k_written": float(
            owner_recall_count / max(positive_count, 1)
        ),
        "production_contract": {
            "training_distribution": "deployment_replay_alike_detect_and_radio_final",
            "stores_mapping_rgb": False,
            "uses_pairwise_image_matching": False,
            "uses_loftr": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
