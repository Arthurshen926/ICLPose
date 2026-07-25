"""Build train-only SfM observation pairs for RGB spatial likelihood pretraining.

The artifact is intentionally not a runtime layout.  It widens the sparse P1
identity signal with registered observations from *train* query images only:

    query observation -> same-track mapping support observation (positive)
    query observation -> RADIO-PCA global landmark ANN tracks (hard negatives)

The mapper/scorer never reads this file.  It contains track identities solely
for supervised pretraining, while the saved neural checkpoint remains target
free at inference time.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_observation_pairs import (
    CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PAIR_FORMAT,
    CandidatePoseRGBSpatialObservationPairs,
    save_candidate_pose_rgb_spatial_observation_pairs,
)
from feature_extract.vfm.localization.context_observation_landmark_bank import (
    sample_spatial_context_descriptors,
)
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    load_spatial_image_context_cache,
)
from feature_extract.vfm.query_to_3d_matching import normalize_rows


_PCA_FORMAT = "radio_intermediate_image_context_pca_v1"
_LANDMARK_BANK_FORMAT = "landmark_map_index_npz"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-query-layout", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--support-observation-index", required=True)
    parser.add_argument("--hard-negative-context-cache", required=True)
    parser.add_argument("--hard-negative-landmark-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-anchors-per-query", type=int, default=1200)
    parser.add_argument("--negative-count", type=int, default=3)
    parser.add_argument("--ann-search-k", type=int, default=64)
    parser.add_argument("--border-margin-px", type=float, default=24.0)
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _stable_hash(*values: object) -> int:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def _image_list_hash(image_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(str(value) for value in image_ids))).encode()).hexdigest()[:16]


def partition_train_query_ids(
    *, query_ids: Sequence[str], fold_count: int, fold_index: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministically reserve one train-query fold without external labels."""

    ids = tuple(sorted(set(str(value) for value in query_ids)))
    count = int(fold_count)
    index = int(fold_index)
    if len(ids) < 2 or count < 2 or count > len(ids) or not 0 <= index < count:
        raise ValueError("observation-pair inner validation fold is invalid")
    validation = tuple(value for position, value in enumerate(ids) if position % count == index)
    train = tuple(value for value in ids if value not in set(validation))
    if not train or not validation:
        raise ValueError("observation-pair inner validation fold is empty")
    return train, validation


def _load_support_observation_index(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    required = {"image_ids", "offsets", "track_ids", "xy", "metadata_json"}
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"support observation index lacks {sorted(missing)}")
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("support observation index metadata is invalid") from error
        image_ids = np.asarray(payload["image_ids"]).astype(str).reshape(-1)
        offsets = np.asarray(payload["offsets"], dtype=np.int64).reshape(-1)
        tracks = np.asarray(payload["track_ids"], dtype=np.int64).reshape(-1)
        xy = np.asarray(payload["xy"], dtype=np.float32)
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != "sfm_track_observation_index_v1"
        or len(image_ids) == 0
        or len(set(image_ids.tolist())) != len(image_ids)
        or offsets.shape != (len(image_ids) + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(tracks)
        or np.any(offsets[1:] < offsets[:-1])
        or xy.shape != (len(tracks), 2)
        or np.any(tracks < 0)
        or np.any(~np.isfinite(xy))
    ):
        raise ValueError("support observation index is invalid")
    image_rows = np.repeat(np.arange(len(image_ids), dtype=np.int32), np.diff(offsets))
    if len(image_rows) != len(tracks):
        raise RuntimeError("support observation index offsets do not cover tracks")
    return image_ids, image_rows, tracks, xy, metadata


def _load_hard_negative_bank(
    *, path: Path, expected_context_cache_sha256: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    required = {"track_ids", "features", "metadata_json"}
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"hard-negative landmark bank lacks {sorted(missing)}")
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("hard-negative landmark bank metadata is invalid") from error
        tracks = np.asarray(payload["track_ids"], dtype=np.int64).reshape(-1)
        features = np.asarray(payload["features"], dtype=np.float32)
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != _LANDMARK_BANK_FORMAT
        or str(metadata.get("context_cache_sha256", ""))
        != str(expected_context_cache_sha256)
        or tracks.size == 0
        or len(np.unique(tracks)) != len(tracks)
        or features.ndim != 2
        or features.shape[0] != len(tracks)
        or np.any(tracks < 0)
        or np.any(~np.isfinite(features))
    ):
        raise ValueError("hard-negative landmark bank is incompatible with context cache")
    normalized, valid = normalize_rows(features)
    if not np.all(valid):
        raise ValueError("hard-negative landmark bank has zero descriptors")
    return tracks, normalized, metadata


def _interior_mask(
    *, image_ids: np.ndarray, xy: np.ndarray, sizes_by_image: Mapping[str, np.ndarray], margin_px: float
) -> np.ndarray:
    ids = np.asarray(image_ids).astype(str).reshape(-1)
    coordinates = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    margin = float(margin_px)
    if margin < 0.0 or len(ids) != len(coordinates):
        raise ValueError("observation interior-mask inputs are invalid")
    output = np.zeros((len(ids),), dtype=bool)
    for image_id in sorted(set(ids.tolist())):
        size = sizes_by_image.get(str(image_id))
        if size is None:
            raise KeyError(f"observation image is absent from context cache: {image_id}")
        width, height = int(size[0]), int(size[1])
        rows = np.flatnonzero(ids == str(image_id))
        values = coordinates[rows]
        output[rows] = (
            (values[:, 0] >= margin)
            & (values[:, 0] <= float(width - 1) - margin)
            & (values[:, 1] >= margin)
            & (values[:, 1] <= float(height - 1) - margin)
        )
    return output


def _support_track_lookup(track_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(track_ids, dtype=np.int64), kind="stable")
    return np.asarray(track_ids, dtype=np.int64)[order], order.astype(np.int64, copy=False)


def _select_support_observation(
    *,
    track_id: int,
    sorted_track_ids: np.ndarray,
    sorted_rows: np.ndarray,
    support_interior: np.ndarray,
    salt: int,
) -> int:
    """Choose one deterministic interior mapping observation for a track."""

    left = int(np.searchsorted(sorted_track_ids, int(track_id), side="left"))
    right = int(np.searchsorted(sorted_track_ids, int(track_id), side="right"))
    if left >= right:
        return -1
    rows = sorted_rows[left:right]
    eligible = rows[np.asarray(support_interior, dtype=bool)[rows]]
    if len(eligible) == 0:
        return -1
    return int(eligible[int(salt) % len(eligible)])


def select_distinct_ann_tracks(
    *,
    ann_track_ids: np.ndarray,
    positive_track_id: int,
    negative_count: int,
) -> np.ndarray:
    """Keep distinct non-positive tracks in ANN order without label leakage."""

    candidates = np.asarray(ann_track_ids, dtype=np.int64).reshape(-1)
    selected: list[int] = []
    for track_id in candidates.tolist():
        if int(track_id) < 0 or int(track_id) == int(positive_track_id) or int(track_id) in selected:
            continue
        selected.append(int(track_id))
        if len(selected) == int(negative_count):
            break
    return np.asarray(selected, dtype=np.int64)


def _query_observation_candidates(
    *,
    query_ids: Sequence[str],
    images_by_name: Mapping[str, Any],
    max_anchors_per_query: int,
    seed: int,
    sizes_by_image: Mapping[str, np.ndarray],
    border_margin_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids_out: list[np.ndarray] = []
    xy_out: list[np.ndarray] = []
    tracks_out: list[np.ndarray] = []
    for query_id in sorted(set(str(value) for value in query_ids)):
        image = images_by_name.get(str(query_id))
        if image is None:
            raise KeyError(f"train query image is absent from COLMAP: {query_id}")
        xy = np.asarray(image.xys, dtype=np.float32).reshape(-1, 2)
        tracks = np.asarray(image.point3d_ids, dtype=np.int64).reshape(-1)
        if xy.shape != (len(tracks), 2):
            raise ValueError("COLMAP query observations are malformed")
        valid = (tracks >= 0) & np.all(np.isfinite(xy), axis=1)
        valid &= _interior_mask(
            image_ids=np.full((len(xy),), str(query_id)),
            xy=xy,
            sizes_by_image=sizes_by_image,
            margin_px=float(border_margin_px),
        )
        rows = np.flatnonzero(valid)
        if len(rows) == 0:
            continue
        limit = int(max_anchors_per_query)
        if limit > 0 and len(rows) > limit:
            generator = np.random.default_rng(_stable_hash(seed, query_id))
            rows = np.sort(generator.choice(rows, size=limit, replace=False))
        ids_out.append(np.full((len(rows),), str(query_id)))
        xy_out.append(xy[rows])
        tracks_out.append(tracks[rows])
    if not ids_out:
        raise ValueError("no interior train-query SfM observations are available")
    return (
        np.concatenate(ids_out),
        np.concatenate(xy_out).astype(np.float32, copy=False),
        np.concatenate(tracks_out).astype(np.int64, copy=False),
    )


def _faiss_search(features: np.ndarray, queries: np.ndarray, *, search_k: int) -> np.ndarray:
    try:
        import faiss
    except ImportError as error:  # pragma: no cover - dependency is part of the runtime image.
        raise RuntimeError("FAISS is required for global landmark hard-negative mining") from error
    if int(search_k) <= 0 or int(search_k) > len(features):
        raise ValueError("hard-negative ANN search_k is invalid")
    index = faiss.IndexFlatIP(int(features.shape[1]))
    index.add(np.ascontiguousarray(features.astype(np.float32, copy=False)))
    _scores, rows = index.search(
        np.ascontiguousarray(queries.astype(np.float32, copy=False)), int(search_k)
    )
    if rows.shape != (len(queries), int(search_k)) or np.any(rows < 0):
        raise RuntimeError("hard-negative ANN search returned incomplete rows")
    return rows.astype(np.int64, copy=False)


def build_candidate_pose_rgb_spatial_observation_pairs(
    *,
    train_query_layout: Path,
    colmap_model_dir: Path,
    support_observation_index: Path,
    hard_negative_context_cache: Path,
    hard_negative_landmark_bank: Path,
    output: Path,
    summary_json: Path,
    max_anchors_per_query: int,
    negative_count: int,
    ann_search_k: int,
    border_margin_px: float,
    inner_validation_fold_count: int,
    inner_validation_fold_index: int,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    """Construct reproducible train-only same-track and ANN-hard-negative rows."""

    if (
        int(max_anchors_per_query) == 0
        or int(max_anchors_per_query) < -1
        or int(negative_count) <= 0
        or int(ann_search_k) < int(negative_count) + 1
        or float(border_margin_px) < 0.0
    ):
        raise ValueError("observation-pair build arguments are invalid")
    output = Path(output)
    summary_path = Path(summary_json)
    if (output.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite observation-pair outputs")

    layout_path = Path(train_query_layout)
    layout = load_candidate_pose_rgb_spatial_layout(layout_path)
    if (
        layout.metadata.get("contains_ground_truth") is not False
        or layout.metadata.get("pose_or_ground_truth_used") is not False
        or layout.metadata.get("render") is not False
        or layout.metadata.get("image_retrieval_or_submap_used") is not False
    ):
        raise ValueError("train-query layout is not target-free")
    train_query_ids = tuple(
        sorted(
            set(
                np.asarray(layout.query_ids)[
                    np.asarray(layout.split_names).astype(str) == "train"
                ]
                .astype(str)
                .tolist()
            )
        )
    )
    if not train_query_ids:
        raise ValueError("train-query layout has no train query images")
    inner_train_ids, inner_validation_ids = partition_train_query_ids(
        query_ids=train_query_ids,
        fold_count=int(inner_validation_fold_count),
        fold_index=int(inner_validation_fold_index),
    )
    split_by_query = {
        **{value: "inner_train" for value in inner_train_ids},
        **{value: "inner_validation" for value in inner_validation_ids},
    }

    cache_path = Path(hard_negative_context_cache)
    context_cache: SpatialImageContextCache = load_spatial_image_context_cache(
        cache_path, expected_format=_PCA_FORMAT
    )
    if (
        context_cache.metadata.get("pose_or_ground_truth_used") is not False
        or context_cache.metadata.get("render") is not False
        or context_cache.metadata.get("image_retrieval_or_submap_used") is not False
        or context_cache.metadata.get("pca_fit_scope")
        != "mapping_support_images_excluding_all_query_splits_v1"
        or 16 not in context_cache.grids
    ):
        raise ValueError("hard-negative context cache violates the mapping-only contract")
    sizes_by_image = {
        str(image_id): np.asarray(size, dtype=np.int64)
        for image_id, size in zip(context_cache.image_ids.tolist(), context_cache.image_sizes)
    }
    if not set(train_query_ids).issubset(sizes_by_image):
        raise ValueError("train query images are absent from hard-negative context cache")

    support_ids, support_image_rows, support_tracks, support_xy, support_metadata = (
        _load_support_observation_index(Path(support_observation_index))
    )
    if set(train_query_ids).intersection(set(support_ids.tolist())):
        raise ValueError("mapping support index overlaps train query images")
    if not set(support_ids.tolist()).issubset(sizes_by_image):
        raise ValueError("support observation images are absent from hard-negative context cache")
    sorted_support_tracks, sorted_support_rows = _support_track_lookup(support_tracks)
    support_interior = _interior_mask(
        image_ids=support_ids[support_image_rows],
        xy=support_xy,
        sizes_by_image=sizes_by_image,
        margin_px=float(border_margin_px),
    )

    bank_tracks, bank_features, bank_metadata = _load_hard_negative_bank(
        path=Path(hard_negative_landmark_bank),
        expected_context_cache_sha256=file_sha256_short(cache_path),
    )
    if bank_features.shape[1] != context_cache.descriptor_dim:
        raise ValueError("hard-negative context cache and landmark bank dimensions differ")

    images_path = Path(colmap_model_dir) / "images.bin"
    images = read_colmap_images_binary(images_path)
    images_by_name = {str(image.image_name): image for image in images.values()}
    query_ids, query_xy, query_tracks = _query_observation_candidates(
        query_ids=train_query_ids,
        images_by_name=images_by_name,
        max_anchors_per_query=int(max_anchors_per_query),
        seed=int(seed),
        sizes_by_image=sizes_by_image,
        border_margin_px=float(border_margin_px),
    )

    positive_rows = np.asarray(
        [
            _select_support_observation(
                track_id=int(track_id),
                sorted_track_ids=sorted_support_tracks,
                sorted_rows=sorted_support_rows,
                support_interior=support_interior,
                salt=_stable_hash(seed, query_id, track_id, "positive"),
            )
            for query_id, track_id in zip(query_ids.tolist(), query_tracks.tolist())
        ],
        dtype=np.int64,
    )
    positive_valid = positive_rows >= 0
    query_ids = query_ids[positive_valid]
    query_xy = query_xy[positive_valid]
    query_tracks = query_tracks[positive_valid]
    positive_rows = positive_rows[positive_valid]
    if len(query_ids) == 0:
        raise ValueError("no train query observations have an interior mapping support observation")

    query_descriptors = sample_spatial_context_descriptors(
        context_cache,
        image_ids=query_ids,
        xy=query_xy,
        grid_size=16,
        boundary_mode="error",
    )
    ann_rows = _faiss_search(
        bank_features, query_descriptors, search_k=int(ann_search_k)
    )
    ann_tracks = bank_tracks[ann_rows]

    selected_anchor_rows: list[int] = []
    negative_support_rows: list[list[int]] = []
    skipped_insufficient_ann = 0
    for row, (query_id, positive_track) in enumerate(zip(query_ids.tolist(), query_tracks.tolist())):
        candidate_tracks = select_distinct_ann_tracks(
            ann_track_ids=ann_tracks[row],
            positive_track_id=int(positive_track),
            negative_count=int(negative_count) * 4,
        )
        selected_rows: list[int] = []
        selected_tracks: set[int] = set()
        for candidate_track in candidate_tracks.tolist():
            if int(candidate_track) in selected_tracks:
                continue
            support_row = _select_support_observation(
                track_id=int(candidate_track),
                sorted_track_ids=sorted_support_tracks,
                sorted_rows=sorted_support_rows,
                support_interior=support_interior,
                salt=_stable_hash(seed, query_id, positive_track, candidate_track, "negative"),
            )
            if support_row < 0:
                continue
            selected_rows.append(int(support_row))
            selected_tracks.add(int(candidate_track))
            if len(selected_rows) == int(negative_count):
                break
        if len(selected_rows) != int(negative_count):
            skipped_insufficient_ann += 1
            continue
        selected_anchor_rows.append(int(row))
        negative_support_rows.append(selected_rows)
    if not selected_anchor_rows:
        raise ValueError("global landmark ANN produced no complete hard-negative rows")

    keep = np.asarray(selected_anchor_rows, dtype=np.int64)
    negative_rows = np.asarray(negative_support_rows, dtype=np.int64)
    selected_query_ids = query_ids[keep]
    selected_query_xy = query_xy[keep]
    selected_tracks = query_tracks[keep]
    selected_positive_rows = positive_rows[keep]
    positive_image_ids = support_ids[support_image_rows[selected_positive_rows]]
    positive_xy = support_xy[selected_positive_rows]
    negative_image_ids = support_ids[support_image_rows[negative_rows]]
    negative_xy = support_xy[negative_rows]
    negative_tracks = support_tracks[negative_rows]
    if np.any(negative_tracks == selected_tracks[:, None]):
        raise RuntimeError("ANN hard-negative row retained its positive track")
    split_names = np.asarray([split_by_query[str(value)] for value in selected_query_ids], dtype=np.str_)
    if set(split_names.tolist()) != {"inner_train", "inner_validation"}:
        raise RuntimeError("observation-pair rows lost an inner train or validation split")

    metadata: dict[str, Any] = {
        "format": CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PAIR_FORMAT,
        "training_only_target_artifact": True,
        "contains_ground_truth": True,
        "contains_validation_or_test_targets": False,
        "runtime_scorer_must_not_load_this_artifact": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "candidate_set": "fixed_positive_same_track_plus_radio_pca_global_landmark_hard_negatives",
        "query_split": "train_only_inner_partition_v1",
        "hard_negative_semantics": "radio_intermediate_pca_global_landmark_ann_distinct_track_v1",
        "negative_count": int(negative_count),
        "ann_search_k": int(ann_search_k),
        "border_margin_px": float(border_margin_px),
        "max_anchors_per_query": int(max_anchors_per_query),
        "seed": int(seed),
        "inner_validation_fold_count": int(inner_validation_fold_count),
        "inner_validation_fold_index": int(inner_validation_fold_index),
        "train_query_layout": str(layout_path.resolve()),
        "train_query_layout_sha256": file_sha256_short(layout_path),
        "colmap_images_bin": str(images_path.resolve()),
        "colmap_images_bin_sha256": file_sha256_short(images_path),
        "support_observation_index": str(Path(support_observation_index).resolve()),
        "support_observation_index_sha256": file_sha256_short(Path(support_observation_index)),
        "hard_negative_context_cache": str(cache_path.resolve()),
        "hard_negative_context_cache_sha256": file_sha256_short(cache_path),
        "hard_negative_landmark_bank": str(Path(hard_negative_landmark_bank).resolve()),
        "hard_negative_landmark_bank_sha256": file_sha256_short(Path(hard_negative_landmark_bank)),
        "train_query_image_count": int(len(train_query_ids)),
        "train_query_image_list_sha256": _image_list_hash(train_query_ids),
        "mapping_support_image_count": int(len(support_ids)),
        "mapping_support_query_overlap_count": 0,
        "hard_negative_descriptor_dim": int(bank_features.shape[1]),
        "hard_negative_context_metadata": {
            "pca_fit_scope": context_cache.metadata.get("pca_fit_scope"),
            "source_image_manifest_sha256": context_cache.metadata.get(
                "source_image_manifest_sha256"
            ),
        },
        "hard_negative_bank_metadata": {
            "descriptor_space_id": bank_metadata.get("descriptor_space_id"),
            "projection_source": bank_metadata.get("projection_source"),
        },
    }
    pairs = CandidatePoseRGBSpatialObservationPairs(
        anchor_ids=np.arange(len(selected_query_ids), dtype=np.int64),
        query_image_ids=selected_query_ids,
        query_xy=selected_query_xy,
        positive_support_image_ids=positive_image_ids,
        positive_support_xy=positive_xy,
        positive_track_ids=selected_tracks,
        negative_support_image_ids=negative_image_ids,
        negative_support_xy=negative_xy,
        negative_track_ids=negative_tracks,
        negative_sources=np.full(
            negative_tracks.shape,
            "radio_intermediate_pca_global_landmark_ann",
            dtype=f"<U{len('radio_intermediate_pca_global_landmark_ann')}",
        ),
        split_names=split_names,
        metadata=metadata,
    )
    save_candidate_pose_rgb_spatial_observation_pairs(pairs, output)
    per_split = Counter(pairs.split_names.tolist())
    summary = {
        "stage": "build_candidate_pose_rgb_spatial_observation_pairs",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "row_count": int(pairs.row_count),
        "negative_count": int(pairs.negative_count),
        "rows_by_inner_split": {key: int(value) for key, value in sorted(per_split.items())},
        "query_images_by_inner_split": {
            "inner_train": int(len(inner_train_ids)),
            "inner_validation": int(len(inner_validation_ids)),
        },
        "candidate_source": "global_radio_intermediate_pca_landmark_ann_no_image_retrieval",
        "discarded": {
            "query_observations_without_interior_mapping_positive": int(
                np.count_nonzero(~positive_valid)
            ),
            "anchors_without_complete_distinct_ann_negatives": int(skipped_insufficient_ann),
        },
        "protocol": {
            "train_query_only": True,
            "contains_validation_or_test_targets": False,
            "runtime_scorer_loads_pairs": False,
            "no_render": True,
            "no_image_retrieval_or_submap": True,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_candidate_pose_rgb_spatial_observation_pairs(
        train_query_layout=Path(args.train_query_layout),
        colmap_model_dir=Path(args.colmap_model_dir),
        support_observation_index=Path(args.support_observation_index),
        hard_negative_context_cache=Path(args.hard_negative_context_cache),
        hard_negative_landmark_bank=Path(args.hard_negative_landmark_bank),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        max_anchors_per_query=int(args.max_anchors_per_query),
        negative_count=int(args.negative_count),
        ann_search_k=int(args.ann_search_k),
        border_margin_px=float(args.border_margin_px),
        inner_validation_fold_count=int(args.inner_validation_fold_count),
        inner_validation_fold_index=int(args.inner_validation_fold_index),
        seed=int(args.seed),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
