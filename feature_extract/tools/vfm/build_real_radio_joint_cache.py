"""Build real-image RADIO joint localization caches from SfM track rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from feature_extract.vfm.matcha_coarse_supervision import (
    MatchaCoarseSupervision,
    cell_offset_labels,
    cell_offset_soft_labels,
)
from feature_extract.vfm.matcha_joint_cache import build_matcha_joint_index_training_set_from_maps
from feature_extract.vfm.matcha_joint_training import (
    MatchaJointTrainingSet,
    save_matcha_joint_training_set_manifest,
)


REFERENCED_MANIFEST_FORMAT = "vfm_real_radio_joint_referenced_manifest_v1"
TRACK_OBSERVATION_INDEX_FORMAT = "sfm_track_observation_index_v1"


def _file_sha256_short(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _first_text(row: Mapping[str, object], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def _optional_float(row: Mapping[str, object], *names: str, default: float = np.nan) -> float:
    text = _first_text(row, *names)
    if not text:
        return float(default)
    try:
        return float(text)
    except ValueError:
        return float(default)


def _optional_int(row: Mapping[str, object], *names: str, default: int = -1) -> int:
    text = _first_text(row, *names)
    if not text:
        return int(default)
    try:
        return int(text)
    except ValueError:
        return int(default)


@dataclass(frozen=True)
class SfMImageTrackObservations:
    track_ids: np.ndarray
    xy: np.ndarray
    image_sizes: np.ndarray
    xyz: np.ndarray
    track_lengths: np.ndarray


@dataclass(frozen=True)
class SfMTrackObservationIndex:
    by_image: dict[str, SfMImageTrackObservations]
    track_xyz_by_id: dict[int, np.ndarray]

    def common_tracks(
        self,
        query_image_id: str,
        reference_image_id: str,
        *,
        query_source_size: tuple[int, int],
        reference_source_size: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        query = self.by_image.get(str(query_image_id))
        reference = self.by_image.get(str(reference_image_id))
        if query is None or reference is None:
            return _empty_landmark_retrieval_supervision()
        track_ids, query_indices, reference_indices = np.intersect1d(
            query.track_ids,
            reference.track_ids,
            assume_unique=True,
            return_indices=True,
        )
        if track_ids.size == 0:
            return _empty_landmark_retrieval_supervision()

        def scale_xy(
            xy: np.ndarray,
            source_sizes: np.ndarray,
            target_size: tuple[int, int],
        ) -> np.ndarray:
            target_width, target_height = map(int, target_size)
            widths = source_sizes[:, 0].astype(np.float64)
            heights = source_sizes[:, 1].astype(np.float64)
            source_aspect = widths / np.maximum(heights, 1.0)
            target_aspect = float(target_width) / max(float(target_height), 1.0)
            if np.any(np.abs(source_aspect - target_aspect) > 1e-3):
                raise ValueError("SfM and source RGB aspect ratios differ; an explicit crop transform is required")
            output = np.asarray(xy, dtype=np.float64).copy()
            output[:, 0] *= max(float(target_width - 1), 1.0) / np.maximum(widths - 1.0, 1.0)
            output[:, 1] *= max(float(target_height - 1), 1.0) / np.maximum(heights - 1.0, 1.0)
            return output

        query_xy = scale_xy(query.xy[query_indices], query.image_sizes[query_indices], query_source_size)
        reference_xy = scale_xy(
            reference.xy[reference_indices],
            reference.image_sizes[reference_indices],
            reference_source_size,
        )
        return {
            "query_xy": query_xy,
            "reference_xy": reference_xy,
            "track_ids": track_ids.astype(np.int64, copy=False),
            "track_xyz": query.xyz[query_indices].astype(np.float64, copy=False),
            "support_view_counts": reference.track_lengths[reference_indices].astype(np.int64, copy=False),
        }


def _empty_landmark_retrieval_supervision() -> dict[str, np.ndarray]:
    return {
        "query_xy": np.zeros((0, 2), dtype=np.float64),
        "reference_xy": np.zeros((0, 2), dtype=np.float64),
        "track_ids": np.zeros((0,), dtype=np.int64),
        "track_xyz": np.zeros((0, 3), dtype=np.float64),
        "support_view_counts": np.zeros((0,), dtype=np.int64),
    }


def _save_track_observation_index_npz(
    index: SfMTrackObservationIndex,
    path: Path,
    *,
    source_jsonl: Path,
) -> None:
    image_ids = sorted(index.by_image)
    offsets = np.zeros((len(image_ids) + 1,), dtype=np.int64)
    for image_index, image_id in enumerate(image_ids):
        offsets[image_index + 1] = offsets[image_index] + int(index.by_image[image_id].track_ids.size)
    track_ids = np.concatenate([index.by_image[image_id].track_ids for image_id in image_ids], axis=0)
    xy = np.concatenate([index.by_image[image_id].xy for image_id in image_ids], axis=0)
    image_sizes = np.concatenate([index.by_image[image_id].image_sizes for image_id in image_ids], axis=0)
    xyz = np.concatenate([index.by_image[image_id].xyz for image_id in image_ids], axis=0)
    track_lengths = np.concatenate([index.by_image[image_id].track_lengths for image_id in image_ids], axis=0)
    unique_track_ids = np.asarray(sorted(index.track_xyz_by_id), dtype=np.int64)
    unique_track_xyz = np.stack([index.track_xyz_by_id[int(track_id)] for track_id in unique_track_ids], axis=0)
    source = Path(source_jsonl)
    metadata = {
        "format": TRACK_OBSERVATION_INDEX_FORMAT,
        "source_path": str(source),
        "source_size": int(source.stat().st_size),
        "source_sha256": _file_sha256_short(source),
        "image_count": int(len(image_ids)),
        "observation_count": int(track_ids.size),
        "track_count": int(unique_track_ids.size),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez(
        temporary,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        image_ids=np.asarray(image_ids, dtype=np.str_),
        offsets=offsets,
        track_ids=track_ids,
        xy=xy,
        image_sizes=image_sizes,
        xyz=xyz,
        track_lengths=track_lengths,
        unique_track_ids=unique_track_ids,
        unique_track_xyz=unique_track_xyz,
    )
    temporary.replace(output)


def _load_track_observation_index_npz(path: Path, *, source_jsonl: Path) -> SfMTrackObservationIndex:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if str(metadata.get("format", "")) != TRACK_OBSERVATION_INDEX_FORMAT:
            raise ValueError(f"unsupported track observation index cache: {path}")
        source = Path(source_jsonl)
        expected_size = int(source.stat().st_size)
        expected_sha = _file_sha256_short(source)
        if int(metadata.get("source_size", -1)) != expected_size or str(metadata.get("source_sha256", "")) != expected_sha:
            raise ValueError(f"stale track observation index cache for {source}: {path}")
        image_ids = np.asarray(data["image_ids"], dtype=str)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        track_ids = np.asarray(data["track_ids"], dtype=np.int64)
        xy = np.asarray(data["xy"], dtype=np.float64)
        image_sizes = np.asarray(data["image_sizes"], dtype=np.int64)
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        track_lengths = np.asarray(data["track_lengths"], dtype=np.int64)
        unique_track_ids = np.asarray(data["unique_track_ids"], dtype=np.int64)
        unique_track_xyz = np.asarray(data["unique_track_xyz"], dtype=np.float64)
    by_image = {
        str(image_id): SfMImageTrackObservations(
            track_ids=track_ids[int(offsets[index]) : int(offsets[index + 1])],
            xy=xy[int(offsets[index]) : int(offsets[index + 1])],
            image_sizes=image_sizes[int(offsets[index]) : int(offsets[index + 1])],
            xyz=xyz[int(offsets[index]) : int(offsets[index + 1])],
            track_lengths=track_lengths[int(offsets[index]) : int(offsets[index + 1])],
        )
        for index, image_id in enumerate(image_ids.tolist())
    }
    track_xyz_by_id = {
        int(track_id): unique_track_xyz[index]
        for index, track_id in enumerate(unique_track_ids.tolist())
    }
    return SfMTrackObservationIndex(by_image=by_image, track_xyz_by_id=track_xyz_by_id)


def load_track_observation_index(
    path: Path,
    *,
    cache_path: Path | None = None,
) -> SfMTrackObservationIndex:
    if cache_path is not None and Path(cache_path).exists():
        return _load_track_observation_index_npz(Path(cache_path), source_jsonl=Path(path))
    grouped: dict[str, dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray, int, float]]] = {}
    xyz_lookup: dict[int, np.ndarray] = {}
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            track_id = int(row["track_id"])
            xyz = np.asarray(row["xyz"], dtype=np.float64).reshape(3)
            xy = np.asarray(row["xy"], dtype=np.float64).reshape(2)
            image_size = np.asarray([int(row["image_width"]), int(row["image_height"])], dtype=np.int64)
            if not np.isfinite(xyz).all() or not np.isfinite(xy).all() or np.any(image_size <= 0):
                continue
            previous = xyz_lookup.get(track_id)
            if previous is not None and not np.allclose(previous, xyz, rtol=0.0, atol=1e-6):
                raise ValueError(f"inconsistent xyz for track {track_id} at {path}:{line_number}")
            xyz_lookup.setdefault(track_id, xyz)
            image_id = str(row["image_id"])
            reprojection_error = float(row.get("reprojection_error", np.inf))
            value = (track_id, xy, image_size, xyz, int(row.get("track_length", 0)), reprojection_error)
            existing = grouped.setdefault(image_id, {}).get(track_id)
            if existing is None or float(value[5]) < float(existing[5]):
                grouped[image_id][track_id] = value
    by_image: dict[str, SfMImageTrackObservations] = {}
    for image_id, by_track in grouped.items():
        values = list(by_track.values())
        values.sort(key=lambda item: int(item[0]))
        track_ids = np.asarray([item[0] for item in values], dtype=np.int64)
        by_image[image_id] = SfMImageTrackObservations(
            track_ids=track_ids,
            xy=np.stack([item[1] for item in values], axis=0),
            image_sizes=np.stack([item[2] for item in values], axis=0),
            xyz=np.stack([item[3] for item in values], axis=0),
            track_lengths=np.asarray([item[4] for item in values], dtype=np.int64),
        )
    if not by_image or not xyz_lookup:
        raise ValueError(f"track observation file contains no finite observations: {path}")
    index = SfMTrackObservationIndex(by_image=by_image, track_xyz_by_id=xyz_lookup)
    if cache_path is not None:
        _save_track_observation_index_npz(index, Path(cache_path), source_jsonl=Path(path))
    return index


def load_track_xyz_lookup(path: Path) -> dict[int, np.ndarray]:
    """Load one canonical XYZ per SfM track from observation JSONL."""
    return load_track_observation_index(Path(path)).track_xyz_by_id


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _image_stem(image_id: str) -> str:
    text = str(image_id).replace("\\", "/").strip("/")
    stem = str(Path(text).with_suffix(""))
    return stem.replace("/", "_")


def _image_token(image_id: str) -> str:
    return str(image_id).replace("\\", "/").strip("/").replace("/", "__")


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(base_dir) / value


def _feature_path_from_template(template: str, *, image_id: str) -> Path:
    if not str(template).strip():
        raise ValueError("feature path missing and no feature_path_template was provided")
    return Path(
        str(template).format(
            image_id=str(image_id),
            image_stem=_image_stem(str(image_id)),
            image_token=_image_token(str(image_id)),
        )
    )


def _load_feature_map(path: Path, *, key: str) -> np.ndarray:
    value = Path(path)
    if not value.exists():
        raise FileNotFoundError(value)
    if value.suffix == ".npy":
        arr = np.load(value)
    else:
        with np.load(value) as data:
            if str(key):
                if str(key) not in data:
                    raise KeyError(f"{value} does not contain feature key {key!r}")
                arr = data[str(key)]
            elif len(data.files) == 1:
                arr = data[data.files[0]]
            else:
                raise ValueError(f"{value} contains multiple arrays; pass --feature_key")
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 3:
        raise ValueError(f"feature map at {value} must have shape (C,H,W)")
    return out


def _load_rgb_hwc(path: Path) -> np.ndarray:
    return np.asarray(Image.open(Path(path)).convert("RGB"), dtype=np.uint8)


def _cell_indices(
    xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    cell_w = float(image_width) / max(float(grid_width), 1.0)
    cell_h = float(image_height) / max(float(grid_height), 1.0)
    col = np.floor(coords[:, 0] / cell_w).astype(np.int64)
    row = np.floor(coords[:, 1] / cell_h).astype(np.int64)
    valid = (
        np.isfinite(coords).all(axis=1)
        & (coords[:, 0] >= 0.0)
        & (coords[:, 0] < float(image_width))
        & (coords[:, 1] >= 0.0)
        & (coords[:, 1] < float(image_height))
        & (col >= 0)
        & (col < int(grid_width))
        & (row >= 0)
        & (row < int(grid_height))
    )
    idx = row * int(grid_width) + col
    idx[~valid] = -1
    return idx.astype(np.int64, copy=False), valid


def _row_xy(row: Mapping[str, object]) -> tuple[np.ndarray | None, np.ndarray | None]:
    query_x = _first_text(row, "query_gt_x", "gt_query_x")
    query_y = _first_text(row, "query_gt_y", "gt_query_y")
    reference_x = _first_text(row, "reference_gt_x", "support_x", "render_x", "gt_reference_x")
    reference_y = _first_text(row, "reference_gt_y", "support_y", "render_y", "gt_reference_y")
    if not query_x or not query_y or not reference_x or not reference_y:
        return None, None
    return (
        np.asarray([float(query_x), float(query_y)], dtype=np.float64),
        np.asarray([float(reference_x), float(reference_y)], dtype=np.float64),
    )


def _dedupe_matches(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for item in items:
        key = (int(item["query_index"]), int(item["reference_index"]))
        existing = by_pair.get(key)
        if existing is None or float(item["uncertainty_px"]) < float(existing["uncertainty_px"]):
            by_pair[key] = item
    return sorted(by_pair.values(), key=lambda value: (int(value["query_index"]), int(value["reference_index"])))


def _dedupe_track_observations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_track: dict[int, dict[str, Any]] = {}
    for item in items:
        track_id = int(item["track_id"])
        if track_id < 0 or not bool(item.get("same_track", False)):
            continue
        existing = by_track.get(track_id)
        if existing is None or float(item["uncertainty_px"]) < float(existing["uncertainty_px"]):
            by_track[track_id] = item
    return [by_track[track_id] for track_id in sorted(by_track)]


def _confidence_from_error(error_px: float, *, positive_error_px: float) -> float:
    if not np.isfinite(float(error_px)):
        return 1.0
    if float(positive_error_px) <= 0.0:
        return 1.0
    return float(np.clip(1.0 - float(error_px) / float(positive_error_px), 0.0, 1.0))


def _build_supervision_for_pair(
    rows: Sequence[Mapping[str, object]],
    *,
    query_image_size: tuple[int, int],
    reference_image_size: tuple[int, int],
    query_grid_hw: tuple[int, int],
    reference_grid_hw: tuple[int, int],
    positive_reprojection_error_px: float,
    require_same_track: bool,
    include_dustbin_rows: bool,
    track_xyz_by_id: Mapping[int, np.ndarray] | None = None,
) -> tuple[MatchaCoarseSupervision | None, dict[str, int], dict[str, np.ndarray]]:
    q_width, q_height = int(query_image_size[0]), int(query_image_size[1])
    r_width, r_height = int(reference_image_size[0]), int(reference_image_size[1])
    q_grid_h, q_grid_w = int(query_grid_hw[0]), int(query_grid_hw[1])
    r_grid_h, r_grid_w = int(reference_grid_hw[0]), int(reference_grid_hw[1])
    positive: list[dict[str, Any]] = []
    no_match: list[dict[str, Any]] = []
    skipped = {
        "missing_xy": 0,
        "out_of_bounds": 0,
        "track_mismatch": 0,
        "dustbin_rows": 0,
    }
    for row in rows:
        query_xy, reference_xy = _row_xy(row)
        if query_xy is None or reference_xy is None:
            skipped["missing_xy"] += 1
            continue
        qidx, qvalid = _cell_indices(
            query_xy[None],
            image_width=q_width,
            image_height=q_height,
            grid_width=q_grid_w,
            grid_height=q_grid_h,
        )
        ridx, rvalid = _cell_indices(
            reference_xy[None],
            image_width=r_width,
            image_height=r_height,
            grid_width=r_grid_w,
            grid_height=r_grid_h,
        )
        qlabels, qlabel_valid = cell_offset_labels(
            query_xy[None],
            image_width=q_width,
            image_height=q_height,
            grid_width=q_grid_w,
            grid_height=q_grid_h,
        )
        rlabels, rlabel_valid = cell_offset_labels(
            reference_xy[None],
            image_width=r_width,
            image_height=r_height,
            grid_width=r_grid_w,
            grid_height=r_grid_h,
        )
        qsoft, _ = cell_offset_soft_labels(
            query_xy[None],
            image_width=q_width,
            image_height=q_height,
            grid_width=q_grid_w,
            grid_height=q_grid_h,
        )
        rsoft, _ = cell_offset_soft_labels(
            reference_xy[None],
            image_width=r_width,
            image_height=r_height,
            grid_width=r_grid_w,
            grid_height=r_grid_h,
        )
        if not (bool(qvalid[0]) and bool(rvalid[0]) and bool(qlabel_valid[0]) and bool(rlabel_valid[0])):
            skipped["out_of_bounds"] += 1
            continue
        qerr = _optional_float(row, "query_reprojection_error", default=np.nan)
        rerr = _optional_float(row, "support_reprojection_error", "reference_reprojection_error", default=np.nan)
        residual = float(np.nanmax(np.asarray([qerr, rerr], dtype=np.float32))) if np.isfinite([qerr, rerr]).any() else 0.0
        item = {
            "query_index": int(qidx[0]),
            "reference_index": int(ridx[0]),
            "query_xy": query_xy.astype(np.float64, copy=False),
            "reference_xy": reference_xy.astype(np.float64, copy=False),
            "query_label": int(qlabels[0]),
            "reference_label": int(rlabels[0]),
            "query_soft": qsoft[0].astype(np.float32, copy=False),
            "reference_soft": rsoft[0].astype(np.float32, copy=False),
            "uncertainty_px": float(residual),
        }
        track_id = _optional_int(row, "track_id")
        support_track_id = _optional_int(row, "support_track_id", "reference_track_id")
        is_dustbin = _parse_bool(row.get("target_is_dustbin", ""))
        same_track = track_id >= 0 and track_id == support_track_id
        if is_dustbin:
            skipped["dustbin_rows"] += 1
            if include_dustbin_rows:
                no_match.append(item)
            continue
        if require_same_track and not same_track:
            skipped["track_mismatch"] += 1
            continue
        item["track_id"] = int(track_id)
        item["same_track"] = bool(same_track)
        item["support_view_count"] = max(0, _optional_int(row, "track_length", default=0))
        item["landmark_xyz"] = np.asarray(
            np.full((3,), np.nan, dtype=np.float64)
            if track_xyz_by_id is None or int(track_id) not in track_xyz_by_id
            else track_xyz_by_id[int(track_id)],
            dtype=np.float64,
        ).reshape(3)
        item["confidence"] = _confidence_from_error(residual, positive_error_px=float(positive_reprojection_error_px))
        positive.append(item)
    retrieval_positive = _dedupe_track_observations(positive)
    positive = _dedupe_matches(retrieval_positive)
    empty_retrieval = {
        "query_xy": np.zeros((0, 2), dtype=np.float64),
        "reference_xy": np.zeros((0, 2), dtype=np.float64),
        "track_ids": np.zeros((0,), dtype=np.int64),
        "track_xyz": np.zeros((0, 3), dtype=np.float64),
        "support_view_counts": np.zeros((0,), dtype=np.int64),
    }
    if not positive:
        return None, skipped, empty_retrieval
    no_match = _dedupe_matches(no_match)
    retrieval = empty_retrieval
    if retrieval_positive:
        retrieval = {
            "query_xy": np.stack([item["query_xy"] for item in retrieval_positive], axis=0),
            "reference_xy": np.stack([item["reference_xy"] for item in retrieval_positive], axis=0),
            "track_ids": np.asarray([item["track_id"] for item in retrieval_positive], dtype=np.int64),
            "track_xyz": np.stack([item["landmark_xyz"] for item in retrieval_positive], axis=0),
            "support_view_counts": np.asarray(
                [item["support_view_count"] for item in retrieval_positive],
                dtype=np.int64,
            ),
        }
    return (
        MatchaCoarseSupervision(
            query_indices=np.asarray([item["query_index"] for item in positive], dtype=np.int64),
            render_indices=np.asarray([item["reference_index"] for item in positive], dtype=np.int64),
            query_xy=np.stack([item["query_xy"] for item in positive], axis=0),
            render_xy=np.stack([item["reference_xy"] for item in positive], axis=0),
            query_offset_labels=np.asarray([item["query_label"] for item in positive], dtype=np.int64),
            render_offset_labels=np.asarray([item["reference_label"] for item in positive], dtype=np.int64),
            roundtrip_errors_px=np.asarray([item["uncertainty_px"] for item in positive], dtype=np.float32),
            query_offset_soft_labels=np.stack([item["query_soft"] for item in positive], axis=0),
            render_offset_soft_labels=np.stack([item["reference_soft"] for item in positive], axis=0),
            confidence_targets=np.asarray([item["confidence"] for item in positive], dtype=np.float32),
            uncertainty_px=np.asarray([item["uncertainty_px"] for item in positive], dtype=np.float32),
            no_match_query_indices=np.asarray([item["query_index"] for item in no_match], dtype=np.int64),
            no_match_render_indices=np.asarray([item["reference_index"] for item in no_match], dtype=np.int64),
            no_match_query_offset_labels=np.asarray([item["query_label"] for item in no_match], dtype=np.int64),
            no_match_render_offset_labels=np.asarray([item["reference_label"] for item in no_match], dtype=np.int64),
            no_match_roundtrip_errors_px=np.asarray([item["uncertainty_px"] for item in no_match], dtype=np.float32),
            no_match_confidence_targets=np.zeros((len(no_match),), dtype=np.float32),
            source="geometry_3dgs_multiview",
            support_view_counts=np.asarray([item["support_view_count"] for item in positive], dtype=np.int64),
            track_ids=np.asarray([item["track_id"] for item in positive], dtype=np.int64),
            landmark_xyz=np.stack([item["landmark_xyz"] for item in positive], axis=0),
        ),
        skipped,
        retrieval,
    )


def _group_rows(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        query_id = _first_text(row, "query_id")
        reference_id = _first_text(row, "reference_image_id", "support_image_id")
        if not query_id or not reference_id:
            continue
        query_feature = _first_text(row, "query_feature_path")
        reference_feature = _first_text(row, "reference_feature_path", "support_feature_path")
        grouped.setdefault((query_id, reference_id, query_feature, reference_feature), []).append(dict(row))
    return grouped


def _group_row_indices(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str, str], list[int]]:
    grouped: dict[tuple[str, str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        query_id = _first_text(row, "query_id")
        reference_id = _first_text(row, "reference_image_id", "support_image_id")
        if not query_id or not reference_id:
            continue
        query_feature = _first_text(row, "query_feature_path")
        reference_feature = _first_text(row, "reference_feature_path", "support_feature_path")
        grouped.setdefault((query_id, reference_id, query_feature, reference_feature), []).append(int(index))
    return grouped


def _has_positive_candidate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    require_same_track: bool,
) -> bool:
    for row in rows:
        query_xy, reference_xy = _row_xy(row)
        if query_xy is None or reference_xy is None:
            continue
        if _parse_bool(row.get("target_is_dustbin", "")):
            continue
        track_id = _optional_int(row, "track_id")
        support_track_id = _optional_int(row, "support_track_id", "reference_track_id")
        same_track = track_id >= 0 and track_id == support_track_id
        if bool(require_same_track) and not same_track:
            continue
        return True
    return False


def _path_for_feature(
    explicit_path: str,
    *,
    image_id: str,
    feature_root: Path,
    feature_path_template: str,
) -> Path:
    path = Path(explicit_path) if explicit_path else _feature_path_from_template(feature_path_template, image_id=image_id)
    return _resolve_path(path, base_dir=feature_root)


def _set_pair_metadata(joint: MatchaJointTrainingSet, *, query_id: str, reference_id: str, split_name: str) -> MatchaJointTrainingSet:
    object.__setattr__(joint, "pair_type_ids", np.asarray([0], dtype=np.int64))
    object.__setattr__(joint, "pair_type_names", np.asarray(["real_real"], dtype=object))
    object.__setattr__(joint, "pair_query_ids", np.asarray([str(query_id)], dtype=object))
    object.__setattr__(joint, "pair_split_names", np.asarray([str(split_name)], dtype=object))
    object.__setattr__(joint, "pair_candidate_ids", np.asarray([str(reference_id)], dtype=object))
    object.__setattr__(joint, "pair_translation_errors_m", np.asarray([0.0], dtype=np.float32))
    object.__setattr__(joint, "pair_rotation_errors_deg", np.asarray([0.0], dtype=np.float32))
    return joint


def _build_joint_set_for_real_pair(
    *,
    rows: Sequence[Mapping[str, object]],
    query_id: str,
    reference_id: str,
    query_feature: np.ndarray,
    reference_feature: np.ndarray,
    query_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    split_name: str,
    hard_negatives_per_match: int,
    roundtrip_heatmap_threshold_px: float,
    positive_reprojection_error_px: float,
    require_same_track: bool,
    include_dustbin_rows: bool,
    track_xyz_by_id: Mapping[int, np.ndarray] | None = None,
    track_observation_index: SfMTrackObservationIndex | None = None,
) -> tuple[MatchaJointTrainingSet, dict[str, int]]:
    if int(query_feature.shape[0]) != int(reference_feature.shape[0]):
        raise ValueError("query/reference feature-map channels must match")
    supervision, skip_counts, retrieval = _build_supervision_for_pair(
        rows,
        query_image_size=(int(query_rgb.shape[1]), int(query_rgb.shape[0])),
        reference_image_size=(int(reference_rgb.shape[1]), int(reference_rgb.shape[0])),
        query_grid_hw=(int(query_feature.shape[1]), int(query_feature.shape[2])),
        reference_grid_hw=(int(reference_feature.shape[1]), int(reference_feature.shape[2])),
        positive_reprojection_error_px=float(positive_reprojection_error_px),
        require_same_track=bool(require_same_track),
        include_dustbin_rows=bool(include_dustbin_rows),
        track_xyz_by_id=track_xyz_by_id,
    )
    if supervision is None or int(supervision.count) == 0:
        raise ValueError("real pair has no valid positive supervision")
    if track_observation_index is not None:
        retrieval = track_observation_index.common_tracks(
            query_id,
            reference_id,
            query_source_size=(int(query_rgb.shape[1]), int(query_rgb.shape[0])),
            reference_source_size=(int(reference_rgb.shape[1]), int(reference_rgb.shape[0])),
        )
        if int(np.asarray(retrieval["track_ids"]).shape[0]) == 0:
            raise ValueError("real pair has no common SfM tracks for landmark retrieval")
    joint = build_matcha_joint_index_training_set_from_maps(
        query_feature,
        reference_feature,
        supervision,
        fine_supervision=supervision,
        query_rgb=query_rgb,
        render_rgb=reference_rgb,
        hard_negatives_per_match=int(hard_negatives_per_match),
        roundtrip_heatmap_threshold_px=float(roundtrip_heatmap_threshold_px),
    )
    retrieval_count = int(np.asarray(retrieval["track_ids"]).shape[0])
    joint = replace(
        joint,
        landmark_sample_pair_indices=np.zeros((retrieval_count,), dtype=np.int64),
        landmark_query_xy=np.asarray(retrieval["query_xy"], dtype=np.float64),
        landmark_reference_xy=np.asarray(retrieval["reference_xy"], dtype=np.float64),
        landmark_track_ids=np.asarray(retrieval["track_ids"], dtype=np.int64),
        landmark_track_xyz=np.asarray(retrieval["track_xyz"], dtype=np.float64),
        landmark_support_view_counts=np.asarray(retrieval["support_view_counts"], dtype=np.int64),
    )
    return _set_pair_metadata(joint, query_id=query_id, reference_id=reference_id, split_name=split_name), skip_counts


class RealRadioReferencedJointSampleProvider:
    """Load one real-image RADIO pair on demand from a referenced manifest."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        feature_cache_size: int = 4,
        rgb_cache_size: int = 8,
        track_xyz_by_id: Mapping[int, np.ndarray] | None = None,
        track_observation_index: SfMTrackObservationIndex | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.metadata = json.loads(self.manifest_path.read_text())
        if str(self.metadata.get("format", "")) != REFERENCED_MANIFEST_FORMAT:
            raise ValueError(f"unsupported real RADIO referenced manifest format in {manifest_path}")
        base_dir = self.manifest_path.parent
        self.rows_csv = _resolve_path(str(self.metadata.get("rows_csv", "")), base_dir=base_dir)
        self.image_root = _resolve_path(str(self.metadata.get("image_root", "")), base_dir=base_dir)
        self.feature_root = _resolve_path(str(self.metadata.get("feature_root", "")), base_dir=base_dir)
        self.feature_path_template = str(self.metadata.get("feature_path_template", "{image_stem}.npz"))
        self.feature_key = str(self.metadata.get("feature_key", "radio_final"))
        self.split_name = str(self.metadata.get("split_name", "train"))
        self.hard_negatives_per_match = int(self.metadata.get("hard_negatives_per_match", 16))
        self.roundtrip_heatmap_threshold_px = float(self.metadata.get("roundtrip_heatmap_threshold_px", 2.0))
        self.positive_reprojection_error_px = float(self.metadata.get("positive_reprojection_error_px", 2.0))
        self.require_same_track = bool(self.metadata.get("require_same_track", True))
        self.include_dustbin_rows = bool(self.metadata.get("include_dustbin_rows", True))
        track_observations_text = str(self.metadata.get("track_observations", "")).strip()
        if track_observation_index is None and track_observations_text:
            track_observation_index = load_track_observation_index(
                _resolve_path(track_observations_text, base_dir=base_dir)
            )
        if track_xyz_by_id is None and track_observation_index is not None:
            track_xyz_by_id = track_observation_index.track_xyz_by_id
        self.track_xyz_by_id = track_xyz_by_id
        self.track_observation_index = track_observation_index
        self.records = [dict(item) for item in self.metadata.get("records", [])]
        if not self.records:
            raise ValueError(f"{manifest_path} contains no referenced pair records")
        self.rows = _read_csv(self.rows_csv)
        self.feature_cache_size = max(0, int(feature_cache_size))
        self.rgb_cache_size = max(0, int(rgb_cache_size))
        self._feature_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._rgb_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = RLock()
        self._landmark_retrieval_audit: dict[str, object] | None = None

    def __len__(self) -> int:
        return int(len(self.records))

    def landmark_retrieval_audit(self) -> dict[str, object]:
        if self._landmark_retrieval_audit is not None:
            return dict(self._landmark_retrieval_audit)
        pair_counts: Counter[int] = Counter()
        positive_row_count = 0
        for record in self.records:
            pair_track_ids: set[int] = set()
            if self.track_observation_index is not None:
                query = self.track_observation_index.by_image.get(str(record.get("query_id", "")))
                reference = self.track_observation_index.by_image.get(str(record.get("reference_image_id", "")))
                if query is not None and reference is not None:
                    pair_track_ids.update(
                        np.intersect1d(query.track_ids, reference.track_ids, assume_unique=True).tolist()
                    )
                    positive_row_count += int(len(pair_track_ids))
            else:
                for row in self._record_rows(record):
                    if _parse_bool(row.get("target_is_dustbin", "")):
                        continue
                    track_id = _optional_int(row, "track_id")
                    support_track_id = _optional_int(row, "support_track_id", "reference_track_id")
                    if track_id < 0 or track_id != support_track_id:
                        continue
                    pair_track_ids.add(int(track_id))
                    positive_row_count += 1
            pair_counts.update(pair_track_ids)
        pair_track_link_count = int(sum(pair_counts.values()))
        unique_track_count = int(len(pair_counts))
        repeated_link_count = max(0, pair_track_link_count - unique_track_count)
        audit = {
            "pair_count": int(len(self.records)),
            "positive_row_count": int(positive_row_count),
            "pair_track_link_count": pair_track_link_count,
            "unique_track_count": unique_track_count,
            "tracks_seen_at_least_twice": int(sum(count >= 2 for count in pair_counts.values())),
            "max_pairs_per_track": int(max(pair_counts.values(), default=0)),
            "expected_first_epoch_history_hit_fraction_without_eviction": float(
                repeated_link_count / max(1, pair_track_link_count)
            ),
            "supervision_source": (
                "sfm_common_track_observations"
                if self.track_observation_index is not None
                else "measurement_csv_track_rows_fallback"
            ),
        }
        self._landmark_retrieval_audit = dict(audit)
        return audit

    def _remember(self, cache: OrderedDict[str, np.ndarray], key: str, value: np.ndarray, *, max_size: int) -> np.ndarray:
        if max_size <= 0:
            return value
        with self._cache_lock:
            cache[str(key)] = value
            while len(cache) > int(max_size):
                cache.popitem(last=False)
        return value

    def _feature(self, path: Path) -> np.ndarray:
        key = str(Path(path))
        with self._cache_lock:
            if key in self._feature_cache:
                value = self._feature_cache.pop(key)
                self._feature_cache[key] = value
                return value
        return self._remember(
            self._feature_cache,
            key,
            _load_feature_map(Path(path), key=self.feature_key),
            max_size=self.feature_cache_size,
        )

    def _rgb(self, image_id: str) -> np.ndarray:
        key = str(image_id)
        with self._cache_lock:
            if key in self._rgb_cache:
                value = self._rgb_cache.pop(key)
                self._rgb_cache[key] = value
                return value
        return self._remember(
            self._rgb_cache,
            key,
            _load_rgb_hwc(Path(self.image_root) / str(image_id)),
            max_size=self.rgb_cache_size,
        )

    def _record_rows(self, record: Mapping[str, object]) -> list[dict[str, str]]:
        row_indices = [int(value) for value in record.get("row_indices", [])]
        rows: list[dict[str, str]] = []
        for row_index in row_indices:
            if row_index < 0 or row_index >= len(self.rows):
                raise IndexError(f"row index {row_index} is out of range for {self.rows_csv}")
            rows.append(dict(self.rows[row_index]))
        if not rows:
            raise ValueError("referenced pair record contains no row_indices")
        return rows

    def get(self, index: int) -> MatchaJointTrainingSet:
        record = self.records[int(index)]
        query_id = str(record.get("query_id", ""))
        reference_id = str(record.get("reference_image_id", ""))
        if not query_id or not reference_id:
            raise ValueError(f"referenced pair record {index} is missing query_id/reference_image_id")
        query_feature = self._feature(
            _path_for_feature(
                str(record.get("query_feature_path", "")),
                image_id=query_id,
                feature_root=self.feature_root,
                feature_path_template=self.feature_path_template,
            )
        )
        reference_feature = self._feature(
            _path_for_feature(
                str(record.get("reference_feature_path", "")),
                image_id=reference_id,
                feature_root=self.feature_root,
                feature_path_template=self.feature_path_template,
            )
        )
        joint, _skip_counts = _build_joint_set_for_real_pair(
            rows=self._record_rows(record),
            query_id=query_id,
            reference_id=reference_id,
            query_feature=query_feature,
            reference_feature=reference_feature,
            query_rgb=self._rgb(query_id),
            reference_rgb=self._rgb(reference_id),
            split_name=str(record.get("split", self.split_name)),
            hard_negatives_per_match=self.hard_negatives_per_match,
            roundtrip_heatmap_threshold_px=self.roundtrip_heatmap_threshold_px,
            positive_reprojection_error_px=self.positive_reprojection_error_px,
            require_same_track=self.require_same_track,
            include_dustbin_rows=self.include_dustbin_rows,
            track_xyz_by_id=self.track_xyz_by_id,
            track_observation_index=self.track_observation_index,
        )
        return joint

    def get_sfm_pair(self, query_id: str, reference_id: str) -> MatchaJointTrainingSet:
        """Materialize an arbitrary real pair directly from SfM common tracks."""

        if self.track_observation_index is None:
            raise ValueError("get_sfm_pair requires a full SfM track observation index")
        query_rgb = self._rgb(str(query_id))
        reference_rgb = self._rgb(str(reference_id))
        retrieval = self.track_observation_index.common_tracks(
            str(query_id),
            str(reference_id),
            query_source_size=(int(query_rgb.shape[1]), int(query_rgb.shape[0])),
            reference_source_size=(int(reference_rgb.shape[1]), int(reference_rgb.shape[0])),
        )
        track_ids = np.asarray(retrieval["track_ids"], dtype=np.int64).reshape(-1)
        if track_ids.size == 0:
            raise ValueError(f"SfM pair {query_id!r}/{reference_id!r} has no common tracks")
        query_xy = np.asarray(retrieval["query_xy"], dtype=np.float64).reshape(-1, 2)
        reference_xy = np.asarray(retrieval["reference_xy"], dtype=np.float64).reshape(-1, 2)
        support_counts = np.asarray(retrieval["support_view_counts"], dtype=np.int64).reshape(-1)
        rows = [
            {
                "query_gt_x": float(query_xy[row, 0]),
                "query_gt_y": float(query_xy[row, 1]),
                "reference_gt_x": float(reference_xy[row, 0]),
                "reference_gt_y": float(reference_xy[row, 1]),
                "track_id": int(track_ids[row]),
                "support_track_id": int(track_ids[row]),
                "track_length": int(support_counts[row]),
                "query_reprojection_error": 0.0,
                "support_reprojection_error": 0.0,
                "target_is_dustbin": False,
            }
            for row in range(int(track_ids.size))
        ]
        query_feature = self._feature(
            _path_for_feature(
                "",
                image_id=str(query_id),
                feature_root=self.feature_root,
                feature_path_template=self.feature_path_template,
            )
        )
        reference_feature = self._feature(
            _path_for_feature(
                "",
                image_id=str(reference_id),
                feature_root=self.feature_root,
                feature_path_template=self.feature_path_template,
            )
        )
        joint, _skip_counts = _build_joint_set_for_real_pair(
            rows=rows,
            query_id=str(query_id),
            reference_id=str(reference_id),
            query_feature=query_feature,
            reference_feature=reference_feature,
            query_rgb=query_rgb,
            reference_rgb=reference_rgb,
            split_name=str(self.split_name),
            hard_negatives_per_match=int(self.hard_negatives_per_match),
            roundtrip_heatmap_threshold_px=float(self.roundtrip_heatmap_threshold_px),
            positive_reprojection_error_px=float(self.positive_reprojection_error_px),
            require_same_track=True,
            include_dustbin_rows=False,
            track_xyz_by_id=self.track_xyz_by_id,
            track_observation_index=self.track_observation_index,
        )
        return joint


def build_real_radio_referenced_joint_manifest(
    *,
    rows_csv: Path,
    image_root: Path,
    feature_root: Path,
    feature_path_template: str,
    feature_key: str,
    output_manifest: Path,
    split_name: str,
    max_pairs: int | None = None,
    hard_negatives_per_match: int = 16,
    roundtrip_heatmap_threshold_px: float = 2.0,
    positive_reprojection_error_px: float = 2.0,
    require_same_track: bool = True,
    include_dustbin_rows: bool = True,
    track_observations: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = _read_csv(Path(rows_csv))
    grouped = _group_row_indices(rows)
    selected_items = list(grouped.items())
    if max_pairs is not None:
        selected_items = selected_items[: int(max_pairs)]
    records: list[dict[str, object]] = []
    skipped = {"empty_candidate_rows": 0}
    for (query_id, reference_id, query_feature_text, reference_feature_text), row_indices in selected_items:
        pair_rows = [rows[int(index)] for index in row_indices]
        if not _has_positive_candidate_rows(pair_rows, require_same_track=bool(require_same_track)):
            skipped["empty_candidate_rows"] += 1
            continue
        records.append(
            {
                "query_id": str(query_id),
                "reference_image_id": str(reference_id),
                "query_feature_path": str(query_feature_text),
                "reference_feature_path": str(reference_feature_text),
                "row_indices": [int(value) for value in row_indices],
                "row_count": int(len(row_indices)),
                "pair_type": "real_real",
                "split": str(split_name),
            }
        )
    if not records:
        raise ValueError(f"no referenced real RADIO joint pairs selected; skipped={skipped}")
    output_manifest_path = Path(output_manifest).resolve()
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": REFERENCED_MANIFEST_FORMAT,
        "rows_csv": str(Path(rows_csv).resolve()),
        "image_root": str(Path(image_root).resolve()),
        "feature_root": str(Path(feature_root).resolve()),
        "feature_path_template": str(feature_path_template),
        "feature_key": str(feature_key),
        "split_name": str(split_name),
        "hard_negatives_per_match": int(hard_negatives_per_match),
        "roundtrip_heatmap_threshold_px": float(roundtrip_heatmap_threshold_px),
        "positive_reprojection_error_px": float(positive_reprojection_error_px),
        "require_same_track": bool(require_same_track),
        "include_dustbin_rows": bool(include_dustbin_rows),
        "input_pair_count": int(len(grouped)),
        "selected_pair_count": int(len(selected_items)),
        "record_count": int(len(records)),
        "sample_count": int(len(records)),
        "cache_format": "referenced",
        "reference_source": "real_image",
        "measurement_supervision": "fine_supervision_from_sfm_tracks",
        "track_observations": "" if track_observations is None else str(Path(track_observations).resolve()),
        "records": records,
    }
    output_manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "stage": "real_radio_joint_cache_builder",
        "elapsed_sec": float(time.perf_counter() - started),
        "rows_csv": str(rows_csv),
        "image_root": str(image_root),
        "feature_root": str(feature_root),
        "feature_path_template": str(feature_path_template),
        "feature_key": str(feature_key),
        "input_pair_count": int(len(grouped)),
        "selected_pair_count": int(len(selected_items)),
        "built_pair_count": int(len(records)),
        "sample_count": int(len(records)),
        "split_name": str(split_name),
        "cache_format": "referenced",
        "reference_source": "real_image",
        "measurement_supervision": "fine_supervision_from_sfm_tracks",
        "track_observations": "" if track_observations is None else str(Path(track_observations).resolve()),
        "skipped": skipped,
        "row_skips": {},
        "outputs": {"joint_cache_manifest": str(output_manifest_path)},
    }


def build_real_radio_joint_cache(
    *,
    rows_csv: Path,
    image_root: Path,
    feature_root: Path,
    feature_path_template: str,
    feature_key: str,
    output_manifest: Path,
    split_name: str,
    max_pairs: int | None = None,
    manifest_mode: str = "sharded_cache",
    hard_negatives_per_match: int = 16,
    roundtrip_heatmap_threshold_px: float = 2.0,
    positive_reprojection_error_px: float = 2.0,
    require_same_track: bool = True,
    include_dustbin_rows: bool = True,
    track_observations: Path | None = None,
) -> dict[str, Any]:
    if str(manifest_mode) == "referenced":
        return build_real_radio_referenced_joint_manifest(
            rows_csv=Path(rows_csv),
            image_root=Path(image_root),
            feature_root=Path(feature_root),
            feature_path_template=str(feature_path_template),
            feature_key=str(feature_key),
            output_manifest=Path(output_manifest),
            split_name=str(split_name),
            max_pairs=max_pairs,
            hard_negatives_per_match=int(hard_negatives_per_match),
            roundtrip_heatmap_threshold_px=float(roundtrip_heatmap_threshold_px),
            positive_reprojection_error_px=float(positive_reprojection_error_px),
            require_same_track=bool(require_same_track),
            include_dustbin_rows=bool(include_dustbin_rows),
            track_observations=track_observations,
        )
    if str(manifest_mode) != "sharded_cache":
        raise ValueError("manifest_mode must be 'sharded_cache' or 'referenced'")
    started = time.perf_counter()
    track_observation_index = (
        None if track_observations is None else load_track_observation_index(Path(track_observations))
    )
    track_xyz_by_id = None if track_observation_index is None else track_observation_index.track_xyz_by_id
    grouped = _group_rows(_read_csv(Path(rows_csv)))
    joint_sets: list[MatchaJointTrainingSet] = []
    skipped: dict[str, int] = {
        "empty_supervision": 0,
        "feature_or_rgb_error": 0,
        "channel_mismatch": 0,
    }
    row_skips: dict[str, int] = {}
    selected_items = list(grouped.items())
    if max_pairs is not None:
        selected_items = selected_items[: int(max_pairs)]
    for (query_id, reference_id, query_feature_text, reference_feature_text), rows in selected_items:
        try:
            query_feature = _load_feature_map(
                _path_for_feature(
                    query_feature_text,
                    image_id=query_id,
                    feature_root=Path(feature_root),
                    feature_path_template=str(feature_path_template),
                ),
                key=str(feature_key),
            )
            reference_feature = _load_feature_map(
                _path_for_feature(
                    reference_feature_text,
                    image_id=reference_id,
                    feature_root=Path(feature_root),
                    feature_path_template=str(feature_path_template),
                ),
                key=str(feature_key),
            )
            query_rgb = _load_rgb_hwc(Path(image_root) / query_id)
            reference_rgb = _load_rgb_hwc(Path(image_root) / reference_id)
        except Exception:
            skipped["feature_or_rgb_error"] += 1
            continue
        try:
            joint, skip_counts = _build_joint_set_for_real_pair(
                rows=rows,
                query_id=query_id,
                reference_id=reference_id,
                query_feature=query_feature,
                reference_feature=reference_feature,
                query_rgb=query_rgb,
                reference_rgb=reference_rgb,
                split_name=str(split_name),
                hard_negatives_per_match=int(hard_negatives_per_match),
                roundtrip_heatmap_threshold_px=float(roundtrip_heatmap_threshold_px),
                positive_reprojection_error_px=float(positive_reprojection_error_px),
                require_same_track=bool(require_same_track),
                include_dustbin_rows=bool(include_dustbin_rows),
                track_xyz_by_id=track_xyz_by_id,
                track_observation_index=track_observation_index,
            )
        except ValueError as exc:
            if "channels must match" in str(exc):
                skipped["channel_mismatch"] += 1
            else:
                skipped["empty_supervision"] += 1
            continue
        for key, value in skip_counts.items():
            row_skips[key] = int(row_skips.get(key, 0) + int(value))
        joint_sets.append(joint)
    if not joint_sets:
        raise ValueError(f"no real RADIO joint samples built; skipped={skipped}, row_skips={row_skips}")
    output_manifest_path = Path(output_manifest).resolve()
    manifest = save_matcha_joint_training_set_manifest(joint_sets, output_manifest_path)
    return {
        "stage": "real_radio_joint_cache_builder",
        "elapsed_sec": float(time.perf_counter() - started),
        "rows_csv": str(rows_csv),
        "image_root": str(image_root),
        "feature_root": str(feature_root),
        "feature_path_template": str(feature_path_template),
        "feature_key": str(feature_key),
        "input_pair_count": int(len(grouped)),
        "selected_pair_count": int(len(selected_items)),
        "built_pair_count": int(len(joint_sets)),
        "sample_count": int(manifest.get("sample_count", 0)),
        "split_name": str(split_name),
        "cache_format": "index_v2",
        "reference_source": "real_image",
        "measurement_supervision": "fine_supervision_from_sfm_tracks",
        "track_observations": "" if track_observations is None else str(Path(track_observations).resolve()),
        "skipped": skipped,
        "row_skips": row_skips,
        "outputs": {"joint_cache_manifest": str(output_manifest_path)},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--feature_path_template", default="{image_stem}.npz")
    parser.add_argument("--feature_key", default="radio_final")
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--split_name", default="train")
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--manifest_mode", choices=("sharded_cache", "referenced"), default="sharded_cache")
    parser.add_argument("--hard_negatives_per_match", type=int, default=16)
    parser.add_argument("--roundtrip_heatmap_threshold_px", type=float, default=2.0)
    parser.add_argument("--positive_reprojection_error_px", type=float, default=2.0)
    parser.add_argument("--allow_track_mismatch", action="store_true")
    parser.add_argument("--exclude_dustbin_rows", action="store_true")
    parser.add_argument("--track_observations", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_real_radio_joint_cache(
        rows_csv=Path(args.rows_csv),
        image_root=Path(args.image_root),
        feature_root=Path(args.feature_root),
        feature_path_template=str(args.feature_path_template),
        feature_key=str(args.feature_key),
        output_manifest=Path(args.output_manifest),
        split_name=str(args.split_name),
        max_pairs=int(args.max_pairs) if int(args.max_pairs) > 0 else None,
        manifest_mode=str(args.manifest_mode),
        hard_negatives_per_match=int(args.hard_negatives_per_match),
        roundtrip_heatmap_threshold_px=float(args.roundtrip_heatmap_threshold_px),
        positive_reprojection_error_px=float(args.positive_reprojection_error_px),
        require_same_track=not bool(args.allow_track_mismatch),
        include_dustbin_rows=not bool(args.exclude_dustbin_rows),
        track_observations=Path(args.track_observations) if str(args.track_observations) else None,
    )
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
