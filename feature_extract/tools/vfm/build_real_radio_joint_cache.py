"""Build real-image RADIO joint localization caches from SfM track rows."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import OrderedDict
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
) -> tuple[MatchaCoarseSupervision | None, dict[str, int]]:
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
        is_dustbin = _parse_bool(row.get("target_is_dustbin", ""))
        same_track = _first_text(row, "track_id") == _first_text(row, "support_track_id", "reference_track_id")
        if is_dustbin:
            skipped["dustbin_rows"] += 1
            if include_dustbin_rows:
                no_match.append(item)
            continue
        if require_same_track and not same_track:
            skipped["track_mismatch"] += 1
            continue
        item["confidence"] = _confidence_from_error(residual, positive_error_px=float(positive_reprojection_error_px))
        positive.append(item)
    positive = _dedupe_matches(positive)
    if not positive:
        return None, skipped
    no_match = _dedupe_matches(no_match)
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
        ),
        skipped,
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
        same_track = _first_text(row, "track_id") == _first_text(row, "support_track_id", "reference_track_id")
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
) -> tuple[MatchaJointTrainingSet, dict[str, int]]:
    if int(query_feature.shape[0]) != int(reference_feature.shape[0]):
        raise ValueError("query/reference feature-map channels must match")
    supervision, skip_counts = _build_supervision_for_pair(
        rows,
        query_image_size=(int(query_rgb.shape[1]), int(query_rgb.shape[0])),
        reference_image_size=(int(reference_rgb.shape[1]), int(reference_rgb.shape[0])),
        query_grid_hw=(int(query_feature.shape[1]), int(query_feature.shape[2])),
        reference_grid_hw=(int(reference_feature.shape[1]), int(reference_feature.shape[2])),
        positive_reprojection_error_px=float(positive_reprojection_error_px),
        require_same_track=bool(require_same_track),
        include_dustbin_rows=bool(include_dustbin_rows),
    )
    if supervision is None or int(supervision.count) == 0:
        raise ValueError("real pair has no valid positive supervision")
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
    return _set_pair_metadata(joint, query_id=query_id, reference_id=reference_id, split_name=split_name), skip_counts


class RealRadioReferencedJointSampleProvider:
    """Load one real-image RADIO pair on demand from a referenced manifest."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        feature_cache_size: int = 4,
        rgb_cache_size: int = 8,
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
        self.records = [dict(item) for item in self.metadata.get("records", [])]
        if not self.records:
            raise ValueError(f"{manifest_path} contains no referenced pair records")
        self.rows = _read_csv(self.rows_csv)
        self.feature_cache_size = max(0, int(feature_cache_size))
        self.rgb_cache_size = max(0, int(rgb_cache_size))
        self._feature_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._rgb_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = RLock()

    def __len__(self) -> int:
        return int(len(self.records))

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
        )
    if str(manifest_mode) != "sharded_cache":
        raise ValueError("manifest_mode must be 'sharded_cache' or 'referenced'")
    started = time.perf_counter()
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
    )
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
