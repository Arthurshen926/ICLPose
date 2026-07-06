from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.measurement_v1.stride4_fine_feature import crop_feature_window


def _read_csv(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _feature_map_chw(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError("feature arrays must have shape (C,H,W), (H,W,C), or (1,C,H,W)")
    first_dim_looks_like_small_channels = arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0] and arr.shape[2] > arr.shape[0]
    last_dim_looks_like_small_channels = arr.shape[-1] <= 16 and arr.shape[0] >= arr.shape[-1] and arr.shape[1] >= arr.shape[-1]
    last_dim_looks_like_large_channels = arr.shape[-1] > arr.shape[0] and arr.shape[-1] > arr.shape[1] and not first_dim_looks_like_small_channels
    if (not first_dim_looks_like_small_channels) and (last_dim_looks_like_small_channels or last_dim_looks_like_large_channels):
        arr = np.moveaxis(arr, -1, 0)
    return np.asarray(arr, dtype=np.float32)


def _load_feature(path: Path, key: str) -> torch.Tensor:
    with np.load(Path(path)) as data:
        if str(key) not in data.files:
            raise KeyError(f"feature key {key!r} missing from {path}")
        return torch.from_numpy(_feature_map_chw(np.asarray(data[str(key)])))


def _load_feature_lru(
    cache: "OrderedDict[tuple[str, str], torch.Tensor]",
    path: Path,
    key: str,
    *,
    capacity: int | None,
    stats: dict[str, int],
) -> torch.Tensor:
    cache_key = (str(path), str(key))
    if cache_key in cache:
        cache.move_to_end(cache_key)
        stats["hit"] = int(stats.get("hit", 0)) + 1
        return cache[cache_key]
    stats["miss"] = int(stats.get("miss", 0)) + 1
    value = _load_feature(path, key)
    cache[cache_key] = value
    if capacity is not None and int(capacity) > 0:
        while len(cache) > int(capacity):
            cache.popitem(last=False)
    return value


def _feature_path(row: Mapping[str, object], *, side: str, feature_name: str) -> Path:
    key = f"{side}_{feature_name}_feature_cache_path"
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing feature cache path column: {key}")
    return Path(value)


def _save_patch(path: Path, key: str, patch: torch.Tensor, *, output_dtype: str) -> None:
    dtype = np.float16 if str(output_dtype).lower() in {"float16", "fp16"} else np.float32
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    np.savez_compressed(tmp, **{str(key): patch.detach().cpu().numpy().astype(dtype, copy=False)})
    generated = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    generated.replace(path)


def _safe_row_id(row: Mapping[str, object], index: int) -> str:
    query = str(row.get("query_id", "")).replace("/", "__").replace("\\", "__").replace(" ", "_")
    anchor = str(row.get("anchor_id", row.get("track_id", ""))).replace("/", "__").replace("\\", "__").replace(" ", "_")
    suffix = f"_{anchor}" if anchor else ""
    return f"{int(index):08d}_{query or 'row'}{suffix}"


def _source_feature_metadata(feature: torch.Tensor, *, image_width: int, image_height: int) -> dict[str, Any]:
    height = int(feature.shape[1])
    width = int(feature.shape[2])
    return {
        "patch_cache_source_feature_height": height,
        "patch_cache_source_feature_width": width,
        "patch_cache_source_feature_pixel_pitch_x": float(image_width - 1) / max(float(width - 1), 1.0),
        "patch_cache_source_feature_pixel_pitch_y": float(image_height - 1) / max(float(height - 1), 1.0),
    }


def _output_dtype_nbytes(output_dtype: str) -> int:
    return 2 if str(output_dtype).lower() in {"float16", "fp16"} else 4


def _patch_side(*, crop_radius_px: float, step_px: float) -> int:
    return int(round((2.0 * float(crop_radius_px)) / float(step_px))) + 1


def _estimated_patch_bytes(feature: torch.Tensor, *, crop_radius_px: float, step_px: float, output_dtype: str) -> int:
    side = _patch_side(crop_radius_px=float(crop_radius_px), step_px=float(step_px))
    return int(feature.shape[0]) * int(side) * int(side) * _output_dtype_nbytes(str(output_dtype))


def materialize_feature_patch_cache_rows(
    *,
    rows_csv: Path,
    output_rows_csv: Path,
    output_patch_dir: Path,
    feature_name: str,
    feature_key: str,
    image_width: int,
    image_height: int,
    crop_radius_px: float,
    step_px: float,
    max_rows: int | None = None,
    output_dtype: str = "float32",
    skip_existing: bool = True,
    source_feature_cache_capacity: int | None = None,
    max_patch_bytes: int | None = 16 * 1024 * 1024,
) -> dict[str, Any]:
    rows = _read_csv(Path(rows_csv), max_rows=max_rows)
    output_rows: list[dict[str, Any]] = []
    patch_count = 0
    reused_count = 0
    source_cache: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()
    source_cache_stats = {"hit": 0, "miss": 0}
    source_feature_shape: list[int] | None = None
    estimated_patch_bytes: int | None = None
    for row_index, row in enumerate(rows):
        item: dict[str, Any] = dict(row)
        row_id = _safe_row_id(row, row_index)
        query_patch_path = Path(output_patch_dir) / f"{row_id}_query_{feature_name}_r{float(crop_radius_px):g}_s{float(step_px):g}.npz"
        render_patch_path = Path(output_patch_dir) / f"{row_id}_render_{feature_name}_r{float(crop_radius_px):g}_s{float(step_px):g}.npz"
        item[f"query_{feature_name}_patch_cache_path"] = str(query_patch_path)
        item[f"render_{feature_name}_patch_cache_path"] = str(render_patch_path)
        item["patch_cache_radius_px"] = float(crop_radius_px)
        item["patch_cache_step_px"] = float(step_px)
        if bool(skip_existing) and query_patch_path.exists() and render_patch_path.exists():
            if source_feature_shape is None:
                query_feature = _load_feature_lru(
                    source_cache,
                    _feature_path(row, side="query", feature_name=str(feature_name)),
                    str(feature_key),
                    capacity=source_feature_cache_capacity,
                    stats=source_cache_stats,
                )
                source_feature_shape = [int(query_feature.shape[1]), int(query_feature.shape[2])]
                item.update(_source_feature_metadata(query_feature, image_width=int(image_width), image_height=int(image_height)))
            reused_count += 2
            output_rows.append(item)
            continue
        query_feature = _load_feature_lru(
            source_cache,
            _feature_path(row, side="query", feature_name=str(feature_name)),
            str(feature_key),
            capacity=source_feature_cache_capacity,
            stats=source_cache_stats,
        ).unsqueeze(0)
        if source_feature_shape is None:
            source_feature_shape = [int(query_feature.shape[2]), int(query_feature.shape[3])]
        item.update(_source_feature_metadata(query_feature[0], image_width=int(image_width), image_height=int(image_height)))
        if estimated_patch_bytes is None:
            estimated_patch_bytes = _estimated_patch_bytes(
                query_feature[0],
                crop_radius_px=float(crop_radius_px),
                step_px=float(step_px),
                output_dtype=str(output_dtype),
            )
        if max_patch_bytes is not None and int(max_patch_bytes) > 0 and int(estimated_patch_bytes) > int(max_patch_bytes):
            raise ValueError(
                "estimated feature patch cache item is too large: "
                f"{int(estimated_patch_bytes)} bytes per patch exceeds max_patch_bytes={int(max_patch_bytes)}. "
                "Increase --max_patch_bytes or reduce crop_radius_px/step resolution/channel count."
            )
        render_feature = _load_feature_lru(
            source_cache,
            _feature_path(row, side="render", feature_name=str(feature_name)),
            str(feature_key),
            capacity=source_feature_cache_capacity,
            stats=source_cache_stats,
        ).unsqueeze(0)
        query_patch, _ = crop_feature_window(
            query_feature,
            torch.tensor([[float(row["center_x"]), float(row["center_y"])]], dtype=torch.float32),
            radius_px=float(crop_radius_px),
            step_px=float(step_px),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        render_patch, _ = crop_feature_window(
            render_feature,
            torch.tensor([[float(row["render_x"]), float(row["render_y"])]], dtype=torch.float32),
            radius_px=float(crop_radius_px),
            step_px=float(step_px),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        _save_patch(query_patch_path, str(feature_key), query_patch[0], output_dtype=str(output_dtype))
        _save_patch(render_patch_path, str(feature_key), render_patch[0], output_dtype=str(output_dtype))
        patch_count += 2
        output_rows.append(item)
    _write_csv(Path(output_rows_csv), output_rows)
    source_height = int(source_feature_shape[0]) if source_feature_shape is not None else 0
    source_width = int(source_feature_shape[1]) if source_feature_shape is not None else 0
    return {
        "stage": "measurement_v1_feature_patch_cache_materializer",
        "input_rows_csv": str(rows_csv),
        "output_rows_csv": str(output_rows_csv),
        "row_count": int(len(rows)),
        "patch_count": int(patch_count),
        "reused_patch_count": int(reused_count),
        "source_feature_cache_hit_count": int(source_cache_stats.get("hit", 0)),
        "source_feature_cache_miss_count": int(source_cache_stats.get("miss", 0)),
        "source_feature_cache_capacity": int(source_feature_cache_capacity) if source_feature_cache_capacity is not None else None,
        "estimated_patch_bytes": int(estimated_patch_bytes) if estimated_patch_bytes is not None else None,
        "max_patch_bytes": int(max_patch_bytes) if max_patch_bytes is not None else None,
        "source_feature_spatial_shape": [source_height, source_width] if source_feature_shape is not None else None,
        "source_feature_pixel_pitch_x": (float(image_width - 1) / max(float(source_width - 1), 1.0)) if source_width else None,
        "source_feature_pixel_pitch_y": (float(image_height - 1) / max(float(source_height - 1), 1.0)) if source_height else None,
        "feature_name": str(feature_name),
        "feature_key": str(feature_key),
        "crop_radius_px": float(crop_radius_px),
        "step_px": float(step_px),
        "output_dtype": str(output_dtype),
    }


def write_feature_patch_cache_summary(path: Path, summary: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n")
