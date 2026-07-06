from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import crop_rgb_window
from feature_extract.vfm.measurement_v1.rgb_patch_training import _load_query_rgb, _load_render_rgb


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


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(base_dir) / value


def _render_cache_by_query(render_cache_manifest_csv: Path | None, *, base_dir: Path) -> dict[str, Path]:
    if render_cache_manifest_csv is None:
        return {}
    out: dict[str, Path] = {}
    for row in _read_csv(Path(render_cache_manifest_csv)):
        query_id = str(row.get("query_id", "")).strip()
        cache_path = str(row.get("rgb_depth_cache_path", "")).strip()
        if query_id and cache_path:
            out[query_id] = _resolve_path(cache_path, base_dir=base_dir)
    return out


def _load_tensor_lru(
    cache: "OrderedDict[str, torch.Tensor]",
    key: str,
    loader: Callable[[], torch.Tensor],
    *,
    capacity: int | None,
    stats: dict[str, int],
) -> torch.Tensor:
    if key in cache:
        cache.move_to_end(key)
        stats["hit"] = int(stats.get("hit", 0)) + 1
        return cache[key]
    stats["miss"] = int(stats.get("miss", 0)) + 1
    value = loader()
    cache[key] = value
    if capacity is not None and int(capacity) > 0:
        while len(cache) > int(capacity):
            cache.popitem(last=False)
    return value


def _save_patch(path: Path, patch: torch.Tensor, *, key: str, output_dtype: str) -> None:
    dtype = np.float16 if str(output_dtype).lower() in {"float16", "fp16"} else np.float32
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    np.savez_compressed(tmp, **{str(key): patch.detach().cpu().numpy().astype(dtype, copy=False)})
    generated = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    generated.replace(path)


def _safe_row_id(row: Mapping[str, object], index: int) -> str:
    query = str(row.get("query_id", "")).replace("/", "__").replace("\\", "__").replace(" ", "_")
    support = str(row.get("support_image_id", "")).replace("/", "__").replace("\\", "__").replace(" ", "_")
    suffix = f"_{support}" if support else ""
    return f"{int(index):08d}_{query or 'row'}{suffix}"


def _source_metadata(*, image_width: int, image_height: int) -> dict[str, Any]:
    return {
        "patch_cache_source_feature_height": int(image_height),
        "patch_cache_source_feature_width": int(image_width),
        "patch_cache_source_feature_pixel_pitch_x": 1.0,
        "patch_cache_source_feature_pixel_pitch_y": 1.0,
    }


def materialize_rgb_native_patch_cache_rows(
    *,
    rows_csv: Path,
    output_rows_csv: Path,
    output_patch_dir: Path,
    image_root: Path,
    image_width: int,
    image_height: int,
    crop_radius_px: float,
    step_px: float,
    query_source: str = "real",
    render_cache_manifest_csv: Path | None = None,
    feature_name: str = "rgb_native",
    feature_key: str = "rgb_native",
    max_rows: int | None = None,
    output_dtype: str = "float32",
    skip_existing: bool = True,
    image_cache_capacity: int | None = 64,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    source = str(query_source)
    if source not in {"real", "real_pair", "render"}:
        raise ValueError("query_source must be 'real', 'real_pair', or 'render'")
    base = Path.cwd() if base_dir is None else Path(base_dir)
    rows = _read_csv(Path(rows_csv), max_rows=max_rows)
    render_by_query = _render_cache_by_query(render_cache_manifest_csv, base_dir=base)
    image_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
    image_cache_stats = {"hit": 0, "miss": 0}
    out_rows: list[dict[str, Any]] = []
    patch_count = 0
    reused_patch_count = 0
    for row_index, row in enumerate(rows):
        item: dict[str, Any] = dict(row)
        row_id = _safe_row_id(row, row_index)
        query_patch_path = Path(output_patch_dir) / f"{row_id}_query_{feature_name}_r{float(crop_radius_px):g}_s{float(step_px):g}.npz"
        render_patch_path = Path(output_patch_dir) / f"{row_id}_render_{feature_name}_r{float(crop_radius_px):g}_s{float(step_px):g}.npz"
        item[f"query_{feature_name}_patch_cache_path"] = str(query_patch_path)
        item[f"render_{feature_name}_patch_cache_path"] = str(render_patch_path)
        item["patch_cache_radius_px"] = float(crop_radius_px)
        item["patch_cache_step_px"] = float(step_px)
        item.update(_source_metadata(image_width=int(image_width), image_height=int(image_height)))
        if bool(skip_existing) and query_patch_path.exists() and render_patch_path.exists():
            reused_patch_count += 2
            out_rows.append(item)
            continue
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError("row missing query_id")
        if source == "real_pair":
            support_id = str(row.get("support_image_id", "")).strip()
            if not support_id:
                raise ValueError("real_pair row missing support_image_id")
            render_path = Path(image_root) / support_id
            query_path = Path(image_root) / query_id
            render_image = _load_tensor_lru(
                image_cache,
                str(render_path),
                lambda p=render_path: _load_query_rgb(p),
                capacity=image_cache_capacity,
                stats=image_cache_stats,
            ).unsqueeze(0)
            query_image = _load_tensor_lru(
                image_cache,
                str(query_path),
                lambda p=query_path: _load_query_rgb(p),
                capacity=image_cache_capacity,
                stats=image_cache_stats,
            ).unsqueeze(0)
            render_xy = [
                float(row.get("support_x", row.get("render_x", ""))),
                float(row.get("support_y", row.get("render_y", ""))),
            ]
        else:
            render_cache = render_by_query.get(query_id)
            if render_cache is None:
                raise ValueError(f"missing render cache for query_id={query_id}")
            render_image = _load_tensor_lru(
                image_cache,
                str(render_cache),
                lambda p=render_cache: _load_render_rgb(p),
                capacity=image_cache_capacity,
                stats=image_cache_stats,
            ).unsqueeze(0)
            query_image = render_image if source == "render" else _load_tensor_lru(
                image_cache,
                str(Path(image_root) / query_id),
                lambda p=Path(image_root) / query_id: _load_query_rgb(p),
                capacity=image_cache_capacity,
                stats=image_cache_stats,
            ).unsqueeze(0)
            render_xy = [float(row["render_x"]), float(row["render_y"])]
        query_patch, _ = crop_rgb_window(
            query_image,
            torch.tensor([[float(row["center_x"]), float(row["center_y"])]], dtype=torch.float32),
            radius_px=float(crop_radius_px),
            step_px=float(step_px),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        render_patch, _ = crop_rgb_window(
            render_image,
            torch.tensor([render_xy], dtype=torch.float32),
            radius_px=float(crop_radius_px),
            step_px=float(step_px),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        _save_patch(query_patch_path, query_patch[0], key=str(feature_key), output_dtype=str(output_dtype))
        _save_patch(render_patch_path, render_patch[0], key=str(feature_key), output_dtype=str(output_dtype))
        patch_count += 2
        out_rows.append(item)
    _write_csv(Path(output_rows_csv), out_rows)
    return {
        "stage": "measurement_v1_rgb_native_patch_cache_materializer",
        "row_count": int(len(rows)),
        "patch_count": int(patch_count),
        "reused_patch_count": int(reused_patch_count),
        "query_source": source,
        "feature_name": str(feature_name),
        "feature_key": str(feature_key),
        "image_cache_hit_count": int(image_cache_stats.get("hit", 0)),
        "image_cache_miss_count": int(image_cache_stats.get("miss", 0)),
        "source_feature_spatial_shape": [int(image_height), int(image_width)],
        "source_feature_pixel_pitch_x": 1.0,
        "source_feature_pixel_pitch_y": 1.0,
        "output_rows_csv": str(output_rows_csv),
        "output_patch_dir": str(output_patch_dir),
    }


def write_rgb_native_patch_cache_summary(path: Path, summary: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n")
