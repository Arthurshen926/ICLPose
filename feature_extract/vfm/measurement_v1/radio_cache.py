from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import _safe_image_stem


class RadioLocalExtractor(Protocol):
    def extract_local(self, rgb: np.ndarray) -> np.ndarray:
        ...


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _render_cache_by_query(render_cache_manifest_csv: Path, *, base_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for row in _read_csv(Path(render_cache_manifest_csv)):
        query_id = str(row.get("query_id", "")).strip()
        cache_path = str(row.get("rgb_depth_cache_path", "")).strip()
        if query_id and cache_path:
            mapping[query_id] = _resolve_path(cache_path, base_dir=base_dir)
    return mapping


def _feature_cache_path(cache_dir: Path, query_id: str, image_width: int, image_height: int, key: str) -> Path:
    return Path(cache_dir) / f"{_safe_image_stem(query_id)}_{int(image_width)}x{int(image_height)}_{str(key)}.npz"


def _output_dtype(dtype: str) -> np.dtype[Any]:
    normalized = str(dtype).strip().lower()
    if normalized in {"float32", "fp32"}:
        return np.dtype(np.float32)
    if normalized in {"float16", "fp16"}:
        return np.dtype(np.float16)
    raise ValueError(f"unsupported RADIO cache output dtype: {dtype}")


def _save_feature_cache(path: Path, key: str, feature: np.ndarray, *, output_dtype: str = "float32") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    np.savez_compressed(tmp, **{str(key): np.asarray(feature, dtype=_output_dtype(output_dtype))})
    generated = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    generated.replace(path)


def _load_render_rgb(path: Path) -> np.ndarray:
    with np.load(Path(path)) as data:
        rgb = np.asarray(data["rgb"], dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"render rgb cache has invalid rgb shape: {rgb.shape}")
    return rgb


def materialize_radio_cache_for_d0_rows(
    *,
    rows_csv: Path,
    output_rows_csv: Path,
    render_cache_manifest_csv: Path,
    render_radio_cache_dir: Path,
    extractor: RadioLocalExtractor,
    image_width: int,
    image_height: int,
    key: str = "radio_dual",
    base_dir: Path | None = None,
    query_radio_cache_dir: Path | None = None,
    skip_existing: bool = True,
    max_rows: int | None = None,
    output_dtype: str = "float32",
) -> dict[str, Any]:
    """Materialize missing render RADIO caches and rewrite D0 rows with paths."""

    base = Path.cwd() if base_dir is None else Path(base_dir)
    dtype_name = _output_dtype(output_dtype).name
    input_rows = _read_csv(Path(rows_csv))
    if max_rows is not None:
        input_rows = input_rows[: int(max_rows)]
    render_cache_by_query = _render_cache_by_query(Path(render_cache_manifest_csv), base_dir=base)
    out_rows: list[dict[str, Any]] = []
    render_written = 0
    render_present = 0
    query_present = 0
    query_missing = 0
    missing_render_rgb = 0
    failed_render_extract = 0
    for row in input_rows:
        query_id = str(row.get("query_id", "")).strip()
        item: dict[str, Any] = dict(row)
        if not query_id:
            out_rows.append(item)
            continue
        query_path_text = str(item.get("query_radio_dual_feature_cache_path", "")).strip()
        if not query_path_text and query_radio_cache_dir is not None:
            item["query_radio_dual_feature_cache_path"] = str(
                _feature_cache_path(Path(query_radio_cache_dir), query_id, int(image_width), int(image_height), str(key))
            )
            query_path_text = str(item["query_radio_dual_feature_cache_path"])
        if query_path_text and _resolve_path(query_path_text, base_dir=base).exists():
            query_present += 1
        else:
            query_missing += 1

        render_path = _feature_cache_path(Path(render_radio_cache_dir), query_id, int(image_width), int(image_height), str(key))
        item["render_radio_dual_feature_cache_path"] = str(render_path)
        if render_path.exists() and bool(skip_existing):
            render_present += 1
            out_rows.append(item)
            continue
        rgb_cache = render_cache_by_query.get(query_id)
        if rgb_cache is None or not rgb_cache.exists():
            missing_render_rgb += 1
            out_rows.append(item)
            continue
        try:
            feature = extractor.extract_local(_load_render_rgb(rgb_cache))
            _save_feature_cache(render_path, str(key), feature, output_dtype=dtype_name)
            render_written += 1
        except Exception:
            failed_render_extract += 1
        out_rows.append(item)
    _write_csv(Path(output_rows_csv), out_rows)
    return {
        "stage": "measurement_v1_radio_cache_materializer",
        "input_row_count": int(len(input_rows)),
        "output_row_count": int(len(out_rows)),
        "query_cache_present_count": int(query_present),
        "query_cache_missing_count": int(query_missing),
        "render_cache_present_count": int(render_present),
        "render_cache_written_count": int(render_written),
        "missing_render_rgb_depth_cache_count": int(missing_render_rgb),
        "failed_render_extract_count": int(failed_render_extract),
        "output_dtype": dtype_name,
        "outputs": {"rows_csv": str(output_rows_csv)},
    }


def write_radio_cache_summary(path: Path, summary: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n")
