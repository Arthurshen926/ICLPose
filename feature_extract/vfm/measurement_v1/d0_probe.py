from __future__ import annotations

import csv
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import _safe_image_stem
from feature_extract.vfm.measurement_v1.decodability import decodability_metrics
from feature_extract.vfm.measurement_v1.local_likelihood import _bilinear_sample, compute_local_likelihood


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    query_key: str
    render_key: str | None = None

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("feature spec name must be non-empty")
        if not str(self.query_key):
            raise ValueError("feature spec query_key must be non-empty")
        if self.render_key is None:
            object.__setattr__(self, "render_key", str(self.query_key))


def parse_feature_specs(value: str) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for raw_item in str(value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 1:
            specs.append(FeatureSpec(name=parts[0], query_key=parts[0], render_key=parts[0]))
        elif len(parts) == 2:
            specs.append(FeatureSpec(name=parts[0], query_key=parts[1], render_key=parts[1]))
        elif len(parts) == 3:
            specs.append(FeatureSpec(name=parts[0], query_key=parts[1], render_key=parts[2]))
        else:
            raise ValueError(f"invalid feature spec: {item}")
    if not specs:
        raise ValueError("at least one feature spec is required")
    return specs


def _read_csv(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


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


def _load_feature(path: Path, key: str) -> tuple[np.ndarray | None, str | None]:
    if not Path(path).exists():
        return None, "missing_cache_file"
    try:
        with np.load(path) as data:
            if str(key) not in data.files:
                return None, f"missing_key:{key}"
            return _feature_map_chw(data[str(key)]), None
    except (OSError, ValueError, KeyError) as exc:
        return None, f"load_error:{type(exc).__name__}"


def _load_feature_lru(
    cache: "OrderedDict[tuple[str, str], tuple[np.ndarray | None, str | None]]",
    path: Path,
    key: str,
    *,
    capacity: int | None,
) -> tuple[np.ndarray | None, str | None]:
    cache_key = (str(path), str(key))
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]
    value = _load_feature(path, key)
    cache[cache_key] = value
    if capacity is not None and int(capacity) > 0:
        while len(cache) > int(capacity):
            cache.popitem(last=False)
    return value


def _resolve_cache_path(
    row: dict[str, str],
    *,
    side: str,
    feature_name: str,
    cache_dir: Path | None,
    image_width: int,
    image_height: int,
    key: str,
) -> Path | None:
    explicit_keys = [
        f"{side}_{feature_name}_feature_cache_path",
        f"{side}_feature_cache_path",
        f"{side}_cache_path",
    ]
    for field in explicit_keys:
        value = str(row.get(field, "")).strip()
        if value:
            return Path(value)
    if cache_dir is None:
        return None
    query_id = str(row.get("query_id", "")).strip()
    if not query_id:
        return None
    stem = _safe_image_stem(query_id)
    candidates = [
        Path(cache_dir) / f"{stem}_{int(image_width)}x{int(image_height)}_{str(key)}.npz",
        Path(cache_dir) / f"{stem}_{int(image_width)}x{int(image_height)}.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _float_from_row(row: dict[str, str], names: Sequence[str]) -> float:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return float(value)
    raise ValueError(f"missing numeric column; expected one of {list(names)}")


def _sample_descriptor(feature_map: np.ndarray, xy_px: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    sampled, valid = _bilinear_sample(feature_map, np.asarray(xy_px, dtype=np.float64).reshape(1, 2), image_width, image_height)
    if not bool(valid[0]):
        return np.zeros((feature_map.shape[0],), dtype=np.float64)
    desc = np.asarray(sampled[0], dtype=np.float64)
    norm = float(np.linalg.norm(desc))
    return desc / norm if norm > 1e-12 else desc


def _is_xy_only_spec(spec: FeatureSpec) -> bool:
    return str(spec.query_key) == "__xy_only__" and str(spec.render_key) == "__xy_only__"


def _xy_only_feature_map(*, image_width: int, image_height: int) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, int(image_width), dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, int(image_height), dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x, grid_y], axis=0).astype(np.float32)


def residual_bin_for_distance(distance_px: float) -> str:
    value = float(distance_px)
    if not np.isfinite(value):
        return "invalid"
    bins = (
        (0.0, 1.0, "0-1px"),
        (1.0, 2.0, "1-2px"),
        (2.0, 4.0, "2-4px"),
        (4.0, 8.0, "4-8px"),
        (8.0, 16.0, "8-16px"),
        (16.0, 32.0, "16-32px"),
        (32.0, 64.0, "32-64px"),
    )
    for lo, hi, label in bins:
        if value >= lo and value < hi:
            return label
    return "64px+"


def _aggregate(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
    }


def _summarize_metrics(metrics: Sequence[dict[str, float | int | str]]) -> dict[str, Any]:
    epe = [float(item["epe_px"]) for item in metrics]
    nll = [float(item["nll"]) for item in metrics]
    entropy = [float(item["entropy"]) for item in metrics]
    ranks = [float(item["gt_rank"]) for item in metrics]
    return {
        "valid_count": int(len(metrics)),
        "epe_mean_px": _aggregate(epe)["mean"],
        "epe_median_px": _aggregate(epe)["median"],
        "epe_p90_px": _aggregate(epe)["p90"],
        "nll_mean": _aggregate(nll)["mean"],
        "entropy_mean": _aggregate(entropy)["mean"],
        "gt_rank_median": _aggregate(ranks)["median"],
        "recall_0p5px": float(np.mean([float(item["recall_0p5px"]) for item in metrics])) if metrics else None,
        "recall_1px": float(np.mean([float(item["recall_1px"]) for item in metrics])) if metrics else None,
        "recall_2px": float(np.mean([float(item["recall_2px"]) for item in metrics])) if metrics else None,
        "recall_5px": float(np.mean([float(item["recall_5px"]) for item in metrics])) if metrics else None,
        "top5_mode_recall_1px": float(np.mean([float(item["top5_mode_recall_1px"]) for item in metrics])) if metrics else None,
        "top5_mode_recall_5px": float(np.mean([float(item["top5_mode_recall_5px"]) for item in metrics])) if metrics else None,
        "peak_second_margin_mean": (
            float(np.mean([float(item["peak_second_margin"]) for item in metrics])) if metrics else None
        ),
    }


def run_d0_probe(
    *,
    rows_csv: Path,
    output_dir: Path,
    feature_specs: Sequence[FeatureSpec],
    image_width: int,
    image_height: int,
    search_radius_px: float,
    step_px: float,
    temperature: float = 1.0,
    query_feature_cache_dir: Path | None = None,
    render_feature_cache_dir: Path | None = None,
    max_rows: int | None = None,
    feature_cache_capacity: int | None = None,
) -> dict[str, Any]:
    """Run a cache-backed D0 token decodability probe without training.

    The probe is intentionally strict: missing cache files or feature keys are
    counted and reported rather than replaced with fabricated measurements.
    """

    rows = _read_csv(Path(rows_csv), max_rows=max_rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    probe_rows: list[dict[str, Any]] = []
    feature_summaries: dict[str, dict[str, Any]] = {}
    for spec in feature_specs:
        per_feature_metrics: list[dict[str, float | int | str]] = []
        missing_reasons: Counter[str] = Counter()
        loaded_feature_cache: OrderedDict[tuple[str, str], tuple[np.ndarray | None, str | None]] = OrderedDict()
        xy_only_feature = _xy_only_feature_map(image_width=int(image_width), image_height=int(image_height)) if _is_xy_only_spec(spec) else None
        for row_idx, row in enumerate(rows):
            if xy_only_feature is not None:
                query_feature = xy_only_feature
                render_feature = xy_only_feature
            else:
                query_path = _resolve_cache_path(
                    row,
                    side="query",
                    feature_name=spec.name,
                    cache_dir=query_feature_cache_dir,
                    image_width=int(image_width),
                    image_height=int(image_height),
                    key=spec.query_key,
                )
                render_path = _resolve_cache_path(
                    row,
                    side="render",
                    feature_name=spec.name,
                    cache_dir=render_feature_cache_dir,
                    image_width=int(image_width),
                    image_height=int(image_height),
                    key=str(spec.render_key),
                )
                if query_path is None:
                    reason = "missing_query_cache_path"
                    missing_reasons[reason] += 1
                    probe_rows.append({"row_index": row_idx, "query_id": row.get("query_id"), "feature": spec.name, "status": reason})
                    continue
                if render_path is None:
                    reason = "missing_render_cache_path"
                    missing_reasons[reason] += 1
                    probe_rows.append({"row_index": row_idx, "query_id": row.get("query_id"), "feature": spec.name, "status": reason})
                    continue
                query_feature, query_error = _load_feature_lru(
                    loaded_feature_cache,
                    query_path,
                    spec.query_key,
                    capacity=feature_cache_capacity,
                )
                if query_error is not None:
                    reason = f"missing_query_key:{spec.query_key}" if query_error.startswith("missing_key:") else f"query_{query_error}"
                    missing_reasons[reason] += 1
                    probe_rows.append({"row_index": row_idx, "query_id": row.get("query_id"), "feature": spec.name, "status": reason})
                    continue
                render_feature, render_error = _load_feature_lru(
                    loaded_feature_cache,
                    render_path,
                    str(spec.render_key),
                    capacity=feature_cache_capacity,
                )
                if render_error is not None:
                    reason = f"missing_render_key:{spec.render_key}" if render_error.startswith("missing_key:") else f"render_{render_error}"
                    missing_reasons[reason] += 1
                    probe_rows.append({"row_index": row_idx, "query_id": row.get("query_id"), "feature": spec.name, "status": reason})
                    continue
                assert query_feature is not None
                assert render_feature is not None
            if query_feature.shape[0] != render_feature.shape[0]:
                reason = "channel_mismatch"
                missing_reasons[reason] += 1
                probe_rows.append({"row_index": row_idx, "query_id": row.get("query_id"), "feature": spec.name, "status": reason})
                continue
            render_xy = np.asarray(
                [_float_from_row(row, ("render_x", "anchor_render_x")), _float_from_row(row, ("render_y", "anchor_render_y"))],
                dtype=np.float64,
            )
            center_xy = np.asarray(
                [
                    _float_from_row(row, ("center_x", "query_center_x", "render_x")),
                    _float_from_row(row, ("center_y", "query_center_y", "render_y")),
                ],
                dtype=np.float64,
            )
            gt_xy = np.asarray(
                [
                    _float_from_row(row, ("query_gt_x", "gt_query_x", "query_x")),
                    _float_from_row(row, ("query_gt_y", "gt_query_y", "query_y")),
                ],
                dtype=np.float64,
            )
            descriptor = _sample_descriptor(render_feature, render_xy, int(image_width), int(image_height))
            likelihood = compute_local_likelihood(
                anchor_descriptor=descriptor,
                query_feature_map=query_feature,
                center_xy_px=center_xy,
                image_width=int(image_width),
                image_height=int(image_height),
                search_radius_px=float(search_radius_px),
                step_px=float(step_px),
                temperature=float(temperature),
                gt_xy_px=gt_xy,
            )
            metrics = decodability_metrics(likelihood, gt_xy_px=gt_xy)
            residual_px = float(np.linalg.norm(gt_xy - center_xy))
            metrics = {
                **metrics,
                "residual_px": residual_px,
                "residual_bin": residual_bin_for_distance(residual_px),
            }
            per_feature_metrics.append(metrics)
            probe_rows.append(
                {
                    "row_index": row_idx,
                    "query_id": row.get("query_id"),
                    "feature": spec.name,
                    "status": "ok",
                    "gt_rank": metrics["gt_rank"],
                    "epe_px": metrics["epe_px"],
                    "nll": metrics["nll"],
                    "entropy": metrics["entropy"],
                    "recall_0p5px": metrics["recall_0p5px"],
                    "recall_1px": metrics["recall_1px"],
                    "recall_2px": metrics["recall_2px"],
                    "recall_5px": metrics["recall_5px"],
                    "mode_error_px": metrics["mode_error_px"],
                    "top5_mode_recall_1px": metrics["top5_mode_recall_1px"],
                    "top5_mode_recall_5px": metrics["top5_mode_recall_5px"],
                    "peak_second_margin": metrics["peak_second_margin"],
                    "residual_px": metrics["residual_px"],
                    "residual_bin": metrics["residual_bin"],
                    "mode_probability": likelihood.mode_probability,
                    "dustbin_probability": likelihood.dustbin_probability,
                }
            )
        by_bin: dict[str, dict[str, Any]] = {}
        for bin_name in sorted({str(item["residual_bin"]) for item in per_feature_metrics}):
            bin_metrics = [item for item in per_feature_metrics if str(item["residual_bin"]) == bin_name]
            by_bin[bin_name] = _summarize_metrics(bin_metrics)
        feature_summaries[spec.name] = {
            "attempted_count": int(len(rows)),
            "missing_count": int(sum(missing_reasons.values())),
            "missing_reasons": dict(sorted(missing_reasons.items())),
            **_summarize_metrics(per_feature_metrics),
            "by_residual_bin": by_bin,
        }
    fieldnames = [
        "row_index",
        "query_id",
        "feature",
        "status",
        "gt_rank",
        "epe_px",
        "nll",
        "entropy",
        "recall_0p5px",
        "recall_1px",
        "recall_2px",
        "recall_5px",
        "mode_error_px",
        "top5_mode_recall_1px",
        "top5_mode_recall_5px",
        "peak_second_margin",
        "residual_px",
        "residual_bin",
        "mode_probability",
        "dustbin_probability",
    ]
    _write_csv(output / "d0_probe_rows.csv", probe_rows, fieldnames)
    summary = {
        "stage": "measurement_v1_d0_probe",
        "row_count": int(len(rows)),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "search_radius_px": float(search_radius_px),
        "step_px": float(step_px),
        "temperature": float(temperature),
        "feature_cache_capacity": int(feature_cache_capacity) if feature_cache_capacity is not None else None,
        "features": feature_summaries,
        "outputs": {
            "d0_probe_rows": str(output / "d0_probe_rows.csv"),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
