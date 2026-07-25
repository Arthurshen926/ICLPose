from __future__ import annotations

import csv
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.io import ImageReadMode, read_image

from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    RGBPatchMeasurementBranch,
    RGBPatchMeasurementPrediction,
    continuous_offset_nll_with_dustbin,
    crop_rgb_window,
    crop_rgb_windows_by_owner,
    crop_rgb_window_with_source_from_output_affine,
    crop_rgb_window_with_source_from_output_homography,
    local_offset_grid,
    residual_delta_gaussian_nll,
)


def _read_csv(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows: list[dict[str, str]] = []
        for row in csv.DictReader(handle):
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= int(max_rows):
                break
        return rows


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(base_dir) / value


def _render_cache_by_query(render_cache_manifest_csv: Path | None, *, base_dir: Path) -> dict[str, Path]:
    if render_cache_manifest_csv is None or not str(render_cache_manifest_csv):
        return {}
    out: dict[str, Path] = {}
    for row in _read_csv(Path(render_cache_manifest_csv)):
        query_id = str(row.get("query_id", "")).strip()
        cache_path = str(row.get("rgb_depth_cache_path", "")).strip()
        if query_id and cache_path:
            out[query_id] = _resolve_path(cache_path, base_dir=base_dir)
    return out


def _float(row: Mapping[str, object], key: str) -> float:
    return float(str(row.get(key, "")).strip())


def _float_any(row: Mapping[str, object], *keys: str) -> float:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return float(value)
    raise ValueError(f"row missing numeric value for any of: {', '.join(keys)}")


def _center_xy(row: Mapping[str, object]) -> tuple[float, float]:
    return (
        _float_any(row, "center_x", "query_center_x"),
        _float_any(row, "center_y", "query_center_y"),
    )


def _optional_bool(row: Mapping[str, object], key: str) -> bool | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"{key} must be a boolean value, got {value!r}")


def _support_from_query_affine(row: Mapping[str, object]) -> torch.Tensor:
    required = ["support_to_query_a00", "support_to_query_a01", "support_to_query_a10", "support_to_query_a11"]
    if any(str(row.get(key, "")).strip() == "" for key in required):
        raise ValueError("local_affine support_patch_warp requires support_to_query_a00/a01/a10/a11")
    support_to_query = torch.tensor(
        [
            [_float(row, "support_to_query_a00"), _float(row, "support_to_query_a01")],
            [_float(row, "support_to_query_a10"), _float(row, "support_to_query_a11")],
        ],
        dtype=torch.float32,
    )
    return torch.linalg.inv(support_to_query)


def _has_local_affine(row: Mapping[str, object]) -> bool:
    required = ["support_to_query_a00", "support_to_query_a01", "support_to_query_a10", "support_to_query_a11"]
    return all(str(row.get(key, "")).strip() != "" for key in required)


def _support_from_query_homography(row: Mapping[str, object]) -> torch.Tensor:
    required = [
        "support_to_query_h00",
        "support_to_query_h01",
        "support_to_query_h02",
        "support_to_query_h10",
        "support_to_query_h11",
        "support_to_query_h12",
        "support_to_query_h20",
        "support_to_query_h21",
        "support_to_query_h22",
    ]
    if any(str(row.get(key, "")).strip() == "" for key in required):
        raise ValueError("local_homography support_patch_warp requires support_to_query_h00...h22")
    support_to_query = torch.tensor(
        [
            [_float(row, "support_to_query_h00"), _float(row, "support_to_query_h01"), _float(row, "support_to_query_h02")],
            [_float(row, "support_to_query_h10"), _float(row, "support_to_query_h11"), _float(row, "support_to_query_h12")],
            [_float(row, "support_to_query_h20"), _float(row, "support_to_query_h21"), _float(row, "support_to_query_h22")],
        ],
        dtype=torch.float32,
    )
    return torch.linalg.inv(support_to_query)


def _has_local_homography(row: Mapping[str, object]) -> bool:
    required = [
        "support_to_query_h00",
        "support_to_query_h01",
        "support_to_query_h02",
        "support_to_query_h10",
        "support_to_query_h11",
        "support_to_query_h12",
        "support_to_query_h20",
        "support_to_query_h21",
        "support_to_query_h22",
    ]
    return all(str(row.get(key, "")).strip() != "" for key in required)


def _load_query_rgb(path: Path) -> torch.Tensor:
    image = read_image(str(Path(path)), mode=ImageReadMode.RGB)
    return image.to(dtype=torch.float32).div_(255.0)


def _load_query_rgb_uint8(path: Path) -> torch.Tensor:
    """Decode RGB without a CPU float conversion for a uint8 GPU image cache."""

    return read_image(str(Path(path)), mode=ImageReadMode.RGB)


def resolve_rgb_image_cache_storage_dtype(value: str) -> torch.dtype:
    """Resolve the explicit full-image cache representation used by RGB crops."""

    normalized = str(value).strip().lower()
    options = {"float16": torch.float16, "uint8": torch.uint8}
    dtype = options.get(normalized)
    if dtype is None:
        raise ValueError("RGB image cache dtype must be one of: float16, uint8")
    return dtype


def _load_render_rgb(path: Path) -> torch.Tensor:
    with np.load(Path(path)) as data:
        arr = np.asarray(data["rgb"], dtype=np.float32)
    if arr.max(initial=0.0) > 2.0:
        arr = arr / 255.0
    return torch.from_numpy(np.moveaxis(arr[..., :3], -1, 0).astype(np.float32, copy=False))


def _split_train_val(
    rows: Sequence[dict[str, str]],
    *,
    val_fraction: float,
    seed: int,
    group_key: str = "",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(rows)
    key = str(group_key).strip()
    if key:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in shuffled:
            value = str(row.get(key, "")).strip()
            if not value:
                raise ValueError(f"group split requested but row missing {key!r}")
            grouped.setdefault(value, []).append(row)
        group_names = sorted(grouped)
        random.Random(int(seed)).shuffle(group_names)
        if len(group_names) == 1:
            return list(shuffled), list(shuffled)
        val_group_count = max(1, int(round(float(val_fraction) * len(group_names)))) if float(val_fraction) > 0.0 else 0
        val_group_count = min(val_group_count, max(len(group_names) - 1, 0))
        train_rows = [row for name in group_names[val_group_count:] for row in grouped[name]]
        val_rows = [row for name in group_names[:val_group_count] for row in grouped[name]]
        if not train_rows and val_rows:
            train_rows = list(val_rows)
        return train_rows, val_rows
    random.Random(int(seed)).shuffle(shuffled)
    if len(shuffled) == 1:
        return list(shuffled), list(shuffled)
    val_count = max(1, int(round(float(val_fraction) * len(shuffled)))) if float(val_fraction) > 0.0 else 0
    val_count = min(val_count, max(len(shuffled) - 1, 0))
    return shuffled[val_count:], shuffled[:val_count]


def _baseline_epe_px(row: Mapping[str, object], *, target_x_key: str, target_y_key: str) -> float:
    center_x, center_y = _center_xy(row)
    return float(np.hypot(_float(row, str(target_x_key)) - center_x, _float(row, str(target_y_key)) - center_y))


def _target_is_dustbin_for_filter(
    row: Mapping[str, object],
    *,
    search_radius_px: float,
    target_x_key: str,
    target_y_key: str,
) -> bool:
    explicit = _optional_bool(row, "target_is_dustbin")
    if explicit is not None:
        return bool(explicit)
    center_x, center_y = _center_xy(row)
    dx = _float(row, str(target_x_key)) - center_x
    dy = _float(row, str(target_y_key)) - center_y
    radius = float(search_radius_px)
    return bool(abs(dx) > radius or abs(dy) > radius)


def _filter_rows_for_training(
    rows: Sequence[dict[str, str]],
    *,
    search_radius_px: float,
    target_x_key: str,
    target_y_key: str,
    target_dustbin_filter: str = "all",
    baseline_epe_min_px: float | None = None,
    baseline_epe_max_px: float | None = None,
    loss_weight_key: str = "",
    min_loss_weight: float | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    dustbin_filter = str(target_dustbin_filter).strip().lower() or "all"
    if dustbin_filter not in {"all", "valid", "dustbin"}:
        raise ValueError("target_dustbin_filter must be one of: all, valid, dustbin")
    min_epe = None if baseline_epe_min_px is None else float(baseline_epe_min_px)
    max_epe = None if baseline_epe_max_px is None else float(baseline_epe_max_px)
    if min_epe is not None and min_epe < 0.0:
        raise ValueError("baseline_epe_min_px must be non-negative")
    if max_epe is not None and max_epe < 0.0:
        raise ValueError("baseline_epe_max_px must be non-negative")
    if min_epe is not None and max_epe is not None and min_epe > max_epe:
        raise ValueError("baseline_epe_min_px must be <= baseline_epe_max_px")
    weight_key = str(loss_weight_key).strip()
    minimum_weight = None if min_loss_weight is None else float(min_loss_weight)
    if minimum_weight is not None and minimum_weight < 0.0:
        raise ValueError("min_loss_weight must be non-negative")
    if minimum_weight is not None and not weight_key:
        raise ValueError("min_loss_weight requires loss_weight_key")
    kept: list[dict[str, str]] = []
    dropped_dustbin = 0
    dropped_residual = 0
    dropped_loss_weight = 0
    for row in rows:
        is_dustbin = _target_is_dustbin_for_filter(
            row,
            search_radius_px=float(search_radius_px),
            target_x_key=str(target_x_key),
            target_y_key=str(target_y_key),
        )
        if dustbin_filter == "valid" and is_dustbin:
            dropped_dustbin += 1
            continue
        if dustbin_filter == "dustbin" and not is_dustbin:
            dropped_dustbin += 1
            continue
        residual = _baseline_epe_px(row, target_x_key=str(target_x_key), target_y_key=str(target_y_key))
        if min_epe is not None and residual < min_epe:
            dropped_residual += 1
            continue
        if max_epe is not None and residual > max_epe:
            dropped_residual += 1
            continue
        if minimum_weight is not None:
            weight_text = str(row.get(weight_key, "")).strip()
            if not weight_text:
                raise ValueError(f"row missing {weight_key!r}")
            if float(weight_text) < minimum_weight:
                dropped_loss_weight += 1
                continue
        kept.append(dict(row))
    return kept, {
        "raw_count": int(len(rows)),
        "kept_count": int(len(kept)),
        "dropped_count": int(len(rows) - len(kept)),
        "dropped_by_dustbin_count": int(dropped_dustbin),
        "dropped_by_residual_count": int(dropped_residual),
        "dropped_by_loss_weight_count": int(dropped_loss_weight),
        "target_dustbin_filter": dustbin_filter,
        "baseline_epe_min_px": min_epe,
        "baseline_epe_max_px": max_epe,
        "loss_weight_key": weight_key,
        "min_loss_weight": minimum_weight,
    }


def _normalised_residual_bin_edges(edges: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in edges)
    if not values:
        return (0.0, 2.0, 5.0, 10.0, 20.0)
    if any(value < 0.0 for value in values):
        raise ValueError("residual sampling bin edges must be non-negative")
    sorted_values = tuple(sorted(set(values)))
    if len(sorted_values) != len(values):
        raise ValueError("residual sampling bin edges must be unique")
    if sorted_values[0] > 0.0:
        sorted_values = (0.0,) + sorted_values
    return sorted_values


def _residual_bin_label(index: int, edges: Sequence[float]) -> str:
    values = tuple(float(value) for value in edges)
    lower = values[int(index)]
    if int(index) + 1 < len(values):
        upper = values[int(index) + 1]
        return f"[{lower:g},{upper:g})"
    return f"[{lower:g},inf)"


def _residual_bin_index(residual_px: float, edges: Sequence[float]) -> int:
    values = tuple(float(value) for value in edges)
    residual = max(0.0, float(residual_px))
    for index in range(len(values) - 1):
        if values[index] <= residual < values[index + 1]:
            return int(index)
    return max(0, len(values) - 1)


@dataclass
class _ResidualBalancedBatchSampler:
    rows: Sequence[dict[str, str]]
    residual_bin_edges_px: Sequence[float]
    target_x_key: str
    target_y_key: str
    search_radius_px: float

    def __post_init__(self) -> None:
        edges = _normalised_residual_bin_edges(self.residual_bin_edges_px)
        object.__setattr__(self, "edges", edges)
        groups: dict[int, list[dict[str, str]]] = {index: [] for index in range(len(edges))}
        dustbin_count = 0
        valid_count = 0
        for row in self.rows:
            residual = _baseline_epe_px(row, target_x_key=str(self.target_x_key), target_y_key=str(self.target_y_key))
            bin_index = _residual_bin_index(residual, edges)
            groups.setdefault(bin_index, []).append(dict(row))
            if _target_is_dustbin_for_filter(
                row,
                search_radius_px=float(self.search_radius_px),
                target_x_key=str(self.target_x_key),
                target_y_key=str(self.target_y_key),
            ):
                dustbin_count += 1
            else:
                valid_count += 1
        non_empty = {index: values for index, values in groups.items() if values}
        object.__setattr__(self, "groups", non_empty)
        object.__setattr__(
            self,
            "summary",
            {
                "enabled": True,
                "row_count": int(len(self.rows)),
                "residual_bin_edges_px": [float(value) for value in edges],
                "bin_count": int(len(edges)),
                "non_empty_bin_count": int(len(non_empty)),
                "valid_count": int(valid_count),
                "dustbin_count": int(dustbin_count),
                "bins": {
                    _residual_bin_label(index, edges): int(len(values))
                    for index, values in sorted(groups.items())
                    if values
                },
            },
        )
        if not non_empty:
            raise ValueError("residual balanced sampler received no non-empty bins")

    def sample_batch(self, *, batch_size: int, rng: random.Random) -> list[dict[str, str]]:
        size = int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        bin_ids = sorted(self.groups)
        out: list[dict[str, str]] = []
        start = int(rng.randrange(len(bin_ids)))
        for index in range(size):
            bin_id = bin_ids[(start + index) % len(bin_ids)]
            bucket = self.groups[bin_id]
            out.append(dict(bucket[rng.randrange(len(bucket))]))
        rng.shuffle(out)
        return out


def _center_baseline_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int | None]:
    if not rows:
        return {"count": 0, "epe_px": None, "epe_median_px": None, "recall_0p5px": None, "recall_1px": None}
    epe = torch.tensor(
        [
            float(np.hypot(_float(row, "query_gt_x") - _center_xy(row)[0], _float(row, "query_gt_y") - _center_xy(row)[1]))
            for row in rows
        ],
        dtype=torch.float32,
    )
    return {
        "count": int(epe.numel()),
        "epe_px": float(torch.mean(epe).item()),
        "epe_median_px": float(torch.median(epe).item()),
        "recall_0p5px": float(torch.mean((epe <= 0.5).float()).item()),
        "recall_1px": float(torch.mean((epe <= 1.0).float()).item()),
        "recall_2px": float(torch.mean((epe <= 2.0).float()).item()),
    }


def _support_patch_source_audit(rows: Sequence[Mapping[str, object]], *, query_source: str) -> dict[str, object]:
    source = str(query_source)
    count = int(len(rows))
    if source == "real_pair":
        missing_support = 0
        same_image = 0
        cross_image = 0
        missing_support_xy = 0
        for row in rows:
            query_id = str(row.get("query_id", "")).strip()
            support_id = str(row.get("support_image_id", "")).strip()
            if not support_id:
                missing_support += 1
                continue
            if support_id == query_id:
                same_image += 1
            else:
                cross_image += 1
            if not str(row.get("support_x", "")).strip() or not str(row.get("support_y", "")).strip():
                missing_support_xy += 1
        return {
            "query_source": source,
            "support_patch_source": "support_image_id",
            "row_count": count,
            "missing_support_image_id_count": int(missing_support),
            "same_image_pair_count": int(same_image),
            "cross_image_pair_count": int(cross_image),
            "missing_support_xy_count": int(missing_support_xy),
        }
    if source in {"render", "render_augmented"}:
        return {
            "query_source": source,
            "support_patch_source": "render_cache_by_query",
            "row_count": count,
            "missing_support_image_id_count": 0,
            "same_image_pair_count": 0,
            "cross_image_pair_count": 0,
            "missing_support_xy_count": 0,
        }
    return {
        "query_source": source,
        "support_patch_source": "render_cache_by_query",
        "row_count": count,
        "missing_support_image_id_count": 0,
        "same_image_pair_count": 0,
        "cross_image_pair_count": 0,
        "missing_support_xy_count": 0,
    }


def _residual_bin_metrics(
    rows: Sequence[Mapping[str, object]],
    epe: torch.Tensor,
    baseline: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float | int | None]:
    if not rows:
        return {}
    raw_values: list[tuple[float, int]] = []
    for idx, row in enumerate(rows):
        text = str(row.get("requested_residual_px", "")).strip()
        if not text:
            continue
        raw_values.append((float(text), int(idx)))
    values_by_bin: dict[float, list[int]] = {}
    unique_values = {value for value, _idx in raw_values}
    grouped_bin_width_px: float | None = None
    if len(unique_values) > 64:
        grouped_bin_width_px = 0.5
        for value, idx in raw_values:
            grouped = round(float(value) / grouped_bin_width_px) * grouped_bin_width_px
            values_by_bin.setdefault(float(grouped), []).append(int(idx))
    else:
        for value, idx in raw_values:
            values_by_bin.setdefault(float(value), []).append(int(idx))
    out: dict[str, float | int | None] = {}
    if grouped_bin_width_px is not None:
        out[f"{prefix}_bin_overflow_grouped_count"] = int(len(unique_values))
        out[f"{prefix}_bin_overflow_grouped_width_px"] = float(grouped_bin_width_px)
    epe_values = epe.detach().cpu().float().reshape(-1)
    baseline_values = baseline.detach().cpu().float().reshape(-1)
    for bin_value in sorted(values_by_bin):
        indices = torch.tensor(values_by_bin[bin_value], dtype=torch.long)
        bin_epe = epe_values[indices]
        bin_baseline = baseline_values[indices]
        key = f"{prefix}_bin_{bin_value:.3f}".replace(".", "p")
        out[f"{key}_count"] = int(bin_epe.numel())
        out[f"{key}_epe_px"] = float(torch.mean(bin_epe).item()) if bin_epe.numel() else None
        out[f"{key}_epe_median_px"] = float(torch.median(bin_epe).item()) if bin_epe.numel() else None
        out[f"{key}_baseline_epe_px"] = float(torch.mean(bin_baseline).item()) if bin_baseline.numel() else None
        out[f"{key}_baseline_epe_median_px"] = float(torch.median(bin_baseline).item()) if bin_baseline.numel() else None
        out[f"{key}_median_improvement_px"] = float(torch.median(bin_baseline - bin_epe).item()) if bin_epe.numel() else None
        out[f"{key}_recall_0p5px"] = float(torch.mean((bin_epe <= 0.5).float()).item()) if bin_epe.numel() else None
        out[f"{key}_recall_1px"] = float(torch.mean((bin_epe <= 1.0).float()).item()) if bin_epe.numel() else None
        out[f"{key}_improve_ratio"] = float(torch.mean((bin_epe < bin_baseline).float()).item()) if bin_epe.numel() else None
        out[f"{key}_worsen_ratio"] = float(torch.mean((bin_epe > bin_baseline).float()).item()) if bin_epe.numel() else None
    return out


def _metric_group_name(value: object) -> str:
    text = str(value).strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return cleaned or "blank"


def _categorical_group_metrics(
    rows: Sequence[Mapping[str, object]],
    epe: torch.Tensor,
    baseline: torch.Tensor,
    *,
    prefix: str,
    group_key: str,
) -> dict[str, float | int | None]:
    if not rows:
        return {}
    key = str(group_key).strip()
    if not key or all(str(row.get(key, "")).strip() == "" for row in rows):
        return {}
    indices_by_group: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        group_name = _metric_group_name(row.get(key, "blank"))
        indices_by_group.setdefault(group_name, []).append(int(idx))
    out: dict[str, float | int | None] = {}
    epe_values = epe.detach().cpu().float().reshape(-1)
    baseline_values = baseline.detach().cpu().float().reshape(-1)
    for group_name in sorted(indices_by_group):
        indices = torch.tensor(indices_by_group[group_name], dtype=torch.long)
        group_epe = epe_values[indices]
        group_baseline = baseline_values[indices]
        metric_prefix = f"{prefix}_{group_name}"
        out[f"{metric_prefix}_count"] = int(group_epe.numel())
        out[f"{metric_prefix}_epe_px"] = float(torch.mean(group_epe).item()) if group_epe.numel() else None
        out[f"{metric_prefix}_epe_median_px"] = float(torch.median(group_epe).item()) if group_epe.numel() else None
        out[f"{metric_prefix}_recall_0p5px"] = float(torch.mean((group_epe <= 0.5).float()).item()) if group_epe.numel() else None
        out[f"{metric_prefix}_recall_1px"] = float(torch.mean((group_epe <= 1.0).float()).item()) if group_epe.numel() else None
        out[f"{metric_prefix}_improve_ratio"] = float(torch.mean((group_epe < group_baseline).float()).item()) if group_epe.numel() else None
    return out


def _prior_scale_batch(rows: Sequence[Mapping[str, object]], *, prior_scale_key: str) -> torch.Tensor | None:
    key = str(prior_scale_key).strip()
    if not key:
        return None
    values = []
    for row in rows:
        text = str(row.get(key, "")).strip()
        if not text:
            raise ValueError(f"prior scale conditioning requested but row missing {key!r}")
        values.append(float(text))
    return torch.tensor(values, dtype=torch.float32)


def _sample_weight_batch(rows: Sequence[Mapping[str, object]], *, loss_weight_key: str) -> torch.Tensor | None:
    key = str(loss_weight_key).strip()
    if not key:
        return None
    values = []
    for row in rows:
        text = str(row.get(key, "")).strip()
        if not text:
            raise ValueError(f"row missing {key!r}")
        values.append(float(text))
    return torch.tensor(values, dtype=torch.float32)


def _configure_head_only_training(
    model: RGBPatchMeasurementBranch,
    *,
    train_dustbin_head_only: bool,
    train_measurement_gate_head_only: bool,
) -> str:
    if bool(train_dustbin_head_only) and bool(train_measurement_gate_head_only):
        raise ValueError("dustbin-head-only and measurement-gate-head-only training are mutually exclusive")
    if not bool(train_dustbin_head_only) and not bool(train_measurement_gate_head_only):
        return "all"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = model.dustbin_head if bool(train_dustbin_head_only) else model.measurement_gate_head
    for parameter in selected.parameters():
        parameter.requires_grad_(True)
    return "dustbin_head" if bool(train_dustbin_head_only) else "measurement_gate_head"


def _passes_measurement_gate(
    metrics: Mapping[str, object],
    *,
    median_epe_key: str,
    improve_ratio_key: str,
    median_epe_threshold_px: float,
    improve_ratio_threshold: float,
) -> dict[str, float | bool | None]:
    median_epe_value = metrics.get(median_epe_key)
    improve_ratio_value = metrics.get(improve_ratio_key)
    median_epe = None if median_epe_value is None else float(median_epe_value)
    improve_ratio = None if improve_ratio_value is None else float(improve_ratio_value)
    epe_passes = median_epe is not None and median_epe <= float(median_epe_threshold_px)
    improve_passes = improve_ratio is not None and improve_ratio >= float(improve_ratio_threshold)
    return {
        "median_epe_px": median_epe,
        "improve_ratio": improve_ratio,
        "median_epe_passes": bool(epe_passes),
        "improve_ratio_passes": bool(improve_passes),
        "passes": bool(epe_passes and improve_passes),
    }


def _residual_bin_gate_summary(
    metrics: Mapping[str, object],
    *,
    prefix: str,
    median_epe_threshold_px: float,
    improve_ratio_threshold: float,
) -> dict[str, dict[str, float | bool | int | None]]:
    out: dict[str, dict[str, float | bool | int | None]] = {}
    marker = f"{prefix}_bin_"
    for key, value in sorted(metrics.items()):
        if not key.startswith(marker) or not key.endswith("_count") or "overflow" in key:
            continue
        base = key[: -len("_count")]
        count = int(value) if value is not None else 0
        median_value = metrics.get(f"{base}_epe_median_px")
        baseline_value = metrics.get(f"{base}_baseline_epe_median_px")
        improve_value = metrics.get(f"{base}_improve_ratio")
        worsen_value = metrics.get(f"{base}_worsen_ratio")
        median_epe = None if median_value is None else float(median_value)
        baseline_epe = None if baseline_value is None else float(baseline_value)
        improve_ratio = None if improve_value is None else float(improve_value)
        worsen_ratio = None if worsen_value is None else float(worsen_value)
        epe_passes = median_epe is not None and median_epe <= float(median_epe_threshold_px)
        improve_passes = improve_ratio is not None and improve_ratio >= float(improve_ratio_threshold)
        bin_name = base[len(marker) :].replace("p", ".")
        out[bin_name] = {
            "count": count,
            "baseline_epe_median_px": baseline_epe,
            "median_epe_px": median_epe,
            "improve_ratio": improve_ratio,
            "worsen_ratio": worsen_ratio,
            "median_epe_passes": bool(epe_passes),
            "improve_ratio_passes": bool(improve_passes),
            "passes": bool(epe_passes and improve_passes),
        }
    return out


def _binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    score_values = scores.detach().cpu().float().reshape(-1)
    label_values = labels.detach().cpu().bool().reshape(-1)
    positives = int(torch.sum(label_values).item())
    negatives = int(label_values.numel() - positives)
    if positives == 0 or negatives == 0:
        return None
    order = torch.argsort(score_values)
    sorted_scores = score_values[order]
    ranks = torch.empty_like(score_values)
    start = 0
    count = int(score_values.numel())
    while start < count:
        end = start + 1
        while end < count and bool(sorted_scores[end] == sorted_scores[start]):
            end += 1
        average_rank = 0.5 * float(start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(torch.sum(ranks[label_values]).item())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)
    return float(auc)


def _binary_ece(probabilities: torch.Tensor, labels: torch.Tensor, *, bin_count: int = 10) -> float | None:
    probs = probabilities.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
    target = labels.detach().cpu().float().reshape(-1)
    if int(probs.numel()) == 0:
        return None
    bins = max(1, int(bin_count))
    ece = 0.0
    for index in range(bins):
        lower = float(index) / float(bins)
        upper = float(index + 1) / float(bins)
        if index == bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(torch.sum(mask).item())
        if count == 0:
            continue
        confidence = float(torch.mean(probs[mask]).item())
        accuracy = float(torch.mean(target[mask]).item())
        ece += float(count) / float(probs.numel()) * abs(confidence - accuracy)
    return float(ece)


class _PatchForwardOnly(nn.Module):
    def __init__(self, model: RGBPatchMeasurementBranch) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        query_patch: torch.Tensor,
        render_patch: torch.Tensor,
        prior_scale_px: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        prior = None if int(prior_scale_px.numel()) == 0 else prior_scale_px
        pred = self.model.forward_from_patches(query_patch, render_patch, prior_scale_px=prior)
        empty_coarse_logits = torch.empty((int(query_patch.shape[0]), 0), device=query_patch.device, dtype=query_patch.dtype)
        empty_coarse_offsets = torch.empty((int(query_patch.shape[0]), 0, 2), device=query_patch.device, dtype=query_patch.dtype)
        offsets = pred.offsets_xy
        if offsets.ndim == 2:
            offsets = offsets.to(device=query_patch.device, dtype=query_patch.dtype).unsqueeze(0).expand(int(query_patch.shape[0]), -1, -1)
        coarse_offsets = pred.coarse_offsets_xy
        if coarse_offsets is not None and coarse_offsets.ndim == 2:
            coarse_offsets = coarse_offsets.to(device=query_patch.device, dtype=query_patch.dtype).unsqueeze(0).expand(
                int(query_patch.shape[0]), -1, -1
            )
        return (
            pred.logits,
            offsets,
            pred.dustbin_logit,
            pred.mean_offset_xy,
            pred.direct_mean_offset_xy,
            pred.direct_log_sigma_xy,
            pred.gated_mean_offset_xy,
            pred.gate_logit,
            empty_coarse_logits if pred.coarse_logits is None else pred.coarse_logits,
            empty_coarse_offsets if coarse_offsets is None else coarse_offsets,
        )


def _forward_patch_prediction(
    *,
    model: RGBPatchMeasurementBranch,
    parallel_forward: nn.DataParallel | None,
    query_patch: torch.Tensor,
    render_patch: torch.Tensor,
    prior_scale: torch.Tensor | None,
    device: torch.device,
) -> Any:
    if parallel_forward is None:
        return model.forward_from_patches(
            query_patch.to(device),
            render_patch.to(device),
            prior_scale_px=None if prior_scale is None else prior_scale.to(device),
        )
    empty_prior = torch.empty((0,), device=device, dtype=query_patch.dtype)
    (
        logits,
        offsets_xy,
        dustbin_logit,
        likelihood_mean,
        direct_mean,
        direct_log_sigma,
        gated_mean,
        gate_logit,
        coarse_logits,
        coarse_offsets,
    ) = parallel_forward(
        query_patch.to(device),
        render_patch.to(device),
        empty_prior if prior_scale is None else prior_scale.to(device),
    )
    coarse_logits_value = None if int(coarse_logits.shape[1]) == 0 else coarse_logits
    coarse_offsets_value = None if int(coarse_offsets.numel()) == 0 else coarse_offsets
    return RGBPatchMeasurementPrediction(
        logits=logits,
        offsets_xy=offsets_xy,
        dustbin_logit=dustbin_logit,
        coarse_logits=coarse_logits_value,
        coarse_offsets_xy=coarse_offsets_value,
        mean_offset_xy=likelihood_mean,
        direct_mean_offset_xy=direct_mean,
        direct_log_sigma_xy=direct_log_sigma,
        gated_mean_offset_xy=gated_mean,
        gate_logit=gate_logit,
        gate_probability=torch.sigmoid(gate_logit),
    )


def _coarse_stage_likelihood_loss(
    pred: RGBPatchMeasurementPrediction,
    target: torch.Tensor,
    *,
    search_radius_px: float,
    target_is_dustbin: torch.Tensor | None,
    sample_weight: torch.Tensor | None,
    dustbin_positive_weight: float,
    target_heatmap_sigma_px: float,
) -> tuple[torch.Tensor | None, RGBPatchMeasurementPrediction | None]:
    if pred.coarse_logits is None or pred.coarse_offsets_xy is None:
        return None, None
    return continuous_offset_nll_with_dustbin(
        pred.coarse_logits,
        pred.coarse_offsets_xy,
        target,
        dustbin_logit=pred.dustbin_logit,
        search_radius_px=float(search_radius_px),
        epe_weight=0.0,
        dustbin_bce_weight=0.0,
        dustbin_positive_weight=float(dustbin_positive_weight),
        target_is_dustbin=target_is_dustbin,
        target_heatmap_sigma_px=float(target_heatmap_sigma_px),
        sample_weight=sample_weight,
    )


def _gate_supervision_loss(
    pred: RGBPatchMeasurementPrediction,
    target: torch.Tensor,
    *,
    target_is_dustbin: torch.Tensor | None,
    sample_weight: torch.Tensor | None,
    center_radius_px: float,
    full_radius_px: float,
    target_mode: str = "residual",
    utility_temperature_px: float = 0.25,
    minimum_update_gain_px: float = 0.1,
    positive_weight: float = 1.0,
    low_residual_threshold_px: float = 1.0,
    low_residual_negative_weight: float = 1.0,
) -> torch.Tensor | None:
    if pred.gate_logit is None:
        return None
    logits = pred.gate_logit.reshape(int(target.shape[0]))
    target_xy = target.to(device=logits.device, dtype=logits.dtype).reshape(int(logits.shape[0]), 2)
    mode = str(target_mode).strip().lower()
    if mode == "residual":
        residual = torch.linalg.norm(target_xy, dim=1)
        center = float(center_radius_px)
        full = float(full_radius_px)
        if full <= center:
            raise ValueError("gate_full_radius_px must be greater than gate_center_radius_px")
        target_gate = ((residual - center) / (full - center)).clamp(0.0, 1.0)
    elif mode in {"utility", "binary_utility"}:
        if pred.mean_offset_xy is None:
            raise ValueError("utility gate supervision requires pred.mean_offset_xy")
        temperature = float(utility_temperature_px)
        if temperature <= 0.0:
            raise ValueError("gate_utility_temperature_px must be positive")
        center_epe = torch.linalg.norm(target_xy, dim=1)
        likelihood_mean = pred.mean_offset_xy.to(device=logits.device, dtype=logits.dtype).reshape(int(logits.shape[0]), 2)
        likelihood_epe = torch.linalg.norm(likelihood_mean.detach() - target_xy, dim=1)
        if mode == "utility":
            target_gate = torch.sigmoid((center_epe - likelihood_epe) / temperature)
        else:
            target_gate = (likelihood_epe + float(minimum_update_gain_px) <= center_epe).to(dtype=logits.dtype)
    else:
        raise ValueError("gate_target_mode must be residual, utility, or binary_utility")
    if target_is_dustbin is None:
        valid_mask = torch.ones_like(target_gate, dtype=torch.bool)
    else:
        valid_mask = ~target_is_dustbin.to(device=logits.device).reshape(int(logits.shape[0])).bool()
    if not torch.any(valid_mask):
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    loss_rows = F.binary_cross_entropy_with_logits(logits, target_gate, reduction="none")
    loss_rows = loss_rows * torch.where(
        target_gate > 0.5,
        torch.full_like(loss_rows, float(positive_weight)),
        torch.ones_like(loss_rows),
    )
    if float(low_residual_negative_weight) != 1.0:
        center_epe = torch.linalg.norm(target_xy, dim=1)
        low_negative = (center_epe <= float(low_residual_threshold_px)) & (target_gate <= 0.5)
        loss_rows = loss_rows * torch.where(
            low_negative,
            torch.full_like(loss_rows, float(low_residual_negative_weight)),
            torch.ones_like(loss_rows),
        )
    weights = None
    if sample_weight is not None:
        weights = sample_weight.to(device=logits.device, dtype=logits.dtype).reshape(int(logits.shape[0])).clamp_min(0.0)
    selected_loss = loss_rows[valid_mask]
    if weights is None:
        return torch.mean(selected_loss)
    selected_weight = weights[valid_mask]
    denom = torch.sum(selected_weight)
    if float(denom.detach().cpu().item()) <= 0.0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return torch.sum(selected_loss * selected_weight) / denom.clamp_min(1e-12)


class TensorImageLRUCache(MutableMapping[str, torch.Tensor]):
    """Byte-bounded tensor cache shared by query and support image owners."""

    def __init__(
        self,
        *,
        max_bytes: int | None = None,
        storage_dtype: torch.dtype | None = None,
    ) -> None:
        if max_bytes is not None and int(max_bytes) <= 0:
            raise ValueError("max_bytes must be positive when provided")
        if storage_dtype not in {
            None,
            torch.float16,
            torch.float32,
            torch.bfloat16,
            torch.uint8,
        }:
            raise ValueError("storage_dtype must be a supported image tensor dtype")
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.storage_dtype = storage_dtype
        self._values: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.current_bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def _tensor_bytes(value: torch.Tensor) -> int:
        return int(value.numel() * value.element_size())

    def __getitem__(self, key: str) -> torch.Tensor:
        value = self._values.pop(key)
        self._values[key] = value
        self.hits += 1
        return value

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        stored = value if self.storage_dtype is None else value.to(dtype=self.storage_dtype)
        if key in self._values:
            previous = self._values.pop(key)
            self.current_bytes -= self._tensor_bytes(previous)
        self._values[key] = stored
        self.current_bytes += self._tensor_bytes(stored)
        while (
            self.max_bytes is not None
            and self.current_bytes > self.max_bytes
            and len(self._values) > 1
        ):
            _old_key, old_value = self._values.popitem(last=False)
            self.current_bytes -= self._tensor_bytes(old_value)
            self.evictions += 1
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    def __delitem__(self, key: str) -> None:
        value = self._values.pop(key)
        self.current_bytes -= self._tensor_bytes(value)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def record_miss(self) -> None:
        self.misses += 1

    def summary(self) -> dict[str, int | str | None]:
        summary: dict[str, int | str | None] = {
            "entry_count": int(len(self)),
            "max_bytes": self.max_bytes,
            "current_bytes": int(self.current_bytes),
            "peak_bytes": int(self.peak_bytes),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "evictions": int(self.evictions),
        }
        if self.storage_dtype is not None:
            summary["storage_dtype"] = str(self.storage_dtype)
        return summary


def _load_tensor_cached(
    cache: MutableMapping[str, torch.Tensor],
    key: str,
    loader,
    *,
    cache_device: torch.device | None = None,
) -> torch.Tensor:
    if key not in cache:
        if isinstance(cache, TensorImageLRUCache):
            cache.record_miss()
        value = loader()
        if cache_device is not None:
            value = value.to(cache_device)
        cache[key] = value
    return cache[key]


def _augment_render_query_image(image: torch.Tensor) -> torch.Tensor:
    values = image.float()
    # Deterministic photometric shift for protocol reproducibility.
    gray = torch.mean(values, dim=1, keepdim=True)
    shifted = 0.75 * values + 0.25 * gray
    shifted = (shifted - 0.5) * 1.25 + 0.5
    h, w = int(shifted.shape[2]), int(shifted.shape[3])
    yy = torch.linspace(-1.0, 1.0, h, device=shifted.device, dtype=shifted.dtype).reshape(1, 1, h, 1)
    xx = torch.linspace(-1.0, 1.0, w, device=shifted.device, dtype=shifted.dtype).reshape(1, 1, 1, w)
    vignette = (1.0 - 0.12 * (xx * xx + yy * yy)).clamp(0.7, 1.0)
    return torch.clamp(shifted * vignette + 0.03, 0.0, 1.0)


def augment_render_template_patch(patch: torch.Tensor, *, mode: str = "none") -> torch.Tensor:
    """Apply training-only render template degradation for real-render alignment."""

    selected = str(mode)
    if selected == "none" or selected == "":
        return patch
    if selected != "realistic":
        raise ValueError("render patch augmentation mode must be 'none' or 'realistic'")
    values = patch.float()
    batch = int(values.shape[0])
    device = values.device
    dtype = values.dtype
    gray = torch.mean(values, dim=1, keepdim=True)
    gray_mix = 0.10 + 0.35 * torch.rand((batch, 1, 1, 1), device=device, dtype=dtype)
    out = values * (1.0 - gray_mix) + gray * gray_mix
    contrast = 0.75 + 0.60 * torch.rand((batch, 1, 1, 1), device=device, dtype=dtype)
    brightness = -0.08 + 0.16 * torch.rand((batch, 1, 1, 1), device=device, dtype=dtype)
    out = (out - 0.5) * contrast + 0.5 + brightness
    gamma = 0.80 + 0.40 * torch.rand((batch, 1, 1, 1), device=device, dtype=dtype)
    out = torch.clamp(out, 0.0, 1.0).pow(gamma)
    blur = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
    blur_mix = 0.05 + 0.35 * torch.rand((batch, 1, 1, 1), device=device, dtype=dtype)
    out = out * (1.0 - blur_mix) + blur * blur_mix
    noise = torch.randn_like(out) * 0.015
    return torch.clamp(out + noise, 0.0, 1.0)


def _crop_cached_rgb_windows_grouped(
    specs: Sequence[tuple[str, Any]],
    centers_xy: Sequence[Sequence[float]],
    *,
    cache: MutableMapping[str, torch.Tensor],
    cache_device: torch.device | None,
    crop_device: torch.device | None = None,
    radius_px: float,
    step_px: float,
    image_width: int,
    image_height: int,
    augment_query_image: bool = False,
) -> torch.Tensor:
    """Crop packed grids from small batches of unique owner images."""

    if len(specs) != len(centers_xy):
        raise ValueError("crop specs and centers must contain the same number of rows")
    if not specs:
        raise ValueError("grouped RGB crop requires at least one row")
    rows_by_key: OrderedDict[str, list[int]] = OrderedDict()
    for row, (key, _loader) in enumerate(specs):
        rows_by_key.setdefault(str(key), []).append(int(row))

    grouped_patches: list[torch.Tensor] = []
    grouped_rows: list[torch.Tensor] = []
    owner_items = list(rows_by_key.items())
    owner_batch_size = 16
    for start in range(0, len(owner_items), owner_batch_size):
        chunk = owner_items[start : start + owner_batch_size]
        images: list[torch.Tensor] = []
        chunk_centers: list[list[float]] = []
        chunk_owners: list[int] = []
        chunk_rows: list[int] = []
        for owner, (key, rows) in enumerate(chunk):
            loader = specs[rows[0]][1]
            images.append(
                _load_tensor_cached(
                    cache,
                    str(key),
                    loader,
                    cache_device=cache_device,
                )
            )
            chunk_rows.extend(rows)
            chunk_owners.extend([int(owner)] * len(rows))
            chunk_centers.extend([centers_xy[row] for row in rows])
        image = torch.stack(images, dim=0)
        if crop_device is not None and image.device != crop_device:
            image = image.to(crop_device)
        if image.dtype == torch.uint8:
            # Keep all cached images compact and decode/upload them once.  The
            # conversion and normalization then happen batched on the crop GPU.
            image = image.to(dtype=torch.float32).div_(255.0)
        if bool(augment_query_image):
            image = _augment_render_query_image(image)
        centers = torch.tensor(
            chunk_centers,
            dtype=torch.float32,
            device=image.device,
        )
        patches, _offsets = crop_rgb_windows_by_owner(
            image,
            torch.tensor(chunk_owners, dtype=torch.long, device=image.device),
            centers,
            radius_px=float(radius_px),
            step_px=float(step_px),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        grouped_patches.append(patches)
        grouped_rows.append(
            torch.tensor(chunk_rows, dtype=torch.long, device=patches.device)
        )
    row_order = torch.cat(grouped_rows, dim=0)
    patches_by_group = torch.cat(grouped_patches, dim=0)
    return patches_by_group[torch.argsort(row_order)].contiguous()


def _stack_unwarped_patch_batch(
    rows: Sequence[Mapping[str, object]],
    *,
    image_root: Path,
    render_cache_by_query: Mapping[str, Path],
    query_crop_width: int,
    query_crop_height: int,
    render_width: int,
    render_height: int,
    crop_radius_px: float,
    step_px: float,
    query_cache: MutableMapping[str, torch.Tensor],
    render_cache: MutableMapping[str, torch.Tensor],
    query_source: str,
    render_patch_augmentation: str,
    target_x_key: str,
    target_y_key: str,
    image_cache_device: torch.device | None,
    crop_device: torch.device | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    source = str(query_source)
    query_specs: list[tuple[str, Any]] = []
    render_specs: list[tuple[str, Any]] = []
    centers: list[list[float]] = []
    anchors: list[list[float]] = []
    targets: list[list[float]] = []
    explicit_dustbin_targets: list[bool | None] = []
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError("row missing query_id")
        query_path = Path(image_root) / query_id
        if source == "real_pair":
            support_id = str(row.get("support_image_id", "")).strip()
            if not support_id:
                raise ValueError("real_pair row missing support_image_id")
            support_path = Path(image_root) / support_id
            query_specs.append(
                (str(query_path), lambda path=query_path: _load_query_rgb(path))
            )
            render_specs.append(
                (str(support_path), lambda path=support_path: _load_query_rgb(path))
            )
            anchor = [
                _float(row, "support_x")
                if str(row.get("support_x", "")).strip()
                else _float(row, "render_x"),
                _float(row, "support_y")
                if str(row.get("support_y", "")).strip()
                else _float(row, "render_y"),
            ]
        else:
            render_path = render_cache_by_query.get(query_id)
            if render_path is None:
                raise ValueError(f"missing render cache for query_id={query_id}")
            render_specs.append(
                (str(render_path), lambda path=render_path: _load_render_rgb(path))
            )
            if source in {"render", "render_augmented"}:
                query_specs.append(
                    (str(render_path), lambda path=render_path: _load_render_rgb(path))
                )
            else:
                query_specs.append(
                    (str(query_path), lambda path=query_path: _load_query_rgb(path))
                )
            anchor = [_float(row, "render_x"), _float(row, "render_y")]
        center_x, center_y = _center_xy(row)
        target_x = _float(row, str(target_x_key))
        target_y = _float(row, str(target_y_key))
        centers.append([center_x, center_y])
        anchors.append(anchor)
        targets.append([target_x - center_x, target_y - center_y])
        explicit_dustbin_targets.append(_optional_bool(row, "target_is_dustbin"))

    query_patch = _crop_cached_rgb_windows_grouped(
        query_specs,
        centers,
        cache=(render_cache if source in {"render", "render_augmented"} else query_cache),
        cache_device=image_cache_device,
        crop_device=crop_device,
        radius_px=float(crop_radius_px),
        step_px=float(step_px),
        image_width=int(query_crop_width),
        image_height=int(query_crop_height),
        augment_query_image=source == "render_augmented",
    )
    render_patch = _crop_cached_rgb_windows_grouped(
        render_specs,
        anchors,
        cache=render_cache,
        cache_device=image_cache_device,
        crop_device=crop_device,
        radius_px=float(crop_radius_px),
        step_px=float(step_px),
        image_width=int(render_width),
        image_height=int(render_height),
    )
    render_patch = augment_render_template_patch(
        render_patch, mode=str(render_patch_augmentation)
    )
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    target_is_dustbin = None
    if any(value is not None for value in explicit_dustbin_targets):
        target_is_dustbin = torch.tensor(
            [bool(value) for value in explicit_dustbin_targets], dtype=torch.bool
        )
    return (
        query_patch,
        render_patch,
        target_tensor,
        torch.linalg.norm(target_tensor, dim=1),
        target_is_dustbin,
    )


def _stack_patch_batch(
    rows: Sequence[Mapping[str, object]],
    *,
    image_root: Path,
    render_cache_by_query: Mapping[str, Path],
    image_width: int,
    image_height: int,
    query_image_width: int | None = None,
    query_image_height: int | None = None,
    render_image_width: int | None = None,
    render_image_height: int | None = None,
    crop_radius_px: float,
    step_px: float,
    query_cache: MutableMapping[str, torch.Tensor],
    render_cache: MutableMapping[str, torch.Tensor],
    query_source: str,
    render_patch_augmentation: str = "none",
    support_patch_warp: str = "none",
    target_x_key: str = "query_gt_x",
    target_y_key: str = "query_gt_y",
    image_cache_device: torch.device | None = None,
    crop_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    query_patches = []
    render_patches = []
    targets = []
    explicit_dustbin_targets: list[bool | None] = []
    source = str(query_source)
    warp_mode = str(support_patch_warp)
    if source not in {"real", "render", "render_augmented", "real_pair"}:
        raise ValueError("query_source must be 'real', 'render', 'render_augmented', or 'real_pair'")
    if warp_mode not in {"none", "local_affine", "local_homography"}:
        raise ValueError("support_patch_warp must be 'none', 'local_affine', or 'local_homography'")
    q_width = int(query_image_width if query_image_width is not None else image_width)
    q_height = int(query_image_height if query_image_height is not None else image_height)
    r_width = int(render_image_width if render_image_width is not None else image_width)
    r_height = int(render_image_height if render_image_height is not None else image_height)
    query_crop_width = r_width if source in {"render", "render_augmented"} else q_width
    query_crop_height = r_height if source in {"render", "render_augmented"} else q_height
    if warp_mode == "none":
        return _stack_unwarped_patch_batch(
            rows,
            image_root=Path(image_root),
            render_cache_by_query=render_cache_by_query,
            query_crop_width=int(query_crop_width),
            query_crop_height=int(query_crop_height),
            render_width=int(r_width),
            render_height=int(r_height),
            crop_radius_px=float(crop_radius_px),
            step_px=float(step_px),
            query_cache=query_cache,
            render_cache=render_cache,
            query_source=source,
            render_patch_augmentation=str(render_patch_augmentation),
            target_x_key=str(target_x_key),
            target_y_key=str(target_y_key),
            image_cache_device=image_cache_device,
            crop_device=crop_device,
        )
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError("row missing query_id")
        query_path = Path(image_root) / query_id
        if source == "real_pair":
            support_id = str(row.get("support_image_id", "")).strip()
            if not support_id:
                raise ValueError("real_pair row missing support_image_id")
            support_path = Path(image_root) / support_id
            render_image = _load_tensor_cached(
                render_cache,
                str(support_path),
                lambda p=support_path: _load_query_rgb(p),
                cache_device=image_cache_device,
            ).unsqueeze(0)
            query_image = _load_tensor_cached(
                query_cache,
                str(query_path),
                lambda p=query_path: _load_query_rgb(p),
                cache_device=image_cache_device,
            ).unsqueeze(0)
        else:
            render_path = render_cache_by_query.get(query_id)
            if render_path is None:
                raise ValueError(f"missing render cache for query_id={query_id}")
            render_image = _load_tensor_cached(
                render_cache,
                str(render_path),
                lambda p=render_path: _load_render_rgb(p),
                cache_device=image_cache_device,
            ).unsqueeze(0)
        if source == "render":
            query_image = render_image
        elif source == "render_augmented":
            query_image = _augment_render_query_image(render_image)
        elif source == "real":
            query_image = _load_tensor_cached(
                query_cache,
                str(query_path),
                lambda p=query_path: _load_query_rgb(p),
                cache_device=image_cache_device,
            ).unsqueeze(0)
        if crop_device is not None:
            if query_image.device != crop_device:
                query_image = query_image.to(crop_device)
            if render_image.device != crop_device:
                render_image = render_image.to(crop_device)
        center = list(_center_xy(row))
        if source == "real_pair":
            anchor = [
                _float(row, "support_x") if str(row.get("support_x", "")).strip() else _float(row, "render_x"),
                _float(row, "support_y") if str(row.get("support_y", "")).strip() else _float(row, "render_y"),
            ]
        else:
            anchor = [_float(row, "render_x"), _float(row, "render_y")]
        target_x = _float(row, str(target_x_key))
        target_y = _float(row, str(target_y_key))
        row_target_is_dustbin = _optional_bool(row, "target_is_dustbin")
        query_patch, _ = crop_rgb_window(
            query_image,
            torch.tensor([center], dtype=torch.float32),
            radius_px=float(crop_radius_px),
            step_px=float(step_px),
            image_width=int(query_crop_width),
            image_height=int(query_crop_height),
        )
        if warp_mode == "local_affine":
            if source != "real_pair":
                raise ValueError("local_affine support_patch_warp is only supported for query_source='real_pair'")
            if bool(row_target_is_dustbin) and not _has_local_affine(row):
                render_patch, _ = crop_rgb_window(
                    render_image,
                    torch.tensor([anchor], dtype=torch.float32),
                    radius_px=float(crop_radius_px),
                    step_px=float(step_px),
                    image_width=int(r_width),
                    image_height=int(r_height),
                )
            else:
                render_patch, _ = crop_rgb_window_with_source_from_output_affine(
                    render_image,
                    torch.tensor([anchor], dtype=torch.float32),
                    _support_from_query_affine(row).reshape(1, 2, 2),
                    radius_px=float(crop_radius_px),
                    step_px=float(step_px),
                    image_width=int(r_width),
                    image_height=int(r_height),
                )
        elif warp_mode == "local_homography":
            if source != "real_pair":
                raise ValueError("local_homography support_patch_warp is only supported for query_source='real_pair'")
            if bool(row_target_is_dustbin) and not _has_local_homography(row):
                render_patch, _ = crop_rgb_window(
                    render_image,
                    torch.tensor([anchor], dtype=torch.float32),
                    radius_px=float(crop_radius_px),
                    step_px=float(step_px),
                    image_width=int(r_width),
                    image_height=int(r_height),
                )
            else:
                render_patch, _ = crop_rgb_window_with_source_from_output_homography(
                    render_image,
                    torch.tensor([[target_x, target_y]], dtype=torch.float32),
                    _support_from_query_homography(row).reshape(1, 3, 3),
                    radius_px=float(crop_radius_px),
                    step_px=float(step_px),
                    image_width=int(r_width),
                    image_height=int(r_height),
                )
        else:
            render_patch, _ = crop_rgb_window(
                render_image,
                torch.tensor([anchor], dtype=torch.float32),
                radius_px=float(crop_radius_px),
                step_px=float(step_px),
                image_width=int(r_width),
                image_height=int(r_height),
            )
        render_patch = augment_render_template_patch(render_patch, mode=str(render_patch_augmentation))
        query_patches.append(query_patch[0])
        render_patches.append(render_patch[0])
        center_x, center_y = _center_xy(row)
        targets.append([target_x - center_x, target_y - center_y])
        explicit_dustbin_targets.append(row_target_is_dustbin)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    target_is_dustbin = None
    if any(value is not None for value in explicit_dustbin_targets):
        target_is_dustbin = torch.tensor([bool(value) for value in explicit_dustbin_targets], dtype=torch.bool)
    return torch.stack(query_patches, dim=0), torch.stack(render_patches, dim=0), target_tensor, torch.linalg.norm(target_tensor, dim=1), target_is_dustbin


def _append_roll_hard_negatives(
    query_patch: torch.Tensor,
    render_patch: torch.Tensor,
    target: torch.Tensor,
    target_is_dustbin: torch.Tensor | None,
    *,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    frac = float(fraction)
    if frac <= 0.0:
        return query_patch, render_patch, target, target_is_dustbin
    batch = int(query_patch.shape[0])
    if batch < 2:
        return query_patch, render_patch, target, target_is_dustbin
    negative_count = max(1, min(batch, int(round(batch * frac))))
    base_dustbin = (
        torch.zeros((batch,), dtype=torch.bool, device=query_patch.device)
        if target_is_dustbin is None
        else target_is_dustbin.to(device=query_patch.device).reshape(batch).bool()
    )
    selected = torch.arange(negative_count, device=query_patch.device)
    rolled_render = torch.roll(render_patch, shifts=1, dims=0)[selected]
    query_negative = query_patch[selected]
    target_negative = torch.zeros((negative_count, 2), dtype=target.dtype, device=target.device)
    negative_dustbin = torch.ones((negative_count,), dtype=torch.bool, device=query_patch.device)
    return (
        torch.cat([query_patch, query_negative], dim=0),
        torch.cat([render_patch, rolled_render], dim=0),
        torch.cat([target, target_negative.to(device=target.device)], dim=0),
        torch.cat([base_dustbin, negative_dustbin], dim=0),
    )


@torch.no_grad()
def _evaluate(
    *,
    model: RGBPatchMeasurementBranch,
    rows: Sequence[Mapping[str, object]],
    image_root: Path,
    render_cache_by_query: Mapping[str, Path],
    image_width: int,
    image_height: int,
    query_image_width: int | None,
    query_image_height: int | None,
    render_image_width: int | None,
    render_image_height: int | None,
    batch_size: int,
    device: torch.device,
    query_cache: MutableMapping[str, torch.Tensor],
    render_cache: MutableMapping[str, torch.Tensor],
    max_eval_rows: int | None,
    epe_weight: float,
    delta_loss_weight: float,
    gated_delta_loss_weight: float,
    gate_supervision_loss_weight: float,
    gate_center_radius_px: float,
    gate_full_radius_px: float,
    gate_target_mode: str,
    gate_utility_temperature_px: float,
    gate_minimum_update_gain_px: float,
    gate_positive_weight: float,
    gate_low_residual_threshold_px: float,
    gate_low_residual_negative_weight: float,
    likelihood_loss_weight: float,
    coarse_likelihood_loss_weight: float,
    dustbin_bce_weight: float,
    dustbin_positive_weight: float,
    target_heatmap_sigma_px: float,
    query_source: str,
    support_patch_warp: str,
    prior_scale_key: str,
    target_x_key: str,
    target_y_key: str,
    loss_weight_key: str,
    image_cache_device: torch.device | None,
) -> dict[str, float | int | None]:
    values = list(rows[: int(max_eval_rows)]) if max_eval_rows is not None else list(rows)
    if not values:
        return {"count": 0, "loss": None, "epe_px": None, "epe_median_px": None, "recall_0p5px": None, "improve_ratio": None}
    model.eval()
    losses: list[float] = []
    epes: list[torch.Tensor] = []
    likelihood_epes: list[torch.Tensor] = []
    mode_epes: list[torch.Tensor] = []
    coarse_epes: list[torch.Tensor] = []
    gated_epes: list[torch.Tensor] = []
    gate_probs: list[torch.Tensor] = []
    baselines: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    dustbin_probs: list[torch.Tensor] = []
    for start in range(0, len(values), int(batch_size)):
        batch_rows = values[start : start + int(batch_size)]
        query_patch, render_patch, target, baseline, target_is_dustbin = _stack_patch_batch(
            batch_rows,
            image_root=image_root,
            render_cache_by_query=render_cache_by_query,
            image_width=int(image_width),
            image_height=int(image_height),
            query_image_width=query_image_width,
            query_image_height=query_image_height,
            render_image_width=render_image_width,
            render_image_height=render_image_height,
            crop_radius_px=model.crop_radius_px,
            step_px=model.step_px,
            query_cache=query_cache,
            render_cache=render_cache,
            query_source=str(query_source),
            render_patch_augmentation="none",
            support_patch_warp=str(support_patch_warp),
            target_x_key=str(target_x_key),
            target_y_key=str(target_y_key),
            image_cache_device=image_cache_device,
        )
        prior_scale = _prior_scale_batch(batch_rows, prior_scale_key=str(prior_scale_key))
        sample_weight = _sample_weight_batch(batch_rows, loss_weight_key=str(loss_weight_key))
        pred0 = model.forward_from_patches(
            query_patch.to(device),
            render_patch.to(device),
            prior_scale_px=None if prior_scale is None else prior_scale.to(device),
        )
        delta_loss, pred = residual_delta_gaussian_nll(
            pred0.direct_mean_offset_xy,
            pred0.direct_log_sigma_xy,
            target.to(device),
            dustbin_logit=pred0.dustbin_logit,
            search_radius_px=model.measurement_search_radius_px,
            target_is_dustbin=None if target_is_dustbin is None else target_is_dustbin.to(device),
            sample_weight=None if sample_weight is None else sample_weight.to(device),
            dustbin_positive_weight=float(dustbin_positive_weight),
        )
        likelihood_loss, likelihood_pred = continuous_offset_nll_with_dustbin(
            pred0.logits,
            pred0.offsets_xy,
            target.to(device),
            dustbin_logit=pred0.dustbin_logit,
            search_radius_px=model.measurement_search_radius_px,
            epe_weight=float(epe_weight),
            dustbin_bce_weight=float(dustbin_bce_weight),
            dustbin_positive_weight=float(dustbin_positive_weight),
            target_is_dustbin=None if target_is_dustbin is None else target_is_dustbin.to(device),
            target_heatmap_sigma_px=float(target_heatmap_sigma_px),
            sample_weight=None if sample_weight is None else sample_weight.to(device),
        )
        coarse_loss, coarse_pred = _coarse_stage_likelihood_loss(
            pred0,
            target.to(device),
            search_radius_px=model.measurement_search_radius_px,
            target_is_dustbin=None if target_is_dustbin is None else target_is_dustbin.to(device),
            sample_weight=None if sample_weight is None else sample_weight.to(device),
            dustbin_positive_weight=float(dustbin_positive_weight),
            target_heatmap_sigma_px=float(target_heatmap_sigma_px),
        )
        gated_delta_loss = None
        if pred0.gated_mean_offset_xy is not None:
            gated_delta_loss, _gated_pred = residual_delta_gaussian_nll(
                pred0.gated_mean_offset_xy,
                pred0.direct_log_sigma_xy,
                target.to(device),
                dustbin_logit=pred0.dustbin_logit,
                search_radius_px=model.measurement_search_radius_px,
                target_is_dustbin=None if target_is_dustbin is None else target_is_dustbin.to(device),
                sample_weight=None if sample_weight is None else sample_weight.to(device),
                dustbin_positive_weight=float(dustbin_positive_weight),
            )
        gate_supervision_loss = _gate_supervision_loss(
            pred0,
            target.to(device),
            target_is_dustbin=None if target_is_dustbin is None else target_is_dustbin.to(device),
            sample_weight=None if sample_weight is None else sample_weight.to(device),
            center_radius_px=float(gate_center_radius_px),
            full_radius_px=float(gate_full_radius_px),
            target_mode=str(gate_target_mode),
            utility_temperature_px=float(gate_utility_temperature_px),
            minimum_update_gain_px=float(gate_minimum_update_gain_px),
            positive_weight=float(gate_positive_weight),
            low_residual_threshold_px=float(gate_low_residual_threshold_px),
            low_residual_negative_weight=float(gate_low_residual_negative_weight),
        )
        if pred0.gated_mean_offset_xy is not None and pred0.gate_probability is not None:
            gated_epes.append(torch.linalg.norm(pred0.gated_mean_offset_xy.detach().cpu() - target.detach().cpu(), dim=1))
            gate_probs.append(pred0.gate_probability.detach().cpu())
        loss = float(delta_loss_weight) * delta_loss + float(likelihood_loss_weight) * likelihood_loss
        if gated_delta_loss is not None:
            loss = loss + float(gated_delta_loss_weight) * gated_delta_loss
        if gate_supervision_loss is not None:
            loss = loss + float(gate_supervision_loss_weight) * gate_supervision_loss
        if coarse_loss is not None:
            loss = loss + float(coarse_likelihood_loss_weight) * coarse_loss
        epe = pred.epe_px.detach().cpu()
        losses.append(float(loss.detach().cpu().item()) * len(batch_rows))
        epes.append(epe)
        likelihood_epes.append(likelihood_pred.epe_px.detach().cpu())
        if coarse_pred is not None:
            coarse_epes.append(coarse_pred.epe_px.detach().cpu())
        if likelihood_pred.mode_offset_xy is None:
            mode_epes.append(likelihood_pred.epe_px.detach().cpu())
        else:
            mode_epes.append(torch.linalg.norm(likelihood_pred.mode_offset_xy.detach().cpu() - target.detach().cpu(), dim=1))
        baselines.append(baseline)
        valid_masks.append((~likelihood_pred.target_is_dustbin).detach().cpu())
        dustbin_probs.append(likelihood_pred.dustbin_probability.detach().cpu())
    epe_all = torch.cat(epes, dim=0)
    likelihood_epe_all = torch.cat(likelihood_epes, dim=0)
    mode_epe_all = torch.cat(mode_epes, dim=0)
    coarse_epe_all = torch.cat(coarse_epes, dim=0) if coarse_epes else None
    gated_epe_all = torch.cat(gated_epes, dim=0) if gated_epes else None
    gate_prob_all = torch.cat(gate_probs, dim=0) if gate_probs else None
    baseline_all = torch.cat(baselines, dim=0)
    valid_mask_all = torch.cat(valid_masks, dim=0).bool()
    dustbin_mask_all = ~valid_mask_all
    dustbin_prob_all = torch.cat(dustbin_probs, dim=0)
    count = int(epe_all.numel())
    valid_count = int(torch.sum(valid_mask_all).item())
    dustbin_count = int(torch.sum(dustbin_mask_all).item())
    valid_epe = epe_all[valid_mask_all]
    valid_likelihood_epe = likelihood_epe_all[valid_mask_all]
    valid_mode_epe = mode_epe_all[valid_mask_all]
    valid_coarse_epe = coarse_epe_all[valid_mask_all] if coarse_epe_all is not None else None
    valid_gated_epe = gated_epe_all[valid_mask_all] if gated_epe_all is not None else None
    valid_baseline = baseline_all[valid_mask_all]
    dustbin_prob = dustbin_prob_all[dustbin_mask_all]
    valid_dustbin_prob = dustbin_prob_all[valid_mask_all]
    predicted_dustbin = dustbin_prob_all >= 0.5
    predicted_valid = ~predicted_dustbin
    predicted_dustbin_count = int(torch.sum(predicted_dustbin).item())
    true_dustbin_positive_count = int(torch.sum(predicted_dustbin & dustbin_mask_all).item())
    validity_target = dustbin_mask_all.to(dtype=dustbin_prob_all.dtype)
    accepted_mask_0p5 = dustbin_prob_all < 0.5
    accepted_valid_mask_0p5 = accepted_mask_0p5 & valid_mask_all
    accepted_dustbin_mask_0p5 = accepted_mask_0p5 & dustbin_mask_all
    accepted_valid_likelihood_epe_0p5 = likelihood_epe_all[accepted_valid_mask_0p5]
    accepted_valid_baseline_0p5 = baseline_all[accepted_valid_mask_0p5]
    accepted_count_0p5 = int(torch.sum(accepted_mask_0p5).item())
    accepted_valid_count_0p5 = int(torch.sum(accepted_valid_mask_0p5).item())
    accepted_dustbin_count_0p5 = int(torch.sum(accepted_dustbin_mask_0p5).item())
    metrics = {
        "count": count,
        "valid_count": valid_count,
        "dustbin_count": dustbin_count,
        "loss": float(sum(losses) / max(count, 1)),
        "epe_px": float(torch.mean(epe_all).item()),
        "epe_median_px": float(torch.median(epe_all).item()),
        "recall_0p5px": float(torch.mean((epe_all <= 0.5).float()).item()),
        "recall_1px": float(torch.mean((epe_all <= 1.0).float()).item()),
        "recall_2px": float(torch.mean((epe_all <= 2.0).float()).item()),
        "improve_ratio": float(torch.mean((epe_all < baseline_all).float()).item()),
        "median_improvement_px": float(torch.median(baseline_all - epe_all).item()),
        "valid_epe_px": float(torch.mean(valid_epe).item()) if valid_count else None,
        "valid_epe_median_px": float(torch.median(valid_epe).item()) if valid_count else None,
        "valid_recall_0p5px": float(torch.mean((valid_epe <= 0.5).float()).item()) if valid_count else None,
        "valid_recall_1px": float(torch.mean((valid_epe <= 1.0).float()).item()) if valid_count else None,
        "valid_improve_ratio": float(torch.mean((valid_epe < valid_baseline).float()).item()) if valid_count else None,
        "likelihood_epe_px": float(torch.mean(likelihood_epe_all).item()),
        "likelihood_epe_median_px": float(torch.median(likelihood_epe_all).item()),
        "likelihood_recall_0p5px": float(torch.mean((likelihood_epe_all <= 0.5).float()).item()),
        "likelihood_recall_1px": float(torch.mean((likelihood_epe_all <= 1.0).float()).item()),
        "likelihood_improve_ratio": float(torch.mean((likelihood_epe_all < baseline_all).float()).item()),
        "likelihood_median_improvement_px": float(torch.median(baseline_all - likelihood_epe_all).item()),
        "likelihood_valid_epe_px": float(torch.mean(valid_likelihood_epe).item()) if valid_count else None,
        "likelihood_valid_epe_median_px": float(torch.median(valid_likelihood_epe).item()) if valid_count else None,
        "likelihood_valid_recall_0p5px": float(torch.mean((valid_likelihood_epe <= 0.5).float()).item()) if valid_count else None,
        "likelihood_valid_recall_1px": float(torch.mean((valid_likelihood_epe <= 1.0).float()).item()) if valid_count else None,
        "likelihood_valid_improve_ratio": (
            float(torch.mean((valid_likelihood_epe < valid_baseline).float()).item()) if valid_count else None
        ),
        "mode_epe_px": float(torch.mean(mode_epe_all).item()),
        "mode_epe_median_px": float(torch.median(mode_epe_all).item()),
        "mode_recall_0p5px": float(torch.mean((mode_epe_all <= 0.5).float()).item()),
        "mode_recall_1px": float(torch.mean((mode_epe_all <= 1.0).float()).item()),
        "mode_improve_ratio": float(torch.mean((mode_epe_all < baseline_all).float()).item()),
        "mode_median_improvement_px": float(torch.median(baseline_all - mode_epe_all).item()),
        "mode_valid_epe_px": float(torch.mean(valid_mode_epe).item()) if valid_count else None,
        "mode_valid_epe_median_px": float(torch.median(valid_mode_epe).item()) if valid_count else None,
        "mode_valid_recall_0p5px": float(torch.mean((valid_mode_epe <= 0.5).float()).item()) if valid_count else None,
        "mode_valid_recall_1px": float(torch.mean((valid_mode_epe <= 1.0).float()).item()) if valid_count else None,
        "mode_valid_improve_ratio": float(torch.mean((valid_mode_epe < valid_baseline).float()).item()) if valid_count else None,
        "dustbin_probability_mean": float(torch.mean(dustbin_prob).item()) if dustbin_count else None,
        "dustbin_recall_0p5": float(torch.mean((dustbin_prob >= 0.5).float()).item()) if dustbin_count else None,
        "valid_dustbin_probability_mean": float(torch.mean(valid_dustbin_prob).item()) if valid_count else None,
        "valid_accept_recall_0p5": float(torch.mean(predicted_valid[valid_mask_all].float()).item()) if valid_count else None,
        "validity_accuracy_0p5": float(torch.mean((predicted_dustbin == dustbin_mask_all).float()).item()),
        "dustbin_precision_0p5": (
            float(true_dustbin_positive_count / predicted_dustbin_count) if predicted_dustbin_count > 0 else None
        ),
        "validity_brier": float(torch.mean((dustbin_prob_all - validity_target) ** 2).item()),
        "validity_ece": _binary_ece(dustbin_prob_all, dustbin_mask_all, bin_count=10),
        "validity_auroc": _binary_auroc(dustbin_prob_all, dustbin_mask_all),
        "dustbin_valid_probability_gap": (
            float(torch.mean(dustbin_prob).item() - torch.mean(valid_dustbin_prob).item()) if dustbin_count and valid_count else None
        ),
        "accepted_count_0p5": accepted_count_0p5,
        "accepted_valid_count_0p5": accepted_valid_count_0p5,
        "accepted_dustbin_count_0p5": accepted_dustbin_count_0p5,
        "accepted_valid_likelihood_epe_median_px_0p5": (
            float(torch.median(accepted_valid_likelihood_epe_0p5).item()) if accepted_valid_count_0p5 else None
        ),
        "accepted_valid_likelihood_epe_px_0p5": (
            float(torch.mean(accepted_valid_likelihood_epe_0p5).item()) if accepted_valid_count_0p5 else None
        ),
        "accepted_valid_likelihood_recall_0p5px_0p5": (
            float(torch.mean((accepted_valid_likelihood_epe_0p5 <= 0.5).float()).item()) if accepted_valid_count_0p5 else None
        ),
        "accepted_valid_likelihood_recall_1px_0p5": (
            float(torch.mean((accepted_valid_likelihood_epe_0p5 <= 1.0).float()).item()) if accepted_valid_count_0p5 else None
        ),
        "accepted_valid_likelihood_improve_ratio_0p5": (
            float(torch.mean((accepted_valid_likelihood_epe_0p5 < accepted_valid_baseline_0p5).float()).item())
            if accepted_valid_count_0p5
            else None
        ),
        "accepted_dustbin_fraction_0p5": (
            float(accepted_dustbin_count_0p5 / accepted_count_0p5) if accepted_count_0p5 > 0 else None
        ),
        "dustbin_reject_recall_0p5": (
            float(torch.mean((~accepted_mask_0p5[dustbin_mask_all]).float()).item()) if dustbin_count else None
        ),
    }
    if coarse_epe_all is not None:
        metrics.update(
            {
                "coarse_epe_px": float(torch.mean(coarse_epe_all).item()),
                "coarse_epe_median_px": float(torch.median(coarse_epe_all).item()),
                "coarse_recall_0p5px": float(torch.mean((coarse_epe_all <= 0.5).float()).item()),
                "coarse_recall_1px": float(torch.mean((coarse_epe_all <= 1.0).float()).item()),
                "coarse_improve_ratio": float(torch.mean((coarse_epe_all < baseline_all).float()).item()),
                "coarse_valid_epe_median_px": float(torch.median(valid_coarse_epe).item()) if valid_count else None,
                "coarse_valid_improve_ratio": (
                    float(torch.mean((valid_coarse_epe < valid_baseline).float()).item()) if valid_count else None
                ),
            }
        )
        metrics.update(_residual_bin_metrics(values, coarse_epe_all, baseline_all, prefix="coarse"))
    if gated_epe_all is not None and valid_gated_epe is not None:
        metrics.update(
            {
                "gated_epe_px": float(torch.mean(gated_epe_all).item()),
                "gated_epe_median_px": float(torch.median(gated_epe_all).item()),
                "gated_recall_0p5px": float(torch.mean((gated_epe_all <= 0.5).float()).item()),
                "gated_recall_1px": float(torch.mean((gated_epe_all <= 1.0).float()).item()),
                "gated_improve_ratio": float(torch.mean((gated_epe_all < baseline_all).float()).item()),
                "gated_median_improvement_px": float(torch.median(baseline_all - gated_epe_all).item()),
                "gated_valid_epe_px": float(torch.mean(valid_gated_epe).item()) if valid_count else None,
                "gated_valid_epe_median_px": float(torch.median(valid_gated_epe).item()) if valid_count else None,
                "gated_valid_recall_0p5px": float(torch.mean((valid_gated_epe <= 0.5).float()).item()) if valid_count else None,
                "gated_valid_recall_1px": float(torch.mean((valid_gated_epe <= 1.0).float()).item()) if valid_count else None,
                "gated_valid_improve_ratio": (
                    float(torch.mean((valid_gated_epe < valid_baseline).float()).item()) if valid_count else None
                ),
                "gate_probability_mean": None if gate_prob_all is None else float(torch.mean(gate_prob_all).item()),
                "gate_probability_valid_mean": (
                    None if gate_prob_all is None or not valid_count else float(torch.mean(gate_prob_all[valid_mask_all]).item())
                ),
            }
        )
        metrics.update(_residual_bin_metrics(values, gated_epe_all, baseline_all, prefix="gated"))
    metrics.update(_residual_bin_metrics(values, likelihood_epe_all, baseline_all, prefix="likelihood"))
    metrics.update(_residual_bin_metrics(values, mode_epe_all, baseline_all, prefix="mode"))
    metrics.update(_residual_bin_metrics(values, epe_all, baseline_all, prefix="direct"))
    metrics.update(
        _categorical_group_metrics(
            values,
            likelihood_epe_all,
            baseline_all,
            prefix="likelihood_policy",
            group_key="measurement_policy",
        )
    )
    metrics.update(
        _categorical_group_metrics(
            values,
            epe_all,
            baseline_all,
            prefix="direct_policy",
            group_key="measurement_policy",
        )
    )
    return metrics


def train_rgb_patch_measurement_branch(
    *,
    rows_csv: Path,
    val_rows_csv: Path | None = None,
    render_cache_manifest_csv: Path | None,
    val_render_cache_manifest_csv: Path | None = None,
    image_root: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    coarse_search_radius_px: float | None = None,
    coarse_step_px: float | None = None,
    query_image_width: int | None = None,
    query_image_height: int | None = None,
    render_image_width: int | None = None,
    render_image_height: int | None = None,
    steps: int = 1000,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = False,
    feature_dim: int = 32,
    hidden_dim: int | None = None,
    input_mode: str = "rgb",
    encoder_arch: str = "simple",
    template_scale_factors: Sequence[float] = (1.0,),
    lr: float = 1e-3,
    epe_weight: float = 0.25,
    delta_loss_weight: float = 1.0,
    gated_delta_loss_weight: float = 0.0,
    gate_supervision_loss_weight: float = 0.0,
    gate_center_radius_px: float = 0.5,
    gate_full_radius_px: float = 2.0,
    gate_target_mode: str = "residual",
    gate_utility_temperature_px: float = 0.25,
    gate_minimum_update_gain_px: float = 0.1,
    gate_positive_weight: float = 1.0,
    gate_low_residual_threshold_px: float = 1.0,
    gate_low_residual_negative_weight: float = 1.0,
    likelihood_loss_weight: float = 0.1,
    coarse_likelihood_loss_weight: float = 0.0,
    dustbin_bce_weight: float = 0.0,
    dustbin_positive_weight: float = 1.0,
    target_heatmap_sigma_px: float = 0.0,
    val_fraction: float = 0.1,
    val_group_key: str = "",
    max_rows: int | None = None,
    max_eval_rows: int | None = 512,
    eval_batch_size: int | None = None,
    seed: int = 0,
    device: str = "cuda",
    base_dir: Path | None = None,
    query_source: str = "real",
    support_patch_warp: str = "none",
    init_checkpoint: Path | None = None,
    train_stage: str = "",
    gate_median_epe_px: float = 0.5,
    gate_improve_ratio: float = 0.8,
    render_patch_augmentation: str = "none",
    hard_negative_fraction: float = 0.0,
    train_dustbin_head_only: bool = False,
    train_measurement_gate_head_only: bool = False,
    condition_on_prior_scale: bool = False,
    prior_scale_key: str = "",
    prior_scale_expert_centers_px: Sequence[float] = (),
    prior_scale_expert_projection: bool = False,
    prior_scale_expert_gate: str = "soft",
    target_x_key: str = "query_gt_x",
    target_y_key: str = "query_gt_y",
    loss_weight_key: str = "",
    min_loss_weight: float | None = None,
    cache_images_on_device: bool = False,
    target_dustbin_filter: str = "all",
    baseline_epe_min_px: float | None = None,
    baseline_epe_max_px: float | None = None,
    residual_balanced_sampling: bool = False,
    residual_sampling_bins_px: Sequence[float] = (),
    data_parallel_device_ids: Sequence[int] = (),
    image_cache_max_gb: float | None = None,
) -> dict[str, Any]:
    base = Path.cwd() if base_dir is None else Path(base_dir)
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(gradient_accumulation_steps) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if (coarse_search_radius_px is None) != (coarse_step_px is None):
        raise ValueError("coarse_search_radius_px and coarse_step_px must be provided together")
    effective_search_radius_px = float(search_radius_px) + (0.0 if coarse_search_radius_px is None else float(coarse_search_radius_px))
    if max_eval_rows is not None and int(max_eval_rows) <= 0:
        max_eval_rows = None
    if eval_batch_size is not None and int(eval_batch_size) <= 0:
        eval_batch_size = None
    raw_rows = _read_csv(Path(rows_csv), max_rows=max_rows)
    if not raw_rows:
        raise ValueError("rows_csv contains no rows")
    rows, row_filter_summary = _filter_rows_for_training(
        raw_rows,
        search_radius_px=float(effective_search_radius_px),
        target_x_key=str(target_x_key),
        target_y_key=str(target_y_key),
        target_dustbin_filter=str(target_dustbin_filter),
        baseline_epe_min_px=baseline_epe_min_px,
        baseline_epe_max_px=baseline_epe_max_px,
        loss_weight_key=str(loss_weight_key),
        min_loss_weight=min_loss_weight,
    )
    if not rows:
        raise ValueError("row filters removed all rows")
    val_row_filter_summary: dict[str, Any] | None = None
    if val_rows_csv is not None and str(val_rows_csv):
        train_rows = list(rows)
        raw_val_rows = _read_csv(Path(val_rows_csv), max_rows=max_rows)
        val_rows, val_row_filter_summary = _filter_rows_for_training(
            raw_val_rows,
            search_radius_px=float(effective_search_radius_px),
            target_x_key=str(target_x_key),
            target_y_key=str(target_y_key),
            target_dustbin_filter=str(target_dustbin_filter),
            baseline_epe_min_px=baseline_epe_min_px,
            baseline_epe_max_px=baseline_epe_max_px,
            loss_weight_key=str(loss_weight_key),
            min_loss_weight=min_loss_weight,
        )
        if not val_rows:
            raise ValueError("val_rows_csv contains no rows")
    else:
        train_rows, val_rows = _split_train_val(rows, val_fraction=float(val_fraction), seed=int(seed), group_key=str(val_group_key))
    render_map = _render_cache_by_query(None if render_cache_manifest_csv is None else Path(render_cache_manifest_csv), base_dir=base)
    val_render_map = (
        render_map
        if val_render_cache_manifest_csv is None or not str(val_render_cache_manifest_csv)
        else _render_cache_by_query(Path(val_render_cache_manifest_csv), base_dir=base)
    )
    q_width = int(query_image_width if query_image_width is not None else image_width)
    q_height = int(query_image_height if query_image_height is not None else image_height)
    r_width = int(render_image_width if render_image_width is not None else image_width)
    r_height = int(render_image_height if render_image_height is not None else image_height)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    image_cache_device = torch_device if bool(cache_images_on_device) else None
    torch.manual_seed(int(seed))
    rng = random.Random(int(seed))
    model = RGBPatchMeasurementBranch(
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        step_px=float(step_px),
        coarse_search_radius_px=None if coarse_search_radius_px is None else float(coarse_search_radius_px),
        coarse_step_px=None if coarse_step_px is None else float(coarse_step_px),
        feature_dim=int(feature_dim),
        hidden_dim=hidden_dim,
        input_mode=str(input_mode),
        encoder_arch=str(encoder_arch),
        template_scale_factors=tuple(float(value) for value in template_scale_factors),
        condition_on_prior_scale=bool(condition_on_prior_scale),
        prior_scale_expert_centers_px=tuple(float(value) for value in prior_scale_expert_centers_px),
        prior_scale_expert_projection=bool(prior_scale_expert_projection),
        prior_scale_expert_gate=str(prior_scale_expert_gate),
    ).to(torch_device)
    if init_checkpoint is not None and str(init_checkpoint):
        state = torch.load(Path(init_checkpoint), map_location=torch_device)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        allow_missing_prior_parameters = bool(condition_on_prior_scale) or bool(tuple(float(value) for value in prior_scale_expert_centers_px))
        allow_missing_gate_parameters = not any(str(key).startswith("measurement_gate_head.") for key in state_dict.keys())
        model.load_state_dict(
            state_dict,
            strict=not (allow_missing_prior_parameters or allow_missing_gate_parameters),
        )
    trainable_scope = _configure_head_only_training(
        model,
        train_dustbin_head_only=bool(train_dustbin_head_only),
        train_measurement_gate_head_only=bool(train_measurement_gate_head_only),
    )
    requested_data_parallel_ids = [int(value) for value in data_parallel_device_ids]
    active_data_parallel_ids: list[int] = []
    parallel_forward: nn.DataParallel | None = None
    if torch_device.type == "cuda" and len(requested_data_parallel_ids) > 1:
        visible_count = int(torch.cuda.device_count())
        usable_ids = [idx for idx in requested_data_parallel_ids if 0 <= int(idx) < visible_count]
        if len(usable_ids) > 1:
            active_data_parallel_ids = usable_ids
            parallel_forward = nn.DataParallel(
                _PatchForwardOnly(model),
                device_ids=active_data_parallel_ids,
                output_device=active_data_parallel_ids[0],
            )
    trainable_parameters = [parameter for parameter in model.parameters() if bool(parameter.requires_grad)]
    if not trainable_parameters:
        raise ValueError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(lr))
    amp_enabled = bool(use_amp) and torch_device.type == "cuda"
    gradient_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameter_count = int(sum(parameter.numel() for parameter in trainable_parameters))
    cache_max_bytes = (
        None
        if image_cache_max_gb is None or float(image_cache_max_gb) <= 0.0
        else int(float(image_cache_max_gb) * (1024**3))
    )
    image_cache = TensorImageLRUCache(max_bytes=cache_max_bytes)
    query_cache: MutableMapping[str, torch.Tensor] = image_cache
    render_cache: MutableMapping[str, torch.Tensor] = image_cache
    residual_sampler: _ResidualBalancedBatchSampler | None = None
    if bool(residual_balanced_sampling):
        residual_sampler = _ResidualBalancedBatchSampler(
            train_rows,
            residual_bin_edges_px=tuple(float(value) for value in residual_sampling_bins_px),
            target_x_key=str(target_x_key),
            target_y_key=str(target_y_key),
            search_radius_px=float(effective_search_radius_px),
        )
    residual_sampling_summary: dict[str, Any] = (
        {
            "enabled": False,
            "residual_bin_edges_px": [float(value) for value in residual_sampling_bins_px],
            "row_count": int(len(train_rows)),
        }
        if residual_sampler is None
        else dict(residual_sampler.summary)
    )
    accumulation_steps = int(gradient_accumulation_steps)
    final_metrics: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    for _micro_step in range(int(steps) * accumulation_steps):
        model.train()
        if residual_sampler is None:
            batch_rows = [train_rows[rng.randrange(len(train_rows))] for _ in range(int(batch_size))]
        else:
            batch_rows = residual_sampler.sample_batch(batch_size=int(batch_size), rng=rng)
        query_patch, render_patch, target, _baseline, target_is_dustbin = _stack_patch_batch(
            batch_rows,
            image_root=Path(image_root),
            render_cache_by_query=render_map,
            image_width=int(image_width),
            image_height=int(image_height),
            query_image_width=int(q_width),
            query_image_height=int(q_height),
            render_image_width=int(r_width),
            render_image_height=int(r_height),
            crop_radius_px=model.crop_radius_px,
            step_px=float(step_px),
            query_cache=query_cache,
            render_cache=render_cache,
            query_source=str(query_source),
            render_patch_augmentation=str(render_patch_augmentation),
            support_patch_warp=str(support_patch_warp),
            target_x_key=str(target_x_key),
            target_y_key=str(target_y_key),
            image_cache_device=image_cache_device,
        )
        query_patch, render_patch, target, target_is_dustbin = _append_roll_hard_negatives(
            query_patch,
            render_patch,
            target,
            target_is_dustbin,
            fraction=float(hard_negative_fraction),
        )
        prior_scale = _prior_scale_batch(batch_rows, prior_scale_key=str(prior_scale_key))
        if prior_scale is not None and int(query_patch.shape[0]) != int(prior_scale.shape[0]):
            extra = int(query_patch.shape[0]) - int(prior_scale.shape[0])
            if extra > 0:
                prior_scale = torch.cat([prior_scale, prior_scale[:extra]], dim=0)
        sample_weight = _sample_weight_batch(batch_rows, loss_weight_key=str(loss_weight_key))
        if sample_weight is not None and int(query_patch.shape[0]) != int(sample_weight.shape[0]):
            extra = int(query_patch.shape[0]) - int(sample_weight.shape[0])
            if extra > 0:
                sample_weight = torch.cat([sample_weight, sample_weight[:extra]], dim=0)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred0 = _forward_patch_prediction(
                model=model,
                parallel_forward=parallel_forward,
                query_patch=query_patch,
                render_patch=render_patch,
                prior_scale=prior_scale,
                device=torch_device,
            )
        delta_loss, pred = residual_delta_gaussian_nll(
            pred0.direct_mean_offset_xy,
            pred0.direct_log_sigma_xy,
            target.to(torch_device),
            dustbin_logit=pred0.dustbin_logit,
            search_radius_px=model.measurement_search_radius_px,
            target_is_dustbin=None if target_is_dustbin is None else target_is_dustbin.to(torch_device),
            sample_weight=None if sample_weight is None else sample_weight.to(torch_device),
            dustbin_positive_weight=float(dustbin_positive_weight),
        )
        likelihood_loss, likelihood_pred = continuous_offset_nll_with_dustbin(
            pred0.logits,
            pred0.offsets_xy,
            target.to(torch_device),
            dustbin_logit=pred0.dustbin_logit,
            search_radius_px=model.measurement_search_radius_px,
            epe_weight=float(epe_weight),
            dustbin_bce_weight=float(dustbin_bce_weight),
            dustbin_positive_weight=float(dustbin_positive_weight),
            target_is_dustbin=None if target_is_dustbin is None else target_is_dustbin.to(torch_device),
            target_heatmap_sigma_px=float(target_heatmap_sigma_px),
            sample_weight=None if sample_weight is None else sample_weight.to(torch_device),
        )
        coarse_loss, coarse_pred = _coarse_stage_likelihood_loss(
            pred0,
            target.to(torch_device),
            search_radius_px=model.measurement_search_radius_px,
            target_is_dustbin=None if target_is_dustbin is None else target_is_dustbin.to(torch_device),
            sample_weight=None if sample_weight is None else sample_weight.to(torch_device),
            dustbin_positive_weight=float(dustbin_positive_weight),
            target_heatmap_sigma_px=float(target_heatmap_sigma_px),
        )
        gated_delta_loss = None
        if pred0.gated_mean_offset_xy is not None:
            gated_delta_loss, _gated_pred = residual_delta_gaussian_nll(
                pred0.gated_mean_offset_xy,
                pred0.direct_log_sigma_xy,
                target.to(torch_device),
                dustbin_logit=pred0.dustbin_logit,
                search_radius_px=model.measurement_search_radius_px,
                target_is_dustbin=None if target_is_dustbin is None else target_is_dustbin.to(torch_device),
                sample_weight=None if sample_weight is None else sample_weight.to(torch_device),
                dustbin_positive_weight=float(dustbin_positive_weight),
            )
        gate_supervision_loss = _gate_supervision_loss(
            pred0,
            target.to(torch_device),
            target_is_dustbin=None if target_is_dustbin is None else target_is_dustbin.to(torch_device),
            sample_weight=None if sample_weight is None else sample_weight.to(torch_device),
            center_radius_px=float(gate_center_radius_px),
            full_radius_px=float(gate_full_radius_px),
            target_mode=str(gate_target_mode),
            utility_temperature_px=float(gate_utility_temperature_px),
            minimum_update_gain_px=float(gate_minimum_update_gain_px),
            positive_weight=float(gate_positive_weight),
            low_residual_threshold_px=float(gate_low_residual_threshold_px),
            low_residual_negative_weight=float(gate_low_residual_negative_weight),
        )
        loss = float(delta_loss_weight) * delta_loss + float(likelihood_loss_weight) * likelihood_loss
        if gated_delta_loss is not None:
            loss = loss + float(gated_delta_loss_weight) * gated_delta_loss
        if gate_supervision_loss is not None:
            loss = loss + float(gate_supervision_loss_weight) * gate_supervision_loss
        if coarse_loss is not None:
            loss = loss + float(coarse_likelihood_loss_weight) * coarse_loss
        gradient_scaler.scale(loss / float(accumulation_steps)).backward()
        if (_micro_step + 1) % accumulation_steps == 0:
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            epe = pred.epe_px.detach().cpu()
            final_metrics = {
                "loss": float(loss.detach().cpu().item()),
                "delta_loss": float(delta_loss.detach().cpu().item()),
                "gated_delta_loss": None if gated_delta_loss is None else float(gated_delta_loss.detach().cpu().item()),
                "gate_supervision_loss": (
                    None if gate_supervision_loss is None else float(gate_supervision_loss.detach().cpu().item())
                ),
                "likelihood_loss": float(likelihood_loss.detach().cpu().item()),
                "coarse_likelihood_loss": None if coarse_loss is None else float(coarse_loss.detach().cpu().item()),
                "epe_px": float(torch.mean(epe).item()),
                "recall_0p5px": float(torch.mean((epe <= 0.5).float()).item()),
                "recall_1px": float(torch.mean((epe <= 1.0).float()).item()),
                "likelihood_epe_px": float(torch.mean(likelihood_pred.epe_px.detach().cpu()).item()),
                "likelihood_recall_0p5px": float(torch.mean((likelihood_pred.epe_px.detach().cpu() <= 0.5).float()).item()),
            }
            if pred0.gated_mean_offset_xy is not None:
                gated_epe = torch.linalg.norm(pred0.gated_mean_offset_xy.detach().cpu() - target.detach().cpu(), dim=1)
                final_metrics.update(
                    {
                        "gated_epe_px": float(torch.mean(gated_epe).item()),
                        "gated_recall_0p5px": float(torch.mean((gated_epe <= 0.5).float()).item()),
                        "gated_recall_1px": float(torch.mean((gated_epe <= 1.0).float()).item()),
                        "gate_probability_mean": (
                            None if pred0.gate_probability is None else float(torch.mean(pred0.gate_probability.detach().cpu()).item())
                        ),
                    }
                )
            if likelihood_pred.mode_offset_xy is not None:
                mode_epe = torch.linalg.norm(likelihood_pred.mode_offset_xy.detach().cpu() - target.detach().cpu(), dim=1)
                final_metrics.update(
                    {
                        "mode_epe_px": float(torch.mean(mode_epe).item()),
                        "mode_recall_0p5px": float(torch.mean((mode_epe <= 0.5).float()).item()),
                        "mode_recall_1px": float(torch.mean((mode_epe <= 1.0).float()).item()),
                    }
                )
            if coarse_pred is not None:
                coarse_epe = coarse_pred.epe_px.detach().cpu()
                final_metrics.update(
                    {
                        "coarse_epe_px": float(torch.mean(coarse_epe).item()),
                        "coarse_recall_0p5px": float(torch.mean((coarse_epe <= 0.5).float()).item()),
                        "coarse_recall_1px": float(torch.mean((coarse_epe <= 1.0).float()).item()),
                    }
                )
    eval_batch = max(1, int(eval_batch_size)) if eval_batch_size is not None else max(1, min(int(batch_size), 16))
    train_eval_rows = train_rows[: int(max_eval_rows)] if max_eval_rows is not None else train_rows
    val_eval_rows = val_rows[: int(max_eval_rows)] if max_eval_rows is not None else val_rows
    train_eval_is_sampled = len(train_eval_rows) < len(train_rows)
    val_eval_is_sampled = len(val_eval_rows) < len(val_rows)
    train_metrics = _evaluate(
        model=model,
        rows=train_eval_rows,
        image_root=Path(image_root),
        render_cache_by_query=render_map,
        image_width=int(image_width),
        image_height=int(image_height),
        query_image_width=int(q_width),
        query_image_height=int(q_height),
        render_image_width=int(r_width),
        render_image_height=int(r_height),
        batch_size=eval_batch,
        device=torch_device,
        query_cache=query_cache,
        render_cache=render_cache,
        max_eval_rows=None,
        epe_weight=float(epe_weight),
        likelihood_loss_weight=float(likelihood_loss_weight),
        coarse_likelihood_loss_weight=float(coarse_likelihood_loss_weight),
        delta_loss_weight=float(delta_loss_weight),
        gated_delta_loss_weight=float(gated_delta_loss_weight),
        gate_supervision_loss_weight=float(gate_supervision_loss_weight),
        gate_center_radius_px=float(gate_center_radius_px),
        gate_full_radius_px=float(gate_full_radius_px),
        gate_target_mode=str(gate_target_mode),
        gate_utility_temperature_px=float(gate_utility_temperature_px),
        gate_minimum_update_gain_px=float(gate_minimum_update_gain_px),
        gate_positive_weight=float(gate_positive_weight),
        gate_low_residual_threshold_px=float(gate_low_residual_threshold_px),
        gate_low_residual_negative_weight=float(gate_low_residual_negative_weight),
        dustbin_bce_weight=float(dustbin_bce_weight),
        dustbin_positive_weight=float(dustbin_positive_weight),
        target_heatmap_sigma_px=float(target_heatmap_sigma_px),
        query_source=str(query_source),
        support_patch_warp=str(support_patch_warp),
        prior_scale_key=str(prior_scale_key),
        target_x_key=str(target_x_key),
        target_y_key=str(target_y_key),
        loss_weight_key=str(loss_weight_key),
        image_cache_device=image_cache_device,
    )
    val_metrics = _evaluate(
        model=model,
        rows=val_eval_rows,
        image_root=Path(image_root),
        render_cache_by_query=val_render_map,
        image_width=int(image_width),
        image_height=int(image_height),
        query_image_width=int(q_width),
        query_image_height=int(q_height),
        render_image_width=int(r_width),
        render_image_height=int(r_height),
        batch_size=eval_batch,
        device=torch_device,
        query_cache=query_cache,
        render_cache=render_cache,
        max_eval_rows=None,
        epe_weight=float(epe_weight),
        likelihood_loss_weight=float(likelihood_loss_weight),
        coarse_likelihood_loss_weight=float(coarse_likelihood_loss_weight),
        delta_loss_weight=float(delta_loss_weight),
        gated_delta_loss_weight=float(gated_delta_loss_weight),
        gate_supervision_loss_weight=float(gate_supervision_loss_weight),
        gate_center_radius_px=float(gate_center_radius_px),
        gate_full_radius_px=float(gate_full_radius_px),
        gate_target_mode=str(gate_target_mode),
        gate_utility_temperature_px=float(gate_utility_temperature_px),
        gate_minimum_update_gain_px=float(gate_minimum_update_gain_px),
        gate_positive_weight=float(gate_positive_weight),
        gate_low_residual_threshold_px=float(gate_low_residual_threshold_px),
        gate_low_residual_negative_weight=float(gate_low_residual_negative_weight),
        dustbin_bce_weight=float(dustbin_bce_weight),
        dustbin_positive_weight=float(dustbin_positive_weight),
        target_heatmap_sigma_px=float(target_heatmap_sigma_px),
        query_source=str(query_source),
        support_patch_warp=str(support_patch_warp),
        prior_scale_key=str(prior_scale_key),
        target_x_key=str(target_x_key),
        target_y_key=str(target_y_key),
        loss_weight_key=str(loss_weight_key),
        image_cache_device=image_cache_device,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "rgb_patch_measurement_branch.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "search_radius_px": float(search_radius_px),
                "context_radius_px": float(context_radius_px),
                "step_px": float(step_px),
                "coarse_search_radius_px": None if coarse_search_radius_px is None else float(coarse_search_radius_px),
                "coarse_step_px": None if coarse_step_px is None else float(coarse_step_px),
                "feature_dim": int(feature_dim),
                "hidden_dim": None if hidden_dim is None else int(hidden_dim),
                "input_mode": str(input_mode),
                "encoder_arch": str(encoder_arch),
                "template_scale_factors": [float(value) for value in template_scale_factors],
                "condition_on_prior_scale": bool(condition_on_prior_scale),
                "prior_scale_expert_centers_px": [float(value) for value in prior_scale_expert_centers_px],
                "prior_scale_expert_projection": bool(prior_scale_expert_projection),
                "prior_scale_expert_gate": str(prior_scale_expert_gate),
            },
        },
        checkpoint,
    )
    summary = {
        "stage": "measurement_v1_rgb_patch_measurement_branch_train",
        "rows_csv": str(rows_csv),
        "val_rows_csv": "" if val_rows_csv is None else str(val_rows_csv),
        "render_cache_manifest_csv": "" if render_cache_manifest_csv is None else str(render_cache_manifest_csv),
        "val_render_cache_manifest_csv": "" if val_render_cache_manifest_csv is None else str(val_render_cache_manifest_csv),
        "image_root": str(image_root),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "query_image_width": int(q_width),
        "query_image_height": int(q_height),
        "render_image_width": int(r_width),
        "render_image_height": int(r_height),
        "query_source": str(query_source),
        "support_patch_warp": str(support_patch_warp),
        "train_stage": str(train_stage) if str(train_stage) else str(query_source),
        "init_checkpoint": "" if init_checkpoint is None else str(init_checkpoint),
        "row_count": int(len(rows)),
        "raw_row_count": int(len(raw_rows)),
        "row_filter": row_filter_summary,
        "val_row_filter": val_row_filter_summary,
        "train_count": int(len(train_rows)),
        "val_count": int(len(val_rows)),
        "train_eval_count": int(len(train_eval_rows)),
        "val_eval_count": int(len(val_eval_rows)),
        "train_eval_is_sampled": bool(train_eval_is_sampled),
        "val_eval_is_sampled": bool(val_eval_is_sampled),
        "max_eval_rows": None if max_eval_rows is None else int(max_eval_rows),
        "val_group_key": str(val_group_key),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(accumulation_steps),
        "effective_batch_size": int(batch_size) * int(accumulation_steps),
        "use_amp": bool(amp_enabled),
        "feature_dim": int(feature_dim),
        "hidden_dim": None if hidden_dim is None else int(hidden_dim),
        "learning_rate": float(lr),
        "seed": int(seed),
        "eval_batch_size": int(eval_batch),
        "cache_images_on_device": bool(cache_images_on_device),
        "image_cache_device": "" if image_cache_device is None else str(image_cache_device),
        "image_cache_max_gb": None if cache_max_bytes is None else float(cache_max_bytes / (1024**3)),
        "image_cache": image_cache.summary(),
        "requested_data_parallel_device_ids": requested_data_parallel_ids,
        "active_data_parallel_device_ids": active_data_parallel_ids,
        "search_radius_px": float(search_radius_px),
        "coarse_search_radius_px": None if coarse_search_radius_px is None else float(coarse_search_radius_px),
        "coarse_step_px": None if coarse_step_px is None else float(coarse_step_px),
        "measurement_search_radius_px": float(model.measurement_search_radius_px),
        "context_radius_px": float(context_radius_px),
        "step_px": float(step_px),
        "input_mode": str(input_mode),
        "encoder_arch": str(encoder_arch),
        "template_scale_factors": [float(value) for value in template_scale_factors],
        "render_patch_augmentation": str(render_patch_augmentation),
        "hard_negative_fraction": float(hard_negative_fraction),
        "train_dustbin_head_only": bool(train_dustbin_head_only),
        "train_measurement_gate_head_only": bool(train_measurement_gate_head_only),
        "trainable_scope": str(trainable_scope),
        "condition_on_prior_scale": bool(condition_on_prior_scale),
        "prior_scale_key": str(prior_scale_key),
        "prior_scale_expert_centers_px": [float(value) for value in prior_scale_expert_centers_px],
        "prior_scale_expert_projection": bool(prior_scale_expert_projection),
        "prior_scale_expert_gate": str(prior_scale_expert_gate),
        "target_x_key": str(target_x_key),
        "target_y_key": str(target_y_key),
        "loss_weight_key": str(loss_weight_key),
        "min_loss_weight": None if min_loss_weight is None else float(min_loss_weight),
        "target_dustbin_filter": str(target_dustbin_filter),
        "baseline_epe_min_px": None if baseline_epe_min_px is None else float(baseline_epe_min_px),
        "baseline_epe_max_px": None if baseline_epe_max_px is None else float(baseline_epe_max_px),
        "residual_balanced_sampling": bool(residual_balanced_sampling),
        "residual_sampling": residual_sampling_summary,
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "delta_loss_weight": float(delta_loss_weight),
        "gated_delta_loss_weight": float(gated_delta_loss_weight),
        "gate_supervision_loss_weight": float(gate_supervision_loss_weight),
        "gate_center_radius_px": float(gate_center_radius_px),
        "gate_full_radius_px": float(gate_full_radius_px),
        "gate_target_mode": str(gate_target_mode),
        "gate_utility_temperature_px": float(gate_utility_temperature_px),
        "gate_minimum_update_gain_px": float(gate_minimum_update_gain_px),
        "gate_positive_weight": float(gate_positive_weight),
        "gate_low_residual_threshold_px": float(gate_low_residual_threshold_px),
        "gate_low_residual_negative_weight": float(gate_low_residual_negative_weight),
        "likelihood_loss_weight": float(likelihood_loss_weight),
        "coarse_likelihood_loss_weight": float(coarse_likelihood_loss_weight),
        "dustbin_bce_weight": float(dustbin_bce_weight),
        "dustbin_positive_weight": float(dustbin_positive_weight),
        "target_heatmap_sigma_px": float(target_heatmap_sigma_px),
        "final_metrics": final_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_center_baseline": _center_baseline_metrics(train_eval_rows),
        "val_center_baseline": _center_baseline_metrics(val_eval_rows),
        "support_patch_source_audit": _support_patch_source_audit(rows, query_source=str(query_source)),
        "acceptance_gate": {
            "split": "val",
            "evaluated_row_count": int(len(val_eval_rows)),
            "total_row_count": int(len(val_rows)),
            "is_sampled": bool(val_eval_is_sampled),
            "median_epe_threshold_px": float(gate_median_epe_px),
            "improve_ratio_threshold": float(gate_improve_ratio),
            "direct_head": _passes_measurement_gate(
                val_metrics,
                median_epe_key="valid_epe_median_px",
                improve_ratio_key="valid_improve_ratio",
                median_epe_threshold_px=float(gate_median_epe_px),
                improve_ratio_threshold=float(gate_improve_ratio),
            ),
            "likelihood_head": _passes_measurement_gate(
                val_metrics,
                median_epe_key="likelihood_valid_epe_median_px",
                improve_ratio_key="likelihood_valid_improve_ratio",
                median_epe_threshold_px=float(gate_median_epe_px),
                improve_ratio_threshold=float(gate_improve_ratio),
            ),
            "mode_head": _passes_measurement_gate(
                val_metrics,
                median_epe_key="mode_valid_epe_median_px",
                improve_ratio_key="mode_valid_improve_ratio",
                median_epe_threshold_px=float(gate_median_epe_px),
                improve_ratio_threshold=float(gate_improve_ratio),
            ),
            "gated_head": _passes_measurement_gate(
                val_metrics,
                median_epe_key="gated_valid_epe_median_px",
                improve_ratio_key="gated_valid_improve_ratio",
                median_epe_threshold_px=float(gate_median_epe_px),
                improve_ratio_threshold=float(gate_improve_ratio),
            ),
            "residual_bin_gates": {
                "direct_head": _residual_bin_gate_summary(
                    val_metrics,
                    prefix="direct",
                    median_epe_threshold_px=float(gate_median_epe_px),
                    improve_ratio_threshold=float(gate_improve_ratio),
                ),
                "likelihood_head": _residual_bin_gate_summary(
                    val_metrics,
                    prefix="likelihood",
                    median_epe_threshold_px=float(gate_median_epe_px),
                    improve_ratio_threshold=float(gate_improve_ratio),
                ),
                "mode_head": _residual_bin_gate_summary(
                    val_metrics,
                    prefix="mode",
                    median_epe_threshold_px=float(gate_median_epe_px),
                    improve_ratio_threshold=float(gate_improve_ratio),
                ),
                "gated_head": _residual_bin_gate_summary(
                    val_metrics,
                    prefix="gated",
                    median_epe_threshold_px=float(gate_median_epe_px),
                    improve_ratio_threshold=float(gate_improve_ratio),
                ),
            },
        },
        "outputs": {"checkpoint": str(checkpoint), "summary": str(output / "summary.json")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
