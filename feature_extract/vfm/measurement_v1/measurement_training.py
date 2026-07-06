from __future__ import annotations

import csv
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
import torch

from feature_extract.vfm.measurement_v1.measurement_branch import (
    Conv3FeatureProjection,
    SharedFeatureProjection,
    continuous_window_nll_and_moments,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import TexturePatchEncoder, template_search_cost_volume_logits
from feature_extract.vfm.measurement_v1.stride4_fine_feature import crop_feature_window, local_template_correlation_logits


FeatureProjection = Union[SharedFeatureProjection, Conv3FeatureProjection, TexturePatchEncoder]


def _read_csv(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= int(max_rows):
                break
        return rows


def _float(row: Mapping[str, object], key: str) -> float:
    return float(str(row.get(key, "")).strip())


def _target_xy(row: Mapping[str, object], *, target_x_key: str, target_y_key: str) -> list[float]:
    return [_float(row, str(target_x_key)), _float(row, str(target_y_key))]


def _sample_weight(row: Mapping[str, object], *, sample_weight_key: str) -> float:
    key = str(sample_weight_key).strip()
    if not key:
        return 1.0
    return _float(row, key)


def _feature_path(row: Mapping[str, object], *, side: str, feature_name: str) -> Path:
    key = f"{side}_{feature_name}_feature_cache_path"
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing feature cache path column: {key}")
    return Path(value)


def _patch_path(row: Mapping[str, object], *, side: str, feature_name: str) -> Path | None:
    key = f"{side}_{feature_name}_patch_cache_path"
    value = str(row.get(key, "")).strip()
    return Path(value) if value else None


def _has_patch_cache(row: Mapping[str, object], *, feature_name: str) -> bool:
    return _patch_path(row, side="query", feature_name=feature_name) is not None and _patch_path(
        row,
        side="render",
        feature_name=feature_name,
    ) is not None


def _load_feature(path: Path, key: str) -> torch.Tensor:
    with np.load(Path(path)) as data:
        arr = np.asarray(data[str(key)], dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"feature cache {path} key {key} must have shape (C,H,W), (H,W,C), or (1,C,H,W)")
    first_dim_looks_like_small_channels = arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0] and arr.shape[2] > arr.shape[0]
    last_dim_looks_like_small_channels = arr.shape[-1] <= 16 and arr.shape[0] >= arr.shape[-1] and arr.shape[1] >= arr.shape[-1]
    last_dim_looks_like_large_channels = arr.shape[-1] > arr.shape[0] and arr.shape[-1] > arr.shape[1] and not first_dim_looks_like_small_channels
    if (not first_dim_looks_like_small_channels) and (last_dim_looks_like_small_channels or last_dim_looks_like_large_channels):
        arr = np.moveaxis(arr, -1, 0)
    return torch.from_numpy(arr.astype(np.float32, copy=False))


def _load_feature_lru(
    cache: "OrderedDict[tuple[str, str], torch.Tensor]",
    path: Path,
    key: str,
    *,
    capacity: int | None,
) -> torch.Tensor:
    cache_key = (str(path), str(key))
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]
    value = _load_feature(path, str(key))
    cache[cache_key] = value
    if capacity is not None and int(capacity) > 0:
        while len(cache) > int(capacity):
            cache.popitem(last=False)
    return value


def _stack_batch(
    rows: Sequence[Mapping[str, object]],
    *,
    feature_name: str,
    feature_key: str,
    target_x_key: str,
    target_y_key: str,
    sample_weight_key: str,
    cache: "OrderedDict[tuple[str, str], torch.Tensor]",
    feature_cache_capacity: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_features = []
    render_features = []
    centers = []
    render_xy = []
    targets = []
    weights = []
    for row in rows:
        query_path = _feature_path(row, side="query", feature_name=feature_name)
        render_path = _feature_path(row, side="render", feature_name=feature_name)
        query_features.append(_load_feature_lru(cache, query_path, str(feature_key), capacity=feature_cache_capacity))
        render_features.append(_load_feature_lru(cache, render_path, str(feature_key), capacity=feature_cache_capacity))
        centers.append([_float(row, "center_x"), _float(row, "center_y")])
        render_xy.append([_float(row, "render_x"), _float(row, "render_y")])
        targets.append(_target_xy(row, target_x_key=str(target_x_key), target_y_key=str(target_y_key)))
        weights.append(_sample_weight(row, sample_weight_key=str(sample_weight_key)))
    return (
        torch.stack(query_features, dim=0).to(device),
        torch.stack(render_features, dim=0).to(device),
        torch.tensor(centers, dtype=torch.float32, device=device),
        torch.tensor(render_xy, dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
    )


def _stack_local_patch_batch(
    rows: Sequence[Mapping[str, object]],
    *,
    feature_name: str,
    feature_key: str,
    target_x_key: str,
    target_y_key: str,
    sample_weight_key: str,
    cache: "OrderedDict[tuple[str, str], torch.Tensor]",
    feature_cache_capacity: int | None,
    image_width: int,
    image_height: int,
    crop_radius_px: float,
    step_px: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rows and all(_has_patch_cache(row, feature_name=str(feature_name)) for row in rows):
        query_patches = []
        render_patches = []
        centers = []
        render_xy = []
        targets = []
        weights = []
        for row in rows:
            query_path = _patch_path(row, side="query", feature_name=str(feature_name))
            render_path = _patch_path(row, side="render", feature_name=str(feature_name))
            if query_path is None or render_path is None:
                raise ValueError("patch cache path unexpectedly missing")
            query_patches.append(_load_feature_lru(cache, query_path, str(feature_key), capacity=feature_cache_capacity))
            render_patches.append(_load_feature_lru(cache, render_path, str(feature_key), capacity=feature_cache_capacity))
            centers.append([_float(row, "center_x"), _float(row, "center_y")])
            render_xy.append([_float(row, "render_x"), _float(row, "render_y")])
            targets.append(_target_xy(row, target_x_key=str(target_x_key), target_y_key=str(target_y_key)))
            weights.append(_sample_weight(row, sample_weight_key=str(sample_weight_key)))
        return (
            torch.stack(query_patches, dim=0).to(device),
            torch.stack(render_patches, dim=0).to(device),
            torch.tensor(centers, dtype=torch.float32, device=device),
            torch.tensor(render_xy, dtype=torch.float32, device=device),
            torch.tensor(targets, dtype=torch.float32, device=device),
            torch.tensor(weights, dtype=torch.float32, device=device),
        )
    query, render, centers, anchors, target, weight = _stack_batch(
        rows,
        feature_name=str(feature_name),
        feature_key=str(feature_key),
        target_x_key=str(target_x_key),
        target_y_key=str(target_y_key),
        sample_weight_key=str(sample_weight_key),
        cache=cache,
        feature_cache_capacity=feature_cache_capacity,
        device=torch.device("cpu"),
    )
    query_patch, _ = crop_feature_window(
        query,
        centers,
        radius_px=float(crop_radius_px),
        step_px=float(step_px),
        image_width=int(image_width),
        image_height=int(image_height),
    )
    render_patch, _ = crop_feature_window(
        render,
        anchors,
        radius_px=float(crop_radius_px),
        step_px=float(step_px),
        image_width=int(image_width),
        image_height=int(image_height),
    )
    return (
        query_patch.to(device),
        render_patch.to(device),
        centers.to(device),
        anchors.to(device),
        target.to(device),
        weight.to(device),
    )


def _measurement_logits(
    *,
    model: FeatureProjection,
    query: torch.Tensor,
    render: torch.Tensor,
    centers: torch.Tensor,
    anchors: torch.Tensor,
    image_width: int,
    image_height: int,
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    crop_before_projection: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_proj = model(query)
    render_proj = model(render)
    if bool(crop_before_projection):
        logits, offsets = template_search_cost_volume_logits(
            query_proj,
            render_proj,
            search_radius_px=float(search_radius_px),
            context_radius_px=float(context_radius_px),
            step_px=float(step_px),
            temperature=1.0,
        )
        sample_xy = centers.reshape(int(centers.shape[0]), 1, 2) + offsets.to(device=centers.device, dtype=centers.dtype).reshape(1, -1, 2)
        if int(sample_xy.shape[0]) == 1:
            sample_xy = sample_xy[0]
        return logits, sample_xy.detach().cpu()
    return local_template_correlation_logits(
        query_proj,
        render_proj,
        query_centers_xy=centers,
        render_anchor_xy=anchors,
        image_width=int(image_width),
        image_height=int(image_height),
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        step_px=float(step_px),
    )


def _split_train_val(
    rows: Sequence[dict[str, str]],
    *,
    val_fraction: float,
    seed: int,
    group_key: str = "",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if str(group_key):
        groups: dict[str, list[dict[str, str]]] = {}
        for row_index, row in enumerate(rows):
            group = str(row.get(str(group_key), "")).strip() or f"__row_{int(row_index)}"
            groups.setdefault(group, []).append(row)
        group_names = list(groups)
        random.Random(int(seed)).shuffle(group_names)
        if len(group_names) <= 1:
            return list(rows), list(rows)
        val_count = max(1, int(round(float(val_fraction) * len(group_names)))) if float(val_fraction) > 0.0 else 0
        val_count = min(val_count, max(len(group_names) - 1, 0))
        val_group_names = set(group_names[:val_count])
        train_rows = [row for name in group_names if name not in val_group_names for row in groups[name]]
        val_rows = [row for name in group_names if name in val_group_names for row in groups[name]]
        return train_rows, val_rows
    shuffled = list(rows)
    random.Random(int(seed)).shuffle(shuffled)
    if len(shuffled) == 1:
        return list(shuffled), list(shuffled)
    val_count = max(1, int(round(float(val_fraction) * len(shuffled)))) if float(val_fraction) > 0.0 else 0
    val_count = min(val_count, max(len(shuffled) - 1, 0))
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:] if val_count > 0 else shuffled
    return train_rows, val_rows


def _group_split_summary(
    train_rows: Sequence[Mapping[str, object]],
    val_rows: Sequence[Mapping[str, object]],
    *,
    group_key: str,
) -> dict[str, Any]:
    if not str(group_key):
        return {
            "val_group_key": "",
            "train_group_count": None,
            "val_group_count": None,
            "group_overlap_count": None,
        }
    train_groups = {str(row.get(str(group_key), "")).strip() for row in train_rows}
    val_groups = {str(row.get(str(group_key), "")).strip() for row in val_rows}
    return {
        "val_group_key": str(group_key),
        "train_group_count": int(len(train_groups)),
        "val_group_count": int(len(val_groups)),
        "group_overlap_count": int(len(train_groups & val_groups)),
    }


def _feature_geometry_summary(
    feature: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    patch_cache_mode: str,
    row: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if feature.ndim != 3:
        raise ValueError("feature must have shape (C,H,W)")
    height = int(feature.shape[1])
    width = int(feature.shape[2])
    source = "precomputed_patch" if str(patch_cache_mode) == "precomputed" else "feature_map"
    if source == "precomputed_patch":
        out: dict[str, Any] = {
            "source": source,
            "input_channel_count": int(feature.shape[0]),
            "input_spatial_shape": [height, width],
            "search_radius_px": float(search_radius_px),
            "context_radius_px": float(context_radius_px),
            "step_px": float(step_px),
        }
        if row is not None:
            source_height = str(row.get("patch_cache_source_feature_height", "")).strip()
            source_width = str(row.get("patch_cache_source_feature_width", "")).strip()
            pitch_x = str(row.get("patch_cache_source_feature_pixel_pitch_x", "")).strip()
            pitch_y = str(row.get("patch_cache_source_feature_pixel_pitch_y", "")).strip()
            if source_height and source_width:
                out["source_feature_spatial_shape"] = [int(float(source_height)), int(float(source_width))]
            if pitch_x:
                out["source_feature_pixel_pitch_x"] = float(pitch_x)
                out["step_to_source_pitch_ratio_x"] = float(step_px) / max(float(pitch_x), 1e-12)
            if pitch_y:
                out["source_feature_pixel_pitch_y"] = float(pitch_y)
                out["step_to_source_pitch_ratio_y"] = float(step_px) / max(float(pitch_y), 1e-12)
            if pitch_x and pitch_y:
                min_source_pitch = min(float(pitch_x), float(pitch_y))
                out["requested_step_sub_source_pitch"] = bool(float(step_px) < 0.5 * min_source_pitch)
                out["search_radius_below_one_source_cell"] = bool(float(search_radius_px) < min_source_pitch)
                out["context_radius_below_one_source_cell"] = bool(float(context_radius_px) < min_source_pitch)
        return out
    pitch_x = float(image_width - 1) / max(float(width - 1), 1.0)
    pitch_y = float(image_height - 1) / max(float(height - 1), 1.0)
    min_pitch = min(pitch_x, pitch_y)
    return {
        "source": source,
        "input_channel_count": int(feature.shape[0]),
        "input_spatial_shape": [height, width],
        "feature_pixel_pitch_x": float(pitch_x),
        "feature_pixel_pitch_y": float(pitch_y),
        "search_radius_feature_cells_x": float(search_radius_px) / max(pitch_x, 1e-12),
        "search_radius_feature_cells_y": float(search_radius_px) / max(pitch_y, 1e-12),
        "context_radius_feature_cells_x": float(context_radius_px) / max(pitch_x, 1e-12),
        "context_radius_feature_cells_y": float(context_radius_px) / max(pitch_y, 1e-12),
        "step_to_pitch_ratio_x": float(step_px) / max(pitch_x, 1e-12),
        "step_to_pitch_ratio_y": float(step_px) / max(pitch_y, 1e-12),
        "requested_step_sub_feature_pitch": bool(float(step_px) < 0.5 * min_pitch),
        "search_radius_below_one_feature_cell": bool(float(search_radius_px) < min_pitch),
    }


def _center_baseline_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    target_x_key: str = "query_gt_x",
    target_y_key: str = "query_gt_y",
) -> dict[str, float | int | None]:
    if not rows:
        return {"count": 0, "epe_px": None, "epe_median_px": None, "recall_0p5px": None, "recall_1px": None, "recall_2px": None}
    epes = []
    for row in rows:
        center = torch.tensor([_float(row, "center_x"), _float(row, "center_y")], dtype=torch.float32)
        target = torch.tensor(_target_xy(row, target_x_key=str(target_x_key), target_y_key=str(target_y_key)), dtype=torch.float32)
        epes.append(torch.linalg.norm(center - target).reshape(1))
    epe_all = torch.cat(epes, dim=0)
    return {
        "count": int(epe_all.numel()),
        "epe_px": float(torch.mean(epe_all).item()),
        "epe_median_px": float(torch.median(epe_all).item()),
        "recall_0p5px": float(torch.mean((epe_all <= 0.5).float()).item()),
        "recall_1px": float(torch.mean((epe_all <= 1.0).float()).item()),
        "recall_2px": float(torch.mean((epe_all <= 2.0).float()).item()),
    }


@torch.no_grad()
def _evaluate_rows(
    *,
    model: FeatureProjection,
    rows: Sequence[Mapping[str, object]],
    feature_name: str,
    feature_key: str,
    target_x_key: str,
    target_y_key: str,
    sample_weight_key: str,
    image_width: int,
    image_height: int,
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    temperature: float,
    batch_size: int,
    cache: "OrderedDict[tuple[str, str], torch.Tensor]",
    feature_cache_capacity: int | None,
    device: torch.device,
    max_eval_rows: int | None,
    crop_before_projection: bool,
) -> dict[str, float | int | None]:
    eval_rows = list(rows[: int(max_eval_rows)]) if max_eval_rows is not None else list(rows)
    if not eval_rows:
        return {"count": 0, "loss": None, "epe_px": None, "recall_0p5px": None, "recall_1px": None}
    losses: list[float] = []
    epes: list[torch.Tensor] = []
    baselines: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(eval_rows), int(batch_size)):
        batch_rows = eval_rows[start : start + int(batch_size)]
        if bool(crop_before_projection):
            query, render, centers, anchors, target, weight = _stack_local_patch_batch(
                batch_rows,
                feature_name=str(feature_name),
                feature_key=str(feature_key),
                target_x_key=str(target_x_key),
                target_y_key=str(target_y_key),
                sample_weight_key=str(sample_weight_key),
                cache=cache,
                feature_cache_capacity=feature_cache_capacity,
                image_width=int(image_width),
                image_height=int(image_height),
                crop_radius_px=float(search_radius_px) + float(context_radius_px),
                step_px=float(step_px),
                device=device,
            )
        else:
            query, render, centers, anchors, target, weight = _stack_batch(
                batch_rows,
                feature_name=str(feature_name),
                feature_key=str(feature_key),
                target_x_key=str(target_x_key),
                target_y_key=str(target_y_key),
                sample_weight_key=str(sample_weight_key),
                cache=cache,
                feature_cache_capacity=feature_cache_capacity,
                device=device,
            )
        logits, sample_xy = _measurement_logits(
            model=model,
            query=query,
            render=render,
            centers=centers,
            anchors=anchors,
            image_width=int(image_width),
            image_height=int(image_height),
            search_radius_px=float(search_radius_px),
            context_radius_px=float(context_radius_px),
            step_px=float(step_px),
            crop_before_projection=bool(crop_before_projection),
        )
        loss, pred = continuous_window_nll_and_moments(
            logits / max(float(temperature), 1e-8),
            sample_xy,
            target,
            confidence=weight if str(sample_weight_key).strip() else None,
        )
        losses.append(float(loss.detach().cpu().item()) * len(batch_rows))
        epe = pred.epe_px if pred.epe_px is not None else torch.linalg.norm(pred.mean_xy_px - target, dim=1)
        epes.append(epe.detach().cpu())
        baseline = torch.linalg.norm(centers - target, dim=1)
        baselines.append(baseline.detach().cpu())
    epe_all = torch.cat(epes, dim=0)
    baseline_all = torch.cat(baselines, dim=0)
    count = int(epe_all.numel())
    return {
        "count": count,
        "loss": float(sum(losses) / max(count, 1)),
        "epe_px": float(torch.mean(epe_all).item()),
        "epe_median_px": float(torch.median(epe_all).item()),
        "epe_p90_px": float(torch.quantile(epe_all, 0.9).item()),
        "center_epe_median_px": float(torch.median(baseline_all).item()),
        "improve_ratio": float(torch.mean((epe_all < baseline_all).float()).item()),
        "recall_0p5px": float(torch.mean((epe_all <= 0.5).float()).item()),
        "recall_1px": float(torch.mean((epe_all <= 1.0).float()).item()),
        "recall_2px": float(torch.mean((epe_all <= 2.0).float()).item()),
        "recall_5px": float(torch.mean((epe_all <= 5.0).float()).item()),
    }


def train_cached_measurement_branch(
    *,
    rows_csv: Path,
    output_dir: Path,
    feature_name: str,
    feature_key: str,
    image_width: int,
    image_height: int,
    search_radius_px: float,
    context_radius_px: float = 0.0,
    step_px: float,
    steps: int = 1000,
    batch_size: int = 16,
    hidden_dim: int = 128,
    output_dim: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.1,
    device: str = "cuda",
    max_rows: int | None = None,
    seed: int = 0,
    val_fraction: float = 0.1,
    max_eval_rows: int | None = 512,
    feature_cache_capacity: int | None = None,
    crop_before_projection: bool = True,
    val_group_key: str = "",
    projection_type: str = "linear1x1",
    target_x_key: str = "query_gt_x",
    target_y_key: str = "query_gt_y",
    sample_weight_key: str = "",
) -> dict[str, Any]:
    rows = _read_csv(Path(rows_csv), max_rows=max_rows)
    if not rows:
        raise ValueError("rows_csv contains no rows")
    train_rows, val_rows = _split_train_val(rows, val_fraction=float(val_fraction), seed=int(seed), group_key=str(val_group_key))
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    rng = random.Random(int(seed))
    cache: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()
    patch_cache_mode = "precomputed" if bool(crop_before_projection) and _has_patch_cache(rows[0], feature_name=str(feature_name)) else "online_crop"
    if not bool(crop_before_projection):
        patch_cache_mode = "full_feature_map"
    first_path = (
        _patch_path(rows[0], side="query", feature_name=str(feature_name))
        if patch_cache_mode == "precomputed"
        else _feature_path(rows[0], side="query", feature_name=feature_name)
    )
    if first_path is None:
        raise ValueError("missing first feature or patch cache path")
    first = _load_feature(first_path, str(feature_key))
    feature_geometry = _feature_geometry_summary(
        first,
        image_width=int(image_width),
        image_height=int(image_height),
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        step_px=float(step_px),
        patch_cache_mode=patch_cache_mode,
        row=rows[0],
    )
    projection = str(projection_type)
    if projection == "linear1x1":
        model: FeatureProjection = SharedFeatureProjection(input_dim=int(first.shape[0]), hidden_dim=int(hidden_dim), output_dim=int(output_dim)).to(torch_device)
    elif projection == "conv3":
        model = Conv3FeatureProjection(input_dim=int(first.shape[0]), hidden_dim=int(hidden_dim), output_dim=int(output_dim)).to(torch_device)
    elif projection in {"texture_rgb", "texture_rgb_graygrad", "texture_norm_graygrad"}:
        if int(first.shape[0]) != 3:
            raise ValueError(f"{projection} requires 3-channel RGB-like input, got {int(first.shape[0])} channels")
        input_mode = projection[len("texture_") :]
        model = TexturePatchEncoder(feature_dim=int(output_dim), hidden_dim=int(hidden_dim), input_mode=input_mode).to(torch_device)
    else:
        raise ValueError("projection_type must be 'linear1x1', 'conv3', 'texture_rgb', 'texture_rgb_graygrad', or 'texture_norm_graygrad'")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    final_metrics: dict[str, float] = {}
    for _step in range(int(steps)):
        model.train()
        batch_rows = [train_rows[rng.randrange(len(train_rows))] for _ in range(int(batch_size))]
        if bool(crop_before_projection):
            query, render, centers, anchors, target, weight = _stack_local_patch_batch(
                batch_rows,
                feature_name=str(feature_name),
                feature_key=str(feature_key),
                target_x_key=str(target_x_key),
                target_y_key=str(target_y_key),
                sample_weight_key=str(sample_weight_key),
                cache=cache,
                feature_cache_capacity=feature_cache_capacity,
                image_width=int(image_width),
                image_height=int(image_height),
                crop_radius_px=float(search_radius_px) + float(context_radius_px),
                step_px=float(step_px),
                device=torch_device,
            )
        else:
            query, render, centers, anchors, target, weight = _stack_batch(
                batch_rows,
                feature_name=str(feature_name),
                feature_key=str(feature_key),
                target_x_key=str(target_x_key),
                target_y_key=str(target_y_key),
                sample_weight_key=str(sample_weight_key),
                cache=cache,
                feature_cache_capacity=feature_cache_capacity,
                device=torch_device,
            )
        logits, sample_xy = _measurement_logits(
            model=model,
            query=query,
            render=render,
            centers=centers,
            anchors=anchors,
            image_width=int(image_width),
            image_height=int(image_height),
            search_radius_px=float(search_radius_px),
            context_radius_px=float(context_radius_px),
            step_px=float(step_px),
            crop_before_projection=bool(crop_before_projection),
        )
        loss, pred = continuous_window_nll_and_moments(
            logits / max(float(temperature), 1e-8),
            sample_xy,
            target,
            confidence=weight if str(sample_weight_key).strip() else None,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            epe = pred.epe_px if pred.epe_px is not None else torch.linalg.norm(pred.mean_xy_px - target, dim=1)
            baseline = torch.linalg.norm(centers - target, dim=1)
            final_metrics = {
                "loss": float(loss.detach().cpu().item()),
                "epe_px": float(torch.mean(epe).detach().cpu().item()),
                "epe_median_px": float(torch.median(epe).detach().cpu().item()),
                "center_epe_median_px": float(torch.median(baseline).detach().cpu().item()),
                "improve_ratio": float(torch.mean((epe < baseline).float()).detach().cpu().item()),
                "recall_0p5px": float(torch.mean((epe <= 0.5).float()).detach().cpu().item()),
                "recall_1px": float(torch.mean((epe <= 1.0).float()).detach().cpu().item()),
            }
    train_metrics = _evaluate_rows(
        model=model,
        rows=train_rows,
        feature_name=str(feature_name),
        feature_key=str(feature_key),
        target_x_key=str(target_x_key),
        target_y_key=str(target_y_key),
        sample_weight_key=str(sample_weight_key),
        image_width=int(image_width),
        image_height=int(image_height),
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        step_px=float(step_px),
        temperature=float(temperature),
        batch_size=max(1, min(int(batch_size), 8)),
        cache=cache,
        feature_cache_capacity=feature_cache_capacity,
        device=torch_device,
        max_eval_rows=max_eval_rows,
        crop_before_projection=bool(crop_before_projection),
    )
    val_metrics = _evaluate_rows(
        model=model,
        rows=val_rows,
        feature_name=str(feature_name),
        feature_key=str(feature_key),
        target_x_key=str(target_x_key),
        target_y_key=str(target_y_key),
        sample_weight_key=str(sample_weight_key),
        image_width=int(image_width),
        image_height=int(image_height),
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        step_px=float(step_px),
        temperature=float(temperature),
        batch_size=max(1, min(int(batch_size), 8)),
        cache=cache,
        feature_cache_capacity=feature_cache_capacity,
        device=torch_device,
        max_eval_rows=max_eval_rows,
        crop_before_projection=bool(crop_before_projection),
    )
    train_eval_rows = train_rows[: int(max_eval_rows)] if max_eval_rows is not None else train_rows
    val_eval_rows = val_rows[: int(max_eval_rows)] if max_eval_rows is not None else val_rows
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "measurement_branch.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "feature_name": str(feature_name),
                "feature_key": str(feature_key),
                "input_dim": int(first.shape[0]),
                "hidden_dim": int(hidden_dim),
                "output_dim": int(output_dim),
                "projection_type": projection,
                "search_radius_px": float(search_radius_px),
                "context_radius_px": float(context_radius_px),
                "step_px": float(step_px),
                "temperature": float(temperature),
                "crop_before_projection": bool(crop_before_projection),
                "feature_geometry": feature_geometry,
                "target_x_key": str(target_x_key),
                "target_y_key": str(target_y_key),
                "sample_weight_key": str(sample_weight_key),
            },
        },
        checkpoint,
    )
    summary = {
        "stage": "measurement_v1_cached_measurement_branch_train",
        "rows_csv": str(rows_csv),
        "row_count": int(len(rows)),
        "train_count": int(len(train_rows)),
        "val_count": int(len(val_rows)),
        **_group_split_summary(train_rows, val_rows, group_key=str(val_group_key)),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "input_dim": int(first.shape[0]),
        "projection_type": projection,
        "target_x_key": str(target_x_key),
        "target_y_key": str(target_y_key),
        "sample_weight_key": str(sample_weight_key),
        "search_radius_px": float(search_radius_px),
        "context_radius_px": float(context_radius_px),
        "step_px": float(step_px),
        "feature_cache_capacity": int(feature_cache_capacity) if feature_cache_capacity is not None else None,
        "crop_before_projection": bool(crop_before_projection),
        "patch_cache_mode": patch_cache_mode,
        "feature_geometry": feature_geometry,
        "final_metrics": final_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_center_baseline": _center_baseline_metrics(
            train_eval_rows,
            target_x_key=str(target_x_key),
            target_y_key=str(target_y_key),
        ),
        "val_center_baseline": _center_baseline_metrics(
            val_eval_rows,
            target_x_key=str(target_x_key),
            target_y_key=str(target_y_key),
        ),
        "outputs": {"checkpoint": str(checkpoint), "summary": str(output / "summary.json")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
