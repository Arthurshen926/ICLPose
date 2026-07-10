from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.dense_depth_measurement_fusion import (
    DENSE_DEPTH_FUSION_FIELDNAMES,
    dense_depth_measurement_summary,
)
from feature_extract.vfm.measurement_v1.measurement_branch import Conv3FeatureProjection, SharedFeatureProjection
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    RGBPatchMeasurementPrediction,
    RGBPatchMeasurementBranch,
    TexturePatchEncoder,
    crop_rgb_window,
    template_search_cost_volume_logits,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    _PatchForwardOnly,
    _forward_patch_prediction,
    _load_query_rgb,
    _load_render_rgb,
    _load_tensor_cached,
    _prior_scale_batch,
    _read_csv,
    _render_cache_by_query,
)


RGB_PATCH_FUSION_FIELDNAMES = [
    *DENSE_DEPTH_FUSION_FIELDNAMES,
    "measurement_mean_dx",
    "measurement_mean_dy",
    "measurement_mode_dx",
    "measurement_mode_dy",
    "measurement_direct_dx",
    "measurement_direct_dy",
    "measurement_gated_dx",
    "measurement_gated_dy",
    "measurement_gate_prob",
    "measurement_peak_dx",
    "measurement_peak_dy",
    "local_cost_peak_prob",
    "local_cost_top2_gap",
    "rgb_patch_prediction_head",
    "measurement_search_radius_px",
    "measurement_fine_search_radius_px",
    "measurement_coarse_search_radius_px",
    "measurement_coarse_step_px",
    "measurement_context_radius_px",
    "measurement_step_px",
]


def _optional_float(row: Mapping[str, object], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            number = float(text)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return float(number)
    return None


def _query_center(row: Mapping[str, object]) -> tuple[float, float]:
    x = _optional_float(row, "query_center_x", "center_x", "query_x")
    y = _optional_float(row, "query_center_y", "center_y", "query_y")
    if x is None or y is None:
        raise ValueError("match row missing query center: expected query_center_x/y, center_x/y, or query_x/y")
    return float(x), float(y)


def _render_anchor(row: Mapping[str, object]) -> tuple[float, float]:
    x = _optional_float(row, "render_x")
    y = _optional_float(row, "render_y")
    if x is None or y is None:
        raise ValueError("match row missing render_x/render_y")
    return float(x), float(y)


def _canonical_reference_source(reference_source: str) -> str:
    source = str(reference_source).strip().lower()
    if source in {"", "render", "render_cache"}:
        return "render_cache"
    if source in {"real_pair", "support_image", "reference_image"}:
        return "real_pair"
    raise ValueError("reference_source must be one of: render_cache, real_pair")


def _real_pair_reference_image_id(row: Mapping[str, object]) -> str:
    for name in ("reference_image_id", "support_image_id"):
        value = str(row.get(name, "")).strip()
        if value:
            return value
    raise ValueError("real-pair row missing reference_image_id/support_image_id")


def _real_pair_reference_anchor(row: Mapping[str, object]) -> tuple[float, float]:
    x = _optional_float(row, "reference_x", "support_x", "render_x")
    y = _optional_float(row, "reference_y", "support_y", "render_y")
    if x is None or y is None:
        raise ValueError("real-pair row missing reference/support anchor coordinates")
    return float(x), float(y)


def _has_explicit_value(row: Mapping[str, object], name: str) -> bool:
    value = row.get(name)
    return value is not None and str(value).strip() != ""


def _valid_render_depth(row: Mapping[str, object]) -> bool:
    depth = _optional_float(row, "render_depth", "depth")
    return bool(depth is not None and np.isfinite(float(depth)) and float(depth) > 1e-6)


def _read_csv_rows(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows: list[dict[str, str]] = []
        for row in csv.DictReader(handle):
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= int(max_rows):
                break
        return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _fieldnames_for_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in RGB_PATCH_FUSION_FIELDNAMES:
        if name not in seen:
            seen.add(name)
            out.append(name)
    for row in rows:
        for name in row.keys():
            if name not in seen:
                seen.add(name)
                out.append(str(name))
    return out


def _canonical_prediction_head(prediction_head: str) -> str:
    head = str(prediction_head).strip().lower()
    if head in {"likelihood", "likelihood_mean", "mean"}:
        return "likelihood_mean"
    if head in {"mode", "likelihood_mode"}:
        return "likelihood_mode"
    if head in {"center", "noop", "no_op"}:
        return "center"
    if head == "direct":
        return "direct"
    if head in {"gated", "center_gated", "gated_likelihood"}:
        return "gated"
    raise ValueError("prediction_head must be one of center, likelihood_mean, likelihood_mode, mode, likelihood, direct, or gated")


class CachedProjectionMeasurementAdapter(nn.Module):
    """Compatibility wrapper for cache-trained template correlation projections."""

    measurement_model_type = "cached_projection"

    def __init__(
        self,
        *,
        projection: nn.Module,
        search_radius_px: float,
        context_radius_px: float,
        step_px: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.projection = projection
        self.search_radius_px = float(search_radius_px)
        self.context_radius_px = float(context_radius_px)
        self.step_px = float(step_px)
        self.temperature = float(temperature)
        if self.search_radius_px < 0.0 or self.context_radius_px < 0.0 or self.step_px <= 0.0:
            raise ValueError("search/context radii and step are invalid")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

    @property
    def crop_radius_px(self) -> float:
        return float(self.search_radius_px + self.context_radius_px)

    def forward_from_patches(
        self,
        query_patch: torch.Tensor,
        render_patch: torch.Tensor,
        prior_scale_px: torch.Tensor | None = None,
    ) -> RGBPatchMeasurementPrediction:
        del prior_scale_px
        if query_patch.ndim != 4 or render_patch.ndim != 4:
            raise ValueError("query_patch and render_patch must have shape (B,C,H,W)")
        if int(query_patch.shape[0]) != int(render_patch.shape[0]):
            raise ValueError("query_patch and render_patch must share batch size")
        query_features = self.projection(query_patch)
        render_features = self.projection(render_patch)
        logits, offsets = template_search_cost_volume_logits(
            query_features,
            render_features,
            search_radius_px=float(self.search_radius_px),
            context_radius_px=float(self.context_radius_px),
            step_px=float(self.step_px),
            temperature=1.0,
        )
        logits = logits / max(float(self.temperature), 1e-8)
        return RGBPatchMeasurementPrediction(
            logits=logits,
            offsets_xy=offsets,
            dustbin_logit=torch.zeros((int(query_patch.shape[0]),), device=logits.device, dtype=logits.dtype),
        )


def _load_cached_projection_measurement_adapter(
    payload: Mapping[str, Any],
    *,
    device: torch.device,
) -> CachedProjectionMeasurementAdapter:
    config = dict(payload.get("config", {}))
    projection_type = str(config.get("projection_type", ""))
    input_dim = int(config.get("input_dim", 0))
    hidden_dim = int(config.get("hidden_dim", 128))
    output_dim = int(config.get("output_dim", config.get("feature_dim", 64)))
    if projection_type == "linear1x1":
        projection: nn.Module = SharedFeatureProjection(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    elif projection_type == "conv3":
        projection = Conv3FeatureProjection(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    elif projection_type in {"texture_rgb", "texture_rgb_graygrad", "texture_norm_graygrad"}:
        if input_dim != 3:
            raise ValueError(f"{projection_type} requires 3-channel RGB-like input, got {input_dim} channels")
        projection = TexturePatchEncoder(
            feature_dim=output_dim,
            hidden_dim=hidden_dim,
            input_mode=projection_type[len("texture_") :],
        )
    else:
        raise ValueError(f"unsupported cached projection_type: {projection_type}")
    state = payload["model"]
    projection.load_state_dict(state, strict=True)
    return CachedProjectionMeasurementAdapter(
        projection=projection.to(device),
        search_radius_px=float(config["search_radius_px"]),
        context_radius_px=float(config.get("context_radius_px", 0.0)),
        step_px=float(config["step_px"]),
        temperature=float(config.get("temperature", 1.0)),
    ).to(device)


def _load_joint_measurement_patch_branch(payload: Mapping[str, Any], *, device: torch.device) -> RGBPatchMeasurementBranch:
    model_config = dict(payload.get("model_config", {}))
    branch_config = dict(model_config.get("measurement_patch_config", {}))
    if not branch_config:
        raise ValueError("joint checkpoint does not contain model_config.measurement_patch_config")
    model = RGBPatchMeasurementBranch(
        search_radius_px=float(branch_config.get("search_radius_px", 2.0)),
        context_radius_px=float(branch_config.get("context_radius_px", 8.0)),
        step_px=float(branch_config.get("step_px", 0.5)),
        coarse_search_radius_px=(
            None
            if float(branch_config.get("coarse_search_radius_px", 0.0)) <= 0.0
            else float(branch_config.get("coarse_search_radius_px"))
        ),
        coarse_step_px=(
            None
            if float(branch_config.get("coarse_step_px", 0.0)) <= 0.0
            else float(branch_config.get("coarse_step_px"))
        ),
        feature_dim=int(branch_config.get("feature_dim", 32)),
        hidden_dim=None if branch_config.get("hidden_dim") is None else int(branch_config.get("hidden_dim")),
        input_mode=str(branch_config.get("input_mode", "rgb")),
        encoder_arch=str(branch_config.get("encoder_arch", "simple")),
    ).to(device)
    state_dict = dict(payload.get("state_dict", {}))
    prefix = "measurement_patch_branch."
    branch_state = {
        str(key)[len(prefix) :]: value
        for key, value in state_dict.items()
        if str(key).startswith(prefix)
    }
    if not branch_state:
        raise ValueError("joint checkpoint state_dict does not contain measurement_patch_branch weights")
    incompatible = model.load_state_dict(branch_state, strict=False)
    allowed_missing = set()
    if int(model.prior_scale_expert_centers.numel()) == 0:
        allowed_missing.add("prior_scale_expert_centers")
    missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "joint checkpoint is incompatible with RGBPatchMeasurementBranch: "
            f"missing={missing}, unexpected={list(incompatible.unexpected_keys)}"
        )
    model.measurement_model_type = "joint_measurement_patch_branch"
    model.eval()
    return model


def load_rgb_patch_measurement_branch(checkpoint: Path, *, device: torch.device) -> nn.Module:
    payload = torch.load(Path(checkpoint), map_location=device)
    config = dict(payload.get("config", {}) if isinstance(payload, dict) else {})
    projection_type = str(config.get("projection_type", ""))
    if isinstance(payload, dict) and "state_dict" in payload and "model_config" in payload:
        return _load_joint_measurement_patch_branch(payload, device=device)
    if isinstance(payload, dict) and "model" in payload and projection_type in {
        "linear1x1",
        "conv3",
        "texture_rgb",
        "texture_rgb_graygrad",
        "texture_norm_graygrad",
    }:
        return _load_cached_projection_measurement_adapter(payload, device=device)
    model = RGBPatchMeasurementBranch(
        search_radius_px=float(config.get("search_radius_px", 2.0)),
        context_radius_px=float(config.get("context_radius_px", 8.0)),
        step_px=float(config.get("step_px", 0.5)),
        coarse_search_radius_px=(
            None if config.get("coarse_search_radius_px") is None else float(config.get("coarse_search_radius_px"))
        ),
        coarse_step_px=None if config.get("coarse_step_px") is None else float(config.get("coarse_step_px")),
        feature_dim=int(config.get("feature_dim", 32)),
        hidden_dim=None if config.get("hidden_dim") is None else int(config.get("hidden_dim")),
        input_mode=str(config.get("input_mode", "rgb")),
        encoder_arch=str(config.get("encoder_arch", "simple")),
        template_scale_factors=tuple(float(value) for value in config.get("template_scale_factors", [1.0])),
        condition_on_prior_scale=bool(config.get("condition_on_prior_scale", False)),
        prior_scale_expert_centers_px=tuple(float(value) for value in config.get("prior_scale_expert_centers_px", [])),
        prior_scale_expert_projection=bool(config.get("prior_scale_expert_projection", False)),
        prior_scale_expert_gate=str(config.get("prior_scale_expert_gate", "soft")),
    ).to(device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = set()
    if int(model.prior_scale_expert_centers.numel()) == 0:
        allowed_missing.add("prior_scale_expert_centers")
    allowed_missing.update(key for key in incompatible.missing_keys if str(key).startswith("measurement_gate_head."))
    missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint is incompatible with RGBPatchMeasurementBranch: "
            f"missing={missing}, unexpected={list(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model


def _load_query_image(
    *,
    row: Mapping[str, object],
    image_root: Path,
    query_cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    query_id = str(row.get("query_id", "")).strip()
    if not query_id:
        raise ValueError("match row missing query_id")
    query_path = Path(image_root) / query_id
    return _load_tensor_cached(query_cache, str(query_path), lambda p=query_path: _load_query_rgb(p))


def _load_render_image(
    *,
    row: Mapping[str, object],
    render_cache_by_query: Mapping[str, Path],
    render_cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    query_id = str(row.get("query_id", "")).strip()
    if not query_id:
        raise ValueError("match row missing query_id")
    render_path = render_cache_by_query.get(query_id)
    if render_path is None:
        raise ValueError(f"missing render cache for query_id={query_id}")
    return _load_tensor_cached(render_cache, str(render_path), lambda p=render_path: _load_render_rgb(p))


def _load_real_pair_reference_image(
    *,
    row: Mapping[str, object],
    image_root: Path,
    reference_cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    reference_id = _real_pair_reference_image_id(row)
    reference_path = Path(image_root) / reference_id
    return _load_tensor_cached(reference_cache, str(reference_path), lambda p=reference_path: _load_query_rgb(p))


def _crop_windows_for_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    image_root: Path,
    render_cache_by_query: Mapping[str, Path],
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    crop_radius_px: float,
    step_px: float,
    query_cache: dict[str, torch.Tensor],
    render_cache: dict[str, torch.Tensor],
    reference_source: str = "render_cache",
) -> tuple[torch.Tensor, torch.Tensor]:
    query_patches: list[torch.Tensor] = []
    render_patches: list[torch.Tensor] = []
    source = _canonical_reference_source(str(reference_source))
    for row in rows:
        query_image = _load_query_image(row=row, image_root=image_root, query_cache=query_cache).unsqueeze(0)
        if source == "real_pair":
            render_image = _load_real_pair_reference_image(
                row=row,
                image_root=image_root,
                reference_cache=render_cache,
            ).unsqueeze(0)
            anchor_xy = _real_pair_reference_anchor(row)
        else:
            render_image = _load_render_image(
                row=row,
                render_cache_by_query=render_cache_by_query,
                render_cache=render_cache,
            ).unsqueeze(0)
            anchor_xy = _render_anchor(row)
        center = torch.tensor([_query_center(row)], dtype=torch.float32)
        anchor = torch.tensor([anchor_xy], dtype=torch.float32)
        query_patch, _ = crop_rgb_window(
            query_image,
            center,
            radius_px=float(crop_radius_px),
            step_px=float(step_px),
            image_width=int(query_image_width),
            image_height=int(query_image_height),
        )
        render_patch, _ = crop_rgb_window(
            render_image,
            anchor,
            radius_px=float(crop_radius_px),
            step_px=float(step_px),
            image_width=int(render_image_width),
            image_height=int(render_image_height),
        )
        query_patches.append(query_patch[0])
        render_patches.append(render_patch[0])
    return torch.stack(query_patches, dim=0), torch.stack(render_patches, dim=0)


def _likelihood_stats(
    logits: torch.Tensor,
    offsets_xy: torch.Tensor,
    *,
    covariance_floor_px2: float,
) -> dict[str, torch.Tensor]:
    probs = F.softmax(logits, dim=1)
    offsets = offsets_xy.to(device=logits.device, dtype=logits.dtype)
    if offsets.ndim == 2:
        offsets = offsets.reshape(1, -1, 2).expand(int(logits.shape[0]), -1, -1)
    elif offsets.ndim == 3:
        if int(offsets.shape[0]) != int(logits.shape[0]):
            raise ValueError("batched offsets must have the same batch size as logits")
    else:
        raise ValueError("offsets_xy must have shape (K,2) or (B,K,2)")
    if int(offsets.shape[1]) != int(logits.shape[1]) or int(offsets.shape[2]) != 2:
        raise ValueError("offsets_xy must align with logits and contain xy pairs")
    mean = torch.sum(probs[..., None] * offsets, dim=1)
    centered = offsets - mean[:, None, :]
    cov = torch.einsum("bk,bki,bkj->bij", probs, centered, centered)
    cov = cov + torch.eye(2, device=logits.device, dtype=logits.dtype).unsqueeze(0) * float(covariance_floor_px2)
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=1)
    entropy_norm = entropy / max(math.log(float(int(probs.shape[1]))), 1e-12)
    top2 = torch.topk(probs, k=2, dim=1).values
    peak_idx = torch.argmax(probs, dim=1)
    batch_idx = torch.arange(int(logits.shape[0]), device=logits.device)
    peak = offsets[batch_idx, peak_idx]
    return {
        "mean": mean,
        "cov": cov,
        "entropy_norm": entropy_norm,
        "peak": peak,
        "peak_prob": top2[:, 0],
        "top2_gap": top2[:, 0] - top2[:, 1],
    }


def _direct_stats(
    mean_offset_xy: torch.Tensor,
    log_sigma_xy: torch.Tensor,
    logits: torch.Tensor,
    offsets_xy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    log_sigma = log_sigma_xy.reshape(int(mean_offset_xy.shape[0]), 2).clamp(-5.0, 3.0)
    sigma = torch.exp(log_sigma).clamp_min(1e-4)
    likelihood = _likelihood_stats(logits, offsets_xy, covariance_floor_px2=1e-4)
    return {
        "mean": mean_offset_xy.reshape(int(mean_offset_xy.shape[0]), 2),
        "cov": torch.diag_embed(sigma * sigma),
        "entropy_norm": likelihood["entropy_norm"],
        "peak": likelihood["peak"],
        "peak_prob": likelihood["peak_prob"],
        "top2_gap": likelihood["top2_gap"],
    }


def _mode_stats(
    logits: torch.Tensor,
    offsets_xy: torch.Tensor,
    *,
    covariance_floor_px2: float,
) -> dict[str, torch.Tensor]:
    likelihood = _likelihood_stats(logits, offsets_xy, covariance_floor_px2=float(covariance_floor_px2))
    return {
        "mean": likelihood["peak"],
        "cov": likelihood["cov"],
        "entropy_norm": likelihood["entropy_norm"],
        "peak": likelihood["peak"],
        "peak_prob": likelihood["peak_prob"],
        "top2_gap": likelihood["top2_gap"],
    }


def apply_rgb_patch_measurements_to_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    image_root: Path,
    render_cache_by_query: Mapping[str, Path],
    model: nn.Module,
    image_width: int | None = None,
    image_height: int | None = None,
    query_image_width: int | None = None,
    query_image_height: int | None = None,
    render_image_width: int | None = None,
    render_image_height: int | None = None,
    batch_size: int = 16,
    device: torch.device | str = "cuda",
    prediction_head: str = "center",
    prior_scale_key: str = "",
    covariance_floor_px2: float = 1e-4,
    data_parallel_device_ids: Sequence[int] = (),
    reference_source: str = "render_cache",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach RGB template-to-search local measurements to match-table rows.

    The function deliberately treats render-side geometry as immutable. It only
    writes query_refined_x/query_refined_y and measurement diagnostics. Existing
    render_x/render_y/render_depth/world_x/world_y/world_z values are preserved.
    """

    head = _canonical_prediction_head(str(prediction_head))
    source = _canonical_reference_source(str(reference_source))
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    q_width = int(query_image_width if query_image_width is not None else image_width if image_width is not None else 0)
    q_height = int(query_image_height if query_image_height is not None else image_height if image_height is not None else 0)
    r_width = int(render_image_width if render_image_width is not None else image_width if image_width is not None else q_width)
    r_height = int(render_image_height if render_image_height is not None else image_height if image_height is not None else q_height)
    if q_width <= 0 or q_height <= 0 or r_width <= 0 or r_height <= 0:
        raise ValueError("query/render image dimensions must be positive")
    model = model.to(torch_device)
    model.eval()
    requested_data_parallel_ids = [int(value) for value in data_parallel_device_ids]
    active_data_parallel_ids: list[int] = []
    parallel_forward: nn.DataParallel | None = None
    if torch_device.type == "cuda" and isinstance(model, RGBPatchMeasurementBranch) and len(requested_data_parallel_ids) > 1:
        visible_count = int(torch.cuda.device_count())
        usable_ids = [idx for idx in requested_data_parallel_ids if 0 <= int(idx) < visible_count]
        if len(usable_ids) > 1:
            active_data_parallel_ids = usable_ids
            parallel_forward = nn.DataParallel(
                _PatchForwardOnly(model),
                device_ids=active_data_parallel_ids,
                output_device=active_data_parallel_ids[0],
            )
    output = [dict(row) for row in rows]
    query_cache: dict[str, torch.Tensor] = {}
    render_cache: dict[str, torch.Tensor] = {}
    batch = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(output), batch):
            batch_rows = output[start : start + batch]
            query_patch, render_patch = _crop_windows_for_rows(
                batch_rows,
                image_root=Path(image_root),
                render_cache_by_query=render_cache_by_query,
                query_image_width=q_width,
                query_image_height=q_height,
                render_image_width=r_width,
                render_image_height=r_height,
                crop_radius_px=model.crop_radius_px,
                step_px=model.step_px,
                query_cache=query_cache,
                render_cache=render_cache,
                reference_source=source,
            )
            prior_scale = _prior_scale_batch(batch_rows, prior_scale_key=str(prior_scale_key))
            pred = _forward_patch_prediction(
                model=model,
                parallel_forward=parallel_forward,
                query_patch=query_patch,
                render_patch=render_patch,
                prior_scale=prior_scale,
                device=torch_device,
            )
            likelihood_stats = _likelihood_stats(pred.logits, pred.offsets_xy, covariance_floor_px2=float(covariance_floor_px2))
            mode_stats = _mode_stats(pred.logits, pred.offsets_xy, covariance_floor_px2=float(covariance_floor_px2))
            direct_stats = None
            gated_stats = None
            if head == "direct":
                if pred.direct_mean_offset_xy is None or pred.direct_log_sigma_xy is None:
                    raise ValueError("checkpoint does not expose direct offset head outputs")
                direct_stats = _direct_stats(pred.direct_mean_offset_xy, pred.direct_log_sigma_xy, pred.logits, pred.offsets_xy)
                stats = direct_stats
            elif head == "gated":
                if pred.gated_mean_offset_xy is None or pred.direct_log_sigma_xy is None:
                    raise ValueError("checkpoint does not expose gated offset head outputs")
                gated_stats = _direct_stats(pred.gated_mean_offset_xy, pred.direct_log_sigma_xy, pred.logits, pred.offsets_xy)
                stats = gated_stats
            elif head == "center":
                stats = dict(likelihood_stats)
                stats["mean"] = torch.zeros_like(likelihood_stats["mean"])
            elif head == "likelihood_mean":
                stats = likelihood_stats
            else:
                stats = mode_stats
            mean = stats["mean"].detach().cpu()
            cov = stats["cov"].detach().cpu()
            entropy = stats["entropy_norm"].detach().cpu()
            peak = stats["peak"].detach().cpu()
            peak_prob = stats["peak_prob"].detach().cpu()
            top2_gap = stats["top2_gap"].detach().cpu()
            likelihood_mean = likelihood_stats["mean"].detach().cpu()
            mode_mean = mode_stats["mean"].detach().cpu()
            direct_mean = None if direct_stats is None else direct_stats["mean"].detach().cpu()
            gated_mean = None if pred.gated_mean_offset_xy is None else pred.gated_mean_offset_xy.detach().cpu()
            gate_prob = None if pred.gate_probability is None else pred.gate_probability.detach().cpu()
            valid_prob = (1.0 - torch.sigmoid(pred.dustbin_logit.detach())).cpu()
            for local_index, row in enumerate(batch_rows):
                cx, cy = _query_center(row)
                dx = float(mean[local_index, 0].item())
                dy = float(mean[local_index, 1].item())
                cov_row = cov[local_index]
                cov_xx = float(cov_row[0, 0].item())
                cov_xy = float(cov_row[0, 1].item())
                cov_yy = float(cov_row[1, 1].item())
                sigma = float(math.sqrt(max(0.5 * (cov_xx + cov_yy), 1e-12)))
                row["query_center_x"] = row.get("query_center_x", row.get("center_x", row.get("query_x", cx)))
                row["query_center_y"] = row.get("query_center_y", row.get("center_y", row.get("query_y", cy)))
                row["query_refined_x"] = float(cx + dx)
                row["query_refined_y"] = float(cy + dy)
                row["measurement_dx"] = dx
                row["measurement_dy"] = dy
                row["measurement_mean_dx"] = float(likelihood_mean[local_index, 0].item())
                row["measurement_mean_dy"] = float(likelihood_mean[local_index, 1].item())
                row["measurement_mode_dx"] = float(mode_mean[local_index, 0].item())
                row["measurement_mode_dy"] = float(mode_mean[local_index, 1].item())
                row["measurement_direct_dx"] = "" if direct_mean is None else float(direct_mean[local_index, 0].item())
                row["measurement_direct_dy"] = "" if direct_mean is None else float(direct_mean[local_index, 1].item())
                row["measurement_gated_dx"] = "" if gated_mean is None else float(gated_mean[local_index, 0].item())
                row["measurement_gated_dy"] = "" if gated_mean is None else float(gated_mean[local_index, 1].item())
                row["measurement_gate_prob"] = "" if gate_prob is None else float(gate_prob[local_index].item())
                row["measurement_cov_xx"] = cov_xx
                row["measurement_cov_xy"] = cov_xy
                row["measurement_cov_yy"] = cov_yy
                row["measurement_sigma_px"] = sigma
                row["measurement_valid_prob"] = float(valid_prob[local_index].item())
                row["local_cost_entropy"] = float(entropy[local_index].item())
                row["measurement_peak_dx"] = float(peak[local_index, 0].item())
                row["measurement_peak_dy"] = float(peak[local_index, 1].item())
                row["local_cost_peak_prob"] = float(peak_prob[local_index].item())
                row["local_cost_top2_gap"] = float(top2_gap[local_index].item())
                row["rgb_patch_prediction_head"] = head
                coarse_radius = getattr(model, "coarse_search_radius_px", None)
                coarse_step = getattr(model, "coarse_step_px", None)
                fine_radius = float(getattr(model, "search_radius_px"))
                total_radius = float(getattr(model, "measurement_search_radius_px", fine_radius))
                row["measurement_search_radius_px"] = total_radius
                row["measurement_fine_search_radius_px"] = fine_radius
                row["measurement_coarse_search_radius_px"] = (
                    "" if coarse_radius is None else float(coarse_radius)
                )
                row["measurement_coarse_step_px"] = "" if coarse_step is None else float(coarse_step)
                row["measurement_context_radius_px"] = float(model.context_radius_px)
                row["measurement_step_px"] = float(model.step_px)
                if not _has_explicit_value(row, "depth_valid"):
                    row["depth_valid"] = _valid_render_depth(row)
    summary = dense_depth_measurement_summary(output)
    summary.update(
        {
            "stage": "measurement_v1_rgb_patch_match_table_fusion",
            "prediction_head": head,
            "measurement_model_type": str(getattr(model, "measurement_model_type", model.__class__.__name__)),
            "reference_source": source,
            "batch_size": int(batch),
            "query_image_width": int(q_width),
            "query_image_height": int(q_height),
            "render_image_width": int(r_width),
            "render_image_height": int(r_height),
            "prior_scale_key": str(prior_scale_key),
            "requested_data_parallel_device_ids": requested_data_parallel_ids,
            "active_data_parallel_device_ids": active_data_parallel_ids,
        }
    )
    return output, summary


def apply_rgb_patch_measurements_to_real_pair_rows(
    *,
    rows_csv: Path,
    image_root: Path,
    checkpoint: Path,
    output_dir: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    query_image_width: int | None = None,
    query_image_height: int | None = None,
    reference_image_width: int | None = None,
    reference_image_height: int | None = None,
    batch_size: int = 16,
    device: str = "cuda",
    max_rows: int | None = None,
    prediction_head: str = "center",
    prior_scale_key: str = "",
    data_parallel_device_ids: Sequence[int] = (),
) -> dict[str, Any]:
    rows = _read_csv_rows(Path(rows_csv), max_rows=max_rows)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    model = load_rgb_patch_measurement_branch(Path(checkpoint), device=torch_device)
    fused_rows, summary = apply_rgb_patch_measurements_to_rows(
        rows,
        image_root=Path(image_root),
        render_cache_by_query={},
        model=model,
        image_width=image_width,
        image_height=image_height,
        query_image_width=query_image_width,
        query_image_height=query_image_height,
        render_image_width=reference_image_width,
        render_image_height=reference_image_height,
        batch_size=int(batch_size),
        device=torch_device,
        prediction_head=str(prediction_head),
        prior_scale_key=str(prior_scale_key),
        data_parallel_device_ids=[int(value) for value in data_parallel_device_ids],
        reference_source="real_pair",
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames_for_rows(fused_rows)
    _write_csv(output / "match_table.csv", fused_rows, fieldnames)
    _write_jsonl(output / "match_table.jsonl", fused_rows)
    summary = {
        **summary,
        "rows_csv": str(rows_csv),
        "render_cache_manifest_csv": "",
        "image_root": str(image_root),
        "checkpoint": str(checkpoint),
        "row_count": int(len(fused_rows)),
        "outputs": {
            "match_table_csv": str(output / "match_table.csv"),
            "match_table_jsonl": str(output / "match_table.jsonl"),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def apply_rgb_patch_measurements_to_match_table(
    *,
    match_table_csv: Path,
    render_cache_manifest_csv: Path,
    image_root: Path,
    checkpoint: Path,
    output_dir: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    query_image_width: int | None = None,
    query_image_height: int | None = None,
    render_image_width: int | None = None,
    render_image_height: int | None = None,
    batch_size: int = 16,
    device: str = "cuda",
    base_dir: Path | None = None,
    max_rows: int | None = None,
    prediction_head: str = "center",
    prior_scale_key: str = "",
    data_parallel_device_ids: Sequence[int] = (),
) -> dict[str, Any]:
    base = Path.cwd() if base_dir is None else Path(base_dir)
    rows = _read_csv_rows(Path(match_table_csv), max_rows=max_rows)
    render_map = _render_cache_by_query(Path(render_cache_manifest_csv), base_dir=base)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    model = load_rgb_patch_measurement_branch(Path(checkpoint), device=torch_device)
    fused_rows, summary = apply_rgb_patch_measurements_to_rows(
        rows,
        image_root=Path(image_root),
        render_cache_by_query=render_map,
        model=model,
        image_width=image_width,
        image_height=image_height,
        query_image_width=query_image_width,
        query_image_height=query_image_height,
        render_image_width=render_image_width,
        render_image_height=render_image_height,
        batch_size=int(batch_size),
        device=torch_device,
        prediction_head=str(prediction_head),
        prior_scale_key=str(prior_scale_key),
        data_parallel_device_ids=[int(value) for value in data_parallel_device_ids],
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames_for_rows(fused_rows)
    _write_csv(output / "match_table.csv", fused_rows, fieldnames)
    _write_jsonl(output / "match_table.jsonl", fused_rows)
    summary = {
        **summary,
        "match_table_csv": str(match_table_csv),
        "render_cache_manifest_csv": str(render_cache_manifest_csv),
        "image_root": str(image_root),
        "checkpoint": str(checkpoint),
        "row_count": int(len(fused_rows)),
        "outputs": {
            "match_table_csv": str(output / "match_table.csv"),
            "match_table_jsonl": str(output / "match_table.jsonl"),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
