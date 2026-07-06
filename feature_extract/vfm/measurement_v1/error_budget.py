from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.measurement_v1.reference_oracle_decomposition import match_from_table_row
from feature_extract.vfm.query_to_3d_matching import estimate_pose_pnp_ransac, pnp_pose_error


@dataclass(frozen=True)
class ErrorBudgetConfig:
    noise_sigmas_px: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    quantization_strides_px: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0, 16.0)
    trials: int = 1
    pnp_reprojection_error_px: float = 12.0
    pnp_iterations: int = 2000
    pnp_min_inliers: int = 6
    rng_seed: int = 0


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _fmt_px(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return f"{text}px"


def _row_xy(row: Mapping[str, object], mode: str) -> np.ndarray | None:
    if mode == "cell_center":
        names = (("cell_center_x", "coarse_cell_center_x", "query_cell_center_x"), ("cell_center_y", "coarse_cell_center_y", "query_cell_center_y"))
    else:
        names = (("query_gt_x", "gt_query_x", "query_x"), ("query_gt_y", "gt_query_y", "query_y"))
    values = []
    for candidates in names:
        item = None
        for name in candidates:
            item = _float_or_none(row.get(name))
            if item is not None:
                break
        if item is None:
            return None
        values.append(float(item))
    return np.asarray(values, dtype=np.float64)


def _apply_noise_and_quantization(
    xy: np.ndarray,
    *,
    noise_sigma_px: float = 0.0,
    quantization_stride_px: float = 0.0,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(xy, dtype=np.float64).reshape(2).copy()
    if float(quantization_stride_px) > 0.0:
        stride = float(quantization_stride_px)
        out = np.round(out / stride) * stride
    if float(noise_sigma_px) > 0.0:
        out = out + rng.normal(0.0, float(noise_sigma_px), size=2)
    return out.astype(np.float64)


def _pose_metrics_for_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    xy_mode: str,
    noise_sigma_px: float,
    quantization_stride_px: float,
    gt_pose_w2c: np.ndarray,
    camera: ColmapCamera,
    config: ErrorBudgetConfig,
    rng: np.random.Generator,
) -> dict[str, object]:
    matches = []
    for row_index, row in enumerate(rows):
        base_xy = _row_xy(row, xy_mode)
        if base_xy is None:
            continue
        xy = _apply_noise_and_quantization(
            base_xy,
            noise_sigma_px=float(noise_sigma_px),
            quantization_stride_px=float(quantization_stride_px),
            rng=rng,
        )
        patched_row = dict(row)
        patched_row.setdefault("render_index", row_index)
        matches.append(match_from_table_row(patched_row, xy=xy, source=f"error_budget_{xy_mode}"))
    pnp = estimate_pose_pnp_ransac(
        matches,
        camera,
        reprojection_error_px=float(config.pnp_reprojection_error_px),
        iterations=int(config.pnp_iterations),
        min_inliers=int(config.pnp_min_inliers),
    )
    error = pnp_pose_error(pnp.pose_w2c if pnp.success else None, gt_pose_w2c)
    return {
        "match_count": int(len(matches)),
        "pnp_success": bool(pnp.success),
        "pnp_inlier_count": int(pnp.inlier_count),
        "translation_error_m": None if not np.isfinite(error.translation_m) else float(error.translation_m),
        "rotation_error_deg": None if not np.isfinite(error.rotation_deg) else float(error.rotation_deg),
    }


def _summarize_pose_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successful = [
        row
        for row in rows
        if bool(row.get("pnp_success")) and _float_or_none(row.get("translation_error_m")) is not None
    ]
    t = np.asarray([float(row["translation_error_m"]) for row in successful], dtype=np.float64)
    r = np.asarray([float(row["rotation_error_deg"]) for row in successful], dtype=np.float64)

    def _success_rate(max_translation_m: float, max_rotation_deg: float) -> float | None:
        if not rows:
            return None
        passed = 0
        for row in rows:
            translation = _float_or_none(row.get("translation_error_m"))
            rotation = _float_or_none(row.get("rotation_error_deg"))
            if (
                bool(row.get("pnp_success"))
                and translation is not None
                and rotation is not None
                and float(translation) <= float(max_translation_m)
                and float(rotation) <= float(max_rotation_deg)
            ):
                passed += 1
        return float(passed / len(rows))

    return {
        "query_trial_count": int(len(rows)),
        "solve_rate": float(len(successful) / len(rows)) if rows else None,
        "median_translation_error_m": float(np.median(t)) if t.size else None,
        "p90_translation_error_m": float(np.percentile(t, 90.0)) if t.size else None,
        "median_rotation_error_deg": float(np.median(r)) if r.size else None,
        "success_3cm_1deg": _success_rate(0.03, 1.0),
        "success_5cm_2deg": _success_rate(0.05, 2.0),
        "success_10cm_5deg": _success_rate(0.10, 5.0),
    }


def run_error_budget_for_rows(
    *,
    rows: Iterable[Mapping[str, object]],
    gt_pose_by_query: Mapping[str, np.ndarray],
    camera: ColmapCamera,
    config: ErrorBudgetConfig = ErrorBudgetConfig(),
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if query_id:
            grouped[query_id].append(row)
    variants: dict[str, list[dict[str, object]]] = defaultdict(list)
    for query_id, query_rows in sorted(grouped.items()):
        gt_pose = gt_pose_by_query.get(query_id)
        if gt_pose is None:
            continue
        for trial in range(max(int(config.trials), 1)):
            for sigma in config.noise_sigmas_px:
                rng = np.random.default_rng(int(config.rng_seed) + 1009 * trial + int(round(float(sigma) * 1000.0)))
                name = f"continuous_noise_{_fmt_px(float(sigma))}"
                variants[name].append(
                    {
                        "query_id": query_id,
                        "trial": int(trial),
                        **_pose_metrics_for_rows(
                            query_rows,
                            xy_mode="continuous",
                            noise_sigma_px=float(sigma),
                            quantization_stride_px=0.0,
                            gt_pose_w2c=np.asarray(gt_pose, dtype=np.float64),
                            camera=camera,
                            config=config,
                            rng=rng,
                        ),
                    }
                )
            for stride in config.quantization_strides_px:
                if float(stride) <= 0.0:
                    continue
                rng = np.random.default_rng(int(config.rng_seed) + 7919 * trial + int(round(float(stride) * 1000.0)))
                name = f"continuous_quant_stride{_fmt_px(float(stride))}"
                variants[name].append(
                    {
                        "query_id": query_id,
                        "trial": int(trial),
                        **_pose_metrics_for_rows(
                            query_rows,
                            xy_mode="continuous",
                            noise_sigma_px=0.0,
                            quantization_stride_px=float(stride),
                            gt_pose_w2c=np.asarray(gt_pose, dtype=np.float64),
                            camera=camera,
                            config=config,
                            rng=rng,
                        ),
                    }
                )
            rng = np.random.default_rng(int(config.rng_seed) + 3571 * trial)
            variants["cell_center_noise_0px"].append(
                {
                    "query_id": query_id,
                    "trial": int(trial),
                    **_pose_metrics_for_rows(
                        query_rows,
                        xy_mode="cell_center",
                        noise_sigma_px=0.0,
                        quantization_stride_px=0.0,
                        gt_pose_w2c=np.asarray(gt_pose, dtype=np.float64),
                        camera=camera,
                        config=config,
                        rng=rng,
                    ),
                }
            )
    return {
        "stage": "measurement_v1_error_budget",
        "query_count": int(len(grouped)),
        "variants": {name: _summarize_pose_rows(values) for name, values in sorted(variants.items())},
        "variant_rows": {name: values for name, values in sorted(variants.items())},
    }
