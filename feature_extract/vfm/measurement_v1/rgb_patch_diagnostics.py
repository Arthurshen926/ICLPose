from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    RGBPatchMeasurementBranch,
    continuous_offset_nll_with_dustbin,
)
from feature_extract.vfm.measurement_v1.candidate_measurement_schema import (
    CandidateIdentityKey,
    CandidateMeasurementCacheKey,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    _prior_scale_batch,
    _read_csv,
    _render_cache_by_query,
    _support_patch_source_audit,
    _stack_patch_batch,
)


DIAGNOSTIC_FIELDNAMES = [
    "row_index",
    "query_id",
    "support_image_id",
    "anchor_id",
    "track_id",
    "support_track_id",
    "policy_row_index",
    "candidate_identity_key",
    "candidate_measurement_cache_key",
    "measurement_checkpoint_sha256",
    "support_view_set_id",
    "source_query_row",
    "candidate_measurement_rank",
    "candidate_score_rank",
    "candidate_role",
    "candidate_prototype_id",
    "candidate_bank_row",
    "candidate_assignment_probability",
    "candidate_retrieval_similarity",
    "candidate_geometry_p01",
    "candidate_geometry_p02",
    "candidate_geometry_p05",
    "split",
    "support_view_rank",
    "support_view_probability",
    "track_length",
    "support_reprojection_error",
    "query_reprojection_error",
    "support_view_angle_deg",
    "assignment_score",
    "pose_selection_score",
    "geometry_p01",
    "geometry_p02",
    "geometry_p05",
    "target_gt_projected_x",
    "target_gt_projected_y",
    "target_gt_projected_residual_px",
    "target_gt_projection_in_front",
    "target_gt_projection_in_image",
    "target_geometry_correct_1px",
    "target_geometry_correct_2px",
    "target_geometry_correct_5px",
    "dustbin_supervision_weight",
    "geometry_supervision_weight",
    "actual_query_observation",
    "target_reason",
    "target_is_dustbin",
    "baseline_epe_px",
    "likelihood_epe_px",
    "mode_epe_px",
    "direct_epe_px",
    "gated_epe_px",
    "improved",
    "mode_improved",
    "direct_improved",
    "center_x",
    "center_y",
    "query_pred_x",
    "query_pred_y",
    "query_mode_x",
    "query_mode_y",
    "query_direct_x",
    "query_direct_y",
    "query_gated_x",
    "query_gated_y",
    "query_gt_x",
    "query_gt_y",
    "target_dx",
    "target_dy",
    "pred_dx",
    "pred_dy",
    "peak_dx",
    "peak_dy",
    "direct_dx",
    "direct_dy",
    "gated_dx",
    "gated_dy",
    "measurement_gate_probability",
    "measurement_geometry_probability",
    "dustbin_probability",
    "likelihood_entropy",
    "likelihood_normalized_entropy",
    "likelihood_peak_probability",
    "likelihood_peak_margin",
    "likelihood_covariance_trace_px2",
    "likelihood_covariance_max_sigma_px",
    "predicted_improvement_px",
    "gt_projected_likelihood_epe_px",
    "gt_projected_mode_epe_px",
    "gt_projected_direct_epe_px",
    "gt_projected_predicted_improvement_px",
    "visualization",
]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in DIAGNOSTIC_FIELDNAMES})


def _bool_text(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _measurement_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    baseline_values = np.asarray(
        [float(row["baseline_epe_px"]) for row in rows], dtype=np.float64
    )
    epe_values = np.asarray(
        [float(row["likelihood_epe_px"]) for row in rows], dtype=np.float64
    )
    mode_epe_values = np.asarray(
        [float(row["mode_epe_px"]) for row in rows], dtype=np.float64
    )
    direct_pairs = [
        (float(row["direct_epe_px"]), float(row["baseline_epe_px"]))
        for row in rows
        if str(row["direct_epe_px"]).strip()
    ]
    direct_epe_values = np.asarray([item[0] for item in direct_pairs], dtype=np.float64)
    direct_baseline_values = np.asarray([item[1] for item in direct_pairs], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "baseline_median_px": float(np.median(baseline_values)) if baseline_values.size else None,
        "likelihood_median_px": float(np.median(epe_values)) if epe_values.size else None,
        "likelihood_p90_px": float(np.percentile(epe_values, 90.0)) if epe_values.size else None,
        "mode_median_px": float(np.median(mode_epe_values)) if mode_epe_values.size else None,
        "mode_p90_px": float(np.percentile(mode_epe_values, 90.0)) if mode_epe_values.size else None,
        "direct_median_px": float(np.median(direct_epe_values)) if direct_epe_values.size else None,
        "direct_p90_px": float(np.percentile(direct_epe_values, 90.0)) if direct_epe_values.size else None,
        "improve_ratio": float(np.mean(epe_values < baseline_values)) if epe_values.size else None,
        "mode_improve_ratio": float(np.mean(mode_epe_values < baseline_values)) if mode_epe_values.size else None,
        "direct_improve_ratio": (
            float(np.mean(direct_epe_values < direct_baseline_values))
            if direct_epe_values.size
            else None
        ),
        "recall_0p5px": float(np.mean(epe_values <= 0.5)) if epe_values.size else None,
        "recall_1px": float(np.mean(epe_values <= 1.0)) if epe_values.size else None,
        "mode_recall_0p5px": float(np.mean(mode_epe_values <= 0.5)) if mode_epe_values.size else None,
        "mode_recall_1px": float(np.mean(mode_epe_values <= 1.0)) if mode_epe_values.size else None,
    }


def _xy_measurement_metrics(
    predicted_xy: Sequence[Sequence[float]],
    target_xy: Sequence[Sequence[float]],
    baseline_epe: Sequence[float],
) -> dict[str, object]:
    predicted = np.asarray(predicted_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    baseline = np.asarray(baseline_epe, dtype=np.float64).reshape(-1)
    epe = np.linalg.norm(predicted - target, axis=1)
    return {
        "count": int(len(epe)),
        "epe_median_px": float(np.median(epe)) if epe.size else None,
        "epe_p90_px": float(np.percentile(epe, 90.0)) if epe.size else None,
        "improve_ratio": float(np.mean(epe < baseline)) if epe.size else None,
        "worsen_ratio": float(np.mean(epe > baseline)) if epe.size else None,
        "recall_0p5px": float(np.mean(epe <= 0.5)) if epe.size else None,
        "recall_1px": float(np.mean(epe <= 1.0)) if epe.size else None,
    }


def _group_measurement_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    target_x_key: str = "query_gt_x",
    target_y_key: str = "query_gt_y",
    baseline_key: str = "baseline_epe_px",
    require_non_dustbin: bool = True,
) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if require_non_dustbin and _bool_text(row.get("target_is_dustbin", "")):
            continue
        if not str(row.get(target_x_key, "")).strip() or not str(
            row.get(target_y_key, "")
        ).strip():
            continue
        if not str(row.get(baseline_key, "")).strip():
            continue
        candidate_identity = str(row.get("candidate_identity_key", "")).strip()
        policy_row = str(row.get("policy_row_index", "")).strip()
        key = (
            candidate_identity
            if candidate_identity
            else policy_row if policy_row else f"row:{row.get('row_index', '')}"
        )
        groups.setdefault(key, []).append(row)

    target_xy: list[list[float]] = []
    baseline: list[float] = []
    learned_top1_xy: list[list[float]] = []
    posterior_mean_xy: list[list[float]] = []
    oracle_best_xy: list[list[float]] = []
    view_counts: list[int] = []
    for group_rows in groups.values():
        first = group_rows[0]
        target = np.asarray(
            [float(first[target_x_key]), float(first[target_y_key])],
            dtype=np.float64,
        )
        predictions = np.asarray(
            [
                [float(row["query_pred_x"]), float(row["query_pred_y"])]
                for row in group_rows
            ],
            dtype=np.float64,
        )
        support_probabilities = np.asarray(
            [
                float(str(row.get("support_view_probability", "1")).strip() or 1.0)
                for row in group_rows
            ],
            dtype=np.float64,
        )
        support_probabilities = np.clip(support_probabilities, 0.0, None)
        if float(np.sum(support_probabilities)) <= 0.0:
            support_probabilities = np.ones_like(support_probabilities)
        support_probabilities /= float(np.sum(support_probabilities))
        top1 = int(np.argmax(support_probabilities))
        oracle = int(np.argmin(np.linalg.norm(predictions - target[None, :], axis=1)))
        target_xy.append(target.tolist())
        baseline.append(float(first[baseline_key]))
        learned_top1_xy.append(predictions[top1].tolist())
        posterior_mean_xy.append(
            np.sum(predictions * support_probabilities[:, None], axis=0).tolist()
        )
        oracle_best_xy.append(predictions[oracle].tolist())
        view_counts.append(int(len(group_rows)))
    return {
        "policy_group_count": int(len(groups)),
        "mean_support_views": float(np.mean(view_counts)) if view_counts else None,
        "learned_top1": _xy_measurement_metrics(
            learned_top1_xy, target_xy, baseline
        ),
        "posterior_mean": _xy_measurement_metrics(
            posterior_mean_xy, target_xy, baseline
        ),
        "oracle_best_view_DIAGNOSTIC_ONLY": _xy_measurement_metrics(
            oracle_best_xy, target_xy, baseline
        ),
    }


def _candidate_geometry_evidence_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Audit whether RGB validity can rescue a non-rank1 candidate.

    This is target-only evaluation. Candidate and support-view selection remain
    frozen and pose-free; the GT geometry flags are never read by the scorer.
    """

    candidate_groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        identity = str(row.get("candidate_identity_key", "")).strip()
        if identity:
            candidate_groups.setdefault(identity, []).append(row)
    if not candidate_groups:
        return None

    candidates: list[dict[str, object]] = []
    for identity, group_rows in candidate_groups.items():
        first = group_rows[0]
        probabilities = np.asarray(
            [1.0 - float(row["dustbin_probability"]) for row in group_rows],
            dtype=np.float64,
        )
        weights = np.asarray(
            [
                float(str(row.get("support_view_probability", "")).strip() or 1.0)
                for row in group_rows
            ],
            dtype=np.float64,
        )
        weights = np.clip(weights, 0.0, None)
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones_like(weights)
        weights /= float(np.sum(weights))
        geometry_values = [
            str(row.get("measurement_geometry_probability", "")).strip()
            for row in group_rows
        ]
        geometry_probabilities = (
            None
            if not geometry_values or any(not value for value in geometry_values)
            else np.asarray([float(value) for value in geometry_values], dtype=np.float64)
        )
        candidate = {
            "identity": identity,
            "token": str(first.get("source_query_row", "")).strip(),
            "measurement_rank": int(first.get("candidate_measurement_rank", 0)),
            "correct_2px": _bool_text(first.get("target_geometry_correct_2px", "")),
            "correct_5px": _bool_text(first.get("target_geometry_correct_5px", "")),
            "validity_max": float(np.max(probabilities)),
            "validity_mean": float(np.mean(probabilities)),
            "validity_posterior": float(np.sum(probabilities * weights)),
            "support_view_count": int(len(group_rows)),
        }
        if geometry_probabilities is not None:
            candidate.update(
                {
                    "geometry_head_max": float(np.max(geometry_probabilities)),
                    "geometry_head_mean": float(np.mean(geometry_probabilities)),
                    "geometry_head_posterior": float(
                        np.sum(geometry_probabilities * weights)
                    ),
                }
            )
        candidates.append(
            candidate
        )
    if any(not str(candidate["token"]) for candidate in candidates):
        raise ValueError("candidate diagnostic row is missing source_query_row")
    token_groups: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        token_groups.setdefault(str(candidate["token"]), []).append(candidate)
    comparable_token_groups: dict[str, list[dict[str, object]]] = {}
    missing_frozen_candidate_count = 0
    duplicate_frozen_candidate_count = 0
    for token, group in token_groups.items():
        frozen_count = sum(
            int(candidate["measurement_rank"]) == 1 for candidate in group
        )
        if frozen_count == 1:
            comparable_token_groups[token] = group
        elif frozen_count == 0:
            missing_frozen_candidate_count += 1
        else:
            duplicate_frozen_candidate_count += 1

    report: dict[str, object] = {
        "candidate_count": int(len(candidates)),
        "token_count": int(len(token_groups)),
        "comparable_token_count": int(len(comparable_token_groups)),
        "missing_frozen_candidate_token_count": int(missing_frozen_candidate_count),
        "duplicate_frozen_candidate_token_count": int(
            duplicate_frozen_candidate_count
        ),
        "mean_support_views": float(
            np.mean([int(candidate["support_view_count"]) for candidate in candidates])
        ),
    }
    for threshold in (2, 5):
        label_key = f"correct_{threshold}px"
        labels = np.asarray([bool(candidate[label_key]) for candidate in candidates])
        threshold_report: dict[str, object] = {}
        aggregations = ["validity_max", "validity_mean", "validity_posterior"]
        if all("geometry_head_max" in candidate for candidate in candidates):
            aggregations.extend(
                ["geometry_head_max", "geometry_head_mean", "geometry_head_posterior"]
            )
        for aggregation in aggregations:
            probabilities = np.asarray(
                [float(candidate[aggregation]) for candidate in candidates],
                dtype=np.float64,
            )
            row_metrics = confidence_metrics(labels, probabilities)
            selected_correct: list[bool] = []
            chosen_correct: list[bool] = []
            oracle_correct: list[bool] = []
            changed: list[bool] = []
            for group in comparable_token_groups.values():
                baseline_candidates = [
                    candidate
                    for candidate in group
                    if int(candidate["measurement_rank"]) == 1
                ]
                baseline = baseline_candidates[0]
                chosen = max(
                    group,
                    key=lambda candidate: (
                        float(candidate[aggregation]),
                        -int(candidate["measurement_rank"]),
                    ),
                )
                selected_correct.append(bool(baseline[label_key]))
                chosen_correct.append(bool(chosen[label_key]))
                oracle_correct.append(any(bool(candidate[label_key]) for candidate in group))
                changed.append(str(chosen["identity"]) != str(baseline["identity"]))
            selected_array = np.asarray(selected_correct, dtype=bool)
            chosen_array = np.asarray(chosen_correct, dtype=bool)
            oracle_array = np.asarray(oracle_correct, dtype=bool)
            changed_array = np.asarray(changed, dtype=bool)
            rescue_eligible = (~selected_array) & oracle_array
            rescued = (~selected_array) & chosen_array
            harmed = selected_array & (~chosen_array)
            row_metrics.update(
                {
                    "frozen_selected_correct_rate": (
                        None if not len(selected_array) else float(np.mean(selected_array))
                    ),
                    "rgb_selected_correct_rate": (
                        None if not len(chosen_array) else float(np.mean(chosen_array))
                    ),
                    "oracle_top_m_correct_rate": (
                        None if not len(oracle_array) else float(np.mean(oracle_array))
                    ),
                    "candidate_change_rate": (
                        None if not len(changed_array) else float(np.mean(changed_array))
                    ),
                    "rescue_eligible_count": int(np.sum(rescue_eligible)),
                    "rescued_count": int(np.sum(rescued)),
                    "rescue_recall": (
                        None
                        if not np.any(rescue_eligible)
                        else float(np.sum(rescued) / np.sum(rescue_eligible))
                    ),
                    "harmed_count": int(np.sum(harmed)),
                    "net_correct_change": int(np.sum(chosen_array) - np.sum(selected_array)),
                }
            )
            threshold_report[aggregation] = row_metrics
        report[f"geometry_correct_{threshold}px"] = threshold_report
    return report


def _residual_update_metrics(
    baseline: np.ndarray, updated: np.ndarray
) -> dict[str, object]:
    before = np.asarray(baseline, dtype=np.float64).reshape(-1)
    after = np.asarray(updated, dtype=np.float64).reshape(-1)
    return {
        "count": int(len(before)),
        "baseline_median_px": float(np.median(before)) if before.size else None,
        "updated_median_px": float(np.median(after)) if after.size else None,
        "updated_p90_px": float(np.percentile(after, 90.0)) if after.size else None,
        "improve_ratio": float(np.mean(after < before)) if after.size else None,
        "worsen_ratio": float(np.mean(after > before)) if after.size else None,
        "updated_recall_0p5px": float(np.mean(after <= 0.5)) if after.size else None,
        "updated_recall_1px": float(np.mean(after <= 1.0)) if after.size else None,
        "updated_recall_2px": float(np.mean(after <= 2.0)) if after.size else None,
        "updated_recall_5px": float(np.mean(after <= 5.0)) if after.size else None,
    }


def _projected_geometry_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    usable = [
        row
        for row in rows
        if str(row.get("target_gt_projected_residual_px", "")).strip()
        and str(row.get("gt_projected_likelihood_epe_px", "")).strip()
    ]
    baseline = np.asarray(
        [float(row["target_gt_projected_residual_px"]) for row in usable],
        dtype=np.float64,
    )
    updated = np.asarray(
        [float(row["gt_projected_likelihood_epe_px"]) for row in usable],
        dtype=np.float64,
    )
    bins = {
        "center_le_1px": baseline <= 1.0,
        "center_1_to_2px": (baseline > 1.0) & (baseline <= 2.0),
        "center_2_to_5px": (baseline > 2.0) & (baseline <= 5.0),
        "center_gt_5px": baseline > 5.0,
    }
    return {
        "scope": "GT_pose_projection_target_only_including_unobserved_tracks",
        "all": _residual_update_metrics(baseline, updated),
        "by_center_residual": {
            name: _residual_update_metrics(baseline[mask], updated[mask])
            for name, mask in bins.items()
        },
    }


def _patch_to_uint8(patch_chw: torch.Tensor) -> np.ndarray:
    arr = patch_chw.detach().cpu().float().clamp(0.0, 1.0).numpy()
    arr = np.moveaxis(arr[:3], 0, -1)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _heatmap_to_uint8(probs: torch.Tensor) -> np.ndarray:
    values = probs.detach().cpu().float().numpy()
    values = values - float(values.min(initial=0.0))
    denom = max(float(values.max(initial=0.0)), 1e-12)
    gray = (values / denom * 255.0 + 0.5).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _save_visualization(
    *,
    path: Path,
    query_patch: torch.Tensor,
    render_patch: torch.Tensor,
    spatial_probs: torch.Tensor,
) -> None:
    query_img = Image.fromarray(_patch_to_uint8(query_patch))
    render_img = Image.fromarray(_patch_to_uint8(render_patch))
    side = int(round(float(spatial_probs.numel()) ** 0.5))
    heatmap = Image.fromarray(_heatmap_to_uint8(spatial_probs.reshape(side, side))).resize(query_img.size, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (query_img.width * 3, query_img.height))
    canvas.paste(query_img, (0, 0))
    canvas.paste(render_img, (query_img.width, 0))
    canvas.paste(heatmap, (query_img.width * 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _load_model(checkpoint: Path, *, device: torch.device) -> RGBPatchMeasurementBranch:
    payload = torch.load(Path(checkpoint), map_location=device)
    config = dict(payload.get("config", {}) if isinstance(payload, dict) else {})
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
    allowed_missing.update(key for key in incompatible.missing_keys if str(key).startswith("geometry_head."))
    missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint is incompatible with RGBPatchMeasurementBranch: "
            f"missing={missing}, unexpected={list(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model


def export_rgb_patch_diagnostics(
    *,
    rows_csv: Path,
    render_cache_manifest_csv: Path | None,
    image_root: Path,
    checkpoint: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    query_source: str = "real",
    support_patch_warp: str = "none",
    max_rows: int | None = None,
    visualize_limit: int = 16,
    batch_size: int = 1,
    cache_images_on_device: bool = False,
    image_cache_max_gb: float | None = None,
    prior_scale_key: str = "",
    target_dustbin_filter: str = "all",
    export_full_likelihood: bool = False,
    device: str = "cuda",
    base_dir: Path | None = None,
) -> dict[str, Any]:
    base = Path.cwd() if base_dir is None else Path(base_dir)
    raw_rows = _read_csv(Path(rows_csv), max_rows=max_rows)
    dustbin_filter = str(target_dustbin_filter).strip().lower() or "all"
    if dustbin_filter not in {"all", "valid", "dustbin"}:
        raise ValueError("target_dustbin_filter must be one of: all, valid, dustbin")
    indexed_rows = [
        (row_index, row)
        for row_index, row in enumerate(raw_rows)
        if dustbin_filter == "all"
        or (dustbin_filter == "dustbin")
        == _bool_text(row.get("target_is_dustbin", ""))
    ]
    rows = [row for _row_index, row in indexed_rows]
    if not rows:
        raise ValueError("target_dustbin_filter removed all diagnostic rows")
    render_map = _render_cache_by_query(None if render_cache_manifest_csv is None else Path(render_cache_manifest_csv), base_dir=base)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    image_cache_device = torch_device if bool(cache_images_on_device) else None
    model = _load_model(Path(checkpoint), device=torch_device)
    checkpoint_sha256 = file_sha256_short(Path(checkpoint))
    cache_max_bytes = (
        None
        if image_cache_max_gb is None or float(image_cache_max_gb) <= 0.0
        else int(float(image_cache_max_gb) * (1024**3))
    )
    image_cache = TensorImageLRUCache(max_bytes=cache_max_bytes)
    query_cache = image_cache
    render_cache = image_cache
    output = Path(output_dir)
    diagnostic_rows: list[dict[str, object]] = []
    likelihood_log_prob_batches: list[np.ndarray] = []
    likelihood_target_batches: list[np.ndarray] = []
    likelihood_offsets_xy: np.ndarray | None = None
    visual_candidates: list[tuple[float, int, Path, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    batch = max(1, int(batch_size))
    with torch.no_grad():
        for batch_start in range(0, len(rows), batch):
            batch_rows = rows[batch_start : batch_start + batch]
            batch_source_indices = [
                row_index
                for row_index, _row in indexed_rows[
                    batch_start : batch_start + batch
                ]
            ]
            query_patch, render_patch, target, baseline, _target_is_dustbin = _stack_patch_batch(
                batch_rows,
                image_root=Path(image_root),
                render_cache_by_query=render_map,
                image_width=int(image_width),
                image_height=int(image_height),
                crop_radius_px=model.crop_radius_px,
                step_px=model.step_px,
                query_cache=query_cache,
                render_cache=render_cache,
                query_source=str(query_source),
                render_patch_augmentation="none",
                support_patch_warp=str(support_patch_warp),
                image_cache_device=image_cache_device,
            )
            prior_scale = _prior_scale_batch(batch_rows, prior_scale_key=str(prior_scale_key))
            pred0 = model.forward_from_patches(
                query_patch.to(torch_device),
                render_patch.to(torch_device),
                prior_scale_px=None if prior_scale is None else prior_scale.to(torch_device),
            )
            _loss, likelihood = continuous_offset_nll_with_dustbin(
                pred0.logits,
                pred0.offsets_xy,
                target.to(torch_device),
                dustbin_logit=pred0.dustbin_logit,
                search_radius_px=model.search_radius_px,
                target_is_dustbin=None if _target_is_dustbin is None else _target_is_dustbin.to(torch_device),
            )
            spatial_probs_batch = torch.exp(likelihood.local_log_probs).detach().cpu()
            if bool(export_full_likelihood):
                batch_offsets = likelihood.offsets_xy.detach().cpu().numpy().astype(
                    np.float32, copy=False
                )
                if likelihood_offsets_xy is None:
                    likelihood_offsets_xy = np.array(batch_offsets, copy=True)
                elif not np.array_equal(likelihood_offsets_xy, batch_offsets):
                    raise RuntimeError("measurement offset support changed between batches")
                likelihood_log_prob_batches.append(
                    likelihood.local_log_probs.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                likelihood_target_batches.append(
                    target.detach().cpu().numpy().astype(np.float32)
                )
            mean_xy_batch = likelihood.mean_offset_xy.detach().cpu()
            epe_batch = likelihood.epe_px.detach().cpu()
            dustbin_batch = likelihood.dustbin_probability.detach().cpu()
            covariance_batch = likelihood.cov_2x2.detach().cpu()
            entropy_batch = -torch.sum(
                spatial_probs_batch
                * torch.log(spatial_probs_batch.clamp_min(1e-12)),
                dim=1,
            )
            normalized_entropy_batch = entropy_batch / max(
                float(np.log(int(spatial_probs_batch.shape[1]))), 1e-12
            )
            top_probabilities = torch.topk(
                spatial_probs_batch,
                k=min(2, int(spatial_probs_batch.shape[1])),
                dim=1,
            ).values
            peak_probability_batch = top_probabilities[:, 0]
            peak_margin_batch = (
                top_probabilities[:, 0] - top_probabilities[:, 1]
                if int(top_probabilities.shape[1]) > 1
                else top_probabilities[:, 0]
            )
            covariance_eigenvalues = torch.linalg.eigvalsh(covariance_batch)
            target_cpu = target.detach().cpu()
            direct_mean_batch = None if pred0.direct_mean_offset_xy is None else pred0.direct_mean_offset_xy.detach().cpu()
            gated_mean_batch = None if pred0.gated_mean_offset_xy is None else pred0.gated_mean_offset_xy.detach().cpu()
            gate_probability_batch = None if pred0.gate_probability is None else pred0.gate_probability.detach().cpu()
            geometry_probability_batch = (
                None
                if pred0.geometry_probability is None
                else pred0.geometry_probability.detach().cpu()
            )
            for local_index, row in enumerate(batch_rows):
                row_index = int(batch_source_indices[local_index])
                spatial_probs = spatial_probs_batch[local_index]
                peak_idx = int(torch.argmax(spatial_probs).item())
                peak_xy = likelihood.offsets_xy[peak_idx].detach().cpu()
                mean_xy = mean_xy_batch[local_index]
                epe = float(epe_batch[local_index].item())
                mode_epe = float(torch.linalg.norm(peak_xy - target_cpu[local_index]).item())
                direct_xy = None if direct_mean_batch is None else direct_mean_batch[local_index]
                direct_epe = None if direct_xy is None else float(torch.linalg.norm(direct_xy - target_cpu[local_index]).item())
                gated_xy = None if gated_mean_batch is None else gated_mean_batch[local_index]
                gated_epe = None if gated_xy is None else float(torch.linalg.norm(gated_xy - target_cpu[local_index]).item())
                baseline_value = float(baseline[local_index].item())
                center_x = float(str(row.get("center_x", "0")).strip())
                center_y = float(str(row.get("center_y", "0")).strip())
                vis_path = ""
                if int(visualize_limit) > 0:
                    candidate_path = output / "visualizations" / f"rank_pending_row{row_index:06d}.png"
                    visual_candidates.append((epe, row_index, candidate_path, query_patch[local_index], render_patch[local_index], spatial_probs))
                candidate_identity_key = str(
                    row.get("candidate_identity_key", "")
                ).strip()
                candidate_cache_key = ""
                if candidate_identity_key:
                    identity = CandidateIdentityKey(
                        query_id=str(row.get("query_id", "")),
                        source_query_row=int(row["source_query_row"]),
                        track_id=int(row["track_id"]),
                        prototype_id=int(row["candidate_prototype_id"]),
                        support_view_set_id=str(row["support_view_set_id"]),
                    )
                    if identity.digest != candidate_identity_key:
                        raise ValueError("candidate identity digest changed before RGB inference")
                    candidate_cache_key = CandidateMeasurementCacheKey(
                        identity=identity,
                        support_image_id=str(row.get("support_image_id", "")),
                        measurement_checkpoint_sha256=checkpoint_sha256,
                    ).digest
                diagnostic_rows.append(
                    {
                        "row_index": int(row_index),
                        "query_id": str(row.get("query_id", "")),
                        "support_image_id": str(row.get("support_image_id", "")),
                        "anchor_id": str(row.get("anchor_id", "")),
                        "track_id": str(row.get("track_id", "")),
                        "support_track_id": str(row.get("support_track_id", "")),
                        "policy_row_index": str(row.get("policy_row_index", "")),
                        "candidate_identity_key": candidate_identity_key,
                        "candidate_measurement_cache_key": candidate_cache_key,
                        "measurement_checkpoint_sha256": checkpoint_sha256,
                        "support_view_set_id": str(row.get("support_view_set_id", "")),
                        "source_query_row": str(row.get("source_query_row", "")),
                        "candidate_measurement_rank": str(
                            row.get("candidate_measurement_rank", "")
                        ),
                        "candidate_score_rank": str(row.get("candidate_score_rank", "")),
                        "candidate_role": str(row.get("candidate_role", "")),
                        "candidate_prototype_id": str(
                            row.get("candidate_prototype_id", "")
                        ),
                        "candidate_bank_row": str(row.get("candidate_bank_row", "")),
                        "candidate_assignment_probability": str(
                            row.get("candidate_assignment_probability", "")
                        ),
                        "candidate_retrieval_similarity": str(
                            row.get("candidate_retrieval_similarity", "")
                        ),
                        "candidate_geometry_p01": str(
                            row.get("candidate_geometry_p01", "")
                        ),
                        "candidate_geometry_p02": str(
                            row.get("candidate_geometry_p02", "")
                        ),
                        "candidate_geometry_p05": str(
                            row.get("candidate_geometry_p05", "")
                        ),
                        "split": str(row.get("split", "")),
                        "support_view_rank": str(row.get("support_view_rank", "")),
                        "support_view_probability": str(row.get("support_view_probability", "")),
                        "track_length": str(row.get("track_length", "")),
                        "support_reprojection_error": str(
                            row.get("support_reprojection_error", "")
                        ),
                        "query_reprojection_error": str(
                            row.get("query_reprojection_error", "")
                        ),
                        "support_view_angle_deg": str(
                            row.get("support_view_angle_deg", "")
                        ),
                        "assignment_score": str(row.get("assignment_score", "")),
                        "pose_selection_score": str(row.get("pose_selection_score", "")),
                        "geometry_p01": str(row.get("geometry_p01", "")),
                        "geometry_p02": str(row.get("geometry_p02", "")),
                        "geometry_p05": str(row.get("geometry_p05", "")),
                        "target_gt_projected_x": str(
                            row.get("target_gt_projected_x", "")
                        ),
                        "target_gt_projected_y": str(
                            row.get("target_gt_projected_y", "")
                        ),
                        "target_gt_projected_residual_px": str(
                            row.get("target_gt_projected_residual_px", "")
                        ),
                        "target_gt_projection_in_front": str(
                            row.get("target_gt_projection_in_front", "")
                        ),
                        "target_gt_projection_in_image": str(
                            row.get("target_gt_projection_in_image", "")
                        ),
                        "target_geometry_correct_1px": str(
                            row.get("target_geometry_correct_1px", "")
                        ),
                        "target_geometry_correct_2px": str(
                            row.get("target_geometry_correct_2px", "")
                        ),
                        "target_geometry_correct_5px": str(
                            row.get("target_geometry_correct_5px", "")
                        ),
                        "dustbin_supervision_weight": str(
                            row.get("dustbin_supervision_weight", "")
                        ),
                        "geometry_supervision_weight": str(
                            row.get("geometry_supervision_weight", "")
                        ),
                        "actual_query_observation": str(
                            row.get("actual_query_observation", "")
                        ),
                        "target_reason": str(row.get("target_reason", "")),
                        "target_is_dustbin": str(row.get("target_is_dustbin", "")),
                        "baseline_epe_px": baseline_value,
                        "likelihood_epe_px": epe,
                        "mode_epe_px": mode_epe,
                        "direct_epe_px": "" if direct_epe is None else direct_epe,
                        "gated_epe_px": "" if gated_epe is None else gated_epe,
                        "improved": bool(epe < baseline_value),
                        "mode_improved": bool(mode_epe < baseline_value),
                        "direct_improved": "" if direct_epe is None else bool(direct_epe < baseline_value),
                        "center_x": center_x,
                        "center_y": center_y,
                        "query_pred_x": center_x + float(mean_xy[0].item()),
                        "query_pred_y": center_y + float(mean_xy[1].item()),
                        "query_mode_x": center_x + float(peak_xy[0].item()),
                        "query_mode_y": center_y + float(peak_xy[1].item()),
                        "query_direct_x": "" if direct_xy is None else center_x + float(direct_xy[0].item()),
                        "query_direct_y": "" if direct_xy is None else center_y + float(direct_xy[1].item()),
                        "query_gated_x": "" if gated_xy is None else center_x + float(gated_xy[0].item()),
                        "query_gated_y": "" if gated_xy is None else center_y + float(gated_xy[1].item()),
                        "query_gt_x": str(row.get("query_gt_x", "")),
                        "query_gt_y": str(row.get("query_gt_y", "")),
                        "target_dx": float(target_cpu[local_index, 0].item()),
                        "target_dy": float(target_cpu[local_index, 1].item()),
                        "pred_dx": float(mean_xy[0].item()),
                        "pred_dy": float(mean_xy[1].item()),
                        "peak_dx": float(peak_xy[0].item()),
                        "peak_dy": float(peak_xy[1].item()),
                        "direct_dx": "" if direct_xy is None else float(direct_xy[0].item()),
                        "direct_dy": "" if direct_xy is None else float(direct_xy[1].item()),
                        "gated_dx": "" if gated_xy is None else float(gated_xy[0].item()),
                        "gated_dy": "" if gated_xy is None else float(gated_xy[1].item()),
                        "measurement_gate_probability": (
                            "" if gate_probability_batch is None else float(gate_probability_batch[local_index].item())
                        ),
                        "measurement_geometry_probability": (
                            ""
                            if geometry_probability_batch is None
                            else float(geometry_probability_batch[local_index].item())
                        ),
                        "dustbin_probability": float(dustbin_batch[local_index].item()),
                        "likelihood_entropy": float(entropy_batch[local_index].item()),
                        "likelihood_normalized_entropy": float(
                            normalized_entropy_batch[local_index].item()
                        ),
                        "likelihood_peak_probability": float(
                            peak_probability_batch[local_index].item()
                        ),
                        "likelihood_peak_margin": float(
                            peak_margin_batch[local_index].item()
                        ),
                        "likelihood_covariance_trace_px2": float(
                            torch.trace(covariance_batch[local_index]).item()
                        ),
                        "likelihood_covariance_max_sigma_px": float(
                            torch.sqrt(
                                covariance_eigenvalues[local_index, -1].clamp_min(0.0)
                            ).item()
                        ),
                        "predicted_improvement_px": float(baseline_value - epe),
                        "gt_projected_likelihood_epe_px": "",
                        "gt_projected_mode_epe_px": "",
                        "gt_projected_direct_epe_px": "",
                        "gt_projected_predicted_improvement_px": "",
                        "visualization": vis_path,
                    }
                )
                projected_x = str(row.get("target_gt_projected_x", "")).strip()
                projected_y = str(row.get("target_gt_projected_y", "")).strip()
                projected_baseline = str(
                    row.get("target_gt_projected_residual_px", "")
                ).strip()
                if projected_x and projected_y and projected_baseline:
                    projected_target = np.asarray(
                        [float(projected_x), float(projected_y)], dtype=np.float64
                    )
                    likelihood_xy = np.asarray(
                        [
                            center_x + float(mean_xy[0].item()),
                            center_y + float(mean_xy[1].item()),
                        ],
                        dtype=np.float64,
                    )
                    mode_xy = np.asarray(
                        [
                            center_x + float(peak_xy[0].item()),
                            center_y + float(peak_xy[1].item()),
                        ],
                        dtype=np.float64,
                    )
                    projected_likelihood_epe = float(
                        np.linalg.norm(likelihood_xy - projected_target)
                    )
                    projected_mode_epe = float(
                        np.linalg.norm(mode_xy - projected_target)
                    )
                    diagnostic_rows[-1][
                        "gt_projected_likelihood_epe_px"
                    ] = projected_likelihood_epe
                    diagnostic_rows[-1][
                        "gt_projected_mode_epe_px"
                    ] = projected_mode_epe
                    diagnostic_rows[-1][
                        "gt_projected_predicted_improvement_px"
                    ] = float(float(projected_baseline) - projected_likelihood_epe)
                    if direct_xy is not None:
                        direct_projected_xy = np.asarray(
                            [
                                center_x + float(direct_xy[0].item()),
                                center_y + float(direct_xy[1].item()),
                            ],
                            dtype=np.float64,
                        )
                        diagnostic_rows[-1][
                            "gt_projected_direct_epe_px"
                        ] = float(
                            np.linalg.norm(direct_projected_xy - projected_target)
                        )
    visual_candidates.sort(key=lambda item: item[0], reverse=True)
    row_to_vis: dict[int, str] = {}
    for rank, (_epe, row_index, _path, query_patch, render_patch, spatial_probs) in enumerate(visual_candidates[: int(visualize_limit)]):
        path = output / "visualizations" / f"worst_{rank:03d}_row{row_index:06d}.png"
        _save_visualization(path=path, query_patch=query_patch, render_patch=render_patch, spatial_probs=spatial_probs)
        row_to_vis[row_index] = str(path)
    for row in diagnostic_rows:
        row["visualization"] = row_to_vis.get(int(row["row_index"]), "")
    diagnostic_rows_path = output / "diagnostic_rows.csv"
    _write_csv(diagnostic_rows_path, diagnostic_rows)
    likelihood_path: Path | None = None
    if bool(export_full_likelihood):
        if likelihood_offsets_xy is None or not likelihood_log_prob_batches:
            raise RuntimeError("full likelihood export produced no spatial likelihoods")
        likelihood_path = output / "candidate_spatial_likelihood_v3.npz"
        full_log_probabilities = np.concatenate(likelihood_log_prob_batches, axis=0)
        target_offsets = np.concatenate(likelihood_target_batches, axis=0)
        if len(full_log_probabilities) != len(diagnostic_rows):
            raise RuntimeError("likelihood rows and diagnostic rows are not aligned")

        def text_array(key: str) -> np.ndarray:
            return np.asarray(
                [str(row.get(key, "")) for row in diagnostic_rows], dtype=np.str_
            )

        def int_array(key: str, *, default: int = -1) -> np.ndarray:
            return np.asarray(
                [int(str(row.get(key, default) or default)) for row in diagnostic_rows],
                dtype=np.int64,
            )

        def float_array(key: str, *, default: float = np.nan) -> np.ndarray:
            values = []
            for row in diagnostic_rows:
                text = str(row.get(key, "")).strip()
                values.append(float(text) if text else float(default))
            return np.asarray(values, dtype=np.float32)

        likelihood_metadata = {
            "format": "candidate_spatial_likelihood_v3",
            "format_version": 3,
            "rows_csv": str(rows_csv),
            "rows_csv_sha256": file_sha256_short(Path(rows_csv)),
            "measurement_checkpoint": str(checkpoint),
            "measurement_checkpoint_sha256": checkpoint_sha256,
            "query_source": str(query_source),
            "support_patch_warp": str(support_patch_warp),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "search_radius_px": float(model.search_radius_px),
            "context_radius_px": float(model.context_radius_px),
            "step_px": float(model.step_px),
            "spatial_probability_semantics": "conditional_on_non_dustbin",
            "dustbin_probability_semantics": "independent_binary_head",
            "support_views_unmarginalized": True,
            "ground_truth_arrays_target_only": [
                "target_offset_xy",
                "target_is_dustbin",
                "target_gt_projected_xy",
                "target_gt_projected_residual_px",
                "target_geometry_correct_1px",
                "target_geometry_correct_2px",
                "target_geometry_correct_5px",
            ],
            "pose_or_ground_truth_used_for_inference": False,
            "render": False,
        }
        np.savez_compressed(
            likelihood_path,
            source_row_indices=np.asarray(
                [int(row["row_index"]) for row in diagnostic_rows], dtype=np.int64
            ),
            query_ids=text_array("query_id"),
            source_query_rows=int_array("source_query_row"),
            candidate_identity_keys=text_array("candidate_identity_key"),
            candidate_measurement_cache_keys=text_array(
                "candidate_measurement_cache_key"
            ),
            candidate_measurement_ranks=int_array("candidate_measurement_rank"),
            candidate_track_ids=int_array("track_id"),
            candidate_prototype_ids=int_array("candidate_prototype_id"),
            support_image_ids=text_array("support_image_id"),
            support_view_ranks=int_array("support_view_rank"),
            support_view_probabilities=float_array("support_view_probability"),
            candidate_prior_probabilities=float_array(
                "candidate_assignment_probability"
            ),
            center_xy=np.stack(
                [float_array("center_x"), float_array("center_y")], axis=1
            ),
            offsets_xy=likelihood_offsets_xy,
            local_log_probabilities=full_log_probabilities,
            dustbin_probabilities=float_array("dustbin_probability"),
            likelihood_entropy=float_array("likelihood_entropy"),
            likelihood_covariance_trace_px2=float_array(
                "likelihood_covariance_trace_px2"
            ),
            measurement_geometry_probabilities=float_array(
                "measurement_geometry_probability"
            ),
            target_offset_xy=target_offsets,
            target_is_dustbin=np.asarray(
                [_bool_text(row["target_is_dustbin"]) for row in diagnostic_rows],
                dtype=bool,
            ),
            target_gt_projected_xy=np.stack(
                [
                    float_array("target_gt_projected_x"),
                    float_array("target_gt_projected_y"),
                ],
                axis=1,
            ),
            target_gt_projected_residual_px=float_array(
                "target_gt_projected_residual_px"
            ),
            target_geometry_correct_1px=np.asarray(
                [_bool_text(row["target_geometry_correct_1px"]) for row in diagnostic_rows],
                dtype=bool,
            ),
            target_geometry_correct_2px=np.asarray(
                [_bool_text(row["target_geometry_correct_2px"]) for row in diagnostic_rows],
                dtype=bool,
            ),
            target_geometry_correct_5px=np.asarray(
                [_bool_text(row["target_geometry_correct_5px"]) for row in diagnostic_rows],
                dtype=bool,
            ),
            metadata_json=np.asarray(
                json.dumps(likelihood_metadata, sort_keys=True), dtype=np.str_
            ),
        )
    valid_rows = [
        row for row in diagnostic_rows if not _bool_text(row["target_is_dustbin"])
    ]
    dustbin_labels = np.asarray(
        [_bool_text(row["target_is_dustbin"]) for row in diagnostic_rows], dtype=bool
    )
    dustbin_probabilities = np.asarray(
        [float(row["dustbin_probability"]) for row in diagnostic_rows], dtype=np.float64
    )
    dustbin_report = confidence_metrics(dustbin_labels, dustbin_probabilities)
    dustbin_report.update(
        {
            "dustbin_count": int(np.sum(dustbin_labels)),
            "valid_count": int(np.sum(~dustbin_labels)),
            "dustbin_probability_mean": (
                None
                if not np.any(dustbin_labels)
                else float(np.mean(dustbin_probabilities[dustbin_labels]))
            ),
            "valid_dustbin_probability_mean": (
                None
                if not np.any(~dustbin_labels)
                else float(np.mean(dustbin_probabilities[~dustbin_labels]))
            ),
        }
    )
    reliable_dustbin_rows = [
        row
        for row in diagnostic_rows
        if float(str(row.get("dustbin_supervision_weight", "0") or "0"))
        > 0.0
    ]
    reliable_dustbin_labels = np.asarray(
        [_bool_text(row["target_is_dustbin"]) for row in reliable_dustbin_rows],
        dtype=bool,
    )
    reliable_dustbin_probabilities = np.asarray(
        [float(row["dustbin_probability"]) for row in reliable_dustbin_rows],
        dtype=np.float64,
    )
    reliable_dustbin_report = confidence_metrics(
        reliable_dustbin_labels, reliable_dustbin_probabilities
    )
    reliable_dustbin_report.update(
        {
            "sample_count": int(len(reliable_dustbin_rows)),
            "dustbin_count": int(np.sum(reliable_dustbin_labels)),
            "valid_count": int(np.sum(~reliable_dustbin_labels)),
            "supervision": "dustbin_supervision_weight_gt_0",
        }
    )
    summary = {
        "stage": "measurement_v1_rgb_patch_diagnostics",
        "row_count": int(len(diagnostic_rows)),
        "raw_row_count": int(len(raw_rows)),
        "target_dustbin_filter": dustbin_filter,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "rows_csv": str(rows_csv),
        "rows_csv_sha256": file_sha256_short(Path(rows_csv)),
        "query_source": str(query_source),
        "support_patch_warp": str(support_patch_warp),
        "candidate_geometry_evidence": _candidate_geometry_evidence_metrics(
            diagnostic_rows
        ),
        "support_patch_source_audit": _support_patch_source_audit(rows, query_source=str(query_source)),
        "batch_size": int(batch),
        "cache_images_on_device": bool(cache_images_on_device),
        "image_cache_device": "" if image_cache_device is None else str(image_cache_device),
        "image_cache_max_gb": None if cache_max_bytes is None else float(cache_max_bytes / (1024**3)),
        "image_cache": image_cache.summary(),
        "prior_scale_key": str(prior_scale_key),
        "metrics_scope": "valid_non_dustbin_rows_only",
        "metrics": _measurement_metrics(valid_rows),
        "group_metrics": _group_measurement_metrics(valid_rows),
        "gt_projected_geometry_metrics_TARGET_ONLY": _projected_geometry_metrics(
            diagnostic_rows
        ),
        "gt_projected_group_metrics_TARGET_ONLY": _group_measurement_metrics(
            diagnostic_rows,
            target_x_key="target_gt_projected_x",
            target_y_key="target_gt_projected_y",
            baseline_key="target_gt_projected_residual_px",
            require_non_dustbin=False,
        ),
        "dustbin_metrics": dustbin_report,
        "reliable_dustbin_metrics": reliable_dustbin_report,
        "all_row_metrics_DIAGNOSTIC_ONLY": _measurement_metrics(diagnostic_rows),
        "outputs": {
            "diagnostic_rows": str(diagnostic_rows_path),
            "diagnostic_rows_sha256": file_sha256_short(
                diagnostic_rows_path
            ),
            "visualizations": str(output / "visualizations"),
            "candidate_spatial_likelihood": (
                None if likelihood_path is None else str(likelihood_path)
            ),
            "candidate_spatial_likelihood_sha256": (
                None
                if likelihood_path is None
                else file_sha256_short(likelihood_path)
            ),
            "summary": str(output / "summary.json"),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
