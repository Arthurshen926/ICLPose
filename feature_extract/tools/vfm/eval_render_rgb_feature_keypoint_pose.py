"""Evaluate sparse matching on RADIO features extracted from rendered RGB."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import (
    _extract_radio_feature_from_rgb,
    _depth_field_from_rgb_source,
    _load_or_extract_render_feature,
    _load_or_render_rgb_depth_cache,
    _render_rgb_and_depth,
    _render_token_cache_path,
    _resolve_render_size,
    _safe_image_stem,
    _select_records,
)
from feature_extract.tools.vfm.build_matcha_joint_cache import _load_or_extract_matcha_joint_feature
from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import (
    _draw_matches,
    _estimate_pose_with_optional_rescore,
    _geometry_row_fields,
    _infer_camera_model_dir,
    _load_camera_with_source,
    _load_query_feature,
    _make_keypoint_detector,
    _parse_default_camera,
    _pose_metrics,
    _project_query_feature,
    _read_rgb,
    _scale_camera,
    _summary,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.correspondence_confidence import (
    CalibratedLogisticConfidence,
    annotate_matches_with_calibrated_confidence,
)
from feature_extract.vfm.coarse_oracle_diagnostics import (
    CoarseOracleRankAccumulator,
    coarse_oracle_candidate_rows,
    coarse_oracle_rank_rows,
)
from feature_extract.vfm.coarse_candidate_ranking import annotate_keypoint_matches_with_coarse_candidate_ranker
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMRenderConfig, load_gaussian_rgb_source_from_ply
from feature_extract.vfm.matcha_coarse_to_fine import (
    apply_cell_reliability_prior_to_matches,
    apply_fine_logit_confidence_to_matches,
    apply_keypoint_cell_prior_to_matches,
    apply_pair_fine_logits_to_matches,
    expand_matches_with_render_local_offsets,
    feature_map_to_coarse_grid,
    matcha_coarse_to_fine_keypoint_matches,
    refine_render_matches_by_local_attention,
    rescore_keypoint_matches_by_feature_similarity,
    retain_topk_matches_per_query,
)
from feature_extract.vfm.matcha_coarse_fine_adapter import (
    load_matcha_coarse_fine_adapter,
    predict_matcha_pair_heads_for_matches,
    project_feature_map_with_matcha_adapter_full as _adapter_project_feature_map_full,
    project_feature_map_with_matcha_keypoint_logits as _adapter_project_keypoint_logits,
    project_feature_map_with_matcha_adapter,
)
from feature_extract.vfm.matcha_keypoint_distillation import AlikeKeypointExtractor
from feature_extract.vfm.matcha_joint_training import (
    load_matcha_joint_model,
    predict_matcha_joint_pair_heads_for_matches,
    project_feature_map_with_matcha_joint_model,
)
from feature_extract.vfm.matcha_joint_cache import _rgb_to_bchw_float
from feature_extract.vfm.matcha_light_fusion import maybe_fuse_feature_map
from feature_extract.vfm.matcha_qkv_attention import apply_qkv_attention_to_feature_maps, load_qkv_attention_checkpoint
from feature_extract.vfm.matcha_rgb_keypoint_detector import (
    MatchaRgbKeypointDetector,
    decode_keypoints_from_logits,
    keypoint_xy_to_feature_cell_indices,
)
from feature_extract.vfm.official_2dgs_renderer import load_official_2dgs_source_from_ply
from feature_extract.vfm.query_to_3d_matching import (
    camera_matrix_and_distortion,
    match_reprojection_errors,
    refit_pose_with_unique_query_inliers,
    reprojection_error_stats,
    soft_order_pnp_matches,
)
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.rendered_keypoint_matching import (
    backproject_depth_to_world,
    bilinear_sample_feature_map,
    dual_softmax_keypoint_matches,
    keypoint_feature_matches_to_pnp_matches,
    mutual_nn_keypoint_matches,
    refine_render_keypoint_matches_by_local_correlation,
)
from feature_extract.vfm.rendered_pose_scoring import (
    annotate_measurement_uncertainty,
    coverage_preserving_match_filter,
    score_pose_hypothesis,
)
from feature_extract.vfm.render_pose_scorer import (
    load_pose_scorer_model,
    pose_candidate_row_from_eval_candidate,
    score_pose_candidates_with_model,
)
from feature_extract.vfm.render_pose_protocol import (
    RenderPoseSelection,
    expand_render_pose_rotation_candidates,
    group_topk_reference_poses,
    group_top_reference_poses,
    parse_rotation_offset_deg,
    parse_rotation_search_offsets_deg,
    parse_world_offset,
    render_pose_error_fields,
    select_render_pose,
)
from feature_extract.vfm.render_pose_residual_solver import solve_render_pose_delta_from_matches
from feature_extract.vfm.tokens import TokenBankManifest


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _coerce_csv_value(value: str) -> object:
    item = str(value)
    if item == "":
        return None
    if item == "True":
        return True
    if item == "False":
        return False
    try:
        number = float(item)
    except ValueError:
        return item
    if not np.isfinite(number):
        return number
    if number.is_integer() and item.strip() == str(int(number)):
        return int(number)
    return number


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    value = Path(path)
    if not value.exists() or value.stat().st_size == 0:
        return []
    with value.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for row in reader:
            parsed: dict[str, object] = {}
            for key, cell in row.items():
                if key is None:
                    continue
                parsed[str(key)] = _coerce_csv_value("" if cell is None else str(cell))
            rows.append(parsed)
        return rows


def _existing_query_ids_from_rows(path: Path) -> set[str]:
    return {str(row["query_id"]) for row in _read_csv_rows(Path(path)) if row.get("query_id") is not None}


def _write_rebuilt_summary_from_rows(output_dir: Path, args: argparse.Namespace, *, elapsed_sec: float = 0.0) -> dict[str, object]:
    rows_path = Path(output_dir) / "rows.csv"
    render_cache_manifest_path = Path(output_dir) / "render_cache_manifest.csv"
    summary_path = Path(output_dir) / "summary.json"
    rows = _read_csv_rows(rows_path)
    render_cache_rows = _read_csv_rows(render_cache_manifest_path)
    summary = {
        "stage": "render_rgb_feature_keypoint_pose",
        "rebuilt_from_rows": True,
        "elapsed_sec": float(elapsed_sec),
        "inputs": {
            "query_manifest": str(args.query_manifest),
            "query_pose_file": str(args.query_pose_file),
            "image_root": str(args.image_root),
            "gaussian_rgb_ply": str(args.gaussian_rgb_ply),
            "matcha_adapter_checkpoint": str(args.matcha_adapter_checkpoint),
            "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
            "candidate_bank": str(args.candidate_bank),
        },
        "config": {
            "matcha_eval_preset": str(args.matcha_eval_preset),
            "match_mode": str(args.match_mode),
            "render_pose_mode": str(args.render_pose_mode),
            "render_pose_world_offset": str(args.render_pose_world_offset),
            "render_pose_rotation_offset_deg": str(args.render_pose_rotation_offset_deg),
            "render_pose_rotation_search_offsets_deg": str(args.render_pose_rotation_search_offsets_deg),
            "render_pose_rotation_search_axis": str(args.render_pose_rotation_search_axis),
            "stream_rows": bool(args.stream_rows),
            "resume_existing_rows": bool(args.resume_existing_rows),
        },
        "metrics": _summary(rows),
        "outputs": {
            "rows": str(rows_path),
            "summary": str(summary_path),
            "render_cache_manifest": str(render_cache_manifest_path),
        },
    }
    summary["metrics"]["render_cache_manifest_row_count"] = int(len(render_cache_rows))
    summary["metrics"].update(_render_lock_diagnostics_from_rows(rows))
    summary["metrics"].update(_fine_confidence_diagnostics_from_rows(rows))
    summary["metrics"].update(_residual_solver_diagnostics_from_rows(rows))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _matcha_query_token_cache_path(
    cache_dir: Path,
    image_id: str,
    *,
    render_width: int,
    render_height: int,
    layer_name: str,
) -> Path:
    generic = _render_token_cache_path(cache_dir, image_id, int(render_width), int(render_height))
    streaming_style = Path(cache_dir) / f"{_safe_image_stem(image_id)}_{int(render_width)}x{int(render_height)}_{str(layer_name)}.npz"
    if not generic.exists() and streaming_style.exists():
        return streaming_style
    return generic


class _StreamingCsvWriter:
    """Write large homogeneous CSV row streams without retaining them in memory."""

    def __init__(self, path: Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._writer: csv.DictWriter | None = None
        self._fieldnames: list[str] | None = None
        self.row_count = 0
        self._append = bool(append)
        if self._append and self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open(newline="") as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration:
                    header = []
                self.row_count = sum(1 for _row in reader)
            self._fieldnames = [str(item) for item in header]

    def writerows(self, rows: Sequence[dict[str, object]]) -> None:
        values = list(rows)
        if not values:
            return
        if self._writer is None:
            mode = "a" if self._append and self._fieldnames else "w"
            if self._fieldnames is None:
                self._fieldnames = sorted({key for row in values for key in row})
            self._handle = self.path.open(mode, newline="")
            self._writer = csv.DictWriter(self._handle, fieldnames=self._fieldnames)
            if mode == "w":
                self._writer.writeheader()
        assert self._writer is not None
        assert self._fieldnames is not None
        extra = sorted({key for row in values for key in row}.difference(self._fieldnames))
        if extra:
            raise ValueError(f"streaming CSV rows introduced new columns after header was written: {extra}")
        self._writer.writerows(values)
        self._handle.flush()
        self.row_count += len(values)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _cache_file_sha1(path: Path | None, *, chunk_size: int = 1024 * 1024) -> str | None:
    if path is None or not Path(path).exists():
        return None
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_numpy_sha1(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha1()
    digest.update(str(value.dtype).encode("utf8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _cache_pose_hash(pose_w2c: np.ndarray) -> str:
    return _cache_numpy_sha1(np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4))


def _render_canvas_camera_from_base(base_camera, *, canvas_width: int, canvas_height: int):
    """Create a wider render canvas camera without scaling focal length."""

    width = int(canvas_width)
    height = int(canvas_height)
    if width < int(base_camera.width) or height < int(base_camera.height):
        raise ValueError("render canvas must be at least as large as the base render camera")
    shift_x = 0.5 * float(width - int(base_camera.width))
    shift_y = 0.5 * float(height - int(base_camera.height))
    params = tuple(float(value) for value in base_camera.params)
    if base_camera.model_id == 1 and len(params) >= 4:
        fx, fy, cx, cy = params[:4]
        canvas_params = (fx, fy, cx + shift_x, cy + shift_y, *params[4:])
    elif base_camera.model_id in {0, 2, 8} and len(params) >= 3:
        focal, cx, cy = params[:3]
        canvas_params = (focal, cx + shift_x, cy + shift_y, *params[3:])
    else:
        raise ValueError(f"unsupported camera model id for render canvas: {base_camera.model_id}")
    from feature_extract.vfm.colmap_tracks import ColmapCamera

    return ColmapCamera(
        camera_id=base_camera.camera_id,
        model_id=base_camera.model_id,
        width=width,
        height=height,
        params=canvas_params,
    )


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        item = float(value)
    except (TypeError, ValueError):
        return None
    return item if np.isfinite(item) else None


def _match_table_rows_for_query(
    *,
    query_id: str,
    matches,
    gt_errors: np.ndarray,
    gt_stride_px: float,
    inlier_mask: np.ndarray | None,
    baseline_reproj_errors: np.ndarray | None,
    render_xy_by_match: dict[int, np.ndarray],
) -> list[dict[str, object]]:
    """Export match-level diagnostics for candidate-mined training."""

    errors = np.asarray(gt_errors, dtype=np.float64).reshape(-1)
    inliers = None if inlier_mask is None else np.asarray(inlier_mask, dtype=bool).reshape(-1)
    baseline_errors = (
        None if baseline_reproj_errors is None else np.asarray(baseline_reproj_errors, dtype=np.float64).reshape(-1)
    )
    stride = float(gt_stride_px) if np.isfinite(gt_stride_px) and float(gt_stride_px) > 0.0 else 16.0
    rows: list[dict[str, object]] = []
    for idx, match in enumerate(matches):
        qxy = np.asarray(match.xy, dtype=np.float64).reshape(2)
        xyz = np.asarray(match.xyz, dtype=np.float64).reshape(3)
        match_render_xy = getattr(match, "render_xy", None)
        render_xy = (
            np.asarray(match_render_xy, dtype=np.float64).reshape(2)
            if match_render_xy is not None
            else np.asarray(render_xy_by_match.get(int(match.track_id), np.full((2,), np.nan)), dtype=np.float64).reshape(2)
        )
        error = float(errors[idx]) if idx < errors.shape[0] and np.isfinite(errors[idx]) else None
        error_stride = None if error is None else float(error / stride)
        baseline_error = (
            None
            if baseline_errors is None or idx >= baseline_errors.shape[0] or not np.isfinite(baseline_errors[idx])
            else float(baseline_errors[idx])
        )
        patch_correct = False if error is None else bool(error <= stride)
        weak_positive = False if error is None else bool(stride < error <= (2.0 * stride))
        pnp_inlier = False if inliers is None or idx >= inliers.shape[0] else bool(inliers[idx])
        hard_negative = False if error is None else bool(error > (2.0 * stride) and pnp_inlier)
        rows.append(
            {
                "query_id": str(query_id),
                "match_index": int(idx),
                "query_index": int(match.token_index),
                "render_index": int(match.track_id),
                "query_x": float(qxy[0]),
                "query_y": float(qxy[1]),
                "render_x": _optional_float(render_xy[0]),
                "render_y": _optional_float(render_xy[1]),
                "xy": [float(qxy[0]), float(qxy[1])],
                "world_x": float(xyz[0]),
                "world_y": float(xyz[1]),
                "world_z": float(xyz[2]),
                "similarity": float(match.similarity),
                "similarity_margin": _optional_float(match.similarity_margin),
                "match_rank": int(idx),
                "token_match_rank": None if match.token_match_rank is None else int(match.token_match_rank),
                "base_render_index": None if match.base_render_index is None else int(match.base_render_index),
                "candidate_render_index": None
                if match.candidate_render_index is None
                else int(match.candidate_render_index),
                "candidate_id": None if match.candidate_id is None else int(match.candidate_id),
                "coarse_rank": None if match.coarse_rank is None else int(match.coarse_rank),
                "coarse_score": _optional_float(match.coarse_score),
                "coarse_score_gap": _optional_float(match.coarse_score_gap),
                "mutual_rank": None if match.mutual_rank is None else int(match.mutual_rank),
                "cell_delta_x": None if match.cell_delta_x is None else int(match.cell_delta_x),
                "cell_delta_y": None if match.cell_delta_y is None else int(match.cell_delta_y),
                "confidence": _optional_float(match.pnp_soft_score),
                "gt_reproj_error_px": error,
                "gt_reproj_error_stride": error_stride,
                "gt_correct_8px": False if error is None else bool(error <= 8.0),
                "gt_correct_16px": False if error is None else bool(error <= 16.0),
                "gt_correct_24px": False if error is None else bool(error <= 24.0),
                "patch_correct": patch_correct,
                "patch_positive_label": patch_correct,
                "strong_positive_label": patch_correct,
                "weak_positive_label": weak_positive,
                "hard_negative_label": hard_negative,
                "ignore_label": weak_positive,
                "pose_usable_label": patch_correct,
                "pnp_inlier": pnp_inlier,
                "baseline_reproj_residual_px": baseline_error,
                "patch_offset_norm_px": _optional_float(match.patch_offset_norm_px),
                "render_depth": _optional_float(match.render_depth),
                "render_alpha": _optional_float(match.render_alpha),
            }
        )
    return rows


def _project_feature_pair_for_render_rgb_eval(
    query_feature: np.ndarray,
    render_feature: np.ndarray,
    *,
    selector_checkpoint: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same optional selector to query and rendered-RGB RADIO maps."""

    if not selector_checkpoint:
        return np.asarray(query_feature, dtype=np.float32), np.asarray(render_feature, dtype=np.float32)
    return (
        _project_query_feature(query_feature, selector_checkpoint, device),
        _project_query_feature(render_feature, selector_checkpoint, device),
    )


def _project_feature_pair_with_matcha_adapter(
    query_feature: np.ndarray,
    render_feature: np.ndarray,
    *,
    adapter_model,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query_selected, query_offsets = project_feature_map_with_matcha_adapter(adapter_model, query_feature, device=str(device))
    render_selected, render_offsets = project_feature_map_with_matcha_adapter(adapter_model, render_feature, device=str(device))
    return query_selected, render_selected, query_offsets, render_offsets


def _project_feature_pair_with_matcha_adapter_full(
    query_feature: np.ndarray,
    render_feature: np.ndarray,
    *,
    adapter_model,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query_selected, query_offsets, query_detector = _adapter_project_feature_map_full(
        adapter_model,
        query_feature,
        device=str(device),
    )
    render_selected, render_offsets, render_detector = _adapter_project_feature_map_full(
        adapter_model,
        render_feature,
        device=str(device),
    )
    query_keypoint = _adapter_project_keypoint_logits(adapter_model, query_feature, device=str(device))
    render_keypoint = _adapter_project_keypoint_logits(adapter_model, render_feature, device=str(device))
    return query_selected, render_selected, query_offsets, render_offsets, query_detector, render_detector, query_keypoint, render_keypoint


def _probability_to_logit(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float32)
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values)).astype(np.float32, copy=False)


def _project_feature_pair_with_matcha_joint_model(
    query_feature: np.ndarray,
    render_feature: np.ndarray,
    *,
    joint_model,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query_selected, query_offsets, query_heatmap = project_feature_map_with_matcha_joint_model(
        joint_model,
        query_feature,
        device=str(device),
    )
    render_selected, render_offsets, render_heatmap = project_feature_map_with_matcha_joint_model(
        joint_model,
        render_feature,
        device=str(device),
    )
    return (
        query_selected,
        render_selected,
        query_offsets,
        render_offsets,
        _probability_to_logit(query_heatmap),
        _probability_to_logit(render_heatmap),
    )


def _iteration_pose_score_value(iteration: dict[str, object]) -> float:
    score = iteration.get("iteration_pose_score")
    value = getattr(score, "score", float("-inf"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _iteration_alignment_score_value(iteration: dict[str, object]) -> float:
    try:
        return float(iteration.get("iteration_alignment_score", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def _select_pose_update_iteration(
    iteration_results: Sequence[dict[str, object]],
    *,
    mode: str,
    score_margin: float,
) -> tuple[dict[str, object], str]:
    if not iteration_results:
        raise ValueError("iteration_results must not be empty")
    if str(mode) == "last":
        return iteration_results[-1], "last"
    value_fn = _iteration_alignment_score_value if str(mode).endswith("alignment") else _iteration_pose_score_value
    scored = [iteration for iteration in iteration_results if str(iteration.get("status", "")) == "ok" and np.isfinite(value_fn(iteration))]
    if not scored:
        return iteration_results[-1], "last:no_scored_pose"
    best = max(scored, key=value_fn)
    if str(mode) in {"best_score", "best_alignment"}:
        return best, str(mode)
    anchor = scored[0]
    if value_fn(best) >= value_fn(anchor) + float(score_margin):
        return best, f"{mode}:accepted"
    return anchor, f"{mode}:kept_initial"


def _pose_candidate_table_rows_for_query(
    *,
    query_id: str,
    render_pose_mode: str,
    candidates: Sequence[dict[str, object]],
    gt_pose_w2c: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate_rank, candidate in enumerate(candidates):
        pnp = candidate.get("pnp")
        render_pose = candidate.get("render_pose")
        pnp_pose = (
            getattr(pnp, "pose_w2c", None)
            if pnp is not None and bool(getattr(pnp, "success", False))
            else None
        )
        translation_error, rotation_error = _pose_metrics(pnp_pose, gt_pose_w2c)
        render_pose_w2c = getattr(render_pose, "pose_w2c", None)
        pnp_render_translation_delta, pnp_render_rotation_delta = _pose_metrics(pnp_pose, render_pose_w2c)
        pose_score = candidate.get("iteration_pose_score")
        feature_row = pose_candidate_row_from_eval_candidate(candidate)
        learned_pose_score = candidate.get("learned_pose_score")
        learned_pose_probability = candidate.get("learned_pose_probability")
        row = {
            "query_id": str(query_id),
            "render_pose_mode": str(render_pose_mode),
            "candidate_rank": int(candidate_rank),
            "initial_render_index": int(candidate.get("initial_render_index", candidate_rank) or 0),
            "initial_render_pose_label": str(candidate.get("initial_render_pose_label", "")),
            "initial_render_candidate_id": str(candidate.get("initial_render_candidate_id", "")),
            "initial_render_reference_image": str(candidate.get("initial_render_reference_image", "")),
            "candidate_pose_update_selection_label": str(candidate.get("candidate_pose_update_selection_label", "")),
            "render_pose_label": "" if render_pose is None else str(getattr(render_pose, "label", "")),
            "render_candidate_id": "" if render_pose is None else str(getattr(render_pose, "candidate_id", "")),
            "render_reference_image": "" if render_pose is None else str(getattr(render_pose, "reference_image", "")),
            "render_translation_error_m": None
            if render_pose is None
            else getattr(render_pose, "render_translation_error_m", None),
            "render_rotation_error_deg": None
            if render_pose is None
            else getattr(render_pose, "render_rotation_error_deg", None),
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
            "pnp_render_translation_delta_m": pnp_render_translation_delta,
            "pnp_render_rotation_delta_deg": pnp_render_rotation_delta,
            "pose_update_selected_score": _iteration_pose_score_value(candidate),
            "pose_update_selected_alignment_score": _iteration_alignment_score_value(candidate),
            "learned_pose_probability": None
            if learned_pose_probability is None
            else float(learned_pose_probability),
            "learned_pose_score": None if learned_pose_score is None else float(learned_pose_score),
            "pose_score_inlier_count": None if pose_score is None else int(getattr(pose_score, "inlier_count", 0)),
            "pose_score_weighted_residual": None
            if pose_score is None
            else float(getattr(pose_score, "weighted_residual", 0.0)),
            "pose_score_confidence_mean": None
            if pose_score is None
            else float(getattr(pose_score, "confidence_mean", 0.0)),
            "pose_score_coverage": None if pose_score is None else float(getattr(pose_score, "coverage", 0.0)),
            "pose_score_degeneracy_penalty": None
            if pose_score is None
            else float(getattr(pose_score, "degeneracy_penalty", 0.0)),
        }
        row.update(feature_row)
        rows.append(row)
    return rows


def _resize_hw(image: np.ndarray, *, height: int, width: int, interpolation: int) -> np.ndarray:
    import cv2

    return cv2.resize(np.asarray(image), (int(width), int(height)), interpolation=interpolation)


def _render_query_feature_alignment_score(
    query_feature: np.ndarray,
    render_feature: np.ndarray,
    *,
    render_alpha: np.ndarray | None,
    render_depth: np.ndarray | None,
    min_alpha: float = 0.1,
) -> float:
    """Score a render pose by same-pixel query/render feature agreement."""

    query = np.asarray(query_feature, dtype=np.float32)
    render = np.asarray(render_feature, dtype=np.float32)
    if query.ndim != 3 or render.ndim != 3 or query.shape[0] != render.shape[0]:
        return float("-inf")
    channels, query_h, query_w = int(query.shape[0]), int(query.shape[1]), int(query.shape[2])
    if render.shape[1:] != (query_h, query_w):
        resized = np.empty((channels, query_h, query_w), dtype=np.float32)
        for channel in range(channels):
            resized[channel] = _resize_hw(render[channel], height=query_h, width=query_w, interpolation=1)
        render = resized
    query_norm = query / np.maximum(np.linalg.norm(query, axis=0, keepdims=True), 1e-6)
    render_norm = render / np.maximum(np.linalg.norm(render, axis=0, keepdims=True), 1e-6)
    mask = np.ones((query_h, query_w), dtype=bool)
    if render_alpha is not None:
        alpha = _resize_hw(np.asarray(render_alpha, dtype=np.float32), height=query_h, width=query_w, interpolation=1)
        mask &= alpha > float(min_alpha)
    if render_depth is not None:
        depth = _resize_hw(np.asarray(render_depth, dtype=np.float32), height=query_h, width=query_w, interpolation=0)
        mask &= np.isfinite(depth) & (depth > 0.0)
    if not np.any(mask):
        return float("-inf")
    cosine = np.sum(query_norm * render_norm, axis=0)
    return float(np.mean(cosine[mask]) + 0.02 * np.mean(mask))


def _load_rgb_keypoint_detector_checkpoint(path: str | Path, *, device: str) -> MatchaRgbKeypointDetector:
    import torch

    checkpoint = torch.load(str(path), map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model = MatchaRgbKeypointDetector()
    model.load_state_dict(state)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    return model.to(torch_device).eval()


def _predict_rgb_detector_keypoints(
    model: MatchaRgbKeypointDetector,
    rgb: np.ndarray,
    *,
    device: str,
    top_k: int,
    threshold: float,
) -> np.ndarray:
    import torch

    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
    det_h = max(8, (orig_h // 8) * 8)
    det_w = max(8, (orig_w // 8) * 8)
    detector_input = image
    if det_h != orig_h or det_w != orig_w:
        detector_input = _resize_hw(image, height=det_h, width=det_w, interpolation=1)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    with torch.no_grad():
        tensor = torch.as_tensor(detector_input.transpose(2, 0, 1)[None], dtype=torch.float32, device=torch_device) / 255.0
        logits = model.to(torch_device).eval()(tensor).detach().cpu()
    xy, _scores, _labels = decode_keypoints_from_logits(
        logits,
        image_width=det_w,
        image_height=det_h,
        top_k=int(top_k),
        threshold=float(threshold),
    )
    if xy.shape[0] and (det_h != orig_h or det_w != orig_w):
        xy = xy.copy()
        xy[:, 0] *= float(orig_w) / float(det_w)
        xy[:, 1] *= float(orig_h) / float(det_h)
    return xy


def _predict_joint_local_fine_logits_for_matches(
    joint_model,
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    matches,
    *,
    device: str,
) -> np.ndarray:
    import torch

    values = list(matches)
    if not values:
        return np.zeros((0, 65), dtype=np.float32)
    qidx = np.asarray([int(match.query_index) for match in values], dtype=np.int64)
    ridx = np.asarray([int(match.render_index) for match in values], dtype=np.int64)
    pairs = np.zeros((len(values),), dtype=np.int64)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    with torch.no_grad():
        qmaps = torch.as_tensor(np.asarray(query_feature_map, dtype=np.float32)[None], device=torch_device)
        rmaps = torch.as_tensor(np.asarray(render_feature_map, dtype=np.float32)[None], device=torch_device)
        logits = joint_model.to(torch_device).eval().local_fine_logits_from_maps(
            qmaps,
            rmaps,
            torch.as_tensor(pairs, dtype=torch.long, device=torch_device),
            torch.as_tensor(qidx, dtype=torch.long, device=torch_device),
            torch.as_tensor(ridx, dtype=torch.long, device=torch_device),
        )
    return logits.detach().cpu().numpy().astype(np.float32, copy=False)


def _predict_joint_local_window_fine_logits_for_matches(
    joint_model,
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    matches,
    *,
    device: str,
) -> np.ndarray:
    import torch

    values = list(matches)
    if not values:
        return np.zeros((0, 64), dtype=np.float32)
    qidx = np.asarray([int(match.query_index) for match in values], dtype=np.int64)
    ridx = np.asarray([int(match.render_index) for match in values], dtype=np.int64)
    pairs = np.zeros((len(values),), dtype=np.int64)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    with torch.no_grad():
        qmaps = torch.as_tensor(np.asarray(query_feature_map, dtype=np.float32)[None], device=torch_device)
        rmaps = torch.as_tensor(np.asarray(render_feature_map, dtype=np.float32)[None], device=torch_device)
        logits = joint_model.to(torch_device).eval().local_window_fine_logits_from_maps(
            qmaps,
            rmaps,
            torch.as_tensor(pairs, dtype=torch.long, device=torch_device),
            torch.as_tensor(qidx, dtype=torch.long, device=torch_device),
            torch.as_tensor(ridx, dtype=torch.long, device=torch_device),
        )
    return logits.detach().cpu().numpy().astype(np.float32, copy=False)


def _predict_joint_patch_corr_fine_logits_for_matches(
    joint_model,
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    query_rgb: np.ndarray,
    render_rgb: np.ndarray,
    matches,
    *,
    target_side: str,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    device: str,
) -> np.ndarray:
    import torch

    values = list(matches)
    if not values:
        return np.zeros((0, 64), dtype=np.float32)
    side = str(target_side)
    if side not in {"render", "query"}:
        raise ValueError("target_side must be 'render' or 'query'")
    if side == "render":
        source_feature = np.asarray(query_feature_map, dtype=np.float32)
        target_feature = np.asarray(render_feature_map, dtype=np.float32)
        source_rgb = _rgb_to_bchw_float(query_rgb, grid_hw=(int(source_feature.shape[1]), int(source_feature.shape[2])))
        target_rgb = _rgb_to_bchw_float(render_rgb, grid_hw=(int(target_feature.shape[1]), int(target_feature.shape[2])))
        source_indices = np.asarray([int(match.query_index) for match in values], dtype=np.int64)
        target_indices = np.asarray([int(match.render_index) for match in values], dtype=np.int64)
        source_xy = np.stack([np.asarray(match.query_xy, dtype=np.float64).reshape(2) for match in values], axis=0)
        source_width = int(query_image_width)
        source_height = int(query_image_height)
    else:
        source_feature = np.asarray(render_feature_map, dtype=np.float32)
        target_feature = np.asarray(query_feature_map, dtype=np.float32)
        source_rgb = _rgb_to_bchw_float(render_rgb, grid_hw=(int(source_feature.shape[1]), int(source_feature.shape[2])))
        target_rgb = _rgb_to_bchw_float(query_rgb, grid_hw=(int(target_feature.shape[1]), int(target_feature.shape[2])))
        source_indices = np.asarray([int(match.render_index) for match in values], dtype=np.int64)
        target_indices = np.asarray([int(match.query_index) for match in values], dtype=np.int64)
        source_xy = np.stack([np.asarray(match.render_xy, dtype=np.float64).reshape(2) for match in values], axis=0)
        source_width = int(render_image_width)
        source_height = int(render_image_height)
    source_xy = source_xy.astype(np.float32, copy=False)
    source_xy[:, 0] *= float(source_rgb.shape[3]) / max(float(source_width), 1.0)
    source_xy[:, 1] *= float(source_rgb.shape[2]) / max(float(source_height), 1.0)
    pairs = np.zeros((len(values),), dtype=np.int64)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    with torch.no_grad():
        logits = joint_model.to(torch_device).eval().patch_corr_fine_logits_from_maps_and_rgb(
            torch.as_tensor(source_feature[None], dtype=torch.float32, device=torch_device),
            torch.as_tensor(target_feature[None], dtype=torch.float32, device=torch_device),
            torch.as_tensor(source_rgb, dtype=torch.float32, device=torch_device),
            torch.as_tensor(target_rgb, dtype=torch.float32, device=torch_device),
            torch.as_tensor(pairs, dtype=torch.long, device=torch_device),
            torch.as_tensor(source_indices, dtype=torch.long, device=torch_device),
            torch.as_tensor(target_indices, dtype=torch.long, device=torch_device),
            query_xy=torch.as_tensor(source_xy, dtype=torch.float32, device=torch_device),
        )
    return logits.detach().cpu().numpy().astype(np.float32, copy=False)


def _keypoint_logits_to_feature_cell_indices(
    logits: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    feature_grid_width: int,
    feature_grid_height: int,
    top_k: int,
    threshold: float,
) -> np.ndarray:
    xy, _scores, _labels = decode_keypoints_from_logits(
        np.asarray(logits, dtype=np.float32),
        image_width=int(image_width),
        image_height=int(image_height),
        top_k=int(top_k),
        threshold=float(threshold),
    )
    return keypoint_xy_to_feature_cell_indices(
        xy,
        image_width=int(image_width),
        image_height=int(image_height),
        feature_grid_width=int(feature_grid_width),
        feature_grid_height=int(feature_grid_height),
    )


def _cell_reliability_from_logits(logits: np.ndarray | None) -> np.ndarray | None:
    if logits is None:
        return None
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim == 1:
        return np.clip(arr.reshape(-1), 0.0, 1.0)
    if arr.ndim == 2:
        return (1.0 / (1.0 + np.exp(-arr.reshape(-1)))).astype(np.float32, copy=False)
    if arr.ndim == 3 and int(arr.shape[0]) == 65:
        values = arr - np.max(arr, axis=0, keepdims=True)
        prob = np.exp(values)
        prob = prob / np.maximum(np.sum(prob, axis=0, keepdims=True), 1e-8)
        return np.clip(1.0 - prob[64].reshape(-1), 0.0, 1.0).astype(np.float32, copy=False)
    if arr.ndim == 3 and int(arr.shape[0]) == 1:
        return (1.0 / (1.0 + np.exp(-arr[0].reshape(-1)))).astype(np.float32, copy=False)
    raise ValueError("reliability logits must have shape (H,W), (65,H,W), (1,H,W), or (N,)")


def _pair_fine_logit_stats(pair_fine_logits: np.ndarray | None) -> dict[str, object]:
    if pair_fine_logits is None:
        return {
            "fine_pair_applied_count": 0,
            "fine_pair_mean_confidence": None,
            "fine_pair_mean_entropy": None,
            "fine_pair_dustbin_rate": None,
        }
    logits = np.asarray(pair_fine_logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 64:
        return {
            "fine_pair_applied_count": 0,
            "fine_pair_mean_confidence": None,
            "fine_pair_mean_entropy": None,
            "fine_pair_dustbin_rate": None,
        }
    spatial = logits[:, :64]
    shifted = spatial - np.max(spatial, axis=1, keepdims=True)
    prob = np.exp(shifted)
    prob = prob / np.maximum(np.sum(prob, axis=1, keepdims=True), 1e-8)
    confidence = np.max(prob, axis=1)
    entropy = -np.sum(prob * np.log(np.maximum(prob, 1e-8)), axis=1)
    dustbin_rate = None
    if logits.shape[1] >= 65:
        full = logits[:, :65] - np.max(logits[:, :65], axis=1, keepdims=True)
        full_prob = np.exp(full)
        full_prob = full_prob / np.maximum(np.sum(full_prob, axis=1, keepdims=True), 1e-8)
        dustbin_rate = float(np.mean(np.argmax(full_prob, axis=1) == 64))
    return {
        "fine_pair_applied_count": int(logits.shape[0]),
        "fine_pair_mean_confidence": float(np.mean(confidence)),
        "fine_pair_mean_entropy": float(np.mean(entropy)),
        "fine_pair_dustbin_rate": dustbin_rate,
    }


def _pnp_match_diagnostics(pnp_matches, inlier_mask: np.ndarray | None) -> dict[str, object]:
    values = list(pnp_matches)
    if not values:
        return {
            "pnp_match_confidence_mean": None,
            "pnp_inlier_confidence_mean": None,
            "pnp_outlier_confidence_mean": None,
            "pnp_match_measurement_sigma_mean": None,
            "pnp_match_patch_offset_applied_ratio": None,
            "pnp_match_patch_offset_norm_mean_px": None,
        }
    confidences = np.asarray(
        [
            float(match.pnp_soft_score)
            if match.pnp_soft_score is not None and np.isfinite(float(match.pnp_soft_score))
            else float(np.clip((float(match.similarity) + 1.0) * 0.5, 0.0, 1.0))
            for match in values
        ],
        dtype=np.float64,
    )
    if inlier_mask is None:
        inliers = np.zeros((len(values),), dtype=bool)
    else:
        inliers = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        if inliers.shape[0] != len(values):
            inliers = np.zeros((len(values),), dtype=bool)
    sigmas = [
        float(match.measurement_sigma_px)
        for match in values
        if match.measurement_sigma_px is not None and np.isfinite(float(match.measurement_sigma_px))
    ]
    applied = [match.patch_offset_applied for match in values if match.patch_offset_applied is not None]
    offset_norms = [
        float(match.patch_offset_norm_px)
        for match in values
        if match.patch_offset_norm_px is not None and np.isfinite(float(match.patch_offset_norm_px))
    ]
    return {
        "pnp_match_confidence_mean": float(np.mean(confidences)),
        "pnp_inlier_confidence_mean": None if not np.any(inliers) else float(np.mean(confidences[inliers])),
        "pnp_outlier_confidence_mean": None if np.all(inliers) else float(np.mean(confidences[~inliers])),
        "pnp_match_measurement_sigma_mean": None if not sigmas else float(np.mean(sigmas)),
        "pnp_match_patch_offset_applied_ratio": None if not applied else float(np.mean([1.0 if item else 0.0 for item in applied])),
        "pnp_match_patch_offset_norm_mean_px": None if not offset_norms else float(np.mean(offset_norms)),
    }


def _coarse_cell_center_xy(
    token_index: int,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
) -> np.ndarray | None:
    if int(grid_width) <= 0 or int(grid_height) <= 0:
        return None
    idx = int(token_index)
    cell_count = int(grid_width) * int(grid_height)
    if idx < 0 or idx >= cell_count:
        return None
    row, col = divmod(idx, int(grid_width))
    return np.asarray(
        [
            (float(col) + 0.5) * float(image_width) / float(grid_width),
            (float(row) + 0.5) * float(image_height) / float(grid_height),
        ],
        dtype=np.float64,
    )


def _project_world_points_to_image(points: np.ndarray, pose_w2c: np.ndarray, camera) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if values.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for fine-offset diagnostics") from exc
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(values, rvec, pose[:3, 3], camera_matrix, distortion)
    return projected.reshape(-1, 2).astype(np.float64, copy=False)


def _sample_scalar_map_nearest(
    values: np.ndarray,
    xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    image = np.asarray(values, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("values must be a scalar HxW image")
    height, width = int(image.shape[0]), int(image.shape[1])
    x = coords[:, 0] * float(width) / max(float(image_width), 1.0)
    y = coords[:, 1] * float(height) / max(float(image_height), 1.0)
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    valid = (
        np.isfinite(coords).all(axis=1)
        & (coords[:, 0] >= 0.0)
        & (coords[:, 0] < float(image_width))
        & (coords[:, 1] >= 0.0)
        & (coords[:, 1] < float(image_height))
        & (xi >= 0)
        & (xi < width)
        & (yi >= 0)
        & (yi < height)
    )
    sampled = np.full((coords.shape[0],), np.nan, dtype=np.float32)
    if np.any(valid):
        sampled[valid] = image[yi[valid], xi[valid]]
    return sampled, valid


def _oracle_render_indices_from_query_gt_depth(
    query_feature_map: np.ndarray,
    query_depth: np.ndarray,
    *,
    query_camera,
    query_pose_w2c: np.ndarray,
    render_camera,
    render_pose_w2c: np.ndarray,
    render_grid_width: int,
    render_grid_height: int,
) -> np.ndarray:
    """Map each query feature cell to the render feature cell containing its GT surface point."""

    query_grid = feature_map_to_coarse_grid(
        query_feature_map,
        image_width=int(query_camera.width),
        image_height=int(query_camera.height),
    )
    query_xy = np.asarray(query_grid.xy, dtype=np.float64)
    depth_values, depth_valid = _sample_scalar_map_nearest(
        query_depth,
        query_xy,
        image_width=int(query_camera.width),
        image_height=int(query_camera.height),
    )
    world, world_valid = backproject_depth_to_world(query_xy, depth_values, query_camera, query_pose_w2c)
    pose = np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4)
    cam_xyz = (pose[:3, :3] @ world.T).T + pose[:3, 3]
    render_xy = _project_world_points_to_image(world, render_pose_w2c, render_camera)
    rgw, rgh = int(render_grid_width), int(render_grid_height)
    oracle = np.full((query_xy.shape[0],), -1, dtype=np.int64)
    if rgw <= 0 or rgh <= 0:
        return oracle
    valid = (
        depth_valid
        & world_valid
        & np.isfinite(render_xy).all(axis=1)
        & np.isfinite(cam_xyz).all(axis=1)
        & (cam_xyz[:, 2] > 1e-6)
        & (render_xy[:, 0] >= 0.0)
        & (render_xy[:, 0] < float(render_camera.width))
        & (render_xy[:, 1] >= 0.0)
        & (render_xy[:, 1] < float(render_camera.height))
    )
    cols = np.floor(render_xy[:, 0] * float(rgw) / max(float(render_camera.width), 1.0)).astype(np.int64)
    rows = np.floor(render_xy[:, 1] * float(rgh) / max(float(render_camera.height), 1.0)).astype(np.int64)
    valid &= (cols >= 0) & (cols < rgw) & (rows >= 0) & (rows < rgh)
    oracle[valid] = rows[valid] * rgw + cols[valid]
    return oracle


def _ece_binary(confidences: np.ndarray, targets: np.ndarray, *, bins: int = 10) -> float | None:
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    tgt = np.asarray(targets, dtype=np.float64).reshape(-1)
    valid = np.isfinite(conf) & np.isfinite(tgt)
    conf = conf[valid]
    tgt = tgt[valid]
    if conf.shape[0] == 0:
        return None
    ece_value = 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for idx in range(int(bins)):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        mask = (conf >= lo) & ((conf < hi) if idx + 1 < len(edges) - 1 else (conf <= hi))
        if not np.any(mask):
            continue
        ece_value += float(np.mean(mask)) * abs(float(np.mean(conf[mask])) - float(np.mean(tgt[mask])))
    return float(ece_value)


def _fine_offset_diagnostics(
    pnp_matches,
    gt_pose_w2c: np.ndarray,
    camera,
    *,
    query_grid_width: int,
    query_grid_height: int,
) -> dict[str, object]:
    values = list(pnp_matches)
    if not values:
        return {
            "fine_offset_eval_count": 0,
            "fine_offset_before_median_px": None,
            "fine_offset_after_median_px": None,
            "fine_offset_improvement_mean_px": None,
            "fine_offset_improved_ratio": None,
            "fine_offset_gt16_before": None,
            "fine_offset_gt16_after": None,
            "fine_offset_confidence_ece_16px": None,
            "fine_offset_uncertainty_error_over_sigma_mean": None,
            "fine_offset_uncertainty_within_1sigma": None,
            "fine_offset_uncertainty_within_2sigma": None,
        }
    points = np.stack([match.xyz for match in values], axis=0).astype(np.float64)
    projected = _project_world_points_to_image(points, gt_pose_w2c, camera)
    before_xy: list[np.ndarray] = []
    after_xy: list[np.ndarray] = []
    confidences: list[float] = []
    sigmas: list[float] = []
    valid_rows: list[int] = []
    for idx, match in enumerate(values):
        center = _coarse_cell_center_xy(
            int(match.token_index),
            image_width=int(camera.width),
            image_height=int(camera.height),
            grid_width=int(query_grid_width),
            grid_height=int(query_grid_height),
        )
        if center is None:
            continue
        xy = np.asarray(match.xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(xy)):
            continue
        before_xy.append(center)
        after_xy.append(xy)
        score = match.pnp_soft_score
        if score is not None and np.isfinite(float(score)):
            confidences.append(float(np.clip(float(score), 0.0, 1.0)))
        else:
            confidences.append(float(np.clip((float(match.similarity) + 1.0) * 0.5, 0.0, 1.0)))
        sigma = match.measurement_sigma_px
        sigmas.append(float(sigma) if sigma is not None and np.isfinite(float(sigma)) else np.nan)
        valid_rows.append(idx)
    if not valid_rows:
        return {
            "fine_offset_eval_count": 0,
            "fine_offset_before_median_px": None,
            "fine_offset_after_median_px": None,
            "fine_offset_improvement_mean_px": None,
            "fine_offset_improved_ratio": None,
            "fine_offset_gt16_before": None,
            "fine_offset_gt16_after": None,
            "fine_offset_confidence_ece_16px": None,
            "fine_offset_uncertainty_error_over_sigma_mean": None,
            "fine_offset_uncertainty_within_1sigma": None,
            "fine_offset_uncertainty_within_2sigma": None,
        }
    row_idx = np.asarray(valid_rows, dtype=np.int64)
    before = np.stack(before_xy, axis=0)
    after = np.stack(after_xy, axis=0)
    target = projected[row_idx]
    before_errors = np.linalg.norm(target - before, axis=1)
    after_errors = np.linalg.norm(target - after, axis=1)
    sigma_values = np.asarray(sigmas, dtype=np.float64)
    sigma_valid = np.isfinite(sigma_values) & (sigma_values > 0.0)
    normalized = after_errors[sigma_valid] / np.maximum(sigma_values[sigma_valid], 1e-6)
    correct16 = (after_errors <= 16.0).astype(np.float64)
    return {
        "fine_offset_eval_count": int(after_errors.shape[0]),
        "fine_offset_before_median_px": float(np.median(before_errors)),
        "fine_offset_after_median_px": float(np.median(after_errors)),
        "fine_offset_improvement_mean_px": float(np.mean(before_errors - after_errors)),
        "fine_offset_improved_ratio": float(np.mean(after_errors < before_errors)),
        "fine_offset_gt16_before": float(np.mean(before_errors <= 16.0)),
        "fine_offset_gt16_after": float(np.mean(after_errors <= 16.0)),
        "fine_offset_confidence_ece_16px": _ece_binary(np.asarray(confidences, dtype=np.float64), correct16),
        "fine_offset_uncertainty_error_over_sigma_mean": None if normalized.shape[0] == 0 else float(np.mean(normalized)),
        "fine_offset_uncertainty_within_1sigma": None if not np.any(sigma_valid) else float(np.mean(after_errors[sigma_valid] <= sigma_values[sigma_valid])),
        "fine_offset_uncertainty_within_2sigma": None if not np.any(sigma_valid) else float(np.mean(after_errors[sigma_valid] <= 2.0 * sigma_values[sigma_valid])),
    }


def _fine_confidence_diagnostics_from_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    def vals(key: str) -> list[float]:
        out = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            try:
                item = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(item):
                out.append(item)
        return out

    def mean_or_none(key: str) -> float | None:
        values = vals(key)
        return None if not values else float(np.mean(values))

    confidence_values = vals("mean_dual_softmax_confidence")
    target_values = vals("pnp_inlier_gt_precision_16px")
    ece = None
    if len(confidence_values) == len(target_values) and confidence_values:
        conf = np.asarray(confidence_values, dtype=np.float64)
        target = np.asarray(target_values, dtype=np.float64)
        ece_value = 0.0
        for lo in np.linspace(0.0, 0.8, 5):
            hi = lo + 0.2
            mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
            if np.any(mask):
                ece_value += float(np.mean(mask)) * abs(float(np.mean(conf[mask])) - float(np.mean(target[mask])))
        ece = float(ece_value)
    inlier_conf = vals("pnp_inlier_confidence_mean")
    outlier_conf = vals("pnp_outlier_confidence_mean")
    confidence_gap = None
    if inlier_conf and outlier_conf:
        confidence_gap = float(np.mean(inlier_conf) - np.mean(outlier_conf))
    match_counts = vals("match_count")
    fine_counts = vals("fine_pair_applied_count")
    fine_ratio = None
    if match_counts and fine_counts and len(match_counts) == len(fine_counts):
        denom = np.maximum(np.asarray(match_counts, dtype=np.float64), 1.0)
        fine_ratio = float(np.mean(np.asarray(fine_counts, dtype=np.float64) / denom))
    return {
        "mean_fine_pair_applied_ratio": fine_ratio,
        "mean_fine_pair_confidence": mean_or_none("fine_pair_mean_confidence"),
        "mean_fine_pair_entropy": mean_or_none("fine_pair_mean_entropy"),
        "mean_fine_pair_dustbin_rate": mean_or_none("fine_pair_dustbin_rate"),
        "mean_fine_offset_before_median_px": mean_or_none("fine_offset_before_median_px"),
        "mean_fine_offset_after_median_px": mean_or_none("fine_offset_after_median_px"),
        "mean_fine_offset_improvement_px": mean_or_none("fine_offset_improvement_mean_px"),
        "mean_fine_offset_improved_ratio": mean_or_none("fine_offset_improved_ratio"),
        "mean_fine_offset_gt16_before": mean_or_none("fine_offset_gt16_before"),
        "mean_fine_offset_gt16_after": mean_or_none("fine_offset_gt16_after"),
        "mean_fine_offset_confidence_ece_16px": mean_or_none("fine_offset_confidence_ece_16px"),
        "mean_fine_offset_uncertainty_error_over_sigma": mean_or_none("fine_offset_uncertainty_error_over_sigma_mean"),
        "mean_fine_offset_uncertainty_within_1sigma": mean_or_none("fine_offset_uncertainty_within_1sigma"),
        "mean_fine_offset_uncertainty_within_2sigma": mean_or_none("fine_offset_uncertainty_within_2sigma"),
        "mean_pnp_confidence": mean_or_none("pnp_match_confidence_mean"),
        "mean_pnp_inlier_confidence": mean_or_none("pnp_inlier_confidence_mean"),
        "mean_pnp_outlier_confidence": mean_or_none("pnp_outlier_confidence_mean"),
        "mean_pnp_confidence_inlier_gap": confidence_gap,
        "mean_pnp_measurement_sigma_px": mean_or_none("pnp_match_measurement_sigma_mean"),
        "mean_pnp_patch_offset_applied_ratio": mean_or_none("pnp_match_patch_offset_applied_ratio"),
        "mean_pnp_patch_offset_norm_px": mean_or_none("pnp_match_patch_offset_norm_mean_px"),
        "confidence_ece_16px": ece,
    }


def _render_lock_diagnostics_from_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    def vals(key: str) -> list[float]:
        out = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            try:
                item = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(item):
                out.append(item)
        return out

    def median_or_none(key: str) -> float | None:
        values = vals(key)
        return None if not values else float(np.median(values))

    def mean_or_none(key: str) -> float | None:
        values = vals(key)
        return None if not values else float(np.mean(values))

    pnp_minus_render_t = []
    pnp_minus_render_r = []
    for row in rows:
        try:
            pnp_t = float(row.get("translation_error_m"))
            render_t = float(row.get("render_translation_error_m"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(pnp_t) and np.isfinite(render_t):
            pnp_minus_render_t.append(abs(pnp_t - render_t))
        try:
            pnp_r = float(row.get("rotation_error_deg"))
            render_r = float(row.get("render_rotation_error_deg"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(pnp_r) and np.isfinite(render_r):
            pnp_minus_render_r.append(abs(pnp_r - render_r))

    locked_rate = None
    if pnp_minus_render_t:
        locked_rate = float(np.mean(np.asarray(pnp_minus_render_t, dtype=np.float64) <= 0.03))

    return {
        "median_render_translation_error_m": median_or_none("render_translation_error_m"),
        "median_render_rotation_error_deg": median_or_none("render_rotation_error_deg"),
        "mean_render_translation_error_m": mean_or_none("render_translation_error_m"),
        "mean_render_rotation_error_deg": mean_or_none("render_rotation_error_deg"),
        "median_abs_pnp_minus_render_translation_error_m": None
        if not pnp_minus_render_t
        else float(np.median(pnp_minus_render_t)),
        "mean_abs_pnp_minus_render_translation_error_m": None
        if not pnp_minus_render_t
        else float(np.mean(pnp_minus_render_t)),
        "median_abs_pnp_minus_render_rotation_error_deg": None
        if not pnp_minus_render_r
        else float(np.median(pnp_minus_render_r)),
        "mean_abs_pnp_minus_render_rotation_error_deg": None
        if not pnp_minus_render_r
        else float(np.mean(pnp_minus_render_r)),
        "locked_to_render_within_3cm_rate": locked_rate,
        "median_pnp_render_translation_delta_m": median_or_none("pnp_render_translation_delta_m"),
        "median_pnp_render_rotation_delta_deg": median_or_none("pnp_render_rotation_delta_deg"),
        "mean_pnp_render_translation_delta_m": mean_or_none("pnp_render_translation_delta_m"),
        "mean_pnp_render_rotation_delta_deg": mean_or_none("pnp_render_rotation_delta_deg"),
    }


def _residual_solver_diagnostics_from_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    def vals(key: str) -> list[float]:
        out = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            try:
                item = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(item):
                out.append(item)
        return out

    def median_or_none(key: str) -> float | None:
        values = vals(key)
        return None if not values else float(np.median(values))

    output = {
        "residual_solver_all_median_translation_error_m": median_or_none("residual_solver_all_translation_error_m"),
        "residual_solver_all_median_rotation_error_deg": median_or_none("residual_solver_all_rotation_error_deg"),
        "residual_solver_oracle16_median_translation_error_m": median_or_none("residual_solver_oracle16_translation_error_m"),
        "residual_solver_oracle16_median_rotation_error_deg": median_or_none("residual_solver_oracle16_rotation_error_deg"),
        "residual_solver_oracle16_mean_match_count": None
        if not vals("residual_solver_oracle16_match_count")
        else float(np.mean(vals("residual_solver_oracle16_match_count"))),
    }
    oracle_prefixes_set: set[str] = set()
    for row in rows:
        for key in row.keys():
            text = str(key)
            if text.startswith("residual_solver_oracle") and text.endswith("_match_count"):
                oracle_prefixes_set.add(text[len("residual_solver_") : -len("_match_count")])
    oracle_prefixes = sorted(oracle_prefixes_set)
    for prefix in oracle_prefixes:
        match_key = f"residual_solver_{prefix}_match_count"
        t_key = f"residual_solver_{prefix}_translation_error_m"
        r_key = f"residual_solver_{prefix}_rotation_error_deg"
        counts = vals(match_key)
        output[f"residual_solver_{prefix}_mean_match_count"] = None if not counts else float(np.mean(counts))
        output[f"residual_solver_{prefix}_median_translation_error_m"] = median_or_none(t_key)
        output[f"residual_solver_{prefix}_median_rotation_error_deg"] = median_or_none(r_key)
    return output


def _run_render_pose_residual_diagnostic(
    *,
    matches,
    gt_errors: np.ndarray,
    render_pose_w2c: np.ndarray,
    gt_pose_w2c: np.ndarray,
    camera,
    oracle_threshold_px: float,
    oracle_thresholds_px: Sequence[float] | None = None,
    max_iterations: int,
) -> dict[str, object]:
    def threshold_key(value: float) -> str:
        text = f"{float(value):g}".replace(".", "p").replace("-", "m")
        return f"oracle{text}"

    if not matches:
        output = {
            "residual_solver_all_match_count": 0,
            "residual_solver_oracle16_match_count": 0,
        }
        thresholds = list(oracle_thresholds_px or [float(oracle_threshold_px)])
        for threshold in thresholds:
            output[f"residual_solver_{threshold_key(float(threshold))}_match_count"] = 0
        return output
    output: dict[str, object] = {
        "residual_solver_all_match_count": int(len(matches)),
    }
    try:
        all_pose = solve_render_pose_delta_from_matches(
            matches,
            np.asarray(render_pose_w2c, dtype=np.float64),
            camera,
            max_iterations=int(max_iterations),
        )
        all_t, all_r = _pose_metrics(all_pose, gt_pose_w2c)
        output.update(
            {
                "residual_solver_all_translation_error_m": all_t,
                "residual_solver_all_rotation_error_deg": all_r,
            }
        )
    except Exception as exc:
        output["residual_solver_all_error"] = str(exc)

    errors = np.asarray(gt_errors, dtype=np.float64).reshape(-1)
    thresholds = [float(item) for item in (oracle_thresholds_px or [float(oracle_threshold_px)])]
    if float(oracle_threshold_px) not in thresholds:
        thresholds.append(float(oracle_threshold_px))
    seen: set[str] = set()
    for threshold in thresholds:
        key = threshold_key(float(threshold))
        if key in seen:
            continue
        seen.add(key)
        oracle_matches = [
            match
            for idx, match in enumerate(matches)
            if idx < errors.shape[0] and np.isfinite(errors[idx]) and errors[idx] <= float(threshold)
        ]
        output[f"residual_solver_{key}_match_count"] = int(len(oracle_matches))
        if len(oracle_matches) >= 3:
            try:
                oracle_pose = solve_render_pose_delta_from_matches(
                    oracle_matches,
                    np.asarray(render_pose_w2c, dtype=np.float64),
                    camera,
                    max_iterations=int(max_iterations),
                )
                oracle_t, oracle_r = _pose_metrics(oracle_pose, gt_pose_w2c)
                output.update(
                    {
                        f"residual_solver_{key}_translation_error_m": oracle_t,
                        f"residual_solver_{key}_rotation_error_deg": oracle_r,
                    }
                )
            except Exception as exc:
                output[f"residual_solver_{key}_error"] = str(exc)
    legacy_key = threshold_key(float(oracle_threshold_px))
    if legacy_key != "oracle16":
        for suffix in ("match_count", "translation_error_m", "rotation_error_deg", "error"):
            source = f"residual_solver_{legacy_key}_{suffix}"
            if source in output:
                output[f"residual_solver_oracle16_{suffix}"] = output[source]
    elif "residual_solver_oracle16_match_count" not in output:
        output["residual_solver_oracle16_match_count"] = 0
    return output


def _normalize_negative_csv_option_args(
    argv: Sequence[str] | None,
    *,
    option_names: set[str],
) -> list[str] | None:
    if argv is None:
        return None
    items = list(argv)
    normalized: list[str] = []
    idx = 0
    while idx < len(items):
        item = str(items[idx])
        if item in option_names and idx + 1 < len(items):
            value = str(items[idx + 1])
            if value.startswith("-") and "," in value:
                normalized.append(f"{item}={value}")
                idx += 2
                continue
        normalized.append(item)
        idx += 1
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--feature_mode", default="radio_final", choices=("radio_final", "radio_dual"))
    parser.add_argument("--radio_fine_intermediate_index", type=int, default=-6)
    parser.add_argument("--radio_coarse_source", default="final", choices=("final", "intermediate"))
    parser.add_argument("--radio_coarse_intermediate_index", type=int, default=-1)
    parser.add_argument("--extract_query_features_from_image", action="store_true")
    parser.add_argument("--selector_checkpoint", default="")
    parser.add_argument("--matcha_adapter_checkpoint", default="")
    parser.add_argument("--matcha_joint_checkpoint", default="")
    parser.add_argument("--matcha_qkv_attention_checkpoint", default="")
    parser.add_argument(
        "--matcha_eval_preset",
        default="none",
        choices=(
            "none",
            "conservative_confidence",
            "radio_matcha_local_search",
            "radio_matcha_local_window",
            "radio_matcha_local_window_no_render_offset",
            "radio_matcha_patch_corr",
            "anti_lock_render_search",
            "anti_lock_post_pair_render_search",
        ),
    )
    parser.add_argument("--matcha_confidence_mode", default="dual_softmax", choices=("dual_softmax", "learned", "blend"))
    parser.add_argument("--matcha_confidence_blend", type=float, default=0.5)
    parser.add_argument("--matcha_min_detector_confidence", type=float, default=0.0)
    parser.add_argument("--matcha_min_keypoint_confidence", type=float, default=0.0)
    parser.add_argument("--matcha_cell_offset_side", default="both", choices=("both", "query", "render", "none"))
    parser.add_argument("--matcha_use_pair_fine_head", action="store_true")
    parser.add_argument("--matcha_pair_fine_side", default="query", choices=("render", "query"))
    parser.add_argument("--matcha_pair_fine_coordinate_mode", default="argmax", choices=("argmax", "softargmax"))
    parser.add_argument("--matcha_use_local_fine_attention", action="store_true")
    parser.add_argument("--matcha_use_local_window_fine_head", action="store_true")
    parser.add_argument("--matcha_local_window_confidence_blend", type=float, default=0.0)
    parser.add_argument("--matcha_use_patch_corr_fine_head", action="store_true")
    parser.add_argument("--matcha_patch_corr_target_side", default="render", choices=("render", "query", "both"))
    parser.add_argument("--matcha_patch_corr_confidence_blend", type=float, default=0.0)
    parser.add_argument("--post_pair_render_refine_radius_px", type=float, default=0.0)
    parser.add_argument("--post_pair_render_refine_step_px", type=float, default=2.0)
    parser.add_argument("--post_pair_render_refine_mode", default="argmax", choices=("argmax", "softargmax"))
    parser.add_argument("--post_pair_render_refine_query_sigma_px", type=float, default=4.0)
    parser.add_argument("--render_side_local_offset_radius_cells", type=int, default=0)
    parser.add_argument("--render_side_local_offset_max_candidates", type=int, default=9)
    parser.add_argument("--render_side_local_offset_top_k_per_query", type=int, default=0)
    parser.add_argument(
        "--matcha_keypoint_proposal_source",
        default="none",
        choices=("none", "adapter", "alike", "rgb_detector"),
        help="Optional MATCHA-style keypoint/cell proposal source for matcha_c2f coarse matching.",
    )
    parser.add_argument(
        "--matcha_keypoint_proposal_mode",
        default="soft",
        choices=("soft", "hard"),
        help="Use detector proposals as a soft confidence prior or as a hard candidate mask.",
    )
    parser.add_argument("--matcha_keypoint_proposal_threshold", type=float, default=0.0)
    parser.add_argument("--matcha_keypoint_proposal_top_k", type=int, default=0)
    parser.add_argument("--matcha_keypoint_prior_boost", type=float, default=0.25)
    parser.add_argument("--matcha_keypoint_prior_penalty", type=float, default=0.0)
    parser.add_argument("--matcha_reliability_prior_source", default="none", choices=("none", "detector", "keypoint", "auto"))
    parser.add_argument("--matcha_reliability_prior_boost", type=float, default=0.25)
    parser.add_argument("--matcha_reliability_prior_penalty", type=float, default=0.0)
    parser.add_argument("--matcha_rgb_keypoint_detector_checkpoint", default="")
    parser.add_argument("--alike_repo", default="/root/matcha")
    parser.add_argument("--alike_model", default="alike-t")
    parser.add_argument("--alike_top_k", type=int, default=4096)
    parser.add_argument("--alike_scores_th", type=float, default=0.1)
    parser.add_argument("--alike_n_limit", type=int, default=8000)
    parser.add_argument("--feature_fusion_mode", default="none", choices=("none", "local_attention"))
    parser.add_argument("--feature_fusion_radius", type=int, default=1)
    parser.add_argument("--feature_fusion_temperature", type=float, default=5.0)
    parser.add_argument("--feature_fusion_alpha", type=float, default=0.5)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument(
        "--render_pose_mode",
        default="gt",
        choices=("gt", "gt_offset", "gt_rotation_offset", "reference_top1", "reference_top5", "reference_top10"),
    )
    parser.add_argument("--render_pose_world_offset", default="0,0,0")
    parser.add_argument("--render_pose_rotation_offset_deg", default="0,0,0")
    parser.add_argument("--render_pose_rotation_search_offsets_deg", default="")
    parser.add_argument("--render_pose_rotation_search_axis", default="y", choices=("x", "y", "z"))
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--reference_top_k", type=int, default=5)
    parser.add_argument("--render_width", type=int, default=0)
    parser.add_argument("--render_height", type=int, default=0)
    parser.add_argument("--render_canvas_scale", type=float, default=1.0)
    parser.add_argument("--render_radius_px", type=float, default=2.0)
    parser.add_argument("--render_depth_epsilon", type=float, default=0.02)
    parser.add_argument("--renderer", default="soft", choices=("soft", "gsplat", "official_2dgs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--render_token_cache_dir", default="")
    parser.add_argument("--skip_existing_render_tokens", action="store_true")
    parser.add_argument("--query_token_cache_dir", default="")
    parser.add_argument("--skip_existing_query_tokens", action="store_true")
    parser.add_argument("--render_rgb_depth_cache_dir", default="")
    parser.add_argument("--skip_existing_render_rgb_depth", action="store_true")
    parser.add_argument("--detector", default="superpoint", choices=("orb", "superpoint", "disk"))
    parser.add_argument("--max_keypoints", type=int, default=1000)
    parser.add_argument("--match_mode", default="mnn", choices=("mnn", "mnn_dual_filter", "dual_softmax", "matcha_c2f"))
    parser.add_argument("--ratio_threshold", type=float, default=0.9)
    parser.add_argument("--dual_softmax_logit_scale", type=float, default=10.0)
    parser.add_argument("--min_dual_softmax_confidence", type=float, default=0.0)
    parser.add_argument("--min_similarity", type=float, default=0.0)
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--matcha_coarse_top_k_per_query", type=int, default=1)
    parser.add_argument("--matcha_coarse_mutual_mode", default="legacy", choices=("legacy", "none", "annotate", "filter"))
    parser.add_argument("--matcha_coarse_local_window_radius_cells", type=int, default=-1)
    parser.add_argument("--matcha_post_confidence_top_k_per_query", type=int, default=0)
    parser.add_argument("--coarse_candidate_ranker_model", default="")
    parser.add_argument("--coarse_candidate_ranker_feature_set", default="coarse", choices=("descriptor", "coarse", "coarse_local"))
    parser.add_argument("--coarse_candidate_ranker_blend", type=float, default=1.0)
    parser.add_argument("--fine_render_search_radius_px", type=float, default=0.0)
    parser.add_argument("--fine_render_search_step_px", type=float, default=1.0)
    parser.add_argument(
        "--matcha_fine_mode",
        default="argmax",
        choices=(
            "argmax",
            "softargmax",
            "bilateral_argmax",
            "bilateral_softargmax",
            "fine_attention_argmax",
            "fine_attention_softargmax",
            "cross_argmax",
            "cross_softargmax",
        ),
    )
    parser.add_argument("--matcha_fine_softmax_temperature", type=float, default=20.0)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=12.0)
    parser.add_argument("--pnp_iterations", type=int, default=2000)
    parser.add_argument("--pnp_min_inliers", type=int, default=6)
    parser.add_argument("--measurement_sigma_px", type=float, default=0.0)
    parser.add_argument("--pnp_soft_order_mode", default="none", choices=("none", "similarity", "margin", "reliability", "confidence", "uncertainty", "composite"))
    parser.add_argument("--pnp_soft_order_top_n", type=int, default=0)
    parser.add_argument("--calibrated_correspondence_confidence_model", default="")
    parser.add_argument("--calibrated_correspondence_feature_set", default="descriptor")
    parser.add_argument("--pose_scorer_model", default="")
    parser.add_argument("--pnp_unique_query_refit", action="store_true")
    parser.add_argument("--render_offset_min_alpha", type=float, default=0.0)
    parser.add_argument("--render_offset_max_depth_delta_m", type=float, default=-1.0)
    parser.add_argument("--render_offset_fallback_to_cell_center", action="store_true")
    parser.add_argument("--coverage_filter_grid", type=int, default=8)
    parser.add_argument("--coverage_filter_max_per_cell", type=int, default=8)
    parser.add_argument("--coverage_filter_min_confidence", type=float, default=-1.0)
    parser.add_argument("--coverage_filter_max_total", type=int, default=512)
    parser.add_argument("--enable_pose_rescore", action="store_true")
    parser.add_argument("--pose_rescore_margin", type=float, default=0.03)
    parser.add_argument("--enable_render_pose_residual_solver", action="store_true")
    parser.add_argument("--residual_solver_oracle_threshold_px", type=float, default=16.0)
    parser.add_argument("--residual_solver_oracle_thresholds_px", default="5,10,16")
    parser.add_argument("--residual_solver_iterations", type=int, default=10)
    parser.add_argument("--pose_update_iterations", type=int, default=1)
    parser.add_argument(
        "--pose_update_selection",
        default="guarded_score",
        choices=("last", "best_score", "guarded_score", "best_alignment", "guarded_alignment"),
        help="How to select the final pose when pose_update_iterations > 1.",
    )
    parser.add_argument("--pose_update_score_margin", type=float, default=0.03)
    parser.add_argument(
        "--pose_update_filter_schedule",
        default="final_only",
        choices=("always", "final_only", "never"),
        help="Apply coverage filtering always, never, or only on the final render-pose update iteration.",
    )
    parser.add_argument("--save_match_table", action="store_true")
    parser.add_argument("--match_table_path", default="")
    parser.add_argument("--match_table_stage", choices=("pose", "candidate"), default="pose")
    parser.add_argument("--save_pose_candidate_table", action="store_true")
    parser.add_argument("--pose_candidate_table_path", default="")
    parser.add_argument("--save_coarse_oracle_table", action="store_true")
    parser.add_argument("--coarse_oracle_table_path", default="")
    parser.add_argument("--save_coarse_oracle_candidate_table", action="store_true")
    parser.add_argument("--coarse_oracle_candidate_table_path", default="")
    parser.add_argument("--coarse_oracle_candidate_top_k", type=int, default=5)
    parser.add_argument("--coarse_oracle_candidate_positive_radius", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_queries", type=int, default=20)
    parser.add_argument("--view_selection", default="prefix", choices=("prefix", "uniform"))
    parser.add_argument("--visualize_limit", type=int, default=5)
    parser.add_argument("--stream_rows", action="store_true")
    parser.add_argument("--resume_existing_rows", action="store_true")
    parser.add_argument("--rebuild_summary_from_rows", action="store_true")
    args = parser.parse_args(
        _normalize_negative_csv_option_args(
            argv,
            option_names={
                "--render_pose_world_offset",
                "--render_pose_rotation_offset_deg",
                "--render_pose_rotation_search_offsets_deg",
            },
        )
    )
    if str(args.feature_mode) == "radio_dual" and str(args.layer_name) == "radio_final":
        args.layer_name = "radio_dual"
    if str(args.render_pose_mode) == "reference_top10":
        args.reference_top_k = 10
    if str(args.matcha_eval_preset) == "conservative_confidence":
        args.matcha_confidence_mode = "learned"
        args.matcha_confidence_blend = 1.0
        args.pnp_soft_order_mode = "confidence"
        args.pnp_soft_order_top_n = 800
        args.coverage_filter_min_confidence = 0.05
        args.measurement_sigma_px = 16.0
    if str(args.matcha_eval_preset) == "radio_matcha_local_search":
        args.matcha_use_pair_fine_head = False
        args.matcha_confidence_mode = "learned"
        args.matcha_confidence_blend = 1.0
        args.fine_render_search_radius_px = 12.0
        args.fine_render_search_step_px = 2.0
        args.matcha_fine_mode = "fine_attention_argmax"
        args.pnp_soft_order_mode = "confidence"
        args.pnp_soft_order_top_n = 800
        args.coverage_filter_min_confidence = 0.05
        args.measurement_sigma_px = 16.0
    if str(args.matcha_eval_preset) in {"radio_matcha_local_window", "radio_matcha_local_window_no_render_offset"}:
        args.matcha_use_pair_fine_head = False
        args.matcha_use_local_window_fine_head = True
        args.matcha_local_window_confidence_blend = 0.5
        args.matcha_confidence_mode = "learned"
        args.matcha_confidence_blend = 1.0
        args.fine_render_search_radius_px = 0.0
        args.post_pair_render_refine_radius_px = 0.0
        args.matcha_pair_fine_coordinate_mode = "softargmax"
        if str(args.matcha_eval_preset) == "radio_matcha_local_window":
            args.render_side_local_offset_radius_cells = 1
            args.render_side_local_offset_max_candidates = 9
            args.render_side_local_offset_top_k_per_query = 3
        else:
            args.render_side_local_offset_radius_cells = 0
            args.render_side_local_offset_top_k_per_query = 0
        args.matcha_fine_mode = "fine_attention_argmax"
        args.pnp_soft_order_mode = "confidence"
        args.pnp_soft_order_top_n = 800
        args.coverage_filter_min_confidence = 0.05
        args.measurement_sigma_px = 16.0
    if str(args.matcha_eval_preset) == "radio_matcha_patch_corr":
        args.matcha_use_pair_fine_head = False
        args.matcha_use_local_window_fine_head = False
        args.matcha_use_patch_corr_fine_head = True
        args.matcha_patch_corr_target_side = "render"
        args.matcha_patch_corr_confidence_blend = 0.0
        args.matcha_confidence_mode = "learned"
        args.matcha_confidence_blend = 1.0
        args.fine_render_search_radius_px = 0.0
        args.post_pair_render_refine_radius_px = 0.0
        args.matcha_fine_mode = "fine_attention_argmax"
        args.pnp_soft_order_mode = "confidence"
        args.pnp_soft_order_top_n = 800
        args.coverage_filter_min_confidence = 0.05
        args.measurement_sigma_px = 16.0
    if str(args.matcha_eval_preset) == "anti_lock_render_search":
        args.matcha_confidence_mode = "learned"
        args.matcha_confidence_blend = 1.0
        args.matcha_pair_fine_side = "query"
        args.fine_render_search_radius_px = 24.0
        args.fine_render_search_step_px = 2.0
        args.matcha_fine_mode = "fine_attention_argmax"
        args.pnp_soft_order_mode = "confidence"
        args.pnp_soft_order_top_n = 800
        args.coverage_filter_min_confidence = 0.05
        args.measurement_sigma_px = 16.0
    if str(args.matcha_eval_preset) == "anti_lock_post_pair_render_search":
        args.matcha_confidence_mode = "learned"
        args.matcha_confidence_blend = 1.0
        args.matcha_pair_fine_side = "query"
        args.fine_render_search_radius_px = 0.0
        args.post_pair_render_refine_radius_px = 24.0
        args.post_pair_render_refine_step_px = 2.0
        args.post_pair_render_refine_mode = "argmax"
        args.pnp_soft_order_mode = "confidence"
        args.pnp_soft_order_top_n = 800
        args.coverage_filter_min_confidence = 0.05
        args.measurement_sigma_px = 16.0
    return args


_parse_render_rgb_eval_args = parse_args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    if bool(args.rebuild_summary_from_rows):
        summary = _write_rebuilt_summary_from_rows(output_dir, args, elapsed_sec=time.perf_counter() - started)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    base_render_width, base_render_height = _resolve_render_size(camera, int(args.render_width), int(args.render_height))
    base_render_camera = _scale_camera(camera, base_render_width, base_render_height)
    canvas_scale = float(args.render_canvas_scale)
    if canvas_scale < 1.0:
        raise ValueError("--render_canvas_scale must be >= 1.0")
    render_width = int(round(float(base_render_width) * canvas_scale))
    render_height = int(round(float(base_render_height) * canvas_scale))
    render_camera = (
        base_render_camera
        if abs(canvas_scale - 1.0) < 1e-9
        else _render_canvas_camera_from_base(
            base_render_camera,
            canvas_width=render_width,
            canvas_height=render_height,
        )
    )
    render_config = GaussianVFMRenderConfig(
        width=render_width,
        height=render_height,
        radius_px=float(args.render_radius_px),
        depth_epsilon=float(args.render_depth_epsilon),
        l2_normalize_pixels=True,
    )
    if str(args.renderer) == "official_2dgs":
        rgb_source = load_official_2dgs_source_from_ply(Path(args.gaussian_rgb_ply))
        depth_field = None
    else:
        rgb_source = load_gaussian_rgb_source_from_ply(Path(args.gaussian_rgb_ply))
        depth_field = _depth_field_from_rgb_source(rgb_source)
    detector_fn = _make_keypoint_detector(args.detector, int(args.max_keypoints), args.device)
    from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor

    radio = RADIOFeatureExtractor(version=args.radio_version, device=args.device, radio_repo=args.radio_repo)
    matcha_adapter_run = (
        load_matcha_coarse_fine_adapter(Path(args.matcha_adapter_checkpoint), device=args.device)
        if str(args.matcha_adapter_checkpoint)
        else None
    )
    matcha_joint_run = (
        load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=args.device)
        if str(args.matcha_joint_checkpoint)
        else None
    )
    if matcha_adapter_run is not None and matcha_joint_run is not None:
        raise ValueError("Use either --matcha_adapter_checkpoint or --matcha_joint_checkpoint, not both")
    if bool(args.matcha_use_patch_corr_fine_head) and matcha_joint_run is None:
        raise ValueError("--matcha_use_patch_corr_fine_head requires --matcha_joint_checkpoint")
    learned_head_summary = matcha_adapter_run.summary if matcha_adapter_run is not None else (matcha_joint_run.summary if matcha_joint_run is not None else {})
    if matcha_adapter_run is not None or matcha_joint_run is not None:
        missing_state_keys = [str(item) for item in learned_head_summary.get("missing_state_keys", [])]
        missing_state_keys = [
            key
            for key in missing_state_keys
            if not key.startswith("local_window_uncertainty_head.")
            and not key.startswith("adapter.pair_fine_uncertainty_head.")
        ]
        learned_head_prefixes = []
        if float(args.matcha_min_detector_confidence) > 0.0:
            learned_head_prefixes.append("detector_head.")
        if float(args.matcha_min_keypoint_confidence) > 0.0:
            learned_head_prefixes.append("keypoint_head.")
        if str(args.matcha_confidence_mode) != "dual_softmax":
            learned_head_prefixes.append("pair_confidence_head.")
        if bool(args.matcha_use_pair_fine_head):
            learned_head_prefixes.append("pair_fine_head.")
        if bool(args.matcha_use_local_window_fine_head):
            learned_head_prefixes.append("local_window_")
        if bool(args.matcha_use_patch_corr_fine_head):
            learned_head_prefixes.append("patch_corr_fine_head.")
        if learned_head_prefixes and any(key.startswith(tuple(learned_head_prefixes)) for key in missing_state_keys):
            raise ValueError(
                "MATCHA learned heads were requested, but the checkpoint is missing trained learned-head weights"
            )
    qkv_attention = None
    if str(args.matcha_qkv_attention_checkpoint):
        qkv_attention, _qkv_summary = load_qkv_attention_checkpoint(Path(args.matcha_qkv_attention_checkpoint), device=args.device)
    calibrated_confidence_model = (
        CalibratedLogisticConfidence.load_json(Path(args.calibrated_correspondence_confidence_model))
        if str(args.calibrated_correspondence_confidence_model)
        else None
    )
    coarse_candidate_ranker_model = (
        CalibratedLogisticConfidence.load_json(Path(args.coarse_candidate_ranker_model))
        if str(args.coarse_candidate_ranker_model)
        else None
    )
    pose_scorer_model = (
        load_pose_scorer_model(Path(args.pose_scorer_model))
        if str(args.pose_scorer_model)
        else None
    )
    pose_scorer_feature_names: list[str] = []
    if str(args.pose_scorer_model):
        pose_scorer_feature_names = list(json.loads(Path(args.pose_scorer_model).read_text()).get("feature_names", []))
    rgb_keypoint_detector = None
    if str(args.matcha_keypoint_proposal_source) == "rgb_detector":
        if str(args.matcha_rgb_keypoint_detector_checkpoint):
            rgb_keypoint_detector = _load_rgb_keypoint_detector_checkpoint(
                args.matcha_rgb_keypoint_detector_checkpoint,
                device=args.device,
            )
        elif matcha_joint_run is not None:
            rgb_keypoint_detector = matcha_joint_run.model.rgb_keypoint_detector
        else:
            raise ValueError(
                "--matcha_rgb_keypoint_detector_checkpoint or --matcha_joint_checkpoint is required for rgb_detector proposals"
            )
    alike_keypoint_extractor = None
    if str(args.matcha_keypoint_proposal_source) == "alike":
        alike_keypoint_extractor = AlikeKeypointExtractor(
            matcha_repo=str(args.alike_repo),
            model_name=str(args.alike_model),
            top_k=int(args.alike_top_k),
            scores_th=float(args.alike_scores_th),
            n_limit=int(args.alike_n_limit),
            device=str(args.device),
        )
    render_cache_dir = Path(args.render_token_cache_dir) if args.render_token_cache_dir else None
    query_cache_dir = Path(args.query_token_cache_dir) if args.query_token_cache_dir else None
    render_rgb_depth_cache_dir = Path(args.render_rgb_depth_cache_dir) if args.render_rgb_depth_cache_dir else None
    reference_top1 = {}
    reference_topk = {}
    if str(args.render_pose_mode) in {"reference_top1", "reference_top5", "reference_top10"}:
        if not str(args.candidate_bank):
            raise ValueError("--candidate_bank is required for reference render pose modes")
        candidate_bank = CandidateHypothesisBank.from_jsonl(Path(args.candidate_bank))
        reference_top1 = group_top_reference_poses(candidate_bank.candidates)
        reference_topk = group_topk_reference_poses(candidate_bank.candidates, top_k=int(args.reference_top_k))
    render_world_offset = parse_world_offset(str(args.render_pose_world_offset))
    render_rotation_offset = parse_rotation_offset_deg(str(args.render_pose_rotation_offset_deg))
    render_rotation_search_offsets = parse_rotation_search_offsets_deg(str(args.render_pose_rotation_search_offsets_deg))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.csv"
    summary_path = output_dir / "summary.json"
    render_cache_manifest_path = output_dir / "render_cache_manifest.csv"
    existing_query_ids = _existing_query_ids_from_rows(rows_path) if bool(args.resume_existing_rows) else set()
    row_writer = (
        _StreamingCsvWriter(rows_path, append=bool(args.resume_existing_rows))
        if bool(args.stream_rows)
        else None
    )
    render_cache_writer = (
        _StreamingCsvWriter(render_cache_manifest_path, append=bool(args.resume_existing_rows))
        if bool(args.stream_rows)
        else None
    )
    coarse_oracle_table_path = (
        (
            Path(args.coarse_oracle_table_path)
            if str(args.coarse_oracle_table_path)
            else output_dir / "coarse_oracle_table.csv"
        )
        if bool(args.save_coarse_oracle_table)
        else None
    )
    coarse_oracle_writer = (
        _StreamingCsvWriter(coarse_oracle_table_path)
        if coarse_oracle_table_path is not None
        else None
    )
    coarse_oracle_accumulator = CoarseOracleRankAccumulator()
    coarse_oracle_candidate_table_path = (
        (
            Path(args.coarse_oracle_candidate_table_path)
            if str(args.coarse_oracle_candidate_table_path)
            else output_dir / "coarse_oracle_candidate_table.csv"
        )
        if bool(args.save_coarse_oracle_candidate_table)
        else None
    )
    coarse_oracle_candidate_writer = (
        _StreamingCsvWriter(coarse_oracle_candidate_table_path)
        if coarse_oracle_candidate_table_path is not None
        else None
    )
    pose_candidate_table_path = (
        (
            Path(args.pose_candidate_table_path)
            if str(args.pose_candidate_table_path)
            else output_dir / "pose_candidate_table.csv"
        )
        if bool(args.save_pose_candidate_table)
        else None
    )
    pose_candidate_table_writer = (
        _StreamingCsvWriter(pose_candidate_table_path, append=bool(args.resume_existing_rows))
        if pose_candidate_table_path is not None and bool(args.stream_rows)
        else None
    )
    coarse_oracle_candidate_row_count = 0
    coarse_oracle_candidate_positive_count = 0
    rows = _read_csv_rows(rows_path) if bool(args.resume_existing_rows) and rows_path.exists() else []
    match_table_rows = []
    pose_candidate_table_rows = (
        _read_csv_rows(pose_candidate_table_path)
        if pose_candidate_table_path is not None
        and bool(args.resume_existing_rows)
        and not bool(args.stream_rows)
        and pose_candidate_table_path.exists()
        else []
    )
    render_cache_rows = (
        _read_csv_rows(render_cache_manifest_path)
        if bool(args.resume_existing_rows) and render_cache_manifest_path.exists()
        else []
    )
    records = _select_records(manifest.records, int(args.max_queries), str(args.view_selection), start_index=int(args.start_index))
    for vis_idx, record in enumerate(records):
        if record.image_id in existing_query_ids:
            continue
        render_cache_written_count = int(len(render_cache_rows))
        gt = gt_by_query.get(record.image_id)
        if gt is None:
            continue
        query_image = _read_rgb(Path(args.image_root) / record.image_id)
        if bool(args.extract_query_features_from_image):
            query_cache_path = (
                None
                if query_cache_dir is None
                else _matcha_query_token_cache_path(
                    query_cache_dir,
                    record.image_id,
                    render_width=int(camera.width),
                    render_height=int(camera.height),
                    layer_name=str(args.layer_name),
                )
            )
            query_feature = _load_or_extract_matcha_joint_feature(
                cache_path=query_cache_path,
                rgb=query_image,
                extractor=radio,
                layer_name=args.layer_name,
                feature_mode=str(args.feature_mode),
                fine_intermediate_index=int(args.radio_fine_intermediate_index),
                coarse_source=str(args.radio_coarse_source),
                coarse_intermediate_index=int(args.radio_coarse_intermediate_index),
                skip_existing=bool(args.skip_existing_query_tokens),
            )
        else:
            query_feature = _load_query_feature(record.token_path, args.layer_name)
        query_feature_raw = query_feature
        try:
            if str(args.render_pose_mode) in {"reference_top5", "reference_top10"}:
                candidates = list(reference_topk.get(str(record.image_id), []))
                if not candidates:
                    raise KeyError(f"reference topK pose not found for query {record.image_id!r}")
                initial_render_poses = []
                for ref_idx, candidate in enumerate(candidates):
                    pose = np.asarray(candidate.pose, dtype=np.float64).reshape(4, 4)
                    t_error, r_error = render_pose_error_fields(pose, gt.pose_w2c)
                    initial_render_poses.append(
                        RenderPoseSelection(
                            pose_w2c=pose,
                            label=f"reference_top{int(ref_idx) + 1}",
                            candidate_id=str(candidate.candidate_id),
                            reference_image=candidate.reference_image,
                            render_translation_error_m=t_error,
                            render_rotation_error_deg=r_error,
                        )
                    )
            else:
                initial_render_poses = [
                    select_render_pose(
                        record.image_id,
                        gt.pose_w2c,
                        mode=str(args.render_pose_mode),
                        world_offset=render_world_offset,
                        rotation_offset_deg=render_rotation_offset,
                        reference_top1=reference_top1,
                    )
                ]
            if render_rotation_search_offsets:
                initial_render_poses = expand_render_pose_rotation_candidates(
                    initial_render_poses,
                    rotation_offsets_deg=render_rotation_search_offsets,
                    axis=str(args.render_pose_rotation_search_axis),
                    gt_pose_w2c=gt.pose_w2c,
                )
        except KeyError:
            rows.append(
                {
                    "query_id": record.image_id,
                    "status": "missing_render_pose",
                    "render_pose_mode": str(args.render_pose_mode),
                    "pnp_success": False,
                }
            )
            continue
        requested_iterations = max(int(args.pose_update_iterations), 1)
        def _run_pose_iteration(current_render_pose: RenderPoseSelection, iteration_index: int) -> dict[str, object]:
            cacheable_iteration = int(iteration_index) == 0
            rgb_depth_cache_path = None
            if render_rgb_depth_cache_dir is not None and cacheable_iteration:
                rgb_depth_cache_path = (
                    render_rgb_depth_cache_dir
                    / f"{_safe_image_stem(record.image_id)}_{_safe_image_stem(current_render_pose.label)}_{render_width}x{render_height}.npz"
                )
            rgb_depth_cache_hit = bool(
                rgb_depth_cache_path is not None
                and bool(args.skip_existing_render_rgb_depth)
                and Path(rgb_depth_cache_path).exists()
            )
            render_rgb, render_depth, render_alpha = _load_or_render_rgb_depth_cache(
                cache_path=rgb_depth_cache_path,
                render_fn=lambda pose=current_render_pose.pose_w2c: _render_rgb_and_depth(
                    rgb_source,
                    depth_field,
                    pose_w2c=pose,
                    camera=camera,
                    config=render_config,
                    renderer=args.renderer,
                    device=args.device,
                ),
                skip_existing=bool(args.skip_existing_render_rgb_depth),
            )
            cache_path = None
            if render_cache_dir is not None and cacheable_iteration:
                cache_path = _render_token_cache_path(
                    render_cache_dir,
                    f"{record.image_id}:{current_render_pose.label}",
                    render_width,
                    render_height,
                )
            render_feature_cache_hit = bool(
                cache_path is not None and bool(args.skip_existing_render_tokens) and Path(cache_path).exists()
            )
            render_feature = _load_or_extract_matcha_joint_feature(
                cache_path=cache_path,
                rgb=render_rgb,
                extractor=radio,
                layer_name=args.layer_name,
                feature_mode=str(args.feature_mode),
                fine_intermediate_index=int(args.radio_fine_intermediate_index),
                coarse_source=str(args.radio_coarse_source),
                coarse_intermediate_index=int(args.radio_coarse_intermediate_index),
                skip_existing=bool(args.skip_existing_render_tokens) and cacheable_iteration,
            )
            render_cache_rows.append(
                {
                    "query_id": record.image_id,
                    "iteration_index": int(iteration_index),
                    "render_pose_label": current_render_pose.label,
                    "render_pose_mode": str(args.render_pose_mode),
                    "render_pose_hash": _cache_pose_hash(current_render_pose.pose_w2c),
                    "render_camera_hash": _cache_numpy_sha1(
                        np.asarray(
                            [
                                float(camera.width),
                                float(camera.height),
                                float(camera.model_id),
                                *[float(value) for value in camera.params],
                            ],
                            dtype=np.float64,
                        )
                    ),
                    "render_width": int(render_width),
                    "render_height": int(render_height),
                    "rgb_depth_cache_path": "" if rgb_depth_cache_path is None else str(rgb_depth_cache_path),
                    "render_token_cache_path": "" if cache_path is None else str(cache_path),
                    "rgb_depth_cache_hit": bool(rgb_depth_cache_hit),
                    "render_feature_cache_hit": bool(render_feature_cache_hit),
                    "rgb_depth_cache_file_sha1": _cache_file_sha1(rgb_depth_cache_path),
                    "render_token_cache_file_sha1": _cache_file_sha1(cache_path),
                    "rgb_sha1": _cache_numpy_sha1(render_rgb),
                    "depth_sha1": _cache_numpy_sha1(render_depth),
                    "alpha_sha1": _cache_numpy_sha1(render_alpha),
                    "feature_sha1": _cache_numpy_sha1(render_feature),
                    "matcha_joint_checkpoint_sha1": _cache_file_sha1(Path(args.matcha_joint_checkpoint))
                    if str(args.matcha_joint_checkpoint)
                    else None,
                    "matcha_adapter_checkpoint_sha1": _cache_file_sha1(Path(args.matcha_adapter_checkpoint))
                    if str(args.matcha_adapter_checkpoint)
                    else None,
                    "gaussian_rgb_ply_size": Path(args.gaussian_rgb_ply).stat().st_size
                    if Path(args.gaussian_rgb_ply).exists()
                    else None,
                    "gaussian_rgb_ply_mtime": Path(args.gaussian_rgb_ply).stat().st_mtime
                    if Path(args.gaussian_rgb_ply).exists()
                    else None,
                }
            )
            query_feature = maybe_fuse_feature_map(
                query_feature_raw,
                mode=str(args.feature_fusion_mode),
                radius=int(args.feature_fusion_radius),
                temperature=float(args.feature_fusion_temperature),
                alpha=float(args.feature_fusion_alpha),
                device=str(args.device),
            )
            render_feature = maybe_fuse_feature_map(
                render_feature,
                mode=str(args.feature_fusion_mode),
                radius=int(args.feature_fusion_radius),
                temperature=float(args.feature_fusion_temperature),
                alpha=float(args.feature_fusion_alpha),
                device=str(args.device),
            )
            query_feature_for_joint_local_fine = np.array(query_feature, dtype=np.float32, copy=False)
            render_feature_for_joint_local_fine = np.array(render_feature, dtype=np.float32, copy=False)
            query_offset_logits = None
            render_offset_logits = None
            query_detector_logits = None
            render_detector_logits = None
            query_keypoint_logits = None
            render_keypoint_logits = None
            matcha_pair_model = None
            matcha_joint_pair_model = None
            if matcha_joint_run is not None:
                (
                    query_feature,
                    render_feature,
                    query_offset_logits,
                    render_offset_logits,
                    query_detector_logits,
                    render_detector_logits,
                ) = _project_feature_pair_with_matcha_joint_model(
                    query_feature,
                    render_feature,
                    joint_model=matcha_joint_run.model,
                    device=str(args.device),
                )
                matcha_joint_pair_model = matcha_joint_run.model
            elif matcha_adapter_run is not None:
                (
                    query_feature,
                    render_feature,
                    query_offset_logits,
                    render_offset_logits,
                    query_detector_logits,
                    render_detector_logits,
                    query_keypoint_logits,
                    render_keypoint_logits,
                ) = _project_feature_pair_with_matcha_adapter_full(
                    query_feature,
                    render_feature,
                    adapter_model=matcha_adapter_run.model,
                    device=str(args.device),
                )
                matcha_pair_model = matcha_adapter_run.model
            else:
                query_feature, render_feature = _project_feature_pair_for_render_rgb_eval(
                    query_feature,
                    render_feature,
                    selector_checkpoint=str(args.selector_checkpoint),
                    device=str(args.device),
                )
            if qkv_attention is not None:
                query_feature, render_feature = apply_qkv_attention_to_feature_maps(
                    qkv_attention,
                    query_feature,
                    render_feature,
                    device=str(args.device),
                )
            iteration_alignment_score = _render_query_feature_alignment_score(
                query_feature,
                render_feature,
                render_alpha=render_alpha,
                render_depth=render_depth,
            )
            if int(query_feature.shape[0]) != int(render_feature.shape[0]):
                return {
                    "status": "descriptor_dim_mismatch",
                    "render_pose": current_render_pose,
                    "query_dim": int(query_feature.shape[0]),
                    "render_dim": int(render_feature.shape[0]),
                    "pnp_success": False,
                }
            query_candidate_indices = None
            render_candidate_indices = None
            if args.match_mode == "matcha_c2f" and str(args.matcha_keypoint_proposal_source) != "none":
                proposal_top_k = (
                    int(args.matcha_keypoint_proposal_top_k)
                    if int(args.matcha_keypoint_proposal_top_k) > 0
                    else int(args.max_keypoints)
                )
                proposal_threshold = float(args.matcha_keypoint_proposal_threshold)
                if str(args.matcha_keypoint_proposal_source) == "adapter":
                    if query_keypoint_logits is not None and render_keypoint_logits is not None:
                        query_candidate_indices = _keypoint_logits_to_feature_cell_indices(
                            query_keypoint_logits,
                            image_width=int(camera.width),
                            image_height=int(camera.height),
                            feature_grid_width=int(query_feature.shape[2]),
                            feature_grid_height=int(query_feature.shape[1]),
                            top_k=proposal_top_k,
                            threshold=proposal_threshold,
                        )
                        render_candidate_indices = _keypoint_logits_to_feature_cell_indices(
                            render_keypoint_logits,
                            image_width=int(render_config.width),
                            image_height=int(render_config.height),
                            feature_grid_width=int(render_feature.shape[2]),
                            feature_grid_height=int(render_feature.shape[1]),
                            top_k=proposal_top_k,
                            threshold=proposal_threshold,
                        )
                elif str(args.matcha_keypoint_proposal_source) == "alike":
                    if alike_keypoint_extractor is None:
                        raise RuntimeError("ALIKE keypoint extractor was not initialized")
                    qxy_proposal, _qproposal_scores = alike_keypoint_extractor(query_image)
                    rxy_proposal, _rproposal_scores = alike_keypoint_extractor(render_rgb)
                    if proposal_top_k > 0:
                        qxy_proposal = qxy_proposal[:proposal_top_k]
                        rxy_proposal = rxy_proposal[:proposal_top_k]
                    query_candidate_indices = keypoint_xy_to_feature_cell_indices(
                        qxy_proposal,
                        image_width=int(camera.width),
                        image_height=int(camera.height),
                        feature_grid_width=int(query_feature.shape[2]),
                        feature_grid_height=int(query_feature.shape[1]),
                    )
                    render_candidate_indices = keypoint_xy_to_feature_cell_indices(
                        rxy_proposal,
                        image_width=int(render_config.width),
                        image_height=int(render_config.height),
                        feature_grid_width=int(render_feature.shape[2]),
                        feature_grid_height=int(render_feature.shape[1]),
                    )
                elif str(args.matcha_keypoint_proposal_source) == "rgb_detector":
                    if rgb_keypoint_detector is None:
                        raise RuntimeError("RGB keypoint detector was not initialized")
                    qxy_proposal = _predict_rgb_detector_keypoints(
                        rgb_keypoint_detector,
                        query_image,
                        device=str(args.device),
                        top_k=proposal_top_k,
                        threshold=proposal_threshold,
                    )
                    rxy_proposal = _predict_rgb_detector_keypoints(
                        rgb_keypoint_detector,
                        render_rgb,
                        device=str(args.device),
                        top_k=proposal_top_k,
                        threshold=proposal_threshold,
                    )
                    query_candidate_indices = keypoint_xy_to_feature_cell_indices(
                        qxy_proposal,
                        image_width=int(camera.width),
                        image_height=int(camera.height),
                        feature_grid_width=int(query_feature.shape[2]),
                        feature_grid_height=int(query_feature.shape[1]),
                    )
                    render_candidate_indices = keypoint_xy_to_feature_cell_indices(
                        rxy_proposal,
                        image_width=int(render_config.width),
                        image_height=int(render_config.height),
                        feature_grid_width=int(render_feature.shape[2]),
                        feature_grid_height=int(render_feature.shape[1]),
                    )
            if args.match_mode == "matcha_c2f":
                query_grid = feature_map_to_coarse_grid(
                    query_feature,
                    image_width=int(camera.width),
                    image_height=int(camera.height),
                )
                render_grid = feature_map_to_coarse_grid(
                    render_feature,
                    image_width=int(render_config.width),
                    image_height=int(render_config.height),
                )
                query_xy = query_grid.xy
                render_xy = render_grid.xy
                qdesc = query_grid.descriptors
                rdesc = render_grid.descriptors
                query_xy_valid = query_xy
                render_xy_valid = render_xy
                cell_offset_side = str(args.matcha_cell_offset_side)
                query_offset_for_matching = query_offset_logits if cell_offset_side in {"both", "query"} else None
                render_offset_for_matching = render_offset_logits if cell_offset_side in {"both", "render"} else None
                kp_matches = matcha_coarse_to_fine_keypoint_matches(
                    query_feature,
                    render_feature,
                    query_image_width=int(camera.width),
                    query_image_height=int(camera.height),
                    render_image_width=int(render_config.width),
                    render_image_height=int(render_config.height),
                    logit_scale=float(args.dual_softmax_logit_scale),
                    min_confidence=float(args.min_dual_softmax_confidence),
                    min_similarity=float(args.min_similarity),
                    max_matches=args.max_matches,
                    fine_search_radius_px=float(args.fine_render_search_radius_px),
                    fine_search_step_px=float(args.fine_render_search_step_px),
                    fine_mode=str(args.matcha_fine_mode),
                    fine_softmax_temperature=float(args.matcha_fine_softmax_temperature),
                    mutual=True,
                    coarse_top_k_per_query=int(args.matcha_coarse_top_k_per_query),
                    coarse_mutual_mode=(
                        None if str(args.matcha_coarse_mutual_mode) == "legacy" else str(args.matcha_coarse_mutual_mode)
                    ),
                    coarse_local_window_radius_cells=(
                        None
                        if int(args.matcha_coarse_local_window_radius_cells) < 0
                        else int(args.matcha_coarse_local_window_radius_cells)
                    ),
                    query_offset_logits=query_offset_for_matching,
                    render_offset_logits=render_offset_for_matching,
                    query_candidate_indices=(
                        query_candidate_indices
                        if str(args.matcha_keypoint_proposal_mode) == "hard"
                        else None
                    ),
                    render_candidate_indices=(
                        render_candidate_indices
                        if str(args.matcha_keypoint_proposal_mode) == "hard"
                        else None
                    ),
                )
            else:
                query_xy, _query_scores = detector_fn(query_image)
                render_xy, _render_scores = detector_fn(render_rgb)
                qdesc, qvalid = bilinear_sample_feature_map(
                    query_feature,
                    query_xy,
                    image_width=int(camera.width),
                    image_height=int(camera.height),
                )
                rdesc, rvalid = bilinear_sample_feature_map(
                    render_feature,
                    render_xy,
                    image_width=int(render_config.width),
                    image_height=int(render_config.height),
                )
                query_xy_valid = query_xy[qvalid]
                render_xy_valid = render_xy[rvalid]
                qdesc = qdesc[qvalid]
                rdesc = rdesc[rvalid]
            if args.match_mode == "dual_softmax":
                kp_matches = dual_softmax_keypoint_matches(
                    query_xy_valid,
                    qdesc,
                    render_xy_valid,
                    rdesc,
                    logit_scale=float(args.dual_softmax_logit_scale),
                    min_confidence=float(args.min_dual_softmax_confidence),
                    min_similarity=float(args.min_similarity),
                    max_matches=args.max_matches,
                )
            elif args.match_mode != "matcha_c2f":
                kp_matches = mutual_nn_keypoint_matches(
                    query_xy_valid,
                    qdesc,
                    render_xy_valid,
                    rdesc,
                    ratio_threshold=float(args.ratio_threshold),
                    min_similarity=float(args.min_similarity),
                    dual_softmax_logit_scale=(
                        float(args.dual_softmax_logit_scale) if args.match_mode == "mnn_dual_filter" else None
                    ),
                    min_dual_softmax_confidence=float(args.min_dual_softmax_confidence),
                    max_matches=args.max_matches,
                )
            learned_confidences = None
            pair_fine_logits = None
            if (matcha_pair_model is not None or matcha_joint_pair_model is not None) and args.match_mode == "matcha_c2f" and kp_matches:
                if query_detector_logits is not None and render_detector_logits is not None and float(args.matcha_min_detector_confidence) > 0.0:
                    qdet = 1.0 / (1.0 + np.exp(-np.asarray(query_detector_logits, dtype=np.float32).reshape(-1)))
                    rdet = 1.0 / (1.0 + np.exp(-np.asarray(render_detector_logits, dtype=np.float32).reshape(-1)))
                    threshold = float(args.matcha_min_detector_confidence)
                    kp_matches = [
                        match
                        for match in kp_matches
                        if 0 <= int(match.query_index) < qdet.shape[0]
                        and 0 <= int(match.render_index) < rdet.shape[0]
                        and float(qdet[int(match.query_index)]) >= threshold
                        and float(rdet[int(match.render_index)]) >= threshold
                    ]
                if query_keypoint_logits is not None and render_keypoint_logits is not None and float(args.matcha_min_keypoint_confidence) > 0.0:
                    qlogits = np.asarray(query_keypoint_logits, dtype=np.float32).reshape(65, -1).T
                    rlogits = np.asarray(render_keypoint_logits, dtype=np.float32).reshape(65, -1).T
                    qlogits = qlogits - np.max(qlogits, axis=1, keepdims=True)
                    rlogits = rlogits - np.max(rlogits, axis=1, keepdims=True)
                    qprob = np.exp(qlogits)
                    rprob = np.exp(rlogits)
                    qprob = qprob / np.maximum(np.sum(qprob, axis=1, keepdims=True), 1e-8)
                    rprob = rprob / np.maximum(np.sum(rprob, axis=1, keepdims=True), 1e-8)
                    qkey = 1.0 - qprob[:, 64]
                    rkey = 1.0 - rprob[:, 64]
                    threshold = float(args.matcha_min_keypoint_confidence)
                    kp_matches = [
                        match
                        for match in kp_matches
                        if 0 <= int(match.query_index) < qkey.shape[0]
                        and 0 <= int(match.render_index) < rkey.shape[0]
                        and float(qkey[int(match.query_index)]) >= threshold
                        and float(rkey[int(match.render_index)]) >= threshold
                    ]
                if kp_matches and (str(args.matcha_confidence_mode) != "dual_softmax" or bool(args.matcha_use_pair_fine_head)):
                    if matcha_joint_pair_model is not None:
                        learned_confidences, pair_fine_logits = predict_matcha_joint_pair_heads_for_matches(
                            matcha_joint_pair_model,
                            query_feature,
                            render_feature,
                            kp_matches,
                            device=str(args.device),
                            fine_target_side=str(args.matcha_pair_fine_side),
                        )
                    else:
                        if bool(args.matcha_use_pair_fine_head) and str(args.matcha_pair_fine_side) != "render":
                            raise ValueError("--matcha_pair_fine_side=query requires --matcha_joint_checkpoint")
                        learned_confidences, pair_fine_logits = predict_matcha_pair_heads_for_matches(
                            matcha_pair_model,
                            query_feature,
                            render_feature,
                            kp_matches,
                            device=str(args.device),
                        )
                if kp_matches and learned_confidences is not None and str(args.matcha_confidence_mode) != "dual_softmax":
                    from dataclasses import replace

                    blend = float(np.clip(float(args.matcha_confidence_blend), 0.0, 1.0))
                    updated_matches = []
                    for match, learned in zip(kp_matches, learned_confidences):
                        learned_value = float(np.clip(float(learned), 0.0, 1.0))
                        if str(args.matcha_confidence_mode) == "blend":
                            base_value = (
                                float(match.dual_softmax_confidence)
                                if match.dual_softmax_confidence is not None
                                else float(np.clip((float(match.similarity) + 1.0) * 0.5, 0.0, 1.0))
                            )
                            confidence = (1.0 - blend) * base_value + blend * learned_value
                        else:
                            confidence = learned_value
                        updated_matches.append(replace(match, dual_softmax_confidence=float(confidence)))
                    kp_matches = updated_matches
                if kp_matches and pair_fine_logits is not None and bool(args.matcha_use_pair_fine_head):
                    kp_matches = apply_pair_fine_logits_to_matches(
                        kp_matches,
                        pair_fine_logits,
                        target_side=str(args.matcha_pair_fine_side),
                        query_image_width=int(camera.width),
                        query_image_height=int(camera.height),
                        query_grid_width=int(query_feature.shape[2]),
                        query_grid_height=int(query_feature.shape[1]),
                        render_image_width=int(render_config.width),
                        render_image_height=int(render_config.height),
                        render_grid_width=int(render_feature.shape[2]),
                        render_grid_height=int(render_feature.shape[1]),
                        coordinate_mode=str(args.matcha_pair_fine_coordinate_mode),
                    )
                if kp_matches and bool(args.matcha_use_local_fine_attention):
                    if matcha_joint_run is None:
                        raise ValueError("--matcha_use_local_fine_attention requires --matcha_joint_checkpoint")
                    local_fine_logits = _predict_joint_local_fine_logits_for_matches(
                        matcha_joint_run.model,
                        query_feature_for_joint_local_fine,
                        render_feature_for_joint_local_fine,
                        kp_matches,
                        device=str(args.device),
                    )
                    kp_matches = apply_pair_fine_logits_to_matches(
                        kp_matches,
                        local_fine_logits,
                        render_image_width=int(render_config.width),
                        render_image_height=int(render_config.height),
                        render_grid_width=int(render_feature.shape[2]),
                        render_grid_height=int(render_feature.shape[1]),
                        coordinate_mode=str(args.matcha_pair_fine_coordinate_mode),
                    )
                if kp_matches and bool(args.matcha_use_local_window_fine_head):
                    if matcha_joint_run is None:
                        raise ValueError("--matcha_use_local_window_fine_head requires --matcha_joint_checkpoint")
                    local_window_fine_logits = _predict_joint_local_window_fine_logits_for_matches(
                        matcha_joint_run.model,
                        query_feature_for_joint_local_fine,
                        render_feature_for_joint_local_fine,
                        kp_matches,
                        device=str(args.device),
                    )
                    pair_fine_logits = local_window_fine_logits
                    kp_matches = apply_pair_fine_logits_to_matches(
                        kp_matches,
                        local_window_fine_logits,
                        render_image_width=int(render_config.width),
                        render_image_height=int(render_config.height),
                        render_grid_width=int(render_feature.shape[2]),
                        render_grid_height=int(render_feature.shape[1]),
                        coordinate_mode=str(args.matcha_pair_fine_coordinate_mode),
                    )
                    if float(args.matcha_local_window_confidence_blend) > 0.0:
                        kp_matches = apply_fine_logit_confidence_to_matches(
                            kp_matches,
                            local_window_fine_logits,
                            blend=float(args.matcha_local_window_confidence_blend),
                        )
                if kp_matches and bool(args.matcha_use_patch_corr_fine_head):
                    if matcha_joint_run is None:
                        raise ValueError("--matcha_use_patch_corr_fine_head requires --matcha_joint_checkpoint")
                    patch_target_side = str(args.matcha_patch_corr_target_side)
                    if patch_target_side in {"render", "both"}:
                        patch_corr_render_logits = _predict_joint_patch_corr_fine_logits_for_matches(
                            matcha_joint_run.model,
                            query_feature_for_joint_local_fine,
                            render_feature_for_joint_local_fine,
                            query_image,
                            render_rgb,
                            kp_matches,
                            target_side="render",
                            query_image_width=int(camera.width),
                            query_image_height=int(camera.height),
                            render_image_width=int(render_config.width),
                            render_image_height=int(render_config.height),
                            device=str(args.device),
                        )
                        pair_fine_logits = patch_corr_render_logits
                        kp_matches = apply_pair_fine_logits_to_matches(
                            kp_matches,
                            patch_corr_render_logits,
                            target_side="render",
                            render_image_width=int(render_config.width),
                            render_image_height=int(render_config.height),
                            render_grid_width=int(render_feature.shape[2]),
                            render_grid_height=int(render_feature.shape[1]),
                            coordinate_mode=str(args.matcha_pair_fine_coordinate_mode),
                        )
                        if float(args.matcha_patch_corr_confidence_blend) > 0.0:
                            kp_matches = apply_fine_logit_confidence_to_matches(
                                kp_matches,
                                patch_corr_render_logits,
                                blend=float(args.matcha_patch_corr_confidence_blend),
                            )
                    if patch_target_side in {"query", "both"} and kp_matches:
                        patch_corr_query_logits = _predict_joint_patch_corr_fine_logits_for_matches(
                            matcha_joint_run.model,
                            query_feature_for_joint_local_fine,
                            render_feature_for_joint_local_fine,
                            query_image,
                            render_rgb,
                            kp_matches,
                            target_side="query",
                            query_image_width=int(camera.width),
                            query_image_height=int(camera.height),
                            render_image_width=int(render_config.width),
                            render_image_height=int(render_config.height),
                            device=str(args.device),
                        )
                        if patch_target_side == "query":
                            pair_fine_logits = patch_corr_query_logits
                        kp_matches = apply_pair_fine_logits_to_matches(
                            kp_matches,
                            patch_corr_query_logits,
                            target_side="query",
                            query_image_width=int(camera.width),
                            query_image_height=int(camera.height),
                            query_grid_width=int(query_feature.shape[2]),
                            query_grid_height=int(query_feature.shape[1]),
                            coordinate_mode=str(args.matcha_pair_fine_coordinate_mode),
                        )
                        if float(args.matcha_patch_corr_confidence_blend) > 0.0:
                            kp_matches = apply_fine_logit_confidence_to_matches(
                                kp_matches,
                                patch_corr_query_logits,
                                blend=float(args.matcha_patch_corr_confidence_blend),
                            )
            post_pair_render_refine_count = 0
            if (
                args.match_mode == "matcha_c2f"
                and float(args.post_pair_render_refine_radius_px) > 0.0
                and kp_matches
            ):
                before_count = int(len(kp_matches))
                kp_matches = refine_render_matches_by_local_attention(
                    kp_matches,
                    query_feature,
                    render_feature,
                    query_image_width=int(camera.width),
                    query_image_height=int(camera.height),
                    render_image_width=int(render_config.width),
                    render_image_height=int(render_config.height),
                    search_radius_px=float(args.post_pair_render_refine_radius_px),
                    step_px=float(args.post_pair_render_refine_step_px),
                    mode=str(args.post_pair_render_refine_mode),
                    temperature=float(args.matcha_fine_softmax_temperature),
                    query_spatial_sigma_px=float(args.post_pair_render_refine_query_sigma_px),
                )
                post_pair_render_refine_count = int(min(before_count, len(kp_matches)))
            if (
                args.match_mode == "matcha_c2f"
                and str(args.matcha_keypoint_proposal_source) != "none"
                and str(args.matcha_keypoint_proposal_mode) == "soft"
                and kp_matches
            ):
                kp_matches = apply_keypoint_cell_prior_to_matches(
                    kp_matches,
                    query_candidate_indices=query_candidate_indices,
                    render_candidate_indices=render_candidate_indices,
                    boost=float(args.matcha_keypoint_prior_boost),
                    penalty=float(args.matcha_keypoint_prior_penalty),
                )
            if args.match_mode == "matcha_c2f" and str(args.matcha_reliability_prior_source) != "none" and kp_matches:
                source = str(args.matcha_reliability_prior_source)
                if source == "auto":
                    source = "keypoint" if query_keypoint_logits is not None and render_keypoint_logits is not None else "detector"
                if source == "keypoint":
                    query_reliability = _cell_reliability_from_logits(query_keypoint_logits)
                    render_reliability = _cell_reliability_from_logits(render_keypoint_logits)
                else:
                    query_reliability = _cell_reliability_from_logits(query_detector_logits)
                    render_reliability = _cell_reliability_from_logits(render_detector_logits)
                kp_matches = apply_cell_reliability_prior_to_matches(
                    kp_matches,
                    query_reliability=query_reliability,
                    render_reliability=render_reliability,
                    boost=float(args.matcha_reliability_prior_boost),
                    penalty=float(args.matcha_reliability_prior_penalty),
                )
            if args.match_mode == "matcha_c2f" and coarse_candidate_ranker_model is not None and kp_matches:
                kp_matches = annotate_keypoint_matches_with_coarse_candidate_ranker(
                    kp_matches,
                    coarse_candidate_ranker_model,
                    feature_set=str(args.coarse_candidate_ranker_feature_set),
                    blend=float(args.coarse_candidate_ranker_blend),
                )
            if args.match_mode == "matcha_c2f" and int(args.matcha_post_confidence_top_k_per_query) > 0 and kp_matches:
                kp_matches = retain_topk_matches_per_query(
                    kp_matches,
                    max_per_query=int(args.matcha_post_confidence_top_k_per_query),
                )
            render_side_local_offset_expanded_match_count = 0
            if args.match_mode == "matcha_c2f" and int(args.render_side_local_offset_radius_cells) > 0 and kp_matches:
                kp_matches = expand_matches_with_render_local_offsets(
                    kp_matches,
                    render_image_width=int(render_config.width),
                    render_image_height=int(render_config.height),
                    render_grid_width=int(render_feature.shape[2]),
                    render_grid_height=int(render_feature.shape[1]),
                    cell_radius=int(args.render_side_local_offset_radius_cells),
                    max_candidates_per_match=int(args.render_side_local_offset_max_candidates),
                )
                kp_matches = rescore_keypoint_matches_by_feature_similarity(
                    kp_matches,
                    query_feature,
                    render_feature,
                    query_image_width=int(camera.width),
                    query_image_height=int(camera.height),
                    render_image_width=int(render_config.width),
                    render_image_height=int(render_config.height),
                )
                if int(args.render_side_local_offset_top_k_per_query) > 0:
                    kp_matches = retain_topk_matches_per_query(
                        kp_matches,
                        max_per_query=int(args.render_side_local_offset_top_k_per_query),
                    )
                render_side_local_offset_expanded_match_count = int(len(kp_matches))
            if float(args.fine_render_search_radius_px) > 0.0 and args.match_mode != "matcha_c2f":
                kp_matches = refine_render_keypoint_matches_by_local_correlation(
                    kp_matches,
                    qdesc,
                    render_feature,
                    image_width=int(render_config.width),
                    image_height=int(render_config.height),
                    search_radius_px=float(args.fine_render_search_radius_px),
                    step_px=float(args.fine_render_search_step_px),
                )
            pnp_matches = keypoint_feature_matches_to_pnp_matches(
                kp_matches,
                render_depth,
                render_camera,
                current_render_pose.pose_w2c,
                image_width=int(render_config.width),
                image_height=int(render_config.height),
                rendered_alpha=render_alpha,
                render_grid_width=int(render_feature.shape[2]),
                render_grid_height=int(render_feature.shape[1]),
                min_render_alpha=float(args.render_offset_min_alpha),
                max_render_depth_delta_m=(
                    None
                    if float(args.render_offset_max_depth_delta_m) < 0.0
                    else float(args.render_offset_max_depth_delta_m)
                ),
                fallback_to_cell_center=bool(args.render_offset_fallback_to_cell_center),
            )
            candidate_pnp_matches_for_table = list(pnp_matches)
            if float(args.measurement_sigma_px) > 0.0:
                pnp_matches = annotate_measurement_uncertainty(
                    pnp_matches,
                    base_sigma_px=float(args.measurement_sigma_px),
                    fine_refined=float(args.fine_render_search_radius_px) > 0.0,
                )
            unfiltered_pnp_match_count = int(len(pnp_matches))
            if calibrated_confidence_model is not None and pnp_matches:
                pnp_matches = annotate_matches_with_calibrated_confidence(
                    pnp_matches,
                    calibrated_confidence_model,
                    feature_set=str(args.calibrated_correspondence_feature_set),
                )
                candidate_pnp_matches_for_table = list(pnp_matches)
            if str(args.pnp_soft_order_mode) != "none":
                pnp_matches = soft_order_pnp_matches(
                    pnp_matches,
                    mode=str(args.pnp_soft_order_mode),
                    max_matches=None if int(args.pnp_soft_order_top_n) <= 0 else int(args.pnp_soft_order_top_n),
                )
            apply_coverage_filter = int(args.coverage_filter_grid) > 0
            if str(args.pose_update_filter_schedule) == "never":
                apply_coverage_filter = False
            elif (
                str(args.pose_update_filter_schedule) == "final_only"
                and requested_iterations > 1
                and int(iteration_index) + 1 < requested_iterations
            ):
                apply_coverage_filter = False
            if apply_coverage_filter:
                pnp_matches = coverage_preserving_match_filter(
                    pnp_matches,
                    camera,
                    grid_size=int(args.coverage_filter_grid),
                    max_per_cell=int(args.coverage_filter_max_per_cell),
                    min_confidence=(
                        None
                        if float(args.coverage_filter_min_confidence) < 0.0
                        else float(args.coverage_filter_min_confidence)
                    ),
                    max_total=None if int(args.coverage_filter_max_total) <= 0 else int(args.coverage_filter_max_total),
                )
            render_xy_by_match = {int(match.render_index): np.asarray(match.render_xy, dtype=np.float64) for match in kp_matches}
            pnp, pose_candidate_label, pose_candidate_score, pose_pnp_matches = _estimate_pose_with_optional_rescore(
                pnp_matches,
                camera,
                enable_rescore=bool(args.enable_pose_rescore),
                pnp_reprojection_error_px=float(args.pnp_reprojection_error_px),
                pnp_iterations=int(args.pnp_iterations),
                pnp_min_inliers=int(args.pnp_min_inliers),
                max_matches=int(args.max_matches),
                rescore_margin=float(args.pose_rescore_margin),
            )
            unique_query_refit_applied = False
            unique_query_refit_inlier_count = None
            unique_query_refit_input_inlier_count = None
            if bool(args.pnp_unique_query_refit) and pnp.success and pnp.pose_w2c is not None:
                unique_query_refit_input_inlier_count = int(pnp.inlier_count)
                refit_pnp = refit_pose_with_unique_query_inliers(
                    pose_pnp_matches,
                    camera,
                    pnp.pose_w2c,
                    pnp.inlier_mask,
                    min_inliers=int(args.pnp_min_inliers),
                    refine_method="LM",
                )
                if refit_pnp.success:
                    pnp = refit_pnp
                    unique_query_refit_applied = True
                    unique_query_refit_inlier_count = int(refit_pnp.inlier_count)
            iteration_pose_score = score_pose_hypothesis(
                pose_pnp_matches,
                pnp.pose_w2c if pnp.success else None,
                camera,
                inlier_threshold_px=float(args.pnp_reprojection_error_px),
                inlier_mask=pnp.inlier_mask,
            )
            return {
                "status": "ok",
                "render_pose": current_render_pose,
                "iteration_index": int(iteration_index),
                "render_rgb": render_rgb,
                "render_depth": render_depth,
                "render_alpha": render_alpha,
                "query_feature": query_feature,
                "render_feature": render_feature,
                "query_xy": query_xy,
                "render_xy": render_xy,
                "qdesc": qdesc,
                "rdesc": rdesc,
                "kp_matches": kp_matches,
                "render_side_local_offset_expanded_match_count": int(render_side_local_offset_expanded_match_count),
                "post_pair_render_refine_count": int(post_pair_render_refine_count),
                "unfiltered_pnp_match_count": unfiltered_pnp_match_count,
                "coverage_filter_applied": bool(apply_coverage_filter),
                "candidate_pnp_matches_for_table": candidate_pnp_matches_for_table,
                "pnp_matches": pnp_matches,
                "render_xy_by_match": render_xy_by_match,
                "pnp": pnp,
                "unique_query_refit_applied": bool(unique_query_refit_applied),
                "unique_query_refit_input_inlier_count": unique_query_refit_input_inlier_count,
                "unique_query_refit_inlier_count": unique_query_refit_inlier_count,
                "pose_candidate_label": pose_candidate_label,
                "pose_candidate_score": pose_candidate_score,
                "iteration_pose_score": iteration_pose_score,
                "iteration_alignment_score": float(iteration_alignment_score),
                "pose_pnp_matches": pose_pnp_matches,
                "fine_pair_stats": _pair_fine_logit_stats(pair_fine_logits),
                "query_candidate_cell_count": (
                    int(query_candidate_indices.shape[0]) if query_candidate_indices is not None else int(query_feature.shape[1] * query_feature.shape[2])
                ),
                "render_candidate_cell_count": (
                    int(render_candidate_indices.shape[0]) if render_candidate_indices is not None else int(render_feature.shape[1] * render_feature.shape[2])
                ),
            }

        candidate_finals: list[dict[str, object]] = []
        candidate_iteration_results: list[list[dict[str, object]]] = []
        for initial_idx, initial_render_pose in enumerate(initial_render_poses):
            iteration_results: list[dict[str, object]] = []
            current_render_pose = initial_render_pose
            for iteration_index in range(requested_iterations):
                iteration = _run_pose_iteration(current_render_pose, iteration_index)
                iteration["initial_render_index"] = int(initial_idx)
                iteration_results.append(iteration)
                if iteration.get("status") != "ok":
                    break
                pnp_iter = iteration["pnp"]
                if iteration_index + 1 >= requested_iterations or not pnp_iter.success or pnp_iter.pose_w2c is None:
                    break
                t_error, r_error = render_pose_error_fields(pnp_iter.pose_w2c, gt.pose_w2c)
                current_render_pose = RenderPoseSelection(
                    pose_w2c=pnp_iter.pose_w2c,
                    label=f"{current_render_pose.label}:pose_update:{iteration_index + 1}",
                    candidate_id=current_render_pose.candidate_id,
                    reference_image=current_render_pose.reference_image,
                    render_translation_error_m=t_error,
                    render_rotation_error_deg=r_error,
                )
            candidate_final, _candidate_selection_label = _select_pose_update_iteration(
                iteration_results,
                mode=str(args.pose_update_selection),
                score_margin=float(args.pose_update_score_margin),
            )
            candidate_final["initial_render_index"] = int(initial_idx)
            candidate_final["initial_render_pose_label"] = str(initial_render_pose.label)
            candidate_final["initial_render_candidate_id"] = str(initial_render_pose.candidate_id)
            candidate_final["initial_render_reference_image"] = str(initial_render_pose.reference_image)
            candidate_final["candidate_pose_update_selection_label"] = str(_candidate_selection_label)
            candidate_final["candidate_iteration_results"] = iteration_results
            candidate_finals.append(candidate_final)
            candidate_iteration_results.append(iteration_results)

        if pose_scorer_model is not None and candidate_finals:
            candidate_finals = score_pose_candidates_with_model(
                candidate_finals,
                pose_scorer_model,
                feature_names=pose_scorer_feature_names,
            )
        query_pose_candidate_rows: list[dict[str, object]] = []
        if pose_candidate_table_path is not None:
            query_pose_candidate_rows = _pose_candidate_table_rows_for_query(
                query_id=str(record.image_id),
                render_pose_mode=str(args.render_pose_mode),
                candidates=candidate_finals,
                gt_pose_w2c=gt.pose_w2c,
            )
        topk_selection_mode = (
            "best_score"
            if pose_scorer_model is not None or str(args.render_pose_mode) in {"reference_top5", "reference_top10"}
            else str(args.pose_update_selection)
        )
        final, pose_update_selection_label = _select_pose_update_iteration(
            candidate_finals,
            mode=topk_selection_mode,
            score_margin=float(args.pose_update_score_margin),
        )
        if str(args.render_pose_mode) in {"reference_top5", "reference_top10"}:
            pose_update_selection_label = f"reference_top{int(final.get('initial_render_index', 0)) + 1}:{pose_update_selection_label}"
        iteration_results = list(final.get("candidate_iteration_results", []))
        if final.get("status") == "descriptor_dim_mismatch":
            rows.append(
                {
                    "query_id": record.image_id,
                    "status": "descriptor_dim_mismatch",
                    "query_dim": int(final["query_dim"]),
                    "render_dim": int(final["render_dim"]),
                    "pnp_success": False,
                    "pose_update_iterations_requested": requested_iterations,
                    "pose_update_iterations_completed": int(len(iteration_results)),
                    "pose_update_selection": str(args.pose_update_selection),
                    "pose_update_selection_label": pose_update_selection_label,
                }
            )
            continue
        render_pose = final["render_pose"]
        render_rgb = final["render_rgb"]
        render_depth = final["render_depth"]
        render_alpha = final["render_alpha"]
        query_feature = final["query_feature"]
        render_feature = final["render_feature"]
        query_xy = final["query_xy"]
        render_xy = final["render_xy"]
        qdesc = final["qdesc"]
        rdesc = final["rdesc"]
        kp_matches = final["kp_matches"]
        render_side_local_offset_expanded_match_count = int(final.get("render_side_local_offset_expanded_match_count", 0))
        unfiltered_pnp_match_count = int(final["unfiltered_pnp_match_count"])
        candidate_pnp_matches_for_table = list(final.get("candidate_pnp_matches_for_table", []))
        pnp_matches = final["pnp_matches"]
        render_xy_by_match = final["render_xy_by_match"]
        pnp = final["pnp"]
        unique_query_refit_applied = bool(final.get("unique_query_refit_applied", False))
        unique_query_refit_input_inlier_count = final.get("unique_query_refit_input_inlier_count")
        unique_query_refit_inlier_count = final.get("unique_query_refit_inlier_count")
        pose_candidate_label = final["pose_candidate_label"]
        pose_candidate_score = final["pose_candidate_score"]
        pose_pnp_matches = final["pose_pnp_matches"]
        fine_pair_stats = dict(final.get("fine_pair_stats", {}))
        translation_error, rotation_error = _pose_metrics(pnp.pose_w2c if pnp.success else None, gt.pose_w2c)
        pnp_render_translation_delta, pnp_render_rotation_delta = _pose_metrics(
            pnp.pose_w2c if pnp.success else None,
            render_pose.pose_w2c,
        )
        pose_update_trace = []
        for iteration in iteration_results:
            iteration_pose = iteration.get("render_pose")
            iteration_pnp = iteration.get("pnp")
            iteration_score = iteration.get("iteration_pose_score")
            iteration_translation_error, iteration_rotation_error = (
                _pose_metrics(iteration_pnp.pose_w2c if iteration_pnp.success else None, gt.pose_w2c)
                if iteration_pnp is not None
                else (None, None)
            )
            pose_update_trace.append(
                {
                    "iteration": int(iteration.get("iteration_index", len(pose_update_trace))),
                    "status": str(iteration.get("status", "")),
                    "render_pose_label": None if iteration_pose is None else str(iteration_pose.label),
                    "render_translation_error_m": (
                        None if iteration_pose is None else iteration_pose.render_translation_error_m
                    ),
                    "render_rotation_error_deg": (
                        None if iteration_pose is None else iteration_pose.render_rotation_error_deg
                    ),
                    "pnp_success": bool(iteration_pnp.success) if iteration_pnp is not None else False,
                    "pnp_inlier_count": int(iteration_pnp.inlier_count) if iteration_pnp is not None else 0,
                    "coverage_filter_applied": bool(iteration.get("coverage_filter_applied", False)),
                    "pose_score": None if iteration_score is None else float(iteration_score.score),
                    "alignment_score": _iteration_alignment_score_value(iteration),
                    "pose_score_inlier_count": None if iteration_score is None else int(iteration_score.inlier_count),
                    "pose_score_coverage": None if iteration_score is None else float(iteration_score.coverage),
                    "pose_score_weighted_residual": (
                        None if iteration_score is None else float(iteration_score.weighted_residual)
                    ),
                    "translation_error_m": iteration_translation_error,
                    "rotation_error_deg": iteration_rotation_error,
                }
            )
        geometry = (
            reprojection_error_stats(
                pose_pnp_matches,
                gt.pose_w2c,
                camera,
                thresholds_px=(5.0, 10.0, 16.0, 32.0),
                pnp_inlier_mask=pnp.inlier_mask,
            )
            if pose_pnp_matches
            else {}
        )
        gt_errors = (
            match_reprojection_errors(pose_pnp_matches, gt.pose_w2c, camera)
            if pose_pnp_matches
            else np.zeros((0,), dtype=np.float64)
        )
        pnp_pose_errors = (
            match_reprojection_errors(pose_pnp_matches, pnp.pose_w2c, camera)
            if pose_pnp_matches and pnp.success and pnp.pose_w2c is not None
            else None
        )
        pnp_match_diagnostics = _pnp_match_diagnostics(pose_pnp_matches, pnp.inlier_mask)
        fine_offset_diagnostics = _fine_offset_diagnostics(
            pose_pnp_matches,
            gt.pose_w2c,
            camera,
            query_grid_width=int(query_feature.shape[2]),
            query_grid_height=int(query_feature.shape[1]),
        )
        residual_solver_diagnostics = (
            _run_render_pose_residual_diagnostic(
                matches=pose_pnp_matches,
                gt_errors=gt_errors,
                render_pose_w2c=render_pose.pose_w2c,
                gt_pose_w2c=gt.pose_w2c,
                camera=camera,
                oracle_threshold_px=float(args.residual_solver_oracle_threshold_px),
                oracle_thresholds_px=[
                    float(item)
                    for item in str(args.residual_solver_oracle_thresholds_px).split(",")
                    if str(item).strip()
                ],
                max_iterations=int(args.residual_solver_iterations),
            )
            if bool(args.enable_render_pose_residual_solver)
            else {}
        )
        if bool(args.save_coarse_oracle_table) or bool(args.save_coarse_oracle_candidate_table):
            query_depth_config = GaussianVFMRenderConfig(
                width=int(camera.width),
                height=int(camera.height),
                radius_px=float(args.render_radius_px),
                depth_epsilon=float(args.render_depth_epsilon),
                l2_normalize_pixels=True,
            )
            _query_gt_rgb, query_gt_depth, _query_gt_alpha = _render_rgb_and_depth(
                rgb_source,
                depth_field,
                pose_w2c=gt.pose_w2c,
                camera=camera,
                config=query_depth_config,
                renderer=args.renderer,
                device=args.device,
            )
            oracle_render_indices = _oracle_render_indices_from_query_gt_depth(
                query_feature,
                query_gt_depth,
                query_camera=camera,
                query_pose_w2c=gt.pose_w2c,
                render_camera=render_camera,
                render_pose_w2c=render_pose.pose_w2c,
                render_grid_width=int(render_feature.shape[2]),
                render_grid_height=int(render_feature.shape[1]),
            )
            if bool(args.save_coarse_oracle_table):
                query_oracle_rows = coarse_oracle_rank_rows(
                    query_feature,
                    render_feature,
                    oracle_render_indices=oracle_render_indices,
                    query_image_width=int(camera.width),
                    query_image_height=int(camera.height),
                    render_image_width=int(render_config.width),
                    render_image_height=int(render_config.height),
                    logit_scale=float(args.dual_softmax_logit_scale),
                )
                for oracle_row in query_oracle_rows:
                    oracle_row.update(
                        {
                            "query_id": record.image_id,
                            "render_pose_mode": str(args.render_pose_mode),
                            "render_pose_label": render_pose.label,
                            "render_translation_error_m": render_pose.render_translation_error_m,
                            "render_rotation_error_deg": render_pose.render_rotation_error_deg,
                            "query_feature_shape": json.dumps(list(query_feature.shape)),
                            "render_feature_shape": json.dumps(list(render_feature.shape)),
                        }
                    )
                coarse_oracle_accumulator.update(query_oracle_rows)
                if coarse_oracle_writer is not None:
                    coarse_oracle_writer.writerows(query_oracle_rows)
            if bool(args.save_coarse_oracle_candidate_table):
                query_candidate_rows = coarse_oracle_candidate_rows(
                    query_feature,
                    render_feature,
                    oracle_render_indices=oracle_render_indices,
                    top_k=int(args.coarse_oracle_candidate_top_k),
                    positive_radius=int(args.coarse_oracle_candidate_positive_radius),
                    query_image_width=int(camera.width),
                    query_image_height=int(camera.height),
                    render_image_width=int(render_config.width),
                    render_image_height=int(render_config.height),
                    logit_scale=float(args.dual_softmax_logit_scale),
                )
                for candidate_row in query_candidate_rows:
                    candidate_row.update(
                        {
                            "query_id": record.image_id,
                            "render_pose_mode": str(args.render_pose_mode),
                            "render_pose_label": render_pose.label,
                            "render_translation_error_m": render_pose.render_translation_error_m,
                            "render_rotation_error_deg": render_pose.render_rotation_error_deg,
                            "query_feature_shape": json.dumps(list(query_feature.shape)),
                            "render_feature_shape": json.dumps(list(render_feature.shape)),
                        }
                    )
                coarse_oracle_candidate_row_count += len(query_candidate_rows)
                coarse_oracle_candidate_positive_count += int(
                    np.sum([bool(row.get("patch_correct", False)) for row in query_candidate_rows])
                )
                if coarse_oracle_candidate_writer is not None:
                    coarse_oracle_candidate_writer.writerows(query_candidate_rows)
        if bool(args.save_match_table):
            if str(args.match_table_stage) == "candidate":
                table_matches = list(candidate_pnp_matches_for_table)
                table_gt_errors = (
                    match_reprojection_errors(table_matches, gt.pose_w2c, camera)
                    if table_matches
                    else np.zeros((0,), dtype=np.float64)
                )
                table_baseline_errors = (
                    match_reprojection_errors(table_matches, pnp.pose_w2c, camera)
                    if table_matches and pnp.success and pnp.pose_w2c is not None
                    else None
                )
                table_inlier_mask = (
                    table_baseline_errors <= float(args.pnp_reprojection_error_px)
                    if table_baseline_errors is not None
                    else None
                )
            else:
                table_matches = pose_pnp_matches
                table_gt_errors = gt_errors
                table_baseline_errors = pnp_pose_errors
                table_inlier_mask = pnp.inlier_mask
            match_table_rows.extend(
                _match_table_rows_for_query(
                    query_id=record.image_id,
                    matches=table_matches,
                    gt_errors=table_gt_errors,
                    gt_stride_px=0.5
                    * (
                        float(camera.width) / max(float(query_feature.shape[2]), 1.0)
                        + float(camera.height) / max(float(query_feature.shape[1]), 1.0)
                    ),
                    inlier_mask=table_inlier_mask,
                    baseline_reproj_errors=table_baseline_errors,
                    render_xy_by_match=render_xy_by_match,
                )
            )
        if vis_idx < int(args.visualize_limit):
            _draw_matches(
                query_image,
                render_rgb,
                pose_pnp_matches,
                gt_errors,
                render_xy_by_match,
                output_dir / "visualizations" / f"{_safe_image_stem(record.image_id)}_matches.png",
            )
        row = {
            "query_id": record.image_id,
            "status": "ok",
            "render_pose_mode": str(args.render_pose_mode),
            "render_pose_label": render_pose.label,
            "initial_render_pose_label": str(final.get("initial_render_pose_label", iteration_results[0]["render_pose"].label)),
            "initial_render_candidate_id": str(final.get("initial_render_candidate_id", "")),
            "initial_render_reference_image": str(final.get("initial_render_reference_image", "")),
            "render_candidate_id": render_pose.candidate_id,
            "render_reference_image": render_pose.reference_image,
            "render_translation_error_m": render_pose.render_translation_error_m,
            "render_rotation_error_deg": render_pose.render_rotation_error_deg,
            "pose_update_iterations_requested": requested_iterations,
            "pose_update_iterations_completed": int(len(iteration_results)),
            "pose_update_selection": str(args.pose_update_selection),
            "pose_update_selection_label": pose_update_selection_label,
            "pose_update_selected_iteration": int(final.get("iteration_index", -1)),
            "pose_update_selected_score": _iteration_pose_score_value(final),
            "pose_update_selected_alignment_score": _iteration_alignment_score_value(final),
            "pose_update_trace": json.dumps(pose_update_trace),
            "query_keypoint_count": int(query_xy.shape[0]),
            "render_keypoint_count": int(render_xy.shape[0]),
            "query_candidate_cell_count": int(final.get("query_candidate_cell_count", query_xy.shape[0])),
            "render_candidate_cell_count": int(final.get("render_candidate_cell_count", render_xy.shape[0])),
            "valid_query_descriptor_count": int(qdesc.shape[0]),
            "valid_render_descriptor_count": int(rdesc.shape[0]),
            "render_width": int(render_width),
            "render_height": int(render_height),
            "query_feature_shape": json.dumps(list(query_feature.shape)),
            "render_feature_shape": json.dumps(list(render_feature.shape)),
            "render_alpha_mean": float(np.mean(render_alpha)),
            "render_depth_valid_fraction": float(np.mean(np.isfinite(render_depth) & (render_depth > 0.0))),
            "match_count": int(len(kp_matches)),
            "render_side_local_offset_expanded_match_count": int(render_side_local_offset_expanded_match_count),
            "post_pair_render_refine_count": int(final.get("post_pair_render_refine_count", 0)),
            "unfiltered_depth_valid_match_count": int(unfiltered_pnp_match_count),
            "depth_valid_match_count": int(len(pnp_matches)),
            "pose_candidate_match_count": int(len(pose_pnp_matches)),
            "pnp_success": bool(pnp.success),
            "pnp_inlier_count": int(pnp.inlier_count),
            "pnp_inlier_ratio": float(pnp.inlier_ratio),
            "unique_query_refit_applied": bool(unique_query_refit_applied),
            "unique_query_refit_input_inlier_count": unique_query_refit_input_inlier_count,
            "unique_query_refit_inlier_count": unique_query_refit_inlier_count,
                "pose_candidate_label": pose_candidate_label,
                "pose_candidate_score": None if pose_candidate_score is None else float(pose_candidate_score.score),
                "learned_pose_probability": (
                    None
                    if final.get("learned_pose_probability") is None
                    else float(final.get("learned_pose_probability"))
                ),
                "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
            "pnp_render_translation_delta_m": pnp_render_translation_delta,
            "pnp_render_rotation_delta_deg": pnp_render_rotation_delta,
            "mean_dual_softmax_confidence": (
                None
                if not kp_matches or kp_matches[0].dual_softmax_confidence is None
                else float(np.mean([float(match.dual_softmax_confidence or 0.0) for match in kp_matches]))
            ),
            **fine_pair_stats,
            **pnp_match_diagnostics,
            **fine_offset_diagnostics,
            **residual_solver_diagnostics,
            **_geometry_row_fields(geometry),
        }
        rows.append(row)
        if row_writer is not None:
            row_writer.writerows([row])
        if pose_candidate_table_path is not None:
            if pose_candidate_table_writer is not None:
                pose_candidate_table_writer.writerows(query_pose_candidate_rows)
            else:
                pose_candidate_table_rows.extend(query_pose_candidate_rows)
        if render_cache_writer is not None:
            new_render_cache_rows = render_cache_rows[render_cache_written_count:]
            render_cache_writer.writerows(new_render_cache_rows)
    if row_writer is not None:
        row_writer.close()
        rows = _read_csv_rows(rows_path)
    else:
        _write_csv(rows_path, rows)
    if render_cache_writer is not None:
        render_cache_writer.close()
        render_cache_rows = _read_csv_rows(render_cache_manifest_path)
    else:
        _write_csv(render_cache_manifest_path, render_cache_rows)
    if pose_candidate_table_writer is not None:
        pose_candidate_table_writer.close()
        pose_candidate_table_rows = _read_csv_rows(pose_candidate_table_path) if pose_candidate_table_path else []
    elif pose_candidate_table_path is not None:
        _write_csv(pose_candidate_table_path, pose_candidate_table_rows)
    match_table_path = None
    if bool(args.save_match_table):
        match_table_path = Path(args.match_table_path) if str(args.match_table_path) else output_dir / "match_table.csv"
        _write_csv(match_table_path, match_table_rows)
    if coarse_oracle_writer is not None:
        coarse_oracle_writer.close()
    if coarse_oracle_candidate_writer is not None:
        coarse_oracle_candidate_writer.close()
    summary = {
        "stage": "render_rgb_feature_keypoint_pose",
        "elapsed_sec": float(time.perf_counter() - started),
        "inputs": {
            "query_manifest": str(args.query_manifest),
            "query_pose_file": str(args.query_pose_file),
            "image_root": str(args.image_root),
            "gaussian_rgb_ply": str(args.gaussian_rgb_ply),
            "selector_checkpoint": str(args.selector_checkpoint),
            "matcha_adapter_checkpoint": str(args.matcha_adapter_checkpoint),
            "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
            "matcha_qkv_attention_checkpoint": str(args.matcha_qkv_attention_checkpoint),
            "query_token_cache_dir": str(args.query_token_cache_dir),
            "render_token_cache_dir": str(args.render_token_cache_dir),
            "render_rgb_depth_cache_dir": str(args.render_rgb_depth_cache_dir),
            "extract_query_features_from_image": bool(args.extract_query_features_from_image),
            "candidate_bank": str(args.candidate_bank),
        },
        "camera": {
            "source": camera_source,
            "width": int(camera.width),
            "height": int(camera.height),
            "base_render_width": int(base_render_width),
            "base_render_height": int(base_render_height),
            "render_width": int(render_width),
            "render_height": int(render_height),
        },
        "config": {
            "detector": args.detector,
            "match_mode": args.match_mode,
            "max_keypoints": int(args.max_keypoints),
            "ratio_threshold": float(args.ratio_threshold),
            "dual_softmax_logit_scale": float(args.dual_softmax_logit_scale),
            "min_dual_softmax_confidence": float(args.min_dual_softmax_confidence),
            "min_similarity": float(args.min_similarity),
            "matcha_coarse_top_k_per_query": int(args.matcha_coarse_top_k_per_query),
            "matcha_coarse_mutual_mode": str(args.matcha_coarse_mutual_mode),
            "matcha_coarse_local_window_radius_cells": int(args.matcha_coarse_local_window_radius_cells),
            "matcha_post_confidence_top_k_per_query": int(args.matcha_post_confidence_top_k_per_query),
            "coarse_candidate_ranker_model": str(args.coarse_candidate_ranker_model),
            "coarse_candidate_ranker_feature_set": str(args.coarse_candidate_ranker_feature_set),
            "coarse_candidate_ranker_blend": float(args.coarse_candidate_ranker_blend),
            "fine_render_search_radius_px": float(args.fine_render_search_radius_px),
            "fine_render_search_step_px": float(args.fine_render_search_step_px),
            "matcha_fine_mode": str(args.matcha_fine_mode),
            "matcha_fine_softmax_temperature": float(args.matcha_fine_softmax_temperature),
            "matcha_eval_preset": str(args.matcha_eval_preset),
            "matcha_confidence_mode": str(args.matcha_confidence_mode),
            "matcha_confidence_blend": float(args.matcha_confidence_blend),
            "matcha_local_window_confidence_blend": float(args.matcha_local_window_confidence_blend),
            "matcha_min_detector_confidence": float(args.matcha_min_detector_confidence),
            "matcha_min_keypoint_confidence": float(args.matcha_min_keypoint_confidence),
            "matcha_cell_offset_side": str(args.matcha_cell_offset_side),
            "matcha_use_pair_fine_head": bool(args.matcha_use_pair_fine_head),
            "matcha_pair_fine_side": str(args.matcha_pair_fine_side),
            "matcha_pair_fine_coordinate_mode": str(args.matcha_pair_fine_coordinate_mode),
            "matcha_use_local_fine_attention": bool(args.matcha_use_local_fine_attention),
            "matcha_use_local_window_fine_head": bool(args.matcha_use_local_window_fine_head),
            "matcha_use_patch_corr_fine_head": bool(args.matcha_use_patch_corr_fine_head),
            "matcha_patch_corr_target_side": str(args.matcha_patch_corr_target_side),
            "matcha_patch_corr_confidence_blend": float(args.matcha_patch_corr_confidence_blend),
            "post_pair_render_refine_radius_px": float(args.post_pair_render_refine_radius_px),
            "post_pair_render_refine_step_px": float(args.post_pair_render_refine_step_px),
            "post_pair_render_refine_mode": str(args.post_pair_render_refine_mode),
            "post_pair_render_refine_query_sigma_px": float(args.post_pair_render_refine_query_sigma_px),
            "render_side_local_offset_radius_cells": int(args.render_side_local_offset_radius_cells),
            "render_side_local_offset_max_candidates": int(args.render_side_local_offset_max_candidates),
            "render_side_local_offset_top_k_per_query": int(args.render_side_local_offset_top_k_per_query),
            "matcha_keypoint_proposal_source": str(args.matcha_keypoint_proposal_source),
            "matcha_keypoint_proposal_mode": str(args.matcha_keypoint_proposal_mode),
            "matcha_keypoint_proposal_threshold": float(args.matcha_keypoint_proposal_threshold),
            "matcha_keypoint_proposal_top_k": int(args.matcha_keypoint_proposal_top_k),
            "matcha_keypoint_prior_boost": float(args.matcha_keypoint_prior_boost),
            "matcha_keypoint_prior_penalty": float(args.matcha_keypoint_prior_penalty),
            "matcha_reliability_prior_source": str(args.matcha_reliability_prior_source),
            "matcha_reliability_prior_boost": float(args.matcha_reliability_prior_boost),
            "matcha_reliability_prior_penalty": float(args.matcha_reliability_prior_penalty),
            "matcha_rgb_keypoint_detector_checkpoint": str(args.matcha_rgb_keypoint_detector_checkpoint),
            "alike_repo": str(args.alike_repo),
            "alike_model": str(args.alike_model),
            "alike_top_k": int(args.alike_top_k),
            "alike_scores_th": float(args.alike_scores_th),
            "alike_n_limit": int(args.alike_n_limit),
            "feature_fusion_mode": str(args.feature_fusion_mode),
            "feature_fusion_radius": int(args.feature_fusion_radius),
            "feature_fusion_temperature": float(args.feature_fusion_temperature),
            "feature_fusion_alpha": float(args.feature_fusion_alpha),
            "measurement_sigma_px": float(args.measurement_sigma_px),
            "pnp_soft_order_mode": str(args.pnp_soft_order_mode),
            "pnp_soft_order_top_n": int(args.pnp_soft_order_top_n),
            "calibrated_correspondence_confidence_model": str(args.calibrated_correspondence_confidence_model),
            "calibrated_correspondence_feature_set": str(args.calibrated_correspondence_feature_set),
            "pose_scorer_model": str(args.pose_scorer_model),
            "pnp_unique_query_refit": bool(args.pnp_unique_query_refit),
            "render_offset_min_alpha": float(args.render_offset_min_alpha),
            "render_offset_max_depth_delta_m": float(args.render_offset_max_depth_delta_m),
            "render_offset_fallback_to_cell_center": bool(args.render_offset_fallback_to_cell_center),
            "coverage_filter_grid": int(args.coverage_filter_grid),
            "coverage_filter_max_per_cell": int(args.coverage_filter_max_per_cell),
            "coverage_filter_min_confidence": float(args.coverage_filter_min_confidence),
            "coverage_filter_max_total": int(args.coverage_filter_max_total),
            "enable_pose_rescore": bool(args.enable_pose_rescore),
            "pose_rescore_margin": float(args.pose_rescore_margin),
            "enable_render_pose_residual_solver": bool(args.enable_render_pose_residual_solver),
            "residual_solver_oracle_threshold_px": float(args.residual_solver_oracle_threshold_px),
            "residual_solver_oracle_thresholds_px": str(args.residual_solver_oracle_thresholds_px),
            "residual_solver_iterations": int(args.residual_solver_iterations),
            "pose_update_iterations": int(args.pose_update_iterations),
            "pose_update_selection": str(args.pose_update_selection),
            "pose_update_score_margin": float(args.pose_update_score_margin),
            "pose_update_filter_schedule": str(args.pose_update_filter_schedule),
            "renderer": args.renderer,
            "render_pose_mode": str(args.render_pose_mode),
            "render_pose_world_offset": str(args.render_pose_world_offset),
            "render_pose_rotation_offset_deg": str(args.render_pose_rotation_offset_deg),
            "render_pose_rotation_search_offsets_deg": str(args.render_pose_rotation_search_offsets_deg),
            "render_pose_rotation_search_axis": str(args.render_pose_rotation_search_axis),
            "render_canvas_scale": float(args.render_canvas_scale),
            "save_match_table": bool(args.save_match_table),
            "match_table_stage": str(args.match_table_stage),
            "save_pose_candidate_table": bool(args.save_pose_candidate_table),
            "save_coarse_oracle_table": bool(args.save_coarse_oracle_table),
            "save_coarse_oracle_candidate_table": bool(args.save_coarse_oracle_candidate_table),
            "coarse_oracle_candidate_top_k": int(args.coarse_oracle_candidate_top_k),
            "coarse_oracle_candidate_positive_radius": int(args.coarse_oracle_candidate_positive_radius),
            "stream_rows": bool(args.stream_rows),
            "resume_existing_rows": bool(args.resume_existing_rows),
            "rebuild_summary_from_rows": bool(args.rebuild_summary_from_rows),
            "start_index": int(args.start_index),
            "view_selection": str(args.view_selection),
            "radio_version": args.radio_version,
        },
        "metrics": _summary(rows),
        "outputs": {
            "rows": str(rows_path),
            "summary": str(summary_path),
            "visualizations": str(output_dir / "visualizations"),
            "render_cache_manifest": str(render_cache_manifest_path),
        },
    }
    summary["metrics"]["render_cache_manifest_row_count"] = int(len(render_cache_rows))
    if match_table_path is not None:
        summary["outputs"]["match_table"] = str(match_table_path)
        summary["metrics"]["match_table_row_count"] = int(len(match_table_rows))
    if pose_candidate_table_path is not None:
        summary["outputs"]["pose_candidate_table"] = str(pose_candidate_table_path)
        summary["metrics"]["pose_candidate_table_row_count"] = int(len(pose_candidate_table_rows))
    if coarse_oracle_table_path is not None:
        summary["outputs"]["coarse_oracle_table"] = str(coarse_oracle_table_path)
        coarse_summary = coarse_oracle_accumulator.summary()
        summary["metrics"].update({f"coarse_oracle_{key}": value for key, value in coarse_summary.items()})
    if coarse_oracle_candidate_table_path is not None:
        summary["outputs"]["coarse_oracle_candidate_table"] = str(coarse_oracle_candidate_table_path)
        summary["metrics"]["coarse_oracle_candidate_table_row_count"] = int(coarse_oracle_candidate_row_count)
        if int(coarse_oracle_candidate_row_count) > 0:
            summary["metrics"]["coarse_oracle_candidate_positive_rate"] = float(
                float(coarse_oracle_candidate_positive_count) / float(coarse_oracle_candidate_row_count)
            )
    summary["metrics"].update(_render_lock_diagnostics_from_rows(rows))
    summary["metrics"].update(_fine_confidence_diagnostics_from_rows(rows))
    summary["metrics"].update(_residual_solver_diagnostics_from_rows(rows))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
