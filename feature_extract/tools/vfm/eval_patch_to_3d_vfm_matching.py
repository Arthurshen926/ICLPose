"""Evaluate patch-level query-token to sparse 3D VFM landmark-set matching."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _limit_submap,
    _load_camera_with_source,
    _load_query_feature,
    _load_reference_submaps,
    _load_track_stats,
    _mean,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c, parse_cambridge_pose_file
from feature_extract.vfm.correspondence_confidence import (
    CalibratedMatchCandidateScorer,
    CalibratedLogisticConfidence,
    annotate_matches_with_calibrated_confidence,
)
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex, count_projected_landmarks, filter_landmarks_by_visibility
from feature_extract.vfm.lowlevel_offset_sidecar import (
    HLocSuperPointKeypointDetector,
    LandmarkConditionedKeypointSelectorConfig,
    LowLevelOffsetSidecarConfig,
    LowLevelSupportBank,
    SuperPointSnapConfig,
    apply_landmark_conditioned_superpoint_snaps_to_matches,
    build_landmark_conditioned_superpoint_candidate_rows,
    apply_superpoint_gt_oracle_snaps_to_matches,
    apply_superpoint_snaps_to_matches,
    apply_lowlevel_offsets_to_matches,
    select_support_observation,
    select_support_superpoint_keypoint,
    superpoint_availability_summary,
)
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_selector_training import SafePairwiseInlierScorer
from feature_extract.vfm.patch_to_3d_matching import (
    PatchTo3DMatchingConfig,
    build_patch_positive_sets,
    evaluate_patch_matches,
    filter_patch_correct_matches,
    filter_landmarks_by_projected_visibility,
    match_query_patches_to_landmarks,
    oracle_patch_positive_matches,
    patch_uncertainty_pnp_threshold,
    patch_positive_set_stats,
)
from feature_extract.vfm.patch_offset_refiner import (
    apply_predicted_patch_offsets,
    load_patch_offset_refiner_checkpoint,
    predict_patch_offsets_for_matches,
    refine_matches_with_oracle_offsets,
)
from feature_extract.vfm.rendered_map_verifier import project_xyz_to_image
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkAmbiguityPruningConfig,
    LandmarkQualityConfig,
    LandmarkMapIndex,
    LocalGeometricConsistencyConfig,
    MapReliabilityConfig,
    PoseRiskConfig,
    SpatialDiversityPnPConfig,
    estimate_pose_pnp_ransac,
    estimate_pose_pnp_fixed,
    estimate_pose_pnp_fixed_robust,
    filter_landmarks_by_reference_images,
    match_reprojection_errors,
    match_spatial_distribution_stats,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
    pose_risk_score,
    selective_localization_summary,
    select_pnp_matches_by_spatial_diversity,
    select_pnp_matches_by_map_reliability,
    soft_order_pnp_matches,
    with_landmark_ambiguity_scores,
)
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap, filter_semidense_by_source_visibility
from feature_extract.vfm.tokens import TokenBankManifest


def _read_image_rgb(path: Path) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for low-level offset sidecar") from exc
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _none_to_float(value):
    return None if value is None else float(value)


def _median_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not present else float(np.median(present))


def _quantile_present(values: list[float | None], q: float) -> float | None:
    present = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not present else float(np.quantile(present, float(q)))


def _rate_present(values: list[bool]) -> float | None:
    return None if not values else float(np.mean([1.0 if value else 0.0 for value in values]))


def _offset_summary_mean(rows: Sequence[dict[str, object]], key: str) -> float | None:
    values = []
    for row in rows:
        offset = row.get("patch_offset_refinement", {})
        if not isinstance(offset, dict) or offset.get(key) is None:
            continue
        value = _safe_float(offset.get(key))
        if value is not None:
            values.append(value)
    return _mean(values)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _fixed_pose_refinement_weights(matches: Sequence[object], mode: str) -> np.ndarray | None:
    mode_value = str(mode)
    if mode_value == "uniform":
        return np.ones((len(matches),), dtype=np.float64)
    weights = []
    for match in matches:
        weight = 1.0
        if mode_value in {"reliability", "composite", "offset_composite", "offset_consistency_composite"}:
            reliability = _safe_float(getattr(match, "map_reliability", None))
            if reliability is not None:
                weight *= max(reliability, 0.05)
            quality = _safe_float(getattr(match, "landmark_quality", None))
            if quality is not None:
                weight *= max(quality, 0.05)
            scale = _safe_float(getattr(match, "pnp_uncertainty_scale", None))
            if scale is not None:
                weight /= max(scale * scale, 1e-3)
        if mode_value in {"match", "composite", "offset_composite", "offset_consistency_composite"}:
            similarity = _safe_float(getattr(match, "similarity", None))
            if similarity is not None:
                weight *= max(similarity, 0.05)
            margin = _safe_float(getattr(match, "similarity_margin", None))
            if margin is not None:
                weight *= 1.0 + min(max(margin, 0.0) * 5.0, 2.0)
            variance = _safe_float(getattr(match, "landmark_variance", None))
            if variance is not None:
                weight /= 1.0 + max(variance, 0.0)
            reproj = _safe_float(getattr(match, "landmark_reprojection_error", None))
            if reproj is not None:
                weight /= 1.0 + max(reproj, 0.0) / 4.0
            ambiguity = _safe_float(getattr(match, "landmark_ambiguity", None))
            if ambiguity is not None:
                weight *= max(1.0 - 0.75 * min(max(ambiguity, 0.0), 1.0), 0.05)
            obs = _safe_float(getattr(match, "observation_count", None))
            if obs is not None:
                weight *= max(np.log1p(max(obs, 0.0)), 1.0)
        if mode_value in {"offset_confidence", "offset_composite", "offset_consistency_composite"}:
            confidence = _safe_float(getattr(match, "patch_offset_confidence", None))
            if confidence is not None:
                confidence = min(max(confidence, 0.0), 1.0)
                weight *= 0.25 + 1.5 * confidence
            sigma = _safe_float(getattr(match, "patch_offset_sigma", None))
            if sigma is not None:
                weight /= max(float(sigma) * float(sigma), 0.25)
            applied = getattr(match, "patch_offset_applied", None)
            if applied is False and confidence is not None and confidence < 0.5:
                weight *= 0.75
        if mode_value == "offset_consistency_composite":
            before = _safe_float(getattr(match, "patch_offset_consistency_before_px", None))
            after = _safe_float(getattr(match, "patch_offset_consistency_after_px", None))
            if before is not None and after is not None:
                increase = max(after - before, 0.0)
                weight /= 1.0 + increase / 2.0
                if after > before:
                    weight *= max(before / max(after, 1e-6), 0.25)
        weights.append(float(np.clip(weight, 0.05, 10.0)))
    return np.asarray(weights, dtype=np.float64)


def _success(row: dict[str, object], translation_m: float, rotation_deg: float) -> bool:
    return bool(
        row["translation_error_m"] is not None
        and float(row["translation_error_m"]) <= float(translation_m)
        and row["rotation_error_deg"] is not None
        and float(row["rotation_error_deg"]) <= float(rotation_deg)
    )


def _matching_config_dict(config: PatchTo3DMatchingConfig) -> dict[str, object]:
    values = dict(config.__dict__)
    quality = values.get("landmark_quality")
    if isinstance(quality, LandmarkQualityConfig):
        values["landmark_quality"] = dict(quality.__dict__)
    reliability = values.get("map_reliability")
    if isinstance(reliability, MapReliabilityConfig):
        values["map_reliability"] = dict(reliability.__dict__)
    local = values.get("local_geometric_consistency")
    if isinstance(local, LocalGeometricConsistencyConfig):
        values["local_geometric_consistency"] = dict(local.__dict__)
    ambiguity = values.get("landmark_ambiguity_pruning")
    if isinstance(ambiguity, LandmarkAmbiguityPruningConfig):
        values["landmark_ambiguity_pruning"] = dict(ambiguity.__dict__)
    return values


def _track_id_set(index: LandmarkMapIndex) -> set[int]:
    return {int(track_id) for track_id in np.asarray(index.track_ids).tolist()}


def _map_reliability_stats(matches, mask: np.ndarray | None = None) -> dict[str, float | int | None]:
    if mask is None:
        selected = np.ones((len(matches),), dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool).reshape(-1)
        if selected.shape[0] != len(matches):
            raise ValueError("map reliability mask must have one value per match")
    values = [
        float(match.map_reliability)
        for match, keep in zip(matches, selected)
        if bool(keep) and match.map_reliability is not None
    ]
    scales = [
        float(match.pnp_uncertainty_scale)
        for match, keep in zip(matches, selected)
        if bool(keep) and match.pnp_uncertainty_scale is not None
    ]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "mean_uncertainty_scale": None,
            "median_uncertainty_scale": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "mean_uncertainty_scale": None if not scales else float(np.mean(scales)),
        "median_uncertainty_scale": None if not scales else float(np.median(scales)),
    }


def _local_consistency_stats(matches, mask: np.ndarray | None = None) -> dict[str, float | int | None]:
    if mask is None:
        selected = np.ones((len(matches),), dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool).reshape(-1)
        if selected.shape[0] != len(matches):
            raise ValueError("local consistency mask must have one value per match")
    supports = [
        int(match.local_consistency_support)
        for match, keep in zip(matches, selected)
        if bool(keep) and match.local_consistency_support is not None
    ]
    scores = [
        float(match.local_consistency_score)
        for match, keep in zip(matches, selected)
        if bool(keep) and match.local_consistency_score is not None
    ]
    if not supports:
        return {
            "count": 0,
            "mean_support": None,
            "median_support": None,
            "mean_score": None,
            "median_score": None,
        }
    return {
        "count": int(len(supports)),
        "mean_support": float(np.mean(supports)),
        "median_support": float(np.median(supports)),
        "mean_score": None if not scores else float(np.mean(scores)),
        "median_score": None if not scores else float(np.median(scores)),
    }


def _full_inlier_mask(matches, pnp_matches, pnp_mask: np.ndarray) -> np.ndarray:
    full = np.zeros((len(matches),), dtype=bool)
    if not matches or not pnp_matches:
        return full
    positions = {id(match): idx for idx, match in enumerate(matches)}
    local_mask = np.asarray(pnp_mask, dtype=bool).reshape(-1)
    for local_idx, pnp_match in enumerate(pnp_matches):
        if local_idx >= local_mask.shape[0] or not bool(local_mask[local_idx]):
            continue
        match_idx = positions.get(id(pnp_match))
        if match_idx is not None:
            full[match_idx] = True
    return full


def _selected_match_indices(matches, pnp_matches) -> list[int]:
    positions = {id(match): idx for idx, match in enumerate(matches)}
    indices = []
    for match in pnp_matches:
        idx = positions.get(id(match))
        if idx is not None:
            indices.append(int(idx))
    return indices


def _load_reference_pose_priors(candidate_bank: str, submap_top_n: int) -> dict[str, list[dict[str, object]]]:
    if not candidate_bank:
        return {}
    if submap_top_n <= 0:
        raise ValueError("submap_top_n must be positive")
    by_query: dict[str, list[dict[str, object]]] = {}
    order = 0
    for line in Path(candidate_bank).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("record_type") == "header":
            continue
        if item.get("record_type", "candidate") != "candidate":
            continue
        query_id = item.get("query_id")
        reference_image = item.get("reference_image")
        if query_id is None or reference_image is None:
            continue
        metadata = dict(item.get("metadata") or {})
        pose_error = dict(item.get("pose_error") or {})
        rank = int(metadata.get("retrieval_rank", len(by_query.get(str(query_id), [])) + 1))
        by_query.setdefault(str(query_id), []).append(
            {
                "rank": rank,
                "order": order,
                "reference_image": str(reference_image),
                "translation_error_m": _safe_float(pose_error.get("translation_m")),
                "rotation_error_deg": _safe_float(pose_error.get("rotation_deg")),
                "pose_cost_m": _safe_float(metadata.get("pose_cost_m")),
            }
        )
        order += 1
    return {
        query_id: sorted(rows, key=lambda item: (int(item["rank"]), int(item["order"])))[:submap_top_n]
        for query_id, rows in by_query.items()
    }


def _reference_prior_summary(candidates: list[dict[str, object]]) -> dict[str, object]:
    if not candidates:
        return {
            "candidate_count": 0,
            "top1_reference_image": None,
            "top1_rank": None,
            "top1_translation_error_m": None,
            "top1_rotation_error_deg": None,
            "top1_pose_cost_m": None,
            "oracle_reference_image": None,
            "oracle_rank": None,
            "oracle_translation_error_m": None,
            "oracle_rotation_error_deg": None,
            "oracle_pose_cost_m": None,
        }
    top1 = candidates[0]

    def oracle_key(item: dict[str, object]) -> tuple[float, float, int]:
        pose_cost = item.get("pose_cost_m")
        translation = item.get("translation_error_m")
        return (
            float(pose_cost) if pose_cost is not None else float("inf"),
            float(translation) if translation is not None else float("inf"),
            int(item["rank"]),
        )

    oracle = min(candidates, key=oracle_key)
    return {
        "candidate_count": len(candidates),
        "top1_reference_image": top1["reference_image"],
        "top1_rank": int(top1["rank"]),
        "top1_translation_error_m": top1.get("translation_error_m"),
        "top1_rotation_error_deg": top1.get("rotation_error_deg"),
        "top1_pose_cost_m": top1.get("pose_cost_m"),
        "oracle_reference_image": oracle["reference_image"],
        "oracle_rank": int(oracle["rank"]),
        "oracle_translation_error_m": oracle.get("translation_error_m"),
        "oracle_rotation_error_deg": oracle.get("rotation_error_deg"),
        "oracle_pose_cost_m": oracle.get("pose_cost_m"),
    }


def _prior_success(prior: dict[str, object], prefix: str, translation_m: float, rotation_deg: float) -> bool | None:
    translation = prior.get(f"{prefix}_translation_error_m")
    rotation = prior.get(f"{prefix}_rotation_error_deg")
    if translation is None or rotation is None:
        return None
    return bool(float(translation) <= float(translation_m) and float(rotation) <= float(rotation_deg))


def main(argv: Optional[Sequence[str]] = None) -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate patch-level query VFM token to sparse 3D landmark matching")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", default="")
    parser.add_argument("--semidense_anchor_npz", default="")
    parser.add_argument("--track_observations", default="")
    parser.add_argument("--visibility_index", default="")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--submap_mode", default="reference_visibility", choices=("reference_visibility", "gt_visible", "none"))
    parser.add_argument("--submap_top_n", type=int, default=5)
    parser.add_argument("--max_submap_landmarks", type=int, default=20000)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--query_token_step", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--mutual_top_k", type=int, default=5)
    parser.add_argument("--match_mode", default="soft_mutual", choices=("nn", "mnn", "soft_mutual"))
    parser.add_argument("--ratio_threshold", type=float, default=0.95)
    parser.add_argument("--disable_ratio_test", action="store_true")
    parser.add_argument("--min_similarity_margin", type=float, default=None)
    parser.add_argument("--min_similarity", type=float, default=0.2)
    parser.add_argument("--max_landmark_variance", type=float, default=None)
    parser.add_argument("--max_landmark_reprojection_error", type=float, default=None)
    parser.add_argument("--max_landmark_ambiguity", type=float, default=None)
    parser.add_argument("--enable_landmark_ambiguity_pruning", action="store_true")
    parser.add_argument("--ambiguity_prune_drop_fraction", type=float, default=None)
    parser.add_argument("--ambiguity_prune_max_score", type=float, default=None)
    parser.add_argument("--ambiguity_prune_close_similarity", type=float, default=0.9)
    parser.add_argument("--ambiguity_prune_reference_size", type=int, default=4096)
    parser.add_argument("--min_distance_to_boundary_px", type=float, default=None)
    parser.add_argument("--min_quality_weighted_similarity", type=float, default=None)
    parser.add_argument("--min_observation_count", type=int, default=2)
    parser.add_argument("--enable_landmark_quality", action="store_true")
    parser.add_argument("--quality_min_score", type=float, default=None)
    parser.add_argument("--quality_min_track_length", type=int, default=None)
    parser.add_argument("--quality_track_weight", type=float, default=1.0)
    parser.add_argument("--quality_variance_weight", type=float, default=1.0)
    parser.add_argument("--quality_reprojection_weight", type=float, default=1.0)
    parser.add_argument("--quality_idf_weight", type=float, default=0.0)
    parser.add_argument("--quality_ambiguity_weight", type=float, default=0.5)
    parser.add_argument("--quality_ambiguity_reference_size", type=int, default=4096)
    parser.add_argument("--enable_map_reliability_prior", action="store_true")
    parser.add_argument("--map_reliability_min_score", type=float, default=None)
    parser.add_argument("--map_reliability_filter_keep_fraction", type=float, default=None)
    parser.add_argument("--map_reliability_pnp_keep_fraction", type=float, default=None)
    parser.add_argument("--map_reliability_track_weight", type=float, default=1.0)
    parser.add_argument("--map_reliability_variance_weight", type=float, default=1.0)
    parser.add_argument("--map_reliability_reprojection_weight", type=float, default=1.0)
    parser.add_argument("--map_reliability_idf_weight", type=float, default=0.0)
    parser.add_argument("--map_reliability_ambiguity_weight", type=float, default=0.5)
    parser.add_argument("--map_reliability_view_angle_weight", type=float, default=0.0)
    parser.add_argument("--map_reliability_ambiguity_reference_size", type=int, default=4096)
    parser.add_argument("--map_reliability_uncertainty_min_scale", type=float, default=0.75)
    parser.add_argument("--map_reliability_uncertainty_max_scale", type=float, default=2.0)
    parser.add_argument("--enable_local_geometric_consistency", action="store_true")
    parser.add_argument("--local_consistency_image_radius_px", type=float, default=96.0)
    parser.add_argument("--local_consistency_xyz_radius_m", type=float, default=2.0)
    parser.add_argument("--local_consistency_min_support", type=int, default=1)
    parser.add_argument("--local_consistency_min_score", type=float, default=None)
    parser.add_argument("--local_consistency_keep_fraction", type=float, default=None)
    parser.add_argument("--local_consistency_max_input_matches", type=int, default=None)
    parser.add_argument("--enable_spatial_diversity_pnp", action="store_true")
    parser.add_argument("--spatial_diversity_grid_rows", type=int, default=4)
    parser.add_argument("--spatial_diversity_grid_cols", type=int, default=4)
    parser.add_argument("--spatial_diversity_max_per_cell", type=int, default=2)
    parser.add_argument("--spatial_diversity_max_matches", type=int, default=None)
    parser.add_argument("--spatial_diversity_min_world_z_range_m", type=float, default=None)
    parser.add_argument("--spatial_diversity_min_planarity_ratio", type=float, default=None)
    parser.add_argument(
        "--spatial_diversity_score_mode",
        default="margin",
        choices=("margin", "pairwise", "reliability", "similarity"),
    )
    parser.add_argument(
        "--pnp_soft_order_mode",
        default="none",
        choices=("none", "similarity", "margin", "reliability", "confidence", "composite"),
    )
    parser.add_argument("--pnp_soft_order_top_n", type=int, default=0)
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--return_all_topk_candidates", action="store_true")
    parser.add_argument(
        "--deduplicate_track_matches",
        action="store_true",
        help="Keep only the highest-scoring match for each query token and 3D track id.",
    )
    parser.add_argument("--match_block_size", type=int, default=256)
    parser.add_argument("--similarity_device", default="cpu")
    parser.add_argument(
        "--match_score_mode",
        default="similarity_quality",
        choices=("similarity", "landmark_quality", "similarity_quality", "similarity_pairwise"),
    )
    parser.add_argument("--safe_pairwise_checkpoint", default="")
    parser.add_argument("--calibrated_candidate_checkpoint", default="")
    parser.add_argument(
        "--calibrated_candidate_feature_set",
        default="descriptor_map",
        choices=("descriptor", "map_stats", "descriptor_map", "full"),
    )
    parser.add_argument("--calibrated_confidence_checkpoint", default="")
    parser.add_argument(
        "--calibrated_confidence_feature_set",
        default="descriptor_map",
        choices=("descriptor", "map_stats", "descriptor_map", "full"),
    )
    parser.add_argument("--pairwise_inlier_weight", type=float, default=0.0)
    parser.add_argument("--pairwise_filter_keep_fraction", type=float, default=None)
    parser.add_argument("--pairwise_filter_min_logit", type=float, default=None)
    parser.add_argument("--pairwise_filter_min_logprob", type=float, default=None)
    parser.add_argument("--pairwise_device", default="")
    parser.add_argument("--pairwise_batch_size", type=int, default=65536)
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--patch_at_k", type=int, default=5)
    parser.add_argument("--pnp_threshold_stride_multiplier", type=float, default=1.5)
    parser.add_argument("--pnp_min_inliers", type=int, default=0)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--pnp_confidence", type=float, default=0.999)
    parser.add_argument("--pnp_method", default="EPNP", choices=("AP3P", "EPNP", "ITERATIVE", "P3P", "SQPNP"))
    parser.add_argument("--pnp_refine_method", default="none", choices=("none", "LM", "VVS"))
    parser.add_argument("--pnp_refine_lm", action="store_true")
    parser.add_argument("--fixed_pose_refine_method", default="auto", choices=("auto", "none", "LM", "VVS", "robust_lm"))
    parser.add_argument("--robust_fixed_pose_loss", default="huber", choices=("linear", "soft_l1", "huber", "cauchy", "arctan"))
    parser.add_argument("--robust_fixed_pose_f_scale_px", type=float, default=4.0)
    parser.add_argument("--robust_fixed_pose_max_nfev", type=int, default=50)
    parser.add_argument(
        "--robust_fixed_pose_weight_mode",
        default="uniform",
        choices=(
            "uniform",
            "match",
            "reliability",
            "composite",
            "offset_confidence",
            "offset_composite",
            "offset_consistency_composite",
        ),
    )
    parser.add_argument(
        "--patch_offset_mode",
        default="none",
        choices=(
            "none",
            "fixed_same_inlier",
            "oracle_all",
            "oracle_patch_positive",
            "oracle_two_pass_inliers",
            "oracle_two_pass_inliers_patch_positive",
            "oracle_same_inlier_fixed",
            "oracle_same_inlier_fixed_patch_positive",
            "learned_confidence",
            "learned_two_pass_inliers",
            "learned_same_inlier_fixed",
            "lowlevel_same_inlier_fixed",
            "superpoint_snap_same_inlier_fixed",
            "superpoint_gt_oracle_same_inlier_fixed",
            "superpoint_gt_oracle_patch_positive_same_inlier_fixed",
            "landmark_conditioned_superpoint_same_inlier_fixed",
        ),
    )
    parser.add_argument("--patch_offset_checkpoint", default="")
    parser.add_argument("--patch_offset_device", default="")
    parser.add_argument("--patch_offset_batch_size", type=int, default=4096)
    parser.add_argument("--patch_offset_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--patch_offset_max_stride", type=float, default=0.5)
    parser.add_argument("--patch_offset_max_sigma", type=float, default=None)
    parser.add_argument("--patch_offset_max_consistency_residual_increase_px", type=float, default=None)
    parser.add_argument("--patch_offset_max_consistency_residual_px", type=float, default=None)
    parser.add_argument("--patch_offset_free_oracle", action="store_true")
    parser.add_argument("--patch_offset_bound_metric", default="l2", choices=("l2", "linf"))
    parser.add_argument("--patch_offset_oracle_noise_px", type=float, default=0.0)
    parser.add_argument("--patch_offset_oracle_seed", type=int, default=0)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--lowlevel_feature_mode", default="gray_ncc", choices=("gray_ncc", "sobel_ncc"))
    parser.add_argument("--lowlevel_template_radius_px", type=int, default=8)
    parser.add_argument("--lowlevel_search_radius_px", type=int, default=8)
    parser.add_argument("--lowlevel_search_step_px", type=int, default=2)
    parser.add_argument("--lowlevel_min_score", type=float, default=0.05)
    parser.add_argument("--lowlevel_min_confidence", type=float, default=0.02)
    parser.add_argument("--superpoint_device", default="")
    parser.add_argument("--superpoint_nms_radius", type=int, default=4)
    parser.add_argument("--superpoint_keypoint_threshold", type=float, default=0.005)
    parser.add_argument("--superpoint_max_keypoints", type=int, default=-1)
    parser.add_argument("--superpoint_remove_borders", type=int, default=4)
    parser.add_argument("--superpoint_snap_radius_stride", type=float, default=0.5)
    parser.add_argument("--superpoint_min_score", type=float, default=0.005)
    parser.add_argument("--superpoint_snap_strategy", default="highest_score", choices=("highest_score", "nearest"))
    parser.add_argument("--superpoint_availability_radii_px", default="2,4,8,12,16")
    parser.add_argument("--superpoint_oracle_max_distance_px", type=float, default=16.0)
    parser.add_argument("--lcks_candidate_radius_stride", type=float, default=0.5)
    parser.add_argument("--lcks_support_radius_px", type=float, default=16.0)
    parser.add_argument("--lcks_min_query_score", type=float, default=0.005)
    parser.add_argument("--lcks_min_support_score", type=float, default=0.005)
    parser.add_argument("--lcks_descriptor_weight", type=float, default=1.0)
    parser.add_argument("--lcks_query_score_weight", type=float, default=0.25)
    parser.add_argument("--lcks_center_penalty_weight", type=float, default=0.15)
    parser.add_argument("--lcks_support_distance_penalty_weight", type=float, default=0.10)
    parser.add_argument("--lcks_score_threshold", type=float, default=0.2)
    parser.add_argument("--lcks_support_selection_strategy", default="nearest", choices=("nearest", "highest_score"))
    parser.add_argument("--risk_coverages", default="0.8,0.9,1.0")
    parser.add_argument(
        "--oracle_match_mode",
        default="none",
        choices=("none", "patch_positives", "filter_gt_correct"),
    )
    parser.add_argument("--oracle_max_positives_per_token", type=int, default=1)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_matches_jsonl", default="")
    parser.add_argument("--output_superpoint_candidate_jsonl", default="")
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    if args.submap_mode == "reference_visibility" and not args.candidate_bank:
        raise ValueError("candidate_bank is required when submap_mode=reference_visibility")
    if args.patch_offset_mode in {
        "lowlevel_same_inlier_fixed",
        "superpoint_snap_same_inlier_fixed",
        "superpoint_gt_oracle_same_inlier_fixed",
        "superpoint_gt_oracle_patch_positive_same_inlier_fixed",
        "landmark_conditioned_superpoint_same_inlier_fixed",
    } and not args.image_root:
        raise ValueError(f"image_root is required for {args.patch_offset_mode}")

    if not args.landmark_bank and not args.semidense_anchor_npz:
        raise ValueError("either --landmark_bank or --semidense_anchor_npz is required")
    if not args.semidense_anchor_npz and not args.track_observations:
        raise ValueError("--track_observations is required when loading --landmark_bank")

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    if args.semidense_anchor_npz:
        semidense_map = SemiDenseAnchorMap.load_npz(Path(args.semidense_anchor_npz))
        landmark_index = semidense_map.to_landmark_index()
        map_source = "semidense_anchor_npz"
    else:
        semidense_map = None
        bank = load_selected_track_bank_npz(Path(args.landmark_bank))
        xyz_by_track, reprojection_error_by_track = _load_track_stats(Path(args.track_observations))
        landmark_index = LandmarkMapIndex.from_track_bank(bank, xyz_by_track, reprojection_error_by_track)
        map_source = "landmark_bank"
    needs_scene_ambiguity = bool(
        args.max_landmark_ambiguity is not None
        or (args.enable_landmark_quality and (args.quality_idf_weight > 0.0 or args.quality_ambiguity_weight > 0.0))
        or (
            args.enable_map_reliability_prior
            and (args.map_reliability_idf_weight > 0.0 or args.map_reliability_ambiguity_weight > 0.0)
        )
    )
    if needs_scene_ambiguity:
        landmark_index = with_landmark_ambiguity_scores(
            landmark_index,
            reference_size=max(args.quality_ambiguity_reference_size, args.map_reliability_ambiguity_reference_size),
            block_size=args.match_block_size,
        )
    visibility_index = None
    if args.visibility_index:
        visibility_index = LandmarkVisibilityIndex.load_npz(Path(args.visibility_index))
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    reference_submaps = _load_reference_submaps(args.candidate_bank, args.submap_top_n)
    reference_pose_priors = _load_reference_pose_priors(args.candidate_bank, args.submap_top_n)
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    config = PatchTo3DMatchingConfig(
        top_k=args.top_k,
        mutual_top_k=args.mutual_top_k,
        match_mode=args.match_mode,
        ratio_threshold=None if args.disable_ratio_test else args.ratio_threshold,
        min_similarity_margin=args.min_similarity_margin,
        min_similarity=args.min_similarity,
        max_landmark_variance=args.max_landmark_variance,
        max_landmark_reprojection_error=args.max_landmark_reprojection_error,
        max_landmark_ambiguity=args.max_landmark_ambiguity,
        min_distance_to_boundary_px=args.min_distance_to_boundary_px,
        min_quality_weighted_similarity=args.min_quality_weighted_similarity,
        min_observation_count=args.min_observation_count,
        query_token_step=args.query_token_step,
        max_matches=args.max_matches,
        block_size=args.match_block_size,
        similarity_device=args.similarity_device,
        match_score_mode=args.match_score_mode,
        return_all_topk_candidates=bool(args.return_all_topk_candidates),
        deduplicate_track_matches=bool(args.deduplicate_track_matches),
        pairwise_filter_keep_fraction=args.pairwise_filter_keep_fraction,
        pairwise_filter_min_logit=args.pairwise_filter_min_logit,
        pairwise_filter_min_logprob=args.pairwise_filter_min_logprob,
        map_reliability=MapReliabilityConfig(
            enabled=bool(args.enable_map_reliability_prior),
            track_weight=float(args.map_reliability_track_weight),
            variance_weight=float(args.map_reliability_variance_weight),
            reprojection_weight=float(args.map_reliability_reprojection_weight),
            idf_weight=float(args.map_reliability_idf_weight),
            ambiguity_weight=float(args.map_reliability_ambiguity_weight),
            view_angle_weight=float(args.map_reliability_view_angle_weight),
            ambiguity_reference_size=int(args.map_reliability_ambiguity_reference_size),
            min_score=args.map_reliability_min_score,
            filter_keep_fraction=args.map_reliability_filter_keep_fraction,
            pnp_keep_fraction=args.map_reliability_pnp_keep_fraction,
            uncertainty_min_scale=float(args.map_reliability_uncertainty_min_scale),
            uncertainty_max_scale=float(args.map_reliability_uncertainty_max_scale),
        ),
        local_geometric_consistency=LocalGeometricConsistencyConfig(
            enabled=bool(args.enable_local_geometric_consistency),
            image_radius_px=float(args.local_consistency_image_radius_px),
            xyz_radius_m=float(args.local_consistency_xyz_radius_m),
            min_support=args.local_consistency_min_support,
            min_score=args.local_consistency_min_score,
            keep_fraction=args.local_consistency_keep_fraction,
            max_input_matches=args.local_consistency_max_input_matches,
        ),
        landmark_ambiguity_pruning=LandmarkAmbiguityPruningConfig(
            enabled=bool(args.enable_landmark_ambiguity_pruning),
            drop_fraction=args.ambiguity_prune_drop_fraction,
            max_score=args.ambiguity_prune_max_score,
            close_similarity_threshold=float(args.ambiguity_prune_close_similarity),
            reference_size=int(args.ambiguity_prune_reference_size),
            block_size=int(args.match_block_size),
        ),
        landmark_quality=LandmarkQualityConfig(
            enabled=bool(args.enable_landmark_quality),
            track_weight=args.quality_track_weight,
            variance_weight=args.quality_variance_weight,
            reprojection_weight=args.quality_reprojection_weight,
            idf_weight=args.quality_idf_weight,
            ambiguity_weight=args.quality_ambiguity_weight,
            ambiguity_reference_size=args.quality_ambiguity_reference_size,
            min_score=args.quality_min_score,
            min_track_length=args.quality_min_track_length,
        ),
    )
    pairwise_scorer = None
    if args.safe_pairwise_checkpoint:
        pairwise_scorer = SafePairwiseInlierScorer.from_checkpoint(
            Path(args.safe_pairwise_checkpoint),
            device=args.pairwise_device or args.similarity_device,
            batch_size=int(args.pairwise_batch_size),
        )
    calibrated_confidence_model = None
    if args.calibrated_confidence_checkpoint:
        calibrated_confidence_model = CalibratedLogisticConfidence.load_json(
            Path(args.calibrated_confidence_checkpoint),
        )
    calibrated_candidate_scorer = None
    if args.calibrated_candidate_checkpoint:
        calibrated_candidate_scorer = CalibratedMatchCandidateScorer(
            model=CalibratedLogisticConfidence.load_json(Path(args.calibrated_candidate_checkpoint)),
            feature_set=args.calibrated_candidate_feature_set,
        )
    offset_run = None
    if args.patch_offset_mode.startswith("learned"):
        if not args.patch_offset_checkpoint:
            raise ValueError("patch_offset_checkpoint is required for learned patch offset modes")
        offset_run = load_patch_offset_refiner_checkpoint(
            Path(args.patch_offset_checkpoint),
            device=args.patch_offset_device or args.similarity_device,
        )
    lowlevel_support_bank = None
    image_cache: dict[str, np.ndarray] = {}
    image_root = Path(args.image_root) if args.image_root else None
    if args.patch_offset_mode in {"lowlevel_same_inlier_fixed", "landmark_conditioned_superpoint_same_inlier_fixed"}:
        lowlevel_support_bank = LowLevelSupportBank.from_jsonl(Path(args.track_observations))
    superpoint_detector = None
    if args.patch_offset_mode in {
        "superpoint_snap_same_inlier_fixed",
        "superpoint_gt_oracle_same_inlier_fixed",
        "superpoint_gt_oracle_patch_positive_same_inlier_fixed",
        "landmark_conditioned_superpoint_same_inlier_fixed",
    }:
        superpoint_detector = HLocSuperPointKeypointDetector(
            device=args.superpoint_device or args.patch_offset_device or args.similarity_device,
            nms_radius=int(args.superpoint_nms_radius),
            keypoint_threshold=float(args.superpoint_keypoint_threshold),
            max_keypoints=int(args.superpoint_max_keypoints),
            remove_borders=int(args.superpoint_remove_borders),
        )
    superpoint_cache = {}

    def get_superpoint_keypoints(image_id: str):
        if superpoint_detector is None:
            raise ValueError("SuperPoint detector is not initialized")
        key = str(image_id)
        if key not in superpoint_cache:
            superpoint_cache[key] = superpoint_detector.detect(get_image(key))
        return superpoint_cache[key]

    superpoint_availability_radii = tuple(
        float(item.strip())
        for item in str(args.superpoint_availability_radii_px).split(",")
        if item.strip()
    )

    def get_image(image_id: str) -> np.ndarray:
        if image_root is None:
            raise ValueError("image_root is required")
        key = str(image_id)
        if key not in image_cache:
            image_cache[key] = _read_image_rgb(image_root / key)
        return image_cache[key]

    rows = []
    match_rows = []
    superpoint_candidate_rows = []
    records = list(manifest.records)
    if args.max_queries > 0:
        records = records[: args.max_queries]
    for record in records:
        query_id = record.image_id
        gt_pose = gt_by_query.get(query_id)
        if gt_pose is None:
            raise ValueError(f"query pose not found for {query_id}")
        gt_visible_submap = filter_landmarks_by_projected_visibility(landmark_index, gt_pose.pose_w2c, camera)
        gt_visible_track_ids = _track_id_set(gt_visible_submap)
        submap = landmark_index
        references = reference_submaps.get(query_id, [])
        if args.submap_mode == "reference_visibility" and not references:
            raise ValueError(f"no reference_visibility candidates found for query_id={query_id}")
        reference_prior = _reference_prior_summary(reference_pose_priors.get(query_id, []))
        visibility_gate = {"full_visible_tracks": None, "bank_visible_tracks": None, "bank_visibility_coverage": None}
        if args.submap_mode == "gt_visible":
            submap = gt_visible_submap
            visibility_gate = {
                "full_visible_tracks": len(submap),
                "bank_visible_tracks": len(submap),
                "bank_visibility_coverage": 1.0 if len(submap) else 0.0,
            }
        elif args.submap_mode == "reference_visibility":
            if visibility_index is not None and semidense_map is not None:
                submap, visibility_gate = filter_semidense_by_source_visibility(
                    semidense_map,
                    visibility_index,
                    references,
                )
            elif visibility_index is None:
                submap = filter_landmarks_by_reference_images(landmark_index, references)
                visibility_gate = {
                    "full_visible_tracks": len(submap),
                    "bank_visible_tracks": len(submap),
                    "bank_visibility_coverage": 1.0 if len(submap) else 0.0,
                }
            else:
                submap, visibility_gate = filter_landmarks_by_visibility(landmark_index, visibility_index, references)
        pre_limit_submap_count = len(submap)
        pre_limit_submap_track_ids = _track_id_set(submap)
        pre_limit_gt_visible_count = len(pre_limit_submap_track_ids.intersection(gt_visible_track_ids))
        submap = _limit_submap(submap, int(args.max_submap_landmarks))
        submap_track_ids = _track_id_set(submap)
        submap_gt_visible_count = len(submap_track_ids.intersection(gt_visible_track_ids))
        gt_visible_count = len(gt_visible_track_ids)
        visible_landmark_recall = (
            float(submap_gt_visible_count / max(gt_visible_count, 1)) if gt_visible_count else None
        )
        pre_limit_visible_landmark_recall = (
            float(pre_limit_gt_visible_count / max(gt_visible_count, 1)) if gt_visible_count else None
        )
        query_feature = _load_query_feature(record.token_path, args.layer_name)
        _channels, token_height, token_width = query_feature.shape
        stride_x = float(camera.width - 1) / max(float(token_width - 1), 1.0)
        stride_y = float(camera.height - 1) / max(float(token_height - 1), 1.0)
        stride = float(max(stride_x, stride_y))
        positives = build_patch_positive_sets(
            submap,
            gt_pose.pose_w2c,
            camera,
            token_width,
            token_height,
            patch_scale=args.patch_scale,
        )
        if args.oracle_match_mode == "patch_positives":
            matches = oracle_patch_positive_matches(
                submap,
                positives,
                max_per_token=int(args.oracle_max_positives_per_token),
            )
        else:
            matches = match_query_patches_to_landmarks(
                query_feature,
                submap,
                config,
                int(camera.width),
                int(camera.height),
                pairwise_inlier_scorer=pairwise_scorer,
                pairwise_inlier_weight=float(args.pairwise_inlier_weight),
                candidate_match_scorer=calibrated_candidate_scorer,
            )
            if args.oracle_match_mode == "filter_gt_correct":
                matches = filter_patch_correct_matches(matches, positives)
        if calibrated_confidence_model is not None:
            matches = annotate_matches_with_calibrated_confidence(
                matches,
                calibrated_confidence_model,
                feature_set=args.calibrated_confidence_feature_set,
            )
        if args.pnp_soft_order_mode != "none":
            matches = soft_order_pnp_matches(matches, mode=args.pnp_soft_order_mode)
        pnp_matches = select_pnp_matches_by_map_reliability(
            matches,
            keep_fraction=config.map_reliability.pnp_keep_fraction if config.map_reliability.enabled else None,
            min_score=None,
        )
        if args.pnp_soft_order_mode != "none" and args.pnp_soft_order_top_n > 0:
            pnp_matches = pnp_matches[: int(args.pnp_soft_order_top_n)]
        pnp_matches = select_pnp_matches_by_spatial_diversity(
            pnp_matches,
            image_width=int(camera.width),
            image_height=int(camera.height),
            config=SpatialDiversityPnPConfig(
                enabled=bool(args.enable_spatial_diversity_pnp),
                grid_rows=int(args.spatial_diversity_grid_rows),
                grid_cols=int(args.spatial_diversity_grid_cols),
                max_per_cell=int(args.spatial_diversity_max_per_cell),
                max_matches=args.spatial_diversity_max_matches,
                score_mode=args.spatial_diversity_score_mode,
                min_world_z_range_m=args.spatial_diversity_min_world_z_range_m,
                min_planarity_ratio=args.spatial_diversity_min_planarity_ratio,
            ),
        )
        selected_indices = _selected_match_indices(matches, pnp_matches)
        pnp_threshold = patch_uncertainty_pnp_threshold(stride, args.pnp_threshold_stride_multiplier)

        def run_pnp(current_pnp_matches):
            return estimate_pose_pnp_ransac(
                current_pnp_matches,
                camera,
                reprojection_error_px=pnp_threshold,
                confidence=float(args.pnp_confidence),
                iterations=args.pnp_iterations,
                min_inliers=int(args.pnp_min_inliers),
                refine_lm=bool(args.pnp_refine_lm),
                pnp_method=args.pnp_method,
                refine_method=args.pnp_refine_method,
            )

        offset_summary = {"mode": args.patch_offset_mode, "refined_count": 0}
        oracle_max_stride = None if bool(args.patch_offset_free_oracle) else float(args.patch_offset_max_stride)
        patch_positive_by_token = {
            int(token_idx): {int(track_id) for track_id in positive.track_ids}
            for token_idx, positive in positives.by_token.items()
        }
        if args.patch_offset_mode in {"oracle_all", "oracle_patch_positive"}:
            matches, offset_summary = refine_matches_with_oracle_offsets(
                matches,
                pose_w2c=gt_pose.pose_w2c,
                camera=camera,
                stride_px=stride,
                inlier_mask=None,
                max_offset_stride=oracle_max_stride,
                bound_metric=args.patch_offset_bound_metric,
                patch_positive_by_token=patch_positive_by_token,
                require_patch_positive=args.patch_offset_mode == "oracle_patch_positive",
                noise_sigma_px=float(args.patch_offset_oracle_noise_px),
                rng_seed=int(args.patch_offset_oracle_seed),
            )
            pnp_matches = [matches[idx] for idx in selected_indices]
            pnp = run_pnp(pnp_matches)
        elif args.patch_offset_mode == "learned_confidence":
            assert offset_run is not None
            offsets, confidences, sigmas = predict_patch_offsets_for_matches(
                query_feature,
                submap,
                matches,
                offset_run.model,
                device=args.patch_offset_device or args.similarity_device,
                batch_size=int(args.patch_offset_batch_size),
            )
            matches, offset_summary = apply_predicted_patch_offsets(
                matches,
                offsets,
                confidences,
                stride_px=stride,
                confidence_threshold=float(args.patch_offset_confidence_threshold),
                inlier_mask=None,
                max_offset_stride=float(args.patch_offset_max_stride),
                sigmas=sigmas,
                max_sigma=args.patch_offset_max_sigma,
            )
            pnp_matches = [matches[idx] for idx in selected_indices]
            pnp = run_pnp(pnp_matches)
        elif args.patch_offset_mode in {
            "fixed_same_inlier",
            "oracle_two_pass_inliers",
            "oracle_two_pass_inliers_patch_positive",
            "oracle_same_inlier_fixed",
            "oracle_same_inlier_fixed_patch_positive",
            "learned_two_pass_inliers",
            "learned_same_inlier_fixed",
            "lowlevel_same_inlier_fixed",
            "superpoint_snap_same_inlier_fixed",
            "superpoint_gt_oracle_same_inlier_fixed",
            "superpoint_gt_oracle_patch_positive_same_inlier_fixed",
            "landmark_conditioned_superpoint_same_inlier_fixed",
        }:
            initial_pnp = run_pnp(pnp_matches)
            initial_full_inlier_mask = _full_inlier_mask(matches, pnp_matches, initial_pnp.inlier_mask)
            if args.patch_offset_mode in {
                "oracle_two_pass_inliers",
                "oracle_two_pass_inliers_patch_positive",
                "oracle_same_inlier_fixed",
                "oracle_same_inlier_fixed_patch_positive",
            }:
                matches, offset_summary = refine_matches_with_oracle_offsets(
                    matches,
                    pose_w2c=gt_pose.pose_w2c,
                    camera=camera,
                    stride_px=stride,
                    inlier_mask=initial_full_inlier_mask,
                    max_offset_stride=oracle_max_stride,
                    bound_metric=args.patch_offset_bound_metric,
                    patch_positive_by_token=patch_positive_by_token,
                    require_patch_positive=args.patch_offset_mode
                    in {"oracle_two_pass_inliers_patch_positive", "oracle_same_inlier_fixed_patch_positive"},
                    noise_sigma_px=float(args.patch_offset_oracle_noise_px),
                    rng_seed=int(args.patch_offset_oracle_seed),
                )
            elif args.patch_offset_mode in {"learned_two_pass_inliers", "learned_same_inlier_fixed"}:
                assert offset_run is not None
                offsets, confidences, sigmas = predict_patch_offsets_for_matches(
                    query_feature,
                    submap,
                    matches,
                    offset_run.model,
                    device=args.patch_offset_device or args.similarity_device,
                    batch_size=int(args.patch_offset_batch_size),
                )
                matches, offset_summary = apply_predicted_patch_offsets(
                    matches,
                    offsets,
                    confidences,
                    stride_px=stride,
                    confidence_threshold=float(args.patch_offset_confidence_threshold),
                    inlier_mask=initial_full_inlier_mask,
                    max_offset_stride=float(args.patch_offset_max_stride),
                    sigmas=sigmas,
                    max_sigma=args.patch_offset_max_sigma,
                    consistency_pose_w2c=initial_pnp.pose_w2c,
                    consistency_camera=camera if initial_pnp.pose_w2c is not None else None,
                    max_consistency_residual_increase_px=args.patch_offset_max_consistency_residual_increase_px,
                    max_consistency_residual_px=args.patch_offset_max_consistency_residual_px,
                )
            elif args.patch_offset_mode == "lowlevel_same_inlier_fixed":
                if lowlevel_support_bank is None:
                    raise ValueError("low-level support bank is not initialized")
                query_viewing_ray_by_track = {}
                if initial_pnp.pose_w2c is not None:
                    camera_center = camera_center_from_pose_w2c(initial_pnp.pose_w2c)
                    for idx, match in enumerate(matches):
                        if not bool(initial_full_inlier_mask[idx]):
                            continue
                        ray = np.asarray(match.xyz, dtype=np.float64).reshape(3) - camera_center
                        norm = max(float(np.linalg.norm(ray)), 1e-12)
                        query_viewing_ray_by_track[int(match.track_id)] = ray / norm
                image_by_id = {query_id: get_image(query_id)}
                for idx, match in enumerate(matches):
                    if not bool(initial_full_inlier_mask[idx]):
                        continue
                    support = select_support_observation(
                        lowlevel_support_bank,
                        int(match.track_id),
                        query_viewing_ray=query_viewing_ray_by_track.get(int(match.track_id)),
                    )
                    if support is not None:
                        try:
                            image_by_id[str(support.image_id)] = get_image(str(support.image_id))
                        except ValueError:
                            pass
                matches, offset_summary = apply_lowlevel_offsets_to_matches(
                    matches,
                    query_image_id=query_id,
                    image_by_id=image_by_id,
                    support_bank=lowlevel_support_bank,
                    inlier_mask=initial_full_inlier_mask,
                    config=LowLevelOffsetSidecarConfig(
                        mode=args.lowlevel_feature_mode,
                        template_radius_px=int(args.lowlevel_template_radius_px),
                        search_radius_px=int(args.lowlevel_search_radius_px),
                        search_step_px=int(args.lowlevel_search_step_px),
                        max_offset_px=float(args.patch_offset_max_stride) * float(stride),
                        min_score=float(args.lowlevel_min_score),
                        min_confidence=float(args.lowlevel_min_confidence),
                    ),
                    query_viewing_ray_by_track=query_viewing_ray_by_track,
                )
            elif args.patch_offset_mode == "superpoint_snap_same_inlier_fixed":
                if superpoint_detector is None:
                    raise ValueError("SuperPoint detector is not initialized")
                query_keypoints = superpoint_detector.detect(get_image(query_id))
                gt_xy_by_match = []
                for match in matches:
                    projected = project_xyz_to_image(np.asarray(match.xyz, dtype=np.float64), gt_pose.pose_w2c, camera)
                    gt_xy_by_match.append(None if projected is None else np.asarray(projected, dtype=np.float64))
                availability = superpoint_availability_summary(
                    gt_xy_by_match,
                    query_keypoints,
                    initial_full_inlier_mask,
                    radii_px=superpoint_availability_radii,
                )
                matches, offset_summary = apply_superpoint_snaps_to_matches(
                    matches,
                    query_keypoints,
                    inlier_mask=initial_full_inlier_mask,
                    config=SuperPointSnapConfig(
                        max_offset_px=float(args.superpoint_snap_radius_stride) * float(stride),
                        min_score=float(args.superpoint_min_score),
                        selection_strategy=args.superpoint_snap_strategy,
                    ),
                )
                offset_summary = {**offset_summary, **availability}
            elif args.patch_offset_mode in {
                "superpoint_gt_oracle_same_inlier_fixed",
                "superpoint_gt_oracle_patch_positive_same_inlier_fixed",
            }:
                if superpoint_detector is None:
                    raise ValueError("SuperPoint detector is not initialized")
                query_keypoints = superpoint_detector.detect(get_image(query_id))
                gt_xy_by_match = []
                for match in matches:
                    projected = project_xyz_to_image(np.asarray(match.xyz, dtype=np.float64), gt_pose.pose_w2c, camera)
                    gt_xy_by_match.append(None if projected is None else np.asarray(projected, dtype=np.float64))
                availability = superpoint_availability_summary(
                    gt_xy_by_match,
                    query_keypoints,
                    initial_full_inlier_mask,
                    radii_px=superpoint_availability_radii,
                )
                matches, offset_summary = apply_superpoint_gt_oracle_snaps_to_matches(
                    matches,
                    query_keypoints,
                    inlier_mask=initial_full_inlier_mask,
                    gt_xy_by_match=gt_xy_by_match,
                    max_distance_px=float(args.superpoint_oracle_max_distance_px),
                    patch_positive_by_token=patch_positive_by_token,
                    require_patch_positive=args.patch_offset_mode
                    == "superpoint_gt_oracle_patch_positive_same_inlier_fixed",
                )
                offset_summary = {**offset_summary, **availability}
            elif args.patch_offset_mode == "landmark_conditioned_superpoint_same_inlier_fixed":
                if lowlevel_support_bank is None:
                    raise ValueError("low-level support bank is not initialized")
                query_keypoints = get_superpoint_keypoints(query_id)
                gt_xy_by_match = []
                for match in matches:
                    projected = project_xyz_to_image(np.asarray(match.xyz, dtype=np.float64), gt_pose.pose_w2c, camera)
                    gt_xy_by_match.append(None if projected is None else np.asarray(projected, dtype=np.float64))
                availability = superpoint_availability_summary(
                    gt_xy_by_match,
                    query_keypoints,
                    initial_full_inlier_mask,
                    radii_px=superpoint_availability_radii,
                )
                query_viewing_ray_by_track = {}
                if initial_pnp.pose_w2c is not None:
                    camera_center = camera_center_from_pose_w2c(initial_pnp.pose_w2c)
                    for idx, match in enumerate(matches):
                        if not bool(initial_full_inlier_mask[idx]):
                            continue
                        ray = np.asarray(match.xyz, dtype=np.float64).reshape(3) - camera_center
                        norm = max(float(np.linalg.norm(ray)), 1e-12)
                        query_viewing_ray_by_track[int(match.track_id)] = ray / norm
                support_by_track = {}
                missing_support_observation = 0
                missing_support_keypoint = 0
                for idx, match in enumerate(matches):
                    if not bool(initial_full_inlier_mask[idx]):
                        continue
                    support_obs = select_support_observation(
                        lowlevel_support_bank,
                        int(match.track_id),
                        query_viewing_ray=query_viewing_ray_by_track.get(int(match.track_id)),
                    )
                    if support_obs is None:
                        missing_support_observation += 1
                        continue
                    support_keypoints = get_superpoint_keypoints(str(support_obs.image_id))
                    support = select_support_superpoint_keypoint(
                        track_id=int(match.track_id),
                        support_image_id=str(support_obs.image_id),
                        observation_xy=np.asarray(support_obs.xy, dtype=np.float64),
                        keypoints=support_keypoints,
                        max_distance_px=float(args.lcks_support_radius_px),
                        min_score=float(args.lcks_min_support_score),
                        strategy=args.lcks_support_selection_strategy,
                    )
                    if support is None:
                        missing_support_keypoint += 1
                        continue
                    support_by_track[int(match.track_id)] = support
                lcks_config = LandmarkConditionedKeypointSelectorConfig(
                    candidate_radius_px=float(args.lcks_candidate_radius_stride) * float(stride),
                    support_radius_px=float(args.lcks_support_radius_px),
                    min_query_score=float(args.lcks_min_query_score),
                    min_support_score=float(args.lcks_min_support_score),
                    descriptor_weight=float(args.lcks_descriptor_weight),
                    query_score_weight=float(args.lcks_query_score_weight),
                    center_penalty_weight=float(args.lcks_center_penalty_weight),
                    support_distance_penalty_weight=float(args.lcks_support_distance_penalty_weight),
                    score_threshold=float(args.lcks_score_threshold),
                    support_selection_strategy=args.lcks_support_selection_strategy,
                )
                if args.output_superpoint_candidate_jsonl:
                    initial_reproj_errors = (
                        [None for _match in matches]
                        if initial_pnp.pose_w2c is None
                        else [
                            None if not np.isfinite(value) else float(value)
                            for value in match_reprojection_errors(matches, initial_pnp.pose_w2c, camera)
                        ]
                    )
                    superpoint_candidate_rows.extend(
                        build_landmark_conditioned_superpoint_candidate_rows(
                            matches,
                            query_keypoints,
                            support_by_track,
                            inlier_mask=initial_full_inlier_mask,
                            config=lcks_config,
                            gt_xy_by_match=gt_xy_by_match,
                            query_id=query_id,
                            stride_px=stride,
                            baseline_reproj_residual_by_match=initial_reproj_errors,
                        )
                    )
                matches, offset_summary = apply_landmark_conditioned_superpoint_snaps_to_matches(
                    matches,
                    query_keypoints,
                    support_by_track,
                    inlier_mask=initial_full_inlier_mask,
                    config=lcks_config,
                    gt_xy_by_match=gt_xy_by_match,
                )
                support_availability = float(len(support_by_track) / max(int(initial_pnp.inlier_count), 1))
                offset_summary = {
                    **offset_summary,
                    **availability,
                    "support_observation_missing_count": int(missing_support_observation),
                    "support_keypoint_missing_count": int(missing_support_keypoint),
                    "support_availability_ratio": support_availability,
                }
            offset_summary = {**offset_summary, "initial_inlier_count": int(initial_pnp.inlier_count)}
            if args.patch_offset_mode in {
                "fixed_same_inlier",
                "oracle_same_inlier_fixed",
                "oracle_same_inlier_fixed_patch_positive",
                "learned_same_inlier_fixed",
                "lowlevel_same_inlier_fixed",
                "superpoint_snap_same_inlier_fixed",
                "superpoint_gt_oracle_same_inlier_fixed",
                "superpoint_gt_oracle_patch_positive_same_inlier_fixed",
                "landmark_conditioned_superpoint_same_inlier_fixed",
            }:
                fixed_indices = np.flatnonzero(initial_full_inlier_mask).astype(np.int64).tolist()
                pnp_matches = [matches[int(idx)] for idx in fixed_indices]
                fixed_refine_method = (
                    args.pnp_refine_method if args.pnp_refine_method != "none" else "LM"
                )
                if args.fixed_pose_refine_method != "auto":
                    fixed_refine_method = args.fixed_pose_refine_method
                if fixed_refine_method == "robust_lm":
                    fixed_weights = _fixed_pose_refinement_weights(
                        pnp_matches,
                        args.robust_fixed_pose_weight_mode,
                    )
                    pnp = estimate_pose_pnp_fixed_robust(
                        pnp_matches,
                        camera,
                        weights=fixed_weights,
                        min_inliers=max(4, int(args.pnp_min_inliers)),
                        initial_pose_w2c=initial_pnp.pose_w2c,
                        pnp_method=args.pnp_method,
                        loss=args.robust_fixed_pose_loss,
                        f_scale_px=float(args.robust_fixed_pose_f_scale_px),
                        max_nfev=int(args.robust_fixed_pose_max_nfev),
                    )
                else:
                    pnp = estimate_pose_pnp_fixed(
                        pnp_matches,
                        camera,
                        min_inliers=max(4, int(args.pnp_min_inliers)),
                        pnp_method=args.pnp_method,
                        refine_method=fixed_refine_method,
                    )
                offset_summary = {
                    **offset_summary,
                    "fixed_same_inlier_count": int(len(fixed_indices)),
                    "fixed_pose_refine_method": fixed_refine_method,
                    "robust_fixed_pose_loss": args.robust_fixed_pose_loss,
                    "robust_fixed_pose_f_scale_px": float(args.robust_fixed_pose_f_scale_px),
                    "robust_fixed_pose_weight_mode": args.robust_fixed_pose_weight_mode,
                }
            else:
                pnp_matches = [matches[idx] for idx in selected_indices]
                pnp = run_pnp(pnp_matches)
        else:
            pnp = run_pnp(pnp_matches)
        full_pnp_inlier_mask = _full_inlier_mask(matches, pnp_matches, pnp.inlier_mask)
        pnp_match_ids = {id(match) for match in pnp_matches}
        pnp_selected_mask = np.asarray([id(match) in pnp_match_ids for match in matches], dtype=bool)
        patch_stats = evaluate_patch_matches(
            matches,
            positives,
            gt_pose.pose_w2c,
            camera,
            stride_px=stride,
            pnp_inlier_mask=full_pnp_inlier_mask,
            top_k=args.patch_at_k,
        )
        positive_stats = patch_positive_set_stats(positives)
        pose_error = pnp_pose_error(pnp.pose_w2c, gt_pose.pose_w2c)
        all_spatial_stats = match_spatial_distribution_stats(
            matches, int(camera.width), int(camera.height), pose_w2c=pnp.pose_w2c
        )
        inlier_spatial_stats = match_spatial_distribution_stats(
            matches,
            int(camera.width),
            int(camera.height),
            full_pnp_inlier_mask,
            pose_w2c=pnp.pose_w2c,
        )
        pnp_residual_stats = pnp_reprojection_residual_stats(
            matches,
            pnp.pose_w2c,
            camera,
            inlier_mask=full_pnp_inlier_mask,
        )
        gt_match_errors = match_reprojection_errors(matches, gt_pose.pose_w2c, camera)
        baseline_match_errors = (
            None
            if pnp.pose_w2c is None
            else match_reprojection_errors(matches, pnp.pose_w2c, camera)
        )
        reliability_all_stats = _map_reliability_stats(matches)
        reliability_pnp_stats = _map_reliability_stats(matches, pnp_selected_mask)
        reliability_inlier_stats = _map_reliability_stats(matches, full_pnp_inlier_mask)
        local_all_stats = _local_consistency_stats(matches)
        local_pnp_stats = _local_consistency_stats(matches, pnp_selected_mask)
        local_inlier_stats = _local_consistency_stats(matches, full_pnp_inlier_mask)
        projected_landmarks = count_projected_landmarks(submap, gt_pose.pose_w2c, camera)
        if args.output_matches_jsonl:
            positive_by_token = positives.by_token
            per_token_seen: dict[int, int] = {}
            for match_idx, match in enumerate(matches):
                positive = positive_by_token.get(int(match.token_index))
                token_rank = per_token_seen.get(int(match.token_index), 0)
                per_token_seen[int(match.token_index)] = token_rank + 1
                gt_error_px = float(gt_match_errors[match_idx])
                gt_error_stride = float(gt_error_px / max(stride, 1e-6))
                patch_correct = bool(positive is not None and int(match.track_id) in positive.track_ids)
                stride_positive = bool(gt_error_px <= stride)
                weak_positive = bool(not (patch_correct or stride_positive) and gt_error_px <= 2.0 * stride)
                pnp_inlier = bool(
                    full_pnp_inlier_mask.shape[0] > match_idx and full_pnp_inlier_mask[match_idx]
                )
                match_rows.append(
                    {
                        "query_id": query_id,
                        "match_index": int(match_idx),
                        "match_rank": int(match_idx),
                        "token_match_rank": int(token_rank),
                        "token_index": int(match.token_index),
                        "track_id": int(match.track_id),
                        "source": match.source,
                        "xy": [float(match.xy[0]), float(match.xy[1])],
                        "xyz": [float(value) for value in np.asarray(match.xyz, dtype=np.float64).reshape(3)],
                        "similarity": float(match.similarity),
                        "similarity_margin": match.similarity_margin,
                        "observation_count": match.observation_count,
                        "visibility_count": match.visibility_count,
                        "landmark_variance": float(match.landmark_variance),
                        "landmark_reprojection_error": match.landmark_reprojection_error,
                        "landmark_ambiguity": match.landmark_ambiguity,
                        "landmark_quality": match.landmark_quality,
                        "quality_weighted_similarity": match.quality_weighted_similarity,
                        "pairwise_inlier_logit": match.pairwise_inlier_logit,
                        "pairwise_inlier_logprob": match.pairwise_inlier_logprob,
                        "pairwise_weighted_similarity": match.pairwise_weighted_similarity,
                        "map_reliability": match.map_reliability,
                        "pnp_uncertainty_scale": match.pnp_uncertainty_scale,
                        "local_consistency_support": match.local_consistency_support,
                        "local_consistency_score": match.local_consistency_score,
                        "pnp_soft_score": match.pnp_soft_score,
                        "patch_offset_confidence": match.patch_offset_confidence,
                        "patch_offset_sigma": match.patch_offset_sigma,
                        "patch_offset_applied": match.patch_offset_applied,
                        "patch_offset_norm_px": match.patch_offset_norm_px,
                        "patch_offset_consistency_before_px": match.patch_offset_consistency_before_px,
                        "patch_offset_consistency_after_px": match.patch_offset_consistency_after_px,
                        "distance_to_boundary_px": match.distance_to_boundary_px,
                        "gt_reproj_error_px": gt_error_px,
                        "gt_reproj_error_stride": gt_error_stride,
                        "baseline_reproj_residual_px": None
                        if baseline_match_errors is None
                        else float(baseline_match_errors[match_idx]),
                        "final_translation_error_m": None
                        if not np.isfinite(pose_error.translation_m)
                        else float(pose_error.translation_m),
                        "final_rotation_error_deg": None
                        if not np.isfinite(pose_error.rotation_deg)
                        else float(pose_error.rotation_deg),
                        "patch_correct": patch_correct,
                        "patch_positive_label": patch_correct,
                        "stride_positive_label": stride_positive,
                        "strong_positive_label": bool(patch_correct or stride_positive),
                        "weak_positive_label": weak_positive,
                        "ignore_label": weak_positive,
                        "hard_negative_label": bool((gt_error_px > 2.0 * stride) and pnp_inlier),
                        "positive_count": 0 if positive is None else int(positive.count),
                        "pnp_inlier": pnp_inlier,
                        "pnp_selected": bool(pnp_selected_mask.shape[0] > match_idx and pnp_selected_mask[match_idx]),
                    }
                )
        row = {
            "query_id": query_id,
            "submap_mode": args.submap_mode,
            "submap_reference_count": len(references),
            "full_visible_tracks": visibility_gate["full_visible_tracks"],
            "bank_visible_tracks": visibility_gate["bank_visible_tracks"],
            "bank_visibility_coverage": visibility_gate["bank_visibility_coverage"],
            "gt_visible_bank_tracks": gt_visible_count,
            "pre_limit_submap_gt_visible_tracks": pre_limit_gt_visible_count,
            "submap_gt_visible_tracks": submap_gt_visible_count,
            "pre_limit_visible_landmark_recall": pre_limit_visible_landmark_recall,
            "visible_landmark_recall": visible_landmark_recall,
            "reference_prior": reference_prior,
            "pre_limit_submap_landmark_count": pre_limit_submap_count,
            "submap_landmark_count": len(submap),
            "projected_landmarks": projected_landmarks,
            "coordinate_audit": {
                "query_feature_shape_chw": [int(v) for v in query_feature.shape],
                "camera_model_id": int(camera.model_id),
                "camera_width": int(camera.width),
                "camera_height": int(camera.height),
                "token_grid_scale_x_px": stride_x,
                "token_grid_scale_y_px": stride_y,
                "query_token_step": int(args.query_token_step),
                "patch_scale": float(args.patch_scale),
            },
            "match_count": len(matches),
            "pnp_match_count": int(pnp.match_count),
            "mean_similarity": _mean([float(match.similarity) for match in matches]),
            "map_reliability": {
                "all_matches": reliability_all_stats,
                "pnp_selected": reliability_pnp_stats,
                "pnp_inliers": reliability_inlier_stats,
            },
            "local_geometric_consistency": {
                "all_matches": local_all_stats,
                "pnp_selected": local_pnp_stats,
                "pnp_inliers": local_inlier_stats,
            },
            "mean_positive_count": _mean([float(item.count) for item in positives.by_token.values()]),
            "nonempty_patch_fraction": _mean([1.0 if item.count > 0 else 0.0 for item in positives.by_token.values()]),
            "positive_set_stats": positive_stats,
            "patch_geometry": patch_stats,
            "all_match_spatial": all_spatial_stats,
            "pnp_inlier_spatial": inlier_spatial_stats,
            "pnp_reprojection": pnp_residual_stats,
            "patch_offset_refinement": offset_summary,
            "pnp_solve": bool(pnp.success),
            "pnp_success": bool(pnp.success),
            "pnp_inlier_count": int(pnp.inlier_count),
            "pnp_inlier_ratio": float(pnp.inlier_ratio),
            "translation_error_m": None if not np.isfinite(pose_error.translation_m) else float(pose_error.translation_m),
            "rotation_error_deg": None if not np.isfinite(pose_error.rotation_deg) else float(pose_error.rotation_deg),
        }
        row["success_10cm_5deg"] = _success(row, 0.10, 5.0)
        row["success_25cm_10deg"] = _success(row, 0.25, 10.0)
        row["success_50cm_10deg"] = _success(row, 0.50, 10.0)
        row["success_1m_10deg"] = _success(row, 1.0, 10.0)
        row["pose_risk"] = pose_risk_score(row, PoseRiskConfig())
        rows.append(row)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))
    if args.output_matches_jsonl:
        matches_path = Path(args.output_matches_jsonl)
        matches_path.parent.mkdir(parents=True, exist_ok=True)
        matches_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in match_rows) + ("\n" if match_rows else ""))
    if args.output_superpoint_candidate_jsonl:
        candidates_path = Path(args.output_superpoint_candidate_jsonl)
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        candidates_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in superpoint_candidate_rows)
            + ("\n" if superpoint_candidate_rows else "")
        )

    labeled_rows = [row for row in rows if row["translation_error_m"] is not None]
    visible_recall_values = [row["visible_landmark_recall"] for row in rows if row["visible_landmark_recall"] is not None]
    reference_prior_rows = [dict(row["reference_prior"]) for row in rows]
    top1_prior_flags_25 = [
        flag
        for flag in (_prior_success(row, "top1", 0.25, 10.0) for row in reference_prior_rows)
        if flag is not None
    ]
    top1_prior_flags_50 = [
        flag
        for flag in (_prior_success(row, "top1", 0.50, 10.0) for row in reference_prior_rows)
        if flag is not None
    ]
    oracle_prior_flags_25 = [
        flag
        for flag in (_prior_success(row, "oracle", 0.25, 10.0) for row in reference_prior_rows)
        if flag is not None
    ]
    oracle_prior_flags_50 = [
        flag
        for flag in (_prior_success(row, "oracle", 0.50, 10.0) for row in reference_prior_rows)
        if flag is not None
    ]
    risk_coverages = tuple(
        float(item.strip())
        for item in str(args.risk_coverages).split(",")
        if item.strip()
    )
    summary = {
        "stage": "patch_to_3d_vfm_matching_baseline",
        "elapsed_sec": float(time.perf_counter() - started),
        "query_count": len(rows),
        "labeled_query_count": len(labeled_rows),
        "landmark_count": len(landmark_index),
        "map_source": map_source,
        "camera": {
            "source": camera_source,
            "model_id": int(camera.model_id),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
        },
        "matching_config": {
            **_matching_config_dict(config),
            "pnp_threshold_stride_multiplier": float(args.pnp_threshold_stride_multiplier),
            "patch_scale": float(args.patch_scale),
            "patch_at_k": int(args.patch_at_k),
            "safe_pairwise_checkpoint": args.safe_pairwise_checkpoint,
            "calibrated_candidate_checkpoint": args.calibrated_candidate_checkpoint,
            "calibrated_candidate_feature_set": args.calibrated_candidate_feature_set,
            "calibrated_confidence_checkpoint": args.calibrated_confidence_checkpoint,
            "calibrated_confidence_feature_set": args.calibrated_confidence_feature_set,
            "pairwise_inlier_weight": float(args.pairwise_inlier_weight),
            "pairwise_filter_keep_fraction": args.pairwise_filter_keep_fraction,
            "pairwise_filter_min_logit": args.pairwise_filter_min_logit,
            "pairwise_filter_min_logprob": args.pairwise_filter_min_logprob,
            "pairwise_device": args.pairwise_device or args.similarity_device,
            "pairwise_batch_size": int(args.pairwise_batch_size),
            "pnp_min_inliers": int(args.pnp_min_inliers),
            "pnp_confidence": float(args.pnp_confidence),
            "pnp_method": args.pnp_method,
            "pnp_refine_method": args.pnp_refine_method,
            "pnp_refine_lm": bool(args.pnp_refine_lm),
            "pnp_soft_order": {
                "mode": args.pnp_soft_order_mode,
                "top_n": None if args.pnp_soft_order_top_n <= 0 else int(args.pnp_soft_order_top_n),
            },
            "patch_offset_refinement": {
                "mode": args.patch_offset_mode,
                "checkpoint": args.patch_offset_checkpoint,
                "confidence_threshold": float(args.patch_offset_confidence_threshold),
                "max_stride": float(args.patch_offset_max_stride),
                "max_sigma": args.patch_offset_max_sigma,
                "max_consistency_residual_increase_px": args.patch_offset_max_consistency_residual_increase_px,
                "max_consistency_residual_px": args.patch_offset_max_consistency_residual_px,
                "free_oracle": bool(args.patch_offset_free_oracle),
                "bound_metric": args.patch_offset_bound_metric,
                "oracle_noise_px": float(args.patch_offset_oracle_noise_px),
                "fixed_pose_refine": {
                    "method": args.fixed_pose_refine_method,
                    "robust_loss": args.robust_fixed_pose_loss,
                    "robust_f_scale_px": float(args.robust_fixed_pose_f_scale_px),
                    "robust_max_nfev": int(args.robust_fixed_pose_max_nfev),
                    "weight_mode": args.robust_fixed_pose_weight_mode,
                    "mean_fixed_refine_count": _offset_summary_mean(rows, "fixed_same_inlier_count"),
                },
                "lowlevel": {
                    "feature_mode": args.lowlevel_feature_mode,
                    "template_radius_px": int(args.lowlevel_template_radius_px),
                    "search_radius_px": int(args.lowlevel_search_radius_px),
                    "search_step_px": int(args.lowlevel_search_step_px),
                    "min_score": float(args.lowlevel_min_score),
                    "min_confidence": float(args.lowlevel_min_confidence),
                },
                "superpoint": {
                    "device": args.superpoint_device or args.patch_offset_device or args.similarity_device,
                    "nms_radius": int(args.superpoint_nms_radius),
                    "keypoint_threshold": float(args.superpoint_keypoint_threshold),
                    "max_keypoints": int(args.superpoint_max_keypoints),
                    "remove_borders": int(args.superpoint_remove_borders),
                    "snap_radius_stride": float(args.superpoint_snap_radius_stride),
                    "min_score": float(args.superpoint_min_score),
                    "snap_strategy": args.superpoint_snap_strategy,
                    "availability_radii_px": [float(value) for value in superpoint_availability_radii],
                    "oracle_max_distance_px": float(args.superpoint_oracle_max_distance_px),
                },
                "landmark_conditioned_keypoint_selector": {
                    "candidate_radius_stride": float(args.lcks_candidate_radius_stride),
                    "support_radius_px": float(args.lcks_support_radius_px),
                    "min_query_score": float(args.lcks_min_query_score),
                    "min_support_score": float(args.lcks_min_support_score),
                    "descriptor_weight": float(args.lcks_descriptor_weight),
                    "query_score_weight": float(args.lcks_query_score_weight),
                    "center_penalty_weight": float(args.lcks_center_penalty_weight),
                    "support_distance_penalty_weight": float(args.lcks_support_distance_penalty_weight),
                    "score_threshold": float(args.lcks_score_threshold),
                    "support_selection_strategy": args.lcks_support_selection_strategy,
                },
                "device": args.patch_offset_device or args.similarity_device,
                "batch_size": int(args.patch_offset_batch_size),
                "mean_refined_count": _mean([
                    float(row.get("patch_offset_refinement", {}).get("refined_count", 0))
                    for row in rows
                ]),
                "mean_initial_inlier_count": _offset_summary_mean(rows, "initial_inlier_count"),
                "mean_fixed_same_inlier_count": _offset_summary_mean(rows, "fixed_same_inlier_count"),
                "mean_offset_applied_ratio": _offset_summary_mean(rows, "offset_applied_ratio"),
                "mean_lowlevel_inlier_count": _offset_summary_mean(rows, "inlier_count"),
                "mean_lowlevel_applied_count": _offset_summary_mean(rows, "applied_count"),
                "mean_lowlevel_skipped_by_inlier_count": _offset_summary_mean(rows, "skipped_by_inlier_count"),
                "mean_lowlevel_missing_support_count": _offset_summary_mean(rows, "missing_support_count"),
                "mean_lowlevel_missing_image_count": _offset_summary_mean(rows, "missing_image_count"),
                "mean_lowlevel_low_confidence_count": _offset_summary_mean(rows, "low_confidence_count"),
                "mean_lowlevel_offset_px": _offset_summary_mean(rows, "mean_offset_px"),
                "mean_lowlevel_score": _offset_summary_mean(rows, "mean_score"),
                "mean_lowlevel_confidence": _offset_summary_mean(rows, "mean_confidence"),
                "mean_learned_offset_px": _offset_summary_mean(rows, "mean_offset_px"),
                "mean_learned_offset_confidence": _offset_summary_mean(rows, "mean_confidence"),
                "mean_learned_offset_applied_confidence": _offset_summary_mean(rows, "mean_applied_confidence"),
                "mean_rejected_by_consistency_count": _offset_summary_mean(rows, "rejected_by_consistency_count"),
                "mean_offset_consistency_evaluated_count": _offset_summary_mean(rows, "consistency_evaluated_count"),
                "mean_offset_consistency_before_px": _offset_summary_mean(rows, "mean_consistency_before_px"),
                "mean_offset_consistency_after_px": _offset_summary_mean(rows, "mean_consistency_after_px"),
                "mean_offset_consistency_increase_px": _offset_summary_mean(rows, "mean_consistency_increase_px"),
                "mean_superpoint_keypoint_count": _offset_summary_mean(rows, "keypoint_count"),
                "mean_superpoint_availability": {
                    f"at_{int(radius) if float(radius).is_integer() else radius:g}px": _offset_summary_mean(
                        rows,
                        f"sp_availability_at_{int(radius) if float(radius).is_integer() else radius:g}px",
                    )
                    for radius in superpoint_availability_radii
                },
                "mean_nearest_sp_distance_px": _offset_summary_mean(rows, "nearest_sp_distance_px_mean"),
                "median_nearest_sp_distance_px": _offset_summary_mean(rows, "nearest_sp_distance_px_median"),
                "mean_gt_to_sp_distance_px": _offset_summary_mean(rows, "mean_gt_to_sp_distance_px"),
                "mean_skipped_by_patch_positive_count": _offset_summary_mean(rows, "skipped_by_patch_positive_count"),
                "mean_too_far_count": _offset_summary_mean(rows, "too_far_count"),
                "mean_snap_error_to_gt_px": _offset_summary_mean(rows, "mean_snap_error_to_gt_px"),
                "mean_snap_selection_accuracy_at_4px": _offset_summary_mean(rows, "snap_selection_accuracy_at_4px"),
                "mean_snap_selection_accuracy_at_8px": _offset_summary_mean(rows, "snap_selection_accuracy_at_8px"),
                "mean_sp_descriptor_top1_accuracy": _offset_summary_mean(rows, "sp_descriptor_top1_accuracy"),
                "mean_sp_descriptor_top5_accuracy": _offset_summary_mean(rows, "sp_descriptor_top5_accuracy"),
                "mean_support_availability_ratio": _offset_summary_mean(rows, "support_availability_ratio"),
                "mean_support_observation_missing_count": _offset_summary_mean(rows, "support_observation_missing_count"),
                "mean_support_keypoint_missing_count": _offset_summary_mean(rows, "support_keypoint_missing_count"),
            },
            "oracle_match_mode": args.oracle_match_mode,
            "oracle_max_positives_per_token": int(args.oracle_max_positives_per_token),
            "spatial_diversity_pnp": {
                "enabled": bool(args.enable_spatial_diversity_pnp),
                "grid_rows": int(args.spatial_diversity_grid_rows),
                "grid_cols": int(args.spatial_diversity_grid_cols),
                "max_per_cell": int(args.spatial_diversity_max_per_cell),
                "max_matches": args.spatial_diversity_max_matches,
                "score_mode": args.spatial_diversity_score_mode,
                "min_world_z_range_m": args.spatial_diversity_min_world_z_range_m,
                "min_planarity_ratio": args.spatial_diversity_min_planarity_ratio,
            },
        },
        "submap": {
            "mode": args.submap_mode,
            "top_n": args.submap_top_n,
            "max_landmarks": args.max_submap_landmarks,
            "mean_reference_count": _mean([float(row["submap_reference_count"]) for row in rows]),
            "mean_landmark_count": _mean([float(row["submap_landmark_count"]) for row in rows]),
            "mean_projected_landmarks": _mean([float(row["projected_landmarks"]) for row in rows]),
        },
        "visible_landmark_recall": {
            "mean": _mean([float(value) for value in visible_recall_values]),
            "median": _median_present([float(value) for value in visible_recall_values]),
            "p25": _quantile_present([float(value) for value in visible_recall_values], 0.25),
            "p75": _quantile_present([float(value) for value in visible_recall_values], 0.75),
            "mean_gt_visible_bank_tracks": _mean([float(row["gt_visible_bank_tracks"]) for row in rows]),
            "mean_submap_gt_visible_tracks": _mean([float(row["submap_gt_visible_tracks"]) for row in rows]),
            "mean_pre_limit_submap_gt_visible_tracks": _mean(
                [float(row["pre_limit_submap_gt_visible_tracks"]) for row in rows]
            ),
            "mean_pre_limit_recall": _mean(
                [
                    float(row["pre_limit_visible_landmark_recall"])
                    for row in rows
                    if row["pre_limit_visible_landmark_recall"] is not None
                ]
            ),
        },
        "reference_prior": {
            "mean_candidate_count": _mean([float(row["candidate_count"]) for row in reference_prior_rows]),
            "top1": {
                "median_translation_error_m": _median_present(
                    [row["top1_translation_error_m"] for row in reference_prior_rows]
                ),
                "median_rotation_error_deg": _median_present(
                    [row["top1_rotation_error_deg"] for row in reference_prior_rows]
                ),
                "success_25cm_10deg": _rate_present(top1_prior_flags_25),
                "success_50cm_10deg": _rate_present(top1_prior_flags_50),
            },
            "oracle": {
                "median_translation_error_m": _median_present(
                    [row["oracle_translation_error_m"] for row in reference_prior_rows]
                ),
                "median_rotation_error_deg": _median_present(
                    [row["oracle_rotation_error_deg"] for row in reference_prior_rows]
                ),
                "success_25cm_10deg": _rate_present(oracle_prior_flags_25),
                "success_50cm_10deg": _rate_present(oracle_prior_flags_50),
            },
        },
        "positive_set_summary": {
            "mean_visible_landmark_count": _mean([float(row["positive_set_stats"]["visible_landmark_count"]) for row in rows]),
            "mean_positive_landmark_count": _mean([float(row["positive_set_stats"]["positive_landmark_count"]) for row in rows]),
            "mean_positives_per_token": _mean([float(row["positive_set_stats"]["mean_positives_per_token"]) for row in rows]),
            "median_positives_per_token": _median_present(
                [float(row["positive_set_stats"]["median_positives_per_token"]) for row in rows]
            ),
            "mean_positives_per_nonempty_token": _mean(
                [float(row["positive_set_stats"]["mean_positives_per_nonempty_token"]) for row in rows]
            ),
            "mean_zero_positive_token_ratio": _mean(
                [float(row["positive_set_stats"]["zero_positive_token_ratio"]) for row in rows]
            ),
            "mean_nonempty_patch_fraction": _mean(
                [float(row["positive_set_stats"]["nonempty_patch_fraction"]) for row in rows]
            ),
            "mean_positive_landmark_density_per_token": _mean(
                [float(row["positive_set_stats"]["positive_landmark_density_per_token"]) for row in rows]
            ),
        },
        "mean_match_count": _mean([float(row["match_count"]) for row in rows]),
        "mean_pnp_match_count": _mean([float(row["pnp_match_count"]) for row in rows]),
        "map_reliability_summary": {
            "mean_all_match_reliability": _mean(
                [
                    float(row["map_reliability"]["all_matches"]["mean"])
                    for row in rows
                    if row["map_reliability"]["all_matches"]["mean"] is not None
                ]
            ),
            "mean_pnp_selected_reliability": _mean(
                [
                    float(row["map_reliability"]["pnp_selected"]["mean"])
                    for row in rows
                    if row["map_reliability"]["pnp_selected"]["mean"] is not None
                ]
            ),
            "mean_pnp_inlier_reliability": _mean(
                [
                    float(row["map_reliability"]["pnp_inliers"]["mean"])
                    for row in rows
                    if row["map_reliability"]["pnp_inliers"]["mean"] is not None
                ]
            ),
            "mean_uncertainty_scale": _mean(
                [
                    float(row["map_reliability"]["all_matches"]["mean_uncertainty_scale"])
                    for row in rows
                    if row["map_reliability"]["all_matches"]["mean_uncertainty_scale"] is not None
                ]
            ),
        },
        "local_geometric_consistency_summary": {
            "mean_all_match_support": _mean(
                [
                    float(row["local_geometric_consistency"]["all_matches"]["mean_support"])
                    for row in rows
                    if row["local_geometric_consistency"]["all_matches"]["mean_support"] is not None
                ]
            ),
            "mean_all_match_score": _mean(
                [
                    float(row["local_geometric_consistency"]["all_matches"]["mean_score"])
                    for row in rows
                    if row["local_geometric_consistency"]["all_matches"]["mean_score"] is not None
                ]
            ),
            "mean_pnp_inlier_support": _mean(
                [
                    float(row["local_geometric_consistency"]["pnp_inliers"]["mean_support"])
                    for row in rows
                    if row["local_geometric_consistency"]["pnp_inliers"]["mean_support"] is not None
                ]
            ),
            "mean_pnp_inlier_score": _mean(
                [
                    float(row["local_geometric_consistency"]["pnp_inliers"]["mean_score"])
                    for row in rows
                    if row["local_geometric_consistency"]["pnp_inliers"]["mean_score"] is not None
                ]
            ),
        },
        "mean_patch_at_1": _mean([float(row["patch_geometry"]["patch_at_1"]) for row in rows]),
        f"mean_patch_at_{int(args.patch_at_k)}": _mean([float(row["patch_geometry"][f"patch_at_{int(args.patch_at_k)}"]) for row in rows]),
        "mean_gt_precision_5px": _mean([float(row["patch_geometry"]["gt_precision_5px"]) for row in rows]),
        "mean_gt_precision_16px": _mean([float(row["patch_geometry"]["gt_precision_16px"]) for row in rows]),
        "mean_gt_precision_stride": _mean([float(row["patch_geometry"]["gt_precision_stride"]) for row in rows]),
        "mean_gt_precision_2stride": _mean([float(row["patch_geometry"]["gt_precision_2stride"]) for row in rows]),
        "median_gt_reproj_median_px": _median_present(
            [row["patch_geometry"]["gt_reproj_median_px"] for row in rows]
        ),
        "mean_pnp_inlier_patch_at_1": _mean(
            [
                float(row["patch_geometry"]["pnp_inlier_patch_at_1"])
                for row in rows
                if row["patch_geometry"]["pnp_inlier_patch_at_1"] is not None
            ]
        ),
        f"mean_pnp_inlier_patch_at_{int(args.patch_at_k)}": _mean(
            [
                float(row["patch_geometry"][f"pnp_inlier_patch_at_{int(args.patch_at_k)}"])
                for row in rows
                if row["patch_geometry"].get(f"pnp_inlier_patch_at_{int(args.patch_at_k)}") is not None
            ]
        ),
        "mean_pnp_inlier_gt_precision_stride": _mean(
            [
                float(row["patch_geometry"]["pnp_inlier_gt_precision_stride"])
                for row in rows
                if row["patch_geometry"]["pnp_inlier_gt_precision_stride"] is not None
            ]
        ),
        "mean_pnp_inlier_count": _mean([float(row["pnp_inlier_count"]) for row in rows]),
        "mean_pnp_inlier_ratio": _mean([float(row["pnp_inlier_ratio"]) for row in rows]),
        "mean_pnp_reproj_inlier_median_px": _mean(
            [
                float(row["pnp_reprojection"]["pnp_reproj_inlier_median_px"])
                for row in rows
                if row["pnp_reprojection"].get("pnp_reproj_inlier_median_px") is not None
            ]
        ),
        "mean_pnp_inlier_bbox_area_frac": _mean(
            [
                float(row["pnp_inlier_spatial"]["bbox_area_frac"])
                for row in rows
                if row["pnp_inlier_spatial"].get("bbox_area_frac") is not None
            ]
        ),
        "mean_pnp_inlier_grid_4x4_occupancy_frac": _mean(
            [
                float(row["pnp_inlier_spatial"]["grid_4x4_occupancy_frac"])
                for row in rows
                if row["pnp_inlier_spatial"].get("grid_4x4_occupancy_frac") is not None
            ]
        ),
        "mean_pnp_inlier_convex_hull_area_frac": _mean(
            [
                float(row["pnp_inlier_spatial"]["convex_hull_area_frac"])
                for row in rows
                if row["pnp_inlier_spatial"].get("convex_hull_area_frac") is not None
            ]
        ),
        "mean_pnp_inlier_depth_range_m": _mean(
            [
                float(row["pnp_inlier_spatial"]["depth_range_m"])
                for row in rows
                if row["pnp_inlier_spatial"].get("depth_range_m") is not None
            ]
        ),
        "mean_pnp_inlier_xyz_planarity_ratio": _mean(
            [
                float(row["pnp_inlier_spatial"]["xyz_planarity_ratio"])
                for row in rows
                if row["pnp_inlier_spatial"].get("xyz_planarity_ratio") is not None
            ]
        ),
        "mean_pnp_inlier_xyz_linearity_ratio": _mean(
            [
                float(row["pnp_inlier_spatial"]["xyz_linearity_ratio"])
                for row in rows
                if row["pnp_inlier_spatial"].get("xyz_linearity_ratio") is not None
            ]
        ),
        "pnp_solve_rate": _mean([1.0 if row["pnp_solve"] else 0.0 for row in rows]),
        "mean_pose_risk": _mean([float(row["pose_risk"]) for row in rows]),
        "selective_localization": {
            "success_25cm_10deg": selective_localization_summary(
                rows,
                coverages=risk_coverages,
                success_key="success_25cm_10deg",
            ),
            "success_50cm_10deg": selective_localization_summary(
                rows,
                coverages=risk_coverages,
                success_key="success_50cm_10deg",
            ),
        },
        "pnp_success_rate": _mean([1.0 if row["pnp_success"] else 0.0 for row in rows]),
        "success_10cm_5deg_all_queries": _mean([1.0 if row["success_10cm_5deg"] else 0.0 for row in rows]),
        "success_25cm_10deg_all_queries": _mean([1.0 if row["success_25cm_10deg"] else 0.0 for row in rows]),
        "success_50cm_10deg_all_queries": _mean([1.0 if row["success_50cm_10deg"] else 0.0 for row in rows]),
        "success_1m_10deg_all_queries": _mean([1.0 if row["success_1m_10deg"] else 0.0 for row in rows]),
        "success_10cm_5deg": _mean([1.0 if row["success_10cm_5deg"] else 0.0 for row in labeled_rows]),
        "success_25cm_10deg": _mean([1.0 if row["success_25cm_10deg"] else 0.0 for row in labeled_rows]),
        "success_50cm_10deg": _mean([1.0 if row["success_50cm_10deg"] else 0.0 for row in labeled_rows]),
        "success_1m_10deg": _mean([1.0 if row["success_1m_10deg"] else 0.0 for row in labeled_rows]),
        "median_translation_error_m": None
        if not labeled_rows
        else float(np.median([float(row["translation_error_m"]) for row in labeled_rows])),
        "median_rotation_error_deg": None
        if not labeled_rows
        else float(np.median([float(row["rotation_error_deg"]) for row in labeled_rows])),
        "inputs": {
            "query_manifest": args.query_manifest,
            "landmark_bank": args.landmark_bank,
            "semidense_anchor_npz": args.semidense_anchor_npz,
            "track_observations": args.track_observations,
            "visibility_index": args.visibility_index,
            "query_pose_file": args.query_pose_file,
            "candidate_bank": args.candidate_bank,
        },
        "outputs": {
            "rows": str(output_jsonl),
            "matches": args.output_matches_jsonl,
            "superpoint_candidates": args.output_superpoint_candidate_jsonl,
        },
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
