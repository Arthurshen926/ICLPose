"""Evaluate Stage C3.1 contextual landmark-maplet matching."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_patch_to_3d_vfm_matching import _load_reference_pose_priors, _reference_prior_summary
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
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex, count_projected_landmarks, filter_landmarks_by_visibility
from feature_extract.vfm.local_maplet_matching import (
    ContextualLandmarkMatchingConfig,
    build_covisibility_maplets,
    build_knn_maplets,
    build_ref_neighborhood_maplets,
    local_maplet_bank_stats,
    match_query_patches_to_contextual_landmarks,
)
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_footprint_matching import observation_index_from_colmap_observations
from feature_extract.vfm.patch_to_3d_matching import (
    build_patch_positive_sets,
    evaluate_patch_matches,
    filter_landmarks_by_projected_visibility,
    patch_positive_set_stats,
    patch_uncertainty_pnp_threshold,
)
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    estimate_pose_pnp_ransac,
    filter_landmarks_by_reference_images,
    match_reprojection_errors,
    match_spatial_distribution_stats,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
)
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def _median_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not present else float(np.median(present))


def _success(row: dict[str, object], translation_m: float, rotation_deg: float) -> bool:
    return bool(
        row["translation_error_m"] is not None
        and float(row["translation_error_m"]) <= float(translation_m)
        and row["rotation_error_deg"] is not None
        and float(row["rotation_error_deg"]) <= float(rotation_deg)
    )


def _track_id_set(index: LandmarkMapIndex) -> set[int]:
    return {int(track_id) for track_id in np.asarray(index.track_ids).tolist()}


def _filter_landmark_index(
    index: LandmarkMapIndex,
    min_observation_count: int,
    max_landmark_variance: float | None,
    max_landmark_reprojection_error: float | None,
) -> LandmarkMapIndex:
    if len(index) == 0:
        return index
    mask = index.observation_counts >= int(min_observation_count)
    if max_landmark_variance is not None:
        mask &= index.mean_variances <= float(max_landmark_variance)
    if max_landmark_reprojection_error is not None:
        mask &= index.reprojection_errors <= float(max_landmark_reprojection_error)
    return index.subset(mask)


def _prior_success(prior: dict[str, object], prefix: str, translation_m: float, rotation_deg: float) -> bool | None:
    translation = prior.get(f"{prefix}_translation_error_m")
    rotation = prior.get(f"{prefix}_rotation_error_deg")
    if translation is None or rotation is None:
        return None
    return bool(float(translation) <= float(translation_m) and float(rotation) <= float(rotation_deg))


def _config_dict(config: ContextualLandmarkMatchingConfig) -> dict[str, object]:
    return dict(config.__dict__)


def _summary_metric_row(label: str, summary: dict[str, object]) -> str:
    def value(key: str) -> str:
        item = summary.get(key)
        if item is None:
            return "n/a"
        if isinstance(item, (int, float)):
            return f"{float(item):.3f}"
        return str(item)

    return (
        f"| {label} | {value('success_25cm_10deg')} | {value('success_50cm_10deg')} | "
        f"{value('median_translation_error_m')} | {value('median_rotation_error_deg')} | "
        f"{value('mean_pnp_inlier_patch_at_1')} | {value('mean_pnp_inlier_count')} |"
    )


def _load_summary(path: str) -> dict[str, object] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text())


def _write_markdown_summary(
    summary: dict[str, object],
    path: Path,
    c25_summary: dict[str, object] | None = None,
    c3_summary: dict[str, object] | None = None,
) -> None:
    lines = [
        "# Stage C3.1 Contextual Landmark Matching",
        "",
        "C3.1 keeps the final PnP correspondence as a single query patch to a single 3D landmark, while using local maplet context only for match scoring.",
        "",
        "## Main Table",
        "",
        "| Method | S@25cm/10deg | S@50cm/10deg | Median t (m) | Median r (deg) | Inlier Patch@1 | Inliers |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if c25_summary is not None:
        lines.append(_summary_metric_row("C2.5 baseline", c25_summary))
    if c3_summary is not None:
        lines.append(_summary_metric_row("C3-new reference footprint", c3_summary))
    descriptor = summary.get("matching_config", {})
    label = (
        f"C3.1 {descriptor.get('score_mode', 'contextual')} "
        f"{summary.get('maplet_type', 'maplet')} k{summary.get('maplet_k', '')} "
        f"q{descriptor.get('query_context', '')} w{descriptor.get('context_weight', '')}"
    )
    lines.append(_summary_metric_row(label, summary))
    lines.extend(
        [
            "",
            "## Maplet Diagnostics",
            "",
            f"- mean neighbors: {float(summary.get('mean_maplet_neighbors', 0.0)):.3f}",
            f"- zero-neighbor ratio: {float(summary.get('mean_zero_neighbor_ratio', 0.0)):.3f}",
            f"- mean context radius: {float(summary.get('mean_context_radius', 0.0)):.3f}",
            f"- mean visible landmark recall: {float(summary.get('mean_visible_landmark_recall', 0.0)):.3f}",
            "",
            "## Inputs",
            "",
        ]
    )
    for key, value in dict(summary.get("inputs") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate Stage C3.1 contextual landmark-maplet matching")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--visibility_index", default="")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--submap_mode", default="reference_visibility", choices=("reference_visibility", "gt_visible", "none"))
    parser.add_argument("--submap_top_n", type=int, default=10)
    parser.add_argument("--max_submap_landmarks", type=int, default=20000)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--maplet_type", default="knn", choices=("knn", "covis", "ref_neighbor"))
    parser.add_argument("--maplet_k", type=int, default=8)
    parser.add_argument("--maplet_radius", type=float, default=0.0)
    parser.add_argument("--ref_neighbor_radius", type=int, default=1)
    parser.add_argument("--query_context", default="3x3", choices=("1x1", "3x3", "5x5"))
    parser.add_argument("--context_weight", type=float, default=0.25)
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--quality_weight", type=float, default=0.0)
    parser.add_argument("--context_pool", default="quality_mean", choices=("mean", "quality_mean", "topk"))
    parser.add_argument(
        "--score_mode",
        default="anchor_context",
        choices=("anchor_only", "context_only", "anchor_context", "anchor_context_quality"),
    )
    parser.add_argument("--match_mode", default="mnn", choices=("nn", "mnn", "soft_mutual"))
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--mutual_top_k", type=int, default=1)
    parser.add_argument("--min_similarity", type=float, default=0.2)
    parser.add_argument("--query_token_step", type=int, default=1)
    parser.add_argument("--min_observation_count", type=int, default=2)
    parser.add_argument("--max_landmark_variance", type=float, default=None)
    parser.add_argument("--max_landmark_reprojection_error", type=float, default=None)
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--match_block_size", type=int, default=1024)
    parser.add_argument("--similarity_device", default="cpu")
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--patch_at_k", type=int, default=5)
    parser.add_argument("--pnp_threshold_stride_multiplier", type=float, default=0.75)
    parser.add_argument("--pnp_min_inliers", type=int, default=0)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--pnp_confidence", type=float, default=0.999)
    parser.add_argument("--pnp_method", default="EPNP", choices=("AP3P", "EPNP", "ITERATIVE", "P3P", "SQPNP"))
    parser.add_argument("--pnp_refine_method", default="LM", choices=("none", "LM", "VVS"))
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_matches_jsonl", default="")
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--summary_md", default="")
    parser.add_argument("--baseline_c25_summary", default="")
    parser.add_argument("--baseline_c3_summary", default="")
    args = parser.parse_args(argv)

    if args.submap_mode == "reference_visibility" and not args.candidate_bank:
        raise ValueError("candidate_bank is required when submap_mode=reference_visibility")

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    xyz_by_track, reprojection_error_by_track = _load_track_stats(Path(args.track_observations))
    colmap_observations = None
    observations_by_image = None
    if args.maplet_type == "ref_neighbor":
        colmap_observations = load_colmap_track_observations_jsonl(Path(args.track_observations))
        observations_by_image = observation_index_from_colmap_observations(colmap_observations)
    bank = load_selected_track_bank_npz(Path(args.landmark_bank))
    landmark_index = LandmarkMapIndex.from_track_bank(bank, xyz_by_track, reprojection_error_by_track)
    landmark_index = _filter_landmark_index(
        landmark_index,
        min_observation_count=int(args.min_observation_count),
        max_landmark_variance=args.max_landmark_variance,
        max_landmark_reprojection_error=args.max_landmark_reprojection_error,
    )
    visibility_index = None
    if args.visibility_index:
        visibility_index = LandmarkVisibilityIndex.load_npz(Path(args.visibility_index))
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    reference_submaps = _load_reference_submaps(args.candidate_bank, args.submap_top_n)
    reference_pose_priors = _load_reference_pose_priors(args.candidate_bank, args.submap_top_n)
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}

    config = ContextualLandmarkMatchingConfig(
        top_k=int(args.top_k),
        mutual_top_k=int(args.mutual_top_k),
        match_mode=args.match_mode,
        query_context=args.query_context,
        context_weight=float(args.context_weight),
        anchor_weight=float(args.anchor_weight),
        quality_weight=float(args.quality_weight),
        score_mode=args.score_mode,
        context_pool=args.context_pool,
        min_similarity=float(args.min_similarity),
        query_token_step=int(args.query_token_step),
        max_matches=None if int(args.max_matches) <= 0 else int(args.max_matches),
        block_size=int(args.match_block_size),
        similarity_device=args.similarity_device,
    )

    rows = []
    match_rows = []
    records = list(manifest.records)
    if args.max_queries > 0:
        records = records[: int(args.max_queries)]
    all_reference_images = []
    if observations_by_image is not None:
        all_reference_images = sorted(observations_by_image.keys())
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
        if args.submap_mode == "gt_visible":
            submap = gt_visible_submap
            references = all_reference_images
            visibility_gate = {
                "full_visible_tracks": len(submap),
                "bank_visible_tracks": len(submap),
                "bank_visibility_coverage": 1.0 if len(submap) else 0.0,
            }
        elif args.submap_mode == "reference_visibility":
            if visibility_index is None:
                submap = filter_landmarks_by_reference_images(landmark_index, references)
                visibility_gate = {
                    "full_visible_tracks": len(submap),
                    "bank_visible_tracks": len(submap),
                    "bank_visibility_coverage": 1.0 if len(submap) else 0.0,
                }
            else:
                submap, visibility_gate = filter_landmarks_by_visibility(landmark_index, visibility_index, references)
        else:
            references = all_reference_images
            visibility_gate = {"full_visible_tracks": None, "bank_visible_tracks": None, "bank_visibility_coverage": None}
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
            patch_scale=float(args.patch_scale),
        )
        if args.maplet_type == "knn":
            maplet_bank = build_knn_maplets(
                submap,
                maplet_k=int(args.maplet_k),
                radius=None if float(args.maplet_radius) <= 0.0 else float(args.maplet_radius),
                context_pool=args.context_pool,
            )
        elif args.maplet_type == "covis":
            maplet_bank = build_covisibility_maplets(
                submap,
                reference_image_ids=references,
                maplet_k=int(args.maplet_k),
                context_pool=args.context_pool,
            )
        else:
            if observations_by_image is None:
                raise RuntimeError("ref_neighbor maplets require loaded observations")
            maplet_bank = build_ref_neighborhood_maplets(
                submap,
                observations_by_image,
                references,
                token_width,
                token_height,
                maplet_k=int(args.maplet_k),
                cell_radius=int(args.ref_neighbor_radius),
                context_pool=args.context_pool,
            )
        matches = match_query_patches_to_contextual_landmarks(
            query_feature,
            submap,
            maplet_bank,
            config,
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        pnp_threshold = patch_uncertainty_pnp_threshold(stride, float(args.pnp_threshold_stride_multiplier))
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=pnp_threshold,
            confidence=float(args.pnp_confidence),
            iterations=int(args.pnp_iterations),
            min_inliers=int(args.pnp_min_inliers),
            pnp_method=args.pnp_method,
            refine_method=args.pnp_refine_method,
        )
        patch_stats = evaluate_patch_matches(
            matches,
            positives,
            gt_pose.pose_w2c,
            camera,
            stride_px=stride,
            pnp_inlier_mask=pnp.inlier_mask,
            top_k=int(args.patch_at_k),
        )
        positive_stats = patch_positive_set_stats(positives)
        pose_error = pnp_pose_error(pnp.pose_w2c, gt_pose.pose_w2c)
        all_spatial_stats = match_spatial_distribution_stats(matches, int(camera.width), int(camera.height))
        inlier_spatial_stats = match_spatial_distribution_stats(
            matches,
            int(camera.width),
            int(camera.height),
            pnp.inlier_mask,
        )
        pnp_residual_stats = pnp_reprojection_residual_stats(
            matches,
            pnp.pose_w2c,
            camera,
            inlier_mask=pnp.inlier_mask,
        )
        projected_landmarks = count_projected_landmarks(submap, gt_pose.pose_w2c, camera)
        maplet_stats = local_maplet_bank_stats(maplet_bank)
        if args.output_matches_jsonl:
            gt_errors = match_reprojection_errors(matches, gt_pose.pose_w2c, camera)
            for match_idx, match in enumerate(matches):
                positive = positives.by_token.get(int(match.token_index))
                match_rows.append(
                    {
                        "query_id": query_id,
                        "match_index": int(match_idx),
                        "token_index": int(match.token_index),
                        "track_id": int(match.track_id),
                        "similarity": float(match.similarity),
                        "source": match.source,
                        "patch_correct": bool(positive is not None and int(match.track_id) in positive.track_ids),
                        "gt_reproj_error_px": float(gt_errors[match_idx]),
                        "gt_reproj_error_stride": float(gt_errors[match_idx] / max(stride, 1e-6)),
                        "pnp_inlier": bool(pnp.inlier_mask[match_idx]) if match_idx < pnp.inlier_mask.shape[0] else False,
                        "local_context_support": match.local_consistency_support,
                        "local_context_score": match.local_consistency_score,
                    }
                )
        reference_prior = _reference_prior_summary(reference_pose_priors.get(query_id, []))
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
            "maplet_bank": maplet_stats,
            "positive_set_stats": positive_stats,
            "match_count": len(matches),
            "pnp_match_count": int(pnp.match_count),
            "mean_similarity": _mean([float(match.similarity) for match in matches]),
            "patch_geometry": patch_stats,
            "all_match_spatial": all_spatial_stats,
            "pnp_inlier_spatial": inlier_spatial_stats,
            "pnp_reprojection": pnp_residual_stats,
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
        rows.append(row)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))
    if args.output_matches_jsonl:
        matches_path = Path(args.output_matches_jsonl)
        matches_path.parent.mkdir(parents=True, exist_ok=True)
        matches_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in match_rows) + ("\n" if match_rows else ""))

    labeled_rows = [row for row in rows if row["translation_error_m"] is not None]
    reference_prior_rows = [dict(row["reference_prior"]) for row in rows]
    top1_prior_flags_25 = [
        flag
        for flag in (_prior_success(row, "top1", 0.25, 10.0) for row in reference_prior_rows)
        if flag is not None
    ]
    oracle_prior_flags_25 = [
        flag
        for flag in (_prior_success(row, "oracle", 0.25, 10.0) for row in reference_prior_rows)
        if flag is not None
    ]
    summary = {
        "stage": "stage_c31_contextual_landmark_matching",
        "elapsed_sec": float(time.perf_counter() - started),
        "query_count": len(rows),
        "labeled_query_count": len(labeled_rows),
        "landmark_count": len(landmark_index),
        "maplet_type": args.maplet_type,
        "maplet_k": int(args.maplet_k),
        "camera": {
            "source": camera_source,
            "model_id": int(camera.model_id),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
        },
        "matching_config": {
            **_config_dict(config),
            "maplet_radius": None if float(args.maplet_radius) <= 0.0 else float(args.maplet_radius),
            "ref_neighbor_radius": int(args.ref_neighbor_radius),
            "pnp_threshold_stride_multiplier": float(args.pnp_threshold_stride_multiplier),
            "pnp_min_inliers": int(args.pnp_min_inliers),
            "pnp_confidence": float(args.pnp_confidence),
            "pnp_method": args.pnp_method,
            "pnp_refine_method": args.pnp_refine_method,
            "patch_scale": float(args.patch_scale),
            "patch_at_k": int(args.patch_at_k),
        },
        "submap": {
            "mode": args.submap_mode,
            "top_n": args.submap_top_n,
            "max_landmarks": args.max_submap_landmarks,
            "mean_reference_count": _mean([float(row["submap_reference_count"]) for row in rows]),
            "mean_landmark_count": _mean([float(row["submap_landmark_count"]) for row in rows]),
            "mean_projected_landmarks": _mean([float(row["projected_landmarks"]) for row in rows]),
        },
        "reference_prior": {
            "top1_success_25cm_10deg": _mean([1.0 if flag else 0.0 for flag in top1_prior_flags_25]),
            "oracle_success_25cm_10deg": _mean([1.0 if flag else 0.0 for flag in oracle_prior_flags_25]),
        },
        "mean_visible_landmark_recall": _mean(
            [float(row["visible_landmark_recall"]) for row in rows if row["visible_landmark_recall"] is not None]
        ),
        "mean_maplet_neighbors": _mean([float(row["maplet_bank"]["mean_neighbors"]) for row in rows]),
        "mean_zero_neighbor_ratio": _mean([float(row["maplet_bank"]["zero_neighbor_ratio"]) for row in rows]),
        "mean_context_radius": _mean([float(row["maplet_bank"]["mean_context_radius"]) for row in rows]),
        "mean_context_feature_variance": _mean(
            [float(row["maplet_bank"]["mean_context_feature_variance"]) for row in rows]
        ),
        "mean_covisibility_strength": _mean(
            [float(row["maplet_bank"]["mean_covisibility_strength"]) for row in rows]
        ),
        "mean_match_count": _mean([float(row["match_count"]) for row in rows]),
        "mean_pnp_match_count": _mean([float(row["pnp_match_count"]) for row in rows]),
        "mean_patch_at_1": _mean([float(row["patch_geometry"]["patch_at_1"]) for row in rows]),
        f"mean_patch_at_{int(args.patch_at_k)}": _mean(
            [float(row["patch_geometry"][f"patch_at_{int(args.patch_at_k)}"]) for row in rows]
        ),
        "mean_pnp_inlier_patch_at_1": _mean(
            [
                float(row["patch_geometry"]["pnp_inlier_patch_at_1"])
                for row in rows
                if row["patch_geometry"]["pnp_inlier_patch_at_1"] is not None
            ]
        ),
        "mean_pnp_inlier_count": _mean([float(row["pnp_inlier_count"]) for row in rows]),
        "mean_pnp_inlier_ratio": _mean([float(row["pnp_inlier_ratio"]) for row in rows]),
        "pnp_solve_rate": _mean([1.0 if row["pnp_solve"] else 0.0 for row in rows]),
        "success_10cm_5deg": _mean([1.0 if row["success_10cm_5deg"] else 0.0 for row in labeled_rows]),
        "success_25cm_10deg": _mean([1.0 if row["success_25cm_10deg"] else 0.0 for row in labeled_rows]),
        "success_50cm_10deg": _mean([1.0 if row["success_50cm_10deg"] else 0.0 for row in labeled_rows]),
        "success_1m_10deg": _mean([1.0 if row["success_1m_10deg"] else 0.0 for row in labeled_rows]),
        "median_translation_error_m": _median_present([row["translation_error_m"] for row in labeled_rows]),
        "median_rotation_error_deg": _median_present([row["rotation_error_deg"] for row in labeled_rows]),
        "inputs": {
            "query_manifest": args.query_manifest,
            "landmark_bank": args.landmark_bank,
            "track_observations": args.track_observations,
            "visibility_index": args.visibility_index,
            "query_pose_file": args.query_pose_file,
            "candidate_bank": args.candidate_bank,
        },
        "outputs": {"rows": str(output_jsonl), "matches": args.output_matches_jsonl},
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.summary_md:
        _write_markdown_summary(
            summary,
            Path(args.summary_md),
            c25_summary=_load_summary(args.baseline_c25_summary),
            c3_summary=_load_summary(args.baseline_c3_summary),
        )


if __name__ == "__main__":
    main()
