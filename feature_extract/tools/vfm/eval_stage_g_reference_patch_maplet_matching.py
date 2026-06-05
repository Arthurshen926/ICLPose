"""Evaluate Stage G reference-patch maplet matching.

This diagnostic tests true patch-to-patch matching: query VFM token descriptor
to reference VFM token descriptor.  Each reference token carries a 3D footprint
defined by existing SfM landmarks observed inside that reference patch.
"""

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
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_footprint_matching import observation_index_from_colmap_observations
from feature_extract.vfm.patch_to_3d_matching import build_patch_positive_sets, patch_positive_set_stats
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, filter_landmarks_by_reference_images
from feature_extract.vfm.query_to_3d_matching import estimate_pose_pnp_ransac, pnp_pose_error
from feature_extract.vfm.reference_patch_maplets import (
    ReferencePatchMapletConfig,
    apply_maplet_verifier_to_matches,
    build_reference_patch_maplet_bank,
    collect_maplet_verifier_training_examples,
    collect_support_selector_training_examples,
    compute_patch_context_feature_map,
    evaluate_reference_patch_maplet_matches,
    expand_reference_patch_maplet_matches_to_query_to_3d,
    fit_maplet_verifier_model,
    fit_support_selector_model,
    match_query_patches_to_reference_patch_maplets,
    project_feature_map_tokens,
    reference_patch_maplet_positive_stats,
    score_maplet_verifier_model,
)
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl
from feature_extract.vfm.vfm_aware_landmarks import load_token_feature_map
from feature_extract.vfm.patch_selector_training import load_safe_patch_selector_checkpoint


def _track_stats_from_observations(observations) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    xyz_by_track: dict[int, np.ndarray] = {}
    errors_by_track: dict[int, list[float]] = {}
    for obs in observations:
        xyz_by_track.setdefault(int(obs.track_id), np.asarray(obs.xyz, dtype=np.float64).reshape(3))
        errors_by_track.setdefault(int(obs.track_id), []).append(float(obs.reprojection_error))
    return xyz_by_track, {
        int(track_id): float(np.mean(errors)) for track_id, errors in errors_by_track.items() if errors
    }


def _load_reference_feature_maps(
    records_by_image: dict[str, object],
    references: Sequence[str],
    layer_name: str,
    selector_run=None,
    selector_device: str = "cpu",
    selector_batch_size: int = 4096,
    patch_context: str = "1x1",
) -> dict[str, np.ndarray]:
    feature_maps: dict[str, np.ndarray] = {}
    for image_id in dict.fromkeys(str(item) for item in references):
        record = records_by_image.get(image_id)
        if record is None:
            continue
        feature_map = load_token_feature_map(record.token_path, layer_name)
        if selector_run is not None:
            feature_map = project_feature_map_tokens(
                feature_map,
                selector_run.model,
                output_dim=int(selector_run.summary.output_dim),
                device=selector_device,
                batch_size=int(selector_batch_size),
                active_group_mask=selector_run.active_group_mask,
            )
        feature_map = compute_patch_context_feature_map(feature_map, context=patch_context)
        feature_maps[image_id] = feature_map
    return feature_maps


def _mean(values: list[float]) -> float:
    return 0.0 if not values else float(np.mean(values))


def _finite_median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return None if not finite else float(np.median(finite))


def _binary_auc(labels: list[float], scores: list[float]) -> float | None:
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    if y.size == 0 or y.shape[0] != s.shape[0]:
        return None
    pos = y > 0.5
    neg = ~pos
    if not np.any(pos) or not np.any(neg):
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    pos_ranks = float(np.sum(ranks[pos]))
    n_pos = float(np.sum(pos))
    n_neg = float(np.sum(neg))
    return float((pos_ranks - n_pos * (n_pos + 1.0) * 0.5) / max(n_pos * n_neg, 1.0))


def _average_precision(labels: list[float], scores: list[float]) -> float | None:
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    if y.size == 0 or y.shape[0] != s.shape[0] or not np.any(y > 0.5):
        return None
    order = np.argsort(-s)
    y_sorted = y[order] > 0.5
    tp = np.cumsum(y_sorted.astype(np.float64))
    precision = tp / np.arange(1, y_sorted.size + 1, dtype=np.float64)
    return float(np.sum(precision[y_sorted]) / max(float(np.sum(y_sorted)), 1.0))


def main(argv: Optional[Sequence[str]] = None) -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate Stage G VFM reference patch maplet matching")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--candidate_bank", required=True)
    parser.add_argument("--submap_top_n", type=int, default=10)
    parser.add_argument("--max_submap_landmarks", type=int, default=20000)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--selector_checkpoint", default="")
    parser.add_argument("--selector_device", default="cpu")
    parser.add_argument("--selector_batch_size", type=int, default=8192)
    parser.add_argument("--patch_context", default="1x1", choices=("1x1", "3x3", "5x5"))
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_similarity", type=float, default=0.0)
    parser.add_argument("--query_token_step", type=int, default=1)
    parser.add_argument("--match_block_size", type=int, default=512)
    parser.add_argument("--max_matches", type=int, default=0)
    parser.add_argument("--min_landmarks_per_maplet", type=int, default=1)
    parser.add_argument("--max_landmarks_per_maplet", type=int, default=64)
    parser.add_argument("--cell_radius", type=int, default=0)
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--run_pnp", action="store_true")
    parser.add_argument("--support_per_maplet", type=int, default=1)
    parser.add_argument(
        "--support_recovery_mode",
        default="quality_point",
        choices=("quality_point", "feature_consistent_point", "soft_xyz", "learned_point"),
    )
    parser.add_argument("--support_selector_train_queries", type=int, default=0)
    parser.add_argument("--support_selector_steps", type=int, default=300)
    parser.add_argument("--support_selector_lr", type=float, default=0.2)
    parser.add_argument("--support_selector_l2", type=float, default=1e-3)
    parser.add_argument("--support_selector_no_balance", action="store_true")
    parser.add_argument("--maplet_verifier_train_queries", type=int, default=0)
    parser.add_argument("--maplet_verifier_mode", default="none", choices=("none", "rerank", "filter", "rerank_filter"))
    parser.add_argument("--maplet_verifier_keep_fraction", type=float, default=0.5)
    parser.add_argument("--maplet_verifier_score_weight", type=float, default=0.5)
    parser.add_argument("--maplet_verifier_steps", type=int, default=300)
    parser.add_argument("--maplet_verifier_lr", type=float, default=0.2)
    parser.add_argument("--maplet_verifier_l2", type=float, default=1e-3)
    parser.add_argument("--maplet_verifier_no_balance", action="store_true")
    parser.add_argument("--max_pnp_matches", type=int, default=2000)
    parser.add_argument("--pnp_threshold_stride", type=float, default=0.75)
    parser.add_argument("--measurement_sigma_stride", type=float, default=0.2886751345948129)
    parser.add_argument("--uncertainty_aware_pnp_order", action="store_true")
    parser.add_argument("--pnp_min_inliers", type=int, default=6)
    parser.add_argument("--pnp_iterations", type=int, default=4000)
    parser.add_argument("--pnp_method", default="EPNP")
    parser.add_argument("--pnp_refine_method", default="LM")
    parser.add_argument("--start_query_index", type=int, default=0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_matches_jsonl", default="")
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    query_manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    query_manifest.validate(verify_checksums=False)
    reference_manifest = TokenBankManifest.from_json(Path(args.reference_manifest))
    reference_manifest.validate(verify_checksums=False)
    reference_records = {record.image_id: record for record in reference_manifest.records}

    colmap_observations = load_colmap_track_observations_jsonl(Path(args.track_observations))
    observations_by_image = observation_index_from_colmap_observations(colmap_observations)
    xyz_by_track, reprojection_error_by_track = _track_stats_from_observations(colmap_observations)
    bank = load_selected_track_bank_npz(Path(args.landmark_bank))
    landmark_index = LandmarkMapIndex.from_track_bank(bank, xyz_by_track, reprojection_error_by_track)

    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    reference_submaps = _load_reference_submaps(args.candidate_bank, int(args.submap_top_n))
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    selector_run = None
    if args.selector_checkpoint:
        selector_run = load_safe_patch_selector_checkpoint(Path(args.selector_checkpoint), device=args.selector_device)

    config = ReferencePatchMapletConfig(
        top_k=int(args.top_k),
        min_similarity=float(args.min_similarity),
        query_token_step=int(args.query_token_step),
        block_size=int(args.match_block_size),
        max_matches=None if int(args.max_matches) <= 0 else int(args.max_matches),
    )

    records = list(query_manifest.records)
    if int(args.start_query_index) > 0:
        records = records[int(args.start_query_index) :]
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    support_selector_model = None
    support_selector_training_summary = None
    maplet_verifier_model = None
    maplet_verifier_training_summary = None
    if args.support_recovery_mode == "learned_point":
        train_count = int(args.support_selector_train_queries)
        if train_count <= 0:
            raise ValueError("--support_selector_train_queries must be positive for learned_point")
        if len(records) <= train_count:
            raise ValueError("learned_point requires held-out eval queries after support selector training")
        train_records = records[:train_count]
        eval_records = records[train_count:]
        train_features = []
        train_labels = []
        for record in train_records:
            query_id = record.image_id
            gt_pose = gt_by_query.get(query_id)
            if gt_pose is None:
                raise ValueError(f"query pose not found for {query_id}")
            references = reference_submaps.get(query_id, [])
            if not references:
                raise ValueError(f"no reference candidates found for {query_id}")
            submap = filter_landmarks_by_reference_images(landmark_index, references)
            submap = _limit_submap(submap, int(args.max_submap_landmarks))
            query_feature = _load_query_feature(record.token_path, args.layer_name)
            if selector_run is not None:
                query_feature = project_feature_map_tokens(
                    query_feature,
                    selector_run.model,
                    output_dim=int(selector_run.summary.output_dim),
                    device=args.selector_device,
                    batch_size=int(args.selector_batch_size),
                    active_group_mask=selector_run.active_group_mask,
                )
            query_feature = compute_patch_context_feature_map(query_feature, context=args.patch_context)
            _channels, token_height, token_width = query_feature.shape
            reference_feature_maps = _load_reference_feature_maps(
                reference_records,
                references,
                args.layer_name,
                selector_run=selector_run,
                selector_device=args.selector_device,
                selector_batch_size=int(args.selector_batch_size),
                patch_context=args.patch_context,
            )
            maplet_bank = build_reference_patch_maplet_bank(
                submap,
                observations_by_image,
                reference_feature_maps,
                references,
                min_landmarks_per_maplet=int(args.min_landmarks_per_maplet),
                max_landmarks_per_maplet=None
                if int(args.max_landmarks_per_maplet) <= 0
                else int(args.max_landmarks_per_maplet),
                cell_radius=int(args.cell_radius),
            )
            positives = build_patch_positive_sets(
                submap,
                gt_pose.pose_w2c,
                camera,
                token_width,
                token_height,
                patch_scale=float(args.patch_scale),
            )
            matches = match_query_patches_to_reference_patch_maplets(
                query_feature,
                maplet_bank,
                config,
                image_width=int(camera.width),
                image_height=int(camera.height),
            )
            features, labels = collect_support_selector_training_examples(matches, maplet_bank, submap, positives)
            if features.shape[0] > 0:
                train_features.append(features)
                train_labels.append(labels)
        if not train_features:
            raise ValueError("no support selector training examples were collected")
        train_x = np.concatenate(train_features, axis=0)
        train_y = np.concatenate(train_labels, axis=0)
        support_selector_model = fit_support_selector_model(
            train_x,
            train_y,
            steps=int(args.support_selector_steps),
            lr=float(args.support_selector_lr),
            l2=float(args.support_selector_l2),
            class_balance=not bool(args.support_selector_no_balance),
        )
        support_selector_training_summary = {
            "train_query_count": int(len(train_records)),
            "eval_query_count": int(len(eval_records)),
            "example_count": int(train_x.shape[0]),
            "positive_fraction": float(np.mean(train_y)) if train_y.size else 0.0,
            "weights": [float(value) for value in support_selector_model.weights.tolist()],
            "bias": float(support_selector_model.bias),
            "feature_names": list(support_selector_model.feature_names),
            "class_balance": not bool(args.support_selector_no_balance),
        }
        records = eval_records
    if int(args.maplet_verifier_train_queries) > 0:
        train_count = int(args.maplet_verifier_train_queries)
        if len(records) <= train_count:
            raise ValueError("maplet verifier requires held-out eval queries after training")
        train_records = records[:train_count]
        eval_records = records[train_count:]
        train_features = []
        train_labels = []
        for record in train_records:
            query_id = record.image_id
            gt_pose = gt_by_query.get(query_id)
            if gt_pose is None:
                raise ValueError(f"query pose not found for {query_id}")
            references = reference_submaps.get(query_id, [])
            if not references:
                raise ValueError(f"no reference candidates found for {query_id}")
            submap = filter_landmarks_by_reference_images(landmark_index, references)
            submap = _limit_submap(submap, int(args.max_submap_landmarks))
            query_feature = _load_query_feature(record.token_path, args.layer_name)
            if selector_run is not None:
                query_feature = project_feature_map_tokens(
                    query_feature,
                    selector_run.model,
                    output_dim=int(selector_run.summary.output_dim),
                    device=args.selector_device,
                    batch_size=int(args.selector_batch_size),
                    active_group_mask=selector_run.active_group_mask,
                )
            query_feature = compute_patch_context_feature_map(query_feature, context=args.patch_context)
            _channels, token_height, token_width = query_feature.shape
            reference_feature_maps = _load_reference_feature_maps(
                reference_records,
                references,
                args.layer_name,
                selector_run=selector_run,
                selector_device=args.selector_device,
                selector_batch_size=int(args.selector_batch_size),
                patch_context=args.patch_context,
            )
            maplet_bank = build_reference_patch_maplet_bank(
                submap,
                observations_by_image,
                reference_feature_maps,
                references,
                min_landmarks_per_maplet=int(args.min_landmarks_per_maplet),
                max_landmarks_per_maplet=None
                if int(args.max_landmarks_per_maplet) <= 0
                else int(args.max_landmarks_per_maplet),
                cell_radius=int(args.cell_radius),
            )
            positives = build_patch_positive_sets(
                submap,
                gt_pose.pose_w2c,
                camera,
                token_width,
                token_height,
                patch_scale=float(args.patch_scale),
            )
            matches = match_query_patches_to_reference_patch_maplets(
                query_feature,
                maplet_bank,
                config,
                image_width=int(camera.width),
                image_height=int(camera.height),
            )
            features, labels = collect_maplet_verifier_training_examples(matches, maplet_bank, positives)
            if features.shape[0] > 0:
                train_features.append(features)
                train_labels.append(labels)
        if not train_features:
            raise ValueError("no maplet verifier training examples were collected")
        train_x = np.concatenate(train_features, axis=0)
        train_y = np.concatenate(train_labels, axis=0)
        maplet_verifier_model = fit_maplet_verifier_model(
            train_x,
            train_y,
            steps=int(args.maplet_verifier_steps),
            lr=float(args.maplet_verifier_lr),
            l2=float(args.maplet_verifier_l2),
            class_balance=not bool(args.maplet_verifier_no_balance),
        )
        train_scores = score_maplet_verifier_model(maplet_verifier_model, train_x)
        maplet_verifier_training_summary = {
            "train_query_count": int(len(train_records)),
            "eval_query_count": int(len(eval_records)),
            "example_count": int(train_x.shape[0]),
            "positive_fraction": float(np.mean(train_y)) if train_y.size else 0.0,
            "train_auroc": _binary_auc(train_y.tolist(), train_scores.tolist()),
            "train_auprc": _average_precision(train_y.tolist(), train_scores.tolist()),
            "weights": [float(value) for value in maplet_verifier_model.weights.tolist()],
            "bias": float(maplet_verifier_model.bias),
            "feature_names": list(maplet_verifier_model.feature_names),
            "class_balance": not bool(args.maplet_verifier_no_balance),
        }
        records = eval_records
    output_rows = []
    match_rows = []
    metrics_by_key: dict[str, list[float]] = {}
    maplet_counts = []
    matched_counts = []
    positive_stats_rows = []
    maplet_positive_stats_rows = []
    maplet_verifier_eval_labels: list[float] = []
    maplet_verifier_eval_scores: list[float] = []

    for record in records:
        query_id = record.image_id
        gt_pose = gt_by_query.get(query_id)
        if gt_pose is None:
            raise ValueError(f"query pose not found for {query_id}")
        references = reference_submaps.get(query_id, [])
        if not references:
            raise ValueError(f"no reference candidates found for {query_id}")
        submap = filter_landmarks_by_reference_images(landmark_index, references)
        submap = _limit_submap(submap, int(args.max_submap_landmarks))
        query_feature = _load_query_feature(record.token_path, args.layer_name)
        if selector_run is not None:
            query_feature = project_feature_map_tokens(
                query_feature,
                selector_run.model,
                output_dim=int(selector_run.summary.output_dim),
                device=args.selector_device,
                batch_size=int(args.selector_batch_size),
                active_group_mask=selector_run.active_group_mask,
            )
        query_feature = compute_patch_context_feature_map(query_feature, context=args.patch_context)
        _channels, token_height, token_width = query_feature.shape
        reference_feature_maps = _load_reference_feature_maps(
            reference_records,
            references,
            args.layer_name,
            selector_run=selector_run,
            selector_device=args.selector_device,
            selector_batch_size=int(args.selector_batch_size),
            patch_context=args.patch_context,
        )
        maplet_bank = build_reference_patch_maplet_bank(
            submap,
            observations_by_image,
            reference_feature_maps,
            references,
            min_landmarks_per_maplet=int(args.min_landmarks_per_maplet),
            max_landmarks_per_maplet=None
            if int(args.max_landmarks_per_maplet) <= 0
            else int(args.max_landmarks_per_maplet),
            cell_radius=int(args.cell_radius),
        )
        positives = build_patch_positive_sets(
            submap,
            gt_pose.pose_w2c,
            camera,
            token_width,
            token_height,
            patch_scale=float(args.patch_scale),
        )
        matches = match_query_patches_to_reference_patch_maplets(
            query_feature,
            maplet_bank,
            config,
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        positive_stats = patch_positive_set_stats(positives)
        maplet_positive_stats = reference_patch_maplet_positive_stats(positives, maplet_bank)
        if maplet_verifier_model is not None:
            verifier_features, verifier_labels = collect_maplet_verifier_training_examples(matches, maplet_bank, positives)
            verifier_scores = score_maplet_verifier_model(maplet_verifier_model, verifier_features)
            maplet_verifier_eval_labels.extend(float(value) for value in verifier_labels.tolist())
            maplet_verifier_eval_scores.extend(float(value) for value in verifier_scores.tolist())
            if args.maplet_verifier_mode != "none":
                keep_fraction = (
                    float(args.maplet_verifier_keep_fraction)
                    if args.maplet_verifier_mode in {"filter", "rerank_filter"}
                    else None
                )
                matches = apply_maplet_verifier_to_matches(
                    matches,
                    maplet_bank,
                    maplet_verifier_model,
                    score_weight=float(args.maplet_verifier_score_weight),
                    keep_fraction=keep_fraction,
                )
        metrics = evaluate_reference_patch_maplet_matches(matches, positives, top_k=int(args.top_k))
        pnp = None
        pnp_match_count = 0
        translation_error = None
        rotation_error = None
        if args.run_pnp:
            base_stride = max(float(positives.stride_x_px), float(positives.stride_y_px))
            measurement_sigma_px = float(args.measurement_sigma_stride) * base_stride
            pnp_matches = expand_reference_patch_maplet_matches_to_query_to_3d(
                matches,
                maplet_bank,
                submap,
                support_per_maplet=int(args.support_per_maplet),
                max_matches=None if int(args.max_pnp_matches) <= 0 else int(args.max_pnp_matches),
                support_recovery_mode=args.support_recovery_mode,
                measurement_sigma_px=measurement_sigma_px,
                uncertainty_aware_sort=bool(args.uncertainty_aware_pnp_order),
                support_selector_model=support_selector_model,
            )
            pnp_match_count = int(len(pnp_matches))
            sigma_values = [
                float(match.measurement_sigma_px)
                for match in pnp_matches
                if match.measurement_sigma_px is not None and np.isfinite(float(match.measurement_sigma_px))
            ]
            threshold = float(args.pnp_threshold_stride) * base_stride
            pnp = estimate_pose_pnp_ransac(
                pnp_matches,
                camera,
                reprojection_error_px=threshold,
                confidence=0.999,
                iterations=int(args.pnp_iterations),
                min_inliers=int(args.pnp_min_inliers),
                pnp_method=args.pnp_method,
                refine_method=args.pnp_refine_method,
            )
            pose_error = pnp_pose_error(pnp.pose_w2c, gt_pose.pose_w2c)
            translation_error = float(pose_error.translation_m)
            rotation_error = float(pose_error.rotation_deg)
        positive_stats_rows.append(positive_stats)
        maplet_positive_stats_rows.append(maplet_positive_stats)
        maplet_counts.append(float(len(maplet_bank)))
        matched_counts.append(float(len(matches)))
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_by_key.setdefault(key, []).append(float(value))
        row = {
            "query_id": query_id,
            "reference_count": int(len(references)),
            "submap_landmark_count": int(len(submap)),
            "maplet_count": int(len(maplet_bank)),
            "match_count": int(len(matches)),
            "positive_set_stats": positive_stats,
            "maplet_positive_stats": maplet_positive_stats,
            **metrics,
        }
        if args.run_pnp:
            row.update(
                {
                    "pnp_match_count": int(pnp_match_count),
                    "measurement_sigma_px_mean": _mean(sigma_values),
                    "measurement_sigma_px_median": _finite_median(sigma_values),
                    "pnp_solve": bool(pnp is not None and pnp.success),
                    "pnp_inlier_count": int(0 if pnp is None else pnp.inlier_count),
                    "pnp_inlier_ratio": float(0.0 if pnp is None else pnp.inlier_ratio),
                    "translation_error_m": translation_error,
                    "rotation_error_deg": rotation_error,
                    "success_10cm_5deg": bool(
                        pnp is not None
                        and pnp.success
                        and translation_error is not None
                        and translation_error <= 0.10
                        and rotation_error is not None
                        and rotation_error <= 5.0
                    ),
                    "success_25cm_10deg": bool(
                        pnp is not None
                        and pnp.success
                        and translation_error is not None
                        and translation_error <= 0.25
                        and rotation_error is not None
                        and rotation_error <= 10.0
                    ),
                    "success_50cm_10deg": bool(
                        pnp is not None
                        and pnp.success
                        and translation_error is not None
                        and translation_error <= 0.50
                        and rotation_error is not None
                        and rotation_error <= 10.0
                    ),
                }
            )
        output_rows.append(row)
        if args.output_matches_jsonl:
            for match in matches:
                support = set(int(item) for item in match.support_track_ids)
                positive = positives.by_token.get(int(match.token_index))
                positive_tracks = set() if positive is None else positive.track_ids
                overlap = support.intersection(positive_tracks)
                match_rows.append(
                    {
                        "query_id": query_id,
                        "token_index": int(match.token_index),
                        "reference_image_id": match.reference_image_id,
                        "reference_token_index": int(match.reference_token_index),
                        "unit_id": int(match.unit_id),
                        "rank": int(match.rank),
                        "similarity": float(match.similarity),
                        "support_count": int(len(support)),
                        "positive_count": int(len(positive_tracks)),
                        "overlap_count": int(len(overlap)),
                        "maplet_correct": bool(len(overlap) > 0),
                    }
                )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if args.output_matches_jsonl:
        match_path = Path(args.output_matches_jsonl)
        match_path.parent.mkdir(parents=True, exist_ok=True)
        with match_path.open("w") as handle:
            for row in match_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "stage": "stage_g_reference_patch_maplet_matching",
        "camera_source": camera_source,
        "inputs": {
            "query_manifest": str(args.query_manifest),
            "reference_manifest": str(args.reference_manifest),
            "landmark_bank": str(args.landmark_bank),
            "track_observations": str(args.track_observations),
            "candidate_bank": str(args.candidate_bank),
            "selector_checkpoint": str(args.selector_checkpoint),
        },
        "outputs": {
            "rows": str(output_path),
            "matches": str(args.output_matches_jsonl),
        },
        "config": vars(args),
        "query_count": int(len(output_rows)),
        "mean_maplet_count": _mean(maplet_counts),
        "mean_match_count": _mean(matched_counts),
        "mean_nonempty_patch_fraction": _mean(
            [float(row.get("nonempty_patch_fraction", 0.0)) for row in positive_stats_rows]
        ),
        "mean_maplet_oracle_covered_positive_token_fraction": _mean(
            [float(row.get("covered_positive_token_fraction", 0.0)) for row in maplet_positive_stats_rows]
        ),
        "mean_positive_maplets_per_token": _mean(
            [float(row.get("mean_positive_maplets_per_token", 0.0)) for row in maplet_positive_stats_rows]
        ),
        "mean_best_maplet_overlap_per_token": _mean(
            [float(row.get("mean_best_overlap_per_token", 0.0)) for row in maplet_positive_stats_rows]
        ),
        "support_selector_training": support_selector_training_summary,
        "maplet_verifier_training": maplet_verifier_training_summary,
        "maplet_verifier_eval": None
        if not maplet_verifier_eval_labels
        else {
            "example_count": int(len(maplet_verifier_eval_labels)),
            "positive_fraction": float(np.mean(maplet_verifier_eval_labels)),
            "auroc": _binary_auc(maplet_verifier_eval_labels, maplet_verifier_eval_scores),
            "auprc": _average_precision(maplet_verifier_eval_labels, maplet_verifier_eval_scores),
        },
        "elapsed_sec": float(time.perf_counter() - started),
    }
    for key, values in metrics_by_key.items():
        summary[f"mean_{key}"] = _mean(values)
    if args.run_pnp:
        labeled_rows = [
            row
            for row in output_rows
            if row.get("translation_error_m") is not None and np.isfinite(float(row["translation_error_m"]))
        ]
        summary.update(
            {
                "pnp_solve_rate": _mean([1.0 if row.get("pnp_solve") else 0.0 for row in output_rows]),
                "mean_pnp_match_count": _mean([float(row.get("pnp_match_count", 0.0)) for row in output_rows]),
                "mean_pnp_inlier_count": _mean([float(row.get("pnp_inlier_count", 0.0)) for row in output_rows]),
                "mean_pnp_inlier_ratio": _mean([float(row.get("pnp_inlier_ratio", 0.0)) for row in output_rows]),
                "mean_measurement_sigma_px": _mean(
                    [float(row.get("measurement_sigma_px_mean", 0.0)) for row in output_rows]
                ),
                "median_measurement_sigma_px": _finite_median(
                    [
                        float(row["measurement_sigma_px_median"])
                        for row in output_rows
                        if row.get("measurement_sigma_px_median") is not None
                    ]
                ),
                "success_10cm_5deg": _mean([1.0 if row.get("success_10cm_5deg") else 0.0 for row in output_rows]),
                "success_25cm_10deg": _mean([1.0 if row.get("success_25cm_10deg") else 0.0 for row in output_rows]),
                "success_50cm_10deg": _mean([1.0 if row.get("success_50cm_10deg") else 0.0 for row in output_rows]),
                "median_translation_error_m": None
                if not labeled_rows
                else float(np.median([float(row["translation_error_m"]) for row in labeled_rows])),
                "median_rotation_error_deg": None
                if not labeled_rows
                else float(np.median([float(row["rotation_error_deg"]) for row in labeled_rows])),
            }
        )
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
