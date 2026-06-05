"""Build VFM-token-aware 2DGS surface anchor maps.

This builder intentionally stops at mapping. It does not run matching or
localization; the output is a persistent VFM-2DGS anchor map plus diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import (
    _load_camera_by_image,
    _load_feature,
    _parse_default_camera,
    _safe_image_stem,
    _select_records,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView, load_gaussian_vfm_source_from_ply
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_mapping import (
    Vfm2DgsAnchorFusionConfig,
    Vfm2DgsMappingConfig,
    Vfm2DgsObservationBank,
    anchor_map_summary,
    build_anchor_covisibility_graph,
    build_anchor_descriptor_index,
    build_surface_elements_from_2dgs_source,
    compute_renderer_token_surface_contribution_buffer,
    compute_token_surface_contribution_buffer,
    compute_token_surface_observations,
    estimate_virtual_cell_max_scale_for_token_projection,
    fuse_token_surface_observations,
    spatial_nms_anchor_map,
    token_surface_observations_from_contribution_buffer,
)


def _stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "max": float(np.max(finite)),
    }


def _strength_counts(values: Sequence[str]) -> dict[str, int]:
    counts = {"strong": 0, "weak": 0}
    for value in values:
        key = str(value)
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VFM-aware 2DGS surface anchor map")
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--view_selection", default="uniform", choices=("prefix", "uniform"))
    parser.add_argument("--max_gaussians", type=int, default=0)
    parser.add_argument(
        "--canonical_vfm_2dgs",
        action="store_true",
        help="Use the mapping protocol intended for VFM-2DGS diagnostics: renderer contributions, surface components, and full debug layers.",
    )
    parser.add_argument(
        "--canonical_token_supply",
        default="grid_top",
        choices=("grid_top", "broad_grid", "surface", "all"),
        help="Token proposal density used by --canonical_vfm_2dgs.",
    )
    parser.add_argument("--surface_min_opacity", type=float, default=0.05)
    parser.add_argument("--surface_max_scale", type=float, default=None)
    parser.add_argument("--surface_adjacency_radius", type=float, default=0.05)
    parser.add_argument("--surface_normal_cosine_threshold", type=float, default=0.8)
    parser.add_argument("--virtual_cell_max_scale", type=float, default=0.0)
    parser.add_argument("--auto_virtual_cell_target_px", type=float, default=0.0)
    parser.add_argument("--auto_virtual_cell_depth_quantile", type=float, default=0.5)
    parser.add_argument("--auto_virtual_cell_max_samples", type=int, default=20000)
    parser.add_argument("--auto_virtual_cell_max_views", type=int, default=16)
    parser.add_argument("--virtual_cell_grid_cap", type=int, default=8)
    parser.add_argument("--token_top_fraction", type=float, default=0.02)
    parser.add_argument("--min_token_saliency", type=float, default=0.0)
    parser.add_argument("--saliency_mode", default="norm", choices=("norm", "local_contrast"))
    parser.add_argument("--token_selection_mode", default="top", choices=("top", "grid_top", "all"))
    parser.add_argument("--token_grid_rows", type=int, default=4)
    parser.add_argument("--token_grid_cols", type=int, default=4)
    parser.add_argument("--max_tokens_per_cell", type=int, default=0)
    parser.add_argument("--min_surface_token_contribution", type=float, default=0.0)
    parser.add_argument("--max_surface_tokens", type=int, default=0)
    parser.add_argument("--surface_token_saliency_power", type=float, default=0.0)
    parser.add_argument("--footprint_radius_px", type=float, default=1.5)
    parser.add_argument("--footprint_sample_grid", type=int, default=1)
    parser.add_argument("--footprint_sample_extent_px", type=float, default=0.0)
    parser.add_argument("--max_projected_disk_radius_px", type=float, default=2.0)
    parser.add_argument("--depth_epsilon", type=float, default=0.05)
    parser.add_argument("--opacity_threshold", type=float, default=0.05)
    parser.add_argument("--view_angle_power", type=float, default=0.0)
    parser.add_argument("--bidirectional_lambda", type=float, default=0.5)
    parser.add_argument("--element_coverage_power", type=float, default=1.0)
    parser.add_argument("--depth_purity_sigma", type=float, default=0.25)
    parser.add_argument("--normal_purity_power", type=float, default=1.0)
    parser.add_argument("--min_purity", type=float, default=0.0)
    parser.add_argument("--max_effective_support_elements", type=float, default=0.0)
    parser.add_argument("--support_mode", default="projection_depth", choices=("projection_depth", "surface_component"))
    parser.add_argument(
        "--token_anchor_competition",
        default="none",
        choices=("none", "winner_component", "winner_element", "winner_neighborhood"),
    )
    parser.add_argument("--token_anchor_neighborhood_hops", type=int, default=1)
    parser.add_argument("--token_anchor_max_support_elements", type=int, default=0)
    parser.add_argument("--min_component_concentration", type=float, default=0.5)
    parser.add_argument("--min_full_component_concentration", type=float, default=0.0)
    parser.add_argument("--max_full_component_count", type=int, default=0)
    parser.add_argument("--max_full_support_elements", type=int, default=0)
    parser.add_argument("--weak_observation_mode", default="drop", choices=("drop", "keep"))
    parser.add_argument("--weak_min_full_component_concentration", type=float, default=0.0)
    parser.add_argument("--weak_max_full_support_elements", type=int, default=0)
    parser.add_argument("--weak_quality_scale", type=float, default=0.25)
    parser.add_argument("--weak_descriptor_weight", type=float, default=0.0)
    parser.add_argument("--min_responsibility", type=float, default=0.01)
    parser.add_argument("--max_elements_per_token", type=int, default=64)
    parser.add_argument("--min_surface_iou", type=float, default=0.2)
    parser.add_argument("--fusion_mode", default="greedy", choices=("greedy", "graph", "surface_first"))
    parser.add_argument("--min_dilated_surface_iou", type=float, default=0.0)
    parser.add_argument("--support_iou_dilation_hops", type=int, default=0)
    parser.add_argument("--min_parent_surface_iou", type=float, default=0.0)
    parser.add_argument("--min_normal_cosine", type=float, default=0.5)
    parser.add_argument("--max_center_distance", type=float, default=0.5)
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--min_descriptor_observations", type=int, default=0)
    parser.add_argument("--support_core_min_observations", type=int, default=0)
    parser.add_argument("--support_core_min_fraction", type=float, default=0.0)
    parser.add_argument("--max_feature_prototypes", type=int, default=4)
    parser.add_argument("--prototype_min_cosine", type=float, default=0.8)
    parser.add_argument("--view_bin_count", type=int, default=4)
    parser.add_argument("--view_bin_feature_mode", default="mean", choices=("mean", "medoid", "consensus_weighted_mean"))
    parser.add_argument("--feature_fusion_mode", default="mean", choices=("mean", "consensus_weighted_mean"))
    parser.add_argument("--feature_consensus_weight_power", type=float, default=1.0)
    parser.add_argument("--min_feature_consensus_cosine", type=float, default=-1.0)
    parser.add_argument("--robust_feature_trim_fraction", type=float, default=0.0)
    parser.add_argument("--surface_first_max_seeds_per_observation", type=int, default=1)
    parser.add_argument("--surface_first_min_seed_weight", type=float, default=0.0)
    parser.add_argument("--spatial_nms_radius", type=float, default=0.0)
    parser.add_argument("--max_anchors", type=int, default=0)
    parser.add_argument("--covisibility_min_score", type=float, default=0.0)
    parser.add_argument("--max_covisibility_neighbors", type=int, default=16)
    parser.add_argument("--no_l2_normalize_features", action="store_true")
    parser.add_argument("--default_camera", default="2,1024,576,883,512,288,0")
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--surface_npz", default="")
    parser.add_argument("--descriptor_index_npz", default="")
    parser.add_argument("--descriptor_faiss_index", default="")
    parser.add_argument("--descriptor_index_no_prototypes", action="store_true")
    parser.add_argument("--contribution_dir", default="")
    parser.add_argument("--contribution_renderer", default="projection_depth_soft", choices=("projection_depth_soft", "gsplat_2dgs"))
    parser.add_argument("--contribution_device", default="cuda")
    parser.add_argument("--observation_bank_npz", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_npz)
    output_dir = output_path.parent
    if bool(args.canonical_vfm_2dgs):
        args.support_mode = "surface_component"
        args.contribution_renderer = "gsplat_2dgs"
        if args.canonical_token_supply == "surface":
            args.token_selection_mode = "surface"
            if int(args.max_surface_tokens) <= 0:
                args.max_surface_tokens = 512
        elif args.canonical_token_supply == "all":
            args.token_selection_mode = "all"
            args.token_top_fraction = 1.0
            args.max_tokens_per_cell = 0
        elif args.canonical_token_supply == "broad_grid":
            args.token_selection_mode = "grid_top"
            args.token_top_fraction = max(float(args.token_top_fraction), 0.05)
            args.max_tokens_per_cell = max(int(args.max_tokens_per_cell), 4)
        else:
            args.token_selection_mode = "grid_top"
        args.min_purity = max(float(args.min_purity), 0.2)
        args.min_component_concentration = max(float(args.min_component_concentration), 0.6)
        args.footprint_sample_grid = max(int(args.footprint_sample_grid), 3)
        args.token_grid_rows = max(int(args.token_grid_rows), 8)
        args.token_grid_cols = max(int(args.token_grid_cols), 8)
        args.support_iou_dilation_hops = max(int(args.support_iou_dilation_hops), 1)
        args.min_dilated_surface_iou = max(float(args.min_dilated_surface_iou), 0.15)
        args.fusion_mode = "graph"
        if not args.contribution_dir:
            args.contribution_dir = str(output_dir / "contributions")
        if not args.surface_npz:
            args.surface_npz = str(output_dir / "surface_elements.npz")
        if not args.observation_bank_npz:
            args.observation_bank_npz = str(output_dir / "observation_bank.npz")
        if not args.descriptor_index_npz:
            args.descriptor_index_npz = str(output_dir / "descriptor_index.npz")
        if float(args.auto_virtual_cell_target_px) <= 0.0 and float(args.virtual_cell_max_scale) <= 0.0:
            args.auto_virtual_cell_target_px = 1.0
    manifest = TokenBankManifest.from_json(Path(args.reference_manifest))
    manifest.validate(verify_checksums=False)
    pose_by_image = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.reference_pose_file))}
    camera_by_image = _load_camera_by_image(args.camera_model_dir)
    fallback_camera = _parse_default_camera(args.default_camera)
    records = [record for record in manifest.records if record.image_id in pose_by_image]
    records = _select_records(records, int(args.max_views), args.view_selection)
    if not records:
        raise ValueError("no reference views with both token features and poses")

    source = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply), max_gaussians=int(args.max_gaussians))
    auto_virtual_cell_max_scale = 0.0
    if float(args.auto_virtual_cell_target_px) > 0.0:
        estimator_records = records[: max(1, min(len(records), int(args.auto_virtual_cell_max_views)))]
        estimator_views = [
            GaussianVFMFeatureView(
                image_id=record.image_id,
                feature_map=_load_feature(Path(record.token_path), args.layer_name),
                pose_w2c=pose_by_image[record.image_id].pose_w2c,
                camera=camera_by_image.get(record.image_id, fallback_camera),
            )
            for record in estimator_records
        ]
        auto_virtual_cell_max_scale = estimate_virtual_cell_max_scale_for_token_projection(
            source,
            estimator_views,
            target_projected_radius_px=float(args.auto_virtual_cell_target_px),
            depth_quantile=float(args.auto_virtual_cell_depth_quantile),
            max_samples=int(args.auto_virtual_cell_max_samples),
            min_opacity=float(args.surface_min_opacity),
        )
        if auto_virtual_cell_max_scale > 0.0:
            args.virtual_cell_max_scale = float(auto_virtual_cell_max_scale)
    surface_elements = build_surface_elements_from_2dgs_source(
        source,
        min_opacity=float(args.surface_min_opacity),
        max_scale=args.surface_max_scale,
        adjacency_radius=float(args.surface_adjacency_radius),
        normal_cosine_threshold=float(args.surface_normal_cosine_threshold),
        virtual_cell_max_scale=float(args.virtual_cell_max_scale),
        virtual_cell_grid_cap=int(args.virtual_cell_grid_cap),
    )
    if args.surface_npz:
        surface_elements.save_npz(Path(args.surface_npz))
    mapping_config = Vfm2DgsMappingConfig(
        token_top_fraction=float(args.token_top_fraction),
        min_token_saliency=float(args.min_token_saliency),
        saliency_mode=str(args.saliency_mode),
        token_selection_mode=str(args.token_selection_mode),
        token_grid_rows=int(args.token_grid_rows),
        token_grid_cols=int(args.token_grid_cols),
        max_tokens_per_cell=int(args.max_tokens_per_cell),
        min_surface_token_contribution=float(args.min_surface_token_contribution),
        max_surface_tokens=int(args.max_surface_tokens),
        surface_token_saliency_power=float(args.surface_token_saliency_power),
        footprint_radius_px=float(args.footprint_radius_px),
        footprint_sample_grid=int(args.footprint_sample_grid),
        footprint_sample_extent_px=float(args.footprint_sample_extent_px),
        max_projected_disk_radius_px=float(args.max_projected_disk_radius_px),
        depth_epsilon=float(args.depth_epsilon),
        opacity_threshold=float(args.opacity_threshold),
        view_angle_power=float(args.view_angle_power),
        bidirectional_lambda=float(args.bidirectional_lambda),
        element_coverage_power=float(args.element_coverage_power),
        depth_purity_sigma=float(args.depth_purity_sigma),
        normal_purity_power=float(args.normal_purity_power),
        min_purity=float(args.min_purity),
        max_effective_support_elements=float(args.max_effective_support_elements),
        support_mode=str(args.support_mode),
        token_anchor_competition=str(args.token_anchor_competition),
        token_anchor_neighborhood_hops=int(args.token_anchor_neighborhood_hops),
        token_anchor_max_support_elements=int(args.token_anchor_max_support_elements),
        min_component_concentration=float(args.min_component_concentration),
        min_full_component_concentration=float(args.min_full_component_concentration),
        max_full_component_count=int(args.max_full_component_count),
        max_full_support_elements=int(args.max_full_support_elements),
        weak_observation_mode=str(args.weak_observation_mode),
        weak_min_full_component_concentration=float(args.weak_min_full_component_concentration),
        weak_max_full_support_elements=int(args.weak_max_full_support_elements),
        weak_quality_scale=float(args.weak_quality_scale),
        weak_descriptor_weight=float(args.weak_descriptor_weight),
        min_responsibility=float(args.min_responsibility),
        max_elements_per_token=int(args.max_elements_per_token),
        l2_normalize_features=not bool(args.no_l2_normalize_features),
    )
    fusion_config = Vfm2DgsAnchorFusionConfig(
        fusion_mode=str(args.fusion_mode),
        min_surface_iou=float(args.min_surface_iou),
        min_dilated_surface_iou=float(args.min_dilated_surface_iou),
        support_iou_dilation_hops=int(args.support_iou_dilation_hops),
        min_parent_surface_iou=float(args.min_parent_surface_iou),
        min_normal_cosine=float(args.min_normal_cosine),
        max_center_distance=float(args.max_center_distance),
        min_observations=int(args.min_observations),
        min_descriptor_observations=int(args.min_descriptor_observations),
        support_core_min_observations=int(args.support_core_min_observations),
        support_core_min_fraction=float(args.support_core_min_fraction),
        l2_normalize_features=not bool(args.no_l2_normalize_features),
        max_feature_prototypes=int(args.max_feature_prototypes),
        prototype_min_cosine=float(args.prototype_min_cosine),
        view_bin_count=int(args.view_bin_count),
        view_bin_feature_mode=str(args.view_bin_feature_mode),
        feature_fusion_mode=str(args.feature_fusion_mode),
        feature_consensus_weight_power=float(args.feature_consensus_weight_power),
        min_feature_consensus_cosine=float(args.min_feature_consensus_cosine),
        robust_feature_trim_fraction=float(args.robust_feature_trim_fraction),
        surface_first_max_seeds_per_observation=int(args.surface_first_max_seeds_per_observation),
        surface_first_min_seed_weight=float(args.surface_first_min_seed_weight),
    )

    observations = []
    per_view = []
    contribution_dir = Path(args.contribution_dir) if args.contribution_dir else None
    aggregate_rejection_stats: dict[str, int] = {}
    aggregate_strength_counts = {"strong": 0, "weak": 0}
    for record in records:
        view = GaussianVFMFeatureView(
            image_id=record.image_id,
            feature_map=_load_feature(Path(record.token_path), args.layer_name),
            pose_w2c=pose_by_image[record.image_id].pose_w2c,
            camera=camera_by_image.get(record.image_id, fallback_camera),
        )
        contribution_summary = None
        if contribution_dir is not None:
            if args.contribution_renderer == "gsplat_2dgs":
                contribution_buffer = compute_renderer_token_surface_contribution_buffer(
                    surface_elements,
                    view,
                    mapping_config,
                    device=str(args.contribution_device),
                    renderer="gsplat_2dgs",
                )
            else:
                contribution_buffer = compute_token_surface_contribution_buffer(surface_elements, view, mapping_config)
            contribution_path = contribution_dir / f"{_safe_image_stem(record.image_id)}.npz"
            contribution_buffer.save_npz(contribution_path)
            view_observations = token_surface_observations_from_contribution_buffer(
                surface_elements,
                view,
                contribution_buffer,
                mapping_config,
            )
            rejection_stats = dict((contribution_buffer.metadata or {}).get("rejection_stats", {}))
            for key, value in rejection_stats.items():
                aggregate_rejection_stats[str(key)] = int(aggregate_rejection_stats.get(str(key), 0)) + int(value)
            contribution_summary = {
                "path": str(contribution_path),
                "renderer": str(contribution_buffer.renderer),
                "token_count": int(len(contribution_buffer)),
                "support_count_stats": _stats(
                    np.diff(contribution_buffer.support_offsets).astype(np.float32, copy=False)
                ),
                "top_alpha_stats": _stats(contribution_buffer.top_alpha),
                "alpha_entropy_stats": _stats(contribution_buffer.alpha_entropy),
                "rejection_stats": rejection_stats,
                "strength_counts": _strength_counts(contribution_buffer.observation_strengths),
                "descriptor_weight_stats": _stats(contribution_buffer.descriptor_weights),
            }
        else:
            view_observations = compute_token_surface_observations(surface_elements, view, mapping_config)
        observations.extend(view_observations)
        view_strength_counts = _strength_counts([obs.observation_strength for obs in view_observations])
        for key, value in view_strength_counts.items():
            aggregate_strength_counts[key] = int(aggregate_strength_counts.get(key, 0)) + int(value)
        row = {
            "image_id": record.image_id,
            "observation_count": int(len(view_observations)),
            "support_count_stats": _stats(
                np.asarray([obs.element_ids.size for obs in view_observations], dtype=np.float32)
            ),
            "purity_stats": _stats(np.asarray([obs.purity_score for obs in view_observations], dtype=np.float32)),
            "quality_stats": _stats(np.asarray([obs.quality_score for obs in view_observations], dtype=np.float32)),
            "strength_counts": view_strength_counts,
            "descriptor_weight_stats": _stats(
                np.asarray([obs.descriptor_weight for obs in view_observations], dtype=np.float32)
            ),
        }
        if contribution_summary is not None:
            row["contribution_buffer"] = contribution_summary
        per_view.append(row)

    anchor_map = fuse_token_surface_observations(
        surface_elements,
        observations,
        fusion_config,
        metadata={
            "stage": "vfm_2dgs_anchor_mapping",
            "mapping_config": mapping_config.to_dict(),
            "surface_metadata": dict(surface_elements.metadata or {}),
        },
    )
    if args.observation_bank_npz:
        observation_bank = Vfm2DgsObservationBank.from_observations(
            observations,
            metadata={
                "stage": "vfm_2dgs_token_observation_layer",
                "mapping_config": mapping_config.to_dict(),
                "view_count": int(len(records)),
            },
        )
        observation_bank.save_npz(Path(args.observation_bank_npz))
    pre_selection_anchor_count = int(len(anchor_map))
    if float(args.spatial_nms_radius) > 0.0 or int(args.max_anchors) > 0:
        anchor_map = spatial_nms_anchor_map(
            anchor_map,
            radius=float(args.spatial_nms_radius),
            max_anchors=int(args.max_anchors),
        )
    if float(args.covisibility_min_score) > 0.0 or int(args.max_covisibility_neighbors) > 0:
        anchor_map = build_anchor_covisibility_graph(
            anchor_map,
            min_score=float(args.covisibility_min_score),
            max_neighbors=int(args.max_covisibility_neighbors),
        )
    anchor_map.save_npz(Path(args.output_npz))
    descriptor_index = None
    if args.descriptor_index_npz or args.descriptor_faiss_index:
        descriptor_index = build_anchor_descriptor_index(
            anchor_map,
            include_prototypes=not bool(args.descriptor_index_no_prototypes),
        )
    if args.descriptor_index_npz:
        descriptor_index.save_npz(Path(args.descriptor_index_npz))
    if args.descriptor_faiss_index:
        descriptor_index.save_faiss(Path(args.descriptor_faiss_index))

    summary = {
        "stage": "vfm_2dgs_anchor_mapping",
        "source_gaussian_count": int(source.xyz.shape[0]),
        "surface_element_count": int(len(surface_elements)),
        "view_count": int(len(records)),
        "mapping_config": mapping_config.to_dict(),
        "fusion_config": fusion_config.to_dict(),
        "selection_config": {
            "spatial_nms_radius": float(args.spatial_nms_radius),
            "max_anchors": int(args.max_anchors),
            "pre_selection_anchor_count": int(pre_selection_anchor_count),
            "post_selection_anchor_count": int(len(anchor_map)),
            "covisibility_min_score": float(args.covisibility_min_score),
            "max_covisibility_neighbors": int(args.max_covisibility_neighbors),
            "covisibility_edge_count": int(anchor_map.covisibility_anchor_ids.shape[0]),
        },
        "canonical_vfm_2dgs": bool(args.canonical_vfm_2dgs),
        "auto_virtual_cell": {
            "target_projected_radius_px": float(args.auto_virtual_cell_target_px),
            "depth_quantile": float(args.auto_virtual_cell_depth_quantile),
            "max_samples": int(args.auto_virtual_cell_max_samples),
            "max_views": int(args.auto_virtual_cell_max_views),
            "estimated_virtual_cell_max_scale": float(auto_virtual_cell_max_scale),
            "effective_virtual_cell_max_scale": float(args.virtual_cell_max_scale),
        },
        "anchor_map": anchor_map_summary(anchor_map, len(observations), len(surface_elements)),
        "observation_support_stats": _stats(np.asarray([obs.element_ids.size for obs in observations], dtype=np.float32)),
        "observation_purity_stats": _stats(np.asarray([obs.purity_score for obs in observations], dtype=np.float32)),
        "observation_quality_stats": _stats(np.asarray([obs.quality_score for obs in observations], dtype=np.float32)),
        "observation_strength_counts": aggregate_strength_counts,
        "observation_descriptor_weight_stats": _stats(
            np.asarray([obs.descriptor_weight for obs in observations], dtype=np.float32)
        ),
        "observation_rejection_stats": aggregate_rejection_stats,
        "per_view": per_view,
        "inputs": {
            "gaussian_ply": str(args.gaussian_ply),
            "reference_manifest": str(args.reference_manifest),
            "reference_pose_file": str(args.reference_pose_file),
            "camera_model_dir": str(args.camera_model_dir),
            "layer_name": str(args.layer_name),
            "view_selection": str(args.view_selection),
        },
        "outputs": {"anchor_map": str(args.output_npz), "summary": str(args.summary_json)},
    }
    if args.surface_npz:
        summary["outputs"]["surface_elements"] = str(args.surface_npz)
    if args.contribution_dir:
        summary["outputs"]["contribution_dir"] = str(args.contribution_dir)
        summary["contribution_renderer"] = {
            "renderer": str(args.contribution_renderer),
            "device": str(args.contribution_device),
        }
    if args.observation_bank_npz:
        summary["outputs"]["observation_bank"] = str(args.observation_bank_npz)
        summary["observation_bank"] = {
            "observation_count": int(len(observations)),
            "feature_dim": int(observations[0].feature.shape[0]) if observations else 0,
        }
    if args.descriptor_index_npz:
        summary["outputs"]["descriptor_index"] = str(args.descriptor_index_npz)
    if args.descriptor_faiss_index:
        summary["outputs"]["descriptor_faiss_index"] = str(args.descriptor_faiss_index)
    if args.descriptor_index_npz or args.descriptor_faiss_index:
        summary["descriptor_index"] = {
            "descriptor_count": int(len(descriptor_index)) if descriptor_index is not None else 0,
            "feature_dim": int(descriptor_index.descriptors.shape[1]) if descriptor_index is not None else 0,
            "include_prototypes": not bool(args.descriptor_index_no_prototypes),
            "faiss_metric": "inner_product",
        }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
