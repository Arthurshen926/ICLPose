"""Fuse multiple VFM-2DGS observation banks into a mapping-only anchor map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.vfm_2dgs_mapping import (
    SurfaceElementMap,
    Vfm2DgsAnchorFusionConfig,
    Vfm2DgsObservationBank,
    anchor_map_summary,
    build_anchor_covisibility_graph,
    build_anchor_descriptor_index,
    fuse_token_surface_observations,
    merge_vfm_2dgs_observation_banks,
    spatial_nms_anchor_map,
    vfm_2dgs_anchor_map_to_semidense,
)
from feature_extract.vfm.semidense_anchor_map import semidense_anchor_map_stats


def _stats(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(np.max(arr)),
    }


def _strength_counts(values: Sequence[str]) -> dict[str, int]:
    counts = {"strong": 0, "weak": 0}
    for value in values:
        key = str(value)
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _parse_source(spec: str) -> tuple[str, Path]:
    if "=" not in str(spec):
        path = Path(spec)
        return path.stem, path
    name, path = str(spec).split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("observation bank source name cannot be empty")
    return name, Path(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse VFM-2DGS observation banks into one anchor map")
    parser.add_argument("--surface_npz", required=True)
    parser.add_argument(
        "--observation_bank",
        action="append",
        required=True,
        help="Observation bank path, optionally as name=/path/to/bank.npz. Repeat for multiple sources.",
    )
    parser.add_argument("--no_deduplicate", action="store_true")
    parser.add_argument("--fusion_mode", default="graph", choices=("greedy", "graph", "surface_first"))
    parser.add_argument("--source_merge_policy", default="allow", choices=("allow", "same_source", "feature_agree"))
    parser.add_argument("--cross_source_min_feature_cosine", type=float, default=0.8)
    parser.add_argument("--min_surface_iou", type=float, default=0.2)
    parser.add_argument("--min_dilated_surface_iou", type=float, default=0.15)
    parser.add_argument("--support_iou_dilation_hops", type=int, default=1)
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
    parser.add_argument("--descriptor_index_npz", default="")
    parser.add_argument("--descriptor_faiss_index", default="")
    parser.add_argument("--descriptor_index_no_prototypes", action="store_true")
    parser.add_argument("--semidense_output_npz", default="")
    parser.add_argument("--semidense_summary_json", default="")
    parser.add_argument(
        "--semidense_descriptor_mode",
        choices=("mean", "prototypes", "view_bins"),
        default="view_bins",
    )
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--merged_observation_bank_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    surface_elements = SurfaceElementMap.load_npz(Path(args.surface_npz))
    parsed_sources = [_parse_source(item) for item in args.observation_bank]
    source_names = [name for name, _path in parsed_sources]
    source_paths = [path for _name, path in parsed_sources]
    banks = [Vfm2DgsObservationBank.load_npz(path) for path in source_paths]
    merged_bank = merge_vfm_2dgs_observation_banks(
        banks,
        source_names=source_names,
        deduplicate=not bool(args.no_deduplicate),
    )
    merged_bank.save_npz(Path(args.merged_observation_bank_npz))

    fusion_config = Vfm2DgsAnchorFusionConfig(
        fusion_mode=str(args.fusion_mode),
        source_merge_policy=str(args.source_merge_policy),
        cross_source_min_feature_cosine=float(args.cross_source_min_feature_cosine),
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
    anchor_map = fuse_token_surface_observations(
        surface_elements,
        merged_bank.to_observations(),
        fusion_config,
        metadata={
            "stage": "vfm_2dgs_multi_source_anchor_mapping",
            "surface_metadata": dict(surface_elements.metadata or {}),
            "observation_merge_metadata": dict(merged_bank.metadata or {}),
        },
    )
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
    semidense_summary = None
    if args.semidense_output_npz:
        semidense = vfm_2dgs_anchor_map_to_semidense(
            anchor_map,
            descriptor_mode=str(args.semidense_descriptor_mode),
        )
        semidense_output_path = Path(args.semidense_output_npz)
        semidense_output_path.parent.mkdir(parents=True, exist_ok=True)
        semidense.save_npz(semidense_output_path)
        semidense_summary = {
            "stage": "vfm_2dgs_to_semidense_export",
            "descriptor_mode": str(args.semidense_descriptor_mode),
            "anchor_count": int(len(semidense)),
            "source_anchor_count": int(len(anchor_map)),
            "feature_dim": int(semidense.feature_dim),
            "inputs": {"anchor_map": str(args.output_npz)},
            "outputs": {"semidense_anchor_npz": str(args.semidense_output_npz)},
            "stats": semidense_anchor_map_stats(
                semidense,
                sparse_landmark_count=max(int(len(anchor_map)), 1),
                source_gaussian_count=max(int(len(anchor_map)), 1),
            ),
        }
        if args.semidense_summary_json:
            semidense_summary["outputs"]["summary"] = str(args.semidense_summary_json)
            semidense_summary_path = Path(args.semidense_summary_json)
            semidense_summary_path.parent.mkdir(parents=True, exist_ok=True)
            semidense_summary_path.write_text(json.dumps(semidense_summary, indent=2, sort_keys=True) + "\n")

    source_selected_counts = {
        str(name): int(sum(1 for item in merged_bank.source_ids if item == name))
        for name in source_names
    }
    summary = {
        "stage": "vfm_2dgs_multi_source_anchor_mapping",
        "surface_element_count": int(len(surface_elements)),
        "source_observation_banks": [
            {"name": str(name), "path": str(path), "observation_count": int(len(bank))}
            for name, path, bank in zip(source_names, source_paths, banks)
        ],
        "merged_observation_bank": {
            "path": str(args.merged_observation_bank_npz),
            "observation_count": int(len(merged_bank)),
            "source_selected_observation_counts": source_selected_counts,
            "duplicate_observation_count": int(dict(merged_bank.metadata or {}).get("duplicate_observation_count", 0)),
            "support_count_stats": _stats(np.diff(merged_bank.support_offsets).astype(np.float32, copy=False)),
            "purity_stats": _stats(merged_bank.purity_scores),
            "quality_stats": _stats(merged_bank.quality_scores),
            "strength_counts": _strength_counts(merged_bank.observation_strengths),
            "descriptor_weight_stats": _stats(merged_bank.descriptor_weights),
        },
        "fusion_config": fusion_config.to_dict(),
        "selection_config": {
            "spatial_nms_radius": float(args.spatial_nms_radius),
            "max_anchors": int(args.max_anchors),
            "pre_selection_anchor_count": pre_selection_anchor_count,
            "post_selection_anchor_count": int(len(anchor_map)),
            "covisibility_min_score": float(args.covisibility_min_score),
            "max_covisibility_neighbors": int(args.max_covisibility_neighbors),
            "covisibility_edge_count": int(anchor_map.covisibility_anchor_ids.shape[0]),
        },
        "anchor_map": anchor_map_summary(anchor_map, len(merged_bank), len(surface_elements)),
        "outputs": {
            "surface_elements": str(args.surface_npz),
            "anchor_map": str(args.output_npz),
            "merged_observation_bank": str(args.merged_observation_bank_npz),
            "summary": str(args.summary_json),
        },
    }
    if args.descriptor_index_npz:
        summary["outputs"]["descriptor_index"] = str(args.descriptor_index_npz)
    if args.descriptor_faiss_index:
        summary["outputs"]["descriptor_faiss_index"] = str(args.descriptor_faiss_index)
    if args.semidense_output_npz:
        summary["outputs"]["semidense_anchor_map"] = str(args.semidense_output_npz)
    if args.semidense_summary_json:
        summary["outputs"]["semidense_summary"] = str(args.semidense_summary_json)
    if descriptor_index is not None:
        summary["descriptor_index"] = {
            "descriptor_count": int(len(descriptor_index)),
            "feature_dim": int(descriptor_index.descriptors.shape[1]) if len(descriptor_index) else 0,
            "include_prototypes": not bool(args.descriptor_index_no_prototypes),
            "faiss_metric": "inner_product",
        }
    if semidense_summary is not None:
        summary["semidense_export"] = {
            "descriptor_mode": str(args.semidense_descriptor_mode),
            "anchor_count": int(semidense_summary["anchor_count"]),
            "source_anchor_count": int(semidense_summary["source_anchor_count"]),
            "feature_dim": int(semidense_summary["feature_dim"]),
        }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
