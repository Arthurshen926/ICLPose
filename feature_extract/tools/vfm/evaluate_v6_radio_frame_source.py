"""Strict RADIO-final feature-source diagnostic for V6 frame alignment.

This evaluator removes the learned metric student from M1.  It uses either a
frozen RADIO-final surface-maplet mapper or an origin-preserving RADIO PCA
projection on both sides, gives the evaluator only the correct chart/region
identity, and then runs the ordinary global frame search.  Ground truth is
consulted only after mode generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.evaluate_v6_maplet_frame_alignment import (
    _aggregate_m1,
    _aggregate_pose_rows,
    _aggregate_retrieval,
    _candidate_retrieved_charts,
    _chart_priors,
    _frame_row,
    _predicted_pose_diagnostic,
    _retrieve,
    _retrieval_metrics,
    _search_config,
    _select_oracle_charts,
    _select_oracle_regions,
    _visible_charts,
)
from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.map_entities import (
    MetricSurfaceChartBank,
    RegionChartIndex,
    RetrievalRegionBank,
    compose_metric_region_atlas,
    merge_metric_atlases,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.maplet_frame_alignment import (
    MapletFrameMatch,
    align_maplet_frame_global,
    ground_truth_chart_frame,
    refine_maplet_frame_matches,
)
from feature_extract.vfm.localization_v6.probability_calibration import (
    V6ProbabilityCalibration,
)


FEATURE_LEVEL = "coarse"
FEATURE_STRIDE = 16


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_regions", required=True)
    parser.add_argument("--region_chart_index", required=True)
    parser.add_argument(
        "--radio_atlas", "--radio_mapper_atlas", required=True
    )
    parser.add_argument("--surface_mapper_checkpoint", default="")
    parser.add_argument("--query_projection_maplets", default="")
    parser.add_argument("--spatial_maplets", default="")
    parser.add_argument("--probability_calibration", default="")
    parser.add_argument("--query_contributor_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--m1_charts_per_query", type=int, default=2)
    parser.add_argument("--m1_regions_per_query", type=int, default=2)
    parser.add_argument("--m3_retrieval_regions", type=int, default=16)
    parser.add_argument("--m3_charts", type=int, default=6)
    parser.add_argument(
        "--support_score_power",
        type=float,
        default=0.0,
        help=(
            "Diagnostic regional-support exponent. Zero tests unbiased "
            "mean correlation across projected scales."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trajectory_ids_from_mapper(
    metadata: Mapping[str, object],
) -> list[str]:
    result = set()
    for key in ("training_images", "validation_images"):
        for image_id in metadata.get(key, []):
            result.add(str(image_id).split("/", 1)[0])
    return sorted(result)


def _validate_contract(
    paths: Mapping[str, Path],
    atlas: MapletFeatureAtlasBank,
    index: RegionChartIndex,
    feature_source_metadata: Mapping[str, object],
    feature_source_kind: str,
    query_trajectories: Sequence[str],
) -> dict[str, object]:
    metadata = dict(atlas.metadata or {})
    transform = str(metadata.get("query_feature_transform", ""))
    expected_transform = {
        "surface_maplet_mapper": "surface_maplet_mapper",
        "raw_radio_pca": "raw_radio_pca",
    }[str(feature_source_kind)]
    if transform != expected_transform:
        raise ValueError("RADIO feature source differs from baked atlas")
    transform_sha256 = str(
        metadata.get("query_feature_transform_sha256", "")
    )
    if feature_source_kind == "surface_maplet_mapper":
        observed_transform_sha256 = _sha256(
            paths["surface_mapper_checkpoint"]
        )
    else:
        observed_transform_sha256 = str(
            feature_source_metadata.get(
                "query_feature_transform_sha256", ""
            )
        )
    if (
        not transform_sha256
        or transform_sha256 != observed_transform_sha256
    ):
        raise ValueError("RADIO feature transform differs from baked atlas")
    if str((index.metadata or {}).get("retrieval_region_sha256", "")) != (
        _sha256(paths["retrieval_regions"])
    ):
        raise ValueError("region-chart index retrieval lineage differs")
    if not np.all(np.isin(index.chart_ids, atlas.maplet_ids)):
        raise ValueError("region-chart index refers to unknown atlas charts")
    if not bool(metadata.get("contributor_geometry_lineage_verified", False)):
        raise ValueError("RADIO mapper atlas geometry lineage is unverified")
    for key in (
        "stores_mapping_rgb",
        "stores_mapping_image_ids",
        "stores_mapping_image_paths",
        "uses_sfm_points",
        "uses_sfm_tracks",
        "uses_alike_descriptors",
        "uses_radio_intermediate",
        "uses_point_correspondence_pnp",
    ):
        if bool(metadata.get(key, False)):
            raise ValueError(f"RADIO mapper atlas violates contract: {key}")
    mapping = {
        str(value)
        for value in metadata.get("mapping_trajectory_ids", [])
    }
    mapper_reference = (
        set(_trajectory_ids_from_mapper(feature_source_metadata))
        if feature_source_kind == "surface_maplet_mapper"
        else set()
    )
    query = {str(value) for value in query_trajectories}
    if query & mapping:
        raise ValueError("strict queries overlap RADIO atlas mapping views")
    if query & mapper_reference:
        raise ValueError("strict queries overlap RADIO mapper references")
    return {
        "atlas_mapping": sorted(mapping),
        "radio_mapper_reference": sorted(mapper_reference),
        "strict_test": sorted(query),
        "strict_test_disjoint_from_atlas_mapping": True,
        "strict_test_disjoint_from_radio_mapper_reference": True,
        "feature_source_kind": str(feature_source_kind),
        "clean_geometry_source_sha256": str(
            metadata.get("clean_geometry_source_sha256", "")
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_json)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError("output exists; pass --force to replace it")
    paths = {
        key: Path(getattr(args, key))
        for key in (
            "retrieval_regions",
            "region_chart_index",
            "radio_atlas",
        )
    }
    source_flags = (
        bool(str(args.surface_mapper_checkpoint)),
        bool(str(args.query_projection_maplets)),
    )
    if sum(source_flags) != 1:
        raise ValueError(
            "provide exactly one RADIO feature transform source"
        )
    if source_flags[0]:
        paths["surface_mapper_checkpoint"] = Path(
            args.surface_mapper_checkpoint
        )
        feature_source_kind = "surface_maplet_mapper"
    else:
        paths["query_projection_maplets"] = Path(
            args.query_projection_maplets
        )
        feature_source_kind = "raw_radio_pca"
    runtime_retrieval_flags = (
        bool(str(args.spatial_maplets)),
        bool(str(args.probability_calibration)),
    )
    if any(runtime_retrieval_flags) and not all(runtime_retrieval_flags):
        raise ValueError(
            "spatial_maplets and probability_calibration are a pair"
        )
    runtime_retrieval = all(runtime_retrieval_flags)
    if runtime_retrieval and feature_source_kind != "surface_maplet_mapper":
        raise ValueError(
            "runtime retrieval requires the surface-maplet mapper source"
        )
    if runtime_retrieval:
        paths["spatial_maplets"] = Path(args.spatial_maplets)
        paths["probability_calibration"] = Path(
            args.probability_calibration
        )
    retrieval_regions = RetrievalRegionBank(
        SurfaceRetrievalMapletBank.load_npz(paths["retrieval_regions"])
    )
    index = RegionChartIndex.load_npz(paths["region_chart_index"])
    atlas = MapletFeatureAtlasBank.load_npz(paths["radio_atlas"])
    chart_bank = MetricSurfaceChartBank(atlas)
    if feature_source_kind == "surface_maplet_mapper":
        mapper, feature_source_metadata = load_surface_maplet_mapper(
            paths["surface_mapper_checkpoint"], device=str(args.device)
        )
        projection_bank = None
    else:
        mapper = None
        projection_bank = SurfaceRetrievalMapletBank.load_npz(
            paths["query_projection_maplets"]
        )
        if projection_bank.query_projection is None:
            raise ValueError("RADIO PCA maplets omit their query projection")
        feature_source_metadata = dict(projection_bank.metadata or {})
    if runtime_retrieval:
        spatial_bank = SurfaceRetrievalMapletBank.load_npz(
            paths["spatial_maplets"]
        )
        calibration = V6ProbabilityCalibration.load_json(
            paths["probability_calibration"]
        )
        calibration_metadata = dict(calibration.metadata)
        if str(
            calibration_metadata.get("identity_bank_sha256", "")
        ) != _sha256(paths["retrieval_regions"]):
            raise ValueError("identity calibration lineage differs")
        if str(
            calibration_metadata.get("spatial_bank_sha256", "")
        ) != _sha256(paths["spatial_maplets"]):
            raise ValueError("spatial calibration lineage differs")
    else:
        spatial_bank = None
        calibration = None
    views = _load_views(
        Path(args.query_contributor_dir),
        atlas,
        Path(args.image_root),
    )
    if int(args.max_queries) > 0:
        indices = np.linspace(
            0,
            len(views) - 1,
            min(int(args.max_queries), len(views)),
            dtype=np.int64,
        )
        views = [views[int(index_value)] for index_value in indices]
    protocol = _validate_contract(
        paths,
        atlas,
        index,
        feature_source_metadata,
        feature_source_kind,
        sorted({str(view.trajectory_id) for view in views}),
    )
    if runtime_retrieval:
        query_ids = {
            str(view.trajectory_id) for view in views
        }
        calibration_ids = {
            str(value)
            for value in calibration.metadata.get(
                "calibration_trajectory_ids", []
            )
        }
        strict_holdout_ids = {
            str(value)
            for value in calibration.metadata.get(
                "strict_holdout_trajectory_ids", []
            )
        }
        if query_ids & calibration_ids:
            raise ValueError(
                "strict queries overlap probability calibration"
            )
        if not query_ids <= strict_holdout_ids:
            raise ValueError(
                "queries are absent from calibration strict holdout"
            )
        protocol["probability_calibration"] = sorted(calibration_ids)
        protocol[
            "strict_test_disjoint_from_probability_calibration"
        ] = True
    chart_rows = []
    refined_chart_rows = []
    composite_rows = []
    refined_rows = []
    query_rows = []
    retrieval_rows = []
    search_config = replace(
        _search_config(FEATURE_LEVEL),
        support_score_power=float(args.support_score_power),
    )
    for query_index, view in enumerate(views):
        mapped = (
            mapper.project(view.radio.numpy()).measurement_context
            if mapper is not None
            else projection_bank.project_query_feature_map(
                view.radio.numpy()
            )
        )
        query_feature = torch.from_numpy(mapped).to(str(args.device))
        visible_ids, visible_counts = _visible_charts(view, atlas)
        if runtime_retrieval:
            retrieval = _retrieve(
                view.radio.numpy(),
                retrieval_regions.feature_bank,
                spatial_bank,
                mapper,
                feature_source_metadata,
                calibration,
                (int(view.camera.width), int(view.camera.height)),
                "topq_nms",
            )
            retrieval_rows.append(
                _retrieval_metrics(
                    retrieval,
                    visible_ids,
                    visible_counts,
                    index,
                )
            )
            retrieved_chart_ids = _candidate_retrieved_charts(
                retrieval,
                index,
                int(args.m3_retrieval_regions),
                int(args.m3_charts),
            )
            retrieved_chart_matches = {}
            for chart_id in retrieved_chart_ids.tolist():
                prior_xy, prior_probability = _chart_priors(
                    int(chart_id), retrieval, index
                )
                retrieved_chart_matches[int(chart_id)] = (
                    align_maplet_frame_global(
                        atlas,
                        int(chart_id),
                        query_feature,
                        feature_level=FEATURE_LEVEL,
                        feature_stride=FEATURE_STRIDE,
                        config=search_config,
                        query_location_priors=(
                            prior_xy if prior_xy.size else None
                        ),
                        query_location_prior_weights=(
                            prior_probability
                            if prior_probability.size
                            else None
                        ),
                        location_prior_sigma_px=96.0,
                        location_prior_strength=0.20,
                    )
                )
            retrieved_chart_pose_errors = _predicted_pose_diagnostic(
                retrieved_chart_matches,
                atlas,
                view,
                maximum_charts=int(args.m3_charts),
            )
        else:
            retrieved_chart_ids = np.zeros((0,), dtype=np.int64)
            retrieved_chart_pose_errors = []
        chart_ids = _select_oracle_charts(
            visible_ids,
            visible_counts,
            atlas,
            int(args.m1_charts_per_query),
        )
        chart_matches: dict[int, tuple[MapletFrameMatch, ...]] = {}
        refined_chart_matches: dict[
            int, tuple[MapletFrameMatch, ...]
        ] = {}
        region_ids = _select_oracle_regions(
            visible_ids,
            visible_counts,
            index,
            int(args.m1_regions_per_query),
        )
        for chart_id in chart_ids.tolist():
            matches = align_maplet_frame_global(
                atlas,
                int(chart_id),
                query_feature,
                feature_level=FEATURE_LEVEL,
                feature_stride=FEATURE_STRIDE,
                config=search_config,
            )
            chart_matches[int(chart_id)] = matches
            refined_matches = refine_maplet_frame_matches(
                atlas,
                matches[:8],
                query_feature,
                iterations=20,
            )
            refined_chart_matches[int(chart_id)] = refined_matches
            target = ground_truth_chart_frame(
                atlas,
                int(chart_id),
                view.pose_w2c,
                view.camera,
                feature_stride=FEATURE_STRIDE,
                feature_level="radio_mapper_gt_affine",
                model="affine",
            )
            if target is not None:
                chart_rows.append(
                    _frame_row(
                        view.image_id,
                        FEATURE_LEVEL,
                        int(chart_id),
                        matches,
                        target,
                        conditioned=False,
                    )
                )
                refined_chart_rows.append(
                    _frame_row(
                        view.image_id,
                        FEATURE_LEVEL,
                        int(chart_id),
                        refined_matches,
                        target,
                        conditioned=False,
                    )
                )
        region_matches: dict[int, tuple[MapletFrameMatch, ...]] = {}
        region_atlases = {}
        for region_id in region_ids.tolist():
            try:
                composite = compose_metric_region_atlas(
                    retrieval_regions,
                    chart_bank,
                    index,
                    int(region_id),
                    resolution=48,
                )
            except ValueError:
                continue
            matches = align_maplet_frame_global(
                composite,
                int(region_id),
                query_feature,
                feature_level=FEATURE_LEVEL,
                feature_stride=FEATURE_STRIDE,
                config=search_config,
            )
            refined = refine_maplet_frame_matches(
                composite,
                matches[:8],
                query_feature,
                iterations=20,
            )
            target = ground_truth_chart_frame(
                composite,
                int(region_id),
                view.pose_w2c,
                view.camera,
                feature_stride=FEATURE_STRIDE,
                feature_level="radio_mapper_composite_gt_affine",
                model="affine",
            )
            if target is not None:
                composite_rows.append(
                    _frame_row(
                        view.image_id,
                        FEATURE_LEVEL,
                        int(region_id),
                        matches,
                        target,
                        conditioned=False,
                    )
                )
                refined_rows.append(
                    _frame_row(
                        view.image_id,
                        FEATURE_LEVEL,
                        int(region_id),
                        refined,
                        target,
                        conditioned=False,
                    )
                )
            region_matches[int(region_id)] = refined
            region_atlases[int(region_id)] = composite
        pose_errors = (
            _predicted_pose_diagnostic(
                region_matches,
                merge_metric_atlases(list(region_atlases.values())),
                view,
                maximum_charts=int(args.m1_regions_per_query),
            )
            if region_atlases
            else []
        )
        chart_pose_errors = _predicted_pose_diagnostic(
            refined_chart_matches,
            atlas,
            view,
            maximum_charts=int(args.m1_charts_per_query),
        )
        query_rows.append(
            {
                "image_id": view.image_id,
                "trajectory_id": view.trajectory_id,
                "visible_chart_count": int(visible_ids.size),
                "m1_chart_ids": chart_ids.tolist(),
                "m1_region_ids": region_ids.tolist(),
                "m3_correct_chart_raw_pose_errors": (
                    _predicted_pose_diagnostic(
                        chart_matches,
                        atlas,
                        view,
                        maximum_charts=int(args.m1_charts_per_query),
                    )
                ),
                "m3_correct_chart_pose_errors": chart_pose_errors,
                "m3_retrieved_chart_ids": retrieved_chart_ids.tolist(),
                "m3_retrieved_chart_pose_errors": (
                    retrieved_chart_pose_errors
                ),
                "m3_correct_region_pose_errors": pose_errors,
            }
        )
        print(
            json.dumps(
                {
                    "progress": f"{query_index + 1}/{len(views)}",
                    "image_id": view.image_id,
                    "correct_region_pose_top1": (
                        pose_errors[0] if pose_errors else None
                    ),
                }
            ),
            flush=True,
        )
    report = {
        "stage": "v6_radio_frame_source_m1_m3_diagnostic",
        "query_count": len(query_rows),
        "protocol": protocol,
        "artifact_sha256": {
            key: _sha256(value) for key, value in paths.items()
        },
        "map_contract": {
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": False,
            "stores_mapping_image_paths": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_stable_point_identities": False,
            "uses_stable_point_correspondences": False,
            "uses_alike_descriptors": False,
            "uses_loftr": False,
            "uses_radio_intermediate": False,
            "uses_point_correspondence_pnp": False,
            "uses_final_point_correspondence_pnp": False,
            "uses_regional_frame_control_pose_solver": True,
            "includes_chart_center_grouped_diagnostic": True,
            "query_feature_source": (
                "RADIO-final through "
                + (
                    "frozen surface-maplet mapper"
                    if feature_source_kind == "surface_maplet_mapper"
                    else "origin-preserving map-only PCA"
                )
            ),
        },
        "m1_correct_chart_global_correlation": _aggregate_m1(chart_rows),
        "m1_correct_chart_local_refinement": _aggregate_m1(
            refined_chart_rows
        ),
        "m1_correct_composite_region_global_correlation": _aggregate_m1(
            composite_rows
        ),
        "m1_correct_composite_region_local_refinement": _aggregate_m1(
            refined_rows
        ),
        "m3_correct_metric_charts_predicted_frame_pose": (
            _aggregate_pose_rows(
                query_rows, "m3_correct_chart_pose_errors"
            )
        ),
        "m3_correct_metric_charts_raw_frame_pose": (
            _aggregate_pose_rows(
                query_rows, "m3_correct_chart_raw_pose_errors"
            )
        ),
        "m3_retrieved_metric_charts_predicted_frame_pose": (
            _aggregate_pose_rows(
                query_rows, "m3_retrieved_chart_pose_errors"
            )
        ),
        "m3_correct_composite_region_predicted_frame_pose": (
            _aggregate_pose_rows(
                query_rows, "m3_correct_region_pose_errors"
            )
        ),
        "diagnostic_scope": (
            "correct identity only; ground truth is evaluated after frame "
            "mode generation and is not a runtime input"
        ),
        "scene_evidence": (
            {"topq_nms": _aggregate_retrieval(retrieval_rows)}
            if retrieval_rows
            else None
        ),
        "configuration": {
            "feature_stride": FEATURE_STRIDE,
            "feature_source_kind": feature_source_kind,
            "m1_charts_per_query": int(args.m1_charts_per_query),
            "m1_regions_per_query": int(args.m1_regions_per_query),
            "composite_region_resolution": 48,
            "local_refinement_iterations": 20,
            "joint_translation_affine_nms": True,
            "maximum_modes_per_spatial_cluster": 2,
            "maximum_appearance_modes": 2,
            "support_score_power": float(args.support_score_power),
            "runtime_retrieval": bool(runtime_retrieval),
            "m3_retrieval_regions": int(args.m3_retrieval_regions),
            "m3_charts": int(args.m3_charts),
            "maximum_pose_hypotheses": 1024,
            "grouped_modes_per_chart": {
                "two_or_three_charts": 5,
                "four_charts": 3,
            },
        },
        "queries": query_rows,
        "m1_rows": chart_rows,
        "m1_composite_rows": composite_rows,
        "m1_composite_refined_rows": refined_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "summary": {
                    key: report[key]
                    for key in (
                        "m1_correct_chart_global_correlation",
                        "m1_correct_chart_local_refinement",
                        "m1_correct_composite_region_global_correlation",
                        "m1_correct_composite_region_local_refinement",
                        "m3_correct_metric_charts_predicted_frame_pose",
                        "m3_correct_metric_charts_raw_frame_pose",
                        "m3_retrieved_metric_charts_predicted_frame_pose",
                        "m3_correct_composite_region_predicted_frame_pose",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
