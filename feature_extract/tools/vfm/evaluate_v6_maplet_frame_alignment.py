"""Strict St Mary's M1/M2/M3 evaluation for V6 maplet-frame alignment.

M1 gives the correct visible metric-chart identity and searches its canonical
atlas over the whole query feature map.  M2 uses GT chart projections only as
an oracle to test the frame-to-pose interface.  M3 evaluates predicted frame
modes for both correct charts and charts reached through runtime RADIO-final
retrieval.  Ground truth never participates in runtime retrieval or predicted
frame search.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.evaluate_v6_retrieval_pose_basin import (
    _region_geometry,
)
from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.map_entities import (
    ChartExpansionPosterior,
    MetricSurfaceChartBank,
    RegionChartIndex,
    RetrievalRegionBank,
    compose_metric_region_atlas,
    expand_region_posterior_to_charts,
    merge_metric_atlases,
    region_chart_conditional_probabilities,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.maplet_frame_alignment import (
    MapletFrameMatch,
    MapletFrameSearchConfig,
    align_maplet_frame_global,
    frame_log_likelihood_ratio,
    frame_matches_to_pose_hypotheses,
    frame_parameter_errors,
    ground_truth_chart_frame,
    marginal_frame_log_likelihood_ratio,
    pose_hypotheses_for_mode_sets,
    refine_maplet_frame_matches,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    MapletRetrievalResult,
    retrieve_candidate_groups,
)
from feature_extract.vfm.localization_v6.metric_encoder import (
    load_v6_metric_encoder,
)
from feature_extract.vfm.localization_v6.probability_calibration import (
    V6ProbabilityCalibration,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    encode_radio_final_regions,
)


LEVEL_STRIDE = {"coarse": 16, "middle": 8, "fine": 4}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_regions", required=True)
    parser.add_argument("--spatial_maplets", required=True)
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--probability_calibration", required=True)
    parser.add_argument("--region_chart_index", required=True)
    parser.add_argument("--atlas_coarse", required=True)
    parser.add_argument("--atlas_middle", required=True)
    parser.add_argument("--atlas_fine", required=True)
    parser.add_argument("--query_metric_encoder", required=True)
    parser.add_argument("--query_contributor_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--m1_charts_per_query", type=int, default=2)
    parser.add_argument("--m1_regions_per_query", type=int, default=2)
    parser.add_argument("--m2_candidate_charts", type=int, default=12)
    parser.add_argument("--m3_retrieval_regions", type=int, default=16)
    parser.add_argument("--m3_charts", type=int, default=24)
    parser.add_argument("--m3_min_charts", type=int, default=12)
    parser.add_argument(
        "--m3_chart_probability_mass", type=float, default=0.95
    )
    parser.add_argument("--m3_level", choices=tuple(LEVEL_STRIDE), default="middle")
    parser.add_argument(
        "--frame_levels",
        nargs="+",
        choices=tuple(LEVEL_STRIDE),
        default=list(LEVEL_STRIDE),
    )
    parser.add_argument(
        "--support_score_power",
        type=float,
        default=0.0,
        help=(
            "Diagnostic regional-support exponent. Zero removes the "
            "cross-scale template-size bias."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _search_config(level: str) -> MapletFrameSearchConfig:
    stride = LEVEL_STRIDE[str(level)]
    base = {
        "coarse": (0.9, 1.5, 2.5, 4.0, 6.5, 10.0, 15.0),
        "middle": (1.8, 3.0, 5.0, 8.0, 13.0, 20.0, 30.0),
        "fine": (3.6, 6.0, 10.0, 16.0, 26.0, 40.0),
    }[str(level)]
    return MapletFrameSearchConfig(
        geometric_mean_sizes=base,
        maximum_modes=16,
        per_transform_peaks=4,
        minimum_template_support_cells={
            "coarse": 1.5,
            "middle": 3.0,
            "fine": 6.0,
        }[str(level)],
        minimum_overlap_fraction=0.55,
        spatial_mode_nms_radius_cells=max(2.0, 32.0 / stride),
        diagnostic_null_score=0.25,
    )


def _validate_contract(
    paths: Mapping[str, Path],
    atlases: Mapping[str, MapletFeatureAtlasBank],
    encoder_metadata: Mapping[str, object],
    index: RegionChartIndex,
    calibration: V6ProbabilityCalibration,
    query_trajectories: Sequence[str],
) -> dict[str, object]:
    coarse = atlases["coarse"]
    for level, atlas in atlases.items():
        if not np.array_equal(atlas.maplet_ids, coarse.maplet_ids):
            raise ValueError(f"{level} atlas chart identities differ")
        if not np.allclose(atlas.centers, coarse.centers, atol=1e-6):
            raise ValueError(f"{level} atlas chart geometry differs")
        for key in (
            "geometry_source_sha256",
            "clean_geometry_source_sha256",
            "clean_source_index_sha256",
            "mapping_trajectory_ids",
            "metric_encoder_sha256",
        ):
            if (atlas.metadata or {}).get(key) != (
                coarse.metadata or {}
            ).get(key):
                raise ValueError(f"{level} atlas lineage differs at {key}")
    expected_map_encoder = str(
        encoder_metadata.get("compatible_map_encoder_sha256", "")
    )
    if (
        not expected_map_encoder
        or expected_map_encoder
        != str((coarse.metadata or {}).get("metric_encoder_sha256", ""))
    ):
        raise ValueError("query student and frozen map encoder differ")
    expected_index_region = str(
        (index.metadata or {}).get("retrieval_region_sha256", "")
    )
    expected_index_chart = str(
        (index.metadata or {}).get("metric_chart_sha256", "")
    )
    if expected_index_region != _sha256(paths["retrieval_regions"]):
        raise ValueError("region-chart index retrieval lineage differs")
    if expected_index_chart != _sha256(paths["atlas_coarse"]):
        raise ValueError("region-chart index chart lineage differs")
    calibration_metadata = dict(calibration.metadata)
    if str(calibration_metadata.get("identity_bank_sha256", "")) != _sha256(
        paths["retrieval_regions"]
    ):
        raise ValueError("identity calibration lineage differs")
    if str(calibration_metadata.get("spatial_bank_sha256", "")) != _sha256(
        paths["spatial_maplets"]
    ):
        raise ValueError("spatial calibration lineage differs")
    mapping = set(
        str(value)
        for value in (coarse.metadata or {}).get(
            "mapping_trajectory_ids", []
        )
    )
    training = set(
        str(value)
        for value in encoder_metadata.get("training_trajectories", [])
    )
    validation = set(
        str(value)
        for value in encoder_metadata.get(
            "validation_trajectories", []
        )
    )
    calibration_ids = set(
        str(value)
        for value in calibration_metadata.get(
            "calibration_trajectory_ids", []
        )
    )
    query = set(str(value) for value in query_trajectories)
    named = {
        "atlas_mapping": mapping,
        "query_student_training": training,
        "query_student_validation": validation,
        "probability_calibration": calibration_ids,
        "strict_test": query,
    }
    allowed_overlap_pairs = {
        frozenset(
            ("query_student_validation", "probability_calibration")
        )
    }
    overlaps = {}
    for first, second in itertools.combinations(named, 2):
        shared = sorted(named[first] & named[second])
        overlaps[f"{first}__{second}"] = shared
        if shared and frozenset((first, second)) not in allowed_overlap_pairs:
            raise ValueError(
                f"strict trajectory protocol overlap {first}/{second}: "
                f"{shared}"
            )
    return {
        "trajectory_sets": {
            key: sorted(value) for key, value in named.items()
        },
        "pairwise_overlaps": overlaps,
        "strict_test_disjoint_from_all_reference_sets": True,
        "allowed_reference_overlap": {
            "query_student_validation__probability_calibration": ["seq11"]
        },
        "query_student_map_encoder_compatible": True,
        "clean_geometry_source_sha256": str(
            (coarse.metadata or {}).get(
                "clean_geometry_source_sha256", ""
            )
        ),
    }


def _retrieve(
    raw_radio: np.ndarray,
    identity_bank: SurfaceRetrievalMapletBank,
    spatial_bank: SurfaceRetrievalMapletBank,
    mapper: object,
    mapper_metadata: Mapping[str, object],
    calibration: V6ProbabilityCalibration,
    image_size_wh: tuple[int, int],
    aggregation: str,
) -> MapletRetrievalResult:
    mapped = mapper.project(raw_radio).measurement_context
    spatial_mapped = spatial_bank.project_query_feature_map(raw_radio)
    _indices, token_xy = select_spatially_balanced_radio_final_regions(
        raw_radio
    )
    config = RadioFinalRegionConfig(
        pool_sizes=tuple(mapper_metadata.get("pool_sizes", (1, 3, 5, 9))),
        pool_weights=tuple(
            mapper_metadata.get(
                "pool_weights", (0.4, 0.3, 0.2, 0.1)
            )
        ),
        global_context_weight=float(
            mapper_metadata.get("global_context_weight", 0.0)
        ),
    )
    descriptors = encode_radio_final_regions(mapped, token_xy, config)
    spatial_descriptors = encode_radio_final_regions(
        spatial_mapped,
        token_xy,
        RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,)),
    )
    region_xy, region_extent = _region_geometry(
        token_xy,
        token_width=int(raw_radio.shape[2]),
        token_height=int(raw_radio.shape[1]),
        image_width=int(image_size_wh[0]),
        image_height=int(image_size_wh[1]),
        config=config,
    )
    return retrieve_candidate_groups(
        descriptors,
        region_xy,
        region_extent,
        identity_bank,
        preliminary_candidates=64,
        maximum_maplets=64,
        maximum_components_per_maplet=16,
        component_nms_distance_m=0.02,
        spatial_query_descriptors=spatial_descriptors,
        spatial_bank=spatial_bank,
        probability_calibration=calibration,
        scene_evidence_aggregation=str(aggregation),
        compute_spatial_modes=False,
    )


def _visible_charts(
    view: object, atlas: MapletFeatureAtlasBank
) -> tuple[np.ndarray, np.ndarray]:
    rows, counts = np.unique(
        np.asarray(view.visible_rows, dtype=np.int64)
        // (atlas.height * atlas.width),
        return_counts=True,
    )
    valid = np.mean(atlas.valid_mask[rows], axis=(1, 2)) >= 0.01
    return atlas.maplet_ids[rows[valid]], counts[valid]


def _retrieval_metrics(
    retrieval: MapletRetrievalResult,
    visible_ids: np.ndarray,
    visible_counts: np.ndarray,
    index: RegionChartIndex,
) -> dict[str, object]:
    total = max(int(np.sum(visible_counts)), 1)
    result: dict[str, object] = {
        "aggregation": retrieval.scene_evidence_aggregation,
    }
    dominant_chart = int(visible_ids[int(np.argmax(visible_counts))])
    dominant_regions = set(
        index.regions_for_charts(
            np.asarray([dominant_chart], dtype=np.int64)
        ).tolist()
    )
    ranked = retrieval.ranked_maplet_ids
    for rank in (1, 5, 16, 64):
        regions = ranked[:rank]
        charts = index.charts_for_regions(regions)
        hit = np.isin(visible_ids, charts)
        result[f"dominant_region_recall_at_{rank}"] = bool(
            any(int(value) in dominant_regions for value in regions)
        )
        result[f"visible_chart_count_at_{rank}"] = int(np.sum(hit))
        result[f"visible_surface_coverage_at_{rank}"] = float(
            np.sum(visible_counts[hit]) / total
        )
    return result


def _select_oracle_charts(
    visible_ids: np.ndarray,
    visible_counts: np.ndarray,
    atlas: MapletFeatureAtlasBank,
    limit: int,
) -> np.ndarray:
    rows = {
        int(value): row
        for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    score = []
    for chart_id, count in zip(visible_ids.tolist(), visible_counts.tolist()):
        row = rows.get(int(chart_id))
        if row is None:
            continue
        valid_fraction = float(np.mean(atlas.valid_mask[row]))
        feature_fraction = float(
            np.mean(np.linalg.norm(atlas.features[row], axis=0) > 0.5)
        )
        score.append(
            (
                float(count)
                * np.sqrt(max(valid_fraction * feature_fraction, 1e-8)),
                int(chart_id),
            )
        )
    score.sort(reverse=True)
    return np.asarray(
        [chart_id for _score, chart_id in score[: int(limit)]],
        dtype=np.int64,
    )


def _select_oracle_regions(
    visible_ids: np.ndarray,
    visible_counts: np.ndarray,
    index: RegionChartIndex,
    limit: int,
) -> np.ndarray:
    """Greedily select regions by marginal visible-chart coverage.

    Retrieval regions and metric charts are different namespaces even when a
    legacy partition happened to assign some of them the same integer.  This
    selector only follows the explicit many-to-many index and therefore does
    not rely on that accidental equality.
    """

    chart_count = {
        int(chart_id): float(count)
        for chart_id, count in zip(
            np.asarray(visible_ids, dtype=np.int64).tolist(),
            np.asarray(visible_counts, dtype=np.float64).tolist(),
        )
    }
    remaining = dict(chart_count)
    associated = {}
    for row, region_id in enumerate(index.region_ids.tolist()):
        charts = index.region_chart_ids[
            index.region_offsets[row] : index.region_offsets[row + 1]
        ]
        visible = tuple(
            int(value)
            for value in charts.tolist()
            if int(value) in chart_count
        )
        if visible:
            associated[int(region_id)] = visible
    selected = []
    for _ in range(max(int(limit), 0)):
        scored = [
            (
                sum(remaining.get(chart_id, 0.0) for chart_id in charts),
                sum(chart_count[chart_id] for chart_id in charts),
                -int(region_id),
                int(region_id),
            )
            for region_id, charts in associated.items()
            if int(region_id) not in selected
        ]
        if not scored:
            break
        marginal, _total, _tie, region_id = max(scored)
        if marginal <= 0.0 and selected:
            break
        selected.append(int(region_id))
        for chart_id in associated[int(region_id)]:
            remaining[chart_id] = 0.0
    return np.asarray(selected, dtype=np.int64)


def _frame_row(
    image_id: str,
    level: str,
    chart_id: int,
    matches: Sequence[MapletFrameMatch],
    target: MapletFrameMatch,
    *,
    conditioned: bool,
) -> dict[str, object]:
    errors = [frame_parameter_errors(match, target) for match in matches]
    return {
        "image_id": image_id,
        "feature_level": str(level),
        "feature_stride": LEVEL_STRIDE[str(level)],
        "chart_id": int(chart_id),
        "conditioned_on_retrieval_region": bool(conditioned),
        "mode_count": len(matches),
        "diagnostic_null_probability": (
            float(matches[0].null_probability) if matches else 1.0
        ),
        "mode_scores": [float(match.score) for match in matches],
        "target": {
            "center_xy": target.query_center_xy.tolist(),
            "scale_xy": target.scale_xy.tolist(),
            "rotation_deg": float(target.in_plane_rotation_deg),
            "canonical_to_query": target.canonical_to_query.tolist(),
            "canonical_homography": (
                target.canonical_homography.tolist()
                if target.canonical_homography is not None
                else None
            ),
        },
        "modes": [
            {
                "center_xy": match.query_center_xy.tolist(),
                "scale_xy": match.scale_xy.tolist(),
                "rotation_deg": float(match.in_plane_rotation_deg),
                "canonical_to_query": (
                    match.canonical_to_query.tolist()
                ),
                "canonical_homography": (
                    match.canonical_homography.tolist()
                    if match.canonical_homography is not None
                    else None
                ),
                "support_fraction": float(match.support_fraction),
                "support_canonical_hull": (
                    match.support_canonical_hull.tolist()
                    if match.support_canonical_hull is not None
                    else None
                ),
                "control_covariance_px": (
                    match.control_covariance_px.tolist()
                    if match.control_covariance_px is not None
                    else None
                ),
                "mode_probability": float(match.probability),
                "identity_probability": float(
                    match.identity_probability
                ),
            }
            for match in matches
        ],
        "mode_errors": errors,
    }


def _chart_priors(
    chart_id: int,
    retrieval: MapletRetrievalResult,
    index: RegionChartIndex,
    regions: RetrievalRegionBank | None = None,
    charts: MetricSurfaceChartBank | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    associated = set(
        index.regions_for_charts(np.asarray([chart_id])).tolist()
    )
    xy = []
    probability = []
    conditional_by_region: dict[int, float] = {}
    if regions is not None and charts is not None:
        for region_id in associated:
            chart_ids, conditional = (
                region_chart_conditional_probabilities(
                    int(region_id), regions, charts, index
                )
            )
            rows = np.flatnonzero(chart_ids == int(chart_id))
            conditional_by_region[int(region_id)] = (
                float(conditional[int(rows[0])]) if rows.size else 0.0
            )
    for group in retrieval.groups:
        for region_id, value in zip(
            group.maplet_ids.tolist(), group.probabilities.tolist()
        ):
            if int(region_id) in associated and float(value) > 0.0:
                xy.append(np.asarray(group.query_region_xy, dtype=np.float32))
                probability.append(
                    float(value)
                    * conditional_by_region.get(int(region_id), 1.0)
                )
    if not xy:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(
            (0,), dtype=np.float32
        )
    return np.stack(xy), np.asarray(probability, dtype=np.float32)


def _region_priors(
    region_id: int,
    retrieval: MapletRetrievalResult,
    *,
    maximum_priors: int = 5,
    overlap_iou: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    observations = []
    for group in retrieval.groups:
        rows = np.flatnonzero(
            np.asarray(group.maplet_ids, dtype=np.int64) == int(region_id)
        )
        if rows.size:
            observations.append(
                (
                    float(group.probabilities[int(rows[0])]),
                    np.asarray(group.query_region_xy, dtype=np.float32),
                    np.maximum(
                        np.asarray(
                            group.query_region_extent, dtype=np.float32
                        ),
                        1e-3,
                    ),
                )
            )
    observations.sort(key=lambda value: -value[0])
    retained = []
    for value in observations:
        lower = value[1] - value[2]
        upper = value[1] + value[2]
        duplicate = False
        for other in retained:
            other_lower = other[1] - other[2]
            other_upper = other[1] + other[2]
            intersection = np.maximum(
                np.minimum(upper, other_upper)
                - np.maximum(lower, other_lower),
                0.0,
            )
            intersection_area = float(np.prod(intersection))
            union = float(
                np.prod(2.0 * value[2])
                + np.prod(2.0 * other[2])
                - intersection_area
            )
            if intersection_area / max(union, 1e-8) > float(overlap_iou):
                duplicate = True
                break
        if not duplicate:
            retained.append(value)
        if len(retained) >= int(maximum_priors):
            break
    if not retained:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(
            (0,), dtype=np.float32
        )
    return (
        np.stack([value[1] for value in retained]),
        np.asarray([value[0] for value in retained], dtype=np.float32),
    )


def _pose_errors(
    hypotheses: Sequence[object], pose_w2c: np.ndarray
) -> list[FramePoseHypothesis]:
    result = []
    for hypothesis in hypotheses:
        error = pnp_pose_error(hypothesis.pose_w2c, pose_w2c)
        result.append(
            {
                "translation_m": float(error.translation_m),
                "rotation_deg": float(error.rotation_deg),
                "score": float(hypothesis.score),
                "reprojection_error_px": float(
                    hypothesis.reprojection_error_px
                ),
                "source_chart_ids": list(hypothesis.source_chart_ids),
                "control_model": str(hypothesis.control_model),
                "seed_model": str(hypothesis.seed_model),
                "factor_cost": (
                    float(hypothesis.factor_cost)
                    if np.isfinite(hypothesis.factor_cost)
                    else None
                ),
                "pose_mode_support_count": int(
                    getattr(hypothesis, "mode_support_count", 1)
                ),
                "pose_mode_member_count": int(
                    getattr(hypothesis, "mode_member_count", 1)
                ),
                "pose_w2c": np.asarray(
                    hypothesis.pose_w2c, dtype=np.float64
                ).reshape(4, 4).tolist(),
            }
        )
    return result


def _m2_oracle(
    view: object,
    chart_ids: np.ndarray,
    atlas: MapletFeatureAtlasBank,
    *,
    model: str,
) -> dict[str, object]:
    matches = {}
    for chart_id in chart_ids.tolist():
        match = ground_truth_chart_frame(
            atlas,
            int(chart_id),
            view.pose_w2c,
            view.camera,
            feature_stride=4,
            feature_level=f"m2_{model}",
            model=str(model),
        )
        if match is not None:
            matches[int(chart_id)] = match
    result = {}
    for count in (1, 2, 3):
        candidates = []
        for ids in itertools.combinations(matches, count):
            hypotheses = frame_matches_to_pose_hypotheses(
                atlas,
                [matches[value] for value in ids],
                view.camera,
                include_grouped_pose=True,
            )
            for hypothesis in hypotheses:
                if len(hypothesis.source_chart_ids) != count:
                    continue
                error = pnp_pose_error(
                    hypothesis.pose_w2c, view.pose_w2c
                )
                candidates.append(
                    {
                        "translation_m": float(error.translation_m),
                        "rotation_deg": float(error.rotation_deg),
                        "chart_ids": list(hypothesis.source_chart_ids),
                        "reprojection_error_px": float(
                            hypothesis.reprojection_error_px
                        ),
                    }
                )
        candidates.sort(
            key=lambda value: (
                value["translation_m"] > 0.30
                or value["rotation_deg"] > 3.0,
                value["translation_m"],
                value["rotation_deg"],
            )
        )
        result[f"{count}_chart"] = (
            candidates[0]
            if candidates
            else {
                "translation_m": None,
                "rotation_deg": None,
                "chart_ids": [],
                "reprojection_error_px": None,
            }
        )
    return result


def _geometry_balanced_chart_subsets(
    ordered: Sequence[tuple[int, Sequence[MapletFrameMatch]]],
    atlas: MapletFeatureAtlasBank,
    *,
    count: int,
    maximum_subsets: int,
) -> list[tuple[int, ...]]:
    """Cover chart supports before ranking alternatives inside each support.

    Prefix-only Cartesian products silently excluded lower-ranked but
    geometrically correct charts from every multi-chart pose.  Candidate
    charts already carry phase evidence, while the 2DGS map supplies their
    metric neighbourhood. For triples/quads, retain the old high-evidence
    prefix and compact local coverage. For pairs, enumerate the finite set
    when the caller's budget permits so distractors cannot delete a support.
    """

    subset_size = int(count)
    limit = max(int(maximum_subsets), 0)
    if subset_size < 2 or limit == 0 or len(ordered) < subset_size:
        return []
    row_by_id = {
        int(value): row
        for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    valid_indices = [
        index
        for index, value in enumerate(ordered)
        if int(value[0]) in row_by_id
    ]
    if len(valid_indices) < subset_size:
        return []
    centers = {
        index: np.asarray(
            atlas.centers[row_by_id[int(ordered[index][0])]],
            dtype=np.float64,
        )
        for index in valid_indices
    }
    evidence = {
        index: marginal_frame_log_likelihood_ratio(ordered[index][1])
        for index in valid_indices
    }

    def subset_metrics(
        subset: tuple[int, ...],
    ) -> tuple[float, float, int, tuple[int, ...]]:
        points = np.stack([centers[index] for index in subset])
        distances = np.linalg.norm(
            points[:, None] - points[None], axis=2
        )
        compactness = float(np.max(distances))
        mass = float(
            np.sum(
                [
                    evidence[index]
                    if np.isfinite(evidence[index])
                    else -50.0
                    for index in subset
                ]
            )
        )
        return compactness, -mass, int(np.sum(subset)), subset

    local_candidates: set[tuple[int, ...]] = set()
    coverage = []
    neighbourhood_size = max(6, subset_size + 2)
    for anchor in valid_indices:
        neighbours = sorted(
            (index for index in valid_indices if index != anchor),
            key=lambda index: (
                float(np.linalg.norm(centers[index] - centers[anchor])),
                index,
            ),
        )[:neighbourhood_size]
        candidates = [
            tuple(sorted((anchor, *others)))
            for others in itertools.combinations(
                neighbours, subset_size - 1
            )
        ]
        if not candidates:
            continue
        local_candidates.update(candidates)
        coverage.append(min(candidates, key=subset_metrics))

    # Keep the established high-posterior combinations as complementary
    # hypotheses, but no longer let that prefix define candidate coverage.
    prefix = valid_indices[: min(len(valid_indices), 8)]
    local_candidates.update(
        tuple(value)
        for value in itertools.combinations(prefix, subset_size)
    )
    if subset_size == 2:
        # With at most 24 phase-screened charts there are only 276 pairs.
        # Covering every pair is affordable and essential under uncertain
        # identity: distractor charts otherwise change the nearest-neighbour
        # graph and can remove a correct two-chart support altogether.  Mode
        # combinations inside each support remain likelihood/geometry ranked.
        local_candidates.update(
            tuple(value)
            for value in itertools.combinations(valid_indices, 2)
        )
    result = []
    seen: set[tuple[int, ...]] = set()
    for subset in coverage:
        if subset not in seen:
            result.append(subset)
            seen.add(subset)
            if len(result) >= limit:
                return result
    for subset in sorted(local_candidates, key=subset_metrics):
        if subset in seen:
            continue
        result.append(subset)
        seen.add(subset)
        if len(result) >= limit:
            break
    return result


def _support_balanced_pose_prefix(
    hypotheses: Sequence[object], limit: int
) -> list[object]:
    """Retain chart supports and solver basins before score duplicates."""

    maximum = max(int(limit), 0)
    representatives = []
    solver_variants = []
    remaining = []
    seen: set[tuple[int, ...]] = set()
    seen_variant: set[tuple[tuple[int, ...], str]] = set()
    for hypothesis in hypotheses:
        support = tuple(
            sorted(
                {
                    int(value)
                    for value in getattr(
                        hypothesis, "source_chart_ids", ()
                    )
                }
            )
        )
        variant = str(
            getattr(hypothesis, "control_model", "default")
        )
        if support not in seen:
            representatives.append(hypothesis)
            seen.add(support)
            seen_variant.add((support, variant))
        elif (support, variant) not in seen_variant:
            # A regional seed and its correlated-factor optimum are distinct
            # coarse basins even though they use the same charts. Preserve one
            # of each before near-duplicate solver outputs.
            solver_variants.append(hypothesis)
            seen_variant.add((support, variant))
        else:
            remaining.append(hypothesis)
    return [*representatives, *solver_variants, *remaining][:maximum]


def _frame_mode_pose_signatures(
    atlas: MapletFeatureAtlasBank,
    matches: Sequence[MapletFrameMatch],
    camera: object,
) -> dict[int, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """Resolve each regional mode to its planar pose branches once."""

    result = {}
    for match in matches:
        hypotheses = frame_matches_to_pose_hypotheses(
            atlas,
            [match],
            camera,
            include_individual_poses=True,
            include_grouped_pose=False,
            include_grouped_center_pose=False,
        )
        signatures = []
        for hypothesis in hypotheses:
            pose = np.asarray(
                hypothesis.pose_w2c, dtype=np.float64
            ).reshape(4, 4)
            signatures.append(
                (
                    -pose[:3, :3].T @ pose[:3, 3],
                    pose[:3, :3],
                )
            )
        result[id(match)] = tuple(signatures)
    return result


def _frame_mode_pose_compatibility(
    first: MapletFrameMatch,
    second: MapletFrameMatch,
    signatures: Mapping[
        int, tuple[tuple[np.ndarray, np.ndarray], ...]
    ],
    *,
    translation_scale_m: float = 2.0,
    rotation_scale_deg: float = 10.0,
) -> float:
    """Minimum normalized SE(3) distance between two planar branches."""

    first_values = signatures.get(id(first), ())
    second_values = signatures.get(id(second), ())
    if not first_values or not second_values:
        return float("inf")
    best = float("inf")
    for first_center, first_rotation in first_values:
        for second_center, second_rotation in second_values:
            translation = float(
                np.linalg.norm(first_center - second_center)
            )
            relative = first_rotation @ second_rotation.T
            rotation = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            (np.trace(relative) - 1.0) * 0.5,
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            best = min(
                best,
                float(
                    np.hypot(
                        translation / max(float(translation_scale_m), 1e-6),
                        rotation / max(float(rotation_scale_deg), 1e-6),
                    )
                ),
            )
    return best


def _geometry_consistent_frame_mode_combinations(
    mode_lists: Sequence[Sequence[MapletFrameMatch]],
    signatures: Mapping[
        int, tuple[tuple[np.ndarray, np.ndarray], ...]
    ],
    *,
    maximum_combinations: int,
    beam_width: int = 64,
) -> list[tuple[MapletFrameMatch, ...]]:
    """Beam-marginalize complete chart-mode distributions.

    Raw frame posterior alone often ranks a locally repetitive facade mode
    above the geometrically compatible one. The beam therefore combines the
    mode likelihood with cross-chart agreement of their planar pose branches.
    It never uses GT and never creates persistent point identities.
    """

    if not mode_lists or int(maximum_combinations) <= 0:
        return []
    beam: list[tuple[float, tuple[MapletFrameMatch, ...]]] = [
        (0.0, ())
    ]
    width = max(int(beam_width), int(maximum_combinations))
    for modes in mode_lists:
        expanded = []
        for _old_score, prefix in beam:
            for match in modes:
                values = (*prefix, match)
                log_evidence = float(
                    np.mean(
                        [
                            frame_log_likelihood_ratio(value)
                            for value in values
                        ]
                    )
                )
                pairwise = [
                    _frame_mode_pose_compatibility(
                        values[first],
                        values[second],
                        signatures,
                    )
                    for first in range(len(values))
                    for second in range(first + 1, len(values))
                ]
                finite = [
                    value for value in pairwise if np.isfinite(value)
                ]
                compatibility = (
                    float(np.mean(finite))
                    if finite
                    else (0.0 if len(values) == 1 else 1e3)
                )
                expanded.append(
                    (
                        log_evidence - 1.5 * compatibility,
                        values,
                    )
                )
        expanded.sort(
            key=lambda value: (
                -value[0],
                tuple(
                    -frame_log_likelihood_ratio(match)
                    for match in value[1]
                ),
            )
        )
        beam = expanded[:width]
    return [
        values
        for _score, values in beam[: int(maximum_combinations)]
    ]


def _likelihood_ranked_frame_mode_combinations(
    mode_lists: Sequence[Sequence[MapletFrameMatch]],
    *,
    maximum_combinations: int,
) -> list[tuple[MapletFrameMatch, ...]]:
    """Retain high-probability regional modes independently of geometry.

    Planar charts have two pose branches.  A wrong branch from two repetitive
    facades can therefore look more mutually compatible than the correct
    image-space frames.  Geometry compatibility remains useful, but it must
    not be the only gate before atlas evidence is evaluated.  This finite
    likelihood list is its complementary proposal family; it contains no GT
    signal and preserves the complete chart-frame observations.
    """

    limit = max(int(maximum_combinations), 0)
    if not mode_lists or limit == 0 or any(not values for values in mode_lists):
        return []
    combinations = itertools.product(*mode_lists)
    return sorted(
        (tuple(values) for values in combinations),
        key=lambda values: (
            -float(
                np.mean(
                    [frame_log_likelihood_ratio(value) for value in values]
                )
            ),
            tuple(-frame_log_likelihood_ratio(value) for value in values),
        ),
    )[:limit]


def _deduplicated_mode_combinations(
    *families: Sequence[Sequence[MapletFrameMatch]],
    maximum_combinations: int,
) -> list[tuple[MapletFrameMatch, ...]]:
    """Merge proposal families without spending budget on duplicate modes."""

    result = []
    seen: set[tuple[int, ...]] = set()
    for family in families:
        for values in family:
            combination = tuple(values)
            key = tuple(id(value) for value in combination)
            if key in seen:
                continue
            result.append(combination)
            seen.add(key)
            if len(result) >= int(maximum_combinations):
                return result
    return result


def _predicted_pose_hypotheses(
    chart_matches: Mapping[int, Sequence[MapletFrameMatch]],
    atlas: MapletFeatureAtlasBank,
    view: object,
    *,
    maximum_charts: int,
    maximum_modes_per_chart_for_groups: int = 8,
    maximum_pose_hypotheses: int = 4096,
) -> list[dict[str, object]]:
    ordered = sorted(
        (
            (int(chart_id), tuple(matches))
            for chart_id, matches in chart_matches.items()
            if matches
        ),
        key=lambda value: -marginal_frame_log_likelihood_ratio(value[1]),
    )[: int(maximum_charts)]
    mode_sets: list[Sequence[MapletFrameMatch]] = []
    refined_mode_set_keys: set[tuple[int, ...]] = set()

    def mode_set_key(
        values: Sequence[MapletFrameMatch],
    ) -> tuple[int, ...]:
        return tuple(id(value) for value in values)

    for _chart_id, matches in ordered:
        # Four single-chart planar modes retain IPPE branch coverage without
        # consuming the budget needed for uncertain multi-chart identity.
        singles = [(match,) for match in matches[:4]]
        mode_sets.extend(singles)
        refined_mode_set_keys.update(mode_set_key(value) for value in singles)
    signature_matches = [
        match
        for _chart_id, matches in ordered
        for match in matches[
            : int(maximum_modes_per_chart_for_groups)
        ]
    ]
    signatures = _frame_mode_pose_signatures(
        atlas, signature_matches, view.camera
    )
    pair_subset_count = len(ordered) * (len(ordered) - 1) // 2
    triple_subset_count = min(
        48,
        math.comb(len(ordered), 3) if len(ordered) >= 3 else 0,
    )
    quad_subset_count = min(
        24,
        math.comb(len(ordered), 4) if len(ordered) >= 4 else 0,
    )
    # A 1024-mode pool was sufficient to cover every chart pair, but it gave
    # each triple only three mode combinations.  That is not a faithful
    # marginalization of the structured frame distribution: on strict q2 the
    # three correct RADIO chart frames form geometry rank 19 (and likelihood
    # rank 28) inside their triple, despite each frame being within roughly
    # one feature cell of its target.  The downstream 2DGS atlas therefore
    # never received the valid basin that it was meant to disambiguate.
    #
    # At the production 4096 budget, retain complementary geometry/appearance
    # branches for every pair, 32 modes for every screened triple, and twelve
    # modes for quads.  The resulting finite distribution is still cheap for
    # the sparse GPU atlas screen.  Smaller diagnostic budgets preserve the
    # historical coverage-first allocation below.
    expanded_distribution = int(maximum_pose_hypotheses) >= 2048
    reserved = (
        4 * len(ordered)
        + pair_subset_count
        + 3 * triple_subset_count
        + 3 * quad_subset_count
    )
    rich_pair_count = min(
        pair_subset_count,
        max(int(maximum_pose_hypotheses) - reserved, 0) // 7,
    )
    historical_rich_pair_count = min(
        pair_subset_count,
        max(1024 - reserved, 0) // 7,
    )
    for count in (2, 3, 4):
        if count == 2:
            maximum_subsets = pair_subset_count
            maximum_combinations = 1
        elif count == 3:
            maximum_subsets = 48
            maximum_combinations = 3
        else:
            maximum_subsets = 24
            maximum_combinations = 3
        subset_indices = _geometry_balanced_chart_subsets(
            ordered,
            atlas,
            count=count,
            maximum_subsets=maximum_subsets,
        )
        for subset_rank, indices in enumerate(subset_indices):
            chart_subset = [ordered[index] for index in indices]
            mode_lists = [
                value[1][
                    : int(maximum_modes_per_chart_for_groups)
                ]
                for value in chart_subset
            ]
            if expanded_distribution:
                if count == 2:
                    geometry_count = 4
                    likelihood_count = 4
                    combination_count = 8
                elif count == 3:
                    geometry_count = 24
                    likelihood_count = 8
                    combination_count = 32
                else:
                    geometry_count = 8
                    likelihood_count = 4
                    combination_count = 12
                combinations = _deduplicated_mode_combinations(
                    _geometry_consistent_frame_mode_combinations(
                        mode_lists,
                        signatures,
                        maximum_combinations=geometry_count,
                    ),
                    _likelihood_ranked_frame_mode_combinations(
                        mode_lists,
                        maximum_combinations=likelihood_count,
                    ),
                    maximum_combinations=combination_count,
                )
                if count == 2 and subset_rank < historical_rich_pair_count:
                    refined_combinations = (
                        _deduplicated_mode_combinations(
                            _geometry_consistent_frame_mode_combinations(
                                mode_lists,
                                signatures,
                                maximum_combinations=4,
                            ),
                            _likelihood_ranked_frame_mode_combinations(
                                mode_lists,
                                maximum_combinations=4,
                            ),
                            maximum_combinations=8,
                        )
                    )
                else:
                    refined_combinations = (
                        _geometry_consistent_frame_mode_combinations(
                            mode_lists,
                            signatures,
                            maximum_combinations=(1 if count == 2 else 3),
                        )
                    )
                refined_mode_set_keys.update(
                    mode_set_key(value) for value in refined_combinations
                )
            elif count == 2 and subset_rank < rich_pair_count:
                # Four geometry-compatible modes plus the four strongest
                # independent frame-likelihood modes give each selected pair
                # complementary branch coverage.  The union is capped at
                # eight and de-duplicated, so the global finite budget is
                # preserved.
                combinations = _deduplicated_mode_combinations(
                    _geometry_consistent_frame_mode_combinations(
                        mode_lists,
                        signatures,
                        maximum_combinations=4,
                    ),
                    _likelihood_ranked_frame_mode_combinations(
                        mode_lists,
                        maximum_combinations=4,
                    ),
                    maximum_combinations=8,
                )
            else:
                combinations = (
                    _geometry_consistent_frame_mode_combinations(
                        mode_lists,
                        signatures,
                        maximum_combinations=maximum_combinations,
                    )
                )
            for combination in combinations:
                mode_sets.append(combination)
                if len(mode_sets) >= int(maximum_pose_hypotheses):
                    break
            if len(mode_sets) >= int(maximum_pose_hypotheses):
                break
        if len(mode_sets) >= int(maximum_pose_hypotheses):
            break
    if expanded_distribution:
        refined_mode_sets = [
            value
            for value in mode_sets
            if mode_set_key(value) in refined_mode_set_keys
        ]
        seed_only_mode_sets = [
            value
            for value in mode_sets
            if mode_set_key(value) not in refined_mode_set_keys
        ]
        hypotheses = [
            *pose_hypotheses_for_mode_sets(
                atlas,
                refined_mode_sets,
                view.camera,
                refine_grouped_pose=True,
            ),
            *pose_hypotheses_for_mode_sets(
                atlas,
                seed_only_mode_sets,
                view.camera,
                refine_grouped_pose=False,
            ),
        ]
        hypotheses.sort(
            key=lambda value: (
                -value.score,
                value.reprojection_error_px,
                -len(value.source_chart_ids),
            )
        )
    else:
        hypotheses = list(
            pose_hypotheses_for_mode_sets(
                atlas, mode_sets, view.camera
            )
        )
    return _support_balanced_pose_prefix(
        hypotheses, int(maximum_pose_hypotheses)
    )


def _predicted_pose_diagnostic(
    chart_matches: Mapping[int, Sequence[MapletFrameMatch]],
    atlas: MapletFeatureAtlasBank,
    view: object,
    *,
    maximum_charts: int,
    maximum_modes_per_chart_for_groups: int = 8,
    maximum_pose_hypotheses: int = 1024,
) -> list[dict[str, object]]:
    hypotheses = _predicted_pose_hypotheses(
        chart_matches,
        atlas,
        view,
        maximum_charts=int(maximum_charts),
        maximum_modes_per_chart_for_groups=int(
            maximum_modes_per_chart_for_groups
        ),
        maximum_pose_hypotheses=int(maximum_pose_hypotheses),
    )
    return _pose_errors(
        hypotheses, view.pose_w2c
    )


def _candidate_retrieved_charts(
    retrieval: MapletRetrievalResult,
    index: RegionChartIndex,
    regions: RetrievalRegionBank,
    charts: MetricSurfaceChartBank,
    region_limit: int,
    chart_limit: int,
    *,
    minimum_charts: int = 12,
    cumulative_probability: float = 0.95,
) -> ChartExpansionPosterior:
    region_ids = retrieval.ranked_maplet_ids[: int(region_limit)]
    evidence_by_id = {
        int(region_id): float(value)
        for region_id, value in zip(
            retrieval.ranked_maplet_ids.tolist(),
            retrieval.evidence.tolist(),
        )
    }
    return expand_region_posterior_to_charts(
        region_ids,
        np.asarray(
            [evidence_by_id.get(int(value), 0.0) for value in region_ids],
            dtype=np.float64,
        ),
        regions,
        charts,
        index,
        minimum_charts=min(int(minimum_charts), int(chart_limit)),
        maximum_charts=int(chart_limit),
        cumulative_probability=float(cumulative_probability),
    )


def _chart_expansion_diagnostics(
    retrieval: MapletRetrievalResult,
    posterior: ChartExpansionPosterior,
    visible_ids: np.ndarray,
    visible_counts: np.ndarray,
    index: RegionChartIndex,
    *,
    region_limit: int,
) -> dict[str, object]:
    """Separate retrieval, index expansion, ranking and truncation losses."""

    if visible_ids.size == 0:
        return {}
    dominant = int(visible_ids[int(np.argmax(visible_counts))])
    dominant_regions = set(
        index.regions_for_charts(np.asarray([dominant])).tolist()
    )
    retrieval_rank = {
        int(value): row + 1
        for row, value in enumerate(
            retrieval.ranked_maplet_ids.tolist()
        )
    }
    dominant_region_rank = min(
        (
            retrieval_rank[int(value)]
            for value in dominant_regions
            if int(value) in retrieval_rank
        ),
        default=None,
    )
    normalized_rank = {
        int(value): row + 1
        for row, value in enumerate(posterior.chart_ids.tolist())
    }
    # Historical degree-biased score retained only to quantify the bug.
    selected_regions = retrieval.ranked_maplet_ids[: int(region_limit)]
    evidence_by_region = {
        int(region_id): float(value)
        for region_id, value in zip(
            retrieval.ranked_maplet_ids.tolist(),
            retrieval.evidence.tolist(),
        )
    }
    expanded = index.charts_for_regions(selected_regions)
    legacy = sorted(
        (
            (
                sum(
                    evidence_by_region.get(int(region_id), 0.0)
                    for region_id in index.regions_for_charts(
                        np.asarray([chart_id])
                    ).tolist()
                ),
                int(chart_id),
            )
            for chart_id in expanded.tolist()
        ),
        key=lambda value: (-value[0], value[1]),
    )
    legacy_rank = {
        chart_id: row + 1
        for row, (_score, chart_id) in enumerate(legacy)
    }
    total = max(float(np.sum(visible_counts)), 1.0)
    result: dict[str, object] = {
        "dominant_chart_id": dominant,
        "dominant_region_rank": dominant_region_rank,
        "dominant_region_in_top_regions": bool(
            dominant_region_rank is not None
            and int(dominant_region_rank) <= int(region_limit)
        ),
        "dominant_chart_in_expanded_candidates": bool(
            dominant in set(expanded.tolist())
        ),
        "dominant_chart_degree_biased_rank": legacy_rank.get(dominant),
        "dominant_chart_normalized_rank": normalized_rank.get(dominant),
        "expanded_chart_count": int(posterior.chart_ids.size),
        "selected_chart_count": int(posterior.selected_chart_ids.size),
        "selected_probability_mass": float(
            posterior.selected_probability_mass
        ),
    }
    for rank in (6, 12, 24):
        chart_ids = posterior.chart_ids[:rank]
        hit = np.isin(visible_ids, chart_ids)
        result[f"dominant_chart_recall_at_{rank}"] = bool(
            dominant in set(chart_ids.tolist())
        )
        result[f"visible_chart_count_at_{rank}"] = int(np.sum(hit))
        result[f"visible_surface_coverage_at_{rank}"] = float(
            np.sum(visible_counts[hit]) / total
        )
    return result


def _aggregate_boolean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0


def _aggregate_retrieval(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result = {}
    for rank in (1, 5, 16, 64):
        result[f"dominant_region_recall_at_{rank}"] = _aggregate_boolean(
            rows, f"dominant_region_recall_at_{rank}"
        )
        result[f"visible_surface_coverage_at_{rank}"] = float(
            np.mean(
                [
                    float(row[f"visible_surface_coverage_at_{rank}"])
                    for row in rows
                ]
            )
        )
        result[f"visible_chart_count_median_at_{rank}"] = float(
            np.median(
                [
                    int(row[f"visible_chart_count_at_{rank}"])
                    for row in rows
                ]
            )
        )
    return result


def _aggregate_chart_expansion(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    local = [
        row["d1_region_chart_expansion"]
        for row in rows
        if row.get("d1_region_chart_expansion")
    ]
    if not local:
        return {"query_count": 0}
    result: dict[str, object] = {"query_count": len(local)}
    for key in (
        "dominant_region_in_top_regions",
        "dominant_chart_in_expanded_candidates",
    ):
        result[f"{key}_fraction"] = float(
            np.mean([bool(row[key]) for row in local])
        )
    for rank in (6, 12, 24):
        result[f"dominant_chart_recall_at_{rank}"] = float(
            np.mean(
                [
                    bool(row[f"dominant_chart_recall_at_{rank}"])
                    for row in local
                ]
            )
        )
        result[f"visible_surface_coverage_at_{rank}"] = float(
            np.mean(
                [
                    float(row[f"visible_surface_coverage_at_{rank}"])
                    for row in local
                ]
            )
        )
    for key in (
        "dominant_region_rank",
        "dominant_chart_degree_biased_rank",
        "dominant_chart_normalized_rank",
        "expanded_chart_count",
        "selected_chart_count",
        "selected_probability_mass",
    ):
        values = [
            float(row[key])
            for row in local
            if row.get(key) is not None
        ]
        result[f"{key}_median"] = (
            float(np.median(values)) if values else None
        )
    return result


def _aggregate_m1(
    rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    result = {}
    for level in LEVEL_STRIDE:
        local = [row for row in rows if row["feature_level"] == level]
        if not local:
            result[level] = {
                "example_count": 0,
                "null_fraction": None,
                **{
                    f"{metric}_at_{rank}{suffix}": None
                    for rank in (1, 5, 16)
                    for metric, suffix in (
                        ("center_recall", "_32px"),
                        ("control_recall", "_32px"),
                        ("best_control_error_median", "_px"),
                        ("best_log_scale_error_median", ""),
                        ("best_rotation_error_median", "_deg"),
                    )
                },
            }
            continue
        level_result: dict[str, object] = {
            "example_count": len(local),
            "null_fraction": float(
                np.mean(
                    [
                        not row["mode_errors"]
                        or float(row["diagnostic_null_probability"]) >= 0.5
                        for row in local
                    ]
                )
            ),
        }
        for rank in (1, 5, 16):
            selected = [
                error
                for row in local
                for error in [
                    min(
                        row["mode_errors"][:rank],
                        key=lambda value: value["control_error_px"],
                        default=None,
                    )
                ]
                if error is not None
            ]
            level_result[f"center_recall_at_{rank}_32px"] = float(
                np.mean(
                    [
                        bool(row["mode_errors"][:rank])
                        and min(
                            value["center_error_px"]
                            for value in row["mode_errors"][:rank]
                        )
                        <= 32.0
                        for row in local
                    ]
                )
            )
            level_result[f"control_recall_at_{rank}_32px"] = float(
                np.mean(
                    [
                        bool(row["mode_errors"][:rank])
                        and min(
                            value["control_error_px"]
                            for value in row["mode_errors"][:rank]
                        )
                        <= 32.0
                        for row in local
                    ]
                )
            )
            level_result[f"best_control_error_median_at_{rank}_px"] = (
                float(
                    np.median(
                        [value["control_error_px"] for value in selected]
                    )
                )
                if selected
                else None
            )
            for field, output_name, suffix in (
                (
                    "log_scale_error",
                    "best_log_scale_error_median",
                    "",
                ),
                (
                    "rotation_error_deg",
                    "best_rotation_error_median",
                    "_deg",
                ),
            ):
                best = [
                    min(
                        (
                            float(value[field])
                            for value in row["mode_errors"][:rank]
                        ),
                        default=np.nan,
                    )
                    for row in local
                ]
                finite = [value for value in best if np.isfinite(value)]
                level_result[
                    f"{output_name}_at_{rank}{suffix}"
                ] = (
                    float(np.median(finite)) if finite else None
                )
        result[level] = level_result
    return result


def _aggregate_pose_rows(
    rows: Sequence[Mapping[str, object]], field: str
) -> dict[str, object]:
    result = {}
    for rank in (1, 5, 16):
        result[f"basin_recall_at_{rank}_30cm_3deg"] = float(
            np.mean(
                [
                    any(
                        float(value["translation_m"]) <= 0.30
                        and float(value["rotation_deg"]) <= 3.0
                        for value in row[field][:rank]
                    )
                    for row in rows
                ]
            )
        )
    result["candidate_oracle_recall_30cm_3deg"] = float(
        np.mean(
            [
                any(
                    float(value["translation_m"]) <= 0.30
                    and float(value["rotation_deg"]) <= 3.0
                    for value in row[field]
                )
                for row in rows
            ]
        )
    )
    result["candidate_count_median"] = float(
        np.median([len(row[field]) for row in rows])
    )
    oracle = [
        min(
            row[field],
            key=lambda value: (
                float(value["translation_m"]) / 0.30
                + float(value["rotation_deg"]) / 3.0
            ),
        )
        for row in rows
        if row[field]
    ]
    result["candidate_oracle_translation_median_m"] = (
        float(
            np.median(
                [float(value["translation_m"]) for value in oracle]
            )
        )
        if oracle
        else None
    )
    result["candidate_oracle_rotation_median_deg"] = (
        float(
            np.median(
                [float(value["rotation_deg"]) for value in oracle]
            )
        )
        if oracle
        else None
    )
    for control_model in (
        "chart_factor",
        "diagnostic_chart_centers",
        "frame_controls",
    ):
        result[
            f"candidate_oracle_{control_model}_recall_30cm_3deg"
        ] = float(
            np.mean(
                [
                    any(
                        value.get("control_model", "frame_controls")
                        == control_model
                        and float(value["translation_m"]) <= 0.30
                        and float(value["rotation_deg"]) <= 3.0
                        for value in row[field]
                    )
                    for row in rows
                ]
            )
        )
    top = [row[field][0] for row in rows if row[field]]
    result["top1_translation_median_m"] = (
        float(np.median([value["translation_m"] for value in top]))
        if top
        else None
    )
    result["top1_rotation_median_deg"] = (
        float(np.median([value["rotation_deg"] for value in top]))
        if top
        else None
    )
    return result


def _oracle_best_frame_modes(
    chart_matches: Mapping[int, Sequence[MapletFrameMatch]],
    atlas: MapletFeatureAtlasBank,
    view: object,
) -> dict[int, tuple[MapletFrameMatch, ...]]:
    """GT-only diagnostic: keep the generated mode with least control error."""

    result = {}
    for chart_id, matches in chart_matches.items():
        if not matches:
            continue
        target = ground_truth_chart_frame(
            atlas,
            int(chart_id),
            view.pose_w2c,
            view.camera,
            feature_stride=int(matches[0].feature_stride),
            feature_level="diagnostic_gt_homography",
            model="homography",
        )
        if target is None:
            continue
        best = min(
            matches,
            key=lambda value: frame_parameter_errors(
                value, target
            )["control_error_px"],
        )
        result[int(chart_id)] = (best,)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_json)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError("output exists; pass --force to replace it")
    paths = {
        key: Path(getattr(args, key))
        for key in (
            "retrieval_regions",
            "spatial_maplets",
            "surface_mapper_checkpoint",
            "probability_calibration",
            "region_chart_index",
            "atlas_coarse",
            "atlas_middle",
            "atlas_fine",
            "query_metric_encoder",
        )
    }
    identity_bank = SurfaceRetrievalMapletBank.load_npz(
        paths["retrieval_regions"]
    )
    retrieval_region_bank = RetrievalRegionBank(identity_bank)
    spatial_bank = SurfaceRetrievalMapletBank.load_npz(
        paths["spatial_maplets"]
    )
    mapper, mapper_metadata = load_surface_maplet_mapper(
        paths["surface_mapper_checkpoint"], device=str(args.device)
    )
    calibration = V6ProbabilityCalibration.load_json(
        paths["probability_calibration"]
    )
    index = RegionChartIndex.load_npz(paths["region_chart_index"])
    atlases = {
        level: MapletFeatureAtlasBank.load_npz(paths[f"atlas_{level}"])
        for level in LEVEL_STRIDE
    }
    chart_banks = {
        level: MetricSurfaceChartBank(atlas)
        for level, atlas in atlases.items()
    }
    model, encoder_metadata = load_v6_metric_encoder(
        paths["query_metric_encoder"], device=str(args.device)
    )
    model.eval()
    views = _load_views(
        Path(args.query_contributor_dir),
        atlases["coarse"],
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
    query_trajectories = sorted(
        {str(view.trajectory_id) for view in views}
    )
    protocol = _validate_contract(
        paths,
        atlases,
        encoder_metadata,
        index,
        calibration,
        query_trajectories,
    )
    aggregation_names = (
        "legacy_sum",
        "topq_nms",
        "noisy_or_nms",
        "block_balanced",
    )
    retrieval_rows = {name: [] for name in aggregation_names}
    m1_rows = []
    m1_refined_rows = []
    m1_composite_rows = []
    m1_composite_refined_rows = []
    query_rows = []
    m3_level = str(args.m3_level)
    frame_levels = tuple(dict.fromkeys(args.frame_levels))
    if m3_level not in frame_levels:
        raise ValueError("m3_level must be included in frame_levels")
    search_configs = {
        level: replace(
            _search_config(level),
            support_score_power=float(args.support_score_power),
        )
        for level in frame_levels
    }
    for query_index, view in enumerate(views):
        raw = view.radio.numpy()
        retrievals = {
            name: _retrieve(
                raw,
                identity_bank,
                spatial_bank,
                mapper,
                mapper_metadata,
                calibration,
                (int(view.camera.width), int(view.camera.height)),
                name,
            )
            for name in aggregation_names
        }
        retrieval = retrievals["topq_nms"]
        visible_ids, visible_counts = _visible_charts(
            view, atlases["coarse"]
        )
        for name, value in retrievals.items():
            retrieval_rows[name].append(
                _retrieval_metrics(
                    value, visible_ids, visible_counts, index
                )
            )
        correct_charts = _select_oracle_charts(
            visible_ids,
            visible_counts,
            atlases["coarse"],
            max(
                int(args.m1_charts_per_query),
                int(args.m2_candidate_charts),
            ),
        )
        correct_regions = _select_oracle_regions(
            visible_ids,
            visible_counts,
            index,
            int(args.m1_regions_per_query),
        )
        with torch.no_grad():
            encoded = model(
                view.radio[None].to(str(args.device)),
                view.rgb[None].to(str(args.device)),
            )
        predicted_by_level: dict[
            str, dict[int, tuple[MapletFrameMatch, ...]]
        ] = {level: {} for level in LEVEL_STRIDE}
        refined_predicted_by_level: dict[
            str, dict[int, tuple[MapletFrameMatch, ...]]
        ] = {level: {} for level in LEVEL_STRIDE}
        composite_predicted_by_level: dict[
            str, dict[int, tuple[MapletFrameMatch, ...]]
        ] = {level: {} for level in LEVEL_STRIDE}
        composite_atlas_by_level: dict[
            str, dict[int, MapletFeatureAtlasBank]
        ] = {level: {} for level in LEVEL_STRIDE}
        for level, stride in LEVEL_STRIDE.items():
            if level not in frame_levels:
                continue
            atlas = atlases[level]
            for chart_id in correct_charts[
                : int(args.m1_charts_per_query)
            ].tolist():
                matches = align_maplet_frame_global(
                    atlas,
                    int(chart_id),
                    encoded[level],
                    encoded["matchability"],
                    feature_level=level,
                    feature_stride=stride,
                    config=search_configs[level],
                )
                predicted_by_level[level][int(chart_id)] = matches
                refined_matches = refine_maplet_frame_matches(
                    atlas,
                    matches[:8],
                    encoded[level],
                    encoded["matchability"],
                    iterations=20,
                )
                refined_predicted_by_level[level][
                    int(chart_id)
                ] = refined_matches
                target = ground_truth_chart_frame(
                    atlas,
                    int(chart_id),
                    view.pose_w2c,
                    view.camera,
                    feature_stride=stride,
                    feature_level="m1_gt_homography",
                    model="homography",
                )
                if target is not None:
                    m1_rows.append(
                        _frame_row(
                            view.image_id,
                            level,
                            int(chart_id),
                            matches,
                            target,
                            conditioned=False,
                        )
                    )
                    m1_refined_rows.append(
                        _frame_row(
                            view.image_id,
                            level,
                            int(chart_id),
                            refined_matches,
                            target,
                            conditioned=False,
                        )
                    )
            for region_id in correct_regions.tolist():
                try:
                    composite = compose_metric_region_atlas(
                        retrieval_region_bank,
                        chart_banks[level],
                        index,
                        int(region_id),
                        resolution=48,
                    )
                except ValueError:
                    continue
                composite_atlas_by_level[level][int(region_id)] = composite
                composite_matches = align_maplet_frame_global(
                    composite,
                    int(region_id),
                    encoded[level],
                    encoded["matchability"],
                    feature_level=f"{level}_composite",
                    feature_stride=stride,
                    config=search_configs[level],
                )
                composite_predicted_by_level[level][
                    int(region_id)
                ] = refine_maplet_frame_matches(
                    composite,
                    composite_matches[:8],
                    encoded[level],
                    encoded["matchability"],
                    iterations=20,
                )
                composite_target = ground_truth_chart_frame(
                    composite,
                    int(region_id),
                    view.pose_w2c,
                    view.camera,
                    feature_stride=stride,
                    feature_level="m1_composite_gt_homography",
                    model="homography",
                )
                if composite_target is not None:
                    m1_composite_rows.append(
                        _frame_row(
                            view.image_id,
                            level,
                            int(region_id),
                            composite_matches,
                            composite_target,
                            conditioned=False,
                        )
                    )
                    m1_composite_refined_rows.append(
                        _frame_row(
                            view.image_id,
                            level,
                            int(region_id),
                            composite_predicted_by_level[level][
                                int(region_id)
                            ],
                            composite_target,
                            conditioned=False,
                        )
                    )
        m2 = {
            model_name: _m2_oracle(
                view,
                correct_charts[: int(args.m2_candidate_charts)],
                atlases["fine"],
                model=model_name,
            )
            for model_name in (
                "oriented_similarity",
                "affine",
                "homography",
            )
        }
        correct_composite_matches = composite_predicted_by_level[m3_level]
        correct_composites = composite_atlas_by_level[m3_level]
        correct_chart_pose_errors = _predicted_pose_diagnostic(
            refined_predicted_by_level[m3_level],
            atlases[m3_level],
            view,
            maximum_charts=int(args.m1_charts_per_query),
        )
        if correct_composites:
            correct_pose_errors = _predicted_pose_diagnostic(
                correct_composite_matches,
                merge_metric_atlases(list(correct_composites.values())),
                view,
                maximum_charts=int(args.m1_regions_per_query),
            )
        else:
            correct_pose_errors = []
        region_row_by_id = {
            int(value): row
            for row, value in enumerate(index.region_ids.tolist())
        }
        retrieved_region_ids = []
        for region_id in retrieval.ranked_maplet_ids[
            : int(args.m3_retrieval_regions)
        ].tolist():
            region_row = region_row_by_id.get(int(region_id))
            if (
                region_row is not None
                and int(index.region_offsets[region_row + 1])
                > int(index.region_offsets[region_row])
            ):
                retrieved_region_ids.append(int(region_id))
            if len(retrieved_region_ids) >= int(args.m3_charts):
                break
        retrieved_matches = {}
        retrieved_composites = {}
        for region_id in retrieved_region_ids:
            try:
                composite = compose_metric_region_atlas(
                    retrieval_region_bank,
                    chart_banks[m3_level],
                    index,
                    int(region_id),
                    resolution=48,
                )
            except ValueError:
                continue
            prior_xy, prior_probability = _region_priors(
                int(region_id), retrieval
            )
            raw_matches = align_maplet_frame_global(
                composite,
                int(region_id),
                encoded[m3_level],
                encoded["matchability"],
                feature_level=f"{m3_level}_composite_retrieved",
                feature_stride=LEVEL_STRIDE[m3_level],
                config=search_configs[m3_level],
                query_location_priors=(
                    prior_xy if prior_xy.size else None
                ),
                query_location_prior_weights=(
                    prior_probability if prior_probability.size else None
                ),
                location_prior_sigma_px=96.0,
                location_prior_strength=0.20,
            )
            retrieved_matches[int(region_id)] = (
                refine_maplet_frame_matches(
                    composite,
                    raw_matches[:8],
                    encoded[m3_level],
                    encoded["matchability"],
                    iterations=20,
                )
            )
            retrieved_composites[int(region_id)] = composite
        if retrieved_composites:
            retrieved_pose_errors = _predicted_pose_diagnostic(
                retrieved_matches,
                merge_metric_atlases(list(retrieved_composites.values())),
                view,
                maximum_charts=int(args.m3_charts),
            )
        else:
            retrieved_pose_errors = []
        retrieved_chart_matches = {}
        retrieved_chart_posterior = _candidate_retrieved_charts(
            retrieval,
            index,
            retrieval_region_bank,
            chart_banks[m3_level],
            int(args.m3_retrieval_regions),
            int(args.m3_charts),
            minimum_charts=int(args.m3_min_charts),
            cumulative_probability=float(
                args.m3_chart_probability_mass
            ),
        )
        retrieved_chart_ids = (
            retrieved_chart_posterior.selected_chart_ids
        )
        for chart_id in retrieved_chart_ids.tolist():
            prior_xy, prior_probability = _chart_priors(
                int(chart_id), retrieval, index
            )
            raw_matches = align_maplet_frame_global(
                atlases[m3_level],
                int(chart_id),
                encoded[m3_level],
                encoded["matchability"],
                feature_level=f"{m3_level}_chart_retrieved",
                feature_stride=LEVEL_STRIDE[m3_level],
                config=search_configs[m3_level],
                query_location_priors=(
                    prior_xy if prior_xy.size else None
                ),
                query_location_prior_weights=(
                    prior_probability if prior_probability.size else None
                ),
                location_prior_sigma_px=96.0,
                location_prior_strength=0.20,
            )
            retrieved_chart_matches[int(chart_id)] = tuple(
                replace(
                    value,
                    identity_probability=(
                        retrieved_chart_posterior.probability_for(
                            int(chart_id)
                        )
                    ),
                )
                for value in refine_maplet_frame_matches(
                    atlases[m3_level],
                    raw_matches[:8],
                    encoded[m3_level],
                    encoded["matchability"],
                    iterations=20,
                )
            )
        retrieved_chart_pose_errors = _predicted_pose_diagnostic(
            retrieved_chart_matches,
            atlases[m3_level],
            view,
            maximum_charts=int(args.m3_charts),
        )
        expansion_diagnostic = _chart_expansion_diagnostics(
            retrieval,
            retrieved_chart_posterior,
            visible_ids,
            visible_counts,
            index,
            region_limit=int(args.m3_retrieval_regions),
        )
        correct_oracle_mode_errors = _predicted_pose_diagnostic(
            _oracle_best_frame_modes(
                refined_predicted_by_level[m3_level],
                atlases[m3_level],
                view,
            ),
            atlases[m3_level],
            view,
            maximum_charts=int(args.m1_charts_per_query),
        )
        retrieved_oracle_mode_matches = _oracle_best_frame_modes(
            retrieved_chart_matches,
            atlases[m3_level],
            view,
        )
        retrieved_oracle_mode_errors = _predicted_pose_diagnostic(
            retrieved_oracle_mode_matches,
            atlases[m3_level],
            view,
            maximum_charts=int(args.m3_charts),
        )
        retrieved_gt_frame_matches = {}
        visible_set = set(int(value) for value in visible_ids.tolist())
        for chart_id in retrieved_chart_ids.tolist():
            if int(chart_id) not in visible_set:
                continue
            target = ground_truth_chart_frame(
                atlases[m3_level],
                int(chart_id),
                view.pose_w2c,
                view.camera,
                feature_stride=LEVEL_STRIDE[m3_level],
                feature_level="diagnostic_runtime_retrieval_gt_homography",
                model="homography",
            )
            if target is not None:
                retrieved_gt_frame_matches[int(chart_id)] = (
                    replace(
                        target,
                        identity_probability=(
                            retrieved_chart_posterior.probability_for(
                                int(chart_id)
                            )
                        ),
                    ),
                )
        retrieved_gt_frame_pose_errors = _predicted_pose_diagnostic(
            retrieved_gt_frame_matches,
            atlases[m3_level],
            view,
            maximum_charts=int(args.m3_charts),
        )
        visible_region_ids = index.regions_for_charts(visible_ids)
        query_row = {
            "image_id": view.image_id,
            "trajectory_id": view.trajectory_id,
            "visible_chart_count": int(visible_ids.size),
            "m1_chart_ids": correct_charts[
                : int(args.m1_charts_per_query)
            ].tolist(),
            "m1_region_ids": correct_regions.tolist(),
            "m2": m2,
            "m3_correct_chart_pose_errors": correct_chart_pose_errors,
            "d3_correct_identity_oracle_mode_pose_errors": (
                correct_oracle_mode_errors
            ),
            "m3_correct_region_pose_errors": correct_pose_errors,
            "m3_retrieved_region_ids": retrieved_region_ids,
            "m3_retrieved_chart_ids": retrieved_chart_ids.tolist(),
            "m3_chart_expansion": {
                "ranked_chart_ids": (
                    retrieved_chart_posterior.chart_ids.tolist()
                ),
                "ranked_chart_probability": (
                    retrieved_chart_posterior.chart_probability.tolist()
                ),
                "selected_probability_mass": float(
                    retrieved_chart_posterior.selected_probability_mass
                ),
            },
            "d1_region_chart_expansion": expansion_diagnostic,
            "d2_retrieved_charts_gt_frame_pose_errors": (
                retrieved_gt_frame_pose_errors
            ),
            "d4_retrieved_predicted_frame_oracle_mode_pose_errors": (
                retrieved_oracle_mode_errors
            ),
            "m3_retrieved_visible_region_count": int(
                np.sum(np.isin(retrieved_region_ids, visible_region_ids))
            ),
            "m3_retrieved_chart_pose_errors": (
                retrieved_chart_pose_errors
            ),
            "m3_retrieved_pose_errors": retrieved_pose_errors,
        }
        query_rows.append(query_row)
        print(
            json.dumps(
                {
                    "progress": f"{query_index + 1}/{len(views)}",
                    "image_id": view.image_id,
                    "m3_retrieved_visible_region_count": query_row[
                        "m3_retrieved_visible_region_count"
                    ],
                    "m3_correct_top1": (
                        correct_pose_errors[0]
                        if correct_pose_errors
                        else None
                    ),
                    "m3_retrieved_top1": (
                        retrieved_pose_errors[0]
                        if retrieved_pose_errors
                        else None
                    ),
                }
            ),
            flush=True,
        )
    m2_summary = {}
    for model_name in (
        "oriented_similarity",
        "affine",
        "homography",
    ):
        model_summary = {}
        for count in (1, 2, 3):
            key = f"{count}_chart"
            values = [
                row["m2"][model_name][key] for row in query_rows
            ]
            valid = [
                value
                for value in values
                if value["translation_m"] is not None
            ]
            model_summary[key] = {
                "solved_fraction": float(len(valid) / max(len(values), 1)),
                "within_30cm_3deg": float(
                    np.mean(
                        [
                            value["translation_m"] is not None
                            and float(value["translation_m"]) <= 0.30
                            and float(value["rotation_deg"]) <= 3.0
                            for value in values
                        ]
                    )
                ),
                "translation_median_m": (
                    float(
                        np.median(
                            [value["translation_m"] for value in valid]
                        )
                    )
                    if valid
                    else None
                ),
                "rotation_median_deg": (
                    float(
                        np.median(
                            [value["rotation_deg"] for value in valid]
                        )
                    )
                    if valid
                    else None
                ),
            }
        m2_summary[model_name] = model_summary
    report = {
        "stage": "v6_maplet_frame_alignment_m1_m2_m3",
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
            "uses_point_correspondence_pnp": False,
            "uses_final_point_correspondence_pnp": False,
            "uses_regional_frame_control_seed_solver": True,
            "uses_chart_block_factor_refinement": True,
            "includes_chart_center_grouped_diagnostic": False,
            "coarse_geometry": (
                "chart_projection_to_IPPE_or_EPNP_seed_then_correlated_chart_factor"
            ),
        },
        "scene_evidence": {
            name: _aggregate_retrieval(rows)
            for name, rows in retrieval_rows.items()
        },
        "m1_correct_chart_global_correlation": _aggregate_m1(m1_rows),
        "m1_correct_chart_local_refinement": _aggregate_m1(
            m1_refined_rows
        ),
        "m1_correct_composite_region_global_correlation": _aggregate_m1(
            m1_composite_rows
        ),
        "m1_correct_composite_region_local_refinement": _aggregate_m1(
            m1_composite_refined_rows
        ),
        "m2_frame_to_pose_oracle": m2_summary,
        "m3_correct_metric_charts_predicted_frame_pose": (
            _aggregate_pose_rows(query_rows, "m3_correct_chart_pose_errors")
        ),
        "m3_correct_composite_region_predicted_frame_pose": _aggregate_pose_rows(
            query_rows, "m3_correct_region_pose_errors"
        ),
        "m3_retrieved_metric_charts_predicted_frame_pose": (
            _aggregate_pose_rows(
                query_rows, "m3_retrieved_chart_pose_errors"
            )
        ),
        "m3_retrieved_composite_region_predicted_frame_pose": _aggregate_pose_rows(
            query_rows, "m3_retrieved_pose_errors"
        ),
        "d1_region_to_chart_coverage": _aggregate_chart_expansion(
            query_rows
        ),
        "d2_actual_retrieval_gt_frame_pose": _aggregate_pose_rows(
            query_rows, "d2_retrieved_charts_gt_frame_pose_errors"
        ),
        "d3_correct_identity_predicted_frame_oracle_mode_pose": (
            _aggregate_pose_rows(
                query_rows,
                "d3_correct_identity_oracle_mode_pose_errors",
            )
        ),
        "d4_actual_retrieval_predicted_frame_oracle_mode_pose": (
            _aggregate_pose_rows(
                query_rows,
                "d4_retrieved_predicted_frame_oracle_mode_pose_errors",
            )
        ),
        "diagnostic_probability_note": (
            "frame-mode null/probabilities are explicitly uncalibrated "
            "diagnostics; they are not production confidence"
        ),
        "configuration": {
            "m1_charts_per_query": int(args.m1_charts_per_query),
            "m1_regions_per_query": int(args.m1_regions_per_query),
            "m2_candidate_charts": int(args.m2_candidate_charts),
            "m3_retrieval_regions": int(args.m3_retrieval_regions),
            "m3_charts": int(args.m3_charts),
            "m3_min_charts": int(args.m3_min_charts),
            "m3_chart_probability_mass": float(
                args.m3_chart_probability_mass
            ),
            "m3_level": m3_level,
            "frame_levels": list(frame_levels),
            "scene_evidence_for_m3": "topq_nms",
            "retrieval_location_prior_sigma_px": 96.0,
            "retrieval_location_prior_strength": 0.20,
            "composite_region_resolution": 48,
            "local_refinement_iterations": 20,
            "local_refinement_model": "projective_homography",
            "local_refinement_fixed_canonical_denominator": True,
            "joint_translation_affine_nms": True,
            "maximum_modes_per_spatial_cluster": 2,
            "maximum_appearance_modes": 2,
            "support_score_power": float(args.support_score_power),
            "maximum_pose_hypotheses": 1024,
            "grouped_modes_per_chart": {
                "two_or_three_charts": 5,
                "four_charts": 3,
            },
        },
        "queries": query_rows,
        "m1_rows": m1_rows,
        "m1_composite_rows": m1_composite_rows,
        "m1_composite_refined_rows": m1_composite_refined_rows,
        "retrieval_rows": retrieval_rows,
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
                        "scene_evidence",
                        "m1_correct_chart_global_correlation",
                        "m1_correct_chart_local_refinement",
                        "m1_correct_composite_region_global_correlation",
                        "m1_correct_composite_region_local_refinement",
                        "m2_frame_to_pose_oracle",
                        "m3_correct_metric_charts_predicted_frame_pose",
                        "m3_correct_composite_region_predicted_frame_pose",
                        "m3_retrieved_metric_charts_predicted_frame_pose",
                        "m3_retrieved_composite_region_predicted_frame_pose",
                        "d1_region_to_chart_coverage",
                        "d2_actual_retrieval_gt_frame_pose",
                        "d3_correct_identity_predicted_frame_oracle_mode_pose",
                        "d4_actual_retrieval_predicted_frame_oracle_mode_pose",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
