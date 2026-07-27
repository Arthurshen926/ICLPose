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
    MapletFrameSearchConfig,
    align_maplet_frame_global,
    frame_matches_to_pose_hypotheses,
    frame_parameter_errors,
    ground_truth_chart_frame,
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
    parser.add_argument("--m3_charts", type=int, default=6)
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
        },
        "modes": [
            {
                "center_xy": match.query_center_xy.tolist(),
                "scale_xy": match.scale_xy.tolist(),
                "rotation_deg": float(match.in_plane_rotation_deg),
                "canonical_to_query": (
                    match.canonical_to_query.tolist()
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
) -> tuple[np.ndarray, np.ndarray]:
    associated = set(
        index.regions_for_charts(np.asarray([chart_id])).tolist()
    )
    xy = []
    probability = []
    for group in retrieval.groups:
        for region_id, value in zip(
            group.maplet_ids.tolist(), group.probabilities.tolist()
        ):
            if int(region_id) in associated and float(value) > 0.0:
                xy.append(np.asarray(group.query_region_xy, dtype=np.float32))
                probability.append(float(value))
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
) -> list[dict[str, object]]:
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


def _predicted_pose_diagnostic(
    chart_matches: Mapping[int, Sequence[MapletFrameMatch]],
    atlas: MapletFeatureAtlasBank,
    view: object,
    *,
    maximum_charts: int,
    maximum_modes_per_chart_for_groups: int = 5,
    maximum_pose_hypotheses: int = 1024,
) -> list[dict[str, object]]:
    ordered = sorted(
        (
            (int(chart_id), tuple(matches))
            for chart_id, matches in chart_matches.items()
            if matches
        ),
        key=lambda value: -float(value[1][0].score),
    )[: int(maximum_charts)]
    mode_sets: list[Sequence[MapletFrameMatch]] = []
    for _chart_id, matches in ordered:
        mode_sets.extend((match,) for match in matches[:16])
    for count in (2, 3, 4):
        for chart_subset in itertools.combinations(ordered[:4], count):
            per_chart_limit = (
                min(
                    int(maximum_modes_per_chart_for_groups),
                    3,
                )
                if count == 4
                else int(maximum_modes_per_chart_for_groups)
            )
            mode_lists = [
                value[1][:per_chart_limit]
                for value in chart_subset
            ]
            for combination in itertools.product(*mode_lists):
                mode_sets.append(combination)
    hypotheses = pose_hypotheses_for_mode_sets(
        atlas, mode_sets, view.camera
    )
    return _pose_errors(
        hypotheses[: int(maximum_pose_hypotheses)], view.pose_w2c
    )


def _candidate_retrieved_charts(
    retrieval: MapletRetrievalResult,
    index: RegionChartIndex,
    region_limit: int,
    chart_limit: int,
) -> np.ndarray:
    region_ids = retrieval.ranked_maplet_ids[: int(region_limit)]
    region_evidence = {
        int(region_id): float(value)
        for region_id, value in zip(
            retrieval.ranked_maplet_ids.tolist(),
            retrieval.evidence.tolist(),
        )
    }
    candidates = index.charts_for_regions(region_ids)
    scored = []
    for chart_id in candidates.tolist():
        associated = index.regions_for_charts(np.asarray([chart_id]))
        evidence = sum(
            region_evidence.get(int(region_id), 0.0)
            for region_id in associated.tolist()
        )
        scored.append((float(evidence), int(chart_id)))
    scored.sort(reverse=True)
    return np.asarray(
        [chart_id for _score, chart_id in scored[: int(chart_limit)]],
        dtype=np.int64,
    )


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
    for control_model in ("frame_controls", "chart_centers"):
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
                    feature_level="m1_gt_affine",
                    model="affine",
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
                    feature_level="m1_composite_gt_affine",
                    model="affine",
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
        retrieved_chart_ids = _candidate_retrieved_charts(
            retrieval,
            index,
            int(args.m3_retrieval_regions),
            int(args.m3_charts),
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
            retrieved_chart_matches[int(chart_id)] = (
                refine_maplet_frame_matches(
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
            "m3_correct_region_pose_errors": correct_pose_errors,
            "m3_retrieved_region_ids": retrieved_region_ids,
            "m3_retrieved_chart_ids": retrieved_chart_ids.tolist(),
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
            "uses_regional_frame_control_pose_solver": True,
            "includes_chart_center_grouped_diagnostic": True,
            "coarse_geometry": (
                "canonical_metric_chart_frame_controls_to_IPPE_or_grouped_pose"
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
            "m3_level": m3_level,
            "frame_levels": list(frame_levels),
            "scene_evidence_for_m3": "topq_nms",
            "retrieval_location_prior_sigma_px": 96.0,
            "retrieval_location_prior_strength": 0.20,
            "composite_region_resolution": 48,
            "local_refinement_iterations": 20,
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
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
