"""Replay V6 rendered-atlas Stage C without recomputing retrieval/frames.

Each input report is one Stage-B proposal branch that stores every pose
hypothesis. This tool reconstructs an equal-budget pool per branch, unions
those finite proposals, optionally shards them across GPUs, and lets the same
RADIO atlas likelihood compare every branch. No mapping RGB, image descriptor,
point identity, or point-PnP input is used.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_v6_radio_frame_source import (
    POSE_MODE_ROTATION_RADIUS_DEG,
    POSE_MODE_TRANSLATION_RADIUS_M,
    STAGE_C_MINIMUM_VIEW_SUPPORTED_CELL_FRACTION,
    _alike_matchability_map,
    _project_surface_spatial_feature_map,
    _sha256,
    _stage_c_pose_pool,
    _stage_c_rows,
)
from feature_extract.tools.vfm.evaluate_v6_maplet_frame_alignment import (
    _predicted_pose_hypotheses,
)
from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.localization.alike_detector_only import (
    AlikeDetectorOnly,
)
from feature_extract.vfm.localization_v6.map_entities import RegionChartIndex
from feature_extract.vfm.localization_v6.atlas_renderer import (
    MINIMUM_MODE_VIEW_DIRECTION_COSINE,
    VIEW_DIRECTION_ANGULAR_STD_FLOOR_DEG,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.maplet_frame_alignment import (
    FramePoseHypothesis,
    MapletFrameMatch,
    factorized_pose_distribution_modes,
    pose_distribution_consensus_modes,
)
from feature_extract.vfm.localization_v6.surface_spatial_projection import (
    load_surface_spatial_projection,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_report", nargs="+", required=True)
    parser.add_argument("--radio_atlas", required=True)
    parser.add_argument("--region_chart_index", required=True)
    parser.add_argument(
        "--frame_spatial_projection_checkpoint", required=True
    )
    parser.add_argument("--query_contributor_dir", required=True)
    parser.add_argument("--spatial_query_token_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--image_id", default="")
    parser.add_argument("--query_index", type=int, default=0)
    parser.add_argument(
        "--pool_size",
        type=int,
        default=256,
        help=(
            "Stage-C proposal budget per source report. Multiple proposal "
            "branches receive equal independent budgets before RADIO atlas "
            "preranking."
        ),
    )
    parser.add_argument(
        "--source_pool_mode",
        choices=("complete", "compressed"),
        default="complete",
        help=(
            "complete exposes every regional Stage-B pose to the cheap RADIO "
            "broad screen; compressed retains the legacy finite pre-prefix."
        ),
    )
    parser.add_argument(
        "--regenerate_pose_distribution_from_frame_modes",
        action="store_true",
        help=(
            "Rebuild the coarse pose distribution from the serialized "
            "query-side structured RADIO frame modes. This is an exact "
            "Stage-B replay and does not read mapping RGB or GT."
        ),
    )
    parser.add_argument(
        "--regenerated_pose_hypotheses", type=int, default=4096
    )
    parser.add_argument(
        "--broad_screen_candidates", type=int, default=256
    )
    parser.add_argument(
        "--broad_screen_radius_cells", type=int, default=6
    )
    parser.add_argument(
        "--broad_screen_maximum_points", type=int, default=192
    )
    parser.add_argument(
        "--pose_candidates",
        type=int,
        default=0,
        help="Candidates refined after full-pool atlas preranking; 0 keeps all.",
    )
    parser.add_argument("--pool_shard_count", type=int, default=1)
    parser.add_argument("--pool_shard_index", type=int, default=0)
    parser.add_argument(
        "--pool_indices",
        default="",
        help="Comma-separated global Stage-C pool indices; overrides sharding.",
    )
    parser.add_argument("--render_charts", type=int, default=24)
    parser.add_argument("--refinement_charts", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument(
        "--maximum_translation_updates", type=int, default=2
    )
    parser.add_argument("--base_stride", type=int, choices=(8, 16), default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--alike_detector_matchability",
        action="store_true",
        help=(
            "Use ALIKE detector scores only as offset-neutral rendered-cell "
            "reliability; RADIO alone determines local displacement."
        ),
    )
    parser.add_argument(
        "--report_view_direction_diagnostic",
        action="store_true",
        help="Report the non-ranking absolute baking-view density.",
    )
    parser.add_argument("--alike_model", default="alike-t")
    parser.add_argument("--alike_detector_top_k", type=int, default=512)
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument(
        "--replace_selected_pose_with_query_gt_diagnostic",
        action="store_true",
        help=(
            "Non-deployable component audit: retain the selected pool "
            "candidate's source charts but replace only its pose by query GT."
        ),
    )
    parser.add_argument("--score_calibration", default="")
    parser.add_argument(
        "--score_calibration_candidate_budget",
        type=int,
        default=0,
        help=(
            "Add this many trajectory-disjoint calibrated candidates to "
            "the preserved exact/local candidate union before refinement."
        ),
    )
    parser.add_argument(
        "--nondeployable_pool_selection_diagnostic",
        action="store_true",
        help=(
            "Mark an explicitly selected pool slice as target-informed or "
            "otherwise non-deployable. The selected pose is not modified."
        ),
    )
    return parser.parse_args(argv)


def _as_hypothesis(row: Mapping[str, object]) -> FramePoseHypothesis:
    factor_cost = row.get("factor_cost", float("inf"))
    return FramePoseHypothesis(
        pose_w2c=np.asarray(row["pose_w2c"], dtype=np.float64).reshape(4, 4),
        score=float(row.get("score", float("-inf"))),
        source_chart_ids=tuple(
            int(value) for value in row.get("source_chart_ids", ())
        ),
        reprojection_error_px=float(
            row.get("reprojection_error_px", float("inf"))
        ),
        positive_depth=bool(row.get("positive_depth", True)),
        control_model=str(row.get("control_model", "chart_factor")),
        seed_model=str(row.get("seed_model", "unknown")),
        factor_cost=(
            float(factor_cost) if factor_cost is not None else float("inf")
        ),
        mode_support_count=int(row.get("pose_mode_support_count", 1)),
        mode_member_count=int(row.get("pose_mode_member_count", 1)),
    )


def _as_frame_match(
    row: Mapping[str, object], chart_id: int
) -> MapletFrameMatch:
    """Deserialize one query-side structured frame sufficient statistic."""

    def optional_array(name: str) -> np.ndarray | None:
        value = row.get(name)
        return (
            None
            if value is None
            else np.asarray(value, dtype=np.float64)
        )

    return MapletFrameMatch(
        chart_id=int(chart_id),
        canonical_to_query=np.asarray(
            row["canonical_to_query"], dtype=np.float64
        ),
        query_center_xy=np.asarray(
            row["query_center_xy"], dtype=np.float64
        ),
        scale_xy=np.asarray(row["scale_xy"], dtype=np.float64),
        in_plane_rotation_deg=float(row["in_plane_rotation_deg"]),
        covariance_xy=np.asarray(
            row["covariance_xy"], dtype=np.float64
        ),
        score=float(row["score"]),
        probability=float(row["probability"]),
        null_probability=float(row["null_probability"]),
        support_fraction=float(row["support_fraction"]),
        feature_level=str(row["feature_level"]),
        feature_stride=int(row["feature_stride"]),
        canonical_homography=optional_array("canonical_homography"),
        support_canonical_hull=optional_array(
            "support_canonical_hull"
        ),
        control_covariance_px=optional_array("control_covariance_px"),
        identity_probability=float(row.get("identity_probability", 1.0)),
    )


def _frame_matches_from_query(
    query: Mapping[str, object],
) -> dict[int, tuple[MapletFrameMatch, ...]]:
    result = {}
    for chart_row in query.get("m3_retrieved_chart_frame_modes", ()):
        chart_id = int(chart_row["chart_id"])
        modes = tuple(
            _as_frame_match(value, chart_id)
            for value in chart_row.get("modes", ())
        )
        if modes:
            result[chart_id] = modes
    return result


def _query_row(
    report: Mapping[str, object], image_id: str, query_index: int
) -> Mapping[str, object]:
    queries = list(report.get("queries", ()))
    if image_id:
        matches = [
            value for value in queries if str(value.get("image_id")) == image_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one source query for {image_id!r}, got {len(matches)}"
            )
        return matches[0]
    if not 0 <= int(query_index) < len(queries):
        raise IndexError("query_index is outside the source report")
    return queries[int(query_index)]


def _pool_indices(args: argparse.Namespace, pool_count: int) -> list[int]:
    if str(args.pool_indices).strip():
        values = [
            int(value)
            for value in str(args.pool_indices).split(",")
            if value.strip()
        ]
    else:
        shards = int(args.pool_shard_count)
        shard = int(args.pool_shard_index)
        if shards <= 0 or not 0 <= shard < shards:
            raise ValueError("invalid pool shard")
        values = [index for index in range(pool_count) if index % shards == shard]
    if any(index < 0 or index >= pool_count for index in values):
        raise IndexError("pool_indices contains an index outside the pool")
    if len(values) != len(set(values)):
        raise ValueError("pool_indices must be unique")
    return values


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    source_paths = [Path(value) for value in args.source_report]
    atlas_path = Path(args.radio_atlas)
    index_path = Path(args.region_chart_index)
    projection_path = Path(args.frame_spatial_projection_checkpoint)
    reports = [
        json.loads(source_path.read_text()) for source_path in source_paths
    ]
    queries = [
        _query_row(report, str(args.image_id), int(args.query_index))
        for report in reports
    ]
    image_ids = {str(query["image_id"]) for query in queries}
    if len(image_ids) != 1:
        raise ValueError(
            "Stage-B proposal reports do not describe the same query"
        )
    image_id = next(iter(image_ids))
    reference_artifact = reports[0].get("artifact_sha256")
    if any(
        report.get("artifact_sha256") != reference_artifact
        for report in reports[1:]
    ):
        raise ValueError(
            "Stage-B proposal reports use different map artifacts"
        )

    atlas = MapletFeatureAtlasBank.load_npz(atlas_path)
    metric_stride = int((atlas.metadata or {}).get("metric_feature_stride", -1))
    if metric_stride != int(args.base_stride):
        raise ValueError(
            f"atlas metric stride {metric_stride} != replay stride {args.base_stride}"
        )
    index = RegionChartIndex.load_npz(index_path)
    projection, projection_metadata = load_surface_spatial_projection(
        projection_path, device=str(args.device)
    )
    projection.eval()

    views = _load_views(
        Path(args.query_contributor_dir),
        atlas,
        Path(args.image_root),
        trajectory_ids=(image_id.split("/", 1)[0],),
    )
    matches = [value for value in views if value.image_id == image_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one contributor view for {image_id!r}, got {len(matches)}"
        )
    view = matches[0]
    spatial_path = (
        Path(args.spatial_query_token_dir)
        / f"{image_id.replace('/', '__')}.npz"
    )
    with np.load(spatial_path, allow_pickle=False) as data:
        radio_final = np.asarray(data["radio_final"], dtype=np.float32)
    query_feature = _project_surface_spatial_feature_map(
        projection, radio_final, device=str(args.device)
    )
    if query_feature.shape[0] != atlas.feature_dim:
        raise ValueError("projected query and map atlas feature dimensions differ")

    query_matchability = None
    detector_metadata: Mapping[str, object] = {}
    detection_count = 0
    if bool(args.alike_detector_matchability):
        detector = AlikeDetectorOnly(
            device=str(args.device),
            matcha_repo=Path(args.matcha_repo),
            model_name=str(args.alike_model),
        )
        detections = detector.detect(
            Path(args.image_root) / image_id,
            image_width=int(view.camera.width),
            image_height=int(view.camera.height),
            top_k=int(args.alike_detector_top_k),
            candidate_top_k=max(4096, int(args.alike_detector_top_k)),
            nms_radius_px=4.0,
            grid_rows=8,
            grid_cols=8,
        )
        detection_count = int(detections.xy.shape[0])
        detector_metadata = dict(detector.metadata)
        query_matchability = _alike_matchability_map(
            detections.xy,
            detections.scores,
            image_width=int(view.camera.width),
            image_height=int(view.camera.height),
            feature_width=int(query_feature.shape[2]),
            feature_height=int(query_feature.shape[1]),
            feature_stride=int(args.base_stride),
        )

    pose_generation_started = time.perf_counter()
    pools = []
    consensus_counts = []
    factorized_counts = []
    raw_counts = []
    for source_index, query in enumerate(queries):
        if bool(args.regenerate_pose_distribution_from_frame_modes):
            frame_matches = _frame_matches_from_query(query)
            if not frame_matches:
                raise ValueError(
                    "source report contains no serialized frame modes"
                )
            generated = _predicted_pose_hypotheses(
                frame_matches,
                atlas,
                view,
                maximum_charts=len(frame_matches),
                maximum_pose_hypotheses=int(
                    args.regenerated_pose_hypotheses
                ),
            )
            raw = [
                replace(
                    value,
                    seed_model=(
                        f"proposal_source_{source_index}:"
                        f"{value.seed_model}"
                    ),
                )
                for value in generated
            ]
        else:
            raw = [
                replace(
                    _as_hypothesis(value),
                    seed_model=(
                        f"proposal_source_{source_index}:"
                        f"{value.get('seed_model', 'unknown')}"
                    ),
                )
                for value in query.get(
                    "m3_retrieved_chart_pose_errors", ()
                )
            ]
        if not raw:
            raise ValueError(
                "a source report contains no replayable Stage-B poses"
            )
        # Recompute derived distributions independently. Proposal scores are
        # branch-internal and must never let one branch consume another
        # branch's finite support budget before RADIO atlas evidence.
        consensus = list(
            pose_distribution_consensus_modes(
                raw,
                translation_radius_m=POSE_MODE_TRANSLATION_RADIUS_M,
                rotation_radius_deg=POSE_MODE_ROTATION_RADIUS_DEG,
            )
        )
        factorized = list(
            factorized_pose_distribution_modes(raw, consensus)
        )
        if str(args.source_pool_mode) == "complete":
            regional_raw = [
                value
                for value in raw
                if len(set(value.source_chart_ids)) >= 2
            ]
            tagged_consensus = [
                replace(
                    value,
                    seed_model=(
                        f"proposal_source_{source_index}:"
                        f"{value.seed_model}"
                    ),
                )
                for value in consensus[:32]
            ]
            tagged_factorized = [
                replace(
                    value,
                    seed_model=(
                        f"proposal_source_{source_index}:"
                        f"{value.seed_model}"
                    ),
                )
                for value in factorized[:32]
            ]
            pools.append(
                [
                    *regional_raw,
                    *tagged_consensus,
                    *tagged_factorized,
                ]
            )
        else:
            pools.append(
                _stage_c_pose_pool(
                    raw,
                    consensus,
                    factorized,
                    int(args.pool_size),
                    minimum_source_charts=2,
                )
            )
        raw_counts.append(len(raw))
        consensus_counts.append(len(consensus))
        factorized_counts.append(len(factorized))
    pool = [value for values in pools for value in values]
    pose_generation_seconds = float(
        time.perf_counter() - pose_generation_started
    )
    selected_indices = _pool_indices(args, len(pool))
    selected = [pool[index] for index in selected_indices]
    if bool(args.replace_selected_pose_with_query_gt_diagnostic):
        selected = [
            replace(
                value,
                pose_w2c=np.asarray(view.pose_w2c, dtype=np.float64),
                control_model="query_gt_component_diagnostic",
                seed_model="QUERY_GT_NOT_DEPLOYABLE",
            )
            for value in selected
        ]
    ranked_chart_lists = [
        [
            int(value["chart_id"])
            for value in query.get(
                "m3_phase_chart_log_evidence", ()
            )
        ]
        for query in queries
    ]
    ranked_chart_union = []
    for rank in range(
        max((len(values) for values in ranked_chart_lists), default=0)
    ):
        for values in ranked_chart_lists:
            if (
                rank < len(values)
                and values[rank] not in ranked_chart_union
            ):
                ranked_chart_union.append(values[rank])
    ranked_chart_ids = np.asarray(
        ranked_chart_union, dtype=np.int64
    )
    pose_candidates = (
        len(selected)
        if int(args.pose_candidates) <= 0
        else min(int(args.pose_candidates), len(selected))
    )
    score_calibration = None
    if str(args.score_calibration):
        from feature_extract.vfm.localization_v6.stage_c_score_calibration import (
            StageCScoreCalibration,
        )

        score_calibration = StageCScoreCalibration.load_json(
            Path(args.score_calibration)
        )
        score_calibration.validate_report_lineage(
            {
                "radio_atlas_sha256": _sha256(atlas_path),
                "image_id": image_id,
            }
        )
    stage_c_audit: dict[str, object] = {}
    stage_c_started = time.perf_counter()
    rows = _stage_c_rows(
        selected,
        atlas,
        ranked_chart_ids,
        query_feature,
        query_matchability,
        view,
        region_chart_index=index,
        maximum_candidates=pose_candidates,
        prerank_pool=len(selected),
        render_charts=int(args.render_charts),
        refinement_charts=int(args.refinement_charts),
        rounds=int(args.rounds),
        maximum_translation_updates=int(
            args.maximum_translation_updates
        ),
        device=str(args.device),
        base_stride=int(args.base_stride),
        preselected_pool=True,
        report_view_direction_diagnostic=bool(
            args.report_view_direction_diagnostic
        ),
        broad_screen_candidates=int(args.broad_screen_candidates),
        broad_screen_radius_cells=int(
            args.broad_screen_radius_cells
        ),
        broad_screen_maximum_points=int(
            args.broad_screen_maximum_points
        ),
        selection_score_calibration=score_calibration,
        calibrated_candidate_budget=int(
            args.score_calibration_candidate_budget
        ),
        stage_c_audit=stage_c_audit,
    )
    stage_c_seconds = float(time.perf_counter() - stage_c_started)
    for row in rows:
        local_index = int(row["broad_screen_source_rank"])
        row["replay_pool_index"] = int(selected_indices[local_index])

    source_reports_deployable = bool(
        all(report.get("deployable_result", True) for report in reports)
    )
    if score_calibration is not None:
        rows = score_calibration.apply(rows)
    payload = {
        "stage": "v6_stage_c_feature_atlas_replay",
        "source_report": str(source_paths[0]),
        "source_report_sha256": _sha256(source_paths[0]),
        "source_reports": [str(value) for value in source_paths],
        "source_report_sha256s": [
            _sha256(value) for value in source_paths
        ],
        "proposal_source_count": len(source_paths),
        "radio_atlas": str(atlas_path),
        "radio_atlas_sha256": _sha256(atlas_path),
        "region_chart_index": str(index_path),
        "frame_spatial_projection_checkpoint": str(projection_path),
        "projection_best_step": int(projection_metadata.get("best_step", -1)),
        "image_id": image_id,
        "pool_size_requested": int(args.pool_size),
        "pool_size_per_source_requested": int(args.pool_size),
        "pool_size_reconstructed": len(pool),
        "source_pool_mode": str(args.source_pool_mode),
        "pose_distribution_regenerated_from_frame_modes": bool(
            args.regenerate_pose_distribution_from_frame_modes
        ),
        "regenerated_pose_hypotheses": int(
            args.regenerated_pose_hypotheses
        ),
        "source_pool_counts": [len(values) for values in pools],
        "pose_generation_seconds": pose_generation_seconds,
        "stage_c_seconds": stage_c_seconds,
        "broad_screen_candidates": int(
            args.broad_screen_candidates
        ),
        "broad_screen_radius_cells": int(
            args.broad_screen_radius_cells
        ),
        "broad_screen_maximum_points": int(
            args.broad_screen_maximum_points
        ),
        "stage_c_audit": stage_c_audit,
        "raw_pose_counts": raw_counts,
        "consensus_mode_counts_recomputed": consensus_counts,
        "factorized_mode_counts_recomputed": factorized_counts,
        "consensus_mode_count_recomputed": int(
            sum(consensus_counts)
        ),
        "factorized_mode_count_recomputed": int(
            sum(factorized_counts)
        ),
        # ``selected_pool_indices`` is retained for artifact compatibility;
        # it denotes the pool slice rendered by the preranker, not the finite
        # exact/local union that proceeds to refinement.
        "selected_pool_indices": selected_indices,
        "evaluated_pool_indices": selected_indices,
        "refined_pool_indices": [
            int(row["replay_pool_index"]) for row in rows
        ],
        "pose_candidates": int(pose_candidates),
        "valid_candidate_count": len(rows),
        "rounds": int(args.rounds),
        "maximum_translation_updates": int(
            args.maximum_translation_updates
        ),
        "render_charts": int(args.render_charts),
        "refinement_charts": int(args.refinement_charts),
        "base_stride": int(args.base_stride),
        "device": str(args.device),
        "alike_detector_matchability": bool(
            args.alike_detector_matchability
        ),
        "alike_detector_local_cell_reliability": bool(
            args.alike_detector_matchability
        ),
        "alike_detector_local_proposal_only": False,
        "alike_detector_offset_prior": False,
        "alike_descriptors_computed_or_stored": False,
        "alike_detection_count": detection_count,
        "alike_detector": dict(detector_metadata),
        "query_gt_pose_component_diagnostic": bool(
            args.replace_selected_pose_with_query_gt_diagnostic
        ),
        "nondeployable_pool_selection_diagnostic": bool(
            args.nondeployable_pool_selection_diagnostic
        ),
        "deployable_result": bool(
            source_reports_deployable
            and not args.replace_selected_pose_with_query_gt_diagnostic
            and not args.nondeployable_pool_selection_diagnostic
        ),
        "source_reports_deployable": source_reports_deployable,
        "score_calibration": str(args.score_calibration),
        "score_calibration_candidate_budget": int(
            args.score_calibration_candidate_budget
        ),
        "score_calibration_sha256": (
            _sha256(Path(args.score_calibration))
            if str(args.score_calibration)
            else None
        ),
        "stage_c_pose_mode_support_count_is_diagnostic_only": True,
        "stage_c_proposal_source_policy": (
            "equal_independent_pool_per_source_then_common_radio_"
            "atlas_likelihood"
        ),
        "stage_c_heldout_identity_scope": (
            "fixed_query_ranked_for_candidate_score_then_bounded_"
            "pose_conditioned_d_optimal_subset_for_se3_refinement"
        ),
        "stage_c_candidate_ranking_score": (
            "local_displacement_marginal_bayes_factor_then_exact_"
            "zero_flow_pair_null_bayes_factor"
        ),
        "stage_c_prerank_selection": (
            "deduplicated_equal_budget_union_of_exact_zero_flow_and_"
            "local_displacement_marginal_rankings"
        ),
        "stage_c_missing_chart_score": (
            "neutral_log_bayes_factor_on_fixed_chart_denominator"
        ),
        "stage_c_minimum_rendered_chart_count": 4,
        "stage_c_absolute_view_direction_density_is_diagnostic_only": True,
        "stage_c_pose_pool_family_allocation": (
            "raw_deep64_then_broad4mode_union_consensus32_"
            "factorized32_default_pool256"
        ),
        "stage_c_raw_pose_family": (
            "leading_pair_supports_up_to_eight_modes_then_broad_"
            "pair_first_regional_supports_up_to_four_modes"
        ),
        "stage_c_minimum_regional_source_charts": 2,
        "stage_c_pose_mode_translation_radius_m": float(
            POSE_MODE_TRANSLATION_RADIUS_M
        ),
        "stage_c_pose_mode_rotation_radius_deg": float(
            POSE_MODE_ROTATION_RADIUS_DEG
        ),
        "stage_c_view_direction_angular_std_floor_deg": float(
            VIEW_DIRECTION_ANGULAR_STD_FLOOR_DEG
        ),
        "stage_c_candidate_conditioned_verification_score": (
            "fixed_chart_exact_pair_null_heldout_plus_0.25_fit_for_"
            "update_acceptance_only"
        ),
        "stage_c_update_evidence": (
            "fixed_chart_exact_zero_flow_pair_null_bayes_factor"
        ),
        "stage_c_se3_optimization_order": (
            "per_level_unified_joint_rotation_direct_fit_selection_"
            "then_translation_pyramid"
        ),
        "stage_c_minimum_mode_view_direction_cosine": float(
            MINIMUM_MODE_VIEW_DIRECTION_COSINE
        ),
        "stage_c_minimum_view_supported_cell_fraction": float(
            STAGE_C_MINIMUM_VIEW_SUPPORTED_CELL_FRACTION
        ),
        "rows": rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "image_id": image_id,
        "pool_count": len(pool),
        "selected_count": len(selected),
        "valid_count": len(rows),
        "top": rows[0] if rows else None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
