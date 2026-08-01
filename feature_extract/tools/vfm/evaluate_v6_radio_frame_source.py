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
    _aggregate_chart_expansion,
    _aggregate_m1,
    _aggregate_pose_rows,
    _aggregate_retrieval,
    _candidate_retrieved_charts,
    _chart_expansion_diagnostics,
    _chart_priors,
    _frame_row,
    _oracle_best_frame_modes,
    _pose_errors,
    _predicted_pose_hypotheses,
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
from feature_extract.vfm.localization.alike_detector_only import (
    AlikeDetectorOnly,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization.continuous_surface_alignment import (
    build_detector_heatmap,
    project_world_points,
)
from feature_extract.vfm.localization_v6.map_entities import (
    MetricSurfaceChartBank,
    RegionChartIndex,
    RetrievalRegionBank,
    compose_metric_region_atlas,
    merge_metric_atlases,
)
from feature_extract.vfm.localization_v6.atlas_pose_alignment import (
    AtlasAlignmentLevel,
    build_radio_final_feature_pyramid,
    refine_pose_with_maplet_atlases,
)
from feature_extract.vfm.localization_v6.atlas_renderer import (
    MINIMUM_MODE_VIEW_DIRECTION_COSINE,
    VIEW_DIRECTION_ANGULAR_STD_FLOOR_DEG,
    atlas_view_direction_log_likelihood,
    atlas_scene_geometry_points,
    render_scene_depth_from_points,
    render_selected_maplet_atlases_fast,
    render_sampled_maplet_atlases,
)
from feature_extract.vfm.localization_v6.heldout_verifier import (
    fixed_chart_local_match_log_bayes_factor,
    fixed_chart_zero_displacement_log_bayes_factor,
    zero_displacement_chart_log_likelihoods,
)
from feature_extract.vfm.localization_v6.local_correlation import (
    build_local_correlation_query_cache,
    local_correlation_distribution,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.maplet_frame_alignment import (
    FramePoseHypothesis,
    MapletFrameMatch,
    align_maplet_frame_global,
    factorized_pose_distribution_modes,
    frame_control_error_px,
    frame_log_likelihood_ratio,
    ground_truth_chart_frame,
    marginal_frame_log_likelihood_ratio,
    pose_distribution_consensus_modes,
    rescale_maplet_frame_match,
    refine_maplet_frame_matches,
    refine_maplet_frame_matches_with_chart_volume,
    refine_maplet_frame_matches_with_local_flow,
    refine_maplet_frame_matches_with_structured_refiner,
)
from feature_extract.vfm.localization_v6.probability_calibration import (
    V6ProbabilityCalibration,
)
from feature_extract.vfm.localization_v6.structured_frame_adapter import (
    load_structured_frame_adapter,
)
from feature_extract.vfm.localization_v6.structured_frame_refiner import (
    StructuredFrameRefiner,
    load_structured_frame_refiner,
)
from feature_extract.vfm.localization_v6.surface_spatial_projection import (
    SurfaceSpatialProjection,
    load_surface_spatial_projection,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


POSE_MODE_TRANSLATION_RADIUS_M = 0.85
POSE_MODE_ROTATION_RADIUS_DEG = 5.0
STAGE_C_VERIFICATION_RADIUS_CELLS = 2
STAGE_C_LOCAL_RADII_CELLS = (6, 4, 3)
ALIKE_MATCHABILITY_FLOOR = 0.20
STAGE_C_MINIMUM_VIEW_SUPPORTED_CELL_FRACTION = 0.50
_STRUCTURED_REFINER_FEATURE_SOURCES = frozenset(
    {"surface_maplet_mapper", "surface_spatial_projection"}
)


def _validate_structured_refiner_feature_source(
    feature_source_kind: str,
) -> None:
    """Reject transforms for which no typed refiner can be trained.

    A structured refiner is conditioned on the already transformed RADIO
    feature lattice.  It is therefore compatible with both supported learned
    transforms (the legacy surface mapper and the current spatial
    projection); requiring the mapper here incorrectly coupled inference to
    an implementation detail of the first refiner trainer.
    """

    if str(feature_source_kind) not in _STRUCTURED_REFINER_FEATURE_SOURCES:
        raise ValueError(
            "structured-frame refiner requires a learned surface feature "
            "transform (surface-maplet mapper or spatial projection)"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_regions", required=True)
    parser.add_argument("--region_chart_index", required=True)
    parser.add_argument(
        "--radio_atlas", "--radio_mapper_atlas", required=True
    )
    parser.add_argument(
        "--coarse_radio_atlas",
        default="",
        help=(
            "Native stride-16 atlas used only for global chart-mode search "
            "when --hierarchical_frame_alignment is explicitly enabled and "
            "--radio_atlas is a phase-matched finer atlas."
        ),
    )
    parser.add_argument(
        "--stage_c_radio_atlas",
        default="",
        help=(
            "Phase-matched RADIO-final atlas used only by rendered Stage-C "
            "SE(3) alignment. The ordinary --radio_atlas remains the native "
            "stride-16 chart atlas for retrieval-to-frame estimation."
        ),
    )
    parser.add_argument(
        "--hierarchical_frame_alignment",
        action="store_true",
        help=(
            "Diagnostic only: refine chart frames themselves on the finer "
            "phase lattice. The production path leaves this disabled and "
            "uses finer tokens only in Stage C."
        ),
    )
    parser.add_argument("--surface_mapper_checkpoint", default="")
    parser.add_argument(
        "--frame_spatial_projection_checkpoint",
        default="",
        help=(
            "Optional phase-preserving RADIO-final transform used only for "
            "chart alignment. Retrieval continues to use the surface mapper."
        ),
    )
    parser.add_argument("--structured_frame_adapter", default="")
    parser.add_argument("--structured_frame_refiner", default="")
    parser.add_argument("--query_projection_maplets", default="")
    parser.add_argument("--spatial_maplets", default="")
    parser.add_argument("--probability_calibration", default="")
    parser.add_argument("--query_contributor_dir", required=True)
    parser.add_argument(
        "--spatial_query_token_dir",
        default="",
        help=(
            "Optional true-phase RADIO-final tokens for frame/atlas "
            "alignment. Retrieval still uses the native contributor token."
        ),
    )
    parser.add_argument(
        "--spatial_feature_stride",
        type=int,
        choices=(8, 16),
        default=16,
    )
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--evaluation_role",
        choices=("strict", "calibration"),
        default="strict",
        help=(
            "strict enforces every trajectory-disjoint deployment gate; "
            "calibration permits labelled non-deployable replay on frozen "
            "training/calibration trajectories for downstream score fitting."
        ),
    )
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--query_shard_count", type=int, default=1)
    parser.add_argument("--query_shard_index", type=int, default=0)
    parser.add_argument("--m1_charts_per_query", type=int, default=2)
    parser.add_argument("--m1_regions_per_query", type=int, default=2)
    parser.add_argument("--m3_retrieval_regions", type=int, default=64)
    parser.add_argument(
        "--m3_frame_screen_charts",
        type=int,
        default=96,
        help=(
            "Expanded metric charts screened by phase-preserving RADIO "
            "frame evidence before expensive structured refinement."
        ),
    )
    parser.add_argument("--m3_charts", type=int, default=24)
    parser.add_argument(
        "--m3_pose_hypotheses",
        type=int,
        default=4096,
        help=(
            "Finite grouped SE(3) hypothesis budget. The production default "
            "preserves pair, triple, and quad frame-mode uncertainty until "
            "the sparse 2DGS atlas screen can compare the pose basins."
        ),
    )
    parser.add_argument("--m3_min_charts", type=int, default=12)
    parser.add_argument(
        "--m3_chart_probability_mass", type=float, default=0.95
    )
    parser.add_argument(
        "--support_score_power",
        type=float,
        default=0.0,
        help=(
            "Diagnostic regional-support exponent. Zero tests unbiased "
            "mean correlation across projected scales."
        ),
    )
    parser.add_argument(
        "--frame_refinement",
        choices=(
            "gradient",
            "local_flow",
            "hybrid",
            "structured",
            "gradient_structured",
            "structured_gradient",
            "structured_flow",
            "chart_volume",
        ),
        default="gradient",
        help=(
            "Chart-level continuous refinement. local_flow retains a local "
            "RADIO displacement posterior per atlas cell and robustly fits "
            "one projective chart factor; hybrid applies it after the "
            "fixed-denominator gradient baseline. structured applies the "
            "trajectory-disjoint complete-chart RADIO refiner."
            " chart_volume performs a learned-free robust structured search "
            "over the complete local correlation volume."
        ),
    )
    parser.add_argument("--run_stage_c", action="store_true")
    parser.add_argument(
        "--stage_c_pose_candidates", type=int, default=16
    )
    parser.add_argument(
        "--stage_c_prerank_pool", type=int, default=256
    )
    parser.add_argument(
        "--stage_c_broad_screen_candidates",
        type=int,
        default=64,
        help=(
            "Sparse rendered-atlas candidates retained before the exact "
            "area-rendered Stage-C likelihood. Set to zero to reproduce the "
            "legacy pre-compressed pool."
        ),
    )
    parser.add_argument(
        "--stage_c_broad_screen_radius_cells", type=int, default=6
    )
    parser.add_argument(
        "--stage_c_broad_screen_maximum_points", type=int, default=192
    )
    parser.add_argument(
        "--stage_c_render_charts",
        type=int,
        default=24,
        help=(
            "Fixed query-ranked chart identity set used to compare pose "
            "candidates. It should cover every chart allowed to generate "
            "the Stage-B distribution."
        ),
    )
    parser.add_argument(
        "--stage_c_refinement_charts",
        type=int,
        default=12,
        help=(
            "Pose-conditioned, geometry-conditioned subset of the fixed "
            "Stage-C scoring identities used for SE(3) optimization. "
            "Candidate comparison still uses every stage_c_render_chart."
        ),
    )
    parser.add_argument("--stage_c_rounds", type=int, default=3)
    parser.add_argument(
        "--stage_c_maximum_translation_updates",
        type=int,
        default=2,
        help=(
            "Maximum accepted translation-bearing SE(3) updates across "
            "the coarse-to-fine atlas trust region."
        ),
    )
    parser.add_argument(
        "--alike_detector_matchability",
        action="store_true",
        help=(
            "Use query-side ALIKE detection scores to seed extra local search "
            "proposals and as an offset-neutral cell reliability signal. "
            "Every proposal is scored and optimized by RADIO-only regional "
            "likelihood; no ALIKE descriptor is computed, stored, or matched."
        ),
    )
    parser.add_argument(
        "--alike_detector_global_proposal",
        action="store_true",
        help=(
            "Use the ALIKE detector heatmap only to generate an additional "
            "global chart-pose proposal branch. Its scores are proposal-local "
            "and must be compared with other branches by Stage-C RADIO atlas "
            "likelihood, never fused as a calibrated match probability."
        ),
    )
    parser.add_argument(
        "--alike_detector_local_proposal_branch",
        action="store_true",
        help=(
            "Preserve a detector-seeded local chart proposal branch until "
            "multi-chart/Stage-C RADIO comparison. ALIKE selects only the "
            "finite initialization basin; optimization and scores are RADIO."
        ),
    )
    parser.add_argument("--alike_model", default="alike-t")
    parser.add_argument("--alike_detector_top_k", type=int, default=512)
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _refine_frame_modes(
    atlas: MapletFeatureAtlasBank,
    matches: Sequence[MapletFrameMatch],
    query_feature: torch.Tensor,
    mode: str,
    structured_refiner: StructuredFrameRefiner | None = None,
    structured_refiner_metadata: Mapping[str, object] | None = None,
    query_matchability: torch.Tensor | None = None,
    detector_guided_proposal_branch: bool = False,
) -> tuple[MapletFrameMatch, ...]:
    name = str(mode)
    values = tuple(matches)
    if name in {"gradient", "hybrid", "gradient_structured"}:
        values = refine_maplet_frame_matches(
            atlas,
            values,
            query_feature,
            query_matchability,
            iterations=20,
        )
    if name in {"local_flow", "hybrid"}:
        values = refine_maplet_frame_matches_with_local_flow(
            atlas,
            values,
            query_feature,
            rounds=2,
            radius_cells=3,
        )
    if name == "chart_volume":
        values = refine_maplet_frame_matches_with_chart_volume(
            atlas,
            values,
            query_feature,
            query_matchability=query_matchability,
            radius_cells=4,
            detector_guided_proposal_branch=bool(
                detector_guided_proposal_branch
            ),
        )
    if name in {"structured", "gradient_structured"}:
        if structured_refiner is None:
            raise ValueError(
                "structured refinement requires a refiner checkpoint"
            )
        metadata = dict(structured_refiner_metadata or {})
        values = refine_maplet_frame_matches_with_structured_refiner(
            atlas,
            values,
            query_feature,
            structured_refiner,
            training_positive_fraction=float(
                metadata.get("training_positive_fraction", 0.75)
            ),
        )
    if name in {"structured_gradient", "structured_flow"}:
        if structured_refiner is None:
            raise ValueError(
                "structured refinement requires a refiner checkpoint"
            )
        metadata = dict(structured_refiner_metadata or {})
        values = refine_maplet_frame_matches_with_structured_refiner(
            atlas,
            values,
            query_feature,
            structured_refiner,
            training_positive_fraction=float(
                metadata.get("training_positive_fraction", 0.75)
            ),
        )
        if name == "structured_gradient":
            values = refine_maplet_frame_matches(
                atlas,
                values,
                query_feature,
                query_matchability,
                iterations=20,
            )
        else:
            values = refine_maplet_frame_matches_with_local_flow(
                atlas,
                values,
                query_feature,
                rounds=2,
                radius_cells=3,
                update_model="homography",
            )
    return values


def _alike_matchability_map(
    xy: np.ndarray,
    scores: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    feature_width: int,
    feature_height: int,
    feature_stride: int,
) -> np.ndarray:
    """Convert detector-only points into a soft query importance field.

    The non-zero floor is intentional: ALIKE may emphasize distinctive
    locations but may not veto RADIO evidence or create point identities.
    """

    heat = build_detector_heatmap(
        xy,
        scores,
        image_width=int(image_width),
        image_height=int(image_height),
        output_width=int(feature_width),
        output_height=int(feature_height),
        sigma_px=max(4.0, 0.75 * float(feature_stride)),
    )
    return np.clip(
        float(ALIKE_MATCHABILITY_FLOOR)
        + (1.0 - float(ALIKE_MATCHABILITY_FLOOR)) * heat,
        float(ALIKE_MATCHABILITY_FLOOR),
        1.0,
    ).astype(np.float32, copy=False)


def _matchability_pyramid(
    matchability: np.ndarray | None,
    query_pyramid: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray | None]:
    """Resample one query-only detector prior onto Stage-C feature grids."""

    if matchability is None:
        return {name: None for name in query_pyramid}
    source = torch.from_numpy(
        np.asarray(matchability, dtype=np.float32)
    )[None, None]
    result: dict[str, np.ndarray | None] = {}
    for name, feature in query_pyramid.items():
        target_shape = tuple(np.asarray(feature).shape[-2:])
        if tuple(source.shape[-2:]) == target_shape:
            value = source
        else:
            value = torch.nn.functional.interpolate(
                source,
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )
        result[name] = np.clip(
            value[0, 0].numpy(),
            float(ALIKE_MATCHABILITY_FLOOR),
            1.0,
        ).astype(np.float32, copy=False)
    return result


@torch.no_grad()
def _project_surface_spatial_feature_map(
    model: SurfaceSpatialProjection,
    radio_final: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Apply the origin-preserving spatial head without changing phase."""

    raw = torch.from_numpy(
        np.asarray(radio_final, dtype=np.float32)
    ).to(device=str(device))
    if raw.ndim != 3:
        raise ValueError("RADIO-final feature map must have shape (C,H,W)")
    projected = model(raw.permute(1, 2, 0)).permute(2, 0, 1)
    return projected.detach().cpu().numpy().astype(np.float32)


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
    retrieval_mapper_metadata: Mapping[str, object] | None = None,
    enforce_strict_disjoint: bool = True,
) -> dict[str, object]:
    metadata = dict(atlas.metadata or {})
    transform = str(metadata.get("query_feature_transform", ""))
    expected_transform = {
        "surface_maplet_mapper": "surface_maplet_mapper",
        "raw_radio_pca": "raw_radio_pca",
        "surface_spatial_projection": "surface_spatial_projection",
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
    elif feature_source_kind == "surface_spatial_projection":
        observed_transform_sha256 = _sha256(
            paths["frame_spatial_projection_checkpoint"]
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
    mapper_reference = set(
        _trajectory_ids_from_mapper(
            retrieval_mapper_metadata
            if retrieval_mapper_metadata is not None
            else feature_source_metadata
        )
    )
    spatial_projection_fit = set()
    if feature_source_kind == "surface_spatial_projection":
        for key in (
            "reference_trajectory_ids",
            "train_query_trajectory_ids",
            "validation_trajectory_ids",
        ):
            spatial_projection_fit.update(
                str(value)
                for value in feature_source_metadata.get(key, [])
            )
    query = {str(value) for value in query_trajectories}
    if query & mapping:
        raise ValueError("strict queries overlap RADIO atlas mapping views")
    if enforce_strict_disjoint and query & mapper_reference:
        raise ValueError("strict queries overlap RADIO mapper references")
    if enforce_strict_disjoint and query & spatial_projection_fit:
        raise ValueError(
            "strict queries overlap spatial-projection fit or selection"
        )
    declared_holdout = {
        str(value)
        for value in feature_source_metadata.get(
            "strict_holdout_trajectory_ids", []
        )
    }
    if (
        enforce_strict_disjoint
        and declared_holdout
        and not query <= declared_holdout
    ):
        raise ValueError(
            "queries are absent from spatial-projection strict holdout"
        )
    return {
        "atlas_mapping": sorted(mapping),
        "radio_mapper_reference": sorted(mapper_reference),
        "spatial_projection_fit_or_selection": sorted(
            spatial_projection_fit
        ),
        "strict_test": sorted(query),
        "strict_test_disjoint_from_atlas_mapping": True,
        "strict_test_disjoint_from_radio_mapper_reference": not bool(
            query & mapper_reference
        ),
        "strict_test_disjoint_from_spatial_projection_fit": not bool(
            query & spatial_projection_fit
        ),
        "query_overlap_with_radio_mapper_reference": sorted(
            query & mapper_reference
        ),
        "query_overlap_with_spatial_projection_fit": sorted(
            query & spatial_projection_fit
        ),
        "evaluation_role": (
            "strict" if enforce_strict_disjoint else "calibration"
        ),
        "feature_source_kind": str(feature_source_kind),
        "clean_geometry_source_sha256": str(
            metadata.get("clean_geometry_source_sha256", "")
        ),
    }


def _validate_shared_atlas_geometry(
    coarse: MapletFeatureAtlasBank,
    fine: MapletFeatureAtlasBank,
) -> None:
    """Reject feature-pyramid levels that do not share one 2DGS chart."""

    for name in (
        "maplet_ids",
        "centers",
        "frames",
        "extents",
        "primitive_ids",
        "xyz",
    ):
        if not np.array_equal(
            np.asarray(getattr(coarse, name)),
            np.asarray(getattr(fine, name)),
        ):
            raise ValueError(
                f"coarse/fine RADIO atlases differ in geometry: {name}"
            )
    coarse_metadata = dict(coarse.metadata or {})
    fine_metadata = dict(fine.metadata or {})
    for key in (
        "geometry_source_sha256",
        "clean_geometry_source_sha256",
        "clean_source_index_sha256",
        "query_feature_transform",
        "query_feature_transform_sha256",
    ):
        if str(coarse_metadata.get(key, "")) != str(
            fine_metadata.get(key, "")
        ):
            raise ValueError(
                f"coarse/fine RADIO atlas lineage differs: {key}"
            )


def _pose_mode_distance(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64).reshape(4, 4)
    second = np.asarray(second, dtype=np.float64).reshape(4, 4)
    first_center = -first[:3, :3].T @ first[:3, 3]
    second_center = -second[:3, :3].T @ second[:3, 3]
    translation = float(np.linalg.norm(first_center - second_center))
    relative = first[:3, :3] @ second[:3, :3].T
    rotation = float(
        np.degrees(
            np.arccos(
                np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
            )
        )
    )
    return translation, rotation


def _diverse_pose_hypotheses(
    hypotheses: Sequence[object],
    limit: int,
    *,
    minimum_source_charts: int = 1,
) -> list[object]:
    """Cover chart-support modes before applying pose-space suppression.

    ``hypotheses`` is ordered by a frame score whose absolute scale depends on
    the number of charts and on the feature transform.  Taking a pose-diverse
    prefix therefore still discards entire chart-support hypotheses before
    the independent rendered-atlas likelihood can inspect them.  First expose
    the best hypothesis for every unordered source-chart set, then use the
    remaining score order to retain alternate modes.
    """

    maximum = max(int(limit), 0)
    if maximum == 0:
        return []
    support_representatives: dict[int, list[object]] = {}
    remaining: dict[int, list[object]] = {}
    seen_supports: set[tuple[int, ...]] = set()
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
        cardinality = len(support)
        if support not in seen_supports:
            support_representatives.setdefault(
                cardinality, []
            ).append(hypothesis)
            seen_supports.add(support)
        else:
            remaining.setdefault(cardinality, []).append(hypothesis)

    def rank(values: Sequence[object]) -> list[object]:
        return sorted(
            values,
            key=lambda value: (
                -float(getattr(value, "score", -np.inf)),
                float(
                    getattr(
                        value, "reprojection_error_px", np.inf
                    )
                ),
            ),
        )

    def interleave(
        groups: Mapping[int, Sequence[object]],
        cardinalities: Sequence[int],
    ) -> list[object]:
        queues = {
            int(cardinality): rank(groups.get(int(cardinality), ()))
            for cardinality in cardinalities
        }
        result = []
        offset = 0
        while any(offset < len(values) for values in queues.values()):
            for cardinality in cardinalities:
                values = queues[int(cardinality)]
                if offset < len(values):
                    result.append(values[offset])
            offset += 1
        return result

    cardinalities = sorted(
        set(support_representatives) | set(remaining)
    )
    minimum = max(int(minimum_source_charts), 1)
    primary = [
        value for value in cardinalities if int(value) >= minimum
    ]
    fallback = [
        value for value in cardinalities if int(value) < minimum
    ]
    # Stage C needs a genuinely regional pose seed.  A single planar chart is
    # intrinsically ambiguous, while two independently retrieved charts are
    # already the minimum regional metric constraint used by the chart-factor
    # solver.  Cover every two/three/four-chart support before admitting
    # alternate solver modes and single-chart fallbacks.
    ordered = [
        *interleave(support_representatives, primary),
        *interleave(remaining, primary),
        *interleave(support_representatives, fallback),
        *interleave(remaining, fallback),
    ]
    selected = []
    for hypothesis in ordered:
        if any(
            (
                lambda value: value[0] < 0.20 and value[1] < 5.0
            )(
                _pose_mode_distance(
                    hypothesis.pose_w2c, retained.pose_w2c
                )
            )
            for retained in selected
        ):
            continue
        selected.append(hypothesis)
        if len(selected) >= maximum:
            break
    return selected


def _multimodal_regional_pose_hypotheses(
    hypotheses: Sequence[object],
    limit: int,
    *,
    minimum_source_charts: int = 2,
    maximum_modes_per_support: int = 8,
    prioritize_pair_supports: bool = False,
) -> list[object]:
    """Keep complete modes inside the strongest regional supports.

    Raw frame hypotheses and derived consensus hypotheses serve different
    purposes.  Applying support coverage to both families spent the raw quota
    on one pose from many supports and deleted the lower-ranked planar branch
    of the strongest support before atlas likelihood could inspect it.  Rank
    supports independently for each 2/3/4-chart cardinality, interleave those
    cardinalities, and retain a finite block of modes per selected support.
    Consensus/factorized families provide the complementary broad coverage.
    """

    maximum = max(int(limit), 0)
    per_support = max(int(maximum_modes_per_support), 1)
    minimum = max(int(minimum_source_charts), 1)
    if maximum == 0:
        return []
    grouped: dict[tuple[int, ...], list[object]] = {}
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
        if len(support) < minimum:
            continue
        grouped.setdefault(support, []).append(hypothesis)

    def hypothesis_key(value: object) -> tuple[float, float]:
        return (
            -float(getattr(value, "score", -np.inf)),
            float(getattr(value, "reprojection_error_px", np.inf)),
        )

    by_cardinality: dict[int, list[tuple[int, ...]]] = {}
    for support, values in grouped.items():
        values.sort(key=hypothesis_key)
        by_cardinality.setdefault(len(support), []).append(support)
    for supports in by_cardinality.values():
        supports.sort(key=lambda support: hypothesis_key(grouped[support][0]))

    cardinalities = sorted(by_cardinality)
    if bool(prioritize_pair_supports):
        # Stage B allocates complete pair coverage because a pair is the
        # smallest non-degenerate regional factor and carries the deepest
        # planar branch ambiguity. Consensus/factorized families separately
        # cover multi-chart structure, so Stage C must not spend its raw
        # budget interleaving triples/quads before broad pair supports.
        support_order = [
            support
            for cardinality in cardinalities
            for support in by_cardinality[cardinality]
        ]
    else:
        support_order = []
        offset = 0
        while any(
            offset < len(by_cardinality[cardinality])
            for cardinality in cardinalities
        ):
            for cardinality in cardinalities:
                values = by_cardinality[cardinality]
                if offset < len(values):
                    support_order.append(values[offset])
            offset += 1

    selected = []
    selected_identities: set[int] = set()
    for support in support_order:
        for hypothesis in grouped[support][:per_support]:
            selected.append(hypothesis)
            selected_identities.add(id(hypothesis))
            if len(selected) >= maximum:
                return selected
    # Sparse supports can leave the finite quota short.  Fill with the broad
    # support-balanced order rather than silently shrinking Stage C.
    for hypothesis in _diverse_pose_hypotheses(
        hypotheses,
        maximum,
        minimum_source_charts=minimum,
    ):
        if id(hypothesis) in selected_identities:
            continue
        selected.append(hypothesis)
        selected_identities.add(id(hypothesis))
        if len(selected) >= maximum:
            break
    return selected


def _stage_c_pose_pool(
    raw_hypotheses: Sequence[object],
    consensus_modes: Sequence[object],
    factorized_modes: Sequence[object],
    limit: int,
    *,
    minimum_source_charts: int = 2,
) -> list[object]:
    """Expose raw, consensus and factorized pose families independently.

    Consensus modes are already support-deduplicated SE(3) probability modes;
    factorized modes preserve the distinct centre/rotation uncertainty of a
    near-planar solution.  Mixing both into one support-cardinality queue made
    factorized proposals consume the consensus quota and, on the strict St
    Mary's replay, removed valid consensus basins that were present in the
    stored Stage-B distribution.

    Stage B deliberately spends its finite budget on both deep planar modes
    of strong chart supports and broad pair-support coverage. Compressing
    those 1024 candidates to 64 raw rows silently undid that design: q9's
    valid basin is the eighth mode of a strong support, while q2's is the
    fourth mode of a broader support. The production pool therefore reserves
    32 consensus and 32 factorized modes and gives the remaining budget to a
    deterministic union of (a) eight-mode depth for the leading supports and
    (b) four-mode breadth across pair supports before larger supports. At the
    default size 256 this is 192 raw + 32 consensus + 32 factorized
    candidates.

    Cross-family pose suppression is intentionally not applied: a factorized
    mode may reuse a raw camera centre with a better consensus rotation, and
    the fixed-support atlas likelihood must compare those alternatives.
    """

    maximum = max(int(limit), 0)
    if maximum == 0:
        return []
    raw = list(raw_hypotheses)
    consensus = list(consensus_modes)
    factorized = list(factorized_modes)
    if maximum < 128:
        raw_budget = maximum // 2
        consensus_budget = min(maximum // 3, len(consensus))
        factorized_budget = min(
            maximum - raw_budget - consensus_budget,
            len(factorized),
        )
    else:
        consensus_budget = min(32, len(consensus))
        factorized_budget = min(32, len(factorized))
        raw_budget = max(
            maximum - consensus_budget - factorized_budget, 0
        )
    deep_raw = _multimodal_regional_pose_hypotheses(
        raw,
        maximum,
        minimum_source_charts=minimum_source_charts,
        maximum_modes_per_support=8,
        prioritize_pair_supports=True,
    )
    broad_raw = _multimodal_regional_pose_hypotheses(
        raw,
        maximum,
        minimum_source_charts=minimum_source_charts,
        maximum_modes_per_support=4,
        prioritize_pair_supports=True,
    )
    raw_order = []
    raw_seen: set[int] = set()

    def append_raw(values: Sequence[object], limit_count: int) -> None:
        for hypothesis in values:
            if len(raw_order) >= int(limit_count):
                return
            identity = id(hypothesis)
            if identity in raw_seen:
                continue
            raw_order.append(hypothesis)
            raw_seen.add(identity)

    # Preserve deep ambiguity for a small leading set, then cover many more
    # supports before spending the tail on additional deep modes.
    append_raw(deep_raw, min(64, maximum))
    append_raw(broad_raw, maximum)
    append_raw(deep_raw, maximum)
    # Both derived constructors already suppress pose duplicates and return a
    # probability-mode order.  Re-running support-set interleaving here is the
    # bug this split is designed to remove.
    selected = [
        *consensus[:consensus_budget],
        *factorized[:factorized_budget],
        *raw_order[:raw_budget],
    ]

    # A short/empty family must not reduce the finite pool. Fill from raw
    # observations first, then the remaining consensus and factorized modes.
    if len(selected) < maximum:
        seen = {id(value) for value in selected}
        fallback = [
            *raw_order,
            *consensus,
            *factorized,
        ]
        for hypothesis in fallback:
            if id(hypothesis) in seen:
                continue
            selected.append(hypothesis)
            seen.add(id(hypothesis))
            if len(selected) >= maximum:
                break
    return selected


def _complete_regional_pose_distribution(
    raw_hypotheses: Sequence[object],
    consensus_modes: Sequence[object],
    factorized_modes: Sequence[object],
) -> list[object]:
    """Expose every regional mode to the sparse rendered-atlas screen.

    Single-chart IPPE branches are useful diagnostics but are not a regional
    localization seed.  Everything supported by at least two chart identities
    must survive until map feature evidence, rather than a fixed CPU prefix,
    performs the reduction.
    """

    regional = [
        value
        for value in raw_hypotheses
        if len(
            {
                int(chart_id)
                for chart_id in getattr(value, "source_chart_ids", ())
            }
        )
        >= 2
    ]
    return [*consensus_modes, *factorized_modes, *regional]


def _source_and_heldout_chart_ids(
    chart_ids: np.ndarray,
    source_chart_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep sources in fit and reserve independent chart components.

    Source-only fitting is geometrically degenerate when a pose seed came
    from several coplanar facade charts. Query-retrieved, pose-visible charts
    may also participate in the update, while a deterministic third of the
    non-source charts remains completely held out for acceptance and ranking.
    """

    rendered = np.unique(np.asarray(chart_ids, dtype=np.int64))
    source = np.intersect1d(
        rendered,
        np.asarray(source_chart_ids, dtype=np.int64),
        assume_unique=False,
    )
    independent = np.setdiff1d(rendered, source, assume_unique=True)
    if independent.size == 0:
        return source, independent
    heldout_count = min(
        max(int(round(0.33 * rendered.size)), 2),
        int(independent.size),
    )
    scored = []
    for chart_id in independent.tolist():
        digest = hashlib.sha256(
            f"v6-stage-c-heldout:{int(chart_id)}".encode()
        ).digest()
        scored.append(
            (int.from_bytes(digest[:8], "little"), int(chart_id))
        )
    heldout = np.asarray(
        [value for _score, value in sorted(scored)[:heldout_count]],
        dtype=np.int64,
    )
    fit = np.setdiff1d(rendered, heldout, assume_unique=True)
    return fit, heldout


def _marginal_frame_log_evidence(
    matches: Sequence[MapletFrameMatch],
) -> float:
    """Marginalize mutually exclusive frame modes against the chart null."""

    return marginal_frame_log_likelihood_ratio(matches)


def _serialized_frame_mode_rows(
    matches_by_chart: Mapping[int, Sequence[MapletFrameMatch]],
) -> list[dict[str, object]]:
    """Save the finite structured distribution for deterministic replay.

    These are query-side transform sufficient statistics, not RGB, mapping
    image identities, local descriptors, point correspondences or GT labels.
    Keeping them makes support-generation and probability bugs reproducible
    without rerunning the expensive RADIO chart correlation stage.
    """

    result = []
    for chart_id, matches in matches_by_chart.items():
        modes = []
        for match in matches:
            modes.append(
                {
                    "canonical_to_query": np.asarray(
                        match.canonical_to_query, dtype=np.float64
                    ).tolist(),
                    "canonical_homography": (
                        None
                        if match.canonical_homography is None
                        else np.asarray(
                            match.canonical_homography, dtype=np.float64
                        ).tolist()
                    ),
                    "query_center_xy": np.asarray(
                        match.query_center_xy, dtype=np.float64
                    ).tolist(),
                    "scale_xy": np.asarray(
                        match.scale_xy, dtype=np.float64
                    ).tolist(),
                    "in_plane_rotation_deg": float(
                        match.in_plane_rotation_deg
                    ),
                    "covariance_xy": np.asarray(
                        match.covariance_xy, dtype=np.float64
                    ).tolist(),
                    "control_covariance_px": (
                        None
                        if match.control_covariance_px is None
                        else np.asarray(
                            match.control_covariance_px, dtype=np.float64
                        ).tolist()
                    ),
                    "support_canonical_hull": (
                        None
                        if match.support_canonical_hull is None
                        else np.asarray(
                            match.support_canonical_hull, dtype=np.float64
                        ).tolist()
                    ),
                    "score": float(match.score),
                    "probability": float(match.probability),
                    "null_probability": float(match.null_probability),
                    "identity_probability": float(
                        match.identity_probability
                    ),
                    "support_fraction": float(match.support_fraction),
                    "feature_level": str(match.feature_level),
                    "feature_stride": int(match.feature_stride),
                }
            )
        result.append(
            {
                "chart_id": int(chart_id),
                "marginal_log_evidence": float(
                    _marginal_frame_log_evidence(matches)
                ),
                "modes": modes,
            }
        )
    return result


def _stage_c_pose_log_score(
    atlas_score: float, hypothesis: object
) -> float:
    """Return the candidate-comparable Stage-C pose likelihood.

    ``mode_support_count`` counts overlapping solver/support hypotheses, not
    independent observations and not calibrated posterior mass.  Adding its
    logarithm made a broad wrong consensus outrank a pose with stronger
    held-out atlas evidence.  Keep it as a diagnostic only; the final ranking
    is the fixed-support atlas likelihood.
    """

    del hypothesis
    return float(atlas_score)


def _geometry_conditioned_chart_subset(
    atlas: MapletFeatureAtlasBank,
    candidate_chart_ids: Sequence[int],
    source_chart_ids: Sequence[int],
    pose_w2c: np.ndarray,
    *,
    limit: int,
) -> np.ndarray:
    """Select query-relevant charts that jointly observe all SE(3) axes.

    Query correlation determines the finite candidate order.  Inside its
    leading 2x budget, a deterministic D-optimal design avoids spending all
    atlas slots on the same facade/depth band.  The Jacobian is evaluated at
    chart centres and scaled to comparable 3 degree / 30 cm perturbations.
    No image, point identity, GT pose, or descriptor is used here.
    """

    maximum = max(int(limit), 0)
    if maximum == 0:
        return np.zeros((0,), dtype=np.int64)
    row_by_id = {
        int(value): row
        for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    ordered = []
    for value in candidate_chart_ids:
        chart_id = int(value)
        if chart_id in row_by_id and chart_id not in ordered:
            ordered.append(chart_id)
    pool = ordered[: max(maximum * 2, maximum)]
    if len(pool) <= maximum:
        return np.asarray(pool, dtype=np.int64)

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rank_by_id = {value: rank for rank, value in enumerate(pool)}

    def scaled_jacobian(chart_id: int) -> np.ndarray | None:
        center = np.asarray(
            atlas.centers[row_by_id[int(chart_id)]], dtype=np.float64
        )
        camera_xyz = pose[:3, :3] @ center + pose[:3, 3]
        x, y, z = camera_xyz.tolist()
        if not np.isfinite(camera_xyz).all() or float(z) <= 1e-4:
            return None
        projection = np.asarray(
            [[1.0 / z, 0.0, -x / (z * z)],
             [0.0, 1.0 / z, -y / (z * z)]],
            dtype=np.float64,
        )
        skew = np.asarray(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )
        jacobian = projection @ np.concatenate(
            [-skew, np.eye(3, dtype=np.float64)], axis=1
        )
        jacobian[:, :3] *= np.deg2rad(3.0)
        jacobian[:, 3:] *= 0.30
        relevance = np.exp(
            -float(rank_by_id[int(chart_id)])
            / max(2.0 * maximum, 1.0)
        )
        return jacobian * np.sqrt(relevance)

    selected = []
    information = np.eye(6, dtype=np.float64) * 1e-3
    # Retain two pose-generating identities when they remain view-supported;
    # the rest of the budget is selected for independent geometry.
    for value in source_chart_ids:
        chart_id = int(value)
        if chart_id not in pool or chart_id in selected:
            continue
        jacobian = scaled_jacobian(chart_id)
        if jacobian is None:
            continue
        selected.append(chart_id)
        information += jacobian.T @ jacobian
        if len(selected) >= min(2, maximum):
            break

    while len(selected) < maximum:
        candidates = []
        baseline = float(np.linalg.slogdet(information)[1])
        for chart_id in pool:
            if chart_id in selected:
                continue
            jacobian = scaled_jacobian(chart_id)
            if jacobian is None:
                continue
            updated = information + jacobian.T @ jacobian
            gain = float(np.linalg.slogdet(updated)[1] - baseline)
            candidates.append(
                (
                    gain,
                    -int(rank_by_id[chart_id]),
                    chart_id,
                    jacobian,
                )
            )
        if not candidates:
            break
        _gain, _negative_rank, chart_id, jacobian = max(
            candidates, key=lambda value: value[:3]
        )
        selected.append(int(chart_id))
        information += jacobian.T @ jacobian
    return np.asarray(selected, dtype=np.int64)


def _pose_visible_render_charts(
    atlas: MapletFeatureAtlasBank,
    ranked_chart_ids: np.ndarray,
    source_chart_ids: Sequence[int],
    pose_w2c: np.ndarray,
    camera: object,
    *,
    limit: int,
    compatible_chart_ids: np.ndarray | None = None,
) -> np.ndarray:
    row_by_id = {
        int(value): row
        for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    # The region graph is an identity expansion graph, not a visibility
    # graph.  Restricting rendering to it left the real candidate with five
    # almost coplanar charts even though many query-retrieved, geometrically
    # independent charts were visible.  Evaluate the globally ranked query
    # identities first and use graph neighbours only as unranked fallbacks.
    fallback = (
        []
        if compatible_chart_ids is None
        else np.asarray(
            compatible_chart_ids, dtype=np.int64
        ).reshape(-1).tolist()
    )
    candidate_order = [
        *source_chart_ids,
        *np.asarray(ranked_chart_ids, dtype=np.int64).reshape(-1).tolist(),
        *fallback,
    ]
    candidates = []
    for value in candidate_order:
        chart_id = int(value)
        if chart_id in row_by_id and chart_id not in candidates:
            candidates.append(chart_id)
    rows = np.asarray(
        [row_by_id[value] for value in candidates], dtype=np.int64
    )
    if rows.size:
        pixels, depth = project_world_points(
            atlas.centers[rows], pose_w2c, camera
        )
        visible = (
            np.isfinite(pixels).all(axis=1)
            & np.isfinite(depth)
            & (depth > 0.0)
            & (pixels[:, 0] >= -0.10 * camera.width)
            & (pixels[:, 0] < 1.10 * camera.width)
            & (pixels[:, 1] >= -0.10 * camera.height)
            & (pixels[:, 1] < 1.10 * camera.height)
        )
        if atlas.mode_features is not None:
            camera_center = (
                -np.asarray(pose_w2c[:3, :3], dtype=np.float64).T
                @ np.asarray(pose_w2c[:3, 3], dtype=np.float64)
            )
            view_supported = []
            for row in rows.tolist():
                direction = (
                    camera_center[None, None]
                    - np.asarray(atlas.xyz[row], dtype=np.float64)
                )
                direction /= np.maximum(
                    np.linalg.norm(direction, axis=2, keepdims=True),
                    1e-8,
                )
                cosine = np.einsum(
                    "khwc,hwc->khw",
                    np.asarray(
                        atlas.mode_view_directions[row],
                        dtype=np.float64,
                    ),
                    direction,
                )
                mode_valid = np.asarray(
                    atlas.mode_valid_mask[row], dtype=bool
                )
                cosine = np.where(mode_valid, cosine, -np.inf)
                cells = (
                    np.asarray(atlas.valid_mask[row], dtype=bool)
                    & np.any(mode_valid, axis=0)
                )
                fraction = (
                    float(
                        np.mean(
                            np.max(cosine, axis=0)[cells]
                            >= float(
                                MINIMUM_MODE_VIEW_DIRECTION_COSINE
                            )
                        )
                    )
                    if np.any(cells)
                    else 0.0
                )
                view_supported.append(
                    fraction
                    >= float(
                        STAGE_C_MINIMUM_VIEW_SUPPORTED_CELL_FRACTION
                    )
                )
            visible &= np.asarray(view_supported, dtype=bool)
        candidates = [
            value
            for value, keep in zip(candidates, visible.tolist())
            if keep
        ]
    supported = set(candidates)
    query_rank_by_id = {
        int(value): rank
        for rank, value in enumerate(
            np.asarray(ranked_chart_ids, dtype=np.int64).reshape(-1).tolist()
        )
    }
    query_rank_limit = max(2 * int(limit), int(limit))
    preferred = [
        value
        for value in candidates
        if value in source_chart_ids
        or query_rank_by_id.get(int(value), query_rank_limit)
        < query_rank_limit
    ]
    if len(preferred) < int(limit):
        preferred.extend(
            value for value in candidates if value not in preferred
        )
    ordered = []
    for value in [
        *[
            int(source)
            for source in source_chart_ids
            if int(source) in supported
        ],
        *preferred,
    ]:
        if int(value) in row_by_id and int(value) not in ordered:
            ordered.append(int(value))
    return _geometry_conditioned_chart_subset(
        atlas,
        ordered,
        source_chart_ids,
        pose_w2c,
        limit=int(limit),
    )


def _stage_c_broad_screen(
    hypotheses: Sequence[object],
    atlas: MapletFeatureAtlasBank,
    comparison_chart_ids: np.ndarray,
    query_feature: np.ndarray,
    view: object,
    scene_geometry: np.ndarray,
    *,
    limit: int,
    radius_cells: int,
    maximum_points: int,
    device: str,
    audit: dict[str, object] | None = None,
) -> list[object]:
    """Cheap RADIO-atlas screen before the full candidate likelihood.

    Stage B already contains an explicit finite coverage policy. Reapplying a
    frame-score prefix before any rendered-map observation deleted valid
    low-ranked pair supports. This screen sees every supplied pose using the
    same fixed chart identities, but reduces correlation radius and raster
    samples. Exact zero-flow and displacement-marginal evidence receive equal
    budgets, so neither a nearly converged mode nor a valid coarse basin can
    monopolize the screen.
    """

    pool = list(hypotheses)
    maximum = max(int(limit), 0)
    if maximum == 0 or len(pool) <= maximum:
        if audit is not None:
            audit["broad_screen_score_rows"] = []
        return pool
    radius = max(int(radius_cells), 1)
    point_limit = max(int(maximum_points), 32)
    query_cache = build_local_correlation_query_cache(
        query_feature,
        radius=radius,
        background_samples=128,
        device=str(device),
    )
    scored = []
    for source_rank, hypothesis in enumerate(pool):
        rendered = render_sampled_maplet_atlases(
            atlas,
            comparison_chart_ids,
            hypothesis.pose_w2c,
            view.camera,
            width=int(query_feature.shape[2]),
            height=int(query_feature.shape[1]),
            maximum_samples=point_limit,
        )
        correlation = local_correlation_distribution(
            rendered,
            query_feature,
            radius=radius,
            # ALIKE is deliberately absent from candidate likelihood. It may
            # generate a proposal branch, but RADIO alone compares branches.
            query_matchability=None,
            maximum_points=point_limit,
            background_samples=128,
            device=str(device),
            query_cache=query_cache,
        )
        rendered_chart_ids = set(
            np.asarray(correlation.maplet_ids, dtype=np.int64).tolist()
        )
        if len(rendered_chart_ids) < 4:
            continue
        exact = fixed_chart_zero_displacement_log_bayes_factor(
            correlation, comparison_chart_ids
        )
        local = fixed_chart_local_match_log_bayes_factor(
            correlation, comparison_chart_ids
        )
        if not np.isfinite(exact) and not np.isfinite(local):
            continue
        scored.append(
            (
                float(exact),
                float(local),
                int(source_rank),
                hypothesis,
            )
        )
    exact_ranked = sorted(
        scored, key=lambda value: (-value[0], -value[1], value[2])
    )
    local_ranked = sorted(
        scored, key=lambda value: (-value[1], -value[0], value[2])
    )
    exact_budget = (maximum + 1) // 2
    local_budget = maximum - exact_budget
    selected = []
    selected_identity: set[int] = set()

    def take(values: Sequence[tuple[object, ...]], budget: int) -> None:
        added = 0
        for value in values:
            if added >= int(budget):
                return
            identity = id(value[3])
            if identity in selected_identity:
                continue
            selected.append(value[3])
            selected_identity.add(identity)
            added += 1

    take(exact_ranked, exact_budget)
    take(local_ranked, local_budget)
    take(exact_ranked, maximum - len(selected))
    take(local_ranked, maximum - len(selected))
    if audit is not None:
        exact_rank = {
            int(value[2]): rank for rank, value in enumerate(exact_ranked)
        }
        local_rank = {
            int(value[2]): rank for rank, value in enumerate(local_ranked)
        }
        selected_source_ranks = {
            int(value[2])
            for value in scored
            if id(value[3]) in selected_identity
        }
        audit["broad_screen_score_rows"] = [
            {
                "source_rank": int(value[2]),
                "exact_zero_flow_score": float(value[0]),
                "local_displacement_score": float(value[1]),
                "exact_rank": int(exact_rank[int(value[2])]),
                "local_rank": int(local_rank[int(value[2])]),
                "selected": bool(
                    int(value[2]) in selected_source_ranks
                ),
            }
            for value in sorted(scored, key=lambda item: item[2])
        ]
    return selected


def _stage_c_rows(
    hypotheses: Sequence[object],
    atlas: MapletFeatureAtlasBank,
    ranked_chart_ids: np.ndarray,
    query_feature: np.ndarray,
    query_matchability: np.ndarray | None,
    view: object,
    *,
    region_chart_index: RegionChartIndex | None = None,
    maximum_candidates: int,
    prerank_pool: int,
    render_charts: int,
    refinement_charts: int | None = None,
    rounds: int,
    maximum_translation_updates: int = 2,
    device: str,
    base_stride: int,
    preselected_pool: bool = False,
    report_view_direction_diagnostic: bool = False,
    broad_screen_candidates: int = 0,
    broad_screen_radius_cells: int = 6,
    broad_screen_maximum_points: int = 192,
    selection_score_calibration: object | None = None,
    calibrated_candidate_budget: int = 0,
    stage_c_audit: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    refinement_chart_limit = (
        int(render_charts)
        if refinement_charts is None
        else int(refinement_charts)
    )
    if int(render_charts) < 2:
        raise ValueError("render_charts must be at least two")
    if refinement_chart_limit < 2:
        raise ValueError("refinement_charts must be at least two")
    refinement_chart_limit = min(
        refinement_chart_limit, int(render_charts)
    )
    resolved_base_stride = int(base_stride)
    verification_radius_cells = max(
        int(STAGE_C_VERIFICATION_RADIUS_CELLS),
        int(np.ceil(32.0 / resolved_base_stride)),
    )
    camera_parameters = np.asarray(view.camera.params, dtype=np.float64)
    focal_pixels = float(camera_parameters[0])
    if int(view.camera.model_id) in {1, 4, 5, 6, 10}:
        focal_pixels = float(np.max(camera_parameters[:2]))
    # The structured coarse modes are clustered inside a five-degree angular
    # basin.  Marginalized pre-ranking must cover that basin in image space;
    # the smaller zero-flow window remains the common final verifier.
    prerank_radius_cells = max(
        int(verification_radius_cells),
        int(
            np.ceil(
                focal_pixels
                * np.tan(np.deg2rad(POSE_MODE_ROTATION_RADIUS_DEG))
                / float(resolved_base_stride)
            )
        ),
    )
    scene_geometry = atlas_scene_geometry_points(atlas)
    query_pyramid = build_radio_final_feature_pyramid(
        query_feature,
        base_stride=resolved_base_stride,
        strides=(
            (16, 8, 4)
            if resolved_base_stride == 16
            else (8, 4)
        ),
    )
    matchability_pyramid = _matchability_pyramid(
        query_matchability, query_pyramid
    )
    alignment_atlases = {
        level: atlas for level in query_pyramid
    }
    if preselected_pool:
        # Replay/debug entry point: the caller has already reconstructed and
        # globally sharded the exact Stage-C pool.  Preserve its order and do
        # not create a second generation of consensus modes or suppress poses
        # against only the current shard.
        consensus_modes: Sequence[FramePoseHypothesis] = ()
        source_rank_by_identity = {
            id(value): rank for rank, value in enumerate(hypotheses)
        }
        pool = list(hypotheses)[: max(int(prerank_pool), 0)]
    else:
        consensus_modes = pose_distribution_consensus_modes(
            tuple(
                value
                for value in hypotheses
                if isinstance(value, FramePoseHypothesis)
            ),
            translation_radius_m=POSE_MODE_TRANSLATION_RADIUS_M,
            rotation_radius_deg=POSE_MODE_ROTATION_RADIUS_DEG,
        )
        factorized_modes = factorized_pose_distribution_modes(
            tuple(
                value
                for value in hypotheses
                if isinstance(value, FramePoseHypothesis)
            ),
            consensus_modes,
        )
        source_rank_by_identity = {
            id(value): rank for rank, value in enumerate(hypotheses)
        }
        source_rank_by_identity.update(
            {
                id(value): -index - 1
                for index, value in enumerate(consensus_modes)
            }
        )
        source_rank_by_identity.update(
            {
                id(value): -len(consensus_modes) - index - 1
                for index, value in enumerate(factorized_modes)
            }
        )
        if int(broad_screen_candidates) > 0:
            # The sparse atlas screen exists specifically to compare the
            # complete structured pose distribution cheaply.  Compressing
            # the raw distribution to ``prerank_pool`` first made that screen
            # a no-op with respect to missing mode combinations.  Expose all
            # regional raw modes plus the two derived probability families;
            # the GPU screen below performs the finite reduction.
            pool = _complete_regional_pose_distribution(
                hypotheses, consensus_modes, factorized_modes
            )
        else:
            pool = _stage_c_pose_pool(
                hypotheses,
                consensus_modes,
                factorized_modes,
                max(int(maximum_candidates), int(prerank_pool)),
                minimum_source_charts=2,
            )
    atlas_chart_ids = {
        int(value) for value in atlas.maplet_ids.tolist()
    }
    comparison_chart_ids = []
    for value in np.asarray(
        ranked_chart_ids, dtype=np.int64
    ).reshape(-1).tolist():
        chart_id = int(value)
        if (
            chart_id in atlas_chart_ids
            and chart_id not in comparison_chart_ids
        ):
            comparison_chart_ids.append(chart_id)
        if len(comparison_chart_ids) >= int(render_charts):
            break
    comparison_chart_ids = np.asarray(
        comparison_chart_ids, dtype=np.int64
    )
    if comparison_chart_ids.size < 2:
        return []
    comparison_fit_ids, comparison_heldout_ids = (
        _source_and_heldout_chart_ids(comparison_chart_ids, ())
    )
    broad_screen_input_count = len(pool)
    source_rank_before_broad_screen = {
        id(value): rank for rank, value in enumerate(pool)
    }
    if (
        int(broad_screen_candidates) > 0
        and len(pool) > int(broad_screen_candidates)
    ):
        pool = _stage_c_broad_screen(
            pool,
            atlas,
            comparison_chart_ids,
            query_feature,
            view,
            scene_geometry,
            limit=int(broad_screen_candidates),
            radius_cells=int(broad_screen_radius_cells),
            maximum_points=int(broad_screen_maximum_points),
            device=str(device),
            audit=stage_c_audit,
        )
    broad_screen_selected_count = len(pool)
    if stage_c_audit is not None:
        stage_c_audit.update(
            {
                "broad_screen_input_count": int(
                    broad_screen_input_count
                ),
                "broad_screen_selected_count": int(
                    broad_screen_selected_count
                ),
                "broad_screen_selected_source_ranks": [
                    int(source_rank_before_broad_screen[id(value)])
                    for value in pool
                ],
            }
        )
    prerank_query_cache = build_local_correlation_query_cache(
        query_feature,
        radius=prerank_radius_cells,
        background_samples=512,
        device=str(device),
    )
    preranked = []
    for coarse_rank, hypothesis in enumerate(pool):
        rendered = render_selected_maplet_atlases_fast(
            atlas,
            comparison_chart_ids,
            hypothesis.pose_w2c,
            view.camera,
            width=int(query_feature.shape[2]),
            height=int(query_feature.shape[1]),
            full_scene_depth=render_scene_depth_from_points(
                scene_geometry,
                hypothesis.pose_w2c,
                view.camera,
                width=int(query_feature.shape[2]),
                height=int(query_feature.shape[1]),
            ),
        )
        correlation = local_correlation_distribution(
            rendered,
            query_feature,
            radius=prerank_radius_cells,
            query_matchability=query_matchability,
            maximum_points=1536,
            device=str(device),
            query_cache=prerank_query_cache,
        )
        rendered_chart_ids = set(
            np.asarray(correlation.maplet_ids, dtype=np.int64).tolist()
        )
        if (
            len(rendered_chart_ids) < 4
            or not rendered_chart_ids.intersection(
                comparison_fit_ids.tolist()
            )
            or not rendered_chart_ids.intersection(
                comparison_heldout_ids.tolist()
            )
        ):
            # The update verifier itself requires at least four independent
            # chart votes. A two-chart accidental match cannot become a
            # production pose merely because all missing charts are neutral.
            continue
        fit_score = (
            fixed_chart_zero_displacement_log_bayes_factor(
                correlation, comparison_fit_ids
            )
        )
        heldout_score = (
            fixed_chart_zero_displacement_log_bayes_factor(
                correlation, comparison_heldout_ids
            )
        )
        # The identity set and fit/held-out split are identical for every
        # pose candidate. Missing charts contribute the explicit uniform-null
        # baseline rather than silently leaving the score denominator.
        verification_score = (
            float(heldout_score + 0.25 * fit_score)
            if np.isfinite(fit_score) and np.isfinite(heldout_score)
            else float("-inf")
        )
        feature_atlas_score = (
            fixed_chart_zero_displacement_log_bayes_factor(
                correlation, comparison_chart_ids
            )
        )
        local_alignment_score = fixed_chart_local_match_log_bayes_factor(
            correlation, comparison_chart_ids
        )
        # View direction already conditions the rendered appearance-mode
        # mixture. Adding its absolute baking-view density again ranks camera
        # poses by proximity to historical map coverage and overwhelms the
        # actual query-map likelihood. Keep that density as a diagnostic only
        # for the finite selected set below.
        atlas_score = float(local_alignment_score)
        pose_mode_log_score = _stage_c_pose_log_score(
            atlas_score, hypothesis
        )
        preranked.append(
            (
                pose_mode_log_score,
                atlas_score,
                float(feature_atlas_score),
                float(local_alignment_score),
                verification_score,
                float(fit_score),
                float(heldout_score),
                float(hypothesis.score),
                int(coarse_rank),
                int(source_rank_by_identity[id(hypothesis)]),
                hypothesis,
                correlation,
            )
        )
    local_ranked = sorted(
        preranked, key=lambda value: (-value[0], -value[1], -value[7])
    )
    exact_ranked = sorted(
        preranked, key=lambda value: (-value[2], -value[1], -value[7])
    )
    local_rank_by_identity = {
        id(value): rank for rank, value in enumerate(local_ranked)
    }
    exact_rank_by_identity = {
        id(value): rank for rank, value in enumerate(exact_ranked)
    }
    # Coarse local-match evidence is deliberately displacement-marginalized;
    # exact evidence is deliberately zero-flow.  Neither dominates the other
    # before refinement (q0 and q1 exercise opposite cases), so preserve a
    # finite union of both orderings instead of collapsing them prematurely.
    requested_candidates = max(int(maximum_candidates), 0)
    exact_budget = (requested_candidates + 1) // 2
    local_budget = requested_candidates - exact_budget
    selected = []
    selected_identity = set()
    selection_branch_by_identity: dict[int, str] = {}
    selection_calibrated_score_by_identity: dict[int, float] = {}

    def _take_branch(
        values: Sequence[tuple[object, ...]],
        budget: int,
        branch: str,
    ) -> None:
        added = 0
        for value in values:
            if added >= int(budget):
                break
            hypothesis_identity = id(value[10])
            if hypothesis_identity in selected_identity:
                continue
            selected.append(value)
            selected_identity.add(hypothesis_identity)
            selection_branch_by_identity[id(value)] = str(branch)
            added += 1

    _take_branch(exact_ranked, exact_budget, "exact_zero_flow")
    _take_branch(local_ranked, local_budget, "local_displacement_marginal")
    # Overlap or invalid candidates can leave one branch short. Fill only to
    # the requested finite budget, retaining deterministic score order.
    _take_branch(exact_ranked, requested_candidates - len(selected), "fill")
    _take_branch(local_ranked, requested_candidates - len(selected), "fill")
    legacy_selected_count = len(selected)
    # A trajectory-disjoint evidence calibrator may supplement, but never
    # replace, the exact/local union above.  The union preserves the two
    # complementary branches that were empirically necessary, while the
    # learned list can recover a strong mode just outside either hard prefix.
    calibrated_budget = max(int(calibrated_candidate_budget), 0)
    if selection_score_calibration is not None and calibrated_budget > 0:
        calibration_rows = [
            {
                "_stage_c_prerank_index": int(index),
                "feature_atlas_score": float(value[2]),
                "final_fit_score": float(value[5]),
                "final_heldout_score": float(value[6]),
                "atlas_prerank_local_alignment_score": float(value[3]),
                "coarse_score": float(value[7]),
                "replay_pool_index": int(value[9]),
            }
            for index, value in enumerate(preranked)
        ]
        calibrated_rows = selection_score_calibration.apply(
            calibration_rows
        )
        for calibrated_row in calibrated_rows[:calibrated_budget]:
            value = preranked[
                int(calibrated_row["_stage_c_prerank_index"])
            ]
            selection_calibrated_score_by_identity[id(value)] = float(
                calibrated_row["calibrated_pose_score"]
            )
            hypothesis_identity = id(value[10])
            if hypothesis_identity in selected_identity:
                continue
            selected.append(value)
            selected_identity.add(hypothesis_identity)
            selection_branch_by_identity[id(value)] = (
                "calibrated_evidence_addition"
            )
    if stage_c_audit is not None:
        stage_c_audit.update(
            {
                "legacy_exact_local_selected_count": int(
                    legacy_selected_count
                ),
                "calibrated_candidate_budget": int(calibrated_budget),
                "post_calibration_union_count": int(len(selected)),
            }
        )
    result = []
    level_names = (
        ("coarse", "middle", "fine")
        if resolved_base_stride == 16
        else ("middle", "fine")
    )
    level_strides = {"coarse": 16, "middle": 8, "fine": 4}
    level_schedule = []
    if int(rounds) < 0:
        raise ValueError("Stage-C rounds must be non-negative")
    for iteration in range(int(rounds)):
        level_name = level_names[min(iteration, len(level_names) - 1)]
        level_schedule.append(
            AtlasAlignmentLevel(
                name=level_name,
                feature_stride=level_strides[level_name],
                correlation_radius=STAGE_C_LOCAL_RADII_CELLS[
                    min(
                        iteration,
                        len(STAGE_C_LOCAL_RADII_CELLS) - 1,
                    )
                ],
                maximum_translation_step_m=max(
                    0.03, 0.08 - 0.025 * iteration
                ),
                maximum_rotation_step_deg=max(
                    1.0, 4.0 - 1.25 * iteration
                ),
                maximum_points=3072,
                translation_search_step_m=max(
                    0.08, 0.25 - 0.07 * iteration
                ),
                # At the native St Mary focal length a one-degree rotation
                # is roughly one middle-level search radius.  Halve it with
                # each finer coordinate-descent level so a valid basin is
                # sampled without jumping across it.
                direct_rotation_step_deg=max(
                    0.25, 1.0 * (0.5 ** iteration)
                ),
            )
        )
    refinement_query_caches = {
        (str(level.name), int(level.correlation_radius)):
        build_local_correlation_query_cache(
            query_pyramid[level.name],
            radius=int(level.correlation_radius),
            background_samples=512,
            device=str(device),
        )
        for level in level_schedule
    }
    verification_query_cache = build_local_correlation_query_cache(
        query_feature,
        radius=verification_radius_cells,
        background_samples=512,
        device=str(device),
    )
    for candidate_index, (
        _prerank_mode_score,
        prerank_score,
        prerank_feature_atlas_score,
        prerank_local_alignment_score,
        prerank_verification_score,
        prerank_fit_score,
        prerank_heldout_score,
        _coarse_score,
        coarse_rank,
        source_rank,
        hypothesis,
        prerank_correlation,
    ) in enumerate(selected):
        alignment_reverted_to_initial_common_score = False
        provisional_accepted_step_count = 0
        prerank_view_direction_score = (
            atlas_view_direction_log_likelihood(
                atlas, comparison_chart_ids, hypothesis.pose_w2c
            )
            if bool(report_view_direction_diagnostic)
            else float("nan")
        )
        if not level_schedule:
            refinement_chart_ids = comparison_chart_ids
            refinement_fit_ids = comparison_fit_ids
            refinement_heldout_ids = comparison_heldout_ids
        else:
            compatible_chart_ids = None
            if region_chart_index is not None:
                source_regions = region_chart_index.regions_for_charts(
                    np.asarray(
                        hypothesis.source_chart_ids, dtype=np.int64
                    )
                )
                compatible_chart_ids = (
                    region_chart_index.charts_for_regions(source_regions)
                )
            # Refinement is a D-optimal subset of the *same* fixed identity
            # universe used to compare candidates.  The previous code could
            # silently introduce region-neighbour charts outside that set;
            # those identities influenced the optimizer without participating
            # in the common candidate score.  Keep source identities when
            # they belong to the scoring universe and constrain all graph
            # fallbacks to that universe as well.
            scoring_identity_set = {
                int(value) for value in comparison_chart_ids.tolist()
            }
            refinement_source_chart_ids = tuple(
                int(value)
                for value in hypothesis.source_chart_ids
                if int(value) in scoring_identity_set
            )
            if compatible_chart_ids is not None:
                compatible_chart_ids = np.asarray(
                    [
                        int(value)
                        for value in np.asarray(
                            compatible_chart_ids, dtype=np.int64
                        ).reshape(-1).tolist()
                        if int(value) in scoring_identity_set
                    ],
                    dtype=np.int64,
                )
            refinement_chart_ids = _pose_visible_render_charts(
                atlas,
                comparison_chart_ids,
                refinement_source_chart_ids,
                hypothesis.pose_w2c,
                view.camera,
                limit=int(refinement_chart_limit),
                compatible_chart_ids=compatible_chart_ids,
            )
            refinement_fit_ids, refinement_heldout_ids = (
                _source_and_heldout_chart_ids(
                    refinement_chart_ids,
                    refinement_source_chart_ids,
                )
            )
            if (
                refinement_fit_ids.size == 0
                or refinement_heldout_ids.size == 0
            ):
                refinement_chart_ids = comparison_chart_ids
                refinement_fit_ids = comparison_fit_ids
                refinement_heldout_ids = comparison_heldout_ids
        if level_schedule:
            alignment = refine_pose_with_maplet_atlases(
                alignment_atlases,
                query_pyramid,
                matchability_pyramid,
                refinement_chart_ids,
                hypothesis.pose_w2c,
                view.camera,
                level_schedule,
                device=str(device),
                minimum_fit_gain=0.01,
                minimum_heldout_gain=0.0,
                fit_maplet_ids=refinement_fit_ids,
                heldout_maplet_ids=refinement_heldout_ids,
                query_caches=refinement_query_caches,
                maximum_committed_translation_updates=int(
                    maximum_translation_updates
                ),
            )
            final_pose_w2c = alignment.refined_pose_w2c
            selected_chart_ids = alignment.selected_chart_ids
            alignment_score = float(alignment.score)
            accepted_step_count = int(alignment.accepted_step_count)
            provisional_accepted_step_count = int(
                alignment.accepted_step_count
            )
            alignment_steps = alignment.steps
            # Refinement levels may use different search radii.  Their
            # normalized displacement probabilities are therefore not
            # comparable scores. Re-evaluate every final pose on the same
            # fixed support as the preranker.
            final_rendered = render_selected_maplet_atlases_fast(
                atlas,
                comparison_chart_ids,
                final_pose_w2c,
                view.camera,
                width=int(query_feature.shape[2]),
                height=int(query_feature.shape[1]),
                full_scene_depth=render_scene_depth_from_points(
                    scene_geometry,
                    final_pose_w2c,
                    view.camera,
                    width=int(query_feature.shape[2]),
                    height=int(query_feature.shape[1]),
                ),
            )
            final_correlation = local_correlation_distribution(
                final_rendered,
                query_feature,
                radius=verification_radius_cells,
                query_matchability=query_matchability,
                maximum_points=1536,
                device=str(device),
                query_cache=verification_query_cache,
            )
            final_fit_score = (
                fixed_chart_zero_displacement_log_bayes_factor(
                    final_correlation, comparison_fit_ids
                )
            )
            final_heldout_score = (
                fixed_chart_zero_displacement_log_bayes_factor(
                    final_correlation, comparison_heldout_ids
                )
            )
            final_verification_score = (
                float(
                    final_heldout_score + 0.25 * final_fit_score
                )
                if np.isfinite(final_fit_score)
                and np.isfinite(final_heldout_score)
                else float("-inf")
            )
            final_feature_atlas_score = (
                fixed_chart_zero_displacement_log_bayes_factor(
                    final_correlation, comparison_chart_ids
                )
            )
            if float(final_feature_atlas_score) < float(
                prerank_feature_atlas_score
            ):
                # Refinement is allowed to use a pose-conditioned D-optimal
                # chart subset, but deployment ranking uses one common query
                # identity set.  Keep the seed as an explicit trust-region
                # fallback when the proposed endpoint worsens that common
                # evidence; otherwise refinement can delete a valid mode.
                alignment_reverted_to_initial_common_score = True
                final_pose_w2c = np.asarray(
                    hypothesis.pose_w2c, dtype=np.float64
                )
                selected_chart_ids = tuple(
                    int(value) for value in comparison_chart_ids.tolist()
                )
                accepted_step_count = 0
                final_fit_score = float(prerank_fit_score)
                final_heldout_score = float(prerank_heldout_score)
                final_verification_score = float(
                    prerank_verification_score
                )
                final_feature_atlas_score = float(
                    prerank_feature_atlas_score
                )
                final_correlation = prerank_correlation
            final_view_direction_score = (
                atlas_view_direction_log_likelihood(
                    atlas, comparison_chart_ids, final_pose_w2c
                )
                if bool(report_view_direction_diagnostic)
                else float("nan")
            )
            final_atlas_score = float(final_feature_atlas_score)
            report_correlation = final_correlation
        else:
            # Fast, exact Stage-C prerank replay: the fixed-support evidence
            # above is already the score at the unchanged pose.  Do not
            # render it a second time merely to report diagnostics.
            final_pose_w2c = np.asarray(
                hypothesis.pose_w2c, dtype=np.float64
            )
            selected_chart_ids = tuple(
                int(value) for value in comparison_chart_ids.tolist()
            )
            alignment_score = float(prerank_verification_score)
            accepted_step_count = 0
            alignment_steps = ()
            final_fit_score = float(prerank_fit_score)
            final_heldout_score = float(prerank_heldout_score)
            final_verification_score = float(
                prerank_verification_score
            )
            final_feature_atlas_score = float(
                prerank_feature_atlas_score
            )
            final_view_direction_score = float(
                prerank_view_direction_score
            )
            final_atlas_score = float(prerank_feature_atlas_score)
            # Each candidate must retain its own distribution.  Referring to
            # the loop-local ``correlation`` here used the final preranked
            # candidate's tensor for every rounds=0 diagnostic row.
            report_correlation = prerank_correlation
        chart_log_likelihoods = (
            zero_displacement_chart_log_likelihoods(
                report_correlation, comparison_chart_ids
            )
        )
        chart_support = {}
        for chart_id in comparison_chart_ids.tolist():
            local = np.asarray(report_correlation.maplet_ids) == int(
                chart_id
            )
            if not np.any(local):
                continue
            surfaces = (
                np.zeros((0,), dtype=np.int64)
                if report_correlation.surface_ids is None
                else np.unique(
                    np.asarray(report_correlation.surface_ids)[local]
                )
            )
            chart_support[int(chart_id)] = {
                "rendered_point_count": int(np.sum(local)),
                "rendered_surface_count": int(
                    np.sum(surfaces >= 0)
                ),
                "mean_null_probability": float(
                    np.mean(report_correlation.null_probability[local])
                ),
                "zero_displacement_log_likelihood": (
                    float(chart_log_likelihoods[int(chart_id)])
                    if int(chart_id) in chart_log_likelihoods
                    else None
                ),
            }
        final_pose_mode_log_score = _stage_c_pose_log_score(
            final_atlas_score, hypothesis
        )
        initial_error = pnp_pose_error(
            hypothesis.pose_w2c, view.pose_w2c
        )
        final_error = pnp_pose_error(
            final_pose_w2c, view.pose_w2c
        )
        result.append(
            {
                "candidate_index": int(candidate_index),
                "atlas_pool_rank": int(coarse_rank),
                "coarse_source_rank": int(source_rank),
                "broad_screen_source_rank": int(
                    source_rank_before_broad_screen[id(hypothesis)]
                ),
                "broad_screen_applied": bool(
                    broad_screen_input_count
                    > broad_screen_selected_count
                ),
                "broad_screen_input_count": int(
                    broad_screen_input_count
                ),
                "broad_screen_selected_count": int(
                    broad_screen_selected_count
                ),
                "source_chart_ids": list(hypothesis.source_chart_ids),
                "control_model": str(hypothesis.control_model),
                "seed_model": str(hypothesis.seed_model),
                "pose_mode_support_count": int(
                    hypothesis.mode_support_count
                ),
                "pose_mode_member_count": int(
                    hypothesis.mode_member_count
                ),
                "fit_chart_ids": list(
                    int(value) for value in comparison_fit_ids
                ),
                "heldout_chart_ids": list(
                    int(value) for value in comparison_heldout_ids
                ),
                "scoring_chart_ids": list(
                    int(value) for value in comparison_chart_ids
                ),
                "refinement_fit_chart_ids": list(
                    int(value) for value in refinement_fit_ids
                ),
                "refinement_heldout_chart_ids": list(
                    int(value) for value in refinement_heldout_ids
                ),
                "render_chart_ids": list(selected_chart_ids),
                "prerank_radius_cells": int(prerank_radius_cells),
                "minimum_rendered_chart_count": 4,
                "prerank_selection_branch": (
                    selection_branch_by_identity[id(selected[candidate_index])]
                ),
                "selection_calibrated_pose_score": (
                    float(
                        selection_calibrated_score_by_identity[
                            id(selected[candidate_index])
                        ]
                    )
                    if id(selected[candidate_index])
                    in selection_calibrated_score_by_identity
                    else None
                ),
                "prerank_local_rank": int(
                    local_rank_by_identity[id(selected[candidate_index])]
                ),
                "prerank_exact_rank": int(
                    exact_rank_by_identity[id(selected[candidate_index])]
                ),
                "coarse_score": float(hypothesis.score),
                "atlas_prerank_score": (
                    float(prerank_score)
                    if np.isfinite(prerank_score)
                    else None
                ),
                "atlas_prerank_feature_score": (
                    float(prerank_feature_atlas_score)
                    if np.isfinite(prerank_feature_atlas_score)
                    else None
                ),
                "atlas_prerank_local_alignment_score": (
                    float(prerank_local_alignment_score)
                    if np.isfinite(prerank_local_alignment_score)
                    else None
                ),
                "atlas_prerank_view_direction_score": (
                    float(prerank_view_direction_score)
                    if np.isfinite(prerank_view_direction_score)
                    else None
                ),
                "verification_prerank_score": (
                    float(prerank_verification_score)
                    if np.isfinite(prerank_verification_score)
                    else None
                ),
                "pose_mode_log_score": (
                    float(final_pose_mode_log_score)
                    if np.isfinite(final_pose_mode_log_score)
                    else None
                ),
                "atlas_score": (
                    float(final_atlas_score)
                    if np.isfinite(final_atlas_score)
                    else None
                ),
                "feature_atlas_score": (
                    float(final_feature_atlas_score)
                    if np.isfinite(final_feature_atlas_score)
                    else None
                ),
                "view_direction_score": (
                    float(final_view_direction_score)
                    if np.isfinite(final_view_direction_score)
                    else None
                ),
                "verification_score": (
                    float(final_verification_score)
                    if np.isfinite(final_verification_score)
                    else None
                ),
                "alignment_internal_score": (
                    float(alignment_score)
                    if np.isfinite(alignment_score)
                    else None
                ),
                "final_fit_score": (
                    float(final_fit_score)
                    if np.isfinite(final_fit_score)
                    else None
                ),
                "final_heldout_score": (
                    float(final_heldout_score)
                    if np.isfinite(final_heldout_score)
                    else None
                ),
                "rendered_point_count": int(
                    report_correlation.xyz.shape[0]
                ),
                "chart_support": chart_support,
                "initial_translation_m": float(
                    initial_error.translation_m
                ),
                "initial_rotation_deg": float(
                    initial_error.rotation_deg
                ),
                "initial_pose_w2c": np.asarray(
                    hypothesis.pose_w2c, dtype=np.float64
                ).tolist(),
                "final_translation_m": float(final_error.translation_m),
                "final_rotation_deg": float(final_error.rotation_deg),
                "final_pose_w2c": np.asarray(
                    final_pose_w2c, dtype=np.float64
                ).tolist(),
                "accepted_step_count": int(accepted_step_count),
                "provisional_accepted_step_count": int(
                    provisional_accepted_step_count
                ),
                "alignment_reverted_to_initial_common_score": bool(
                    alignment_reverted_to_initial_common_score
                ),
                "steps": [
                    {
                        **step.__dict__,
                        "condition_number": (
                            float(step.condition_number)
                            if np.isfinite(step.condition_number)
                            else None
                        ),
                        "normalized_update_disagreement": (
                            float(step.normalized_update_disagreement)
                            if np.isfinite(
                                step.normalized_update_disagreement
                            )
                            else None
                        ),
                    }
                    for step in alignment_steps
                ],
            }
        )
    result.sort(
        key=lambda value: (
            -(
                float(value["pose_mode_log_score"])
                if value["pose_mode_log_score"] is not None
                else -np.inf
            ),
            -(
                float(value["atlas_score"])
                if value["atlas_score"] is not None
                else -np.inf
            ),
            -float(value["coarse_score"]),
        )
    )
    return result


def _aggregate_stage_c(
    query_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = [
        list(value.get("stage_c_atlas_alignment", []))
        for value in query_rows
    ]
    top = [value[0] for value in rows if value]
    oracle = [
        min(
            value,
            key=lambda item: (
                float(item["final_translation_m"]) / 0.30
                + float(item["final_rotation_deg"]) / 3.0
            ),
        )
        for value in rows
        if value
    ]
    return {
        "query_count": len(rows),
        "solved_fraction": float(
            np.mean([bool(value) for value in rows])
        )
        if rows
        else 0.0,
        "top1_recall_30cm_3deg": float(
            np.mean(
                [
                    bool(value)
                    and float(value[0]["final_translation_m"]) <= 0.30
                    and float(value[0]["final_rotation_deg"]) <= 3.0
                    for value in rows
                ]
            )
        )
        if rows
        else 0.0,
        "candidate_oracle_recall_30cm_3deg": float(
            np.mean(
                [
                    any(
                        float(item["final_translation_m"]) <= 0.30
                        and float(item["final_rotation_deg"]) <= 3.0
                        for item in value
                    )
                    for value in rows
                ]
            )
        )
        if rows
        else 0.0,
        "top1_translation_median_m": (
            float(
                np.median(
                    [float(value["final_translation_m"]) for value in top]
                )
            )
            if top
            else None
        ),
        "top1_rotation_median_deg": (
            float(
                np.median(
                    [float(value["final_rotation_deg"]) for value in top]
                )
            )
            if top
            else None
        ),
        "top1_translation_p90_m": (
            float(
                np.percentile(
                    [float(value["final_translation_m"]) for value in top],
                    90,
                )
            )
            if top
            else None
        ),
        "top1_rotation_p90_deg": (
            float(
                np.percentile(
                    [float(value["final_rotation_deg"]) for value in top],
                    90,
                )
            )
            if top
            else None
        ),
        "candidate_oracle_translation_median_m": (
            float(
                np.median(
                    [
                        float(value["final_translation_m"])
                        for value in oracle
                    ]
                )
            )
            if oracle
            else None
        ),
        "candidate_oracle_translation_p90_m": (
            float(
                np.percentile(
                    [
                        float(value["final_translation_m"])
                        for value in oracle
                    ],
                    90,
                )
            )
            if oracle
            else None
        ),
        "candidate_oracle_rotation_p90_deg": (
            float(
                np.percentile(
                    [float(value["final_rotation_deg"]) for value in oracle],
                    90,
                )
            )
            if oracle
            else None
        ),
        "top1_initial_translation_median_m": (
            float(
                np.median(
                    [float(value["initial_translation_m"]) for value in top]
                )
            )
            if top
            else None
        ),
        "top1_initial_rotation_median_deg": (
            float(
                np.median(
                    [float(value["initial_rotation_deg"]) for value in top]
                )
            )
            if top
            else None
        ),
        "top1_initial_translation_p90_m": (
            float(
                np.percentile(
                    [float(value["initial_translation_m"]) for value in top],
                    90,
                )
            )
            if top
            else None
        ),
        "top1_initial_rotation_p90_deg": (
            float(
                np.percentile(
                    [float(value["initial_rotation_deg"]) for value in top],
                    90,
                )
            )
            if top
            else None
        ),
        "candidate_oracle_rotation_median_deg": (
            float(
                np.median(
                    [float(value["final_rotation_deg"]) for value in oracle]
                )
            )
            if oracle
            else None
        ),
        "candidate_improved_fraction": float(
            np.mean(
                [
                    float(item["final_translation_m"])
                    < float(item["initial_translation_m"])
                    for value in rows
                    for item in value
                ]
            )
        )
        if any(rows)
        else 0.0,
        "accepted_step_fraction": float(
            np.mean(
                [
                    int(item["accepted_step_count"]) > 0
                    for value in rows
                    for item in value
                ]
            )
        )
        if any(rows)
        else 0.0,
    }


def _aggregate_projective_gap(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    values = [
        float(row["gt_affine_to_homography_control_error_px"])
        for row in rows
        if row.get("gt_affine_to_homography_control_error_px") is not None
    ]
    return {
        "chart_count": len(values),
        "median_control_error_px": (
            float(np.median(values)) if values else None
        ),
        "p90_control_error_px": (
            float(np.quantile(values, 0.90)) if values else None
        ),
        "fraction_above_8px": (
            float(np.mean(np.asarray(values) > 8.0)) if values else None
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if int(args.m3_charts) <= 0 or int(args.m3_pose_hypotheses) <= 0:
        raise ValueError("M3 chart and pose-hypothesis budgets must be positive")
    spatial_feature_stride = int(args.spatial_feature_stride)
    has_spatial_query_tokens = bool(str(args.spatial_query_token_dir))
    hierarchical_frame_alignment = bool(
        args.hierarchical_frame_alignment
    )
    if hierarchical_frame_alignment and (
        not has_spatial_query_tokens or spatial_feature_stride >= 16
    ):
        raise ValueError(
            "hierarchical frame alignment requires true finer-phase query "
            "tokens with --spatial_feature_stride 8"
        )
    stage_c_phase_only = (
        has_spatial_query_tokens and not hierarchical_frame_alignment
    )
    feature_stride = (
        spatial_feature_stride
        if hierarchical_frame_alignment
        else 16 if stage_c_phase_only else spatial_feature_stride
    )
    feature_level = "coarse" if feature_stride == 16 else "middle"
    global_feature_stride = (
        16 if hierarchical_frame_alignment else feature_stride
    )
    global_feature_level = (
        "coarse" if global_feature_stride == 16 else feature_level
    )
    stage_c_feature_stride = (
        spatial_feature_stride
        if has_spatial_query_tokens
        else feature_stride
    )
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
    if hierarchical_frame_alignment:
        if not str(args.coarse_radio_atlas):
            raise ValueError(
                "hierarchical frame alignment requires an explicit "
                "--coarse_radio_atlas; one atlas cannot represent both "
                "native and phase-interleaved RADIO lattices"
            )
        paths["coarse_radio_atlas"] = Path(args.coarse_radio_atlas)
    elif str(args.coarse_radio_atlas):
        raise ValueError(
            "--coarse_radio_atlas is only valid for hierarchical alignment"
        )
    if str(args.stage_c_radio_atlas):
        if hierarchical_frame_alignment:
            raise ValueError(
                "hierarchical alignment already uses --radio_atlas for its "
                "fine/Stage-C level; do not also provide "
                "--stage_c_radio_atlas"
            )
        if not has_spatial_query_tokens:
            raise ValueError(
                "--stage_c_radio_atlas requires phase-matched spatial query "
                "tokens"
            )
        paths["stage_c_radio_atlas"] = Path(args.stage_c_radio_atlas)
    if (
        bool(args.run_stage_c)
        and stage_c_phase_only
        and spatial_feature_stride < 16
        and "stage_c_radio_atlas" not in paths
    ):
        raise ValueError(
            "finer-phase Stage C requires an explicit phase-matched "
            "--stage_c_radio_atlas; a native atlas cannot be paired with a "
            "different RADIO lattice"
        )
    has_mapper = bool(str(args.surface_mapper_checkpoint))
    frame_source_flags = (
        bool(str(args.query_projection_maplets)),
        bool(str(args.frame_spatial_projection_checkpoint)),
    )
    if sum(frame_source_flags) > 1:
        raise ValueError("provide at most one dedicated frame transform")
    if has_mapper:
        paths["surface_mapper_checkpoint"] = Path(
            args.surface_mapper_checkpoint
        )
    if bool(str(args.frame_spatial_projection_checkpoint)):
        paths["frame_spatial_projection_checkpoint"] = Path(
            args.frame_spatial_projection_checkpoint
        )
        feature_source_kind = "surface_spatial_projection"
    elif bool(str(args.query_projection_maplets)):
        paths["query_projection_maplets"] = Path(
            args.query_projection_maplets
        )
        feature_source_kind = "raw_radio_pca"
    elif has_mapper:
        feature_source_kind = "surface_maplet_mapper"
    else:
        raise ValueError(
            "provide a surface mapper or a dedicated frame transform"
        )
    if str(args.structured_frame_adapter):
        if feature_source_kind != "surface_maplet_mapper":
            raise ValueError(
                "structured-frame adapter requires surface-maplet mapper"
            )
        paths["structured_frame_adapter"] = Path(
            args.structured_frame_adapter
        )
    if str(args.structured_frame_refiner):
        _validate_structured_refiner_feature_source(feature_source_kind)
        paths["structured_frame_refiner"] = Path(
            args.structured_frame_refiner
        )
    if str(args.frame_refinement) in {
        "structured",
        "gradient_structured",
        "structured_gradient",
        "structured_flow",
    } and "structured_frame_refiner" not in paths:
        raise ValueError(
            "structured frame_refinement requires "
            "--structured_frame_refiner"
        )
    runtime_retrieval_flags = (
        bool(str(args.spatial_maplets)),
        bool(str(args.probability_calibration)),
    )
    if any(runtime_retrieval_flags) and not all(runtime_retrieval_flags):
        raise ValueError(
            "spatial_maplets and probability_calibration are a pair"
        )
    runtime_retrieval = all(runtime_retrieval_flags)
    if runtime_retrieval and not has_mapper:
        raise ValueError(
            "runtime retrieval requires --surface_mapper_checkpoint"
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
    coarse_atlas = (
        MapletFeatureAtlasBank.load_npz(paths["coarse_radio_atlas"])
        if "coarse_radio_atlas" in paths
        else atlas
    )
    stage_c_atlas = (
        MapletFeatureAtlasBank.load_npz(paths["stage_c_radio_atlas"])
        if "stage_c_radio_atlas" in paths
        else atlas
    )
    if hierarchical_frame_alignment:
        _validate_shared_atlas_geometry(coarse_atlas, atlas)
        fine_stride = int(
            (atlas.metadata or {}).get("metric_feature_stride", -1)
        )
        coarse_stride = int(
            (coarse_atlas.metadata or {}).get(
                "metric_feature_stride", -1
            )
        )
        if fine_stride != feature_stride:
            raise ValueError(
                "fine RADIO atlas stride differs from spatial query stride"
            )
        if coarse_stride != global_feature_stride:
            raise ValueError(
                "coarse RADIO atlas stride differs from global query stride"
            )
    if stage_c_atlas is not atlas:
        _validate_shared_atlas_geometry(atlas, stage_c_atlas)
        stage_c_atlas_stride = int(
            (stage_c_atlas.metadata or {}).get(
                "metric_feature_stride", -1
            )
        )
        if stage_c_atlas_stride != stage_c_feature_stride:
            raise ValueError(
                "Stage-C RADIO atlas stride differs from its query lattice"
            )
    chart_bank = MetricSurfaceChartBank(atlas)
    mapper = None
    retrieval_mapper_metadata: Mapping[str, object] | None = None
    if has_mapper:
        mapper, loaded_mapper_metadata = load_surface_maplet_mapper(
            paths["surface_mapper_checkpoint"], device=str(args.device)
        )
        retrieval_mapper_metadata = dict(loaded_mapper_metadata)
        declared_identity_mapper = str(
            (retrieval_regions.feature_bank.metadata or {}).get(
                "surface_mapper_sha256", ""
            )
        )
        if (
            runtime_retrieval
            and (
                not declared_identity_mapper
                or declared_identity_mapper
                != _sha256(paths["surface_mapper_checkpoint"])
            )
        ):
            raise ValueError(
                "retrieval-region map and online mapper lineage differ"
            )
    projection_bank = None
    spatial_projection_model = None
    if feature_source_kind == "surface_maplet_mapper":
        if mapper is None or retrieval_mapper_metadata is None:
            raise ValueError("frame mapper source is unavailable")
        feature_source_metadata = retrieval_mapper_metadata
    elif feature_source_kind == "surface_spatial_projection":
        spatial_projection_model, loaded_projection_metadata = (
            load_surface_spatial_projection(
                paths["frame_spatial_projection_checkpoint"],
                device=str(args.device),
            )
        )
        spatial_projection_model.eval()
        feature_source_metadata = dict(loaded_projection_metadata)
    else:
        projection_bank = None
        projection_bank = SurfaceRetrievalMapletBank.load_npz(
            paths["query_projection_maplets"]
        )
        if projection_bank.query_projection is None:
            raise ValueError("RADIO PCA maplets omit their query projection")
        feature_source_metadata = dict(projection_bank.metadata or {})
    if "structured_frame_adapter" in paths:
        frame_adapter, frame_adapter_metadata = (
            load_structured_frame_adapter(
                paths["structured_frame_adapter"],
                device=str(args.device),
            )
        )
        if str(
            frame_adapter_metadata.get(
                "compatible_radio_atlas_sha256", ""
            )
        ) != _sha256(paths["radio_atlas"]):
            raise ValueError("structured-frame adapter atlas differs")
        if str(
            frame_adapter_metadata.get(
                "surface_mapper_checkpoint_sha256", ""
            )
        ) != _sha256(paths["surface_mapper_checkpoint"]):
            raise ValueError("structured-frame adapter mapper differs")
    else:
        frame_adapter = None
        frame_adapter_metadata = {}
    if "structured_frame_refiner" in paths:
        frame_refiner, frame_refiner_metadata = (
            load_structured_frame_refiner(
                paths["structured_frame_refiner"],
                device=str(args.device),
            )
        )
        if str(
            frame_refiner_metadata.get(
                "compatible_radio_atlas_sha256", ""
            )
        ) != _sha256(paths["radio_atlas"]):
            raise ValueError("structured-frame refiner atlas differs")
        refiner_feature_kind = str(
            frame_refiner_metadata.get("query_feature_transform", "")
        )
        if refiner_feature_kind:
            if refiner_feature_kind != str(feature_source_kind):
                raise ValueError(
                    "structured-frame refiner feature transform differs"
                )
            refiner_feature_path = (
                paths["frame_spatial_projection_checkpoint"]
                if refiner_feature_kind == "surface_spatial_projection"
                else paths["surface_mapper_checkpoint"]
            )
            if str(
                frame_refiner_metadata.get(
                    "query_feature_transform_sha256", ""
                )
            ) != _sha256(refiner_feature_path):
                raise ValueError(
                    "structured-frame refiner feature lineage differs"
                )
        elif str(
            frame_refiner_metadata.get(
                "surface_mapper_checkpoint_sha256", ""
            )
        ) != _sha256(paths["surface_mapper_checkpoint"]):
            # Backward compatibility for the pre-typed refiner artifacts.
            raise ValueError("structured-frame refiner mapper differs")
    else:
        frame_refiner = None
        frame_refiner_metadata = {}
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
        if str(
            calibration_metadata.get("surface_mapper_sha256", "")
        ) != _sha256(paths["surface_mapper_checkpoint"]):
            raise ValueError("probability calibration mapper differs")
        if str(
            (spatial_bank.metadata or {}).get(
                "query_feature_transform_sha256", ""
            )
        ) != str(
            (atlas.metadata or {}).get(
                "query_feature_transform_sha256", ""
            )
        ):
            raise ValueError(
                "retrieval spatial bank and frame atlas transforms differ"
            )
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
    if (
        int(args.query_shard_count) <= 0
        or not 0
        <= int(args.query_shard_index)
        < int(args.query_shard_count)
    ):
        raise ValueError("invalid query shard")
    views = [
        view
        for row, view in enumerate(views)
        if row % int(args.query_shard_count)
        == int(args.query_shard_index)
    ]
    if not views:
        raise ValueError("query shard is empty")
    protocol = _validate_contract(
        paths,
        atlas,
        index,
        feature_source_metadata,
        feature_source_kind,
        sorted({str(view.trajectory_id) for view in views}),
        retrieval_mapper_metadata,
        enforce_strict_disjoint=(str(args.evaluation_role) == "strict"),
    )
    identity_mapping = {
        str(value)
        for value in (
            retrieval_regions.feature_bank.metadata or {}
        ).get("mapping_trajectory_ids", [])
    }
    query_trajectory_ids = {
        str(view.trajectory_id) for view in views
    }
    if query_trajectory_ids & identity_mapping:
        raise ValueError(
            "strict queries overlap retrieval-region mapping views"
        )
    protocol["retrieval_region_mapping"] = sorted(identity_mapping)
    protocol[
        "strict_test_disjoint_from_retrieval_region_mapping"
    ] = True
    if frame_adapter is not None:
        adapter_reference = {
            str(value)
            for key in (
                "training_trajectory_ids",
                "validation_trajectory_ids",
            )
            for value in frame_adapter_metadata.get(key, [])
        }
        query_trajectory_ids = {
            str(view.trajectory_id) for view in views
        }
        if (
            str(args.evaluation_role) == "strict"
            and query_trajectory_ids & adapter_reference
        ):
            raise ValueError(
                "strict queries overlap structured-frame adapter references"
            )
        protocol["structured_frame_adapter_reference"] = sorted(
            adapter_reference
        )
        protocol[
            "strict_test_disjoint_from_structured_frame_adapter"
        ] = not bool(query_trajectory_ids & adapter_reference)
    if frame_refiner is not None:
        refiner_reference = {
            str(value)
            for key in (
                "training_trajectory_ids",
                "validation_trajectory_ids",
            )
            for value in frame_refiner_metadata.get(key, [])
        }
        query_trajectory_ids = {
            str(view.trajectory_id) for view in views
        }
        if (
            str(args.evaluation_role) == "strict"
            and query_trajectory_ids & refiner_reference
        ):
            raise ValueError(
                "strict queries overlap structured-frame refiner references"
            )
        protocol["structured_frame_refiner_reference"] = sorted(
            refiner_reference
        )
        protocol[
            "strict_test_disjoint_from_structured_frame_refiner"
        ] = not bool(query_trajectory_ids & refiner_reference)
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
        if (
            str(args.evaluation_role) == "strict"
            and query_ids & calibration_ids
        ):
            raise ValueError(
                "strict queries overlap probability calibration"
            )
        if (
            str(args.evaluation_role) == "strict"
            and not query_ids <= strict_holdout_ids
        ):
            raise ValueError(
                "queries are absent from calibration strict holdout"
            )
        protocol["probability_calibration"] = sorted(calibration_ids)
        protocol["query_overlap_with_probability_calibration"] = sorted(
            query_ids & calibration_ids
        )
        protocol[
            "strict_test_disjoint_from_probability_calibration"
        ] = not bool(query_ids & calibration_ids)
    chart_rows = []
    refined_chart_rows = []
    composite_rows = []
    refined_rows = []
    query_rows = []
    retrieval_rows = []
    search_config = replace(
        _search_config(global_feature_level),
        support_score_power=float(args.support_score_power),
    )
    alike_detector = (
        AlikeDetectorOnly(
            device=str(args.device),
            matcha_repo=Path(args.matcha_repo),
            model_name=str(args.alike_model),
        )
        if bool(
            args.alike_detector_matchability
            or args.alike_detector_global_proposal
            or args.alike_detector_local_proposal_branch
        )
        else None
    )
    alike_detector_metadata = (
        dict(alike_detector.metadata) if alike_detector is not None else {}
    )
    for query_index, view in enumerate(views):
        global_raw_radio = view.radio.numpy()
        if has_spatial_query_tokens:
            spatial_token_path = (
                Path(args.spatial_query_token_dir)
                / f"{view.image_id.replace('/', '__')}.npz"
            )
            if not spatial_token_path.exists():
                raise FileNotFoundError(
                    f"spatial query token is missing: {spatial_token_path}"
                )
            with np.load(
                spatial_token_path, allow_pickle=False
            ) as spatial_token:
                raw_radio = np.asarray(
                    spatial_token["radio_final"], dtype=np.float32
                )
        else:
            raw_radio = global_raw_radio
        if has_spatial_query_tokens:
            expected_scale = (
                16.0 / float(spatial_feature_stride)
            )
            observed_scale = np.asarray(raw_radio.shape[-2:]) / np.asarray(
                global_raw_radio.shape[-2:]
            )
            if not np.allclose(
                observed_scale,
                expected_scale,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "phase RADIO token shape does not match its declared "
                    "feature stride"
                )

        def project_frame_feature(source: np.ndarray) -> np.ndarray:
            if feature_source_kind == "surface_spatial_projection":
                if spatial_projection_model is None:
                    raise ValueError(
                        "spatial frame projection is unavailable"
                    )
                return _project_surface_spatial_feature_map(
                    spatial_projection_model,
                    source,
                    device=str(args.device),
                )
            if feature_source_kind == "surface_maplet_mapper":
                if mapper is None:
                    raise ValueError(
                        "surface frame mapper is unavailable"
                    )
                return mapper.project(source).measurement_context
            if projection_bank is None:
                raise ValueError("RADIO PCA projection is unavailable")
            return projection_bank.project_query_feature_map(source)

        mapped_global = project_frame_feature(global_raw_radio)
        mapped_spatial = (
            project_frame_feature(raw_radio)
            if has_spatial_query_tokens
            else mapped_global
        )
        mapped = (
            mapped_spatial
            if hierarchical_frame_alignment
            else mapped_global
        )
        stage_c_mapped = (
            mapped_spatial
            if has_spatial_query_tokens
            else mapped
        )
        query_feature_global = torch.from_numpy(mapped_global).to(
            str(args.device)
        )
        query_feature = torch.from_numpy(mapped).to(str(args.device))
        if frame_adapter is not None:
            with torch.no_grad():
                query_feature_global = frame_adapter(
                    query_feature_global[None]
                )[0]
                query_feature = (
                    frame_adapter(query_feature[None])[0]
                    if hierarchical_frame_alignment
                    else query_feature_global
                )
                stage_c_feature = (
                    frame_adapter(
                        torch.from_numpy(mapped_spatial)
                        .to(str(args.device))[None]
                    )[0]
                    if has_spatial_query_tokens
                    else query_feature
                )
            mapped_global = (
                query_feature_global.detach().cpu().numpy()
            )
            mapped = query_feature.detach().cpu().numpy()
            stage_c_mapped = (
                stage_c_feature.detach().cpu().numpy()
            )
        query_matchability_global = None
        query_matchability = None
        stage_c_query_matchability = None
        alike_detection_count = 0
        if alike_detector is not None:
            detections = alike_detector.detect(
                Path(args.image_root) / view.image_id,
                image_width=int(view.camera.width),
                image_height=int(view.camera.height),
                top_k=int(args.alike_detector_top_k),
                candidate_top_k=max(
                    4096, int(args.alike_detector_top_k)
                ),
                nms_radius_px=4.0,
                grid_rows=8,
                grid_cols=8,
            )
            alike_detection_count = int(detections.xy.shape[0])
            global_matchability_array = _alike_matchability_map(
                detections.xy,
                detections.scores,
                image_width=int(view.camera.width),
                image_height=int(view.camera.height),
                feature_width=int(mapped_global.shape[2]),
                feature_height=int(mapped_global.shape[1]),
                feature_stride=int(global_feature_stride),
            )
            frame_matchability_array = (
                _alike_matchability_map(
                    detections.xy,
                    detections.scores,
                    image_width=int(view.camera.width),
                    image_height=int(view.camera.height),
                    feature_width=int(mapped.shape[2]),
                    feature_height=int(mapped.shape[1]),
                    feature_stride=int(feature_stride),
                )
                if hierarchical_frame_alignment
                else global_matchability_array
            )
            stage_c_query_matchability = (
                _alike_matchability_map(
                    detections.xy,
                    detections.scores,
                    image_width=int(view.camera.width),
                    image_height=int(view.camera.height),
                    feature_width=int(stage_c_mapped.shape[2]),
                    feature_height=int(stage_c_mapped.shape[1]),
                    feature_stride=int(stage_c_feature_stride),
                )
                if (
                    stage_c_mapped.shape[-2:]
                    != global_matchability_array.shape
                    or stage_c_feature_stride != global_feature_stride
                )
                else global_matchability_array
            )
            # A query-only detector heatmap is not map/query identity
            # evidence.  Passing it into the global chart search allowed the
            # spatial arrangement of ALIKE peaks to vote for chart location
            # before RADIO agreement and measurably destroyed valid pose
            # basins on repeated facades.  Keep global search RADIO-only;
            # detector scores enter only after RADIO has defined a local
            # window, where they are collapsed to offset-neutral cell
            # reliability by the chart/atlas correlation modules.
            if bool(args.alike_detector_global_proposal):
                query_matchability_global = torch.from_numpy(
                    global_matchability_array
                ).to(str(args.device))
            query_matchability = torch.from_numpy(
                frame_matchability_array
            ).to(str(args.device))

        def refinement_seeds(
            matches: Sequence[MapletFrameMatch],
        ) -> tuple[MapletFrameMatch, ...]:
            if not hierarchical_frame_alignment:
                return tuple(matches)
            return tuple(
                rescale_maplet_frame_match(
                    value,
                    feature_stride=feature_stride,
                    feature_level=feature_level,
                )
                for value in matches
            )
        visible_ids, visible_counts = _visible_charts(view, atlas)
        if runtime_retrieval:
            retrieval = _retrieve(
                view.radio.numpy(),
                retrieval_regions.feature_bank,
                spatial_bank,
                mapper,
                retrieval_mapper_metadata or {},
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
            retrieved_chart_posterior = _candidate_retrieved_charts(
                retrieval,
                index,
                retrieval_regions,
                chart_bank,
                int(args.m3_retrieval_regions),
                int(args.m3_frame_screen_charts),
                minimum_charts=int(args.m3_frame_screen_charts),
                cumulative_probability=float(
                    args.m3_chart_probability_mass
                ),
            )
            frame_screen_chart_ids = (
                retrieved_chart_posterior.selected_chart_ids
            )
            raw_matches_by_chart = {}
            phase_chart_evidence = []
            for chart_id in frame_screen_chart_ids.tolist():
                prior_xy, prior_probability = _chart_priors(
                    int(chart_id),
                    retrieval,
                    index,
                    retrieval_regions,
                    chart_bank,
                )
                raw_matches = align_maplet_frame_global(
                    coarse_atlas,
                    int(chart_id),
                    query_feature_global,
                    query_matchability_global,
                    feature_level=global_feature_level,
                    feature_stride=global_feature_stride,
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
                identity_probability = (
                    retrieved_chart_posterior.probability_for(
                        int(chart_id)
                    )
                )
                conditioned_raw = tuple(
                    replace(
                        value,
                        identity_probability=identity_probability,
                    )
                    for value in raw_matches
                )
                raw_matches_by_chart[int(chart_id)] = conditioned_raw
                phase_chart_evidence.append(
                    (
                        _marginal_frame_log_evidence(conditioned_raw),
                        int(chart_id),
                    )
                )
            phase_chart_evidence.sort(
                key=lambda value: (-value[0], value[1])
            )
            retrieved_chart_ids = np.asarray(
                [
                    chart_id
                    for _evidence, chart_id in phase_chart_evidence[
                        : int(args.m3_charts)
                    ]
                ],
                dtype=np.int64,
            )
            retrieved_chart_matches = {}
            for chart_id in retrieved_chart_ids.tolist():
                raw_matches = raw_matches_by_chart[int(chart_id)]
                refined_matches = _refine_frame_modes(
                    atlas,
                    refinement_seeds(raw_matches[:8]),
                    query_feature,
                    str(args.frame_refinement),
                    frame_refiner,
                    frame_refiner_metadata,
                    query_matchability,
                    bool(args.alike_detector_local_proposal_branch),
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
                    for value in refined_matches
                )
            retrieved_chart_hypotheses = _predicted_pose_hypotheses(
                retrieved_chart_matches,
                atlas,
                view,
                maximum_charts=int(args.m3_charts),
                maximum_pose_hypotheses=int(args.m3_pose_hypotheses),
            )
            retrieved_chart_pose_errors = _pose_errors(
                retrieved_chart_hypotheses, view.pose_w2c
            )
            retrieved_consensus_mode_errors = _pose_errors(
                pose_distribution_consensus_modes(
                    retrieved_chart_hypotheses,
                    translation_radius_m=(
                        POSE_MODE_TRANSLATION_RADIUS_M
                    ),
                    rotation_radius_deg=POSE_MODE_ROTATION_RADIUS_DEG,
                ),
                view.pose_w2c,
            )
            stage_c_rows = (
                _stage_c_rows(
                    retrieved_chart_hypotheses,
                    stage_c_atlas,
                    np.asarray(
                        [
                            chart_id
                            for _evidence, chart_id in phase_chart_evidence
                        ],
                        dtype=np.int64,
                    ),
                    stage_c_mapped,
                    stage_c_query_matchability,
                    view,
                    region_chart_index=index,
                    maximum_candidates=int(
                        args.stage_c_pose_candidates
                    ),
                    prerank_pool=int(args.stage_c_prerank_pool),
                    render_charts=int(args.stage_c_render_charts),
                    refinement_charts=int(
                        args.stage_c_refinement_charts
                    ),
                    rounds=int(args.stage_c_rounds),
                    maximum_translation_updates=int(
                        args.stage_c_maximum_translation_updates
                    ),
                    device=str(args.device),
                    base_stride=stage_c_feature_stride,
                    broad_screen_candidates=int(
                        args.stage_c_broad_screen_candidates
                    ),
                    broad_screen_radius_cells=int(
                        args.stage_c_broad_screen_radius_cells
                    ),
                    broad_screen_maximum_points=int(
                        args.stage_c_broad_screen_maximum_points
                    ),
                )
                if bool(args.run_stage_c)
                else []
            )
            expansion_diagnostic = _chart_expansion_diagnostics(
                retrieval,
                retrieved_chart_posterior,
                visible_ids,
                visible_counts,
                index,
                region_limit=int(args.m3_retrieval_regions),
            )
            retrieved_oracle_mode_errors = _predicted_pose_diagnostic(
                _oracle_best_frame_modes(
                    retrieved_chart_matches, atlas, view
                ),
                atlas,
                view,
                maximum_charts=int(args.m3_charts),
                maximum_pose_hypotheses=int(args.m3_pose_hypotheses),
            )
            visible_set = set(
                int(value) for value in visible_ids.tolist()
            )
            retrieved_gt_frame_matches = {}
            for chart_id in retrieved_chart_ids.tolist():
                if int(chart_id) not in visible_set:
                    continue
                target = ground_truth_chart_frame(
                    atlas,
                    int(chart_id),
                    view.pose_w2c,
                    view.camera,
                    feature_stride=feature_stride,
                    feature_level=(
                        "diagnostic_runtime_retrieval_gt_homography"
                    ),
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
                atlas,
                view,
                maximum_charts=int(args.m3_charts),
                maximum_pose_hypotheses=int(args.m3_pose_hypotheses),
            )
        else:
            frame_screen_chart_ids = np.zeros((0,), dtype=np.int64)
            phase_chart_evidence = []
            retrieved_chart_ids = np.zeros((0,), dtype=np.int64)
            retrieved_chart_pose_errors = []
            retrieved_consensus_mode_errors = []
            stage_c_rows = []
            expansion_diagnostic = {}
            retrieved_oracle_mode_errors = []
            retrieved_gt_frame_pose_errors = []
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
                coarse_atlas,
                int(chart_id),
                query_feature_global,
                query_matchability_global,
                feature_level=global_feature_level,
                feature_stride=global_feature_stride,
                config=search_config,
            )
            chart_matches[int(chart_id)] = matches
            refined_matches = _refine_frame_modes(
                atlas,
                refinement_seeds(matches[:8]),
                query_feature,
                str(args.frame_refinement),
                frame_refiner,
                frame_refiner_metadata,
                query_matchability,
                bool(args.alike_detector_local_proposal_branch),
            )
            refined_chart_matches[int(chart_id)] = refined_matches
            raw_target = ground_truth_chart_frame(
                atlas,
                int(chart_id),
                view.pose_w2c,
                view.camera,
                feature_stride=global_feature_stride,
                feature_level="radio_mapper_global_gt_homography",
                model="homography",
            )
            target = ground_truth_chart_frame(
                atlas,
                int(chart_id),
                view.pose_w2c,
                view.camera,
                feature_stride=feature_stride,
                feature_level="radio_mapper_gt_homography",
                model="homography",
            )
            if target is not None:
                affine_target = ground_truth_chart_frame(
                    atlas,
                    int(chart_id),
                    view.pose_w2c,
                    view.camera,
                    feature_stride=feature_stride,
                    feature_level="diagnostic_gt_affine",
                    model="affine",
                )
                projective_gap = (
                    frame_control_error_px(affine_target, target)
                    if affine_target is not None
                    else None
                )
                for destination, row_level, values, row_target in (
                    (
                        chart_rows,
                        global_feature_level,
                        matches,
                        raw_target,
                    ),
                    (
                        refined_chart_rows,
                        feature_level,
                        refined_matches,
                        target,
                    ),
                ):
                    if row_target is None:
                        continue
                    row = _frame_row(
                            view.image_id,
                            row_level,
                            int(chart_id),
                            values,
                            row_target,
                            conditioned=False,
                    )
                    row[
                        "gt_affine_to_homography_control_error_px"
                    ] = projective_gap
                    destination.append(row)
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
                query_feature_global,
                query_matchability_global,
                feature_level=global_feature_level,
                feature_stride=global_feature_stride,
                config=search_config,
            )
            refined = _refine_frame_modes(
                composite,
                refinement_seeds(matches[:8]),
                query_feature,
                str(args.frame_refinement),
                frame_refiner,
                frame_refiner_metadata,
                query_matchability,
                bool(args.alike_detector_local_proposal_branch),
            )
            raw_target = ground_truth_chart_frame(
                composite,
                int(region_id),
                view.pose_w2c,
                view.camera,
                feature_stride=global_feature_stride,
                feature_level=(
                    "radio_mapper_composite_global_gt_homography"
                ),
                model="homography",
            )
            target = ground_truth_chart_frame(
                composite,
                int(region_id),
                view.pose_w2c,
                view.camera,
                feature_stride=feature_stride,
                feature_level="radio_mapper_composite_gt_homography",
                model="homography",
            )
            if target is not None and raw_target is not None:
                composite_rows.append(
                    _frame_row(
                        view.image_id,
                        global_feature_level,
                        int(region_id),
                        matches,
                        raw_target,
                        conditioned=False,
                    )
                )
                refined_rows.append(
                    _frame_row(
                        view.image_id,
                        feature_level,
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
        correct_oracle_mode_errors = _predicted_pose_diagnostic(
            _oracle_best_frame_modes(
                refined_chart_matches, atlas, view
            ),
            atlas,
            view,
            maximum_charts=int(args.m1_charts_per_query),
        )
        query_rows.append(
            {
                "image_id": view.image_id,
                "trajectory_id": view.trajectory_id,
                "alike_detection_count": int(alike_detection_count),
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
                "d3_correct_identity_oracle_mode_pose_errors": (
                    correct_oracle_mode_errors
                ),
                "m3_retrieved_chart_ids": retrieved_chart_ids.tolist(),
                "m3_retrieved_chart_frame_modes": (
                    _serialized_frame_mode_rows(retrieved_chart_matches)
                ),
                "m3_frame_screen_chart_ids": (
                    frame_screen_chart_ids.tolist()
                ),
                "m3_phase_chart_log_evidence": [
                    {
                        "chart_id": int(chart_id),
                        "log_evidence": (
                            float(evidence)
                            if np.isfinite(evidence)
                            else None
                        ),
                    }
                    for evidence, chart_id in phase_chart_evidence
                ],
                "m3_retrieved_chart_pose_errors": (
                    retrieved_chart_pose_errors
                ),
                "m3_retrieved_se3_consensus_mode_pose_errors": (
                    retrieved_consensus_mode_errors
                ),
                "d1_region_chart_expansion": expansion_diagnostic,
                "d2_retrieved_charts_gt_frame_pose_errors": (
                    retrieved_gt_frame_pose_errors
                ),
                "d4_retrieved_predicted_frame_oracle_mode_pose_errors": (
                    retrieved_oracle_mode_errors
                ),
                "stage_c_atlas_alignment": stage_c_rows,
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
            "uses_alike_detector_scores": bool(
                alike_detector is not None
            ),
            "uses_loftr": False,
            "uses_radio_intermediate": False,
            "uses_point_correspondence_pnp": False,
            "uses_final_point_correspondence_pnp": False,
            "uses_regional_frame_control_seed_solver": True,
            "uses_chart_block_factor_refinement": True,
            "uses_se3_mode_probability_marginalization": True,
            "uses_maplet_disjoint_atlas_verification": True,
            "uses_area_invariant_surface_evidence": True,
            "uses_map_mode_view_support_domain": True,
            "uses_phase_aware_chart_screening_after_region_retrieval": True,
            "uses_coarse_to_fine_radio_final_frame_alignment": (
                hierarchical_frame_alignment
            ),
            "includes_chart_center_grouped_diagnostic": False,
            "final_pose_estimator": (
                "rendered_maplet_atlas_correlation_iterative_se3"
                if bool(args.run_stage_c)
                else "not_run_stage_b_diagnostic_only"
            ),
            "query_feature_source": (
                "RADIO-final through "
                + (
                    (
                        "frozen surface-maplet mapper and structured "
                        "chart-transform adapter"
                    )
                    if frame_adapter is not None
                    else "frozen surface-maplet mapper"
                    if feature_source_kind == "surface_maplet_mapper"
                    else (
                        "phase-preserving surface-spatial projection "
                        "(retrieval mapper separated)"
                    )
                    if feature_source_kind
                    == "surface_spatial_projection"
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
        "m3_retrieved_se3_consensus_mode_pose": (
            _aggregate_pose_rows(
                query_rows,
                "m3_retrieved_se3_consensus_mode_pose_errors",
            )
        ),
        "m3_correct_composite_region_predicted_frame_pose": (
            _aggregate_pose_rows(
                query_rows, "m3_correct_region_pose_errors"
            )
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
        "d6_gt_affine_vs_homography": _aggregate_projective_gap(
            chart_rows
        ),
        "stage_c_runtime_atlas_alignment": (
            _aggregate_stage_c(query_rows)
            if bool(args.run_stage_c)
            else None
        ),
        "diagnostic_scope": (
            (
                "non-deployable calibration replay; runtime candidates are "
                "generated before target pose errors are attached"
                if str(args.evaluation_role) == "calibration"
                else "runtime retrieval and correct-identity decomposition; "
                "ground truth is used only by explicitly named D2/D3/D4 "
                "diagnostics after candidates have been generated"
            )
        ),
        "deployable_result": bool(str(args.evaluation_role) == "strict"),
        "scene_evidence": (
            {"topq_nms": _aggregate_retrieval(retrieval_rows)}
            if retrieval_rows
            else None
        ),
        "configuration": {
            "evaluation_role": str(args.evaluation_role),
            "feature_stride": feature_stride,
            "feature_level": feature_level,
            "global_feature_stride": global_feature_stride,
            "global_feature_level": global_feature_level,
            "stage_c_feature_stride": stage_c_feature_stride,
            "stage_c_phase_only": stage_c_phase_only,
            "hierarchical_frame_alignment": (
                hierarchical_frame_alignment
            ),
            "uses_distinct_phase_matched_atlas_levels": (
                hierarchical_frame_alignment
                or stage_c_atlas is not atlas
            ),
            "hierarchical_frame_schedule": (
                [
                    "stride16_global_chart_mode_search",
                    f"stride{feature_stride}_local_chart_residual",
                    f"stride{stage_c_feature_stride}_rendered_atlas_se3",
                ]
                if hierarchical_frame_alignment
                else (
                    [
                        "stride16_native_chart_mode_search_and_residual",
                        (
                            f"stride{stage_c_feature_stride}_phase_matched_"
                            "rendered_atlas_se3"
                        ),
                    ]
                    if stage_c_phase_only
                    else [f"stride{feature_stride}_single_scale"]
                )
            ),
            "stage_c_radio_atlas": (
                str(paths["stage_c_radio_atlas"])
                if "stage_c_radio_atlas" in paths
                else str(paths["radio_atlas"])
            ),
            "spatial_query_token_dir": (
                str(args.spatial_query_token_dir)
                if str(args.spatial_query_token_dir)
                else None
            ),
            "feature_source_kind": feature_source_kind,
            "alike_detector_matchability": bool(
                alike_detector is not None
            ),
            "alike_detector_global_frame_weighting": bool(
                args.alike_detector_global_proposal
            ),
            "alike_detector_global_proposal_only": bool(
                args.alike_detector_global_proposal
            ),
            "alike_detector_local_cell_reliability": bool(
                alike_detector is not None
            ),
            "alike_detector_local_proposal_only": bool(
                args.alike_detector_local_proposal_branch
            ),
            "alike_detector_local_proposal_selection": (
                "detector_guided_initialization_then_radio_optimization"
                if bool(args.alike_detector_local_proposal_branch)
                else "radio_ranked_union"
            ),
            "alike_detector_offset_prior": False,
            "alike_detector": (
                alike_detector_metadata
                if alike_detector is not None
                else None
            ),
            "alike_detector_top_k": int(args.alike_detector_top_k),
            "alike_matchability_floor": (
                float(ALIKE_MATCHABILITY_FLOOR)
                if alike_detector is not None
                else None
            ),
            "alike_descriptors_computed_or_stored": False,
            "uses_structured_frame_adapter": bool(
                frame_adapter is not None
            ),
            "uses_structured_frame_refiner": bool(
                frame_refiner is not None
            ),
            "query_shard_count": int(args.query_shard_count),
            "query_shard_index": int(args.query_shard_index),
            "m1_charts_per_query": int(args.m1_charts_per_query),
            "m1_regions_per_query": int(args.m1_regions_per_query),
            "composite_region_resolution": 48,
            "local_refinement_iterations": 20,
            "local_refinement_mode": str(args.frame_refinement),
            "local_refinement_model": (
                "local_multimodal_radio_flow_projective_homography"
                if str(args.frame_refinement) == "local_flow"
                else (
                    "fixed_denominator_gradient_then_local_multimodal_"
                    "radio_flow_projective_homography"
                    if str(args.frame_refinement) == "hybrid"
                    else (
                        "trajectory_disjoint_complete_chart_radio_refiner"
                        if str(args.frame_refinement) == "structured"
                        else (
                            "fixed_denominator_gradient_then_trajectory_"
                            "disjoint_complete_chart_radio_refiner"
                            if str(args.frame_refinement)
                            == "gradient_structured"
                            else (
                                "trajectory_disjoint_complete_chart_radio_"
                                "refiner_then_fixed_denominator_projective_"
                                "alignment"
                                if str(args.frame_refinement)
                                == "structured_gradient"
                                else (
                                    "trajectory_disjoint_complete_chart_"
                                    "radio_refiner_then_robust_regional_"
                                    "radio_flow_homography"
                                    if str(args.frame_refinement)
                                    == "structured_flow"
                                    else (
                                        "complete_chart_radio_correlation_"
                                        "volume_robust_affine_factor"
                                        if str(args.frame_refinement)
                                        == "chart_volume"
                                        else "projective_homography"
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "local_refinement_fixed_canonical_denominator": True,
            "joint_translation_affine_nms": True,
            "maximum_modes_per_spatial_cluster": 2,
            "maximum_appearance_modes": 2,
            "support_score_power": float(args.support_score_power),
            "runtime_retrieval": bool(runtime_retrieval),
            "m3_retrieval_regions": int(args.m3_retrieval_regions),
            "m3_frame_screen_charts": int(
                args.m3_frame_screen_charts
            ),
            "m3_charts": int(args.m3_charts),
            "m3_min_charts": int(args.m3_min_charts),
            "m3_chart_probability_mass": float(
                args.m3_chart_probability_mass
            ),
            "m3_chart_selection": (
                "retrieval_region_expansion_then_phase_preserving_"
                "radio_frame_mode_marginalization"
            ),
            "run_stage_c": bool(args.run_stage_c),
            "stage_c_pose_candidates": int(
                args.stage_c_pose_candidates
            ),
            "stage_c_prerank_pool": int(args.stage_c_prerank_pool),
            "stage_c_broad_screen_candidates": int(
                args.stage_c_broad_screen_candidates
            ),
            "stage_c_broad_screen_radius_cells": int(
                args.stage_c_broad_screen_radius_cells
            ),
            "stage_c_broad_screen_maximum_points": int(
                args.stage_c_broad_screen_maximum_points
            ),
            "stage_c_broad_screen": (
                "complete_regional_pose_distribution_then_sparse_"
                "chart_balanced_2dgs_feature_render"
            ),
            "stage_c_render_charts": int(args.stage_c_render_charts),
            "stage_c_refinement_charts": int(
                args.stage_c_refinement_charts
            ),
            "stage_c_rounds": int(args.stage_c_rounds),
            "stage_c_maximum_translation_updates": int(
                args.stage_c_maximum_translation_updates
            ),
            "stage_c_pose_mode_translation_radius_m": (
                POSE_MODE_TRANSLATION_RADIUS_M
            ),
            "stage_c_pose_mode_rotation_radius_deg": (
                POSE_MODE_ROTATION_RADIUS_DEG
            ),
            "stage_c_verification_radius_cells": (
                max(
                    int(STAGE_C_VERIFICATION_RADIUS_CELLS),
                    int(np.ceil(32.0 / stage_c_feature_stride)),
                )
            ),
            "stage_c_local_radii_cells": list(
                STAGE_C_LOCAL_RADII_CELLS
            ),
            "stage_c_final_score_uses_fixed_support": True,
            "stage_c_missing_chart_score": (
                "neutral_log_bayes_factor_on_fixed_chart_denominator"
            ),
            "stage_c_minimum_rendered_chart_count": 4,
            "stage_c_minimum_mode_view_direction_cosine": float(
                MINIMUM_MODE_VIEW_DIRECTION_COSINE
            ),
            "stage_c_minimum_view_supported_cell_fraction": float(
                STAGE_C_MINIMUM_VIEW_SUPPORTED_CELL_FRACTION
            ),
            "stage_c_render_identity_scope": (
                "fixed_query_ranked_for_candidate_score_then_pose_"
                "conditioned_d_optimal_for_se3_refinement"
            ),
            "stage_c_pose_mode_support_count_is_diagnostic_only": True,
            "stage_c_ranking_score": (
                "local_displacement_marginal_bayes_factor_then_exact_"
                "zero_flow_pair_null_bayes_factor"
            ),
            "stage_c_prerank_selection": (
                "deduplicated_equal_budget_union_of_exact_zero_flow_and_"
                "local_displacement_marginal_rankings"
            ),
            "stage_c_absolute_view_direction_density_is_diagnostic_only": (
                True
            ),
            "stage_c_pose_pool_family_allocation": (
                "raw_deep64_then_broad4mode_union_consensus32_"
                "factorized32_default_pool256"
            ),
            "stage_c_raw_pose_family": (
                "leading_pair_supports_up_to_eight_modes_then_broad_"
                "pair_first_regional_supports_up_to_four_modes"
            ),
            "stage_c_minimum_regional_source_charts": 2,
            "stage_c_view_direction_angular_std_floor_deg": float(
                VIEW_DIRECTION_ANGULAR_STD_FLOOR_DEG
            ),
            "stage_c_se3_optimization_order": (
                "per_level_unified_joint_rotation_direct_fit_selection_"
                "then_translation_pyramid"
            ),
            "stage_c_direct_rotation_verification": (
                "fixed_chart_fit_select_then_heldout_accept"
            ),
            "stage_c_minimum_rotation_fit_log_gain": 0.01,
            "stage_c_minimum_translation_fit_log_gain": 0.05,
            "stage_c_maximum_committed_translation_updates": 1,
            "stage_c_direct_translation_enabled": False,
            "stage_c_translation_verification": (
                "analytic_disjoint_flow_fixed_chart_consensus"
            ),
            "stage_c_update_evidence": (
                "fixed_chart_exact_zero_flow_pair_null_bayes_factor"
            ),
            "stage_c_direct_rotation_schedule_deg": [
                max(0.25, 1.0 * (0.5 ** iteration))
                for iteration in range(int(args.stage_c_rounds))
            ],
            "stage_c_verification_weighting": (
                "fixed_chart_pair_null_fit_heldout_for_update_"
                "acceptance_only"
            ),
            "maximum_pose_hypotheses": int(args.m3_pose_hypotheses),
            "grouped_frame_mode_search": {
                "modes_per_chart": 8,
                "method": (
                    "bounded_beam_with_planar_pose_branch_compatibility"
                ),
                "two_chart_supports": (
                    f"all_pairs_within_phase_top{int(args.m3_charts)}"
                ),
                "maximum_subsets": {
                    "two_charts": int(args.m3_charts)
                    * max(int(args.m3_charts) - 1, 0)
                    // 2,
                    "three_charts": 48,
                    "four_charts": 24,
                },
                "combinations_per_chart_subset": {
                    "two_charts": (
                        "geometry_top4_union_likelihood_top4"
                    ),
                    "three_charts": (
                        "geometry_top24_union_likelihood_top8"
                    ),
                    "four_charts": (
                        "geometry_top8_union_likelihood_top4"
                    ),
                },
            },
        },
        "queries": query_rows,
        "m1_rows": chart_rows,
        "m1_refined_rows": refined_chart_rows,
        "m1_composite_rows": composite_rows,
        "m1_composite_refined_rows": refined_rows,
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
                        "m1_correct_chart_global_correlation",
                        "m1_correct_chart_local_refinement",
                        "m1_correct_composite_region_global_correlation",
                        "m1_correct_composite_region_local_refinement",
                        "m3_correct_metric_charts_predicted_frame_pose",
                        "m3_correct_metric_charts_raw_frame_pose",
                        "m3_retrieved_metric_charts_predicted_frame_pose",
                        "m3_retrieved_se3_consensus_mode_pose",
                        "m3_correct_composite_region_predicted_frame_pose",
                        "d1_region_to_chart_coverage",
                        "d2_actual_retrieval_gt_frame_pose",
                        "d3_correct_identity_predicted_frame_oracle_mode_pose",
                        "d4_actual_retrieval_predicted_frame_oracle_mode_pose",
                        "d6_gt_affine_vs_homography",
                        "stage_c_runtime_atlas_alignment",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
