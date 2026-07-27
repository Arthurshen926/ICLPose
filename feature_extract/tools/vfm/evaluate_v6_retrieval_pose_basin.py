"""Evaluate V6 maplet retrieval and experimental coarse-pose basin coverage.

This is deliberately separate from the oracle-maplet local-correlation
evaluator.  Ground truth is used only after hypotheses have been generated, so
the report measures whether the runtime retrieval/proposal chain can actually
enter the fine alignment basin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
    _load_raw_final,
)
from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.cambridge_pose_lattice import (
    parse_cambridge_pose_file,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_pose_proposal import (
    propose_regional_center_pseudo_pnp_poses,
)
from feature_extract.vfm.localization_v6.maplet_footprint_pose import (
    propose_maplet_footprint_poses,
    score_maplet_footprint_pose,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    retrieve_candidate_groups,
)
from feature_extract.vfm.localization_v6.metric_encoder import (
    load_v6_metric_encoder,
)
from feature_extract.vfm.localization_v6.maplet_pose_voting import (
    AnonymousMapletPoseVoteBank,
    vote_maplet_poses,
)
from feature_extract.vfm.localization_v6.probability_calibration import (
    V6ProbabilityCalibration,
    probability_calibration_sha256,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.query_to_3d_matching import (
    camera_matrix_and_distortion,
    pnp_pose_error,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    encode_radio_final_regions,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--query_camera_manifest", required=True)
    parser.add_argument("--surface_mapper_checkpoint", default="")
    parser.add_argument("--maplets", required=True)
    parser.add_argument(
        "--probability_calibration",
        required=True,
        help=(
            "Frozen trajectory-disjoint identity/spatial null calibration."
        ),
    )
    parser.add_argument(
        "--spatial_maplets",
        default="",
        help=(
            "Optional separate RADIO spatial bank. The primary bank retrieves "
            "maplet identity; this bank selects within-maplet surface modes."
        ),
    )
    parser.add_argument(
        "--spatial_metric_encoder_checkpoint",
        default="",
        help=(
            "Query encoder for an exact_canonical_v6_metric_surface_texture "
            "spatial bank. The primary RADIO bank still retrieves identity."
        ),
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--query_trajectory_ids", nargs="*", default=[])
    parser.add_argument("--preliminary_candidates", type=int, default=64)
    parser.add_argument("--maximum_maplets", type=int, default=64)
    parser.add_argument(
        "--maximum_components_per_maplet",
        type=int,
        default=16,
        help="Distinct query-conditioned surface-location modes retained per maplet.",
    )
    parser.add_argument(
        "--component_nms_distance_m", type=float, default=0.02
    )
    parser.add_argument(
        "--proposal_trials",
        type=int,
        default=0,
        help=(
            "Regional-center pseudo-PnP diagnostic only: optional stochastic "
            "grouped hypotheses. Ignored by region-level coarse methods."
        ),
    )
    parser.add_argument(
        "--proposal_refinement_candidates", type=int, default=32
    )
    parser.add_argument(
        "--proposal_em_iterations",
        type=int,
        default=0,
        help=(
            "Regional-center pseudo-PnP diagnostic only: experimental "
            "likelihood EM. Ignored by region-level coarse methods."
        ),
    )
    parser.add_argument("--maximum_modes", type=int, default=16)
    parser.add_argument("--pose_vote_bank", default="")
    parser.add_argument(
        "--coarse_pose_method",
        choices=(
            "maplet_footprint",
            "anonymous_view_mode",
            "regional_center_pseudo_pnp",
        ),
        default="anonymous_view_mode",
        help=(
            "Anonymous view-mode voting is the region-level default. "
            "Footprint CEM remains experimental; regional-center pseudo-PnP "
            "is a diagnostic baseline only."
        ),
    )
    parser.add_argument("--footprint_cem_iterations", type=int, default=3)
    parser.add_argument("--footprint_particles_per_mode", type=int, default=64)
    parser.add_argument(
        "--footprint_candidates_per_region", type=int, default=16
    )
    parser.add_argument("--atlas_geometry", default="")
    parser.add_argument("--visibility_contributor_dir", default="")
    parser.add_argument("--image_root", default="")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument(
        "--allow_reference_trajectory_diagnostic",
        action="store_true",
        help=(
            "Allow query trajectories used by probability calibration or "
            "pose-vote map construction. This is a replay diagnostic and "
            "must not be reported as independent validation."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _region_geometry(
    token_xy: np.ndarray,
    *,
    token_width: int,
    token_height: int,
    image_width: int,
    image_height: int,
    config: RadioFinalRegionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray(
        [image_width / token_width, image_height / token_height],
        dtype=np.float32,
    )
    center = (np.asarray(token_xy, dtype=np.float32) + 0.5) * scale - 0.5
    weights = np.asarray(config.pool_weights, dtype=np.float64)
    sizes = np.asarray(config.pool_sizes, dtype=np.float64)
    effective_half_size = 0.5 * float(
        np.sum(weights * sizes) / max(np.sum(weights), 1e-8)
    )
    # The extent represents the descriptor's pooled support, not a keypoint
    # measurement.  It therefore enters proposal likelihood as uncertainty.
    extent = np.broadcast_to(
        scale[None] * effective_half_size, center.shape
    ).copy()
    return center, extent


def _geometric_diversity(
    maplet_ids: np.ndarray, bank: SurfaceRetrievalMapletBank
) -> dict[str, float]:
    row_by_id = {
        int(value): int(row)
        for row, value in enumerate(bank.maplet_ids.tolist())
    }
    rows = np.asarray(
        [row_by_id[int(value)] for value in maplet_ids if int(value) in row_by_id],
        dtype=np.int64,
    )
    if rows.size < 2:
        return {
            "candidate_center_rank": 0.0,
            "candidate_center_spread_m": 0.0,
            "candidate_normal_resultant": 1.0,
        }
    centered = bank.centers[rows] - np.mean(bank.centers[rows], axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.sum(singular > max(float(singular[0]), 1e-8) * 0.05))
    return {
        "candidate_center_rank": float(rank),
        "candidate_center_spread_m": float(
            np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
        ),
        "candidate_normal_resultant": float(
            np.linalg.norm(np.mean(bank.normals[rows], axis=0))
        ),
    }


def _binary_average_precision(
    labels: list[bool], scores: list[float]
) -> float | None:
    if not labels or not any(labels):
        return None
    label = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-score, kind="stable")
    ranked = label[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(np.sum(precision[ranked]) / np.sum(ranked))


def _region_identity_diagnostics(
    groups,
    view,
    atlas: MapletFeatureAtlasBank,
    bank: SurfaceRetrievalMapletBank,
) -> dict[str, float | int]:
    """Separate region identity recall from the coarse-pose observation model."""

    surface_ids = np.asarray(view.visible_rows, dtype=np.int64)
    surface_xy = np.asarray(view.image_xy, dtype=np.float64)
    flat_cells = atlas.height * atlas.width
    surface_maplet_rows = surface_ids // flat_cells
    surface_maplet_ids = atlas.maplet_ids[surface_maplet_rows]
    row_by_id = {
        int(value): int(row)
        for row, value in enumerate(bank.maplet_ids.tolist())
    }
    labeled = 0
    hits = {1: 0, 5: 0, 64: 0}
    dominant_hits = {1: 0, 5: 0, 64: 0}
    correct_mass = []
    maplet_center_residual = []
    component_center_residual = []
    component_map_residual = []
    component_oracle_residual = []
    surface_center_residual = []
    identity_null_labels: list[bool] = []
    identity_null_scores: list[float] = []
    spatial_null_labels: list[bool] = []
    spatial_null_scores: list[float] = []
    spatial_null_true_identity_labels: list[bool] = []
    spatial_null_true_identity_scores: list[float] = []
    spatial_surface_errors = []
    spatial_rank1 = []
    spatial_rank5 = []
    spatial_mode_entropy = []
    for group in groups:
        extent = np.maximum(
            np.asarray(group.query_region_extent, dtype=np.float64), 1.0
        )
        normalized = np.abs(
            surface_xy
            - np.asarray(group.query_region_xy, dtype=np.float64)[None]
        ) / extent[None]
        inside = np.max(normalized, axis=1) <= 1.0
        identity_null_labels.append(not bool(np.any(inside)))
        identity_null_scores.append(float(group.null_probability))
        if not np.any(inside):
            if group.spatial_null_probabilities is not None:
                spatial_null_scores.extend(
                    np.asarray(
                        group.spatial_null_probabilities, dtype=np.float64
                    ).tolist()
                )
                spatial_null_labels.extend(
                    [True] * int(group.maplet_ids.size)
                )
            continue
        local_ids, local_counts = np.unique(
            surface_maplet_ids[inside], return_counts=True
        )
        order = np.argsort(-local_counts, kind="stable")
        # Very broad overlapping maplets can cover the same RADIO region.
        # Retaining the strongest eight labels avoids declaring every nearby
        # facade a correct identity while preserving genuine multilayer cases.
        true_ids = local_ids[order[:8]]
        dominant_id = int(local_ids[order[0]])
        labeled += 1
        candidate_ids = np.asarray(group.maplet_ids, dtype=np.int64)
        true_candidate = np.isin(candidate_ids, true_ids)
        for rank in hits:
            hits[rank] += int(
                np.any(true_candidate[: min(rank, true_candidate.size)])
            )
            dominant_hits[rank] += int(
                dominant_id
                in candidate_ids[: min(rank, candidate_ids.size)]
            )
        correct_mass.append(
            float(np.sum(group.probabilities[true_candidate]))
        )
        correct_rows = np.flatnonzero(true_candidate)
        if (
            group.component_offsets is not None
            and group.component_centers is not None
            and group.component_probabilities is not None
            and group.spatial_null_probabilities is not None
        ):
            offsets = np.asarray(group.component_offsets, dtype=np.int64)
            component_centers = np.asarray(
                group.component_centers, dtype=np.float64
            )
            component_probability = np.asarray(
                group.component_probabilities, dtype=np.float64
            )
            spatial_null = np.asarray(
                group.spatial_null_probabilities, dtype=np.float64
            )
            local_surface_ids = surface_ids[inside]
            local_surface_maplet_ids = surface_maplet_ids[inside]
            local_surface_xyz = atlas.xyz.reshape(-1, 3)[local_surface_ids]
            for candidate, maplet_id in enumerate(candidate_ids.tolist()):
                begin, end = (
                    int(offsets[candidate]),
                    int(offsets[candidate + 1]),
                )
                target = local_surface_maplet_ids == int(maplet_id)
                identity_is_true = bool(np.any(target))
                resolved = False
                map_error = float("inf")
                if end > begin and np.any(target):
                    probability = component_probability[begin:end]
                    order_by_probability = np.argsort(
                        -probability, kind="stable"
                    )
                    distance = np.linalg.norm(
                        component_centers[begin:end, None]
                        - local_surface_xyz[target][None],
                        axis=2,
                    )
                    per_mode_error = np.min(distance, axis=1)
                    map_error = float(
                        per_mode_error[order_by_probability[0]]
                    )
                    resolved = map_error <= 0.30
                    spatial_surface_errors.append(map_error)
                    spatial_rank1.append(
                        float(per_mode_error[order_by_probability[0]] <= 0.30)
                    )
                    spatial_rank5.append(
                        float(
                            np.any(
                                per_mode_error[
                                    order_by_probability[:5]
                                ]
                                <= 0.30
                            )
                        )
                    )
                    distribution = np.r_[
                        probability, spatial_null[candidate]
                    ]
                    distribution /= max(float(np.sum(distribution)), 1e-12)
                    spatial_mode_entropy.append(
                        float(
                            -np.sum(
                                distribution
                                * np.log(np.maximum(distribution, 1e-12))
                            )
                        )
                    )
                spatial_null_labels.append(not resolved)
                spatial_null_scores.append(float(spatial_null[candidate]))
                if identity_is_true:
                    spatial_null_true_identity_labels.append(not resolved)
                    spatial_null_true_identity_scores.append(
                        float(spatial_null[candidate])
                    )
        if correct_rows.size:
            selected_ids = candidate_ids[correct_rows]
            bank_rows = np.asarray(
                [row_by_id[int(value)] for value in selected_ids],
                dtype=np.int64,
            )
            maplet_pixels, maplet_depth = project_world_points(
                bank.centers[bank_rows], view.pose_w2c, view.camera
            )
            maplet_valid = (
                np.isfinite(maplet_pixels).all(axis=1)
                & np.isfinite(maplet_depth)
                & (maplet_depth > 0.0)
            )
            if np.any(maplet_valid):
                maplet_center_residual.append(
                    float(
                        np.min(
                            np.linalg.norm(
                                maplet_pixels[maplet_valid]
                                - np.asarray(group.query_region_xy)[None],
                                axis=1,
                            )
                        )
                    )
                )
            if group.candidate_centers is not None:
                component_pixels, component_depth = project_world_points(
                    np.asarray(group.candidate_centers)[correct_rows],
                    view.pose_w2c,
                    view.camera,
                )
                component_valid = (
                    np.isfinite(component_pixels).all(axis=1)
                    & np.isfinite(component_depth)
                    & (component_depth > 0.0)
                )
                if np.any(component_valid):
                    component_center_residual.append(
                        float(
                            np.min(
                                np.linalg.norm(
                                    component_pixels[component_valid]
                                    - np.asarray(group.query_region_xy)[None],
                                    axis=1,
                                )
                            )
                        )
                    )
            if (
                group.component_offsets is not None
                and group.component_centers is not None
                and group.component_probabilities is not None
            ):
                offsets = np.asarray(group.component_offsets, dtype=np.int64)
                all_centers = np.asarray(
                    group.component_centers, dtype=np.float64
                )
                conditional = np.asarray(
                    group.component_probabilities, dtype=np.float64
                )
                map_centers = []
                oracle_centers = []
                for candidate in correct_rows.tolist():
                    begin, end = (
                        int(offsets[candidate]),
                        int(offsets[candidate + 1]),
                    )
                    if end <= begin:
                        continue
                    local = conditional[begin:end]
                    map_centers.append(
                        all_centers[begin + int(np.argmax(local))]
                    )
                    oracle_centers.extend(all_centers[begin:end])
                for values, destination in (
                    (map_centers, component_map_residual),
                    (oracle_centers, component_oracle_residual),
                ):
                    pixels, depth = project_world_points(
                        np.asarray(values), view.pose_w2c, view.camera
                    )
                    valid = (
                        np.isfinite(pixels).all(axis=1)
                        & np.isfinite(depth)
                        & (depth > 0.0)
                    )
                    if np.any(valid):
                        destination.append(
                            float(
                                np.min(
                                    np.linalg.norm(
                                        pixels[valid]
                                        - np.asarray(
                                            group.query_region_xy
                                        )[None],
                                        axis=1,
                                    )
                                )
                            )
                        )
        local_surface_ids = surface_ids[inside]
        local_surface_maplet_ids = surface_maplet_ids[inside]
        surface_centers = []
        for maplet_id in true_ids.tolist():
            keep = local_surface_maplet_ids == int(maplet_id)
            rows = local_surface_ids[keep]
            if rows.size:
                surface_centers.append(
                    np.mean(
                        atlas.xyz.reshape(-1, 3)[rows],
                        axis=0,
                    )
                )
        if surface_centers:
            pixels, depth = project_world_points(
                np.asarray(surface_centers),
                view.pose_w2c,
                view.camera,
            )
            valid = (
                np.isfinite(pixels).all(axis=1)
                & np.isfinite(depth)
                & (depth > 0.0)
            )
            if np.any(valid):
                surface_center_residual.append(
                    float(
                        np.min(
                            np.linalg.norm(
                                pixels[valid]
                                - np.asarray(group.query_region_xy)[None],
                                axis=1,
                            )
                        )
                    )
                )
    denominator = max(labeled, 1)
    return {
        "labeled_region_count": int(labeled),
        "region_label_coverage": float(labeled / max(len(groups), 1)),
        "region_true_maplet_recall_at_1": float(hits[1] / denominator),
        "region_true_maplet_recall_at_5": float(hits[5] / denominator),
        "region_true_maplet_recall_at_64": float(hits[64] / denominator),
        "region_dominant_maplet_recall_at_1": float(
            dominant_hits[1] / denominator
        ),
        "region_dominant_maplet_recall_at_5": float(
            dominant_hits[5] / denominator
        ),
        "region_dominant_maplet_recall_at_64": float(
            dominant_hits[64] / denominator
        ),
        "identity_null_auprc": _binary_average_precision(
            identity_null_labels, identity_null_scores
        ),
        "spatial_null_auprc": _binary_average_precision(
            spatial_null_labels, spatial_null_scores
        ),
        "spatial_null_true_identity_auprc": _binary_average_precision(
            spatial_null_true_identity_labels,
            spatial_null_true_identity_scores,
        ),
        "spatial_null_example_count": int(len(spatial_null_labels)),
        "spatial_null_true_identity_example_count": int(
            len(spatial_null_true_identity_labels)
        ),
        "spatial_null_prevalence": (
            float(np.mean(spatial_null_labels))
            if spatial_null_labels
            else None
        ),
        "spatial_null_true_identity_prevalence": (
            float(np.mean(spatial_null_true_identity_labels))
            if spatial_null_true_identity_labels
            else None
        ),
        "spatial_surface_error_median_m": (
            float(np.median(spatial_surface_errors))
            if spatial_surface_errors
            else None
        ),
        "spatial_surface_error_sample_count": int(
            len(spatial_surface_errors)
        ),
        "spatial_rank1_within_30cm": (
            float(np.mean(spatial_rank1)) if spatial_rank1 else None
        ),
        "spatial_rank5_within_30cm": (
            float(np.mean(spatial_rank5)) if spatial_rank5 else None
        ),
        "spatial_mode_entropy_mean": (
            float(np.mean(spatial_mode_entropy))
            if spatial_mode_entropy
            else None
        ),
        "region_true_probability_mass_mean": (
            float(np.mean(correct_mass)) if correct_mass else 0.0
        ),
        "correct_maplet_center_reprojection_median_px": (
            float(np.median(maplet_center_residual))
            if maplet_center_residual
            else None
        ),
        "correct_component_center_reprojection_median_px": (
            float(np.median(component_center_residual))
            if component_center_residual
            else None
        ),
        "correct_map_component_reprojection_median_px": (
            float(np.median(component_map_residual))
            if component_map_residual
            else None
        ),
        "correct_oracle_component_reprojection_median_px": (
            float(np.median(component_oracle_residual))
            if component_oracle_residual
            else None
        ),
        "oracle_surface_center_reprojection_median_px": (
            float(np.median(surface_center_residual))
            if surface_center_residual
            else None
        ),
    }


def _diagnostic_pnp(
    xyz: list[np.ndarray],
    xy: list[np.ndarray],
    camera,
    gt_pose_w2c: np.ndarray,
) -> dict[str, float | int | None]:
    if len(xyz) < 6:
        return {
            "correspondence_count": len(xyz),
            "translation_m": None,
            "rotation_deg": None,
            "inlier_count": 0,
        }
    matrix, distortion = camera_matrix_and_distortion(camera)
    # OpenCV's RANSAC RNG is global and otherwise makes layered oracle rows
    # vary between identical evaluator runs.
    cv2.setRNGSeed(194917)
    success, rotation, translation, inliers = cv2.solvePnPRansac(
        np.asarray(xyz, dtype=np.float64),
        np.asarray(xy, dtype=np.float64),
        matrix,
        distortion,
        iterationsCount=2000,
        reprojectionError=8.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return {
            "correspondence_count": len(xyz),
            "translation_m": None,
            "rotation_deg": None,
            "inlier_count": 0,
        }
    if inliers is not None and len(inliers) >= 6:
        keep = np.asarray(inliers, dtype=np.int64).reshape(-1)
        rotation, translation = cv2.solvePnPRefineLM(
            np.asarray(xyz, dtype=np.float64)[keep],
            np.asarray(xy, dtype=np.float64)[keep],
            matrix,
            distortion,
            rotation,
            translation,
        )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = cv2.Rodrigues(rotation)[0]
    pose[:3, 3] = np.asarray(translation).reshape(3)
    error = pnp_pose_error(pose, gt_pose_w2c)
    return {
        "correspondence_count": len(xyz),
        "translation_m": float(error.translation_m),
        "rotation_deg": float(error.rotation_deg),
        "inlier_count": int(len(inliers)) if inliers is not None else 0,
    }


def _coarse_observation_oracles(
    groups,
    view,
    atlas: MapletFeatureAtlasBank,
    identity_bank: SurfaceRetrievalMapletBank,
    spatial_bank: SurfaceRetrievalMapletBank,
) -> dict[str, dict[str, float | int | None]]:
    """Measure which coarse observation model first destroys the pose basin."""

    surface_ids = np.asarray(view.visible_rows, dtype=np.int64)
    surface_xy = np.asarray(view.image_xy, dtype=np.float64)
    surface_xyz = atlas.xyz.reshape(-1, 3)[surface_ids]
    flat_cells = atlas.height * atlas.width
    surface_maplet_ids = atlas.maplet_ids[surface_ids // flat_cells]
    bank_row_by_id = {
        int(value): int(row)
        for row, value in enumerate(identity_bank.maplet_ids.tolist())
    }
    spatial_row_by_id = {
        int(value): int(row)
        for row, value in enumerate(spatial_bank.maplet_ids.tolist())
    }
    exact: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
    maplet_center: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
    oracle_component: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
    candidate_oracle_component: dict[
        tuple[int, float, float, float],
        tuple[float, np.ndarray, np.ndarray],
    ] = {}
    map_component: dict[
        tuple[int, float, float, float],
        tuple[float, np.ndarray, np.ndarray],
    ] = {}
    for group in groups:
        query_xy = np.asarray(group.query_region_xy, dtype=np.float64)
        extent = np.maximum(
            np.asarray(group.query_region_extent, dtype=np.float64), 1.0
        )
        inside = np.max(np.abs(surface_xy - query_xy[None]) / extent[None], axis=1)
        rows = np.flatnonzero(inside <= 1.0)
        if rows.size == 0:
            continue
        nearest = int(
            rows[
                np.argmin(
                    np.linalg.norm(surface_xy[rows] - query_xy[None], axis=1)
                )
            ]
        )
        surface_id = int(surface_ids[nearest])
        surface_residual = float(
            np.linalg.norm(surface_xy[nearest] - query_xy)
        )
        previous = exact.get(surface_id)
        if previous is None or surface_residual < previous[0]:
            exact[surface_id] = (
                surface_residual,
                np.asarray(surface_xyz[nearest], dtype=np.float64),
                query_xy,
            )
        maplet_id = int(surface_maplet_ids[nearest])
        bank_row = bank_row_by_id.get(maplet_id)
        if bank_row is None:
            continue
        projected, depth = project_world_points(
            identity_bank.centers[bank_row : bank_row + 1],
            view.pose_w2c,
            view.camera,
        )
        if np.isfinite(projected).all() and float(depth[0]) > 0.0:
            residual = float(np.linalg.norm(projected[0] - query_xy))
            previous = maplet_center.get(maplet_id)
            if previous is None or residual < previous[0]:
                maplet_center[maplet_id] = (
                    residual,
                    np.asarray(
                        identity_bank.centers[bank_row], dtype=np.float64
                    ),
                    query_xy,
                )
        spatial_row = spatial_row_by_id.get(maplet_id)
        if spatial_row is not None:
            begin, end = (
                int(spatial_bank.descriptor_offsets[spatial_row]),
                int(spatial_bank.descriptor_offsets[spatial_row + 1]),
            )
            component_xyz = np.asarray(
                spatial_bank.descriptor_centers[begin:end],
                dtype=np.float64,
            )
            projected, depth = project_world_points(
                component_xyz, view.pose_w2c, view.camera
            )
            valid = (
                np.isfinite(projected).all(axis=1)
                & np.isfinite(depth)
                & (depth > 0.0)
            )
            if np.any(valid):
                local_rows = np.flatnonzero(valid)
                local = int(
                    local_rows[
                        np.argmin(
                            np.linalg.norm(
                                projected[local_rows] - query_xy[None],
                                axis=1,
                            )
                        )
                    ]
                )
                component_id = begin + local
                residual = float(
                    np.linalg.norm(projected[local] - query_xy)
                )
                previous = oracle_component.get(component_id)
                if previous is None or residual < previous[0]:
                    oracle_component[component_id] = (
                        residual,
                        component_xyz[local],
                        query_xy,
                    )
        candidate = np.flatnonzero(
            np.asarray(group.maplet_ids, dtype=np.int64) == maplet_id
        )
        if (
            candidate.size
            and group.component_offsets is not None
            and group.component_centers is not None
            and group.component_probabilities is not None
        ):
            candidate = int(candidate[0])
            offsets = np.asarray(group.component_offsets, dtype=np.int64)
            local_begin, local_end = (
                int(offsets[candidate]),
                int(offsets[candidate + 1]),
            )
            if local_end <= local_begin:
                continue
            candidate_positions = np.asarray(
                group.component_centers[local_begin:local_end],
                dtype=np.float64,
            )
            candidate_pixels, candidate_depth = project_world_points(
                candidate_positions, view.pose_w2c, view.camera
            )
            candidate_valid = (
                np.isfinite(candidate_pixels).all(axis=1)
                & np.isfinite(candidate_depth)
                & (candidate_depth > 0.0)
            )
            if np.any(candidate_valid):
                valid_rows = np.flatnonzero(candidate_valid)
                oracle_local = int(
                    valid_rows[
                        np.argmin(
                            np.linalg.norm(
                                candidate_pixels[valid_rows]
                                - query_xy[None],
                                axis=1,
                            )
                        )
                    ]
                )
                oracle_position = candidate_positions[oracle_local]
                oracle_residual = float(
                    np.linalg.norm(
                        candidate_pixels[oracle_local] - query_xy
                    )
                )
                oracle_key = (
                    maplet_id,
                    *np.round(oracle_position, decimals=4).tolist(),
                )
                previous = candidate_oracle_component.get(oracle_key)
                if previous is None or oracle_residual < previous[0]:
                    candidate_oracle_component[oracle_key] = (
                        oracle_residual,
                        oracle_position,
                        query_xy,
                    )
            probability = np.asarray(
                group.component_probabilities[local_begin:local_end],
                dtype=np.float64,
            )
            local = int(np.argmax(probability))
            position = np.asarray(
                group.component_centers[local_begin + local], dtype=np.float64
            )
            projected, depth = project_world_points(
                position[None], view.pose_w2c, view.camera
            )
            if np.isfinite(projected).all() and float(depth[0]) > 0.0:
                residual = float(np.linalg.norm(projected[0] - query_xy))
                component_key = (
                    maplet_id,
                    *np.round(position, decimals=4).tolist(),
                )
                previous = map_component.get(component_key)
                if previous is None or residual < previous[0]:
                    map_component[component_key] = (
                        residual,
                        position,
                        query_xy,
                    )

    def solve(values):
        ordered = [values[key] for key in sorted(values)]
        return _diagnostic_pnp(
            [row[1] for row in ordered],
            [row[2] for row in ordered],
            view.camera,
            view.pose_w2c,
        )

    return {
        "nearest_exact_visible_surface": solve(exact),
        "true_maplet_center": solve(maplet_center),
        "true_maplet_oracle_spatial_component": solve(oracle_component),
        "retrieved_true_maplet_oracle_spatial_component": solve(
            candidate_oracle_component
        ),
        "true_maplet_descriptor_map_component": solve(map_component),
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"query_count": 0}

    def fraction(predicate) -> float:
        return float(np.mean([bool(predicate(row)) for row in rows]))

    success_rows = [row for row in rows if int(row["hypothesis_count"]) > 0]
    output: dict[str, object] = {
        "query_count": len(rows),
        "proposal_success_fraction": float(len(success_rows) / len(rows)),
        "median_hypothesis_count": float(
            np.median([int(row["hypothesis_count"]) for row in rows])
        ),
        "probability_conservation_max_error": float(
            np.max([float(row["probability_conservation_error"]) for row in rows])
        ),
        "spatial_probability_conservation_max_error": float(
            np.max(
                [
                    float(row["spatial_probability_conservation_error"])
                    for row in rows
                ]
            )
        ),
        "spatial_available_candidate_fraction": float(
            np.mean(
                [
                    float(row["spatial_available_candidate_fraction"])
                    for row in rows
                ]
            )
        ),
    }
    exact_rows = [row for row in rows if "visible_surface_coverage" in row]
    if exact_rows:
        output.update(
            {
                "exact_visibility_query_count": len(exact_rows),
                "any_visible_maplet_hit": float(
                    np.mean(
                        [
                            int(row["visible_maplet_hit_count"]) > 0
                            for row in exact_rows
                        ]
                    )
                ),
                "visible_maplet_recall_mean": float(
                    np.mean(
                        [float(row["visible_maplet_recall"]) for row in exact_rows]
                    )
                ),
                "visible_surface_coverage_mean": float(
                    np.mean(
                        [
                            float(row["visible_surface_coverage"])
                            for row in exact_rows
                        ]
                    )
                ),
                "geometrically_independent_visible_hits_mean": float(
                    np.mean(
                        [
                            float(row["geometrically_independent_visible_hits"])
                            for row in exact_rows
                        ]
                    )
                ),
            }
        )
        for key in (
            "region_label_coverage",
            "region_true_maplet_recall_at_1",
            "region_true_maplet_recall_at_5",
            "region_true_maplet_recall_at_64",
            "region_dominant_maplet_recall_at_1",
            "region_dominant_maplet_recall_at_5",
            "region_dominant_maplet_recall_at_64",
            "region_true_probability_mass_mean",
            "identity_null_auprc",
            "spatial_null_auprc",
            "spatial_null_true_identity_auprc",
            "spatial_null_prevalence",
            "spatial_null_true_identity_prevalence",
            "spatial_surface_error_median_m",
            "spatial_rank1_within_30cm",
            "spatial_rank5_within_30cm",
            "spatial_mode_entropy_mean",
            "correct_maplet_center_reprojection_median_px",
            "correct_component_center_reprojection_median_px",
            "correct_map_component_reprojection_median_px",
            "correct_oracle_component_reprojection_median_px",
            "oracle_surface_center_reprojection_median_px",
        ):
            values = [
                float(row[key])
                for row in exact_rows
                if row.get(key) is not None
            ]
            output[key] = float(np.mean(values)) if values else None
        for key in (
            "spatial_null_example_count",
            "spatial_null_true_identity_example_count",
            "spatial_surface_error_sample_count",
        ):
            output[key] = int(
                np.sum(
                    [
                        int(row.get(key, 0))
                        for row in exact_rows
                    ]
                )
            )
        oracle_names = sorted(
            {
                name
                for row in exact_rows
                for name in row.get("coarse_observation_oracles", {})
            }
        )
        output["coarse_observation_oracle_summary"] = {}
        for name in oracle_names:
            oracle_rows = [
                row["coarse_observation_oracles"][name]
                for row in exact_rows
                if name in row.get("coarse_observation_oracles", {})
            ]
            valid = [
                value
                for value in oracle_rows
                if value.get("translation_m") is not None
                and value.get("rotation_deg") is not None
            ]
            output["coarse_observation_oracle_summary"][name] = {
                "solved_fraction": float(len(valid) / len(exact_rows)),
                "translation_median_m": (
                    float(
                        np.median(
                            [float(value["translation_m"]) for value in valid]
                        )
                    )
                    if valid
                    else None
                ),
                "rotation_median_deg": (
                    float(
                        np.median(
                            [float(value["rotation_deg"]) for value in valid]
                        )
                    )
                    if valid
                    else None
                ),
                "within_30cm_3deg": float(
                    np.mean(
                        [
                            value.get("translation_m") is not None
                            and float(value["translation_m"]) <= 0.30
                            and float(value["rotation_deg"]) <= 3.0
                            for value in oracle_rows
                        ]
                    )
                ),
                "correspondence_count_median": float(
                    np.median(
                        [
                            int(value["correspondence_count"])
                            for value in oracle_rows
                        ]
                    )
                ),
            }
    for rank in (1, 5, 16):
        for translation_m in (0.05, 0.20, 0.30):
            key = f"oracle_at_{rank}_{int(translation_m * 100)}cm_3deg"
            output[key] = fraction(
                lambda row, rank=rank, translation_m=translation_m: any(
                    float(error["translation_m"]) <= translation_m
                    and float(error["rotation_deg"]) <= 3.0
                    for error in row["hypothesis_errors"][:rank]
                )
            )
    output["top1_30cm_3deg"] = fraction(
        lambda row: bool(row["hypothesis_errors"])
        and float(row["hypothesis_errors"][0]["translation_m"]) <= 0.30
        and float(row["hypothesis_errors"][0]["rotation_deg"]) <= 3.0
    )
    top1_translation = [
        float(row["hypothesis_errors"][0]["translation_m"])
        for row in success_rows
    ]
    top1_rotation = [
        float(row["hypothesis_errors"][0]["rotation_deg"])
        for row in success_rows
    ]
    output["top1_translation_median_m"] = (
        float(np.median(top1_translation)) if top1_translation else None
    )
    output["top1_translation_p90_m"] = (
        float(np.quantile(top1_translation, 0.9)) if top1_translation else None
    )
    output["top1_rotation_median_deg"] = (
        float(np.median(top1_rotation)) if top1_rotation else None
    )
    output["top1_rotation_p90_deg"] = (
        float(np.quantile(top1_rotation, 0.9)) if top1_rotation else None
    )
    footprint_margins = [
        float(row["diagnostic_footprint_gt_minus_best_proposal"])
        for row in rows
        if row.get("diagnostic_footprint_gt_minus_best_proposal")
        is not None
    ]
    output["diagnostic_footprint_gt_preferred_fraction"] = (
        float(np.mean(np.asarray(footprint_margins) > 0.0))
        if footprint_margins
        else None
    )
    output["diagnostic_footprint_gt_minus_best_proposal_median"] = (
        float(np.median(footprint_margins))
        if footprint_margins
        else None
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_json)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite V6 retrieval-pose report")
    bank = SurfaceRetrievalMapletBank.load_npz(Path(args.maplets))
    spatial_bank = (
        SurfaceRetrievalMapletBank.load_npz(Path(args.spatial_maplets))
        if str(args.spatial_maplets)
        else bank
    )
    probability_calibration_path = Path(args.probability_calibration)
    probability_calibration = V6ProbabilityCalibration.load_json(
        probability_calibration_path
    )
    calibration_metadata = dict(probability_calibration.metadata)
    expected_identity_hash = str(
        calibration_metadata.get("identity_bank_sha256", "")
    )
    expected_spatial_hash = str(
        calibration_metadata.get("spatial_bank_sha256", "")
    )
    observed_identity_hash = hashlib.sha256(
        Path(args.maplets).read_bytes()
    ).hexdigest()
    observed_spatial_hash = hashlib.sha256(
        Path(args.spatial_maplets or args.maplets).read_bytes()
    ).hexdigest()
    if expected_identity_hash and expected_identity_hash != observed_identity_hash:
        raise ValueError("probability calibration identity-bank lineage differs")
    if expected_spatial_hash and expected_spatial_hash != observed_spatial_hash:
        raise ValueError("probability calibration spatial-bank lineage differs")
    pose_vote_bank = (
        AnonymousMapletPoseVoteBank.load_npz(Path(args.pose_vote_bank))
        if str(args.pose_vote_bank)
        else None
    )
    if (
        str(args.coarse_pose_method)
        in {"maplet_footprint", "anonymous_view_mode"}
        and pose_vote_bank is None
    ):
        raise ValueError(
            f"{args.coarse_pose_method} requires --pose_vote_bank"
        )
    exact_visibility = {}
    atlas_geometry = None
    if str(args.atlas_geometry) or str(args.visibility_contributor_dir):
        if not (
            str(args.atlas_geometry)
            and str(args.visibility_contributor_dir)
            and str(args.image_root)
        ):
            raise ValueError(
                "exact visibility requires atlas_geometry, "
                "visibility_contributor_dir and image_root"
            )
        atlas_geometry = MapletFeatureAtlasBank.load_npz(
            Path(args.atlas_geometry)
        )
        exact_visibility = {
            view.image_id: view
            for view in _load_views(
                Path(args.visibility_contributor_dir),
                atlas_geometry,
                Path(args.image_root),
            )
        }
    spatial_metric_model = None
    spatial_metric_level = ""
    spatial_metric_representation = str(
        (spatial_bank.metadata or {}).get("representation", "")
    )
    if (
        spatial_metric_representation
        == "exact_canonical_v6_metric_surface_texture"
    ):
        if not str(args.spatial_metric_encoder_checkpoint):
            raise ValueError(
                "metric spatial bank requires its query metric encoder"
            )
        spatial_metric_path = Path(args.spatial_metric_encoder_checkpoint)
        spatial_metric_model, _metric_metadata = load_v6_metric_encoder(
            spatial_metric_path, device=str(args.device)
        )
        spatial_metric_model.eval()
        digest = hashlib.sha256(spatial_metric_path.read_bytes()).hexdigest()
        expected_digest = str(
            (spatial_bank.metadata or {}).get(
                "query_metric_encoder_sha256", ""
            )
        )
        if not expected_digest or digest != expected_digest:
            raise ValueError(
                "metric spatial bank/query encoder lineage differs"
            )
        spatial_metric_level = str(
            (spatial_bank.metadata or {}).get("metric_feature_level", "")
        )
        if spatial_metric_level not in {"fine", "middle", "coarse"}:
            raise ValueError("metric spatial bank has invalid feature level")
        if not str(args.image_root):
            raise ValueError("metric spatial query encoding requires image_root")
    mapper = None
    mapper_metadata: dict[str, object] = {}
    if bank.query_projection is None or (
        spatial_metric_model is None
        and spatial_bank.query_projection is None
    ):
        if not str(args.surface_mapper_checkpoint):
            raise ValueError(
                "retrieval bank requires a surface mapper checkpoint"
            )
        mapper, loaded_mapper_metadata = load_surface_maplet_mapper(
            Path(args.surface_mapper_checkpoint), device=str(args.device)
        )
        mapper_metadata = dict(loaded_mapper_metadata)
    config = RadioFinalRegionConfig(
        pool_sizes=tuple(mapper_metadata.get("pool_sizes", (1, 3, 5, 9))),
        pool_weights=tuple(
            mapper_metadata.get("pool_weights", (0.4, 0.3, 0.2, 0.1))
        ),
        global_context_weight=float(
            mapper_metadata.get("global_context_weight", 0.0)
        ),
    )
    map_representation = str(
        (bank.metadata or {}).get("representation", "")
    )
    spatial_map_representation = str(
        (spatial_bank.metadata or {}).get("representation", "")
    )
    identity_texture_retrieval = map_representation in {
        "canonical_spatial_radio_final_mixture_per_maplet",
        "exact_canonical_radio_final_spatial_texture",
    }
    spatial_texture_retrieval = spatial_map_representation in {
        "canonical_spatial_radio_final_mixture_per_maplet",
        "exact_canonical_radio_final_spatial_texture",
    }
    query_region_config = (
        RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,))
        if identity_texture_retrieval
        else config
    )
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate()
    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.query_pose_file))
    }
    camera_by_image, camera_audit = _load_query_camera_manifest(
        Path(args.query_camera_manifest)
    )
    records = [
        record
        for record in manifest.records
        if record.image_id in pose_by_image
        and record.image_id in camera_by_image
    ]
    requested_trajectories = {
        str(value) for value in args.query_trajectory_ids
    }
    if requested_trajectories:
        records = [
            record
            for record in records
            if record.image_id.split("/", 1)[0] in requested_trajectories
        ]
    if exact_visibility:
        records = [
            record
            for record in records
            if record.image_id in exact_visibility
        ]
    if int(args.max_queries) > 0:
        # Evenly sample the trajectory instead of silently evaluating only its
        # beginning, which is usually a much easier and less diverse subset.
        indices = np.linspace(
            0, len(records) - 1, int(args.max_queries), dtype=np.int64
        )
        records = [records[int(index)] for index in indices.tolist()]
    query_trajectory_ids = sorted(
        {record.image_id.split("/", 1)[0] for record in records}
    )
    calibration_overlap = sorted(
        set(query_trajectory_ids)
        & set(calibration_metadata["calibration_trajectory_ids"])
    )
    pose_vote_mapping_ids = sorted(
        str(value)
        for value in (
            (pose_vote_bank.metadata or {}).get(
                "mapping_trajectory_ids", []
            )
            if pose_vote_bank is not None
            else []
        )
    )
    pose_vote_mapping_overlap = sorted(
        set(query_trajectory_ids) & set(pose_vote_mapping_ids)
    )
    if (
        calibration_overlap or pose_vote_mapping_overlap
    ) and not bool(args.allow_reference_trajectory_diagnostic):
        raise ValueError(
            "independent evaluation overlaps map/calibration trajectories; "
            f"calibration={calibration_overlap}, "
            f"pose_vote_map={pose_vote_mapping_overlap}. Use "
            "--allow_reference_trajectory_diagnostic only for replay "
            "diagnostics."
        )
    rows: list[dict[str, object]] = []
    for query_index, record in enumerate(records):
        raw = _load_raw_final(Path(record.token_path), "radio_final")
        mapped = (
            bank.project_query_feature_map(raw)
            if bank.query_projection is not None
            else mapper.project(raw).measurement_context
        )
        metric_token_xy = None
        if spatial_metric_model is not None:
            camera = camera_by_image[record.image_id]
            image = Image.open(
                Path(args.image_root) / record.image_id
            ).convert("RGB").resize(
                (camera.width, camera.height), Image.Resampling.BILINEAR
            )
            rgb = torch.from_numpy(
                (np.asarray(image, dtype=np.float32) / 255.0)
                .transpose(2, 0, 1)
                .copy()
            )[None].to(str(args.device))
            radio = torch.from_numpy(
                np.asarray(raw, dtype=np.float32)
            )[None].to(str(args.device))
            with torch.no_grad():
                spatial_mapped = spatial_metric_model(radio, rgb)[
                    spatial_metric_level
                ][0].cpu().numpy()
        else:
            spatial_mapped = (
                spatial_bank.project_query_feature_map(raw)
                if spatial_bank.query_projection is not None
                else mapper.project(raw).measurement_context
            )
        _indices, token_xy = select_spatially_balanced_radio_final_regions(raw)
        if spatial_metric_model is not None:
            metric_token_xy = (
                (token_xy + 0.5)
                * np.asarray(
                    [
                        spatial_mapped.shape[2] / raw.shape[2],
                        spatial_mapped.shape[1] / raw.shape[1],
                    ],
                    dtype=np.float32,
                )[None]
                - 0.5
            )
        descriptors = encode_radio_final_regions(
            mapped, token_xy, query_region_config
        )
        spatial_descriptors = encode_radio_final_regions(
            spatial_mapped,
            token_xy if metric_token_xy is None else metric_token_xy,
            RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,)),
        )
        region_xy, region_extent = _region_geometry(
            token_xy,
            token_width=int(raw.shape[2]),
            token_height=int(raw.shape[1]),
            image_width=int(camera_by_image[record.image_id].width),
            image_height=int(camera_by_image[record.image_id].height),
            config=query_region_config,
        )
        retrieval = retrieve_candidate_groups(
            descriptors,
            region_xy,
            region_extent,
            bank,
            preliminary_candidates=int(args.preliminary_candidates),
            maximum_maplets=int(args.maximum_maplets),
            maximum_components_per_maplet=int(
                args.maximum_components_per_maplet
            ),
            component_nms_distance_m=float(
                args.component_nms_distance_m
            ),
            set_rerank_strength=0.0,
            spatial_query_descriptors=(
                spatial_descriptors if spatial_bank is not bank else None
            ),
            spatial_bank=(
                spatial_bank if spatial_bank is not bank else None
            ),
            probability_calibration=probability_calibration,
        )
        conservation_error = max(
            (
                abs(
                    float(np.sum(group.probabilities))
                    + float(group.null_probability)
                    - 1.0
                )
                for group in retrieval.groups
            ),
            default=0.0,
        )
        spatial_conservation_errors = []
        spatial_available_count = 0
        spatial_candidate_count = 0
        for group in retrieval.groups:
            if (
                group.component_offsets is None
                or group.component_probabilities is None
                or group.spatial_null_probabilities is None
            ):
                continue
            offsets = np.asarray(group.component_offsets, dtype=np.int64)
            spatial_null = np.asarray(
                group.spatial_null_probabilities, dtype=np.float64
            )
            available = np.asarray(group.spatial_available, dtype=bool)
            spatial_available_count += int(np.sum(available))
            spatial_candidate_count += int(available.size)
            for candidate in range(group.maplet_ids.size):
                begin, end = int(offsets[candidate]), int(offsets[candidate + 1])
                spatial_conservation_errors.append(
                    abs(
                        float(
                            np.sum(
                                group.component_probabilities[begin:end]
                            )
                        )
                        + float(spatial_null[candidate])
                        - 1.0
                    )
                )
        spatial_conservation_error = max(
            spatial_conservation_errors, default=0.0
        )
        view_mode_hypotheses = (
            vote_maplet_poses(
                descriptors,
                bank,
                pose_vote_bank,
                candidate_groups=retrieval.groups,
                image_size_wh=(
                    int(camera_by_image[record.image_id].width),
                    int(camera_by_image[record.image_id].height),
                ),
                camera=camera_by_image[record.image_id],
                region_match_probability=np.asarray(
                    [
                        1.0 - float(group.null_probability)
                        for group in retrieval.groups
                    ],
                    dtype=np.float64,
                ),
                maximum_modes=int(args.maximum_modes),
            )
            if pose_vote_bank is not None
            else tuple()
        )
        if str(args.coarse_pose_method) == "maplet_footprint":
            hypotheses = propose_maplet_footprint_poses(
                retrieval.groups,
                bank,
                camera_by_image[record.image_id],
                view_mode_hypotheses,
                maximum_modes=int(args.maximum_modes),
                candidates_per_region=int(
                    args.footprint_candidates_per_region
                ),
                cem_iterations=int(args.footprint_cem_iterations),
                particles_per_mode=int(
                    args.footprint_particles_per_mode
                ),
                seed=int(args.seed) + query_index,
            )
        elif str(args.coarse_pose_method) == "anonymous_view_mode":
            hypotheses = view_mode_hypotheses
        else:
            hypotheses = propose_regional_center_pseudo_pnp_poses(
                retrieval.groups,
                bank,
                camera_by_image[record.image_id],
                trials=int(args.proposal_trials),
                maximum_modes=int(args.maximum_modes),
                refinement_candidates=int(
                    args.proposal_refinement_candidates
                ),
                em_iterations=int(args.proposal_em_iterations),
                seed=int(args.seed) + query_index,
            )
        # Diagnostic only: GT is evaluated strictly after every hypothesis has
        # been generated.  Comparing its footprint likelihood with proposal
        # likelihoods separates a failed pose search from a misspecified
        # footprint observation model without leaking GT into localization.
        gt_footprint_score, gt_footprint_support = (
            score_maplet_footprint_pose(
                pose_by_image[record.image_id],
                retrieval.groups,
                bank,
                camera_by_image[record.image_id],
                candidates_per_region=int(
                    args.footprint_candidates_per_region
                ),
            )
        )
        errors = []
        proposal_footprint_scores = []
        for hypothesis in hypotheses:
            error = pnp_pose_error(
                hypothesis.pose_w2c, pose_by_image[record.image_id]
            )
            footprint_score, footprint_support = score_maplet_footprint_pose(
                hypothesis.pose_w2c,
                retrieval.groups,
                bank,
                camera_by_image[record.image_id],
                candidates_per_region=int(
                    args.footprint_candidates_per_region
                ),
            )
            proposal_footprint_scores.append(float(footprint_score))
            errors.append(
                {
                    "translation_m": float(error.translation_m),
                    "rotation_deg": float(error.rotation_deg),
                    "proposal_score": float(hypothesis.score),
                    "supporting_group_count": int(
                        hypothesis.supporting_group_count
                    ),
                    "proposal_source": str(hypothesis.source),
                    "diagnostic_footprint_log_likelihood_ratio": float(
                        footprint_score
                    ),
                    "diagnostic_footprint_supporting_region_count": int(
                        footprint_support
                    ),
                }
            )
        row = {
            "image_id": record.image_id,
            "query_region_count": len(retrieval.groups),
            "ranked_maplet_count": int(retrieval.ranked_maplet_ids.size),
            "ranked_maplet_ids": retrieval.ranked_maplet_ids.tolist(),
            "probability_conservation_error": float(conservation_error),
            "spatial_probability_conservation_error": float(
                spatial_conservation_error
            ),
            "spatial_available_candidate_fraction": float(
                spatial_available_count / max(spatial_candidate_count, 1)
            ),
            "hypothesis_count": len(hypotheses),
            "hypothesis_errors": errors,
            "diagnostic_footprint_ground_truth_log_likelihood_ratio": float(
                gt_footprint_score
            ),
            "diagnostic_footprint_ground_truth_supporting_region_count": int(
                gt_footprint_support
            ),
            "diagnostic_footprint_gt_minus_best_proposal": (
                float(
                    gt_footprint_score
                    - max(proposal_footprint_scores)
                )
                if proposal_footprint_scores
                else None
            ),
            **_geometric_diversity(retrieval.ranked_maplet_ids, bank),
        }
        if record.image_id in exact_visibility:
            assert atlas_geometry is not None
            exact_view = exact_visibility[record.image_id]
            surface_ids = exact_view.visible_rows
            maplet_rows = surface_ids // (
                atlas_geometry.height * atlas_geometry.width
            )
            unique_rows, visible_counts = np.unique(
                maplet_rows, return_counts=True
            )
            visible_ids = atlas_geometry.maplet_ids[unique_rows]
            hit = np.isin(visible_ids, retrieval.ranked_maplet_ids)
            hit_ids = visible_ids[hit]
            diversity = _geometric_diversity(hit_ids, bank)
            row.update(
                {
                    "visible_maplet_count": int(visible_ids.size),
                    "visible_maplet_hit_count": int(np.sum(hit)),
                    "visible_maplet_recall": float(
                        np.sum(hit) / max(visible_ids.size, 1)
                    ),
                    "visible_surface_coverage": float(
                        np.sum(visible_counts[hit])
                        / max(np.sum(visible_counts), 1)
                    ),
                    "geometrically_independent_visible_hits": float(
                        diversity["candidate_center_rank"]
                    ),
                    **_region_identity_diagnostics(
                        retrieval.groups,
                        exact_view,
                        atlas_geometry,
                        bank,
                    ),
                    "coarse_observation_oracles": (
                        _coarse_observation_oracles(
                            retrieval.groups,
                            exact_view,
                            atlas_geometry,
                            bank,
                            spatial_bank,
                        )
                    ),
                }
            )
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "stage": "v6_retrieval_and_coarse_pose_dashboard",
        **_aggregate(rows),
        "camera_audit": camera_audit,
        "query_trajectory_ids": query_trajectory_ids,
        "calibration_trajectory_overlap": calibration_overlap,
        "pose_vote_mapping_trajectory_overlap": (
            pose_vote_mapping_overlap
        ),
        "independent_trajectory_protocol": not bool(
            calibration_overlap or pose_vote_mapping_overlap
        ),
        "reference_trajectory_overlap_allowed_for_diagnostic": bool(
            args.allow_reference_trajectory_diagnostic
        ),
        "set_reranker_strength": 0.0,
        "preliminary_candidates": int(args.preliminary_candidates),
        "maximum_maplets": int(args.maximum_maplets),
        "maximum_components_per_maplet": int(
            args.maximum_components_per_maplet
        ),
        "component_nms_distance_m": float(args.component_nms_distance_m),
        "diagnostic_regional_center_pseudo_pnp": {
            "enabled": (
                str(args.coarse_pose_method)
                == "regional_center_pseudo_pnp"
            ),
            "proposal_trials": int(args.proposal_trials),
            "proposal_refinement_candidates": int(
                args.proposal_refinement_candidates
            ),
            "proposal_em_iterations": int(args.proposal_em_iterations),
            "deterministic_ransac_prefixes": [
                12,
                16,
                24,
                32,
                48,
                64,
                96,
                "all",
            ],
        },
        "uses_canonical_spatial_retrieval_texture": bool(
            spatial_texture_retrieval
        ),
        "uses_embedded_query_projection": bool(
            bank.query_projection is not None
        ),
        "uses_separate_spatial_surface_bank": bool(
            spatial_bank is not bank
        ),
        "uses_metric_spatial_surface_bank": bool(
            spatial_metric_model is not None
        ),
        "spatial_metric_feature_level": spatial_metric_level,
        "identity_bank_maplet_count": len(bank),
        "spatial_bank_maplet_count": len(spatial_bank),
        "identity_bank_is_never_spatially_intersected": True,
        "identity_bank_has_canonical_tangent_frames": bool(
            (bank.metadata or {}).get(
                "has_canonical_tangent_frames", False
            )
        ),
        "coarse_pose_method": str(args.coarse_pose_method),
        "coarse_pose_method_status": (
            "experimental_region_level"
            if str(args.coarse_pose_method)
            in {"anonymous_view_mode", "maplet_footprint"}
            else "diagnostic_pseudo_correspondence_only"
        ),
        "regional_center_pseudo_pnp_is_diagnostic_only": True,
        "probability_calibration_sha256": (
            probability_calibration_sha256(probability_calibration_path)
        ),
        "probability_calibration_trajectory_ids": list(
            calibration_metadata["calibration_trajectory_ids"]
        ),
        "spatial_calibration_condition": str(
            calibration_metadata.get(
                "spatial_calibration_condition",
                "legacy_or_unspecified",
            )
        ),
        "probability_calibration_metrics": {
            "identity": dict(
                calibration_metadata.get("identity_metrics", {})
            ),
            "conditional_spatial": dict(
                calibration_metadata.get("spatial_metrics", {})
            ),
        },
        "footprint_cem_iterations": int(args.footprint_cem_iterations),
        "footprint_particles_per_mode": int(
            args.footprint_particles_per_mode
        ),
        "footprint_candidates_per_region": int(
            args.footprint_candidates_per_region
        ),
        "production_contract": {
            "map_representation": map_representation,
            "spatial_map_representation": spatial_map_representation,
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": False,
            "uses_pairwise_image_matching": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "ground_truth_used_after_hypothesis_generation_only": True,
            "identity_bank_intersected_with_spatial_bank": False,
            "has_explicit_conditional_spatial_null": True,
            "mode_truncation_mass_transferred_to_spatial_null": True,
            "uses_virtual_surface_mean": False,
            "uses_fixed_pose_null_multiplier": False,
            "ground_truth_footprint_scoring_is_diagnostic_only": True,
            "fine_correlation_typed_null_is_runtime_qualified": False,
            "iterative_fine_alignment_is_runtime_closed": False,
            "end_to_end_production_qualified": False,
        },
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
