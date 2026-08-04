"""Evaluate V8.1 O1--O4 oracles and corrected region/surface inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_v6_retrieval_pose_basin import (
    _region_geometry,
    _region_identity_diagnostics,
)
from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_atlas import MapletFeatureAtlasBank
from feature_extract.vfm.localization_v6.maplet_pose_voting import (
    AnonymousMapletPoseVoteBank,
    vote_maplet_poses,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    MapletRetrievalResult,
    QueryMapletGroup,
    retrieve_candidate_groups,
)
from feature_extract.vfm.localization_v6.probability_calibration import (
    V6ProbabilityCalibration,
)
from feature_extract.vfm.localization_v6.se3_update import se3_exp
from feature_extract.vfm.localization_v8.structured_maplet_graph import (
    PhysicalMapletGraph,
    build_query_region_graph,
    refine_structured_maplet_pose,
    score_structured_maplet_pose,
)
from feature_extract.vfm.localization_v81.region_surface_graph import (
    RegionEvidenceGraph,
    build_region_evidence_graph,
    refine_region_surface_pose,
    score_region_surface_pose,
)
from feature_extract.vfm.localization_v81.virtual_pose_lattice import (
    VirtualPoseLattice,
    expand_virtual_pose_seeds,
    rank_pose_candidates_by_region_centres,
    rank_virtual_pose_lattice,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    encode_radio_final_regions,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--atlas_geometry", required=True)
    parser.add_argument("--identity_bank", required=True)
    parser.add_argument("--spatial_bank", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--probability_calibration", required=True)
    parser.add_argument("--physical_graph", required=True)
    parser.add_argument("--pose_vote_bank", required=True)
    parser.add_argument("--virtual_pose_lattice", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--candidates_per_group", type=int, default=12)
    parser.add_argument("--maximum_pose_modes", type=int, default=64)
    parser.add_argument("--virtual_coarse_poses", type=int, default=256)
    parser.add_argument("--virtual_subdivision_seeds", type=int, default=64)
    parser.add_argument("--virtual_full_score_poses", type=int, default=64)
    parser.add_argument("--lattice_random_samples", type=int, default=128)
    parser.add_argument("--skip_local_oracles", action="store_true")
    parser.add_argument("--refine_seeds", type=int, default=2)
    parser.add_argument("--refine_stages", type=int, default=5)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _pose_error(pose: np.ndarray, target: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(pose, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    estimate_center = -estimate[:3, :3].T @ estimate[:3, 3]
    truth_center = -truth[:3, :3].T @ truth[:3, 3]
    relative = estimate[:3, :3] @ truth[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return {
        "translation_m": float(np.linalg.norm(estimate_center - truth_center)),
        "rotation_deg": float(np.degrees(np.arccos(cosine))),
    }


def _pose_metrics(values: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not values:
        return {}
    translation = np.asarray([value["translation_m"] for value in values])
    rotation = np.asarray([value["rotation_deg"] for value in values])
    return {
        f"{prefix}_translation_median_m": float(np.median(translation)),
        f"{prefix}_translation_p90_m": float(np.quantile(translation, 0.9)),
        f"{prefix}_rotation_median_deg": float(np.median(rotation)),
        f"{prefix}_rotation_p90_deg": float(np.quantile(rotation, 0.9)),
        f"{prefix}_1m_10deg": float(np.mean((translation <= 1.0) & (rotation <= 10.0))),
        f"{prefix}_50cm_5deg": float(np.mean((translation <= 0.5) & (rotation <= 5.0))),
        f"{prefix}_30cm_3deg": float(np.mean((translation <= 0.3) & (rotation <= 3.0))),
    }


def _local_pose_lattice(
    ground_truth: np.ndarray, *, random_samples: int, seed: int
) -> list[np.ndarray]:
    perturbations = [np.zeros((6,), dtype=np.float64)]
    for axis in range(6):
        magnitudes = (
            np.deg2rad([1.0, 3.0, 5.0, 10.0])
            if axis < 3
            else np.asarray([0.10, 0.30, 0.50, 1.00])
        )
        for magnitude in magnitudes:
            for sign in (-1.0, 1.0):
                value = np.zeros((6,), dtype=np.float64)
                value[axis] = sign * float(magnitude)
                perturbations.append(value)
    rng = np.random.default_rng(int(seed))
    for index in range(max(int(random_samples), 0)):
        rotation = rng.normal(size=3)
        rotation /= max(float(np.linalg.norm(rotation)), 1e-8)
        rotation *= np.deg2rad(0.5 + 14.5 * (index + 0.5) / max(random_samples, 1))
        translation = rng.normal(size=3)
        translation /= max(float(np.linalg.norm(translation)), 1e-8)
        translation *= 0.05 + 1.45 * ((index * 37) % max(random_samples, 1) + 0.5) / max(random_samples, 1)
        perturbations.append(np.r_[rotation, translation])
    return [se3_exp(value) @ np.asarray(ground_truth) for value in perturbations]


def _oracle_identity_for_current_supports(
    retrieval: MapletRetrievalResult,
    view,
    atlas: MapletFeatureAtlasBank,
) -> MapletRetrievalResult:
    flat_cells = int(atlas.height * atlas.width)
    visible_maplet = atlas.maplet_ids[
        np.asarray(view.visible_rows, dtype=np.int64) // flat_cells
    ]
    visible_xy = np.asarray(view.image_xy, dtype=np.float64)
    groups = []
    evidence: dict[int, float] = {}
    for observation in retrieval.groups:
        extent = np.maximum(np.asarray(observation.query_region_extent), 1.0)
        inside = np.max(
            np.abs(visible_xy - observation.query_region_xy[None]) / extent[None],
            axis=1,
        ) <= 1.0
        if np.any(inside):
            identity, counts = np.unique(visible_maplet[inside], return_counts=True)
            maplet_id = int(identity[int(np.argmax(counts))])
            candidate_ids = np.asarray([maplet_id], dtype=np.int64)
            probability = np.asarray([1.0], dtype=np.float32)
            null = 0.0
            evidence[maplet_id] = evidence.get(maplet_id, 0.0) + 1.0
        else:
            candidate_ids = np.zeros((0,), dtype=np.int64)
            probability = np.zeros((0,), dtype=np.float32)
            null = 1.0
        groups.append(
            QueryMapletGroup(
                query_region_xy=np.asarray(observation.query_region_xy),
                query_region_extent=np.asarray(observation.query_region_extent),
                maplet_ids=candidate_ids,
                probabilities=probability,
                null_probability=null,
                omitted_probability=0.0,
            )
        )
    ranked = sorted(evidence, key=lambda value: (-evidence[value], value))
    return MapletRetrievalResult(
        tuple(groups),
        np.asarray(ranked, dtype=np.int64),
        np.asarray([evidence[value] for value in ranked], dtype=np.float32),
        scene_evidence_aggregation="oracle_current_support_identity",
    )


def _oracle_support_and_identity(
    view, atlas: MapletFeatureAtlasBank
) -> tuple[MapletRetrievalResult, np.ndarray]:
    flat_cells = int(atlas.height * atlas.width)
    maplet_ids = atlas.maplet_ids[
        np.asarray(view.visible_rows, dtype=np.int64) // flat_cells
    ]
    image_xy = np.asarray(view.image_xy, dtype=np.float64)
    groups = []
    ranked = []
    for maplet_id in np.unique(maplet_ids).tolist():
        local = image_xy[maplet_ids == int(maplet_id)]
        if local.shape[0] < 3:
            continue
        low = np.quantile(local, 0.05, axis=0)
        high = np.quantile(local, 0.95, axis=0)
        extent = np.maximum(0.5 * (high - low), 2.0)
        groups.append(
            QueryMapletGroup(
                query_region_xy=0.5 * (low + high),
                query_region_extent=extent,
                maplet_ids=np.asarray([maplet_id], dtype=np.int64),
                probabilities=np.asarray([1.0], dtype=np.float32),
                null_probability=0.0,
                omitted_probability=0.0,
            )
        )
        ranked.append(int(maplet_id))
    # Orthogonal oracle descriptors ensure that O1 evaluates maplet observation
    # geometry, not the support-grouping heuristic.
    descriptor = np.eye(len(groups), dtype=np.float32)
    retrieval = MapletRetrievalResult(
        tuple(groups),
        np.asarray(ranked, dtype=np.int64),
        np.ones((len(ranked),), dtype=np.float32),
        scene_evidence_aggregation="oracle_support_and_identity",
    )
    return retrieval, descriptor


def _score_lattice(
    lattice: list[np.ndarray],
    truth: np.ndarray,
    scorer: Callable[[np.ndarray], tuple[float, dict[str, float]]],
) -> tuple[list[tuple[float, np.ndarray, dict[str, float]]], dict[str, object]]:
    scored = []
    for pose in lattice:
        score, parts = scorer(pose)
        scored.append((float(score), pose, parts))
    scored.sort(key=lambda value: -value[0])
    errors = [_pose_error(value[1], truth) for value in scored]
    top16 = errors[:16]
    ground_truth_score = float(scorer(truth)[0])
    non_ground_truth_scores = [
        value[0]
        for value, error in zip(scored, errors)
        if error["translation_m"] > 1e-6 or error["rotation_deg"] > 1e-6
    ]
    return scored, {
        "top1": errors[0],
        "ground_truth_rank": 1 + sum(value[0] > ground_truth_score for value in scored),
        "top16_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in top16)),
        "top16_50cm_5deg": bool(any(x["translation_m"] <= 0.5 and x["rotation_deg"] <= 5.0 for x in top16)),
        "top16_30cm_3deg": bool(any(x["translation_m"] <= 0.3 and x["rotation_deg"] <= 3.0 for x in top16)),
        "score_margin_gt_minus_best_non_gt": float(
            ground_truth_score - max(non_ground_truth_scores, default=ground_truth_score)
        ),
    }


def _refine_best(
    scored: list[tuple[float, np.ndarray, dict[str, float]]],
    query: RegionEvidenceGraph,
    physical: PhysicalMapletGraph,
    camera,
    truth: np.ndarray,
    seeds: int,
    stages: int,
) -> dict[str, object]:
    candidates = []
    for _score, pose, _parts in scored[: max(int(seeds), 0)]:
        refined_pose, refined_score, parts = refine_region_surface_pose(
            pose, query, physical, camera, stages=int(stages)
        )
        candidates.append((refined_score, refined_pose, parts))
    candidates.sort(key=lambda value: -value[0])
    if not candidates:
        return {"top1": None}
    return {
        "top1": _pose_error(candidates[0][1], truth),
        "score": float(candidates[0][0]),
        "score_parts": candidates[0][2],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite V8.1 oracle evaluation")
    identity_path = Path(args.identity_bank)
    spatial_path = Path(args.spatial_bank)
    mapper_path = Path(args.surface_mapper)
    identity = SurfaceRetrievalMapletBank.load_npz(identity_path)
    spatial = SurfaceRetrievalMapletBank.load_npz(spatial_path)
    atlas = MapletFeatureAtlasBank.load_npz(Path(args.atlas_geometry))
    physical = PhysicalMapletGraph.load_npz(Path(args.physical_graph))
    votes = AnonymousMapletPoseVoteBank.load_npz(Path(args.pose_vote_bank))
    virtual_lattice = (
        VirtualPoseLattice.load_npz(Path(args.virtual_pose_lattice))
        if str(args.virtual_pose_lattice)
        else None
    )
    calibration = V6ProbabilityCalibration.load_json(Path(args.probability_calibration))
    mapper, mapper_metadata = load_surface_maplet_mapper(mapper_path, device=args.device)
    hashes = {
        "identity_bank": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
        "spatial_bank": hashlib.sha256(spatial_path.read_bytes()).hexdigest(),
        "surface_mapper": hashlib.sha256(mapper_path.read_bytes()).hexdigest(),
    }
    calibration_metadata = dict(calibration.metadata or {})
    for name, key in (
        ("identity_bank", "identity_bank_sha256"),
        ("spatial_bank", "spatial_bank_sha256"),
        ("surface_mapper", "surface_mapper_sha256"),
    ):
        if hashes[name] != str(calibration_metadata.get(key, "")):
            raise ValueError(f"calibration {name} lineage differs")
    if physical.feature_bank_sha256 != hashes["identity_bank"]:
        raise ValueError("physical graph and canonical feature bank differ")
    if virtual_lattice is not None:
        if virtual_lattice.physical_graph_sha256 != hashlib.sha256(Path(args.physical_graph).read_bytes()).hexdigest():
            raise ValueError("virtual lattice physical-graph lineage differs")
        if virtual_lattice.pose_vote_bank_sha256 != hashlib.sha256(Path(args.pose_vote_bank).read_bytes()).hexdigest():
            raise ValueError("virtual lattice pose-vote lineage differs")
    strict = {"seq3", "seq5", "seq13"}
    vote_overlap = strict & set((votes.metadata or {}).get("mapping_trajectory_ids", []))
    if vote_overlap:
        raise ValueError(f"strict trajectories leaked into pose votes: {sorted(vote_overlap)}")
    views = _load_views(Path(args.contributors), atlas, Path(args.image_root))
    views = views[int(args.shard_index) :: int(args.shard_count)]
    if int(args.max_queries) > 0:
        positions = np.linspace(0, len(views) - 1, int(args.max_queries), dtype=np.int64)
        views = [views[int(value)] for value in positions.tolist()]
    config = RadioFinalRegionConfig(
        pool_sizes=tuple(mapper_metadata.get("pool_sizes", (1, 3, 5, 9))),
        pool_weights=tuple(mapper_metadata.get("pool_weights", (0.4, 0.3, 0.2, 0.1))),
        global_context_weight=float(mapper_metadata.get("global_context_weight", 0.0)),
    )
    rows = []
    for query_index, view in enumerate(views):
        raw = view.radio.numpy()
        mapped = mapper.project(raw).measurement_context
        spatial_mapped = spatial.project_query_feature_map(raw)
        _indices, token_xy = select_spatially_balanced_radio_final_regions(raw)
        descriptors = encode_radio_final_regions(mapped, token_xy, config)
        spatial_descriptors = encode_radio_final_regions(
            spatial_mapped,
            token_xy,
            RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,)),
        )
        region_xy, region_extent = _region_geometry(
            token_xy,
            token_width=int(raw.shape[2]),
            token_height=int(raw.shape[1]),
            image_width=int(view.camera.width),
            image_height=int(view.camera.height),
            config=config,
        )
        retrieval = retrieve_candidate_groups(
            descriptors,
            region_xy,
            region_extent,
            identity,
            preliminary_candidates=64,
            maximum_maplets=64,
            maximum_components_per_maplet=16,
            component_nms_distance_m=0.02,
            spatial_query_descriptors=spatial_descriptors,
            spatial_bank=spatial,
            probability_calibration=calibration,
            compute_spatial_modes=False,
        )
        current_graph = build_region_evidence_graph(
            retrieval,
            physical,
            descriptors,
            image_size_wh=(view.camera.width, view.camera.height),
            candidates_per_group=int(args.candidates_per_group),
        )
        o1 = o2 = o3 = None
        if not bool(args.skip_local_oracles):
            oracle_identity_retrieval = _oracle_identity_for_current_supports(
                retrieval, view, atlas
            )
            oracle_identity_graph = build_region_evidence_graph(
                oracle_identity_retrieval,
                physical,
                descriptors,
                image_size_wh=(view.camera.width, view.camera.height),
                candidates_per_group=int(args.candidates_per_group),
            )
            oracle_support_retrieval, oracle_descriptor = _oracle_support_and_identity(
                view, atlas
            )
            oracle_support_graph = build_region_evidence_graph(
                oracle_support_retrieval,
                physical,
                oracle_descriptor,
                image_size_wh=(view.camera.width, view.camera.height),
                candidates_per_group=int(args.candidates_per_group),
            )
            local_lattice = _local_pose_lattice(
                view.pose_w2c,
                random_samples=int(args.lattice_random_samples),
                seed=191 + query_index,
            )
            scorer = lambda graph: lambda pose: score_region_surface_pose(
                pose, graph, physical, view.camera
            )
            o1_scored, o1 = _score_lattice(
                local_lattice, view.pose_w2c, scorer(oracle_support_graph)
            )
            o2_scored, o2 = _score_lattice(
                local_lattice, view.pose_w2c, scorer(oracle_identity_graph)
            )
            o3_scored, o3 = _score_lattice(
                local_lattice, view.pose_w2c, scorer(current_graph)
            )
            o1["refined"] = _refine_best(
                o1_scored, oracle_support_graph, physical, view.camera, view.pose_w2c,
                int(args.refine_seeds), int(args.refine_stages),
            )
            o2["refined"] = _refine_best(
                o2_scored, oracle_identity_graph, physical, view.camera, view.pose_w2c,
                int(args.refine_seeds), int(args.refine_stages),
            )
            o3["refined"] = _refine_best(
                o3_scored, current_graph, physical, view.camera, view.pose_w2c,
                int(args.refine_seeds), int(args.refine_stages),
            )

        hypotheses = vote_maplet_poses(
            descriptors,
            identity,
            votes,
            candidate_groups=retrieval.groups,
            image_size_wh=(view.camera.width, view.camera.height),
            camera=view.camera,
            maximum_modes=int(args.maximum_pose_modes),
            candidates_per_region=8,
        )
        frozen_query = build_query_region_graph(
            retrieval,
            physical,
            image_size_wh=(view.camera.width, view.camera.height),
            candidates_per_region=8,
        )
        frozen_scored = []
        corrected_scored = []
        for hypothesis in hypotheses:
            frozen_score, frozen_parts = score_structured_maplet_pose(
                hypothesis.pose_w2c, frozen_query, physical, view.camera
            )
            corrected_score, corrected_parts = score_region_surface_pose(
                hypothesis.pose_w2c, current_graph, physical, view.camera
            )
            frozen_scored.append((frozen_score, hypothesis.pose_w2c, frozen_parts))
            corrected_scored.append((corrected_score, hypothesis.pose_w2c, corrected_parts))
        frozen_scored.sort(key=lambda value: -value[0])
        corrected_scored.sort(key=lambda value: -value[0])
        frozen_refined = []
        for _score, pose, _parts in frozen_scored[: int(args.refine_seeds)]:
            refined_pose, refined_score, refined_parts = refine_structured_maplet_pose(
                pose, frozen_query, physical, view.camera, stages=int(args.refine_stages)
            )
            frozen_refined.append((refined_score, refined_pose, refined_parts))
        frozen_refined.sort(key=lambda value: -value[0])
        o4 = {
            "proposal_count": len(hypotheses),
            "frozen_v80_top1": _pose_error(frozen_scored[0][1], view.pose_w2c) if frozen_scored else None,
            "frozen_v80_refined_top1": _pose_error(frozen_refined[0][1], view.pose_w2c) if frozen_refined else None,
            "corrected_v81_top1": _pose_error(corrected_scored[0][1], view.pose_w2c) if corrected_scored else None,
            "corrected_v81_refined_top1": _refine_best(
                corrected_scored, current_graph, physical, view.camera, view.pose_w2c,
                int(args.refine_seeds), int(args.refine_stages),
            )["top1"],
        }
        o5 = None
        if virtual_lattice is not None:
            coarse_indices, coarse_scores = rank_virtual_pose_lattice(
                virtual_lattice,
                current_graph,
                maximum_poses=int(args.virtual_coarse_poses),
            )
            coarse_poses = virtual_lattice.poses_w2c(coarse_indices)
            structured_seed_indices, structured_seed_scores = rank_pose_candidates_by_region_centres(
                coarse_poses,
                current_graph,
                physical,
                view.camera,
                maximum_poses=int(args.virtual_subdivision_seeds),
                device=str(args.device),
            )
            structured_seed_poses = coarse_poses[structured_seed_indices]
            subdivision = expand_virtual_pose_seeds(
                structured_seed_poses,
                translation_step_m=0.5,
                rotation_step_deg=3.0,
            )
            subdivision_indices, subdivision_scores = rank_pose_candidates_by_region_centres(
                subdivision,
                current_graph,
                physical,
                view.camera,
                maximum_poses=int(args.virtual_full_score_poses),
                device=str(args.device),
            )
            subdivided_poses = subdivision[subdivision_indices]
            virtual_scored = []
            for pose in subdivided_poses:
                score, parts = score_region_surface_pose(
                    pose, current_graph, physical, view.camera
                )
                virtual_scored.append((score, pose, parts))
            virtual_scored.sort(key=lambda value: -value[0])
            virtual_refined = _refine_best(
                virtual_scored,
                current_graph,
                physical,
                view.camera,
                view.pose_w2c,
                int(args.refine_seeds),
                int(args.refine_stages),
            )
            coarse_errors = [_pose_error(pose, view.pose_w2c) for pose in coarse_poses[:16]]
            coarse_all_errors = [_pose_error(pose, view.pose_w2c) for pose in coarse_poses]
            structured_seed_errors = [
                _pose_error(pose, view.pose_w2c) for pose in structured_seed_poses
            ]
            subdivision_errors = [_pose_error(pose, view.pose_w2c) for pose in subdivided_poses[:16]]
            subdivision_all_errors = [_pose_error(pose, view.pose_w2c) for pose in subdivided_poses]
            truth_center = -view.pose_w2c[:3, :3].T @ view.pose_w2c[:3, 3]
            nearest_translation = float(np.min(np.linalg.norm(virtual_lattice.positions - truth_center[None], axis=1)))
            relative = np.einsum(
                "nij,kj->nki", virtual_lattice.rotations_w2c, view.pose_w2c[:3, :3]
            )
            rotation_cosine = np.clip(
                (np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5,
                -1.0,
                1.0,
            )
            nearest_rotation = float(np.min(np.degrees(np.arccos(rotation_cosine))))
            o5 = {
                "lattice_pose_count": virtual_lattice.pose_count,
                "oracle_nearest_translation_m": nearest_translation,
                "oracle_nearest_rotation_deg": nearest_rotation,
                "coarse_top16_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in coarse_errors)),
                "coarse_top16_50cm_5deg": bool(any(x["translation_m"] <= 0.5 and x["rotation_deg"] <= 5.0 for x in coarse_errors)),
                "coarse_top64_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in coarse_all_errors[:64])),
                "coarse_top256_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in coarse_all_errors[:256])),
                "coarse_top1024_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in coarse_all_errors[:1024])),
                "coarse_top4096_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in coarse_all_errors[:4096])),
                "coarse_top8192_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in coarse_all_errors[:8192])),
                "structured_coarse_top16_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in structured_seed_errors[:16])),
                "structured_coarse_top64_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in structured_seed_errors[:64])),
                "structured_coarse_top256_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in structured_seed_errors[:256])),
                "structured_coarse_top1024_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in structured_seed_errors[:1024])),
                "subdivided_top16_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in subdivision_errors)),
                "subdivided_top16_50cm_5deg": bool(any(x["translation_m"] <= 0.5 and x["rotation_deg"] <= 5.0 for x in subdivision_errors)),
                "subdivided_top64_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in subdivision_all_errors[:64])),
                "subdivided_top256_1m_10deg": bool(any(x["translation_m"] <= 1.0 and x["rotation_deg"] <= 10.0 for x in subdivision_all_errors[:256])),
                "coarse_retrieval_top1": coarse_errors[0] if coarse_errors else None,
                "subdivided_retrieval_top1": subdivision_errors[0] if subdivision_errors else None,
                "corrected_graph_top1": _pose_error(virtual_scored[0][1], view.pose_w2c) if virtual_scored else None,
                "corrected_graph_refined_top1": virtual_refined["top1"],
                "coarse_score_top1": float(coarse_scores[0]) if coarse_scores.size else None,
                "structured_coarse_score_top1": float(structured_seed_scores[0]) if structured_seed_scores.size else None,
                "subdivision_score_top1": float(subdivision_scores[0]) if subdivision_scores.size else None,
            }
        identity_metrics = _region_identity_diagnostics(
            retrieval.groups, view, atlas, identity
        )
        row = {
            "image_id": view.image_id,
            "trajectory_id": view.trajectory_id,
            "raw_support_count": len(retrieval.groups),
            "evidence_group_count": current_graph.group_count,
            "evidence_group_size_median": float(np.median(np.diff(current_graph.member_offsets))),
            "evidence_group_size_maximum": int(np.max(np.diff(current_graph.member_offsets))),
            "query_undirected_edge_count": int(current_graph.edge_source.size),
            "grouping_feature_similarity": current_graph.grouping_feature_similarity,
            "o1_oracle_support_identity": o1,
            "o2_current_support_oracle_identity": o2,
            "o3_current_posterior_dense_local_lattice": o3,
            "o4_current_proposal_score": o4,
            "o5_virtual_pose_lattice": o5,
            **identity_metrics,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "stage": "v81_oracles_and_corrected_region_surface_graph",
        "query_count": len(rows),
        "strict_test_trajectory_ids": sorted({view.trajectory_id for view in views}),
        "pose_vote_trajectory_overlap": sorted(vote_overlap),
        "map_stores_one_vfm_feature_type": True,
        "runtime_stores_mapping_images": False,
        "uses_point_correspondence_or_pnp": False,
        "local_lattice_is_ground_truth_centered_diagnostic_only": not bool(args.skip_local_oracles),
        "local_lattice_pose_count": None if args.skip_local_oracles else 49 + int(args.lattice_random_samples),
        "uses_query_independent_virtual_pose_lattice": virtual_lattice is not None,
        "raw_support_count_median": float(np.median([row["raw_support_count"] for row in rows])) if rows else None,
        "evidence_group_count_median": float(np.median([row["evidence_group_count"] for row in rows])) if rows else None,
        **_pose_metrics([row["o1_oracle_support_identity"]["top1"] for row in rows if row["o1_oracle_support_identity"]], "o1_lattice_top1"),
        **_pose_metrics([row["o2_current_support_oracle_identity"]["top1"] for row in rows if row["o2_current_support_oracle_identity"]], "o2_lattice_top1"),
        **_pose_metrics([row["o3_current_posterior_dense_local_lattice"]["top1"] for row in rows if row["o3_current_posterior_dense_local_lattice"]], "o3_lattice_top1"),
        **_pose_metrics([row["o1_oracle_support_identity"]["refined"]["top1"] for row in rows if row["o1_oracle_support_identity"]], "o1_refined_top1"),
        **_pose_metrics([row["o2_current_support_oracle_identity"]["refined"]["top1"] for row in rows if row["o2_current_support_oracle_identity"]], "o2_refined_top1"),
        **_pose_metrics([row["o3_current_posterior_dense_local_lattice"]["refined"]["top1"] for row in rows if row["o3_current_posterior_dense_local_lattice"]], "o3_refined_top1"),
        **_pose_metrics([row["o4_current_proposal_score"]["frozen_v80_refined_top1"] for row in rows if row["o4_current_proposal_score"]["frozen_v80_refined_top1"]], "o4_frozen_v80_refined_top1"),
        **_pose_metrics([row["o4_current_proposal_score"]["corrected_v81_refined_top1"] for row in rows if row["o4_current_proposal_score"]["corrected_v81_refined_top1"]], "o4_corrected_v81_refined_top1"),
        **_pose_metrics([row["o5_virtual_pose_lattice"]["corrected_graph_top1"] for row in rows if row["o5_virtual_pose_lattice"] and row["o5_virtual_pose_lattice"]["corrected_graph_top1"]], "o5_virtual_graph_top1"),
        **_pose_metrics([row["o5_virtual_pose_lattice"]["corrected_graph_refined_top1"] for row in rows if row["o5_virtual_pose_lattice"] and row["o5_virtual_pose_lattice"]["corrected_graph_refined_top1"]], "o5_virtual_graph_refined_top1"),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
