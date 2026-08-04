"""Evaluate V8 single-feature query/maplet graph pose inference.

This evaluator is intentionally trajectory-strict.  Mapping pose statistics
only propose anonymous SE(3) modes; the query/physical graph likelihood ranks
them without reference images, image IDs, point correspondences, or PnP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

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
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.maplet_pose_voting import (
    AnonymousMapletPoseVoteBank,
    vote_maplet_poses,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    retrieve_candidate_groups,
)
from feature_extract.vfm.localization_v6.probability_calibration import (
    V6ProbabilityCalibration,
)
from feature_extract.vfm.localization_v8.structured_maplet_graph import (
    PhysicalMapletGraph,
    build_query_region_graph,
    refine_structured_maplet_pose,
    score_structured_maplet_pose,
)
from feature_extract.vfm.localization_v8.multi_teacher_student import (
    load_maplet_retrieval_adaptor,
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
    parser.add_argument("--localization_adaptor", default="")
    parser.add_argument("--probability_calibration", required=True)
    parser.add_argument("--physical_graph", required=True)
    parser.add_argument("--pose_vote_bank", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--maximum_pose_modes", type=int, default=64)
    parser.add_argument("--candidates_per_region", type=int, default=8)
    parser.add_argument("--pair_weight", type=float, default=0.5)
    parser.add_argument("--refine_seeds", type=int, default=2)
    parser.add_argument("--refine_stages", type=int, default=4)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _pose_metrics(errors: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not errors:
        return {}
    translation = np.asarray([x["translation_m"] for x in errors])
    rotation = np.asarray([x["rotation_deg"] for x in errors])
    return {
        f"{prefix}_translation_median_m": float(np.median(translation)),
        f"{prefix}_translation_p90_m": float(np.quantile(translation, 0.9)),
        f"{prefix}_rotation_median_deg": float(np.median(rotation)),
        f"{prefix}_rotation_p90_deg": float(np.quantile(rotation, 0.9)),
        f"{prefix}_30cm_3deg": float(np.mean((translation <= 0.30) & (rotation <= 3.0))),
        f"{prefix}_1m_10deg": float(np.mean((translation <= 1.0) & (rotation <= 10.0))),
    }


def _error(pose: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(pose, dtype=np.float64)
    target = np.asarray(ground_truth, dtype=np.float64)
    estimate_center = -estimate[:3, :3].T @ estimate[:3, 3]
    target_center = -target[:3, :3].T @ target[:3, 3]
    relative = estimate[:3, :3] @ target[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return {
        "translation_m": float(np.linalg.norm(estimate_center - target_center)),
        "rotation_deg": float(np.degrees(np.arccos(cosine))),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite V8 graph evaluation")
    identity_path = Path(args.identity_bank)
    identity = SurfaceRetrievalMapletBank.load_npz(identity_path)
    spatial = SurfaceRetrievalMapletBank.load_npz(Path(args.spatial_bank))
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper), device=str(args.device)
    )
    localization_adaptor = None
    if str(args.localization_adaptor):
        localization_adaptor, _adaptor_metadata = load_maplet_retrieval_adaptor(
            Path(args.localization_adaptor), device=str(args.device)
        )
    calibration = V6ProbabilityCalibration.load_json(
        Path(args.probability_calibration)
    )
    atlas = MapletFeatureAtlasBank.load_npz(Path(args.atlas_geometry))
    physical = PhysicalMapletGraph.load_npz(Path(args.physical_graph))
    votes = AnonymousMapletPoseVoteBank.load_npz(Path(args.pose_vote_bank))
    import hashlib
    if physical.feature_bank_sha256 != hashlib.sha256(identity_path.read_bytes()).hexdigest():
        raise ValueError("physical graph and canonical RADIO bank lineage differ")
    test_trajectories = {"seq3", "seq5", "seq13"}
    overlap = test_trajectories & set((votes.metadata or {}).get("mapping_trajectory_ids", []))
    if overlap:
        raise ValueError(f"strict test trajectory leaked into pose proposals: {sorted(overlap)}")
    views = _load_views(Path(args.contributors), atlas, Path(args.image_root))
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid query shard")
    views = views[int(args.shard_index) :: int(args.shard_count)]
    if int(args.max_queries) > 0:
        positions = np.linspace(0, len(views) - 1, int(args.max_queries), dtype=np.int64)
        views = [views[int(x)] for x in positions.tolist()]
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
        if localization_adaptor is not None:
            descriptors = localization_adaptor.project_numpy(
                descriptors, device=str(args.device)
            )
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
        hypotheses = vote_maplet_poses(
            descriptors,
            identity,
            votes,
            candidate_groups=retrieval.groups,
            image_size_wh=(view.camera.width, view.camera.height),
            camera=view.camera,
            maximum_modes=int(args.maximum_pose_modes),
            candidates_per_region=int(args.candidates_per_region),
        )
        query_graph = build_query_region_graph(
            retrieval,
            physical,
            image_size_wh=(view.camera.width, view.camera.height),
            candidates_per_region=int(args.candidates_per_region),
        )
        scored = []
        for hypothesis in hypotheses:
            score, parts = score_structured_maplet_pose(
                hypothesis.pose_w2c,
                query_graph,
                physical,
                view.camera,
                pair_weight=float(args.pair_weight),
            )
            scored.append((score, hypothesis, parts))
        scored.sort(key=lambda x: -x[0])
        gt_score, gt_parts = score_structured_maplet_pose(
            view.pose_w2c,
            query_graph,
            physical,
            view.camera,
            pair_weight=float(args.pair_weight),
        )
        baseline_errors = [_error(x.pose_w2c, view.pose_w2c) for x in hypotheses]
        graph_errors = [_error(x[1].pose_w2c, view.pose_w2c) for x in scored]
        seed_candidates = []
        if hypotheses:
            seed_candidates.append(hypotheses[0])
        seed_candidates.extend(x[1] for x in scored[: max(int(args.refine_seeds), 0)])
        unique_seeds = []
        for seed in seed_candidates:
            if any(np.allclose(seed.pose_w2c, old.pose_w2c) for old in unique_seeds):
                continue
            unique_seeds.append(seed)
            if len(unique_seeds) >= max(int(args.refine_seeds), 0):
                break
        refined = []
        for seed in unique_seeds:
            pose, score, parts = refine_structured_maplet_pose(
                seed.pose_w2c,
                query_graph,
                physical,
                view.camera,
                pair_weight=float(args.pair_weight),
                stages=int(args.refine_stages),
            )
            refined.append((score, pose, parts))
        refined.sort(key=lambda x: -x[0])
        refined_errors = [_error(x[1], view.pose_w2c) for x in refined]
        gt_rank = 1 + sum(float(x[0]) > float(gt_score) for x in scored)
        identity_metrics = _region_identity_diagnostics(
            retrieval.groups, view, atlas, identity
        )
        row = {
            "image_id": view.image_id,
            "query_region_count": len(query_graph.xy),
            "query_edge_count": int(query_graph.edge_source.size),
            "pose_hypothesis_count": len(hypotheses),
            "baseline_top1": baseline_errors[0] if baseline_errors else None,
            "graph_top1": graph_errors[0] if graph_errors else None,
            "graph_refined_top1": refined_errors[0] if refined_errors else None,
            "baseline_oracle": min(baseline_errors, key=lambda x: x["translation_m"] + x["rotation_deg"] / 10.0) if baseline_errors else None,
            "graph_oracle": min(graph_errors, key=lambda x: x["translation_m"] + x["rotation_deg"] / 10.0) if graph_errors else None,
            "ground_truth_graph_score": float(gt_score),
            "ground_truth_graph_score_parts": gt_parts,
            "ground_truth_rank_among_proposals": int(gt_rank),
            "best_proposal_graph_score": float(scored[0][0]) if scored else None,
            "best_proposal_graph_score_parts": scored[0][2] if scored else None,
            "best_refined_graph_score": float(refined[0][0]) if refined else None,
            "best_refined_graph_score_parts": refined[0][2] if refined else None,
            **identity_metrics,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    baseline_top1 = [x["baseline_top1"] for x in rows if x["baseline_top1"]]
    graph_top1 = [x["graph_top1"] for x in rows if x["graph_top1"]]
    refined_top1 = [x["graph_refined_top1"] for x in rows if x["graph_refined_top1"]]
    report = {
        "stage": "v8_single_feature_structured_maplet_graph",
        "query_count": len(rows),
        "strict_test_trajectory_ids": sorted({x.trajectory_id for x in views}),
        "pose_proposal_trajectory_overlap": sorted(overlap),
        "map_stores_one_vfm_feature_type": True,
        "physical_graph_stores_descriptor_arrays": False,
        "physical_graph_node_count": int(physical.maplet_ids.size),
        "physical_graph_edge_count": int(physical.edge_source.size),
        "pair_weight": float(args.pair_weight),
        **_pose_metrics(baseline_top1, "baseline_top1"),
        **_pose_metrics(graph_top1, "graph_top1"),
        **_pose_metrics(refined_top1, "graph_refined_top1"),
        "ground_truth_preferred_to_all_proposals_fraction": float(np.mean([x["ground_truth_rank_among_proposals"] == 1 for x in rows])) if rows else 0.0,
        "ground_truth_graph_rank_median": float(np.median([x["ground_truth_rank_among_proposals"] for x in rows])) if rows else None,
        "region_true_maplet_recall_at_1": float(np.mean([x["region_true_maplet_recall_at_1"] for x in rows])) if rows else None,
        "region_true_maplet_recall_at_5": float(np.mean([x["region_true_maplet_recall_at_5"] for x in rows])) if rows else None,
        "region_true_maplet_recall_at_64": float(np.mean([x["region_true_maplet_recall_at_64"] for x in rows])) if rows else None,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
