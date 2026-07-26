"""Pose-free online localization against a track-free 2DGS surface map."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_cameras_binary
from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
    VfmGuidedLoFTRMatcher,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
    LocalFeatureFrame,
    SurfaceAnchorObservationIndex,
    SurfaceMapletMatchConfig,
    SurfacePoseConfig,
    SurfacePoseResult,
    VfmSurfaceObservationIndex,
    build_maplet_conditioned_surface_anchor_candidate_pool,
    build_pose_guided_surface_anchor_candidate_pool,
    build_seeded_surface_anchor_candidate_pool,
    build_support_layout_guided_local_query,
    build_vfm_surface_observation_candidate_pool,
    build_vfm_support_guided_local_query,
    estimate_vfm_query_to_support_layout,
    generate_grouped_surface_pose_hypotheses,
    lift_pairwise_matches_to_surface_anchors,
    lift_pairwise_matches_with_2dgs_depth,
    match_radio_final_regions_to_maplets,
    predict_vfm_support_anchor_query_points,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    StableSurfaceAnchorMap,
    VfmSurfaceMapletBank,
    encode_radio_final_regions,
)
from feature_extract.vfm.surface_depth_bank import TwoDGSSurfaceDepthBank
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_mapping import Vfm2DgsObservationBank


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_image_root", required=True)
    parser.add_argument("--radio_final_layer", default="radio_final")
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--local_descriptor_bank", required=True)
    parser.add_argument("--observation_bank", required=True)
    parser.add_argument(
        "--surface_depth_bank",
        default="",
        help="Optional official-2DGS mapping-view depth manifest.",
    )
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument(
        "--loftr_hloc_root",
        default="third_party/Hierarchical-Localization",
    )
    parser.add_argument(
        "--loftr_checkpoint",
        default="/root/.cache/torch/hub/checkpoints/loftr_outdoor.ckpt",
    )
    parser.add_argument("--loftr_weights", default="outdoor", choices=("outdoor", "indoor"))
    parser.add_argument("--loftr_match_threshold", type=float, default=0.2)
    parser.add_argument("--loftr_resize_width", type=int, default=960)
    parser.add_argument("--loftr_resize_height", type=int, default=540)
    parser.add_argument("--loftr_pair_batch_size", type=int, default=4)
    parser.add_argument(
        "--loftr_anchor_maximum_distance_px",
        type=float,
        default=5.0,
    )
    parser.add_argument("--loftr_depth_maximum_points", type=int, default=768)
    parser.add_argument("--loftr_depth_query_nms_radius_px", type=float, default=2.0)
    parser.add_argument(
        "--loftr_cross_modal_max_translation_m",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--loftr_cross_modal_max_rotation_deg",
        type=float,
        default=10.0,
    )
    parser.add_argument("--disable_vfm_guided_loftr", action="store_true")
    parser.add_argument("--query_region_rows", type=int, default=8)
    parser.add_argument("--query_region_cols", type=int, default=8)
    parser.add_argument("--query_regions_per_cell", type=int, default=2)
    parser.add_argument("--maplet_top_k", type=int, default=8)
    parser.add_argument("--maximum_support_views", type=int, default=64)
    parser.add_argument("--maximum_layout_models", type=int, default=4096)
    parser.add_argument("--maximum_maplets_per_region", type=int, default=3)
    parser.add_argument("--local_top_k", type=int, default=2048)
    parser.add_argument("--local_candidate_top_k", type=int, default=8192)
    parser.add_argument("--local_nms_radius_px", type=float, default=2.0)
    parser.add_argument("--guided_local_search_radius_px", type=float, default=32.0)
    parser.add_argument("--guided_local_max_points", type=int, default=512)
    parser.add_argument("--anchor_top_l", type=int, default=5)
    parser.add_argument("--vfm_support_views", type=int, default=4)
    parser.add_argument("--vfm_global_support_views", type=int, default=64)
    parser.add_argument("--vfm_max_query_points", type=int, default=768)
    parser.add_argument("--vfm_minimum_similarity", type=float, default=0.20)
    parser.add_argument(
        "--mapping_pose_translation_prior_weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--mapping_pose_rotation_prior_weight",
        type=float,
        default=0.10,
    )
    parser.add_argument("--hypothesis_count", type=int, default=256)
    parser.add_argument("--minimum_fit_groups", type=int, default=8)
    parser.add_argument("--minimum_verification_groups", type=int, default=4)
    parser.add_argument("--heldout_stride", type=int, default=5)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _camera_from_model(model_dir: Path):
    cameras = read_colmap_cameras_binary(Path(model_dir) / "cameras.bin")
    if not cameras:
        raise ValueError("camera model contains no cameras")
    structural = {
        (int(camera.model_id), int(camera.width), int(camera.height), len(camera.params))
        for camera in cameras.values()
    }
    if len(structural) != 1:
        raise ValueError("training camera model has incompatible intrinsic families")
    model_id, width, height, parameter_count = next(iter(structural))
    parameter_matrix = np.asarray(
        [camera.params for camera in cameras.values()],
        dtype=np.float64,
    ).reshape(-1, int(parameter_count))
    # Test-image camera IDs live in a file that also stores test poses.  To keep
    # inference pose-free, use the robust median of training-camera intrinsics.
    return ColmapCamera(
        camera_id=0,
        model_id=int(model_id),
        width=int(width),
        height=int(height),
        params=tuple(float(value) for value in np.median(parameter_matrix, axis=0).tolist()),
    )


def _load_raw_final(path: Path, layer_name: str) -> np.ndarray:
    with np.load(Path(path)) as data:
        if str(layer_name) not in data:
            raise ValueError(f"RADIO-final layer {layer_name!r} is absent from {path}")
        feature = np.asarray(data[str(layer_name)], dtype=np.float32)
    if feature.ndim == 4 and int(feature.shape[0]) == 1:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError("RADIO-final query feature must have shape (C,H,W)")
    return feature


def _pose_record(result, image_id: str, diagnostics: dict[str, object]) -> dict[str, object]:
    hypotheses = [
        {
            "pose_w2c": hypothesis.pose_w2c.reshape(-1).tolist(),
            "generation_score": float(hypothesis.generation_score),
            "verification_score": float(hypothesis.verification_score),
            "inlier_count": int(hypothesis.inlier_count),
            "source": str(hypothesis.source),
        }
        for hypothesis in result.hypotheses[:16]
    ]
    return {
        "image_id": str(image_id),
        "success": bool(result.success),
        "pose_w2c": result.pose_w2c.reshape(-1).tolist() if result.success else None,
        "failure_reason": result.failure_reason,
        "fit_group_count": int(np.sum(result.fit_mask)),
        "verification_group_count": int(np.sum(result.verification_mask)),
        "hypothesis_count": int(len(result.hypotheses)),
        "hypotheses": hypotheses,
        "diagnostics": diagnostics,
    }


def _select_pose_by_geometric_support(
    result: SurfacePoseResult,
    minimum_selected_inliers: int,
) -> SurfacePoseResult:
    """Select within one fixed pool without a retrieved-pose proximity prior."""

    if not result.hypotheses:
        return result
    ranked = sorted(
        result.hypotheses,
        key=lambda hypothesis: (
            int(hypothesis.inlier_count),
            float(hypothesis.verification_score),
            float(hypothesis.generation_score),
        ),
        reverse=True,
    )
    selected = ranked[0]
    success = int(selected.inlier_count) >= int(minimum_selected_inliers)
    return SurfacePoseResult(
        success=bool(success),
        pose_w2c=selected.pose_w2c if success else np.eye(4),
        hypotheses=tuple(ranked),
        fit_mask=result.fit_mask,
        verification_mask=result.verification_mask,
        failure_reason=None if success else "insufficient_selected_pose_inliers",
    )


def _direct_support_result_key(
    result: SurfacePoseResult,
    *,
    layout_inliers: int,
    layout_median_residual: float,
    support_rank: int,
) -> tuple[float, ...]:
    if not result.hypotheses:
        return (
            float(bool(result.success)),
            -1.0,
            -np.inf,
            -np.inf,
            float(layout_inliers),
            -float(layout_median_residual),
            -float(support_rank),
        )
    selected = result.hypotheses[0]
    verification_groups = max(int(np.sum(result.verification_mask)), 1)
    fit_groups = max(int(np.sum(result.fit_mask)), 1)
    return (
        float(bool(result.success)),
        float(selected.inlier_count),
        float(selected.verification_score) / float(verification_groups),
        float(selected.generation_score) / float(fit_groups),
        float(layout_inliers),
        -float(layout_median_residual),
        -float(support_rank),
    )


def _pose_disagreement(
    first_pose_w2c: np.ndarray,
    second_pose_w2c: np.ndarray,
) -> tuple[float, float]:
    first = np.asarray(first_pose_w2c, dtype=np.float64).reshape(4, 4)
    second = np.asarray(second_pose_w2c, dtype=np.float64).reshape(4, 4)
    first_center = -first[:3, :3].T @ first[:3, 3]
    second_center = -second[:3, :3].T @ second[:3, 3]
    translation = float(np.linalg.norm(first_center - second_center))
    relative = first[:3, :3] @ second[:3, :3].T
    cosine = float(
        np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    )
    return translation, float(np.degrees(np.arccos(cosine)))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if (
        float(args.loftr_cross_modal_max_translation_m) <= 0.0
        or float(args.loftr_cross_modal_max_rotation_deg) <= 0.0
        or int(args.loftr_depth_maximum_points) <= 0
        or float(args.loftr_depth_query_nms_radius_px) < 0.0
    ):
        raise ValueError("LoFTR 2DGS depth-verification limits are invalid")
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite query localization outputs")

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    if not records:
        raise ValueError("query manifest contains no selected records")
    maplets = VfmSurfaceMapletBank.load_npz(Path(args.maplets))
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    local_bank = AnchorLocalDescriptorBank.load_npz(Path(args.local_descriptor_bank))
    observation_bank = Vfm2DgsObservationBank.load_npz(Path(args.observation_bank))
    camera = _camera_from_model(Path(args.camera_model_dir))
    observation_index = VfmSurfaceObservationIndex.from_bank(observation_bank)
    anchor_observation_index = SurfaceAnchorObservationIndex.from_anchor_map(
        anchors
    )
    surface_depth_bank = (
        TwoDGSSurfaceDepthBank.load_json(Path(args.surface_depth_bank))
        if str(args.surface_depth_bank)
        else None
    )
    if surface_depth_bank is not None and (
        surface_depth_bank.width != int(camera.width)
        or surface_depth_bank.height != int(camera.height)
    ):
        raise ValueError("2DGS surface depth bank and camera-model grids differ")
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint),
        device=str(args.device),
    )
    for metadata, name in (
        (maplets.metadata, "maplet bank"),
        (anchors.metadata, "anchor map"),
        (local_bank.metadata, "local descriptor bank"),
        (observation_bank.metadata, "surface observation bank"),
        (mapper_metadata, "surface mapper"),
    ):
        if bool(metadata.get("uses_radio_intermediate", False)):
            raise ValueError(f"{name} illegally uses RADIO intermediate")
        if bool(metadata.get("uses_sfm_points", False)):
            raise ValueError(f"{name} illegally uses SfM points")
        if bool(metadata.get("uses_sfm_tracks", False)):
            raise ValueError(f"{name} illegally uses SfM tracks")
    pool_sizes = tuple(int(value) for value in mapper_metadata.get("pool_sizes", (1, 3, 5, 9)))
    pool_weights = tuple(
        float(value) for value in mapper_metadata.get("pool_weights", (0.4, 0.3, 0.2, 0.1))
    )
    region_config = RadioFinalRegionConfig(pool_sizes=pool_sizes, pool_weights=pool_weights)
    match_config = SurfaceMapletMatchConfig(
        top_k=int(args.maplet_top_k),
        maximum_support_views=int(args.maximum_support_views),
        maximum_layout_models=int(args.maximum_layout_models),
    )
    pose_config = SurfacePoseConfig(
        hypothesis_count=int(args.hypothesis_count),
        minimum_fit_groups=int(args.minimum_fit_groups),
        minimum_verification_groups=int(args.minimum_verification_groups),
        heldout_stride=int(args.heldout_stride),
    )
    vfm_pose_config = SurfacePoseConfig(
        hypothesis_count=int(args.hypothesis_count),
        minimum_fit_groups=int(args.minimum_fit_groups),
        minimum_verification_groups=int(args.minimum_verification_groups),
        heldout_stride=int(args.heldout_stride),
        reprojection_sigma_px=10.0,
        inlier_threshold_px=24.0,
    )
    mapping_pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.mapping_pose_file))
    }
    image_root = Path(args.query_image_root)
    alike = AlikeDenseObservationExtractor(
        device=str(args.device),
        matcha_repo=Path(args.matcha_repo),
        model_name=str(args.alike_model_name),
    )
    loftr = (
        None
        if bool(args.disable_vfm_guided_loftr)
        else VfmGuidedLoFTRMatcher(
            device=str(args.device),
            hloc_root=Path(args.loftr_hloc_root),
            checkpoint=Path(args.loftr_checkpoint),
            weights=str(args.loftr_weights),
            match_threshold=float(args.loftr_match_threshold),
            resize_width=int(args.loftr_resize_width),
            resize_height=int(args.loftr_resize_height),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    successes = 0
    failure_counts: dict[str, int] = {}
    with output_path.open("w") as output:
        for index, record in enumerate(records):
            raw = _load_raw_final(record.token_path, str(args.radio_final_layer))
            mapped = mapper.project(raw).coarse_descriptors
            vfm_candidate_pool, vfm_support_views, vfm_support_scores = (
                build_vfm_surface_observation_candidate_pool(
                    mapped,
                    query_image_size=(int(camera.width), int(camera.height)),
                    observation_bank=observation_bank,
                    maximum_global_support_views=int(args.vfm_global_support_views),
                    maximum_support_views=int(args.vfm_support_views),
                    maximum_query_points=int(args.vfm_max_query_points),
                    top_l=int(args.anchor_top_l),
                    minimum_similarity=float(args.vfm_minimum_similarity),
                    observation_index=observation_index,
                )
            )
            vfm_pose_result = generate_grouped_surface_pose_hypotheses(
                vfm_candidate_pool,
                camera,
                vfm_pose_config,
            )
            _token_indices, region_xy = select_spatially_balanced_radio_final_regions(
                raw,
                grid_rows=int(args.query_region_rows),
                grid_cols=int(args.query_region_cols),
                regions_per_cell=int(args.query_regions_per_cell),
            )
            region_descriptors = encode_radio_final_regions(
                mapped,
                region_xy,
                region_config,
            )
            grid_size = (int(mapped.shape[2]), int(mapped.shape[1]))
            maplet_match = match_radio_final_regions_to_maplets(
                region_xy,
                region_descriptors,
                grid_size,
                maplets,
                match_config,
            )
            image_path = image_root / record.image_id
            direct_hypothesis_count = max(
                64,
                int(np.ceil(int(args.hypothesis_count) / max(len(vfm_support_views), 1))),
            )
            direct_pose_config = replace(
                pose_config,
                hypothesis_count=direct_hypothesis_count,
            )
            loftr_result = None
            loftr_pool = None
            loftr_selected_support_view = None
            loftr_depth_result = None
            loftr_depth_pool = None
            loftr_depth_selected_support_view = None
            loftr_support_runs: list[dict[str, object]] = []
            if loftr is not None and vfm_support_views:
                pair_matches = loftr.match_support_views(
                    image_path,
                    [image_root / support for support in vfm_support_views],
                    camera_width=int(camera.width),
                    camera_height=int(camera.height),
                    batch_size=int(args.loftr_pair_batch_size),
                )
                for support_rank, (support_view_id, matches) in enumerate(
                    zip(vfm_support_views, pair_matches)
                ):
                    lifted_pool = lift_pairwise_matches_to_surface_anchors(
                        matches.query_xy,
                        matches.support_xy,
                        matches.confidence,
                        support_view_id=support_view_id,
                        observation_index=anchor_observation_index,
                        maximum_support_distance_px=float(
                            args.loftr_anchor_maximum_distance_px
                        ),
                    )
                    lifted_result = generate_grouped_surface_pose_hypotheses(
                        lifted_pool,
                        camera,
                        direct_pose_config,
                    )
                    lifted_result = _select_pose_by_geometric_support(
                        lifted_result,
                        direct_pose_config.minimum_selected_inliers,
                    )
                    depth_pool = None
                    depth_result = None
                    if surface_depth_bank is not None:
                        support_pose = mapping_pose_by_image.get(
                            str(support_view_id)
                        )
                        if support_pose is None:
                            raise KeyError(
                                f"mapping pose missing for VFM support {support_view_id}"
                            )
                        depth_pool = lift_pairwise_matches_with_2dgs_depth(
                            matches.query_xy,
                            matches.support_xy,
                            matches.confidence,
                            support_pose_w2c=support_pose,
                            support_depth=surface_depth_bank.depth_for_image(
                                str(support_view_id)
                            ),
                            camera=camera,
                            maximum_points=int(
                                args.loftr_depth_maximum_points
                            ),
                            query_nms_radius_px=float(
                                args.loftr_depth_query_nms_radius_px
                            ),
                        )
                        depth_result = generate_grouped_surface_pose_hypotheses(
                            depth_pool,
                            camera,
                            direct_pose_config,
                        )
                        depth_result = _select_pose_by_geometric_support(
                            depth_result,
                            direct_pose_config.minimum_selected_inliers,
                        )
                    cross_modal_translation = float("inf")
                    cross_modal_rotation = float("inf")
                    if (
                        lifted_result.success
                        and depth_result is not None
                        and depth_result.success
                    ):
                        (
                            cross_modal_translation,
                            cross_modal_rotation,
                        ) = _pose_disagreement(
                            lifted_result.pose_w2c,
                            depth_result.pose_w2c,
                        )
                    loftr_support_runs.append(
                        {
                            "support_rank": int(support_rank),
                            "support_view_id": str(support_view_id),
                            "pair_match_count": int(len(matches.confidence)),
                            "pool": lifted_pool,
                            "result": lifted_result,
                            "depth_pool": depth_pool,
                            "depth_result": depth_result,
                            "cross_modal_translation_m": float(
                                cross_modal_translation
                            ),
                            "cross_modal_rotation_deg": float(
                                cross_modal_rotation
                            ),
                        }
                    )
                if loftr_support_runs:
                    selected_loftr_run = max(
                        loftr_support_runs,
                        key=lambda run: _direct_support_result_key(
                            run["result"],
                            layout_inliers=int(len(run["pool"])),
                            layout_median_residual=0.0,
                            support_rank=int(run["support_rank"]),
                        ),
                    )
                    loftr_result = selected_loftr_run["result"]
                    loftr_pool = selected_loftr_run["pool"]
                    loftr_selected_support_view = str(
                        selected_loftr_run["support_view_id"]
                    )
                depth_support_runs = [
                    run
                    for run in loftr_support_runs
                    if isinstance(run["depth_result"], SurfacePoseResult)
                ]
                if depth_support_runs:
                    all_consensus_depth_runs = [
                        run
                        for run in depth_support_runs
                        if run["depth_result"].success
                        and run["result"].success
                        and np.isfinite(
                            float(run["cross_modal_translation_m"])
                        )
                        and np.isfinite(
                            float(run["cross_modal_rotation_deg"])
                        )
                    ]
                    consensus_depth_runs = [
                        run
                        for run in all_consensus_depth_runs
                        if float(run["cross_modal_translation_m"])
                        <= float(
                            args.loftr_cross_modal_max_translation_m
                        )
                        and float(run["cross_modal_rotation_deg"])
                        <= float(args.loftr_cross_modal_max_rotation_deg)
                    ]
                    selected_depth_run = None
                    if consensus_depth_runs:
                        selected_depth_run = min(
                            consensus_depth_runs,
                            key=lambda run: (
                                float(run["cross_modal_translation_m"])
                                + 0.10
                                * float(run["cross_modal_rotation_deg"]),
                                -int(
                                    run["result"].hypotheses[0].inlier_count
                                ),
                                -int(
                                    run["depth_result"].hypotheses[
                                        0
                                    ].inlier_count
                                ),
                                int(run["support_rank"]),
                            ),
                        )
                    elif not all_consensus_depth_runs:
                        selected_depth_run = max(
                            depth_support_runs,
                            key=lambda run: _direct_support_result_key(
                                run["depth_result"],
                                layout_inliers=int(len(run["depth_pool"])),
                                layout_median_residual=0.0,
                                support_rank=int(run["support_rank"]),
                            ),
                        )
                    if selected_depth_run is not None:
                        loftr_depth_result = selected_depth_run[
                            "depth_result"
                        ]
                        loftr_depth_pool = selected_depth_run["depth_pool"]
                        loftr_depth_selected_support_view = str(
                            selected_depth_run["support_view_id"]
                        )
            if loftr_depth_result is not None and loftr_depth_result.success:
                detected_query = LocalFeatureFrame(
                    image_id=record.image_id,
                    keypoints_xy=np.zeros((0, 2), dtype=np.float32),
                    descriptors=np.zeros(
                        (0, local_bank.feature_dim),
                        dtype=np.float32,
                    ),
                    scores=np.zeros((0,), dtype=np.float32),
                )
            else:
                detected = alike.detect(
                    image_path,
                    image_width=int(camera.width),
                    image_height=int(camera.height),
                    top_k=int(args.local_top_k),
                    candidate_top_k=int(args.local_candidate_top_k),
                    nms_radius_px=float(args.local_nms_radius_px),
                    grid_rows=8,
                    grid_cols=8,
                )
                detected_query = LocalFeatureFrame(
                    image_id=record.image_id,
                    keypoints_xy=detected.xy,
                    descriptors=detected.descriptors,
                    scores=detected.scores,
                )
            direct_layout_matrix = None
            direct_layout_translation = None
            direct_layout_inliers = 0
            direct_layout_median_residual = float("inf")
            direct_guided_result = None
            direct_guided_pool = None
            direct_dense_guided_result = None
            direct_dense_guided_pool = None
            direct_guided_search_radius_px = float(
                args.guided_local_search_radius_px
            )
            direct_selected_support_view = None
            direct_support_runs: list[dict[str, object]] = []
            for support_rank, support_view_id in enumerate(vfm_support_views):
                (
                    layout_matrix,
                    layout_translation,
                    layout_inliers,
                    layout_median_residual,
                ) = estimate_vfm_query_to_support_layout(
                    mapped,
                    observation_bank,
                    support_view_id,
                    minimum_similarity=float(args.vfm_minimum_similarity),
                    observation_index=observation_index,
                )
                support_run: dict[str, object] = {
                    "support_rank": int(support_rank),
                    "support_view_id": str(support_view_id),
                    "layout_matrix": layout_matrix,
                    "layout_translation": layout_translation,
                    "layout_inliers": int(layout_inliers),
                    "layout_median_residual": float(layout_median_residual),
                    "dense_pool": None,
                    "dense_result": None,
                    "sparse_pool": None,
                    "sparse_result": None,
                }
                direct_support_runs.append(support_run)
                if (
                    layout_matrix is None
                    or (
                        loftr_depth_result is not None
                        and loftr_depth_result.success
                    )
                ):
                    continue
                (
                    predicted_anchor_xy,
                    support_anchor_descriptors,
                    predicted_anchor_ids,
                ) = predict_vfm_support_anchor_query_points(
                    query_image_size=(int(camera.width), int(camera.height)),
                    query_grid_size=grid_size,
                    support_view_id=support_view_id,
                    query_to_support_matrix=layout_matrix,
                    query_to_support_translation=layout_translation,
                    anchors=anchors,
                    descriptor_bank=local_bank,
                    maximum_points=min(
                        int(args.guided_local_max_points),
                        256,
                    ),
                )
                if len(predicted_anchor_xy) < 4:
                    continue
                (
                    measured_anchor_xy,
                    measured_anchor_descriptors,
                    measured_anchor_scores,
                    _measured_image_hash,
                ) = alike.match_descriptor_points(
                    image_path,
                    predicted_anchor_xy,
                    support_anchor_descriptors,
                    image_width=int(camera.width),
                    image_height=int(camera.height),
                    search_radius_px=int(
                        np.ceil(direct_guided_search_radius_px)
                    ),
                    search_step_px=1,
                )
                direct_dense_query = LocalFeatureFrame(
                    image_id=record.image_id,
                    keypoints_xy=measured_anchor_xy,
                    descriptors=measured_anchor_descriptors,
                    scores=measured_anchor_scores,
                )
                dense_pool = build_seeded_surface_anchor_candidate_pool(
                    direct_dense_query,
                    predicted_anchor_ids,
                    descriptor_bank=local_bank,
                    anchors=anchors,
                    top_l=int(args.anchor_top_l),
                )
                dense_result = generate_grouped_surface_pose_hypotheses(
                    dense_pool,
                    camera,
                    direct_pose_config,
                )
                dense_result = _select_pose_by_geometric_support(
                    dense_result,
                    direct_pose_config.minimum_selected_inliers,
                )
                support_run["dense_pool"] = dense_pool
                support_run["dense_result"] = dense_result

            dense_support_runs = [
                run
                for run in direct_support_runs
                if isinstance(run["dense_result"], SurfacePoseResult)
            ]
            if dense_support_runs:
                selected_dense_run = max(
                    dense_support_runs,
                    key=lambda run: _direct_support_result_key(
                        run["dense_result"],
                        layout_inliers=int(run["layout_inliers"]),
                        layout_median_residual=float(run["layout_median_residual"]),
                        support_rank=int(run["support_rank"]),
                    ),
                )
                direct_dense_guided_result = selected_dense_run["dense_result"]
                direct_dense_guided_pool = selected_dense_run["dense_pool"]
                direct_layout_matrix = selected_dense_run["layout_matrix"]
                direct_layout_translation = selected_dense_run["layout_translation"]
                direct_layout_inliers = int(selected_dense_run["layout_inliers"])
                direct_layout_median_residual = float(
                    selected_dense_run["layout_median_residual"]
                )
                direct_selected_support_view = str(
                    selected_dense_run["support_view_id"]
                )

            if (
                direct_dense_guided_result is None
                or not direct_dense_guided_result.success
            ) and not (
                loftr_depth_result is not None
                and loftr_depth_result.success
            ):
                for support_run in direct_support_runs:
                    layout_matrix = support_run["layout_matrix"]
                    if layout_matrix is None:
                        continue
                    direct_local_query, direct_seed_anchor_ids = (
                        build_vfm_support_guided_local_query(
                            detected_query,
                            query_image_size=(int(camera.width), int(camera.height)),
                            query_grid_size=grid_size,
                            support_view_id=str(support_run["support_view_id"]),
                            query_to_support_matrix=layout_matrix,
                            query_to_support_translation=support_run[
                                "layout_translation"
                            ],
                            anchors=anchors,
                            descriptor_bank=local_bank,
                            search_radius_px=direct_guided_search_radius_px,
                            maximum_query_points=int(args.guided_local_max_points),
                        )
                    )
                    if not np.all(direct_seed_anchor_ids >= 0):
                        continue
                    sparse_pool = build_seeded_surface_anchor_candidate_pool(
                        direct_local_query,
                        direct_seed_anchor_ids,
                        descriptor_bank=local_bank,
                        anchors=anchors,
                        top_l=int(args.anchor_top_l),
                    )
                    sparse_result = generate_grouped_surface_pose_hypotheses(
                        sparse_pool,
                        camera,
                        direct_pose_config,
                    )
                    sparse_result = _select_pose_by_geometric_support(
                        sparse_result,
                        direct_pose_config.minimum_selected_inliers,
                    )
                    support_run["sparse_pool"] = sparse_pool
                    support_run["sparse_result"] = sparse_result
                sparse_support_runs = [
                    run
                    for run in direct_support_runs
                    if isinstance(run["sparse_result"], SurfacePoseResult)
                ]
                if sparse_support_runs:
                    selected_sparse_run = max(
                        sparse_support_runs,
                        key=lambda run: _direct_support_result_key(
                            run["sparse_result"],
                            layout_inliers=int(run["layout_inliers"]),
                            layout_median_residual=float(
                                run["layout_median_residual"]
                            ),
                            support_rank=int(run["support_rank"]),
                        ),
                    )
                    direct_guided_result = selected_sparse_run["sparse_result"]
                    direct_guided_pool = selected_sparse_run["sparse_pool"]
                    if (
                        direct_dense_guided_result is None
                        or not direct_dense_guided_result.success
                    ):
                        direct_layout_matrix = selected_sparse_run["layout_matrix"]
                        direct_layout_translation = selected_sparse_run[
                            "layout_translation"
                        ]
                        direct_layout_inliers = int(
                            selected_sparse_run["layout_inliers"]
                        )
                        direct_layout_median_residual = float(
                            selected_sparse_run["layout_median_residual"]
                        )
                        direct_selected_support_view = str(
                            selected_sparse_run["support_view_id"]
                        )
            local_query, seed_anchor_ids = build_support_layout_guided_local_query(
                detected_query,
                query_image_size=(int(camera.width), int(camera.height)),
                maplet_match=maplet_match,
                maplets=maplets,
                anchors=anchors,
                descriptor_bank=local_bank,
                maximum_maplets_per_region=int(args.maximum_maplets_per_region),
                search_radius_px=float(args.guided_local_search_radius_px),
                maximum_query_points=int(args.guided_local_max_points),
            )
            if np.all(seed_anchor_ids >= 0):
                candidate_pool = build_seeded_surface_anchor_candidate_pool(
                    local_query,
                    seed_anchor_ids,
                    descriptor_bank=local_bank,
                    anchors=anchors,
                    top_l=int(args.anchor_top_l),
                )
            else:
                candidate_pool = build_maplet_conditioned_surface_anchor_candidate_pool(
                    local_query,
                    query_image_size=(int(camera.width), int(camera.height)),
                    query_region_xy=region_xy,
                    query_region_grid_size=grid_size,
                    maplet_match=maplet_match,
                    maplets=maplets,
                    descriptor_bank=local_bank,
                    anchors=anchors,
                    top_l=int(args.anchor_top_l),
                    maximum_maplets_per_region=int(args.maximum_maplets_per_region),
                )
            pose_result = generate_grouped_surface_pose_hypotheses(
                candidate_pool,
                camera,
                pose_config,
            )
            pose_guided_pool = None
            pose_guided_result = None
            pose_guided_initial_pose = None
            pose_guided_initial_source = None
            if vfm_support_views and vfm_support_views[0] in mapping_pose_by_image:
                pose_guided_initial_pose = mapping_pose_by_image[vfm_support_views[0]]
                pose_guided_initial_source = "vfm_retrieved_mapping_view"
            elif vfm_pose_result.hypotheses:
                pose_guided_initial_pose = vfm_pose_result.hypotheses[0].pose_w2c
                pose_guided_initial_source = "vfm_surface_pnp"
            if pose_guided_initial_pose is not None:
                pose_guided_pool = build_pose_guided_surface_anchor_candidate_pool(
                    detected_query,
                    pose_guided_initial_pose,
                    camera,
                    descriptor_bank=local_bank,
                    anchors=anchors,
                    top_l=int(args.anchor_top_l),
                    search_radius_px=32.0,
                    maximum_query_points=int(args.guided_local_max_points),
                )
                pose_guided_result = generate_grouped_surface_pose_hypotheses(
                    pose_guided_pool,
                    camera,
                    pose_config,
                )
            if pose_guided_result is not None and pose_guided_result.success:
                pose_result = pose_guided_result
            elif vfm_pose_result.success:
                pose_result = vfm_pose_result
            if direct_guided_result is not None and direct_guided_result.success:
                pose_result = direct_guided_result
            if (
                direct_dense_guided_result is not None
                and direct_dense_guided_result.success
            ):
                pose_result = direct_dense_guided_result
            if loftr_result is not None and loftr_result.success:
                pose_result = loftr_result
            if loftr_depth_result is not None and loftr_depth_result.success:
                pose_result = loftr_depth_result
            successes += int(pose_result.success)
            if not pose_result.success:
                key = str(pose_result.failure_reason or "unknown")
                failure_counts[key] = int(failure_counts.get(key, 0)) + 1
            diagnostics = {
                "query_region_count": int(len(region_xy)),
                "support_view_id": maplet_match.support_view_id,
                "support_view_score": float(maplet_match.support_view_score),
                "support_transform_matrix": (
                    maplet_match.support_transform_matrix.reshape(-1).tolist()
                    if maplet_match.support_transform_matrix is not None
                    else None
                ),
                "support_transform_translation": (
                    maplet_match.support_transform_translation.reshape(-1).tolist()
                    if maplet_match.support_transform_translation is not None
                    else None
                ),
                "maplet_non_null_fraction": float(
                    np.mean(
                        np.max(maplet_match.candidate_probabilities, axis=1)
                        > maplet_match.null_probabilities
                    )
                ),
                "detected_local_keypoint_count": int(len(detected_query.keypoints_xy)),
                "guided_local_keypoint_count": int(len(local_query.keypoints_xy)),
                "guided_seed_count": int(np.sum(seed_anchor_ids >= 0)),
                "local_groups_with_candidates": int(np.sum(np.any(candidate_pool.valid_mask, axis=1))),
                "mean_local_null_probability": (
                    float(np.mean(candidate_pool.null_probabilities))
                    if len(candidate_pool.null_probabilities)
                    else 1.0
                ),
                "vfm_surface_group_count": int(len(vfm_candidate_pool)),
                "vfm_surface_groups_with_candidates": int(
                    np.sum(np.any(vfm_candidate_pool.valid_mask, axis=1))
                ),
                "vfm_pose_success": bool(vfm_pose_result.success),
                "vfm_pose_failure_reason": vfm_pose_result.failure_reason,
                "vfm_pose_best_inlier_count": (
                    int(vfm_pose_result.hypotheses[0].inlier_count)
                    if vfm_pose_result.hypotheses
                    else 0
                ),
                "pose_guided_surface_group_count": (
                    int(len(pose_guided_pool)) if pose_guided_pool is not None else 0
                ),
                "pose_guided_pose_success": bool(
                    pose_guided_result is not None and pose_guided_result.success
                ),
                "pose_guided_pose_best_inlier_count": (
                    int(pose_guided_result.hypotheses[0].inlier_count)
                    if pose_guided_result is not None and pose_guided_result.hypotheses
                    else 0
                ),
                "pose_guided_initial_source": pose_guided_initial_source,
                "direct_vfm_layout_inlier_count": int(direct_layout_inliers),
                "direct_vfm_layout_median_residual_tokens": float(
                    direct_layout_median_residual
                ),
                "direct_vfm_selected_support_view": direct_selected_support_view,
                "direct_vfm_evaluated_support_count": int(
                    len(direct_support_runs)
                ),
                "direct_vfm_support_diagnostics": [
                    {
                        "support_rank": int(run["support_rank"]),
                        "support_view_id": str(run["support_view_id"]),
                        "layout_inlier_count": int(run["layout_inliers"]),
                        "layout_median_residual_tokens": float(
                            run["layout_median_residual"]
                        ),
                        "dense_measurement_group_count": (
                            int(len(run["dense_pool"]))
                            if run["dense_pool"] is not None
                            else 0
                        ),
                        "dense_pose_success": bool(
                            isinstance(run["dense_result"], SurfacePoseResult)
                            and run["dense_result"].success
                        ),
                        "dense_best_inlier_count": (
                            int(run["dense_result"].hypotheses[0].inlier_count)
                            if isinstance(run["dense_result"], SurfacePoseResult)
                            and run["dense_result"].hypotheses
                            else 0
                        ),
                        "sparse_measurement_group_count": (
                            int(len(run["sparse_pool"]))
                            if run["sparse_pool"] is not None
                            else 0
                        ),
                        "sparse_pose_success": bool(
                            isinstance(run["sparse_result"], SurfacePoseResult)
                            and run["sparse_result"].success
                        ),
                        "sparse_best_inlier_count": (
                            int(run["sparse_result"].hypotheses[0].inlier_count)
                            if isinstance(run["sparse_result"], SurfacePoseResult)
                            and run["sparse_result"].hypotheses
                            else 0
                        ),
                    }
                    for run in direct_support_runs
                ],
                "direct_vfm_guided_surface_group_count": (
                    int(len(direct_guided_pool)) if direct_guided_pool is not None else 0
                ),
                "direct_vfm_guided_search_radius_px": float(
                    direct_guided_search_radius_px
                ),
                "direct_vfm_guided_pose_success": bool(
                    direct_guided_result is not None and direct_guided_result.success
                ),
                "direct_vfm_guided_pose_best_inlier_count": (
                    int(direct_guided_result.hypotheses[0].inlier_count)
                    if direct_guided_result is not None
                    and direct_guided_result.hypotheses
                    else 0
                ),
                "direct_vfm_dense_measurement_group_count": (
                    int(len(direct_dense_guided_pool))
                    if direct_dense_guided_pool is not None
                    else 0
                ),
                "direct_vfm_dense_measurement_pose_success": bool(
                    direct_dense_guided_result is not None
                    and direct_dense_guided_result.success
                ),
                "direct_vfm_dense_measurement_best_inlier_count": (
                    int(direct_dense_guided_result.hypotheses[0].inlier_count)
                    if direct_dense_guided_result is not None
                    and direct_dense_guided_result.hypotheses
                    else 0
                ),
                "vfm_support_views": list(vfm_support_views),
                "vfm_support_scores": vfm_support_scores.tolist(),
                "vfm_guided_loftr_selected_support_view": (
                    loftr_selected_support_view
                ),
                "vfm_guided_loftr_surface_group_count": (
                    int(len(loftr_pool)) if loftr_pool is not None else 0
                ),
                "vfm_guided_loftr_support_diagnostics": [
                    {
                        "support_rank": int(run["support_rank"]),
                        "support_view_id": str(run["support_view_id"]),
                        "pair_match_count": int(run["pair_match_count"]),
                        "lifted_surface_group_count": int(len(run["pool"])),
                        "pose_success": bool(run["result"].success),
                        "best_inlier_count": (
                            int(run["result"].hypotheses[0].inlier_count)
                            if run["result"].hypotheses
                            else 0
                        ),
                        "depth_surface_group_count": (
                            int(len(run["depth_pool"]))
                            if run["depth_pool"] is not None
                            else 0
                        ),
                        "depth_pose_success": bool(
                            isinstance(
                                run["depth_result"],
                                SurfacePoseResult,
                            )
                            and run["depth_result"].success
                        ),
                        "depth_best_inlier_count": (
                            int(
                                run["depth_result"].hypotheses[
                                    0
                                ].inlier_count
                            )
                            if isinstance(
                                run["depth_result"],
                                SurfacePoseResult,
                            )
                            and run["depth_result"].hypotheses
                            else 0
                        ),
                        "cross_modal_translation_m": (
                            float(run["cross_modal_translation_m"])
                            if np.isfinite(
                                float(run["cross_modal_translation_m"])
                            )
                            else None
                        ),
                        "cross_modal_rotation_deg": (
                            float(run["cross_modal_rotation_deg"])
                            if np.isfinite(
                                float(run["cross_modal_rotation_deg"])
                            )
                            else None
                        ),
                    }
                    for run in loftr_support_runs
                ],
                "vfm_guided_loftr_depth_selected_support_view": (
                    loftr_depth_selected_support_view
                ),
                "vfm_guided_loftr_depth_surface_group_count": (
                    int(len(loftr_depth_pool))
                    if loftr_depth_pool is not None
                    else 0
                ),
                "selected_pose_branch": (
                    "vfm_guided_loftr_2dgs_depth_surface"
                    if loftr_depth_result is not None
                    and pose_result is loftr_depth_result
                    else (
                        "vfm_guided_loftr_2dgs_anchor"
                        if loftr_result is not None and pose_result is loftr_result
                        else (
                            "direct_vfm_layout_dense_alike_measurement"
                            if direct_dense_guided_result is not None
                            and pose_result is direct_dense_guided_result
                            else (
                                "direct_vfm_layout_alike_surface_anchor"
                                if direct_guided_result is not None
                                and pose_result is direct_guided_result
                                else (
                                    "vfm_guided_alike_surface_anchor"
                                    if pose_guided_result is not None
                                    and pose_result is pose_guided_result
                                    else (
                                        "vfm_surface_observation"
                                        if pose_result is vfm_pose_result
                                        else "alike_surface_anchor"
                                    )
                                )
                            )
                        )
                    )
                ),
            }
            output.write(json.dumps(_pose_record(pose_result, record.image_id, diagnostics)) + "\n")
            output.flush()
            if (index + 1) % 10 == 0 or index + 1 == len(records):
                print(
                    json.dumps(
                        {
                            "completed": int(index + 1),
                            "total": int(len(records)),
                            "successes": int(successes),
                            "elapsed_seconds": float(time.time() - started),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    summary = {
        "stage": "localize_2dgs_surface_queries",
        "query_count": int(len(records)),
        "success_count": int(successes),
        "success_rate": float(successes / len(records)),
        "failure_counts": failure_counts,
        "elapsed_seconds": float(time.time() - started),
        "production_contract": {
            "query_ground_truth_used": False,
            "vfm_layer": "radio_final",
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "fit_and_verification_groups_disjoint": True,
            "fixed_candidate_pool_per_hypothesis": True,
        },
        "inputs": {
            "query_manifest": str(args.query_manifest),
            "surface_mapper_checkpoint": str(args.surface_mapper_checkpoint),
            "surface_mapper_checkpoint_sha256": file_sha256_short(
                Path(args.surface_mapper_checkpoint)
            ),
            "maplets": str(args.maplets),
            "maplets_sha256": file_sha256_short(Path(args.maplets)),
            "anchors": str(args.anchors),
            "anchors_sha256": file_sha256_short(Path(args.anchors)),
            "local_descriptor_bank": str(args.local_descriptor_bank),
            "local_descriptor_bank_sha256": file_sha256_short(
                Path(args.local_descriptor_bank)
            ),
            "observation_bank": str(args.observation_bank),
            "observation_bank_sha256": file_sha256_short(Path(args.observation_bank)),
            "surface_depth_bank": (
                str(args.surface_depth_bank)
                if surface_depth_bank is not None
                else None
            ),
            "surface_depth_bank_sha256": (
                file_sha256_short(Path(args.surface_depth_bank))
                if surface_depth_bank is not None
                else None
            ),
            "camera_model_dir": str(args.camera_model_dir),
            "mapping_pose_file": str(args.mapping_pose_file),
        },
        "config": {
            "region_grid": [
                int(args.query_region_rows),
                int(args.query_region_cols),
                int(args.query_regions_per_cell),
            ],
            "maplet_match": match_config.__dict__,
            "maximum_maplets_per_region": int(args.maximum_maplets_per_region),
            "local_top_k": int(args.local_top_k),
            "guided_local_search_radius_px": float(args.guided_local_search_radius_px),
            "guided_local_max_points": int(args.guided_local_max_points),
            "anchor_top_l": int(args.anchor_top_l),
            "vfm_support_views": int(args.vfm_support_views),
            "vfm_global_support_views": int(args.vfm_global_support_views),
            "vfm_max_query_points": int(args.vfm_max_query_points),
            "vfm_minimum_similarity": float(args.vfm_minimum_similarity),
            "mapping_pose_prior": {
                "source": "vfm_retrieved_mapping_view",
                "translation_weight": float(
                    args.mapping_pose_translation_prior_weight
                ),
                "rotation_weight": float(args.mapping_pose_rotation_prior_weight),
                "uses_query_ground_truth": False,
                "direct_support_selection_uses_prior": False,
                "fallback_pose_guidance_only": True,
            },
            "direct_support_pose": direct_pose_config.__dict__,
            "vfm_guided_loftr": (
                None
                if loftr is None
                else {
                    **loftr.metadata,
                    "pair_batch_size": int(args.loftr_pair_batch_size),
                    "anchor_maximum_distance_px": float(
                        args.loftr_anchor_maximum_distance_px
                    ),
                    "depth_surface_enabled": bool(
                        surface_depth_bank is not None
                    ),
                    "depth_maximum_points": int(
                        args.loftr_depth_maximum_points
                    ),
                    "depth_query_nms_radius_px": float(
                        args.loftr_depth_query_nms_radius_px
                    ),
                    "cross_modal_max_translation_m": float(
                        args.loftr_cross_modal_max_translation_m
                    ),
                    "cross_modal_max_rotation_deg": float(
                        args.loftr_cross_modal_max_rotation_deg
                    ),
                }
            ),
            "pose": pose_config.__dict__,
            "vfm_pose": vfm_pose_config.__dict__,
        },
        "output_jsonl": str(output_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
