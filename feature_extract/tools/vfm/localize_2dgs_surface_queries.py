"""Localize query images against a feature-only 2DGS surface map.

Runtime inputs are deliberately limited to query RGB/RADIO-final tokens and
frozen map artifacts.  Mapping RGB, SfM points/tracks, pairwise image matchers,
and view-depth lifting are not accepted by this program.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_cameras_binary
from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
)
from feature_extract.vfm.localization.surface_anchor_set_matcher import (
    load_surface_anchor_set_matcher,
    match_query_to_surface_anchors,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
    LocalFeatureFrame,
    SurfaceMapletMatchConfig,
    SurfacePoseConfig,
    SurfacePoseResult,
    VfmSurfaceObservationIndex,
    build_maplet_conditioned_surface_anchor_candidate_pool,
    build_pose_guided_surface_anchor_candidate_pool,
    build_seeded_surface_anchor_candidate_pool,
    build_vfm_surface_observation_candidate_pool,
    estimate_vfm_query_to_support_layout,
    generate_grouped_surface_pose_hypotheses,
    match_radio_final_regions_to_maplets,
    predict_vfm_support_anchor_query_points,
    score_surface_pose,
    select_vfm_surface_feature_modes,
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
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_mapping import Vfm2DgsObservationBank


FORBIDDEN_RUNTIME_FLAGS = (
    "uses_mapping_rgb_at_inference",
    "uses_mapping_image_retrieval",
    "uses_pairwise_image_matching",
    "uses_sfm_points",
    "uses_sfm_tracks",
    "uses_radio_intermediate",
    "uses_view_depth_at_inference",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_image_root", required=True)
    parser.add_argument("--radio_final_layer", default="radio_final")
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--local_descriptor_bank", required=True)
    parser.add_argument(
        "--radio_surface_feature_bank",
        required=True,
        help=(
            "RADIO-final features attached to 2DGS surface observations; "
            "contains no mapping RGB."
        ),
    )
    parser.add_argument("--surface_anchor_matcher_checkpoint", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument("--query_region_rows", type=int, default=8)
    parser.add_argument("--query_region_cols", type=int, default=8)
    parser.add_argument("--query_regions_per_cell", type=int, default=2)
    parser.add_argument("--maplet_top_k", type=int, default=8)
    parser.add_argument("--maximum_support_views", type=int, default=64)
    parser.add_argument("--maximum_layout_models", type=int, default=4096)
    parser.add_argument("--maximum_feature_modes", type=int, default=8)
    parser.add_argument("--maximum_global_feature_modes", type=int, default=64)
    parser.add_argument("--feature_mode_minimum_similarity", type=float, default=0.20)
    parser.add_argument("--local_top_k", type=int, default=2048)
    parser.add_argument("--local_candidate_top_k", type=int, default=8192)
    parser.add_argument("--local_nms_radius_px", type=float, default=2.0)
    parser.add_argument("--anchor_top_l", type=int, default=20)
    parser.add_argument("--verification_anchor_top_l", type=int, default=10)
    parser.add_argument("--layout_guided_maximum_points", type=int, default=256)
    parser.add_argument("--layout_guided_search_radius_px", type=int, default=18)
    parser.add_argument("--layout_guided_seed_prior_logit", type=float, default=32.0)
    parser.add_argument("--disable_layout_guided_measurement", action="store_true")
    parser.add_argument("--hypothesis_count", type=int, default=256)
    parser.add_argument("--minimum_fit_groups", type=int, default=8)
    parser.add_argument("--minimum_verification_groups", type=int, default=4)
    parser.add_argument("--heldout_stride", type=int, default=5)
    parser.add_argument(
        "--set_pose_maximum_mean_null",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--set_pose_minimum_confident_groups",
        type=int,
        default=12,
    )
    parser.add_argument("--pose_refinement_radius_px", type=float, default=24.0)
    parser.add_argument("--pose_refinement_maximum_points", type=int, default=768)
    parser.add_argument("--disable_pose_refinement", action="store_true")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _camera_from_model(model_dir: Path) -> ColmapCamera:
    """Load intrinsics only; no points, tracks, images, or poses are read."""

    cameras = read_colmap_cameras_binary(Path(model_dir) / "cameras.bin")
    if not cameras:
        raise ValueError("camera model contains no cameras")
    structural = {
        (
            int(camera.model_id),
            int(camera.width),
            int(camera.height),
            len(camera.params),
        )
        for camera in cameras.values()
    }
    if len(structural) != 1:
        raise ValueError("camera model has incompatible intrinsic families")
    model_id, width, height, parameter_count = next(iter(structural))
    parameters = np.asarray(
        [camera.params for camera in cameras.values()], dtype=np.float64
    ).reshape(-1, int(parameter_count))
    return ColmapCamera(
        camera_id=0,
        model_id=int(model_id),
        width=int(width),
        height=int(height),
        params=tuple(
            float(value) for value in np.median(parameters, axis=0).tolist()
        ),
    )


def _load_raw_final(path: Path, layer_name: str) -> np.ndarray:
    with np.load(Path(path)) as data:
        if str(layer_name) not in data:
            raise ValueError(
                f"RADIO-final layer {layer_name!r} is absent from {path}"
            )
        feature = np.asarray(data[str(layer_name)], dtype=np.float32)
    if feature.ndim == 4 and int(feature.shape[0]) == 1:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError("RADIO-final query feature must have shape (C,H,W)")
    return feature


def _validate_metadata(metadata: Mapping[str, object], name: str) -> None:
    illegal = [key for key in FORBIDDEN_RUNTIME_FLAGS if bool(metadata.get(key))]
    if illegal:
        raise ValueError(f"{name} violates map-only runtime contract: {illegal}")


def _retrieved_anchor_ids(
    maplet_match,
    maplets: VfmSurfaceMapletBank,
) -> np.ndarray:
    maplet_rows = {
        int(maplet_id): int(row)
        for row, maplet_id in enumerate(maplets.maplet_ids.tolist())
    }
    retrieved = {
        int(value)
        for value in maplet_match.candidate_maplet_ids.reshape(-1).tolist()
        if int(value) >= 0
    }
    output: set[int] = set()
    for maplet_id in retrieved:
        row = maplet_rows.get(maplet_id)
        if row is None:
            continue
        start = int(maplets.anchor_offsets[row])
        end = int(maplets.anchor_offsets[row + 1])
        output.update(int(value) for value in maplets.anchor_ids[start:end])
    return np.asarray(sorted(output), dtype=np.int64)


def _retrieved_support_mode_ids(
    maplet_match,
    maplets: VfmSurfaceMapletBank,
) -> tuple[str, ...]:
    row_by_id = {
        int(maplet_id): int(row)
        for row, maplet_id in enumerate(maplets.maplet_ids.tolist())
    }
    output: set[str] = set()
    for maplet_id_value in maplet_match.candidate_maplet_ids.reshape(-1):
        row = row_by_id.get(int(maplet_id_value))
        if row is None:
            continue
        start = int(maplets.view_offsets[row])
        end = int(maplets.view_offsets[row + 1])
        output.update(maplets.view_image_ids[start:end])
    return tuple(sorted(output))


def _fixed_pose_evidence(
    pool,
    result: SurfacePoseResult,
    camera: ColmapCamera,
    config: SurfacePoseConfig,
) -> tuple[float, int]:
    if not result.success or not result.hypotheses or len(pool) == 0:
        return float("-inf"), 0
    score, residuals, _posterior = score_surface_pose(
        pool,
        result.pose_w2c,
        camera,
        np.ones((len(pool),), dtype=bool),
        config,
    )
    valid_residuals = np.where(pool.valid_mask, residuals, np.inf)
    inlier_count = int(
        np.sum(
            np.min(valid_residuals, axis=1)
            <= float(config.inlier_threshold_px)
        )
    )
    return float(score / max(len(pool), 1)), inlier_count


def _result_key(result: SurfacePoseResult) -> tuple[float, ...]:
    if not result.hypotheses:
        return (float(result.success), float("-inf"), 0.0, float("-inf"))
    best = result.hypotheses[0]
    verification_count = max(int(np.sum(result.verification_mask)), 1)
    fit_count = max(int(np.sum(result.fit_mask)), 1)
    return (
        float(result.success),
        float(best.inlier_count),
        float(best.verification_score) / verification_count,
        float(best.generation_score) / fit_count,
    )


def _pose_record(
    *,
    image_id: str,
    result: SurfacePoseResult,
    branch: str,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
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
        "pose_w2c": (
            result.pose_w2c.reshape(-1).tolist() if result.success else None
        ),
        "failure_reason": result.failure_reason,
        "branch": str(branch),
        "fit_group_count": int(np.sum(result.fit_mask)),
        "verification_group_count": int(np.sum(result.verification_mask)),
        "hypothesis_count": len(result.hypotheses),
        "hypotheses": hypotheses,
        "diagnostics": dict(diagnostics),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite localization outputs")
    if (
        int(args.local_candidate_top_k) < int(args.local_top_k)
        or int(args.maplet_top_k) <= 0
        or int(args.anchor_top_l) <= 0
        or int(args.verification_anchor_top_l) <= 0
        or int(args.maximum_feature_modes) <= 0
        or int(args.maximum_global_feature_modes) <= 0
        or float(args.feature_mode_minimum_similarity) <= 0.0
        or not 0.0 <= float(args.set_pose_maximum_mean_null) <= 1.0
        or int(args.set_pose_minimum_confident_groups) <= 0
    ):
        raise ValueError("localization candidate limits are invalid")

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate()
    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    if not records:
        raise ValueError("query manifest is empty")
    query_root = Path(args.query_image_root).resolve(strict=True)
    query_image_paths: dict[str, Path] = {}
    for record in records:
        relative = Path(record.image_id)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"query image escapes query_image_root: {record.image_id}"
            )
        path = query_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"query image is absent: {path}")
        query_image_paths[record.image_id] = path

    maplets = VfmSurfaceMapletBank.load_npz(Path(args.maplets))
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    local_bank = AnchorLocalDescriptorBank.load_npz(
        Path(args.local_descriptor_bank)
    )
    surface_features = Vfm2DgsObservationBank.load_npz(
        Path(args.radio_surface_feature_bank)
    )
    surface_feature_index = VfmSurfaceObservationIndex.from_bank(
        surface_features
    )
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint), device=str(args.device)
    )
    anchor_matcher, anchor_matcher_payload = load_surface_anchor_set_matcher(
        Path(args.surface_anchor_matcher_checkpoint), device=str(args.device)
    )
    for metadata, name in (
        (maplets.metadata, "maplet bank"),
        (anchors.metadata, "anchor map"),
        (local_bank.metadata, "local descriptor bank"),
        (surface_features.metadata, "RADIO surface feature bank"),
        (mapper_metadata, "surface mapper"),
        (anchor_matcher_payload.get("metadata") or {}, "surface anchor matcher"),
    ):
        _validate_metadata(metadata, name)
    if int(local_bank.feature_dim) != int(anchor_matcher.config.descriptor_dim):
        raise ValueError("anchor matcher and local descriptor dimensions differ")
    if set(local_bank.anchor_ids.tolist()) - set(anchors.anchor_ids.tolist()):
        raise ValueError("local descriptor bank contains unknown surface anchors")

    camera = _camera_from_model(Path(args.camera_model_dir))
    pool_sizes = tuple(
        int(value) for value in mapper_metadata.get("pool_sizes", (1, 3, 5, 9))
    )
    pool_weights = tuple(
        float(value)
        for value in mapper_metadata.get(
            "pool_weights", (0.4, 0.3, 0.2, 0.1)
        )
    )
    region_config = RadioFinalRegionConfig(
        pool_sizes=pool_sizes, pool_weights=pool_weights
    )
    maplet_config = SurfaceMapletMatchConfig(
        top_k=int(args.maplet_top_k),
        maximum_support_views=int(args.maximum_support_views),
        maximum_layout_models=int(args.maximum_layout_models),
        enable_support_layout=False,
    )
    pose_config = SurfacePoseConfig(
        hypothesis_count=int(args.hypothesis_count),
        minimum_fit_groups=int(args.minimum_fit_groups),
        minimum_verification_groups=int(args.minimum_verification_groups),
        heldout_stride=int(args.heldout_stride),
    )
    alike = AlikeDenseObservationExtractor(
        device=str(args.device),
        matcha_repo=Path(args.matcha_repo),
        model_name=str(args.alike_model_name),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    successes = 0
    failure_counts: dict[str, int] = {}
    branch_counts: dict[str, int] = {}
    query_file_audit: list[dict[str, object]] = []
    with output_path.open("w") as output:
        for query_index, record in enumerate(records):
            query_started = time.time()
            stage_started = time.time()
            raw = _load_raw_final(
                record.token_path, str(args.radio_final_layer)
            )
            mapped = mapper.project(raw).coarse_descriptors
            _indices, region_xy = select_spatially_balanced_radio_final_regions(
                raw,
                grid_rows=int(args.query_region_rows),
                grid_cols=int(args.query_region_cols),
                regions_per_cell=int(args.query_regions_per_cell),
            )
            region_descriptors = encode_radio_final_regions(
                mapped, region_xy, region_config
            )
            grid_size = (int(mapped.shape[2]), int(mapped.shape[1]))
            maplet_match = match_radio_final_regions_to_maplets(
                region_xy,
                region_descriptors,
                grid_size,
                maplets,
                maplet_config,
            )
            coarse_runtime = time.time() - stage_started
            stage_started = time.time()
            image_path = query_image_paths[record.image_id]
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
            query = LocalFeatureFrame(
                image_id=record.image_id,
                keypoints_xy=detected.xy,
                descriptors=detected.descriptors,
                scores=detected.scores,
            )
            query_feature_runtime = time.time() - stage_started
            stage_started = time.time()
            anchor_pool, matcher_diagnostics = match_query_to_surface_anchors(
                model=anchor_matcher,
                query=query,
                query_image_size=(int(camera.width), int(camera.height)),
                query_region_xy=region_xy,
                query_region_grid_size=grid_size,
                maplet_match=maplet_match,
                maplets=maplets,
                anchors=anchors,
                descriptor_bank=local_bank,
                device=str(args.device),
                top_l=int(args.anchor_top_l),
            )
            anchor_set_runtime = time.time() - stage_started
            stage_started = time.time()
            maximum_set_probability = np.max(
                np.where(
                    anchor_pool.valid_mask,
                    anchor_pool.candidate_probabilities,
                    0.0,
                ),
                axis=1,
            )
            confident_set_groups = int(
                np.sum(
                    maximum_set_probability
                    > anchor_pool.null_probabilities
                )
            )
            mean_set_null = (
                float(np.mean(anchor_pool.null_probabilities))
                if len(anchor_pool)
                else 1.0
            )
            set_pose_gate_passed = bool(
                mean_set_null
                <= float(args.set_pose_maximum_mean_null)
                and confident_set_groups
                >= int(args.set_pose_minimum_confident_groups)
            )
            if set_pose_gate_passed:
                initial_result = generate_grouped_surface_pose_hypotheses(
                    anchor_pool, camera, pose_config
                )
            else:
                initial_result = SurfacePoseResult(
                    success=False,
                    pose_w2c=np.eye(4, dtype=np.float64),
                    hypotheses=(),
                    fit_mask=np.zeros((len(anchor_pool),), dtype=bool),
                    verification_mask=np.zeros(
                        (len(anchor_pool),), dtype=bool
                    ),
                    failure_reason="set_matcher_confidence_gate",
                )
            initial_pose_runtime = time.time() - stage_started
            stage_started = time.time()
            fixed_verification_pool = (
                build_maplet_conditioned_surface_anchor_candidate_pool(
                    query,
                    query_image_size=(
                        int(camera.width),
                        int(camera.height),
                    ),
                    query_region_xy=region_xy,
                    query_region_grid_size=grid_size,
                    maplet_match=maplet_match,
                    maplets=maplets,
                    descriptor_bank=local_bank,
                    anchors=anchors,
                    top_l=int(args.verification_anchor_top_l),
                    maximum_maplets_per_region=3,
                )
            )
            fixed_pool_runtime = time.time() - stage_started
            result = initial_result
            selection_pool = fixed_verification_pool
            branch = "radio_maplet_alike_set_ot_grouped_pnp"
            layout_pool_size = 0
            layout_success_count = 0
            selected_layout_view_id = None
            selected_layout_score = None
            refinement_pool_size = 0
            retrieved_anchor_ids = _retrieved_anchor_ids(maplet_match, maplets)
            allowed_mode_ids = _retrieved_support_mode_ids(
                maplet_match, maplets
            )
            if bool(args.disable_layout_guided_measurement):
                feature_mode_ids = ()
                feature_mode_scores = np.zeros((0,), dtype=np.float32)
            else:
                feature_mode_ids, feature_mode_scores = (
                    select_vfm_surface_feature_modes(
                        mapped,
                        surface_features,
                        maximum_global_modes=int(
                            args.maximum_global_feature_modes
                        ),
                        maximum_modes=int(args.maximum_feature_modes),
                        allowed_mode_ids=allowed_mode_ids,
                        observation_index=surface_feature_index,
                    )
                )
            feature_mode_selection_runtime = (
                time.time() - stage_started - fixed_pool_runtime
            )
            stage_started = time.time()
            pose_candidates: list[
                tuple[
                    float,
                    int,
                    float,
                    int,
                    SurfacePoseResult,
                    str,
                    str | None,
                ]
            ] = []
            initial_fixed_score, initial_fixed_inliers = _fixed_pose_evidence(
                fixed_verification_pool,
                initial_result,
                camera,
                pose_config,
            )
            if initial_result.success:
                pose_candidates.append(
                    (
                        initial_fixed_score,
                        initial_fixed_inliers,
                        float("-inf"),
                        -1,
                        initial_result,
                        "radio_maplet_alike_set_ot_grouped_pnp",
                        None,
                    )
                )
            feature_mode_diagnostics: list[dict[str, object]] = []
            for feature_mode_rank, (
                layout_view_id,
                feature_mode_score,
            ) in enumerate(zip(feature_mode_ids, feature_mode_scores.tolist())):
                (
                    layout_matrix,
                    layout_translation,
                    layout_inliers,
                    layout_median_residual,
                ) = estimate_vfm_query_to_support_layout(
                    mapped,
                    surface_features,
                    str(layout_view_id),
                    minimum_similarity=float(
                        args.feature_mode_minimum_similarity
                    ),
                    observation_index=surface_feature_index,
                )
                mode_diagnostic: dict[str, object] = {
                    "rank": int(feature_mode_rank),
                    "mode_id": str(layout_view_id),
                    "vfm_score": float(feature_mode_score),
                    "layout_inliers": int(layout_inliers),
                    "layout_median_residual_tokens": (
                        float(layout_median_residual)
                        if np.isfinite(layout_median_residual)
                        else None
                    ),
                    "seed_count": 0,
                    "pose_success": False,
                    "pose_inliers": 0,
                    "fixed_evidence_per_group": None,
                    "fixed_evidence_inliers": 0,
                }
                feature_mode_diagnostics.append(mode_diagnostic)
                if (
                    layout_matrix is None
                    or layout_translation is None
                ):
                    continue
                (
                    predicted_xy,
                    support_descriptors,
                    seed_anchor_ids,
                ) = predict_vfm_support_anchor_query_points(
                    query_image_size=(int(camera.width), int(camera.height)),
                    query_grid_size=grid_size,
                    support_view_id=str(layout_view_id),
                    query_to_support_matrix=layout_matrix,
                    query_to_support_translation=layout_translation,
                    anchors=anchors,
                    descriptor_bank=local_bank,
                    allowed_anchor_ids=retrieved_anchor_ids,
                    maximum_points=int(args.layout_guided_maximum_points),
                )
                mode_diagnostic["seed_count"] = int(len(seed_anchor_ids))
                if len(predicted_xy) >= 4:
                    (
                        measured_xy,
                        measured_descriptors,
                        measured_scores,
                        _measured_image_hash,
                    ) = alike.match_descriptor_points(
                        image_path,
                        predicted_xy,
                        support_descriptors,
                        image_width=int(camera.width),
                        image_height=int(camera.height),
                        search_radius_px=int(
                            args.layout_guided_search_radius_px
                        ),
                        search_step_px=1,
                    )
                    layout_query = LocalFeatureFrame(
                        image_id=record.image_id,
                        keypoints_xy=measured_xy,
                        descriptors=measured_descriptors,
                        scores=measured_scores,
                    )
                    layout_pool = build_seeded_surface_anchor_candidate_pool(
                        layout_query,
                        seed_anchor_ids,
                        descriptor_bank=local_bank,
                        anchors=anchors,
                        top_l=int(args.anchor_top_l),
                        seed_prior_logit=float(
                            args.layout_guided_seed_prior_logit
                        ),
                    )
                    layout_pool_size += len(layout_pool)
                    layout_result = generate_grouped_surface_pose_hypotheses(
                        layout_pool, camera, pose_config
                    )
                    layout_success_count += int(layout_result.success)
                    mode_diagnostic["pose_success"] = bool(
                        layout_result.success
                    )
                    mode_diagnostic["pose_inliers"] = int(
                        layout_result.hypotheses[0].inlier_count
                        if layout_result.hypotheses
                        else 0
                    )
                    fixed_score, fixed_inliers = _fixed_pose_evidence(
                        fixed_verification_pool,
                        layout_result,
                        camera,
                        pose_config,
                    )
                    mode_diagnostic["fixed_evidence_per_group"] = (
                        float(fixed_score)
                        if np.isfinite(fixed_score)
                        else None
                    )
                    mode_diagnostic["fixed_evidence_inliers"] = int(
                        fixed_inliers
                    )
                    if layout_result.success:
                        pose_candidates.append(
                            (
                                fixed_score,
                                fixed_inliers,
                                float(feature_mode_score),
                                -int(feature_mode_rank),
                                layout_result,
                                (
                                    "radio_maplet_feature_mode_alike_"
                                    "anchor_measurement"
                                ),
                                str(layout_view_id),
                            )
                        )
            if not pose_candidates and feature_mode_ids:
                for feature_mode_rank, (
                    layout_view_id,
                    feature_mode_score,
                ) in enumerate(
                    zip(feature_mode_ids[:4], feature_mode_scores[:4].tolist())
                ):
                    (
                        layout_matrix,
                        layout_translation,
                        _layout_inliers,
                        _layout_median_residual,
                    ) = estimate_vfm_query_to_support_layout(
                        mapped,
                        surface_features,
                        str(layout_view_id),
                        minimum_similarity=float(
                            args.feature_mode_minimum_similarity
                        ),
                        observation_index=surface_feature_index,
                    )
                    if (
                        layout_matrix is None
                        or layout_translation is None
                    ):
                        continue
                    (
                        predicted_xy,
                        support_descriptors,
                        seed_anchor_ids,
                    ) = predict_vfm_support_anchor_query_points(
                        query_image_size=(
                            int(camera.width),
                            int(camera.height),
                        ),
                        query_grid_size=grid_size,
                        support_view_id=str(layout_view_id),
                        query_to_support_matrix=layout_matrix,
                        query_to_support_translation=layout_translation,
                        anchors=anchors,
                        descriptor_bank=local_bank,
                        allowed_anchor_ids=None,
                        maximum_points=int(
                            args.layout_guided_maximum_points
                        ),
                    )
                    feature_mode_diagnostics[
                        feature_mode_rank
                    ]["expanded_seed_count"] = int(len(seed_anchor_ids))
                    if len(predicted_xy) < 4:
                        continue
                    (
                        measured_xy,
                        measured_descriptors,
                        measured_scores,
                        _measured_image_hash,
                    ) = alike.match_descriptor_points(
                        image_path,
                        predicted_xy,
                        support_descriptors,
                        image_width=int(camera.width),
                        image_height=int(camera.height),
                        search_radius_px=int(
                            args.layout_guided_search_radius_px
                        ),
                        search_step_px=1,
                    )
                    expanded_query = LocalFeatureFrame(
                        image_id=record.image_id,
                        keypoints_xy=measured_xy,
                        descriptors=measured_descriptors,
                        scores=measured_scores,
                    )
                    expanded_pool = (
                        build_seeded_surface_anchor_candidate_pool(
                            expanded_query,
                            seed_anchor_ids,
                            descriptor_bank=local_bank,
                            anchors=anchors,
                            top_l=int(args.anchor_top_l),
                            seed_prior_logit=float(
                                args.layout_guided_seed_prior_logit
                            ),
                        )
                    )
                    layout_pool_size += len(expanded_pool)
                    expanded_result = (
                        generate_grouped_surface_pose_hypotheses(
                            expanded_pool, camera, pose_config
                        )
                    )
                    layout_success_count += int(expanded_result.success)
                    feature_mode_diagnostics[
                        feature_mode_rank
                    ]["expanded_pose_success"] = bool(
                        expanded_result.success
                    )
                    feature_mode_diagnostics[
                        feature_mode_rank
                    ]["expanded_pose_inliers"] = int(
                        expanded_result.hypotheses[0].inlier_count
                        if expanded_result.hypotheses
                        else 0
                    )
                    fixed_score, fixed_inliers = _fixed_pose_evidence(
                        fixed_verification_pool,
                        expanded_result,
                        camera,
                        pose_config,
                    )
                    if expanded_result.success:
                        pose_candidates.append(
                            (
                                fixed_score,
                                fixed_inliers,
                                float(feature_mode_score),
                                -int(feature_mode_rank),
                                expanded_result,
                                (
                                    "radio_feature_mode_expanded_stable_"
                                    "anchor_fallback"
                                ),
                                str(layout_view_id),
                            )
                        )
            direct_surface_fallback_group_count = 0
            direct_surface_fallback_inliers = 0
            if not pose_candidates:
                (
                    direct_surface_pool,
                    direct_surface_mode_ids,
                    direct_surface_mode_scores,
                ) = build_vfm_surface_observation_candidate_pool(
                    mapped,
                    query_image_size=(
                        int(camera.width),
                        int(camera.height),
                    ),
                    observation_bank=surface_features,
                    maximum_global_support_views=int(
                        args.maximum_global_feature_modes
                    ),
                    maximum_support_views=min(
                        4, int(args.maximum_feature_modes)
                    ),
                    maximum_query_points=768,
                    top_l=5,
                    minimum_similarity=float(
                        args.feature_mode_minimum_similarity
                    ),
                    observation_index=surface_feature_index,
                )
                direct_surface_fallback_group_count = len(
                    direct_surface_pool
                )
                direct_surface_result = (
                    generate_grouped_surface_pose_hypotheses(
                        direct_surface_pool, camera, pose_config
                    )
                )
                direct_surface_fallback_inliers = int(
                    direct_surface_result.hypotheses[0].inlier_count
                    if direct_surface_result.hypotheses
                    else 0
                )
                if direct_surface_result.success:
                    fixed_score, fixed_inliers = _fixed_pose_evidence(
                        fixed_verification_pool,
                        direct_surface_result,
                        camera,
                        pose_config,
                    )
                    pose_candidates.append(
                        (
                            fixed_score,
                            fixed_inliers,
                            float(
                                direct_surface_mode_scores[0]
                                if len(direct_surface_mode_scores)
                                else -np.inf
                            ),
                            0,
                            direct_surface_result,
                            (
                                "radio_final_2dgs_surface_observation_"
                                "fallback"
                            ),
                            (
                                str(direct_surface_mode_ids[0])
                                if direct_surface_mode_ids
                                else None
                            ),
                        )
                    )
            if pose_candidates:
                selected_candidate = max(
                    pose_candidates,
                    key=lambda item: (
                        float(item[0]),
                        int(item[1]),
                        float(item[2]),
                        int(item[3]),
                    ),
                )
                result = selected_candidate[4]
                branch = selected_candidate[5]
                selected_layout_view_id = selected_candidate[6]
                if selected_layout_view_id is not None:
                    selected_layout_score = float(selected_candidate[2])
            feature_mode_measurement_runtime = time.time() - stage_started
            stage_started = time.time()
            refinement_fixed_evidence_delta = None
            if result.success and not bool(args.disable_pose_refinement):
                refinement_pool = (
                    build_pose_guided_surface_anchor_candidate_pool(
                        query,
                        result.pose_w2c,
                        camera,
                        local_bank,
                        anchors,
                        allowed_anchor_ids=retrieved_anchor_ids,
                        top_l=int(args.anchor_top_l),
                        search_radius_px=float(args.pose_refinement_radius_px),
                        maximum_query_points=int(
                            args.pose_refinement_maximum_points
                        ),
                    )
                )
                refinement_pool_size = len(refinement_pool)
                refinement_result = generate_grouped_surface_pose_hypotheses(
                    refinement_pool, camera, pose_config
                )
                if refinement_result.success:
                    base_fixed_score = score_surface_pose(
                        selection_pool,
                        result.pose_w2c,
                        camera,
                        np.ones((len(selection_pool),), dtype=bool),
                        pose_config,
                    )[0]
                    refinement_fixed_score = score_surface_pose(
                        selection_pool,
                        refinement_result.pose_w2c,
                        camera,
                        np.ones((len(selection_pool),), dtype=bool),
                        pose_config,
                    )[0]
                    refinement_fixed_evidence_delta = float(
                        refinement_fixed_score - base_fixed_score
                    )
                    if refinement_fixed_score > base_fixed_score:
                        result = refinement_result
                        branch = (
                            "radio_maplet_alike_set_ot_pose_guided_feature_em"
                        )
            refinement_runtime = time.time() - stage_started
            if result.success:
                successes += 1
            else:
                reason = str(result.failure_reason or "unknown")
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            branch_counts[branch] = branch_counts.get(branch, 0) + 1
            diagnostics = {
                **matcher_diagnostics,
                "query_feature_count": len(query.keypoints_xy),
                "retrieved_maplet_count": len(
                    set(
                        int(value)
                        for value in maplet_match.candidate_maplet_ids.reshape(
                            -1
                        )
                        .tolist()
                        if int(value) >= 0
                    )
                ),
                "retrieved_anchor_count": len(retrieved_anchor_ids),
                "maplet_layout_feature_view_id": None,
                "maplet_layout_score": None,
                "initial_pool_size": len(anchor_pool),
                "set_pose_gate_passed": bool(set_pose_gate_passed),
                "set_pose_confident_group_count": int(
                    confident_set_groups
                ),
                "fixed_verification_pool_size": len(
                    fixed_verification_pool
                ),
                "initial_fixed_evidence_per_group": (
                    float(initial_fixed_score)
                    if np.isfinite(initial_fixed_score)
                    else None
                ),
                "initial_fixed_evidence_inliers": int(initial_fixed_inliers),
                "layout_pool_size": int(layout_pool_size),
                "layout_mode_count": len(feature_mode_ids),
                "layout_success_count": int(layout_success_count),
                "direct_surface_fallback_group_count": int(
                    direct_surface_fallback_group_count
                ),
                "direct_surface_fallback_inliers": int(
                    direct_surface_fallback_inliers
                ),
                "feature_mode_diagnostics": feature_mode_diagnostics,
                "selected_layout_feature_view_id": selected_layout_view_id,
                "selected_layout_feature_score": selected_layout_score,
                "refinement_pool_size": int(refinement_pool_size),
                "refinement_fixed_evidence_delta": (
                    refinement_fixed_evidence_delta
                ),
                "runtime_seconds": float(time.time() - query_started),
                "stage_runtime_seconds": {
                    "coarse_radio_maplet": float(coarse_runtime),
                    "query_alike": float(query_feature_runtime),
                    "anchor_set_matcher": float(anchor_set_runtime),
                    "initial_grouped_pnp": float(initial_pose_runtime),
                    "fixed_verification_pool": float(fixed_pool_runtime),
                    "feature_mode_selection": float(
                        feature_mode_selection_runtime
                    ),
                    "feature_mode_measurement_and_pnp": float(
                        feature_mode_measurement_runtime
                    ),
                    "pose_refinement": float(refinement_runtime),
                },
                "uses_mapping_rgb_at_inference": False,
                "uses_mapping_image_retrieval": False,
                "uses_pairwise_image_matching": False,
                "uses_sfm_points": False,
                "uses_sfm_tracks": False,
                "uses_radio_intermediate": False,
                "uses_view_depth_at_inference": False,
            }
            output.write(
                json.dumps(
                    _pose_record(
                        image_id=record.image_id,
                        result=result,
                        branch=branch,
                        diagnostics=diagnostics,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            output.flush()
            query_file_audit.append(
                {
                    "image_id": record.image_id,
                    "path_relative_to_query_root": str(
                        image_path.relative_to(query_root)
                    ),
                    "sha256": detected.image_sha256,
                }
            )
            print(
                json.dumps(
                    {
                        "query": query_index + 1,
                        "query_count": len(records),
                        "image_id": record.image_id,
                        "success": result.success,
                        "branch": branch,
                        "runtime_seconds": diagnostics["runtime_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    artifacts = {
        name: {
            "path": str(path),
            "sha256": file_sha256_short(Path(path)),
        }
        for name, path in (
            ("query_manifest", args.query_manifest),
            ("surface_mapper_checkpoint", args.surface_mapper_checkpoint),
            ("maplets", args.maplets),
            ("anchors", args.anchors),
            ("local_descriptor_bank", args.local_descriptor_bank),
            ("radio_surface_feature_bank", args.radio_surface_feature_bank),
            (
                "surface_anchor_matcher_checkpoint",
                args.surface_anchor_matcher_checkpoint,
            ),
        )
    }
    summary = {
        "stage": "localize_2dgs_surface_queries_map_only",
        "query_count": len(records),
        "success_count": int(successes),
        "success_rate": float(successes / len(records)),
        "failure_counts": failure_counts,
        "branch_counts": branch_counts,
        "runtime_seconds": float(time.time() - started),
        "artifacts": artifacts,
        "config": {
            "radio_final_layer": str(args.radio_final_layer),
            "region_grid": [
                int(args.query_region_rows),
                int(args.query_region_cols),
                int(args.query_regions_per_cell),
            ],
            "maplet_match": maplet_config.__dict__,
            "anchor_set_matcher": anchor_matcher.config.to_dict(),
            "local_top_k": int(args.local_top_k),
            "anchor_top_l": int(args.anchor_top_l),
            "layout_guided_feature_measurement": {
                "maximum_global_feature_modes": int(
                    args.maximum_global_feature_modes
                ),
                "maximum_feature_modes": int(args.maximum_feature_modes),
                "minimum_vfm_similarity": float(
                    args.feature_mode_minimum_similarity
                ),
                "maximum_points": int(args.layout_guided_maximum_points),
                "search_radius_px": int(args.layout_guided_search_radius_px),
                "seed_prior_logit": float(
                    args.layout_guided_seed_prior_logit
                ),
                "support_source": (
                    "stored_radio_final_2dgs_feature_modes_and_"
                    "stable_anchor_prototypes"
                ),
                "query_rgb_only": True,
            },
            "pose": pose_config.__dict__,
            "pose_refinement": {
                "enabled": not bool(args.disable_pose_refinement),
                "search_radius_px": float(args.pose_refinement_radius_px),
                "maximum_points": int(args.pose_refinement_maximum_points),
                "scope": "radio_retrieved_maplet_anchors_only",
            },
        },
        "runtime_access_audit": {
            "query_image_root": str(query_root),
            "opened_query_image_count": len(query_file_audit),
            "opened_query_images": query_file_audit,
            "mapping_image_paths_accepted_by_cli": False,
        },
        "production_contract": {
            "map_representation": (
                "2dgs_surface_maplets_stable_anchors_and_feature_prototypes"
            ),
            "coarse_match": "radio_final_region_to_maplet",
            "fine_match": (
                "alike_query_set_to_anchor_set_cross_attention_ot_dustbin"
            ),
            "geometry": "grouped_pnp_with_heldout_feature_evidence",
            "support_mode_semantics": (
                "map_feature_mode_label_only_no_mapping_image_access"
            ),
            "uses_mapping_rgb_at_inference": False,
            "uses_mapping_image_retrieval": False,
            "uses_pairwise_image_matching": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
            "uses_view_depth_at_inference": False,
        },
        "output_jsonl": str(output_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
