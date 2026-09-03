"""Match query RADIO tokens directly to a canonical metric plane UV atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _camera_inventory,
    _mutual_matches,
    _radio,
    _records,
    _region_token_support,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
)
from feature_extract.vfm.localization_goal_maplet.chart_local_radio_projection import (
    load_chart_local_radio_projection,
    project_chart_local_radio,
)
from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import (
    load_mapping_subtoken_head,
)
from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import (
    _undistort_simple_radial_xy,
)


def _clip_chart_coordinate_to_original_cell(
    prototype_uv_m: np.ndarray,
    predicted_offset_m: np.ndarray,
    cell_size_m: float,
) -> np.ndarray:
    """Return the predicted coordinate clipped to the prototype's half-open cell."""
    prototype = np.asarray(prototype_uv_m, np.float64).reshape(-1, 2)
    offset = np.asarray(predicted_offset_m, np.float64).reshape(-1, 2)
    size = float(cell_size_m)
    if prototype.shape != offset.shape or not (
        np.all(np.isfinite(prototype)) and np.all(np.isfinite(offset))
    ):
        raise ValueError("chart coordinate arrays differ or are nonfinite")
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("metric atlas cell size differs")
    lower = np.floor(prototype / size) * size
    upper = np.nextafter(lower + size, lower)
    return np.clip(prototype + offset, lower, upper)


def _region_token_measurements(
    labels: np.ndarray,
    region: int,
    token_grid: tuple[int, int] = (36, 64),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return token support and the centroid of observed same-plane pixels."""

    value = np.asarray(labels, np.int32)
    token, visible = _region_token_support(value, int(region), token_grid=token_grid)
    height, width = map(int, token_grid)
    if value.shape != (height * 4, width * 4):
        raise ValueError("plane mask/token-grid geometry differs")
    measurement = np.empty((len(token), 2), np.float64)
    mask = value == int(region)
    for row, token_id in enumerate(token.tolist()):
        ty, tx = divmod(int(token_id), width)
        yy, xx = np.nonzero(mask[ty * 4 : (ty + 1) * 4, tx * 4 : (tx + 1) * 4])
        if not len(xx):
            raise AssertionError("visible plane token has no observed pixel")
        measurement[row] = (float(np.mean(xx + tx * 4)), float(np.mean(yy + ty * 4)))
    return token, visible, measurement


def _metric_homography_filter(
    query_tokens: np.ndarray,
    plane_uv_m: np.ndarray,
    *,
    threshold_m: float,
) -> np.ndarray:
    keep, _, _ = _metric_homography_filter_with_projection(
        query_tokens, plane_uv_m, threshold_m=threshold_m,
    )
    return keep


def _metric_homography_filter_with_projection(
    query_tokens: np.ndarray,
    plane_uv_m: np.ndarray,
    *,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the frozen inlier mask and coarse chart UV at every source token."""
    if len(query_tokens) < 4:
        return (
            np.zeros(len(query_tokens), bool),
            np.zeros((len(query_tokens), 2), np.float64),
            np.zeros(len(query_tokens), bool),
        )
    query_xy = np.c_[query_tokens % 64, query_tokens // 64].astype(np.float64)
    target_uv = np.asarray(plane_uv_m, np.float64).reshape(-1, 2)
    cv2.setRNGSeed(260901)
    homography, mask = cv2.findHomography(
        query_xy,
        target_uv,
        cv2.RANSAC,
        float(threshold_m),
        maxIters=2000,
        confidence=0.995,
    )
    if homography is None or mask is None:
        return (
            np.zeros(len(query_tokens), bool),
            np.zeros((len(query_tokens), 2), np.float64),
            np.zeros(len(query_tokens), bool),
        )
    projected = cv2.perspectiveTransform(
        query_xy.reshape(-1, 1, 2), np.asarray(homography, np.float64),
    ).reshape(-1, 2)
    finite = np.all(np.isfinite(projected), axis=1)
    context_valid = finite & (int(np.sum(mask)) >= 4)
    projected[~context_valid] = target_uv[~context_valid]
    # Preserve the historical correspondence mask exactly.  Reliability is a
    # separate context bit and must not silently change the frozen inventory.
    return mask.reshape(-1).astype(bool), projected, context_valid


def _core_seeded_metric_homography_filter(
    query_tokens: np.ndarray,
    plane_uv_m: np.ndarray,
    visible_fraction: np.ndarray,
    *,
    threshold_m: float,
) -> np.ndarray:
    """Fit on plane-interior tokens, then admit geometrically consistent halo.

    Boundary tokens remain useful for coverage but can no longer determine the
    plane warp when at least four fully observed interior tokens are present.
    No hidden/occluded pixel is added and an invalid core fit fails closed.
    """

    token = np.asarray(query_tokens, np.int64).reshape(-1)
    target = np.asarray(plane_uv_m, np.float64).reshape(-1, 2)
    fraction = np.asarray(visible_fraction, np.float64).reshape(-1)
    if not (len(token) == len(target) == len(fraction)):
        raise ValueError("core/halo metric homography arrays differ")
    if len(token) < 4:
        return np.zeros(len(token), bool)
    query_xy = np.c_[token % 64, token // 64].astype(np.float64)
    core = fraction >= 1.0 - 1e-7
    seed = np.flatnonzero(core) if int(np.sum(core)) >= 4 else np.arange(len(token))
    cv2.setRNGSeed(260901)
    homography, mask = cv2.findHomography(
        query_xy[seed], target[seed], cv2.RANSAC, float(threshold_m),
        maxIters=2000, confidence=0.995,
    )
    if homography is None or mask is None or int(np.sum(mask)) < 4:
        return np.zeros(len(token), bool)
    projected = cv2.perspectiveTransform(
        query_xy.reshape(-1, 1, 2), np.asarray(homography, np.float64),
    ).reshape(-1, 2)
    residual = np.linalg.norm(projected - target, axis=1)
    return np.all(np.isfinite(projected), axis=1) & (residual <= float(threshold_m))


def _top_distinct_hypotheses(
    tokens: np.ndarray,
    scores: np.ndarray,
    plane_rows: np.ndarray,
    texel_rows: np.ndarray,
    *,
    maximum_per_token: int,
) -> np.ndarray:
    chosen: list[int] = []
    for token in np.unique(tokens):
        candidates = np.flatnonzero(tokens == token)
        order = candidates[np.lexsort((texel_rows[candidates], plane_rows[candidates], -scores[candidates]))]
        seen: set[tuple[int, int]] = set()
        for row in order.tolist():
            key = (int(plane_rows[row]), int(texel_rows[row]))
            if key in seen:
                continue
            seen.add(key)
            chosen.append(row)
            if len(seen) == int(maximum_per_token):
                break
    return np.asarray(chosen, np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane_uv_atlas", type=Path, required=True)
    parser.add_argument("--query_plane_dir", type=Path, required=True)
    parser.add_argument("--plane_ranking", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--radio_projection", type=Path,
        help="Mapping-only projection bound by the supplied projected atlas.",
    )
    parser.add_argument("--query_camera_inventory", type=Path, required=True)
    parser.add_argument("--topk_planes", type=int, default=10)
    parser.add_argument("--hypotheses_per_query_token", type=int, default=3)
    parser.add_argument("--homography_threshold_m", type=float, default=1.0)
    parser.add_argument(
        "--query_support_policy", choices=("uniform", "core_seeded_halo_verified"),
        default="uniform",
    )
    parser.add_argument(
        "--query_measurement_policy",
        choices=("token_center", "observed_plane_pixel_centroid", "mapping_pair_subtoken"),
        default="token_center",
    )
    parser.add_argument(
        "--mapping_subtoken_head", type=Path,
        help="Mapping-only pair head; required by mapping_pair_subtoken.",
    )
    parser.add_argument(
        "--planar_map", type=Path,
        help="Finite-plane geometry required by a joint chart-UV coordinate head.",
    )
    parser.add_argument(
        "--surface_coordinate_mean_policy",
        choices=("conditional_match_mean", "match_marginal_mean"),
        default="conditional_match_mean",
    )
    parser.add_argument("--output_correspondences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_correspondences.exists():
        raise FileExistsError("refusing to overwrite plane UV correspondence artifact")
    if int(args.hypotheses_per_query_token) < 1 or float(args.homography_threshold_m) <= 0:
        raise ValueError("invalid plane UV correspondence configuration")
    if (args.query_measurement_policy == "mapping_pair_subtoken") != (args.mapping_subtoken_head is not None):
        raise ValueError("mapping pair subtoken policy/head must be enabled together")

    with np.load(args.plane_uv_atlas, allow_pickle=False) as data:
        atlas_meta = json.loads(str(data["metadata_json"].item()))
        atlas_names = [
            "plane_texel_offsets", "texel_uv_m", "world_points", "radio_features",
            "view_support", "token_support", "texel_identity", "prototype_rank",
        ]
        if atlas_meta.get("artifact_type") in (
            "goal_maplet_metric_plane_uv_radio_atlas_v4",
            "goal_maplet_metric_plane_uv_radio_atlas_v5",
            "goal_maplet_metric_plane_uv_radio_atlas_v6",
            "goal_maplet_metric_plane_uv_radio_atlas_v7",
            "goal_maplet_metric_plane_uv_radio_atlas_v8",
        ):
            atlas_names += ["prototype_view_direction_world", "prototype_observation_range_m"]
        if atlas_meta.get("artifact_type") == "goal_maplet_metric_plane_uv_radio_atlas_v5":
            atlas_names += ["prototype_surface_height_m", "prototype_surface_height_std_m"]
        if atlas_meta.get("artifact_type") in (
            "goal_maplet_metric_plane_uv_radio_atlas_v6",
            "goal_maplet_metric_plane_uv_radio_atlas_v7",
            "goal_maplet_metric_plane_uv_radio_atlas_v8",
        ):
            atlas_names += [
                "prototype_surface_height_m", "prototype_surface_height_std_m",
                "prototype_surface_height_applied_m", "prototype_surface_height_valid",
            ]
        if atlas_meta.get("artifact_type") == "goal_maplet_metric_plane_uv_radio_atlas_v8":
            atlas_names += [
                "prototype_world_covariance_m2", "prototype_plane_pixel_purity",
                "prototype_plane_depth_dispersion_m",
            ]
        atlas = {name: np.asarray(data[name]) for name in atlas_names}
    if (
        atlas_meta.get("artifact_type") not in (
            "goal_maplet_metric_plane_uv_radio_atlas_v2",
            "goal_maplet_metric_plane_uv_radio_atlas_v3",
            "goal_maplet_metric_plane_uv_radio_atlas_v4",
            "goal_maplet_metric_plane_uv_radio_atlas_v5",
            "goal_maplet_metric_plane_uv_radio_atlas_v6",
            "goal_maplet_metric_plane_uv_radio_atlas_v7",
            "goal_maplet_metric_plane_uv_radio_atlas_v8",
        )
        or atlas_meta.get("uses_query_pose_depth_or_ground_truth") is not False
        or arrays_sha256(atlas) != atlas_meta.get("arrays_sha256")
    ):
        raise ValueError("metric plane UV atlas contract differs")
    ranking = json.loads(args.plane_ranking.read_text())
    if (
        ranking.get("artifact_type")
        not in (
            "goal_maplet_direct_radio_to_finite_plane_ranking_v1",
            "goal_maplet_direct_radio_to_finite_plane_ranking_v2",
            "goal_maplet_context_fused_direct_radio_to_finite_plane_ranking_v1",
            "goal_maplet_context_fused_direct_radio_to_finite_plane_ranking_v2",
        )
        or ranking.get("uses_pose_or_ground_truth") is not False
        or ranking.get("contains_postlabel_fields") is not False
    ):
        raise ValueError("plane ranking is not pose/label-free")
    radio_records = _records(args.radio_manifest)
    cameras, camera_meta = _camera_inventory(args.query_camera_inventory)
    projection_meta = None
    projection_weight = None
    if args.radio_projection is not None:
        projection_weight, projection_meta = load_chart_local_radio_projection(args.radio_projection)
    expected_projection = atlas_meta.get("chart_local_radio_projection_content_sha256")
    actual_projection = None if projection_meta is None else projection_meta.get("content_sha256")
    if expected_projection != actual_projection:
        raise ValueError("query RADIO projection does not match the atlas projection")
    subtoken_head = None
    subtoken_meta = None
    surface_planes = None
    use_surface_coordinate = False
    use_geometric_context = False
    use_homography_context = False
    if args.mapping_subtoken_head is not None:
        subtoken_head, subtoken_meta = load_mapping_subtoken_head(args.mapping_subtoken_head)
        if (
            subtoken_meta.get("radio_projection_content_sha256") != actual_projection
            or int(subtoken_meta.get("feature_dimension", -1))
            != int(atlas["radio_features"].shape[1])
            or subtoken_meta.get("mapping_validation_gate_pass") is not True
        ):
            raise ValueError("mapping subtoken head is incompatible or failed mapping validation")
        use_surface_coordinate = subtoken_meta.get("artifact_type") in {
            "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_head_v7",
            "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_context_head_v9",
            "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_deep_context_head_v10",
            "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_homography_context_head_v11",
        }
        use_geometric_context = (
            subtoken_meta.get("artifact_type") in {
                "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_context_head_v9",
                "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_deep_context_head_v10",
                "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_homography_context_head_v11",
            }
        )
        use_homography_context = (
            subtoken_meta.get("artifact_type")
            == "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_homography_context_head_v11"
        )
        if use_geometric_context and (
            subtoken_meta.get("geometric_context_enabled") is not True
            or int(subtoken_meta.get("geometric_context_dimension", -1))
            != (8 if use_homography_context else 5)
        ):
            raise ValueError("mapping surface-coordinate context contract differs")
        if use_homography_context and (
            args.query_support_policy != "uniform"
            or abs(float(subtoken_meta.get("homography_context_threshold_m", -1.0))
                   - float(args.homography_threshold_m)) > 1e-12
        ):
            raise ValueError("mapping homography context differs from the frozen coarse fit")
        if use_surface_coordinate:
            if args.planar_map is None:
                raise ValueError("surface-coordinate head requires its finite planar map")
            if (
                file_sha256(args.planar_map) != subtoken_meta.get("planar_map_file_sha256")
                or file_sha256(args.planar_map) != atlas_meta.get("planar_map_file_sha256")
            ):
                raise ValueError("surface-coordinate planar map lineage differs")
            surface_planes = GeometryNativePlanarMap.load_npz(args.planar_map)
        elif args.planar_map is not None:
            raise ValueError("planar map is only consumed by a surface-coordinate head")
    if not use_surface_coordinate and args.surface_coordinate_mean_policy != "conditional_match_mean":
        raise ValueError("surface-coordinate mean policy requires a surface-coordinate head")

    names, point_rows, token_rows, measurement_rows, provenance_rows, matrices, radial = [], [], [], [], [], [], []
    prototype_rows, visible_rows, score_rows = [], [], []
    covariance_rows, purity_rows, dispersion_rows = [], [], []
    measurement_variance_rows, match_probability_rows = [], []
    centroid_covariance_rows = []
    chart_uv_measurement_rows, chart_uv_cell_lower_rows = [], []
    diagnostic_rows = []
    for query in ranking["rows"]:
        name = str(query["image"])
        if name not in cameras:
            raise ValueError("query camera inventory lacks query")
        model_id, width, height, params = cameras[name]
        # Keep this import local to make the phase boundary explicit: camera
        # inventory contains intrinsics only and has already excluded pose.
        from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics

        K, k1 = _scaled_intrinsics(model_id, params, width, height)
        planes, _ = QueryPlaneRegions.load_npz(args.query_plane_dir / name)
        query_feature = _radio(name, radio_records)
        if projection_weight is not None:
            query_feature = project_chart_local_radio(query_feature, projection_weight)
        all_points, all_tokens, all_measurements, all_scores, all_planes, all_texels, all_regions = [], [], [], [], [], [], []
        all_query_descriptor, all_map_descriptor = [], []
        all_homography_projection = []
        all_homography_valid = []
        all_prototypes, all_visible = [], []
        all_covariance, all_purity, all_dispersion = [], [], []
        for region_row in query["regions"]:
            region = int(region_row["region"])
            if args.query_measurement_policy == "observed_plane_pixel_centroid":
                qtoken, qvisible, qmeasurement = _region_token_measurements(planes.labels, region)
            else:
                qtoken, qvisible = _region_token_support(planes.labels, region)
                qmeasurement = np.c_[(qtoken % 64) * 4 + 1.5, (qtoken // 64) * 4 + 1.5]
            if len(qtoken) < 4:
                continue
            qfeature = query_feature[qtoken]
            for rank, plane in enumerate(region_row["top10"][: int(args.topk_planes)]):
                plane = int(plane)
                lo, hi = map(int, atlas["plane_texel_offsets"][plane : plane + 2])
                if hi - lo < 4:
                    continue
                qi, ti, score = _mutual_matches(qfeature, atlas["radio_features"][lo:hi].astype(np.float32))
                if len(qi) < 4:
                    continue
                if args.query_support_policy == "core_seeded_halo_verified":
                    keep = _core_seeded_metric_homography_filter(
                        qtoken[qi], atlas["texel_uv_m"][lo + ti], qvisible[qi],
                        threshold_m=float(args.homography_threshold_m),
                    )
                    homography_projection = np.full((len(qi), 2), np.nan)
                elif use_homography_context:
                    keep, homography_projection, homography_valid = _metric_homography_filter_with_projection(
                        qtoken[qi], atlas["texel_uv_m"][lo + ti],
                        threshold_m=float(args.homography_threshold_m),
                    )
                else:
                    keep = _metric_homography_filter(
                        qtoken[qi], atlas["texel_uv_m"][lo + ti],
                        threshold_m=float(args.homography_threshold_m),
                    )
                    homography_projection = np.full((len(qi), 2), np.nan)
                    homography_valid = np.zeros(len(qi), bool)
                selected = np.flatnonzero(keep)
                if not len(selected):
                    continue
                prototype = lo + ti[selected]
                all_points.append(atlas["world_points"][prototype])
                all_tokens.append(qtoken[qi[selected]])
                all_measurements.append(qmeasurement[qi[selected]])
                all_scores.append(score[selected] - 0.02 * rank)
                all_planes.append(np.full(len(selected), plane, np.int64))
                all_texels.append(atlas["texel_identity"][prototype].astype(np.int64))
                all_regions.append(np.full(len(selected), region, np.int64))
                all_prototypes.append(prototype.astype(np.int64))
                all_visible.append(qvisible[qi[selected]].astype(np.float64))
                all_query_descriptor.append(qfeature[qi[selected]].astype(np.float32))
                all_map_descriptor.append(atlas["radio_features"][prototype].astype(np.float32))
                if use_homography_context:
                    all_homography_projection.append(homography_projection[selected])
                    all_homography_valid.append(homography_valid[selected])
                if atlas_meta.get("artifact_type") == "goal_maplet_metric_plane_uv_radio_atlas_v8":
                    all_covariance.append(atlas["prototype_world_covariance_m2"][prototype])
                    all_purity.append(atlas["prototype_plane_pixel_purity"][prototype])
                    all_dispersion.append(atlas["prototype_plane_depth_dispersion_m"][prototype])
        if all_points:
            world = np.concatenate(all_points)
            token = np.concatenate(all_tokens)
            measurement = np.concatenate(all_measurements)
            score = np.concatenate(all_scores)
            plane_row = np.concatenate(all_planes)
            texel_row = np.concatenate(all_texels)
            region_row = np.concatenate(all_regions)
            prototype_row = np.concatenate(all_prototypes)
            visible_fraction = np.concatenate(all_visible)
            chosen = _top_distinct_hypotheses(
                token, score, plane_row, texel_row,
                maximum_per_token=int(args.hypotheses_per_query_token),
            )
            world, token, measurement = world[chosen], token[chosen], measurement[chosen]
            provenance = np.c_[region_row[chosen], plane_row[chosen], texel_row[chosen]]
            prototype_row = prototype_row[chosen]
            visible_fraction = visible_fraction[chosen]
            match_score = score[chosen]
            if subtoken_head is not None:
                qdescriptor = np.concatenate(all_query_descriptor)[chosen]
                mdescriptor = np.concatenate(all_map_descriptor)[chosen]
                with torch.no_grad():
                    head_arguments = [
                        torch.from_numpy(qdescriptor), torch.from_numpy(mdescriptor),
                        torch.from_numpy(token.astype(np.int64)),
                    ]
                    if use_geometric_context:
                        token_center_xy = np.c_[
                            (token % 64) * 4 + 1.5, (token // 64) * 4 + 1.5,
                        ]
                        ideal = _undistort_simple_radial_xy(
                            token_center_xy,
                            np.broadcast_to(K, (len(token), 3, 3)),
                            np.full(len(token), k1, np.float64),
                        )
                        ray = np.c_[ideal, np.ones(len(ideal), np.float64)]
                        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)
                        query_normal = np.asarray(
                            planes.normals_camera[provenance[:, 0]], np.float64,
                        )
                        incidence = np.abs(np.sum(query_normal * ray, axis=1))
                        prototype_uv_context = np.asarray(
                            atlas["texel_uv_m"][prototype_row], np.float64,
                        )
                        cell_size_context = float(atlas_meta.get("cell_size_m", 0.5))
                        phase = 2.0 * (
                            prototype_uv_context / cell_size_context
                            - np.floor(prototype_uv_context / cell_size_context) - 0.5
                        )
                        context = np.c_[ideal, phase, incidence].astype(np.float32)
                        if use_homography_context:
                            projected_uv = np.concatenate(all_homography_projection)[chosen]
                            projected_valid = np.concatenate(all_homography_valid)[chosen]
                            residual = np.clip(
                                (projected_uv - prototype_uv_context)
                                / float(args.homography_threshold_m), -2.0, 2.0,
                            )
                            residual[~projected_valid] = 0.0
                            if not np.all(np.isfinite(residual)):
                                raise ValueError("selected mapping homography context is nonfinite")
                            context = np.c_[
                                context, residual, projected_valid.astype(np.float32),
                            ].astype(np.float32)
                        head_arguments.append(torch.from_numpy(context))
                    head_output = subtoken_head(*head_arguments)
                mean, variance = head_output[:2]
                match_logit = head_output[4] if use_surface_coordinate else head_output[2]
                token_center = np.c_[(token % 64) * 4 + 1.5, (token // 64) * 4 + 1.5]
                coordinate_shrinkage = float(subtoken_meta.get("coordinate_shrinkage", 1.0))
                if not (0.0 <= coordinate_shrinkage <= 1.0):
                    raise ValueError("mapping subtoken coordinate shrinkage differs")
                offset = mean.numpy().astype(np.float64)
                affine_matrix = subtoken_meta.get("coordinate_affine_matrix")
                affine_bias = subtoken_meta.get("coordinate_affine_bias_px")
                if affine_matrix is None:
                    offset = coordinate_shrinkage * offset
                else:
                    matrix = np.asarray(affine_matrix, np.float64)
                    bias = np.asarray(affine_bias, np.float64)
                    if matrix.shape != (2, 2) or bias.shape != (2,) or not (
                        np.all(np.isfinite(matrix)) and np.all(np.isfinite(bias))
                    ):
                        raise ValueError("mapping subtoken coordinate affine calibration differs")
                    offset = offset @ matrix + bias
                measurement = token_center + np.clip(offset, -2.0, 2.0)
                variance_scale = float(subtoken_meta.get("measurement_variance_scale", 1.0))
                if not np.isfinite(variance_scale) or variance_scale <= 0.0:
                    raise ValueError("mapping subtoken variance scale differs")
                measurement_variance = variance_scale * variance.numpy().reshape(-1)
                match_probability = torch.sigmoid(match_logit).numpy().reshape(-1)
                if use_surface_coordinate:
                    assert surface_planes is not None
                    chart_offset_shrinkage = float(
                        subtoken_meta.get("chart_uv_coordinate_shrinkage", 1.0)
                    )
                    if not (0.0 <= chart_offset_shrinkage <= 1.0):
                        raise ValueError("mapping chart-UV coordinate shrinkage differs")
                    chart_offset = (
                        chart_offset_shrinkage
                        * head_output[2].numpy().astype(np.float64)
                    )
                    if args.surface_coordinate_mean_policy == "match_marginal_mean":
                        chart_offset = chart_offset * match_probability[:, None]
                    prototype_uv = np.asarray(
                        atlas["texel_uv_m"][prototype_row], np.float64,
                    )
                    cell_size_m = float(atlas_meta.get("cell_size_m", 0.5))
                    chart_uv_measurement = _clip_chart_coordinate_to_original_cell(
                        prototype_uv, chart_offset, cell_size_m,
                    )
                    chart_offset = chart_uv_measurement - prototype_uv
                    chart_uv_cell_lower = (
                        np.floor(prototype_uv / cell_size_m) * cell_size_m
                    )
                    chart_variance_scale = float(
                        subtoken_meta.get("chart_uv_measurement_variance_scale", 1.0)
                    )
                    if not np.isfinite(chart_variance_scale) or chart_variance_scale <= 0.0:
                        raise ValueError("mapping chart-UV variance scale differs")
                    chart_variance = (
                        chart_variance_scale
                        * head_output[3].numpy().reshape(-1).astype(np.float64)
                    )
                    plane_index = provenance[:, 1]
                    if np.any((plane_index < 0) | (plane_index >= len(surface_planes.plane_ids))):
                        raise ValueError("surface-coordinate plane row is outside planar map")
                    tangent = np.asarray(surface_planes.frames_world[plane_index, :2], np.float64)
                    world = world + np.einsum("ni,nij->nj", chart_offset, tangent)
                    tangent_projector = np.einsum("nki,nkj->nij", tangent, tangent)
                    centroid_covariance = chart_variance[:, None, None] * tangent_projector
                else:
                    centroid_covariance = np.zeros((len(chosen), 3, 3), np.float64)
                    chart_uv_measurement = np.zeros((len(chosen), 2), np.float64)
                    chart_uv_cell_lower = np.zeros((len(chosen), 2), np.float64)
            else:
                measurement_variance = np.zeros(len(chosen), np.float64)
                match_probability = np.ones(len(chosen), np.float64)
                centroid_covariance = np.zeros((len(chosen), 3, 3), np.float64)
                chart_uv_measurement = np.zeros((len(chosen), 2), np.float64)
                chart_uv_cell_lower = np.zeros((len(chosen), 2), np.float64)
            if atlas_meta.get("artifact_type") == "goal_maplet_metric_plane_uv_radio_atlas_v8":
                covariance = np.concatenate(all_covariance)[chosen]
                purity = np.concatenate(all_purity)[chosen]
                dispersion = np.concatenate(all_dispersion)[chosen]
            else:
                covariance = np.zeros((len(chosen), 3, 3), np.float64)
                purity = np.ones(len(chosen), np.float64)
                dispersion = np.zeros(len(chosen), np.float64)
        else:
            world = np.zeros((0, 3), np.float64)
            token = np.zeros(0, np.int64)
            measurement = np.zeros((0, 2), np.float64)
            provenance = np.zeros((0, 3), np.int64)
            prototype_row = np.zeros(0, np.int64)
            visible_fraction = np.zeros(0, np.float64)
            match_score = np.zeros(0, np.float64)
            covariance = np.zeros((0, 3, 3), np.float64)
            purity = np.zeros(0, np.float64)
            dispersion = np.zeros(0, np.float64)
            measurement_variance = np.zeros(0, np.float64)
            match_probability = np.zeros(0, np.float64)
            centroid_covariance = np.zeros((0, 3, 3), np.float64)
            chart_uv_measurement = np.zeros((0, 2), np.float64)
            chart_uv_cell_lower = np.zeros((0, 2), np.float64)
        names.append(name); point_rows.append(world); token_rows.append(token); measurement_rows.append(measurement)
        provenance_rows.append(provenance); matrices.append(K); radial.append(k1)
        prototype_rows.append(prototype_row); visible_rows.append(visible_fraction); score_rows.append(match_score)
        covariance_rows.append(covariance); purity_rows.append(purity); dispersion_rows.append(dispersion)
        measurement_variance_rows.append(measurement_variance); match_probability_rows.append(match_probability)
        centroid_covariance_rows.append(centroid_covariance)
        chart_uv_measurement_rows.append(chart_uv_measurement)
        chart_uv_cell_lower_rows.append(chart_uv_cell_lower)
        diagnostic_rows.append({
            "name": name,
            "correspondence_count": int(len(token)),
            "unique_query_token_count": int(len(np.unique(token))),
            "physical_plane_count": int(len(np.unique(provenance[:, 1]))) if len(provenance) else 0,
            "metric_texel_count": int(len(np.unique(provenance[:, 2]))) if len(provenance) else 0,
        })

    offsets = np.r_[0, np.cumsum([len(row) for row in token_rows])].astype(np.int64)
    arrays = {
        "names": np.asarray(names),
        "correspondence_offsets": offsets,
        "world_points": np.concatenate(point_rows).astype(np.float64),
        "query_tokens": np.concatenate(token_rows).astype(np.int64),
        "provenance_region_plane_atlas_row": np.concatenate(provenance_rows).astype(np.int64),
        "camera_matrices": np.asarray(matrices, np.float64),
        "radial_k1": np.asarray(radial, np.float64),
        "prototype_atlas_row": np.concatenate(prototype_rows).astype(np.int64),
        "query_plane_visible_fraction": np.concatenate(visible_rows).astype(np.float32),
        "radio_match_score": np.concatenate(score_rows).astype(np.float32),
        "prototype_world_covariance_m2": np.concatenate(covariance_rows).astype(np.float32),
        "prototype_plane_pixel_purity": np.concatenate(purity_rows).astype(np.float32),
        "prototype_plane_depth_dispersion_m": np.concatenate(dispersion_rows).astype(np.float32),
    }
    use_subtoken_measurement = args.query_measurement_policy != "token_center"
    if use_subtoken_measurement:
        arrays["query_measurements_xy"] = np.concatenate(measurement_rows).astype(np.float32)
    if args.query_measurement_policy == "mapping_pair_subtoken":
        arrays["query_measurement_variance_px2"] = np.concatenate(measurement_variance_rows).astype(np.float32)
        arrays["correspondence_match_probability"] = np.concatenate(match_probability_rows).astype(np.float32)
        if use_surface_coordinate:
            arrays["prototype_centroid_covariance_world_m2"] = np.concatenate(
                centroid_covariance_rows,
            ).astype(np.float32)
            arrays["prototype_chart_uv_measurement_m"] = np.concatenate(
                chart_uv_measurement_rows,
            ).astype(np.float64)
            arrays["prototype_chart_uv_cell_lower_m"] = np.concatenate(
                chart_uv_cell_lower_rows,
            ).astype(np.float64)
    metadata = {
        "artifact_type": (
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7"
            if use_surface_coordinate
            else "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5"
            if args.query_measurement_policy == "mapping_pair_subtoken"
            else "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4"
            if args.query_measurement_policy == "observed_plane_pixel_centroid"
            else "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v3"
        ),
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "correspondence_count": int(offsets[-1]),
        "pose_or_ground_truth_opened": False,
        "query_depth_or_scale_used": False,
        "correspondence_semantics": "query_RADIO_to_view_independent_metric_plane_UV_texels",
        "hypotheses_per_query_token": int(args.hypotheses_per_query_token),
        "topk_planes": int(args.topk_planes),
        "homography_threshold_m": float(args.homography_threshold_m),
        "query_support_policy": str(args.query_support_policy),
        "query_boundary_token_role": (
            "verified_halo_only" if args.query_support_policy == "core_seeded_halo_verified"
            else "equal_to_core"
        ),
        "prototype_geometry_binding": "explicit_atlas_row_not_texel_identity",
        "query_support_weight_available": "visible_fraction_for_soft_core_halo_weighting",
        "prototype_geometry_uncertainty_available": bool(
            atlas_meta.get("artifact_type") == "goal_maplet_metric_plane_uv_radio_atlas_v8"
        ),
        "plane_uv_atlas_file_sha256": file_sha256(args.plane_uv_atlas),
        "plane_uv_atlas_content_sha256": atlas_meta.get("content_sha256"),
        "plane_ranking_file_sha256": file_sha256(args.plane_ranking),
        "query_camera_only_inventory_file_sha256": file_sha256(args.query_camera_inventory),
        "query_camera_only_inventory_content_sha256": camera_meta.get("content_sha256"),
        "radio_manifest_file_sha256_in_order": [file_sha256(path) for path in args.radio_manifest],
        "chart_local_radio_projection_file_sha256": (
            None if args.radio_projection is None else file_sha256(args.radio_projection)
        ),
        "chart_local_radio_projection_content_sha256": actual_projection,
        "runtime_map_stores_source_rgb": False,
    }
    if use_subtoken_measurement:
        metadata.update(
            query_measurement_semantics=(
                "centroid_of_observed_same_plane_region_pixels_inside_each_4x4_RADIO_token"
                if args.query_measurement_policy == "observed_plane_pixel_centroid"
                else (
                    "mapping_only_joint_query_subtoken_and_continuous_chartUV_surface_coordinate_mean"
                    if use_surface_coordinate
                    else "mapping_only_RADIO_query_to_anonymous_atlas_prototype_continuous_subtoken_mean_inside_original_4x4_token"
                )
            ),
            hidden_or_occluded_pixels_added_to_query_measurement=0,
        )
    if args.query_measurement_policy == "mapping_pair_subtoken":
        metadata.update(
            query_measurement_uncertainty_semantics=(
                "predicted_query_centroid_variance_px2_plus_tangent_chartUV_centroid_covariance_world_m2_not_surface_footprint"
                if use_surface_coordinate
                else "predicted_isotropic_centroid_measurement_variance_px2_not_surface_footprint"
            ),
            correspondence_null_semantics="mapping_only_same_plane_different_metric_cell_probability_head_no_runtime_threshold",
            mapping_subtoken_head_file_sha256=file_sha256(args.mapping_subtoken_head),
            mapping_subtoken_head_content_sha256=subtoken_meta.get("content_sha256"),
            mapping_subtoken_coordinate_shrinkage=float(subtoken_meta.get("coordinate_shrinkage", 1.0)),
            mapping_subtoken_measurement_variance_scale=float(subtoken_meta.get("measurement_variance_scale", 1.0)),
            mapping_subtoken_validation_route=subtoken_meta.get("validation_mapping_route"),
            mapping_subtoken_prototype_policy=subtoken_meta.get("prototype_policy"),
            mapping_subtoken_shrinkage_calibration=subtoken_meta.get("coordinate_shrinkage_calibration"),
        )
        if use_surface_coordinate:
            metadata.update(
                mapping_chart_uv_measurement_variance_scale=float(
                    subtoken_meta["chart_uv_measurement_variance_scale"]
                ),
                mapping_chart_uv_coordinate_shrinkage=float(
                    subtoken_meta.get("chart_uv_coordinate_shrinkage", 1.0)
                ),
                planar_map_file_sha256=file_sha256(args.planar_map),
                continuous_map_coordinate_semantics=(
                    "anonymous_atlas_centroid_plus_predicted_metric_offset_in_finite_plane_tangent_frame"
                ),
                surface_coordinate_mean_policy=str(args.surface_coordinate_mean_policy),
                chart_uv_metric_cell_support_enforced=True,
                chart_uv_metric_cell_size_m=float(atlas_meta.get("cell_size_m", 0.5)),
                mapping_surface_geometric_context_semantics=(
                    subtoken_meta.get("geometric_context_semantics")
                    if use_geometric_context else "none"
                ),
            )
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output_correspondences.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_correspondences, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = {
        "artifact_type": "goal_maplet_metric_plane_uv_radio_correspondence_build_v1",
        "frozen_correspondence_file_sha256": file_sha256(args.output_correspondences),
        "frozen_correspondence_content_sha256": metadata["content_sha256"],
        "query_count": int(len(names)),
        "median_correspondence_count": float(np.median([row["correspondence_count"] for row in diagnostic_rows])),
        "median_unique_query_token_count": float(np.median([row["unique_query_token_count"] for row in diagnostic_rows])),
        "pose_or_ground_truth_opened": False,
        "rows": diagnostic_rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
