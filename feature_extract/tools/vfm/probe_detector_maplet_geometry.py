"""Probe candidate-conditioned maplet geometry on real detector proposals."""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.local_maplet_matching import load_local_maplet_support_index_npz
from feature_extract.vfm.localization.detector_landmark_proposals import (
    summarize_detector_proposal_geometry,
    summarize_ranked_detector_proposal_geometry,
)
from feature_extract.vfm.localization.candidate_maplet_schema import (
    CANDIDATE_MAPLET_STATIC_FEATURE_NAMES,
    CANDIDATE_MAPLET_STATIC_SCHEMA_VERSION,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import (
    UniqueTrackCandidateSet,
    binary_average_precision,
)
from feature_extract.vfm.localization.local_assignment_linear import selective_switch_scores
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationFeatureStore,
    aggregate_maplet_view_evidence,
    canonical_rows_for_track_candidates,
    load_support_observation_geometry_index_npz,
    score_maplet_support_view,
    select_dense_query_context_neighborhood,
    select_support_maplet_view_indices,
)
from feature_extract.vfm.localization.real_image_observation_features import (
    spatially_diverse_detection_indices,
)


STATIC_FEATURE_NAMES = CANDIDATE_MAPLET_STATIC_FEATURE_NAMES
QUERY_POINT_SELECTION_POLICY = (
    "pose_keep_priority_then_full_detector_spatial_fallback_v2"
)


def _float_list(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--query_context_detector_cache", default=None)
    parser.add_argument("--support_feature_cache", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--feature_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--feature_scope",
        default="full_probe",
        choices=("full_probe", "static_only"),
        help="static_only builds the exact 17 fields consumed by candidate-maplet training",
    )
    parser.add_argument(
        "--build_only",
        action="store_true",
        help="write the validated feature artifact without fitting or pose evaluation",
    )
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help=(
            "build model inputs without loading or writing GT residuals/labels; "
            "requires --build_only"
        ),
    )
    parser.add_argument("--baseline_strategy", default="alike_support_top2_mean")
    parser.add_argument("--candidate_top_k", type=int, default=10)
    parser.add_argument("--query_points_per_image", type=int, default=128)
    parser.add_argument("--query_neighborhood_radius_px", type=float, default=96.0)
    parser.add_argument("--max_query_neighbors", type=int, default=48)
    parser.add_argument("--support_view_count", type=int, default=2)
    parser.add_argument("--support_view_candidate_count", type=int, default=2)
    parser.add_argument(
        "--support_view_selection",
        default="coverage",
        choices=("coverage", "anchor_similarity", "anchor_coverage"),
    )
    parser.add_argument("--support_view_coverage_weight", type=float, default=0.1)
    parser.add_argument("--max_maplet_tracks", type=int, default=33)
    parser.add_argument("--min_descriptor_similarity", type=float, default=0.20)
    parser.add_argument("--max_tentative_matches", type=int, default=16)
    parser.add_argument("--geometry_threshold_px", type=float, default=8.0)
    parser.add_argument("--positive_threshold_px", type=float, default=2.0)
    parser.add_argument("--train_query_count", type=int, default=60)
    parser.add_argument("--validation_query_count", type=int, default=15)
    parser.add_argument("--c_values", type=_float_list, default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument(
        "--switch_margin_thresholds",
        type=_float_list,
        default=(0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7),
    )
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    parser.add_argument("--maplet_view_cache_size", type=int, default=4096)
    return parser.parse_args(argv)


def _query_neighborhood(
    xy: np.ndarray,
    anchor: int,
    *,
    radius_px: float,
    max_points: int,
) -> tuple[np.ndarray, int]:
    points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if int(anchor) < 0 or int(anchor) >= len(points):
        raise ValueError("query neighborhood anchor is out of range")
    distances2 = np.sum((points - points[int(anchor)]) ** 2, axis=1)
    rows = np.flatnonzero(distances2 <= float(radius_px) ** 2)
    order = np.lexsort((rows, distances2[rows]))
    rows = rows[order[: min(int(max_points), len(order))]]
    anchor_positions = np.flatnonzero(rows == int(anchor))
    if anchor_positions.size != 1:
        raise RuntimeError("query neighborhood dropped its anchor")
    return rows.astype(np.int64), int(anchor_positions[0])


def _pose_gate(candidate: dict[str, object], baseline: dict[str, object]) -> bool:
    lower = ("median_translation_m_success", "p90_translation_m_success", "median_rotation_deg_success")
    higher = ("success_rate", "recall_25cm_2deg", "recall_10cm_5deg", "recall_5cm_5deg")
    tolerance = 1e-12
    try:
        candidate_values = {
            name: float(candidate[name]) for name in (*lower, *higher)
        }
        baseline_values = {
            name: float(baseline[name]) for name in (*lower, *higher)
        }
    except (KeyError, TypeError, ValueError):
        return False
    if not all(
        np.isfinite(value)
        for value in (*candidate_values.values(), *baseline_values.values())
    ):
        return False
    no_regression = all(
        candidate_values[name] <= baseline_values[name] + tolerance for name in lower
    )
    no_regression &= all(
        candidate_values[name] + tolerance >= baseline_values[name] for name in higher
    )
    improves = any(
        candidate_values[name] < baseline_values[name] - tolerance for name in lower
    )
    improves |= any(
        candidate_values[name] > baseline_values[name] + tolerance for name in higher
    )
    return bool(no_regression and improves)


def _load_inputs(args: argparse.Namespace) -> dict[str, object]:
    with np.load(Path(args.proposals), allow_pickle=False) as data:
        supervision_keys = {
            "nearest_visible_bank_rows",
            "nearest_visible_track_ids",
            "nearest_visible_residuals_px",
            "candidate_gt_residuals_px",
        }
        proposals = {
            key: np.asarray(data[key])
            for key in data.files
            if not bool(args.inference_only) or key not in supervision_keys
        }
    with np.load(Path(args.detector_query_cache), allow_pickle=False) as data:
        query_cache = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
        query_metadata = json.loads(str(data["metadata_json"].item()))
    context_cache = None
    context_metadata = None
    if args.query_context_detector_cache:
        with np.load(Path(args.query_context_detector_cache), allow_pickle=False) as data:
            context_cache = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
            context_metadata = json.loads(str(data["metadata_json"].item()))
    with np.load(Path(args.support_feature_cache), allow_pickle=False) as data:
        support_tracks = np.asarray(data["track_ids"], dtype=np.int64)
        support_descriptors = np.asarray(data["descriptors"], dtype=np.float32)
        support_scores = np.asarray(data["detector_scores"], dtype=np.float32)
        support_metadata = json.loads(str(data["metadata_json"].item()))
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(args.support_geometry_index)
    )
    landmark_index, landmark_metadata = load_landmark_index_npz(Path(args.projected_landmark_bank))
    maplet_index, maplet_metadata = load_local_maplet_support_index_npz(Path(args.maplet_support_index))

    query_ids = np.asarray(proposals["query_ids"]).astype(str)
    query_xy = np.asarray(proposals["xy"], dtype=np.float32)
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    query_image_sizes: dict[str, tuple[int, int]] = {}
    for query_id in dict.fromkeys(query_ids.tolist()):
        image = images_by_name.get(str(query_id))
        if image is None:
            raise KeyError(f"query image missing from COLMAP model: {query_id}")
        camera = cameras[int(image.camera_id)]
        query_image_sizes[str(query_id)] = (int(camera.width), int(camera.height))
    cache_ids = np.repeat(
        np.asarray(query_cache["image_ids"]).astype(str),
        np.diff(np.asarray(query_cache["offsets"], dtype=np.int64)),
    )
    if not np.array_equal(query_ids, cache_ids) or not np.array_equal(query_xy, query_cache["xy"]):
        raise ValueError("proposal rows and detector query cache differ")
    baseline_key = f"strategy__{args.baseline_strategy}"
    if baseline_key not in proposals:
        raise ValueError(f"proposal artifact is missing {baseline_key}")
    if str(query_metadata.get("alike_checkpoint_sha256", "")) != str(
        support_metadata.get("model_checkpoint_sha256", "")
    ):
        raise ValueError("query and support ALIKE checkpoints differ")
    if context_cache is not None and context_metadata is not None:
        if context_metadata.get("format") != "alike_dense_query_context_cache_v1":
            raise ValueError("unsupported dense query context cache")
        if str(context_metadata.get("alike_checkpoint_sha256", "")) != str(
            query_metadata.get("alike_checkpoint_sha256", "")
        ):
            raise ValueError("anchor and context ALIKE checkpoints differ")
        if not np.array_equal(context_cache["image_ids"].astype(str), query_cache["image_ids"].astype(str)):
            raise ValueError("anchor and context query image lists differ")
        expected_hashes = query_metadata.get("image_sha256_by_id")
        actual_hashes = context_metadata.get("image_sha256_by_id")
        if not isinstance(expected_hashes, dict) or expected_hashes != actual_hashes:
            raise ValueError("anchor and context query image contents differ")
        if int(context_cache["local_descriptors"].shape[1]) != int(query_cache["local_descriptors"].shape[1]):
            raise ValueError("anchor and context local descriptor dimensions differ")
    if str(geometry_metadata.get("support_feature_cache_sha256", "")) != file_sha256_short(
        Path(args.support_feature_cache)
    ):
        raise ValueError("support geometry index references a different feature cache")
    if not np.array_equal(maplet_index.anchor_track_ids, landmark_index.track_ids):
        raise ValueError("maplet anchors and projected landmark bank rows differ")
    maplet_source_hash = str(maplet_metadata.get("source_landmark_index_sha256", ""))
    if maplet_source_hash and maplet_source_hash != file_sha256_short(Path(args.projected_landmark_bank)):
        raise ValueError("maplet index was built from a different landmark bank")
    candidate_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    proposals["canonical_bank_row_indices"] = canonical_rows_for_track_candidates(
        candidate_tracks,
        landmark_index.track_ids,
    )
    feature_store = SupportObservationFeatureStore(
        geometry,
        source_track_ids=support_tracks,
        source_descriptors=support_descriptors,
        source_detector_scores=support_scores,
    )
    return {
        "proposals": proposals,
        "query_cache": query_cache,
        "query_metadata": query_metadata,
        "query_context_cache": context_cache,
        "query_context_metadata": context_metadata,
        "geometry_metadata": geometry_metadata,
        "landmark_index": landmark_index,
        "landmark_metadata": landmark_metadata,
        "maplet_index": maplet_index,
        "maplet_metadata": maplet_metadata,
        "feature_store": feature_store,
        "query_ids": query_ids,
        "query_xy": query_xy,
        "query_image_sizes": query_image_sizes,
    }


def _select_query_rows(
    *,
    query_ids: np.ndarray,
    xy: np.ndarray,
    detector_scores: np.ndarray,
    pose_keep_mask: np.ndarray,
    image_sizes_by_id: dict[str, tuple[int, int]],
    top_k: int,
) -> np.ndarray:
    selected: list[int] = []
    for query_id in dict.fromkeys(query_ids.tolist()):
        image_rows = np.flatnonzero(query_ids == str(query_id))
        image_size = image_sizes_by_id.get(str(query_id))
        if image_size is None:
            raise KeyError(f"image dimensions missing for query: {query_id}")
        image_width, image_height = image_size
        preferred_rows = image_rows[pose_keep_mask[image_rows]]
        preferred_local = spatially_diverse_detection_indices(
            xy[preferred_rows],
            detector_scores[preferred_rows],
            top_k=min(int(top_k), len(preferred_rows)),
            nms_radius_px=0.0,
            image_width=int(image_width),
            image_height=int(image_height),
            grid_rows=4,
            grid_cols=4,
        )
        image_selected = preferred_rows[preferred_local].tolist()
        missing = min(int(top_k), len(image_rows)) - len(image_selected)
        if missing > 0:
            selected_set = set(int(row) for row in image_selected)
            fallback_rows = np.asarray(
                [int(row) for row in image_rows.tolist() if int(row) not in selected_set],
                dtype=np.int64,
            )
            fallback_local = spatially_diverse_detection_indices(
                xy[fallback_rows],
                detector_scores[fallback_rows],
                top_k=min(int(missing), len(fallback_rows)),
                nms_radius_px=0.0,
                image_width=int(image_width),
                image_height=int(image_height),
                grid_rows=4,
                grid_cols=4,
            )
            image_selected.extend(fallback_rows[fallback_local].tolist())
        selected.extend(image_selected)
    return np.asarray(selected, dtype=np.int64)


def _expected_feature_metadata(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "proposals": Path(args.proposals),
        "detector_query_cache": Path(args.detector_query_cache),
        "support_feature_cache": Path(args.support_feature_cache),
        "support_geometry_index": Path(args.support_geometry_index),
        "projected_landmark_bank": Path(args.projected_landmark_bank),
        "maplet_support_index": Path(args.maplet_support_index),
    }
    if args.query_context_detector_cache:
        paths["query_context_detector_cache"] = Path(args.query_context_detector_cache)
    metadata = {
        "format": "detector_maplet_geometry_features_v1",
        **{f"{name}_sha256": file_sha256_short(path) for name, path in paths.items()},
        "baseline_strategy": str(args.baseline_strategy),
        "candidate_top_k": int(args.candidate_top_k),
        "query_points_per_image": int(args.query_points_per_image),
        "query_neighborhood_radius_px": float(args.query_neighborhood_radius_px),
        "max_query_neighbors": int(args.max_query_neighbors),
        "support_view_count": int(args.support_view_count),
        "support_view_candidate_count": int(args.support_view_candidate_count),
        "support_view_selection": str(args.support_view_selection),
        "support_view_coverage_weight": float(args.support_view_coverage_weight),
        "max_maplet_tracks": int(args.max_maplet_tracks),
        "min_descriptor_similarity": float(args.min_descriptor_similarity),
        "max_tentative_matches": int(args.max_tentative_matches),
        "geometry_threshold_px": float(args.geometry_threshold_px),
        "positive_threshold_px": float(args.positive_threshold_px),
        "query_point_selection": QUERY_POINT_SELECTION_POLICY,
        "query_image_dimensions": "colmap_per_image",
        "query_context_source": (
            "dense_alike_detector_cache" if args.query_context_detector_cache else "global_anchor_detector_cache"
        ),
    }
    if str(args.feature_scope) != "full_probe":
        metadata["feature_scope"] = str(args.feature_scope)
        metadata["static_feature_schema_version"] = int(
            CANDIDATE_MAPLET_STATIC_SCHEMA_VERSION
        )
    if bool(args.inference_only):
        metadata["supervision_mode"] = "none_inference_only"
    return metadata


def _extract_or_load_features(
    args: argparse.Namespace,
    inputs: dict[str, object],
) -> tuple[dict[str, np.ndarray], tuple[str, ...], bool, float]:
    artifact = Path(args.feature_artifact)
    expected = _expected_feature_metadata(args)
    if artifact.exists():
        with np.load(artifact, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            mismatches = {
                key: {"expected": value, "actual": metadata.get(key)}
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"stale maplet geometry feature artifact: {json.dumps(mismatches, sort_keys=True)}")
            payload = {key: np.asarray(data[key]) for key in data.files if key not in {"metadata_json", "feature_names"}}
            names = tuple(str(value) for value in data["feature_names"].tolist())
        return payload, names, True, 0.0

    start = time.time()
    proposals = inputs["proposals"]
    query_cache = inputs["query_cache"]
    landmark_index = inputs["landmark_index"]
    maplet_index = inputs["maplet_index"]
    feature_store = inputs["feature_store"]
    query_ids = inputs["query_ids"]
    query_xy = inputs["query_xy"]
    if not isinstance(proposals, dict) or not isinstance(query_cache, dict):
        raise TypeError("invalid loaded input payload")
    baseline = np.asarray(proposals[f"strategy__{args.baseline_strategy}"], dtype=np.float32)
    coarse = np.asarray(proposals["coarse_scores"], dtype=np.float32)
    candidate_rows = np.asarray(proposals["canonical_bank_row_indices"], dtype=np.int64)
    candidate_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    candidate_residuals = (
        None
        if bool(args.inference_only)
        else np.asarray(proposals["candidate_gt_residuals_px"], dtype=np.float32)
    )
    detector_scores = np.asarray(query_cache["detector_scores"], dtype=np.float32)
    detector_dispersions = np.asarray(query_cache["detector_dispersions"], dtype=np.float32)
    query_descriptors = np.asarray(query_cache["local_descriptors"], dtype=np.float32)
    query_descriptors /= np.maximum(np.linalg.norm(query_descriptors, axis=1, keepdims=True), 1e-8)
    context_cache = inputs.get("query_context_cache")
    context_image_position: dict[str, int] = {}
    context_descriptors = None
    if isinstance(context_cache, dict):
        context_image_position = {
            str(image_id): int(index)
            for index, image_id in enumerate(context_cache["image_ids"].astype(str).tolist())
        }
        context_descriptors = np.asarray(context_cache["local_descriptors"], dtype=np.float32)
        context_descriptors /= np.maximum(
            np.linalg.norm(context_descriptors, axis=1, keepdims=True), 1e-8
        )
    selected_rows = _select_query_rows(
        query_ids=query_ids,
        xy=query_xy,
        detector_scores=detector_scores,
        pose_keep_mask=np.asarray(proposals["pose_keep_mask"], dtype=bool),
        image_sizes_by_id=inputs["query_image_sizes"],
        top_k=int(args.query_points_per_image),
    )
    selected_from_pose_keep = np.asarray(
        proposals["pose_keep_mask"], dtype=bool
    )[selected_rows]
    selected_columns = np.full((len(selected_rows), int(args.candidate_top_k)), -1, dtype=np.int64)
    for local_row, global_row in enumerate(selected_rows.tolist()):
        valid = np.flatnonzero((candidate_rows[global_row] >= 0) & np.isfinite(baseline[global_row]))
        order = valid[np.argsort(-baseline[global_row, valid], kind="stable")]
        kept = order[: int(args.candidate_top_k)]
        selected_columns[local_row, : len(kept)] = kept

    if str(args.feature_scope) == "static_only":
        feature_names = tuple(STATIC_FEATURE_NAMES)
    else:
        _empty, geometry_names = aggregate_maplet_view_evidence(
            np.zeros((0, 19), dtype=np.float32)
        )
        feature_names = tuple(STATIC_FEATURE_NAMES) + tuple(geometry_names)
    features = np.zeros(
        (len(selected_rows), int(args.candidate_top_k), len(feature_names)),
        dtype=np.float32,
    )
    labels = (
        None
        if bool(args.inference_only)
        else np.zeros((len(selected_rows), int(args.candidate_top_k)), dtype=bool)
    )
    valid_edges = selected_columns >= 0
    view_cache: OrderedDict[int, tuple[tuple[str, object], ...]] = OrderedDict()

    rows_by_image: dict[str, np.ndarray] = {}
    for query_id in dict.fromkeys(query_ids.tolist()):
        rows_by_image[str(query_id)] = np.flatnonzero(query_ids == str(query_id))
    previous_query = None
    completed_images = 0
    for local_row, global_row in enumerate(selected_rows.tolist()):
        query_id = str(query_ids[global_row])
        if query_id != previous_query:
            completed_images += 1
            previous_query = query_id
            print(
                json.dumps(
                    {
                        "stage": "maplet_geometry_feature_extraction",
                        "completed_query_images": int(completed_images - 1),
                        "total_query_images": int(len(rows_by_image)),
                        "elapsed_seconds": float(time.time() - start),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if str(args.feature_scope) == "full_probe":
            image_rows = rows_by_image[query_id]
            if isinstance(context_cache, dict) and context_descriptors is not None:
                context_position = context_image_position[query_id]
                context_start = int(context_cache["offsets"][context_position])
                context_end = int(context_cache["offsets"][context_position + 1])
                neighborhood_xy, neighborhood_descriptors, neighborhood_anchor = (
                    select_dense_query_context_neighborhood(
                        anchor_xy=query_xy[global_row],
                        anchor_descriptor=query_descriptors[global_row],
                        context_xy=context_cache["xy"][context_start:context_end],
                        context_descriptors=context_descriptors[context_start:context_end],
                        context_scores=context_cache["detector_scores"][context_start:context_end],
                        radius_px=float(args.query_neighborhood_radius_px),
                        max_points=int(args.max_query_neighbors),
                    )
                )
            else:
                anchor_in_image = int(np.searchsorted(image_rows, global_row))
                neighborhood_local, neighborhood_anchor = _query_neighborhood(
                    query_xy[image_rows],
                    anchor_in_image,
                    radius_px=float(args.query_neighborhood_radius_px),
                    max_points=int(args.max_query_neighbors),
                )
                neighborhood_rows = image_rows[neighborhood_local]
                neighborhood_xy = query_xy[neighborhood_rows]
                neighborhood_descriptors = query_descriptors[neighborhood_rows]
        row_columns = selected_columns[local_row]
        valid_columns = row_columns[row_columns >= 0]
        coarse_best = float(np.max(coarse[global_row, valid_columns]))
        baseline_best = float(np.max(baseline[global_row, valid_columns]))
        baseline_order = valid_columns[np.argsort(-baseline[global_row, valid_columns], kind="stable")]
        baseline_rank_by_column = {int(column): rank for rank, column in enumerate(baseline_order.tolist())}
        for compact_column, proposal_column in enumerate(row_columns.tolist()):
            if proposal_column < 0:
                continue
            bank_row = int(candidate_rows[global_row, proposal_column])
            track_id = int(candidate_tracks[global_row, proposal_column])
            maplet_row = bank_row
            maplet_neighbors = maplet_index.neighbor_track_ids[maplet_row]
            maplet_tracks = np.concatenate(
                [np.asarray([track_id], dtype=np.int64), maplet_neighbors[maplet_neighbors >= 0]]
            )[: int(args.max_maplet_tracks)]
            static = np.asarray(
                [
                    coarse[global_row, proposal_column],
                    baseline[global_row, proposal_column],
                    float(proposal_column) / max(float(candidate_rows.shape[1] - 1), 1.0),
                    float(baseline_rank_by_column[proposal_column]) / max(float(len(valid_columns) - 1), 1.0),
                    coarse_best - coarse[global_row, proposal_column],
                    baseline_best - baseline[global_row, proposal_column],
                    np.clip(detector_scores[global_row], 0.0, 1.0),
                    np.log10(max(float(detector_scores[global_row]), 1e-8)),
                    np.log1p(max(float(detector_dispersions[global_row]), 0.0)),
                    np.log1p(float(landmark_index.observation_counts[bank_row])),
                    np.log1p(max(float(landmark_index.mean_variances[bank_row]), 0.0)),
                    min(max(float(landmark_index.reprojection_errors[bank_row]), 0.0), 10.0),
                    min(max(float(landmark_index.feature_ambiguities[bank_row]), 0.0), 10.0),
                    float(maplet_index.maplets.neighbor_counts[maplet_row]) / max(float(maplet_index.maplets.maplet_k), 1.0),
                    np.log1p(max(float(maplet_index.maplets.context_radius[maplet_row]), 0.0)),
                    np.log1p(max(float(maplet_index.maplets.covisibility_strength[maplet_row]), 0.0)),
                    np.log1p(max(float(maplet_index.maplets.context_feature_variance[maplet_row]), 0.0)),
                ],
                dtype=np.float32,
            )
            if labels is not None:
                assert candidate_residuals is not None
                labels[local_row, compact_column] = bool(
                    candidate_residuals[global_row, proposal_column]
                    <= float(args.positive_threshold_px)
                )
            if str(args.feature_scope) == "static_only":
                features[local_row, compact_column] = static
                continue
            cached_views = view_cache.get(maplet_row)
            if cached_views is None:
                image_indices = maplet_index.support_image_indices[
                    maplet_row, : int(args.support_view_candidate_count)
                ]
                cached_list = []
                for image_index in image_indices.tolist():
                    if int(image_index) < 0:
                        continue
                    support_image_id = maplet_index.support_image_ids[int(image_index)]
                    cached_list.append(
                        (
                            str(support_image_id),
                            feature_store.maplet_view(str(support_image_id), maplet_tracks),
                        )
                    )
                cached_views = tuple(cached_list)
                view_cache[maplet_row] = cached_views
                if len(view_cache) > int(args.maplet_view_cache_size):
                    view_cache.popitem(last=False)
            else:
                view_cache.move_to_end(maplet_row)
            selected_view_indices = select_support_maplet_view_indices(
                [support_view for _image_id, support_view in cached_views],
                query_anchor_descriptor=query_descriptors[global_row],
                anchor_track_id=track_id,
                expected_maplet_track_count=len(maplet_tracks),
                top_k=int(args.support_view_count),
                strategy=str(args.support_view_selection),
                coverage_weight=float(args.support_view_coverage_weight),
            )
            selected_views = tuple(cached_views[int(index)] for index in selected_view_indices.tolist())
            view_features = [
                score_maplet_support_view(
                    query_xy=neighborhood_xy,
                    query_descriptors=neighborhood_descriptors,
                    query_anchor_index=int(neighborhood_anchor),
                    support_view=support_view,
                    anchor_track_id=track_id,
                    expected_maplet_track_count=len(maplet_tracks),
                    min_similarity=float(args.min_descriptor_similarity),
                    max_tentative_matches=int(args.max_tentative_matches),
                    geometry_threshold_px=float(args.geometry_threshold_px),
                )
                for _image_id, support_view in selected_views
            ]
            aggregated, _names = aggregate_maplet_view_evidence(
                np.stack(view_features, axis=0) if view_features else np.zeros((0, 19), dtype=np.float32)
            )
            features[local_row, compact_column] = np.concatenate([static, aggregated])
    payload = {
        "selected_rows": selected_rows,
        "selected_columns": selected_columns,
        "features": features,
        "valid_edges": valid_edges,
        "selected_from_pose_keep": selected_from_pose_keep,
    }
    if labels is not None:
        payload["labels"] = labels
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        artifact,
        **payload,
        feature_names=np.asarray(feature_names, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(expected, sort_keys=True), dtype=np.str_),
    )
    return payload, feature_names, False, float(time.time() - start)


def _compact_candidates(
    proposals: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    selected_columns: np.ndarray,
) -> UniqueTrackCandidateSet:
    valid = selected_columns >= 0
    safe_columns = np.maximum(selected_columns, 0)
    rows = np.take_along_axis(proposals["canonical_bank_row_indices"][selected_rows], safe_columns, axis=1)
    tracks = np.take_along_axis(proposals["candidate_track_ids"][selected_rows], safe_columns, axis=1)
    prototypes = np.take_along_axis(proposals["candidate_prototype_ids"][selected_rows], safe_columns, axis=1)
    scores = np.take_along_axis(proposals["coarse_scores"][selected_rows], safe_columns, axis=1)
    rows[~valid] = -1
    tracks[~valid] = -1
    prototypes[~valid] = -1
    scores[~valid] = -np.inf
    return UniqueTrackCandidateSet(rows, tracks, prototypes, scores)


def _compact_values(
    values: np.ndarray,
    selected_rows: np.ndarray,
    selected_columns: np.ndarray,
) -> np.ndarray:
    valid = selected_columns >= 0
    safe_columns = np.maximum(selected_columns, 0)
    output = np.take_along_axis(np.asarray(values)[selected_rows], safe_columns, axis=1).astype(np.float32)
    output[~valid] = -np.inf
    return output


def _split_masks(query_ids: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    unique = tuple(dict.fromkeys(query_ids.tolist()))
    train_end = int(args.train_query_count)
    validation_end = train_end + int(args.validation_query_count)
    if train_end <= 0 or validation_end >= len(unique):
        raise ValueError("split counts must leave a non-empty test block")
    split = {
        "train": list(unique[:train_end]),
        "validation": list(unique[train_end:validation_end]),
        "test": list(unique[validation_end:]),
    }
    masks = {
        name: np.isin(query_ids, np.asarray(values, dtype=np.str_))
        for name, values in split.items()
    }
    return split, masks


def _identity_metrics(
    *,
    nearest_residuals: np.ndarray,
    candidate_residuals: np.ndarray,
    scores: np.ndarray,
    query_ids: np.ndarray,
    labels: np.ndarray,
    valid_edges: np.ndarray,
) -> dict[str, object]:
    row_has_positive = np.any(labels & valid_edges, axis=1)
    max_score = np.max(np.where(valid_edges, scores, -np.inf), axis=1)
    return {
        "geometry": summarize_ranked_detector_proposal_geometry(
            nearest_landmark_residuals=nearest_residuals,
            candidate_residuals=candidate_residuals,
            candidate_scores=scores,
            query_ids=query_ids.tolist(),
            thresholds_px=(1.0, 2.0, 5.0, 8.0),
            top_ls=(1, 5, 10),
        ),
        "proposal_pool_availability": summarize_detector_proposal_geometry(
            nearest_landmark_residuals=nearest_residuals,
            candidate_residuals=np.where(
                valid_edges, candidate_residuals, np.inf
            ),
            query_ids=query_ids.tolist(),
            thresholds_px=(1.0, 2.0, 5.0, 8.0),
            top_ls=(1, 5, 10, int(candidate_residuals.shape[1])),
        ),
        "pair_positive_average_precision": binary_average_precision(labels[valid_edges], scores[valid_edges]),
        "wrong_pool_rejection_average_precision": binary_average_precision(~row_has_positive, -max_score),
        "row_positive_rate": float(np.mean(row_has_positive)),
    }


def _fit_model(
    features: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[valid])
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=int(max_iter),
        random_state=int(seed),
        solver="lbfgs",
    )
    model.fit(train_features, labels[valid].astype(np.int64))
    return scaler, model


def _predict_model(
    scaler: StandardScaler,
    model: LogisticRegression,
    features: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    output = np.full(valid.shape, -np.inf, dtype=np.float32)
    flat_valid = valid.reshape(-1)
    probabilities = model.predict_proba(scaler.transform(features.reshape(-1, features.shape[-1])[flat_valid]))[:, 1]
    output.reshape(-1)[flat_valid] = probabilities.astype(np.float32)
    return output


def _pose_inputs(
    proposals: dict[str, np.ndarray],
    query_ids: np.ndarray,
    query_xy: np.ndarray,
    selected_rows: np.ndarray,
) -> tuple[list[ColmapTrackObservation], list[str]]:
    nearest_tracks = np.asarray(proposals["nearest_visible_track_ids"], dtype=np.int64)
    observations = [
        ColmapTrackObservation(
            track_id=int(nearest_tracks[row]),
            image_id=str(query_ids[row]),
            point2d_idx=int(row),
            xy=(float(query_xy[row, 0]), float(query_xy[row, 1])),
            xyz=np.zeros((3,), dtype=np.float64),
            track_length=1,
            reprojection_error=0.0,
        )
        for row in selected_rows.tolist()
    ]
    return observations, query_ids[selected_rows].tolist()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.inference_only) and not bool(args.build_only):
        raise ValueError("--inference_only requires --build_only")
    if min(
        int(args.candidate_top_k),
        int(args.query_points_per_image),
        int(args.max_query_neighbors),
        int(args.support_view_count),
        int(args.max_maplet_tracks),
    ) <= 0:
        raise ValueError("probe counts must be positive")
    if int(args.support_view_candidate_count) < int(args.support_view_count):
        raise ValueError("support_view_candidate_count must be at least support_view_count")
    np.random.seed(int(args.seed))
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(args)
    feature_payload, feature_names, cache_hit, feature_seconds = _extract_or_load_features(args, inputs)
    if bool(args.build_only):
        artifact_path = Path(args.feature_artifact)
        build_summary = {
            "stage": "candidate_maplet_feature_artifact_build",
            "protocol": {
                "feature_scope": str(args.feature_scope),
                "candidate_top_k": int(args.candidate_top_k),
                "query_points_per_image": int(args.query_points_per_image),
                "image_retrieval": False,
                "submap": False,
                "render": False,
                "supervision_loaded": not bool(args.inference_only),
            },
            "artifact": {
                "path": str(artifact_path),
                "sha256": file_sha256_short(artifact_path),
                "cache_hit": bool(cache_hit),
                "build_seconds": float(feature_seconds),
                "selected_query_point_count": int(len(feature_payload["selected_rows"])),
                "pose_keep_priority_point_count": int(
                    np.sum(feature_payload["selected_from_pose_keep"])
                ),
                "full_detector_fallback_point_count": int(
                    np.sum(~np.asarray(feature_payload["selected_from_pose_keep"], dtype=bool))
                ),
                "candidate_edge_count": int(np.sum(feature_payload["valid_edges"])),
                "feature_count": int(len(feature_names)),
                "feature_names": list(feature_names),
                "positive_edge_rate": (
                    None
                    if bool(args.inference_only)
                    else float(
                        np.mean(
                            np.asarray(feature_payload["labels"], dtype=bool)[
                                np.asarray(
                                    feature_payload["valid_edges"], dtype=bool
                                )
                            ]
                        )
                    )
                ),
            },
            "input_manifest": _expected_feature_metadata(args),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(build_summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(build_summary, indent=2, sort_keys=True))
        return
    proposals = inputs["proposals"]
    landmark_index = inputs["landmark_index"]
    query_ids_all = inputs["query_ids"]
    query_xy_all = inputs["query_xy"]
    if not isinstance(proposals, dict):
        raise TypeError("invalid proposal payload")
    selected_rows = np.asarray(feature_payload["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(feature_payload["selected_columns"], dtype=np.int64)
    features = np.asarray(feature_payload["features"], dtype=np.float32)
    labels = np.asarray(feature_payload["labels"], dtype=bool)
    valid_edges = np.asarray(feature_payload["valid_edges"], dtype=bool)
    candidates = _compact_candidates(proposals, selected_rows, selected_columns)
    baseline_scores = _compact_values(
        proposals[f"strategy__{args.baseline_strategy}"], selected_rows, selected_columns
    )
    candidate_residuals = _compact_values(
        proposals["candidate_gt_residuals_px"], selected_rows, selected_columns
    )
    candidate_residuals[~valid_edges] = np.inf
    nearest_residuals = np.asarray(proposals["nearest_visible_residuals_px"], dtype=np.float32)[selected_rows]
    selected_query_ids = np.asarray(query_ids_all, dtype=np.str_)[selected_rows]
    split, split_masks = _split_masks(selected_query_ids, args)

    cameras = read_colmap_cameras_binary(Path(args.colmap_model_dir) / "cameras.bin")
    images = read_colmap_images_binary(Path(args.colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    pose_observations, pose_query_ids = _pose_inputs(
        proposals, query_ids_all, query_xy_all, selected_rows
    )

    def subset_pose(name: str, mask: np.ndarray, scores: np.ndarray) -> dict[str, object]:
        rows = np.flatnonzero(mask)
        subset = UniqueTrackCandidateSet(
            candidates.bank_row_indices[rows],
            candidates.track_ids[rows],
            candidates.prototype_ids[rows],
            candidates.coarse_scores[rows],
        )
        summary, pose_rows = _evaluate_pose_strategy(
            strategy=str(name),
            scores=scores[rows],
            candidates=subset,
            query_observations=[pose_observations[int(row)] for row in rows.tolist()],
            query_ids=[pose_query_ids[int(row)] for row in rows.tolist()],
            landmark_index=landmark_index,
            cameras=cameras,
            images_by_name=images_by_name,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
        )
        (output_dir / f"pose_rows_{name}.json").write_text(
            json.dumps(pose_rows, indent=2, sort_keys=True) + "\n"
        )
        return summary

    baseline = {}
    for split_name, mask in split_masks.items():
        baseline[split_name] = {
            "identity": _identity_metrics(
                nearest_residuals=nearest_residuals[mask],
                candidate_residuals=candidate_residuals[mask],
                scores=baseline_scores[mask],
                query_ids=selected_query_ids[mask],
                labels=labels[mask],
                valid_edges=valid_edges[mask],
            ),
            "pose": subset_pose(f"baseline_{split_name}", mask, baseline_scores),
        }

    static_dim = len(STATIC_FEATURE_NAMES)
    groups = {
        "static": np.arange(static_dim, dtype=np.int64),
        "static_plus_maplet_geometry": np.arange(features.shape[-1], dtype=np.int64),
    }
    group_results: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    for group_name, feature_columns in groups.items():
        group_features = features[:, :, feature_columns]
        trials = []
        fitted = []
        train_valid = valid_edges & split_masks["train"][:, None]
        for c_value in args.c_values:
            scaler, model = _fit_model(
                group_features,
                labels,
                train_valid,
                c_value=float(c_value),
                max_iter=int(args.max_iter),
                seed=int(args.seed),
            )
            scores = _predict_model(scaler, model, group_features, valid_edges)
            validation_identity = _identity_metrics(
                nearest_residuals=nearest_residuals[split_masks["validation"]],
                candidate_residuals=candidate_residuals[split_masks["validation"]],
                scores=scores[split_masks["validation"]],
                query_ids=selected_query_ids[split_masks["validation"]],
                labels=labels[split_masks["validation"]],
                valid_edges=valid_edges[split_masks["validation"]],
            )
            validation_pose = subset_pose(
                f"{group_name}_c{float(c_value):g}_validation",
                split_masks["validation"],
                scores,
            )
            recall1 = float(
                validation_identity["geometry"]["thresholds_px"]["2"]["recall_at_1_given_mappable"]
            )
            trial = {
                "c": float(c_value),
                "validation_identity": validation_identity,
                "validation_pose": validation_pose,
                "passes_pose_gate_vs_baseline": _pose_gate(validation_pose, baseline["validation"]["pose"]),
                "selection_key": [
                    float(_pose_gate(validation_pose, baseline["validation"]["pose"])),
                    recall1,
                    float(validation_identity["pair_positive_average_precision"]),
                    -float(validation_pose["median_translation_m_success"]),
                ],
            }
            trials.append(trial)
            fitted.append((scaler, model, scores))
        best_index = max(range(len(trials)), key=lambda index: tuple(trials[index]["selection_key"]))
        best_trial = trials[best_index]
        scaler, model, scores = fitted[best_index]
        predictions[group_name] = scores
        split_results = {}
        for split_name, mask in split_masks.items():
            identity = _identity_metrics(
                nearest_residuals=nearest_residuals[mask],
                candidate_residuals=candidate_residuals[mask],
                scores=scores[mask],
                query_ids=selected_query_ids[mask],
                labels=labels[mask],
                valid_edges=valid_edges[mask],
            )
            pose = (
                best_trial["validation_pose"]
                if split_name == "validation"
                else subset_pose(f"{group_name}_{split_name}", mask, scores)
            )
            split_results[split_name] = {
                "identity": identity,
                "pose": pose,
                "passes_pose_gate_vs_baseline": _pose_gate(pose, baseline[split_name]["pose"]),
            }
        group_results[group_name] = {
            "feature_count": int(len(feature_columns)),
            "chosen_c": float(best_trial["c"]),
            "trials": trials,
            "splits": split_results,
            "model": {
                "coef": model.coef_[0].astype(float).tolist(),
                "intercept": model.intercept_.astype(float).tolist(),
                "scaler_mean": scaler.mean_.astype(float).tolist(),
                "scaler_scale": scaler.scale_.astype(float).tolist(),
            },
        }

    full = group_results["static_plus_maplet_geometry"]
    static = group_results["static"]
    full_test = full["splits"]["test"]
    static_test = static["splits"]["test"]
    maplet_incremental_gate = _pose_gate(full_test["pose"], static_test["pose"])
    selective_results: dict[str, object] = {}
    for selective_name, reference_scores, reference_summary in (
        ("geometry_over_baseline", baseline_scores, baseline),
        ("geometry_over_static", predictions["static"], static["splits"]),
    ):
        trials = []
        resolved_trials = []
        for threshold in args.switch_margin_thresholds:
            _selected, resolved, switched, margins = selective_switch_scores(
                predictions["static_plus_maplet_geometry"],
                reference_scores,
                margin_threshold=float(threshold),
                valid_mask=valid_edges,
            )
            validation_identity = _identity_metrics(
                nearest_residuals=nearest_residuals[split_masks["validation"]],
                candidate_residuals=candidate_residuals[split_masks["validation"]],
                scores=resolved[split_masks["validation"]],
                query_ids=selected_query_ids[split_masks["validation"]],
                labels=labels[split_masks["validation"]],
                valid_edges=valid_edges[split_masks["validation"]],
            )
            validation_pose = subset_pose(
                f"{selective_name}_margin{float(threshold):g}_validation",
                split_masks["validation"],
                resolved,
            )
            trial = {
                "margin_threshold": float(threshold),
                "validation_switch_count": int(np.sum(switched[split_masks["validation"]])),
                "validation_margin_median": float(
                    np.median(margins[split_masks["validation"]][np.isfinite(margins[split_masks["validation"]])])
                ),
                "validation_identity": validation_identity,
                "validation_pose": validation_pose,
                "passes_pose_gate": _pose_gate(
                    validation_pose,
                    reference_summary["validation"]["pose"],
                ),
            }
            trials.append(trial)
            resolved_trials.append(resolved)
        eligible = [index for index, trial in enumerate(trials) if bool(trial["passes_pose_gate"])]
        if eligible:
            chosen_index = max(
                eligible,
                key=lambda index: (
                    float(trials[index]["validation_pose"]["recall_10cm_5deg"]),
                    float(trials[index]["validation_pose"]["recall_5cm_5deg"]),
                    -float(trials[index]["validation_pose"]["median_translation_m_success"]),
                    -float(trials[index]["validation_pose"]["p90_translation_m_success"]),
                    -float(trials[index]["margin_threshold"]),
                ),
            )
            chosen_threshold = float(trials[chosen_index]["margin_threshold"])
            resolved = resolved_trials[chosen_index]
            validation = {
                "identity": trials[chosen_index]["validation_identity"],
                "pose": trials[chosen_index]["validation_pose"],
                "switch_count": int(trials[chosen_index]["validation_switch_count"]),
                "passes_pose_gate": True,
            }
            test_identity = _identity_metrics(
                nearest_residuals=nearest_residuals[split_masks["test"]],
                candidate_residuals=candidate_residuals[split_masks["test"]],
                scores=resolved[split_masks["test"]],
                query_ids=selected_query_ids[split_masks["test"]],
                labels=labels[split_masks["test"]],
                valid_edges=valid_edges[split_masks["test"]],
            )
            test_pose = subset_pose(
                f"{selective_name}_test",
                split_masks["test"],
                resolved,
            )
            _selected, _resolved, switched, _margins = selective_switch_scores(
                predictions["static_plus_maplet_geometry"],
                reference_scores,
                margin_threshold=chosen_threshold,
                valid_mask=valid_edges,
            )
            test = {
                "identity": test_identity,
                "pose": test_pose,
                "switch_count": int(np.sum(switched[split_masks["test"]])),
                "passes_pose_gate": _pose_gate(test_pose, reference_summary["test"]["pose"]),
            }
        else:
            chosen_index = None
            chosen_threshold = None
            validation = {
                **reference_summary["validation"],
                "switch_count": 0,
                "passes_pose_gate": False,
            }
            test = {
                **reference_summary["test"],
                "switch_count": 0,
                "passes_pose_gate": False,
            }
        selective_results[selective_name] = {
            "reference": "baseline" if selective_name.endswith("baseline") else "static",
            "trials": trials,
            "chosen_trial_index": chosen_index,
            "chosen_margin_threshold": chosen_threshold,
            "validation": validation,
            "test": test,
            "promote": bool(
                validation["passes_pose_gate"] and test["passes_pose_gate"]
            ),
        }
    np.savez(
        output_dir / "candidate_scores.npz",
        selected_rows=selected_rows,
        selected_columns=selected_columns,
        baseline_scores=baseline_scores,
        static_scores=predictions["static"],
        static_plus_maplet_geometry_scores=predictions["static_plus_maplet_geometry"],
        labels=labels,
        valid_edges=valid_edges,
        feature_names=np.asarray(feature_names, dtype=np.str_),
    )
    summary = {
        "stage": "detector_candidate_conditioned_maplet_geometry_probe",
        "protocol": {
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "supervision_loaded": True,
            "query_pose_used_by_features": False,
            "set_valued_positive_threshold_px": float(args.positive_threshold_px),
            "global_proposal_source": "RADIO top-L with ALIKE per-track support baseline",
            "support_source": "real SfM observations in candidate maplet support views",
            "support_view_selection": str(args.support_view_selection),
            "support_view_candidate_count": int(args.support_view_candidate_count),
            "support_view_count": int(args.support_view_count),
            "query_context_source": (
                "dense_alike_detector_cache"
                if args.query_context_detector_cache
                else "global_anchor_detector_cache"
            ),
        },
        "split": {"strategy": "contiguous_temporal_blocks_v1", **split},
        "feature_artifact": {
            "path": str(args.feature_artifact),
            "cache_hit": bool(cache_hit),
            "extraction_seconds": float(feature_seconds),
            "selected_query_point_count": int(len(selected_rows)),
            "candidate_edge_count": int(np.sum(valid_edges)),
            "feature_count": int(features.shape[-1]),
            "feature_names": list(feature_names),
        },
        "baseline": baseline,
        "models": group_results,
        "selective_switch": selective_results,
        "gate": {
            "static_plus_geometry_vs_baseline_validation": bool(full["splits"]["validation"]["passes_pose_gate_vs_baseline"]),
            "static_plus_geometry_vs_baseline_test": bool(full["splits"]["test"]["passes_pose_gate_vs_baseline"]),
            "maplet_geometry_incremental_vs_static_test": bool(maplet_incremental_gate),
            "selective_geometry_over_baseline": bool(
                selective_results["geometry_over_baseline"]["promote"]
            ),
            "selective_geometry_over_static": bool(
                selective_results["geometry_over_static"]["promote"]
            ),
            "promote_to_assignment_training": bool(
                selective_results["geometry_over_baseline"]["promote"]
            ),
        },
        "runtime_seconds": float(time.time() - start),
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "candidate_scores": str(output_dir / "candidate_scores.npz"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
