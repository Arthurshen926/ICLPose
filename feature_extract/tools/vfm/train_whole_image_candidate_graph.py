"""Train a whole-image candidate/null graph on frozen P23 evidence."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.local_maplet_matching import (
    LocalMapletSupportIndex,
    load_local_maplet_support_index_npz,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.whole_image_candidate_graph import (
    WholeImageCandidateGraph,
    WholeImageLatentCandidateGraph,
    set_identity_nll,
    system_hard_negative_margin_loss,
)
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


RELATION_FEATURE_NAMES = (
    "query_delta_x",
    "query_delta_y",
    "query_distance",
    "track_direction_x",
    "track_direction_y",
    "track_direction_z",
    "track_distance_log1p",
    "same_physical_track",
    "source_maplet_contains_target",
    "target_maplet_contains_source",
    "mutual_maplet_relation",
    "source_covisibility_strength",
    "target_covisibility_strength",
)

POSE_RELATION_FEATURE_NAMES = (
    "initial_pose_available",
    "source_in_front",
    "target_in_front",
    "source_reprojection_delta_x",
    "source_reprojection_delta_y",
    "source_reprojection_distance_log1p",
    "target_reprojection_delta_x",
    "target_reprojection_delta_y",
    "target_reprojection_distance_log1p",
    "projected_track_delta_x",
    "projected_track_delta_y",
    "projected_track_distance",
    "observed_projected_delta_error_x",
    "observed_projected_delta_error_y",
    "observed_projected_delta_error",
)


def relation_feature_names(*, pose_conditioned: bool) -> tuple[str, ...]:
    return RELATION_FEATURE_NAMES + (
        POSE_RELATION_FEATURE_NAMES if bool(pose_conditioned) else ()
    )


def _cross_query_xyz_neighbors(
    xyz: np.ndarray,
    *,
    candidate_count: int,
    neighbor_k: int,
) -> np.ndarray:
    """Select 3D-near candidates from different query-token groups."""

    values = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    total = int(len(values))
    k = int(neighbor_k)
    if total <= int(candidate_count) or k <= 0:
        raise ValueError("relation graph requires multiple query groups")
    search_k = min(total, max(k + int(candidate_count) + 1, 4 * k))
    tree = cKDTree(values)
    _distance, candidates = tree.query(values, k=search_k)
    candidates = np.asarray(candidates, dtype=np.int64).reshape(total, search_k)
    output = np.empty((total, k), dtype=np.int64)
    for source in range(total):
        source_group = int(source // int(candidate_count))
        accepted = [
            int(target)
            for target in candidates[source].tolist()
            if int(target // int(candidate_count)) != source_group
        ]
        if len(accepted) < k:
            _all_distance, all_candidates = tree.query(values[source], k=total)
            accepted = [
                int(target)
                for target in np.asarray(all_candidates).reshape(-1).tolist()
                if int(target // int(candidate_count)) != source_group
            ]
        if len(accepted) < k:
            raise ValueError("not enough cross-query candidate neighbors")
        output[source] = np.asarray(accepted[:k], dtype=np.int64)
    return output


def build_explicit_relation_graph(
    query_xy_normalized: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_bank_rows: np.ndarray,
    candidate_track_ids: np.ndarray,
    maplet_index: LocalMapletSupportIndex,
    *,
    neighbor_k: int,
    projected_xy_normalized: np.ndarray | None = None,
    candidate_in_front: np.ndarray | None = None,
    initial_pose_available: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sparse no-pose query/track/maplet relation factors."""

    xy = np.asarray(query_xy_normalized, dtype=np.float32)
    xyz = np.asarray(candidate_xyz, dtype=np.float32)
    bank_rows = np.asarray(candidate_bank_rows, dtype=np.int64)
    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    if xyz.ndim != 4 or xyz.shape[3] != 3:
        raise ValueError("candidate XYZ must have shape (I, Q, L, 3)")
    image_count, query_count, candidate_count, _ = xyz.shape
    if xy.shape != (image_count, query_count, 2):
        raise ValueError("relation query coordinates are not aligned")
    if bank_rows.shape != xyz.shape[:3] or tracks.shape != bank_rows.shape:
        raise ValueError("relation candidate identities are not aligned")
    if not np.array_equal(
        np.asarray(maplet_index.anchor_track_ids, dtype=np.int64)[bank_rows], tracks
    ):
        raise ValueError("maplet rows and candidate tracks are not aligned")
    pose_conditioned = projected_xy_normalized is not None
    if pose_conditioned != (candidate_in_front is not None):
        raise ValueError("pose-conditioned relations require projections and depths")
    if pose_conditioned:
        projected_xy = np.asarray(projected_xy_normalized, dtype=np.float32)
        in_front = np.asarray(candidate_in_front, dtype=bool)
        if projected_xy.shape != (*xyz.shape[:3], 2):
            raise ValueError("pose-conditioned projections are not aligned")
        if in_front.shape != xyz.shape[:3]:
            raise ValueError("pose-conditioned depth mask is not aligned")
        if initial_pose_available is None:
            pose_available = np.ones((image_count,), dtype=bool)
        else:
            pose_available = np.asarray(initial_pose_available, dtype=bool).reshape(-1)
            if pose_available.shape != (image_count,):
                raise ValueError("initial pose availability is not aligned")
    else:
        if initial_pose_available is not None:
            raise ValueError("pose availability cannot be supplied without projections")
        projected_xy = None
        in_front = None
        pose_available = np.zeros((image_count,), dtype=bool)
    feature_names = relation_feature_names(pose_conditioned=pose_conditioned)

    total = query_count * candidate_count
    neighbors = np.empty(
        (image_count, query_count, candidate_count, int(neighbor_k)),
        dtype=np.int64,
    )
    relation = np.empty(
        (*neighbors.shape, len(feature_names)), dtype=np.float32
    )
    covisibility = np.asarray(
        maplet_index.maplets.covisibility_strength, dtype=np.float32
    )
    positive_covisibility = covisibility[np.isfinite(covisibility) & (covisibility > 0.0)]
    covisibility_scale = (
        max(float(np.percentile(positive_covisibility, 90.0)), 1e-6)
        if positive_covisibility.size
        else 1.0
    )
    normalized_covisibility = np.clip(
        covisibility / covisibility_scale, 0.0, 1.0
    )

    for image_index in range(image_count):
        flat_xyz = xyz[image_index].reshape(total, 3)
        flat_rows = bank_rows[image_index].reshape(total)
        flat_tracks = tracks[image_index].reshape(total)
        flat_neighbors = _cross_query_xyz_neighbors(
            flat_xyz,
            candidate_count=candidate_count,
            neighbor_k=int(neighbor_k),
        )
        neighbors[image_index] = flat_neighbors.reshape(
            query_count, candidate_count, int(neighbor_k)
        )

        source_group = np.arange(total, dtype=np.int64) // candidate_count
        target_group = flat_neighbors // candidate_count
        query_delta = xy[image_index, target_group] - xy[image_index, source_group, None]
        query_distance = np.linalg.norm(query_delta, axis=2, keepdims=True)
        track_delta = flat_xyz[flat_neighbors] - flat_xyz[:, None, :]
        track_distance = np.linalg.norm(track_delta, axis=2, keepdims=True)
        track_direction = track_delta / np.maximum(track_distance, 1e-8)
        same_track = flat_tracks[flat_neighbors] == flat_tracks[:, None]

        source_maplets = maplet_index.neighbor_track_ids[flat_rows]
        target_tracks = flat_tracks[flat_neighbors]
        source_contains = np.any(
            source_maplets[:, None, :] == target_tracks[:, :, None], axis=2
        )
        target_maplets = maplet_index.neighbor_track_ids[flat_rows[flat_neighbors]]
        target_contains = np.any(
            target_maplets == flat_tracks[:, None, None], axis=2
        )
        source_covisibility = normalized_covisibility[flat_rows][:, None]
        target_covisibility = normalized_covisibility[flat_rows[flat_neighbors]]
        feature_blocks = [
            query_delta,
            query_distance,
            track_direction,
            np.log1p(track_distance),
            same_track[:, :, None].astype(np.float32),
            source_contains[:, :, None].astype(np.float32),
            target_contains[:, :, None].astype(np.float32),
            (source_contains & target_contains)[:, :, None].astype(np.float32),
            np.broadcast_to(source_covisibility[:, :, None], (*same_track.shape, 1)),
            target_covisibility[:, :, None],
        ]
        if pose_conditioned:
            assert projected_xy is not None and in_front is not None
            flat_projected = projected_xy[image_index].reshape(total, 2)
            flat_in_front = in_front[image_index].reshape(total)
            source_projection_delta = (
                flat_projected - xy[image_index, source_group]
            )
            target_projection_delta = (
                flat_projected[flat_neighbors] - xy[image_index, target_group]
            )
            projected_delta = (
                flat_projected[flat_neighbors] - flat_projected[:, None, :]
            )
            delta_error = query_delta - projected_delta
            edge_valid = (
                pose_available[image_index]
                & flat_in_front[:, None]
                & flat_in_front[flat_neighbors]
                & np.all(np.isfinite(source_projection_delta), axis=1)[:, None]
                & np.all(np.isfinite(target_projection_delta), axis=2)
                & np.all(np.isfinite(projected_delta), axis=2)
            )
            source_delta = np.broadcast_to(
                source_projection_delta[:, None, :], (*same_track.shape, 2)
            ).copy()
            source_distance = np.broadcast_to(
                np.linalg.norm(source_projection_delta, axis=1)[:, None, None],
                (*same_track.shape, 1),
            ).copy()
            target_distance = np.linalg.norm(
                target_projection_delta, axis=2, keepdims=True
            )
            projected_distance = np.linalg.norm(
                projected_delta, axis=2, keepdims=True
            )
            delta_error_distance = np.linalg.norm(
                delta_error, axis=2, keepdims=True
            )
            for value in (
                source_delta,
                source_distance,
                target_projection_delta,
                target_distance,
                projected_delta,
                projected_distance,
                delta_error,
                delta_error_distance,
            ):
                value[~edge_valid] = 0.0
            feature_blocks.extend(
                [
                    np.full(
                        (*same_track.shape, 1),
                        float(pose_available[image_index]),
                        dtype=np.float32,
                    ),
                    np.broadcast_to(
                        flat_in_front[:, None, None], (*same_track.shape, 1)
                    ).astype(np.float32),
                    flat_in_front[flat_neighbors, None].astype(np.float32),
                    source_delta,
                    np.log1p(source_distance),
                    target_projection_delta,
                    np.log1p(target_distance),
                    projected_delta,
                    projected_distance,
                    delta_error,
                    delta_error_distance,
                ]
            )
        features = np.concatenate(feature_blocks, axis=2)
        relation[image_index] = features.reshape(
            query_count,
            candidate_count,
            int(neighbor_k),
            len(feature_names),
        )
    return neighbors, relation


def _load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files if k != "metadata_json"}
        metadata = {} if "metadata_json" not in z.files else json.loads(str(z["metadata_json"].item()))
    return arrays, metadata


def _load_initial_pose_lookup(
    path: Path,
    *,
    strategy: str,
    expected_image_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray | None], dict[str, object]]:
    """Load only success/pose fields from a frozen pose-row artifact."""

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("initial pose rows must be split-indexed JSON")
    lookup: dict[str, np.ndarray | None] = {}
    source_splits: list[str] = []
    for split_name in ("train", "validation", "test"):
        split_payload = payload.get(split_name)
        if not isinstance(split_payload, dict) or strategy not in split_payload:
            raise ValueError(
                f"initial pose rows are missing {split_name!r}/{strategy!r}"
            )
        rows = split_payload[strategy]
        if not isinstance(rows, list):
            raise ValueError("initial pose strategy rows must be a list")
        source_splits.append(split_name)
        for row in rows:
            if not isinstance(row, dict) or "query_id" not in row:
                raise ValueError("initial pose row is missing query_id")
            query_id = str(row["query_id"])
            if query_id in lookup:
                raise ValueError(f"duplicate initial pose row for {query_id!r}")
            pose = row.get("pose_w2c")
            if not bool(row.get("success", False)) or pose is None:
                lookup[query_id] = None
                continue
            matrix = np.asarray(pose, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"invalid initial pose matrix for {query_id!r}")
            lookup[query_id] = matrix

    expected = {str(value) for value in np.asarray(expected_image_ids).reshape(-1)}
    actual = set(lookup)
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(
            "initial pose rows must cover the candidate image set exactly: "
            f"missing={missing}, extra={extra}"
        )
    return lookup, {
        "strategy": str(strategy),
        "source_splits": source_splits,
        "fields_consumed": ["query_id", "success", "pose_w2c"],
        "gt_error_fields_consumed": False,
        "successful_pose_count": int(sum(value is not None for value in lookup.values())),
        "image_count": int(len(lookup)),
    }


def _project_candidates_from_initial_pose(
    candidate_xyz: np.ndarray,
    image_ids: np.ndarray,
    pose_lookup: dict[str, np.ndarray | None],
    *,
    colmap_model_dir: Path,
    image_width: float,
    image_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project every candidate with a frozen, inference-produced initial pose."""

    cameras = read_colmap_cameras_binary(Path(colmap_model_dir) / "cameras.bin")
    images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
    image_by_name = {}
    for record in images.values():
        if record.image_name in image_by_name:
            raise ValueError(f"duplicate COLMAP image name {record.image_name!r}")
        image_by_name[record.image_name] = record

    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    if xyz.ndim != 4 or xyz.shape[3] != 3:
        raise ValueError("candidate XYZ grid is not aligned")
    projected = np.zeros((*xyz.shape[:3], 2), dtype=np.float32)
    in_front = np.zeros(xyz.shape[:3], dtype=bool)
    available = np.zeros((xyz.shape[0],), dtype=bool)
    camera_ids = np.full((xyz.shape[0],), -1, dtype=np.int64)
    normalization = np.asarray(
        [float(image_width) - 1.0, float(image_height) - 1.0], dtype=np.float64
    )
    for image_index, value in enumerate(np.asarray(image_ids).reshape(-1)):
        query_id = str(value)
        record = image_by_name.get(query_id)
        if record is None:
            raise ValueError(f"initial-pose query is absent from COLMAP: {query_id!r}")
        camera = cameras.get(int(record.camera_id))
        if camera is None:
            raise ValueError(f"COLMAP camera is absent for {query_id!r}")
        if (
            int(camera.width) != int(round(float(image_width)))
            or int(camera.height) != int(round(float(image_height)))
        ):
            raise ValueError("declared image dimensions disagree with COLMAP camera")
        camera_ids[image_index] = int(camera.camera_id)
        pose = pose_lookup[query_id]
        if pose is None:
            continue
        points = xyz[image_index].reshape(-1, 3)
        pixels = project_world_to_image(points, pose, camera)
        camera_xyz = points @ pose[:3, :3].T + pose[:3, 3][None]
        finite = np.all(np.isfinite(pixels), axis=1) & np.isfinite(camera_xyz[:, 2])
        front = finite & (camera_xyz[:, 2] > 1e-6)
        normalized = np.zeros_like(pixels)
        normalized[finite] = pixels[finite] / normalization[None]
        projected[image_index] = normalized.reshape(*xyz.shape[1:3], 2)
        in_front[image_index] = front.reshape(xyz.shape[1:3])
        available[image_index] = True
    return projected, in_front, available, camera_ids


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int(np.count_nonzero(labels))
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / positive_count)


def _metrics(candidate_logits, null_logits, labels, prior_probability=None):
    with torch.no_grad():
        probs = torch.softmax(torch.cat([candidate_logits, null_logits.unsqueeze(2)], 2), 2)
        candidate = probs[:, :, :-1]
        null = probs[:, :, -1]
        any_positive = torch.any(labels, dim=2)
        correct_mass = torch.sum(candidate * labels.float(), dim=2)
        target_mass = torch.where(any_positive, correct_mass, null)
        order = torch.argsort(candidate, dim=2, descending=True)
        ranked = torch.gather(labels, 2, order)
        top1 = ranked[:, :, 0]
        state_unknown = null >= torch.max(candidate, dim=2).values
        state_correct = torch.where(any_positive, top1 & ~state_unknown, state_unknown)
        result = {
            "identity_nll": float(torch.mean(-torch.log(target_mass.clamp_min(1e-12))).cpu()),
            "candidate_top1_correct_rate": float(torch.mean(top1.float()).cpu()),
            "state_top1_correct_rate": float(torch.mean(state_correct.float()).cpu()),
            "unknown_top1_rate": float(torch.mean(state_unknown.float()).cpu()),
            "correct_available_rate": float(torch.mean(any_positive.float()).cpu()),
        }
        available_count = int(torch.count_nonzero(any_positive).cpu())
        for rank in (1, 5, 10, 20):
            capped = min(rank, candidate.shape[2])
            recalled = torch.any(ranked[:, :, :capped], dim=2)
            result[f"candidate_recall_at_{rank}_given_available"] = (
                float(torch.mean(recalled[any_positive].float()).cpu())
                if available_count
                else float("nan")
            )
        result["null_average_precision"] = _average_precision(
            null.cpu().numpy(), (~any_positive).cpu().numpy()
        )
        if prior_probability is not None:
            prior = torch.as_tensor(
                prior_probability,
                device=candidate_logits.device,
                dtype=candidate_logits.dtype,
            )
            if prior.shape != candidate.shape:
                raise ValueError("metric prior is not aligned")
            prior_top1 = torch.argmax(prior, dim=2, keepdim=True)
            prior_correct = torch.gather(labels, 2, prior_top1)[:, :, 0]
            predicted_top1 = torch.argmax(candidate, dim=2, keepdim=True)
            predicted_correct = torch.gather(labels, 2, predicted_top1)[:, :, 0]
            rescue = any_positive & ~prior_correct
            preserve = any_positive & prior_correct
            result["rank2_20_rescue_rate"] = (
                float(torch.mean(predicted_correct[rescue].float()).cpu())
                if bool(torch.any(rescue))
                else float("nan")
            )
            result["wrong_switch_rate"] = (
                float(torch.mean((~predicted_correct[preserve]).float()).cpu())
                if bool(torch.any(preserve))
                else float("nan")
            )
        return result


def _checkpoint_selection_key(metrics: dict, mode: str) -> tuple[float, ...]:
    """Return a lexicographic key where larger always means a better checkpoint."""

    nll = float(metrics["identity_nll"])
    top1 = float(metrics["candidate_top1_correct_rate"])
    rescue = float(metrics.get("rank2_20_rescue_rate", float("nan")))
    wrong_switch = float(metrics.get("wrong_switch_rate", float("nan")))
    if mode == "identity_nll":
        return (-nll,)
    if mode == "candidate_top1_correct_rate":
        return (top1, -nll)
    if mode == "net_rescue":
        net_rescue = rescue - wrong_switch
        if not np.isfinite(net_rescue):
            net_rescue = float("-inf")
        return (net_rescue, top1, -nll)
    raise ValueError(f"unsupported checkpoint selection metric: {mode}")


def _require_equal(name: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _validate_manifests(
    args,
    candidate_meta: dict,
    score_meta: dict,
    landmark_meta: dict,
) -> dict:
    hashes = {
        "candidate_artifact_sha256": file_sha256_short(Path(args.candidate_artifact)),
        "score_artifact_sha256": file_sha256_short(Path(args.score_artifact)),
        "proposals_sha256": file_sha256_short(Path(args.proposals)),
        "landmark_bank_sha256": file_sha256_short(Path(args.landmark_bank)),
        "split_json_sha256": file_sha256_short(Path(args.split_json)),
    }
    if args.maplet_support_index is not None:
        hashes["maplet_support_index_sha256"] = file_sha256_short(
            Path(args.maplet_support_index)
        )
    _require_equal(
        "candidate artifact format",
        candidate_meta.get("format"),
        "detector_maplet_geometry_features_v1",
    )
    _require_equal(
        "score artifact format",
        score_meta.get("format"),
        "candidate_maplet_ensemble_scores_v2",
    )
    score_manifest = dict(score_meta.get("data_manifest") or {})
    required_splits = {"train", "validation", "test"}
    exported_splits = set(score_meta.get("prediction_splits") or ())
    if not required_splits.issubset(exported_splits):
        raise ValueError(
            "score artifact must contain frozen train/validation/test predictions; "
            f"found {sorted(exported_splits)}"
        )
    for metadata, label in (
        (candidate_meta, "candidate artifact"),
        (score_manifest, "score artifact"),
    ):
        _require_equal(
            f"{label} proposals hash",
            metadata.get("proposals_sha256"),
            hashes["proposals_sha256"],
        )
        _require_equal(
            f"{label} landmark bank hash",
            metadata.get("projected_landmark_bank_sha256"),
            hashes["landmark_bank_sha256"],
        )
    _require_equal(
        "score feature artifact hash",
        score_manifest.get("feature_artifact_sha256"),
        hashes["candidate_artifact_sha256"],
    )
    _require_equal(
        "candidate/score top-k",
        candidate_meta.get("candidate_top_k"),
        score_manifest.get("candidate_top_k"),
    )
    if not landmark_meta.get("descriptor_space_id"):
        raise ValueError("landmark bank is missing descriptor_space_id")
    return hashes


def _validate_latent_manifest(
    path: Path,
    latent_meta: dict,
    score_meta: dict,
) -> str:
    _require_equal(
        "candidate latent format",
        latent_meta.get("format"),
        "candidate_maplet_view_latents_v1",
    )
    _require_equal(
        "candidate latent data manifest",
        latent_meta.get("data_manifest"),
        score_meta.get("data_manifest"),
    )
    _require_equal(
        "candidate latent prediction splits",
        set(latent_meta.get("prediction_splits") or ()),
        {"train", "validation", "test"},
    )
    checkpoint_sha256 = latent_meta.get("checkpoint_sha256")
    if checkpoint_sha256 not in set(score_meta.get("checkpoint_sha256") or ()):
        raise ValueError("candidate latent checkpoint is not part of the score ensemble")
    return file_sha256_short(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--score_artifact", required=True)
    parser.add_argument(
        "--candidate_latents",
        default=None,
        help="optional single-checkpoint per-support-view candidate latent artifact",
    )
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", default=None)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--model_dim", type=int, default=96)
    parser.add_argument("--neighbor_k", type=int, default=8)
    parser.add_argument("--explicit_relations", action="store_true")
    parser.add_argument("--pose_conditioned_relations", action="store_true")
    parser.add_argument("--initial_pose_rows", default=None)
    parser.add_argument("--initial_pose_strategy", default="frozen_baseline")
    parser.add_argument("--colmap_model_dir", default=None)
    parser.add_argument("--hard_negative_loss_weight", type=float, default=0.0)
    parser.add_argument("--hard_negative_margin", type=float, default=0.2)
    parser.add_argument("--hard_negative_k", type=int, default=3)
    parser.add_argument("--hard_negative_preserve_weight", type=float, default=1.0)
    parser.add_argument(
        "--selection_metric",
        choices=("identity_nll", "candidate_top1_correct_rate", "net_rescue"),
        default="identity_nll",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_width", type=float, default=1024.0)
    parser.add_argument("--image_height", type=float, default=576.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if int(args.batch_size) <= 0 or int(args.neighbor_k) <= 0:
        raise ValueError("batch_size and neighbor_k must be positive")
    if float(args.hard_negative_loss_weight) < 0.0:
        raise ValueError("hard-negative loss weight must be non-negative")
    if int(args.hard_negative_k) <= 0:
        raise ValueError("hard-negative k must be positive")
    if float(args.hard_negative_preserve_weight) < 0.0:
        raise ValueError("hard-negative preserve weight must be non-negative")
    if bool(args.explicit_relations) and args.maplet_support_index is None:
        raise ValueError("explicit relations require a maplet support index")
    if bool(args.pose_conditioned_relations) and not bool(args.explicit_relations):
        raise ValueError("pose-conditioned relations require explicit relations")
    if bool(args.pose_conditioned_relations) and (
        args.initial_pose_rows is None or args.colmap_model_dir is None
    ):
        raise ValueError(
            "pose-conditioned relations require initial pose rows and COLMAP model"
        )
    if not bool(args.pose_conditioned_relations) and (
        args.initial_pose_rows is not None or args.colmap_model_dir is not None
    ):
        raise ValueError(
            "initial pose inputs are only valid with pose-conditioned relations"
        )
    if float(args.image_width) <= 1.0 or float(args.image_height) <= 1.0:
        raise ValueError("image dimensions must be greater than one pixel")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    candidates, candidate_meta = _load(Path(args.candidate_artifact))
    scores, score_meta = _load(Path(args.score_artifact))
    latent_arrays = None
    latent_meta = None
    if args.candidate_latents is not None:
        latent_arrays, latent_meta = _load(Path(args.candidate_latents))
    proposals, _ = _load(Path(args.proposals))
    landmark, landmark_meta = load_landmark_index_npz(Path(args.landmark_bank))
    maplet_index = None
    maplet_metadata = None
    if args.maplet_support_index is not None:
        maplet_index, maplet_metadata = load_local_maplet_support_index_npz(
            Path(args.maplet_support_index)
        )
        if not np.array_equal(maplet_index.anchor_track_ids, landmark.track_ids):
            raise ValueError("maplet support index and landmark bank rows differ")
        source_hash = str(maplet_metadata.get("source_landmark_index_sha256", ""))
        if source_hash and source_hash != file_sha256_short(Path(args.landmark_bank)):
            raise ValueError("maplet support index references a different landmark bank")
    input_hashes = _validate_manifests(
        args, candidate_meta, score_meta, landmark_meta
    )
    if bool(args.pose_conditioned_relations):
        model_dir = Path(args.colmap_model_dir)
        input_hashes.update(
            {
                "initial_pose_rows_sha256": file_sha256_short(
                    Path(args.initial_pose_rows)
                ),
                "colmap_cameras_sha256": file_sha256_short(
                    model_dir / "cameras.bin"
                ),
                "colmap_images_sha256": file_sha256_short(
                    model_dir / "images.bin"
                ),
            }
        )
    if latent_arrays is not None and latent_meta is not None:
        input_hashes["candidate_latents_sha256"] = _validate_latent_manifest(
            Path(args.candidate_latents), latent_meta, score_meta
        )
    selected_rows = candidates["selected_rows"].astype(np.int64)
    selected_columns = candidates["selected_columns"].astype(np.int64)
    if selected_columns.ndim != 2:
        raise ValueError("selected_columns must have shape (rows, candidates)")
    row_count, candidate_count = selected_columns.shape
    query_count = int(candidate_meta.get("query_points_per_image", 0))
    if query_count <= 0 or row_count % query_count:
        raise ValueError("candidate rows do not form fixed whole-image blocks")
    if int(candidate_meta.get("candidate_top_k", 0)) != candidate_count:
        raise ValueError("candidate count disagrees with candidate artifact manifest")
    for key in ("features", "labels", "valid_edges"):
        if candidates[key].shape[:2] != selected_columns.shape:
            raise ValueError(f"candidate array {key!r} is not aligned")
    if not np.all(candidates["valid_edges"]):
        raise ValueError("whole-image graph v1 requires a dense valid candidate pool")
    query_ids = proposals["query_ids"][selected_rows].astype(str)
    query_xy = proposals["xy"][selected_rows].astype(np.float32)
    image_count = len(query_ids) // query_count
    query_id_blocks = query_ids.reshape(image_count, query_count)
    if not np.all(query_id_blocks == query_id_blocks[:, :1]):
        raise ValueError("candidate rows are not contiguous whole images")
    image_ids = query_id_blocks[:, 0]
    static = candidates["features"].astype(np.float32)
    if static.ndim != 3 or static.shape[:2] != (row_count, candidate_count):
        raise ValueError("candidate feature tensor is not aligned")
    if not np.all(np.isfinite(static)):
        raise ValueError("candidate features contain non-finite values")
    prefix = "ensemble"
    typed = np.stack(
        [
            scores[f"{prefix}__set_candidate_probability"],
            scores[f"{prefix}__anchor_assignment_probability"],
            scores[f"{prefix}__geometry_p01px"],
            scores[f"{prefix}__geometry_p02px"],
            scores[f"{prefix}__geometry_p05px"],
            scores[f"{prefix}__candidate_visibility_probability"],
            scores[f"{prefix}__support_view_probability_0"],
            scores[f"{prefix}__support_view_probability_1"],
        ],
        axis=2,
    ).astype(np.float32)
    if typed.shape[:2] != (row_count, candidate_count):
        raise ValueError("typed score evidence is not aligned with candidates")
    latent_mode = latent_arrays is not None
    scalar_typed_count = 6 if latent_mode else typed.shape[2]
    features = np.concatenate([static, typed[:, :, :scalar_typed_count]], axis=2).reshape(
        image_count, query_count, candidate_count, -1
    )
    candidate_view_latents = None
    support_view_probability = None
    support_view_mask = None
    latent_mean = None
    latent_std = None
    if latent_mode:
        assert latent_arrays is not None and latent_meta is not None
        view_count = int(latent_meta.get("support_view_count", 0))
        expected_view_count = typed.shape[2] - scalar_typed_count
        if view_count != expected_view_count or view_count <= 0:
            raise ValueError("latent support-view count disagrees with score evidence")
        latent_keys = tuple(
            f"candidate_view_embedding_{view_rank}" for view_rank in range(view_count)
        )
        if not all(key in latent_arrays for key in latent_keys):
            raise ValueError("candidate latent artifact is missing support views")
        latent_values = np.stack(
            [latent_arrays[key] for key in latent_keys], axis=1
        ).astype(np.float32)
        expected_edge_count = row_count * candidate_count
        if latent_values.shape[:2] != (expected_edge_count, view_count):
            raise ValueError("candidate latent arrays are not aligned with candidate edges")
        if int(latent_meta.get("model_dim", 0)) != latent_values.shape[2]:
            raise ValueError("candidate latent dimension disagrees with its manifest")
        if not np.all(np.isfinite(latent_values)):
            raise ValueError("candidate latent artifact contains missing/non-finite values")
        candidate_view_latents = latent_values.reshape(
            image_count,
            query_count,
            candidate_count,
            view_count,
            latent_values.shape[2],
        )
        support_view_probability = typed[:, :, scalar_typed_count:].reshape(
            image_count, query_count, candidate_count, view_count
        )
        support_view_mask = np.isfinite(support_view_probability)
        if not np.all(support_view_mask):
            raise ValueError("whole-image latent graph v1 requires every support view")
        if np.any(support_view_probability < 0.0):
            raise ValueError("support-view probabilities must be non-negative")
        support_mass = np.sum(support_view_probability, axis=3)
        if not np.allclose(support_mass, 1.0, atol=2e-5, rtol=2e-5):
            raise ValueError("support-view probabilities are not normalized")
    labels = candidates["labels"].astype(bool).reshape(
        image_count, query_count, candidate_count
    )
    prior = scores[f"{prefix}__set_candidate_probability"].astype(np.float32).reshape(
        image_count, query_count, candidate_count
    )
    dustbin_full = scores[
        f"{prefix}__set_dustbin_probability_DIAGNOSTIC_ONLY"
    ].astype(np.float32)
    if dustbin_full.shape != (row_count, candidate_count):
        raise ValueError("dustbin score tensor is not aligned with candidates")
    if not np.allclose(dustbin_full, dustbin_full[:, :1], atol=1e-7, rtol=1e-6):
        raise ValueError("per-query dustbin probability is not constant over candidates")
    null = dustbin_full[:, 0].reshape(image_count, query_count)
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(prior)) or not np.all(np.isfinite(null)):
        raise ValueError("full-split graph inputs contain non-finite values")
    if np.any(prior < 0.0) or np.any(null < 0.0):
        raise ValueError("candidate/null prior probabilities must be non-negative")
    prior_mass = np.sum(prior, axis=2) + null
    if not np.allclose(prior_mass, 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError(
            "candidate/null prior probability mass is not normalized: "
            f"range=({prior_mass.min():.8f}, {prior_mass.max():.8f})"
        )
    bank_rows = np.take_along_axis(
        proposals["bank_row_indices"][selected_rows], selected_columns, axis=1
    ).reshape(image_count, query_count, candidate_count)
    candidate_xyz = landmark.xyz[bank_rows]
    candidate_track_ids = landmark.track_ids[bank_rows]
    total_candidates = query_count * candidate_count
    if int(args.neighbor_k) > total_candidates:
        raise ValueError("neighbor_k exceeds candidates in a whole image")
    xy_normalized = query_xy.reshape(image_count, query_count, 2) / np.asarray(
        [float(args.image_width) - 1.0, float(args.image_height) - 1.0],
        dtype=np.float32,
    )
    if np.any(xy_normalized < -1e-4) or np.any(xy_normalized > 1.0001):
        raise ValueError("query coordinates lie outside the declared image dimensions")
    relation_features = None
    relation_mean = None
    relation_std = None
    relation_names: tuple[str, ...] = ()
    initial_pose_manifest = None
    initial_pose_camera_ids = None
    if bool(args.explicit_relations):
        if maplet_index is None:
            raise RuntimeError("explicit relation maplet index was not loaded")
        projected_xy_normalized = None
        candidate_in_front = None
        initial_pose_available = None
        if bool(args.pose_conditioned_relations):
            pose_lookup, initial_pose_manifest = _load_initial_pose_lookup(
                Path(args.initial_pose_rows),
                strategy=str(args.initial_pose_strategy),
                expected_image_ids=image_ids,
            )
            (
                projected_xy_normalized,
                candidate_in_front,
                initial_pose_available,
                initial_pose_camera_ids,
            ) = _project_candidates_from_initial_pose(
                candidate_xyz,
                image_ids,
                pose_lookup,
                colmap_model_dir=Path(args.colmap_model_dir),
                image_width=float(args.image_width),
                image_height=float(args.image_height),
            )
        neighbor_indices, relation_features = build_explicit_relation_graph(
            xy_normalized,
            candidate_xyz,
            bank_rows,
            candidate_track_ids,
            maplet_index,
            neighbor_k=int(args.neighbor_k),
            projected_xy_normalized=projected_xy_normalized,
            candidate_in_front=candidate_in_front,
            initial_pose_available=initial_pose_available,
        )
        relation_names = relation_feature_names(
            pose_conditioned=bool(args.pose_conditioned_relations)
        )
    else:
        flat_xyz = candidate_xyz.reshape(image_count, total_candidates, 3)
        neighbor_indices = np.empty(
            (image_count, total_candidates, int(args.neighbor_k)), dtype=np.int64
        )
        for image_index in range(image_count):
            _distance, indices = cKDTree(flat_xyz[image_index]).query(
                flat_xyz[image_index], k=int(args.neighbor_k)
            )
            neighbor_indices[image_index] = np.asarray(indices).reshape(
                total_candidates, int(args.neighbor_k)
            )
        neighbor_indices = neighbor_indices.reshape(
            image_count, query_count, candidate_count, int(args.neighbor_k)
        )
    split = json.loads(Path(args.split_json).read_text())
    indices_by_split = {
        name: np.flatnonzero(np.isin(image_ids, split[name]))
        for name in ("train", "validation", "test")
    }
    assigned = np.zeros((image_count,), dtype=np.int64)
    for name, indices in indices_by_split.items():
        if len(indices) == 0:
            raise ValueError(f"split {name!r} has no whole images")
        assigned[indices] += 1
    if not np.all(assigned == 1):
        raise ValueError("split JSON must partition every candidate image exactly once")
    train_indices = indices_by_split["train"]
    mean = features[train_indices].mean(axis=(0, 1, 2), keepdims=True)
    std = features[train_indices].std(axis=(0, 1, 2), keepdims=True)
    std = np.maximum(std, 1e-4)
    features = (features - mean) / std
    if relation_features is not None:
        relation_mean = relation_features[train_indices].mean(
            axis=(0, 1, 2, 3), keepdims=True
        )
        relation_std = relation_features[train_indices].std(
            axis=(0, 1, 2, 3), keepdims=True
        )
        relation_std = np.maximum(relation_std, 1e-4)
        relation_features = (relation_features - relation_mean) / relation_std
    if candidate_view_latents is not None:
        latent_mean = candidate_view_latents[train_indices].mean(
            axis=(0, 1, 2, 3), keepdims=True
        )
        latent_std = candidate_view_latents[train_indices].std(
            axis=(0, 1, 2, 3), keepdims=True
        )
        latent_std = np.maximum(latent_std, 1e-4)
        candidate_view_latents = (
            candidate_view_latents - latent_mean
        ) / latent_std
    device = torch.device(args.device)
    if candidate_view_latents is None:
        model = WholeImageCandidateGraph(
            features.shape[-1],
            model_dim=int(args.model_dim),
            relation_dim=(
                0 if relation_features is None else int(relation_features.shape[-1])
            ),
        ).to(device)
    else:
        model = WholeImageLatentCandidateGraph(
            candidate_view_latents.shape[-1],
            features.shape[-1],
            model_dim=int(args.model_dim),
            relation_dim=(
                0 if relation_features is None else int(relation_features.shape[-1])
            ),
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    rng = random.Random(int(args.seed))

    def forward(index):
        ids = np.asarray(index, dtype=np.int64)
        common = (
            torch.from_numpy(xy_normalized[ids]).to(device),
            torch.from_numpy(neighbor_indices[ids]).to(device),
            torch.from_numpy(prior[ids]).to(device),
            torch.from_numpy(null[ids]).to(device),
        )
        relation = (
            None
            if relation_features is None
            else torch.from_numpy(relation_features[ids]).to(device)
        )
        if candidate_view_latents is None:
            return model(
                torch.from_numpy(features[ids]).to(device),
                *common,
                relation_features=relation,
            )
        assert support_view_probability is not None and support_view_mask is not None
        return model(
            torch.from_numpy(candidate_view_latents[ids]).to(device),
            torch.from_numpy(features[ids]).to(device),
            torch.from_numpy(support_view_probability[ids]).to(device),
            torch.from_numpy(support_view_mask[ids]).to(device),
            *common,
            relation_features=relation,
        )

    best_state = None
    best_epoch = -1
    best_selection_key: tuple[float, ...] | None = None
    history = []
    baseline_metrics = {
        name: _metrics(
            torch.from_numpy(np.log(np.maximum(prior[indices], 1e-12))).to(device),
            torch.from_numpy(np.log(np.maximum(null[indices], 1e-12))).to(device),
            torch.from_numpy(labels[indices]).to(device),
            prior_probability=prior[indices],
        )
        for name, indices in indices_by_split.items()
    }
    for epoch in range(int(args.epochs)):
        model.train()
        order = train_indices.tolist()
        rng.shuffle(order)
        losses = []
        identity_losses = []
        hard_negative_losses = []
        for start in range(0, len(order), int(args.batch_size)):
            batch_indices = order[start : start + int(args.batch_size)]
            candidate_logits, null_logits = forward(batch_indices)
            target = torch.from_numpy(labels[batch_indices]).to(device)
            identity_loss = set_identity_nll(candidate_logits, null_logits, target)
            hard_negative_loss = system_hard_negative_margin_loss(
                candidate_logits,
                target,
                torch.from_numpy(prior[batch_indices]).to(device),
                margin=float(args.hard_negative_margin),
                hard_negative_k=int(args.hard_negative_k),
                preserve_weight=float(args.hard_negative_preserve_weight),
            )
            loss = identity_loss + float(
                args.hard_negative_loss_weight
            ) * hard_negative_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            identity_losses.append(float(identity_loss.detach().cpu()))
            hard_negative_losses.append(float(hard_negative_loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_logits, validation_null = forward(indices_by_split["validation"])
            validation_metrics = _metrics(
                validation_logits,
                validation_null,
                torch.from_numpy(labels[indices_by_split["validation"]]).to(device),
                prior_probability=prior[indices_by_split["validation"]],
            )
        selection_key = _checkpoint_selection_key(
            validation_metrics, str(args.selection_metric)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_identity_loss": float(np.mean(identity_losses)),
                "train_hard_negative_loss": float(np.mean(hard_negative_losses)),
                "validation": validation_metrics,
                "validation_selection_key": list(selection_key),
            }
        )
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    metrics = {}
    posterior = {}
    with torch.no_grad():
        for name in ("train", "validation", "test"):
            logits, null_logits = forward(indices_by_split[name])
            metrics[name] = _metrics(
                logits,
                null_logits,
                torch.from_numpy(labels[indices_by_split[name]]).to(device),
                prior_probability=prior[indices_by_split[name]],
            )
            posterior[name] = torch.softmax(torch.cat([logits, null_logits.unsqueeze(2)], 2), 2).cpu().numpy()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if relation_features is not None and bool(args.pose_conditioned_relations):
        graph_stage = (
            "whole_image_latent_pose_relation_candidate_graph_v3"
            if latent_mode
            else "whole_image_pose_relation_candidate_graph_v3"
        )
    elif relation_features is not None:
        graph_stage = (
            "whole_image_latent_relation_candidate_graph_v2"
            if latent_mode
            else "whole_image_relation_candidate_graph_v2"
        )
    else:
        graph_stage = (
            "whole_image_latent_candidate_graph_v1"
            if latent_mode
            else "whole_image_candidate_graph_v1"
        )
    checkpoint = output / "best.pt"
    torch.save(
        {
            "format": graph_stage,
            "model_state_dict": best_state,
            "input_dim": features.shape[-1],
            "latent_dim": (
                None
                if candidate_view_latents is None
                else int(candidate_view_latents.shape[-1])
            ),
            "model_dim": int(args.model_dim),
            "feature_mean": mean,
            "feature_std": std,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "relation_dim": (
                0 if relation_features is None else int(relation_features.shape[-1])
            ),
            "relation_feature_names": (
                [] if relation_features is None else list(relation_names)
            ),
            "relation_mean": relation_mean,
            "relation_std": relation_std,
            "neighbor_k": int(args.neighbor_k),
            "hard_negative_loss_weight": float(args.hard_negative_loss_weight),
            "hard_negative_margin": float(args.hard_negative_margin),
            "hard_negative_k": int(args.hard_negative_k),
            "hard_negative_preserve_weight": float(
                args.hard_negative_preserve_weight
            ),
            "selection_metric": str(args.selection_metric),
            "best_selection_key": list(best_selection_key),
            "best_epoch": best_epoch,
            "input_hashes": input_hashes,
            "descriptor_space_id": landmark_meta.get("descriptor_space_id"),
        },
        checkpoint,
    )
    posterior_path = output / "posterior.npz"
    posterior_metadata = {
        "format": (
            f"{graph_stage}_posterior"
            if relation_features is not None
            else (
                "whole_image_latent_candidate_graph_posterior_v1"
                if latent_mode
                else "whole_image_candidate_graph_posterior_v1"
            )
        ),
        "checkpoint_sha256": file_sha256_short(checkpoint),
        "input_hashes": input_hashes,
        "descriptor_space_id": landmark_meta.get("descriptor_space_id"),
        "query_count": query_count,
        "candidate_count": candidate_count,
    }
    np.savez_compressed(
        posterior_path,
        image_ids=image_ids,
        metadata_json=np.asarray(json.dumps(posterior_metadata, sort_keys=True), dtype=np.str_),
        **{f"{name}_indices": indices_by_split[name] for name in indices_by_split},
        **{f"{name}_posterior": posterior[name] for name in posterior},
    )
    metric_delta = {
        name: {
            key: metrics[name][key] - baseline_metrics[name][key]
            for key in metrics[name]
            if key in baseline_metrics[name]
            and np.isfinite(metrics[name][key])
            and np.isfinite(baseline_metrics[name][key])
        }
        for name in metrics
    }
    summary = {
        "stage": graph_stage,
        "best_epoch": best_epoch,
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "metric_delta_graph_minus_prior": metric_delta,
        "history": history,
        "protocol": {
            "target": "2px_set_valued_identity",
            "pose_input": bool(args.pose_conditioned_relations),
            "test_used_for_selection": False,
            "explicit_null": True,
            "query_graph": True,
            "explicit_relation_graph": bool(relation_features is not None),
            "relation_feature_names": (
                [] if relation_features is None else list(relation_names)
            ),
            "relation_neighbors_exclude_same_query_group": bool(
                relation_features is not None
            ),
            "maplet_relation": bool(relation_features is not None),
            "pose_conditioned_relations": bool(args.pose_conditioned_relations),
            "initial_pose_manifest": initial_pose_manifest,
            "initial_pose_strategy": (
                str(args.initial_pose_strategy)
                if bool(args.pose_conditioned_relations)
                else None
            ),
            "initial_pose_camera_ids": (
                None
                if initial_pose_camera_ids is None
                else sorted(
                    int(value)
                    for value in np.unique(initial_pose_camera_ids)
                    if int(value) >= 0
                )
            ),
            "hard_negative_loss_weight": float(args.hard_negative_loss_weight),
            "hard_negative_margin": float(args.hard_negative_margin),
            "hard_negative_k": int(args.hard_negative_k),
            "hard_negative_preserve_weight": float(
                args.hard_negative_preserve_weight
            ),
            "selection_metric": str(args.selection_metric),
            "best_selection_key": list(best_selection_key),
            "candidate_latent_input": bool(latent_mode),
            "support_view_mixture": bool(latent_mode),
            "support_view_marginalization": (
                "logsumexp_at_candidate_logit" if latent_mode else None
            ),
            "landmark_xyz_knn": int(args.neighbor_k),
            "query_count": query_count,
            "candidate_count": candidate_count,
            "image_dimensions": [float(args.image_width), float(args.image_height)],
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
        },
        "inputs": {
            **input_hashes,
            "descriptor_space_id": landmark_meta.get("descriptor_space_id"),
        },
        "outputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256_short(checkpoint),
            "posterior": str(posterior_path),
            "posterior_sha256": file_sha256_short(posterior_path),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
