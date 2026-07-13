"""Materialize candidate-specific real RGB rows from a frozen top-M selection."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.local_maplet_matching import load_local_maplet_support_index_npz
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.measurement_v1.candidate_measurement_schema import (
    CandidateIdentityKey,
    support_view_set_id,
)
from feature_extract.vfm.measurement_v1.candidate_measurement_selection import (
    audit_candidate_measurement_selection,
)
from feature_extract.vfm.measurement_v1.real_real_tracks import (
    REAL_REAL_MEASUREMENT_FIELDNAMES,
)
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


CANDIDATE_MEASUREMENT_EXTRA_FIELDNAMES = [
    "candidate_identity_key",
    "support_view_set_id",
    "candidate_measurement_rank",
    "candidate_score_rank",
    "candidate_role",
    "candidate_prototype_id",
    "candidate_bank_row",
    "candidate_assignment_probability",
    "candidate_retrieval_similarity",
    "candidate_geometry_p01",
    "candidate_geometry_p02",
    "candidate_geometry_p05",
    "source_query_row",
    "split",
    "support_view_rank",
    "support_view_probability",
    "actual_query_observation",
    "center_residual_px",
    "target_gt_projected_x",
    "target_gt_projected_y",
    "target_gt_projected_residual_px",
    "target_gt_projection_in_front",
    "target_gt_projection_in_image",
    "target_geometry_correct_1px",
    "target_geometry_correct_2px",
    "target_geometry_correct_5px",
    "dustbin_supervision_weight",
    "geometry_supervision_weight",
    "target_reason",
    "measurement_loss_weight",
]


def _load_selection(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    artifact_format = metadata.get("format")
    if artifact_format not in {
        "candidate_measurement_selection_v1",
        "candidate_evidence_v3",
    }:
        raise ValueError("unsupported candidate measurement selection artifact")
    required = {
        "selected_rows",
        "query_ids",
        "query_xy",
        "split_names",
        "candidate_valid",
        "candidate_roles",
        "candidate_score_ranks",
        "candidate_track_ids",
        "candidate_prototype_ids",
        "candidate_bank_rows",
        "candidate_coarse_similarities",
        "candidate_geometry_probabilities",
        "candidate_support_view_probabilities",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"candidate measurement selection lacks arrays: {sorted(missing)}")
    if artifact_format == "candidate_evidence_v3":
        if "candidate_prior_probabilities" not in arrays:
            raise ValueError("candidate evidence V3 lacks candidate prior probabilities")
        if "candidate_target_gt_residuals_px" not in arrays:
            raise ValueError("candidate evidence V3 lacks target-only GT residuals")
        arrays["candidate_scores"] = arrays["candidate_prior_probabilities"]
        arrays["candidate_gt_residuals_px"] = arrays[
            "candidate_target_gt_residuals_px"
        ]
    else:
        for key in ("candidate_scores", "candidate_gt_residuals_px"):
            if key not in arrays:
                raise ValueError(f"candidate measurement selection lacks array: {key}")
    return arrays, metadata


def _within_margin(xy: np.ndarray, *, width: int, height: int, margin: float) -> bool:
    x, y = float(xy[0]), float(xy[1])
    return margin <= x < float(width) - margin and margin <= y < float(height) - margin


def _frame_gap(a: str, b: str) -> int | None:
    pattern = re.compile(r"(?:^|/)(seq[^/]+)/frame(\d+)\.[^.]+$")
    match_a = pattern.search(str(a))
    match_b = pattern.search(str(b))
    if match_a is None or match_b is None or match_a.group(1) != match_b.group(1):
        return None
    return abs(int(match_a.group(2)) - int(match_b.group(2)))


def _sequence_name(query_id: str) -> str:
    match = re.search(r"(?:^|/)(seq[^/]+)/", str(query_id))
    return "unknown" if match is None else str(match.group(1))


def _query_track_observations(image: Any) -> dict[int, tuple[float, float]]:
    return {
        int(track_id): (float(xy[0]), float(xy[1]))
        for track_id, xy in zip(image.point3d_ids.tolist(), image.xys.tolist())
        if int(track_id) >= 0
    }


def _validate_input_hashes(
    metadata: Mapping[str, Any],
    *,
    support_geometry_index: Path,
    maplet_support_index: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
) -> dict[str, str]:
    actual: dict[str, str] = {
        "support_geometry_index_sha256": file_sha256_short(Path(support_geometry_index)),
        "maplet_support_index_sha256": file_sha256_short(Path(maplet_support_index)),
        "projected_landmark_bank_sha256": file_sha256_short(Path(projected_landmark_bank)),
    }
    if metadata.get("format") == "candidate_evidence_v3":
        actual.update(
            {
                "colmap_images_sha256": file_sha256_short(
                    Path(colmap_model_dir) / "images.bin"
                ),
                "colmap_cameras_sha256": file_sha256_short(
                    Path(colmap_model_dir) / "cameras.bin"
                ),
                "colmap_points3d_sha256": file_sha256_short(
                    Path(colmap_model_dir) / "points3D.bin"
                ),
            }
        )
    mismatches = {
        key: {"selection": metadata.get(key), "input": value}
        for key, value in actual.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"candidate measurement row inputs are stale: {json.dumps(mismatches, sort_keys=True)}"
        )
    return actual


def build_candidate_measurement_rows(
    *,
    selection_artifact: Path,
    support_geometry_index: Path,
    maplet_support_index: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    split_name: str,
    output_rows_csv: Path,
    search_radius_px: float = 6.0,
    context_radius_px: float = 12.0,
    support_views_per_candidate: int = 4,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    """Build real-real RGB rows for every candidate, without pose-based selection."""

    if str(split_name) not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation, or test")
    if float(search_radius_px) <= 0.0 or float(context_radius_px) < 0.0:
        raise ValueError("measurement radii are invalid")
    if int(support_views_per_candidate) <= 0:
        raise ValueError("support_views_per_candidate must be positive")
    arrays, metadata = _load_selection(Path(selection_artifact))
    expected_hashes = _validate_input_hashes(
        metadata,
        support_geometry_index=Path(support_geometry_index),
        maplet_support_index=Path(maplet_support_index),
        projected_landmark_bank=Path(projected_landmark_bank),
        colmap_model_dir=Path(colmap_model_dir),
    )
    geometry_index, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    maplet_index, maplet_metadata = load_local_maplet_support_index_npz(
        Path(maplet_support_index)
    )
    landmark_index, landmark_metadata = load_landmark_index_npz(
        Path(projected_landmark_bank)
    )
    if not np.array_equal(maplet_index.anchor_track_ids, landmark_index.track_ids):
        raise ValueError("maplet and canonical landmark track rows differ")
    track_to_canonical = {
        int(track_id): int(row)
        for row, track_id in enumerate(landmark_index.track_ids.tolist())
    }
    images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
    cameras = read_colmap_cameras_binary(Path(colmap_model_dir) / "cameras.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}

    query_ids = np.asarray(arrays["query_ids"]).astype(str)
    query_xy = np.asarray(arrays["query_xy"], dtype=np.float32)
    split_names = np.asarray(arrays["split_names"]).astype(str)
    selected_rows = np.asarray(arrays["selected_rows"], dtype=np.int64)
    valid = np.asarray(arrays["candidate_valid"], dtype=bool)
    roles = np.asarray(arrays["candidate_roles"]).astype(str)
    score_ranks = np.asarray(arrays["candidate_score_ranks"], dtype=np.int64)
    track_ids = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    prototype_ids = np.asarray(arrays["candidate_prototype_ids"], dtype=np.int64)
    bank_rows = np.asarray(arrays["candidate_bank_rows"], dtype=np.int64)
    assignment_scores = np.asarray(arrays["candidate_scores"], dtype=np.float32)
    retrieval_similarities = np.asarray(
        arrays["candidate_coarse_similarities"], dtype=np.float32
    )
    stored_gt_residuals = np.asarray(
        arrays["candidate_gt_residuals_px"], dtype=np.float32
    )
    geometry_probabilities = np.asarray(
        arrays["candidate_geometry_probabilities"], dtype=np.float32
    )
    support_probabilities = np.asarray(
        arrays["candidate_support_view_probabilities"], dtype=np.float32
    )
    candidate_shape = valid.shape
    for name, value in (
        ("roles", roles),
        ("score_ranks", score_ranks),
        ("track_ids", track_ids),
        ("prototype_ids", prototype_ids),
        ("bank_rows", bank_rows),
        ("assignment_scores", assignment_scores),
        ("retrieval_similarities", retrieval_similarities),
        ("stored_gt_residuals", stored_gt_residuals),
    ):
        if value.shape != candidate_shape:
            raise ValueError(f"candidate selection shape mismatch: {name}")
    if geometry_probabilities.shape != (*candidate_shape, 3):
        raise ValueError("candidate geometry probabilities must have three channels")
    if support_probabilities.ndim != 3 or support_probabilities.shape[:2] != candidate_shape:
        raise ValueError("candidate support-view probabilities have incompatible shape")

    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    skipped_by_group: Counter[str] = Counter()
    target_reasons: Counter[str] = Counter()
    candidates_seen = 0
    candidates_with_support = 0
    candidate_rows_by_rank: Counter[str] = Counter()
    support_available = np.zeros(candidate_shape, dtype=bool)
    margin = float(search_radius_px) + float(context_radius_px)
    query_observation_cache: dict[str, dict[int, tuple[float, float]]] = {}
    stop = False
    for query_row in range(len(query_ids)):
        if split_names[query_row] != str(split_name):
            continue
        query_id = str(query_ids[query_row])
        query_image = images_by_name.get(query_id)
        if query_image is None:
            skipped["query_missing_from_colmap"] += int(np.sum(valid[query_row]))
            continue
        query_camera = cameras.get(int(query_image.camera_id))
        if query_camera is None:
            skipped["query_camera_missing"] += int(np.sum(valid[query_row]))
            continue
        if query_id not in query_observation_cache:
            query_observation_cache[query_id] = _query_track_observations(query_image)
        pose_w2c = np.eye(4, dtype=np.float64)
        pose_w2c[:3, :3] = qvec_to_rotmat(query_image.qvec)
        pose_w2c[:3, 3] = np.asarray(query_image.tvec, dtype=np.float64).reshape(3)
        center = np.asarray(query_xy[query_row], dtype=np.float64)
        for measurement_column in range(candidate_shape[1]):
            if not bool(valid[query_row, measurement_column]):
                continue
            if max_candidates is not None and candidates_seen >= int(max_candidates):
                stop = True
                break
            candidates_seen += 1
            measurement_rank = int(measurement_column + 1)
            track_id = int(track_ids[query_row, measurement_column])
            prototype_id = int(prototype_ids[query_row, measurement_column])
            canonical_row = track_to_canonical.get(track_id)
            if canonical_row is None:
                skipped["track_missing_from_canonical_bank"] += 1
                continue
            landmark_xyz = np.asarray(landmark_index.xyz[canonical_row], dtype=np.float64)
            projected_xy = project_world_to_image(
                landmark_xyz[None, :], pose_w2c, query_camera
            )[0]
            camera_xyz = pose_w2c[:3, :3] @ landmark_xyz + pose_w2c[:3, 3]
            in_front = bool(camera_xyz[2] > 1e-6)
            in_image = bool(
                in_front
                and 0.0 <= float(projected_xy[0]) < float(query_camera.width)
                and 0.0 <= float(projected_xy[1]) < float(query_camera.height)
            )
            gt_residual = float(np.linalg.norm(center - projected_xy))
            stored_residual = float(stored_gt_residuals[query_row, measurement_column])
            if math.isfinite(stored_residual) and not math.isclose(
                stored_residual, gt_residual, rel_tol=1e-4, abs_tol=1e-3
            ):
                raise ValueError(
                    "candidate GT projection residual changed: "
                    f"query={query_id}, track={track_id}, stored={stored_residual}, recomputed={gt_residual}"
                )
            query_target = query_observation_cache[query_id].get(track_id)
            actual_observation = query_target is not None
            if actual_observation:
                target_xy = np.asarray(query_target, dtype=np.float64)
                center_residual = float(np.linalg.norm(center - target_xy))
                target_is_dustbin = center_residual > float(search_radius_px)
                target_reason = "out_of_window" if target_is_dustbin else "valid_observation"
                dustbin_supervision_weight = 1.0
            else:
                target_xy = center.copy()
                center_residual = float("inf")
                target_is_dustbin = True
                target_reason = "track_not_observed"
                dustbin_supervision_weight = float(gt_residual > float(search_radius_px))
            target_reasons[target_reason] += 1

            support_indices = np.asarray(
                maplet_index.support_image_indices[canonical_row], dtype=np.int64
            )
            posterior = support_probabilities[query_row, measurement_column]
            posterior_count = min(len(posterior), len(support_indices))
            learned_order = np.argsort(-posterior[:posterior_count], kind="stable").tolist()
            support_order = learned_order + [
                index for index in range(posterior_count, len(support_indices))
            ]
            usable_supports: list[dict[str, Any]] = []
            for support_rank in support_order:
                image_index = int(support_indices[int(support_rank)])
                if image_index < 0:
                    continue
                support_id = str(maplet_index.support_image_ids[image_index])
                if support_id == query_id:
                    skipped["same_image_support_forbidden"] += 1
                    continue
                support_image = images_by_name.get(support_id)
                if support_image is None:
                    skipped["support_image_missing_from_colmap"] += 1
                    continue
                support_camera = cameras.get(int(support_image.camera_id))
                if support_camera is None:
                    skipped["support_camera_missing"] += 1
                    continue
                support_row = int(
                    geometry_index.geometry_rows_for_tracks(
                        support_id, np.asarray([track_id], dtype=np.int64)
                    )[0]
                )
                if support_row < 0:
                    skipped["support_view_missing_anchor"] += 1
                    continue
                support_xy = np.asarray(geometry_index.xy[support_row], dtype=np.float64)
                if not _within_margin(
                    center,
                    width=int(query_camera.width),
                    height=int(query_camera.height),
                    margin=margin,
                ) or not _within_margin(
                    support_xy,
                    width=int(support_camera.width),
                    height=int(support_camera.height),
                    margin=margin,
                ):
                    skipped["crop_near_boundary"] += 1
                    continue
                usable_supports.append(
                    {
                        "support_id": support_id,
                        "support_xy": support_xy,
                        "support_row": support_row,
                        "support_rank": int(support_rank),
                        "support_probability": (
                            float(posterior[int(support_rank)])
                            if int(support_rank) < posterior_count
                            else None
                        ),
                    }
                )
                if len(usable_supports) >= int(support_views_per_candidate):
                    break
            if not usable_supports:
                skipped["no_usable_support_view"] += 1
                group = (
                    f"{_sequence_name(query_id)}:candidate_rank_{measurement_rank}:"
                    "no_usable_support_view"
                )
                skipped_by_group[group] += 1
                continue
            candidates_with_support += 1
            support_available[query_row, measurement_column] = True
            candidate_rows_by_rank[str(measurement_rank)] += 1
            view_set_id = support_view_set_id(
                support["support_id"] for support in usable_supports
            )
            identity = CandidateIdentityKey(
                query_id=query_id,
                source_query_row=int(selected_rows[query_row]),
                track_id=track_id,
                prototype_id=prototype_id,
                support_view_set_id=view_set_id,
            )
            for support in usable_supports:
                support_id = str(support["support_id"])
                support_xy = np.asarray(support["support_xy"], dtype=np.float64)
                support_row = int(support["support_row"])
                frame_gap = _frame_gap(support_id, query_id)
                row = {
                    "query_id": query_id,
                    "support_image_id": support_id,
                    "track_id": track_id,
                    "support_track_id": track_id,
                    "track_length": int(landmark_index.observation_counts[canonical_row]),
                    "support_x": float(support_xy[0]),
                    "support_y": float(support_xy[1]),
                    "render_x": float(support_xy[0]),
                    "render_y": float(support_xy[1]),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "query_gt_x": float(target_xy[0]),
                    "query_gt_y": float(target_xy[1]),
                    "requested_residual_px": (
                        float(search_radius_px) + 1.0
                        if not math.isfinite(center_residual)
                        else center_residual
                    ),
                    "target_is_dustbin": bool(target_is_dustbin),
                    "support_reprojection_error": float(
                        geometry_index.reprojection_errors[support_row]
                    ),
                    "query_reprojection_error": float(
                        landmark_index.reprojection_errors[canonical_row]
                    ),
                    "support_frame_gap": "" if frame_gap is None else int(frame_gap),
                    # Pose-derived view angle is intentionally unavailable to the RGB scorer.
                    "support_view_angle_deg": "",
                    "candidate_identity_key": identity.digest,
                    "support_view_set_id": view_set_id,
                    "candidate_measurement_rank": measurement_rank,
                    "candidate_score_rank": int(score_ranks[query_row, measurement_column]),
                    "candidate_role": str(roles[query_row, measurement_column]),
                    "candidate_prototype_id": prototype_id,
                    "candidate_bank_row": int(bank_rows[query_row, measurement_column]),
                    "candidate_assignment_probability": float(
                        assignment_scores[query_row, measurement_column]
                    ),
                    "candidate_retrieval_similarity": float(
                        retrieval_similarities[query_row, measurement_column]
                    ),
                    "candidate_geometry_p01": float(
                        geometry_probabilities[query_row, measurement_column, 0]
                    ),
                    "candidate_geometry_p02": float(
                        geometry_probabilities[query_row, measurement_column, 1]
                    ),
                    "candidate_geometry_p05": float(
                        geometry_probabilities[query_row, measurement_column, 2]
                    ),
                    "source_query_row": int(selected_rows[query_row]),
                    "split": str(split_name),
                    "support_view_rank": int(support["support_rank"]),
                    "support_view_probability": (
                        ""
                        if support["support_probability"] is None
                        else float(support["support_probability"])
                    ),
                    "actual_query_observation": bool(actual_observation),
                    "center_residual_px": center_residual,
                    "target_gt_projected_x": float(projected_xy[0]),
                    "target_gt_projected_y": float(projected_xy[1]),
                    "target_gt_projected_residual_px": gt_residual,
                    "target_gt_projection_in_front": in_front,
                    "target_gt_projection_in_image": in_image,
                    "target_geometry_correct_1px": bool(gt_residual <= 1.0),
                    "target_geometry_correct_2px": bool(gt_residual <= 2.0),
                    "target_geometry_correct_5px": bool(gt_residual <= 5.0),
                    "dustbin_supervision_weight": dustbin_supervision_weight,
                    "geometry_supervision_weight": 1.0,
                    "target_reason": target_reason,
                    "measurement_loss_weight": float(
                        max(geometry_probabilities[query_row, measurement_column, 2], 1e-3)
                    ),
                }
                rows.append(row)
        if stop:
            break

    output = Path(output_rows_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        *REAL_REAL_MEASUREMENT_FIELDNAMES,
        *CANDIDATE_MEASUREMENT_EXTRA_FIELDNAMES,
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    candidate_support_rate = (
        0.0 if candidates_seen == 0 else float(candidates_with_support / candidates_seen)
    )
    split_mask = split_names == str(split_name)
    support_constrained_audit = audit_candidate_measurement_selection(
        query_ids=query_ids[split_mask],
        split_names=split_names[split_mask],
        selected_residuals=stored_gt_residuals[split_mask],
        pool_residuals=stored_gt_residuals[split_mask],
        valid=valid[split_mask] & support_available[split_mask],
        oracle_pool_definition="measured_top_m_before_support_filter",
    )
    summary = {
        "stage": "candidate_specific_real_rgb_measurement_rows",
        "protocol": {
            "candidate_selection_frozen": True,
            "pose_used_for_candidate_or_support_selection": False,
            "pose_derived_features_exposed_to_rgb_scorer": False,
            "ground_truth_pose_target_only": True,
            "same_image_support_forbidden": True,
            "support_view_aggregation": "none_rows_remain_multimodal",
            "render": False,
            "image_retrieval": False,
            "submap": False,
        },
        "split": str(split_name),
        "candidate_count": int(candidates_seen),
        "candidates_with_support": int(candidates_with_support),
        "candidate_support_rate": candidate_support_rate,
        "support_constrained_rescue_audit": support_constrained_audit,
        "candidate_count_with_support_by_measurement_rank": dict(
            sorted(candidate_rows_by_rank.items())
        ),
        "output_rows": int(len(rows)),
        "unique_candidate_identity_count": int(
            len({str(row["candidate_identity_key"]) for row in rows})
        ),
        "target_reasons": dict(sorted(target_reasons.items())),
        "skipped": dict(sorted(skipped.items())),
        "missing_support_by_sequence_and_candidate_rank": dict(
            sorted(skipped_by_group.items())
        ),
        "search_radius_px": float(search_radius_px),
        "context_radius_px": float(context_radius_px),
        "support_views_per_candidate": int(support_views_per_candidate),
        "inputs": {
            "selection_artifact": str(selection_artifact),
            "selection_artifact_sha256": file_sha256_short(Path(selection_artifact)),
            **expected_hashes,
            "colmap_images_sha256": file_sha256_short(
                Path(colmap_model_dir) / "images.bin"
            ),
            "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
            "support_geometry_format": geometry_metadata.get("format"),
            "maplet_format": maplet_metadata.get("version"),
        },
        "outputs": {
            "rows_csv": str(output),
            "rows_csv_sha256": file_sha256_short(output),
            "summary": str(output.with_suffix(".summary.json")),
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
