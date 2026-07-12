from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    camera_center_from_qvec_tvec,
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
    viewing_ray_from_camera_center,
)
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image
from feature_extract.vfm.local_maplet_matching import (
    load_local_maplet_support_index_npz,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.measurement_v1.real_real_tracks import (
    REAL_REAL_MEASUREMENT_FIELDNAMES,
)


SELECTED_POLICY_EXTRA_FIELDNAMES = [
    "policy_row_index",
    "source_query_row",
    "split",
    "support_view_rank",
    "support_view_probability",
    "assignment_score",
    "pose_selection_score",
    "geometry_p01",
    "geometry_p02",
    "geometry_p05",
    "switched_from_baseline",
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


def _load_policy_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files if key != "metadata_json"}
        metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("format") != "pose_safe_selected_policy_v1":
        raise ValueError("unsupported selected-policy artifact")
    required = {
        "selected_rows",
        "query_ids",
        "query_xy",
        "selected_track_ids",
        "selected_canonical_rows",
        "selected_assignment_scores",
        "selected_pose_selection_scores",
        "selected_gt_residuals_px",
        "selected_geometry_probabilities",
        "selected_support_view_probabilities",
        "switched_from_baseline",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"selected-policy artifact is missing arrays: {sorted(missing)}")
    count = int(len(arrays["selected_rows"]))
    for key in required - {"selected_rows"}:
        if int(len(arrays[key])) != count:
            raise ValueError(f"selected-policy array has incompatible row count: {key}")
    return arrays, metadata


def _frame_gap(a: str, b: str) -> int | None:
    pattern = re.compile(r"(?:^|/)(seq[^/]+)/frame(\d+)\.[^.]+$")
    match_a = pattern.search(str(a))
    match_b = pattern.search(str(b))
    if match_a is None or match_b is None or match_a.group(1) != match_b.group(1):
        return None
    return abs(int(match_a.group(2)) - int(match_b.group(2)))


def _within_margin(xy: Sequence[float], *, width: int, height: int, margin: float) -> bool:
    x, y = float(xy[0]), float(xy[1])
    return margin <= x < float(width) - margin and margin <= y < float(height) - margin


def _query_track_observations(image) -> dict[int, tuple[float, float]]:
    return {
        int(track_id): (float(xy[0]), float(xy[1]))
        for track_id, xy in zip(image.point3d_ids.tolist(), image.xys.tolist())
        if int(track_id) >= 0
    }


def _geometry_probability_columns(metadata: Mapping[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for index, key in enumerate(metadata.get("geometry_probability_keys", [])):
        text = str(key)
        for threshold in ("01", "02", "05"):
            if text.endswith(f"geometry_p{threshold}px"):
                output[threshold] = int(index)
    return output


def build_selected_policy_measurement_rows(
    *,
    policy_artifact: Path,
    support_geometry_index: Path,
    maplet_support_index: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    split_json: Path,
    split_name: str,
    output_rows_csv: Path,
    search_radius_px: float = 6.0,
    context_radius_px: float = 12.0,
    minimum_geometry_p05: float = 0.0,
    support_views_per_row: int = 1,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Materialize real RGB rows from one frozen S4 assignment policy."""

    if str(split_name) not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation, or test")
    if float(search_radius_px) <= 0.0 or float(context_radius_px) < 0.0:
        raise ValueError("measurement radii must be non-negative with positive search radius")
    if not 0.0 <= float(minimum_geometry_p05) <= 1.0:
        raise ValueError("minimum_geometry_p05 must be in [0, 1]")
    if int(support_views_per_row) <= 0:
        raise ValueError("support_views_per_row must be positive")

    arrays, policy_metadata = _load_policy_artifact(Path(policy_artifact))
    geometry_index, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    maplet_index, maplet_metadata = load_local_maplet_support_index_npz(
        Path(maplet_support_index)
    )
    landmark_index, landmark_metadata = load_landmark_index_npz(
        Path(projected_landmark_bank)
    )
    expected_hashes = {
        "support_geometry_index_sha256": file_sha256_short(Path(support_geometry_index)),
        "maplet_support_index_sha256": file_sha256_short(Path(maplet_support_index)),
        "projected_landmark_bank_sha256": file_sha256_short(Path(projected_landmark_bank)),
        "split_json_sha256": file_sha256_short(Path(split_json)),
    }
    mismatches = {
        key: {"policy": policy_metadata.get(key), "input": value}
        for key, value in expected_hashes.items()
        if policy_metadata.get(key) not in {None, value}
    }
    if mismatches:
        raise ValueError(f"selected-policy measurement inputs are stale: {json.dumps(mismatches, sort_keys=True)}")
    if not np.array_equal(maplet_index.anchor_track_ids, landmark_index.track_ids):
        raise ValueError("maplet and landmark track rows differ")
    if geometry_metadata.get("format") != "support_observation_geometry_index_npz":
        raise ValueError("unsupported support geometry index")

    split = json.loads(Path(split_json).read_text())
    allowed_query_ids = {str(value) for value in split[str(split_name)]}
    images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
    cameras = read_colmap_cameras_binary(Path(colmap_model_dir) / "cameras.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    query_observation_cache: dict[str, dict[int, tuple[float, float]]] = {}
    geometry_columns = _geometry_probability_columns(policy_metadata)
    geometry_probabilities = np.asarray(
        arrays["selected_geometry_probabilities"], dtype=np.float32
    )
    view_probabilities = np.asarray(
        arrays["selected_support_view_probabilities"], dtype=np.float32
    )
    if view_probabilities.ndim != 2 or view_probabilities.shape[1] <= 0:
        raise ValueError("selected policy has no support-view posterior")
    if geometry_probabilities.ndim != 2 or "05" not in geometry_columns:
        raise ValueError("selected policy has no p(residual<=5px) output")

    query_ids = np.asarray(arrays["query_ids"]).astype(str)
    query_xy = np.asarray(arrays["query_xy"], dtype=np.float32)
    track_ids = np.asarray(arrays["selected_track_ids"], dtype=np.int64)
    canonical_rows = np.asarray(arrays["selected_canonical_rows"], dtype=np.int64)
    selected_rows = np.asarray(arrays["selected_rows"], dtype=np.int64)
    gt_projected_residuals = np.asarray(
        arrays["selected_gt_residuals_px"], dtype=np.float32
    )
    if gt_projected_residuals.shape != (len(query_ids),):
        raise ValueError("selected policy GT residuals have incompatible shape")
    assignment_scores = np.asarray(arrays["selected_assignment_scores"], dtype=np.float32)
    pose_scores = np.asarray(arrays["selected_pose_selection_scores"], dtype=np.float32)
    switched = np.asarray(arrays["switched_from_baseline"], dtype=bool)
    p05 = geometry_probabilities[:, geometry_columns["05"]]

    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    selected_policy_rows = 0
    observed_policy_rows = 0
    margin = float(search_radius_px) + float(context_radius_px)
    for policy_row in range(len(query_ids)):
        query_id = str(query_ids[policy_row])
        if query_id not in allowed_query_ids:
            continue
        if not np.isfinite(p05[policy_row]) or p05[policy_row] < float(minimum_geometry_p05):
            skipped["below_geometry_p05_threshold"] += 1
            continue
        selected_policy_rows += 1
        query_image = images_by_name.get(query_id)
        if query_image is None:
            skipped["query_missing_from_colmap"] += 1
            continue
        query_camera = cameras.get(int(query_image.camera_id))
        if query_camera is None:
            skipped["query_camera_missing"] += 1
            continue
        center = np.asarray(query_xy[policy_row], dtype=np.float64)
        gt_projected_residual = float(gt_projected_residuals[policy_row])
        has_gt_projected_residual = bool(np.isfinite(gt_projected_residual))
        canonical_row = int(canonical_rows[policy_row])
        track_id = int(track_ids[policy_row])
        if canonical_row < 0 or canonical_row >= len(landmark_index):
            skipped["invalid_canonical_row"] += 1
            continue
        if int(landmark_index.track_ids[canonical_row]) != track_id:
            raise ValueError("selected canonical row and track identity disagree")
        pose_w2c = np.eye(4, dtype=np.float64)
        pose_w2c[:3, :3] = qvec_to_rotmat(query_image.qvec)
        pose_w2c[:3, 3] = np.asarray(query_image.tvec, dtype=np.float64).reshape(3)
        landmark_xyz = np.asarray(
            landmark_index.xyz[canonical_row], dtype=np.float64
        ).reshape(3)
        gt_projected_xy = project_world_to_image(
            landmark_xyz[None, :], pose_w2c, query_camera
        )[0]
        camera_xyz = pose_w2c[:3, :3] @ landmark_xyz + pose_w2c[:3, 3]
        gt_projection_in_front = bool(camera_xyz[2] > 1e-6)
        gt_projection_in_image = bool(
            gt_projection_in_front
            and 0.0 <= float(gt_projected_xy[0]) < float(query_camera.width)
            and 0.0 <= float(gt_projected_xy[1]) < float(query_camera.height)
        )
        recomputed_gt_residual = float(np.linalg.norm(center - gt_projected_xy))
        if has_gt_projected_residual and not np.isclose(
            recomputed_gt_residual,
            gt_projected_residual,
            rtol=1e-4,
            atol=1e-3,
        ):
            raise ValueError(
                "selected-policy GT residual disagrees with direct COLMAP projection: "
                f"query={query_id}, track={track_id}, artifact={gt_projected_residual}, "
                f"recomputed={recomputed_gt_residual}"
            )
        camera_center = camera_center_from_qvec_tvec(query_image.qvec, query_image.tvec)
        query_ray = viewing_ray_from_camera_center(
            landmark_index.xyz[canonical_row], camera_center
        )
        if query_id not in query_observation_cache:
            query_observation_cache[query_id] = _query_track_observations(query_image)
        query_target = query_observation_cache[query_id].get(track_id)
        actual_observation = query_target is not None
        if actual_observation:
            observed_policy_rows += 1
            target_xy = np.asarray(query_target, dtype=np.float64)
            center_residual = float(np.linalg.norm(center - target_xy))
            target_is_dustbin = center_residual > float(search_radius_px)
            target_reason = "out_of_window" if target_is_dustbin else "valid_observation"
        else:
            target_xy = center.copy()
            center_residual = float("inf")
            target_is_dustbin = True
            target_reason = "track_not_observed"
        # Sparse SfM non-observation is not a reliable visibility negative.  It can
        # supervise geometric correctness through the GT pose residual, but must
        # not dominate the RGB dustbin head as a synthetic no-match label.
        if actual_observation:
            dustbin_supervision_weight = 1.0
        elif has_gt_projected_residual and gt_projected_residual > float(search_radius_px):
            dustbin_supervision_weight = 1.0
        else:
            dustbin_supervision_weight = 0.0
        reasons[target_reason] += 1

        support_indices = maplet_index.support_image_indices[canonical_row]
        posterior_count = min(view_probabilities.shape[1], len(support_indices))
        ranked_views = np.argsort(
            -view_probabilities[policy_row, :posterior_count], kind="stable"
        )
        emitted_for_policy = 0
        for view_rank in ranked_views.tolist():
            image_index = int(support_indices[int(view_rank)])
            if image_index < 0:
                continue
            support_id = str(maplet_index.support_image_ids[image_index])
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
            support_xy = geometry_index.xy[support_row]
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
            support_ray = geometry_index.viewing_rays[support_row]
            view_angle = ""
            if np.all(np.isfinite(support_ray)):
                cosine = float(np.clip(np.dot(query_ray, support_ray), -1.0, 1.0))
                view_angle = float(math.degrees(math.acos(cosine)))
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
                    if not np.isfinite(center_residual)
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
                "support_view_angle_deg": view_angle,
                "policy_row_index": int(policy_row),
                "source_query_row": int(selected_rows[policy_row]),
                "split": str(split_name),
                "support_view_rank": int(view_rank),
                "support_view_probability": float(
                    view_probabilities[policy_row, int(view_rank)]
                ),
                "assignment_score": float(assignment_scores[policy_row]),
                "pose_selection_score": float(pose_scores[policy_row]),
                "geometry_p01": (
                    ""
                    if "01" not in geometry_columns
                    else float(geometry_probabilities[policy_row, geometry_columns["01"]])
                ),
                "geometry_p02": (
                    ""
                    if "02" not in geometry_columns
                    else float(geometry_probabilities[policy_row, geometry_columns["02"]])
                ),
                "geometry_p05": float(p05[policy_row]),
                "switched_from_baseline": bool(switched[policy_row]),
                "actual_query_observation": bool(actual_observation),
                "center_residual_px": center_residual,
                "target_gt_projected_x": float(gt_projected_xy[0]),
                "target_gt_projected_y": float(gt_projected_xy[1]),
                "target_gt_projected_residual_px": (
                    "" if not has_gt_projected_residual else gt_projected_residual
                ),
                "target_gt_projection_in_front": bool(gt_projection_in_front),
                "target_gt_projection_in_image": bool(gt_projection_in_image),
                "target_geometry_correct_1px": bool(
                    has_gt_projected_residual and gt_projected_residual <= 1.0
                ),
                "target_geometry_correct_2px": bool(
                    has_gt_projected_residual and gt_projected_residual <= 2.0
                ),
                "target_geometry_correct_5px": bool(
                    has_gt_projected_residual and gt_projected_residual <= 5.0
                ),
                "dustbin_supervision_weight": float(dustbin_supervision_weight),
                "geometry_supervision_weight": float(has_gt_projected_residual),
                "target_reason": target_reason,
                "measurement_loss_weight": float(max(p05[policy_row], 1e-3)),
            }
            rows.append(row)
            emitted_for_policy += 1
            if emitted_for_policy >= int(support_views_per_row):
                break
            if max_rows is not None and len(rows) >= int(max_rows):
                break
        if emitted_for_policy == 0:
            skipped["no_usable_support_view"] += 1
        if max_rows is not None and len(rows) >= int(max_rows):
            break

    fieldnames = [*REAL_REAL_MEASUREMENT_FIELDNAMES, *SELECTED_POLICY_EXTRA_FIELDNAMES]
    output = Path(output_rows_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    finite_residuals = np.asarray(
        [float(row["center_residual_px"]) for row in rows if np.isfinite(row["center_residual_px"])],
        dtype=np.float64,
    )
    summary = {
        "stage": "selected_policy_real_rgb_measurement_rows",
        "protocol": {
            "render": False,
            "image_retrieval": False,
            "submap": False,
            "assignment_policy_frozen": True,
            "support_view_selection": "learned_posterior",
            "query_target": "actual_colmap_track_observation_target_only",
            "unobserved_or_out_of_window_target": "dustbin",
            "geometry_correct_target": "frozen_policy_gt_pose_projection_residual",
            "unobserved_dustbin_supervision": "ignored_when_gt_projection_is_in_window",
        },
        "split": str(split_name),
        "selected_policy_rows": int(selected_policy_rows),
        "actual_observation_policy_rows": int(observed_policy_rows),
        "output_rows": int(len(rows)),
        "valid_rows": int(sum(not bool(row["target_is_dustbin"]) for row in rows)),
        "dustbin_rows": int(sum(bool(row["target_is_dustbin"]) for row in rows)),
        "dustbin_supervised_rows": int(
            sum(float(row["dustbin_supervision_weight"]) > 0.0 for row in rows)
        ),
        "ambiguous_unobserved_rows": int(
            sum(float(row["dustbin_supervision_weight"]) == 0.0 for row in rows)
        ),
        "geometry_supervised_rows": int(
            sum(float(row["geometry_supervision_weight"]) > 0.0 for row in rows)
        ),
        "target_reasons": dict(sorted(reasons.items())),
        "skipped": dict(sorted(skipped.items())),
        "center_residual_median_px": (
            None if finite_residuals.size == 0 else float(np.median(finite_residuals))
        ),
        "center_residual_p90_px": (
            None if finite_residuals.size == 0 else float(np.quantile(finite_residuals, 0.9))
        ),
        "search_radius_px": float(search_radius_px),
        "context_radius_px": float(context_radius_px),
        "minimum_geometry_p05": float(minimum_geometry_p05),
        "support_views_per_row": int(support_views_per_row),
        "inputs": {
            "policy_artifact": str(policy_artifact),
            "policy_artifact_sha256": file_sha256_short(Path(policy_artifact)),
            "support_geometry_index": str(support_geometry_index),
            "support_geometry_index_sha256": expected_hashes[
                "support_geometry_index_sha256"
            ],
            "maplet_support_index": str(maplet_support_index),
            "maplet_support_index_sha256": expected_hashes[
                "maplet_support_index_sha256"
            ],
            "projected_landmark_bank": str(projected_landmark_bank),
            "projected_landmark_bank_sha256": expected_hashes[
                "projected_landmark_bank_sha256"
            ],
            "colmap_model_dir": str(colmap_model_dir),
            "colmap_images_sha256": file_sha256_short(
                Path(colmap_model_dir) / "images.bin"
            ),
            "split_json": str(split_json),
            "split_json_sha256": expected_hashes["split_json_sha256"],
            "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
            "maplet_format": maplet_metadata.get("format"),
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
