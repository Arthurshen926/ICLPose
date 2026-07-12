"""Coverage and pose oracle diagnostics for grouped top-L landmark proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera, ColmapImageObservation, qvec_to_rotmat
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)


@dataclass(frozen=True)
class TopLOracleObservation:
    query_id: str
    track_id: int
    correct_rank: int | None
    xy: np.ndarray
    xyz: np.ndarray


def _grid_coverage(observations: Sequence[TopLOracleObservation], *, width: int, height: int) -> float:
    if not observations or int(width) <= 0 or int(height) <= 0:
        return 0.0
    occupied = set()
    for observation in observations:
        col = int(np.clip(np.floor(float(observation.xy[0]) / float(width) * 4.0), 0, 3))
        row = int(np.clip(np.floor(float(observation.xy[1]) / float(height) * 4.0), 0, 3))
        occupied.add((row, col))
    return float(len(occupied)) / 16.0


def summarize_topl_oracle(
    observations: Sequence[TopLOracleObservation],
    *,
    top_ls: Sequence[int],
    cameras_by_query: Mapping[str, ColmapCamera] | None = None,
    images_by_query: Mapping[str, ColmapImageObservation] | None = None,
    reprojection_error_px: float = 4.0,
    iterations: int = 1000,
) -> dict[str, object]:
    values = list(observations)
    query_ids = sorted({str(observation.query_id) for observation in values})
    by_query = {
        query_id: [observation for observation in values if str(observation.query_id) == query_id]
        for query_id in query_ids
    }
    output: dict[str, object] = {
        "observation_count": int(len(values)),
        "query_count": int(len(query_ids)),
        "top_l": {},
    }
    for top_l in top_ls:
        limit = int(top_l)
        if limit <= 0:
            raise ValueError("top_ls must contain positive integers")
        positive_counts: list[int] = []
        coverages: list[float] = []
        eligible_count = 0
        pnp_success_count = 0
        translation_errors: list[float] = []
        rotation_errors: list[float] = []
        selected_total = 0
        for query_id in query_ids:
            selected = [
                observation
                for observation in by_query[query_id]
                if observation.correct_rank is not None and int(observation.correct_rank) <= limit
            ]
            deduplicated: list[TopLOracleObservation] = []
            seen_tracks: set[int] = set()
            for observation in selected:
                if int(observation.track_id) in seen_tracks:
                    continue
                seen_tracks.add(int(observation.track_id))
                deduplicated.append(observation)
            selected = deduplicated
            selected_total += int(len(selected))
            positive_counts.append(int(len(selected)))
            image = None if images_by_query is None else images_by_query.get(query_id)
            camera = None if cameras_by_query is None else cameras_by_query.get(query_id)
            width = 0 if camera is None else int(camera.width)
            height = 0 if camera is None else int(camera.height)
            coverages.append(_grid_coverage(selected, width=width, height=height))
            if len(selected) < 4:
                continue
            eligible_count += 1
            if image is None or camera is None:
                continue
            matches = [
                QueryTo3DMatch(
                    token_index=index,
                    xy=np.asarray(observation.xy, dtype=np.float64),
                    track_id=int(observation.track_id),
                    xyz=np.asarray(observation.xyz, dtype=np.float64),
                    similarity=1.0,
                    ratio=0.0,
                    landmark_variance=0.0,
                    source="topl_correct_track_oracle",
                )
                for index, observation in enumerate(selected)
            ]
            pnp = estimate_pose_pnp_ransac(
                matches,
                camera,
                reprojection_error_px=float(reprojection_error_px),
                iterations=int(iterations),
                min_inliers=4,
                refine_method="LM",
            )
            if not pnp.success or pnp.pose_w2c is None:
                continue
            pnp_success_count += 1
            gt_pose = np.eye(4, dtype=np.float64)
            gt_pose[:3, :3] = qvec_to_rotmat(image.qvec)
            gt_pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64).reshape(3)
            error = pnp_pose_error(pnp.pose_w2c, gt_pose)
            translation_errors.append(float(error.translation_m))
            rotation_errors.append(float(error.rotation_deg))
        counts = np.asarray(positive_counts, dtype=np.float64)
        coverage_values = np.asarray(coverages, dtype=np.float64)
        pnp_denominator = max(int(len(query_ids)), 1)
        output["top_l"][str(limit)] = {
            "correct_observation_recall": 0.0 if not values else float(selected_total / len(values)),
            "mean_positive_proposal_count": 0.0 if counts.size == 0 else float(counts.mean()),
            "median_positive_proposal_count": 0.0 if counts.size == 0 else float(np.median(counts)),
            "query_with_at_least_4_positive_rate": float(eligible_count / pnp_denominator),
            "median_grid_coverage_4x4": 0.0 if coverage_values.size == 0 else float(np.median(coverage_values)),
            "oracle_pnp_success_rate_all_queries": float(pnp_success_count / pnp_denominator),
            "oracle_pnp_success_rate_eligible": (
                0.0 if eligible_count == 0 else float(pnp_success_count / eligible_count)
            ),
            "oracle_translation_median_m": (
                None if not translation_errors else float(np.median(np.asarray(translation_errors)))
            ),
            "oracle_rotation_median_deg": (
                None if not rotation_errors else float(np.median(np.asarray(rotation_errors)))
            ),
        }
    return output
