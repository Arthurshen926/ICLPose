"""Pose-level evaluation helpers for real-image RADIO localization."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from feature_extract.vfm.cambridge_pose_lattice import CambridgePoseRecord
from feature_extract.vfm.colmap_tracks import ColmapCamera, ColmapImageObservation, ColmapTrackObservation
from feature_extract.vfm.localization.model import SelectorCoarseMeasurementModel
from feature_extract.vfm.localization.pipeline import (
    RealRadioLocalizationPair,
    _load_feature_map,
    _load_rgb_chw,
    _resolve_path,
)
from feature_extract.vfm.localization.schemas import CoarseProposal, MeasurementResult
from feature_extract.vfm.measurement_v1.rgb_patch_pose_proxy import scaled_colmap_camera
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    SpatialDiversityPnPConfig,
    estimate_pose_pnp_ransac,
    match_spatial_distribution_stats,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
    select_pnp_matches_by_spatial_diversity,
)


@dataclass(frozen=True)
class NearestSupportObservation:
    observation: ColmapTrackObservation
    distance_px: float


class SupportObservationIndex:
    """Nearest-neighbor lookup over COLMAP observations grouped by image id."""

    def __init__(self, observations: Sequence[ColmapTrackObservation]) -> None:
        by_image: dict[str, list[ColmapTrackObservation]] = {}
        for observation in observations:
            by_image.setdefault(str(observation.image_id), []).append(observation)
        self._by_image = {image_id: tuple(items) for image_id, items in by_image.items()}
        self._xy_by_image = {
            image_id: np.asarray([item.xy for item in items], dtype=np.float64)
            for image_id, items in self._by_image.items()
        }

    def nearest(
        self,
        image_id: str,
        xy: np.ndarray | Sequence[float],
        *,
        max_distance_px: float,
    ) -> NearestSupportObservation | None:
        items = self._by_image.get(str(image_id))
        if not items:
            return None
        query_xy = np.asarray(xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(query_xy)):
            return None
        distances = np.linalg.norm(self._xy_by_image[str(image_id)] - query_xy[None, :], axis=1)
        best = int(np.argmin(distances))
        distance = float(distances[best])
        if distance > float(max_distance_px):
            return None
        return NearestSupportObservation(observation=items[best], distance_px=distance)


def build_support_observation_index(observations: Sequence[ColmapTrackObservation]) -> SupportObservationIndex:
    return SupportObservationIndex(observations)


@dataclass(frozen=True)
class ClosedLoopProposalRecord:
    query_id: str
    reference_image_id: str
    proposal: CoarseProposal
    measurement: MeasurementResult | None
    proposal_index: int


def _score_for_pnp(record: ClosedLoopProposalRecord) -> float:
    if record.measurement is not None and record.measurement.confidence is not None:
        return float(record.measurement.confidence)
    if record.proposal.confidence is not None:
        return float(record.proposal.confidence)
    return float(record.proposal.score)


def _measured_query_xy(record: ClosedLoopProposalRecord) -> np.ndarray:
    if record.measurement is not None:
        return np.asarray(record.measurement.measured_query_xy, dtype=np.float64).reshape(2)
    return np.asarray(record.proposal.query_xy, dtype=np.float64).reshape(2)


def _measured_reference_xy(record: ClosedLoopProposalRecord) -> np.ndarray:
    if record.measurement is not None:
        return np.asarray(record.measurement.measured_reference_xy, dtype=np.float64).reshape(2)
    return np.asarray(record.proposal.reference_xy, dtype=np.float64).reshape(2)


def convert_proposals_to_query_3d_matches(
    records: Sequence[ClosedLoopProposalRecord],
    *,
    observation_index: SupportObservationIndex,
    max_support_distance_px: float,
) -> tuple[list[dict[str, Any]], dict[str, list[QueryTo3DMatch]]]:
    rows: list[dict[str, Any]] = []
    matches_by_query: dict[str, list[QueryTo3DMatch]] = {}
    for record in records:
        query_xy = _measured_query_xy(record)
        reference_xy = _measured_reference_xy(record)
        nearest = observation_index.nearest(
            record.reference_image_id,
            reference_xy,
            max_distance_px=float(max_support_distance_px),
        )
        score = float(_score_for_pnp(record))
        base_row: dict[str, Any] = {
            "query_id": str(record.query_id),
            "reference_image_id": str(record.reference_image_id),
            "proposal_index": int(record.proposal_index),
            "query_x": float(query_xy[0]),
            "query_y": float(query_xy[1]),
            "reference_x": float(reference_xy[0]),
            "reference_y": float(reference_xy[1]),
            "score": score,
        }
        if nearest is None:
            rows.append({**base_row, "association_status": "missing_support_observation"})
            continue
        observation = nearest.observation
        match = QueryTo3DMatch(
            token_index=int(record.proposal_index),
            xy=query_xy.astype(np.float64, copy=False),
            track_id=int(observation.track_id),
            xyz=np.asarray(observation.xyz, dtype=np.float64).reshape(3),
            similarity=score,
            ratio=1.0,
            landmark_variance=0.0,
            source="real_radio_closed_loop",
            observation_count=int(observation.track_length),
            landmark_reprojection_error=float(observation.reprojection_error),
            pnp_soft_score=score,
            patch_offset_confidence=None if record.measurement is None else record.measurement.confidence,
            measurement_sigma_px=None if record.measurement is None else record.measurement.uncertainty_px,
            coarse_rank=record.proposal.rank,
            coarse_score=float(record.proposal.score),
        )
        matches_by_query.setdefault(str(record.query_id), []).append(match)
        rows.append(
            {
                **base_row,
                "association_status": "matched",
                "track_id": int(observation.track_id),
                "support_observation_distance_px": float(nearest.distance_px),
                "support_track_length": int(observation.track_length),
                "support_reprojection_error": float(observation.reprojection_error),
            }
        )
    return rows, matches_by_query


def deduplicate_query_3d_matches(matches: Sequence[QueryTo3DMatch]) -> list[QueryTo3DMatch]:
    best: dict[int, QueryTo3DMatch] = {}
    for match in matches:
        key = int(match.track_id)
        existing = best.get(key)
        current_score = 0.0 if match.pnp_soft_score is None else float(match.pnp_soft_score)
        existing_score = -float("inf") if existing is None or existing.pnp_soft_score is None else float(existing.pnp_soft_score)
        if existing is None or current_score > existing_score:
            best[key] = match
    return [best[key] for key in sorted(best)]


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = [float(row[key]) for row in rows if np.isfinite(float(row.get(key, float("inf"))))]
    return np.asarray(values, dtype=np.float64)


def summarize_pose_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    success_rows = [row for row in values if bool(row.get("success", False))]
    translations = _finite_values(success_rows, "translation_error_m")
    rotations = _finite_values(success_rows, "rotation_error_deg")
    failure_counts: dict[str, int] = {}
    for row in values:
        if bool(row.get("success", False)):
            continue
        reason = str(row.get("failure_reason", "") or "unknown")
        failure_counts[reason] = failure_counts.get(reason, 0) + 1

    def percentile(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if arr.size else float("inf")

    def recall(max_t: float, max_r: float) -> float:
        if not values:
            return 0.0
        passed = sum(
            1
            for row in success_rows
            if float(row.get("translation_error_m", float("inf"))) <= max_t
            and float(row.get("rotation_error_deg", float("inf"))) <= max_r
        )
        return float(passed / len(values))

    match_counts = np.asarray([float(row.get("match_count", 0)) for row in values], dtype=np.float64)
    inlier_counts = np.asarray([float(row.get("inlier_count", 0)) for row in values], dtype=np.float64)
    return {
        "query_count": int(len(values)),
        "success_count": int(len(success_rows)),
        "success_rate": float(len(success_rows) / len(values)) if values else 0.0,
        "median_translation_error_m": percentile(translations, 50.0),
        "translation_error_p90_m": percentile(translations, 90.0),
        "median_rotation_error_deg": percentile(rotations, 50.0),
        "rotation_error_p90_deg": percentile(rotations, 90.0),
        "recall_0_25m_2deg": recall(0.25, 2.0),
        "recall_0_5m_5deg": recall(0.5, 5.0),
        "recall_5m_10deg": recall(5.0, 10.0),
        "median_match_count": float(np.median(match_counts)) if match_counts.size else 0.0,
        "median_inlier_count": float(np.median(inlier_counts)) if inlier_counts.size else 0.0,
        "failure_counts": failure_counts,
    }


def evaluate_query_poses(
    matches_by_query: Mapping[str, Sequence[QueryTo3DMatch]],
    *,
    cameras_by_query: Mapping[str, ColmapCamera],
    gt_poses_by_query: Mapping[str, CambridgePoseRecord],
    pnp_reprojection_error_px: float = 8.0,
    pnp_iterations: int = 1000,
    pnp_confidence: float = 0.999,
    pnp_min_inliers: int = 4,
    spatial_diversity: SpatialDiversityPnPConfig | None = None,
) -> list[dict[str, Any]]:
    pose_rows: list[dict[str, Any]] = []
    for query_id in sorted(matches_by_query):
        raw_matches = deduplicate_query_3d_matches(matches_by_query[query_id])
        camera = cameras_by_query.get(query_id)
        gt = gt_poses_by_query.get(query_id)
        if camera is None:
            pose_rows.append(
                {
                    "query_id": query_id,
                    "success": False,
                    "match_count": 0,
                    "inlier_count": 0,
                    "translation_error_m": float("inf"),
                    "rotation_error_deg": float("inf"),
                    "failure_reason": "missing_camera",
                }
            )
            continue
        if gt is None:
            pose_rows.append(
                {
                    "query_id": query_id,
                    "success": False,
                    "match_count": 0,
                    "inlier_count": 0,
                    "translation_error_m": float("inf"),
                    "rotation_error_deg": float("inf"),
                    "failure_reason": "missing_gt_pose",
                }
            )
            continue
        matches = raw_matches
        if spatial_diversity is not None:
            matches = select_pnp_matches_by_spatial_diversity(
                matches,
                int(camera.width),
                int(camera.height),
                spatial_diversity,
            )
        if len(matches) < int(pnp_min_inliers):
            pose_rows.append(
                {
                    "query_id": query_id,
                    "success": False,
                    "match_count": int(len(matches)),
                    "inlier_count": 0,
                    "translation_error_m": float("inf"),
                    "rotation_error_deg": float("inf"),
                    "failure_reason": "insufficient_matches",
                }
            )
            continue
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=float(pnp_reprojection_error_px),
            confidence=float(pnp_confidence),
            iterations=int(pnp_iterations),
            min_inliers=int(pnp_min_inliers),
            refine_method="LM",
        )
        error = pnp_pose_error(pnp.pose_w2c if pnp.success else None, gt.pose_w2c)
        spatial_all = match_spatial_distribution_stats(matches, int(camera.width), int(camera.height))
        spatial_inliers = match_spatial_distribution_stats(
            matches,
            int(camera.width),
            int(camera.height),
            pnp.inlier_mask,
        )
        residuals = pnp_reprojection_residual_stats(matches, pnp.pose_w2c, camera, inlier_mask=pnp.inlier_mask)
        pose_rows.append(
            {
                "query_id": query_id,
                "success": bool(pnp.success),
                "match_count": int(len(matches)),
                "inlier_count": int(pnp.inlier_count),
                "inlier_ratio": float(pnp.inlier_ratio),
                "translation_error_m": float(error.translation_m),
                "rotation_error_deg": float(error.rotation_deg),
                "failure_reason": "" if pnp.success else "pnp_failed",
                "all_grid_4x4_occupancy_frac": spatial_all.get("grid_4x4_occupancy_frac"),
                "inlier_grid_4x4_occupancy_frac": spatial_inliers.get("grid_4x4_occupancy_frac"),
                **residuals,
            }
        )
    return pose_rows


def write_mapping_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            name = str(key)
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_mapping_rows_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _cameras_by_query_from_colmap(
    *,
    cameras: Mapping[int, ColmapCamera],
    colmap_images: Mapping[int, ColmapImageObservation],
    image_root: Path,
) -> dict[str, ColmapCamera]:
    out: dict[str, ColmapCamera] = {}
    for image in colmap_images.values():
        camera = cameras.get(int(image.camera_id))
        if camera is None:
            continue
        image_path = Path(image_root) / image.image_name
        if image_path.exists():
            width, height = _image_size(image_path)
            out[str(image.image_name)] = scaled_colmap_camera(camera, image_width=width, image_height=height)
        else:
            out[str(image.image_name)] = camera
    return out


def _bridge_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    matched = [row for row in rows if row.get("association_status") == "matched"]
    distances = _finite_values(matched, "support_observation_distance_px")
    return {
        "proposal_count": int(total),
        "associated_match_count": int(len(matched)),
        "association_rate": float(len(matched) / total) if total else 0.0,
        "support_observation_distance_median_px": float(np.median(distances)) if distances.size else None,
        "support_observation_distance_p90_px": float(np.percentile(distances, 90.0)) if distances.size else None,
    }


def run_real_radio_pose_localization_eval(
    pairs: Sequence[RealRadioLocalizationPair],
    *,
    image_root: Path,
    feature_root: Path,
    output_dir: Path,
    feature_mapper,
    coarse_matcher,
    measurement_branch,
    cameras: Mapping[int, ColmapCamera],
    colmap_images: Mapping[int, ColmapImageObservation],
    colmap_observations: Sequence[ColmapTrackObservation],
    gt_poses_by_query: Mapping[str, CambridgePoseRecord],
    feature_key: str = "",
    max_pairs: int | None = None,
    max_support_distance_px: float = 6.0,
    pnp_reprojection_error_px: float = 8.0,
    pnp_iterations: int = 1000,
    pnp_confidence: float = 0.999,
    pnp_min_inliers: int = 4,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_pairs = list(pairs)
    if max_pairs is not None:
        selected_pairs = selected_pairs[: int(max_pairs)]

    model = SelectorCoarseMeasurementModel(feature_mapper, coarse_matcher, measurement_branch)
    records: list[ClosedLoopProposalRecord] = []
    proposal_rows: list[dict[str, Any]] = []
    for pair in selected_pairs:
        query_rgb = _load_rgb_chw(Path(image_root) / pair.query_id)
        reference_rgb = _load_rgb_chw(Path(image_root) / pair.reference_image_id)
        query_feature = _load_feature_map(
            _resolve_path(pair.query_feature_path, base_dir=Path(feature_root)),
            key=str(feature_key),
        )
        reference_feature = _load_feature_map(
            _resolve_path(pair.reference_feature_path, base_dir=Path(feature_root)),
            key=str(feature_key),
        )
        result = model.match_pair(
            query_feature,
            reference_feature,
            query_image_size=(int(query_rgb.shape[2]), int(query_rgb.shape[1])),
            reference_image_size=(int(reference_rgb.shape[2]), int(reference_rgb.shape[1])),
            query_rgb=query_rgb,
            reference_rgb=reference_rgb,
        )
        measurements = {
            (id(item.proposal), int(item.proposal.query_index), int(item.proposal.reference_index)): item
            for item in result.measurements
        }
        for proposal_index, proposal in enumerate(result.coarse_proposals):
            measurement = measurements.get((id(proposal), int(proposal.query_index), int(proposal.reference_index)))
            records.append(
                ClosedLoopProposalRecord(
                    query_id=pair.query_id,
                    reference_image_id=pair.reference_image_id,
                    proposal=proposal,
                    measurement=measurement,
                    proposal_index=proposal_index,
                )
            )
            proposal_rows.append(
                {
                    "query_id": pair.query_id,
                    "reference_image_id": pair.reference_image_id,
                    "proposal_index": int(proposal_index),
                    "query_x": float(proposal.query_xy[0]),
                    "query_y": float(proposal.query_xy[1]),
                    "reference_x": float(proposal.reference_xy[0]),
                    "reference_y": float(proposal.reference_xy[1]),
                    "coarse_score": float(proposal.score),
                    "measurement_confidence": (
                        "" if measurement is None or measurement.confidence is None else float(measurement.confidence)
                    ),
                }
            )

    observation_index = build_support_observation_index(colmap_observations)
    match_rows, matches_by_query = convert_proposals_to_query_3d_matches(
        records,
        observation_index=observation_index,
        max_support_distance_px=float(max_support_distance_px),
    )
    cameras_by_query = _cameras_by_query_from_colmap(cameras=cameras, colmap_images=colmap_images, image_root=Path(image_root))
    pose_rows = evaluate_query_poses(
        matches_by_query,
        cameras_by_query=cameras_by_query,
        gt_poses_by_query=gt_poses_by_query,
        pnp_reprojection_error_px=float(pnp_reprojection_error_px),
        pnp_iterations=int(pnp_iterations),
        pnp_confidence=float(pnp_confidence),
        pnp_min_inliers=int(pnp_min_inliers),
    )

    write_mapping_rows_csv(output / "proposals.csv", proposal_rows)
    write_mapping_rows_jsonl(output / "proposals.jsonl", proposal_rows)
    write_mapping_rows_csv(output / "matches_2d3d.csv", match_rows)
    write_mapping_rows_jsonl(output / "matches_2d3d.jsonl", match_rows)
    write_mapping_rows_csv(output / "pose_rows.csv", pose_rows)
    write_mapping_rows_jsonl(output / "pose_rows.jsonl", pose_rows)
    summary = {
        "stage": "real_radio_pose_localization",
        "pair_count": int(len(selected_pairs)),
        "bridge": _bridge_summary(match_rows),
        "pose": summarize_pose_rows(pose_rows),
        "max_support_distance_px": float(max_support_distance_px),
        "pnp_reprojection_error_px": float(pnp_reprojection_error_px),
        "pnp_iterations": int(pnp_iterations),
        "pnp_confidence": float(pnp_confidence),
        "pnp_min_inliers": int(pnp_min_inliers),
        "outputs": {
            "proposals_csv": str(output / "proposals.csv"),
            "proposals_jsonl": str(output / "proposals.jsonl"),
            "matches_2d3d_csv": str(output / "matches_2d3d.csv"),
            "matches_2d3d_jsonl": str(output / "matches_2d3d.jsonl"),
            "pose_rows_csv": str(output / "pose_rows.csv"),
            "pose_rows_jsonl": str(output / "pose_rows.jsonl"),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
