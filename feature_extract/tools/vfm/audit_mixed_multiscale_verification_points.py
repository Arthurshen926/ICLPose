"""Join SfM labels after freezing mixed verification-point proposals.

This is a diagnostic boundary: the builder writes only target-free real-image
appearance and immutable full-bank candidates.  This command evaluates their
coverage/rank only after that artifact has been frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.detector_landmark_proposals import (
    candidate_reprojection_residuals,
    nearest_visible_landmarks,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.mixed_verification_points import (
    POINT_SOURCE_ALIKE,
    POINT_SOURCE_RADIO_FINAL,
    POINT_SOURCE_RADIO_INTERMEDIATE,
    load_mixed_verification_points,
)


_ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
_RANK_BUCKET_NAMES = ("rank_1", "rank_2_5", "rank_6_10", "rank_11_20", "missing")


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not splits or len(set(splits)) != len(splits) or set(splits) - _ALLOWED_SPLITS:
        raise ValueError("audit_splits must be a non-empty unique split list")
    return splits


def _parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or any(not np.isfinite(item) or item <= 0.0 for item in thresholds)
    ):
        raise ValueError("thresholds_px must be a non-empty unique positive list")
    return thresholds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification_points", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audit_splits", default="validation")
    parser.add_argument("--thresholds_px", default="2,5,8")
    return parser.parse_args(argv)


def _pose_w2c(image) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(image.qvec)
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
    return pose


def _first_positive_ranks(residuals: np.ndarray, threshold_px: float) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(float(threshold_px)) or float(threshold_px) <= 0.0:
        raise ValueError("candidate residual ranks have invalid inputs")
    positive = values <= float(threshold_px)
    ranks = np.full((values.shape[0],), -1, dtype=np.int64)
    has_positive = np.any(positive, axis=1)
    ranks[has_positive] = np.argmax(positive[has_positive], axis=1).astype(np.int64) + 1
    return ranks


def _rank_bucket_counts(ranks: np.ndarray) -> dict[str, int]:
    values = np.asarray(ranks, dtype=np.int64).reshape(-1)
    return {
        "rank_1": int(np.count_nonzero(values == 1)),
        "rank_2_5": int(np.count_nonzero((values >= 2) & (values <= 5))),
        "rank_6_10": int(np.count_nonzero((values >= 6) & (values <= 10))),
        "rank_11_20": int(np.count_nonzero((values >= 11) & (values <= 20))),
        "missing": int(np.count_nonzero(values < 0)),
    }


def _coverage_metrics(
    *,
    nearest_residuals: np.ndarray,
    candidate_residuals: np.ndarray,
    coarse_scores: np.ndarray,
    threshold_px: float,
) -> dict[str, object]:
    nearest = np.asarray(nearest_residuals, dtype=np.float32).reshape(-1)
    residuals = np.asarray(candidate_residuals, dtype=np.float32)
    scores = np.asarray(coarse_scores, dtype=np.float32)
    if residuals.ndim != 2 or residuals.shape != scores.shape or residuals.shape[0] != len(nearest):
        raise ValueError("mixed verification coverage arrays are incompatible")
    mappable = nearest <= float(threshold_px)
    ranks = _first_positive_ranks(residuals, float(threshold_px))
    retrieved = ranks > 0
    positive_mask = residuals <= float(threshold_px)
    positive_scores = np.where(positive_mask, scores, -np.inf).max(axis=1)
    wrong_scores = np.where(~positive_mask & np.isfinite(residuals), scores, -np.inf).max(axis=1)
    finite_pairs = np.isfinite(positive_scores) & np.isfinite(wrong_scores)
    score_margins = positive_scores[finite_pairs] - wrong_scores[finite_pairs]
    return {
        "point_count": int(len(nearest)),
        "mappable_point_count": int(np.count_nonzero(mappable)),
        "mappable_point_rate": float(np.mean(mappable)) if len(mappable) else 0.0,
        "proposal_positive_point_count": int(np.count_nonzero(retrieved)),
        "proposal_positive_point_rate": float(np.mean(retrieved)) if len(retrieved) else 0.0,
        "recall_at_1_given_mappable": (
            0.0 if not np.any(mappable) else float(np.mean(ranks[mappable] == 1))
        ),
        "recall_at_5_given_mappable": (
            0.0 if not np.any(mappable) else float(np.mean((ranks[mappable] > 0) & (ranks[mappable] <= 5)))
        ),
        "recall_at_10_given_mappable": (
            0.0 if not np.any(mappable) else float(np.mean((ranks[mappable] > 0) & (ranks[mappable] <= 10)))
        ),
        "recall_at_20_given_mappable": (
            0.0 if not np.any(mappable) else float(np.mean(retrieved[mappable]))
        ),
        "median_first_positive_rank_when_retrieved": (
            None if not np.any(retrieved) else float(np.median(ranks[retrieved]))
        ),
        "rank_buckets": _rank_bucket_counts(ranks),
        "coarse_positive_vs_best_wrong_pair_count": int(np.count_nonzero(finite_pairs)),
        "coarse_positive_beats_best_wrong_rate": (
            None if not np.any(finite_pairs) else float(np.mean(score_margins > 0.0))
        ),
        "coarse_positive_minus_best_wrong_median": (
            None if not len(score_margins) else float(np.median(score_margins))
        ),
    }


def _metrics_by_source(
    *,
    point_sources: np.ndarray,
    nearest_residuals: np.ndarray,
    candidate_residuals: np.ndarray,
    coarse_scores: np.ndarray,
    thresholds_px: Sequence[float],
) -> dict[str, object]:
    sources = np.asarray(point_sources).astype(str).reshape(-1)
    result: dict[str, object] = {}
    for source in (
        "all",
        POINT_SOURCE_ALIKE,
        POINT_SOURCE_RADIO_INTERMEDIATE,
        POINT_SOURCE_RADIO_FINAL,
    ):
        mask = np.ones((len(sources),), dtype=bool) if source == "all" else sources == source
        result[source] = {
            "point_count": int(np.count_nonzero(mask)),
            "thresholds": {
                f"{float(threshold):g}": _coverage_metrics(
                    nearest_residuals=nearest_residuals[mask],
                    candidate_residuals=candidate_residuals[mask],
                    coarse_scores=coarse_scores[mask],
                    threshold_px=float(threshold),
                )
                for threshold in thresholds_px
            },
        }
    return result


def audit_mixed_multiscale_verification_points(
    *,
    verification_points: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    output_dir: Path,
    audit_splits: Sequence[str],
    thresholds_px: Sequence[float],
) -> dict[str, object]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite mixed verification audit output")
    requested_splits = set(audit_splits)
    points = load_mixed_verification_points(Path(verification_points))
    if not requested_splits <= set(points.split_names.tolist()):
        raise ValueError("mixed verification artifact does not contain every requested audit split")
    bank, bank_metadata = load_landmark_index_npz(Path(projected_landmark_bank))
    if str(points.metadata.get("projected_landmark_bank_sha256", "")) != str(
        file_sha256_short(Path(projected_landmark_bank))
    ):
        raise ValueError("mixed verification artifact references a different landmark bank")
    if str(points.metadata.get("descriptor_space_id", "")) != str(
        bank_metadata.get("descriptor_space_id", "")
    ):
        raise ValueError("mixed verification artifact descriptor space differs from bank")
    valid_candidates = points.candidate_track_ids >= 0
    safe_rows = np.maximum(points.candidate_bank_rows, 0)
    if np.any(
        bank.track_ids[safe_rows[valid_candidates]]
        != points.candidate_track_ids[valid_candidates]
    ):
        raise ValueError("mixed verification candidate tracks differ from landmark bank")

    selected = np.isin(points.split_names, tuple(requested_splits))
    if not np.any(selected):
        raise ValueError("mixed verification audit has no requested points")
    model_dir = Path(colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    selected_indices = np.flatnonzero(selected)
    nearest_rows = np.full((len(selected_indices),), -1, dtype=np.int64)
    nearest_tracks = np.full((len(selected_indices),), -1, dtype=np.int64)
    nearest_residuals = np.full((len(selected_indices),), np.inf, dtype=np.float32)
    candidate_residuals = np.full(
        (len(selected_indices), points.candidate_track_ids.shape[1]), np.inf, dtype=np.float32
    )
    local_index = {int(point_index): offset for offset, point_index in enumerate(selected_indices.tolist())}
    target_query_ids = sorted(set(points.query_ids[selected].tolist()))
    for query_id in target_query_ids:
        image = images_by_name.get(str(query_id))
        if image is None:
            raise KeyError(f"COLMAP model is missing query image {query_id}")
        query_indices = selected_indices[points.query_ids[selected_indices] == str(query_id)]
        local_rows = np.asarray([local_index[int(value)] for value in query_indices], dtype=np.int64)
        camera = cameras[int(image.camera_id)]
        nearest_bank, nearest_track, nearest_error = nearest_visible_landmarks(
            points.xy[query_indices], bank, _pose_w2c(image), camera
        )
        residuals = candidate_reprojection_residuals(
            points.xy[query_indices],
            points.candidate_bank_rows[query_indices],
            bank,
            _pose_w2c(image),
            camera,
        )
        nearest_rows[local_rows] = nearest_bank
        nearest_tracks[local_rows] = nearest_track
        nearest_residuals[local_rows] = nearest_error
        candidate_residuals[local_rows] = residuals
    metrics = _metrics_by_source(
        point_sources=points.point_sources[selected],
        nearest_residuals=nearest_residuals,
        candidate_residuals=candidate_residuals,
        coarse_scores=points.candidate_coarse_similarities[selected],
        thresholds_px=thresholds_px,
    )
    per_split: dict[str, object] = {}
    for split in sorted(requested_splits):
        mask = points.split_names[selected] == split
        per_split[split] = _metrics_by_source(
            point_sources=points.point_sources[selected][mask],
            nearest_residuals=nearest_residuals[mask],
            candidate_residuals=candidate_residuals[mask],
            coarse_scores=points.candidate_coarse_similarities[selected][mask],
            thresholds_px=thresholds_px,
        )
    summary: dict[str, object] = {
        "stage": "audit_frozen_mixed_multiscale_verification_point_coverage",
        "inputs": {
            "verification_points": str(Path(verification_points)),
            "verification_points_sha256": file_sha256_short(Path(verification_points)),
            "projected_landmark_bank": str(Path(projected_landmark_bank)),
            "projected_landmark_bank_sha256": file_sha256_short(Path(projected_landmark_bank)),
            "colmap_model_dir": str(model_dir),
            "colmap_cameras_sha256": file_sha256_short(model_dir / "cameras.bin"),
            "colmap_images_sha256": file_sha256_short(model_dir / "images.bin"),
        },
        "audit_target_splits": sorted(requested_splits),
        "thresholds_px": [float(value) for value in thresholds_px],
        "target_query_count": int(len(target_query_ids)),
        "target_point_count": int(len(selected_indices)),
        "all_requested_split_metrics": metrics,
        "per_split_metrics": per_split,
        "protocol": {
            "prediction_artifact_target_free": True,
            "candidate_set_frozen_before_target_join": True,
            "target_labels_joined_after_freeze": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "materialized_target_splits": sorted(requested_splits),
            "test_target_labels_materialized": "test" in requested_splits,
            "rank_bucket_names": list(_RANK_BUCKET_NAMES),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = audit_mixed_multiscale_verification_points(
        verification_points=Path(args.verification_points),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        colmap_model_dir=Path(args.colmap_model_dir),
        output_dir=Path(args.output_dir),
        audit_splits=_parse_splits(args.audit_splits),
        thresholds_px=_parse_thresholds(args.thresholds_px),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
