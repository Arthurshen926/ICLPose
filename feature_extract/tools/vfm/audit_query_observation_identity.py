"""Audit exact query-observation track supervision and its PnP oracle value."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.probe_detector_maplet_geometry import _identity_metrics
from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    canonical_rows_for_track_candidates,
)
from feature_extract.vfm.localization.query_observation_identity import (
    RegisteredQueryObservationTargets,
    registered_candidate_identity_labels,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


def _float_list(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed or any(item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("expected positive comma-separated floats")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--feature_artifact", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_strategy", default="alike_support_top2_mean")
    parser.add_argument("--observation_thresholds_px", type=_float_list, default=(1.0, 2.0, 3.0, 5.0))
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _compact(values: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return np.take_along_axis(np.asarray(values)[rows], columns, axis=1)


def _subset_targets(
    targets: RegisteredQueryObservationTargets, mask: np.ndarray
) -> RegisteredQueryObservationTargets:
    keep = np.asarray(mask, dtype=bool)
    return RegisteredQueryObservationTargets(
        track_ids=targets.track_ids[keep],
        distances_px=targets.distances_px[keep],
        supervised=targets.supervised[keep],
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(Path(args.proposals), allow_pickle=False) as data:
        proposals = {key: np.asarray(data[key]) for key in data.files}
    with np.load(Path(args.feature_artifact), allow_pickle=False) as data:
        selected_rows = np.asarray(data["selected_rows"], dtype=np.int64)
        selected_columns = np.asarray(data["selected_columns"], dtype=np.int64)
    if np.any(selected_columns < 0):
        raise ValueError("identity audit requires a fixed valid candidate pool")
    split = json.loads(Path(args.split_json).read_text())
    for name in ("train", "validation", "test"):
        if name not in split:
            raise ValueError(f"split is missing {name}")

    query_ids = np.asarray(proposals["query_ids"])[selected_rows].astype(str)
    query_xy = np.asarray(proposals["xy"], dtype=np.float32)[selected_rows]
    candidate_tracks = _compact(
        proposals["candidate_track_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    candidate_prototypes = _compact(
        proposals["candidate_prototype_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    candidate_residuals = _compact(
        proposals["candidate_gt_residuals_px"], selected_rows, selected_columns
    ).astype(np.float32)
    baseline_scores = _compact(
        proposals[f"strategy__{args.baseline_strategy}"], selected_rows, selected_columns
    ).astype(np.float32)
    nearest_residuals = np.asarray(
        proposals["nearest_visible_residuals_px"], dtype=np.float32
    )[selected_rows]
    nearest_tracks = np.asarray(
        proposals["nearest_visible_track_ids"], dtype=np.int64
    )[selected_rows]

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    landmark_index, _metadata = load_landmark_index_npz(
        Path(args.projected_landmark_bank)
    )
    canonical = canonical_rows_for_track_candidates(
        candidate_tracks, landmark_index.track_ids
    )
    candidates = UniqueTrackCandidateSet(
        canonical,
        candidate_tracks,
        candidate_prototypes,
        baseline_scores,
    )
    query_observations = [
        ColmapTrackObservation(
            track_id=int(nearest_tracks[row]),
            image_id=str(query_ids[row]),
            point2d_idx=int(selected_rows[row]),
            xy=(float(query_xy[row, 0]), float(query_xy[row, 1])),
            xyz=np.zeros((3,), dtype=np.float64),
            track_length=1,
            reprojection_error=0.0,
        )
        for row in range(len(query_ids))
    ]

    max_threshold = max(float(value) for value in args.observation_thresholds_px)
    nearest_targets = registered_query_observation_targets(
        query_ids=query_ids,
        query_xy=query_xy,
        images_by_name=images_by_name,
        max_distance_px=max_threshold,
    )
    summary: dict[str, object] = {
        "stage": "registered_query_observation_identity_audit",
        "protocol": {
            "query_pose_used": False,
            "query_registered_observation_identity_target_only": True,
            "render": False,
            "image_retrieval": False,
            "test_used_for_selection": False,
        },
        "thresholds_px": {},
    }
    pose_rows: dict[str, object] = {}
    for threshold in args.observation_thresholds_px:
        supervised = nearest_targets.distances_px <= float(threshold)
        threshold_targets = RegisteredQueryObservationTargets(
            track_ids=np.where(supervised, nearest_targets.track_ids, -1),
            distances_px=np.where(supervised, nearest_targets.distances_px, np.inf),
            supervised=supervised,
        )
        labels = registered_candidate_identity_labels(candidate_tracks, threshold_targets)
        threshold_summary: dict[str, object] = {"splits": {}}
        exact_scores = np.full(baseline_scores.shape, -np.inf, dtype=np.float32)
        exact_scores[labels] = 1.0
        hybrid_scores = baseline_scores.copy()
        rows_with_exact = np.any(labels, axis=1)
        hybrid_scores[rows_with_exact] = -np.inf
        hybrid_scores[labels] = 1.0
        for split_name in ("train", "validation", "test", "all"):
            split_mask = (
                np.ones((len(query_ids),), dtype=bool)
                if split_name == "all"
                else np.isin(query_ids, np.asarray(split[split_name], dtype=np.str_))
            )
            split_targets = _subset_targets(threshold_targets, split_mask)
            identity_summary = summarize_registered_candidate_identity(
                labels[split_mask], split_targets
            )
            identity_summary["registered_observation_distance_median_px"] = (
                None
                if not np.any(split_targets.supervised)
                else float(
                    np.median(
                        split_targets.distances_px[split_targets.supervised]
                    )
                )
            )
            threshold_summary["splits"][split_name] = identity_summary
            if split_name == "all":
                continue
            rows = np.flatnonzero(split_mask)
            subset = UniqueTrackCandidateSet(
                candidates.bank_row_indices[rows],
                candidates.track_ids[rows],
                candidates.prototype_ids[rows],
                candidates.coarse_scores[rows],
            )
            threshold_summary["splits"][split_name]["pose"] = {}
            for strategy, scores in (
                ("baseline", baseline_scores),
                ("exact_identity_only_oracle", exact_scores),
                ("exact_identity_hybrid_oracle", hybrid_scores),
            ):
                pose_summary, rows_output = _evaluate_pose_strategy(
                    strategy=f"registered_{threshold:g}px_{strategy}_{split_name}",
                    scores=scores[rows],
                    candidates=subset,
                    query_observations=[query_observations[int(row)] for row in rows],
                    query_ids=query_ids[rows].tolist(),
                    landmark_index=landmark_index,
                    cameras=cameras,
                    images_by_name=images_by_name,
                    reprojection_error_px=float(args.pnp_reprojection_error_px),
                    iterations=int(args.pnp_iterations),
                )
                threshold_summary["splits"][split_name]["pose"][strategy] = pose_summary
                pose_rows[f"{threshold:g}px/{split_name}/{strategy}"] = rows_output
            threshold_summary["splits"][split_name]["hybrid_geometry_identity"] = (
                _identity_metrics(
                    nearest_residuals=nearest_residuals[split_mask],
                    candidate_residuals=candidate_residuals[split_mask],
                    scores=hybrid_scores[split_mask],
                    query_ids=query_ids[split_mask],
                    labels=candidate_residuals[split_mask] <= 2.0,
                    valid_edges=np.ones_like(candidate_residuals[split_mask], dtype=bool),
                )
            )
        summary["thresholds_px"][str(float(threshold))] = threshold_summary
    summary["artifacts"] = {
        "proposals_sha256": file_sha256_short(Path(args.proposals)),
        "feature_artifact_sha256": file_sha256_short(Path(args.feature_artifact)),
        "projected_landmark_bank_sha256": file_sha256_short(
            Path(args.projected_landmark_bank)
        ),
        "split_sha256": file_sha256_short(Path(args.split_json)),
        "summary": str(output_dir / "summary.json"),
        "pose_rows": str(output_dir / "pose_rows.json"),
    }
    (output_dir / "pose_rows.json").write_text(
        json.dumps(pose_rows, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
