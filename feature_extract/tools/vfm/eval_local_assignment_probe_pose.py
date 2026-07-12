"""Re-evaluate PnP strategies from a compact local-assignment probe artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_arrays", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--strategies",
        default=(
            "coarse_prototype,all_support_best,all_support_top2_mean,all_support_top4_mean,"
            "coarse_all_support_best_50_50,proposal_oracle"
        ),
    )
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _validate_probe_binding(
    *,
    probe_path: Path,
    landmark_metadata: dict[str, object],
    candidates: UniqueTrackCandidateSet,
    landmark_index,
) -> dict[str, object]:
    summary_path = probe_path.parent / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"probe summary is required for descriptor-space validation: {summary_path}")
    summary = json.loads(summary_path.read_text())
    descriptor = summary.get("descriptor_space")
    if not isinstance(descriptor, dict):
        raise ValueError("probe summary is missing descriptor_space")
    expected_id = str(descriptor.get("descriptor_space_id", ""))
    actual_id = str(landmark_metadata.get("descriptor_space_id", ""))
    if not expected_id or expected_id != actual_id:
        raise ValueError(
            f"probe/landmark descriptor spaces differ: {expected_id!r} vs {actual_id!r}"
        )
    valid = candidates.valid_mask
    rows = candidates.bank_row_indices[valid]
    if np.any(rows >= len(landmark_index)):
        raise ValueError("probe candidate bank row is out of range")
    if not np.array_equal(candidates.track_ids[valid], landmark_index.track_ids[rows]):
        raise ValueError("probe candidate track ids are not aligned with the supplied landmark bank")
    if not np.array_equal(candidates.prototype_ids[valid], landmark_index.prototype_ids[rows]):
        raise ValueError("probe candidate prototype ids are not aligned with the supplied landmark bank")
    return {
        "probe_summary": str(summary_path),
        "descriptor_space_id": actual_id,
        "candidate_alignment_checked": True,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    probe_path = Path(args.probe_arrays)
    with np.load(probe_path, allow_pickle=False) as data:
        query_ids = tuple(str(item) for item in data["query_ids"].tolist())
        point2d_indices = np.asarray(data["query_point2d_indices"], dtype=np.int64)
        query_xy = np.asarray(data["query_xy"], dtype=np.float64)
        correct_track_ids = np.asarray(data["correct_track_ids"], dtype=np.int64)
        candidates = UniqueTrackCandidateSet(
            bank_row_indices=np.asarray(data["bank_row_indices"], dtype=np.int64),
            track_ids=np.asarray(data["candidate_track_ids"], dtype=np.int64),
            prototype_ids=np.asarray(data["candidate_prototype_ids"], dtype=np.int64),
            coarse_scores=np.asarray(data["coarse_scores"], dtype=np.float32),
        )
        strategy_scores = {
            key[len("strategy__") :]: np.asarray(data[key], dtype=np.float32)
            for key in data.files
            if key.startswith("strategy__")
        }
    if not (
        len(query_ids)
        == point2d_indices.shape[0]
        == query_xy.shape[0]
        == correct_track_ids.shape[0]
        == candidates.query_count
    ):
        raise ValueError("probe query arrays have inconsistent lengths")

    landmark_index, landmark_metadata = load_landmark_index_npz(Path(args.projected_landmark_bank))
    binding = _validate_probe_binding(
        probe_path=probe_path,
        landmark_metadata=landmark_metadata,
        candidates=candidates,
        landmark_index=landmark_index,
    )
    query_observations = [
        ColmapTrackObservation(
            track_id=int(track_id),
            image_id=str(query_id),
            point2d_idx=int(point2d_index),
            xy=(float(xy[0]), float(xy[1])),
            xyz=np.zeros((3,), dtype=np.float64),
            track_length=1,
            reprojection_error=0.0,
        )
        for query_id, point2d_index, xy, track_id in zip(
            query_ids,
            point2d_indices,
            query_xy,
            correct_track_ids,
        )
    ]
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    strategies = tuple(item.strip() for item in str(args.strategies).split(",") if item.strip())
    summaries: dict[str, object] = {}
    rows: dict[str, object] = {}
    for strategy in strategies:
        if strategy != "proposal_oracle" and strategy not in strategy_scores:
            raise ValueError(f"strategy is not present in probe artifact: {strategy}")
        summary, strategy_rows = _evaluate_pose_strategy(
            strategy=strategy,
            scores=None if strategy == "proposal_oracle" else strategy_scores[strategy],
            candidates=candidates,
            query_observations=query_observations,
            query_ids=query_ids,
            landmark_index=landmark_index,
            cameras=cameras,
            images_by_name=images_by_name,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
        )
        summaries[strategy] = summary
        rows[strategy] = strategy_rows
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "pose_rows.json"
    rows_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    output = {
        "stage": "s4_l0_assignment_probe_pose_replay",
        "binding": binding,
        "pnp": {
            "reprojection_error_px": float(args.pnp_reprojection_error_px),
            "iterations": int(args.pnp_iterations),
            "rng_seed": "sha256(query_id)",
        },
        "pose": summaries,
        "outputs": {"pose_rows": str(rows_path), "summary": str(output_dir / "summary.json")},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
