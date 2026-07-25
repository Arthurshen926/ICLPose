"""Build train-only coherent-repeat candidate-edge targets.

The frozen RGB layout is never altered.  Correct and coherent-wrong projected
offsets are read only from the already train-only target artifact, then used to
select difficult positive/negative candidate identities that both explain the
same query anchor under different poses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


# Keep the documented direct script invocation self-contained, matching the
# RGB likelihood trainer's torchrun entry point.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    CANDIDATE_POSE_RGB_SPATIAL_HARD_REPEAT_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_MULTI_HARD_REPEAT_FORMAT,
    CandidatePoseRGBSpatialHardRepeatTargets,
    save_candidate_pose_rgb_spatial_hard_repeat_targets,
    select_coherent_hard_repeat_candidate_edges,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    load_candidate_pose_rgb_spatial_training_targets,
)


def build_candidate_pose_rgb_spatial_hard_repeat_targets(
    *,
    rgb_spatial_layout: Path,
    rgb_spatial_targets: Path,
    positive_radius_px: float,
    negative_radius_px: float,
    output: Path,
    summary_json: Path,
    force: bool,
    max_negatives_per_source_pair: int = 1,
) -> dict[str, object]:
    """Materialize distinct correct/coherent-wrong candidate pairs for training."""

    layout_path = Path(rgb_spatial_layout)
    target_path = Path(rgb_spatial_targets)
    output_path = Path(output)
    summary_path = Path(summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite hard-repeat target output")
    positive_radius = float(positive_radius_px)
    negative_radius = float(negative_radius_px)
    negative_cap = int(max_negatives_per_source_pair)
    if (
        not np.isfinite(positive_radius)
        or not np.isfinite(negative_radius)
        or positive_radius <= 0.0
        or negative_radius <= 0.0
        or negative_cap < 0
    ):
        raise ValueError("hard-repeat radii and negative cap are invalid")

    layout = load_candidate_pose_rgb_spatial_layout(layout_path)
    targets = load_candidate_pose_rgb_spatial_training_targets(target_path)
    layout_sha256 = file_sha256_short(layout_path)
    if str(targets.metadata.get("rgb_spatial_layout_sha256", "")) != layout_sha256:
        raise ValueError("hard-repeat targets would join a stale RGB layout")
    if targets.candidate_count != layout.candidate_count:
        raise ValueError("hard-repeat targets and layout candidate count differ")
    exact_identity_mode = (
        str(targets.metadata.get("spatial_supervision_mode", ""))
        == "registered_exact_identity"
    )
    target_radius = float(targets.metadata.get("spatial_search_radius_px", 0.0))
    if positive_radius > target_radius or negative_radius > target_radius:
        raise ValueError("hard-repeat radii exceed the frozen local spatial support")
    layout_row_by_source = {
        int(source_id): row
        for row, source_id in enumerate(np.asarray(layout.source_point_ids, dtype=np.int64).tolist())
    }
    target_row_by_source = {
        int(source_id): row
        for row, source_id in enumerate(np.asarray(targets.source_point_ids, dtype=np.int64))
    }

    source_ids: list[np.ndarray] = []
    query_ids: list[np.ndarray] = []
    pair_ids: list[np.ndarray] = []
    positive_candidates: list[np.ndarray] = []
    negative_candidates: list[np.ndarray] = []
    positive_offsets: list[np.ndarray] = []
    negative_offsets: list[np.ndarray] = []
    pair_counts: list[int] = []
    source_pair_counts: list[int] = []
    for pair_index, (pair_id, query_id) in enumerate(
        zip(targets.pair_ids.tolist(), targets.pair_query_ids.tolist())
    ):
        start, stop = targets.pair_point_offsets[pair_index : pair_index + 2].tolist()
        pair_sources = np.asarray(targets.pair_source_point_ids[int(start) : int(stop)], dtype=np.int64)
        try:
            layout_rows = np.asarray(
                [layout_row_by_source[int(source_id)] for source_id in pair_sources.tolist()],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("hard-repeat pair references a source absent from the layout") from error
        try:
            target_rows = np.asarray(
                [target_row_by_source[int(source_id)] for source_id in pair_sources.tolist()],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("hard-repeat pair references a source absent from training targets") from error
        if (
            np.any(np.asarray(layout.split_names[layout_rows]).astype(str) != "train")
            or np.any(np.asarray(layout.query_ids[layout_rows]).astype(str) != str(query_id))
        ):
            raise ValueError("hard-repeat target pair is not a train-only layout group")
        correct_offsets = np.asarray(targets.correct_projection_offsets_xy[int(start) : int(stop)], dtype=np.float32)
        correct_valid = np.asarray(targets.correct_projection_valid[int(start) : int(stop)], dtype=bool)
        wrong_offsets = np.asarray(
            targets.coherent_wrong_projection_offsets_xy[int(start) : int(stop)], dtype=np.float32
        )
        wrong_valid = np.asarray(targets.coherent_wrong_projection_valid[int(start) : int(stop)], dtype=bool)
        positions, positive, negative = select_coherent_hard_repeat_candidate_edges(
            correct_offsets_xy=correct_offsets,
            correct_valid=correct_valid,
            wrong_offsets_xy=wrong_offsets,
            wrong_valid=wrong_valid,
            candidate_prior_probabilities=np.asarray(
                layout.candidate_prior_probabilities[layout_rows], dtype=np.float32
            ),
            positive_radius_px=positive_radius,
            negative_radius_px=negative_radius,
            positive_candidate_mask=(
                np.asarray(targets.spatial_target_observed[target_rows], dtype=bool)
                if exact_identity_mode
                else None
            ),
            max_negatives_per_source_pair=negative_cap,
        )
        count = int(len(positions))
        pair_counts.append(count)
        source_pair_counts.append(int(len(np.unique(positions))))
        if not count:
            continue
        source_ids.append(pair_sources[positions])
        query_ids.append(np.full((count,), str(query_id), dtype=layout.query_ids.dtype))
        pair_ids.append(np.full((count,), int(pair_id), dtype=np.int64))
        positive_candidates.append(positive)
        negative_candidates.append(negative)
        positive_offsets.append(correct_offsets[positions, positive])
        negative_offsets.append(wrong_offsets[positions, negative])

    if not source_ids:
        raise ValueError("hard-repeat target builder selected no coherent candidate pairs")
    artifact = CandidatePoseRGBSpatialHardRepeatTargets(
        source_point_ids=np.concatenate(source_ids, axis=0),
        query_ids=np.concatenate(query_ids, axis=0),
        pair_ids=np.concatenate(pair_ids, axis=0),
        positive_candidate_indices=np.concatenate(positive_candidates, axis=0),
        negative_candidate_indices=np.concatenate(negative_candidates, axis=0),
        positive_offsets_xy=np.concatenate(positive_offsets, axis=0),
        negative_offsets_xy=np.concatenate(negative_offsets, axis=0),
        metadata={
            "format": (
                CANDIDATE_POSE_RGB_SPATIAL_HARD_REPEAT_FORMAT
                if negative_cap == 1
                else CANDIDATE_POSE_RGB_SPATIAL_MULTI_HARD_REPEAT_FORMAT
            ),
            "training_only_target_artifact": True,
            "contains_ground_truth": True,
            "contains_validation_or_test_targets": False,
            "runtime_layout_is_target_free": True,
            "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
            "render": False,
            "image_retrieval_or_submap_used": False,
            "rgb_spatial_layout_sha256": layout_sha256,
            "rgb_spatial_targets_sha256": file_sha256_short(target_path),
            "projection_space_id": str(layout.metadata["projection_space_id"]),
            "descriptor_space_id": str(layout.metadata["descriptor_space_id"]),
            "candidate_count": int(layout.candidate_count),
            "positive_radius_px": positive_radius,
            "negative_radius_px": negative_radius,
            "max_negatives_per_source_pair": negative_cap,
            "negative_selection": "all_or_capped_distinct_coherent_wrong_local_candidates_v1",
            "selection": (
                "registered_exact_track_correct_local_candidate_vs_different_"
                "coherent_wrong_local_candidate_min_linf_then_frozen_prior_v1"
                if exact_identity_mode and negative_cap == 1
                else (
                    "registered_exact_track_correct_local_candidate_vs_all_or_capped_"
                    "distinct_coherent_wrong_local_candidates_min_linf_then_frozen_prior_v2"
                    if exact_identity_mode
                    else "correct_pose_local_candidate_vs_all_or_capped_distinct_"
                    "coherent_wrong_local_candidates_min_linf_then_frozen_prior_v2"
                )
            ),
            "negative_excludes_all_correct_pose_local_candidates": not exact_identity_mode,
            "positive_requires_registered_exact_track": exact_identity_mode,
            "inputs": {
                "rgb_spatial_layout": str(layout_path),
                "rgb_spatial_targets": str(target_path),
                "train_only_pair_source": str(targets.metadata.get("train_pairs_sha256", "")),
            },
        },
    )
    save_candidate_pose_rgb_spatial_hard_repeat_targets(artifact, output_path)
    summary = {
        "stage": "build_candidate_pose_rgb_spatial_hard_repeat_targets",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "target_count": int(artifact.count),
        "source_pair_count": int(sum(source_pair_counts)),
        "pair_count": int(len(targets.pair_ids)),
        "nonempty_pair_count": int(sum(count > 0 for count in pair_counts)),
        "median_targets_per_nonempty_pair": float(
            np.median([count for count in pair_counts if count > 0])
        ),
        "median_source_pairs_per_nonempty_pair": float(
            np.median(
                [
                    source_count
                    for count, source_count in zip(pair_counts, source_pair_counts)
                    if count > 0
                ]
            )
        ),
        "positive_radius_px": positive_radius,
        "negative_radius_px": negative_radius,
        "max_negatives_per_source_pair": negative_cap,
        "exact_identity_mode": exact_identity_mode,
        "protocol": {
            "train_only": True,
            "runtime_scorer_must_not_load_hard_repeat_targets": True,
            "negative_is_current_coherent_wrong_candidate_not_random_support_permutation": True,
            "multiple_distinct_wrong_candidates_per_source_pair": bool(negative_cap != 1),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--rgb-spatial-targets", required=True)
    parser.add_argument("--positive-radius-px", type=float, default=4.0)
    parser.add_argument("--negative-radius-px", type=float, default=4.0)
    parser.add_argument(
        "--max-negatives-per-source-pair",
        type=int,
        default=1,
        help="0 retains all coherent-wrong candidates; 1 reproduces the legacy single-negative target.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_candidate_pose_rgb_spatial_hard_repeat_targets(
        rgb_spatial_layout=Path(args.rgb_spatial_layout),
        rgb_spatial_targets=Path(args.rgb_spatial_targets),
        positive_radius_px=float(args.positive_radius_px),
        negative_radius_px=float(args.negative_radius_px),
        max_negatives_per_source_pair=int(args.max_negatives_per_source_pair),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
