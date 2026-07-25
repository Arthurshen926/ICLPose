"""Convert current hard-pose anchors into broad identity-only supervision.

The input hard-pose artifact is train-only and contains correct/coherent-wrong
projections.  This converter deliberately does not serialize those projections
into its output.  It keeps only the fixed real-image query/support candidate
layout selected by the hard-pose miner, so the regular candidate identity LLR
trainer can learn a target-free appearance factor from many current coherent
repeat modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_pose_pairs import (
    load_candidate_pose_rgb_spatial_hard_pose_pairs,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_observation_pairs import (
    CandidatePoseRGBSpatialObservationPairs,
    load_candidate_pose_rgb_spatial_observation_pairs,
    save_candidate_pose_rgb_spatial_observation_pairs,
)


_NEGATIVE_SOURCE = "current_hard_pose_radio_pca_ann_v1"


def _split_by_query(pairs: CandidatePoseRGBSpatialObservationPairs) -> dict[str, str]:
    """Require the broad source's query split to be query-disjoint."""

    result: dict[str, str] = {}
    for query_id in np.unique(pairs.query_image_ids).tolist():
        values = np.unique(pairs.split_names[pairs.query_image_ids == str(query_id)])
        if len(values) != 1:
            raise ValueError("observation-pair inner split leaks across a query image")
        result[str(query_id)] = str(values[0])
    return result


def build_candidate_pose_rgb_spatial_hard_pose_identity_pairs(
    *,
    hard_pose_pairs: Path,
    observation_pairs: Path,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, object]:
    """Write target-bearing identity rows from fixed current hard-pose inputs."""

    hard_path = Path(hard_pose_pairs)
    source_path = Path(observation_pairs)
    output_path = Path(output)
    summary_path = Path(summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite hard-pose identity-pair output")

    hard = load_candidate_pose_rgb_spatial_hard_pose_pairs(hard_path)
    source = load_candidate_pose_rgb_spatial_observation_pairs(source_path)
    expected_source_sha = str(hard.metadata.get("source_observation_pairs_sha256", ""))
    if expected_source_sha != file_sha256_short(source_path):
        raise ValueError("hard-pose rows do not derive from this observation-pair source")
    if hard.candidate_count != source.negative_count + 1:
        raise ValueError("hard-pose and observation-pair candidate cardinalities differ")
    if str(hard.metadata.get("hard_pose_source", "")).strip() == "":
        raise ValueError("hard-pose rows lack current coherent-wrong provenance")

    split_by_query = _split_by_query(source)
    if not set(hard.query_image_ids.tolist()).issubset(split_by_query):
        raise ValueError("hard-pose rows reference a query absent from broad supervision")
    expected_splits = np.asarray(
        [split_by_query[str(query_id)] for query_id in hard.query_image_ids.tolist()],
        dtype=hard.split_names.dtype,
    )
    if not np.array_equal(expected_splits, hard.split_names):
        raise ValueError("hard-pose rows cross the source query-grouped split")

    metadata = dict(source.metadata)
    metadata.update(
        {
            "hard_pose_identity_pair_source": "current_coherent_wrong_pose_mined_anchor_v1",
            "hard_pose_pairs_sha256": file_sha256_short(hard_path),
            "hard_pose_pair_format": str(hard.metadata.get("format", "")),
            "hard_pose_source": str(hard.metadata.get("hard_pose_source", "")),
            "hard_pose_anchor_jitter_radius_px": int(
                hard.metadata.get("anchor_jitter_radius_px", 0)
            ),
            "hard_pose_min_positive_wrong_delta_px": float(
                hard.metadata.get("min_wrong_positive_offset_delta_px", 0.0)
            ),
            "hard_pose_projection_targets_serialized": False,
            "runtime_scorer_must_not_load_this_artifact": True,
        }
    )
    artifact = CandidatePoseRGBSpatialObservationPairs(
        anchor_ids=np.asarray(hard.row_ids, dtype=np.int64),
        query_image_ids=np.asarray(hard.query_image_ids).astype(str),
        query_xy=np.asarray(hard.query_xy, dtype=np.float32),
        positive_support_image_ids=np.asarray(hard.support_image_ids[:, 0]).astype(str),
        positive_support_xy=np.asarray(hard.support_xy[:, 0], dtype=np.float32),
        positive_track_ids=np.asarray(hard.candidate_track_ids[:, 0], dtype=np.int64),
        negative_support_image_ids=np.asarray(hard.support_image_ids[:, 1:]).astype(str),
        negative_support_xy=np.asarray(hard.support_xy[:, 1:], dtype=np.float32),
        negative_track_ids=np.asarray(hard.candidate_track_ids[:, 1:], dtype=np.int64),
        negative_sources=np.full(
            (hard.row_count, hard.candidate_count - 1), _NEGATIVE_SOURCE, dtype=np.str_
        ),
        split_names=np.asarray(hard.split_names).astype(str),
        metadata=metadata,
    )
    save_candidate_pose_rgb_spatial_observation_pairs(artifact, output_path)
    summary = {
        "stage": "build_candidate_pose_rgb_spatial_hard_pose_identity_pairs",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "row_count": int(artifact.row_count),
        "candidate_count": int(artifact.negative_count + 1),
        "query_count": int(len(np.unique(artifact.query_image_ids))),
        "split_counts": {
            split: int(np.count_nonzero(artifact.split_names == split))
            for split in ("inner_train", "inner_validation")
        },
        "lineage": {
            "hard_pose_pairs_sha256": file_sha256_short(hard_path),
            "observation_pairs_sha256": file_sha256_short(source_path),
        },
        "protocol": {
            "train_only": True,
            "fixed_candidate_inputs_selected_before_identity_training": True,
            "correct_or_wrong_pose_not_serialized_to_output": True,
            "runtime_scorer_must_not_load_output": True,
            "render": False,
            "image_retrieval_or_submap_used": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-pose-pairs", required=True)
    parser.add_argument("--observation-pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_candidate_pose_rgb_spatial_hard_pose_identity_pairs(
        hard_pose_pairs=Path(args.hard_pose_pairs),
        observation_pairs=Path(args.observation_pairs),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
