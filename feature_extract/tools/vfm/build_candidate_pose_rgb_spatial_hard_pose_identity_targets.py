"""Build direct correct-vs-coherent-wrong identity targets without pose fields."""

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
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_pose_identity import (
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_IDENTITY_TARGET_FORMAT,
    CandidatePoseRGBSpatialHardPoseIdentityTargets,
    save_candidate_pose_rgb_spatial_hard_pose_identity_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_pose_pairs import (
    load_candidate_pose_rgb_spatial_hard_pose_pairs,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_observation_pairs import (
    load_candidate_pose_rgb_spatial_observation_pairs,
)


def build_candidate_pose_rgb_spatial_hard_pose_identity_targets(
    *,
    hard_pose_pairs: Path,
    hard_pose_identity_pairs: Path,
    coherent_wrong_local_radius_px: float,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, object]:
    """Materialize distinct wrong-candidate masks after train-only projection."""

    hard_path = Path(hard_pose_pairs)
    identity_path = Path(hard_pose_identity_pairs)
    output_path = Path(output)
    summary_path = Path(summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite hard-pose identity target output")
    radius = float(coherent_wrong_local_radius_px)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("coherent-wrong local radius is invalid")

    hard = load_candidate_pose_rgb_spatial_hard_pose_pairs(hard_path)
    identity = load_candidate_pose_rgb_spatial_observation_pairs(identity_path)
    if str(identity.metadata.get("hard_pose_pairs_sha256", "")) != file_sha256_short(hard_path):
        raise ValueError("identity-pair layout does not derive from the supplied hard-pose rows")
    if hard.candidate_count != identity.negative_count + 1:
        raise ValueError("hard-pose and identity-pair candidate counts differ")
    hard_row_by_anchor = {
        int(anchor): row for row, anchor in enumerate(np.asarray(hard.row_ids, dtype=np.int64))
    }
    if len(hard_row_by_anchor) != hard.row_count:
        raise ValueError("hard-pose row IDs are not unique")
    try:
        hard_rows = np.asarray(
            [hard_row_by_anchor[int(anchor)] for anchor in identity.anchor_ids.tolist()], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("identity-pair layout references an unknown hard-pose row") from error
    if (
        not np.array_equal(identity.query_image_ids, hard.query_image_ids[hard_rows])
        or not np.allclose(identity.query_xy, hard.query_xy[hard_rows], atol=1e-5, rtol=0.0)
        or not np.array_equal(identity.positive_track_ids, hard.candidate_track_ids[hard_rows, 0])
        or not np.array_equal(identity.negative_track_ids, hard.candidate_track_ids[hard_rows, 1:])
        or not np.array_equal(identity.positive_support_image_ids, hard.support_image_ids[hard_rows, 0])
        or not np.array_equal(identity.negative_support_image_ids, hard.support_image_ids[hard_rows, 1:])
        or not np.allclose(identity.positive_support_xy, hard.support_xy[hard_rows, 0], atol=1e-5, rtol=0.0)
        or not np.allclose(identity.negative_support_xy, hard.support_xy[hard_rows, 1:], atol=1e-5, rtol=0.0)
        or not np.array_equal(identity.split_names, hard.split_names[hard_rows])
    ):
        raise ValueError("identity-pair layout drifts from fixed hard-pose visual inputs")
    if (
        not np.all(hard.spatial_target_observed[hard_rows, 0])
        or np.any(hard.spatial_target_observed[hard_rows, 1:])
    ):
        raise ValueError("hard-pose identity target requires original candidate slot zero positive")

    wrong_local = (
        np.asarray(hard.coherent_wrong_projection_valid[hard_rows], dtype=bool)
        & (
            np.max(
                np.abs(np.asarray(hard.coherent_wrong_projection_offsets_xy[hard_rows], dtype=np.float32)),
                axis=2,
            )
            <= radius
        )
    )
    wrong_local[:, 0] = False
    selected = np.flatnonzero(np.any(wrong_local[:, 1:], axis=1))
    if len(selected) == 0:
        raise ValueError("no distinct coherent-wrong candidates are local at the requested radius")
    artifact = CandidatePoseRGBSpatialHardPoseIdentityTargets(
        anchor_ids=np.asarray(identity.anchor_ids[selected], dtype=np.int64),
        query_image_ids=np.asarray(identity.query_image_ids[selected]).astype(str),
        hard_negative_candidate_mask=np.asarray(wrong_local[selected], dtype=bool),
        metadata={
            "format": CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_IDENTITY_TARGET_FORMAT,
            "training_only_target_artifact": True,
            "contains_ground_truth": True,
            "contains_validation_or_test_targets": False,
            "runtime_scorer_must_not_load_this_artifact": True,
            "runtime_layout_is_target_free": True,
            "pose_projection_or_residual_serialized": False,
            "positive_candidate_is_original_slot_zero": True,
            "render": False,
            "image_retrieval_or_submap_used": False,
            "hard_pose_identity_pairs_sha256": file_sha256_short(identity_path),
            "hard_pose_pairs_sha256": file_sha256_short(hard_path),
            "candidate_count": int(hard.candidate_count),
            "coherent_wrong_local_radius_px": radius,
            "selection": "distinct_candidate_coherent_wrong_projection_linf_local_v1",
            "hard_pose_source": str(hard.metadata.get("hard_pose_source", "")),
        },
    )
    save_candidate_pose_rgb_spatial_hard_pose_identity_targets(artifact, output_path)
    counts = np.sum(artifact.hard_negative_candidate_mask, axis=1)
    summary = {
        "stage": "build_candidate_pose_rgb_spatial_hard_pose_identity_targets",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "target_count": int(artifact.count),
        "candidate_count": int(artifact.candidate_count),
        "query_count": int(len(np.unique(artifact.query_image_ids))),
        "negative_count_quantiles": np.quantile(counts, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist(),
        "coherent_wrong_local_radius_px": radius,
        "protocol": {
            "target_sidecar_only": True,
            "no_pose_projection_or_residual_serialized": True,
            "distinct_wrong_candidate_only": True,
            "runtime_scorer_must_not_load_output": True,
        },
    }
    split_by_query: dict[str, str] = {}
    for query_id in np.unique(identity.query_image_ids).tolist():
        split = np.unique(identity.split_names[identity.query_image_ids == str(query_id)])
        if len(split) != 1:
            raise ValueError("identity-pair split leaks across query")
        split_by_query[str(query_id)] = str(split[0])
    summary["split_counts"] = {
        split: int(
            np.count_nonzero(
                np.asarray([split_by_query[str(query_id)] for query_id in artifact.query_image_ids])
                == split
            )
        )
        for split in ("inner_train", "inner_validation")
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-pose-pairs", required=True)
    parser.add_argument("--hard-pose-identity-pairs", required=True)
    parser.add_argument("--coherent-wrong-local-radius-px", type=float, default=8.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_candidate_pose_rgb_spatial_hard_pose_identity_targets(
        hard_pose_pairs=Path(args.hard_pose_pairs),
        hard_pose_identity_pairs=Path(args.hard_pose_identity_pairs),
        coherent_wrong_local_radius_px=float(args.coherent_wrong_local_radius_px),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
