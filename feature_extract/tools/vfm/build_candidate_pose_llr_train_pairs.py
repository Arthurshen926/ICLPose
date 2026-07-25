"""Build train-only correct-pose versus coherent-wrong-pose LLR pairs.

The coherent wrong poses are selected by a previous target-side hard-mode
mining pass.  This builder is the only consumer of that target-bearing
artifact.  It copies no candidate labels, residuals, pose errors, or target
rows into the runtime scorer; it writes only train query IDs and a correct /
coherent-wrong pose pair for each selected train mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import qvec_to_rotmat, read_colmap_images_binary
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    load_mixed_verification_points,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    grouped_hypothesis_semantic_manifest,
    validate_grouped_hypothesis_semantic_match,
)


TRAIN_PAIR_FORMAT = "candidate_pose_llr_train_pairs_v1"


def validate_train_pair_splits(*, query_ids: np.ndarray, split_names: np.ndarray) -> None:
    """Require every target-joined pose pair to belong to the train split."""

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    splits = np.asarray(split_names).astype(str).reshape(-1)
    if len(ids) == 0 or ids.shape != splits.shape or np.any(ids == ""):
        raise ValueError("train-pair query rows are invalid")
    if np.any(splits != "train"):
        raise ValueError("candidate pose-LLR training rejects validation/test target rows")


def _pose_w2c(image: object) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(np.asarray(getattr(image, "qvec"), dtype=np.float64))
    pose[:3, 3] = np.asarray(getattr(image, "tvec"), dtype=np.float64).reshape(3)
    return pose


def _load_hypothesis_poses(
    paths: Sequence[Path], *, evaluation_label: str, expected_train_query_ids: set[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, str]]]:
    """Load target-free source poses indexed in the hard-mode JSON contract."""

    poses_by_query: dict[str, np.ndarray] = {}
    scores_by_query: dict[str, np.ndarray] = {}
    manifests: list[dict[str, str]] = []
    if not expected_train_query_ids:
        raise ValueError("hard-mode target rows select no train queries")
    for path in paths:
        with np.load(Path(path), allow_pickle=False) as payload:
            required = {
                "query_ids",
                "split_names",
                "evaluation_labels",
                "poses_w2c",
                "verification_log_likelihood_means",
                "metadata_json",
            }
            missing = required - set(payload.files)
            if missing:
                raise ValueError(f"hard-mode pose source lacks {sorted(missing)}")
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
            query_ids = np.asarray(payload["query_ids"]).astype(str)
            splits = np.asarray(payload["split_names"]).astype(str)
            labels = np.asarray(payload["evaluation_labels"]).astype(str)
            poses = np.asarray(payload["poses_w2c"], dtype=np.float64)
            scores = np.asarray(payload["verification_log_likelihood_means"], dtype=np.float64)
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("format") != "grouped_pose_hypotheses_inference_only_v1"
            or metadata.get("contains_target_fields") is not False
            or poses.shape != (len(query_ids), 4, 4)
            or splits.shape != query_ids.shape
            or labels.shape != query_ids.shape
            or scores.shape != query_ids.shape
            or not np.isfinite(poses).all()
        ):
            raise ValueError("hard-mode source is not a valid target-free hypothesis artifact")
        for query_id in sorted(set(query_ids.tolist())):
            if str(query_id) not in expected_train_query_ids:
                continue
            rows = np.flatnonzero(
                (query_ids == str(query_id)) & (labels == str(evaluation_label))
            )
            if len(rows) == 0:
                continue
            if np.any(splits[rows] != "train"):
                raise ValueError("hard-mode pose source leaks validation/test hypothesis rows")
            if str(query_id) in poses_by_query:
                raise ValueError(f"hard-mode source repeats query {query_id!r}")
            poses_by_query[str(query_id)] = poses[rows].copy()
            scores_by_query[str(query_id)] = scores[rows].copy()
        manifests.append({"path": str(path), "sha256": file_sha256_short(Path(path))})
    if not poses_by_query:
        raise ValueError("hard-mode source has no train query poses")
    missing = sorted(expected_train_query_ids.difference(poses_by_query))
    if missing:
        raise ValueError(f"hard-mode source is missing selected train queries: {missing[:5]}")
    return poses_by_query, scores_by_query, manifests


def _hard_mode_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "query_ids",
            "hard_mode_ids_TARGET_ONLY",
            "hard_mode_query_ids_TARGET_ONLY",
            "hard_mode_support_group_counts_TARGET_ONLY",
            "metadata_json",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"hard-mode artifact lacks {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]).copy() for key in required if key != "metadata_json"}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != "pose_conditioned_system_hard_modes_v2"
        or metadata.get("training_only_target_artifact") is not True
        or metadata.get("ground_truth_joined_after_generation") is not True
        or metadata.get("pose_or_ground_truth_used_for_hypothesis_generation") is not False
        or metadata.get("split_names") != ["train"]
    ):
        raise ValueError("hard-mode artifact is not a train-only post-inference target join")
    rows_path = Path(path).parent / "per_query_TARGET_ONLY.json"
    if not rows_path.is_file():
        raise FileNotFoundError("hard-mode artifact needs its paired per_query_TARGET_ONLY.json")
    raw_rows = json.loads(rows_path.read_text())
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("hard-mode per-query target rows are invalid")
    return metadata, raw_rows, arrays


def _mode_sources_from_metadata(metadata: Mapping[str, Any]) -> tuple[Path, ...]:
    inputs = metadata.get("inputs")
    values = inputs.get("hypothesis_artifacts") if isinstance(inputs, Mapping) else None
    if not isinstance(values, list) or not values:
        raise ValueError("hard-mode artifact lacks source hypothesis manifests")
    paths: list[Path] = []
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError("hard-mode source hypothesis manifest is invalid")
        path = Path(str(item.get("path", "")))
        expected_sha = str(item.get("sha256", ""))
        if not path.is_file() or not expected_sha or file_sha256_short(path) != expected_sha:
            raise ValueError("hard-mode source hypothesis manifest is stale")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("hard-mode source hypothesis paths repeat")
    return tuple(paths)


def _hypothesis_semantic_manifest_from_path(path: Path) -> dict[str, object]:
    """Read only the target-free generation contract from a hypothesis NPZ."""

    with np.load(Path(path), allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError("hypothesis artifact lacks metadata for semantic lineage")
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    return grouped_hypothesis_semantic_manifest(metadata)


def validate_train_pair_hypothesis_sources_match_reference(
    *, source_paths: Sequence[Path], reference_path: Path
) -> dict[str, object]:
    """Require train hard-mode poses to use held-out inference semantics exactly."""

    paths = tuple(Path(path) for path in source_paths)
    reference = Path(reference_path)
    if not paths:
        raise ValueError("candidate pose-LLR training needs at least one hypothesis source")
    if not reference.is_file():
        raise FileNotFoundError("candidate pose-LLR reference hypothesis is missing")
    expected = _hypothesis_semantic_manifest_from_path(reference)
    sources: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("candidate pose-LLR source hypothesis is missing")
        observed = _hypothesis_semantic_manifest_from_path(path)
        validate_grouped_hypothesis_semantic_match(expected=expected, observed=observed)
        sources.append(
            {
                "path": str(path),
                "sha256": file_sha256_short(path),
                "semantic_hash": str(observed["semantic_hash"]),
            }
        )
    return {
        "semantic_manifest": expected["semantic_manifest"],
        "semantic_hash": str(expected["semantic_hash"]),
        "reference_hypothesis_artifact": {
            "path": str(reference),
            "sha256": file_sha256_short(reference),
        },
        "source_hypothesis_artifacts": sources,
    }


def build_candidate_pose_llr_train_pairs(
    *,
    verification_points_path: Path,
    hard_modes_path: Path,
    reference_hypothesis_path: Path,
    colmap_model_dir: Path,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, object]:
    """Materialize pair targets while enforcing the train-only boundary."""

    output = Path(output)
    summary_json = Path(summary_json)
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite candidate pose-LLR train pairs")
    points = load_mixed_verification_points(Path(verification_points_path))
    if (
        points.metadata.get("format") != MIXED_VERIFICATION_POINTS_FORMAT
        or points.metadata.get("contains_ground_truth") is not False
        or points.metadata.get("pose_or_ground_truth_used") is not False
        or points.metadata.get("render") is not False
    ):
        raise ValueError("verification points violate the target-free real-image contract")
    hard_metadata, per_query, hard_arrays = _hard_mode_rows(Path(hard_modes_path))
    evaluation_label = str(hard_metadata.get("evaluation_label", ""))
    if not evaluation_label:
        raise ValueError("hard-mode artifact lacks its frozen evaluation label")
    pose_paths = _mode_sources_from_metadata(hard_metadata)
    hypothesis_lineage = validate_train_pair_hypothesis_sources_match_reference(
        source_paths=pose_paths,
        reference_path=Path(reference_hypothesis_path),
    )
    expected_train_queries: set[str] = set()
    for query_row in per_query:
        if not isinstance(query_row, Mapping):
            raise ValueError("hard-mode per-query entry is invalid")
        query_id = str(query_row.get("query_id", ""))
        split_name = str(query_row.get("split_name", ""))
        if not query_id or split_name != "train":
            raise ValueError("hard-mode per-query rows leak validation/test targets")
        expected_train_queries.add(query_id)
    poses_by_query, scores_by_query, pose_manifests = _load_hypothesis_poses(
        pose_paths,
        evaluation_label=evaluation_label,
        expected_train_query_ids=expected_train_queries,
    )
    images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}

    query_ids: list[str] = []
    split_names: list[str] = []
    correct_poses: list[np.ndarray] = []
    wrong_poses: list[np.ndarray] = []
    hypothesis_indices: list[int] = []
    support_counts: list[int] = []
    for query_row in per_query:
        if not isinstance(query_row, Mapping):
            raise ValueError("hard-mode per-query entry is invalid")
        query_id = str(query_row.get("query_id", ""))
        split_name = str(query_row.get("split_name", ""))
        modes = query_row.get("selected_modes_TARGET_ONLY")
        if not query_id or split_name != "train" or not isinstance(modes, list):
            raise ValueError("hard-mode per-query rows leak validation/test targets")
        point_rows = points.rows_for_query(query_id)
        if len(point_rows) == 0 or np.any(points.split_names[point_rows] != "train"):
            raise ValueError("hard-mode query is absent from train-only verification points")
        poses = poses_by_query.get(query_id)
        scores = scores_by_query.get(query_id)
        image = images_by_name.get(query_id)
        if poses is None or scores is None or image is None:
            raise ValueError("hard-mode query cannot resolve its train pose source")
        gt_pose = _pose_w2c(image)
        for mode in modes:
            if not isinstance(mode, Mapping):
                raise ValueError("hard-mode selection entry is invalid")
            index = int(mode.get("hypothesis_index", -1))
            if not 0 <= index < len(poses):
                raise ValueError("hard-mode hypothesis index is outside its target-free source")
            recorded_score = float(mode.get("score", np.nan))
            if not np.isfinite(recorded_score) or not np.isclose(
                recorded_score, scores[index], rtol=1e-7, atol=1e-7
            ):
                raise ValueError("hard-mode selection does not match its frozen source hypothesis")
            query_ids.append(query_id)
            split_names.append("train")
            correct_poses.append(gt_pose)
            wrong_poses.append(poses[index])
            hypothesis_indices.append(index)
            support_counts.append(int(mode.get("consistent_group_count", 0)))
    validate_train_pair_splits(
        query_ids=np.asarray(query_ids), split_names=np.asarray(split_names)
    )
    if not query_ids:
        raise ValueError("hard-mode mining selected no train pose pairs")

    expected_query_ids = np.asarray(hard_arrays["hard_mode_query_ids_TARGET_ONLY"]).astype(str)
    expected_support = np.asarray(
        hard_arrays["hard_mode_support_group_counts_TARGET_ONLY"], dtype=np.int64
    )
    if (
        expected_query_ids.shape != (len(query_ids),)
        or expected_support.shape != (len(query_ids),)
        or not np.array_equal(expected_query_ids, np.asarray(query_ids))
        or not np.array_equal(expected_support, np.asarray(support_counts, dtype=np.int64))
    ):
        raise ValueError("hard-mode per-query rows do not reproduce structured mode membership")

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": TRAIN_PAIR_FORMAT,
        "contains_ground_truth": True,
        "contains_validation_or_test_targets": False,
        "training_only_target_artifact": True,
        "correct_pose_source": "model_train_images_bin_train_query_only",
        "coherent_wrong_pose_source": "train_only_post_inference_hard_modes_v2",
        "wrong_pose_hypotheses_target_free_before_posthoc_train_join": True,
        "verification_points_are_target_free": True,
        "historical_candidate_layout_used_only_to_mine_wrong_pose_modes": True,
        "hypothesis_semantic_lineage": hypothesis_lineage,
        "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
        "inputs": {
            "verification_points": str(verification_points_path),
            "verification_points_sha256": file_sha256_short(Path(verification_points_path)),
            "hard_modes": str(hard_modes_path),
            "hard_modes_sha256": file_sha256_short(Path(hard_modes_path)),
            "hard_mode_per_query": str(Path(hard_modes_path).parent / "per_query_TARGET_ONLY.json"),
            "hard_mode_per_query_sha256": file_sha256_short(
                Path(hard_modes_path).parent / "per_query_TARGET_ONLY.json"
            ),
            "hypothesis_artifacts": pose_manifests,
            "colmap_images_bin": str(Path(colmap_model_dir) / "images.bin"),
            "colmap_images_bin_sha256": file_sha256_short(Path(colmap_model_dir) / "images.bin"),
        },
        "pair_count": int(len(query_ids)),
        "query_count": int(len(set(query_ids))),
        "evaluation_label": evaluation_label,
    }
    np.savez_compressed(
        output,
        query_ids=np.asarray(query_ids),
        split_names=np.asarray(split_names),
        coherent_wrong_hypothesis_indices=np.asarray(hypothesis_indices, dtype=np.int64),
        correct_poses_w2c=np.stack(correct_poses).astype(np.float64),
        coherent_wrong_poses_w2c=np.stack(wrong_poses).astype(np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "stage": "build_candidate_pose_llr_train_pairs",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "pair_count": int(len(query_ids)),
        "query_count": int(len(set(query_ids))),
        "protocol": {
            "train_only": True,
            "verification_points_target_free": True,
            "runtime_scorer_must_not_load_target_pairs": True,
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-points", required=True)
    parser.add_argument("--hard-modes", required=True)
    parser.add_argument("--reference-hypothesis-artifact", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_candidate_pose_llr_train_pairs(
        verification_points_path=Path(args.verification_points),
        hard_modes_path=Path(args.hard_modes),
        reference_hypothesis_path=Path(args.reference_hypothesis_artifact),
        colmap_model_dir=Path(args.colmap_model_dir),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
