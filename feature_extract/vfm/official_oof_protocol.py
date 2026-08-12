"""Machine-checkable official-train OOF protocol for GoalMaplet localization."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


STMARYS_ROUTE_FOLDS_V1: tuple[tuple[str, ...], ...] = (
    ("seq2",),
    ("seq4",),
    ("seq1", "seq8", "seq14"),
    ("seq7", "seq12"),
    ("seq6", "seq9", "seq10", "seq11"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_id_sha256(image_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in image_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_cambridge_image_ids(path: Path) -> list[str]:
    image_ids = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith("seq") and "/" in fields[0]:
            image_ids.append(fields[0])
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"duplicate image IDs in pose file: {path}")
    return image_ids


def parse_token_manifest_image_ids(path: Path) -> list[str]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"token manifest has no records list: {path}")
    image_ids = [str(record["image_id"]) for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"duplicate image IDs in token manifest: {path}")
    return image_ids


def _trajectory_counts(image_ids: Sequence[str]) -> dict[str, int]:
    return dict(
        sorted(Counter(value.split("/", 1)[0] for value in image_ids).items())
    )


def validate_route_folds(
    train_image_ids: Sequence[str],
    folds: Sequence[Sequence[str]] = STMARYS_ROUTE_FOLDS_V1,
) -> list[dict[str, object]]:
    train_trajectories = set(_trajectory_counts(train_image_ids))
    held_trajectories = [trajectory for fold in folds for trajectory in fold]
    duplicates = sorted(
        trajectory
        for trajectory, count in Counter(held_trajectories).items()
        if count != 1
    )
    if duplicates:
        raise ValueError(f"trajectories occur in multiple folds: {duplicates}")
    if set(held_trajectories) != train_trajectories:
        missing = sorted(train_trajectories - set(held_trajectories))
        extra = sorted(set(held_trajectories) - train_trajectories)
        raise ValueError(f"fold partition mismatch: missing={missing}, extra={extra}")

    rows = []
    all_ids = set(train_image_ids)
    for index, fold in enumerate(folds):
        held = tuple(sorted(str(value) for value in fold))
        query_ids = sorted(
            image_id
            for image_id in train_image_ids
            if image_id.split("/", 1)[0] in held
        )
        mapping_ids = sorted(all_ids - set(query_ids))
        rows.append(
            {
                "fold_id": f"fold{index}",
                "held_query_trajectories": list(held),
                "mapping_trajectories": sorted(train_trajectories - set(held)),
                "query_count": len(query_ids),
                "mapping_count": len(mapping_ids),
                "query_image_ids_sha256": ordered_id_sha256(query_ids),
                "mapping_image_ids_sha256": ordered_id_sha256(mapping_ids),
            }
        )
    if sum(int(row["query_count"]) for row in rows) != len(train_image_ids):
        raise AssertionError("OOF queries do not cover official train exactly once")
    return rows


def build_stmarys_protocol(
    *,
    train_pose_file: Path,
    test_pose_file: Path,
    train_token_manifest: Path,
    test_token_manifest: Path,
    extra_train_manifests: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    train_ids = parse_cambridge_image_ids(train_pose_file)
    test_ids = parse_cambridge_image_ids(test_pose_file)
    train_set = set(train_ids)
    test_set = set(test_ids)
    overlap = sorted(train_set & test_set)
    if overlap:
        raise ValueError(f"official train/test overlap: {overlap[:8]}")

    train_token_ids = parse_token_manifest_image_ids(train_token_manifest)
    test_token_ids = parse_token_manifest_image_ids(test_token_manifest)
    if set(train_token_ids) != train_set:
        raise ValueError("train RADIO token manifest does not equal official train")
    if set(test_token_ids) != test_set:
        raise ValueError("test RADIO token manifest does not equal official test")
    for name, image_ids in extra_train_manifests.items():
        if set(image_ids) != train_set or len(image_ids) != len(train_ids):
            raise ValueError(f"{name} does not equal official train")

    folds = validate_route_folds(train_ids)
    return {
        "artifact_type": "goal_maplet_official_train_oof_protocol_v1",
        "benchmark": "CambridgeLandmarks/StMarysChurch",
        "official_train": {
            "count": len(train_ids),
            "trajectory_counts": _trajectory_counts(train_ids),
            "image_ids_sha256": ordered_id_sha256(train_ids),
            "pose_file": str(train_pose_file),
            "pose_file_sha256": file_sha256(train_pose_file),
            "token_manifest": str(train_token_manifest),
            "token_manifest_sha256": file_sha256(train_token_manifest),
        },
        "official_test": {
            "count": len(test_ids),
            "trajectory_counts": _trajectory_counts(test_ids),
            "image_ids_sha256": ordered_id_sha256(test_ids),
            "pose_file": str(test_pose_file),
            "pose_file_sha256": file_sha256(test_pose_file),
            "token_manifest": str(test_token_manifest),
            "token_manifest_sha256": file_sha256(test_token_manifest),
            "use": "frozen_final_evaluation_only",
            "historical_exposure": (
                "The current project used this split in pre-G23 diagnostics; "
                "it is the standard benchmark test, not a newly untouched set."
            ),
        },
        "development": {
            "method": "five_fold_route_grouped_out_of_fold",
            "uses_permanent_validation_split": False,
            "each_official_train_frame_is_oof_query_once": True,
            "folds": folds,
            "screening_geometry_protocol": (
                "fixed_full_train_2dgs_non_publishable_screening"
            ),
            "confirmation_geometry_protocol": (
                "fold_rebuilt_2dgs_excluding_held_query_trajectories"
            ),
        },
        "final_fit": {
            "train_count": len(train_ids),
            "training_data": "all_official_train",
            "hyperparameters": "frozen_from_train_only_oof",
            "test_feedback_allowed": False,
        },
        "integrity": {
            "official_train_test_disjoint": True,
            "train_token_ids_equal_pose_ids": True,
            "test_token_ids_equal_pose_ids": True,
            "extra_train_manifest_counts": {
                name: len(image_ids) for name, image_ids in extra_train_manifests.items()
            },
        },
    }
