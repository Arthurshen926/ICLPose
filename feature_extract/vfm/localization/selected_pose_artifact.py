"""Build strict selected-pose artifacts from frozen grouped hypotheses.

The grouped-hypothesis export intentionally retains every generated pose.  A
cross-fit scorer, however, sometimes needs one fixed external source pose per
query.  This module extracts only the inference-time optional choice and
records the source lineage without carrying target errors forward.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


SELECTED_POSE_ARTIFACT_FORMAT = "selected_pose_inference_only_v1"
GROUPED_HYPOTHESIS_ARTIFACT_FORMAT = "grouped_pose_hypotheses_inference_only_v1"


def canonical_manifest_sha256(value: object) -> str:
    """Return the short canonical hash used by selected-pose artifacts."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _metadata(payload: Mapping[str, np.ndarray], path: Path) -> dict[str, object]:
    if "metadata_json" not in payload:
        raise ValueError(f"{path}: grouped hypothesis artifact has no metadata")
    try:
        metadata = json.loads(str(payload["metadata_json"].item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: grouped hypothesis metadata is invalid") from exc
    if metadata.get("format") != GROUPED_HYPOTHESIS_ARTIFACT_FORMAT:
        raise ValueError(f"{path}: unsupported grouped hypothesis artifact format")
    if bool(metadata.get("contains_target_fields")) or bool(
        metadata.get("pose_or_ground_truth_used_for_generation")
    ):
        raise ValueError(f"{path}: grouped hypothesis artifact is not target-free")
    inputs = metadata.get("inputs")
    config = metadata.get("grouped_config")
    if not isinstance(inputs, dict) or not isinstance(config, dict):
        raise ValueError(f"{path}: grouped hypothesis lineage is incomplete")
    return metadata


def _selected_rows_from_payload(
    payload: Mapping[str, np.ndarray],
    *,
    path: Path,
    evaluation_label: str,
) -> list[dict[str, object]]:
    required = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "chosen_for_optional_pose",
        "poses_w2c",
        "verification_effective_group_counts",
        "verification_strict_inlier_counts",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"{path}: grouped hypothesis artifact misses fields: {sorted(missing)}"
        )
    query_ids = np.asarray(payload["query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(payload["split_names"]).astype(str).reshape(-1)
    labels = np.asarray(payload["evaluation_labels"]).astype(str).reshape(-1)
    chosen = np.asarray(payload["chosen_for_optional_pose"], dtype=bool).reshape(-1)
    poses = np.asarray(payload["poses_w2c"], dtype=np.float64)
    effective_groups = np.asarray(
        payload["verification_effective_group_counts"], dtype=np.int64
    ).reshape(-1)
    strict_inliers = np.asarray(
        payload["verification_strict_inlier_counts"], dtype=np.int64
    ).reshape(-1)
    count = len(query_ids)
    if (
        split_names.shape != (count,)
        or labels.shape != (count,)
        or chosen.shape != (count,)
        or poses.shape != (count, 4, 4)
        or effective_groups.shape != (count,)
        or strict_inliers.shape != (count,)
    ):
        raise ValueError(f"{path}: grouped hypothesis arrays have incompatible shapes")
    selected_label = str(evaluation_label)
    if not selected_label:
        unique_labels = np.unique(labels)
        if len(unique_labels) != 1:
            raise ValueError(
                f"{path}: multiple hypothesis labels require an explicit selection"
            )
        selected_label = str(unique_labels[0])
    rows: list[dict[str, object]] = []
    identities = sorted(
        {
            (str(split_name), str(query_id))
            for split_name, query_id, label in zip(split_names, query_ids, labels)
            if str(label) == selected_label
        }
    )
    if not identities:
        raise ValueError(f"{path}: selected hypothesis label has no rows")
    for split_name, query_id in identities:
        mask = (
            (split_names == split_name)
            & (query_ids == query_id)
            & (labels == selected_label)
            & chosen
        )
        indices = np.flatnonzero(mask)
        if len(indices) != 1:
            raise ValueError(
                f"{path}: {split_name}/{query_id} has {len(indices)} chosen poses"
            )
        index = int(indices[0])
        pose = poses[index]
        success = bool(np.all(np.isfinite(pose)))
        if not success and not np.all(np.isnan(pose)):
            raise ValueError(f"{path}: failed selected pose is neither finite nor NaN")
        group_count = max(0, int(effective_groups[index]))
        inlier_count = max(0, int(strict_inliers[index]))
        rows.append(
            {
                "query_id": query_id,
                "split_name": split_name,
                "evaluation_label": selected_label,
                "success": success,
                "pose_w2c": pose.copy() if success else None,
                # These are held-out group counts, not re-estimated PnP counts.
                "match_count": max(group_count, inlier_count),
                "inlier_count": min(group_count, inlier_count),
            }
        )
    return rows


def selected_pose_rows_from_grouped_hypotheses(
    hypothesis_paths: Sequence[Path],
    *,
    evaluation_label: str = "",
    expected_query_ids: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Extract one frozen optional pose per query with strict shared lineage."""

    paths = tuple(Path(path) for path in hypothesis_paths)
    if not paths:
        raise ValueError("at least one grouped hypothesis artifact is required")
    if len(set(paths)) != len(paths):
        raise ValueError("grouped hypothesis artifact paths must be unique")

    all_rows: list[dict[str, object]] = []
    source_shards: list[dict[str, str]] = []
    canonical_inputs: dict[str, object] | None = None
    canonical_config: dict[str, object] | None = None
    selected_label = str(evaluation_label)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        payload = _load_npz(path)
        metadata = _metadata(payload, path)
        inputs = dict(metadata["inputs"])
        config = dict(metadata["grouped_config"])
        if canonical_inputs is None:
            canonical_inputs = inputs
            canonical_config = config
        elif inputs != canonical_inputs or config != canonical_config:
            raise ValueError("grouped hypothesis artifacts have incompatible lineage")
        local_rows = _selected_rows_from_payload(
            payload,
            path=path,
            evaluation_label=selected_label,
        )
        local_labels = {str(row["evaluation_label"]) for row in local_rows}
        if len(local_labels) != 1:
            raise RuntimeError("selected pose extraction produced ambiguous labels")
        local_label = next(iter(local_labels))
        if selected_label and local_label != selected_label:
            raise ValueError(f"{path}: selected label differs from requested label")
        selected_label = local_label
        all_rows.extend(local_rows)
        source_shards.append(
            {"path": str(path), "sha256": file_sha256_short(path)}
        )
    identities = [
        (str(row["split_name"]), str(row["query_id"])) for row in all_rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("grouped hypothesis shards contain duplicate query poses")
    if expected_query_ids is not None:
        expected = {
            (str(split_name), str(query_id))
            for split_name, query_ids in expected_query_ids.items()
            for query_id in query_ids
        }
        actual = set(identities)
        if actual != expected:
            raise ValueError(
                "grouped hypothesis source coverage differs from expected splits: "
                f"missing={sorted(expected - actual)[:5]}, "
                f"extra={sorted(actual - expected)[:5]}"
            )
    if canonical_inputs is None or canonical_config is None:
        raise RuntimeError("grouped hypothesis source extraction has no lineage")
    source_manifest = {
        "stage": "selected_pose_from_grouped_hypotheses_v1",
        "inputs": canonical_inputs,
        "protocol": {
            "source_pose_kind": "frozen_chosen_optional_grouped_hypothesis",
            "selection_field": "chosen_for_optional_pose",
            "ground_truth_available_to_source_selector": False,
            "source_hypothesis_format": GROUPED_HYPOTHESIS_ARTIFACT_FORMAT,
            "source_grouped_config": canonical_config,
        },
        "execution": {
            "source_query_count": int(len(all_rows)),
            "source_shards": source_shards,
        },
    }
    return sorted(
        all_rows, key=lambda row: (str(row["split_name"]), str(row["query_id"]))
    ), source_manifest


def write_selected_pose_artifact(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    source_manifest: Mapping[str, object],
) -> None:
    """Write a selected-pose artifact accepted by the immutable-source loader."""

    if not rows:
        raise ValueError("selected pose artifact requires at least one row")
    identities: set[tuple[str, str, str]] = set()
    poses = np.full((len(rows), 4, 4), np.nan, dtype=np.float64)
    success = np.zeros((len(rows),), dtype=bool)
    match_counts = np.zeros((len(rows),), dtype=np.int64)
    inlier_counts = np.zeros((len(rows),), dtype=np.int64)
    for index, row in enumerate(rows):
        identity = (
            str(row["split_name"]),
            str(row["evaluation_label"]),
            str(row["query_id"]),
        )
        if identity in identities:
            raise ValueError(f"duplicate selected pose identity: {identity}")
        identities.add(identity)
        row_success = bool(row["success"])
        match_count = int(row["match_count"])
        inlier_count = int(row["inlier_count"])
        if match_count < 0 or not 0 <= inlier_count <= match_count:
            raise ValueError("selected pose match/inlier counts are invalid")
        success[index] = row_success
        match_counts[index] = match_count
        inlier_counts[index] = inlier_count
        if row_success:
            pose = np.asarray(row["pose_w2c"], dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(pose)):
                raise ValueError("successful selected pose is non-finite")
            poses[index] = pose
        elif row.get("pose_w2c") is not None:
            raise ValueError("failed selected pose must not carry a pose matrix")
    manifest = dict(source_manifest)
    metadata = {
        "format": SELECTED_POSE_ARTIFACT_FORMAT,
        "row_count": int(len(rows)),
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_reestimated_during_replay": False,
        "source_manifest_sha256": canonical_manifest_sha256(manifest),
        "source_manifest": manifest,
    }
    np.savez_compressed(
        path,
        query_ids=np.asarray([str(row["query_id"]) for row in rows]),
        split_names=np.asarray([str(row["split_name"]) for row in rows]),
        evaluation_labels=np.asarray(
            [str(row["evaluation_label"]) for row in rows]
        ),
        success=success,
        poses_w2c=poses,
        match_counts=match_counts,
        inlier_counts=inlier_counts,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
