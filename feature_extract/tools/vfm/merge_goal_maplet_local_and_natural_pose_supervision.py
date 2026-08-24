"""Merge GT-local probes with pose-free natural hard negatives by query ID."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.train_evaluate_goal_maplet_fulltoken_pose_ranker import (
    LOCAL_LABEL_SCHEMA,
    _load_feature_artifact,
    _load_local_supervision_dataset,
)
from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    FULLTOKEN_POSE_RANKING_CHANNELS,
    FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)


MERGED_SUPERVISION_SEMANTICS = (
    "gt_anchor_plus_local_multiscale_and_pose_free_natural_hard_negatives_v1"
)
FEATURE_SCHEMA = "goal_maplet_compact_fulltoken_pose_ranking_features_v1"


def _float16_ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return representable-step distance, treating signed zero as identical."""
    first = np.asarray(left)
    second = np.asarray(right)
    if first.dtype != np.float16 or second.dtype != np.float16 or first.shape != second.shape:
        raise ValueError("anchor evidence must have matching float16 arrays")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("anchor evidence must be finite")

    def _ordered(value: np.ndarray) -> np.ndarray:
        bits = value.view(np.uint16).astype(np.int32)
        sign = (bits & 0x8000) != 0
        magnitude = bits & 0x7FFF
        return np.where(sign, 0x8000 - magnitude, 0x8000 + magnitude)

    return np.abs(_ordered(first) - _ordered(second))


def _validate_anchor_evidence(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    """Bound float16 replay drift without hiding semantic-channel changes.

    Stable mass/evidence/token channels may move by one representable step.
    Signed phase cosine is a reduction around zero, where an equally tiny
    absolute perturbation spans multiple float16 subnormal/near-zero steps;
    it therefore uses the same half-epsilon absolute bound instead.
    """
    distance = _float16_ulp_distance(left, right)
    maximum = int(np.max(distance, initial=0))
    first = np.asarray(left, dtype=np.float16)
    second = np.asarray(right, dtype=np.float16)
    absolute = np.abs(first.astype(np.float32) - second.astype(np.float32))
    maximum_absolute = float(np.max(absolute, initial=0.0))
    half_epsilon = float(2.0 ** -11)
    if maximum_absolute > half_epsilon:
        raise ValueError("local and natural rendered GT anchor evidence differs in absolute value")
    phase_channels = tuple(
        index for index, name in enumerate(FULLTOKEN_POSE_RANKING_CHANNELS)
        if "signed_phase_cosine" in name
    )
    stable_channels = tuple(
        index for index in range(len(FULLTOKEN_POSE_RANKING_CHANNELS))
        if index not in phase_channels
    )
    if first.ndim != 3 or first.shape[0] != len(FULLTOKEN_POSE_RANKING_CHANNELS):
        raise ValueError("anchor evidence channel layout differs")
    maximum_stable_ulp = int(np.max(distance[list(stable_channels)], initial=0))
    if maximum_stable_ulp > 1:
        raise ValueError("stable local/natural GT anchor evidence differs by over one ULP")
    changed = int(np.count_nonzero(distance))
    return {
        "maximum_float16_ulp_distance": maximum,
        "maximum_stable_channel_float16_ulp_distance": maximum_stable_ulp,
        "maximum_absolute_difference": maximum_absolute,
        "changed_value_count": changed,
        "value_count": int(distance.size),
    }


def _parse_route_limits(values: list[str]) -> list[tuple[str, int]]:
    result = []
    for value in values:
        route, separator, count = str(value).partition(":")
        if not separator or not route or not count.isdigit() or int(count) <= 0:
            raise ValueError("route limits must use ROUTE:POSITIVE_COUNT")
        result.append((route, int(count)))
    if not result or len({route for route, _ in result}) != len(result):
        raise ValueError("route limits must be nonempty and unique")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_dataset", required=True)
    parser.add_argument("--local_features", required=True)
    parser.add_argument("--local_feature_manifest", required=True)
    parser.add_argument("--natural_dataset", action="append", required=True)
    parser.add_argument("--natural_features", action="append", required=True)
    parser.add_argument("--natural_feature_manifest", action="append", required=True)
    parser.add_argument("--route_limits", nargs="+", required=True)
    parser.add_argument("--output_features", required=True)
    parser.add_argument("--output_labels", required=True)
    parser.add_argument("--output_manifest", required=True)
    args = parser.parse_args()

    natural_arguments = (
        list(args.natural_dataset), list(args.natural_features),
        list(args.natural_feature_manifest),
    )
    if len({len(value) for value in natural_arguments}) != 1:
        raise ValueError("natural dataset/feature/manifest counts differ")
    output_feature = Path(args.output_features)
    output_label = Path(args.output_labels)
    output_manifest = Path(args.output_manifest)
    partial = output_feature.with_suffix(".partial.npy")
    if output_feature.suffix != ".npy" or output_label.suffix != ".npz":
        raise ValueError("merged feature/label outputs must use .npy/.npz")
    if any(path.exists() for path in (output_feature, output_label, output_manifest, partial)):
        raise FileExistsError("refusing to overwrite merged supervision artifacts")

    local_path = Path(args.local_dataset)
    local_arrays, local_metadata = _load_local_supervision_dataset(local_path)
    local_features, local_manifest = _load_feature_artifact(
        Path(args.local_features), Path(args.local_feature_manifest),
        dataset_path=local_path,
        dataset_content_sha256=str(local_metadata["content_sha256"]),
    )
    natural_by_id: dict[
        str, tuple[dict[str, np.ndarray], int, np.ndarray, int, dict[str, object]]
    ] = {}
    natural_bindings = []
    for dataset_value, feature_value, manifest_value in zip(*natural_arguments):
        dataset_path = Path(dataset_value)
        arrays, metadata = load_pose_candidate_dataset(
            dataset_path, require_rendered_targets=False,
        )
        if (
            metadata.get("candidate_zero_is_diagnostic_gt_anchor") is not True
            or metadata.get("candidate_pool_frozen_before_target_pose_opened") is not True
        ):
            raise ValueError("natural hard-negative pool was not frozen before labels")
        features, manifest = _load_feature_artifact(
            Path(feature_value), Path(manifest_value), dataset_path=dataset_path,
            dataset_content_sha256=str(metadata["content_sha256"]),
        )
        query_range = manifest.get("query_range")
        if features.shape[1] != arrays["candidate_valid"].shape[1]:
            raise ValueError("natural hard-negative candidate counts differ")
        if features.shape[0] == arrays["candidate_valid"].shape[0]:
            dataset_rows = np.arange(features.shape[0], dtype=np.int64)
        elif (
            isinstance(query_range, list) and len(query_range) == 2
            and int(query_range[1]) - int(query_range[0]) == features.shape[0]
            and 0 <= int(query_range[0]) <= int(query_range[1]) <= arrays["image_ids"].size
        ):
            dataset_rows = np.arange(int(query_range[0]), int(query_range[1]), dtype=np.int64)
        else:
            raise ValueError("natural hard-negative feature query range differs")
        for feature_row, row in enumerate(dataset_rows.tolist()):
            image_id = arrays["image_ids"][row]
            key = str(image_id)
            if key in natural_by_id:
                raise ValueError("duplicate natural hard-negative query")
            natural_by_id[key] = (arrays, row, features, feature_row, metadata)
        natural_bindings.append({
            "dataset_file_sha256": file_sha256(dataset_path),
            "dataset_content_sha256": metadata["content_sha256"],
            "feature_file_sha256": manifest["feature_file_sha256"],
            "feature_manifest_file_sha256": file_sha256(Path(manifest_value)),
        })

    selected = []
    for route, count in _parse_route_limits(list(args.route_limits)):
        rows = [
            row for row, image_id in enumerate(local_arrays["image_ids"].tolist())
            if str(image_id).split("/", 1)[0] == route
        ]
        if len(rows) < count:
            raise ValueError(f"local supervision route {route} lacks {count} rows")
        selected.extend(rows[:count])
    selected = np.asarray(selected, dtype=np.int64)
    image_ids = np.asarray(local_arrays["image_ids"][selected], dtype=str)
    if any(str(image_id) not in natural_by_id for image_id in image_ids.tolist()):
        raise ValueError("natural hard negatives do not cover every selected local query")

    first_arrays, _, first_features, _, _ = natural_by_id[str(image_ids[0])]
    natural_count = int(first_arrays["candidate_valid"].shape[1])
    local_count = int(local_arrays["candidate_valid"].shape[1])
    merged_count = local_count + natural_count - 1
    feature_shape = (
        int(selected.size), merged_count, len(FULLTOKEN_POSE_RANKING_CHANNELS), 36, 64,
    )
    output_feature.parent.mkdir(parents=True, exist_ok=True)
    merged_features = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float16, shape=feature_shape,
    )
    poses = np.empty((selected.size, merged_count, 4, 4), dtype=np.float64)
    translation = np.empty((selected.size, merged_count), dtype=np.float32)
    rotation = np.empty_like(translation)
    valid = np.empty((selected.size, merged_count), dtype=bool)
    anchor_replay_rows = []
    for output_row, local_row in enumerate(selected.tolist()):
        image_id = str(local_arrays["image_ids"][local_row])
        natural_arrays, natural_row, natural_features, natural_feature_row, _ = natural_by_id[image_id]
        if int(natural_arrays["candidate_valid"].shape[1]) != natural_count:
            raise ValueError("natural hard-negative candidate counts differ")
        if not np.array_equal(
            local_arrays["candidate_poses_w2c"][local_row, 0],
            natural_arrays["candidate_poses_w2c"][natural_row, 0],
        ):
            raise ValueError("local and natural GT anchor poses differ")
        replay = _validate_anchor_evidence(
            np.asarray(local_features[local_row, 0]),
            np.asarray(natural_features[natural_feature_row, 0]),
        )
        anchor_replay_rows.append({"image_id": image_id, **replay})
        for destination, source in (
            (poses, "candidate_poses_w2c"),
            (translation, "translation_m"),
            (rotation, "rotation_deg"),
            (valid, "candidate_valid"),
        ):
            destination[output_row, :local_count] = local_arrays[source][local_row]
            destination[output_row, local_count:] = natural_arrays[source][natural_row, 1:]
        merged_features[output_row, :local_count] = local_features[local_row]
        merged_features[output_row, local_count:] = natural_features[natural_feature_row, 1:]
    merged_features.flush()
    del merged_features
    os.replace(partial, output_feature)

    label_arrays = {
        "image_ids": image_ids,
        "source_query_rows": np.asarray(local_arrays["source_query_rows"][selected]),
        "candidate_poses_w2c": poses,
        "translation_m": translation,
        "rotation_deg": rotation,
        "candidate_valid": valid,
    }
    label_metadata = {
        "artifact_type": LOCAL_LABEL_SCHEMA,
        "content_sha256": arrays_sha256(label_arrays),
        "supervision_semantics": MERGED_SUPERVISION_SEMANTICS,
        "local_candidate_count_including_anchor": local_count,
        "natural_candidate_count_including_duplicate_anchor": natural_count,
        "merged_candidate_count": merged_count,
        "natural_candidate_start_row": local_count,
        "local_source_dataset_file_sha256": file_sha256(local_path),
        "local_source_dataset_content_sha256": local_metadata["content_sha256"],
        "natural_source_bindings": natural_bindings,
        "duplicate_gt_anchor_replay_contract": (
            "same_pose_stable_channels_one_float16_ulp_phase_channels_half_epsilon_v2"
        ),
        "duplicate_gt_anchor_replay_maximum_float16_ulp_distance": max(
            int(row["maximum_float16_ulp_distance"]) for row in anchor_replay_rows
        ),
        "duplicate_gt_anchor_replay_changed_value_count": sum(
            int(row["changed_value_count"]) for row in anchor_replay_rows
        ),
        "duplicate_gt_anchor_replay_value_count": sum(
            int(row["value_count"]) for row in anchor_replay_rows
        ),
        "duplicate_gt_anchor_replay_maximum_absolute_difference": max(
            float(row["maximum_absolute_difference"]) for row in anchor_replay_rows
        ),
        "uses_gt_for_training_candidate_generation": True,
        "natural_candidate_pools_frozen_before_target_pose_opened": True,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
    }
    output_label.parent.mkdir(parents=True, exist_ok=True)
    temporary_label = output_label.with_name(output_label.name + ".tmp.npz")
    np.savez_compressed(
        temporary_label, **label_arrays,
        metadata_json=np.asarray(json.dumps(label_metadata, sort_keys=True)),
    )
    os.replace(temporary_label, output_label)
    manifest = {
        "artifact_type": FEATURE_SCHEMA,
        "feature_semantics": FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
        "feature_channels": list(FULLTOKEN_POSE_RANKING_CHANNELS),
        "feature_shape": list(feature_shape),
        "feature_dtype": "float16",
        "feature_file": str(output_feature.resolve()),
        "feature_file_sha256": file_sha256(output_feature),
        "dataset_file_sha256": file_sha256(output_label),
        "dataset_content_sha256": label_metadata["content_sha256"],
        "supervision_semantics": MERGED_SUPERVISION_SEMANTICS,
        "production_eligible": False,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "query_count": int(selected.size),
        "candidate_count": merged_count,
        "feature_file_sha256": manifest["feature_file_sha256"],
        "label_content_sha256": label_metadata["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
