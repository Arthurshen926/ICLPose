"""Train paired mode-relation LLRs on fixed query edges and option sets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.mode_relation import (
    EDGE_FAMILIES,
    FEATURE_NAMES,
    ModeRelationLikelihoodRatioArtifact,
    analytic_relation_score,
)


POSITIVE_SOURCE = "exact_gt_configuration"


def _load(paths: list[Path]) -> tuple[dict[str, np.ndarray], dict]:
    required = (
        "features", "targets", "image_ids", "trajectories", "source_types",
        "edge_left_group_rows", "edge_right_group_rows", "edge_families", "edge_roles",
        "candidate_indices", "pose_translation_m", "pose_rotation_deg", "relation_null_types",
    )
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            current = json.loads(str(np.asarray(data["metadata_json"]).item()))
            for key in required:
                parts[key].append(np.asarray(data[key]))
        if current.get("artifact_type") != "goal_maplet_mode_relation_samples_v1":
            raise ValueError("relation training requires mode-relation samples v1")
        if current.get("pairing_contract") != "same_image_same_query_edge_fixed_options_v1":
            raise ValueError("relation sample pairing contract differs")
        if current.get("edge_contract") != "query_only_fit_tree_disjoint_verify_v1":
            raise ValueError("relation sample edge contract differs")
        if tuple(current.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("relation sample feature contract differs")
        if metadata is None:
            metadata = current
        else:
            for key in (
                "physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256",
                "child_eligibility_sha256", "candidate_pool_sha256", "temperature",
                "maximum_modes", "runtime_maximum_children", "pairing_contract", "edge_contract",
                "option_contract",
            ):
                if current.get(key) != metadata.get(key):
                    raise ValueError(f"relation sample shards differ: {key}")
    arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    count = arrays["features"].shape[0]
    if arrays["features"].shape != (count, len(FEATURE_NAMES)):
        raise ValueError("relation sample feature shape differs")
    if any(value.shape[0] != count for value in arrays.values()):
        raise ValueError("relation sample arrays differ")
    for key in ("image_ids", "trajectories", "source_types"):
        arrays[key] = arrays[key].astype(str)
    return arrays, dict(metadata)


def _pairs(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    image = arrays["image_ids"]
    left = np.asarray(arrays["edge_left_group_rows"], dtype=np.int64)
    right = np.asarray(arrays["edge_right_group_rows"], dtype=np.int64)
    role = np.asarray(arrays["edge_roles"], dtype=np.int64)
    source = arrays["source_types"]
    target = np.asarray(arrays["targets"], dtype=np.int64)
    positive_by_key: dict[tuple[str, int, int, int], int] = {}
    for row in np.flatnonzero(mask & (source == POSITIVE_SOURCE) & (target == 1)).tolist():
        key = (str(image[row]), int(left[row]), int(right[row]), int(role[row]))
        if key in positive_by_key:
            raise ValueError(f"duplicate exact relation positive: {key}")
        positive_by_key[key] = row
    positive, negative = [], []
    for row in np.flatnonzero(mask & (source != POSITIVE_SOURCE) & (target == 0)).tolist():
        key = (str(image[row]), int(left[row]), int(right[row]), int(role[row]))
        if key in positive_by_key:
            positive.append(positive_by_key[key])
            negative.append(row)
    if not positive:
        raise ValueError("no exact within-edge relation pairs")
    positive = np.asarray(positive, dtype=np.int64)
    negative = np.asarray(negative, dtype=np.int64)
    return {
        "positive_features": np.asarray(arrays["features"], dtype=np.float32)[positive],
        "negative_features": np.asarray(arrays["features"], dtype=np.float32)[negative],
        "image_ids": image[negative], "trajectories": arrays["trajectories"][negative],
        "source_types": source[negative],
        "edge_families": np.asarray(arrays["edge_families"], dtype=np.int64)[negative],
        "edge_roles": role[negative],
        "edge_left_group_rows": left[negative], "edge_right_group_rows": right[negative],
        "candidate_indices": np.asarray(arrays["candidate_indices"], dtype=np.int64)[negative],
    }


def _balanced_weight(image: np.ndarray, source: np.ndarray, family: np.ndarray) -> np.ndarray:
    key = np.asarray([f"{a}\0{b}\0{c}" for a, b, c in zip(image, source, family)])
    _, inverse, count = np.unique(key, return_inverse=True, return_counts=True)
    weight = 1.0 / count[inverse]
    return weight * weight.size / max(float(np.sum(weight)), 1e-12)


def _fit(rows: dict[str, np.ndarray]):
    difference = rows["positive_features"] - rows["negative_features"]
    x = np.concatenate([difference, -difference], axis=0)
    y = np.concatenate([np.ones(difference.shape[0]), np.zeros(difference.shape[0])]).astype(np.int64)
    weight = _balanced_weight(rows["image_ids"], rows["source_types"], rows["edge_families"])
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.125, fit_intercept=False, max_iter=4000, random_state=214133),
    )
    with threadpool_limits(limits=8):
        estimator.fit(x, y, logisticregression__sample_weight=np.concatenate([weight, weight]))
    return estimator


def _raw(estimator, features: np.ndarray) -> np.ndarray:
    return np.asarray(estimator.decision_function(features), dtype=np.float64).reshape(-1)


def _calibrate(estimator, rows: dict[str, np.ndarray]) -> tuple[float, float]:
    key = np.asarray([
        f"{image}\0{left}\0{right}\0{role}"
        for image, left, right, role in zip(
            rows["image_ids"], rows["edge_left_group_rows"],
            rows["edge_right_group_rows"], rows["edge_roles"],
        )
    ])
    _, first = np.unique(key, return_index=True)
    positive = rows["positive_features"][np.sort(first)]
    negative = rows["negative_features"]
    x = np.concatenate([_raw(estimator, positive), _raw(estimator, negative)])[:, None]
    y = np.concatenate([np.ones(positive.shape[0]), np.zeros(negative.shape[0])]).astype(np.int64)
    weight = np.where(y == 1, 0.5 / positive.shape[0], 0.5 / negative.shape[0])
    weight *= y.size / np.sum(weight)
    calibrator = LogisticRegression(C=1.0e6, max_iter=2000, random_state=214133)
    calibrator.fit(x, y, sample_weight=weight)
    return float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])


def _report(estimator, scale: float, intercept: float, rows: dict[str, np.ndarray]) -> dict:
    positive = scale * _raw(estimator, rows["positive_features"]) + intercept
    negative = scale * _raw(estimator, rows["negative_features"]) + intercept
    margin = positive - negative
    analytic_margin = analytic_relation_score(
        rows["positive_features"], np.ones(margin.shape, dtype=bool),
    ) - analytic_relation_score(rows["negative_features"], np.ones(margin.shape, dtype=bool))

    def breakdown(values: np.ndarray, labels: np.ndarray, names=None) -> dict:
        result = {}
        for value in sorted(set(labels.tolist())):
            selected = labels == value
            key = names[int(value)] if names is not None else str(value)
            result[key] = {
                "pair_count": int(np.sum(selected)),
                "concordance": float(np.mean(values[selected] > 0.0)),
                "margin_median": float(np.median(values[selected])),
            }
        return result

    aggregate: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for row, (image, source, candidate) in enumerate(zip(
        rows["image_ids"], rows["source_types"], rows["candidate_indices"],
    )):
        aggregate[(str(image), str(source), int(candidate))].append(row)
    aggregate_margin = np.asarray([
        float(np.median(margin[np.asarray(indices, dtype=np.int64)]))
        for indices in aggregate.values()
    ])
    return {
        "pair_count": int(margin.size), "query_count": int(np.unique(rows["image_ids"]).size),
        "pair_concordance": float(np.mean(margin > 0.0)),
        "margin_median": float(np.median(margin)),
        "analytic_pair_concordance": float(np.mean(analytic_margin > 0.0)),
        "aggregate_candidate_count": int(aggregate_margin.size),
        "aggregate_concordance": float(np.mean(aggregate_margin > 0.0)),
        "source": breakdown(margin, rows["source_types"]),
        "edge_family": breakdown(margin, rows["edge_families"], EDGE_FAMILIES),
        "edge_role": breakdown(margin, rows["edge_roles"], ("fit", "verify")),
        "trajectory": breakdown(margin, rows["trajectories"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--calibration_trajectories", nargs="+", default=["seq9"])
    parser.add_argument("--validation_trajectories", nargs="+", default=["seq10"])
    parser.add_argument("--excluded_trajectories", nargs="+", default=["seq11", "seq12", "seq14"])
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    model_path, summary_path = Path(args.output_model), Path(args.summary_json)
    if not args.force and (model_path.exists() or summary_path.exists()):
        raise FileExistsError("refusing to overwrite relation artifact")
    calibration, validation, excluded = (
        set(args.calibration_trajectories), set(args.validation_trajectories), set(args.excluded_trajectories),
    )
    if calibration & validation or (calibration | validation) & excluded:
        raise ValueError("relation trajectory partitions must be disjoint")
    arrays, sample_metadata = _load([Path(value) for value in args.samples])
    trajectory = arrays["trajectories"]
    masks = {
        "train": ~np.isin(trajectory, list(calibration | validation | excluded)),
        "calibration": np.isin(trajectory, list(calibration)),
        "validation": np.isin(trajectory, list(validation)),
    }
    partitions = {name: _pairs(arrays, mask) for name, mask in masks.items()}
    estimator = _fit(partitions["train"])
    scale, intercept = _calibrate(estimator, partitions["calibration"])
    metadata = {
        "artifact_type": "goal_maplet_mode_relation_likelihood_ratio_v1",
        "feature_names": list(FEATURE_NAMES), "edge_families": list(EDGE_FAMILIES),
        "pairing_contract": sample_metadata["pairing_contract"],
        "edge_contract": sample_metadata["edge_contract"],
        "option_contract": sample_metadata["option_contract"],
        "runtime_maximum_children": int(sample_metadata["runtime_maximum_children"]),
        "sample_sha256": [file_sha256(Path(value)) for value in args.samples],
        "training_trajectories": sorted(set(trajectory[masks["train"]].tolist())),
        "calibration_trajectories": list(args.calibration_trajectories),
        "validation_trajectories": list(args.validation_trajectories),
        "excluded_trajectories": list(args.excluded_trajectories),
        "uses_gt_at_runtime": False, "uses_absolute_pose_features": False,
        "uses_mapping_rgb": False, "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False, "stores_mapping_image_ids": False,
        "stored_downstream_embedding_count": 0, "uses_alike_descriptors": False,
        "uses_radio_intermediate": False, "uses_sfm_points": False,
        "uses_sfm_tracks": False, "uses_point_correspondences": False,
        **{key: sample_metadata[key] for key in (
            "physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256",
            "child_eligibility_sha256", "candidate_pool_sha256", "temperature", "maximum_modes",
        )},
    }
    artifact = ModeRelationLikelihoodRatioArtifact(estimator, scale, intercept, metadata)
    artifact.save(model_path)
    result = {
        "stage": "train_goal_maplet_mode_relation_likelihood_ratio_v1",
        "model": str(model_path), "calibration_scale": scale,
        "calibration_intercept": intercept, "metadata": metadata,
        "reports": {name: _report(estimator, scale, intercept, rows) for name, rows in partitions.items()},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
