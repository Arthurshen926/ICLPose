"""Train an additive pose factor from same-query, same-group contrasts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from feature_extract.vfm.localization_goal_maplet.child_local_factor import FEATURE_NAMES, NULL_TYPES
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pose_likelihood_ratio import PoseLikelihoodRatioArtifact


NEGATIVE_SOURCES = (
    "frozen_graph_top1",
    "top32_near_miss_pose",
    "top32_low_rotation_phase_pose",
    "retrieved_hard_wrong_child",
)


def _load(paths: list[Path]) -> tuple[dict[str, np.ndarray], dict]:
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata = None
    required = (
        "features", "targets", "image_ids", "trajectories", "source_types",
        "group_rows", "child_rows", "truth_child_rows", "candidate_indices",
        "truth_child_runtime_ranks",
        "pose_translation_m", "pose_rotation_deg",
        "oracle_child_center_offset_px",
    )
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            current = json.loads(str(np.asarray(data["metadata_json"]).item()))
            for key in required:
                parts[key].append(np.asarray(data[key]))
        if current.get("artifact_type") != "goal_maplet_child_local_factor_samples_v3":
            raise ValueError("pose-likelihood training requires paired factor samples v3")
        if current.get("pairing_contract") != "same_image_same_query_group_exact_v1":
            raise ValueError("factor sample pairing contract differs")
        if current.get("feature_input_contract") != "deployment_query_group_center_v1":
            raise ValueError("factor sample deployment-replay contract differs")
        if tuple(current.get("null_types", ())) != NULL_TYPES:
            raise ValueError("factor null contract differs")
        if metadata is None:
            metadata = current
        else:
            for key in (
                "physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256",
                "child_eligibility_sha256", "candidate_pool_sha256", "temperature", "maximum_modes",
                "pairing_contract",
                "feature_input_contract",
                "runtime_maximum_children",
            ):
                if current.get(key) != metadata.get(key):
                    raise ValueError(f"factor sample shards differ: {key}")
    arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    count = arrays["features"].shape[0]
    if arrays["features"].shape != (count, len(FEATURE_NAMES)):
        raise ValueError("factor sample feature contract differs")
    if any(np.asarray(value).shape[0] != count for key, value in arrays.items() if key != "features"):
        raise ValueError("factor sample arrays differ")
    for key in ("image_ids", "trajectories", "source_types"):
        arrays[key] = arrays[key].astype(str)
    return arrays, metadata


def _pairs(arrays: dict[str, np.ndarray], trajectory_mask: np.ndarray) -> dict[str, np.ndarray]:
    feature = np.asarray(arrays["features"], dtype=np.float32)
    target = np.asarray(arrays["targets"], dtype=np.int64)
    image = arrays["image_ids"]
    group = np.asarray(arrays["group_rows"], dtype=np.int64)
    source = arrays["source_types"]
    positive_by_key: dict[tuple[str, int], int] = {}
    positive_mask = trajectory_mask & (source == "exact_gt_pose") & (target == NULL_TYPES.index("valid"))
    for row in np.flatnonzero(positive_mask).tolist():
        key = (str(image[row]), int(group[row]))
        if key in positive_by_key:
            raise ValueError(f"duplicate exact positive factor: {key}")
        positive_by_key[key] = row
    positive, negative = [], []
    for row in np.flatnonzero(trajectory_mask & np.isin(source, NEGATIVE_SOURCES)).tolist():
        # A correct graph Top-1 is not a negative.  Missing field is a map
        # availability event and cannot teach pose incompatibility.
        if target[row] in (NULL_TYPES.index("valid"), NULL_TYPES.index("field_missing")):
            continue
        key = (str(image[row]), int(group[row]))
        if key in positive_by_key:
            positive.append(positive_by_key[key])
            negative.append(row)
    positive = np.asarray(positive, dtype=np.int64)
    negative = np.asarray(negative, dtype=np.int64)
    if positive.size == 0:
        raise ValueError("no exact within-group pose-likelihood pairs")
    return {
        "positive_features": feature[positive],
        "negative_features": feature[negative],
        "image_ids": image[negative],
        "trajectories": arrays["trajectories"][negative],
        "source_types": source[negative],
        "group_rows": group[negative],
        "candidate_indices": np.asarray(arrays["candidate_indices"], dtype=np.int64)[negative],
        "translation_m": np.asarray(arrays["pose_translation_m"], dtype=np.float64)[negative],
        "rotation_deg": np.asarray(arrays["pose_rotation_deg"], dtype=np.float64)[negative],
    }


def _balanced_weight(image: np.ndarray, source: np.ndarray) -> np.ndarray:
    _, inverse, count = np.unique(image, return_inverse=True, return_counts=True)
    weight = 1.0 / count[inverse]
    for value in np.unique(source):
        mask = source == value
        weight[mask] /= max(float(np.sum(weight[mask])), 1e-12)
    return weight * image.size / max(float(np.sum(weight)), 1e-12)


def _pair_fit(rows: dict[str, np.ndarray]):
    difference = rows["positive_features"] - rows["negative_features"]
    x = np.concatenate([difference, -difference], axis=0)
    y = np.concatenate([
        np.ones(difference.shape[0], dtype=np.int64),
        np.zeros(difference.shape[0], dtype=np.int64),
    ])
    weight = _balanced_weight(rows["image_ids"], rows["source_types"])
    weight = np.concatenate([weight, weight])
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.125, fit_intercept=False, max_iter=4000, random_state=194917),
    )
    with threadpool_limits(limits=8):
        estimator.fit(x, y, logisticregression__sample_weight=weight)
    return estimator


def _raw_score(estimator, feature: np.ndarray) -> np.ndarray:
    return np.asarray(estimator.decision_function(feature), dtype=np.float64).reshape(-1)


def _calibrate(estimator, rows: dict[str, np.ndarray]) -> tuple[float, float]:
    # Each positive group appears once; negatives retain their structured
    # families.  This is an independent-trajectory absolute density-ratio
    # calibration, not a refit of the paired ranking direction.
    keys = np.asarray([
        f"{image}\0{group}" for image, group in zip(rows["image_ids"], rows["group_rows"])
    ])
    _, first = np.unique(keys, return_index=True)
    positive = rows["positive_features"][np.sort(first)]
    negative = rows["negative_features"]
    x = np.concatenate([_raw_score(estimator, positive), _raw_score(estimator, negative)])[:, None]
    y = np.concatenate([
        np.ones(positive.shape[0], dtype=np.int64),
        np.zeros(negative.shape[0], dtype=np.int64),
    ])
    # Equal positive/negative prior makes zero the explicit neutral evidence
    # point even when every positive has several structured negatives.
    weight = np.where(y == 1, 0.5 / max(positive.shape[0], 1), 0.5 / max(negative.shape[0], 1))
    weight *= y.size / np.sum(weight)
    calibrator = LogisticRegression(C=1.0e6, max_iter=2000, random_state=194917)
    calibrator.fit(x, y, sample_weight=weight)
    return float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])


def _score(estimator, scale: float, intercept: float, feature: np.ndarray) -> np.ndarray:
    return scale * _raw_score(estimator, feature) + intercept


def _report(estimator, scale: float, intercept: float, rows: dict[str, np.ndarray]) -> dict:
    positive = _score(estimator, scale, intercept, rows["positive_features"])
    negative = _score(estimator, scale, intercept, rows["negative_features"])
    margin = positive - negative
    probability = np.exp(-np.logaddexp(0.0, -margin))
    pair_weight = _balanced_weight(rows["image_ids"], rows["source_types"])
    pair_nll = -np.log(np.maximum(probability, 1e-12))
    source_report = {}
    for value in sorted(set(rows["source_types"].tolist())):
        mask = rows["source_types"] == value
        source_report[value] = {
            "pair_count": int(np.sum(mask)),
            "concordance": float(np.mean(margin[mask] > 0.0)),
            "margin_median": float(np.median(margin[mask])),
            "pair_nll": float(np.mean(pair_nll[mask])),
        }
    trajectory_report = {}
    for value in sorted(set(rows["trajectories"].tolist())):
        mask = rows["trajectories"] == value
        trajectory_report[value] = {
            "pair_count": int(np.sum(mask)),
            "concordance": float(np.mean(margin[mask] > 0.0)),
            "margin_median": float(np.median(margin[mask])),
        }
    # Add scores over all paired groups for each concrete candidate family.
    aggregate = []
    aggregate_sources = defaultdict(list)
    key_rows: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for row, (image, source, candidate) in enumerate(zip(
        rows["image_ids"], rows["source_types"], rows["candidate_indices"]
    )):
        key_rows[(str(image), str(source), int(candidate))].append(row)
    for (_, source, _), indices in key_rows.items():
        selected = np.asarray(indices, dtype=np.int64)
        value = float(np.mean(positive[selected]) - np.mean(negative[selected]))
        aggregate.append(value)
        aggregate_sources[source].append(value)
    point_score = np.concatenate([positive, negative])
    point_target = np.concatenate([
        np.ones(positive.size, dtype=np.int64), np.zeros(negative.size, dtype=np.int64)
    ])
    point_probability = np.exp(-np.logaddexp(0.0, -point_score))
    return {
        "pair_count": int(margin.size),
        "query_count": int(np.unique(rows["image_ids"]).size),
        "pair_concordance": float(np.mean(margin > 0.0)),
        "pair_nll": float(np.sum(pair_weight * pair_nll) / np.sum(pair_weight)),
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "aggregate_candidate_count": int(len(aggregate)),
        "aggregate_concordance": float(np.mean(np.asarray(aggregate) > 0.0)),
        "aggregate_by_source": {
            key: {
                "count": len(value),
                "concordance": float(np.mean(np.asarray(value) > 0.0)),
                "margin_median": float(np.median(value)),
            } for key, value in sorted(aggregate_sources.items())
        },
        "point_nll": float(np.mean(-np.log(np.maximum(np.where(point_target, point_probability, 1.0 - point_probability), 1e-12)))),
        "point_auprc": float(average_precision_score(point_target, point_probability)),
        "source": source_report,
        "trajectory": trajectory_report,
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
        raise FileExistsError("refusing to overwrite pose-likelihood artifact")
    calibration = set(args.calibration_trajectories)
    validation = set(args.validation_trajectories)
    excluded = set(args.excluded_trajectories)
    if calibration & validation or (calibration | validation) & excluded:
        raise ValueError("pose-likelihood trajectory partitions must be disjoint")
    arrays, sample_metadata = _load([Path(value) for value in args.samples])
    trajectory = arrays["trajectories"]
    masks = {
        "train": ~np.isin(trajectory, list(calibration | validation | excluded)),
        "calibration": np.isin(trajectory, list(calibration)),
        "validation": np.isin(trajectory, list(validation)),
    }
    partitions = {name: _pairs(arrays, mask) for name, mask in masks.items()}
    estimator = _pair_fit(partitions["train"])
    scale, intercept = _calibrate(estimator, partitions["calibration"])
    metadata = {
        "artifact_type": "goal_maplet_pose_likelihood_ratio_v1",
        "feature_names": list(FEATURE_NAMES),
        "pairing_contract": sample_metadata["pairing_contract"],
        "feature_input_contract": sample_metadata["feature_input_contract"],
        "negative_sources": list(NEGATIVE_SOURCES),
        "runtime_maximum_children": int(sample_metadata["runtime_maximum_children"]),
        "sample_sha256": [file_sha256(Path(value)) for value in args.samples],
        "training_trajectories": sorted(set(trajectory[masks["train"]].tolist())),
        "calibration_trajectories": list(args.calibration_trajectories),
        "validation_trajectories": list(args.validation_trajectories),
        "excluded_trajectories": list(args.excluded_trajectories),
        "uses_gt_at_runtime": False,
        "uses_absolute_pose_features": False,
        "uses_mapping_rgb": False,
        "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False,
        "stores_mapping_image_ids": False,
        "stored_downstream_embedding_count": 0,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_point_correspondences": False,
        **{key: sample_metadata[key] for key in (
            "physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256",
            "child_eligibility_sha256", "candidate_pool_sha256", "temperature", "maximum_modes",
        )},
    }
    artifact = PoseLikelihoodRatioArtifact(estimator, scale, intercept, metadata)
    artifact.save(model_path)
    result = {
        "stage": "train_goal_maplet_pose_likelihood_ratio_v1",
        "model": str(model_path),
        "calibration_scale": scale,
        "calibration_intercept": intercept,
        "metadata": metadata,
        "reports": {
            name: _report(estimator, scale, intercept, rows)
            for name, rows in partitions.items()
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
