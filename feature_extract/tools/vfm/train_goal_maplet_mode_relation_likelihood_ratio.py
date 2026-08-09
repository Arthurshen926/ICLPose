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
        if current.get("artifact_type") not in (
            "goal_maplet_mode_relation_samples_v1",
            "goal_maplet_mode_relation_samples_v2",
        ):
            raise ValueError("relation training requires mode-relation samples")
        if current.get("pairing_contract") not in (
            "same_image_same_query_edge_fixed_options_v1",
            "same_image_same_runtime_query_edge_adaptive_options_v2",
        ):
            raise ValueError("relation sample pairing contract differs")
        if current.get("edge_contract") not in (
            "query_only_fit_tree_disjoint_verify_v1",
            "query_only_complete_link_fit_tree_disjoint_verify_v2",
            "query_only_complete_link_cluster_collapse_fit_verify_v3",
        ):
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
                "endpoint_state_budget",
                "endpoint_hierarchy_calibration_sha256",
                "endpoint_hierarchy_fit_trajectories",
                "endpoint_hierarchy_validation_trajectories",
                "physical_instance_readout_sha256", "endpoint_state_policy",
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


def _concatenate_pairs(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Join paired supervision sets while preserving their common schema."""

    keys = set(parts[0])
    if any(set(part) != keys for part in parts[1:]):
        raise ValueError("relation direction pair schemas differ")
    return {
        key: np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
        for key in sorted(keys)
    }


def _raw(estimator, features: np.ndarray) -> np.ndarray:
    return np.asarray(estimator.decision_function(features), dtype=np.float64).reshape(-1)


def _fit_affine_calibration(
    positive_raw: np.ndarray,
    negative_raw: np.ndarray,
) -> tuple[float, float]:
    positive_raw = np.asarray(positive_raw, dtype=np.float64).reshape(-1)
    negative_raw = np.asarray(negative_raw, dtype=np.float64).reshape(-1)
    if positive_raw.size == 0 or negative_raw.size == 0:
        raise ValueError("affine relation calibration has an empty class")
    x = np.concatenate([positive_raw, negative_raw])[:, None]
    y = np.concatenate([np.ones(positive_raw.size), np.zeros(negative_raw.size)]).astype(np.int64)
    weight = np.where(y == 1, 0.5 / positive_raw.size, 0.5 / negative_raw.size)
    weight *= y.size / np.sum(weight)
    calibrator = LogisticRegression(C=1.0e6, max_iter=2000, random_state=214133)
    calibrator.fit(x, y, sample_weight=weight)
    return float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])


def _calibrate(
    estimator, rows: dict[str, np.ndarray],
) -> tuple[float, float, np.ndarray, np.ndarray]:
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
    scale, intercept = _fit_affine_calibration(
        _raw(estimator, positive), _raw(estimator, negative),
    )
    family_scale = np.full((len(EDGE_FAMILIES),), scale, dtype=np.float64)
    family_intercept = np.full((len(EDGE_FAMILIES),), intercept, dtype=np.float64)
    for family in range(len(EDGE_FAMILIES)):
        selected = rows["edge_families"] == family
        if int(np.sum(selected)) >= 8:
            family_scale[family], family_intercept[family] = _fit_affine_calibration(
                _raw(estimator, rows["positive_features"][selected]),
                _raw(estimator, rows["negative_features"][selected]),
            )
    return scale, intercept, family_scale, family_intercept


def _report(
    estimator,
    scale: float,
    intercept: float,
    family_scale: np.ndarray,
    family_intercept: np.ndarray,
    rows: dict[str, np.ndarray],
) -> dict:
    family = np.asarray(rows["edge_families"], dtype=np.int64)
    positive = family_scale[family] * _raw(estimator, rows["positive_features"]) + family_intercept[family]
    negative = family_scale[family] * _raw(estimator, rows["negative_features"]) + family_intercept[family]
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
    parser.add_argument(
        "--direction_samples", nargs="+", default=None,
        help=(
            "Optional prior-policy paired samples used only to estimate the "
            "proposal-invariant relation direction. Calibration and lineage "
            "always come from --samples."
        ),
    )
    parser.add_argument(
        "--direction_model", default=None,
        help=(
            "Optional previously cross-fitted relation artifact supplying only "
            "the proposal-invariant paired direction; it is recalibrated on "
            "the current endpoint policy."
        ),
    )
    parser.add_argument("--calibration_trajectories", nargs="+", default=["seq9"])
    parser.add_argument("--validation_trajectories", nargs="+", default=["seq10"])
    parser.add_argument("--excluded_trajectories", nargs="+", default=["seq11", "seq12", "seq14"])
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.direction_samples and args.direction_model:
        raise ValueError("choose either direction samples or a frozen direction model")
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
    direction_parts = [partitions["train"]]
    direction_metadata = []
    direction_sha256 = []
    direction_trajectories = set(trajectory[masks["train"]].tolist())
    if args.direction_samples:
        prior_arrays, prior_metadata = _load([Path(value) for value in args.direction_samples])
        for key in (
            "physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256",
            "child_eligibility_sha256",
        ):
            if prior_metadata.get(key) != sample_metadata.get(key):
                raise ValueError(f"relation direction sample lineage differs: {key}")
        prior_mask = ~np.isin(
            prior_arrays["trajectories"], list(calibration | validation | excluded),
        )
        direction_parts.append(_pairs(prior_arrays, prior_mask))
        direction_metadata.append({
            "option_contract": prior_metadata.get("option_contract"),
            "endpoint_state_policy": prior_metadata.get("endpoint_state_policy", "mass_adaptive_g16"),
        })
        direction_sha256 = [file_sha256(Path(value)) for value in args.direction_samples]
        direction_trajectories.update(prior_arrays["trajectories"][prior_mask].tolist())
    direction_train = _concatenate_pairs(*direction_parts)
    frozen_direction_sha256 = None
    if args.direction_model:
        frozen_direction = ModeRelationLikelihoodRatioArtifact.load(Path(args.direction_model))
        for key in (
            "physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256",
            "child_eligibility_sha256",
        ):
            if frozen_direction.metadata.get(key) != sample_metadata.get(key):
                raise ValueError(f"frozen relation direction lineage differs: {key}")
        estimator = frozen_direction.pair_estimator
        frozen_direction_sha256 = file_sha256(Path(args.direction_model))
    else:
        estimator = _fit(direction_train)
    scale, intercept, family_scale, family_intercept = _calibrate(
        estimator, partitions["calibration"],
    )
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
        "relation_direction": "shared_paired_direction",
        "relation_direction_sampling": (
            "frozen_prior_policy_paired_direction_current_policy_recalibration"
            if args.direction_model else
            "current_plus_prior_policy_paired_physical_endpoints"
            if args.direction_samples else "current_policy_paired_physical_endpoints"
        ),
        "frozen_direction_model_sha256": frozen_direction_sha256,
        "direction_sample_sha256": direction_sha256,
        "direction_sample_contracts": direction_metadata,
        "direction_training_trajectories": sorted(direction_trajectories),
        "relation_calibration": "edge_family_specific_affine",
        **{key: sample_metadata[key] for key in (
            "physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256",
            "child_eligibility_sha256", "candidate_pool_sha256", "temperature", "maximum_modes",
        )},
    }
    if "endpoint_state_budget" in sample_metadata:
        metadata["endpoint_state_budget"] = int(sample_metadata["endpoint_state_budget"])
    if sample_metadata.get("endpoint_hierarchy_calibration_sha256") is not None:
        metadata["endpoint_hierarchy_calibration_sha256"] = str(
            sample_metadata["endpoint_hierarchy_calibration_sha256"]
        )
        metadata["endpoint_hierarchy_fit_trajectories"] = list(
            sample_metadata.get("endpoint_hierarchy_fit_trajectories", ())
        )
        metadata["endpoint_hierarchy_validation_trajectories"] = list(
            sample_metadata.get("endpoint_hierarchy_validation_trajectories", ())
        )
    if sample_metadata.get("physical_instance_readout_sha256") is not None:
        metadata["physical_instance_readout_sha256"] = str(
            sample_metadata["physical_instance_readout_sha256"]
        )
    metadata["endpoint_state_policy"] = str(
        sample_metadata.get("endpoint_state_policy", "mass_adaptive_g16")
    )
    artifact = ModeRelationLikelihoodRatioArtifact(
        estimator, scale, intercept, metadata, family_scale, family_intercept,
    )
    artifact.save(model_path)
    result = {
        "stage": "train_goal_maplet_mode_relation_likelihood_ratio_v1",
        "model": str(model_path), "calibration_scale": scale,
        "calibration_intercept": intercept, "metadata": metadata,
        "family_calibration_scale": family_scale.tolist(),
        "family_calibration_intercept": family_intercept.tolist(),
        "reports": {
            name: _report(
                estimator, scale, intercept, family_scale, family_intercept, rows,
            ) for name, rows in {**partitions, "direction_train": direction_train}.items()
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
