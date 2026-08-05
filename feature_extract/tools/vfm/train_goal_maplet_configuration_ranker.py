"""Train lightweight query-grouped Goal-Maplet configuration rankers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_extract.vfm.localization_goal_maplet.configuration_ranker import (
    BASE_FEATURE_NAMES,
    EXACT_FEATURE_NAMES,
    ConfigurationRankerArtifact,
    candidate_runtime_features,
    source_pool_sha256,
)


MODE = "actual_parent_actual_child"


def _dataset(payload: dict, validation: set[str], excluded: set[str], include_exact: bool):
    partitions = {"train": [], "validation": []}
    for row in payload["rows"]:
        trajectory = str(row["image_id"]).split("/", 1)[0]
        if trajectory in excluded:
            continue
        split = "validation" if trajectory in validation else "train"
        features = candidate_runtime_features(row, mode_name=MODE, include_exact=include_exact)
        details = row["mode_details"][MODE]
        translation = np.asarray([item["translation_m"] for item in details], dtype=np.float64)
        rotation = np.asarray([item["rotation_deg"] for item in details], dtype=np.float64)
        if include_exact:
            keep = np.asarray(
                row["ranking_diagnostics"][MODE]["cascade_exact_evaluated"], dtype=bool
            )
            features, translation, rotation = features[keep], translation[keep], rotation[keep]
        target = ((translation <= 0.5) & (rotation <= 5.0)).astype(np.int64)
        partitions[split].append({
            "image_id": str(row["image_id"]),
            "trajectory": trajectory,
            "features": features,
            "target": target,
            "translation": translation,
            "rotation": rotation,
        })
    return partitions


def _fit_rows(rows, estimator):
    x = np.concatenate([row["features"] for row in rows], axis=0)
    y = np.concatenate([row["target"] for row in rows], axis=0)
    weights = np.concatenate([
        np.full(row["target"].shape, 1.0 / max(row["target"].size, 1), dtype=np.float64)
        for row in rows
    ])
    positive = max(float(np.sum(weights[y == 1])), 1e-8)
    negative = max(float(np.sum(weights[y == 0])), 1e-8)
    weights[y == 1] *= 0.5 / positive
    weights[y == 0] *= 0.5 / negative
    estimator.fit(x, y, **({"sample_weight": weights} if not hasattr(estimator, "steps") else {"logisticregression__sample_weight": weights}))
    return estimator


def _evaluate(rows, estimator):
    selected_t, selected_r, oracle_t, oracle_r, success, promotion = [], [], [], [], [], []
    probabilities = []
    targets = []
    for row in rows:
        probability = np.asarray(estimator.predict_proba(row["features"])[:, 1])
        selected = int(np.argmax(probability))
        utility = row["translation"] / 0.5 + row["rotation"] / 5.0
        oracle = int(np.argmin(utility))
        selected_t.append(float(row["translation"][selected]))
        selected_r.append(float(row["rotation"][selected]))
        oracle_t.append(float(row["translation"][oracle]))
        oracle_r.append(float(row["rotation"][oracle]))
        success.append(bool(row["target"][selected]))
        promotion.append(bool(selected == oracle))
        probabilities.extend(probability.tolist())
        targets.extend(row["target"].tolist())
    t = np.asarray(selected_t, dtype=np.float64)
    ot = np.asarray(oracle_t, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (p >= lower) & (p < lower + 0.1 if lower < 0.9 else p <= 1.0)
        if np.any(mask):
            calibration_error += float(np.mean(mask)) * abs(float(np.mean(p[mask]) - np.mean(y[mask])))
    return {
        "query_count": len(rows),
        "top1_translation_median_m": float(np.median(t)),
        "top1_translation_p90_m": float(np.percentile(t, 90.0)),
        "top1_rotation_median_deg": float(np.median(selected_r)),
        "top1_rotation_p90_deg": float(np.percentile(selected_r, 90.0)),
        "top1_0p5m_5deg_fraction": float(np.mean(success)),
        "oracle_translation_median_m": float(np.median(ot)),
        "oracle_translation_p90_m": float(np.percentile(ot, 90.0)),
        "median_selection_regret_m": float(np.median(t - ot)),
        "oracle_exact_promotion_fraction": float(np.mean(promotion)),
        "catastrophic_2m_or_10deg_fraction": float(np.mean((t > 2.0) | (np.asarray(selected_r) > 10.0))),
        "candidate_brier": float(np.mean(np.square(p - y))),
        "candidate_ece_10bin": float(calibration_error),
    }


def _evaluate_fixed_ranking(rows, policy: str):
    class FixedEstimator:
        @staticmethod
        def predict_proba(features):
            column = 5 if policy == "cheap_identity" else 0
            score = np.asarray(features[:, column], dtype=np.float64)
            score = score - np.max(score)
            probability = np.exp(np.clip(score, -40.0, 0.0))
            probability /= max(float(np.sum(probability)), 1e-12)
            return np.stack([1.0 - probability, probability], axis=1)

    return _evaluate(rows, FixedEstimator())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--validation_trajectories", nargs="+", default=["seq9", "seq10", "seq12", "seq14"])
    parser.add_argument("--excluded_trajectories", nargs="+", default=["seq11"])
    parser.add_argument("--feature_set", choices=("base", "exact"), default="base")
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_model), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite configuration ranker")
    payload = json.loads(Path(args.candidate_pool).read_text())
    validation = set(args.validation_trajectories)
    excluded = set(args.excluded_trajectories)
    if validation & excluded:
        raise ValueError("validation and excluded trajectories must be disjoint")
    include_exact = args.feature_set == "exact"
    partitions = _dataset(payload, validation, excluded, include_exact)
    if not partitions["train"] or not partitions["validation"]:
        raise ValueError("configuration ranker requires non-empty trajectory-disjoint splits")
    estimators = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, class_weight=None, random_state=194917),
        ),
        "hist_gbdt": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            max_depth=3,
            min_samples_leaf=16,
            l2_regularization=1.0,
            random_state=194917,
        ),
    }
    reports = {}
    fitted = {}
    for name, estimator in estimators.items():
        fitted[name] = _fit_rows(partitions["train"], estimator)
        reports[name] = {
            "train": _evaluate(partitions["train"], fitted[name]),
            "validation": _evaluate(partitions["validation"], fitted[name]),
        }
    baselines = {
        name: {
            split: _evaluate_fixed_ranking(partitions[split], name)
            for split in ("train", "validation")
        }
        for name in ("proposal", "cheap_identity")
    }
    selected_name = min(
        reports,
        key=lambda name: (
            reports[name]["validation"]["top1_translation_median_m"],
            reports[name]["validation"]["top1_translation_p90_m"],
        ),
    )
    metadata = {
        "artifact_type": "goal_maplet_configuration_ranker_v1",
        "feature_names": list(EXACT_FEATURE_NAMES if include_exact else BASE_FEATURE_NAMES),
        "selected_estimator": selected_name,
        "candidate_pool_sha256": source_pool_sha256(Path(args.candidate_pool)),
        "physical_map_sha256": payload.get("physical_map_sha256"),
        "canonical_field_sha256": payload.get("canonical_field_sha256"),
        "typed_graph_sha256": payload.get("typed_graph_sha256"),
        "field_feature_contract_sha256": payload.get("field_feature_contract_sha256"),
        "validation_trajectories": list(args.validation_trajectories),
        "excluded_trajectories": list(args.excluded_trajectories),
        "target": "candidate_within_0.5m_5deg",
        "feature_set": str(args.feature_set),
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
    }
    artifact = ConfigurationRankerArtifact(fitted[selected_name], metadata)
    artifact.save(output)
    result = {
        "stage": "train_goal_maplet_configuration_ranker",
        "selected_estimator": selected_name,
        "candidate_pool": str(args.candidate_pool),
        "model": str(output),
        "metadata": metadata,
        "reports": reports,
        "baselines": baselines,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
