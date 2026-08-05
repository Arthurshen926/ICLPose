"""Train query-set pairwise Goal-Maplet configuration rankers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from feature_extract.vfm.localization_goal_maplet.configuration_ranker import (
    BASE_CONFIGURATION_FEATURE_NAMES,
    BASE_FEATURE_NAMES,
    EXACT_CONFIGURATION_FEATURE_NAMES,
    EXACT_FEATURE_NAMES,
    ConfigurationPairwiseRankerArtifact,
    candidate_runtime_features,
    source_pool_sha256,
)


MODE = "actual_parent_actual_child"


def _dataset(
    payload: dict, validation: set[str], excluded: set[str],
    include_exact: bool, include_configuration: bool,
):
    partitions = {"train": [], "validation": []}
    for row in payload["rows"]:
        trajectory = str(row["image_id"]).split("/", 1)[0]
        if trajectory in excluded:
            continue
        feature = candidate_runtime_features(
            row, mode_name=MODE, include_exact=include_exact,
            include_configuration=include_configuration,
        )
        details = row["mode_details"][MODE]
        translation = np.asarray([item["translation_m"] for item in details], dtype=np.float64)
        rotation = np.asarray([item["rotation_deg"] for item in details], dtype=np.float64)
        if include_exact:
            keep = np.asarray(row["ranking_diagnostics"][MODE]["cascade_exact_evaluated"], dtype=bool)
            feature, translation, rotation = feature[keep], translation[keep], rotation[keep]
        utility = translation / 0.5 + rotation / 5.0
        partitions["validation" if trajectory in validation else "train"].append({
            "image_id": str(row["image_id"]), "features": feature,
            "translation": translation, "rotation": rotation, "utility": utility,
        })
    return partitions


def _pairs(rows: list[dict], *, maximum_pairs_per_query: int = 384):
    x, y, weight = [], [], []
    rng = np.random.default_rng(194917)
    for row in rows:
        left, right = np.triu_indices(row["utility"].size, 1)
        gap = np.abs(row["utility"][left] - row["utility"][right])
        keep = gap >= 0.05
        left, right, gap = left[keep], right[keep], gap[keep]
        if left.size > int(maximum_pairs_per_query):
            selected = rng.choice(left.size, size=int(maximum_pairs_per_query), replace=False)
            left, right, gap = left[selected], right[selected], gap[selected]
        if left.size == 0:
            continue
        difference = row["features"][left] - row["features"][right]
        preference = row["utility"][left] < row["utility"][right]
        # Equal query mass; large utility gaps matter more but cannot dominate.
        current_weight = np.clip(gap, 0.1, 4.0) / max(left.size, 1)
        x.extend((difference, -difference))
        y.extend((preference, ~preference))
        weight.extend((current_weight, current_weight))
    return (
        np.concatenate(x).astype(np.float32),
        np.concatenate(y).astype(np.int64),
        np.concatenate(weight).astype(np.float64),
    )


def _evaluate(rows: list[dict], artifact: ConfigurationPairwiseRankerArtifact) -> dict:
    translation, rotation, oracle_translation = [], [], []
    utility_regret, translation_regret, promotion, success = [], [], [], []
    for row in rows:
        score = artifact.score_candidates(row["features"])
        selected = int(np.argmax(score))
        oracle = int(np.argmin(row["utility"]))
        translation.append(float(row["translation"][selected]))
        rotation.append(float(row["rotation"][selected]))
        oracle_translation.append(float(row["translation"][oracle]))
        utility_regret.append(float(row["utility"][selected] - row["utility"][oracle]))
        translation_regret.append(float(row["translation"][selected] - row["translation"][oracle]))
        promotion.append(bool(selected == oracle))
        success.append(bool(row["translation"][selected] <= 0.5 and row["rotation"][selected] <= 5.0))
    t, r = np.asarray(translation), np.asarray(rotation)
    tr, ur = np.asarray(translation_regret), np.asarray(utility_regret)
    return {
        "query_count": len(rows),
        "top1_translation_median_m": float(np.median(t)),
        "top1_translation_p90_m": float(np.percentile(t, 90.0)),
        "top1_rotation_median_deg": float(np.median(r)),
        "top1_rotation_p90_deg": float(np.percentile(r, 90.0)),
        "top1_0p5m_5deg_fraction": float(np.mean(success)),
        "catastrophic_2m_or_10deg_fraction": float(np.mean((t > 2.0) | (r > 10.0))),
        "oracle_translation_median_m": float(np.median(oracle_translation)),
        "median_selection_regret_m": float(np.median(tr)),
        "p90_selection_regret_m": float(np.percentile(tr, 90.0)),
        "median_normalized_utility_regret": float(np.median(ur)),
        "p90_normalized_utility_regret": float(np.percentile(ur, 90.0)),
        "oracle_exact_promotion_fraction": float(np.mean(promotion)),
    }


def _baseline(rows: list[dict], feature_column: int) -> dict:
    class Baseline:
        metadata = {"feature_names": list(BASE_FEATURE_NAMES)}
        @staticmethod
        def score_candidates(features):
            return np.asarray(features[:, feature_column], dtype=np.float64)
    return _evaluate(rows, Baseline())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--validation_trajectories", nargs="+", default=["seq9", "seq10", "seq12", "seq14"])
    parser.add_argument("--excluded_trajectories", nargs="+", default=["seq11"])
    parser.add_argument(
        "--feature_set", choices=("base", "exact", "configuration", "configuration_exact"),
        default="base",
    )
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_model), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite pairwise configuration ranker")
    payload = json.loads(Path(args.candidate_pool).read_text())
    include_exact = args.feature_set in ("exact", "configuration_exact")
    include_configuration = args.feature_set in ("configuration", "configuration_exact")
    if include_configuration:
        evidence_contract = dict(payload.get("configuration_evidence_contract", {}))
        if not (
            bool(evidence_contract.get("outer_cross_fit"))
            or bool(evidence_contract.get("factor_training_pool_disjoint"))
        ):
            raise ValueError(
                "configuration-factor ranker training requires outer-cross-fit "
                "or a factor-training-disjoint candidate pool"
            )
    partitions = _dataset(
        payload, set(args.validation_trajectories), set(args.excluded_trajectories),
        include_exact, include_configuration,
    )
    x, y, weight = _pairs(partitions["train"])
    estimators = {
        "pairwise_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=194917)
        ),
        "pairwise_hist_gbdt": HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=200, max_leaf_nodes=15, max_depth=3,
            min_samples_leaf=32, l2_regularization=3.0, random_state=194917,
        ),
    }
    reports, fitted = {}, {}
    names = list(
        EXACT_CONFIGURATION_FEATURE_NAMES if include_exact and include_configuration
        else EXACT_FEATURE_NAMES if include_exact
        else BASE_CONFIGURATION_FEATURE_NAMES if include_configuration
        else BASE_FEATURE_NAMES
    )
    for name, estimator in estimators.items():
        fit_args = {"logisticregression__sample_weight": weight} if hasattr(estimator, "steps") else {"sample_weight": weight}
        with threadpool_limits(limits=4):
            fitted[name] = estimator.fit(x, y, **fit_args)
        artifact = ConfigurationPairwiseRankerArtifact(fitted[name], {
            "artifact_type": "goal_maplet_configuration_pairwise_ranker_v2", "feature_names": names,
        })
        reports[name] = {split: _evaluate(rows, artifact) for split, rows in partitions.items()}
    selected_name = min(reports, key=lambda name: (
        reports[name]["validation"]["median_selection_regret_m"],
        reports[name]["validation"]["p90_selection_regret_m"],
        reports[name]["validation"]["top1_translation_p90_m"],
    ))
    metadata = {
        "artifact_type": "goal_maplet_configuration_pairwise_ranker_v2",
        "feature_names": names, "selected_estimator": selected_name,
        "candidate_pool_sha256": source_pool_sha256(Path(args.candidate_pool)),
        "physical_map_sha256": payload.get("physical_map_sha256"),
        "canonical_field_sha256": payload.get("canonical_field_sha256"),
        "typed_graph_sha256": payload.get("typed_graph_sha256"),
        "field_feature_contract_sha256": payload.get("field_feature_contract_sha256"),
        "validation_trajectories": list(args.validation_trajectories),
        "excluded_trajectories": list(args.excluded_trajectories),
        "target": "within_query_pairwise_continuous_pose_utility",
        "utility": "translation_m/0.5+rotation_deg/5",
        "feature_set": str(args.feature_set), "uses_gt_at_runtime": False,
        "configuration_evidence_training_contract": (
            payload.get("configuration_evidence_contract") if include_configuration else None
        ),
        "requires_child_local_factor_calibrator": bool(include_configuration),
        "uses_absolute_pose_features": False, "uses_mapping_rgb": False,
        "stores_mapping_rgb": False, "stores_mapping_image_paths": False,
        "stores_mapping_image_ids": False, "stored_downstream_embedding_count": 0,
        "uses_alike_descriptors": False, "uses_radio_intermediate": False,
        "uses_sfm_points": False, "uses_sfm_tracks": False, "uses_point_correspondences": False,
    }
    ConfigurationPairwiseRankerArtifact(fitted[selected_name], metadata).save(output)
    result = {
        "stage": "train_goal_maplet_configuration_pairwise_ranker_v2",
        "selected_estimator": selected_name, "model": str(output), "metadata": metadata,
        "pair_count": int(x.shape[0]), "query_count": {key: len(value) for key, value in partitions.items()},
        "reports": reports,
        "baselines": {
            "proposal": {split: _baseline(rows, 0) for split, rows in partitions.items()},
            "cheap_identity": {split: _baseline(rows, 5) for split, rows in partitions.items()},
        },
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
