"""Train basin/safety heads as a bounded residual on a predefined pose score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_extract.vfm.localization_goal_maplet.latent_selector import (
    FEATURE_NAMES,
    MODE,
    LatentSafetySelectorArtifact,
    latent_selector_features,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _rows(payload: dict, trajectories: set[str]) -> list[dict]:
    output = []
    for row in payload["rows"]:
        trajectory = str(row["image_id"]).split("/", 1)[0]
        if trajectory not in trajectories:
            continue
        details = row["mode_details"][MODE]
        evidence = row["ranking_diagnostics"][MODE]["configuration_evidence_v3"]
        output.append({
            "image_id": row["image_id"],
            "features": latent_selector_features(row),
            "base": np.asarray(evidence[payload["selector_base_feature"]], dtype=np.float64),
            "translation": np.asarray([item["translation_m"] for item in details], dtype=np.float64),
            "rotation": np.asarray([item["rotation_deg"] for item in details], dtype=np.float64),
        })
    return output


def _fit_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature, basin, catastrophic, weight = [], [], [], []
    for row in rows:
        count = row["features"].shape[0]
        feature.append(row["features"])
        basin.append((row["translation"] <= 0.5) & (row["rotation"] <= 5.0))
        catastrophic.append((row["translation"] > 2.0) | (row["rotation"] > 10.0))
        weight.append(np.full((count,), 1.0 / max(count, 1), dtype=np.float64))
    return (
        np.concatenate(feature).astype(np.float32),
        np.concatenate(basin).astype(np.int64),
        np.concatenate(catastrophic).astype(np.int64),
        np.concatenate(weight).astype(np.float64),
    )


def _metrics(rows: list[dict], artifact: LatentSafetySelectorArtifact, policy: str) -> dict:
    translation, rotation, oracle_translation = [], [], []
    for row in rows:
        if policy == "base":
            score = row["base"]
        else:
            score, _, _ = artifact.score_candidates(
                row["features"], row["base"], safety_veto=policy == "safety_residual",
            )
        selected = int(np.argmax(score))
        utility = row["translation"] / 0.5 + row["rotation"] / 5.0
        oracle = int(np.argmin(utility))
        translation.append(float(row["translation"][selected]))
        rotation.append(float(row["rotation"][selected]))
        oracle_translation.append(float(row["translation"][oracle]))
    t, r, o = np.asarray(translation), np.asarray(rotation), np.asarray(oracle_translation)
    return {
        "query_count": len(rows),
        "translation_median_m": float(np.median(t)),
        "translation_p90_m": float(np.percentile(t, 90.0)),
        "rotation_median_deg": float(np.median(r)),
        "rotation_p90_deg": float(np.percentile(r, 90.0)),
        "top1_0p5m_5deg_fraction": float(np.mean((t <= 0.5) & (r <= 5.0))),
        "catastrophic_2m_or_10deg_fraction": float(np.mean((t > 2.0) | (r > 10.0))),
        "median_selection_regret_m": float(np.median(t - o)),
        "p90_selection_regret_m": float(np.percentile(t - o, 90.0)),
    }


def _head_report(rows: list[dict], artifact: LatentSafetySelectorArtifact) -> dict:
    feature, basin, catastrophic, weight = _fit_arrays(rows)
    basin_probability, catastrophic_probability = artifact.predict_heads(feature)
    return {
        "basin_auprc": float(average_precision_score(basin, basin_probability, sample_weight=weight)),
        "catastrophic_auprc": float(average_precision_score(
            catastrophic, catastrophic_probability, sample_weight=weight,
        )),
        "basin_prevalence": float(np.average(basin, weights=weight)),
        "catastrophic_prevalence": float(np.average(catastrophic, weights=weight)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument(
        "--base_feature", default="configuration_fixed_valid_factor_mass_mean",
    )
    parser.add_argument(
        "--training_trajectories", nargs="+", default=["seq1", "seq2", "seq4", "seq6", "seq7", "seq8"],
    )
    parser.add_argument("--validation_trajectories", nargs="+", default=["seq12", "seq14"])
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_model), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite latent safety selector")
    payload = json.loads(Path(args.candidate_pool).read_text())
    contract = dict(payload.get("configuration_evidence_contract", {}))
    if not bool(contract.get("outer_cross_fit")):
        raise ValueError("latent safety selector training requires outer-cross-fit factor evidence")
    if args.base_feature not in set(contract.get("feature_names", ())):
        raise ValueError("latent selector base feature is absent")
    payload["selector_base_feature"] = str(args.base_feature)
    train = _rows(payload, set(args.training_trajectories))
    validation = _rows(payload, set(args.validation_trajectories))
    x, basin, catastrophic, weight = _fit_arrays(train)
    estimators = []
    for target in (basin, catastrophic):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.05, class_weight="balanced", max_iter=2000, random_state=194917,
            ),
        )
        estimator.fit(x, target, logisticregression__sample_weight=weight)
        estimators.append(estimator)
    metadata = {
        "artifact_type": "goal_maplet_latent_safety_selector_v1",
        "feature_names": list(FEATURE_NAMES),
        "base_feature": str(args.base_feature),
        "residual_scale": 0.25,
        "catastrophic_veto_probability": 0.50,
        "training_trajectories": list(args.training_trajectories),
        "validation_trajectories": list(args.validation_trajectories),
        "candidate_pool_sha256": file_sha256(Path(args.candidate_pool)),
        "configuration_evidence_contract": contract,
        "physical_map_sha256": payload.get("physical_map_sha256"),
        "canonical_field_sha256": payload.get("canonical_field_sha256"),
        "typed_graph_sha256": payload.get("typed_graph_sha256"),
        "field_feature_contract_sha256": payload.get("field_feature_contract_sha256"),
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
    artifact = LatentSafetySelectorArtifact(estimators[0], estimators[1], metadata)
    artifact.save(output)
    result = {
        "stage": "train_goal_maplet_latent_safety_selector_v1",
        "model": str(output),
        "metadata": metadata,
        "query_count": {"train": len(train), "validation": len(validation)},
        "candidate_count": int(x.shape[0]),
        "heads": {
            "train": _head_report(train, artifact),
            "validation": _head_report(validation, artifact),
        },
        "reports": {
            split: {
                policy: _metrics(rows, artifact, policy)
                for policy in ("base", "residual", "safety_residual")
            }
            for split, rows in (("train", train), ("validation", validation))
        },
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
