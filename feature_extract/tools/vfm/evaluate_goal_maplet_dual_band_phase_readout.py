"""Fit and evaluate a tiny coordinate-free G19-B dual-band pose ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


COMPONENT_NAMES = (
    "mapper_cosine",
    "context_cosine",
    "horizontal_phase",
    "vertical_phase",
    "horizontal_phase_step2",
    "vertical_phase_step2",
    "diagonal_down_phase",
    "diagonal_up_phase",
    "coverage",
)
IDENTITY_PHASE_INDICES = tuple(range(8))
SINGLE_SCALE_PHASE_INDICES = (2, 3)
MULTISCALE_PHASE_INDICES = tuple(range(2, 8))


def _load(path: Path, mode_name: str) -> dict[str, object]:
    report = json.loads(Path(path).read_text())
    contract = dict(report.get("surface_verification_contract") or {})
    if not bool(contract.get("phase_preserving_dual_band", False)):
        raise ValueError("G19-B evaluation requires dual-band surface evidence")
    rows = []
    for row in sorted(report.get("rows", []), key=lambda item: str(item["image_id"])):
        details = list(row.get("mode_details", {}).get(str(mode_name), []))
        diagnostic = row.get("ranking_diagnostics", {}).get(str(mode_name), {})
        components = list(diagnostic.get("surface_phase_components_preorder") or [])
        order = list(diagnostic.get("surface_alignment_original_indices") or [])
        evaluated = len(components)
        if not details or evaluated <= 0 or len(order) < evaluated or len(details) < evaluated:
            raise ValueError(f"incomplete G19-B evidence: {row.get('image_id')}")
        translation = np.full((evaluated,), np.inf, dtype=np.float64)
        rotation = np.full((evaluated,), np.inf, dtype=np.float64)
        for ranked_index, original_index in enumerate(order[:evaluated]):
            if int(original_index) >= evaluated:
                raise ValueError("evaluated G19-B candidate order is not closed")
            translation[int(original_index)] = float(details[ranked_index]["translation_m"])
            rotation[int(original_index)] = float(details[ranked_index]["rotation_deg"])
        feature = np.asarray([
            [float(item[name]) for name in COMPONENT_NAMES] for item in components
        ], dtype=np.float64)
        rows.append({
            "image_id": str(row["image_id"]),
            "trajectory_id": str(row["image_id"]).split("/", 1)[0],
            "feature": feature,
            "translation_m": translation,
            "rotation_deg": rotation,
        })
    return {"contract": contract, "rows": rows, "report": report}


def _metrics(rows: list[dict[str, object]], selected: list[int]) -> dict[str, object]:
    translation = np.asarray([
        np.asarray(row["translation_m"])[index] for row, index in zip(rows, selected)
    ], dtype=np.float64)
    rotation = np.asarray([
        np.asarray(row["rotation_deg"])[index] for row, index in zip(rows, selected)
    ], dtype=np.float64)
    return {
        "query_count": int(len(rows)),
        "translation_m": {
            "median": float(np.median(translation)),
            "p90": float(np.percentile(translation, 90.0)),
        },
        "rotation_deg": {
            "median": float(np.median(rotation)),
            "p90": float(np.percentile(rotation, 90.0)),
        },
        "strict_0.5m_5deg": float(np.mean((translation <= 0.5) & (rotation <= 5.0))),
        "success_1m_10deg": float(np.mean((translation <= 1.0) & (rotation <= 10.0))),
        "catastrophic_rate": float(np.mean((translation > 5.0) | (rotation > 30.0))),
        "selected_original_indices": [int(value) for value in selected],
    }


def _policy_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    index = {name: offset for offset, name in enumerate(COMPONENT_NAMES)}
    policies = {
        "frozen_candidate_top1": [0 for _row in rows],
        "mapper_cosine": [
            int(np.argmax(np.asarray(row["feature"])[:, index["mapper_cosine"]]))
            for row in rows
        ],
        "context_cosine": [
            int(np.argmax(np.asarray(row["feature"])[:, index["context_cosine"]]))
            for row in rows
        ],
        "directional_phase_only": [
            int(np.argmax(np.mean(np.asarray(row["feature"])[:, 2:4], axis=1)))
            for row in rows
        ],
        "multiscale_directional_phase_only": [
            int(np.argmax(np.mean(np.asarray(row["feature"])[:, 2:8], axis=1)))
            for row in rows
        ],
        "dual_band_fixed": [
            int(np.argmax(
                0.50 * np.asarray(row["feature"])[:, 0]
                + 0.20 * np.asarray(row["feature"])[:, 1]
                + 0.15 * np.asarray(row["feature"])[:, 2]
                + 0.15 * np.asarray(row["feature"])[:, 3]
            ))
            for row in rows
        ],
    }
    return {name: _metrics(rows, selected) for name, selected in policies.items()}


def _fit_pairwise(
    rows: list[dict[str, object]],
    component_indices: tuple[int, ...] = IDENTITY_PHASE_INDICES,
) -> tuple[StandardScaler, LogisticRegression, dict[str, object]]:
    difference, labels, weights = [], [], []
    usable_query_count = 0
    for row in rows:
        feature = np.asarray(row["feature"], dtype=np.float64)[:, component_indices]
        translation = np.asarray(row["translation_m"], dtype=np.float64)
        rotation = np.asarray(row["rotation_deg"], dtype=np.float64)
        usable = (translation <= 1.0) & (rotation <= 10.0)
        if not np.any(usable):
            continue
        quality = translation / 0.5 + rotation / 5.0
        target = int(np.argmin(quality))
        usable_query_count += 1
        for negative in range(feature.shape[0]):
            if negative == target:
                continue
            delta = feature[target] - feature[negative]
            phase_negative = bool(
                0.5 <= translation[negative] <= 2.5 and rotation[negative] <= 10.0
            )
            weight = 2.0 if phase_negative else 1.0
            difference.extend([delta, -delta])
            labels.extend([1, 0])
            weights.extend([weight, weight])
    if not difference:
        raise ValueError("G19-B has no usable pairwise training rows")
    x = np.asarray(difference, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    sample_weight = np.asarray(weights, dtype=np.float64)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        fit_intercept=False,
        max_iter=2000,
        random_state=1902,
        solver="liblinear",
    )
    model.fit(scaler.transform(x), y, sample_weight=sample_weight)
    return scaler, model, {
        "usable_query_count": int(usable_query_count),
        "symmetric_pair_count": int(x.shape[0]),
        "phase_negative_weight": 2.0,
    }


def _apply_pairwise(
    rows: list[dict[str, object]],
    scaler: StandardScaler,
    model: LogisticRegression,
    component_indices: tuple[int, ...] = IDENTITY_PHASE_INDICES,
) -> tuple[dict[str, object], list[list[float]]]:
    selected, scores = [], []
    for row in rows:
        feature = np.asarray(row["feature"], dtype=np.float64)[:, component_indices]
        value = model.decision_function(scaler.transform(feature))
        selected.append(int(np.argmax(value)))
        scores.append(np.asarray(value, dtype=np.float64).tolist())
    report = _metrics(rows, selected)
    report["candidate_scores"] = scores
    return report, scores


def _leave_one_trajectory_out(
    rows: list[dict[str, object]],
    component_indices: tuple[int, ...] = IDENTITY_PHASE_INDICES,
) -> dict[str, object]:
    held_rows: list[dict[str, object]] = []
    held_selected: list[int] = []
    folds = []
    trajectories = sorted({str(row["trajectory_id"]) for row in rows})
    for trajectory in trajectories:
        fit_rows = [row for row in rows if str(row["trajectory_id"]) != trajectory]
        selection_rows = [row for row in rows if str(row["trajectory_id"]) == trajectory]
        scaler, model, fit_report = _fit_pairwise(fit_rows, component_indices)
        metrics, _scores = _apply_pairwise(
            selection_rows, scaler, model, component_indices,
        )
        held_rows.extend(selection_rows)
        held_selected.extend(metrics["selected_original_indices"])
        folds.append({
            "held_trajectory_id": trajectory,
            "fit": fit_report,
            "metrics": {key: value for key, value in metrics.items() if key != "candidate_scores"},
        })
    return {"aggregate": _metrics(held_rows, held_selected), "folds": folds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_report", required=True)
    parser.add_argument("--selection_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_model", default="")
    parser.add_argument("--mode_name", default="actual_parent_actual_child")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    model_output = Path(args.output_model) if args.output_model else None
    if (output.exists() or (model_output is not None and model_output.exists())) and not bool(args.force):
        raise FileExistsError("refusing to overwrite G19-B evaluation")
    train_path, selection_path = Path(args.train_report), Path(args.selection_report)
    train = _load(train_path, str(args.mode_name))
    selection = _load(selection_path, str(args.mode_name))
    train_trajectories = {str(row["trajectory_id"]) for row in train["rows"]}
    selection_trajectories = {str(row["trajectory_id"]) for row in selection["rows"]}
    if train_trajectories & selection_trajectories:
        raise ValueError("G19-B train and selection trajectories overlap")
    for key in ("physical_instance_readout_sha256",):
        if train["contract"].get(key) != selection["contract"].get(key):
            raise ValueError(f"G19-B report lineage differs: {key}")
    for key in ("physical_map_sha256", "canonical_field_sha256"):
        if train["report"].get(key) != selection["report"].get(key):
            raise ValueError(f"G19-B report lineage differs: {key}")
    variants = {
        "identity_multiscale_phase_pairwise": IDENTITY_PHASE_INDICES,
        "single_scale_phase_pairwise": SINGLE_SCALE_PHASE_INDICES,
        "multiscale_phase_pairwise": MULTISCALE_PHASE_INDICES,
    }
    fitted = {}
    for name, indices in variants.items():
        scaler, model, fit_report = _fit_pairwise(train["rows"], indices)
        cross_trajectory = _leave_one_trajectory_out(train["rows"], indices)
        train_metrics, _ = _apply_pairwise(train["rows"], scaler, model, indices)
        selection_metrics, _ = _apply_pairwise(
            selection["rows"], scaler, model, indices,
        )
        effective_weight = model.coef_[0] / np.maximum(scaler.scale_, 1.0e-12)
        fitted[name] = {
            "indices": indices,
            "fit": {
                **fit_report,
                "type": "fixed_l2_pairwise_logistic_on_candidate_component_differences",
                "component_names": [COMPONENT_NAMES[index] for index in indices],
                "C": 0.1,
                "standardizer_mean": scaler.mean_.tolist(),
                "standardizer_scale": scaler.scale_.tolist(),
                "coefficient": model.coef_[0].tolist(),
                "effective_raw_component_weight": effective_weight.tolist(),
                "intercept": 0.0,
            },
            "leave_one_trajectory_out": cross_trajectory,
            "train": train_metrics,
            "selection": selection_metrics,
        }

    def cv_key(name: str) -> tuple[float, ...]:
        metric = fitted[name]["leave_one_trajectory_out"]["aggregate"]
        return (
            float(metric["catastrophic_rate"]),
            float(metric["translation_m"]["p90"]),
            -float(metric["success_1m_10deg"]),
            -float(metric["strict_0.5m_5deg"]),
        )

    phase_names = ("single_scale_phase_pairwise", "multiscale_phase_pairwise")
    training_preselected_policy = min(phase_names, key=cv_key)

    def promotion_key(name: str) -> tuple[float, ...]:
        metric = fitted[name]["selection"]
        return (
            float(metric["catastrophic_rate"]),
            -float(metric["success_1m_10deg"]),
            float(metric["translation_m"]["p90"]),
            -float(metric["strict_0.5m_5deg"]),
            float(metric["translation_m"]["median"]),
        )

    promotion_selected_policy = min(phase_names, key=promotion_key)
    selected_fit = fitted[promotion_selected_policy]["fit"]
    policy_payload = {
        "artifact_type": "goal_maplet_phase_readout_policy_v1",
        "selected_variant": promotion_selected_policy,
        "component_names": selected_fit["component_names"],
        "standardizer_mean": selected_fit["standardizer_mean"],
        "standardizer_scale": selected_fit["standardizer_scale"],
        "coefficient": selected_fit["coefficient"],
        "intercept": 0.0,
        "physical_map_sha256": train["report"].get("physical_map_sha256"),
        "canonical_field_sha256": train["report"].get("canonical_field_sha256"),
        "physical_instance_readout_sha256": train["contract"].get(
            "physical_instance_readout_sha256"
        ),
        "training_report_sha256": file_sha256(train_path),
        "promotion_report_sha256": file_sha256(selection_path),
        "selection_used_for_parameter_training": False,
        "absolute_image_coordinates_used": False,
        "stored_downstream_embedding_count": 0,
        "uses_point_correspondences": False,
        "uses_pnp": False,
    }
    policy_sha256 = None
    if model_output is not None:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        model_output.write_text(json.dumps(policy_payload, indent=2, sort_keys=True) + "\n")
        policy_sha256 = file_sha256(model_output)
    result = {
        "stage": "g19_b_coordinate_free_dual_band_phase_readout",
        "component_names": list(COMPONENT_NAMES),
        "selected_policy": f"proposal_identity_plus_{promotion_selected_policy}",
        "training_preselected_policy": training_preselected_policy,
        "promotion_selected_policy": promotion_selected_policy,
        "phase_structure_selection_uses_seq11": False,
        "phase_structure_selection_key": (
            "training_leave_one_trajectory_out:catastrophic,p90,-1m,-strict"
        ),
        "promotion_uses_seq11": True,
        "promotion_key": "catastrophic,-1m_success,p90,-strict,median",
        "selected_policy_rationale": (
            "The proposal already supplies low-frequency physical identity; "
            "surface verification adds only coordinate-free directional phase."
        ),
        "training_trajectory_ids": sorted(train_trajectories),
        "selection_trajectory_ids": sorted(selection_trajectories),
        "selection_used_for_training": False,
        "variant_fits": {
            name: {
                "fit": value["fit"],
                "training_leave_one_trajectory_out": value["leave_one_trajectory_out"],
            }
            for name, value in fitted.items()
        },
        "train_policies": {
            **_policy_metrics(train["rows"]),
            **{name: value["train"] for name, value in fitted.items()},
        },
        "selection_policies": {
            **_policy_metrics(selection["rows"]),
            **{name: value["selection"] for name, value in fitted.items()},
        },
        "deployment_contract": {
            "stored_map_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "absolute_image_coordinates_used": False,
            "discrete_correspondences_used": False,
            "uses_pnp": False,
        },
        "lineage": {
            "train_report_sha256": file_sha256(train_path),
            "selection_report_sha256": file_sha256(selection_path),
            "physical_instance_readout_sha256": train["contract"].get(
                "physical_instance_readout_sha256"
            ),
            "physical_map_sha256": train["report"].get("physical_map_sha256"),
            "canonical_field_sha256": train["report"].get("canonical_field_sha256"),
            "phase_readout_policy_path": str(model_output) if model_output is not None else None,
            "phase_readout_policy_sha256": policy_sha256,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "train_policies"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
