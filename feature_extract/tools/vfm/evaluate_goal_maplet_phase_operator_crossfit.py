"""Compare frozen directional/Jacobian phase on one feature-cross-fitted fold.

Ranking and abstention are deliberately separate here: this evaluator accepts
no null or conditional-energy policy.  It verifies the held trajectory was
excluded from the feature mapper, canonical field, physical readout, typed
graph and validity calibration before comparing the two fixed phase operators.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import kendalltau, pearsonr
import torch

from feature_extract.tools.vfm.evaluate_goal_maplet_conditional_energy import (
    _candidate_generator_contract,
)
from feature_extract.tools.vfm.fit_evaluate_goal_maplet_conditional_energy import (
    _map_query_overlap_audit,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


MODE = "actual_parent_actual_child"
BANDS = ((0.25, 0.5), (0.5, 1.0), (1.0, 2.5))


def _trajectory(image_id: str) -> str:
    return str(image_id).replace("\\", "/").split("/", 1)[0]


def _pose_key(item: dict[str, object]) -> tuple[float, ...]:
    return tuple(
        np.asarray(item["pose_w2c"], dtype=np.float64)
        .round(8)
        .reshape(-1)
        .tolist()
    )


def _strict(item: dict[str, object]) -> bool:
    return bool(
        float(item["translation_m"]) <= 0.5
        and float(item["rotation_deg"]) <= 5.0
    )


def _summary(items: list[dict[str, object]]) -> dict[str, object]:
    if not items:
        raise ValueError("phase report contains no evaluated queries")
    translation = np.asarray(
        [float(item["translation_m"]) for item in items], dtype=np.float64
    )
    rotation = np.asarray(
        [float(item["rotation_deg"]) for item in items], dtype=np.float64
    )
    return {
        "query_count": len(items),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.percentile(translation, 90.0)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90.0)),
        "strict_0.5m_5deg": float(np.mean((translation <= 0.5) & (rotation <= 5.0))),
        "within_1m_10deg": float(np.mean((translation <= 1.0) & (rotation <= 10.0))),
        "catastrophic_rate": float(np.mean((translation > 5.0) | (rotation > 30.0))),
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if len(left) < 2 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan"), float("nan")
    return float(kendalltau(left, right).statistic), float(pearsonr(left, right).statistic)


def _finite_distribution(values: Iterable[float]) -> dict[str, object]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {"count": 0, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10.0)),
        "p90": float(np.percentile(array, 90.0)),
    }


def _operator_rows(report: dict[str, object], mode_name: str) -> dict[str, list[dict[str, object]]]:
    result = {}
    for row in report.get("rows", ()):  # type: ignore[union-attr]
        image_id = str(row["image_id"])
        details = [
            item
            for item in row.get("mode_details", {}).get(mode_name, ())
            if item.get("surface_alignment_score") is not None
        ]
        if not details:
            raise ValueError(f"missing phase-ranked candidates: {image_id}")
        result[image_id] = details
    return result


def _candidate_availability(
    report: dict[str, object], mode_name: str,
) -> dict[str, dict[str, object]]:
    result = {}
    for row in report.get("rows", ()):  # type: ignore[union-attr]
        image_id = str(row["image_id"])
        details = list(row.get("mode_details", {}).get(mode_name, ()))
        evaluated = [
            item for item in details
            if item.get("surface_alignment_score") is not None
        ]
        if not evaluated:
            raise ValueError(f"missing phase-evaluated candidates: {image_id}")
        strict = lambda item: _strict(item)
        one_m = lambda item: bool(
            float(item["translation_m"]) <= 1.0
            and float(item["rotation_deg"]) <= 10.0
        )
        result[image_id] = {
            "evaluated_candidate_count": len(evaluated),
            "full_candidate_count": len(details),
            "strict_available_evaluated": any(strict(item) for item in evaluated),
            "strict_available_full": any(strict(item) for item in details),
            "one_m_available_evaluated": any(one_m(item) for item in evaluated),
            "one_m_available_full": any(one_m(item) for item in details),
        }
    return result


def _candidate_coverage_summary(
    query_results: list[dict[str, object]],
) -> dict[str, object]:
    if not query_results:
        raise ValueError("candidate coverage contains no queries")
    result: dict[str, object] = {"query_count": len(query_results)}
    for name, success_key in (("strict", "strict"), ("one_m", "one_m")):
        available_evaluated = np.asarray([
            bool(row["candidate_availability"][f"{name}_available_evaluated"])
            for row in query_results
        ])
        available_full = np.asarray([
            bool(row["candidate_availability"][f"{name}_available_full"])
            for row in query_results
        ])
        selected = np.asarray([
            bool(row["directional"][success_key]) for row in query_results
        ])
        available_count = int(np.sum(available_evaluated))
        result[name] = {
            "phase_evaluated_oracle_rate": float(np.mean(available_evaluated)),
            "full_pool_oracle_rate": float(np.mean(available_full)),
            "directional_selected_rate": float(np.mean(selected)),
            "directional_capture_of_available": (
                float(np.sum(selected & available_evaluated)) / available_count
                if available_count else None
            ),
            "no_candidate_in_full_pool_count": int(np.sum(~available_full)),
            "candidate_only_outside_phase_topn_count": int(
                np.sum(available_full & ~available_evaluated)
            ),
            "phase_ranking_miss_count": int(
                np.sum(available_evaluated & ~selected)
            ),
        }
    return result


def _pairwise_pose_concordance(details: list[dict[str, object]]) -> dict[str, object]:
    quality = np.asarray([
        float(item["translation_m"]) + 0.02 * float(item["rotation_deg"])
        for item in details
    ])
    score = np.asarray([float(item["surface_alignment_score"]) for item in details])
    wins = ties = total = 0
    for left in range(len(details)):
        for right in range(left + 1, len(details)):
            if abs(float(quality[left] - quality[right])) <= 1.0e-9:
                continue
            total += 1
            expected = quality[left] < quality[right]
            delta = float(score[left] - score[right])
            if abs(delta) <= 1.0e-12:
                ties += 1
            elif (delta > 0.0) == expected:
                wins += 1
    return {"wins": wins, "ties": ties, "total": total}


def _oracle_band_concordance(details: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    quality = np.asarray([
        float(item["translation_m"]) + 0.02 * float(item["rotation_deg"])
        for item in details
    ])
    oracle = int(np.argmin(quality))
    oracle_score = float(details[oracle]["surface_alignment_score"])
    result = {}
    for low, high in BANDS:
        wins = ties = total = 0
        for index, item in enumerate(details):
            translation = float(item["translation_m"])
            if index == oracle or not (low < translation <= high):
                continue
            if quality[index] <= quality[oracle] + 1.0e-9:
                continue
            total += 1
            delta = oracle_score - float(item["surface_alignment_score"])
            if abs(delta) <= 1.0e-12:
                ties += 1
            elif delta > 0.0:
                wins += 1
        result[f"{low:.2f}_{high:.2f}m"] = {"wins": wins, "ties": ties, "total": total}
    return result


def _compare_rows(
    directional: dict[str, object],
    jacobian: dict[str, object],
    *,
    mode_name: str = MODE,
) -> dict[str, object]:
    left = _operator_rows(directional, mode_name)
    right = _operator_rows(jacobian, mode_name)
    availability = _candidate_availability(directional, mode_name)
    if set(left) != set(right):
        raise ValueError("phase operators evaluated different queries")
    top1 = {"both_right": 0, "directional_right_jacobian_wrong": 0,
            "jacobian_right_directional_wrong": 0, "both_wrong": 0,
            "same_top1_pose": 0}
    correlations = {"kendall": [], "pearson": []}
    method_top1 = {"directional": [], "jacobian": []}
    pairwise = {name: {"wins": 0, "ties": 0, "total": 0} for name in method_top1}
    bands = {
        name: {f"{low:.2f}_{high:.2f}m": {"wins": 0, "ties": 0, "total": 0}
               for low, high in BANDS}
        for name in method_top1
    }
    per_trajectory = defaultdict(lambda: {"directional": [], "jacobian": []})
    query_results = []
    for image_id in sorted(left):
        left_by_pose = {_pose_key(item): item for item in left[image_id]}
        right_by_pose = {_pose_key(item): item for item in right[image_id]}
        if set(left_by_pose) != set(right_by_pose):
            raise ValueError(f"phase operators evaluated different poses: {image_id}")
        keys = sorted(left_by_pose)
        left_score = np.asarray([
            float(left_by_pose[key]["surface_alignment_score"]) for key in keys
        ])
        right_score = np.asarray([
            float(right_by_pose[key]["surface_alignment_score"]) for key in keys
        ])
        kendall, pearson = _safe_correlation(left_score, right_score)
        correlations["kendall"].append(kendall)
        correlations["pearson"].append(pearson)
        left_top = left[image_id][0]
        right_top = right[image_id][0]
        method_top1["directional"].append(left_top)
        method_top1["jacobian"].append(right_top)
        trajectory = _trajectory(image_id)
        per_trajectory[trajectory]["directional"].append(left_top)
        per_trajectory[trajectory]["jacobian"].append(right_top)
        left_ok, right_ok = _strict(left_top), _strict(right_top)
        if left_ok and right_ok:
            top1["both_right"] += 1
        elif left_ok:
            top1["directional_right_jacobian_wrong"] += 1
        elif right_ok:
            top1["jacobian_right_directional_wrong"] += 1
        else:
            top1["both_wrong"] += 1
        if _pose_key(left_top) == _pose_key(right_top):
            top1["same_top1_pose"] += 1
        query_results.append({
            "image_id": image_id,
            "trajectory_id": trajectory,
            "directional": {
                "translation_m": float(left_top["translation_m"]),
                "rotation_deg": float(left_top["rotation_deg"]),
                "surface_alignment_score": float(
                    left_top["surface_alignment_score"]
                ),
                "strict": left_ok,
                "one_m": bool(
                    float(left_top["translation_m"]) <= 1.0
                    and float(left_top["rotation_deg"]) <= 10.0
                ),
                "pose_key": list(_pose_key(left_top)),
            },
            "jacobian": {
                "translation_m": float(right_top["translation_m"]),
                "rotation_deg": float(right_top["rotation_deg"]),
                "surface_alignment_score": float(
                    right_top["surface_alignment_score"]
                ),
                "strict": right_ok,
                "one_m": bool(
                    float(right_top["translation_m"]) <= 1.0
                    and float(right_top["rotation_deg"]) <= 10.0
                ),
                "pose_key": list(_pose_key(right_top)),
            },
            "same_top1_pose": _pose_key(left_top) == _pose_key(right_top),
            "candidate_availability": availability[image_id],
            "operator_score_kendall": kendall if np.isfinite(kendall) else None,
            "operator_score_pearson": pearson if np.isfinite(pearson) else None,
        })
        for name, details in (("directional", left[image_id]), ("jacobian", right[image_id])):
            value = _pairwise_pose_concordance(details)
            for key in pairwise[name]:
                pairwise[name][key] += int(value[key])
            value_bands = _oracle_band_concordance(details)
            for band, counts in value_bands.items():
                for key in bands[name][band]:
                    bands[name][band][key] += int(counts[key])
    for name in pairwise:
        total = pairwise[name]["total"]
        pairwise[name]["concordance"] = (
            (pairwise[name]["wins"] + 0.5 * pairwise[name]["ties"]) / total
            if total else None
        )
        for counts in bands[name].values():
            total = counts["total"]
            counts["concordance"] = (
                (counts["wins"] + 0.5 * counts["ties"]) / total if total else None
            )
    result = {
        "metrics": {name: _summary(items) for name, items in method_top1.items()},
        "per_trajectory": {
            trajectory: {name: _summary(items) for name, items in values.items()}
            for trajectory, values in sorted(per_trajectory.items())
        },
        "top1_complementarity": {**top1, "query_count": len(left)},
        "operator_score_correlation_per_query": {
            name: _finite_distribution(values) for name, values in correlations.items()
        },
        "pose_quality_pairwise_concordance": pairwise,
        "oracle_vs_negative_distance_band_concordance": bands,
        "query_results": query_results,
    }
    result["candidate_coverage_and_ranking_headroom"] = (
        _candidate_coverage_summary(query_results)
    )
    return result


def _checkpoint_metadata(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or not isinstance(value.get("metadata"), dict):
        raise ValueError(f"checkpoint lacks metadata: {path}")
    return dict(value["metadata"])


def _assert_disjoint(values: Iterable[object], heldout: set[str], label: str) -> None:
    overlap = heldout.intersection(str(value) for value in values)
    if overlap:
        raise ValueError(f"held trajectory leaked into {label}: {sorted(overlap)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directional_report", required=True)
    parser.add_argument("--jacobian_report", required=True)
    parser.add_argument("--directional_policy", required=True)
    parser.add_argument("--jacobian_policy", required=True)
    parser.add_argument("--canonical_field_summary", required=True)
    parser.add_argument("--mapping_contributors", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--physical_instance_readout", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--validity_summary", required=True)
    parser.add_argument("--typed_graph_summary", required=True)
    parser.add_argument("--heldout_trajectories", nargs="+", required=True)
    parser.add_argument(
        "--geometry_protocol",
        choices=("fixed_train_2dgs", "fold_rebuilt_2dgs"),
        required=True,
    )
    parser.add_argument("--mode_name", default=MODE)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite phase cross-fit evaluation")
    directional_path = Path(args.directional_report)
    jacobian_path = Path(args.jacobian_report)
    directional = json.loads(directional_path.read_text())
    jacobian = json.loads(jacobian_path.read_text())
    if _candidate_generator_contract(directional) != _candidate_generator_contract(jacobian):
        raise ValueError("phase reports use different candidate generators")
    left_surface = dict(directional.get("surface_verification_contract") or {})
    right_surface = dict(jacobian.get("surface_verification_contract") or {})
    if left_surface.get("candidate_pool_sha256") != right_surface.get("candidate_pool_sha256"):
        raise ValueError("phase reports use different candidate pools")
    if left_surface.get("conditional_pose_energy_sha256") is not None or right_surface.get(
        "conditional_pose_energy_sha256"
    ) is not None:
        raise ValueError("phase ranking audit cannot include a null/joint energy")
    policies = {
        "directional": (Path(args.directional_policy), left_surface),
        "jacobian": (Path(args.jacobian_policy), right_surface),
    }
    for name, (path, surface) in policies.items():
        policy = json.loads(path.read_text())
        if file_sha256(path) != surface.get("phase_readout_policy_sha256"):
            raise ValueError(f"{name} policy and report differ")
        if policy.get("operator_parameters_refit") is not False:
            raise ValueError(f"{name} operator was not frozen for cross-fit")
        if not (
            bool(policy.get("map_crossfit_lineage_rebind"))
            or bool(policy.get("feature_pipeline_crossfit_binding"))
            or bool(policy.get("map_crossfit_binding"))
        ):
            raise ValueError(f"{name} policy lacks map-cross-fit binding lineage")
        for key in (
            "physical_map_sha256",
            "canonical_field_sha256",
            "physical_instance_readout_sha256",
        ):
            if policy.get(key) != directional.get(key):
                raise ValueError(f"{name} policy map lineage differs: {key}")

    heldout = {str(value) for value in args.heldout_trajectories}
    query_trajectories = {
        _trajectory(str(row["image_id"])) for row in directional.get("rows", ())
    }
    if not query_trajectories or not query_trajectories.issubset(heldout):
        raise ValueError("evaluation queries are not confined to the held fold")
    overlap = _map_query_overlap_audit(
        directional,
        canonical_field_summary=Path(args.canonical_field_summary),
        mapping_contributors=Path(args.mapping_contributors),
    )
    if not overlap["map_query_image_disjoint"]:
        raise ValueError("held queries occur in canonical-field contributors")
    field_summary = json.loads(Path(args.canonical_field_summary).read_text())
    _assert_disjoint(field_summary.get("mapping_trajectory_ids", ()), heldout, "canonical field")
    mapper_metadata = _checkpoint_metadata(Path(args.surface_mapper))
    for key in (
        "training_trajectory_ids",
        "validation_trajectory_ids",
        "prototype_trajectory_ids",
    ):
        _assert_disjoint(mapper_metadata.get(key, ()), heldout, f"surface mapper {key}")
    if not heldout.issubset(set(mapper_metadata.get("strict_holdout_trajectory_ids", ()))):
        raise ValueError("surface mapper did not declare the held trajectory")
    readout_metadata = _checkpoint_metadata(Path(args.physical_instance_readout))
    for key in (
        "training_trajectory_ids",
        "selection_trajectory_ids",
        "validation_trajectory_ids",
    ):
        _assert_disjoint(readout_metadata.get(key, ()), heldout, f"physical readout {key}")
    expected_mapper = file_sha256(Path(args.surface_mapper))
    expected_readout = file_sha256(Path(args.physical_instance_readout))
    for key, expected in (
        ("physical_map_sha256", directional.get("physical_map_sha256")),
        ("canonical_field_sha256", directional.get("canonical_field_sha256")),
        ("surface_mapper_sha256", expected_mapper),
    ):
        if readout_metadata.get(key) != expected:
            raise ValueError(f"physical readout lineage differs: {key}")
    validity = json.loads(Path(args.validity_summary).read_text())
    calibration = dict(validity.get("calibration_metadata") or {})
    _assert_disjoint(calibration.get("fit_trajectory_ids", ()), heldout, "validity calibration")
    if validity.get("calibration_sha256") != directional.get("validity_calibration_sha256"):
        raise ValueError("validity summary and candidate generator differ")
    for key, expected in (
        ("physical_map_sha256", directional.get("physical_map_sha256")),
        ("canonical_field_sha256", directional.get("canonical_field_sha256")),
        ("physical_instance_readout_sha256", expected_readout),
        ("surface_mapper_file_sha256", expected_mapper),
    ):
        if calibration.get(key) != expected:
            raise ValueError(f"validity calibration lineage differs: {key}")
    graph = json.loads(Path(args.typed_graph_summary).read_text())
    if graph.get("typed_graph_sha256") != directional.get("typed_graph_sha256"):
        raise ValueError("typed-graph summary and candidate generator differ")
    for key, expected in (
        ("physical_map_sha256", directional.get("physical_map_sha256")),
        ("canonical_field_sha256", directional.get("canonical_field_sha256")),
    ):
        if graph.get(key) != expected:
            raise ValueError(f"typed-graph lineage differs: {key}")
    if dict(graph.get("metadata") or {}).get(
        "physical_instance_readout_sha256"
    ) != expected_readout:
        raise ValueError("typed graph and physical readout differ")
    contract = json.loads(Path(args.field_feature_contract).read_text())
    if contract.get("content_sha256") != directional.get("field_feature_contract_sha256"):
        raise ValueError("feature contract and phase report differ")
    if contract.get("canonical_field_sha256") != directional.get(
        "canonical_field_sha256"
    ) or contract.get("query_readout_sha256") != expected_mapper:
        raise ValueError("feature contract map/readout lineage differs")
    comparison = _compare_rows(
        directional, jacobian, mode_name=str(args.mode_name)
    )
    result = {
        "stage": "goal_maplet_phase_operator_outer_feature_crossfit_evaluation",
        "heldout_trajectories": sorted(heldout),
        "geometry_protocol": str(args.geometry_protocol),
        "geometry_was_outer_crossfit": args.geometry_protocol == "fold_rebuilt_2dgs",
        "feature_pipeline_outer_crossfit": True,
        "ranking_and_null_separated": True,
        "null_or_abstention_evaluated": False,
        "operator_parameters_refit": False,
        "production_promotion_allowed": False,
        "production_blocker": (
            "fixed 2DGS geometry is not outer-cross-fitted"
            if args.geometry_protocol == "fixed_train_2dgs"
            else "method-selection folds are not an untouched final test"
        ),
        "candidate_generator_contract": _candidate_generator_contract(directional),
        "map_query_overlap_audit": overlap,
        "artifacts": {
            "directional_report": str(directional_path),
            "directional_report_sha256": file_sha256(directional_path),
            "jacobian_report": str(jacobian_path),
            "jacobian_report_sha256": file_sha256(jacobian_path),
            "surface_mapper_sha256": file_sha256(Path(args.surface_mapper)),
            "physical_instance_readout_sha256": file_sha256(
                Path(args.physical_instance_readout)
            ),
            "canonical_field_summary_sha256": file_sha256(
                Path(args.canonical_field_summary)
            ),
        },
        **comparison,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "heldout_trajectories": result["heldout_trajectories"],
        "geometry_protocol": result["geometry_protocol"],
        "metrics": result["metrics"],
        "top1_complementarity": result["top1_complementarity"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
