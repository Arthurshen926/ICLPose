"""Merge disjoint outer-feature-cross-fit phase-basin reports exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.tools.vfm.audit_goal_maplet_phase_basin import (
    _component_metrics,
    _sequence_metrics,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _trajectory(image_id: str) -> str:
    return str(image_id).replace("\\", "/").split("/", 1)[0]


def _validate_crossfit_lineage(
    reports: list[dict[str, object]],
    evaluations: list[dict[str, object]],
) -> bool:
    """Bind every basin fold to a separately audited phase-evaluation fold."""

    if len(reports) != len(evaluations):
        raise ValueError("basin reports and cross-fit evaluations differ in count")
    evaluation_by_field: dict[str, dict[str, object]] = {}
    for evaluation in evaluations:
        if evaluation.get("stage") != (
            "goal_maplet_phase_operator_outer_feature_crossfit_evaluation"
        ):
            raise ValueError("basin lineage input is not a feature-cross-fit evaluation")
        if not bool(evaluation.get("feature_pipeline_outer_crossfit")):
            raise ValueError("basin lineage input did not pass feature cross-fit")
        contract = dict(evaluation.get("candidate_generator_contract") or {})
        field_sha = str(contract.get("canonical_field_sha256", ""))
        if not field_sha or field_sha in evaluation_by_field:
            raise ValueError("cross-fit evaluations lack unique canonical fields")
        evaluation_by_field[field_sha] = evaluation

    geometry_crossfit = []
    for report in reports:
        field_sha = str(report.get("canonical_field_sha256", ""))
        evaluation = evaluation_by_field.get(field_sha)
        if evaluation is None:
            raise ValueError("basin canonical field has no audited cross-fit evaluation")
        contract = dict(evaluation.get("candidate_generator_contract") or {})
        if str(report.get("physical_map_sha256", "")) != str(
            contract.get("physical_map_sha256", "")
        ):
            raise ValueError("basin and cross-fit evaluation use different physical maps")
        basin_images = {
            str(row["image_id"]) for row in report.get("ground_truth", ())
        }
        evaluation_images = {
            str(row["image_id"]) for row in evaluation.get("query_results", ())
        }
        if basin_images != evaluation_images:
            raise ValueError("basin and cross-fit evaluation query sets differ")
        basin_trajectories = {_trajectory(image_id) for image_id in basin_images}
        heldout = {
            str(value) for value in evaluation.get("heldout_trajectories", ())
        }
        if basin_trajectories != heldout:
            raise ValueError("basin queries do not match the audited held fold")
        geometry_crossfit.append(bool(evaluation.get("geometry_was_outer_crossfit")))
    return all(geometry_crossfit)


def _merge_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    if not reports:
        raise ValueError("no basin reports")
    factors = {int(report["render_supersample_factor"]) for report in reports}
    physical = {str(report["physical_map_sha256"]) for report in reports}
    policy_types = {str(report["phase_readout_artifact_type"]) for report in reports}
    magnitudes_t = {
        tuple(float(value) for value in report["translation_magnitudes"])
        for report in reports
    }
    magnitudes_r = {
        tuple(float(value) for value in report["rotation_magnitudes_deg"])
        for report in reports
    }
    if len(factors) != 1 or factors != {2}:
        raise ValueError("basin folds must use the same factor-2 renderer")
    if len(physical) != 1 or len(policy_types) != 1:
        raise ValueError("basin folds use different physical maps or operator types")
    if len(magnitudes_t) != 1 or len(magnitudes_r) != 1:
        raise ValueError("basin folds use different perturbation grids")
    fields = [str(report["canonical_field_sha256"]) for report in reports]
    if len(fields) != len(set(fields)):
        raise ValueError("outer-feature-cross-fit basin folds reused a canonical field")
    rows = [dict(row) for report in reports for row in report.get("rows", ())]
    ground_truth = [
        dict(row) for report in reports for row in report.get("ground_truth", ())
    ]
    image_ids = [str(row["image_id"]) for row in ground_truth]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("outer-map-cross-fit basin query rows overlap")
    translation = list(next(iter(magnitudes_t)))
    rotation = list(next(iter(magnitudes_r)))
    trajectories = sorted({_trajectory(image_id) for image_id in image_ids})
    per_trajectory: dict[str, object] = {}
    for trajectory in trajectories:
        selected_rows = [
            row for row in rows if _trajectory(str(row["image_id"])) == trajectory
        ]
        per_trajectory[trajectory] = _sequence_metrics(
            selected_rows, translation, rotation
        )
    component_names = (
        "jacobian_phase_visible",
        "jacobian_log_scale_agreement",
        "jacobian_observability",
    )
    result = {
        "query_count": len(image_ids),
        "trajectory_ids": trajectories,
        "physical_map_sha256": next(iter(physical)),
        "canonical_field_sha256_per_fold": fields,
        "phase_readout_artifact_type": next(iter(policy_types)),
        "render_supersample_factor": 2,
        "translation_magnitudes": translation,
        "rotation_magnitudes_deg": rotation,
        "summary": _sequence_metrics(rows, translation, rotation),
        "per_trajectory": per_trajectory,
        "component_summary": {
            component: _component_metrics(
                rows, ground_truth, component, translation, rotation
            )
            for component in component_names
        },
        "rows": rows,
        "ground_truth": ground_truth,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--crossfit_evaluations", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite merged basin report")
    paths = [Path(value) for value in args.inputs]
    reports = [json.loads(path.read_text()) for path in paths]
    evaluation_paths = [Path(value) for value in args.crossfit_evaluations]
    evaluations = [json.loads(path.read_text()) for path in evaluation_paths]
    geometry_was_outer_crossfit = _validate_crossfit_lineage(reports, evaluations)
    result = {
        "stage": "goal_maplet_phase_basin_outer_feature_crossfit_merged",
        "feature_pipeline_outer_crossfit": True,
        "feature_lineage_verified_against_phase_evaluations": True,
        "geometry_was_outer_crossfit": geometry_was_outer_crossfit,
        "production_promotion_allowed": False,
        "production_blocker": (
            "method-selection folds are not an untouched final test"
            if geometry_was_outer_crossfit
            else "fixed 2DGS geometry is not outer-cross-fitted"
        ),
        "fold_inputs": [
            {"path": str(path), "sha256": file_sha256(path)} for path in paths
        ],
        "crossfit_evaluation_inputs": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in evaluation_paths
        ],
        **_merge_reports(reports),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "phase_readout_artifact_type": result["phase_readout_artifact_type"],
        "query_count": result["query_count"],
        "summary": result["summary"],
        "per_trajectory": result["per_trajectory"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
