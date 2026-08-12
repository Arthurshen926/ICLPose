"""Attribute OOF localization failures to generator, pruning, verifier or refinement.

This is an oracle-only analysis.  Ground-truth pose errors are read only after
the deployed candidate/refinement decisions have been made; they never alter a
candidate list, winner or acceptance decision.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


MODE_KEY = "actual_parent_actual_child"
RAW_STAGE = "retained_seed_heap"
STRUCTURAL_PRESCREEN_STAGE = "structural_equivalence_prescreen"
SPARSE_SCORED_STAGE = "full_map_sparse_seed_vfm_likelihood"
EXACT_POOL_STAGE = "exact_verification_pool_selection"
EXACT_RANKED_STAGE = "full_surface_ranked"
RETAINED_STAGE = "topn_after_nms"
VIEW_CHART_STAGE = "mapping_view_anchor_beam"

THRESHOLDS = {
    "strict_0.5m_5deg": (0.5, 5.0, "within_0_5m_5deg_count"),
    "loose_1m_10deg": (1.0, 10.0, "within_1m_10deg_count"),
}


def _wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count <= 0:
        return [float("nan"), float("nan")]
    probability = float(successes) / float(count)
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    radius = z * sqrt(
        probability * (1.0 - probability) / count + z * z / (4.0 * count**2)
    ) / denominator
    return [float(center - radius), float(center + radius)]


def _diagnostics(row: dict[str, object]) -> dict[str, object]:
    value = row.get("proposal_diagnostics", {})
    if isinstance(value, dict) and MODE_KEY in value:
        value = value[MODE_KEY]
    return value if isinstance(value, dict) else {}


def _stage_has_basin(
    diagnostics: dict[str, object], stage: str, count_field: str,
) -> bool | None:
    value = diagnostics.get(stage)
    if not isinstance(value, dict) or count_field not in value:
        return None
    return int(value[count_field]) > 0


def _candidate_top1_success(
    row: dict[str, object], translation_m: float, rotation_deg: float,
) -> bool | None:
    details = row.get("mode_details", {})
    modes = details.get(MODE_KEY, []) if isinstance(details, dict) else []
    if not modes:
        return None
    top1 = modes[0]
    return bool(
        float(top1["translation_m"]) <= float(translation_m)
        and float(top1["rotation_deg"]) <= float(rotation_deg)
    )


def _classify(
    candidate: dict[str, object],
    final: dict[str, object],
    *,
    translation_m: float,
    rotation_deg: float,
    count_field: str,
) -> tuple[str, dict[str, bool | None]]:
    diagnostic = _diagnostics(candidate)
    stages = {
        "view_chart": _stage_has_basin(diagnostic, VIEW_CHART_STAGE, count_field),
        "raw_proposal": _stage_has_basin(diagnostic, RAW_STAGE, count_field),
        "structural_prescreen": _stage_has_basin(
            diagnostic, STRUCTURAL_PRESCREEN_STAGE, count_field
        ),
        "sparse_scored_pool": _stage_has_basin(
            diagnostic, SPARSE_SCORED_STAGE, count_field
        ),
        "exact_pool": _stage_has_basin(diagnostic, EXACT_POOL_STAGE, count_field),
        "exact_ranked": _stage_has_basin(diagnostic, EXACT_RANKED_STAGE, count_field),
        "retained_topn": _stage_has_basin(diagnostic, RETAINED_STAGE, count_field),
        "candidate_top1": _candidate_top1_success(
            candidate, translation_m, rotation_deg
        ),
        "final_selected": bool(
            float(final["final_translation_m"]) <= float(translation_m)
            and float(final["final_rotation_deg"]) <= float(rotation_deg)
        ),
    }
    if stages["final_selected"]:
        return "success", stages
    if stages["raw_proposal"] is None:
        return "U_missing_stage_diagnostics", stages
    if not stages["raw_proposal"]:
        return "G_generator_absence", stages
    if stages["structural_prescreen"] is False:
        return "S_structural_prescreen_pruning", stages
    if stages["sparse_scored_pool"] is False:
        return "S_structural_prescreen_pruning", stages
    if stages["exact_pool"] is None:
        return "U_missing_stage_diagnostics", stages
    if not stages["exact_pool"]:
        return "S_sparse_screen_or_exact_pool_pruning", stages
    if stages["exact_ranked"] is None or stages["retained_topn"] is None:
        return "U_missing_stage_diagnostics", stages
    if not stages["exact_ranked"] or not stages["retained_topn"]:
        return "V_exact_verifier_ranking_or_nms", stages
    if stages["candidate_top1"]:
        return "R_refinement_or_gate_regression", stages
    return "R_refinement_or_final_selection_failure", stages


def _summarize(rows: list[dict[str, object]], threshold: str) -> dict[str, object]:
    classifications = [str(row["taxonomy"][threshold]["class"]) for row in rows]
    names = sorted(set(classifications))
    count = len(rows)
    return {
        "query_count": count,
        "classes": {
            name: {
                "count": classifications.count(name),
                "rate": float(classifications.count(name) / count) if count else 0.0,
                "wilson95": _wilson(classifications.count(name), count),
            }
            for name in names
        },
    }


def build_failure_taxonomy(
    candidate_reports: list[tuple[str, Path]],
    final_report_path: Path,
    protocol_path: Path,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text())
    expected = protocol["official_train"]
    candidate_by_id: dict[str, dict[str, object]] = {}
    source_records = []
    expected_fold_ids = {
        str(value["fold_id"]) for value in protocol["development"]["folds"]
    }
    if {fold_id for fold_id, _ in candidate_reports} != expected_fold_ids:
        raise ValueError("candidate reports do not cover the declared OOF folds")
    for fold_id, path in candidate_reports:
        report = json.loads(path.read_text())
        rows = list(report.get("rows", []))
        for row in rows:
            image_id = str(row["image_id"])
            if image_id in candidate_by_id:
                raise ValueError(f"duplicate OOF candidate row: {image_id}")
            candidate_by_id[image_id] = row
        source_records.append({
            "fold_id": fold_id,
            "path": str(path),
            "sha256": file_sha256(path),
            "query_count": len(rows),
        })
    final_report = json.loads(final_report_path.read_text())
    if str(final_report.get("postselection_evidence_contract", "")) != (
        "proposal_to_pruning_to_exact_ranking_to_refinement_to_winner_v1"
    ):
        raise ValueError("final report lacks the complete post-selection contract")
    final_rows = list(final_report.get("rows", []))
    final_by_id = {str(row["image_id"]): row for row in final_rows}
    if len(final_by_id) != len(final_rows):
        raise ValueError("duplicate final post-selection row")
    candidate_ids = sorted(candidate_by_id)
    if set(candidate_by_id) != set(final_by_id):
        raise ValueError("candidate and final reports contain different query IDs")
    if (
        len(candidate_ids) != int(expected["count"])
        or ordered_id_sha256(candidate_ids) != str(expected["image_ids_sha256"])
    ):
        raise ValueError("failure taxonomy does not cover official train exactly")

    rows = []
    for image_id in candidate_ids:
        candidate = candidate_by_id[image_id]
        final = final_by_id[image_id]
        taxonomy = {}
        for name, (translation, rotation, field) in THRESHOLDS.items():
            classification, stages = _classify(
                candidate,
                final,
                translation_m=translation,
                rotation_deg=rotation,
                count_field=field,
            )
            taxonomy[name] = {"class": classification, "stage_basin": stages}
        rows.append({
            "image_id": image_id,
            "trajectory_id": image_id.split("/", 1)[0],
            "final_translation_m": float(final["final_translation_m"]),
            "final_rotation_deg": float(final["final_rotation_deg"]),
            "taxonomy": taxonomy,
        })
    trajectories = sorted({str(row["trajectory_id"]) for row in rows})
    return {
        "artifact_type": "goal_maplet_oof_failure_taxonomy_v1",
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "candidate_reports": source_records,
        "final_postselection_report": str(final_report_path),
        "final_postselection_report_sha256": file_sha256(final_report_path),
        "oracle_only_attribution": True,
        "selection_uses_ground_truth": False,
        "attribution_uses_ground_truth_after_selection": True,
        "class_definitions": {
            "G": "no correct basin among all generated raw states",
            "S": "a raw correct basin is removed before the exact pool",
            "V": "the exact pool contains a correct basin but exact ranking/NMS removes it",
            "R": "a retained correct basin is not the final successful selected pose",
        },
        "integrity": {
            "query_count": len(rows),
            "image_ids_sha256": ordered_id_sha256(candidate_ids),
        },
        "summary": {
            threshold: _summarize(rows, threshold) for threshold in THRESHOLDS
        },
        "per_trajectory": {
            trajectory: {
                threshold: _summarize(
                    [row for row in rows if row["trajectory_id"] == trajectory],
                    threshold,
                )
                for threshold in THRESHOLDS
            }
            for trajectory in trajectories
        },
        "rows": rows,
    }


def _parse_fold_reports(values: Sequence[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        fold_id, separator, path = value.partition("=")
        if not separator or not fold_id or not path:
            raise ValueError("candidate reports must use foldN=/path/report.json")
        result.append((fold_id, Path(path)))
    if len({fold_id for fold_id, _ in result}) != len(result):
        raise ValueError("duplicate candidate fold report")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--candidate_reports", nargs="+", required=True)
    parser.add_argument("--final_postselection_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite failure-taxonomy report")
    payload = build_failure_taxonomy(
        _parse_fold_reports(args.candidate_reports),
        Path(args.final_postselection_report),
        Path(args.protocol),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact_type": payload["artifact_type"],
        "integrity": payload["integrity"],
        "summary": payload["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
