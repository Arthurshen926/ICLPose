"""Report post-selection OOF success across fixed acquisition-extrapolation strata."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


STRATA = (
    "center_distance_stratum", "view_direction_stratum",
    "height_offset_stratum", "outside_mapping_view_manifold_proxy",
)


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


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    count = len(rows)
    strict = sum(
        float(row["final_translation_m"]) <= 0.5
        and float(row["final_rotation_deg"]) <= 5.0 for row in rows
    )
    loose = sum(
        float(row["final_translation_m"]) <= 1.0
        and float(row["final_rotation_deg"]) <= 10.0 for row in rows
    )
    catastrophic = sum(
        float(row["final_translation_m"]) > 2.0
        or float(row["final_rotation_deg"]) > 20.0 for row in rows
    )
    def rate(value: int) -> dict[str, object]:
        return {
            "count": int(value), "rate": float(value / count) if count else 0.0,
            "wilson95": _wilson(value, count),
        }
    return {
        "query_count": count,
        "strict_success": rate(strict),
        "loose_success": rate(loose),
        "catastrophic_failure": rate(catastrophic),
    }


def evaluate_strata(strata_path: Path, postselection_path: Path) -> dict[str, object]:
    strata = json.loads(strata_path.read_text())
    report = json.loads(postselection_path.read_text())
    if strata.get("artifact_type") != (
        "goal_maplet_oof_acquisition_extrapolation_strata_v1"
    ):
        raise ValueError("unsupported acquisition strata")
    if str(report.get("postselection_evidence_contract", "")) != (
        "proposal_to_pruning_to_exact_ranking_to_refinement_to_winner_v1"
    ):
        raise ValueError("post-selection report lacks the deployed evidence contract")
    strata_by_id = {str(row["image_id"]): row for row in strata["rows"]}
    result_rows = []
    for row in report.get("rows", []):
        image_id = str(row["image_id"])
        if image_id not in strata_by_id:
            raise ValueError(f"post-selection query has no acquisition stratum: {image_id}")
        result_rows.append({**row, **{
            key: strata_by_id[image_id][key] for key in STRATA
        }})
    image_ids = [str(row["image_id"]) for row in result_rows]
    if (
        len(image_ids) != len(set(image_ids))
        or len(image_ids) != int(strata["query_count"])
        or ordered_id_sha256(image_ids) != str(strata["image_ids_sha256"])
    ):
        raise ValueError("stratified evaluation does not cover OOF queries once")
    return {
        "artifact_type": "goal_maplet_oof_extrapolation_stratified_evaluation_v1",
        "acquisition_strata": str(strata_path),
        "acquisition_strata_sha256": file_sha256(strata_path),
        "protocol_sha256": str(strata["protocol_sha256"]),
        "postselection_report": str(postselection_path),
        "postselection_report_sha256": file_sha256(postselection_path),
        "selection_uses_strata_or_ground_truth": False,
        "integrity": {
            "query_count": len(result_rows),
            "image_ids_sha256": ordered_id_sha256(image_ids),
        },
        "summary": _summary(result_rows),
        "by_stratum": {
            field: {
                str(value): _summary([
                    row for row in result_rows if row[field] == value
                ])
                for value in sorted({row[field] for row in result_rows}, key=str)
            }
            for field in STRATA
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strata", required=True)
    parser.add_argument("--postselection_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite extrapolation evaluation")
    payload = evaluate_strata(Path(args.strata), Path(args.postselection_report))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
