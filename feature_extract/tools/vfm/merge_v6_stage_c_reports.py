"""Merge disjoint V6 Stage-C replay shards without using GT for ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--score_calibration", default="")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_score(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    return float(value) if value is not None and np.isfinite(value) else -np.inf


def _rank_key(row: Mapping[str, object]) -> tuple[float, float, float, int]:
    return (
        -_finite_score(row, "pose_mode_log_score"),
        -_finite_score(row, "atlas_score"),
        -_finite_score(row, "coarse_score"),
        int(row.get("replay_pool_index", -1)),
    )


def _percentile(values: Sequence[float], q: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, q)) if finite.size else None


def _aggregate(query_results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    solved = [value for value in query_results if value.get("rows")]
    top = [value["rows"][0] for value in solved]
    oracle = [
        min(
            value["rows"],
            key=lambda row: (
                float(row["final_translation_m"]) / 0.30
                + float(row["final_rotation_deg"]) / 3.0
            ),
        )
        for value in solved
    ]

    def recall(rows: Sequence[Mapping[str, object]]) -> float:
        return (
            float(
                sum(
                    float(row["final_translation_m"]) <= 0.30
                    and float(row["final_rotation_deg"]) <= 3.0
                    for row in rows
                )
                / len(query_results)
            )
            if query_results
            else 0.0
        )

    return {
        "query_count": len(query_results),
        "solved_fraction": (
            float(len(solved) / len(query_results)) if query_results else 0.0
        ),
        "top1_recall_30cm_3deg": recall(top),
        "top1_translation_median_m": _percentile(
            [float(row["final_translation_m"]) for row in top], 50.0
        ),
        "top1_translation_p90_m": _percentile(
            [float(row["final_translation_m"]) for row in top], 90.0
        ),
        "top1_rotation_median_deg": _percentile(
            [float(row["final_rotation_deg"]) for row in top], 50.0
        ),
        "top1_rotation_p90_deg": _percentile(
            [float(row["final_rotation_deg"]) for row in top], 90.0
        ),
        "candidate_oracle_recall_30cm_3deg": recall(oracle),
        "candidate_oracle_translation_median_m": _percentile(
            [float(row["final_translation_m"]) for row in oracle], 50.0
        ),
        "candidate_oracle_rotation_median_deg": _percentile(
            [float(row["final_rotation_deg"]) for row in oracle], 50.0
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = [Path(value) for value in args.input_json]
    reports = [json.loads(path.read_text()) for path in paths]
    if any(report.get("stage") != "v6_stage_c_feature_atlas_replay" for report in reports):
        raise ValueError("every input must be a V6 Stage-C replay report")

    score_calibration = None
    if str(args.score_calibration):
        from feature_extract.vfm.localization_v6.stage_c_score_calibration import (
            StageCScoreCalibration,
        )

        score_calibration = StageCScoreCalibration.load_json(
            Path(args.score_calibration)
        )
    grouped: dict[str, list[tuple[Path, Mapping[str, object]]]] = {}
    for path, report in zip(paths, reports):
        grouped.setdefault(str(report["image_id"]), []).append((path, report))

    query_results = []
    shard_rows = []
    for image_id, values in sorted(grouped.items()):
        first = values[0][1]
        if score_calibration is not None:
            score_calibration.validate_report_lineage(first)
        invariant_keys = (
            "radio_atlas_sha256",
            "region_chart_index",
            "frame_spatial_projection_checkpoint",
            "source_report_sha256s",
            "source_pool_mode",
            "pool_size_reconstructed",
            "base_stride",
            "rounds",
            "render_charts",
            "refinement_charts",
            "maximum_translation_updates",
            "alike_detector_matchability",
            "query_gt_pose_component_diagnostic",
        )
        for _path, report in values[1:]:
            if any(report.get(key) != first.get(key) for key in invariant_keys):
                raise ValueError(f"Stage-C shard contract differs for {image_id}")
        evaluated = []
        rows = []
        for path, report in values:
            local_evaluated = [int(value) for value in report.get("evaluated_pool_indices", ())]
            if set(evaluated).intersection(local_evaluated):
                raise ValueError(f"Stage-C shards overlap for {image_id}")
            evaluated.extend(local_evaluated)
            rows.extend(dict(value) for value in report.get("rows", ()))
            shard_rows.append(
                {
                    "image_id": image_id,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "evaluated_candidate_count": len(local_evaluated),
                    "broad_screen_input_count": int(
                        report.get("stage_c_audit", {}).get(
                            "broad_screen_input_count", 0
                        )
                    ),
                    "broad_screen_selected_count": int(
                        report.get("stage_c_audit", {}).get(
                            "broad_screen_selected_count", 0
                        )
                    ),
                    "refined_candidate_count": len(report.get("rows", ())),
                }
            )
        pool_count = int(first["pool_size_reconstructed"])
        evaluated_set = set(evaluated)
        complete = evaluated_set == set(range(pool_count))
        if len(values) > 1 and not complete:
            raise ValueError(
                f"merged Stage-C shards do not cover the complete pool for {image_id}"
            )
        replay_indices = [int(row["replay_pool_index"]) for row in rows]
        if len(replay_indices) != len(set(replay_indices)):
            raise ValueError(f"duplicate refined Stage-C candidate for {image_id}")
        rows = (
            score_calibration.apply(rows)
            if score_calibration is not None
            else sorted(rows, key=_rank_key)
        )
        query_results.append(
            {
                "image_id": image_id,
                "complete_pool_coverage": bool(complete),
                "evaluated_candidate_count": len(evaluated_set),
                "pool_size_reconstructed": pool_count,
                "rows": rows,
            }
        )

    payload = {
        "stage": "v6_stage_c_feature_atlas_replay_merge",
        "ranking_uses_ground_truth": False,
        "oracle_metrics_are_diagnostic_only": True,
        "input_reports": shard_rows,
        "query_results": query_results,
        "summary": _aggregate(query_results),
        "score_calibration": str(args.score_calibration),
        "score_calibration_sha256": (
            _sha256(Path(args.score_calibration))
            if str(args.score_calibration)
            else None
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(output)
    print(json.dumps({"output": str(output), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
