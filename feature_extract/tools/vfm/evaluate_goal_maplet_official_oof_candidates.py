"""Validate and aggregate route-grouped official-train OOF pose candidates."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from feature_extract.vfm.official_oof_protocol import (
    file_sha256,
    ordered_id_sha256,
    parse_cambridge_image_ids,
)


K_VALUES = (1, 2, 4, 8, 16, 32)
MODE_KEY = "actual_parent_actual_child"
STAGE_NAMES = (
    "mapping_view_anchor_beam",
    "retained_seed_heap",
    "structural_equivalence_prescreen",
    "continuous_visibility_chart",
    "full_map_sparse_seed_vfm_likelihood",
    "exact_verification_pool_selection",
    "full_surface_ranked",
    "topn_after_nms",
)


def _wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count <= 0:
        return [float("nan"), float("nan")]
    probability = successes / count
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    radius = z * sqrt(
        probability * (1.0 - probability) / count + z * z / (4.0 * count**2)
    ) / denominator
    return [float(center - radius), float(center + radius)]


def _rate(successes: int, count: int) -> dict[str, object]:
    return {
        "count": int(successes),
        "rate": float(successes / count) if count else float("nan"),
        "wilson95": _wilson(successes, count),
    }


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"median": float("nan"), "p90": float("nan")}
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _stage_summary(rows: Sequence[dict[str, object]], stage_name: str) -> dict[str, object]:
    stages = []
    for row in rows:
        diagnostic = row.get("proposal_diagnostics", {})
        if isinstance(diagnostic, dict) and MODE_KEY in diagnostic:
            diagnostic = diagnostic[MODE_KEY]
        if isinstance(diagnostic, dict) and isinstance(diagnostic.get(stage_name), dict):
            stages.append(diagnostic[stage_name])
    present = len(stages)
    strict = sum(int(value.get("within_0_5m_5deg_count", 0)) > 0 for value in stages)
    loose = sum(int(value.get("within_1m_10deg_count", 0)) > 0 for value in stages)
    result: dict[str, object] = {
        "stage_present_query_count": present,
        "strict_basin_survival": _rate(strict, present),
        "loose_basin_survival": _rate(loose, present),
        "pose_count": _quantiles(float(value.get("pose_count", 0)) for value in stages),
    }
    for suffix in ("0_5m_5deg", "1m_10deg"):
        field = f"unique_pose_basin_count_{suffix}"
        available = [float(value[field]) for value in stages if field in value]
        result[field] = {
            "available_query_count": len(available),
            **_quantiles(available),
        }
    return result


def _row_modes(row: dict[str, object]) -> list[dict[str, object]]:
    details = row.get("mode_details", {})
    if not isinstance(details, dict) or not isinstance(details.get(MODE_KEY), list):
        raise ValueError(f"row {row.get('image_id')} has no {MODE_KEY} mode details")
    modes = list(details[MODE_KEY])
    ranks = [int(value["rank"]) for value in modes]
    if ranks != list(range(1, len(modes) + 1)):
        raise ValueError(f"row {row.get('image_id')} has non-contiguous mode ranks")
    return modes


def _candidate_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    count = len(rows)
    strict_by_k = {value: 0 for value in K_VALUES}
    loose_by_k = {value: 0 for value in K_VALUES}
    top1_translation = []
    top1_rotation = []
    catastrophic = 0
    empty = 0
    retained_opportunity = {"strict": 0, "loose": 0}
    retained_opportunity_captured = {"strict": 0, "loose": 0}
    for row in rows:
        modes = _row_modes(row)
        if not modes:
            empty += 1
            continue
        translation = np.asarray(
            [value["translation_m"] for value in modes], dtype=np.float64
        )
        rotation = np.asarray([value["rotation_deg"] for value in modes], dtype=np.float64)
        top1_translation.append(float(translation[0]))
        top1_rotation.append(float(rotation[0]))
        catastrophic += int(translation[0] > 2.0 or rotation[0] > 20.0)
        diagnostic = row.get("proposal_diagnostics", {})
        if isinstance(diagnostic, dict) and MODE_KEY in diagnostic:
            diagnostic = diagnostic[MODE_KEY]
        retained = diagnostic.get("topn_after_nms", {}) if isinstance(
            diagnostic, dict
        ) else {}
        for name, field, top1_success in (
            (
                "strict", "within_0_5m_5deg_count",
                bool(translation[0] <= 0.5 and rotation[0] <= 5.0),
            ),
            (
                "loose", "within_1m_10deg_count",
                bool(translation[0] <= 1.0 and rotation[0] <= 10.0),
            ),
        ):
            if isinstance(retained, dict) and int(retained.get(field, 0)) > 0:
                retained_opportunity[name] += 1
                retained_opportunity_captured[name] += int(top1_success)
        for top_k in K_VALUES:
            limit = min(top_k, len(modes))
            strict_by_k[top_k] += int(np.any(
                (translation[:limit] <= 0.5) & (rotation[:limit] <= 5.0)
            ))
            loose_by_k[top_k] += int(np.any(
                (translation[:limit] <= 1.0) & (rotation[:limit] <= 10.0)
            ))
    # Empty candidate sets are failures for every operational metric.
    result = {
        "query_count": count,
        "empty_candidate_count": empty,
        "top1_strict_success": _rate(strict_by_k[1], count),
        "top1_loose_success": _rate(loose_by_k[1], count),
        "top1_catastrophic_failure": _rate(catastrophic + empty, count),
        "top1_translation_m": _quantiles(top1_translation),
        "top1_rotation_deg": _quantiles(top1_rotation),
        "per_query_elapsed_sec": _quantiles(
            float(row["elapsed_sec"]) for row in rows if "elapsed_sec" in row
        ),
        "total_reported_query_elapsed_sec": float(sum(
            float(row.get("elapsed_sec", 0.0)) for row in rows
        )),
        "topk_basin_recall": {
            str(top_k): {
                "strict": _rate(strict_by_k[top_k], count),
                "loose": _rate(loose_by_k[top_k], count),
            }
            for top_k in K_VALUES
        },
        "retained_basin_opportunity_capture": {
            name: {
                "available_correct_basin_count": int(retained_opportunity[name]),
                "top1_capture": _rate(
                    retained_opportunity_captured[name], retained_opportunity[name]
                ),
            }
            for name in ("strict", "loose")
        },
        "stage_survival": {
            stage: _stage_summary(rows, stage) for stage in STAGE_NAMES
        },
    }
    return result


def _parse_fold_reports(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        fold_id, separator, path = value.partition("=")
        if not separator or not fold_id or not path:
            raise ValueError("fold reports must use foldN=/path/to/report.json")
        if fold_id in result:
            raise ValueError(f"duplicate fold report: {fold_id}")
        result[fold_id] = Path(path)
    return result


def evaluate_official_oof_candidates(
    protocol_path: Path,
    fold_report_paths: dict[str, Path],
) -> dict[str, object]:
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol.get("artifact_type") != "goal_maplet_official_train_oof_protocol_v1":
        raise ValueError("unsupported official OOF protocol")
    train = protocol["official_train"]
    pose_file = Path(train["pose_file"])
    if file_sha256(pose_file) != str(train["pose_file_sha256"]):
        raise ValueError("official train pose file changed after protocol creation")
    train_ids = parse_cambridge_image_ids(pose_file)
    folds = list(protocol["development"]["folds"])
    expected_fold_ids = {str(value["fold_id"]) for value in folds}
    if set(fold_report_paths) != expected_fold_ids:
        raise ValueError(
            f"fold report mismatch: expected={sorted(expected_fold_ids)}, "
            f"actual={sorted(fold_report_paths)}"
        )

    all_rows = []
    seen_ids = set()
    per_fold = []
    for fold in folds:
        fold_id = str(fold["fold_id"])
        held = set(str(value) for value in fold["held_query_trajectories"])
        expected_ids = {
            value for value in train_ids if value.split("/", 1)[0] in held
        }
        path = fold_report_paths[fold_id]
        report = json.loads(path.read_text())
        rows = list(report.get("rows", []))
        image_ids = [str(value["image_id"]) for value in rows]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"duplicate query image in {fold_id}")
        if set(image_ids) != expected_ids:
            missing = sorted(expected_ids - set(image_ids))[:8]
            extra = sorted(set(image_ids) - expected_ids)[:8]
            raise ValueError(f"{fold_id} query mismatch: missing={missing}, extra={extra}")
        if ordered_id_sha256(image_ids) != str(fold["query_image_ids_sha256"]):
            raise ValueError(f"{fold_id} query hash does not match protocol")
        overlap = seen_ids & set(image_ids)
        if overlap:
            raise ValueError(f"OOF query appears in multiple folds: {sorted(overlap)[:8]}")
        seen_ids.update(image_ids)
        all_rows.extend(rows)
        per_fold.append({
            "fold_id": fold_id,
            "held_query_trajectories": sorted(held),
            "source_report": str(path),
            "source_report_sha256": file_sha256(path),
            "summary": _candidate_summary(rows),
        })
    if seen_ids != set(train_ids):
        raise ValueError("OOF reports do not cover official train exactly once")

    return {
        "artifact_type": "goal_maplet_official_train_oof_candidate_evaluation_v1",
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "selection_uses_ground_truth": False,
        "evaluation_uses_ground_truth": True,
        "thresholds": {
            "strict": "translation<=0.5m and rotation<=5deg",
            "loose": "translation<=1m and rotation<=10deg",
            "catastrophic": "translation>2m or rotation>20deg or empty candidate set",
        },
        "integrity": {
            "official_train_query_count": len(train_ids),
            "oof_query_count": len(all_rows),
            "each_official_train_frame_evaluated_exactly_once": True,
            "oof_image_ids_sha256": ordered_id_sha256(seen_ids),
        },
        "summary": _candidate_summary(all_rows),
        "folds": per_fold,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--fold_reports", required=True, nargs="+")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite OOF evaluation: {output}")
    result = evaluate_official_oof_candidates(
        Path(args.protocol), _parse_fold_reports(args.fold_reports)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact_type": result["artifact_type"],
        "integrity": result["integrity"],
        "summary": result["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
