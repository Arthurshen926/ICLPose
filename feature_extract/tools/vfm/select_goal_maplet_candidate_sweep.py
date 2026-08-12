"""Select a final candidate budget from full official-train OOF development runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.tools.vfm.freeze_goal_maplet_g23_configuration import (
    _candidate_configuration,
)


def _parse(values: Sequence[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("reports must use tag=/path/report.json")
        result.append((name, Path(path)))
    if len({name for name, _ in result}) != len(result):
        raise ValueError("candidate sweep tags must be unique")
    return result


def _selection_key(report: dict[str, object]) -> tuple[object, ...]:
    summary = report["summary"]
    return (
        int(summary["top1_catastrophic_failure"]["count"]),
        -int(summary["topk_basin_recall"]["32"]["strict"]["count"]),
        -int(summary["topk_basin_recall"]["32"]["loose"]["count"]),
        -int(summary["top1_strict_success"]["count"]),
        -int(summary["top1_loose_success"]["count"]),
        float(summary["top1_translation_m"]["p90"]),
        float(summary["top1_rotation_deg"]["p90"]),
        float(summary["total_reported_query_elapsed_sec"]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite candidate sweep selection")
    protocol_path = Path(args.protocol)
    protocol_sha = file_sha256(protocol_path)
    candidates = []
    for tag, path in _parse(args.reports):
        report = json.loads(path.read_text())
        if (
            report.get("artifact_type")
            != "goal_maplet_official_train_oof_candidate_evaluation_v1"
            or str(report.get("protocol_sha256")) != protocol_sha
        ):
            raise ValueError(f"invalid candidate sweep report: {path}")
        source_paths = [Path(value["source_report"]) for value in report["folds"]]
        source_configurations = [
            _candidate_configuration(json.loads(source.read_text()))
            for source in source_paths
        ]
        if any(
            configuration != source_configurations[0]
            for configuration in source_configurations[1:]
        ):
            raise ValueError(f"candidate configuration differs by fold: {path}")
        candidates.append({
            "tag": tag, "path": str(path), "sha256": file_sha256(path),
            "selection_key": list(_selection_key(report)),
            "candidate_configuration": source_configurations[0],
            "inference_arguments": source_configurations[0]["inference_arguments"],
            "summary": report["summary"],
        })
    recommended = min(candidates, key=lambda value: tuple(value["selection_key"]))
    payload = {
        "artifact_type": "goal_maplet_candidate_oof_development_selection_v1",
        "protocol": str(protocol_path), "protocol_sha256": protocol_sha,
        "metrics_are_development_after_hyperparameter_selection": True,
        "official_test_used_for_selection": False,
        "selection_order": [
            "minimum_top1_catastrophic_count",
            "maximum_strict_basin_recall_at_32",
            "maximum_loose_basin_recall_at_32",
            "maximum_top1_strict_success",
            "maximum_top1_loose_success",
            "minimum_translation_p90", "minimum_rotation_p90",
            "minimum_total_compute",
        ],
        "recommended_tag": recommended["tag"],
        "recommended": recommended, "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
