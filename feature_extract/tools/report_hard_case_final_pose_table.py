#!/usr/bin/env python3
"""Report final-pose localization metrics on hard-case query subsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_retrieval_dataset import list_colmap_split_samples  # noqa: E402
from feature_extract.localizability.pose_cache_report import (  # noqa: E402
    build_pose_cache_comparison,
)


def _parse_label_path_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Expected LABEL=PATH, got: {value}")
    return label, path


def hard_case_query_names(path: str | Path) -> list[str]:
    """Return unique sample names from a hard-case JSONL artifact."""

    names: list[str] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row.get("sample_name") or row.get("query_image_name") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


def build_hard_case_final_pose_report(
    *,
    hard_cases: Sequence[tuple[str, str | Path]],
    caches: Sequence[tuple[str, str | Path]],
    gt_poses_by_name: Mapping[str, np.ndarray],
    protocol: str = "hard_case_final_pose_comparison",
) -> dict:
    """Build hard-case x method final-pose metrics from existing pose caches."""

    cases = []
    for case_label, case_path in hard_cases:
        query_names = hard_case_query_names(case_path)
        comparison = build_pose_cache_comparison(
            caches=caches,
            gt_poses_by_name=gt_poses_by_name,
            query_names=query_names,
            protocol=f"{protocol}:{case_label}",
        )
        cases.append(
            {
                "case": str(case_label),
                "hard_case_path": str(case_path),
                "num_case_queries": int(len(query_names)),
                "query_names": query_names,
                "rows": comparison["rows"],
            }
        )
    return {
        "protocol": str(protocol),
        "num_cases": int(len(cases)),
        "num_caches": int(len(caches)),
        "caches": [{"label": str(label), "path": str(path)} for label, path in caches],
        "cases": cases,
    }


def format_hard_case_final_pose_markdown(report: Mapping[str, object]) -> str:
    """Render a compact hard-case final-pose markdown table."""

    lines = [
        "# Hard-Case Final-Pose Report",
        "",
        f"- protocol: `{report.get('protocol')}`",
        "",
        "| case | method | n | rot med deg | trans med mm | trans mean mm | R@1deg/100mm | R@5deg/250mm | solver success |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report.get("cases", []):  # type: ignore[union-attr]
        case_map = dict(case)
        for row in case_map.get("rows", []):
            row_map = dict(row)
            metrics = dict(row_map.get("metrics", {}))
            success = row_map.get("solver_success_frac")
            success_text = "" if success is None else f"{float(success) * 100.0:.1f}%"
            lines.append(
                "| {case} | {method} | {n:d} | {rot:.3f} | {trans_med:.1f} | {trans_mean:.1f} | "
                "{r100:.1f} | {r250:.1f} | {success} |".format(
                    case=str(case_map.get("case", "")),
                    method=str(row_map.get("label", "")),
                    n=int(row_map.get("num_samples", 0)),
                    rot=float(metrics.get("rot_median", float("nan"))),
                    trans_med=float(metrics.get("trans_median", float("nan"))),
                    trans_mean=float(metrics.get("trans_mean", float("nan"))),
                    r100=float(metrics.get("joint_1deg_100mm", float("nan"))),
                    r250=float(metrics.get("joint_5deg_250mm", float("nan"))),
                    success=success_text,
                )
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-case", action="append", required=True, help="Hard-case spec LABEL=PATH. Repeatable.")
    parser.add_argument("--cache", action="append", required=True, help="Pose cache spec LABEL=PATH. Repeatable.")
    parser.add_argument("--colmap-dir", required=True)
    parser.add_argument("--query-split", required=True)
    parser.add_argument("--protocol", default="hard_case_final_pose_comparison")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = list_colmap_split_samples(args.colmap_dir, args.query_split)
    gt_poses_by_name = {
        str(sample["image_name"]): np.asarray(sample["pose_w2c"], dtype=np.float32)
        for sample in samples
    }
    report = build_hard_case_final_pose_report(
        hard_cases=[_parse_label_path_spec(spec) for spec in args.hard_case],
        caches=[_parse_label_path_spec(spec) for spec in args.cache],
        gt_poses_by_name=gt_poses_by_name,
        protocol=str(args.protocol),
    )
    report["colmap_dir"] = str(args.colmap_dir)
    report["query_split"] = str(args.query_split)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(format_hard_case_final_pose_markdown(report), encoding="utf-8")
    print(json.dumps({"num_cases": report["num_cases"], "num_caches": report["num_caches"]}, indent=2))


if __name__ == "__main__":
    main()
