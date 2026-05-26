"""Calibrate a two-score fusion weight on a query split and evaluate held out queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.score_fusion import (
    calibrate_two_score_fusion,
    rows_from_json_payload,
    rows_to_json_payload,
    slice_score_rows_by_query_ids,
)
from feature_extract.vfm.score_table import group_rows_by_query


def _load_rows(path: Path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"score rows file must be a non-empty JSON list: {path}")
    return rows_from_json_payload(payload)


def _query_set_digest(query_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(query_id) for query_id in query_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_query_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"query-id file must be a non-empty JSON list: {path}")
    return sorted(str(item) for item in payload)


def _validate_explicit_query_ids(
    calibration: Sequence[str],
    evaluation: Sequence[str],
    available: set[str],
) -> None:
    if not calibration or not evaluation:
        raise ValueError("explicit calibration and evaluation query ids must be non-empty")
    overlap = set(calibration).intersection(evaluation)
    if overlap:
        raise ValueError(f"explicit split overlaps at query {sorted(overlap)[0]!r}")
    unknown = (set(calibration) | set(evaluation)) - available
    if unknown:
        raise ValueError(f"explicit split contains unknown query {sorted(unknown)[0]!r}")


def _split_query_ids(
    query_ids: Sequence[str],
    calibration_fraction: float,
    split_seed: int,
    calibration_prefixes: Sequence[str],
    evaluation_prefixes: Sequence[str],
) -> tuple[list[str], list[str], str]:
    all_queries = sorted(str(query_id) for query_id in query_ids)
    if calibration_prefixes or evaluation_prefixes:
        calibration = [
            query_id for query_id in all_queries if any(query_id.startswith(prefix) for prefix in calibration_prefixes)
        ]
        evaluation = [
            query_id for query_id in all_queries if any(query_id.startswith(prefix) for prefix in evaluation_prefixes)
        ]
        if not calibration or not evaluation:
            raise ValueError("prefix split produced an empty calibration or evaluation set")
        overlap = set(calibration).intersection(evaluation)
        if overlap:
            raise ValueError(f"prefix split overlaps at query {sorted(overlap)[0]!r}")
        return calibration, evaluation, "query_prefix_holdout"

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    shuffled = list(all_queries)
    random.Random(split_seed).shuffle(shuffled)
    split = int(round(len(shuffled) * float(calibration_fraction)))
    split = min(max(split, 1), len(shuffled) - 1)
    return sorted(shuffled[:split]), sorted(shuffled[split:]), "query_random_holdout"


def _report_to_dict(report):
    return report.to_dict()


def _alpha_reports_to_dict(alpha_reports):
    return {
        f"{alpha:.6f}": {
            "calibration": _report_to_dict(item.calibration),
            "evaluation": _report_to_dict(item.evaluation),
        }
        for alpha, item in sorted(alpha_reports.items())
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate a two-score fusion alpha on held-out query splits")
    parser.add_argument("--primary_rows", required=True)
    parser.add_argument("--secondary_rows", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--metric", default="pred_cost_m")
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--calibration_prefix", action="append", default=[])
    parser.add_argument("--evaluation_prefix", action="append", default=[])
    parser.add_argument("--calibration_queries")
    parser.add_argument("--evaluation_queries")
    parser.add_argument("--normalization", default="zscore", choices=("none", "zscore", "minmax", "rank_percentile"))
    parser.add_argument("--alignment", default="candidate_id", choices=("candidate_id", "query_rank", "init_lattice_id"))
    parser.add_argument("--method_prefix", required=True)
    parser.add_argument("--output_rows", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--output_split_queries")
    args = parser.parse_args(argv)

    primary_path = Path(args.primary_rows)
    secondary_path = Path(args.secondary_rows)
    primary_rows = _load_rows(primary_path)
    secondary_rows = _load_rows(secondary_path)
    primary_queries = set(group_rows_by_query(primary_rows))
    secondary_queries = set(group_rows_by_query(secondary_rows))
    if primary_queries != secondary_queries:
        raise ValueError("primary and secondary query sets must match")
    if bool(args.calibration_queries) != bool(args.evaluation_queries):
        raise ValueError("--calibration_queries and --evaluation_queries must be provided together")
    if args.calibration_queries:
        if args.calibration_prefix or args.evaluation_prefix:
            raise ValueError("explicit query files cannot be combined with prefix splits")
        calibration_ids = _load_query_ids(Path(args.calibration_queries))
        evaluation_ids = _load_query_ids(Path(args.evaluation_queries))
        _validate_explicit_query_ids(calibration_ids, evaluation_ids, set(primary_queries))
        split_protocol = "query_explicit_holdout"
    else:
        calibration_ids, evaluation_ids, split_protocol = _split_query_ids(
            sorted(primary_queries),
            calibration_fraction=args.calibration_fraction,
            split_seed=args.split_seed,
            calibration_prefixes=args.calibration_prefix,
            evaluation_prefixes=args.evaluation_prefix,
        )
    result = calibrate_two_score_fusion(
        primary_rows,
        secondary_rows,
        alphas=args.alphas,
        metric=args.metric,
        calibration_query_ids=calibration_ids,
        evaluation_query_ids=evaluation_ids,
        method_prefix=args.method_prefix,
        normalization=args.normalization,
        alignment=args.alignment,
    )

    output_rows = Path(args.output_rows)
    output_rows.parent.mkdir(parents=True, exist_ok=True)
    evaluation_rows = slice_score_rows_by_query_ids(result.selected_rows, evaluation_ids)
    output_rows.write_text(json.dumps(rows_to_json_payload(evaluation_rows), indent=2, sort_keys=True) + "\n")

    report = _report_to_dict(result.evaluation_report)
    report["selected_alpha"] = float(result.selected_alpha)
    report["calibration_report"] = _report_to_dict(result.calibration_report)
    report["alpha_reports"] = _alpha_reports_to_dict(result.alpha_reports)
    report["split"] = {
        "protocol": split_protocol,
        "split_seed": int(args.split_seed),
        "calibration_fraction": float(args.calibration_fraction),
        "calibration_query_count": len(calibration_ids),
        "evaluation_query_count": len(evaluation_ids),
        "calibration_query_sha256": _query_set_digest(calibration_ids),
        "evaluation_query_sha256": _query_set_digest(evaluation_ids),
        "calibration_prefixes": list(args.calibration_prefix),
        "evaluation_prefixes": list(args.evaluation_prefix),
        "calibration_queries_path": args.calibration_queries,
        "evaluation_queries_path": args.evaluation_queries,
    }
    report["selection"] = {
        "selected_on": "calibration",
        "selection_metric": args.metric,
        "alpha_grid": [float(alpha) for alpha in args.alphas],
        "selected_alpha": float(result.selected_alpha),
    }
    report["inputs"] = {
        "alignment": args.alignment,
        "normalization": args.normalization,
        "primary_rows": {
            "path": str(primary_path),
            "sha256": file_sha256_short(primary_path),
        },
        "secondary_rows": {
            "path": str(secondary_path),
            "sha256": file_sha256_short(secondary_path),
        },
    }
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_split_queries:
        split_payload = {
            "protocol": split_protocol,
            "calibration_query_ids": sorted(calibration_ids),
            "evaluation_query_ids": sorted(evaluation_ids),
            "calibration_query_sha256": _query_set_digest(calibration_ids),
            "evaluation_query_sha256": _query_set_digest(evaluation_ids),
        }
        output_split = Path(args.output_split_queries)
        output_split.parent.mkdir(parents=True, exist_ok=True)
        output_split.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
