"""Fit a held-out linear scorer over rendered-map evidence diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.evidence_linear_scorer import fit_heldout_evidence_linear_scorer
from feature_extract.vfm.score_fusion import rows_from_json_payload, rows_to_json_payload
from feature_extract.vfm.score_table import group_rows_by_query


DEFAULT_FEATURES = (
    "baseline_score",
    "evidence_score",
    "mean_similarity",
    "inlier_fraction",
    "match_count_log1p",
    "visibility_fraction",
    "risk",
    "empty_evidence",
)


def _load_rows(path: Path):
    payload = json.loads(path.read_text())
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

    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    shuffled = list(all_queries)
    random.Random(split_seed).shuffle(shuffled)
    split = int(round(len(shuffled) * float(calibration_fraction)))
    split = min(max(split, 1), len(shuffled) - 1)
    return sorted(shuffled[:split]), sorted(shuffled[split:]), "query_random_holdout"


def _report_to_dict(report):
    return report.to_dict()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fit held-out linear scorer for rendered-map evidence rows")
    parser.add_argument("--baseline_rows", required=True)
    parser.add_argument("--evidence_rows", required=True)
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument(
        "--target",
        default="negative_query_zscore_cost",
        choices=(
            "negative_cost",
            "negative_query_centered_cost",
            "negative_query_zscore_cost",
            "negative_query_rank_cost",
        ),
    )
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--calibration_prefix", action="append", default=[])
    parser.add_argument("--evaluation_prefix", action="append", default=[])
    parser.add_argument("--calibration_queries")
    parser.add_argument("--evaluation_queries")
    parser.add_argument("--alignment", default="candidate_id", choices=("candidate_id", "query_rank", "init_lattice_id"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--output_rows", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--output_split_queries")
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline_rows)
    evidence_path = Path(args.evidence_rows)
    baseline_rows = _load_rows(baseline_path)
    evidence_rows = _load_rows(evidence_path)
    baseline_queries = set(group_rows_by_query(baseline_rows))
    evidence_queries = set(group_rows_by_query(evidence_rows))
    if baseline_queries != evidence_queries:
        raise ValueError("baseline and evidence query sets must match")

    if bool(args.calibration_queries) != bool(args.evaluation_queries):
        raise ValueError("--calibration_queries and --evaluation_queries must be provided together")
    if args.calibration_queries:
        if args.calibration_prefix or args.evaluation_prefix:
            raise ValueError("explicit query files cannot be combined with prefix splits")
        calibration_ids = _load_query_ids(Path(args.calibration_queries))
        evaluation_ids = _load_query_ids(Path(args.evaluation_queries))
        _validate_explicit_query_ids(calibration_ids, evaluation_ids, set(baseline_queries))
        split_protocol = "query_explicit_holdout"
    else:
        calibration_ids, evaluation_ids, split_protocol = _split_query_ids(
            sorted(baseline_queries),
            calibration_fraction=args.calibration_fraction,
            split_seed=args.split_seed,
            calibration_prefixes=args.calibration_prefix,
            evaluation_prefixes=args.evaluation_prefix,
        )

    result = fit_heldout_evidence_linear_scorer(
        baseline_rows,
        evidence_rows,
        calibration_query_ids=calibration_ids,
        evaluation_query_ids=evaluation_ids,
        feature_names=args.features,
        method=args.method,
        l2=args.l2,
        alignment=args.alignment,
        target=args.target,
    )

    output_rows = Path(args.output_rows)
    output_rows.parent.mkdir(parents=True, exist_ok=True)
    output_rows.write_text(json.dumps(rows_to_json_payload(result.evaluation_rows), indent=2, sort_keys=True) + "\n")

    report = _report_to_dict(result.evaluation_report)
    report["calibration_report"] = _report_to_dict(result.calibration_report)
    report["model"] = result.model.to_dict()
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
        "target": args.target,
        "features": list(args.features),
        "l2": float(args.l2),
    }
    report["inputs"] = {
        "alignment": args.alignment,
        "baseline_rows": {
            "path": str(baseline_path),
            "sha256": file_sha256_short(baseline_path),
        },
        "evidence_rows": {
            "path": str(evidence_path),
            "sha256": file_sha256_short(evidence_path),
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
