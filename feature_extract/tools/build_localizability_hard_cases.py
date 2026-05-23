#!/usr/bin/env python3
"""Export protocol hard-case candidate-table subsets for POFD-FS audits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.hard_cases import build_hard_case_masks, summarize_hard_case_masks  # noqa: E402
from feature_extract.localizability.score_calibrator import load_candidate_table_jsonl  # noqa: E402
from feature_extract.tools.eval_localizability_protocol_controls import (  # noqa: E402
    _candidate_table_to_tensors,
    _field_aliases,
)


def _group_rows_by_sample(rows: Sequence[dict]) -> "OrderedDict[str, list[dict]]":
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row["sample_name"]), []).append(row)
    return grouped


def build_hard_case_subset_rows(
    rows: Sequence[dict],
    *,
    retrieval_topk: int = 10,
) -> dict[str, list[dict] | dict[str, float]]:
    """Return full candidate-row groups for each hard-case mask."""

    sample_names, tensors = _candidate_table_to_tensors(rows)
    fields = _field_aliases(tensors)
    valid = tensors["valid"].bool()
    masks = build_hard_case_masks(
        tensors["score"].float(),
        tensors["pose_cost_m"].float(),
        basin_label=tensors["in_basin"].bool(),
        valid_mask=valid,
        pnp_inliers=fields.get("pnp_inliers"),
        delta_trans_m=fields.get("delta_trans_m"),
        delta_rot_deg=fields.get("delta_rot_deg"),
        retrieval_topk=int(retrieval_topk),
    )
    grouped = _group_rows_by_sample(rows)
    out: dict[str, list[dict] | dict[str, float]] = {}
    for case_name, mask in masks.items():
        selected_samples = {sample for sample, keep in zip(sample_names, mask.detach().cpu().bool().tolist()) if keep}
        case_rows: list[dict] = []
        for sample_name, sample_rows in grouped.items():
            if sample_name in selected_samples:
                case_rows.extend(dict(row) for row in sample_rows)
        out[case_name] = case_rows
    out.update({f"{name}_summary": metrics for name, metrics in summarize_hard_case_masks(masks).items()})
    return out


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row)) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--retrieval-topk", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_candidate_table_jsonl(args.candidate_table)
    subsets = build_hard_case_subset_rows(rows, retrieval_topk=int(args.retrieval_topk))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, value in sorted(subsets.items()):
        if name.endswith("_summary"):
            summary[name[: -len("_summary")]] = value
            continue
        _write_jsonl(out_dir / f"{name}.jsonl", value)  # type: ignore[arg-type]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
