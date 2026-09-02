"""Evaluate a frozen plane ranking against a separately opened post-label report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--postlabel_authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum_purity", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite post-label plane ranking evaluation")
    ranking = json.loads(args.ranking.read_text())
    labels = json.loads(args.postlabel_authority.read_text())
    label_rows = {str(row["image"]): row for row in labels["rows"]}
    records = []
    for query in ranking["rows"]:
        label_query = label_rows[str(query["image"])]
        if len(query["regions"]) != len(label_query["regions"]):
            raise ValueError("ranking and label region inventories differ")
        for frozen, label in zip(query["regions"], label_query["regions"]):
            if int(frozen["region"]) != int(label["region"]):
                raise ValueError("ranking and label region rows differ")
            gt = int(label.get("gt_plane", -1))
            purity = float(label.get("gt_purity", 0.0))
            if gt < 0 or purity < args.minimum_purity:
                continue
            top = [int(row) for row in frozen["top10"]]
            rank = top.index(gt) + 1 if gt in top else -1
            records.append((int(label["pixels"]), rank))
    weights = np.asarray([row[0] for row in records], np.float64)
    ranks = np.asarray([row[1] for row in records], np.int64)
    report = {
        "artifact_type": "goal_maplet_frozen_plane_ranking_postlabel_evaluation_v1",
        "ranking_file_sha256": file_sha256(args.ranking),
        "postlabel_authority_file_sha256": file_sha256(args.postlabel_authority),
        "evaluated_region_count": len(records),
        "minimum_gt_purity": float(args.minimum_purity),
        "gt_opened_after_ranking_frozen": True,
        "weighted_recall_at_1": float(np.average(ranks == 1, weights=weights)) if len(ranks) else 0.0,
        "weighted_recall_at_5": float(np.average((ranks > 0) & (ranks <= 5), weights=weights)) if len(ranks) else 0.0,
        "weighted_recall_at_10": float(np.average(ranks > 0, weights=weights)) if len(ranks) else 0.0,
        "production_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
