#!/usr/bin/env python3
"""Evaluate POFD-FS candidate ranking from standardized banks or saved scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.candidate_bank import candidate_bank_from_npz  # noqa: E402
from feature_extract.localizability.metrics import ranking_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, help="Standardized localizability candidate bank .npz")
    parser.add_argument("--scores-npz", default=None, help="Optional .npz containing candidate scores")
    parser.add_argument("--score-key", default="scores")
    parser.add_argument(
        "--score-mode",
        choices=("input", "negative_cost", "retrieval"),
        default="input",
        help="negative_cost is an oracle sanity check; retrieval reads retrieval_scores_candidates if present.",
    )
    parser.add_argument("--basin-trans-m", type=float, default=0.25)
    parser.add_argument("--basin-rot-deg", type=float, default=10.0)
    parser.add_argument("--topk", default="1,5")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _load_scores(args: argparse.Namespace, bank) -> torch.Tensor:
    if args.score_mode == "negative_cost":
        return -bank.pose_cost_m.float()
    source_path = args.scores_npz or bank.metadata.source_path
    if not source_path:
        raise ValueError("--scores-npz is required when no source path is recorded")
    with np.load(source_path, allow_pickle=True) as data:
        if args.score_mode == "retrieval":
            key = "retrieval_scores_candidates"
        else:
            key = args.score_key
        if key not in data:
            raise KeyError(f"Score key {key!r} not found in {source_path}")
        return torch.as_tensor(data[key], dtype=torch.float32)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = candidate_bank_from_npz(args.bank)
    scores = _load_scores(args, bank)
    basin = bank.basin_label(args.basin_trans_m, args.basin_rot_deg)
    topk = tuple(int(part) for part in str(args.topk).split(",") if part)
    metrics = ranking_metrics(
        scores,
        bank.pose_cost_m,
        valid_mask=bank.valid_mask,
        basin_label=basin,
        topk=topk,
    )
    row = {
        "split": "eval",
        "bank": str(args.bank),
        "score_mode": args.score_mode,
        "score_key": args.score_key,
        "num_samples": len(bank.sample_names),
    }
    row.update({key: float(value.detach().cpu()) for key, value in metrics.items()})
    (out_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
