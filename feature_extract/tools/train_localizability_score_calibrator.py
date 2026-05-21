#!/usr/bin/env python3
"""Train a lightweight POFD-FS hypothesis score calibrator from candidate tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.losses import (  # noqa: E402
    basin_bce_loss,
    online_score_hard_negative_loss,
    pose_distance_soft_rank_loss,
)
from feature_extract.localizability.failure_replay import (  # noqa: E402
    batch_failure_pairs,
    cached_failure_pair_margin_loss,
    load_mined_score_hard_pairs,
)
from feature_extract.localizability.metrics import ranking_metrics  # noqa: E402
from feature_extract.localizability.score_calibrator import (  # noqa: E402
    HypothesisScoreCalibrator,
    group_candidate_table_rows,
    load_candidate_table_jsonl,
    normalize_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-table", required=True)
    parser.add_argument("--val-table", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feature-keys", default="score,delta_trans_m,delta_rot_deg")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--model-type", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--score-residual-weight", type=float, default=0.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--rank-temperature-m", type=float, default=0.05)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--hard-cost-gap-m", type=float, default=0.12)
    parser.add_argument("--hard-margin", type=float, default=0.08)
    parser.add_argument("--basin-weight", type=float, default=0.5)
    parser.add_argument("--replay-pairs", default=None)
    parser.add_argument("--replay-weight", type=float, default=0.0)
    parser.add_argument("--replay-margin", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def _feature_keys(value: str) -> tuple[str, ...]:
    keys = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not keys:
        raise ValueError("--feature-keys must not be empty")
    return keys


def _eval_scores(scores: torch.Tensor, table) -> dict:
    metrics = ranking_metrics(
        scores,
        table.pose_cost_m,
        valid_mask=table.valid_mask,
        basin_label=table.basin_label,
        topk=(1, 5),
    )
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    keys = _feature_keys(args.feature_keys)
    train = group_candidate_table_rows(load_candidate_table_jsonl(args.train_table), feature_keys=keys)
    val = group_candidate_table_rows(load_candidate_table_jsonl(args.val_table), feature_keys=keys)
    train_x, val_x, mean, std = normalize_features(train.features, val.features)

    score_key_index = keys.index("score") if "score" in keys else 0
    model = HypothesisScoreCalibrator(
        feature_dim=len(keys),
        model_type=args.model_type,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        score_residual_weight=float(args.score_residual_weight),
        score_feature_index=score_key_index,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    raw_train_scores = train.features[..., score_key_index]
    raw_val_scores = val.features[..., score_key_index]
    replay_active = replay_positive = replay_negative = None
    if args.replay_pairs and float(args.replay_weight) > 0.0:
        replay_rows = load_mined_score_hard_pairs(args.replay_pairs)
        replay_active, replay_positive, replay_negative = batch_failure_pairs(train.sample_names, replay_rows)
    best = {"pred_cost_m": float("inf")}
    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        raw_row = {
            "split": "raw",
            "step": 0,
            "train": _eval_scores(raw_train_scores, train),
            "val": _eval_scores(raw_val_scores, val),
        }
        log.write(json.dumps(raw_row) + "\n")
        print(json.dumps(raw_row), flush=True)
        for step in range(1, int(args.max_steps) + 1):
            scores = model(train_x)
            rank_loss, _rank = pose_distance_soft_rank_loss(
                scores,
                train.pose_cost_m,
                valid_mask=train.valid_mask,
                temperature_m=float(args.rank_temperature_m),
            )
            hard_loss, _hard = online_score_hard_negative_loss(
                scores,
                train.pose_cost_m,
                valid_mask=train.valid_mask,
                cost_gap_m=float(args.hard_cost_gap_m),
                margin=float(args.hard_margin),
            )
            replay_loss = scores.sum() * 0.0
            replay_stats = {"failure_pair_active_frac": torch.tensor(0.0)}
            if replay_active is not None:
                replay_loss, replay_stats = cached_failure_pair_margin_loss(
                    scores,
                    positive_idx=replay_positive,
                    negative_idx=replay_negative,
                    active_mask=replay_active,
                    margin=float(args.replay_margin),
                )
            loss = (
                float(args.rank_weight) * rank_loss
                + float(args.hard_weight) * hard_loss
                + float(args.basin_weight) * basin_bce_loss(scores, train.basin_label, valid_mask=train.valid_mask)
                + float(args.replay_weight) * replay_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % int(args.eval_every) == 0 or step == int(args.max_steps):
                with torch.no_grad():
                    train_scores = model(train_x)
                    val_scores = model(val_x)
                    row = {
                        "split": "eval",
                        "step": int(step),
                        "loss": float(loss.detach().cpu()),
                        "replay_loss": float(replay_loss.detach().cpu()),
                        "replay_active_frac": float(replay_stats["failure_pair_active_frac"].detach().cpu()),
                        "train": _eval_scores(train_scores, train),
                        "val": _eval_scores(val_scores, val),
                    }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                if row["val"]["pred_cost_m"] < best["pred_cost_m"]:
                    best = row["val"]
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "feature_keys": keys,
                            "feature_mean": mean,
                            "feature_std": std,
                            "model_type": args.model_type,
                            "hidden_dim": int(args.hidden_dim),
                            "num_layers": int(args.num_layers),
                            "dropout": float(args.dropout),
                            "score_residual_weight": float(args.score_residual_weight),
                            "step": int(step),
                            "metrics": row,
                        },
                        out_dir / "best.pth",
                    )
    summary = {"best_val": best, "feature_keys": list(keys), "raw_val": raw_row["val"], "raw_train": raw_row["train"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
