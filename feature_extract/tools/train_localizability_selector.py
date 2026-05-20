#!/usr/bin/env python3
"""Train a compact POFD-FS selector from pre-rendered query/candidate features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.candidate_bank import candidate_bank_from_npz  # noqa: E402
from feature_extract.localizability.losses import (  # noqa: E402
    basin_bce_loss,
    channel_sparsity_loss,
    online_score_hard_negative_loss,
    pose_distance_soft_rank_loss,
    spatial_utility_entropy_loss,
)
from feature_extract.localizability.metrics import ranking_metrics  # noqa: E402
from feature_extract.localizability.scorer import PoseHypothesisScorer  # noqa: E402
from feature_extract.localizability.selector import LocalizationFeatureSelector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--features-npz", required=True, help="NPZ with query_feature and candidate_feature")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--score-mode", choices=("same_pixel", "local_corr", "pair_matcher_local"), default="local_corr")
    parser.add_argument("--score-radius", type=int, default=12)
    parser.add_argument("--score-temperature", type=float, default=0.05)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--basin-weight", type=float, default=0.5)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--utility-entropy-weight", type=float, default=0.01)
    parser.add_argument("--basin-trans-m", type=float, default=0.25)
    parser.add_argument("--basin-rot-deg", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260518)
    return parser.parse_args()


def _load_features(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=True) as data:
        query = torch.as_tensor(data["query_feature"], dtype=torch.float32)
        render = torch.as_tensor(data["candidate_feature"], dtype=torch.float32)
    if query.ndim != 4 or render.ndim != 5:
        raise ValueError("query_feature must be (B,C,H,W), candidate_feature must be (B,K,C,H,W)")
    return query, render


def _select_rows(tensor: torch.Tensor | None, idx: torch.Tensor):
    return None if tensor is None else tensor[idx]


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = candidate_bank_from_npz(args.bank)
    query_feature, candidate_feature = _load_features(args.features_npz)
    if query_feature.shape[0] != len(bank.sample_names) or candidate_feature.shape[:2] != bank.candidate_pose.shape[:2]:
        raise ValueError("Feature NPZ shapes do not match candidate bank")
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    selector = LocalizationFeatureSelector(
        in_channels=query_feature.shape[1],
        out_channels=args.out_dim,
        group_size=args.group_size,
    ).to(device)
    scorer = PoseHypothesisScorer(mode=args.score_mode, radius=args.score_radius, temperature=args.score_temperature).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=1.0e-5)
    num_samples = query_feature.shape[0]
    log_path = out_dir / "train_log.jsonl"

    def evaluate(step: int, split: str = "eval") -> dict:
        selector.eval()
        rows = []
        with torch.no_grad():
            for start in range(0, num_samples, args.batch_size):
                idx = torch.arange(start, min(num_samples, start + args.batch_size))
                q = query_feature[idx].to(device)
                r = candidate_feature[idx].to(device)
                q_out = selector(q)
                bsz, num_candidates = r.shape[:2]
                r_out = selector(r.reshape(bsz * num_candidates, r.shape[2], r.shape[3], r.shape[4]))
                render_z = r_out["z"].reshape(bsz, num_candidates, args.out_dim, r.shape[3], r.shape[4])
                scores, _aux = scorer(q_out["z"], render_z, query_utility=q_out["utility"])
                metrics = ranking_metrics(
                    scores.cpu(),
                    bank.pose_cost_m[idx],
                    valid_mask=_select_rows(bank.valid_mask, idx),
                    basin_label=bank.basin_label(args.basin_trans_m, args.basin_rot_deg)[idx],
                    topk=(1, 5),
                )
                rows.append({key: value.detach().cpu() for key, value in metrics.items()})
        out = {"step": int(step), "split": split}
        for key in rows[0]:
            out[key] = float(torch.stack([row[key] for row in rows]).mean())
        selector.train()
        return out

    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, int(args.max_steps) + 1):
            selector.train()
            idx = torch.randint(0, num_samples, (int(args.batch_size),))
            q = query_feature[idx].to(device)
            r = candidate_feature[idx].to(device)
            q_out = selector(q)
            bsz, num_candidates = r.shape[:2]
            r_out = selector(r.reshape(bsz * num_candidates, r.shape[2], r.shape[3], r.shape[4]))
            render_z = r_out["z"].reshape(bsz, num_candidates, args.out_dim, r.shape[3], r.shape[4])
            scores, _aux = scorer(q_out["z"], render_z, query_utility=q_out["utility"])
            costs = bank.pose_cost_m[idx].to(device)
            valid = _select_rows(bank.valid_mask, idx)
            valid = valid.to(device) if valid is not None else None
            basin = bank.basin_label(args.basin_trans_m, args.basin_rot_deg)[idx].to(device)
            rank_loss, _rank_metrics = pose_distance_soft_rank_loss(scores, costs, valid_mask=valid)
            hard_loss, _hard_metrics = online_score_hard_negative_loss(scores, costs, valid_mask=valid)
            loss = (
                float(args.rank_weight) * rank_loss
                + float(args.hard_weight) * hard_loss
                + float(args.basin_weight) * basin_bce_loss(scores, basin, valid_mask=valid)
                + float(args.sparsity_weight) * channel_sparsity_loss(q_out["channel_gate"])
                + float(args.utility_entropy_weight) * spatial_utility_entropy_loss(q_out["utility"])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            if step % int(args.eval_every) == 0 or step == int(args.max_steps):
                row = evaluate(step)
                row["loss"] = float(loss.detach().cpu())
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
    torch.save({"selector_state_dict": selector.state_dict(), "args": vars(args)}, out_dir / "checkpoint.pth")


if __name__ == "__main__":
    main()
