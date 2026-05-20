#!/usr/bin/env python3
"""Export standardized POFD-FS candidate banks from existing pose-candidate caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.tools.eval_cpr_buckets import build_model_and_data, map_pose_gt_for_batch  # noqa: E402
from feature_extract.train_impl import load_config, move_batch_to_device, pose_error_tensors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--candidate-render-batch-size", type=int, default=4)
    parser.add_argument(
        "--pose-candidate-cache",
        default=None,
        help="Optional override for dataset.<split>_pose_candidate_cache before building the loader.",
    )
    parser.add_argument("--rot-cost-weight", type=float, default=0.1)
    parser.add_argument("--candidate-source", default="controlled")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.pose_candidate_cache:
        cfg.setdefault("dataset", {})[f"{args.split}_pose_candidate_cache"] = str(args.pose_candidate_cache)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    _model, loader, map_renderer = build_model_and_data(cfg, args, device)
    sample_names = []
    pose_gt_rows = []
    candidate_rows = []
    valid_rows = []
    retrieval_score_rows = []
    trans_rows = []
    rot_rows = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if "pose_init_candidates" not in batch:
            raise KeyError("Batch does not contain pose_init_candidates; check dataset pose candidate cache config")
        pose_gt = map_pose_gt_for_batch(map_renderer, batch, device)
        candidates = batch["pose_init_candidates"].float()
        bsz, num_candidates = candidates.shape[:2]
        pose_gt_flat = pose_gt[:, None].expand(-1, num_candidates, -1, -1).reshape(-1, 4, 4)
        _, rot_err_deg, trans_err_m = pose_error_tensors(candidates.reshape(-1, 4, 4), pose_gt_flat)
        trans_err_m = trans_err_m.reshape(bsz, num_candidates)
        rot_err_deg = rot_err_deg.reshape(bsz, num_candidates)
        sample_names.extend([str(name) for name in batch["sample_name"]])
        pose_gt_rows.append(pose_gt.detach().cpu())
        candidate_rows.append(candidates.detach().cpu())
        valid_rows.append(batch.get("candidate_valid_mask", torch.ones(bsz, num_candidates, device=device, dtype=torch.bool)).detach().cpu())
        retrieval_score_rows.append(
            batch.get("retrieval_scores_candidates", torch.zeros(bsz, num_candidates, device=device)).detach().cpu()
        )
        trans_rows.append(trans_err_m.detach().cpu())
        rot_rows.append(rot_err_deg.detach().cpu())
    pose_gt_all = torch.cat(pose_gt_rows, dim=0)
    candidates_all = torch.cat(candidate_rows, dim=0)
    valid_all = torch.cat(valid_rows, dim=0)
    retrieval_scores = torch.cat(retrieval_score_rows, dim=0)
    trans_all = torch.cat(trans_rows, dim=0)
    rot_all = torch.cat(rot_rows, dim=0)
    pose_cost = trans_all + float(args.rot_cost_weight) * torch.deg2rad(rot_all)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        sample_names=np.asarray(sample_names),
        pose_gt=pose_gt_all.numpy().astype(np.float32),
        candidates=candidates_all.numpy().astype(np.float32),
        valid_mask=valid_all.numpy().astype(bool),
        pose_cost_m=pose_cost.numpy().astype(np.float32),
        trans_err_m=trans_all.numpy().astype(np.float32),
        rot_err_deg=rot_all.numpy().astype(np.float32),
        retrieval_scores_candidates=retrieval_scores.numpy().astype(np.float32),
        scene=np.asarray(args.scene or cfg.get("dataset", {}).get("scene", "")),
        candidate_source=np.asarray(args.candidate_source),
        metadata=np.asarray(
            [
                {
                    "config": str(Path(args.config).resolve()),
                    "split": args.split,
                    "rot_cost_weight": float(args.rot_cost_weight),
                    "num_samples": len(sample_names),
                }
            ],
            dtype=object,
        ),
    )
    print(json.dumps({"out": str(out), "num_samples": len(sample_names), "num_candidates": int(candidates_all.shape[1])}))


if __name__ == "__main__":
    main()
