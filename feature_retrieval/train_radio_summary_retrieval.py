#!/usr/bin/env python3
"""Train a location-aware retrieval embedding from RADIO summary cache."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_retrieval_dataset import list_colmap_split_samples  # noqa: E402
from feature_extract.radio_summary_init_export import extract_radio_summary_descriptors  # noqa: E402
from feature_retrieval.retrieval_v5 import (  # noqa: E402
    ProjectionHead,
    ProjectionHeadWithPose,
    compute_recall,
    evaluate_retrieval,
    geodesic_distance,
    train_contrastive,
)


class SummaryRetrievalData:
    def __init__(self, *, feature_dir: str, colmap_dir: str, split_file: str, device: torch.device):
        samples = list_colmap_split_samples(colmap_dir, split_file)
        used, desc = extract_radio_summary_descriptors(feature_dir, samples)
        poses = torch.stack([torch.as_tensor(sample["pose_w2c"], dtype=torch.float32) for sample in used], dim=0)
        self.features = desc.to(device)
        self.rotations = poses[:, :3, :3].to(device)
        self.translations = self._camera_centers_from_w2c(poses).to(device)
        self.names = [sample["image_name"] for sample in used]
        self.samples = used
        self.N = len(used)
        print(f"[{Path(split_file).name}] Loaded {self.N} samples, features {tuple(self.features.shape)}")

    @staticmethod
    def _camera_centers_from_w2c(poses: torch.Tensor) -> torch.Tensor:
        rot = poses[:, :3, :3]
        trans = poses[:, :3, 3]
        return -torch.bmm(rot.transpose(1, 2), trans.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def evaluate_and_pack(query_data, train_data, query_embed, train_embed):
    results = evaluate_retrieval(
        query_embed,
        query_data.translations,
        query_data.rotations,
        train_embed,
        train_data.translations,
        train_data.rotations,
        topk=(1, 3, 5, 10),
    )
    recalls = compute_recall(results["top1_rot_err"], results["top1_trans_err"])
    return {
        "top1_rot_median": float(results["top1_rot_median"]),
        "top1_trans_median_mm": float(results["top1_trans_median"]),
        "top1_rot_mean": float(results["top1_rot_mean"]),
        "top1_trans_mean_mm": float(results["top1_trans_mean"]),
        "r5_1": float(recalls["R@5°/1m"]["combined"]),
        "r10_2": float(recalls["R@10°/2m"]["combined"]),
        "r15_5": float(recalls["R@15°/5m"]["combined"]),
        "r25_5": float(recalls["R@25°/5m"]["combined"]),
        "top3_oracle_rot_median": float(results["top3_oracle_rot_median"]),
        "top3_oracle_trans_median_mm": float(results["top3_oracle_trans_median"]),
        "top5_oracle_rot_median": float(results["top5_oracle_rot_median"]),
        "top5_oracle_trans_median_mm": float(results["top5_oracle_trans_median"]),
        "top10_oracle_rot_median": float(results["top10_oracle_rot_median"]),
        "top10_oracle_trans_median_mm": float(results["top10_oracle_trans_median"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RADIO-summary location-aware retrieval embedding")
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--loss_fn", choices=["ntxent", "multisim", "ap"], default="ntxent")
    parser.add_argument("--pos_radius", type=float, default=2.0)
    parser.add_argument("--neg_radius", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--with_pose_aux", action="store_true")
    parser.add_argument("--lambda_pose", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data = SummaryRetrievalData(
        feature_dir=args.feature_dir,
        colmap_dir=args.colmap_dir,
        split_file=args.train_split,
        device=device,
    )
    query_data = SummaryRetrievalData(
        feature_dir=args.feature_dir,
        colmap_dir=args.colmap_dir,
        split_file=args.query_split,
        device=device,
    )

    with torch.no_grad():
        raw = evaluate_and_pack(query_data, train_data, query_data.features, train_data.features)
    print(
        "Raw RADIO summary: "
        f"{raw['top1_rot_median']:.2f}deg/{raw['top1_trans_median_mm']:.0f}mm "
        f"R@10/2={raw['r10_2']:.1f}%"
    )

    train_args = SimpleNamespace(**vars(args))
    model = train_contrastive(train_data, query_data, train_args)
    model.eval()
    with torch.no_grad():
        if args.with_pose_aux:
            query_embed, _, _ = model(query_data.features)
            train_embed, _, _ = model(train_data.features)
        else:
            query_embed = model(query_data.features)
            train_embed = model(train_data.features)
        learned = evaluate_and_pack(query_data, train_data, query_embed, train_embed)

    report = {
        "args": vars(args),
        "raw": raw,
        "learned": learned,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "raw": raw,
            "learned": learned,
        },
        out_dir / "model_best.pt",
    )
    print(
        "Learned retrieval: "
        f"{learned['top1_rot_median']:.2f}deg/{learned['top1_trans_median_mm']:.0f}mm "
        f"R@10/2={learned['r10_2']:.1f}% "
        f"top10-oracle={learned['top10_oracle_rot_median']:.2f}deg/"
        f"{learned['top10_oracle_trans_median_mm']:.0f}mm"
    )
    print(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
