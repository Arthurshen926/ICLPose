"""Pretrain the HCD codec on extracted RADIO features (no 3DGS involved).

This standalone script trains the encoder-decoder on 2D feature maps so that
the compact representation preserves as much RADIO information as possible
before any 3DGS distillation begins.

Usage:
    python radio_gs/scripts/train_codec.py \
        --feature_dir output/radio_features/room_0/backbone/ \
        --output_dir output/radio_gs/codec_pretrain/ \
        --epochs 50 --batch_size 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radio_gs.models.hcd_codec import HCDCodec


class FeatureMapDataset(Dataset):
    """Load pre-extracted RADIO feature .pt files for codec pretraining."""

    def __init__(self, feature_dir: str, max_files: int | None = None):
        self.paths: List[Path] = sorted(
            Path(feature_dir).glob("*.pt"),
            key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[-1].isdigit() else 0,
        )
        if max_files:
            self.paths = self.paths[:max_files]
        assert len(self.paths) > 0, f"No .pt files in {feature_dir}"
        print(f"[CodecDataset] {len(self.paths)} feature maps from {feature_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        feat = torch.load(self.paths[idx], map_location="cpu")
        if feat.dim() == 4:
            feat = feat.squeeze(0)
        return feat.float()


def train_codec(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    codec = HCDCodec(
        input_dim=args.input_dim,
        bottleneck_dim=args.bottleneck_dim,
        dual_stream=args.dual_stream,
    ).to(device)

    dataset = FeatureMapDataset(args.feature_dir, max_files=args.max_files)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    optimizer = optim.AdamW(codec.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_cos = -1.0

    for epoch in range(1, args.epochs + 1):
        # Train
        codec.train()
        train_loss, train_cos, n = 0.0, 0.0, 0
        for batch in tqdm(train_loader, desc=f"Train E{epoch:03d}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()

            recon = codec(batch)
            losses = codec.compute_reconstruction_loss(batch, recon)
            loss = losses["total"]

            loss.backward()
            nn.utils.clip_grad_norm_(codec.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                cos = F.cosine_similarity(recon.flatten(2), batch.flatten(2), dim=1).mean()

            train_loss += loss.item()
            train_cos += cos.item()
            n += 1

        scheduler.step()
        avg_loss = train_loss / max(n, 1)
        avg_cos = train_cos / max(n, 1)

        # Validate
        codec.eval()
        val_cos, val_psnr, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = codec(batch)
                cos = F.cosine_similarity(recon.flatten(2), batch.flatten(2), dim=1).mean()
                mse = F.mse_loss(recon, batch)
                psnr = -10.0 * np.log10(mse.item() + 1e-8)
                val_cos += cos.item()
                val_psnr += psnr
                nv += 1

        val_cos /= max(nv, 1)
        val_psnr /= max(nv, 1)

        is_best = val_cos > best_val_cos
        if is_best:
            best_val_cos = val_cos
        marker = " ★" if is_best else ""

        print(f"[E{epoch:03d}] train_loss={avg_loss:.4f} train_cos={avg_cos:.4f} | "
              f"val_cos={val_cos:.4f} val_psnr={val_psnr:.1f}dB{marker}")

        # Save
        state = {
            "epoch": epoch,
            "codec_state_dict": codec.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_cosine": val_cos,
            "val_psnr": val_psnr,
        }
        torch.save(state, os.path.join(args.output_dir, "latest.pth"))
        if is_best:
            torch.save(state, os.path.join(args.output_dir, "best.pth"))

    print(f"\nDone! Best val cosine: {best_val_cos:.4f}")
    print(f"Checkpoints: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pretrain HCD codec on RADIO features")
    parser.add_argument("--feature_dir", required=True, help="Directory with .pt feature files")
    parser.add_argument("--output_dir", default="output/radio_gs/codec_pretrain/")
    parser.add_argument("--input_dim", type=int, default=1280)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--dual_stream", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_codec(args)


if __name__ == "__main__":
    main()
