"""Train RADIO-GS feature field via distillation.

Supports Architecture A (Explicit per-Gaussian) and Architecture B (Hybrid
DCFF-style), with optional HCD codec compression and FeatSharp-3D integration.

Usage:
    python radio_gs/scripts/train_feature_field.py \
        --config radio_gs/configs/replica_explicit.yaml \
        [--resume path/to/checkpoint.pth] \
        [--warmstart path/to/weights.pth]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radio_gs.config import RadioGSConfig, load_config
from radio_gs.losses.distillation_loss import (
    DistillationLoss,
    MultiViewConsistencyLoss,
    TotalVariationLoss,
)
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TB = True
except ImportError:
    _HAS_TB = False


# ===================================================================
# Dataset
# ===================================================================

class SimpleRadioDataset(Dataset):
    """Loads pre-extracted RADIO features + poses for distillation training."""

    def __init__(
        self,
        feature_dir: str,
        pose_file: str,
        depth_dir: Optional[str] = None,
        split: str = "train",
    ):
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.split = split

        # --- discover feature files (backbone/rgb_{idx}.pt) ---------------
        backbone_dir = self.feature_dir / "backbone"
        if not backbone_dir.exists():
            backbone_dir = self.feature_dir  # fallback: features at root
        self.feature_paths: List[Path] = sorted(
            backbone_dir.glob("rgb_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        assert len(self.feature_paths) > 0, (
            f"No feature files found in {backbone_dir}"
        )

        # --- load poses (traj_w_c.txt: one 4x4 c2w per line) --------------
        self.poses_w2c = self._load_poses(pose_file)
        assert len(self.poses_w2c) >= len(self.feature_paths), (
            f"Fewer poses ({len(self.poses_w2c)}) than features "
            f"({len(self.feature_paths)})"
        )

        # --- optional depth maps ------------------------------------------
        if self.depth_dir is not None and self.depth_dir.exists():
            self.depth_paths: Optional[List[Path]] = sorted(
                self.depth_dir.glob("*.png"),
                key=lambda p: int(p.stem.split("_")[-1])
                if p.stem.split("_")[-1].isdigit()
                else 0,
            )
        else:
            self.depth_paths = None

    # ------------------------------------------------------------------
    def _load_poses(self, pose_file: str) -> np.ndarray:
        """Load poses from traj_w_c.txt → convert c2w to w2c."""
        raw = np.loadtxt(pose_file).reshape(-1, 4, 4).astype(np.float32)
        # Invert c2w → w2c
        w2c = np.linalg.inv(raw)
        return w2c

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        radio_feat = torch.load(
            self.feature_paths[idx], map_location="cpu"
        )  # [C, Hp, Wp]
        if radio_feat.dim() == 4:
            radio_feat = radio_feat.squeeze(0)

        pose_w2c = torch.from_numpy(self.poses_w2c[idx])  # [4, 4]

        depth: Optional[torch.Tensor] = None
        if self.depth_paths is not None and idx < len(self.depth_paths):
            import cv2

            d = cv2.imread(str(self.depth_paths[idx]), cv2.IMREAD_UNCHANGED)
            if d is not None:
                depth = torch.from_numpy(d.astype(np.float32) / 1000.0)

        out: Dict[str, torch.Tensor] = {
            "radio_features": radio_feat,
            "pose_w2c": pose_w2c,
            "frame_idx": torch.tensor(idx, dtype=torch.long),
        }
        if depth is not None:
            out["depth"] = depth
        return out


# ===================================================================
# Trainer
# ===================================================================

class RadioGSTrainer:
    """Training loop for RADIO-GS feature field distillation."""

    def __init__(self, config: RadioGSConfig) -> None:
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training mode: "latent" trains in 64d space with frozen decoder,
        # "decoded" (default/legacy) trains through decoder in 1280d space
        self.train_mode = getattr(config, "train_mode", "decoded")

        # Reproducibility
        self._set_seed(getattr(config, "seed", 42))

        # Output directories
        self.output_dir = Path(getattr(config, "output_dir", "output/radio_gs"))
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.vis_dir = self.output_dir / "visualizations"
        for d in (self.ckpt_dir, self.log_dir, self.vis_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Components
        self.model = self.build_model(config).to(self.device)
        self.codec = self._build_codec(config).to(self.device)
        self.renderer = FeatureFieldRenderer(
            image_height=getattr(config, "feature_height", 30),
            image_width=getattr(config, "feature_width", 40),
            fx=getattr(config, "fx", 320.0) * getattr(config, "feature_width", 40) / getattr(config, "image_width", 640),
            fy=getattr(config, "fy", 320.0) * getattr(config, "feature_height", 30) / getattr(config, "image_height", 480),
            cx=getattr(config, "cx", 319.5) * getattr(config, "feature_width", 40) / getattr(config, "image_width", 640),
            cy=getattr(config, "cy", 239.5) * getattr(config, "feature_height", 30) / getattr(config, "image_height", 480),
            max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
            use_2dgs=getattr(config, "use_2dgs", False),
        ).to(self.device)
        self.sharpener = FeatSharp3D(
            mode=getattr(config, "featsharp_mode", "analytical"),
            feature_dim=self._resolve_latent_dim(config),
            strength=getattr(config, "featsharp_strength", 0.5),
        ).to(self.device)

        # In latent mode, freeze codec entirely
        if self.train_mode == "latent":
            for p in self.codec.parameters():
                p.requires_grad = False
            self._log("Latent mode: codec frozen, training in 64d space")

        # Losses
        self.distill_loss_fn = DistillationLoss(
            l2_weight=getattr(config, "l2_weight", 1.0),
            cosine_weight=getattr(config, "cosine_weight", 0.5),
        )
        self.mv_loss_fn = MultiViewConsistencyLoss()
        self.tv_loss_fn = TotalVariationLoss()

        # Feature norm regularization weight
        self.feat_norm_weight = getattr(config, "feat_norm_weight", 0.0)

        # Optimizer with separate LR groups
        self.optimizer = self._build_optimizer(config)
        self.scheduler = self._build_scheduler(config)
        self.scaler = GradScaler()

        # Datasets + loaders
        self.train_dataset, self.val_dataset = self.build_dataset(config)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=getattr(config, "batch_size", 4),
            shuffle=True,
            num_workers=getattr(config, "num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(config, "num_workers", 4),
            pin_memory=True,
        )

        # Logging
        self.writer: Optional[SummaryWriter] = None
        if _HAS_TB:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Tracking
        self.start_epoch = 1
        self.global_step = 0
        self.best_cosine = -1.0

        self._log(f"Model params: {self._count_params(self.model):.2f}M")
        self._log(f"Codec params: {self._count_params(self.codec):.2f}M")
        self._log(f"Sharpener mode: {self.sharpener.mode}")

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_latent_dim(config: RadioGSConfig) -> int:
        arch = getattr(config, "architecture", "explicit")
        if arch == "hybrid":
            return getattr(config, "hybrid_latent_dim", 16)
        return getattr(config, "latent_dim", 64)

    def build_model(self, config: RadioGSConfig) -> nn.Module:
        arch = getattr(config, "architecture", "explicit")
        if arch == "explicit":
            model = ExplicitFeatureGaussian(
                latent_dim=getattr(config, "latent_dim", 64),
            )
        elif arch == "hybrid":
            model = HybridFeatureGaussian(
                latent_dim=getattr(config, "hybrid_latent_dim", 16),
                hash_levels=getattr(config, "hash_levels", 16),
                hash_features_per_level=getattr(config, "hash_features_per_level", 2),
                hash_log2_size=getattr(config, "hash_log2_size", 19),
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        ply_path = getattr(config, "ply_path", None)
        if ply_path:
            self._log(f"Loading geometry from {ply_path}")
            model.load_from_ply(ply_path)

        return model

    @staticmethod
    def _build_codec(config: RadioGSConfig) -> nn.Module:
        return HCDCodec(
            input_dim=getattr(config, "radio_feature_dim", 1280),
            bottleneck_dim=getattr(config, "bottleneck_dim", 64),
            dual_stream=getattr(config, "dual_stream", True),
        )

    def _build_optimizer(self, config: RadioGSConfig) -> optim.Optimizer:
        param_groups = [
            {
                "params": self.model.trainable_parameters(),
                "lr": getattr(config, "lr_features", 1e-3),
                "name": "features",
            },
        ]
        # Only add decoder to optimizer if not in latent mode (decoder is frozen)
        if self.train_mode != "latent":
            param_groups.append(
                {
                    "params": self.codec.decoder.parameters(),
                    "lr": getattr(config, "lr_decoder", 1e-4),
                    "name": "decoder",
                }
            )
        if self.sharpener.mode not in ("analytical", "none"):
            param_groups.append(
                {
                    "params": self.sharpener.parameters(),
                    "lr": getattr(config, "lr_heads", 1e-4),
                    "name": "sharpener",
                }
            )
        return optim.AdamW(
            param_groups,
            weight_decay=getattr(config, "weight_decay", 1e-5),
            betas=(0.9, 0.999),
        )

    def _build_scheduler(
        self, config: RadioGSConfig
    ) -> optim.lr_scheduler._LRScheduler:
        warmup_epochs = getattr(config, "warmup_epochs", 5)
        total_epochs = getattr(config, "epochs", 100)

        cosine = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=1e-6,
        )
        if warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=warmup_epochs,
            )
            return optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        return cosine

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def build_dataset(
        self, config: RadioGSConfig
    ) -> Tuple[SimpleRadioDataset, SimpleRadioDataset]:
        feature_dir = getattr(config, "feature_dir", "")
        scene = getattr(config, "scene", "room_0")
        scene_root = Path("dataset") / scene
        train_split = getattr(config, "train_split", "Sequence_1")
        val_split = getattr(config, "val_split", "Sequence_2")
        depth_dir = getattr(config, "depth_dir", None)

        # Val features use a separate directory
        val_feature_dir = feature_dir.replace(train_split, val_split)
        if not Path(val_feature_dir).exists():
            val_feature_dir = feature_dir  # fallback: same dir

        train_ds = SimpleRadioDataset(
            feature_dir=feature_dir,
            pose_file=str(scene_root / train_split / "traj_w_c.txt"),
            depth_dir=depth_dir,
            split="train",
        )
        val_ds = SimpleRadioDataset(
            feature_dir=val_feature_dir,
            pose_file=str(scene_root / val_split / "traj_w_c.txt"),
            depth_dir=None,
            split="val",
        )
        self._log(f"Train: {len(train_ds)} frames  |  Val: {len(val_ds)} frames")
        return train_ds, val_ds

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        if self.train_mode != "latent":
            self.codec.train()
        self.sharpener.train()

        loss_accum = {"total": 0.0, "distill": 0.0, "compact": 0.0, "tv": 0.0}
        cos_accum = 0.0
        n_batches = 0
        log_every = getattr(self.cfg, "log_every", 100)

        pbar = tqdm(
            self.train_loader,
            desc=f"Train E{epoch:03d}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            gt_features = batch["radio_features"].to(self.device)   # [B, C, Hp, Wp]
            pose_w2c = batch["pose_w2c"].to(self.device)         # [B, 4, 4]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                # Render compact features from 3DGS
                result = self.renderer.render_features_batch(
                    self.model, pose_w2c
                )
                rendered_compact = result["feature_map"]  # [B, D, Hf, Wf]

                # Sharpen rendered features
                rendered_compact = self.sharpener(rendered_compact)

                if self.train_mode == "latent":
                    # LATENT MODE: gt_features are already 64d (pre-encoded)
                    gt_compact = gt_features
                    if gt_compact.shape[-2:] != rendered_compact.shape[-2:]:
                        gt_compact = F.interpolate(
                            gt_compact,
                            size=rendered_compact.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    # Primary loss: cosine + L2 in latent space
                    l_cos = 1.0 - F.cosine_similarity(
                        rendered_compact.float().flatten(2),
                        gt_compact.float().flatten(2),
                        dim=1,
                    ).mean()
                    l_l2 = F.mse_loss(rendered_compact.float(), gt_compact.float())
                    l2_w = getattr(self.cfg, "l2_weight", 1.0)
                    cos_w = getattr(self.cfg, "cosine_weight", 0.5)
                    l_distill = l2_w * l_l2 + cos_w * l_cos

                    l_compact = torch.tensor(0.0, device=self.device)

                    # Feature norm regularization
                    l_feat_norm = torch.tensor(0.0, device=self.device)
                    if self.feat_norm_weight > 0:
                        feat_norms = rendered_compact.float().norm(dim=1).mean()
                        gt_norms = gt_compact.float().norm(dim=1).mean()
                        l_feat_norm = (feat_norms - gt_norms).abs()

                else:
                    # DECODED MODE (legacy V1/V2): compare in 1280d space
                    gt_radio = gt_features
                    with torch.no_grad():
                        gt_compact = self.codec.encoder(gt_radio)

                    decoded = self.codec.decoder(rendered_compact)

                    if decoded.shape[-2:] != gt_radio.shape[-2:]:
                        gt_radio_rs = F.interpolate(
                            gt_radio,
                            size=decoded.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    else:
                        gt_radio_rs = gt_radio

                    distill_dict = self.distill_loss_fn(decoded, gt_radio_rs)
                    l_distill = distill_dict["total"]

                    if gt_compact.shape[-2:] != rendered_compact.shape[-2:]:
                        gt_compact_rs = F.interpolate(
                            gt_compact,
                            size=rendered_compact.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    else:
                        gt_compact_rs = gt_compact
                    l_compact = F.mse_loss(rendered_compact, gt_compact_rs)
                    l_feat_norm = torch.tensor(0.0, device=self.device)

                l_tv = self.tv_loss_fn(rendered_compact)

                adaptor_w = getattr(self.cfg, "adaptor_weight", 0.1)
                tv_w = getattr(self.cfg, "tv_weight", 0.01)
                loss = l_distill + adaptor_w * l_compact + tv_w * l_tv
                if self.feat_norm_weight > 0:
                    loss = loss + self.feat_norm_weight * l_feat_norm

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_clip = getattr(self.cfg, "grad_clip", 10.0)
            nn.utils.clip_grad_norm_(
                self._all_trainable_params(), max_norm=grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Compute monitoring cosine in appropriate space
            with torch.no_grad():
                if self.train_mode == "latent":
                    cos_sim = F.cosine_similarity(
                        rendered_compact.detach().float().flatten(2),
                        gt_compact.detach().float().flatten(2),
                        dim=1,
                    ).mean()
                else:
                    cos_sim = F.cosine_similarity(
                        decoded.detach().float().flatten(2),
                        gt_radio_rs.detach().float().flatten(2),
                        dim=1,
                    ).mean()

            loss_accum["total"] += loss.item()
            loss_accum["distill"] += l_distill.item()
            loss_accum["compact"] += l_compact.item()
            loss_accum["tv"] += l_tv.item()
            cos_accum += cos_sim.item()
            n_batches += 1
            self.global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}", cos=f"{cos_sim.item():.4f}"
            )

            # Periodic logging
            if self.global_step % log_every == 0 and self.writer is not None:
                self.writer.add_scalar(
                    "train/loss", loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/distill", l_distill.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/compact", l_compact.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/tv", l_tv.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/cosine", cos_sim.item(), self.global_step
                )
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr", lr, self.global_step)

        # Epoch averages
        if n_batches == 0:
            return {}
        metrics = {k: v / n_batches for k, v in loss_accum.items()}
        metrics["cosine"] = cos_accum / n_batches
        lr = self.optimizer.param_groups[0]["lr"]
        self._log(
            f"[Train E{epoch:03d}] loss={metrics['total']:.4f} "
            f"cosine={metrics['cosine']:.4f} lr={lr:.2e}"
        )
        return metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.codec.eval()
        self.sharpener.eval()

        cos_latent_accum = 0.0
        cos_decoded_accum = 0.0
        mse_accum = 0.0
        n = 0

        for batch in tqdm(
            self.val_loader, desc=f"Val   E{epoch:03d}", leave=False, dynamic_ncols=True
        ):
            gt_features = batch["radio_features"].to(self.device)
            pose_w2c = batch["pose_w2c"].to(self.device)

            rendered_compact = self.renderer.render_features_batch(self.model, pose_w2c)["feature_map"]
            rendered_compact = self.sharpener(rendered_compact)

            if self.train_mode == "latent":
                # gt_features are 64d
                gt_compact = gt_features
                if gt_compact.shape[-2:] != rendered_compact.shape[-2:]:
                    gt_compact = F.interpolate(
                        gt_compact, size=rendered_compact.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                cos_latent = F.cosine_similarity(
                    rendered_compact.float().flatten(2),
                    gt_compact.float().flatten(2),
                    dim=1,
                ).mean()
                cos_latent_accum += cos_latent.item()

                # Also decode and compare to 1280d GT for monitoring
                decoded = self.codec.decoder(rendered_compact)
                # Load 1280d GT for this frame
                gt_1280_path = self._get_1280d_val_path(batch)
                if gt_1280_path is not None:
                    gt_1280 = torch.load(gt_1280_path).float().unsqueeze(0).to(self.device)
                    if gt_1280.shape[-2:] != decoded.shape[-2:]:
                        gt_1280 = F.interpolate(
                            gt_1280, size=decoded.shape[-2:],
                            mode="bilinear", align_corners=False,
                        )
                    cos_dec = F.cosine_similarity(
                        decoded.float().flatten(2),
                        gt_1280.float().flatten(2),
                        dim=1,
                    ).mean()
                    mse = F.mse_loss(decoded.float(), gt_1280.float())
                    cos_decoded_accum += cos_dec.item()
                    mse_accum += mse.item()
                else:
                    cos_decoded_accum += cos_latent.item()
                    mse_accum += F.mse_loss(rendered_compact.float(), gt_compact.float()).item()
            else:
                # Decoded mode: gt_features are 1280d
                gt_radio = gt_features
                decoded = self.codec.decoder(rendered_compact)
                if decoded.shape[-2:] != gt_radio.shape[-2:]:
                    gt_radio = F.interpolate(
                        gt_radio, size=decoded.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                cos_dec = F.cosine_similarity(
                    decoded.float().flatten(2),
                    gt_radio.float().flatten(2),
                    dim=1,
                ).mean()
                mse = F.mse_loss(decoded.float(), gt_radio.float())
                cos_decoded_accum += cos_dec.item()
                cos_latent_accum += cos_dec.item()
                mse_accum += mse.item()

            n += 1

        if n == 0:
            return {}

        avg_cos_latent = cos_latent_accum / n
        avg_cos_decoded = cos_decoded_accum / n
        avg_mse = mse_accum / n
        psnr = -10.0 * np.log10(avg_mse + 1e-8)

        # Primary metric for best model selection: latent cosine in latent mode
        primary_cos = avg_cos_latent if self.train_mode == "latent" else avg_cos_decoded

        metrics = {
            "cosine": primary_cos,
            "cosine_latent": avg_cos_latent,
            "cosine_decoded": avg_cos_decoded,
            "mse": avg_mse,
            "psnr": psnr,
        }

        if self.writer is not None:
            self.writer.add_scalar("val/cosine_latent", avg_cos_latent, epoch)
            self.writer.add_scalar("val/cosine_decoded", avg_cos_decoded, epoch)
            self.writer.add_scalar("val/psnr", psnr, epoch)

        self._save_vis(epoch)

        self._log(
            f"[Val E{epoch:03d}] cos_latent={avg_cos_latent:.4f} "
            f"cos_decoded={avg_cos_decoded:.4f} psnr={psnr:.2f}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "codec_state_dict": self.codec.state_dict(),
            "sharpener_state_dict": self.sharpener.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_cosine": self.best_cosine,
            "metrics": metrics,
        }
        torch.save(state, self.ckpt_dir / "latest.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")

    def load_checkpoint(self, path: str, resume: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "codec_state_dict" in ckpt:
            self.codec.load_state_dict(ckpt["codec_state_dict"], strict=False)
        if "sharpener_state_dict" in ckpt:
            self.sharpener.load_state_dict(
                ckpt["sharpener_state_dict"], strict=False
            )

        if resume:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.global_step = ckpt.get("global_step", 0)
            self.best_cosine = ckpt.get("best_cosine", -1.0)
            self._log(f"Resumed from epoch {self.start_epoch - 1}")
        else:
            self._log(f"Warmstart: loaded model weights from {path}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        total_epochs = getattr(self.cfg, "epochs", 100)
        eval_every = getattr(self.cfg, "eval_every", 5)
        save_every = getattr(self.cfg, "save_every", 10)

        self._log(
            f"Starting training: epochs {self.start_epoch}→{total_epochs}, "
            f"eval_every={eval_every}, save_every={save_every}"
        )

        for epoch in range(self.start_epoch, total_epochs + 1):
            train_metrics = self.train_epoch(epoch)

            if epoch % eval_every == 0 or epoch == total_epochs:
                val_metrics = self.validate(epoch)
                is_best = val_metrics.get("cosine", -1) > self.best_cosine
                if is_best:
                    self.best_cosine = val_metrics["cosine"]
                    self._log(
                        f"  ★ New best! cosine={self.best_cosine:.4f} "
                        f"psnr={val_metrics.get('psnr', 0):.2f}"
                    )
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            elif epoch % save_every == 0:
                self.save_checkpoint(epoch, train_metrics)

            self.scheduler.step()

        self._log("Training complete.")
        if self.writer is not None:
            self.writer.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_trainable_params(self):
        """Gather all trainable parameters for gradient clipping."""
        params = list(self.model.trainable_parameters())
        if self.train_mode != "latent":
            params += list(self.codec.decoder.parameters())
        if self.sharpener.mode not in ("analytical", "none"):
            params += list(self.sharpener.parameters())
        return params

    def _get_1280d_val_path(self, batch) -> Optional[str]:
        """In latent mode, try to locate the original 1280d feature for monitoring."""
        try:
            idx = batch["frame_idx"].item()
            val_1280_dir = getattr(self.cfg, "val_1280d_dir", None)
            if val_1280_dir is None:
                # Derive from feature_dir: replace 64d with 1280d
                feat_dir = getattr(self.cfg, "feature_dir", "")
                val_split = getattr(self.cfg, "val_split", "Sequence_2")
                train_split = getattr(self.cfg, "train_split", "Sequence_1")
                val_1280_dir = feat_dir.replace("64d", "1280d").replace(train_split, val_split)
            p = Path(val_1280_dir) / "backbone" / f"rgb_{idx}.pt"
            if not p.exists():
                p = Path(val_1280_dir) / f"rgb_{idx}.pt"
            return str(p) if p.exists() else None
        except Exception:
            return None

    @staticmethod
    def _count_params(module: nn.Module) -> float:
        return sum(p.numel() for p in module.parameters()) / 1e6

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file = self.log_dir / "training.log"
        with open(log_file, "a") as f:
            f.write(line + "\n")

    @torch.no_grad()
    def _save_vis(self, epoch: int) -> None:
        """Save PCA visualisation for the first validation sample."""
        try:
            sample = self.val_dataset[0]
            gt = sample["radio_features"].unsqueeze(0).to(self.device)
            pose = sample["pose_w2c"].unsqueeze(0).to(self.device)

            rendered = self.renderer.render_features_batch(self.model, pose)["feature_map"]
            rendered = self.sharpener(rendered)
            decoded = self.codec.decoder(rendered)

            if decoded.shape[-2:] != gt.shape[-2:]:
                gt = F.interpolate(
                    gt, size=decoded.shape[-2:], mode="bilinear", align_corners=False
                )

            # Simple 3-component PCA → RGB image
            for tag, feat in [("gt", gt), ("decoded", decoded)]:
                flat = feat[0].float().flatten(1)           # [C, H*W]
                mean = flat.mean(dim=1, keepdim=True)
                centered = flat - mean
                U, S, _ = torch.pca_lowrank(centered.T, q=3)  # [H*W, 3]
                rgb = U.T.reshape(3, *decoded.shape[-2:])
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                if self.writer is not None:
                    self.writer.add_image(f"val/{tag}", rgb, epoch)
        except Exception:
            pass  # visualisation is best-effort


# ===================================================================
# Entry point
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RADIO-GS feature field via distillation."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--warmstart", default=None, help="Load model weights only"
    )
    parser.add_argument(
        "--pretrained_codec", default=None,
        help="Path to pretrained HCD codec checkpoint (from train_codec.py)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = RadioGSTrainer(config)

    # Load pretrained codec first (before resume/warmstart which may override)
    if args.pretrained_codec:
        ckpt = torch.load(args.pretrained_codec, map_location=trainer.device)
        trainer.codec.load_state_dict(ckpt["codec_state_dict"])
        trainer._log(f"Loaded pretrained codec from {args.pretrained_codec}")

    if args.resume:
        trainer.load_checkpoint(args.resume, resume=True)
    elif args.warmstart:
        trainer.load_checkpoint(args.warmstart, resume=False)

    trainer.train()


if __name__ == "__main__":
    main()
