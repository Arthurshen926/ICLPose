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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radio_gs.config import RadioGSConfig, load_config
from radio_gs.heads.depth_head import DepthHead, DepthLoss
from radio_gs.heads.segmentation_head import SegmentationHead, SegmentationLoss, compute_miou
from radio_gs.losses.distillation_loss import (
    DistillationLoss,
    MultiViewConsistencyLoss,
    TotalVariationLoss,
)
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
from radio_gs.models.siglip_projection import SigLIP2FeatureProjection
from radio_gs.models.screen_refiner import ScreenSpaceRefiner
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
        semantics_dir: Optional[str] = None,
        rgb_dir: Optional[str] = None,
        feature_size: Optional[tuple] = None,
        split: str = "train",
    ):
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.semantics_dir = Path(semantics_dir) if semantics_dir else None
        self.rgb_dir = Path(rgb_dir) if rgb_dir else None
        self.feature_size = feature_size  # (H, W) for downsampling RGB
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
        if self.semantics_dir is not None and self.semantics_dir.exists():
            self.semantics_paths: Optional[List[Path]] = sorted(
                self.semantics_dir.glob("semantic_class_*.png"),
                key=lambda p: int(p.stem.split("_")[-1])
                if p.stem.split("_")[-1].isdigit()
                else 0,
            )
        else:
            self.semantics_paths = None

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
                depth = torch.from_numpy(d.astype(np.float32) / 1000.0).clone()

        semantics: Optional[torch.Tensor] = None
        if self.semantics_paths is not None and idx < len(self.semantics_paths):
            from PIL import Image

            with Image.open(self.semantics_paths[idx]) as sem_img:
                sem = np.array(sem_img, dtype=np.int64)
            semantics = torch.from_numpy(sem).clone()

        # --- optional RGB guide (downsampled to feature resolution) --------
        rgb_guide: Optional[torch.Tensor] = None
        if self.rgb_dir is not None:
            import cv2

            rgb_path = self.rgb_dir / f"rgb_{idx}.png"
            if rgb_path.exists():
                img = cv2.imread(str(rgb_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.feature_size is not None:
                        img = cv2.resize(img, (self.feature_size[1], self.feature_size[0]))
                    rgb_guide = torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0

        out: Dict[str, torch.Tensor] = {
            "radio_features": radio_feat,
            "pose_w2c": pose_w2c,
            "frame_idx": torch.tensor(idx, dtype=torch.long),
        }
        if depth is not None:
            out["depth"] = depth
        if semantics is not None:
            out["semantics"] = semantics
        if rgb_guide is not None:
            out["rgb_guide"] = rgb_guide
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

        # Optional screen-space refiner (corrects alpha-blending artifacts)
        self.use_refiner = getattr(config, "use_refiner", False)
        self.refiner_rgb_guide = getattr(config, "refiner_rgb_guide", False)
        self.refiner_depth_guide = getattr(config, "refiner_depth_guide", False)
        self.self_guided = getattr(config, "self_guided", False)
        self.train_sh = getattr(config, "train_sh", False)
        self.rgb_loss_weight = getattr(config, "rgb_loss_weight", 0.0)
        self._is_hybrid = getattr(config, "architecture", "explicit") == "hybrid"

        # Enable SH training if requested
        if self.train_sh and hasattr(self.model, "enable_sh_training"):
            self.model.enable_sh_training()
            self._log("Joint RGB training: SH coefficients unfrozen")

        if self.use_refiner:
            extra_ch = 0
            if self.refiner_rgb_guide:
                extra_ch += 3
            if self.refiner_depth_guide:
                extra_ch += 3 if getattr(config, "refiner_depth_grad", False) else 1
            self.refiner = ScreenSpaceRefiner(
                latent_dim=self._resolve_latent_dim(config),
                hidden_dim=getattr(config, "refiner_hidden_dim", 128),
                num_blocks=getattr(config, "refiner_num_blocks", 4),
                dropout=getattr(config, "refiner_dropout", 0.1),
                extra_channels=extra_ch,
                norm_type=getattr(config, "refiner_norm_type", "gn"),
            ).to(self.device)
        else:
            self.refiner = None

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
        self.depth_loss_weight = getattr(config, "depth_loss_weight", 0.0)
        self.geom_depth_loss_weight = getattr(config, "geom_depth_loss_weight", 0.0)
        self.depth_alpha_threshold = getattr(config, "depth_alpha_threshold", 0.05)
        self.depth_head: Optional[DepthHead] = None
        self.depth_supervision_loss: Optional[DepthLoss] = None
        self.geom_depth_supervision_loss: Optional[DepthLoss] = None
        self.seg_loss_weight = getattr(config, "seg_loss_weight", 0.0)
        self.seg_head: Optional[SegmentationHead] = None
        self.seg_loss_fn: Optional[SegmentationLoss] = None
        self.siglip_alignment_weight = getattr(config, "siglip_alignment_weight", 0.0)
        self.siglip_projection: Optional[SigLIP2FeatureProjection] = None
        if self.depth_loss_weight > 0 or self.geom_depth_loss_weight > 0:
            self.depth_head = DepthHead(
                feature_dim=getattr(config, "radio_feature_dim", 1280),
                hidden_dim=getattr(config, "depth_head_hidden_dim", 256),
                num_layers=getattr(config, "depth_head_num_layers", 3),
                head_type=getattr(config, "depth_head_type", "mlp"),
            ).to(self.device)
            self.depth_supervision_loss = DepthLoss(
                loss_type=getattr(config, "depth_supervision_loss_type", "scale_invariant"),
                weight=1.0,
            )
            self.geom_depth_supervision_loss = DepthLoss(
                loss_type=getattr(
                    config,
                    "geom_depth_supervision_loss_type",
                    getattr(config, "depth_supervision_loss_type", "scale_invariant"),
                ),
                weight=1.0,
            )
        if self.seg_loss_weight > 0:
            self.seg_head = SegmentationHead(
                feature_dim=getattr(config, "radio_feature_dim", 1280),
                num_classes=getattr(config, "seg_num_classes", 40),
                hidden_dim=getattr(config, "seg_head_hidden_dim", 256),
                num_layers=getattr(config, "seg_head_num_layers", 2),
                head_type=getattr(config, "seg_head_type", "mlp"),
            ).to(self.device)
            self.seg_loss_fn = SegmentationLoss(
                loss_type=getattr(config, "seg_loss_type", "ce"),
                ignore_index=getattr(config, "seg_ignore_index", 255),
            )
        if self.siglip_alignment_weight > 0:
            proj_path = Path(
                getattr(
                    config,
                    "siglip_projection_weights",
                    "output/radio_gs/siglip2_feat_projection.pth",
                )
            )
            if not proj_path.is_absolute():
                proj_path = Path(__file__).resolve().parents[2] / proj_path
            if not proj_path.exists():
                raise FileNotFoundError(
                    f"SigLIP2 projection weights not found: {proj_path}"
                )
            self.siglip_projection = SigLIP2FeatureProjection().to(self.device)
            self.siglip_projection.load_state_dict(
                torch.load(proj_path, map_location="cpu")
            )
            self.siglip_projection.eval()
            for param in self.siglip_projection.parameters():
                param.requires_grad = False

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
            collate_fn=self._collate_batch,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(config, "num_workers", 4),
            pin_memory=True,
            collate_fn=self._collate_batch,
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
        if self.use_refiner and self.refiner is not None:
            self._log(f"Refiner params: {self._count_params(self.refiner):.2f}M")
        if self.depth_head is not None:
            self._log(f"Depth aux head params: {self._count_params(self.depth_head):.2f}M")
        if self.seg_head is not None:
            self._log(f"Seg aux head params: {self._count_params(self.seg_head):.2f}M")

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
                train_sh=getattr(config, "train_sh", False),
            )
        elif arch == "hybrid":
            model = HybridFeatureGaussian(
                latent_dim=getattr(config, "hybrid_latent_dim", 16),
                hash_output_dim=getattr(config, "hash_output_dim", 48),
                fine_dim=getattr(config, "fine_dim", 64),
                coarse_dim=getattr(config, "coarse_dim", 64),
                output_dim=getattr(config, "hybrid_output_dim", 128),
                num_levels=getattr(config, "hash_levels", 16),
                features_per_level=getattr(config, "hash_features_per_level", 2),
                log2_hashmap_size=getattr(config, "hash_log2_size", 19),
                base_resolution=getattr(config, "hash_base_resolution", 16),
                max_resolution=getattr(config, "hash_max_resolution", 2048),
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
        arch = getattr(config, "architecture", "explicit")
        # Feature embeddings (always trainable)
        feature_params = [self.model._feature if arch == "explicit" else self.model._latent]
        param_groups = [
            {
                "params": feature_params,
                "lr": getattr(config, "lr_features", 1e-3),
                "name": "features",
            },
        ]
        # Hybrid architecture: hash grid + screen-space decoders
        if arch == "hybrid":
            param_groups.append({
                "params": list(self.model.hash_field.parameters()),
                "lr": getattr(config, "lr_hash", 1e-3),
                "name": "hash_field",
            })
            hybrid_decoder_params = (
                list(self.model.fine_decoder.parameters())
                + list(self.model.coarse_decoder.parameters())
                + list(self.model.fusion_head.parameters())
            )
            param_groups.append({
                "params": hybrid_decoder_params,
                "lr": getattr(config, "lr_decoder", 1e-4),
                "name": "hybrid_decoders",
            })
        # SH params (separate group for joint RGB training)
        if self.train_sh and hasattr(self.model, "_sh_dc_param"):
            sh_params = [self.model._sh_dc_param]
            if hasattr(self.model, "_sh_rest_param") and self.model._sh_rest_param is not None:
                sh_params.append(self.model._sh_rest_param)
            param_groups.append(
                {
                    "params": sh_params,
                    "lr": getattr(config, "lr_sh", 5e-4),
                    "name": "sh_colors",
                }
            )
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
        if self.use_refiner and self.refiner is not None:
            param_groups.append(
                {
                    "params": self.refiner.parameters(),
                    "lr": getattr(config, "lr_refiner", 5e-4),
                    "name": "refiner",
                }
            )
        if self.depth_head is not None:
            param_groups.append(
                {
                    "params": self.depth_head.parameters(),
                    "lr": getattr(config, "lr_heads", 1e-4),
                    "name": "depth_head",
                }
            )
        if self.seg_head is not None:
            param_groups.append(
                {
                    "params": self.seg_head.parameters(),
                    "lr": getattr(config, "lr_heads", 1e-4),
                    "name": "seg_head",
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
    ) -> Tuple[Dataset, Dataset]:
        feature_dir = getattr(config, "feature_dir", "")
        scene = getattr(config, "scene", "room_0")
        scene_root = Path("dataset") / scene
        train_split = getattr(config, "train_split", "Sequence_1")
        val_split = getattr(config, "val_split", "Sequence_2")
        depth_dir = getattr(config, "depth_dir", None)
        semantics_dir = getattr(config, "semantics_dir", None)
        mixed_split = getattr(config, "mixed_split", False)

        # Val features use a separate directory
        val_feature_dir = feature_dir.replace(train_split, val_split)
        if not Path(val_feature_dir).exists():
            val_feature_dir = feature_dir  # fallback: same dir

        # RGB guide setup
        rgb_dir_train = rgb_dir_val = None
        feature_size = None
        if self.refiner_rgb_guide:
            feature_size = (
                getattr(config, "feature_height", 30),
                getattr(config, "feature_width", 40),
            )
            rgb_dir_train = str(scene_root / train_split / "rgb")
            rgb_dir_val = str(scene_root / val_split / "rgb")

        if mixed_split:
            # Merge both sequences and random 80/20 split
            val_depth_dir = depth_dir.replace(train_split, val_split) if depth_dir else None
            val_semantics_dir = (
                semantics_dir.replace(train_split, val_split) if semantics_dir else None
            )
            ds_seq1 = SimpleRadioDataset(
                feature_dir=feature_dir,
                pose_file=str(scene_root / train_split / "traj_w_c.txt"),
                depth_dir=depth_dir,
                semantics_dir=semantics_dir,
                rgb_dir=rgb_dir_train,
                feature_size=feature_size,
                split="train",
            )
            ds_seq2 = SimpleRadioDataset(
                feature_dir=val_feature_dir,
                pose_file=str(scene_root / val_split / "traj_w_c.txt"),
                depth_dir=val_depth_dir,
                semantics_dir=val_semantics_dir,
                rgb_dir=rgb_dir_val,
                feature_size=feature_size,
                split="train",
            )
            combined = ConcatDataset([ds_seq1, ds_seq2])
            total = len(combined)
            train_ratio = getattr(config, "mixed_train_ratio", 0.8)
            train_size = int(train_ratio * total)
            val_size = total - train_size
            seed = getattr(config, "mixed_seed", 42)
            gen = torch.Generator().manual_seed(seed)
            train_ds, val_ds = torch.utils.data.random_split(
                combined, [train_size, val_size], generator=gen
            )
            self._log(
                f"Mixed split: {total} total → Train: {train_size} | Val: {val_size} "
                f"(ratio={train_ratio}, seed={seed})"
            )
            return train_ds, val_ds

        train_ds = SimpleRadioDataset(
            feature_dir=feature_dir,
            pose_file=str(scene_root / train_split / "traj_w_c.txt"),
            depth_dir=depth_dir,
            semantics_dir=semantics_dir,
            rgb_dir=rgb_dir_train,
            feature_size=feature_size,
            split="train",
        )
        val_ds = SimpleRadioDataset(
            feature_dir=val_feature_dir,
            pose_file=str(scene_root / val_split / "traj_w_c.txt"),
            depth_dir=depth_dir.replace(train_split, val_split) if depth_dir else None,
            semantics_dir=(
                semantics_dir.replace(train_split, val_split) if semantics_dir else None
            ),
            rgb_dir=rgb_dir_val,
            feature_size=feature_size,
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
        if self.use_refiner and self.refiner is not None:
            self.refiner.train()
        if self.depth_head is not None:
            self.depth_head.train()
        if self.seg_head is not None:
            self.seg_head.train()

        loss_accum = {
            "total": 0.0,
            "distill": 0.0,
            "compact": 0.0,
            "tv": 0.0,
            "rgb": 0.0,
            "depth_gt": 0.0,
            "depth_geom": 0.0,
            "seg_aux": 0.0,
            "siglip_align": 0.0,
        }
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
            if self._is_hybrid:
                gt_features = gt_features.float()
                pose_w2c = pose_w2c.float()

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=not self._is_hybrid):
                # Render compact features (and optionally RGB) from 3DGS
                rendered_rgb = None
                l_rgb = torch.tensor(0.0, device=self.device)

                if self.self_guided and self.train_sh:
                    # Joint rendering with SH training: features + RGB, backprop RGB loss
                    result = self.renderer.render_features_and_rgb(
                        self.model, pose_w2c
                    )
                    rendered_compact = result["feature_map"]
                    rendered_rgb = result["rgb"]

                    gt_rgb = batch.get("rgb_guide")
                    if gt_rgb is not None:
                        gt_rgb = gt_rgb.to(self.device)
                        l_rgb = F.l1_loss(rendered_rgb.float(), gt_rgb.float())
                elif self.self_guided:
                    # Self-guided with frozen SH: render RGB as guide (no gradient)
                    result = self.renderer.render_features_and_rgb(
                        self.model, pose_w2c
                    )
                    rendered_compact = result["feature_map"]
                    rendered_rgb = result["rgb"].detach()
                else:
                    result = self.renderer.render_features_batch(
                        self.model, pose_w2c
                    )
                    rendered_compact = result["feature_map"]

                # Sharpen rendered features
                rendered_compact = self.sharpener(rendered_compact)

                # Apply screen-space refiner if enabled
                if self.use_refiner and self.refiner is not None:
                    guide = self._build_guide(batch, result, rendered_rgb)
                    rendered_compact = self.refiner(rendered_compact, guide=guide)

                # Hybrid architecture: decode via hash grid + fusion
                if self._is_hybrid:
                    from radio_gs.models.hybrid_gaussian import unproject_depth_to_positions
                    depth_map = result["depth_map"].float()
                    position_map = unproject_depth_to_positions(
                        depth_map, pose_w2c.float(), self.renderer.K.float(),
                        depth_map.shape[1], depth_map.shape[2],
                    )
                    position_map = self._normalize_positions(position_map)
                    rendered_compact = self.model.decode_screen_space(
                        rendered_compact.float(), position_map
                    )

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
                    decoded_for_depth = self.codec.decoder(rendered_compact)

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
                    decoded_for_depth = decoded

                depth_losses = self._compute_depth_aux_losses(
                    batch=batch,
                    render_result=result,
                    decoded=decoded_for_depth,
                )
                seg_losses = self._compute_seg_aux_losses(
                    batch=batch,
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                )
                l_siglip = self._compute_siglip_alignment_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio_rs if self.train_mode != "latent" else None,
                )

                l_tv = self.tv_loss_fn(rendered_compact)

                adaptor_w = getattr(self.cfg, "adaptor_weight", 0.1)
                tv_w = getattr(self.cfg, "tv_weight", 0.01)
                loss = l_distill + adaptor_w * l_compact + tv_w * l_tv
                if self.feat_norm_weight > 0:
                    loss = loss + self.feat_norm_weight * l_feat_norm
                if self.rgb_loss_weight > 0:
                    loss = loss + self.rgb_loss_weight * l_rgb
                loss = loss + depth_losses["total"] + seg_losses["total"] + l_siglip

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
            loss_accum["rgb"] += l_rgb.item()
            loss_accum["depth_gt"] += depth_losses["depth_gt"].item()
            loss_accum["depth_geom"] += depth_losses["depth_geom"].item()
            loss_accum["seg_aux"] += seg_losses["total"].item()
            loss_accum["siglip_align"] += l_siglip.item()
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
                self.writer.add_scalar(
                    "train/depth_gt", depth_losses["depth_gt"].item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/depth_geom", depth_losses["depth_geom"].item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/seg_aux", seg_losses["total"].item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/siglip_align", l_siglip.item(), self.global_step
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
        if self.use_refiner and self.refiner is not None:
            self.refiner.eval()
        if self.depth_head is not None:
            self.depth_head.eval()
        if self.seg_head is not None:
            self.seg_head.eval()

        cos_latent_accum = 0.0
        cos_decoded_accum = 0.0
        mse_accum = 0.0
        depth_gt_accum = 0.0
        depth_geom_accum = 0.0
        seg_aux_accum = 0.0
        seg_aux_miou_accum = 0.0
        siglip_align_accum = 0.0
        n = 0

        for batch in tqdm(
            self.val_loader, desc=f"Val   E{epoch:03d}", leave=False, dynamic_ncols=True
        ):
            gt_features = batch["radio_features"].to(self.device)
            pose_w2c = batch["pose_w2c"].to(self.device)

            rendered_rgb = None
            if self.self_guided:
                val_result = self.renderer.render_features_and_rgb(self.model, pose_w2c)
                rendered_rgb = val_result["rgb"]
            else:
                val_result = self.renderer.render_features_batch(self.model, pose_w2c)
            rendered_compact = val_result["feature_map"]
            rendered_compact = self.sharpener(rendered_compact)
            if self.use_refiner and self.refiner is not None:
                guide = self._build_guide(batch, val_result, rendered_rgb=rendered_rgb)
                rendered_compact = self.refiner(rendered_compact, guide=guide)

            # Hybrid decode: latent + hash grid → fused output
            if self._is_hybrid:
                from radio_gs.models.hybrid_gaussian import unproject_depth_to_positions
                depth_map = val_result["depth_map"].float()
                position_map = unproject_depth_to_positions(
                    depth_map, pose_w2c.float(), self.renderer.K.float(),
                    depth_map.shape[1], depth_map.shape[2],
                )
                position_map = self._normalize_positions(position_map)
                rendered_compact = self.model.decode_screen_space(
                    rendered_compact.float(), position_map
                )

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
                decoded_for_depth = decoded
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
                decoded_for_depth = decoded

            depth_losses = self._compute_depth_aux_losses(
                batch=batch,
                render_result=val_result,
                decoded=decoded_for_depth,
            )
            seg_losses = self._compute_seg_aux_losses(
                batch=batch,
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
            )
            depth_gt_accum += depth_losses["depth_gt"].item()
            depth_geom_accum += depth_losses["depth_geom"].item()
            seg_aux_accum += seg_losses["total"].item()
            seg_aux_miou_accum += seg_losses["miou"]
            siglip_align_accum += self._compute_siglip_alignment_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                target=gt_radio if self.train_mode != "latent" else None,
            ).item()

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
            "depth_gt": depth_gt_accum / n,
            "depth_geom": depth_geom_accum / n,
            "seg_aux": seg_aux_accum / n,
            "seg_aux_miou": seg_aux_miou_accum / n,
            "siglip_align": siglip_align_accum / n,
        }

        if self.writer is not None:
            self.writer.add_scalar("val/cosine_latent", avg_cos_latent, epoch)
            self.writer.add_scalar("val/cosine_decoded", avg_cos_decoded, epoch)
            self.writer.add_scalar("val/psnr", psnr, epoch)
            self.writer.add_scalar("val/depth_gt", metrics["depth_gt"], epoch)
            self.writer.add_scalar("val/depth_geom", metrics["depth_geom"], epoch)
            self.writer.add_scalar("val/seg_aux", metrics["seg_aux"], epoch)
            self.writer.add_scalar("val/seg_aux_miou", metrics["seg_aux_miou"], epoch)
            self.writer.add_scalar("val/siglip_align", metrics["siglip_align"], epoch)

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
        if self.use_refiner and self.refiner is not None:
            state["refiner_state_dict"] = self.refiner.state_dict()
        if self.depth_head is not None:
            state["depth_head_state_dict"] = self.depth_head.state_dict()
        if self.seg_head is not None:
            state["seg_head_state_dict"] = self.seg_head.state_dict()
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
        if "refiner_state_dict" in ckpt and self.use_refiner and self.refiner is not None:
            try:
                self.refiner.load_state_dict(
                    ckpt["refiner_state_dict"], strict=False
                )
            except RuntimeError as e:
                self._log(f"Refiner state_dict size mismatch (architecture changed), "
                          f"starting refiner from scratch: {e}")
        if "depth_head_state_dict" in ckpt and self.depth_head is not None:
            try:
                self.depth_head.load_state_dict(
                    ckpt["depth_head_state_dict"], strict=False
                )
            except RuntimeError as e:
                self._log(
                    f"Depth head state_dict size mismatch, starting depth head from scratch: {e}"
                )
        if "seg_head_state_dict" in ckpt and self.seg_head is not None:
            try:
                self.seg_head.load_state_dict(
                    ckpt["seg_head_state_dict"], strict=False
                )
            except RuntimeError as e:
                self._log(
                    f"Seg head state_dict size mismatch, starting seg head from scratch: {e}"
                )

        if resume:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                self._log(f"Optimizer state mismatch (new param groups?), "
                          f"re-initializing optimizer: {e}")
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (ValueError, KeyError) as e:
                self._log(f"Scheduler state mismatch, re-initializing: {e}")
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
        if self.use_refiner and self.refiner is not None:
            params += list(self.refiner.parameters())
        if self.depth_head is not None:
            params += list(self.depth_head.parameters())
        if self.seg_head is not None:
            params += list(self.seg_head.parameters())
        return params

    @staticmethod
    def _collate_batch(batch):
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            return torch.stack([item.clone() for item in batch], dim=0)
        if isinstance(elem, dict):
            return {
                key: RadioGSTrainer._collate_batch([item[key] for item in batch])
                for key in elem
            }
        if isinstance(elem, (int, float)):
            return torch.tensor(batch)
        return batch

    @staticmethod
    def _resize_map(
        x: torch.Tensor,
        size: Tuple[int, int],
        is_mask: bool = False,
    ) -> torch.Tensor:
        """Resize a dense [B,H,W] or [B,1,H,W] map to the target spatial size."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[-2:] == size:
            return x.float()
        if is_mask:
            return F.interpolate(x.float(), size=size, mode="nearest")
        if x.shape[-2] >= size[0] and x.shape[-1] >= size[1]:
            return F.interpolate(x.float(), size=size, mode="area")
        return F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False)

    def _compute_depth_aux_losses(
        self,
        batch: Dict[str, torch.Tensor],
        render_result: Dict[str, torch.Tensor],
        decoded: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = decoded.sum() * 0.0
        losses = {
            "total": zero,
            "depth_gt": zero,
            "depth_geom": zero,
        }
        if self.depth_head is None:
            return losses

        gt_depth = batch.get("depth")
        if gt_depth is None:
            return losses

        pred_depth = self.depth_head(decoded.float())
        target_size = pred_depth.shape[-2:]
        gt_depth = self._resize_map(gt_depth.to(self.device).float(), target_size)
        alpha = self._resize_map(
            render_result["alpha_map"].to(self.device).float(),
            target_size,
        )
        valid_mask = (gt_depth > 0) & (alpha > self.depth_alpha_threshold)

        if self.depth_loss_weight > 0 and valid_mask.any():
            assert self.depth_supervision_loss is not None
            losses["depth_gt"] = self.depth_supervision_loss(pred_depth, gt_depth, valid_mask)

        if self.geom_depth_loss_weight > 0:
            geom_key = "geom_depth" if "geom_depth" in render_result else "depth_map"
            geom_depth = self._resize_map(
                render_result[geom_key].detach().to(self.device).float(),
                target_size,
            )
            geom_mask = (geom_depth > 0) & (alpha > self.depth_alpha_threshold)
            geom_mask = geom_mask & valid_mask
            if geom_mask.any():
                assert self.geom_depth_supervision_loss is not None
                losses["depth_geom"] = self.geom_depth_supervision_loss(
                    pred_depth, geom_depth, geom_mask
                )

        losses["total"] = (
            self.depth_loss_weight * losses["depth_gt"]
            + self.geom_depth_loss_weight * losses["depth_geom"]
        )
        return losses

    def _project_siglip_features(self, features: torch.Tensor) -> torch.Tensor:
        assert self.siglip_projection is not None
        B, C, H, W = features.shape
        feat_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, C).float()
        projected = self.siglip_projection(feat_flat)
        projected = projected.permute(0, 2, 1).reshape(B, -1, H, W)
        return F.normalize(projected, dim=1)

    def _compute_siglip_alignment_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            self.siglip_projection is None
            or self.siglip_alignment_weight <= 0
            or decoded is None
            or target is None
        ):
            return torch.tensor(0.0, device=self.device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        pred_siglip = self._project_siglip_features(decoded)
        with torch.no_grad():
            target_siglip = self._project_siglip_features(target)
        return self.siglip_alignment_weight * F.mse_loss(pred_siglip, target_siglip)

    def _compute_seg_aux_losses(
        self,
        batch: Dict[str, torch.Tensor],
        decoded: Optional[torch.Tensor],
    ) -> Dict[str, float | torch.Tensor]:
        zero = torch.tensor(0.0, device=self.device)
        losses: Dict[str, float | torch.Tensor] = {
            "total": zero,
            "miou": 0.0,
        }
        if self.seg_head is None or self.seg_loss_fn is None or decoded is None:
            return losses
        gt_sem = batch.get("semantics")
        if gt_sem is None:
            return losses
        gt_sem = gt_sem.to(self.device).long()
        if gt_sem.shape[-2:] != decoded.shape[-2:]:
            gt_sem = self._resize_map(
                gt_sem.unsqueeze(1).float(), decoded.shape[-2:], is_mask=True
            ).squeeze(1).long()
        seg_logits = self.seg_head(decoded.float())
        seg_loss = self.seg_loss_fn(seg_logits, gt_sem)
        losses["total"] = self.seg_loss_weight * seg_loss
        with torch.no_grad():
            pred = seg_logits.argmax(dim=1)
            losses["miou"] = compute_miou(
                pred,
                gt_sem,
                num_classes=getattr(self.cfg, "seg_num_classes", 40),
                ignore_index=getattr(self.cfg, "seg_ignore_index", 255),
            )
        return losses

    def _normalize_positions(self, position_map: torch.Tensor) -> torch.Tensor:
        """Normalize world-space positions to [0, 1] using scene bounds from Gaussians."""
        if not hasattr(self, "_scene_bounds"):
            xyz = self.model.get_xyz()
            margin = 0.1
            self._scene_bounds = (
                xyz.min(dim=0).values - margin,
                xyz.max(dim=0).values + margin,
            )
        lo, hi = self._scene_bounds
        extent = (hi - lo).clamp(min=1e-6)
        # position_map: [B, 3, H, W]
        lo_v = lo.view(1, 3, 1, 1)
        extent_v = extent.view(1, 3, 1, 1)
        return ((position_map - lo_v) / extent_v).clamp(0.0, 1.0)

    def _build_guide(
        self,
        batch: dict,
        render_result: dict,
        rendered_rgb: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Build the guide signal for the screen-space refiner.

        Supports RGB guide (GT or self-rendered), depth guide, or both.
        Returns None if no guide is configured.
        """
        parts = []

        if self.refiner_rgb_guide:
            if self.self_guided and rendered_rgb is not None:
                parts.append(rendered_rgb.detach())
            else:
                rgb = batch.get("rgb_guide")
                if rgb is not None:
                    parts.append(rgb.to(self.device))
                else:
                    # Zero-pad if RGB not available
                    B, _, H, W = render_result["feature_map"].shape
                    parts.append(torch.zeros(B, 3, H, W, device=self.device))

        if self.refiner_depth_guide:
            depth = render_result["depth_map"]  # [B, H, W]
            depth = depth.unsqueeze(1)           # [B, 1, H, W]
            # Normalize depth to [0, 1] per-batch for stable input
            dmin = depth.amin(dim=(2, 3), keepdim=True)
            dmax = depth.amax(dim=(2, 3), keepdim=True)
            depth = (depth - dmin) / (dmax - dmin + 1e-6)

            if getattr(self.cfg, "refiner_depth_grad", False):
                # 3ch guide: depth + spatial gradients (edge info)
                dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
                dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
                dx = F.pad(dx, (0, 1, 0, 0))  # pad right
                dy = F.pad(dy, (0, 0, 0, 1))  # pad bottom
                # Scale gradients for visibility (they're typically small)
                dx = dx * 10.0
                dy = dy * 10.0
                parts.append(torch.cat([depth, dx, dy], dim=1))  # 3ch
            else:
                parts.append(depth)  # 1ch

        if not parts:
            return None
        return torch.cat(parts, dim=1)

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
            if self.use_refiner and self.refiner is not None:
                rgb_guide = sample.get("rgb_guide")
                if rgb_guide is not None:
                    rgb_guide = rgb_guide.unsqueeze(0).to(self.device)
                rendered = self.refiner(rendered, guide=rgb_guide)
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
    else:
        # Check config for resume_from / warmstart_from
        resume_from = getattr(config, "resume_from", None) or None
        warmstart_from = getattr(config, "warmstart_from", None) or None
        if resume_from:
            trainer.load_checkpoint(resume_from, resume=False)
            trainer._log(f"Warmstart from config: {resume_from}")
        elif warmstart_from:
            trainer.load_checkpoint(warmstart_from, resume=False)

    trainer.train()


if __name__ == "__main__":
    main()
