#!/usr/bin/env python3
"""
Scene Coordinate Regression (SCR) Training Script
==================================================
Trains a SceneCoordNet to predict 3D world coordinates from RADIO features,
then evaluates via PnP+RANSAC pose solving.

Pipeline:
  1. Load pre-cached RADIO fine_geo features (64, 68, 120)
  2. Load pre-rendered depth cache (depth, valid_mask, pose_w2c)
  3. Compute GT 3D world coordinates via unprojection
  4. Train: SCR_network(features) → predicted_3D  (B, 3, H, W)
  5. Loss: smooth_L1 on valid pixels + confidence BCE
  6. Eval: PnP+RANSAC pose from predicted coordinates

Usage:
    python scripts/train_scene_coord.py --config configs/scr_radio_oh_v1.yaml
    python scripts/train_scene_coord.py --config configs/scr_radio_oh_v1.yaml --gpu 1
"""

import os
import sys
import math
import time
import argparse
import logging
import glob as glob_mod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from ic_models.scene_coord_net import SceneCoordNet, SceneCoordNetV2
from data.radio_loc_dataset import (
    read_colmap_cameras,
    read_colmap_images,
    colmap_to_w2c,
    camera_params_to_intrinsics,
)

logger = logging.getLogger(__name__)


# =============================================================================
#  Geometry Utilities
# =============================================================================

def compute_world_coords(
    depth: torch.Tensor,
    pose_w2c: torch.Tensor,
    intrinsics: Dict[str, float],
) -> torch.Tensor:
    """Unproject depth map to 3D world coordinates.

    Args:
        depth: (H, W) depth map.
        pose_w2c: (4, 4) world-to-camera pose.
        intrinsics: dict with fx, fy, cx, cy.

    Returns:
        (3, H, W) world coordinates.
    """
    H, W = depth.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    u = torch.arange(W, dtype=torch.float32, device=depth.device)
    v = torch.arange(H, dtype=torch.float32, device=depth.device)
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')

    X_cam = (u_grid - cx) / fx * depth
    Y_cam = (v_grid - cy) / fy * depth
    Z_cam = depth

    pts_cam = torch.stack([X_cam, Y_cam, Z_cam], dim=-1)  # (H, W, 3)

    c2w = torch.linalg.inv(pose_w2c.float())
    R = c2w[:3, :3]
    t = c2w[:3, 3]

    pts_world = torch.einsum('ij,hwj->hwi', R, pts_cam) + t

    return pts_world.permute(2, 0, 1)  # (3, H, W)


def compute_pose_error(
    pose_pred: np.ndarray,
    pose_gt: np.ndarray,
) -> Tuple[float, float]:
    """Compute rotation (degrees) and translation (mm) errors.

    Args:
        pose_pred: (4, 4) predicted w2c pose.
        pose_gt: (4, 4) ground-truth w2c pose.

    Returns:
        (rot_err_deg, trans_err_mm)
    """
    R_pred = pose_pred[:3, :3]
    R_gt = pose_gt[:3, :3]
    R_rel = R_pred.T @ R_gt
    trace = np.clip(np.trace(R_rel), -1.0 + 1e-7, 3.0 - 1e-7)
    rot_err = np.degrees(np.arccos((trace - 1.0) / 2.0))

    t_pred = pose_pred[:3, 3]
    t_gt = pose_gt[:3, 3]
    trans_err = np.linalg.norm(t_pred - t_gt) * 1000.0  # mm

    return float(rot_err), float(trans_err)


# =============================================================================
#  Dataset
# =============================================================================

class SCRDataset(Dataset):
    """Dataset for Scene Coordinate Regression training.

    Loads pre-cached RADIO features and depth maps, computes GT 3D coordinates.
    Normalizes coordinates using training-set statistics for stable learning.

    Args:
        feature_dir: root dir with fine_geo/ subdir.
        depth_cache_dir: path to pre-rendered depth cache (.pt dicts).
        colmap_dir: COLMAP sparse model (cameras.bin, images.bin).
        split_file: Cambridge Landmarks format split file.
        intrinsics: dict with fx, fy, cx, cy at feature resolution.
        feature_hw: expected feature resolution (H, W).
        coord_mean: (3,) mean for coordinate normalization (None = compute).
        coord_std: (3,) std for coordinate normalization (None = compute).
    """

    def __init__(
        self,
        feature_dir: str,
        depth_cache_dir: str,
        colmap_dir: str,
        split_file: str,
        intrinsics: Dict[str, float],
        feature_hw: Tuple[int, int] = (68, 120),
        coarse_hw: Optional[Tuple[int, int]] = None,
        coord_mean: Optional[torch.Tensor] = None,
        coord_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.feature_hw = tuple(feature_hw)
        self.coarse_hw = tuple(coarse_hw) if coarse_hw else None
        self.intrinsics = intrinsics
        H, W = self.feature_hw

        # ------------------------------------------------------------------
        # Parse split file (Cambridge Landmarks format)
        # ------------------------------------------------------------------
        split_names = set()
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Visual') or line.startswith('ImageFile'):
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    img_name = parts[0]
                    base = os.path.splitext(img_name)[0]
                    split_names.add(base + '.png')
                    split_names.add(base + '.jpg')

        # ------------------------------------------------------------------
        # Load COLMAP model
        # ------------------------------------------------------------------
        cameras = read_colmap_cameras(os.path.join(colmap_dir, 'cameras.bin'))
        images = read_colmap_images(os.path.join(colmap_dir, 'images.bin'))

        # ------------------------------------------------------------------
        # Auto-detect feature file naming
        # ------------------------------------------------------------------
        fine_geo_dir = os.path.join(feature_dir, 'fine_geo')
        if not os.path.isdir(fine_geo_dir):
            raise RuntimeError(f"Feature directory not found: {fine_geo_dir}")

        sample_file = os.listdir(fine_geo_dir)[0] if os.listdir(fine_geo_dir) else ''
        use_colmap_id_naming = sample_file.startswith('rgb_')

        # Coarse feature directory (optional, for v2)
        coarse_sem_dir = os.path.join(feature_dir, 'coarse_sem')
        has_coarse = self.coarse_hw is not None and os.path.isdir(coarse_sem_dir)
        if self.coarse_hw is not None and not os.path.isdir(coarse_sem_dir):
            logger.warning(
                f"Coarse feature dir not found: {coarse_sem_dir}. "
                "Falling back to no coarse features."
            )
            has_coarse = False
        self.has_coarse = has_coarse

        # ------------------------------------------------------------------
        # Build sample list
        # ------------------------------------------------------------------
        depth_cache_exists = os.path.isdir(depth_cache_dir)
        if not depth_cache_exists:
            logger.warning(
                f"Depth cache dir not found: {depth_cache_dir}. "
                "Samples without depth will be skipped."
            )

        self.samples: List[Dict] = []
        skipped_no_feature = 0
        skipped_no_depth = 0

        for img_id in sorted(images.keys()):
            meta = images[img_id]
            if meta.name not in split_names:
                continue

            # Find feature file
            if use_colmap_id_naming:
                fine_pattern = os.path.join(
                    fine_geo_dir, f'rgb_{img_id}_fine_geo_*.pt')
                fine_matches = glob_mod.glob(fine_pattern)
                if not fine_matches:
                    skipped_no_feature += 1
                    continue
                fine_path = fine_matches[0]

                # Coarse feature (optional)
                coarse_path = None
                if has_coarse:
                    coarse_pattern = os.path.join(
                        coarse_sem_dir, f'rgb_{img_id}_coarse_sem_*.pt')
                    coarse_matches = glob_mod.glob(coarse_pattern)
                    if coarse_matches:
                        coarse_path = coarse_matches[0]
            else:
                stem = os.path.splitext(os.path.basename(meta.name))[0]
                fine_path = os.path.join(fine_geo_dir, f'{stem}.pt')
                if not os.path.isfile(fine_path):
                    skipped_no_feature += 1
                    continue

                coarse_path = None
                if has_coarse:
                    cp = os.path.join(coarse_sem_dir, f'{stem}.pt')
                    if os.path.isfile(cp):
                        coarse_path = cp

            # Find depth cache file — indexed by sanitized image name
            stem_for_depth = meta.name.replace('/', '_').replace('\\', '_')
            stem_for_depth = os.path.splitext(stem_for_depth)[0]
            depth_path = os.path.join(depth_cache_dir, f'{stem_for_depth}.pt')
            if not os.path.isfile(depth_path):
                skipped_no_depth += 1
                continue

            pose_w2c = colmap_to_w2c(meta.qvec, meta.tvec).astype(np.float32)

            self.samples.append({
                'img_id': img_id,
                'image_name': meta.name,
                'fine_path': fine_path,
                'coarse_path': coarse_path,
                'depth_path': depth_path,
                'pose_w2c': pose_w2c,
            })

        if skipped_no_feature > 0:
            logger.warning(f"Skipped {skipped_no_feature} images (no features)")
        if skipped_no_depth > 0:
            logger.warning(f"Skipped {skipped_no_depth} images (no depth cache)")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found. feature_dir={feature_dir}, "
                f"depth_cache_dir={depth_cache_dir}, split_file={split_file}"
            )

        # ------------------------------------------------------------------
        # Pre-load everything into memory (features are small: ~64×68×120×2B)
        # ------------------------------------------------------------------
        logger.info(f"Loading {len(self.samples)} samples into memory...")
        self._features: List[torch.Tensor] = []
        self._coarse_features: List[Optional[torch.Tensor]] = []
        self._world_coords: List[torch.Tensor] = []
        self._valid_masks: List[torch.Tensor] = []
        self._poses_w2c: List[torch.Tensor] = []

        H_c, W_c = self.coarse_hw if self.coarse_hw else (0, 0)

        for s in self.samples:
            feat = torch.load(s['fine_path'], map_location='cpu', weights_only=True)
            feat = feat.float()
            if feat.shape[1:] != (H, W):
                feat = F.interpolate(
                    feat.unsqueeze(0), (H, W),
                    mode='bilinear', align_corners=False,
                ).squeeze(0)
            self._features.append(feat)

            # Coarse features (optional)
            if has_coarse and s['coarse_path'] is not None:
                cfeat = torch.load(
                    s['coarse_path'], map_location='cpu', weights_only=True)
                cfeat = cfeat.float()
                if cfeat.shape[1:] != (H_c, W_c):
                    cfeat = F.interpolate(
                        cfeat.unsqueeze(0), (H_c, W_c),
                        mode='bilinear', align_corners=False,
                    ).squeeze(0)
                self._coarse_features.append(cfeat)
            else:
                self._coarse_features.append(None)

            depth_data = torch.load(
                s['depth_path'], map_location='cpu', weights_only=True)
            depth = depth_data['depth'].float()
            valid = depth_data['valid_mask'].bool()
            pose_w2c_t = torch.from_numpy(s['pose_w2c']).float()

            # Additional validity: depth must be > 0.1m
            valid = valid & (depth > 0.1)

            world_coords = compute_world_coords(depth, pose_w2c_t, intrinsics)
            self._world_coords.append(world_coords)
            self._valid_masks.append(valid)
            self._poses_w2c.append(pose_w2c_t)

        # ------------------------------------------------------------------
        # Compute or accept normalization statistics
        # ------------------------------------------------------------------
        if coord_mean is not None and coord_std is not None:
            self.coord_mean = coord_mean
            self.coord_std = coord_std
        else:
            logger.info("Computing coordinate normalization statistics...")
            all_valid_coords = []
            for wc, vm in zip(self._world_coords, self._valid_masks):
                # wc: (3, H, W), vm: (H, W)
                valid_pts = wc[:, vm]  # (3, N_valid)
                all_valid_coords.append(valid_pts)
            all_valid_coords = torch.cat(all_valid_coords, dim=1)  # (3, N)
            self.coord_mean = all_valid_coords.mean(dim=1)  # (3,)
            self.coord_std = all_valid_coords.std(dim=1).clamp(min=1e-3)  # (3,)
            logger.info(
                f"  coord_mean={self.coord_mean.tolist()}, "
                f"coord_std={self.coord_std.tolist()}"
            )

        # Normalize stored world coords
        mean = self.coord_mean.view(3, 1, 1)
        std = self.coord_std.view(3, 1, 1)
        self._world_coords_norm = [
            (wc - mean) / std for wc in self._world_coords
        ]

        logger.info(
            f"SCRDataset ready: {len(self.samples)} samples, "
            f"feature_hw={self.feature_hw}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'features': self._features[idx],               # (64, H, W)
            'world_coords_norm': self._world_coords_norm[idx],  # (3, H, W)
            'world_coords': self._world_coords[idx],       # (3, H, W)
            'valid_mask': self._valid_masks[idx],           # (H, W)
            'pose_w2c': self._poses_w2c[idx],               # (4, 4)
            'image_name': self.samples[idx]['image_name'],
        }
        if self.has_coarse and self._coarse_features[idx] is not None:
            item['coarse_features'] = self._coarse_features[idx]
        return item


def scr_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate that stacks tensors and preserves strings."""
    collated = {}
    for key in batch[0]:
        vals = [d[key] for d in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[key] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], str):
            collated[key] = vals
        else:
            collated[key] = vals
    return collated


# =============================================================================
#  Visualization
# =============================================================================

def save_coord_visualization(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence: torch.Tensor,
    save_path: str,
    coord_mean: torch.Tensor,
    coord_std: torch.Tensor,
) -> None:
    """Save visualization of predicted vs GT 3D coordinates and confidence.

    Saves a simple .pt file with the tensors (avoids matplotlib dependency).
    For actual image visualization, use a separate notebook/script.

    Args:
        pred_coords: (3, H, W) normalized predicted coords.
        gt_coords: (3, H, W) normalized GT coords.
        valid_mask: (H, W) valid pixel mask.
        confidence: (1, H, W) confidence map.
        save_path: output .pt path.
        coord_mean: (3,) normalization mean.
        coord_std: (3,) normalization std.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        mean = coord_mean.view(3, 1, 1)
        std = coord_std.view(3, 1, 1)
        pred_denorm = (pred_coords.cpu() * std + mean)
        gt_denorm = (gt_coords.cpu() * std + mean)

        # Normalize XYZ to [0, 1] for RGB visualization
        all_coords = torch.cat([
            pred_denorm[:, valid_mask.cpu()],
            gt_denorm[:, valid_mask.cpu()],
        ], dim=1)
        if all_coords.shape[1] > 0:
            vmin = all_coords.min(dim=1, keepdim=True).values.view(3, 1, 1)
            vmax = all_coords.max(dim=1, keepdim=True).values.view(3, 1, 1)
            rng = (vmax - vmin).clamp(min=1e-3)
        else:
            vmin, rng = torch.zeros(3, 1, 1), torch.ones(3, 1, 1)

        pred_rgb = ((pred_denorm - vmin) / rng).clamp(0, 1)
        gt_rgb = ((gt_denorm - vmin) / rng).clamp(0, 1)

        # Mask invalid pixels
        mask_3 = valid_mask.cpu().unsqueeze(0).expand(3, -1, -1).float()
        pred_rgb = pred_rgb * mask_3
        gt_rgb = gt_rgb * mask_3

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(gt_rgb.permute(1, 2, 0).numpy())
        axes[0].set_title('GT Coordinates (XYZ→RGB)')
        axes[0].axis('off')

        axes[1].imshow(pred_rgb.permute(1, 2, 0).numpy())
        axes[1].set_title('Predicted Coordinates')
        axes[1].axis('off')

        axes[2].imshow(confidence.squeeze(0).cpu().numpy(), cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('Confidence')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    except ImportError:
        # Fallback: save raw tensors
        torch.save({
            'pred_coords': pred_coords.cpu(),
            'gt_coords': gt_coords.cpu(),
            'valid_mask': valid_mask.cpu(),
            'confidence': confidence.cpu(),
        }, save_path.replace('.png', '.pt'))


# =============================================================================
#  Trainer
# =============================================================================

class SCRTrainer:
    """Scene Coordinate Regression trainer."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Output directories
        exp_name = config.get('exp_name', 'scr_default')
        base_output = config.get('output_dir', 'output')
        self.output_dir = Path(base_output) / exp_name
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.vis_dir = self.output_dir / 'visualizations'
        self.log_dir = self.output_dir / 'logs'
        for d in [self.ckpt_dir, self.vis_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Training config
        tc = config.get('training', {})
        self.total_epochs = tc.get('epochs', 200)
        self.grad_clip = tc.get('grad_clip', 1.0)
        self.val_every = tc.get('val_every', 5)
        self.save_every = tc.get('save_every', 10)
        self.vis_every = tc.get('vis_every', 10)
        self.num_vis_samples = tc.get('num_vis_samples', 4)

        loss_cfg = tc.get('loss', {})
        self.coord_weight = loss_cfg.get('coord_weight', 1.0)
        self.confidence_weight = loss_cfg.get('confidence_weight', 0.1)
        self.reproj_threshold = loss_cfg.get('reproj_threshold', 10.0)

        # Intrinsics
        ic = config.get('intrinsics', {})
        self.intrinsics = {
            'fx': ic.get('fx', 104.6),
            'fy': ic.get('fy', 105.4),
            'cx': ic.get('cx', 60.0),
            'cy': ic.get('cy', 34.0),
        }

        # Initialize components
        self._init_datasets()
        self._init_model()
        self._init_optimizer()

        # AMP
        self.scaler = GradScaler()

        # Tracking
        self.epoch = 0
        self.global_step = 0
        self.best_trans_median = float('inf')

    def _init_datasets(self):
        dc = self.config.get('dataset', {})
        tc = self.config.get('training', {})
        feature_hw = tuple(dc.get('feature_hw', [68, 120]))
        coarse_hw_cfg = dc.get('coarse_hw', None)
        coarse_hw = tuple(coarse_hw_cfg) if coarse_hw_cfg else None

        logger.info("Loading training dataset...")
        self.train_dataset = SCRDataset(
            feature_dir=dc['feature_dir'],
            depth_cache_dir=dc['depth_cache_dir'],
            colmap_dir=dc['colmap_dir'],
            split_file=dc['train_split'],
            intrinsics=self.intrinsics,
            feature_hw=feature_hw,
            coarse_hw=coarse_hw,
        )

        logger.info("Loading test dataset...")
        self.test_dataset = SCRDataset(
            feature_dir=dc['feature_dir'],
            depth_cache_dir=dc['depth_cache_dir'],
            colmap_dir=dc['colmap_dir'],
            split_file=dc['test_split'],
            intrinsics=self.intrinsics,
            feature_hw=feature_hw,
            coarse_hw=coarse_hw,
            coord_mean=self.train_dataset.coord_mean,
            coord_std=self.train_dataset.coord_std,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=tc.get('batch_size', 32),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=scr_collate_fn,
        )
        self.val_loader = DataLoader(
            self.test_dataset,
            batch_size=tc.get('batch_size', 32),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=scr_collate_fn,
        )

        logger.info(
            f"  Train: {len(self.train_dataset)} samples, "
            f"Test: {len(self.test_dataset)} samples"
        )

    def _init_model(self):
        mc = self.config.get('model', {})
        model_version = mc.get('version', 1)
        self.model_version = model_version

        if model_version == 2:
            self.model = SceneCoordNetV2(
                in_channels=mc.get('feature_dim', 64),
                hidden_channels=mc.get('hidden_dim', 256),
                num_res_blocks=mc.get('num_res_blocks', 6),
                use_pos_encoding=mc.get('use_position_encoding', True),
            ).to(self.device)
        else:
            self.model = SceneCoordNet(
                in_channels=mc.get('feature_dim', 64),
                hidden_channels=mc.get('hidden_dim', 128),
                num_res_blocks=mc.get('num_res_blocks', 5),
                use_pos_encoding=mc.get('use_position_encoding', True),
            ).to(self.device)

        logger.info(f"Model: {self.model}")
        logger.info(f"  Parameters: {self.model.param_count():,}")

    def _init_optimizer(self):
        tc = self.config.get('training', {})
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=tc.get('lr', 1e-3),
            weight_decay=tc.get('weight_decay', 1e-5),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=tc.get('epochs', 200),
            eta_min=1e-6,
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        pred_coords: torch.Tensor,
        pred_confidence: torch.Tensor,
        gt_coords_norm: torch.Tensor,
        gt_coords: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SCR training loss.

        Args:
            pred_coords: (B, 3, H, W) predicted normalized 3D coordinates.
            pred_confidence: (B, 1, H, W) predicted confidence in [0, 1].
            gt_coords_norm: (B, 3, H, W) GT normalized coordinates.
            gt_coords: (B, 3, H, W) GT coordinates (unnormalized, for reproj).
            valid_mask: (B, H, W) boolean mask.

        Returns:
            (loss, metrics_dict)
        """
        B = pred_coords.shape[0]
        mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        n_valid = mask.sum().clamp(min=1.0)

        # --- Coordinate loss: smooth L1 on valid pixels ---
        coord_diff = pred_coords - gt_coords_norm
        coord_loss_map = F.smooth_l1_loss(
            pred_coords, gt_coords_norm, reduction='none')  # (B, 3, H, W)
        coord_loss = (coord_loss_map * mask).sum() / (n_valid * 3.0)

        # --- Confidence loss: BCE with reprojection error supervision ---
        # Denormalize predictions for reprojection error
        mean = self.train_dataset.coord_mean.to(pred_coords.device).view(1, 3, 1, 1)
        std = self.train_dataset.coord_std.to(pred_coords.device).view(1, 3, 1, 1)
        pred_denorm = pred_coords * std + mean

        # Per-pixel L2 error in world coords as proxy for reprojection quality
        coord_err = torch.norm(pred_denorm - gt_coords, dim=1, keepdim=True)  # (B, 1, H, W)

        # Convert world-coord error to approximate pixel error using average focal length
        avg_depth = gt_coords[:, 2:3, :, :].clamp(min=0.5)  # Z component
        avg_f = (self.intrinsics['fx'] + self.intrinsics['fy']) / 2.0
        reproj_err_approx = coord_err * avg_f / avg_depth  # approximate pixel error

        # Target confidence: 1 where reproj error < threshold, 0 otherwise
        conf_target = (reproj_err_approx < self.reproj_threshold).float()
        conf_target = conf_target.detach()

        conf_loss = F.binary_cross_entropy(
            pred_confidence, conf_target, reduction='none')  # (B, 1, H, W)
        conf_loss = (conf_loss * mask).sum() / n_valid

        # --- Total ---
        total = self.coord_weight * coord_loss + self.confidence_weight * conf_loss

        metrics = {
            'total_loss': total.item(),
            'coord_loss': coord_loss.item(),
            'conf_loss': conf_loss.item(),
            'conf_mean': pred_confidence.mean().item(),
            'coord_err_mean': coord_err[valid_mask.unsqueeze(1).expand_as(coord_err)].mean().item()
                if valid_mask.any() else 0.0,
        }

        return total, metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_metrics: Dict[str, List[float]] = {}

        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            gt_coords_norm = batch['world_coords_norm'].to(self.device)
            gt_coords = batch['world_coords'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                if self.model_version == 2 and 'coarse_features' in batch:
                    coarse = batch['coarse_features'].to(self.device)
                    pred_coords, pred_conf = self.model(features, coarse)
                else:
                    pred_coords, pred_conf = self.model(features)

            # Loss in fp32 for numerical stability
            pred_coords_f32 = pred_coords.float()
            pred_conf_f32 = pred_conf.float()

            loss, metrics = self.compute_loss(
                pred_coords_f32, pred_conf_f32,
                gt_coords_norm, gt_coords, valid_mask,
            )

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1

            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

        avg = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

        logger.info(
            f"[Train E{epoch:03d}]  loss={avg.get('total_loss', 0):.4f}  "
            f"coord={avg.get('coord_loss', 0):.4f}  "
            f"conf={avg.get('conf_loss', 0):.4f}  "
            f"conf_mean={avg.get('conf_mean', 0):.3f}  "
            f"coord_err={avg.get('coord_err_mean', 0):.3f}m"
        )

        return avg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        torch.cuda.empty_cache()

        all_rot_errs = []
        all_trans_errs = []
        all_coord_losses = []
        all_successes = []

        # Denormalization params
        mean = self.train_dataset.coord_mean.to(self.device).view(1, 3, 1, 1)
        std = self.train_dataset.coord_std.to(self.device).view(1, 3, 1, 1)

        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            gt_coords_norm = batch['world_coords_norm'].to(self.device)
            gt_coords = batch['world_coords'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            poses_gt = batch['pose_w2c'].numpy()

            # Forward pass
            if self.model_version == 2 and 'coarse_features' in batch:
                coarse = batch['coarse_features'].to(self.device)
                pred_coords, pred_conf = self.model(features, coarse)
            else:
                pred_coords, pred_conf = self.model(features)
            pred_coords = pred_coords.float()
            pred_conf = pred_conf.float()

            # Coordinate loss for monitoring
            mask_f = valid_mask.unsqueeze(1).float()
            n_valid = mask_f.sum().clamp(min=1.0)
            coord_loss = (
                F.smooth_l1_loss(pred_coords, gt_coords_norm, reduction='none') * mask_f
            ).sum() / (n_valid * 3.0)
            all_coord_losses.append(coord_loss.item())

            # Denormalize for PnP
            pred_denorm = pred_coords * std + mean

            # PnP+RANSAC per sample
            B = features.shape[0]
            for b in range(B):
                coords_3hw = pred_denorm[b].cpu().numpy()  # (3, H, W)
                conf_hw = pred_conf[b, 0].cpu().numpy()     # (H, W)
                pose_gt_b = poses_gt[b]                      # (4, 4)

                H, W = conf_hw.shape
                K = np.array([
                    [self.intrinsics['fx'], 0.0, self.intrinsics['cx']],
                    [0.0, self.intrinsics['fy'], self.intrinsics['cy']],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)

                # Use pixel centers
                us = np.arange(W, dtype=np.float64) + 0.5
                vs = np.arange(H, dtype=np.float64) + 0.5
                grid_v, grid_u = np.meshgrid(vs, us, indexing='ij')
                pixels = np.stack([grid_u, grid_v], axis=-1)  # (H, W, 2)

                pose_pred, n_inliers, ok = SceneCoordNet._solve_pnp_single(
                    coords_3hw, conf_hw, pixels, K, conf_thresh=0.5,
                )

                all_successes.append(ok)
                if ok:
                    rot_err, trans_err = compute_pose_error(pose_pred, pose_gt_b)
                    all_rot_errs.append(rot_err)
                    all_trans_errs.append(trans_err)

        # Aggregate metrics
        val_metrics: Dict[str, float] = {}
        val_metrics['val_coord_loss'] = float(np.mean(all_coord_losses))
        val_metrics['val_pnp_success_rate'] = float(np.mean(all_successes)) * 100

        if all_rot_errs:
            rot = np.array(all_rot_errs)
            trans = np.array(all_trans_errs)
            val_metrics.update({
                'val_rot_mean': float(np.mean(rot)),
                'val_rot_median': float(np.median(rot)),
                'val_trans_mean': float(np.mean(trans)),
                'val_trans_median': float(np.median(trans)),
                'val_pct_1deg_50mm': float(np.mean((rot < 1.0) & (trans < 50.0)) * 100),
                'val_pct_5deg_100mm': float(np.mean((rot < 5.0) & (trans < 100.0)) * 100),
            })

            logger.info(
                f"[Val  E{epoch:03d}]  "
                f"rot={val_metrics['val_rot_mean']:.2f}° "
                f"(med {val_metrics['val_rot_median']:.2f}°)  "
                f"trans={val_metrics['val_trans_mean']:.1f}mm "
                f"(med {val_metrics['val_trans_median']:.1f}mm)  "
                f"<1°/50mm={val_metrics['val_pct_1deg_50mm']:.1f}%  "
                f"<5°/100mm={val_metrics['val_pct_5deg_100mm']:.1f}%  "
                f"PnP_ok={val_metrics['val_pnp_success_rate']:.1f}%  "
                f"coord_loss={val_metrics['val_coord_loss']:.4f}"
            )
        else:
            logger.warning(
                f"[Val  E{epoch:03d}]  No successful PnP solves! "
                f"coord_loss={val_metrics['val_coord_loss']:.4f}"
            )

        return val_metrics

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, epoch: int) -> None:
        self.model.eval()
        n = min(self.num_vis_samples, len(self.test_dataset))

        mean = self.train_dataset.coord_mean
        std = self.train_dataset.coord_std

        for i in range(n):
            sample = self.test_dataset[i]
            feat = sample['features'].unsqueeze(0).to(self.device)

            with torch.no_grad(), autocast():
                if self.model_version == 2 and 'coarse_features' in sample:
                    coarse = sample['coarse_features'].unsqueeze(0).to(self.device)
                    pred_coords, pred_conf = self.model(feat, coarse)
                else:
                    pred_coords, pred_conf = self.model(feat)

            save_path = str(self.vis_dir / f'epoch{epoch:03d}_sample{i}.png')
            save_coord_visualization(
                pred_coords=pred_coords[0].float().cpu(),
                gt_coords=sample['world_coords_norm'],
                valid_mask=sample['valid_mask'],
                confidence=pred_conf[0].float().cpu(),
                save_path=save_path,
                coord_mean=mean,
                coord_std=std,
            )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_trans_median': self.best_trans_median,
            'coord_mean': self.train_dataset.coord_mean,
            'coord_std': self.train_dataset.coord_std,
            'config': self.config,
        }
        torch.save(ckpt, self.ckpt_dir / 'latest.pth')
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')
        if epoch % self.save_every == 0:
            torch.save(ckpt, self.ckpt_dir / f'epoch_{epoch:03d}.pth')

    def _load_checkpoint(self, path: str):
        logger.info(f"[Resume] Loading from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.epoch = ckpt['epoch'] + 1
        self.global_step = ckpt.get('global_step', 0)
        self.best_trans_median = ckpt.get('best_trans_median', float('inf'))
        logger.info(f"  Resumed at epoch {self.epoch}, step {self.global_step}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, resume_path: Optional[str] = None):
        if resume_path:
            self._load_checkpoint(resume_path)

        logger.info(
            f"\n{'=' * 60}\n"
            f"  Scene Coordinate Regression Training\n"
            f"  Model: {self.model}\n"
            f"  Epochs: {self.total_epochs}\n"
            f"  Batch size: {self.config.get('training', {}).get('batch_size', 32)}\n"
            f"  LR: {self.config.get('training', {}).get('lr', 1e-3)}\n"
            f"  Output: {self.output_dir}\n"
            f"{'=' * 60}"
        )

        epoch_times = []
        train_start = time.time()

        for epoch in range(self.epoch, self.total_epochs):
            self.epoch = epoch
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = {}
            if (epoch + 1) % self.val_every == 0 or epoch == self.total_epochs - 1:
                val_metrics = self.validate(epoch)

            # Visualize
            if (epoch + 1) % self.vis_every == 0 or epoch == self.total_epochs - 1:
                self.visualize(epoch)

            # Scheduler
            self.scheduler.step()

            # Checkpoint
            is_best = False
            trans_med = val_metrics.get('val_trans_median', float('inf'))
            if trans_med < self.best_trans_median:
                self.best_trans_median = trans_med
                is_best = True
                logger.info(
                    f"  ★ New best: trans_median={trans_med:.1f}mm  "
                    f"rot_median={val_metrics.get('val_rot_median', -1):.2f}°"
                )

            self._save_checkpoint(epoch, is_best)

            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            avg_epoch = np.mean(epoch_times[-5:])
            remaining = (self.total_epochs - epoch - 1) * avg_epoch
            eta_min = remaining / 60
            total_elapsed = (time.time() - train_start) / 60

            logger.info(
                f"  Epoch {epoch}: {elapsed:.0f}s  "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}  "
                f"ETA: {eta_min:.0f}min  [{total_elapsed:.0f}min elapsed]"
            )

        total_time = (time.time() - train_start) / 60
        logger.info(
            f"\nTraining complete in {total_time:.0f}min! "
            f"Best trans_median: {self.best_trans_median:.1f}mm"
        )


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Scene Coordinate Regression network')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID')
    args = parser.parse_args()

    # GPU selection
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
        ],
    )

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Add file handler after we know output dir
    exp_name = config.get('exp_name', 'scr_default')
    base_output = config.get('output_dir', 'output')
    log_file = Path(base_output) / exp_name / f'{exp_name}_train.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S'))
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Config: {args.config}")
    logger.info(f"Experiment: {exp_name}")

    trainer = SCRTrainer(config)
    trainer.train(resume_path=args.resume)


if __name__ == '__main__':
    main()
