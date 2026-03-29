#!/usr/bin/env python3
"""
RadioPoseNet 训练 + 评估脚本 (OldHospital)
=============================================

Training pipeline:
  1. 加载 RADIO PCA 64d 查询特征 (预提取, 从磁盘)
  2. 加载 2DGS 模型 (几何 + 64d 特征) 用于实时渲染参考特征和深度
  3. 训练: 添加位姿噪声 → 渲染参考特征/深度 → RadioPoseNet → flow + pose loss
  4. 评估: 最近邻初始化 → render → forward → outer-loop 精化

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_radio_pose.py \\
        --config configs/radio_pose_oh.yaml

    # 仅评估
    CUDA_VISIBLE_DEVICES=0 python scripts/train_radio_pose.py \\
        --config configs/radio_pose_oh.yaml --eval_only \\
        --checkpoint output/radio_pose_oh/checkpoints/best.pth
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_models.radio_pose_net import RadioPoseNet
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer
from modules.lie_algebra import se3_exp, se3_log


# ==============================================================================
#  Dataset: RADIO features + poses for Cambridge Landmarks
# ==============================================================================

class RadioPoseDataset(Dataset):
    """
    Dataset for RADIO-based pose estimation on Cambridge Landmarks.

    Loads:
      - RADIO PCA 64d features from disk
      - GT camera poses (w2c) from traj_w_c.txt
      - Adds SE(3) noise to create training pairs

    The renderer is called in the Trainer, not here, since it needs CUDA.
    """

    def __init__(
        self,
        feature_dir: str,
        traj_path: str,
        frame_indices: Optional[List[int]] = None,
        noise_rot_deg: float = 8.0,
        noise_trans_m: float = 0.25,
        is_train: bool = True,
    ):
        self.feature_dir = Path(feature_dir)
        self.is_train = is_train
        self.noise_rot_deg = noise_rot_deg
        self.noise_trans_m = noise_trans_m

        # Discover available features
        pattern = re.compile(r'rgb_(\d+)_fine_radio_(\d+)x(\d+)x(\d+)\.pt')
        self.features = {}  # idx → filename
        feat_dir = self.feature_dir / 'fine_radio'
        for f in sorted(feat_dir.iterdir()):
            m = pattern.match(f.name)
            if m:
                idx = int(m.group(1))
                self.features[idx] = f

        # Load trajectory (c2w matrices, 16 values per line)
        poses_c2w = []
        with open(traj_path) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) == 16:
                    poses_c2w.append(np.array(vals).reshape(4, 4))
        self.poses_c2w = poses_c2w

        # Filter to available indices
        if frame_indices is not None:
            self.indices = [i for i in frame_indices if i in self.features and i < len(poses_c2w)]
        else:
            self.indices = sorted([i for i in self.features.keys() if i < len(poses_c2w)])

        print(f"  [RadioPoseDataset] {len(self.indices)} frames, "
              f"noise={noise_rot_deg:.1f}°/{noise_trans_m:.2f}m, train={is_train}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]

        # Load feature
        feat = torch.load(str(self.features[frame_idx]),
                          map_location='cpu', weights_only=True).float()  # (64, 68, 120)

        # GT pose: c2w → w2c
        c2w = torch.from_numpy(self.poses_c2w[frame_idx].astype(np.float32))
        w2c = torch.linalg.inv(c2w)

        # Add SE(3) noise for training
        if self.is_train:
            noisy_w2c = self._add_noise(w2c)
        else:
            noisy_w2c = w2c.clone()  # eval: exact GT (override externally for NN init)

        return {
            'query_feat': feat,        # (64, H, W)
            'pose_gt': w2c,            # (4, 4) GT w2c
            'initial_pose': noisy_w2c, # (4, 4) noisy w2c
            'frame_idx': frame_idx,
        }

    def _add_noise(self, pose_w2c: torch.Tensor) -> torch.Tensor:
        """Add random SE(3) noise to pose."""
        # Random rotation (axis-angle)
        axis = torch.randn(3)
        axis = axis / (axis.norm() + 1e-8)
        angle_deg = torch.empty(1).uniform_(0, self.noise_rot_deg).item()
        angle_rad = angle_deg * math.pi / 180.0
        omega = axis * angle_rad

        # Random translation
        direction = torch.randn(3)
        direction = direction / (direction.norm() + 1e-8)
        magnitude = torch.empty(1).uniform_(0, self.noise_trans_m).item()
        trans = direction * magnitude

        xi = torch.cat([trans, omega])  # (6,)
        delta_T = se3_exp(xi)  # (4, 4)

        return delta_T @ pose_w2c


def collate_radio(batch):
    return {
        'query_feat': torch.stack([b['query_feat'] for b in batch]),
        'pose_gt': torch.stack([b['pose_gt'] for b in batch]),
        'initial_pose': torch.stack([b['initial_pose'] for b in batch]),
        'frame_idx': [b['frame_idx'] for b in batch],
    }


# ==============================================================================
#  Loss Functions
# ==============================================================================

def flow_loss_fn(
    pred_flows: List[torch.Tensor],
    gt_flow: torch.Tensor,
    gt_mask: torch.Tensor,
    gamma: float = 0.8,
) -> torch.Tensor:
    """RAFT-style sequence flow loss with gamma decay."""
    n_preds = len(pred_flows)
    total = torch.tensor(0.0, device=gt_flow.device)
    for i, flow_pred in enumerate(pred_flows):
        weight = gamma ** (n_preds - 1 - i)
        if flow_pred.shape[-2:] != gt_flow.shape[-2:]:
            flow_resized = F.interpolate(gt_flow, flow_pred.shape[-2:],
                                          mode='bilinear', align_corners=False)
            mask_resized = F.interpolate(gt_mask, flow_pred.shape[-2:], mode='nearest')
            sH, sW = flow_pred.shape[-2:]
            gH, gW = gt_flow.shape[-2:]
            flow_resized[:, 0] *= sW / gW
            flow_resized[:, 1] *= sH / gH
        else:
            flow_resized = gt_flow
            mask_resized = gt_mask

        diff = (flow_pred - flow_resized).abs()
        n_valid = mask_resized.sum().clamp(min=1.0)
        loss_i = (diff * mask_resized).sum() / (n_valid * 2)
        total = total + weight * loss_i
    return total


def contrastive_matching_loss(
    q_feat: torch.Tensor,
    r_feat: torch.Tensor,
    gt_flow: torch.Tensor,
    gt_mask: torch.Tensor,
    temperature: float = 0.07,
    n_samples: int = 256,
) -> torch.Tensor:
    """InfoNCE contrastive loss for discriminative domain projections.

    For each sampled Q pixel, the GT matching R pixel (from GT flow) is the positive,
    all other R pixels are negatives. Features should be L2-normalized.

    Args:
        q_feat: (B, C, H, W) projected query features (L2-normalized)
        r_feat: (B, C, H, W) projected reference features (L2-normalized)
        gt_flow: (B, 2, H, W) GT optical flow (u, v in pixels)
        gt_mask: (B, 1, H, W) valid depth mask
        temperature: softmax temperature (lower = sharper)
        n_samples: number of Q pixels to sample per batch element
    Returns:
        loss: scalar InfoNCE loss
    """
    B, C, H, W = q_feat.shape
    N = H * W
    device = q_feat.device

    # Flatten spatial dims
    q_flat = q_feat.reshape(B, C, N)   # (B, C, N)
    r_flat = r_feat.reshape(B, C, N)   # (B, C, N)

    # Compute target R pixel for each Q pixel via GT flow
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij')
    target_x = (grid_x.unsqueeze(0) + gt_flow[:, 0]).round().long()  # (B, H, W)
    target_y = (grid_y.unsqueeze(0) + gt_flow[:, 1]).round().long()

    valid = (gt_mask.squeeze(1) > 0.5) & \
            (target_x >= 0) & (target_x < W) & \
            (target_y >= 0) & (target_y < H)
    target_idx = (target_y * W + target_x).clamp(0, N - 1)  # (B, H, W)

    loss = torch.tensor(0.0, device=device)
    count = 0

    for b in range(B):
        v = valid[b].flatten()  # (N,)
        if v.sum() < 10:
            continue

        valid_indices = v.nonzero(as_tuple=True)[0]
        # Randomly sample n_samples Q pixels
        n_pick = min(n_samples, valid_indices.shape[0])
        perm = torch.randperm(valid_indices.shape[0], device=device)[:n_pick]
        sampled = valid_indices[perm]  # indices into flattened Q

        q_sampled = q_flat[b, :, sampled]  # (C, n_pick)
        targets = target_idx[b].flatten()[sampled]  # (n_pick,)

        # Cosine similarity of sampled Q with ALL R
        sim = torch.mm(q_sampled.t(), r_flat[b]) / temperature  # (n_pick, N)

        # InfoNCE: cross-entropy with GT target
        loss = loss + F.cross_entropy(sim, targets)
        count += 1

    return loss / max(count, 1)


def confidence_regularization_loss(
    conf_fine: torch.Tensor,
    flow_fine: torch.Tensor = None,
    gt_flow: torch.Tensor = None,
    gt_mask: torch.Tensor = None,
    target_range: Tuple[float, float] = (0.05, 0.95),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    置信度正则化损失 — 防止 confidence 坍塌到 0 或 1.

    1. Coverage: 惩罚 conf 均值偏离 [target_low, target_high]
    2. Calibration: 让 confidence 与 flow 精度正相关 (高 conf → 低 flow error)

    Args:
        conf_fine: (B, 1, H, W) confidence map
        flow_fine: predicted flow (for calibration)
        gt_flow: GT flow (for calibration)
        gt_mask: valid mask
        target_range: desired conf mean range
    """
    device = conf_fine.device
    metrics = {}

    conf_mean = conf_fine.mean()
    metrics['conf_mean'] = conf_mean.item()
    metrics['conf_std'] = conf_fine.std().item()

    # Coverage: push mean into target range
    target_low, target_high = target_range
    if conf_mean < target_low:
        coverage_loss = (target_low - conf_mean) ** 2
    elif conf_mean > target_high:
        coverage_loss = (conf_mean - target_high) ** 2
    else:
        coverage_loss = torch.tensor(0.0, device=device)

    # Calibration: confidence should align with flow accuracy
    cal_loss = torch.tensor(0.0, device=device)
    if flow_fine is not None and gt_flow is not None:
        flow_err = torch.norm(flow_fine.detach() - gt_flow.detach(), dim=1, keepdim=True)
        flow_err_norm = (flow_err / 10.0).clamp(0, 1)
        target_conf = 1.0 - flow_err_norm
        if gt_mask is not None:
            n_valid = gt_mask.sum().clamp(min=1.0)
            cal_loss = ((conf_fine - target_conf).pow(2) * gt_mask).sum() / n_valid
        else:
            cal_loss = (conf_fine - target_conf).pow(2).mean()

    total = coverage_loss + 0.1 * cal_loss
    metrics['conf_coverage'] = coverage_loss.item()
    metrics['conf_cal'] = cal_loss.item()
    return total, metrics


def pose_loss_fn(
    delta_xi: torch.Tensor,
    pose_init: torch.Tensor,
    pose_gt: torch.Tensor,
    rot_weight: float = 1.0,
    trans_weight: float = 10.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cosine rotation loss + L1 translation loss."""
    with torch.amp.autocast('cuda', enabled=False):
        delta_xi_f = delta_xi.float()
        pose_init_f = pose_init.float()
        pose_gt_f = pose_gt.float()

        T_delta = se3_exp(delta_xi_f)  # (B, 4, 4)
        pred_pose = T_delta @ pose_init_f
        T_rel = pose_gt_f @ torch.linalg.inv(pred_pose)

        R_rel = T_rel[:, :3, :3]
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        rot_loss = (1.0 - cos_angle).mean()

        t_rel = T_rel[:, :3, 3]
        trans_loss = t_rel.abs().mean()

        total = rot_weight * rot_loss + trans_weight * trans_loss

        # Metrics
        angle_deg = torch.acos(cos_angle).mean().item() * 180.0 / math.pi
        trans_m = t_rel.norm(dim=1).mean().item()

    return total, {'rot_deg': angle_deg, 'trans_m': trans_m}


def xi_direct_loss(
    delta_xi: torch.Tensor,
    pose_init: torch.Tensor,
    pose_gt: torch.Tensor,
) -> torch.Tensor:
    """Direct supervision of se3 vector: L1 loss on xi_pred vs xi_gt.

    GT delta_xi = se3_log(pose_gt @ inv(pose_init))
    This gives the geometry solver a direct gradient signal without needing
    to backprop through se3_exp → pose comparison.
    """
    with torch.amp.autocast('cuda', enabled=False):
        delta_xi_f = delta_xi.float()
        pose_init_f = pose_init.float()
        pose_gt_f = pose_gt.float()

        T_gt = pose_gt_f @ torch.linalg.inv(pose_init_f)
        xi_gt = se3_log(T_gt)  # (B, 6)

        return F.l1_loss(delta_xi_f, xi_gt.detach())


# ==============================================================================
#  Rendering Helper
# ==============================================================================

class RadioRenderer:
    """Renders 64d RADIO features + depth from a 2DGS model."""

    def __init__(
        self,
        ply_path: str,
        feature_model_path: str,
        device: torch.device,
        img_hw: Tuple[int, int] = (1080, 1920),
        fx: float = 1170.0,
        fy: float = 1170.0,
        cx: float = 960.0,
        cy: float = 540.0,
    ):
        self.device = device
        self.img_hw = img_hw
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Load Gaussian model
        print(f"  [RadioRenderer] Loading PLY: {ply_path}")
        ckpt = torch.load(feature_model_path, map_location='cpu', weights_only=True)
        feat_dim = ckpt['feature_dim']

        self.gs_model = GaussianFeatureModel(feature_dim=feat_dim)
        self.gs_model.load_ply(ply_path)
        self.gs_model._loc_feature = nn.Parameter(
            ckpt['loc_feature'].to(device))
        self.gs_model = self.gs_model.to(device)
        self.gs_model.eval()

        n_gauss = self.gs_model.get_xyz.shape[0]
        print(f"  [RadioRenderer] {n_gauss} Gaussians, {feat_dim}d features")

    def render_features(
        self, viewmat: torch.Tensor,
        feat_hw: Tuple[int, int] = (68, 120),
    ) -> torch.Tensor:
        """Render 64d features at target resolution.
        Args:
            viewmat: (4, 4) or (B, 4, 4) w2c pose
        Returns:
            (D, fH, fW) or (B, D, fH, fW) feature map
        """
        batched = viewmat.dim() == 3
        if not batched:
            viewmat = viewmat.unsqueeze(0)

        B = viewmat.shape[0]
        fH, fW = feat_hw
        # Scale intrinsics to rendering resolution
        render_fx = self.fx * fW / self.img_hw[1]
        render_fy = self.fy * fH / self.img_hw[0]
        render_cx = self.cx * fW / self.img_hw[1]
        render_cy = self.cy * fH / self.img_hw[0]

        result = FeatureRenderer.render_features_batch(
            self.gs_model, viewmat,
            fx=render_fx, fy=render_fy,
            cx=render_cx, cy=render_cy,
            img_height=fH, img_width=fW,
            norm_feat_before_render=True,
            norm_feat_after_render=True,
        )
        feat = result['feature_map']  # (B, D, fH, fW)
        if not batched:
            feat = feat.squeeze(0)
        return feat

    def render_depth(
        self, viewmat: torch.Tensor,
        depth_hw: Tuple[int, int] = (68, 120),
    ) -> torch.Tensor:
        """Render depth map.
        Args:
            viewmat: (4, 4) or (B, 4, 4) w2c pose
        Returns:
            (H, W) or (B, H, W) depth map
        """
        batched = viewmat.dim() == 3
        if not batched:
            viewmat = viewmat.unsqueeze(0)

        B = viewmat.shape[0]
        dH, dW = depth_hw
        render_fx = self.fx * dW / self.img_hw[1]
        render_fy = self.fy * dH / self.img_hw[0]
        render_cx = self.cx * dW / self.img_hw[1]
        render_cy = self.cy * dH / self.img_hw[0]

        depths = []
        for i in range(B):
            d = FeatureRenderer.render_depth(
                self.gs_model, viewmat[i],
                fx=render_fx, fy=render_fy,
                cx=render_cx, cy=render_cy,
                img_height=dH, img_width=dW,
            )
            depths.append(d)
        depth = torch.stack(depths)  # (B, H, W)

        if not batched:
            depth = depth.squeeze(0)
        return depth

    def render_all(
        self, viewmat: torch.Tensor,
        feat_hw: Tuple[int, int] = (68, 120),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render features + depth. Returns (feat, depth)."""
        feat = self.render_features(viewmat, feat_hw)
        depth = self.render_depth(viewmat, feat_hw)
        return feat, depth


# ==============================================================================
#  Attention Heatmap Visualization (RGB overlay, bilateral Q↔R)
# ==============================================================================

def _build_image_index() -> List[str]:
    """Build sequential index → RGB image path mapping for OldHospital."""
    base = Path('dataset/OldHospital')
    all_images = []
    for seq_num in range(1, 10):
        seq_dir = base / f'seq{seq_num}'
        if seq_dir.is_dir():
            imgs = sorted([f for f in seq_dir.iterdir() if f.suffix == '.png'])
            all_images.extend([str(p) for p in imgs])
    return all_images


def save_attention_heatmap_rgb(
    attn_q2r: torch.Tensor,
    attn_r2q: torch.Tensor,
    coarse_hw: Tuple[int, int],
    query_img_path: str,
    rendered_feat: torch.Tensor,
    renderer,
    pose_w2c: torch.Tensor,
    save_path: str,
    query_pixels: Optional[List[Tuple[int, int]]] = None,
    title: str = '',
):
    """
    Visualize bidirectional cross-attention overlaid on RGB images.

    Creates a 2-row figure:
      Row 1: Q→R attention — for selected query pixels, show where they attend on the rendered image
      Row 2: R→Q attention — for selected reference pixels, show where they attend on the query image

    Args:
        attn_q2r: (N_q, N_r) Q attends to R
        attn_r2q: (N_r, N_q) R attends to Q
        coarse_hw: (H, W) spatial resolution
        query_img_path: path to query RGB image
        rendered_feat: (C, fH, fW) rendered feature map (use PCA→RGB for display)
        renderer: RadioRenderer for rendering RGB-like vis
        pose_w2c: (4, 4) w2c pose used for rendering
        save_path: output path
        query_pixels: list of (row, col) to highlight
        title: plot title
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    H, W = coarse_hw
    attn_q2r_np = attn_q2r.cpu().numpy()  # (N_q, N_r)
    attn_r2q_np = attn_r2q.cpu().numpy()  # (N_r, N_q)

    # Load query RGB image, resize to coarse resolution for overlay
    query_rgb = np.array(Image.open(query_img_path).resize((W, H), Image.BILINEAR)) / 255.0

    # Create a pseudo-RGB from rendered features (first 3 PCA dims)
    feat = rendered_feat.cpu().float()
    if feat.shape[0] >= 3:
        vis_feat = feat[:3]  # (3, fH, fW)
    else:
        vis_feat = feat[:1].expand(3, -1, -1)
    # Normalize to [0, 1]
    vis_feat = F.interpolate(vis_feat.unsqueeze(0), (H, W), mode='bilinear', align_corners=False)[0]
    vmin, vmax = vis_feat.min(), vis_feat.max()
    if vmax > vmin:
        vis_feat = (vis_feat - vmin) / (vmax - vmin)
    render_rgb = vis_feat.permute(1, 2, 0).numpy()  # (H, W, 3)

    if query_pixels is None:
        query_pixels = [
            (H // 4, W // 4),
            (H // 4, 3 * W // 4),
            (3 * H // 4, W // 4),
            (3 * H // 4, 3 * W // 4),
        ]

    n_pts = len(query_pixels)
    fig, axes = plt.subplots(2, n_pts + 1, figsize=(4 * (n_pts + 1), 8))

    # ── Row 0: Q→R (query pixel → where it looks in rendered map) ──
    # Overall entropy on query image
    entropy_q2r = -(attn_q2r_np * np.log(attn_q2r_np + 1e-10)).sum(axis=1).reshape(H, W)
    axes[0, 0].imshow(query_rgb)
    im0 = axes[0, 0].imshow(entropy_q2r, cmap='jet', alpha=0.5)
    axes[0, 0].set_title('Query: Attention Entropy', fontsize=9)
    axes[0, 0].axis('off')

    for i, (qr, qc) in enumerate(query_pixels):
        q_idx = qr * W + qc
        attn_map = attn_q2r_np[q_idx].reshape(H, W)
        # Show on rendered image: this is where the query pixel attends
        axes[0, i + 1].imshow(render_rgb)
        axes[0, i + 1].imshow(attn_map, cmap='hot', alpha=0.6, vmin=0)
        # Mark source pixel position on rendered image
        axes[0, i + 1].plot(qc, qr, 'c+', markersize=12, markeredgewidth=2)
        axes[0, i + 1].set_title(f'Q→R: query({qr},{qc})\non rendered map', fontsize=8)
        axes[0, i + 1].axis('off')

    # ── Row 1: R→Q (reference pixel → where it looks in query) ──
    entropy_r2q = -(attn_r2q_np * np.log(attn_r2q_np + 1e-10)).sum(axis=1).reshape(H, W)
    axes[1, 0].imshow(render_rgb)
    axes[1, 0].imshow(entropy_r2q, cmap='jet', alpha=0.5)
    axes[1, 0].set_title('Rendered: Attention Entropy', fontsize=9)
    axes[1, 0].axis('off')

    for i, (qr, qc) in enumerate(query_pixels):
        r_idx = qr * W + qc
        attn_map = attn_r2q_np[r_idx].reshape(H, W)
        # Show on query image: this is where the reference pixel attends
        axes[1, i + 1].imshow(query_rgb)
        axes[1, i + 1].imshow(attn_map, cmap='hot', alpha=0.6, vmin=0)
        axes[1, i + 1].plot(qc, qr, 'g+', markersize=12, markeredgewidth=2)
        axes[1, i + 1].set_title(f'R→Q: ref({qr},{qc})\non query image', fontsize=8)
        axes[1, i + 1].axis('off')

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
#  Trainer
# ==============================================================================

class RadioPoseTrainer:
    """Training and evaluation for RadioPoseNet on OldHospital."""

    def __init__(self, config: Dict, eval_only: bool = False,
                 checkpoint: str = None):
        self.config = config
        self.device = torch.device('cuda')
        self.eval_only = eval_only

        torch.backends.cudnn.benchmark = True

        # Output dirs
        exp_name = config.get('exp_name', 'radio_pose_oh')
        self.output_dir = Path(config.get('output_dir', f'output/{exp_name}'))
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.vis_dir = self.output_dir / 'visualizations'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        if not eval_only:
            self.log_dir = self.output_dir / 'logs'
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
            with open(self.output_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        self._init_renderer()
        self._init_model()
        if not eval_only:
            self._init_datasets()
            self._init_optimizer()

        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _init_renderer(self):
        rc = self.config['renderer']
        self.feat_hw = tuple(rc.get('feat_hw', [68, 120]))
        self.renderer = RadioRenderer(
            ply_path=rc['ply_path'],
            feature_model_path=rc['feature_model_path'],
            device=self.device,
            img_hw=tuple(rc.get('img_hw', [1080, 1920])),
            fx=rc.get('fx', 1170.0),
            fy=rc.get('fy', 1170.0),
            cx=rc.get('cx', 960.0),
            cy=rc.get('cy', 540.0),
        )
        print(f"  ✓ Renderer loaded, feat_hw={self.feat_hw}")

    def _init_model(self):
        mc = self.config.get('model', {})
        rc = self.config['renderer']
        intrinsics = {
            'fx': rc.get('fx', 1170.0),
            'fy': rc.get('fy', 1170.0),
            'cx': rc.get('cx', 960.0),
            'cy': rc.get('cy', 540.0),
        }
        self.model = RadioPoseNet(
            feat_dim=mc.get('feat_dim', 64),
            hidden_dim=mc.get('hidden_dim', 128),
            n_heads=mc.get('n_heads', 4),
            n_attn_layers=mc.get('n_attn_layers', 2),
            ffn_dim=mc.get('ffn_dim', 128),
            local_radius=mc.get('local_radius', 4),
            fine_iters=mc.get('fine_iters', 4),
            damping=mc.get('damping', 1e-3),
            coarse_hw=tuple(mc.get('coarse_hw', [17, 30])),
            fine_hw=tuple(mc.get('fine_hw', [34, 60])),
            intrinsics=intrinsics,
            img_hw=tuple(rc.get('img_hw', [1080, 1920])),
            irls_iters=mc.get('irls_iters', 0),
            conf_floor=mc.get('conf_floor', 0.0),
            detach_conf_in_solver=mc.get('detach_conf_in_solver', False),
            solver_hw=tuple(mc['solver_hw']) if 'solver_hw' in mc else None,
            sequential_solve=mc.get('sequential_solve', False),
            solver_trans_scale=mc.get('solver_trans_scale', 1.0),
            shared_projection=mc.get('shared_projection', False),
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  ✓ RadioPoseNet: {n_params / 1e6:.2f}M params, "
              f"fine_iters={mc.get('fine_iters', 4)}")

    def _init_datasets(self):
        dc = self.config['data']
        feature_dir = dc['feature_dir']
        traj_path = dc['traj_path']

        # Load train/test split indices
        train_indices = None
        test_indices = None
        train_idx_path = os.path.join(feature_dir, 'train_indices.npy')
        test_idx_path = os.path.join(feature_dir, 'test_indices.npy')

        if os.path.exists(train_idx_path):
            train_indices = np.load(train_idx_path).tolist()
            print(f"  Loaded train indices: {len(train_indices)} frames")
        if os.path.exists(test_idx_path):
            test_indices = np.load(test_idx_path).tolist()
            print(f"  Loaded test indices: {len(test_indices)} frames")

        self.train_dataset = RadioPoseDataset(
            feature_dir=feature_dir,
            traj_path=traj_path,
            frame_indices=train_indices,
            noise_rot_deg=dc.get('noise_rot_deg', 8.0),
            noise_trans_m=dc.get('noise_trans_m', 0.25),
            is_train=True,
        )
        self.val_dataset = RadioPoseDataset(
            feature_dir=feature_dir,
            traj_path=traj_path,
            frame_indices=test_indices if test_indices is not None else train_indices,
            noise_rot_deg=dc.get('val_noise_rot_deg', 8.0),
            noise_trans_m=dc.get('val_noise_trans_m', 0.25),
            is_train=True,  # enable noise for validation too
        )

        bs = dc.get('batch_size', 16)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=bs, shuffle=True,
            num_workers=dc.get('num_workers', 4),
            collate_fn=collate_radio, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=dc.get('val_batch_size', 8),
            shuffle=False, num_workers=2,
            collate_fn=collate_radio, pin_memory=True)

    def _init_optimizer(self):
        tc = self.config.get('training', {})
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=tc.get('lr', 2e-4),
            weight_decay=tc.get('weight_decay', 1e-5))
        self.total_epochs = tc.get('epochs', 100)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.total_epochs, eta_min=tc.get('min_lr', 1e-6))
        self.grad_clip = tc.get('grad_clip', 1.0)
        self.use_amp = tc.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        lc = tc.get('loss', {})
        self.flow_weight = lc.get('flow_weight', 1.0)
        self.pose_weight = lc.get('pose_weight', 1.0)
        self.rot_weight = lc.get('rot_weight', 1.0)
        self.trans_weight = lc.get('trans_weight', 10.0)
        self.gamma = lc.get('gamma', 0.8)
        self.conf_reg_weight = lc.get('conf_reg_weight', 0.1)
        self.xi_weight = lc.get('xi_weight', 1.0)  # direct xi supervision
        self.contrast_weight = lc.get('contrast_weight', 0.5)  # contrastive matching loss
        self.phase1_epochs = tc.get('phase1_epochs', 10)  # flow-only warmup
        self.outer_iters = tc.get('outer_iters', 1)
        self.val_outer_iters = tc.get('val_outer_iters', 3)
        self.pose_loss_last_only = tc.get('pose_loss_last_only', False)

        # Noise curriculum
        nc = tc.get('noise_curriculum', {})
        self.use_noise_curriculum = nc.get('enabled', False)
        self.noise_warmup_epochs = nc.get('warmup_epochs', 40)
        self.noise_base_rot = nc.get('base_rot_deg', 2.0)
        self.noise_base_trans = nc.get('base_trans_m', 0.05)
        self.noise_max_rot = nc.get('max_rot_deg', 8.0)
        self.noise_max_trans = nc.get('max_trans_m', 0.25)

    @torch.no_grad()
    def _render_batch(self, poses_w2c):
        """Render features + depth for a batch of poses."""
        feat = self.renderer.render_features(poses_w2c, self.feat_hw)
        depth = self.renderer.render_depth(poses_w2c, self.feat_hw)
        return feat, depth

    def _update_noise_for_epoch(self, epoch: int):
        """Linearly ramp noise from base to max over warmup_epochs."""
        if not self.use_noise_curriculum:
            return
        ratio = min(1.0, epoch / max(1, self.noise_warmup_epochs))
        cur_rot = self.noise_base_rot + (self.noise_max_rot - self.noise_base_rot) * ratio
        cur_trans = self.noise_base_trans + (self.noise_max_trans - self.noise_base_trans) * ratio
        # Update train dataset noise parameters
        ds = getattr(self.train_dataset, 'dataset', self.train_dataset)
        ds.noise_rot_deg = cur_rot
        ds.noise_trans_m = cur_trans

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        self._update_noise_for_epoch(epoch)
        total_loss_sum = 0.0
        n_batches = 0
        phase = 'flow_only' if epoch < self.phase1_epochs else 'joint'

        pbar = tqdm(self.train_loader, desc=f'Train E{epoch:03d} [{phase}]')
        for batch in pbar:
            query_feat = batch['query_feat'].to(self.device)   # (B, 64, 68, 120)
            pose_gt = batch['pose_gt'].to(self.device)         # (B, 4, 4)
            pose_cur = batch['initial_pose'].to(self.device)   # (B, 4, 4)
            B = query_feat.shape[0]

            total_loss = torch.tensor(0.0, device=self.device)

            for outer_iter in range(self.outer_iters):
                # Render at current pose
                render_feat, depth = self._render_batch(pose_cur)

                # Compute GT flow at fine resolution
                with torch.no_grad():
                    gt_flow, gt_mask = self.model.compute_gt_flow(
                        pose_cur, pose_gt, depth, self.model.FINE_HW)

                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    result = self.model(query_feat, render_feat, depth)

                    # Flow loss (RAFT-style sequence loss)
                    loss_flow = flow_loss_fn(
                        result['fine_flow_preds'], gt_flow, gt_mask, self.gamma)

                    # Coarse flow loss
                    gt_flow_c, gt_mask_c = self.model.compute_gt_flow(
                        pose_cur, pose_gt, depth, self.model.COARSE_HW)
                    diff_c = (result['flow_coarse'] - gt_flow_c).abs()
                    n_valid_c = gt_mask_c.sum().clamp(min=1.0)
                    loss_flow_c = (diff_c * gt_mask_c).sum() / (n_valid_c * 2)

                    loss = self.flow_weight * (loss_flow + 0.3 * loss_flow_c)

                    # Confidence regularization (runs in ALL phases to prevent collapse)
                    conf_loss, conf_metrics = confidence_regularization_loss(
                        result['conf_fine'],
                        flow_fine=result['flow_fine'],
                        gt_flow=gt_flow,
                        gt_mask=gt_mask,
                    )
                    loss = loss + self.conf_reg_weight * conf_loss

                    # Contrastive matching loss — trains domain projections to be discriminative
                    if 'q_coarse_proj' in result:
                        loss_contrast = contrastive_matching_loss(
                            result['q_coarse_proj'], result['r_coarse_proj'],
                            gt_flow_c, gt_mask_c,
                            temperature=0.07, n_samples=256)
                        loss = loss + self.contrast_weight * loss_contrast

                    # Pose loss (after warmup, optionally only at last outer iter)
                    apply_pose = (phase == 'joint' and 'delta_xi' in result)
                    if self.pose_loss_last_only:
                        apply_pose = apply_pose and (outer_iter == self.outer_iters - 1)
                    if apply_pose:
                        loss_pose, pose_metrics = pose_loss_fn(
                            result['delta_xi'], pose_cur, pose_gt,
                            self.rot_weight, self.trans_weight)
                        loss = loss + self.pose_weight * loss_pose

                        # Direct xi supervision (stronger gradient signal)
                        loss_xi = xi_direct_loss(
                            result['delta_xi'], pose_cur, pose_gt)
                        loss = loss + self.xi_weight * loss_xi

                total_loss = total_loss + loss

                # Update pose for next outer iteration
                if outer_iter < self.outer_iters - 1 and 'delta_xi' in result:
                    with torch.no_grad():
                        T_delta = se3_exp(result['delta_xi'].detach())
                        pose_cur = T_delta @ pose_cur

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_sum += total_loss.item()
            n_batches += 1
            self.global_step += 1

            pbar.set_postfix(loss=f'{total_loss.item():.4f}',
                             conf=f'{conf_metrics.get("conf_mean", 0):.3f}')

        return total_loss_sum / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, epoch: int):
        self.model.eval()
        rot_errors, trans_errors = [], []
        init_rot_errors, init_trans_errors = [], []

        for batch in tqdm(self.val_loader, desc=f'Val E{epoch:03d}'):
            query_feat = batch['query_feat'].to(self.device)
            pose_gt = batch['pose_gt'].to(self.device)
            pose_cur = batch['initial_pose'].to(self.device)

            # Record initial pose errors
            for b in range(pose_gt.shape[0]):
                pred_c2w = torch.inverse(pose_cur[b])
                gt_c2w = torch.inverse(pose_gt[b])
                init_trans_errors.append(
                    (pred_c2w[:3, 3] - gt_c2w[:3, 3]).norm().item() * 100)
                R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
                trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
                cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
                init_rot_errors.append(
                    torch.acos(cos_a).item() * 180.0 / math.pi)

            for outer_iter in range(self.val_outer_iters):
                render_feat, depth = self._render_batch(pose_cur)
                result = self.model(query_feat, render_feat, depth)
                if 'delta_xi' in result:
                    T_delta = se3_exp(result['delta_xi'])
                    pose_cur = T_delta @ pose_cur

            # Compute errors
            for b in range(pose_gt.shape[0]):
                pred_c2w = torch.inverse(pose_cur[b])
                gt_c2w = torch.inverse(pose_gt[b])
                pos_err = (pred_c2w[:3, 3] - gt_c2w[:3, 3]).norm().item() * 100  # cm
                R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
                trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
                cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
                rot_err = torch.acos(cos_a).item() * 180.0 / math.pi
                rot_errors.append(rot_err)
                trans_errors.append(pos_err)

        med_rot = np.median(rot_errors)
        med_trans = np.median(trans_errors)
        init_med_rot = np.median(init_rot_errors)
        init_med_trans = np.median(init_trans_errors)
        return med_rot, med_trans, rot_errors, trans_errors, init_med_rot, init_med_trans

    def _save_checkpoint(self, path: str, metrics: dict = None):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'config': self.config,
            'metrics': metrics,
        }
        torch.save(state, path)

    def _load_checkpoint(self, path: str):
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if not self.eval_only and 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.epoch = ckpt.get('epoch', 0) + 1
            self.global_step = ckpt.get('global_step', 0)
        print(f"  Loaded: epoch={ckpt.get('epoch', '?')}, "
              f"metrics={ckpt.get('metrics', {})}")

    def _visualize_attention(self, epoch: int, n_samples: int = 4):
        """Save bidirectional attention heatmaps overlaid on RGB images."""
        self.model.eval()
        vis_dir = self.vis_dir / f'epoch_{epoch:03d}'
        vis_dir.mkdir(exist_ok=True)

        # Build image index for RGB overlay
        image_index = _build_image_index()

        count = 0
        for batch in self.val_loader:
            query_feat = batch['query_feat'].to(self.device)
            pose_cur = batch['initial_pose'].to(self.device)

            render_feat, depth = self._render_batch(pose_cur)
            result = self.model(query_feat, render_feat, depth)

            attn_q2r = result.get('attn_weights_q2r')
            attn_r2q = result.get('attn_weights_r2q')
            if attn_q2r is None or attn_r2q is None:
                break

            for b in range(min(attn_q2r.shape[0], n_samples - count)):
                frame_idx = batch['frame_idx'][b]
                # Get RGB image path
                img_path = image_index[frame_idx] if frame_idx < len(image_index) else None
                if img_path is None or not os.path.exists(img_path):
                    continue

                save_attention_heatmap_rgb(
                    attn_q2r=attn_q2r[b],
                    attn_r2q=attn_r2q[b],
                    coarse_hw=self.model.COARSE_HW,
                    query_img_path=img_path,
                    rendered_feat=render_feat[b].detach(),
                    renderer=self.renderer,
                    pose_w2c=pose_cur[b],
                    save_path=str(vis_dir / f'attn_frame{frame_idx}.png'),
                    title=f'Epoch {epoch}, Frame {frame_idx}',
                )
                count += 1
                if count >= n_samples:
                    return

    def train(self):
        print(f"\n{'='*60}")
        print(f"Training RadioPoseNet — {self.total_epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(self.epoch, self.total_epochs):
            self.epoch = epoch

            # Train
            avg_loss = self._train_one_epoch(epoch)
            self.scheduler.step()

            # Validate
            med_rot, med_trans, _, _, init_med_rot, init_med_trans = self._validate(epoch)
            lr = self.scheduler.get_last_lr()[0]

            print(f"[E{epoch:03d}] loss={avg_loss:.4f}, "
                  f"init: {init_med_trans:.1f}cm/{init_med_rot:.2f}° → "
                  f"val: {med_trans:.1f}cm/{med_rot:.2f}°, lr={lr:.6f}")

            if not self.eval_only:
                self.writer.add_scalar('train/loss', avg_loss, epoch)
                self.writer.add_scalar('val/median_trans_cm', med_trans, epoch)
                self.writer.add_scalar('val/median_rot_deg', med_rot, epoch)

            # Save checkpoints
            metric = med_trans + med_rot * 100  # combined metric
            self._save_checkpoint(str(self.ckpt_dir / 'latest.pth'),
                                  {'med_trans': med_trans, 'med_rot': med_rot})
            if metric < self.best_val_metric:
                self.best_val_metric = metric
                self._save_checkpoint(str(self.ckpt_dir / 'best.pth'),
                                      {'med_trans': med_trans, 'med_rot': med_rot})
                print(f"  ★ New best: {med_trans:.1f}cm / {med_rot:.2f}°")

            # Visualize attention heatmaps periodically
            if epoch % 10 == 0 or epoch == self.total_epochs - 1:
                self._visualize_attention(epoch)

        print(f"\nTraining complete. Best: {self.best_val_metric:.2f}")

    def evaluate(self):
        """Full evaluation with visualization."""
        print("\n" + "=" * 60)
        print("Evaluating RadioPoseNet on OldHospital test set")
        print("=" * 60 + "\n")

        med_rot, med_trans, rot_errors, trans_errors, init_med_rot, init_med_trans = self._validate(self.epoch)
        self._visualize_attention(self.epoch, n_samples=8)

        print(f"\n{'─'*40}")
        print(f"  Initial pose error:    {init_med_trans:.1f} cm / {init_med_rot:.2f}°")
        print(f"  Median position error: {med_trans:.1f} cm")
        print(f"  Median rotation error: {med_rot:.2f}°")
        print(f"  Mean position error:   {np.mean(trans_errors):.1f} cm")
        print(f"  Mean rotation error:   {np.mean(rot_errors):.2f}°")
        print(f"{'─'*40}")

        return med_trans, med_rot


# ==============================================================================
#  Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train/Eval RadioPoseNet')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    trainer = RadioPoseTrainer(
        config, eval_only=args.eval_only, checkpoint=args.checkpoint)

    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
