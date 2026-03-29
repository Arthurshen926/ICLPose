#!/usr/bin/env python3
"""
PoseRefiner Training Script for OldHospital (Cambridge Landmarks).

Uses the same data pipeline and renderer as RadioPoseNet training,
but with the improved PoseRefiner architecture featuring:
  - Convex upsampling to 4× solver resolution
  - Depth-normalized geometry solver
  - 8 GRU fine iterations
  - Optional translation MLP head

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_refiner.py \\
        --config configs/refiner_oldhospital.yaml

    # Resume training
    python scripts/train_refiner.py --config configs/refiner_oldhospital.yaml \\
        --resume output/refiner_oh_v1/checkpoints/latest.pth

    # Warmstart (load model weights only)
    python scripts/train_refiner.py --config configs/refiner_oldhospital.yaml \\
        --warmstart output/radio_pose_oh_v15/checkpoints/latest.pth
"""

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_models.pose_refiner import PoseRefiner
from modules.lie_algebra import se3_exp
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer


# ==============================================================================
#  Dataset
# ==============================================================================

class RadioPoseDataset(Dataset):
    """RADIO feature pose estimation dataset for Cambridge Landmarks."""

    def __init__(self, feature_dir, traj_path, frame_indices=None,
                 noise_rot_deg=8.0, noise_trans_m=0.25, is_train=True):
        self.feature_dir = Path(feature_dir)
        self.is_train = is_train
        self.noise_rot_deg = noise_rot_deg
        self.noise_trans_m = noise_trans_m

        pattern = re.compile(r'rgb_(\d+)_fine_radio_(\d+)x(\d+)x(\d+)\.pt')
        self.features = {}
        feat_dir = self.feature_dir / 'fine_radio'
        for f in sorted(feat_dir.iterdir()):
            m = pattern.match(f.name)
            if m:
                self.features[int(m.group(1))] = f

        poses_c2w = []
        with open(traj_path) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) == 16:
                    poses_c2w.append(np.array(vals).reshape(4, 4))
        self.poses_c2w = poses_c2w

        if frame_indices is not None:
            self.indices = [i for i in frame_indices
                           if i in self.features and i < len(poses_c2w)]
        else:
            self.indices = sorted([i for i in self.features.keys()
                                   if i < len(poses_c2w)])

        print(f"  [RadioPoseDataset] {len(self.indices)} frames, "
              f"noise={noise_rot_deg:.1f}°/{noise_trans_m:.2f}m, train={is_train}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]
        feat = torch.load(str(self.features[frame_idx]),
                          map_location='cpu', weights_only=True).float()
        c2w = torch.from_numpy(self.poses_c2w[frame_idx].astype(np.float32))
        w2c = torch.linalg.inv(c2w)

        if self.is_train:
            noisy_w2c = self._add_noise(w2c)
        else:
            noisy_w2c = w2c.clone()

        return {
            'query_feat': feat,
            'pose_gt': w2c,
            'initial_pose': noisy_w2c,
            'frame_idx': frame_idx,
        }

    def _add_noise(self, pose_w2c):
        axis = torch.randn(3)
        axis = axis / (axis.norm() + 1e-8)
        angle_rad = torch.empty(1).uniform_(0, self.noise_rot_deg).item() * math.pi / 180.0
        omega = axis * angle_rad

        direction = torch.randn(3)
        direction = direction / (direction.norm() + 1e-8)
        magnitude = torch.empty(1).uniform_(0, self.noise_trans_m).item()
        trans = direction * magnitude

        xi = torch.cat([trans, omega])
        delta_T = se3_exp(xi)
        return delta_T @ pose_w2c


def collate_radio(batch):
    return {
        'query_feat': torch.stack([b['query_feat'] for b in batch]),
        'pose_gt': torch.stack([b['pose_gt'] for b in batch]),
        'initial_pose': torch.stack([b['initial_pose'] for b in batch]),
        'frame_idx': [b['frame_idx'] for b in batch],
    }


# ==============================================================================
#  Renderer
# ==============================================================================

class RadioRenderer:
    """Renders 64d RADIO features + depth from a 2DGS model."""

    def __init__(self, ply_path, feature_model_path, device,
                 img_hw=(1080, 1920), fx=1663.12, fy=1663.12,
                 cx=960.0, cy=540.0):
        self.device = device
        self.img_hw = img_hw
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

        ckpt = torch.load(feature_model_path, map_location='cpu', weights_only=True)
        feat_dim = ckpt['feature_dim']
        self.gs_model = GaussianFeatureModel(feature_dim=feat_dim)
        self.gs_model.load_ply(ply_path)
        self.gs_model._loc_feature = nn.Parameter(ckpt['loc_feature'].to(device))
        self.gs_model = self.gs_model.to(device)
        self.gs_model.eval()
        n_gauss = self.gs_model.get_xyz.shape[0]
        print(f"  [RadioRenderer] {n_gauss} Gaussians, {feat_dim}d features")

    def render_features(self, viewmat, feat_hw=(68, 120)):
        batched = viewmat.dim() == 3
        if not batched:
            viewmat = viewmat.unsqueeze(0)
        fH, fW = feat_hw
        render_fx = self.fx * fW / self.img_hw[1]
        render_fy = self.fy * fH / self.img_hw[0]
        render_cx = self.cx * fW / self.img_hw[1]
        render_cy = self.cy * fH / self.img_hw[0]
        result = FeatureRenderer.render_features_batch(
            self.gs_model, viewmat,
            fx=render_fx, fy=render_fy, cx=render_cx, cy=render_cy,
            img_height=fH, img_width=fW,
            norm_feat_before_render=True, norm_feat_after_render=True)
        feat = result['feature_map']
        return feat if batched else feat.squeeze(0)

    def render_depth(self, viewmat, depth_hw=(68, 120)):
        batched = viewmat.dim() == 3
        if not batched:
            viewmat = viewmat.unsqueeze(0)
        dH, dW = depth_hw
        render_fx = self.fx * dW / self.img_hw[1]
        render_fy = self.fy * dH / self.img_hw[0]
        render_cx = self.cx * dW / self.img_hw[1]
        render_cy = self.cy * dH / self.img_hw[0]
        depths = []
        for i in range(viewmat.shape[0]):
            d = FeatureRenderer.render_depth(
                self.gs_model, viewmat[i],
                fx=render_fx, fy=render_fy, cx=render_cx, cy=render_cy,
                img_height=dH, img_width=dW)
            depths.append(d)
        depth = torch.stack(depths)
        return depth if batched else depth.squeeze(0)

    def render_all(self, viewmat, feat_hw=(68, 120)):
        feat = self.render_features(viewmat, feat_hw)
        depth = self.render_depth(viewmat, feat_hw)
        return feat, depth


# ==============================================================================
#  Loss Functions
# ==============================================================================

def flow_loss_fn(pred_flows, gt_flow, gt_mask, gamma=0.8):
    """RAFT-style sequence flow loss with gamma decay."""
    n_preds = len(pred_flows)
    total = torch.tensor(0.0, device=gt_flow.device)
    for i, flow_pred in enumerate(pred_flows):
        weight = gamma ** (n_preds - 1 - i)
        if flow_pred.shape[-2:] != gt_flow.shape[-2:]:
            sH, sW = flow_pred.shape[-2:]
            gH, gW = gt_flow.shape[-2:]
            flow_resized = F.interpolate(gt_flow, (sH, sW), mode='bilinear', align_corners=False)
            mask_resized = F.interpolate(gt_mask, (sH, sW), mode='nearest')
            flow_resized[:, 0] *= sW / gW
            flow_resized[:, 1] *= sH / gH
        else:
            flow_resized = gt_flow
            mask_resized = gt_mask
        diff = (flow_pred - flow_resized).abs()
        n_valid = mask_resized.sum().clamp(min=1.0)
        total = total + weight * (diff * mask_resized).sum() / (n_valid * 2)
    return total


def confidence_regularization_loss(
    conf_fine: torch.Tensor,
    flow_fine: torch.Tensor = None,
    gt_flow: torch.Tensor = None,
    gt_mask: torch.Tensor = None,
    target_range=(0.05, 0.95),
):
    """Confidence regularization: coverage + calibration (matches v15)."""
    device = conf_fine.device
    metrics = {}

    conf_mean = conf_fine.mean()
    metrics['conf_mean'] = conf_mean.item()
    metrics['conf_std'] = conf_fine.std().item()

    # Coverage: push mean into target range
    lo, hi = target_range
    if conf_mean < lo:
        coverage_loss = (lo - conf_mean) ** 2
    elif conf_mean > hi:
        coverage_loss = (conf_mean - hi) ** 2
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


def contrastive_matching_loss(q_feat, r_feat, gt_flow, gt_mask,
                              temperature=0.07, n_samples=256):
    """InfoNCE contrastive loss for discriminative projections."""
    B, C, H, W = q_feat.shape
    N = H * W
    device = q_feat.device
    q_flat = q_feat.reshape(B, C, N)
    r_flat = r_feat.reshape(B, C, N)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
    target_x = (grid_x.unsqueeze(0) + gt_flow[:, 0]).round().long()
    target_y = (grid_y.unsqueeze(0) + gt_flow[:, 1]).round().long()
    valid = ((gt_mask.squeeze(1) > 0.5) &
             (target_x >= 0) & (target_x < W) &
             (target_y >= 0) & (target_y < H))
    target_idx = (target_y * W + target_x).clamp(0, N - 1)

    loss = torch.tensor(0.0, device=device)
    count = 0
    for b in range(B):
        v = valid[b].flatten()
        if v.sum() < 10:
            continue
        valid_indices = v.nonzero(as_tuple=True)[0]
        n_pick = min(n_samples, valid_indices.shape[0])
        perm = torch.randperm(valid_indices.shape[0], device=device)[:n_pick]
        sampled = valid_indices[perm]
        q_s = q_flat[b, :, sampled]
        targets = target_idx[b].flatten()[sampled]
        sim = (q_s.T @ r_flat[b]) / temperature
        labels = targets
        loss = loss + F.cross_entropy(sim, labels)
        count += 1
    return loss / max(count, 1)


def pose_loss_fn(delta_xi, pose_init, pose_gt, rot_weight=1.0, trans_weight=10.0):
    """Cosine rotation loss + L1 translation loss."""
    with torch.no_grad():
        T_delta_gt = pose_gt @ torch.linalg.inv(pose_init)

    T_pred = se3_exp(delta_xi)
    R_pred = T_pred[:, :3, :3]
    R_gt = T_delta_gt[:, :3, :3]
    t_pred = T_pred[:, :3, 3]
    t_gt = T_delta_gt[:, :3, 3]

    cos_sim = (R_pred * R_gt).sum(dim=(1, 2)) / 3.0
    rot_loss = (1.0 - cos_sim.clamp(-1, 1)).mean()
    trans_loss = (t_pred - t_gt).abs().mean()

    loss = rot_weight * rot_loss + trans_weight * trans_loss
    return loss, {'rot_loss': rot_loss.item(), 'trans_loss': trans_loss.item()}


def xi_direct_loss(delta_xi, pose_init, pose_gt):
    """Direct L1 supervision on se(3) vector."""
    from modules.lie_algebra import se3_log
    with torch.no_grad():
        T_delta_gt = pose_gt @ torch.linalg.inv(pose_init)
        xi_gt = se3_log(T_delta_gt)
    return F.l1_loss(delta_xi, xi_gt)


# ==============================================================================
#  Trainer
# ==============================================================================

class RefinerTrainer:
    """Training and evaluation for PoseRefiner on OldHospital."""

    def __init__(self, config, eval_only=False, checkpoint=None, warmstart=None):
        self.config = config
        self.device = torch.device('cuda')
        self.eval_only = eval_only
        torch.backends.cudnn.benchmark = True

        exp_name = config.get('exp_name', 'refiner_oh')
        self.output_dir = Path(config.get('output_dir', f'output/{exp_name}'))
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            self._load_checkpoint(checkpoint, resume=True)
        elif warmstart:
            self._load_checkpoint(warmstart, resume=False)

    def _init_renderer(self):
        rc = self.config['renderer']
        self.feat_hw = tuple(rc.get('feat_hw', [68, 120]))
        self.renderer = RadioRenderer(
            ply_path=rc['ply_path'],
            feature_model_path=rc['feature_model_path'],
            device=self.device,
            img_hw=tuple(rc.get('img_hw', [1080, 1920])),
            fx=rc.get('fx', 1663.12), fy=rc.get('fy', 1663.12),
            cx=rc.get('cx', 960.0), cy=rc.get('cy', 540.0))
        print(f"  ✓ Renderer loaded, feat_hw={self.feat_hw}")

    def _init_model(self):
        mc = self.config.get('model', {})
        rc = self.config['renderer']
        intrinsics = {
            'fx': rc.get('fx', 1663.12), 'fy': rc.get('fy', 1663.12),
            'cx': rc.get('cx', 960.0), 'cy': rc.get('cy', 540.0),
        }
        self.model = PoseRefiner(
            in_dim=mc.get('in_dim', 64),
            match_dim=mc.get('match_dim', 64),
            hidden_dim=mc.get('hidden_dim', 128),
            n_heads=mc.get('n_heads', 4),
            n_attn_layers=mc.get('n_attn_layers', 2),
            ffn_dim=mc.get('ffn_dim', 128),
            local_radius=mc.get('local_radius', 4),
            fine_iters=mc.get('fine_iters', 8),
            damping=mc.get('damping', 1e-3),
            coarse_hw=tuple(mc.get('coarse_hw', [17, 30])),
            fine_hw=tuple(mc.get('fine_hw', [34, 60])),
            solver_upsample=mc.get('solver_upsample', 4),
            solver_hw=tuple(mc['solver_hw']) if 'solver_hw' in mc else None,
            intrinsics=intrinsics,
            img_hw=tuple(rc.get('img_hw', [1080, 1920])),
            depth_normalize=mc.get('depth_normalize', True),
            sequential_solve=mc.get('sequential_solve', True),
            detach_conf=mc.get('detach_conf', True),
            conf_floor=mc.get('conf_floor', 0.1),
            use_trans_head=mc.get('use_trans_head', False),
            trans_head_mode=mc.get('trans_head_mode', 'replace'),
            solver_trans_scale=mc.get('solver_trans_scale', 0.0),
            use_flow_scale_head=mc.get('use_flow_scale_head', False),
            raw_coarse_corr=mc.get('raw_coarse_corr', False),
        ).to(self.device)
        self.model.update_damping = mc.get('update_damping', False)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  ✓ PoseRefiner: {n_params / 1e6:.2f}M params, "
              f"fine_iters={mc.get('fine_iters', 8)}, "
              f"solver_hw={self.model.SOLVER_HW}, "
              f"trans_scale={mc.get('solver_trans_scale', 0.0)}")

    def _init_datasets(self):
        dc = self.config['data']
        feature_dir = dc['feature_dir']
        traj_path = dc['traj_path']

        train_indices, test_indices = None, None
        train_idx_path = os.path.join(feature_dir, 'train_indices.npy')
        test_idx_path = os.path.join(feature_dir, 'test_indices.npy')
        if os.path.exists(train_idx_path):
            train_indices = np.load(train_idx_path).tolist()
            print(f"  Loaded train indices: {len(train_indices)} frames")
        if os.path.exists(test_idx_path):
            test_indices = np.load(test_idx_path).tolist()
            print(f"  Loaded test indices: {len(test_indices)} frames")

        self.train_dataset = RadioPoseDataset(
            feature_dir=feature_dir, traj_path=traj_path,
            frame_indices=train_indices,
            noise_rot_deg=dc.get('noise_rot_deg', 8.0),
            noise_trans_m=dc.get('noise_trans_m', 0.25), is_train=True)
        self.val_dataset = RadioPoseDataset(
            feature_dir=feature_dir, traj_path=traj_path,
            frame_indices=test_indices or train_indices,
            noise_rot_deg=dc.get('val_noise_rot_deg', 8.0),
            noise_trans_m=dc.get('val_noise_trans_m', 0.25), is_train=True)

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
        self.total_epochs = tc.get('epochs', 200)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.total_epochs, eta_min=tc.get('min_lr', 1e-6))
        self.grad_clip = tc.get('grad_clip', 1.0)
        self.use_amp = tc.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        lc = tc.get('loss', {})
        self.flow_weight = lc.get('flow_weight', 1.0)
        self.pose_weight = lc.get('pose_weight', 0.0)
        self.rot_weight = lc.get('rot_weight', 1.0)
        self.trans_weight = lc.get('trans_weight', 10.0)
        self.gamma = lc.get('gamma', 0.8)
        self.conf_reg_weight = lc.get('conf_reg_weight', 0.1)
        self.xi_weight = lc.get('xi_weight', 0.0)
        self.contrast_weight = lc.get('contrast_weight', 0.5)
        self.phase1_epochs = tc.get('phase1_epochs', 200)
        self.outer_iters = tc.get('outer_iters', 1)
        self.val_outer_iters = tc.get('val_outer_iters', 3)
        self.trans_start_epoch = tc.get('trans_start_epoch', 50)
        self.grad_accum_steps = tc.get('grad_accum_steps', 1)

        nc = tc.get('noise_curriculum', {})
        self.use_noise_curriculum = nc.get('enabled', False)
        self.noise_warmup_epochs = nc.get('warmup_epochs', 40)
        self.noise_base_rot = nc.get('base_rot_deg', 2.0)
        self.noise_base_trans = nc.get('base_trans_m', 0.05)
        self.noise_max_rot = nc.get('max_rot_deg', 8.0)
        self.noise_max_trans = nc.get('max_trans_m', 0.25)

    @torch.no_grad()
    def _render_batch(self, poses_w2c):
        feat = self.renderer.render_features(poses_w2c, self.feat_hw)
        depth = self.renderer.render_depth(poses_w2c, self.feat_hw)
        return feat, depth

    def _update_noise_for_epoch(self, epoch):
        if not self.use_noise_curriculum:
            return
        ratio = min(1.0, epoch / max(1, self.noise_warmup_epochs))
        cur_rot = self.noise_base_rot + (self.noise_max_rot - self.noise_base_rot) * ratio
        cur_trans = self.noise_base_trans + (self.noise_max_trans - self.noise_base_trans) * ratio
        ds = getattr(self.train_dataset, 'dataset', self.train_dataset)
        ds.noise_rot_deg = cur_rot
        ds.noise_trans_m = cur_trans

    def _train_one_epoch(self, epoch):
        self.model.train()
        self._update_noise_for_epoch(epoch)
        total_loss_sum = 0.0
        n_batches = 0
        phase = 'flow_only' if epoch < self.phase1_epochs else 'joint'

        pbar = tqdm(self.train_loader, desc=f'Train E{epoch:03d} [{phase}]')
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(pbar):
            query_feat = batch['query_feat'].to(self.device)
            pose_gt = batch['pose_gt'].to(self.device)
            pose_cur = batch['initial_pose'].to(self.device)

            total_loss_value = 0.0

            for outer_iter in range(self.outer_iters):
                render_feat, depth = self._render_batch(pose_cur)

                with torch.no_grad():
                    gt_flow, gt_mask = self.model.compute_gt_flow(
                        pose_cur, pose_gt, depth, self.model.FINE_HW)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    result = self.model(query_feat, render_feat, depth)

                    # Flow loss (RAFT-style sequence)
                    loss_flow = flow_loss_fn(
                        result['fine_flow_preds'], gt_flow, gt_mask, self.gamma)

                    # Coarse flow loss
                    gt_flow_c, gt_mask_c = self.model.compute_gt_flow(
                        pose_cur, pose_gt, depth, self.model.COARSE_HW)
                    diff_c = (result['flow_coarse'] - gt_flow_c).abs()
                    n_valid_c = gt_mask_c.sum().clamp(min=1.0)
                    loss_flow_c = (diff_c * gt_mask_c).sum() / (n_valid_c * 2)

                    loss = self.flow_weight * (loss_flow + 0.3 * loss_flow_c)

                    # Confidence regularization (with calibration)
                    conf_loss, conf_metrics = confidence_regularization_loss(
                        result['conf_fine'],
                        flow_fine=result['flow_fine'],
                        gt_flow=gt_flow,
                        gt_mask=gt_mask)
                    loss = loss + self.conf_reg_weight * conf_loss

                    # Contrastive matching loss
                    if self.contrast_weight > 0 and 'q_coarse_proj' in result:
                        loss_contrast = contrastive_matching_loss(
                            result['q_coarse_proj'], result['r_coarse_proj'],
                            gt_flow_c, gt_mask_c, temperature=0.07, n_samples=256)
                        loss = loss + self.contrast_weight * loss_contrast

                    # Pose loss (after warmup)
                    if phase == 'joint' and 'delta_xi' in result:
                        loss_pose, _ = pose_loss_fn(
                            result['delta_xi'], pose_cur, pose_gt,
                            self.rot_weight, self.trans_weight)
                        loss = loss + self.pose_weight * loss_pose
                        if self.xi_weight > 0:
                            loss = loss + self.xi_weight * xi_direct_loss(
                                result['delta_xi'], pose_cur, pose_gt)

                # Backward per outer iteration to save memory
                # (no cross-iteration gradient since pose_cur is detached)
                scaled_loss = loss / (self.outer_iters * self.grad_accum_steps)
                self.scaler.scale(scaled_loss).backward()
                total_loss_value += loss.item()

                if outer_iter < self.outer_iters - 1 and 'delta_xi' in result:
                    with torch.no_grad():
                        xi_update = result['delta_xi'].detach()
                        # Zero translation during flow-only phase
                        if phase == 'flow_only' and self.model.solver_trans_scale > 0:
                            xi_update = torch.cat([
                                torch.zeros_like(xi_update[:, :3]),
                                xi_update[:, 3:]
                            ], dim=1)
                        pose_cur = se3_exp(xi_update) @ pose_cur

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss_sum += total_loss_value
            n_batches += 1
            self.global_step += 1
            pbar.set_postfix(loss=f'{total_loss_value:.4f}',
                             conf=f'{conf_metrics.get("conf_mean", 0):.3f}')

        return total_loss_sum / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        rot_errors, trans_errors = [], []
        init_rot_errors, init_trans_errors = [], []
        # Per-iteration tracking
        per_iter_rot = {i: [] for i in range(self.val_outer_iters)}

        for batch in tqdm(self.val_loader, desc=f'Val E{epoch:03d}'):
            query_feat = batch['query_feat'].to(self.device)
            pose_gt = batch['pose_gt'].to(self.device)
            pose_cur = batch['initial_pose'].to(self.device)

            for b in range(pose_gt.shape[0]):
                pred_c2w = torch.inverse(pose_cur[b])
                gt_c2w = torch.inverse(pose_gt[b])
                init_trans_errors.append(
                    (pred_c2w[:3, 3] - gt_c2w[:3, 3]).norm().item() * 100)
                R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
                trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
                cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
                init_rot_errors.append(torch.acos(cos_a).item() * 180.0 / math.pi)

            for outer_iter in range(self.val_outer_iters):
                render_feat, depth = self._render_batch(pose_cur)
                result = self.model(query_feat, render_feat, depth)
                if 'delta_xi' in result:
                    xi_val = result['delta_xi']
                    # Zero translation during flow-only phase
                    phase_val = 'flow_only' if epoch < self.phase1_epochs else 'joint'
                    if phase_val == 'flow_only' and self.model.solver_trans_scale > 0:
                        xi_val = torch.cat([
                            torch.zeros_like(xi_val[:, :3]),
                            xi_val[:, 3:]
                        ], dim=1)
                    pose_cur = se3_exp(xi_val) @ pose_cur
                # Track per-iteration rotation error
                for b in range(pose_gt.shape[0]):
                    pred_c2w = torch.inverse(pose_cur[b])
                    gt_c2w = torch.inverse(pose_gt[b])
                    R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
                    trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
                    cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
                    per_iter_rot[outer_iter].append(
                        torch.acos(cos_a).item() * 180.0 / math.pi)

            for b in range(pose_gt.shape[0]):
                pred_c2w = torch.inverse(pose_cur[b])
                gt_c2w = torch.inverse(pose_gt[b])
                pos_err = (pred_c2w[:3, 3] - gt_c2w[:3, 3]).norm().item() * 100
                R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
                trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
                cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
                rot_err = torch.acos(cos_a).item() * 180.0 / math.pi
                rot_errors.append(rot_err)
                trans_errors.append(pos_err)

        med_rot = np.median(rot_errors)
        med_trans = np.median(trans_errors)
        # Log per-iteration diagnostics
        iter_str = ' | '.join(
            f'I{i}={np.median(per_iter_rot[i]):.2f}°'
            for i in range(self.val_outer_iters))
        print(f'  [Iter] {iter_str}')
        init_med_rot = np.median(init_rot_errors)
        init_med_trans = np.median(init_trans_errors)
        return med_rot, med_trans, rot_errors, trans_errors, init_med_rot, init_med_trans

    def _save_checkpoint(self, path, metrics=None):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'config': self.config,
            'metrics': metrics,
        }, path)

    def _load_checkpoint(self, path, resume=True):
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Handle loading from RadioPoseNet checkpoints (warmstart)
        state = ckpt.get('model_state_dict', ckpt)
        model_state = self.model.state_dict()

        # Smart radius mapping for corr_encoder conv weights
        # When local_radius changes (e.g. 4→8), map the (2r_old+1)² channels
        # from the checkpoint to the correct positions in the (2r_new+1)² layout
        model_radius = self.model.local_radius if hasattr(self.model, 'local_radius') else None
        for k in list(state.keys()):
            if k in model_state and state[k].shape != model_state[k].shape:
                # Check if this is a corr_encoder conv weight with radius mismatch
                if 'corr_encoder' in k and state[k].dim() == 4 and model_state[k].dim() == 4:
                    old_shape = state[k].shape  # [C_out, (2r_old+1)²+extra, kH, kW]
                    new_shape = model_state[k].shape  # [C_out, (2r_new+1)²+extra, kH, kW]
                    if old_shape[0] == new_shape[0] and old_shape[2:] == new_shape[2:]:
                        # Determine old and new radius from channel counts
                        # channels = (2r+1)² + extra (extra = 3 for flow+conf)
                        extra = 3  # flow_x, flow_y, conf
                        old_corr = old_shape[1] - extra
                        new_corr = new_shape[1] - extra
                        import math
                        r_old = (int(math.sqrt(old_corr)) - 1) // 2
                        r_new = (int(math.sqrt(new_corr)) - 1) // 2
                        if (2*r_old+1)**2 == old_corr and (2*r_new+1)**2 == new_corr and r_new > r_old:
                            print(f"  Mapping corr_encoder weights: r={r_old} → r={r_new}")
                            # Initialize with zeros
                            new_weight = torch.zeros_like(model_state[k])
                            # Map old radius channels to new radius positions
                            old_w = 2*r_old + 1
                            new_w = 2*r_new + 1
                            for dy_old in range(-r_old, r_old+1):
                                for dx_old in range(-r_old, r_old+1):
                                    old_idx = (dy_old + r_old) * old_w + (dx_old + r_old)
                                    new_idx = (dy_old + r_new) * new_w + (dx_old + r_new)
                                    new_weight[:, new_idx] = state[k][:, old_idx]
                            # Copy extra channels (flow + conf)
                            new_weight[:, new_corr:] = state[k][:, old_corr:]
                            state[k] = new_weight
                            print(f"    Mapped {old_corr} corr channels + {extra} extra → {new_corr}+{extra}")
                            continue

        # Filter out remaining size-mismatched keys
        skipped = []
        for k in list(state.keys()):
            if k in model_state and state[k].shape != model_state[k].shape:
                skipped.append(f"{k} (ckpt={list(state[k].shape)} vs model={list(model_state[k].shape)})")
                del state[k]
        if skipped:
            print(f"  Skipped {len(skipped)} size-mismatched keys:")
            for s in skipped:
                print(f"    {s}")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} (new modules)")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (ignored)")

        if resume and not self.eval_only:
            if 'optimizer_state_dict' in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception:
                    print("  ⚠ Could not load optimizer state (architecture changed)")
            self.epoch = ckpt.get('epoch', 0) + 1
            self.global_step = ckpt.get('global_step', 0)

        print(f"  Loaded: epoch={ckpt.get('epoch', '?')}, metrics={ckpt.get('metrics', {})}")

    def train(self):
        print(f"\n{'='*60}")
        print(f"Training PoseRefiner — {self.total_epochs} epochs")
        print(f"  Solver: {self.model.SOLVER_HW}, depth_norm={self.model.depth_normalize}")
        print(f"  Outer iters: train={self.outer_iters}, val={self.val_outer_iters}")
        print(f"{'='*60}\n")

        for epoch in range(self.epoch, self.total_epochs):
            self.epoch = epoch
            avg_loss = self._train_one_epoch(epoch)
            self.scheduler.step()

            med_rot, med_trans, _, _, init_med_rot, init_med_trans = self._validate(epoch)
            lr = self.scheduler.get_last_lr()[0]

            print(f"[E{epoch:03d}] loss={avg_loss:.4f}, "
                  f"init: {init_med_trans:.1f}cm/{init_med_rot:.2f}° → "
                  f"val: {med_trans:.1f}cm/{med_rot:.2f}°, lr={lr:.6f}")

            if not self.eval_only:
                self.writer.add_scalar('train/loss', avg_loss, epoch)
                self.writer.add_scalar('val/median_trans_cm', med_trans, epoch)
                self.writer.add_scalar('val/median_rot_deg', med_rot, epoch)

            metric = med_trans + med_rot * 100
            self._save_checkpoint(str(self.ckpt_dir / 'latest.pth'),
                                  {'med_trans': med_trans, 'med_rot': med_rot})
            if metric < self.best_val_metric:
                self.best_val_metric = metric
                self._save_checkpoint(str(self.ckpt_dir / 'best.pth'),
                                      {'med_trans': med_trans, 'med_rot': med_rot})
                print(f"  ★ New best: {med_trans:.1f}cm / {med_rot:.2f}°")

        print(f"\nTraining complete. Best metric: {self.best_val_metric:.2f}")

    def evaluate(self):
        print(f"\n{'='*60}")
        print(f"Evaluating PoseRefiner on test set")
        print(f"{'='*60}\n")
        med_rot, med_trans, rot_errors, trans_errors, init_med_rot, init_med_trans = \
            self._validate(self.epoch)
        print(f"\n{'─'*40}")
        print(f"  Initial:  {init_med_trans:.1f}cm / {init_med_rot:.2f}°")
        print(f"  Final:    {med_trans:.1f}cm / {med_rot:.2f}°")
        print(f"  Mean:     {np.mean(trans_errors):.1f}cm / {np.mean(rot_errors):.2f}°")
        print(f"{'─'*40}")
        return med_trans, med_rot


# ==============================================================================
#  Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train/Eval PoseRefiner')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (full state)')
    parser.add_argument('--warmstart', type=str, default=None,
                        help='Load model weights only (reset optimizer)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    trainer = RefinerTrainer(
        config,
        eval_only=args.eval_only,
        checkpoint=args.resume,
        warmstart=args.warmstart,
    )

    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
