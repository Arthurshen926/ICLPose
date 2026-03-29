#!/usr/bin/env python3
"""
SingleScalePoseNet 训练脚本
============================
单尺度 DA3 特征 + GRU 迭代 + 几何求解

训练策略:
  Phase 1 (warmup): 仅 flow loss (RAFT-style sequence loss)
  Phase 2 (joint): flow loss + pose loss (cosine rotation + L1 translation)

用法:
    python scripts/train_single_scale.py --config configs/exp060_single_scale.yaml

    # 恢复训练
    python scripts/train_single_scale.py --config configs/exp060_single_scale.yaml \\
        --resume output/exp060/checkpoints/latest.pth
"""

import os
import sys
import math
import time
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ic_models.single_scale_pose_net import SingleScalePoseNet
from ic_models.transformer_pose_net import TransformerPoseNet
from ic_models.transformer_pose_net_v2 import TransformerPoseNetV2
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4


# ==============================================================================
#  Loss Functions
# ==============================================================================

def flow_loss(
    pred: Dict[str, torch.Tensor],
    gt_flow: torch.Tensor,
    gt_mask: torch.Tensor = None,
    gamma: float = 0.8,
    huber_delta: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    RAFT-style sequence loss on fine flow predictions.
    Each GRU iteration's flow gets weighted, later iters weighted more.
    """
    flow_preds = pred.get('fine_flow_preds', [pred['flow_fine']])
    n_iters = len(flow_preds)

    def _masked_huber(flow_pred, flow_gt, mask=None):
        diff = flow_pred - flow_gt
        abs_diff = diff.abs()
        loss_map = torch.where(
            abs_diff <= huber_delta,
            0.5 * diff.pow(2) / huber_delta,
            abs_diff - 0.5 * huber_delta,
        )
        if mask is not None:
            n_valid = mask.sum().clamp(min=1.0)
            return (loss_map * mask).sum() / (n_valid * 2)
        return loss_map.mean()

    total = torch.tensor(0.0, device=gt_flow.device)
    for i, iter_flow in enumerate(flow_preds):
        w = gamma ** (n_iters - 1 - i)
        total = total + w * _masked_huber(iter_flow, gt_flow, gt_mask)

    # Final iteration metrics
    final_flow = flow_preds[-1]
    if gt_mask is not None:
        n_valid = gt_mask.sum().clamp(min=1.0)
        epe = (torch.norm(final_flow - gt_flow, dim=1, keepdim=True) * gt_mask).sum() / n_valid
    else:
        epe = torch.norm(final_flow - gt_flow, dim=1).mean()

    return total, {
        'flow_loss': total.item(),
        'flow_epe': epe.item(),
    }


def confidence_regularization_loss(
    conf: torch.Tensor,
    target_low: float = 0.05,
    target_high: float = 0.95,
) -> torch.Tensor:
    """Prevent confidence collapse (all 0 or all 1)."""
    conf_mean = conf.mean()
    if conf_mean < target_low:
        return (target_low - conf_mean) ** 2
    elif conf_mean > target_high:
        return (conf_mean - target_high) ** 2
    return torch.tensor(0.0, device=conf.device)


def pose_loss(
    delta_xi: torch.Tensor,
    pose_init: torch.Tensor,
    pose_gt: torch.Tensor,
    rot_weight: float = 1.0,
    trans_weight: float = 10.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Pose loss: cosine rotation + L1 translation."""
    with torch.cuda.amp.autocast(enabled=False):
        delta_xi = delta_xi.float()
        pose_init = pose_init.float()
        pose_gt = pose_gt.float()

        T_delta = se3_exp(delta_xi)
        pose_pred = torch.bmm(T_delta, pose_init)

        R_pred = pose_pred[:, :3, :3]
        R_gt = pose_gt[:, :3, :3]
        R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)

        rot_loss_val = (1.0 - cos_angle).mean()  # cosine loss

        cos_for_metric = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        rot_err_deg = torch.acos(cos_for_metric) * 180.0 / math.pi

        t_pred = pose_pred[:, :3, 3]
        t_gt = pose_gt[:, :3, 3]
        trans_err = torch.norm(t_pred - t_gt, dim=1)

        loss = rot_loss_val * rot_weight + trans_err.mean() * trans_weight

    return loss, {
        'rot_err_deg': rot_err_deg.mean().item(),
        'trans_err_mm': (trans_err * 1000).mean().item(),
        'pose_loss': loss.item(),
    }


# ==============================================================================
#  Trainer
# ==============================================================================

class SingleScaleTrainer:
    """SingleScalePoseNet trainer."""

    def __init__(self, config: Dict, resume_path: str = None,
                 warmstart_path: str = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        exp_name = config.get('exp_name', 'exp_single_scale')
        self.output_dir = Path(config.get('output_dir', f'output/{exp_name}'))
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.writer = SummaryWriter(str(self.log_dir))

        tc = config.get('training', {})
        self.use_amp = tc.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self._init_renderer()
        self._init_model()
        self._init_datasets()
        self._init_optimizer()

        self.epoch = 0
        self.global_step = 0
        self.best_val_rot = float('inf')

        if resume_path:
            self._load_checkpoint(resume_path)
        elif warmstart_path:
            self._warmstart(warmstart_path)

    def _init_renderer(self):
        rc = self.config['renderer']
        triplane_path = rc.get('triplane_model_path')
        scale_model_paths = rc.get('scale_model_paths', {})

        print("[Trainer] Loading renderer...")
        self.renderer = MultiScaleRenderer(
            ply_path=rc['ply_path'],
            scale_model_paths=scale_model_paths,
            device=self.device,
            img_height=rc.get('img_height', 1080),
            img_width=rc.get('img_width', 1920),
            fx=rc.get('fx', 1663.12),
            fy=rc.get('fy', 1663.12),
            cx=rc.get('cx', 960.0),
            cy=rc.get('cy', 540.0),
            triplane_model_path=triplane_path,
            scale_resolutions=rc.get('scale_resolutions', None),
        )
        print(f"  Renderer loaded (scales: {self.renderer.scale_names})")

    def _init_model(self):
        mc = self.config.get('model', {})
        rc = self.config.get('renderer', {})
        intrinsics = {
            'fx': rc.get('fx', 1663.12),
            'fy': rc.get('fy', 1663.12),
            'cx': rc.get('cx', 960.0),
            'cy': rc.get('cy', 540.0),
        }
        model_type = mc.get('type', 'single_scale')

        if model_type == 'transformer':
            self.model = TransformerPoseNet(
                feat_dim=mc.get('feat_dim', 64),
                feat_in_dim=mc.get('feat_in_dim', 64),
                n_heads=mc.get('n_heads', 8),
                n_layers=mc.get('n_layers', 4),
                ffn_dim=mc.get('ffn_dim', 256),
                dropout=mc.get('dropout', 0.0),
                fine_iters=mc.get('fine_iters', 4),
                attn_hw=tuple(mc.get('attn_hw', [35, 61])),
                fine_hw=tuple(mc.get('fine_hw', [69, 121])),
                damping=mc.get('damping', 1e-3),
                intrinsics=intrinsics,
                img_hw=(rc.get('img_height', 1080), rc.get('img_width', 1920)),
                irls_iters=mc.get('irls_iters', 3),
                irls_huber_k=mc.get('irls_huber_k', 1.345),
                robust_kernel=mc.get('robust_kernel', 'huber'),
                pixel_stride=mc.get('pixel_stride', 1),
            ).to(self.device)
            model_name = "TransformerPoseNet"
            model_info = (f"n_layers={mc.get('n_layers', 4)}, "
                         f"n_heads={mc.get('n_heads', 8)}, "
                         f"attn_hw={mc.get('attn_hw', [35, 61])}")
        elif model_type == 'transformer_v2':
            self.model = TransformerPoseNetV2(
                d_model=mc.get('d_model', 128),
                feat_in_dim=mc.get('feat_in_dim', 64),
                n_heads=mc.get('n_heads', 4),
                n_layers=mc.get('n_layers', 6),
                ffn_dim=mc.get('ffn_dim', 256),
                dropout=mc.get('dropout', 0.0),
                attn_hw=tuple(mc.get('attn_hw', [35, 61])),
                fine_hw=tuple(mc.get('fine_hw', [69, 121])),
                damping=mc.get('damping', 1e-3),
                intrinsics=intrinsics,
                img_hw=(rc.get('img_height', 1080), rc.get('img_width', 1920)),
                irls_iters=mc.get('irls_iters', 3),
                irls_huber_k=mc.get('irls_huber_k', 1.345),
                robust_kernel=mc.get('robust_kernel', 'huber'),
                pixel_stride=mc.get('pixel_stride', 1),
            ).to(self.device)
            model_name = "TransformerPoseNetV2"
            model_info = (f"d_model={mc.get('d_model', 128)}, "
                         f"n_layers={mc.get('n_layers', 6)}, "
                         f"n_heads={mc.get('n_heads', 4)}")
        else:
            self.model = SingleScalePoseNet(
                hidden_dim=mc.get('hidden_dim', 128),
                decode_dim=mc.get('decode_dim', 64),
                feat_in_dim=mc.get('feat_in_dim', 64),
                local_radius=mc.get('local_radius', 4),
                corr_dilations=tuple(mc.get('corr_dilations', [1, 2, 4])),
                fine_iters=mc.get('fine_iters', 8),
                damping=mc.get('damping', 1e-3),
                fine_hw=tuple(mc.get('fine_hw', [69, 121])),
                intrinsics=intrinsics,
                img_hw=(rc.get('img_height', 1080), rc.get('img_width', 1920)),
                irls_iters=mc.get('irls_iters', 3),
                irls_huber_k=mc.get('irls_huber_k', 1.345),
                robust_kernel=mc.get('robust_kernel', 'huber'),
                pixel_stride=mc.get('pixel_stride', 1),
            ).to(self.device)
            model_name = "SingleScalePoseNet"
            model_info = (f"fine_iters={mc.get('fine_iters', 8)}, "
                         f"dilations={mc.get('corr_dilations', [1,2,4])}")

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] {model_name}: {n_params/1e6:.3f}M params, {model_info}")

    def _init_datasets(self):
        dc = self.config['data']
        mc = self.config.get('model', {})
        flow_res = tuple(mc.get('fine_hw', [69, 121]))

        if dc.get('val_feature_dir'):
            self.train_dataset = PoseDatasetV4(
                feature_base_dir=dc['train_feature_dir'],
                traj_path=dc['train_traj_path'],
                depth_dir=dc.get('train_depth_dir'),
                noise_rot_deg=dc.get('noise_rot_deg', 8.0),
                noise_trans_m=dc.get('noise_trans_m', 0.25),
                is_train=True,
                flow_resolution=flow_res,
                scale_names=['fine'],
                cache_in_memory=dc.get('cache_features', False),
            )
            self.val_dataset = PoseDatasetV4(
                feature_base_dir=dc['val_feature_dir'],
                traj_path=dc['val_traj_path'],
                depth_dir=dc.get('val_depth_dir'),
                noise_rot_deg=dc.get('val_noise_rot_deg', 8.0),
                noise_trans_m=dc.get('val_noise_trans_m', 0.25),
                is_train=False,
                netvlad_poses_path=dc.get('val_netvlad_poses'),
                flow_resolution=flow_res,
                scale_names=['fine'],
                cache_in_memory=dc.get('cache_features', False),
            )
        else:
            full_dataset = PoseDatasetV4(
                feature_base_dir=dc['train_feature_dir'],
                traj_path=dc['train_traj_path'],
                depth_dir=dc.get('train_depth_dir'),
                noise_rot_deg=dc.get('noise_rot_deg', 8.0),
                noise_trans_m=dc.get('noise_trans_m', 0.25),
                is_train=True,
                flow_resolution=flow_res,
                scale_names=['fine'],
                cache_in_memory=dc.get('cache_features', False),
            )
            n_total = len(full_dataset)
            n_val = max(1, int(n_total * dc.get('val_split_ratio', 0.1)))
            n_train = n_total - n_val
            self.train_dataset, self.val_dataset = \
                torch.utils.data.random_split(
                    full_dataset, [n_train, n_val],
                    generator=torch.Generator().manual_seed(42))
            print(f"[Data] Auto-split: {n_train} train + {n_val} val")

        n_workers = dc.get('num_workers', 4)
        use_persistent = n_workers > 0
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=dc.get('batch_size', 4),
            shuffle=True,
            num_workers=n_workers,
            collate_fn=collate_v4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=use_persistent,
            prefetch_factor=2 if n_workers > 0 else None,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=dc.get('val_batch_size', 4),
            shuffle=False,
            num_workers=min(n_workers, 4),
            collate_fn=collate_v4,
            pin_memory=True,
            persistent_workers=min(n_workers, 4) > 0,
        )
        print(f"[Data] Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def _init_optimizer(self):
        tc = self.config.get('training', {})
        base_lr = float(tc.get('lr', 2e-4))
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=float(tc.get('weight_decay', 1e-5)),
        )
        self.total_epochs = int(tc.get('epochs', 100))
        self.warmup_epochs = int(tc.get('warmup_epochs', 0))

        # LR scheduler: optional warmup + cosine annealing
        if self.warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=self.warmup_epochs)
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.total_epochs - self.warmup_epochs,
                eta_min=float(tc.get('min_lr', 1e-6)))
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs])
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.total_epochs,
                eta_min=float(tc.get('min_lr', 1e-6)))

        self.grad_clip = tc.get('grad_clip', 1.0)
        self.phase1_epochs = tc.get('phase1_epochs', 10)

        lc = tc.get('loss', {})
        self.pose_weight = lc.get('pose_weight', 1.0)
        self.rot_weight = lc.get('rot_weight', 1.0)
        self.trans_weight = lc.get('trans_weight', 10.0)
        self.gamma = lc.get('gamma', 0.8)
        self.huber_delta = lc.get('huber_delta', 5.0)
        self.conf_reg_weight = lc.get('conf_reg_weight', 0.01)

        # Noise curriculum
        nc = tc.get('noise_curriculum', {})
        self.use_noise_curriculum = nc.get('enabled', False)
        self.noise_warmup_epochs = nc.get('warmup_epochs', 30)
        dc = self.config['data']
        self.noise_rot_max = dc.get('noise_rot_deg', 8.0)
        self.noise_trans_max = dc.get('noise_trans_m', 0.25)
        self.noise_rot_min = nc.get('start_rot_deg', 2.0)
        self.noise_trans_min = nc.get('start_trans_m', 0.05)

        # Outer iterations
        self.num_outer_iters = tc.get('outer_iters', 1)
        self.val_outer_iters = tc.get('val_outer_iters', 3)
        self.gamma_outer = lc.get('gamma_outer', 0.8)

        # Gradient accumulation
        self.grad_accum_steps = tc.get('gradient_accumulation_steps', 1)

    # ------------------------------------------------------------------
    #  Rendering
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _render_batch(
        self, poses_w2c: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Render fine features + depth from 3DGS."""
        result = self.renderer.render_batch(
            poses_w2c, scales=['fine'], return_depth=True)
        render_feats = {'fine': result['fine_feat']}
        depth = result.get('depth_map')
        return render_feats, depth

    @torch.no_grad()
    def _compute_gt_flow(
        self, pose_init, pose_gt, depth,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GT flow at fine resolution."""
        flow, mask = self.model.compute_gt_flow(
            pose_init, pose_gt, depth, self.model.FINE_HW)
        return flow, mask

    # ------------------------------------------------------------------
    #  Training step
    # ------------------------------------------------------------------
    def _train_step(
        self, batch: Dict, use_pose_loss: bool,
    ) -> Dict[str, float]:
        """Single training step with optional outer iteration."""
        query_feats = {k: v.to(self.device) for k, v in batch['query_feats'].items()}
        pose_gt = batch['pose_gt'].to(self.device)
        pose_cur = batch['initial_pose'].to(self.device)

        N = self.num_outer_iters
        all_metrics = {}

        for outer_i in range(N):
            # 1. Render
            render_feats, depth = self._render_batch(pose_cur)

            # 2. Forward
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.model(query_feats, render_feats, depth)

            # 3. GT flow
            gt_flow, gt_mask = self._compute_gt_flow(pose_cur, pose_gt, depth)

            # 4. Loss
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                f_loss, f_metrics = flow_loss(
                    pred, gt_flow, gt_mask,
                    gamma=self.gamma, huber_delta=self.huber_delta)

                iter_loss = f_loss

                # Confidence regularization
                conf_loss = confidence_regularization_loss(pred['conf_fine'])
                iter_loss = iter_loss + self.conf_reg_weight * conf_loss

                # Pose loss
                p_metrics = {}
                if use_pose_loss and 'delta_xi' in pred:
                    p_loss, p_metrics = pose_loss(
                        pred['delta_xi'], pose_cur, pose_gt,
                        rot_weight=self.rot_weight,
                        trans_weight=self.trans_weight,
                    )
                    iter_loss = iter_loss + self.pose_weight * p_loss

            if torch.isnan(iter_loss) or torch.isinf(iter_loss):
                return {'nan_step': True}

            iter_w = self.gamma_outer ** (N - 1 - outer_i)
            scaled_loss = iter_w * iter_loss / (N * self.grad_accum_steps)
            self.scaler.scale(scaled_loss).backward()

            if outer_i == N - 1:
                all_metrics = f_metrics.copy()
                all_metrics['conf_mean'] = pred['conf_fine'].mean().item()
                all_metrics.update(p_metrics)

            # Update pose for next iteration
            if outer_i < N - 1 and 'delta_xi' in pred:
                with torch.no_grad():
                    T_delta = se3_exp(pred['delta_xi'].float())
                    pose_cur = torch.bmm(T_delta, pose_cur.float()).detach()

        all_metrics['total_loss'] = iter_loss.item()
        return all_metrics

    def train_epoch(self, epoch: int):
        self.model.train()
        use_pose_loss = epoch >= self.phase1_epochs

        epoch_metrics = {}
        pbar = tqdm(self.train_loader,
                     desc=f"Epoch {epoch}/{self.total_epochs}",
                     leave=False)

        for batch_idx, batch in enumerate(pbar):
            # Gradient accumulation: only zero grads at accumulation boundary
            if batch_idx % self.grad_accum_steps == 0:
                self.optimizer.zero_grad()

            metrics = self._train_step(batch, use_pose_loss)

            if metrics.get('nan_step'):
                self.global_step += 1
                continue

            # Step optimizer at accumulation boundary
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    valid_grads = True
                    for p in self.model.parameters():
                        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                            valid_grads = False
                            break
                    if not valid_grads:
                        self.optimizer.zero_grad()
                        self.scaler.update()
                        self.global_step += 1
                        continue
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.global_step += 1

            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k != 'nan_step':
                    epoch_metrics.setdefault(k, []).append(v)

            pbar_dict = {'loss': f"{metrics.get('total_loss', 0):.4f}"}
            if 'rot_err_deg' in metrics:
                pbar_dict['rot'] = f"{metrics['rot_err_deg']:.2f}°"
            pbar.set_postfix(pbar_dict)

            # Log GPU memory on first step
            if self.global_step == 1 and torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"\n  [GPU Mem] allocated={alloc:.2f}GB "
                      f"reserved={reserved:.2f}GB peak={peak:.2f}GB")

            if self.global_step % 50 == 0:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and k != 'nan_step':
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)

        avg = {k: np.mean(v) for k, v in epoch_metrics.items()}
        phase = "Phase2(flow+pose)" if use_pose_loss else "Phase1(flow only)"
        msg = (f"[Train E{epoch}] {phase}  "
               f"loss={avg.get('total_loss', 0):.4f}  "
               f"flow_epe={avg.get('flow_epe', 0):.2f}")
        if 'rot_err_deg' in avg:
            msg += f"  rot={avg['rot_err_deg']:.2f}°  trans={avg['trans_err_mm']:.1f}mm"
        print(msg)
        return avg

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        torch.cuda.empty_cache()
        self.model.eval()
        all_rot_errs, all_trans_errs = [], []
        N = self.val_outer_iters

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            query_feats = {k: v.to(self.device) for k, v in batch['query_feats'].items()}
            pose_gt = batch['pose_gt'].to(self.device)
            pose_cur = batch['initial_pose'].to(self.device)

            for outer_i in range(N):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    render_feats, depth = self._render_batch(pose_cur)
                    pred = self.model(query_feats, render_feats, depth)

                if outer_i < N - 1 and 'delta_xi' in pred:
                    with torch.cuda.amp.autocast(enabled=False):
                        T_delta = se3_exp(pred['delta_xi'].float())
                        pose_cur = torch.bmm(T_delta, pose_cur.float())

            if 'delta_xi' in pred:
                with torch.cuda.amp.autocast(enabled=False):
                    T_delta = se3_exp(pred['delta_xi'].float())
                    pose_pred = torch.bmm(T_delta, pose_cur.float())

                    R_pred = pose_pred[:, :3, :3]
                    R_gt = pose_gt.float()[:, :3, :3]
                    R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
                    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
                    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
                    rot_err = torch.acos(cos_angle) * 180.0 / math.pi

                    t_pred = pose_pred[:, :3, 3]
                    t_gt = pose_gt.float()[:, :3, 3]
                    trans_err = torch.norm(t_pred - t_gt, dim=1) * 1000

                all_rot_errs.extend(rot_err.cpu().tolist())
                all_trans_errs.extend(trans_err.cpu().tolist())

        val_metrics = {}
        if all_rot_errs:
            rot = np.array(all_rot_errs)
            trans = np.array(all_trans_errs)
            val_metrics = {
                'val_rot_mean': float(np.nanmean(rot)),
                'val_rot_median': float(np.nanmedian(rot)),
                'val_trans_mean': float(np.nanmean(trans)),
                'val_trans_median': float(np.nanmedian(trans)),
                'val_pct_1deg': float(np.mean(rot < 1.0) * 100),
                'val_pct_5deg': float(np.mean(rot < 5.0) * 100),
            }
            iters_str = f"  ({N} iters)" if N > 1 else ""
            print(f"[Val E{epoch}]  rot={val_metrics['val_rot_mean']:.2f}° "
                  f"(med {val_metrics['val_rot_median']:.2f}°)  "
                  f"trans={val_metrics['val_trans_mean']:.1f}mm  "
                  f"<1°={val_metrics['val_pct_1deg']:.1f}%{iters_str}")

            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, epoch)

        return val_metrics

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_rot': self.best_val_rot,
            'config': self.config,
        }
        torch.save(ckpt, self.ckpt_dir / 'latest.pth')
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')
        if epoch % 10 == 0:
            torch.save(ckpt, self.ckpt_dir / f'epoch_{epoch:03d}.pth')

    def _load_checkpoint(self, path: str):
        print(f"[Resume] Loading from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.epoch = ckpt['epoch'] + 1
        self.global_step = ckpt['global_step']
        self.best_val_rot = ckpt.get('best_val_rot', float('inf'))
        print(f"  Resumed at epoch {self.epoch}")

    def _warmstart(self, path: str):
        print(f"[Warmstart] Loading model weights from {path}")
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt['model_state_dict']
        model_state = self.model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and v.shape == model_state[k].shape}
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        if missing:
            print(f"  Missing: {missing}")
        print(f"  Loaded {len(filtered)} keys, fresh optimizer")

    def _update_noise(self, epoch: int):
        if not self.use_noise_curriculum:
            return
        ratio = min(1.0, epoch / max(1, self.noise_warmup_epochs))
        cur_rot = self.noise_rot_min + (self.noise_rot_max - self.noise_rot_min) * ratio
        cur_trans = self.noise_trans_min + (self.noise_trans_max - self.noise_trans_min) * ratio
        ds = getattr(self.train_dataset, 'dataset', self.train_dataset)
        ds.noise_rot_deg = cur_rot
        ds.noise_trans_m = cur_trans
        print(f"  [Curriculum] rot={cur_rot:.1f}° trans={cur_trans:.3f}m")

    def train(self):
        print(f"\n{'='*60}")
        print(f"  SingleScalePoseNet Training")
        print(f"  Epochs: {self.total_epochs} (Phase1: {self.phase1_epochs})")
        print(f"  Outer iters: {self.num_outer_iters} train / {self.val_outer_iters} val")
        print(f"  Batch size: {self.config['data'].get('batch_size', 4)}")
        print(f"  Grad accum: {self.grad_accum_steps}")
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}\n")

        train_start = time.time()

        for epoch in range(self.epoch, self.total_epochs):
            self.epoch = epoch
            t0 = time.time()

            self._update_noise(epoch)
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            self.scheduler.step()

            is_best = False
            if val_metrics and val_metrics.get('val_rot_mean', float('inf')) < self.best_val_rot:
                self.best_val_rot = val_metrics['val_rot_mean']
                is_best = True
                print(f"  ★ New best: rot={self.best_val_rot:.2f}°")

            self._save_checkpoint(epoch, is_best)

            elapsed = time.time() - t0
            total_min = (time.time() - train_start) / 60
            print(f"  Epoch {epoch}: {elapsed:.0f}s  "
                  f"lr={self.optimizer.param_groups[0]['lr']:.6f}  "
                  f"[{total_min:.0f}min elapsed]\n")

        print(f"\nTraining complete! Best val rot: {self.best_val_rot:.2f}°")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train SingleScalePoseNet')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--warmstart', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = SingleScaleTrainer(config, resume_path=args.resume,
                                 warmstart_path=args.warmstart)
    trainer.train()


if __name__ == '__main__':
    main()
