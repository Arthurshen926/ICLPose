#!/usr/bin/env python3
"""
MSFlowPoseNet 训练脚本
========================
SD-Primary Coarse-to-Fine Multi-Scale Flow Pose Estimation

训练策略:
  Phase 1 (warmup): 仅多尺度 flow loss, 不使用 pose loss
  Phase 2 (joint):  flow loss + pose loss (渐进增大权重)

改进:
  - 外层迭代精化: 每步 N 次 render→forward→update pose, 逐步缩小残差
  - 噪声课程学习: 从小扰动到大扰动渐进
  - RAFT 风格 sequence loss: fine 层每次 GRU 迭代都监督, 后期权重更大
  - Pose loss warmup: Phase 2 初期 pose_weight 从小值渐增到目标值

用法:
    python scripts/train_ms_flow.py --config configs/exp030_ms_flow_v2.yaml

    # 恢复训练
    python scripts/train_ms_flow.py --config configs/exp030_ms_flow_v2.yaml \\
        --resume output/exp030/checkpoints/latest.pth
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

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4


# ==============================================================================
#  Loss Functions
# ==============================================================================

def multiscale_flow_loss(
    pred: Dict[str, torch.Tensor],
    gt_flows: Dict[str, torch.Tensor],
    gt_masks: Dict[str, torch.Tensor] = None,
    weights: Dict[str, float] = None,
    gamma: float = 0.8,
    use_huber: bool = True,
    huber_delta: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    多尺度 flow 监督损失 (masked L1/Huber + masked endpoint error).
    Fine 层使用 RAFT-style sequence loss: 对每次 GRU 迭代都计算 loss,
    后面的迭代权重更大 (gamma^(N-1-i)).

    Args:
        pred: model output with flow_coarse, flow_mid, flow_fine, fine_flow_preds
        gt_flows: {
            'coarse': (B, 2, cH, cW),
            'mid':    (B, 2, mH, mW),
            'fine':   (B, 2, fH, fW),
        }
        gt_masks: {
            'coarse': (B, 1, cH, cW),  # 0/1 validity mask
            'mid':    (B, 1, mH, mW),
            'fine':   (B, 1, fH, fW),
        }
        weights: per-scale loss weights
        gamma: RAFT sequence loss 衰减系数 (越大越均匀, 0.8=标准RAFT)
        use_huber: 使用 Huber loss 代替 L1 (对 flow 大误差更鲁棒)
        huber_delta: Huber loss 阈值 (像素), 超过此值的误差线性增长而非二次

    Returns:
        total_loss, metrics_dict
    """
    if weights is None:
        weights = {'coarse': 0.1, 'mid': 0.3, 'fine': 1.0}

    total = torch.tensor(0.0, device=pred['flow_fine'].device)
    metrics = {}

    def _masked_flow_loss(flow_pred, flow_gt, mask=None):
        """Compute masked per-pixel flow loss (L1 or Huber)."""
        diff = flow_pred - flow_gt
        if use_huber:
            # Huber loss per-pixel: smooth at small errors, linear at large
            abs_diff = diff.abs()
            loss_map = torch.where(
                abs_diff <= huber_delta,
                0.5 * diff.pow(2) / huber_delta,
                abs_diff - 0.5 * huber_delta,
            )
        else:
            loss_map = diff.abs()

        if mask is not None:
            n_valid = mask.sum().clamp(min=1.0)
            return (loss_map * mask).sum() / (n_valid * 2)
        return loss_map.mean()

    scale_map = {
        'coarse': 'flow_coarse',
        'mid': 'flow_mid',
        'fine': 'flow_fine',
    }

    for scale_name, pred_key in scale_map.items():
        if scale_name not in gt_flows or pred_key not in pred:
            continue
        gt_flow = gt_flows[scale_name]
        mask = gt_masks.get(scale_name) if gt_masks else None
        w = weights.get(scale_name, 1.0)

        # ── Fine 层: RAFT-style sequence loss (对每次迭代都监督) ──
        if scale_name == 'fine' and 'fine_flow_preds' in pred:
            fine_preds = pred['fine_flow_preds']
            n_iters = len(fine_preds)
            seq_loss = torch.tensor(0.0, device=gt_flow.device)

            for i, iter_flow in enumerate(fine_preds):
                # 权重: gamma^(N-1-i), 最后一次迭代权重=1.0
                iter_w = gamma ** (n_iters - 1 - i)
                iter_loss = _masked_flow_loss(iter_flow, gt_flow, mask)
                seq_loss = seq_loss + iter_w * iter_loss

            total = total + w * seq_loss

            # 最终迭代的 metrics
            pred_flow = fine_preds[-1]
            if mask is not None:
                n_valid = mask.sum().clamp(min=1.0)
                epe_map = torch.norm(pred_flow - gt_flow, dim=1, keepdim=True)
                epe = (epe_map * mask).sum() / n_valid
                diff_final = (pred_flow - gt_flow).abs()
                l1_final = (diff_final * mask).sum() / (n_valid * 2)
                valid_ratio = mask.mean().item()
            else:
                l1_final = F.l1_loss(pred_flow, gt_flow)
                epe = torch.norm(pred_flow - gt_flow, dim=1).mean()
                valid_ratio = 1.0

            metrics[f'flow_{scale_name}_l1'] = l1_final.item()
            metrics[f'flow_{scale_name}_epe'] = epe.item()
            metrics[f'flow_{scale_name}_valid'] = valid_ratio
            metrics['flow_fine_seq_loss'] = seq_loss.item()
            continue

        # ── Mid: sequence loss when mid_iters > 1 ──
        if scale_name == 'mid' and 'mid_flow_preds' in pred and len(pred['mid_flow_preds']) > 1:
            mid_preds = pred['mid_flow_preds']
            n_iters = len(mid_preds)
            seq_loss = torch.tensor(0.0, device=gt_flow.device)
            for i, iter_flow in enumerate(mid_preds):
                iter_w = gamma ** (n_iters - 1 - i)
                seq_loss = seq_loss + iter_w * _masked_flow_loss(iter_flow, gt_flow, mask)
            total = total + w * seq_loss

            pred_flow = mid_preds[-1]
            if mask is not None:
                n_valid = mask.sum().clamp(min=1.0)
                epe_map = torch.norm(pred_flow - gt_flow, dim=1, keepdim=True)
                epe = (epe_map * mask).sum() / n_valid
                valid_ratio = mask.mean().item()
            else:
                epe = torch.norm(pred_flow - gt_flow, dim=1).mean()
                valid_ratio = 1.0

            metrics[f'flow_{scale_name}_l1'] = _masked_flow_loss(pred_flow, gt_flow, mask).item()
            metrics[f'flow_{scale_name}_epe'] = epe.item()
            metrics[f'flow_{scale_name}_valid'] = valid_ratio
            metrics['flow_mid_seq_loss'] = seq_loss.item()
            continue

        # ── Coarse / Mid(single iter): 标准单次 loss ──
        pred_flow = pred[pred_key]
        l1 = _masked_flow_loss(pred_flow, gt_flow, mask)

        if mask is not None:
            # EPE for monitoring
            epe_map = torch.norm(pred_flow - gt_flow, dim=1, keepdim=True)
            n_valid = mask.sum().clamp(min=1.0)
            epe = (epe_map * mask).sum() / n_valid
            valid_ratio = mask.mean().item()
        else:
            epe = torch.norm(pred_flow - gt_flow, dim=1).mean()
            valid_ratio = 1.0

        total = total + w * l1

        metrics[f'flow_{scale_name}_l1'] = l1.item()
        metrics[f'flow_{scale_name}_epe'] = epe.item()
        metrics[f'flow_{scale_name}_valid'] = valid_ratio

    metrics['flow_total'] = total.item()
    return total, metrics


def confidence_regularization_loss(
    pred: Dict[str, torch.Tensor],
    gt_flows: Dict[str, torch.Tensor] = None,
    gt_masks: Dict[str, torch.Tensor] = None,
    conf_coverage_range: Tuple[float, float] = (0.05, 0.95),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    置信度正则化损失.

    防止置信度坍塌 (全 0 或全 1):
      1. Coverage: 鼓励置信度均值接近合理范围 (soft penalty when mean < 0.3 or > 0.9)
      2. Calibration (可选): 高置信度处 flow 误差应更小 — 鼓励置信度与实际精度对齐

    只对 fine 层的置信度计算，因为精细层直接输入几何求解器。

    Args:
        pred: model output with conf_fine (B, 1, H, W)
        gt_flows: optional, for calibration loss
        gt_masks: optional

    Returns:
        loss, metrics
    """
    conf = pred.get('conf_fine')
    if conf is None:
        return torch.tensor(0.0), {}

    device = conf.device
    metrics = {}

    # ── 1. Coverage loss: 防止均值极端 ──
    # For directional confidence (B,2,H,W), average across both channels
    conf_mean = conf.mean()
    metrics['conf_mean'] = conf_mean.item()
    metrics['conf_std'] = conf.std().item()

    # Soft penalty: push mean toward [target_low, target_high] range
    # Configurable per dataset — indoor scenes can use wider range
    target_low, target_high = conf_coverage_range
    if conf_mean < target_low:
        coverage_loss = (target_low - conf_mean) ** 2
    elif conf_mean > target_high:
        coverage_loss = (conf_mean - target_high) ** 2
    else:
        coverage_loss = torch.tensor(0.0, device=device)

    # ── 2. Calibration loss: 置信度应与 flow 精度一致 ──
    cal_loss = torch.tensor(0.0, device=device)
    if gt_flows is not None and 'fine' in gt_flows:
        flow_pred = pred.get('flow_fine')
        flow_gt = gt_flows['fine']
        mask = gt_masks.get('fine') if gt_masks else None

        if flow_pred is not None:
            # Per-pixel flow error (detached, as supervision signal)
            flow_err = torch.norm(flow_pred.detach() - flow_gt.detach(), dim=1, keepdim=True)
            # Normalize error to [0, 1] range roughly (clamp at 10 pixels)
            flow_err_norm = (flow_err / 10.0).clamp(0, 1)
            # Target confidence: high where error is low
            # For directional conf (B,2,H,W), expand target to match
            target_conf = 1.0 - flow_err_norm
            if conf.shape[1] == 2:
                target_conf = target_conf.expand_as(conf)

            if mask is not None:
                n_valid = mask.sum().clamp(min=1.0)
                cal_loss = ((conf - target_conf).pow(2) * mask).sum() / n_valid
            else:
                cal_loss = (conf - target_conf).pow(2).mean()

    total = coverage_loss + 0.1 * cal_loss
    metrics['conf_coverage_loss'] = coverage_loss.item()
    metrics['conf_cal_loss'] = cal_loss.item()
    metrics['conf_reg_total'] = total.item()

    return total, metrics


def diversity_regularization_loss(
    pred: Dict[str, torch.Tensor],
    scales: Tuple[str, ...] = ('coarse', 'mid', 'fine'),
    max_samples: int = 64,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Diversity regularization: penalize inter-pixel cosine similarity collapse.

    When decoded features collapse to a single direction (cosine similarity → 1),
    the correlation volume becomes uninformative and the flow head cannot learn.
    This loss pushes off-diagonal cosine similarity DOWN by penalizing the mean.

    Args:
        pred: model output containing decoded_q_{scale} tensors (B, C, H, W)
        scales: which decoder outputs to regularize
        max_samples: randomly sample this many spatial positions (saves memory)

    Returns:
        loss, metrics dict
    """
    device = None
    total_loss = torch.tensor(0.0)
    metrics = {}
    n_terms = 0

    for scale in scales:
        key = f'decoded_q_{scale}'
        feats = pred.get(key)
        if feats is None:
            continue
        if device is None:
            device = feats.device
            total_loss = total_loss.to(device)

        B, C, H, W = feats.shape
        HW = H * W

        # Reshape to (B, C, HW) — already L2-normalized by ScaleDecoder
        f = feats.reshape(B, C, HW)

        # Subsample spatial positions if too many (saves memory for large maps)
        if HW > max_samples:
            idx = torch.randperm(HW, device=device)[:max_samples]
            f = f[:, :, idx]
            n = max_samples
        else:
            n = HW

        # Cosine similarity matrix: (B, n, n)
        sim = torch.bmm(f.permute(0, 2, 1), f)

        # Off-diagonal mean — this is what we want to minimize
        mask = 1.0 - torch.eye(n, device=device).unsqueeze(0)
        off_diag_mean = (sim * mask).sum() / (mask.sum() * B)

        # Penalty: ReLU(off_diag_mean - target) squared
        # Target 0.5 = healthy diversity; penalty kicks in above 0.7
        target = 0.5
        penalty = F.relu(off_diag_mean - target) ** 2

        total_loss = total_loss + penalty
        n_terms += 1
        metrics[f'div_{scale}_sim'] = off_diag_mean.item()
        metrics[f'div_{scale}_penalty'] = penalty.item()

    if n_terms > 0:
        total_loss = total_loss / n_terms

    metrics['div_total'] = total_loss.item()
    return total_loss, metrics


def pose_loss(
    delta_xi: torch.Tensor,
    pose_init: torch.Tensor,
    pose_gt: torch.Tensor,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
    rot_loss_type: str = 'acos',
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    位姿损失: 预测更新后的位姿与 GT 的误差.

    delta_xi → se3_exp → pose_pred = T_delta @ pose_init

    **全部在 fp32 下计算**，避免 acos 和 se3_exp 在 fp16 下产生 NaN。

    rot_loss_type:
      - 'acos': loss = acos(cos_angle), 梯度 = 1/sin(θ), 在小角度梯度爆炸
      - 'cosine': loss = 1 - cos(θ), 梯度 = sin(θ), 平滑稳定 (CorrPoseNet 风格)

    Returns:
        loss, metrics
    """
    # 强制 fp32 — acos 的梯度在 cos_angle≈±1 时趋于无穷，
    # fp16 精度不足 (eps≈1e-3) 会导致 clamp 失效 → NaN
    with torch.cuda.amp.autocast(enabled=False):
        delta_xi = delta_xi.float()
        pose_init = pose_init.float()
        pose_gt = pose_gt.float()

        B = delta_xi.shape[0]

        # 应用位姿更新
        T_delta = se3_exp(delta_xi)                    # (B, 4, 4)
        pose_pred = torch.bmm(T_delta, pose_init)      # (B, 4, 4)

        # 旋转: cos_angle ∈ [-1, 1]
        R_pred = pose_pred[:, :3, :3]
        R_gt = pose_gt[:, :3, :3]
        R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)

        # 旋转损失
        if rot_loss_type == 'cosine':
            # CorrPoseNet 风格: 1-cos(θ), gradient ∝ sin(θ), 在 θ→0 时平滑趋零
            rot_loss_val = (1.0 - cos_angle).mean()
        else:
            # 原始: acos(cos_angle), gradient ∝ 1/sin(θ), 小角度梯度爆炸
            cos_clamped = cos_angle.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
            rot_loss_val = torch.acos(cos_clamped).mean()

        # 旋转误差度数 (用于 metrics, 用更紧的 clamp)
        cos_for_metric = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        rot_err_deg = torch.acos(cos_for_metric) * 180.0 / math.pi

        # 平移误差 (米)
        t_pred = pose_pred[:, :3, 3]
        t_gt = pose_gt[:, :3, 3]
        trans_err = torch.norm(t_pred - t_gt, dim=1)

        # Loss: rot + trans
        loss = rot_loss_val * rot_weight + trans_err.mean() * trans_weight

    return loss, {
        'rot_err_deg': rot_err_deg.mean().item(),
        'trans_err_mm': (trans_err * 1000).mean().item(),
        'pose_loss': loss.item(),
    }


# ==============================================================================
#  Trainer Class
# ==============================================================================

class MSFlowTrainer:
    """MSFlowPoseNet 训练器."""

    def __init__(self, config: Dict, resume_path: str = None,
                 warmstart_path: str = None, continue_path: str = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 性能优化
        torch.backends.cudnn.benchmark = True

        # 设置输出目录
        exp_name = config.get('exp_name', 'exp029_ms_flow')
        self.output_dir = Path(config.get('output_dir', f'output/{exp_name}'))
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 保存 config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.writer = SummaryWriter(str(self.log_dir))

        # AMP 混合精度
        tc = config.get('training', {})
        self.use_amp = tc.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 初始化
        self._init_renderer()
        self._init_model()
        self._init_datasets()
        self._init_optimizer()

        self.epoch = 0
        self.global_step = 0
        self.best_val_rot = float('inf')

        if resume_path:
            self._load_checkpoint(resume_path)
        elif continue_path:
            self._continue_training(continue_path)
        elif warmstart_path:
            self._warmstart(warmstart_path)

    def _init_renderer(self):
        """加载 3DGS 渲染器."""
        rc = self.config['renderer']
        ply_path = rc['ply_path']
        scale_model_paths = rc['scale_model_paths']

        print("[Trainer] 加载 MultiScaleRenderer...")
        self.renderer = MultiScaleRenderer(
            ply_path=ply_path,
            scale_model_paths=scale_model_paths,
            device=self.device,
            img_height=rc.get('img_height', 480),
            img_width=rc.get('img_width', 640),
            fx=rc.get('fx', 320.0),
            fy=rc.get('fy', 320.0),
            cx=rc.get('cx', 319.5),
            cy=rc.get('cy', 239.5),
        )
        # v1 uses default SCALE_RESOLUTIONS (7×10, 15×20, 35×46)
        print("  ✓ Renderer loaded (v1 resolutions)")

    def _init_model(self):
        """创建 MSFlowPoseNet."""
        mc = self.config.get('model', {})
        rc = self.config.get('renderer', {})
        intrinsics = {
            'fx': rc.get('fx', 320.0),
            'fy': rc.get('fy', 320.0),
            'cx': rc.get('cx', 319.5),
            'cy': rc.get('cy', 239.5),
        }
        self.model = MSFlowPoseNet(
            hidden_dim=mc.get('hidden_dim', 128),
            decode_dim=mc.get('decode_dim', 64),
            local_radius=mc.get('local_radius', 4),
            damping=mc.get('damping', 1e-3),
            coarse_hw=tuple(mc.get('coarse_hw', [7, 10])),
            mid_hw=tuple(mc.get('mid_hw', [15, 20])),
            fine_hw=tuple(mc.get('fine_hw', [35, 46])),
            fine_iters=mc.get('fine_iters', 4),
            mid_iters=mc.get('mid_iters', 1),
            corr_temperature=mc.get('corr_temperature', 1.0),
            intrinsics=intrinsics,
            img_hw=(rc.get('img_height', 480), rc.get('img_width', 640)),
            coarse_in_dim=mc.get('coarse_in_dim', 512),
            mid_in_dim=mc.get('mid_in_dim', 512),
            fine_sd_in_dim=mc.get('fine_sd_in_dim', 512),
            fine_dino_in_dim=mc.get('fine_dino_in_dim', 768),
            irls_iters=mc.get('irls_iters', 0),
            irls_huber_k=mc.get('irls_huber_k', 1.345),
            deep_flow_head=mc.get('deep_flow_head', False),
            cross_scale_context=mc.get('cross_scale_context', False),
            cross_scale_dim=mc.get('cross_scale_dim', 32),
            pose_refinement=mc.get('pose_refinement', False),
            corr_dilations=tuple(mc['corr_dilations']) if mc.get('corr_dilations') else None,
            geometry_upsample=mc.get('geometry_upsample', 1),
            multiscale_consistency=mc.get('multiscale_consistency', False),
            ms_consistency_sigma=mc.get('ms_consistency_sigma', 1.0),
            pixel_stride=mc.get('pixel_stride', 1),
            adaptive_damping=mc.get('adaptive_damping', False),
            adaptive_damping_max=mc.get('adaptive_damping_max', 0.1),
            adaptive_damping_cond_thresh=mc.get('adaptive_damping_cond_thresh', 1e4),
            positional_encoding=mc.get('positional_encoding', False),
            pe_mode=mc.get('pe_mode', 'concat'),
            pe_dim=mc.get('pe_dim', 32),
            skip_coarse_flow=mc.get('skip_coarse_flow', False),
            learnable_temperature=mc.get('learnable_temperature', False),
            directional_confidence=mc.get('directional_confidence', False),
            dino_all_scales=mc.get('dino_all_scales', False),
            dino_replace_sd=mc.get('dino_replace_sd', False),
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        irls_str = f", irls_iters={mc.get('irls_iters', 0)}" if mc.get('irls_iters', 0) > 0 else ""
        dfh_str = ", deep_flow_head" if mc.get('deep_flow_head', False) else ""
        csc_str = ", cross_scale_ctx" if mc.get('cross_scale_context', False) else ""
        pr_str = ", pose_refine" if mc.get('pose_refinement', False) else ""
        cd_str = f", corr_dilations={mc['corr_dilations']}" if mc.get('corr_dilations') else ""
        gu_str = f", geo_up={mc['geometry_upsample']}×" if mc.get('geometry_upsample', 1) > 1 else ""
        msc_str = ", ms_consistency" if mc.get('multiscale_consistency', False) else ""
        pe_str = f", pos_enc({mc.get('pe_mode', 'concat')},{mc.get('pe_dim', 32)})" if mc.get('positional_encoding', False) else ""
        scf_str = ", skip_coarse_flow" if mc.get('skip_coarse_flow', False) else ""
        lt_str = ", learnable_temp" if mc.get('learnable_temperature', False) else ""
        dc_str = ", dir_conf" if mc.get('directional_confidence', False) else ""
        ad_str = ", adaptive_damp" if mc.get('adaptive_damping', False) else ""
        das_str = ", dino_all_scales" if mc.get('dino_all_scales', False) else ""
        drs_str = ", dino_replace_sd" if mc.get('dino_replace_sd', False) else ""
        print(f"[Model] MSFlowPoseNet: {n_params/1e6:.2f}M trainable params, "
              f"fine_iters={mc.get('fine_iters', 4)}, "
              f"mid_iters={mc.get('mid_iters', 1)}, "
              f"corr_temp={mc.get('corr_temperature', 1.0)}{irls_str}{dfh_str}{csc_str}{pr_str}{cd_str}{gu_str}{msc_str}{pe_str}{scf_str}{lt_str}{dc_str}{ad_str}{das_str}{drs_str}")

    def _init_datasets(self):
        """准备训练/验证数据集."""
        dc = self.config['data']

        if dc.get('val_feature_dir'):
            # 独立验证集 (例如 Sequence 2)
            self.train_dataset = PoseDatasetV4(
                feature_base_dir=dc['train_feature_dir'],
                traj_path=dc['train_traj_path'],
                depth_dir=dc.get('train_depth_dir'),
                noise_rot_deg=dc.get('noise_rot_deg', 15.0),
                noise_trans_m=dc.get('noise_trans_m', 0.5),
                is_train=True,
            )
            self.val_dataset = PoseDatasetV4(
                feature_base_dir=dc['val_feature_dir'],
                traj_path=dc['val_traj_path'],
                depth_dir=dc.get('val_depth_dir'),
                noise_rot_deg=dc.get('val_noise_rot_deg', 15.0),
                noise_trans_m=dc.get('val_noise_trans_m', 0.5),
                is_train=False,
                netvlad_poses_path=dc.get('val_netvlad_poses'),
            )
        else:
            # 从训练集自动拆分 val (最后 10%)
            full_dataset = PoseDatasetV4(
                feature_base_dir=dc['train_feature_dir'],
                traj_path=dc['train_traj_path'],
                depth_dir=dc.get('train_depth_dir'),
                noise_rot_deg=dc.get('noise_rot_deg', 15.0),
                noise_trans_m=dc.get('noise_trans_m', 0.5),
                is_train=True,
            )
            n_total = len(full_dataset)
            n_val = max(1, int(n_total * dc.get('val_split_ratio', 0.1)))
            n_train = n_total - n_val
            self.train_dataset, self.val_dataset = \
                torch.utils.data.random_split(
                    full_dataset, [n_train, n_val],
                    generator=torch.Generator().manual_seed(42))
            # val subset 使用较小噪声以更好检测进步
            print(f"[Data] Auto-split: {n_train} train + {n_val} val from {n_total} total")

        # persistent_workers=False so noise curriculum updates propagate to workers
        # (persistent workers keep stale copies of dataset attributes)
        n_workers = dc.get('num_workers', 4)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=dc.get('batch_size', 4),
            shuffle=True,
            num_workers=n_workers,
            collate_fn=collate_v4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=2 if n_workers > 0 else None,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=dc.get('val_batch_size', 4),
            shuffle=False,
            num_workers=2,
            collate_fn=collate_v4,
            pin_memory=True,
            persistent_workers=False,
        )

        print(f"[Data] Train: {len(self.train_dataset)} samples, "
              f"Val: {len(self.val_dataset)} samples")

    def _init_optimizer(self):
        tc = self.config.get('training', {})
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=tc.get('lr', 1e-4),
            weight_decay=tc.get('weight_decay', 1e-5),
        )
        total_epochs = tc.get('epochs', 100)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=tc.get('min_lr', 1e-6),
        )
        self.grad_clip = tc.get('grad_clip', 1.0)
        self.total_epochs = total_epochs
        self.phase1_epochs = tc.get('phase1_epochs', 20)

        # Loss weights
        lc = tc.get('loss', {})
        self.flow_weights = lc.get('flow_weights',
                                    {'coarse': 0.1, 'mid': 0.3, 'fine': 1.0})
        self.pose_weight = lc.get('pose_weight', 1.0)
        self.rot_weight = lc.get('rot_weight', 1.0)
        self.trans_weight = lc.get('trans_weight', 1.0)
        self.rot_loss_type = lc.get('rot_loss_type', 'cosine')  # 'cosine' (stable) or 'acos'
        self.gamma = lc.get('gamma', 0.8)  # RAFT sequence loss decay
        self.use_huber = lc.get('use_huber', True)
        self.huber_delta = lc.get('huber_delta', 5.0)
        self.conf_reg_weight = lc.get('conf_reg_weight', 0.01)
        self.conf_coverage_range = tuple(lc.get('conf_coverage_range', [0.05, 0.95]))
        self.div_reg_weight = lc.get('div_reg_weight', 0.0)  # diversity loss weight

        # ── 噪声课程学习 ──
        nc = tc.get('noise_curriculum', {})
        self.use_noise_curriculum = nc.get('enabled', False)
        self.noise_warmup_epochs = nc.get('warmup_epochs', 30)
        dc = self.config['data']
        self.noise_rot_max = dc.get('noise_rot_deg', 8.0)
        self.noise_trans_max = dc.get('noise_trans_m', 0.25)
        self.noise_rot_min = nc.get('noise_rot_min', 2.0)
        self.noise_trans_min = nc.get('noise_trans_min', 0.05)

        # ── Pose loss warmup ──
        pw = tc.get('pose_warmup', {})
        self.use_pose_warmup = pw.get('enabled', False)
        self.pose_warmup_epochs = pw.get('warmup_epochs', 10)
        self.pose_weight_min = pw.get('min_weight', 0.01)

        # ── 外层迭代精化 (Outer-Loop Iterative Refinement) ──
        self.num_outer_iters = tc.get('outer_iters', 1)
        self.val_outer_iters = tc.get('val_outer_iters', 1)
        self.gamma_outer = lc.get('gamma_outer', 0.8)

    @torch.no_grad()
    def _render_batch(
        self,
        poses_w2c: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        渲染一个 batch 的多尺度特征 + 深度.

        Returns:
            render_feats: {'coarse': (B,1280,8,10), 'mid': ..., 'fine_sd': ..., 'fine_dino': ...}
            depth: (B, 35, 46)
        """
        scales = ['coarse', 'mid', 'fine_sd', 'fine_dino']
        result = self.renderer.render_batch(
            poses_w2c, scales=scales, return_depth=True)

        render_feats = {
            'coarse': result['coarse_feat'],
            'mid': result['mid_feat'],
            'fine_sd': result['fine_sd_feat'],
            'fine_dino': result['fine_dino_feat'],
        }
        depth = result.get('depth_map')  # (B, 35, 46)
        return render_feats, depth

    @torch.no_grad()
    def _compute_gt_flows(
        self,
        pose_init: torch.Tensor,
        pose_gt: torch.Tensor,
        depth: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """计算各尺度的 GT flow 和有效性 mask (不需要梯度)."""
        gt_flows = {}
        gt_masks = {}

        # 使用模型实例属性获取分辨率
        for name, hw in [('coarse', self.model.COARSE_HW),
                          ('mid', self.model.MID_HW),
                          ('fine', self.model.FINE_HW)]:
            flow, mask = self.model.compute_gt_flow(
                pose_init, pose_gt, depth, hw)
            gt_flows[name] = flow
            gt_masks[name] = mask

        return gt_flows, gt_masks

    # ------------------------------------------------------------------
    #  Noise curriculum helper
    # ------------------------------------------------------------------
    def _update_noise_for_epoch(self, epoch: int):
        """根据课程学习策略动态调整数据集噪声参数.

        线性从 (noise_rot_min, noise_trans_min) 增长到 (noise_rot_max, noise_trans_max)
        在 warmup_epochs 个 epoch 内完成。
        """
        if not self.use_noise_curriculum:
            return

        ratio = min(1.0, epoch / max(1, self.noise_warmup_epochs))
        cur_rot = self.noise_rot_min + (self.noise_rot_max - self.noise_rot_min) * ratio
        cur_trans = self.noise_trans_min + (self.noise_trans_max - self.noise_trans_min) * ratio

        # 获取底层 PoseDatasetV4 对象 (处理 Subset 包装)
        def _set_noise(ds, rot, trans):
            real = getattr(ds, 'dataset', ds)  # Subset.dataset or ds itself
            real.noise_rot_deg = rot
            real.noise_trans_m = trans

        _set_noise(self.train_dataset, cur_rot, cur_trans)
        # 验证集保持固定最大噪声, 这样验证指标可比较
        print(f"  [Curriculum] noise_rot={cur_rot:.1f}° noise_trans={cur_trans:.3f}m (ratio={ratio:.2f})")

    # ------------------------------------------------------------------
    #  Effective pose weight (warmup)
    # ------------------------------------------------------------------
    def _effective_pose_weight(self, epoch: int) -> float:
        """Phase2 pose loss 权重从 min_weight 线性增长到 pose_weight."""
        if not self.use_pose_warmup:
            return self.pose_weight
        phase2_epoch = epoch - self.phase1_epochs  # 从 phase2 开始计
        ratio = min(1.0, phase2_epoch / max(1, self.pose_warmup_epochs))
        return self.pose_weight_min + (self.pose_weight - self.pose_weight_min) * ratio

    def _train_step(
        self,
        batch: Dict,
        use_pose_loss: bool,
        effective_pose_weight: float = 1.0,
    ) -> Dict[str, float]:
        """单步训练 (含外层迭代精化).

        每次外层迭代:
          1. 从 pose_cur 渲染多尺度特征
          2. 前向推理得到 flow + delta_xi
          3. 计算 flow loss + pose loss
          4. 按 gamma_outer^(N-1-i) 加权后立即 backward (节省显存)
          5. 用 delta_xi 更新 pose_cur (detach, 切断迭代间梯度)

        Returns:
            Dict[str, float]: 最后一次迭代的 metrics (已在内部完成 backward)
        """
        query_feats = {k: v.to(self.device) for k, v in batch['query_feats'].items()}
        pose_gt = batch['pose_gt'].to(self.device)
        pose_cur = batch['initial_pose'].to(self.device)
        depth_gt = batch.get('depth')

        N = self.num_outer_iters
        all_metrics = {}
        total_loss_val = 0.0

        for outer_i in range(N):
            # 1. 渲染
            render_feats, depth_rendered = self._render_batch(pose_cur)
            depth = depth_rendered
            if depth is None and depth_gt is not None:
                depth = depth_gt.to(self.device)

            # 2. Forward (with AMP)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.model(query_feats, render_feats, depth)

            # 3. GT flows from current pose
            gt_flows, gt_masks = self._compute_gt_flows(pose_cur, pose_gt, depth)

            # 4. Flow loss
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                flow_loss, flow_metrics = multiscale_flow_loss(
                    pred, gt_flows, gt_masks, self.flow_weights, gamma=self.gamma,
                    use_huber=self.use_huber, huber_delta=self.huber_delta)

                iter_loss = flow_loss

                # Confidence regularization
                conf_loss, conf_metrics = confidence_regularization_loss(
                    pred, gt_flows, gt_masks,
                    conf_coverage_range=self.conf_coverage_range)
                iter_loss = iter_loss + self.conf_reg_weight * conf_loss

                # Diversity regularization (prevents decoder feature collapse)
                if self.div_reg_weight > 0:
                    div_loss, div_metrics = diversity_regularization_loss(pred)
                    iter_loss = iter_loss + self.div_reg_weight * div_loss
                else:
                    div_metrics = {}

                # Pose loss (Phase 2)
                if use_pose_loss and 'delta_xi' in pred:
                    p_loss, p_metrics = pose_loss(
                        pred['delta_xi'], pose_cur, pose_gt,
                        rot_weight=self.rot_weight,
                        trans_weight=self.trans_weight,
                        rot_loss_type=self.rot_loss_type,
                    )
                    iter_loss = iter_loss + effective_pose_weight * p_loss

            # NaN check
            if torch.isnan(iter_loss) or torch.isinf(iter_loss):
                return {'nan_step': True}

            # 5. 外层 sequence loss 加权: gamma_outer^(N-1-i)
            iter_w = self.gamma_outer ** (N - 1 - outer_i)
            scaled_loss = iter_w * iter_loss / N
            self.scaler.scale(scaled_loss).backward()

            total_loss_val += iter_loss.item()

            # 6. 记录最后一次迭代的 metrics
            if outer_i == N - 1:
                all_metrics = flow_metrics.copy()
                all_metrics.update(conf_metrics)
                all_metrics.update(div_metrics)
                if use_pose_loss and 'delta_xi' in pred:
                    all_metrics.update(p_metrics)
                    all_metrics['eff_pose_w'] = effective_pose_weight

            # 7. 更新 pose (除最后一次外)
            if outer_i < N - 1 and 'delta_xi' in pred:
                with torch.no_grad():
                    T_delta = se3_exp(pred['delta_xi'].float())
                    pose_cur = torch.bmm(T_delta, pose_cur.float()).detach()

        all_metrics['total_loss'] = total_loss_val / N
        return all_metrics

    def train_epoch(self, epoch: int):
        """训练一个 epoch."""
        self.model.train()
        use_pose_loss = epoch >= self.phase1_epochs
        eff_pw = self._effective_pose_weight(epoch) if use_pose_loss else 0.0

        epoch_metrics = {}
        pbar = tqdm(self.train_loader,
                     desc=f"Epoch {epoch}/{self.total_epochs}",
                     leave=False)

        for batch in pbar:
            self.optimizer.zero_grad()

            # _train_step 内部处理 autocast + per-iteration backward
            metrics = self._train_step(batch, use_pose_loss, eff_pw)

            # NaN guard
            if metrics.get('nan_step'):
                self.global_step += 1
                continue

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                # 检查 unscale 后梯度是否包含 NaN/Inf
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
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1

            # 累积 metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k != 'nan_step':
                    epoch_metrics.setdefault(k, []).append(v)

            # 更新 progress bar
            pbar_dict = {'loss': f"{metrics.get('total_loss', 0):.4f}"}
            if 'rot_err_deg' in metrics:
                pbar_dict['rot'] = f"{metrics['rot_err_deg']:.2f}°"
                pbar_dict['trans'] = f"{metrics['trans_err_mm']:.1f}mm"
            pbar.set_postfix(pbar_dict)

            # TensorBoard
            if self.global_step % 50 == 0:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and k != 'nan_step':
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)
                self.writer.add_scalar(
                    'train/lr', self.optimizer.param_groups[0]['lr'],
                    self.global_step)

        # Epoch averages
        avg = {k: np.mean(v) for k, v in epoch_metrics.items()}
        phase = "Phase2(flow+pose)" if use_pose_loss else "Phase1(flow only)"
        msg = (f"[Train E{epoch}] {phase}  "
               f"loss={avg.get('total_loss', 0):.4f}  "
               f"flow={avg.get('flow_total', 0):.4f}")
        if 'rot_err_deg' in avg:
            msg += f"  rot={avg['rot_err_deg']:.2f}°  trans={avg['trans_err_mm']:.1f}mm"
        if 'div_coarse_sim' in avg:
            msg += f"  div_sim=[{avg['div_coarse_sim']:.3f}/{avg['div_mid_sim']:.3f}/{avg['div_fine_sim']:.3f}]"
        if self.num_outer_iters > 1:
            msg += f"  ({self.num_outer_iters} outer iters)"
        print(msg)

        return avg

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证 (支持外层迭代精化)."""
        torch.cuda.empty_cache()
        self.model.eval()
        all_rot_errs, all_trans_errs = [], []
        all_flow_epe = []
        val_metrics = {}
        N = self.val_outer_iters

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            query_feats = {k: v.to(self.device) for k, v in batch['query_feats'].items()}
            pose_gt = batch['pose_gt'].to(self.device)
            pose_cur = batch['initial_pose'].to(self.device)

            # ── 外层迭代精化 ──
            for outer_i in range(N):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    render_feats, depth = self._render_batch(pose_cur)
                    pred = self.model(query_feats, render_feats, depth)

                # 除最后一次外, 用预测 delta_xi 更新 pose
                if outer_i < N - 1 and 'delta_xi' in pred:
                    with torch.cuda.amp.autocast(enabled=False):
                        T_delta = se3_exp(pred['delta_xi'].float())
                        pose_cur = torch.bmm(T_delta, pose_cur.float())

            # ── 最终迭代评估 ──
            # Flow EPE (residual flow from final pose to GT)
            if depth is not None:
                gt_fine, gt_mask = self.model.compute_gt_flow(
                    pose_cur, pose_gt, depth, self.model.FINE_HW)
                epe_map = torch.norm(
                    pred['flow_fine'].float() - gt_fine.float(),
                    dim=1, keepdim=True)
                n_valid = gt_mask.sum().clamp(min=1.0)
                epe = (epe_map * gt_mask).sum() / n_valid
                all_flow_epe.append(epe.item())

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
                    trans_err = torch.norm(t_pred - t_gt, dim=1) * 1000  # mm

                all_rot_errs.extend(rot_err.cpu().tolist())
                all_trans_errs.extend(trans_err.cpu().tolist())

        if all_rot_errs:
            rot = np.array(all_rot_errs)
            trans = np.array(all_trans_errs)
            val_metrics = {
                'val_rot_mean': float(np.mean(rot)),
                'val_rot_median': float(np.median(rot)),
                'val_trans_mean': float(np.mean(trans)),
                'val_trans_median': float(np.median(trans)),
                'val_pct_1deg': float(np.mean(rot < 1.0) * 100),
                'val_pct_5deg': float(np.mean(rot < 5.0) * 100),
            }
            if all_flow_epe:
                val_metrics['val_flow_epe'] = float(np.mean(all_flow_epe))
            iters_str = f"  ({N} iters)" if N > 1 else ""
            print(f"[Val E{epoch}]  rot={val_metrics['val_rot_mean']:.2f}° "
                  f"(med {val_metrics['val_rot_median']:.2f}°)  "
                  f"trans={val_metrics['val_trans_mean']:.1f}mm  "
                  f"<1°={val_metrics['val_pct_1deg']:.1f}%"
                  f"  flow_epe={val_metrics.get('val_flow_epe', -1):.2f}"
                  f"{iters_str}")

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
        self.epoch = ckpt['epoch'] + 1
        self.global_step = ckpt['global_step']
        self.best_val_rot = ckpt.get('best_val_rot', float('inf'))

        # If config epochs > checkpoint's original epochs, extend with fresh cosine cycle
        old_epochs = ckpt.get('config', {}).get('training', {}).get('epochs', self.total_epochs)
        if self.total_epochs > old_epochs and self.epoch >= old_epochs:
            remaining = self.total_epochs - self.epoch
            tc = self.config.get('training', {})
            ext_lr = tc.get('continue_lr', tc.get('min_lr', 1e-6) * 10)
            for pg in self.optimizer.param_groups:
                pg['lr'] = ext_lr
                pg['initial_lr'] = ext_lr
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(remaining, 1),
                eta_min=tc.get('min_lr', 1e-6),
            )
            print(f"  Extended training: fresh cosine LR={ext_lr} over {remaining} epochs")
        else:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"  Resumed at epoch {self.epoch}, step {self.global_step}")

    def _warmstart(self, path: str):
        """Warmstart: 只加载模型权重, 保持新的 optimizer/scheduler/epoch.
        Handles size mismatches gracefully (e.g., when corr_dilations changes
        the corr_encoder input channels)."""
        print(f"[Warmstart] Loading model weights from {path}")
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt['model_state_dict']

        # Filter out keys with size mismatch
        model_state = self.model.state_dict()
        filtered_state = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_state and v.shape != model_state[k].shape:
                skipped.append(f"{k}: ckpt {list(v.shape)} vs model {list(model_state[k].shape)}")
            else:
                filtered_state[k] = v

        if skipped:
            print(f"  Skipped {len(skipped)} size-mismatched keys (random init):")
            for s in skipped:
                print(f"    {s}")

        missing, unexpected = self.model.load_state_dict(filtered_state, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        src_epoch = ckpt.get('epoch', '?')
        print(f"  Loaded weights from epoch {src_epoch}, fresh optimizer")

    def _continue_training(self, path: str):
        """Continue training: load model + optimizer state, fresh scheduler.
        The scheduler is created to cover REMAINING epochs only (T_max = total - start),
        starting at config LR and decaying to min_lr. This avoids the LR jump
        that occurs when stepping a full-length cosine schedule forward."""
        print(f"[Continue] Loading model + optimizer from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.epoch = ckpt['epoch'] + 1
        self.global_step = ckpt.get('global_step', 0)
        self.best_val_rot = ckpt.get('best_val_rot', float('inf'))

        tc = self.config.get('training', {})
        remaining = self.total_epochs - self.epoch
        cont_lr = tc.get('continue_lr', tc.get('lr', 1e-4))
        min_lr = tc.get('min_lr', 1e-6)

        # Reset optimizer LR to config value, then create fresh cosine schedule
        for pg in self.optimizer.param_groups:
            pg['lr'] = cont_lr
            pg['initial_lr'] = cont_lr
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(remaining, 1), eta_min=min_lr,
        )

        old_tmax = ckpt.get('config', {}).get('training', {}).get('epochs', '?')
        print(f"  Restored model+optimizer from epoch {ckpt['epoch']}")
        print(f"  Fresh scheduler: T_max={remaining} (remaining of {self.total_epochs}), LR={cont_lr}")
        print(f"  Continuing from epoch {self.epoch}, best_val_rot={self.best_val_rot:.2f}°")

    def train(self):
        """完整训练循环."""
        print(f"\n{'='*60}")
        print(f"  MSFlowPoseNet Training")
        print(f"  Epochs: {self.total_epochs} (Phase1: {self.phase1_epochs})")
        print(f"  Outer iters: {self.num_outer_iters} train / {self.val_outer_iters} val")
        print(f"  Output: {self.output_dir}")
        print(f"  AMP: {self.use_amp}")
        print(f"{'='*60}\n")

        epoch_times = []
        train_start = time.time()

        for epoch in range(self.epoch, self.total_epochs):
            self.epoch = epoch
            t0 = time.time()

            # 噪声课程学习: 每个 epoch 开始前更新数据集噪声
            self._update_noise_for_epoch(epoch)

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Scheduler step
            self.scheduler.step()

            # Checkpoint
            is_best = False
            if val_metrics and val_metrics.get('val_rot_mean', float('inf')) < self.best_val_rot:
                self.best_val_rot = val_metrics['val_rot_mean']
                is_best = True
                print(f"  ★ New best: rot={self.best_val_rot:.2f}°")

            self._save_checkpoint(epoch, is_best)

            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            avg_epoch = np.mean(epoch_times[-5:])  # 最近5个epoch平均
            remaining = (self.total_epochs - epoch - 1) * avg_epoch
            eta_min = remaining / 60
            total_elapsed = (time.time() - train_start) / 60
            print(f"  Epoch {epoch}: {elapsed:.0f}s  "
                  f"lr={self.optimizer.param_groups[0]['lr']:.6f}  "
                  f"ETA: {eta_min:.0f}min  "
                  f"[{total_elapsed:.0f}min elapsed]\n")

        total_time = (time.time() - train_start) / 60
        print(f"\nTraining complete in {total_time:.0f}min! Best val rot: {self.best_val_rot:.2f}°")
        self.writer.close()


# ==============================================================================
#  Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MSFlowPoseNet')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--warmstart', type=str, default=None,
                        help='Warmstart from checkpoint (load model weights only)')
    parser.add_argument('--continue_training', type=str, default=None,
                        help='Continue training (load model+optimizer, fresh scheduler)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = MSFlowTrainer(config, resume_path=args.resume,
                            warmstart_path=args.warmstart,
                            continue_path=args.continue_training)
    trainer.train()


if __name__ == '__main__':
    main()
