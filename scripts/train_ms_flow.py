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
from modules.localizability_head import localizability_loss
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

    # Guard against NaN in metrics
    rot_mean = rot_err_deg.mean().item()
    trans_mean = (trans_err * 1000).mean().item()
    if not math.isfinite(rot_mean):
        rot_mean = float('nan')
    if not math.isfinite(trans_mean):
        trans_mean = float('nan')
    return loss, {
        'rot_err_deg': rot_mean,
        'trans_err_mm': trans_mean,
        'pose_loss': loss.item(),
    }


def flow_consistency_loss(
    pred: Dict[str, torch.Tensor],
    coarse_hw: Tuple[int, int],
    mid_hw: Tuple[int, int],
    fine_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Cross-scale flow consistency regularization.

    Penalizes disagreement between coarse→fine and mid→fine upsampled flows.
    Encourages the cascade to produce coherent predictions across scales,
    which helps with repetitive textures where coarse context disambiguates.

    Uses Huber loss for robustness (some disagreement at boundaries is natural).

    Returns:
        loss: scalar
        metrics: dict
    """
    flow_fine = pred.get('flow_fine')
    flow_coarse = pred.get('flow_coarse')
    flow_mid = pred.get('flow_mid')

    if flow_fine is None or flow_coarse is None:
        return torch.tensor(0.0), {}

    fH, fW = fine_hw
    cH, cW = coarse_hw
    mH, mW = mid_hw
    device = flow_fine.device

    total = torch.tensor(0.0, device=device)
    metrics = {}

    # Upsample coarse flow to fine resolution (rescale pixel values)
    flow_c_up = F.interpolate(
        flow_coarse.detach(), size=(fH, fW),
        mode='bilinear', align_corners=False)
    flow_c_up[:, 0] *= fW / cW
    flow_c_up[:, 1] *= fH / cH

    # Huber loss between fine and upsampled coarse
    diff_cf = F.huber_loss(flow_fine, flow_c_up, delta=2.0, reduction='mean')
    total = total + diff_cf
    metrics['flow_cons_cf'] = diff_cf.item()

    if flow_mid is not None:
        flow_m_up = F.interpolate(
            flow_mid.detach(), size=(fH, fW),
            mode='bilinear', align_corners=False)
        flow_m_up[:, 0] *= fW / mW
        flow_m_up[:, 1] *= fH / mH

        diff_mf = F.huber_loss(flow_fine, flow_m_up, delta=2.0, reduction='mean')
        total = total + diff_mf
        metrics['flow_cons_mf'] = diff_mf.item()
        total = total / 2.0  # average the two terms

    metrics['flow_cons_total'] = total.item()
    return total, metrics


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
            depth_pe_dim=mc.get('depth_pe_dim', 0),
            skip_coarse_flow=mc.get('skip_coarse_flow', False),
            learnable_temperature=mc.get('learnable_temperature', False),
            directional_confidence=mc.get('directional_confidence', False),
            dino_all_scales=mc.get('dino_all_scales', False),
            dino_replace_sd=mc.get('dino_replace_sd', False),
            localizability_prior=mc.get('localizability_prior', False),
            attention_coarse=mc.get('attention_coarse', False),
            attention_coarse_heads=mc.get('attention_coarse_heads', 4),
            attention_coarse_layers=mc.get('attention_coarse_layers', 2),
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
        dpe_str = f"+depth{mc.get('depth_pe_dim')}" if mc.get('depth_pe_dim', 0) > 0 and mc.get('positional_encoding', False) else ""
        scf_str = ", skip_coarse_flow" if mc.get('skip_coarse_flow', False) else ""
        lt_str = ", learnable_temp" if mc.get('learnable_temperature', False) else ""
        dc_str = ", dir_conf" if mc.get('directional_confidence', False) else ""
        ad_str = ", adaptive_damp" if mc.get('adaptive_damping', False) else ""
        das_str = ", dino_all_scales" if mc.get('dino_all_scales', False) else ""
        drs_str = ", dino_replace_sd" if mc.get('dino_replace_sd', False) else ""
        loc_str = ", loc_prior" if mc.get('localizability_prior', False) else ""
        attn_str = ", attn_coarse" if mc.get('attention_coarse', False) else ""
        print(f"[Model] MSFlowPoseNet: {n_params/1e6:.2f}M trainable params, "
              f"fine_iters={mc.get('fine_iters', 4)}, "
              f"mid_iters={mc.get('mid_iters', 1)}, "
              f"corr_temp={mc.get('corr_temperature', 1.0)}{irls_str}{dfh_str}{csc_str}{pr_str}{cd_str}{gu_str}{msc_str}{pe_str}{dpe_str}{scf_str}{lt_str}{dc_str}{ad_str}{das_str}{drs_str}{loc_str}{attn_str}")

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
        base_lr = tc.get('lr', 1e-4)
        weight_decay = tc.get('weight_decay', 1e-5)

        # ── Per-scale learning rates ──
        psl = tc.get('per_scale_lr', {})
        if psl.get('enabled', False):
            coarse_scale = psl.get('coarse_scale', 0.5)
            mid_scale = psl.get('mid_scale', 0.75)
            fine_scale = psl.get('fine_scale', 1.0)

            coarse_params, mid_params, fine_params, other_params = [], [], [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith(('coarse_dec.', 'coarse_head.', 'coarse_pe.',
                                    'coarse_dino_')):
                    coarse_params.append(param)
                elif name.startswith(('temp_coarse',)):
                    coarse_params.append(param)
                elif name.startswith(('mid_dec.', 'mid_head.', 'mid_pe.',
                                      'mid_context.', 'mid_dino_')):
                    mid_params.append(param)
                elif name.startswith(('temp_mid',)):
                    mid_params.append(param)
                elif name.startswith(('fine_dec.', 'fine_head.', 'fine_context.',
                                      'fine_pe.', 'context_net.', 'temp_fine')):
                    fine_params.append(param)
                else:
                    # GRU, cross_scale_ctx, pose_refine_head, etc.
                    fine_params.append(param)

            param_groups = [
                {'params': coarse_params, 'lr': base_lr * coarse_scale, 'name': 'coarse'},
                {'params': mid_params, 'lr': base_lr * mid_scale, 'name': 'mid'},
                {'params': fine_params, 'lr': base_lr * fine_scale, 'name': 'fine'},
            ]
            # Filter empty groups
            param_groups = [g for g in param_groups if g['params']]

            print(f"[Optimizer] Per-scale LR: coarse={base_lr*coarse_scale:.6f} "
                  f"({len(coarse_params)} tensors), mid={base_lr*mid_scale:.6f} "
                  f"({len(mid_params)} tensors), fine={base_lr*fine_scale:.6f} "
                  f"({len(fine_params)} tensors)")

            self.optimizer = optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=base_lr,
                weight_decay=weight_decay,
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
        self.flow_cons_weight = lc.get('flow_consistency_weight', 0.0)
        self.loc_loss_weight = lc.get('localizability_weight', 0.0)

        # ── 噪声课程学习 ──
        nc = tc.get('noise_curriculum', {})
        self.use_noise_curriculum = nc.get('enabled', False)
        self.noise_warmup_epochs = nc.get('warmup_epochs', 30)
        dc = self.config['data']
        self.noise_rot_max = dc.get('noise_rot_deg', 8.0)
        self.noise_trans_max = dc.get('noise_trans_m', 0.25)
        # Accept both naming conventions: noise_rot_min / start_noise_rot / start_rot_deg
        self.noise_rot_min = nc.get('noise_rot_min', nc.get('start_noise_rot', nc.get('start_rot_deg', 2.0)))
        self.noise_trans_min = nc.get('noise_trans_min', nc.get('start_noise_trans', nc.get('start_trans_m', 0.05)))

        # ── Pose loss warmup ──
        pw = tc.get('pose_warmup', {})
        self.use_pose_warmup = pw.get('enabled', False)
        self.pose_warmup_epochs = pw.get('warmup_epochs', 10)
        self.pose_weight_min = pw.get('min_weight', 0.01)

        # ── 外层迭代精化 (Outer-Loop Iterative Refinement) ──
        self.num_outer_iters = tc.get('outer_iters', 1)
        self.val_outer_iters = tc.get('val_outer_iters', 1)
        self.gamma_outer = lc.get('gamma_outer', 0.8)

        # ── Depth-warp mode: bypass 3DGS feature rendering ──
        self.use_depth_warp = tc.get('use_depth_warp', False)
        self.use_cross_frame_warp = tc.get('use_cross_frame_warp', False)
        if self.use_depth_warp or self.use_cross_frame_warp:
            from modules.depth_warp import backward_warp_features
            self._backward_warp = backward_warp_features
            self._scale_intrinsics = self.renderer.get_scale_intrinsics()
            if self.use_cross_frame_warp:
                self._init_cross_frame_cache()
                print(f"  ✓ Cross-frame warp mode enabled (warp from nearest neighbor)")
            else:
                print(f"  ✓ Depth-warp mode enabled (bypass 3DGS feature rendering)")

        # ── Pre-loaded rendered reference features (for render-extract training) ──
        self._ref_feature_cache = None  # {frame_idx: {scale: (C,H,W)}}
        ref_feature_dir = tc.get('ref_feature_dir', None)
        if ref_feature_dir:
            self._init_ref_feature_cache(ref_feature_dir)

        # ── Reference feature augmentation (train-only) ──
        ref_aug = tc.get('ref_feat_augmentation', {})
        self.ref_aug_enabled = ref_aug.get('enabled', False)
        self.ref_aug_noise_std = ref_aug.get('noise_std', 0.2)
        self.ref_aug_spatial_dropout = ref_aug.get('spatial_dropout', 0.15)
        self.ref_aug_channel_jitter = ref_aug.get('channel_jitter', 0.1)
        if self.ref_aug_enabled:
            print(f"  ✓ Ref feature augmentation: noise={self.ref_aug_noise_std}, "
                  f"dropout={self.ref_aug_spatial_dropout}, ch_jitter={self.ref_aug_channel_jitter}")

        # ── Pose-warp augmentation (train-only): perturb source pose before warping ──
        pose_aug = tc.get('pose_warp_augmentation', {})
        self.pose_aug_enabled = pose_aug.get('enabled', False)
        self.pose_aug_rot_deg = pose_aug.get('rot_deg', 5.0)
        self.pose_aug_trans_m = pose_aug.get('trans_m', 0.3)
        if self.pose_aug_enabled:
            print(f"  ✓ Pose-warp augmentation: rot={self.pose_aug_rot_deg}°, trans={self.pose_aug_trans_m}m")

    def _augment_ref_feats(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation to reference features (training only).

        Simulates imperfect reference features to bridge domain gap between
        same-frame warp (training) and cross-frame warp (inference).
        """
        augmented = {}
        for scale, feat in feats.items():
            f = feat  # (B, C, H, W)
            # 1. Gaussian noise
            if self.ref_aug_noise_std > 0:
                f = f + torch.randn_like(f) * self.ref_aug_noise_std
            # 2. Spatial dropout (zero entire spatial locations)
            if self.ref_aug_spatial_dropout > 0:
                B, C, H, W = f.shape
                mask = torch.rand(B, 1, H, W, device=f.device) > self.ref_aug_spatial_dropout
                f = f * mask.float()
            # 3. Channel-wise jitter (scale each channel randomly)
            if self.ref_aug_channel_jitter > 0:
                B, C, H, W = f.shape
                jitter = 1.0 + (torch.rand(B, C, 1, 1, device=f.device) - 0.5) * 2 * self.ref_aug_channel_jitter
                f = f * jitter
            augmented[scale] = f
        return augmented

    def _perturb_source_pose(self, pose_gt: torch.Tensor) -> torch.Tensor:
        """Perturb the GT pose before warping to simulate cross-frame geometric differences.

        Creates realistic occlusion/distortion artifacts matching actual nearest-neighbor warping.
        """
        from modules.lie_algebra import se3_exp
        B = pose_gt.shape[0]
        noise_rot_rad = self.pose_aug_rot_deg * np.pi / 180.0
        xi = torch.zeros(B, 6, device=pose_gt.device)
        xi[:, :3] = torch.randn(B, 3, device=pose_gt.device) * self.pose_aug_trans_m
        xi[:, 3:] = torch.randn(B, 3, device=pose_gt.device) * noise_rot_rad
        delta = se3_exp(xi)  # (B, 4, 4)
        return torch.bmm(delta, pose_gt)

    def _init_ref_feature_cache(self, ref_feature_dir: str):
        """Pre-load rendered reference features for render-extract training.

        When set, depth-warp uses these pre-computed rendered features
        instead of query features as the warp source, bridging the
        domain gap between training and render-extract inference.
        """
        from data.dataset_v4 import PoseDatasetV4
        dc = self.config['data']
        cache_ds = PoseDatasetV4(
            feature_base_dir=ref_feature_dir,
            traj_path=dc['train_traj_path'],
            noise_rot_deg=0.0,
            noise_trans_m=0.0,
            is_train=False,
        )
        print(f"[RefFeatureCache] Loading {len(cache_ds)} rendered ref features from {ref_feature_dir}...")
        self._ref_feature_cache = {}
        for i in range(len(cache_ds)):
            item = cache_ds[i]
            frame_idx = item['frame_idx']
            self._ref_feature_cache[frame_idx] = {
                k: v for k, v in item['query_feats'].items()
            }
        print(f"[RefFeatureCache] Cached {len(self._ref_feature_cache)} frames")

    @torch.no_grad()
    def _get_ref_features_batch(self, frame_indices, device):
        """Look up cached rendered reference features for a batch."""
        ref = {scale: [] for scale in self._ref_feature_cache[frame_indices[0]].keys()}
        for idx in frame_indices:
            for scale in ref:
                ref[scale].append(self._ref_feature_cache[idx][scale])
        return {scale: torch.stack(ref[scale]).to(device) for scale in ref}

    def _init_cross_frame_cache(self):
        """Build a cache of all training features + poses for cross-frame retrieval."""
        dc = self.config['data']
        from data.dataset_v4 import PoseDatasetV4, load_poses_c2w
        # Load all poses in c2w for position-based nearest neighbor
        poses_c2w = load_poses_c2w(dc['train_traj_path'])
        all_positions = torch.from_numpy(poses_c2w[:, :3, 3].astype(np.float32))
        all_poses_w2c = torch.from_numpy(np.stack([
            np.linalg.inv(p.astype(np.float64)).astype(np.float32)
            for p in poses_c2w
        ]))  # (N_all, 4, 4)

        # Load all features
        cache_ds = PoseDatasetV4(
            feature_base_dir=dc['train_feature_dir'],
            traj_path=dc['train_traj_path'],
            noise_rot_deg=0.0,
            noise_trans_m=0.0,
            is_train=False,
        )
        print(f"[CrossFrameCache] Loading {len(cache_ds)} frames...")
        self._cf_features = []  # list of {scale: (C,H,W)}
        for i in range(len(cache_ds)):
            item = cache_ds[i]
            self._cf_features.append({
                k: v for k, v in item['query_feats'].items()
            })
        self._cf_positions = all_positions  # (N, 3) world positions
        self._cf_poses_w2c = all_poses_w2c  # (N, 4, 4)
        print(f"[CrossFrameCache] Cached {len(self._cf_features)} frames")

    @torch.no_grad()
    def _get_cross_frame_refs(self, frame_indices, device):
        """Get features and poses from nearest neighbor (excluding self).

        Args:
            frame_indices: list of frame indices in current batch
        Returns:
            ref_feats: {scale: (B, C, H, W)} on device
            ref_poses: (B, 4, 4) w2c poses of reference frames
        """
        B = len(frame_indices)
        query_pos = self._cf_positions[frame_indices]  # (B, 3)

        # Compute distances to all frames
        dists = torch.cdist(query_pos, self._cf_positions)  # (B, N)
        # Exclude self (set self-distance to inf)
        for b in range(B):
            dists[b, frame_indices[b]] = float('inf')
        # Find nearest
        _, nn_idx = dists.min(dim=1)  # (B,)

        ref_feats_list = {scale: [] for scale in self._cf_features[0].keys()}
        ref_poses = []
        for b in range(B):
            idx = nn_idx[b].item()
            for scale in ref_feats_list:
                ref_feats_list[scale].append(self._cf_features[idx][scale])
            ref_poses.append(self._cf_poses_w2c[idx])

        ref_feats = {
            scale: torch.stack(ref_feats_list[scale]).to(device)
            for scale in ref_feats_list
        }
        ref_poses_tensor = torch.stack(ref_poses).to(device)
        return ref_feats, ref_poses_tensor

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
            # 1. 渲染 (or depth-warp) — depth-warp uses fp32 to avoid fp16 overflow
            if self.use_cross_frame_warp:
                frame_indices = batch.get('frame_idx', list(range(pose_gt.shape[0])))
                with torch.cuda.amp.autocast(enabled=False):
                    depth_rendered = self.renderer.render_depth_batch(pose_cur.float())
                    ref_feats, ref_poses = self._get_cross_frame_refs(
                        frame_indices, self.device)
                    render_feats = self._backward_warp(
                        {k: v.float() for k, v in ref_feats.items()},
                        depth_rendered, ref_poses.float(), pose_cur.float(),
                        self._scale_intrinsics)
                depth = depth_rendered
            elif self.use_depth_warp:
                with torch.cuda.amp.autocast(enabled=False):
                    depth_rendered = self.renderer.render_depth_batch(pose_cur.float())
                    # Pose-warp augmentation: perturb source pose to simulate cross-frame warp
                    source_pose = pose_gt.float()
                    if self.pose_aug_enabled and self.model.training:
                        source_pose = self._perturb_source_pose(source_pose)
                    # Use pre-loaded rendered refs if available, else use query feats
                    if self._ref_feature_cache is not None:
                        frame_indices = batch.get('frame_idx', list(range(pose_gt.shape[0])))
                        warp_src = self._get_ref_features_batch(frame_indices, self.device)
                        warp_src = {k: v.float() for k, v in warp_src.items()}
                    else:
                        warp_src = {k: v.float() for k, v in query_feats.items()}
                    render_feats = self._backward_warp(
                        warp_src,
                        depth_rendered, source_pose, pose_cur.float(),
                        self._scale_intrinsics)
                depth = depth_rendered
            else:
                render_feats, depth_rendered = self._render_batch(pose_cur)
                depth = depth_rendered
            if depth is None and depth_gt is not None:
                depth = depth_gt.to(self.device)

            # 1b. Reference feature augmentation (training only)
            if self.ref_aug_enabled and self.model.training:
                render_feats = self._augment_ref_feats(render_feats)

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

                # Flow consistency regularization (cross-scale agreement)
                flow_cons_metrics = {}
                if self.flow_cons_weight > 0:
                    fc_loss, flow_cons_metrics = flow_consistency_loss(
                        pred, self.model.COARSE_HW, self.model.MID_HW, self.model.FINE_HW)
                    iter_loss = iter_loss + self.flow_cons_weight * fc_loss

                # Localizability prior loss
                loc_metrics = {}
                if self.loc_loss_weight > 0 and 'loc_score' in pred:
                    l_loss, loc_metrics = localizability_loss(
                        pred['loc_score'], pred['flow_fine'],
                        gt_flows['fine'], gt_masks.get('fine'))
                    iter_loss = iter_loss + self.loc_loss_weight * l_loss

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
                all_metrics.update(flow_cons_metrics)
                all_metrics.update(loc_metrics)
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
                # Depth-warp must run in fp32 (coordinate math overflows in fp16)
                if self.use_cross_frame_warp:
                    frame_indices = batch.get('frame_idx', list(range(pose_gt.shape[0])))
                    with torch.cuda.amp.autocast(enabled=False):
                        depth = self.renderer.render_depth_batch(pose_cur.float())
                        ref_feats, ref_poses = self._get_cross_frame_refs(
                            frame_indices, self.device)
                        render_feats = self._backward_warp(
                            {k: v.float() for k, v in ref_feats.items()},
                            depth, ref_poses.float(), pose_cur.float(),
                            self._scale_intrinsics)
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        pred = self.model(query_feats, render_feats, depth)
                elif self.use_depth_warp:
                    with torch.cuda.amp.autocast(enabled=False):
                        depth = self.renderer.render_depth_batch(pose_cur.float())
                        # Use rendered refs if available
                        if self._ref_feature_cache is not None:
                            frame_indices = batch.get('frame_idx', list(range(pose_gt.shape[0])))
                            warp_src = self._get_ref_features_batch(frame_indices, self.device)
                            warp_src = {k: v.float() for k, v in warp_src.items()}
                        else:
                            warp_src = {k: v.float() for k, v in query_feats.items()}
                        render_feats = self._backward_warp(
                            warp_src,
                            depth, pose_gt.float(), pose_cur.float(),
                            self._scale_intrinsics)
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        pred = self.model(query_feats, render_feats, depth)
                else:
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
            # Joint metrics: percentage where BOTH rotation AND translation thresholds are met
            joint_01_53 = float(np.mean((rot < 0.1) & (trans < 5.3)) * 100)
            joint_1_50 = float(np.mean((rot < 1.0) & (trans < 50.0)) * 100)
            joint_5_100 = float(np.mean((rot < 5.0) & (trans < 100.0)) * 100)
            val_metrics = {
                'val_rot_mean': float(np.nanmean(rot)),
                'val_rot_median': float(np.nanmedian(rot)),
                'val_trans_mean': float(np.nanmean(trans)),
                'val_trans_median': float(np.nanmedian(trans)),
                'val_pct_1deg': float(np.mean(rot < 1.0) * 100),
                'val_pct_5deg': float(np.mean(rot < 5.0) * 100),
                'val_joint_01deg_53mm': joint_01_53,
                'val_joint_1deg_50mm': joint_1_50,
                'val_joint_5deg_100mm': joint_5_100,
            }
            if all_flow_epe:
                val_metrics['val_flow_epe'] = float(np.mean(all_flow_epe))
            iters_str = f"  ({N} iters)" if N > 1 else ""
            print(f"[Val E{epoch}]  rot={val_metrics['val_rot_mean']:.2f}° "
                  f"(med {val_metrics['val_rot_median']:.2f}°)  "
                  f"trans={val_metrics['val_trans_mean']:.1f}mm  "
                  f"<1°={val_metrics['val_pct_1deg']:.1f}%  "
                  f"joint@0.1°/5.3mm={joint_01_53:.1f}%  "
                  f"joint@1°/50mm={joint_1_50:.1f}%"
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

            # Checkpoint — track best by rot_mean + joint metric
            is_best = False
            if val_metrics and val_metrics.get('val_rot_mean', float('inf')) < self.best_val_rot:
                self.best_val_rot = val_metrics['val_rot_mean']
                is_best = True
                joint_str = f"  joint@0.1°/5.3mm={val_metrics.get('val_joint_01deg_53mm', 0):.1f}%"
                print(f"  ★ New best: rot={self.best_val_rot:.2f}°{joint_str}")

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
        print(f"Final metrics: {val_metrics}")
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
