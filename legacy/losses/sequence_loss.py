"""
Sequence Loss for Iterative Render-and-Compare Pose Estimation
===============================================================
参考:
  - RAFT (Teed & Deng, ECCV 2020): Exponentially increasing weights
  - DROID-SLAM (Teed & Deng, NeurIPS 2021): Confidence-weighted flow loss

关键设计:
  1. Sequence Loss: L = Σ_{k=1}^{K} γ^{K-k} · L_pose(P_k, P_gt)
     - 越靠后的迭代权重越大 (γ < 1 时)
     - 鼓励先粗后细的收敛行为
  
  2. Pose Loss: 旋转用 geodesic，平移用 L1
     - L_pose = L_rot(R_k, R_gt) + λ_t · L_trans(t_k, t_gt)
  
  3. Flow Loss (可选): 显式 2D-3D 对应监督
     - L_flow = mean(conf · ||f_pred - f_gt|| - λ · log(conf))
     - 提供 1610× 更稠密的梯度信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from modules.lie_algebra import compute_gt_flow


# ---------- Rotation metrics ----------

def rotation_geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    计算旋转的测地线距离 L = arccos((tr(R_pred^T R_gt) - 1) / 2)
    
    Args:
        R_pred: (B, 3, 3) 预测旋转矩阵 (从 w2c 的 [:3, :3] 提取)
        R_gt:   (B, 3, 3) GT 旋转矩阵
    
    Returns:
        (B,) 每个样本的测地线距离 (弧度)
    """
    R_diff = R_pred.transpose(-1, -2) @ R_gt  # (B, 3, 3)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]  # (B,)
    # clamp for numerical stability
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)  # (B,)
    return angle


def translation_loss(t_pred: torch.Tensor, t_gt: torch.Tensor, mode: str = 'l1') -> torch.Tensor:
    """
    计算平移损失
    
    Args:
        t_pred: (B, 3) 预测平移
        t_gt:   (B, 3) GT 平移
        mode: 'l1', 'l2', 'smooth_l1'
    
    Returns:
        (B,) 每个样本的平移损失
    """
    if mode == 'l1':
        return torch.abs(t_pred - t_gt).sum(dim=-1)
    elif mode == 'l2':
        return torch.norm(t_pred - t_gt, dim=-1)
    elif mode == 'smooth_l1':
        return F.smooth_l1_loss(t_pred, t_gt, reduction='none').sum(dim=-1)
    else:
        raise ValueError(f"Unknown translation loss mode: {mode}")


# ---------- Flow Loss ----------

def masked_flow_l1_loss(
    flow_pred: torch.Tensor,
    flow_gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    简单的 masked L1 flow 损失 (无置信度加权)
    
    L = mean_valid( ||f_pred - f_gt||_1 )
    
    不使用置信度加权，避免网络通过降低置信度逃避学习。
    
    Args:
        flow_pred: (B, 2, H, W) 预测的 flow
        flow_gt:   (B, 2, H, W) GT flow
        valid_mask: (B, 1, H, W) 有效像素掩码 (depth > 0)
    
    Returns:
        scalar loss
    """
    flow_error = torch.abs(flow_pred - flow_gt).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    
    if valid_mask is not None:
        valid_mask = valid_mask.float()
        num_valid = valid_mask.sum().clamp(min=1.0)
        loss = (flow_error * valid_mask).sum() / num_valid
    else:
        loss = flow_error.mean()
    
    return loss


def confidence_weighted_flow_loss(
    flow_pred: torch.Tensor,
    flow_gt: torch.Tensor,
    log_confidence: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    confidence_penalty: float = 0.1,
) -> torch.Tensor:
    """
    DROID-SLAM 风格的置信度加权 flow 损失 (保留但默认不使用)
    
    L = mean_valid( conf * ||f_pred - f_gt||_1 - λ * log(conf) )
    """
    flow_error = torch.abs(flow_pred - flow_gt).sum(dim=1, keepdim=True)
    confidence = torch.exp(log_confidence)
    weighted_error = confidence * flow_error - confidence_penalty * log_confidence
    
    if valid_mask is not None:
        valid_mask = valid_mask.float()
        num_valid = valid_mask.sum().clamp(min=1.0)
        loss = (weighted_error * valid_mask).sum() / num_valid
    else:
        loss = weighted_error.mean()
    
    return loss


# ---------- Sequence Loss ----------

class SequenceLoss(nn.Module):
    """
    Sequence Loss for iterative pose refinement
    
    L = Σ_{k=1}^{K} γ^{K-k} · [L_pose(P_k, P_gt) + λ_flow · L_flow(k)]
    
    Args:
        gamma: 权重衰减因子 (0 < γ ≤ 1), 默认 0.8
            - γ = 1.0: 所有步等权重
            - γ = 0.8: 最后一步权重 1.0, 倒数第二步 0.8, ...
        lambda_translation: 平移损失权重
        lambda_flow: flow 损失权重
        translation_mode: 平移损失模式 ('l1', 'l2', 'smooth_l1')
        confidence_penalty: flow 损失中防止置信度坍塌的惩罚系数
        use_flow_loss: 是否使用 flow 损失
    """
    
    def __init__(
        self,
        gamma: float = 0.8,
        lambda_translation: float = 1.0,
        lambda_flow: float = 0.1,
        translation_mode: str = 'l1',
        confidence_penalty: float = 0.1,
        use_flow_loss: bool = True,
    ):
        super().__init__()
        self.gamma = gamma
        self.lambda_translation = lambda_translation
        self.lambda_flow = lambda_flow
        self.translation_mode = translation_mode
        self.confidence_penalty = confidence_penalty
        self.use_flow_loss = use_flow_loss
    
    def forward(
        self,
        predictions: Dict[str, object],
        pose_gt: torch.Tensor,
        depth_gt: Optional[torch.Tensor] = None,
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算 Sequence Loss
        
        Args:
            predictions: ICPoseNetV3 forward 输出 dict
                - 'poses': [(B, 4, 4)] K+1 个位姿
                - 'flow_list': [(B, 2, H, W)] K 个 flow
                - 'log_conf_list': [(B, 1, H, W)] K 个置信度
            pose_gt: (B, 4, 4) GT 位姿 (w2c)
            depth_gt: (B, H, W) GT 深度图 (用于 flow GT)
            intrinsics: {'fx':, 'fy':, 'cx':, 'cy':}
        
        Returns:
            dict:
                'total_loss': scalar
                'pose_loss': scalar (sequence-weighted)
                'rotation_loss': scalar
                'translation_loss': scalar
                'flow_loss': scalar
                'per_step_rotation': list[scalar]
                'per_step_translation': list[scalar]
        """
        poses = predictions['poses']         # K+1 个位姿 (P₀, P₁, ..., P_K)
        flow_list = predictions.get('flow_list', [])
        log_conf_list = predictions.get('log_conf_list', [])
        
        K = len(poses) - 1  # 迭代次数
        R_gt = pose_gt[:, :3, :3]   # (B, 3, 3)
        t_gt = pose_gt[:, :3, 3]    # (B, 3)
        
        total_pose_loss = 0.0
        total_rot_loss = 0.0
        total_trans_loss = 0.0
        total_flow_loss = 0.0
        
        per_step_rot = []
        per_step_trans = []
        per_step_flow = []
        
        for k in range(K):
            # 权重: γ^{K-1-k} (最后一步权重为 1, 倒数第二步为 γ, ...)
            weight = self.gamma ** (K - 1 - k)
            
            # 位姿损失
            P_k = poses[k + 1]  # 第 k 步的输出位姿 (k=0 是初始位姿)
            R_k = P_k[:, :3, :3]
            t_k = P_k[:, :3, 3]
            
            rot_loss = rotation_geodesic_loss(R_k, R_gt).mean()
            trans_loss = translation_loss(
                t_k, t_gt, mode=self.translation_mode
            ).mean()
            
            step_pose_loss = rot_loss + self.lambda_translation * trans_loss
            total_pose_loss = total_pose_loss + weight * step_pose_loss
            total_rot_loss = total_rot_loss + weight * rot_loss
            total_trans_loss = total_trans_loss + weight * trans_loss
            
            per_step_rot.append(rot_loss.detach())
            per_step_trans.append(trans_loss.detach())
            
            # Flow 损失
            if (self.use_flow_loss and depth_gt is not None 
                    and intrinsics is not None and k < len(flow_list)):
                
                flow_pred = flow_list[k]
                log_conf = log_conf_list[k]
                
                # 【关键修复】GT flow 基于修正前位姿 poses[k] 而非修正后 P_k
                # FlowHead 看到的是 poses[k] 处渲染的残差，应该预测 GT→poses[k] 的 flow
                P_before = poses[k].detach()
                flow_gt_data = compute_gt_flow(
                    depth_gt, pose_gt, P_before, intrinsics
                )
                flow_gt = flow_gt_data['flow']          # (B, 2, H, W)
                valid_mask = flow_gt_data['valid_mask']  # (B, 1, H, W)
                
                # 确保分辨率匹配
                if flow_pred.shape[-2:] != flow_gt.shape[-2:]:
                    flow_gt = F.interpolate(
                        flow_gt, size=flow_pred.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                    valid_mask = F.interpolate(
                        valid_mask, size=flow_pred.shape[-2:],
                        mode='nearest'
                    )
                
                # 使用纯 L1 损失，不用 confidence 加权(避免 confidence 坍塌)
                step_flow_loss = masked_flow_l1_loss(
                    flow_pred, flow_gt,
                    valid_mask=valid_mask,
                )
                
                total_flow_loss = total_flow_loss + weight * step_flow_loss
                per_step_flow.append(step_flow_loss.detach())
        
        # 总损失
        total_loss = total_pose_loss + self.lambda_flow * total_flow_loss
        
        # 计算最终步的监控指标 (不带 sequence 加权)
        with torch.no_grad():
            final_pose = poses[-1]
            final_rot_err = rotation_geodesic_loss(
                final_pose[:, :3, :3], R_gt
            ).mean() * (180.0 / 3.14159265)  # 转为度
            final_trans_err = torch.norm(
                final_pose[:, :3, 3] - t_gt, dim=-1
            ).mean()
        
        return {
            'total_loss': total_loss,
            'pose_loss': total_pose_loss.detach() if torch.is_tensor(total_pose_loss) else torch.tensor(0.0),
            'rotation_loss': total_rot_loss.detach() if torch.is_tensor(total_rot_loss) else torch.tensor(0.0),
            'translation_loss': total_trans_loss.detach() if torch.is_tensor(total_trans_loss) else torch.tensor(0.0),
            'flow_loss': total_flow_loss.detach() if torch.is_tensor(total_flow_loss) else torch.tensor(0.0),
            'per_step_rotation': per_step_rot,
            'per_step_translation': per_step_trans,
            'per_step_flow': per_step_flow,
            # 监控指标
            'final_rotation_error_deg': final_rot_err,
            'final_translation_error_m': final_trans_err,
        }


class PoseOnlySequenceLoss(nn.Module):
    """
    简化版: 只有位姿损失, 不需要 flow / depth
    适合初期调试或没有深度图时使用
    """
    
    def __init__(
        self,
        gamma: float = 0.8,
        lambda_translation: float = 1.0,
        translation_mode: str = 'l1',
    ):
        super().__init__()
        self.gamma = gamma
        self.lambda_translation = lambda_translation
        self.translation_mode = translation_mode
    
    def forward(
        self,
        predictions: Dict[str, object],
        pose_gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: 包含 'poses' key (list of (B, 4, 4))
            pose_gt: (B, 4, 4) GT w2c
        """
        poses = predictions['poses']
        K = len(poses) - 1
        R_gt = pose_gt[:, :3, :3]
        t_gt = pose_gt[:, :3, 3]
        
        total_loss = 0.0
        for k in range(K):
            weight = self.gamma ** (K - 1 - k)
            P_k = poses[k + 1]
            
            rot_loss = rotation_geodesic_loss(P_k[:, :3, :3], R_gt).mean()
            trans_loss = translation_loss(
                P_k[:, :3, 3], t_gt, mode=self.translation_mode
            ).mean()
            
            total_loss = total_loss + weight * (
                rot_loss + self.lambda_translation * trans_loss
            )
        
        # 最终步误差
        with torch.no_grad():
            final = poses[-1]
            rot_err_deg = rotation_geodesic_loss(
                final[:, :3, :3], R_gt
            ).mean() * (180.0 / 3.14159265)
            trans_err_m = torch.norm(final[:, :3, 3] - t_gt, dim=-1).mean()
        
        return {
            'total_loss': total_loss,
            'final_rotation_error_deg': rot_err_deg,
            'final_translation_error_m': trans_err_m,
        }
