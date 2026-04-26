"""
Coarse-to-Fine 位姿损失函数

借鉴 MaRepo 的关键设计:
1. 中间层辅助损失 - 多阶段监督
2. 动态损失权重 - 前期使用软损失，后期切换到硬损失
3. 旋转预热策略 - 从 L1/Cosine 逐渐过渡到测地距离
4. Dyntanh - 动态 tanh 软裁剪

参考: reference/marepo/loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


class PoseLossC2F(nn.Module):
    """
    Coarse-to-Fine 位姿损失
    
    特点:
    1. 多阶段损失 - 对每个中间阶段计算损失
    2. 阶段权重衰减 - 浅层阶段权重较低，深层阶段权重较高
    3. 旋转预热 - 训练初期使用软损失（L1/Cosine），后期切换到测地距离
    4. 动态 tanh 软裁剪 - 限制大误差的影响
    """
    
    def __init__(
        self,
        num_stages: int = 4,
        stage_weights: List[float] = None,  # 每个阶段的权重
        rotation_loss_type: str = 'geodesic',
        translation_loss_type: str = 'l2',
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        # 预热配置
        warmup_epochs: int = 50,  # 预热 epoch 数
        warmup_rotation_loss: str = 'l1',  # 预热期间的旋转损失
        # Dyntanh 配置（借鉴 MaRepo）
        use_dyntanh: bool = True,
        soft_clamp: float = 100.0,  # 初始软裁剪值
        soft_clamp_min: float = 10.0,  # 最终软裁剪值
        total_iterations: int = 100000,
    ):
        super().__init__()
        
        self.num_stages = num_stages
        self.rotation_loss_type = rotation_loss_type
        self.translation_loss_type = translation_loss_type
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        
        # 阶段权重（默认: 浅层 0.5, 深层 1.0）
        if stage_weights is None:
            # 线性增加权重: [0.25, 0.5, 0.75, 1.0]
            stage_weights = [(i + 1) / num_stages for i in range(num_stages)]
        self.stage_weights = stage_weights
        
        # 预热配置
        self.warmup_epochs = warmup_epochs
        self.warmup_rotation_loss = warmup_rotation_loss
        self.current_epoch = 0
        
        # Dyntanh 配置
        self.use_dyntanh = use_dyntanh
        self.soft_clamp = soft_clamp
        self.soft_clamp_min = soft_clamp_min
        self.total_iterations = total_iterations
        self.current_iteration = 0
        
        print(f"📐 PoseLossC2F initialized:")
        print(f"   - Num stages: {num_stages}")
        print(f"   - Stage weights: {stage_weights}")
        print(f"   - Rotation loss: {rotation_loss_type} (warmup: {warmup_rotation_loss})")
        print(f"   - Translation loss: {translation_loss_type}")
        print(f"   - Warmup epochs: {warmup_epochs}")
        print(f"   - Dyntanh: {use_dyntanh} (clamp: {soft_clamp} → {soft_clamp_min})")
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch（用于预热调度）"""
        self.current_epoch = epoch
    
    def set_iteration(self, iteration: int):
        """设置当前迭代（用于 dyntanh 调度）"""
        self.current_iteration = iteration
    
    @property
    def is_warmup(self) -> bool:
        """是否在预热阶段"""
        return self.current_epoch < self.warmup_epochs
    
    @property
    def warmup_progress(self) -> float:
        """预热进度 [0, 1]"""
        if self.warmup_epochs == 0:
            return 1.0
        return min(self.current_epoch / self.warmup_epochs, 1.0)
    
    def get_current_clamp(self) -> float:
        """获取当前的 dyntanh 裁剪值"""
        if not self.use_dyntanh:
            return float('inf')
        
        progress = self.current_iteration / max(self.total_iterations, 1)
        # 使用圆形调度（MaRepo 的方法）
        schedule_weight = 1 - math.sqrt(1 - min(progress, 1.0) ** 2)
        clamp = (1 - schedule_weight) * self.soft_clamp + schedule_weight * self.soft_clamp_min
        
        return clamp
    
    def dyntanh(self, errors: torch.Tensor) -> torch.Tensor:
        """
        动态 tanh 软裁剪（借鉴 MaRepo）
        
        当误差较大时，使用 tanh 软裁剪来限制其影响
        """
        clamp = self.get_current_clamp()
        return clamp * torch.tanh(errors / clamp)
    
    def compute_rotation_loss(
        self,
        R_pred: torch.Tensor,
        R_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算旋转损失（支持预热调度）
        
        预热期间: 使用 L1 或 Cosine 损失（更平滑）
        预热后: 使用测地距离（更精确）
        """
        if self.is_warmup:
            # 预热期间使用软损失
            if self.warmup_rotation_loss == 'l1':
                # L1: 直接比较旋转矩阵
                loss = torch.abs(R_pred - R_gt).sum(dim=(-2, -1))
            elif self.warmup_rotation_loss == 'cosine':
                # Cosine: 展平后计算余弦相似度
                R_pred_flat = R_pred.reshape(*R_pred.shape[:-2], 9)
                R_gt_flat = R_gt.reshape(*R_gt.shape[:-2], 9)
                cos_sim = F.cosine_similarity(R_pred_flat, R_gt_flat, dim=-1)
                loss = 1.0 - cos_sim
            elif self.warmup_rotation_loss == 'quaternion':
                # 四元数 L1
                q_pred = rotation_matrix_to_quaternion(R_pred)
                q_gt = rotation_matrix_to_quaternion(R_gt)
                dist1 = torch.abs(q_pred - q_gt).sum(dim=-1)
                dist2 = torch.abs(q_pred + q_gt).sum(dim=-1)
                loss = torch.minimum(dist1, dist2)
            else:
                # 默认使用 L1
                loss = torch.abs(R_pred - R_gt).sum(dim=(-2, -1))
            
            # 线性混合到测地距离
            if self.warmup_progress > 0.5:
                # 后半段预热开始混入测地距离
                geo_loss = self._geodesic_loss(R_pred, R_gt)
                mix_ratio = (self.warmup_progress - 0.5) * 2  # [0, 1]
                loss = (1 - mix_ratio) * loss + mix_ratio * geo_loss
        else:
            # 预热后使用目标损失
            if self.rotation_loss_type == 'geodesic':
                loss = self._geodesic_loss(R_pred, R_gt)
            elif self.rotation_loss_type == 'l1':
                loss = torch.abs(R_pred - R_gt).sum(dim=(-2, -1))
            elif self.rotation_loss_type == 'l2':
                loss = torch.sqrt(((R_pred - R_gt) ** 2).sum(dim=(-2, -1)))
            else:
                loss = self._geodesic_loss(R_pred, R_gt)
        
        return loss
    
    def _geodesic_loss(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """测地距离损失（角度，单位：度）"""
        R_rel = torch.matmul(R_pred.transpose(-2, -1), R_gt)
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        trace = torch.clamp(trace, -1.0, 3.0)
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        return theta * 180.0 / math.pi
    
    def compute_translation_loss(
        self,
        t_pred: torch.Tensor,
        t_gt: torch.Tensor,
    ) -> torch.Tensor:
        """计算平移损失"""
        if self.translation_loss_type == 'l1':
            loss = torch.abs(t_pred - t_gt).sum(dim=-1)
        elif self.translation_loss_type == 'l2':
            loss = torch.norm(t_pred - t_gt, dim=-1)
        elif self.translation_loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(t_pred, t_gt, reduction='none').sum(dim=-1)
        else:
            loss = torch.norm(t_pred - t_gt, dim=-1)
        
        return loss
    
    def forward(
        self,
        pose_stages: List[torch.Tensor],  # List[(B, 4, 4)]
        pose_gt: torch.Tensor,  # (B, 4, 4)
        return_components: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        计算多阶段位姿损失
        
        Args:
            pose_stages: 每个阶段的预测位姿
            pose_gt: 真值位姿
            return_components: 是否返回分量
        
        Returns:
            losses: {
                'loss': 总损失,
                'rotation_loss': 旋转损失,
                'translation_loss': 平移损失,
                'stage_losses': List[stage_loss],
            }
        """
        R_gt = pose_gt[:, :3, :3]
        t_gt = pose_gt[:, :3, 3]
        
        total_loss = 0.0
        total_rot_loss = 0.0
        total_trans_loss = 0.0
        stage_losses = []
        
        num_stages = min(len(pose_stages), self.num_stages)
        
        for i, pose_pred in enumerate(pose_stages):
            R_pred = pose_pred[:, :3, :3]
            t_pred = pose_pred[:, :3, 3]
            
            # 计算损失
            rot_loss = self.compute_rotation_loss(R_pred, R_gt)
            trans_loss = self.compute_translation_loss(t_pred, t_gt)
            
            # 应用 dyntanh（如果启用）
            if self.use_dyntanh:
                rot_loss = self.dyntanh(rot_loss)
                trans_loss = self.dyntanh(trans_loss)
            
            # 取均值
            rot_loss = rot_loss.mean()
            trans_loss = trans_loss.mean()
            
            # 加权
            stage_loss = self.rotation_weight * rot_loss + self.translation_weight * trans_loss
            
            # 阶段权重
            stage_weight = self.stage_weights[i] if i < len(self.stage_weights) else 1.0
            weighted_stage_loss = stage_weight * stage_loss
            
            total_loss += weighted_stage_loss
            total_rot_loss += stage_weight * rot_loss
            total_trans_loss += stage_weight * trans_loss
            stage_losses.append(stage_loss.item())
        
        # 归一化
        weight_sum = sum(self.stage_weights[:num_stages])
        total_loss = total_loss / weight_sum
        total_rot_loss = total_rot_loss / weight_sum
        total_trans_loss = total_trans_loss / weight_sum
        
        losses = {
            'loss': total_loss,
            'rotation_loss': total_rot_loss,
            'translation_loss': total_trans_loss,
        }
        
        if return_components:
            losses['stage_losses'] = stage_losses
        
        return losses


class PoseLossMapFree(nn.Module):
    """
    MapFree 风格的位姿损失（借鉴 MaRepo）
    
    特点:
    1. 旋转使用角度误差（弧度）
    2. 平移使用 L1
    3. 可选 tanh 软裁剪
    """
    
    def __init__(
        self,
        soft_clamp: bool = False,
        rotation_scale: float = 45.0,  # 旋转软裁剪尺度
        translation_scale: float = 100.0,  # 平移软裁剪尺度
    ):
        super().__init__()
        self.soft_clamp = soft_clamp
        self.rotation_scale = rotation_scale
        self.translation_scale = translation_scale
        
        if soft_clamp:
            self.tanh = nn.Tanh()
    
    def forward(
        self,
        pose_pred: torch.Tensor,  # (B, 4, 4) or (B, 3, 4)
        pose_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, trans_loss, rot_loss
        """
        t_pred = pose_pred[:, :3, 3]
        t_gt = pose_gt[:, :3, 3]
        R_pred = pose_pred[:, :3, :3]
        R_gt = pose_gt[:, :3, :3]
        
        # 平移 L1 损失
        trans_loss = F.l1_loss(t_pred, t_gt)
        
        # 旋转角度损失
        rot_loss = self._rotation_angle_loss(R_pred, R_gt)
        
        if self.soft_clamp:
            loss = (
                self.translation_scale * self.tanh(trans_loss / self.translation_scale) +
                self.rotation_scale * self.tanh(rot_loss / self.rotation_scale)
            )
        else:
            loss = trans_loss + rot_loss
        
        return loss, trans_loss, rot_loss
    
    def _rotation_angle_loss(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """旋转角度损失（弧度）"""
        residual = R_pred.transpose(1, 2) @ R_gt
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        cosine = torch.clamp(cosine, -0.99999, 0.99999)
        R_err = torch.acos(cosine)  # 弧度
        return F.l1_loss(R_err, torch.zeros_like(R_err))


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 → 四元数 (w, x, y, z)"""
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    
    # trace > 0
    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
    
    # 其他情况...（简化版）
    mask = ~(trace > 0)
    if mask.any():
        s = torch.sqrt(1.0 + R[mask, 0, 0] - R[mask, 1, 1] - R[mask, 2, 2] + 1e-8) * 2
        q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        q[mask, 1] = 0.25 * s
        q[mask, 2] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
        q[mask, 3] = (R[mask, 0, 2] + R[mask, 2, 0]) / s
    
    q = q.reshape(*batch_shape, 4)
    return F.normalize(q, dim=-1)


if __name__ == '__main__':
    """测试"""
    print("=== 测试 PoseLossC2F ===")
    
    loss_fn = PoseLossC2F(
        num_stages=4,
        warmup_epochs=10,
        use_dyntanh=True,
    )
    
    B = 4
    pose_stages = [torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone() for _ in range(4)]
    pose_gt = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
    
    # 添加一些噪声
    for i, pose in enumerate(pose_stages):
        pose[:, :3, 3] = torch.randn(B, 3) * (0.1 * (4 - i))
    
    # 测试预热阶段
    loss_fn.set_epoch(5)
    losses = loss_fn(pose_stages, pose_gt)
    print(f"Warmup (epoch 5): loss={losses['loss']:.4f}, rot={losses['rotation_loss']:.4f}, trans={losses['translation_loss']:.4f}")
    
    # 测试预热后
    loss_fn.set_epoch(100)
    losses = loss_fn(pose_stages, pose_gt)
    print(f"After warmup (epoch 100): loss={losses['loss']:.4f}, rot={losses['rotation_loss']:.4f}, trans={losses['translation_loss']:.4f}")
    
    print("\n=== 测试 PoseLossMapFree ===")
    loss_fn_mf = PoseLossMapFree(soft_clamp=True)
    
    pose_pred = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()
    pose_pred[:, :3, 3] = torch.randn(B, 3) * 0.5
    
    loss, trans_loss, rot_loss = loss_fn_mf(pose_pred, pose_gt)
    print(f"MapFree: loss={loss:.4f}, trans={trans_loss:.4f}, rot={rot_loss:.4f}")
