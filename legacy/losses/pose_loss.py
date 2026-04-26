"""
位姿损失函数模块
提供旋转和平移的多种损失计算方式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class PoseLoss(nn.Module):
    """
    位姿损失类
    支持多种旋转损失和平移损失的组合
    
    旋转损失类型:
        - geodesic: 测地距离（在SO(3)流形上的距离）
        - l2: L2距离（直接比较旋转矩阵）
        - cosine: 余弦相似度损失
        - quaternion: 四元数L2距离
    
    平移损失类型:
        - l1: L1距离（曼哈顿距离）
        - l2: L2距离（欧几里得距离）
        - smooth_l1: Smooth L1损失
    """
    
    def __init__(
        self,
        rotation_loss_type: str = 'geodesic',
        translation_loss_type: str = 'l2',
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        reduction: str = 'mean',
    ):
        """
        初始化位姿损失
        
        参数:
            rotation_loss_type: 旋转损失类型 ['geodesic', 'l2', 'cosine', 'quaternion']
            translation_loss_type: 平移损失类型 ['l1', 'l2', 'smooth_l1']
            rotation_weight: 旋转损失权重
            translation_weight: 平移损失权重
            reduction: 损失归约方式 ['mean', 'sum', 'none']
        """
        super().__init__()
        
        self.rotation_loss_type = rotation_loss_type
        self.translation_loss_type = translation_loss_type
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.reduction = reduction
        
        # 验证损失类型
        valid_rot_types = ['geodesic', 'l2', 'cosine', 'quaternion']
        valid_trans_types = ['l1', 'l2', 'smooth_l1']
        
        if rotation_loss_type not in valid_rot_types:
            raise ValueError(
                f"rotation_loss_type必须是{valid_rot_types}之一，"
                f"但得到: {rotation_loss_type}"
            )
        
        if translation_loss_type not in valid_trans_types:
            raise ValueError(
                f"translation_loss_type必须是{valid_trans_types}之一，"
                f"但得到: {translation_loss_type}"
            )
        
        print(f"[PoseLoss] 旋转损失: {rotation_loss_type} (权重={rotation_weight})")
        print(f"[PoseLoss] 平移损失: {translation_loss_type} (权重={translation_weight})")
    
    def rotation_matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """
        将旋转矩阵转换为四元数
        
        参数:
            R: [..., 3, 3] 旋转矩阵
            
        返回:
            q: [..., 4] 四元数 (w, x, y, z)
        """
        batch_shape = R.shape[:-2]
        R = R.reshape(-1, 3, 3)
        
        # 使用Shepperd方法
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        
        q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
        
        # Case 1: trace > 0
        mask = trace > 0
        s = torch.sqrt(trace[mask] + 1.0) * 2
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
        
        # Case 2: R[0,0] is the largest diagonal
        mask = (~(trace > 0)) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask, 0, 0] - R[mask, 1, 1] - R[mask, 2, 2]) * 2
        q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        q[mask, 1] = 0.25 * s
        q[mask, 2] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
        q[mask, 3] = (R[mask, 0, 2] + R[mask, 2, 0]) / s
        
        # Case 3: R[1,1] is the largest diagonal
        mask = (~(trace > 0)) & (~((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]))) & (R[:, 1, 1] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask, 1, 1] - R[mask, 0, 0] - R[mask, 2, 2]) * 2
        q[mask, 0] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
        q[mask, 1] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
        q[mask, 2] = 0.25 * s
        q[mask, 3] = (R[mask, 1, 2] + R[mask, 2, 1]) / s
        
        # Case 4: R[2,2] is the largest diagonal
        mask = (~(trace > 0)) & (~((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]))) & (~(R[:, 1, 1] > R[:, 2, 2]))
        s = torch.sqrt(1.0 + R[mask, 2, 2] - R[mask, 0, 0] - R[mask, 1, 1]) * 2
        q[mask, 0] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
        q[mask, 1] = (R[mask, 0, 2] + R[mask, 2, 0]) / s
        q[mask, 2] = (R[mask, 1, 2] + R[mask, 2, 1]) / s
        q[mask, 3] = 0.25 * s
        
        q = q.reshape(*batch_shape, 4)
        return q / torch.norm(q, dim=-1, keepdim=True)
    
    def compute_rotation_loss_geodesic(
        self, 
        R_pred: torch.Tensor, 
        R_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        计算测地距离损失（SO(3)流形上的距离）
        
        参数:
            R_pred: [..., 3, 3] 预测的旋转矩阵
            R_gt: [..., 3, 3] 真值旋转矩阵
            
        返回:
            loss: [...] 旋转损失（角度，单位：度）
        """
        # 计算相对旋转 R_rel = R_pred^T @ R_gt
        R_rel = torch.matmul(R_pred.transpose(-2, -1), R_gt)
        
        # 计算迹
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        
        # 测地距离: theta = arccos((trace(R_rel) - 1) / 2)
        # 为数值稳定性，限制trace的范围
        trace = torch.clamp(trace, -1.0, 3.0)
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        theta = torch.acos(cos_theta)  # 弧度
        theta_deg = theta * 180.0 / np.pi  # 转换为度
        
        return theta_deg
    
    def compute_rotation_loss_l2(
        self, 
        R_pred: torch.Tensor, 
        R_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        计算L2距离损失（直接比较旋转矩阵）
        
        参数:
            R_pred: [..., 3, 3] 预测的旋转矩阵
            R_gt: [..., 3, 3] 真值旋转矩阵
            
        返回:
            loss: [...] 旋转损失
        """
        diff = R_pred - R_gt
        loss = torch.sqrt((diff ** 2).sum(dim=(-2, -1)))
        return loss
    
    def compute_rotation_loss_cosine(
        self, 
        R_pred: torch.Tensor, 
        R_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        计算余弦相似度损失
        
        参数:
            R_pred: [..., 3, 3] 预测的旋转矩阵
            R_gt: [..., 3, 3] 真值旋转矩阵
            
        返回:
            loss: [...] 旋转损失
        """
        # 展平旋转矩阵
        R_pred_flat = R_pred.reshape(*R_pred.shape[:-2], 9)
        R_gt_flat = R_gt.reshape(*R_gt.shape[:-2], 9)
        
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(R_pred_flat, R_gt_flat, dim=-1)
        
        # 转换为损失（1 - cos_sim）
        loss = 1.0 - cos_sim
        
        return loss
    
    def compute_rotation_loss_quaternion(
        self, 
        R_pred: torch.Tensor, 
        R_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        计算四元数L2距离损失
        
        参数:
            R_pred: [..., 3, 3] 预测的旋转矩阵
            R_gt: [..., 3, 3] 真值旋转矩阵
            
        返回:
            loss: [...] 旋转损失
        """
        # 转换为四元数
        q_pred = self.rotation_matrix_to_quaternion(R_pred)
        q_gt = self.rotation_matrix_to_quaternion(R_gt)
        
        # 四元数有两个等价表示：q和-q
        # 选择距离更近的那个
        dist1 = torch.norm(q_pred - q_gt, dim=-1)
        dist2 = torch.norm(q_pred + q_gt, dim=-1)
        
        loss = torch.minimum(dist1, dist2)
        
        return loss
    
    def compute_translation_loss_l1(
        self, 
        t_pred: torch.Tensor, 
        t_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        计算L1距离损失（曼哈顿距离）
        
        参数:
            t_pred: [..., 3] 预测的平移向量
            t_gt: [..., 3] 真值平移向量
            
        返回:
            loss: [...] 平移损失
        """
        diff = torch.abs(t_pred - t_gt)
        loss = diff.sum(dim=-1)
        return loss
    
    def compute_translation_loss_l2(
        self, 
        t_pred: torch.Tensor, 
        t_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        计算L2距离损失（欧几里得距离）
        
        参数:
            t_pred: [..., 3] 预测的平移向量
            t_gt: [..., 3] 真值平移向量
            
        返回:
            loss: [...] 平移损失
        """
        diff = t_pred - t_gt
        loss = torch.norm(diff, dim=-1)
        return loss
    
    def compute_translation_loss_smooth_l1(
        self, 
        t_pred: torch.Tensor, 
        t_gt: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        计算Smooth L1损失
        
        参数:
            t_pred: [..., 3] 预测的平移向量
            t_gt: [..., 3] 真值平移向量
            beta: Smooth L1的平滑参数
            
        返回:
            loss: [...] 平移损失
        """
        diff = t_pred - t_gt
        abs_diff = torch.abs(diff)
        
        # Smooth L1: 0.5 * x^2 / beta (if |x| < beta), |x| - 0.5 * beta (otherwise)
        loss = torch.where(
            abs_diff < beta,
            0.5 * (diff ** 2) / beta,
            abs_diff - 0.5 * beta
        )
        
        loss = loss.sum(dim=-1)
        return loss
    
    def forward(
        self,
        pose_pred: torch.Tensor,
        pose_gt: torch.Tensor,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播，计算位姿损失
        
        参数:
            pose_pred: [..., 4, 4] 预测的位姿矩阵
            pose_gt: [..., 4, 4] 真值位姿矩阵
            return_components: 是否返回损失分量
            
        返回:
            losses: 包含损失的字典
                - loss: 总损失
                - rotation_loss: 旋转损失（如果return_components=True）
                - translation_loss: 平移损失（如果return_components=True）
        """
        # 提取旋转和平移
        R_pred = pose_pred[..., :3, :3]
        t_pred = pose_pred[..., :3, 3]
        
        R_gt = pose_gt[..., :3, :3]
        t_gt = pose_gt[..., :3, 3]
        
        # 计算旋转损失
        if self.rotation_loss_type == 'geodesic':
            rotation_loss = self.compute_rotation_loss_geodesic(R_pred, R_gt)
        elif self.rotation_loss_type == 'l2':
            rotation_loss = self.compute_rotation_loss_l2(R_pred, R_gt)
        elif self.rotation_loss_type == 'cosine':
            rotation_loss = self.compute_rotation_loss_cosine(R_pred, R_gt)
        elif self.rotation_loss_type == 'quaternion':
            rotation_loss = self.compute_rotation_loss_quaternion(R_pred, R_gt)
        else:
            raise ValueError(f"未知的旋转损失类型: {self.rotation_loss_type}")
        
        # 计算平移损失
        if self.translation_loss_type == 'l1':
            translation_loss = self.compute_translation_loss_l1(t_pred, t_gt)
        elif self.translation_loss_type == 'l2':
            translation_loss = self.compute_translation_loss_l2(t_pred, t_gt)
        elif self.translation_loss_type == 'smooth_l1':
            translation_loss = self.compute_translation_loss_smooth_l1(t_pred, t_gt)
        else:
            raise ValueError(f"未知的平移损失类型: {self.translation_loss_type}")
        
        # 应用权重
        weighted_rotation_loss = self.rotation_weight * rotation_loss
        weighted_translation_loss = self.translation_weight * translation_loss
        
        # 总损失
        total_loss = weighted_rotation_loss + weighted_translation_loss
        
        # 归约
        if self.reduction == 'mean':
            total_loss = total_loss.mean()
            rotation_loss = rotation_loss.mean()
            translation_loss = translation_loss.mean()
        elif self.reduction == 'sum':
            total_loss = total_loss.sum()
            rotation_loss = rotation_loss.sum()
            translation_loss = translation_loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f"未知的归约方式: {self.reduction}")
        
        # 构建输出字典
        losses = {'loss': total_loss}
        
        if return_components:
            losses['rotation_loss'] = rotation_loss
            losses['translation_loss'] = translation_loss
        
        return losses


if __name__ == '__main__':
    """测试代码"""
    print("测试PoseLoss模块\n")
    
    # 创建测试数据
    batch_size = 4
    
    # 生成随机旋转矩阵（通过正交化）
    def random_rotation_matrix(batch_size):
        # 生成随机矩阵
        A = torch.randn(batch_size, 3, 3)
        # QR分解得到正交矩阵
        Q, R = torch.linalg.qr(A)
        # 确保行列式为1（而不是-1）
        det = torch.det(Q)
        Q = Q * det.unsqueeze(-1).unsqueeze(-1)
        return Q
    
    R_gt = random_rotation_matrix(batch_size)
    t_gt = torch.randn(batch_size, 3)
    
    # 添加小扰动作为预测
    R_pred = R_gt + torch.randn_like(R_gt) * 0.1
    # 重新正交化
    U, _, Vt = torch.linalg.svd(R_pred)
    R_pred = torch.matmul(U, Vt)
    
    t_pred = t_gt + torch.randn_like(t_gt) * 0.1
    
    # 构建4x4位姿矩阵
    pose_gt = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    pose_gt[:, :3, :3] = R_gt
    pose_gt[:, :3, 3] = t_gt
    
    pose_pred = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    pose_pred[:, :3, :3] = R_pred
    pose_pred[:, :3, 3] = t_pred
    
    print(f"位姿形状: {pose_gt.shape}\n")
    
    # 测试不同损失类型
    loss_configs = [
        ('geodesic', 'l2', 1.0, 1.0),
        ('l2', 'l2', 1.0, 1.0),
        ('cosine', 'l1', 1.0, 1.0),
        ('quaternion', 'smooth_l1', 1.0, 1.0),
        ('geodesic', 'l2', 10.0, 1.0),  # 旋转权重更大
    ]
    
    for rot_type, trans_type, rot_w, trans_w in loss_configs:
        print(f"测试配置: rotation={rot_type}, translation={trans_type}, "
              f"rot_weight={rot_w}, trans_weight={trans_w}")
        
        loss_fn = PoseLoss(
            rotation_loss_type=rot_type,
            translation_loss_type=trans_type,
            rotation_weight=rot_w,
            translation_weight=trans_w,
            reduction='mean',
        )
        
        losses = loss_fn(pose_pred, pose_gt, return_components=True)
        
        print(f"  总损失: {losses['loss'].item():.4f}")
        print(f"  旋转损失: {losses['rotation_loss'].item():.4f}")
        print(f"  平移损失: {losses['translation_loss'].item():.4f}\n")
    
    # 测试完美匹配情况
    print("测试完美匹配（预测=真值）:")
    loss_fn = PoseLoss(
        rotation_loss_type='geodesic',
        translation_loss_type='l2',
        rotation_weight=1.0,
        translation_weight=1.0,
    )
    
    losses = loss_fn(pose_gt, pose_gt, return_components=True)
    print(f"  总损失: {losses['loss'].item():.6f}")
    print(f"  旋转损失: {losses['rotation_loss'].item():.6f}")
    print(f"  平移损失: {losses['translation_loss'].item():.6f}\n")
    
    print("测试完成!")

class PoseLossKendall(nn.Module):
    """
    基于Kendall不确定性的位姿损失（自动学习权重）
    
    参考: Kendall et al. "Geometric Loss Functions for Camera Pose Regression with Deep Learning" CVPR 2017
    
    核心思想:
        - 使用可学习的log_var参数自动平衡旋转和平移损失
        - 损失公式: L = exp(-log_var) * loss + log_var
        - log_var越大，该项权重越小，正则项增大防止无限增大
    
    优势:
        - 不需要手动调整rotation_weight和translation_weight
        - 网络自动学习最优权重平衡
        - 数值稳定（使用log_var而非直接的variance）
    """
    
    def __init__(
        self,
        rotation_loss_type: str = 'rotation_6d',
        translation_loss_type: str = 'l2',
        reduction: str = 'mean',
        init_log_var_rotation: float = 0.0,
        init_log_var_translation: float = 0.0,
    ):
        """
        初始化Kendall位姿损失
        
        参数:
            rotation_loss_type: 旋转损失类型 ['rotation_6d', 'geodesic', 'quaternion']
            translation_loss_type: 平移损失类型 ['l1', 'l2', 'smooth_l1']
            reduction: 损失归约方式 ['mean', 'sum']
            init_log_var_rotation: 旋转log_var初始值（默认0，即初始权重=1）
            init_log_var_translation: 平移log_var初始值
        """
        super().__init__()
        
        self.rotation_loss_type = rotation_loss_type
        self.translation_loss_type = translation_loss_type
        self.reduction = reduction
        
        # 可学习的不确定性参数（log scale）
        self.log_var_rotation = nn.Parameter(torch.tensor(init_log_var_rotation, dtype=torch.float32))
        self.log_var_translation = nn.Parameter(torch.tensor(init_log_var_translation, dtype=torch.float32))
        
        print(f"[PoseLossKendall] 旋转损失: {rotation_loss_type} (自动权重)")
        print(f"[PoseLossKendall] 平移损失: {translation_loss_type} (自动权重)")
        print(f"[PoseLossKendall] 初始log_var: rot={init_log_var_rotation:.3f}, trans={init_log_var_translation:.3f}")
    
    def compute_rotation_loss_6d(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """
        计算6D旋转表示的损失（使用geodesic distance返回度数）
        
        注意：为了与原始训练保持一致的损失量级，使用geodesic distance而非Frobenius范数
        geodesic返回度数(0-180)，而Frobenius范数仅返回0-3，量级相差太大
        
        参数:
            R_pred: (B, 3, 3) 预测旋转矩阵
            R_gt: (B, 3, 3) 真值旋转矩阵
            
        返回:
            loss: 标量，旋转损失（度数）
        """
        # 使用geodesic distance（与原始训练一致）
        R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta_rad = torch.acos(cos_theta)
        theta_deg = theta_rad * 180.0 / 3.14159265359  # 转换为度数
        
        if self.reduction == 'mean':
            return theta_deg.mean()
        elif self.reduction == 'sum':
            return theta_deg.sum()
        else:
            return theta_deg
    
    def compute_rotation_loss_geodesic(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """
        计算测地距离（SO(3)流形上的距离）
        
        参数:
            R_pred: (B, 3, 3) 预测旋转矩阵
            R_gt: (B, 3, 3) 真值旋转矩阵
            
        返回:
            loss: 标量，测地距离（弧度）
        """
        # 相对旋转: R_rel = R_pred @ R_gt^T
        R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))
        
        # 计算旋转角度: theta = arccos((trace(R_rel) - 1) / 2)
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 数值稳定
        
        theta = torch.acos(cos_theta)  # 弧度
        
        if self.reduction == 'mean':
            return theta.mean()
        elif self.reduction == 'sum':
            return theta.sum()
        else:
            return theta
    
    def compute_translation_loss(self, t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
        """
        计算平移损失
        
        参数:
            t_pred: (B, 3) 预测平移
            t_gt: (B, 3) 真值平移
            
        返回:
            loss: 标量，平移损失
        """
        if self.translation_loss_type == 'l1':
            loss = F.l1_loss(t_pred, t_gt, reduction='none').sum(dim=1)
        elif self.translation_loss_type == 'l2':
            loss = F.mse_loss(t_pred, t_gt, reduction='none').sum(dim=1).sqrt()
        elif self.translation_loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(t_pred, t_gt, reduction='none').sum(dim=1)
        else:
            raise ValueError(f"不支持的平移损失类型: {self.translation_loss_type}")
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def forward(
        self,
        pose_pred: torch.Tensor,
        pose_gt: torch.Tensor,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            pose_pred: (B, 4, 4) 预测位姿矩阵，或(B, 3, 3)和(B, 3)的元组
            pose_gt: (B, 4, 4) 真值位姿矩阵
            return_components: 是否返回各分量损失
            
        返回:
            losses: 字典，包含总损失和各分量损失
        """
        # 解析位姿
        if isinstance(pose_pred, tuple):
            R_pred, t_pred = pose_pred
        else:
            R_pred = pose_pred[:, :3, :3]
            t_pred = pose_pred[:, :3, 3]
        
        R_gt = pose_gt[:, :3, :3]
        t_gt = pose_gt[:, :3, 3]
        
        # 计算各分量损失
        if self.rotation_loss_type == 'rotation_6d':
            rotation_loss = self.compute_rotation_loss_6d(R_pred, R_gt)
        elif self.rotation_loss_type == 'geodesic':
            rotation_loss = self.compute_rotation_loss_geodesic(R_pred, R_gt)
        else:
            raise ValueError(f"不支持的旋转损失类型: {self.rotation_loss_type}")
        
        translation_loss = self.compute_translation_loss(t_pred, t_gt)
        
        # Kendall's Loss加权
        # L = exp(-log_var) * loss + log_var
        weighted_rotation_loss = torch.exp(-self.log_var_rotation) * rotation_loss + self.log_var_rotation
        weighted_translation_loss = torch.exp(-self.log_var_translation) * translation_loss + self.log_var_translation
        
        total_loss = weighted_rotation_loss + weighted_translation_loss
        
        # 返回结果
        losses = {
            'loss': total_loss,
            'rotation_loss': rotation_loss,
            'translation_loss': translation_loss,
            'weighted_rotation_loss': weighted_rotation_loss,
            'weighted_translation_loss': weighted_translation_loss,
            'log_var_rotation': self.log_var_rotation,
            'log_var_translation': self.log_var_translation,
            'effective_weight_rotation': torch.exp(-self.log_var_rotation),
            'effective_weight_translation': torch.exp(-self.log_var_translation),
        }
        
        if not return_components:
            return losses['loss']
        
        return losses


if __name__ == '__main__':
    print("测试位姿损失函数...")
    print("测试已移至 test_6d_rotation.py")
