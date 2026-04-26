"""
位置编码模块
包含2D和3D位置编码
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding2D(nn.Module):
    """
    2D Sine/Cosine位置编码
    
    用于为图像特征添加位置信息，告诉网络特征点在画面中的 (u, v) 坐标
    参考：DETR的位置编码实现
    
    输入:
        - coords_2d: [B, N, 2] 归一化到[0,1]的2D坐标 (u, v)
    输出:
        - pos_emb: [B, N, embed_dim] 位置编码
    """
    
    def __init__(self, embed_dim=256, temperature=10000):
        """
        Args:
            embed_dim: 编码维度（必须是偶数）
            temperature: 温度系数
        """
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim必须是偶数"
        
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # 频率系数
        dim_t = torch.arange(embed_dim // 4, dtype=torch.float32)
        dim_t = self.temperature ** (2 * dim_t / (embed_dim // 4))
        self.register_buffer('dim_t', dim_t)
        
    def forward(self, coords_2d):
        """
        生成2D位置编码
        
        Args:
            coords_2d: [B, N, 2] 2D坐标，范围[0, 1]
                      coords_2d[..., 0] = u (水平坐标)
                      coords_2d[..., 1] = v (垂直坐标)
                      
        Returns:
            pos_emb: [B, N, embed_dim] 位置编码
        """
        B, N, _ = coords_2d.shape
        device = coords_2d.device
        
        # 归一化坐标到[0, 1]范围（如果还没归一化）
        # coords_2d应该已经是归一化的
        
        # 提取u和v坐标
        u = coords_2d[..., 0]  # [B, N]
        v = coords_2d[..., 1]  # [B, N]
        
        # 生成频率
        dim_t = self.dim_t  # [embed_dim//4]
        
        # 计算位置编码
        # u方向: embed_dim//2 维度
        pos_u = u.unsqueeze(-1) / dim_t  # [B, N, embed_dim//4]
        pos_u = torch.cat([pos_u.sin(), pos_u.cos()], dim=-1)  # [B, N, embed_dim//2]
        
        # v方向: embed_dim//2 维度
        pos_v = v.unsqueeze(-1) / dim_t  # [B, N, embed_dim//4]
        pos_v = torch.cat([pos_v.sin(), pos_v.cos()], dim=-1)  # [B, N, embed_dim//2]
        
        # 拼接
        pos_emb = torch.cat([pos_u, pos_v], dim=-1)  # [B, N, embed_dim]
        
        return pos_emb


class PositionalEncoding3D(nn.Module):
    """
    3D位置编码
    
    用于为3D点云特征添加空间位置信息
    使用Sine/Cosine编码3D坐标 (x, y, z)
    
    输入:
        - coords_3d: [B, N, 3] 世界坐标系下的3D坐标 (x, y, z)
    输出:
        - pos_emb: [B, N, embed_dim] 位置编码
    """
    
    def __init__(self, embed_dim=256, temperature=10000, normalize=True,
                 scale_factor=1.0):
        """
        Args:
            embed_dim: 编码维度（必须能被3整除才能均匀分配给x,y,z）
            temperature: 温度系数
            normalize: 是否归一化3D坐标
            scale_factor: 归一化缩放因子
        """
        super().__init__()
        assert embed_dim % 3 == 0, "embed_dim应该能被3整除"
        
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.normalize = normalize
        self.scale_factor = scale_factor
        
        # 每个维度的编码维度
        self.dim_per_axis = embed_dim // 3
        
        # 频率系数（每个轴用dim_per_axis//2个sin和cos）
        dim_t = torch.arange(self.dim_per_axis // 2, dtype=torch.float32)
        dim_t = self.temperature ** (2 * dim_t / (self.dim_per_axis // 2))
        self.register_buffer('dim_t', dim_t)
        
    def forward(self, coords_3d):
        """
        生成3D位置编码
        
        Args:
            coords_3d: [B, N, 3] 3D坐标
                      coords_3d[..., 0] = x
                      coords_3d[..., 1] = y
                      coords_3d[..., 2] = z
                      
        Returns:
            pos_emb: [B, N, embed_dim] 位置编码
        """
        B, N, _ = coords_3d.shape
        device = coords_3d.device
        
        # 可选的归一化
        if self.normalize:
            # 归一化到[-1, 1]范围
            coords_norm = coords_3d * self.scale_factor
            # 简单的tanh归一化
            coords_norm = torch.tanh(coords_norm)
        else:
            coords_norm = coords_3d
        
        # 提取x, y, z坐标
        x = coords_norm[..., 0]  # [B, N]
        y = coords_norm[..., 1]  # [B, N]
        z = coords_norm[..., 2]  # [B, N]
        
        # 生成频率
        dim_t = self.dim_t  # [dim_per_axis//2]
        
        # 计算位置编码
        # x方向
        pos_x = x.unsqueeze(-1) / dim_t  # [B, N, dim_per_axis//2]
        pos_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)  # [B, N, dim_per_axis]
        
        # y方向
        pos_y = y.unsqueeze(-1) / dim_t  # [B, N, dim_per_axis//2]
        pos_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1)  # [B, N, dim_per_axis]
        
        # z方向
        pos_z = z.unsqueeze(-1) / dim_t  # [B, N, dim_per_axis//2]
        pos_z = torch.cat([pos_z.sin(), pos_z.cos()], dim=-1)  # [B, N, dim_per_axis]
        
        # 拼接
        pos_emb = torch.cat([pos_x, pos_y, pos_z], dim=-1)  # [B, N, embed_dim]
        
        return pos_emb


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码（作为替代方案）
    使用MLP将坐标映射到编码空间
    """
    
    def __init__(self, input_dim, embed_dim=256, hidden_dim=128):
        """
        Args:
            input_dim: 输入维度（2D为2，3D为3）
            embed_dim: 输出编码维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, coords):
        """
        Args:
            coords: [B, N, input_dim] 坐标
            
        Returns:
            pos_emb: [B, N, embed_dim] 位置编码
        """
        return self.mlp(coords)


if __name__ == "__main__":
    # 测试2D位置编码
    print("测试2D位置编码...")
    pos_enc_2d = PositionalEncoding2D(embed_dim=256)
    
    # 创建测试坐标
    coords_2d = torch.rand(4, 100, 2)  # [B=4, N=100, 2]
    pos_emb_2d = pos_enc_2d(coords_2d)
    print(f"  输入形状: {coords_2d.shape}")
    print(f"  输出形状: {pos_emb_2d.shape}")
    print(f"  ✓ 2D位置编码测试通过")
    
    # 测试3D位置编码
    print("\n测试3D位置编码...")
    pos_enc_3d = PositionalEncoding3D(embed_dim=258)  # 可被3整除
    
    coords_3d = torch.randn(4, 100, 3)  # [B=4, N=100, 3]
    pos_emb_3d = pos_enc_3d(coords_3d)
    print(f"  输入形状: {coords_3d.shape}")
    print(f"  输出形状: {pos_emb_3d.shape}")
    print(f"  ✓ 3D位置编码测试通过")
    
    # 测试可学习位置编码
    print("\n测试可学习位置编码...")
    learnable_enc = LearnablePositionalEncoding(input_dim=2, embed_dim=256)
    pos_emb_learnable = learnable_enc(coords_2d)
    print(f"  输入形状: {coords_2d.shape}")
    print(f"  输出形状: {pos_emb_learnable.shape}")
    print(f"  ✓ 可学习位置编码测试通过")
