"""
位姿回归模块 - 借鉴 MaRepo 架构
参考: reference/marepo/transformer/transformer.py

MaRepo 的核心设计:
1. 场景坐标(SC)使用 NeRF 位置编码
2. 像素坐标使用焦距归一化的正弦位置编码
3. 纯自注意力 Transformer（12层 + skip connection）
4. 残差卷积块进行特征降维
5. 使用场景均值进行坐标归一化

关键改进:
- 使用 NeRF-style 位置编码来增强 3D 坐标表达
- 焦距归一化的像素位置编码（相机感知）
- 更深的自注意力网络 + 周期性 skip connection
- 6D 或 9D 旋转表示
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def get_nerf_embedder(num_freqs=5, input_dims=3, include_input=True):
    """
    NeRF 风格的位置编码
    
    参考: NeRF 论文和 marepo/transformer/position_encoding_nerf.py
    
    Args:
        num_freqs: 频率数量 (L in the paper)
        input_dims: 输入维度
        include_input: 是否包含原始输入
    
    Returns:
        embed_fn: 编码函数
        out_dim: 输出维度
    """
    freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
    
    def embed_fn(x):
        """
        x: (..., input_dims)
        out: (..., out_dim)
        """
        out = []
        if include_input:
            out.append(x)
        for freq in freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)
    
    out_dim = input_dims * (1 + 2 * num_freqs) if include_input else input_dims * 2 * num_freqs
    return embed_fn, out_dim


class LinearAttention(nn.Module):
    """
    线性注意力（来自 LoFTR / MaRepo）
    
    复杂度 O(N) 而非 O(N^2)
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    def elu_feature_map(self, x):
        return F.elu(x) + 1
    
    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """
        Args:
            queries: (B, N, H, D)
            keys: (B, N, H, D)
            values: (B, N, H, D)
        Returns:
            (B, N, H, D)
        """
        Q = self.elu_feature_map(queries)
        K = self.elu_feature_map(keys)
        
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None].float()
            values = values * kv_mask[:, :, None, None].float()
        
        # K^T @ V
        KV = torch.einsum('bnhd,bnhv->bhdv', K, values)  # (B, H, D, D)
        # Q @ (K^T @ V)
        Z = 1.0 / (torch.einsum('bnhd,bhd->bnh', Q, K.sum(dim=1)) + self.eps)
        V = torch.einsum('bnhd,bhdv,bnh->bnhv', Q, KV, Z)
        
        return V


class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码层（借鉴 MaRepo）
    
    - 多头自注意力（线性或标准）
    - 前馈网络
    - Layer Norm
    """
    def __init__(self, d_model, nhead, use_linear_attention=True, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 投影层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 注意力
        if use_linear_attention:
            self.attention = LinearAttention()
        else:
            self.attention = None  # 使用标准注意力
        
        self.merge = nn.Linear(d_model, d_model, bias=False)
        
        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )
        
        # Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, C)
            mask: (B, N) True=valid, False=invalid
        """
        B = x.size(0)
        
        # 多头注意力
        query = self.q_proj(x).view(B, -1, self.nhead, self.head_dim)  # (B, N, H, D)
        key = self.k_proj(x).view(B, -1, self.nhead, self.head_dim)
        value = self.v_proj(x).view(B, -1, self.nhead, self.head_dim)
        
        if self.attention is not None:
            # 线性注意力
            attn = self.attention(query, key, value, q_mask=mask, kv_mask=mask)
        else:
            # 标准注意力
            scores = torch.einsum('bnhd,bmhd->bhnm', query, key) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn = torch.einsum('bhnm,bmhd->bnhd', attn_weights, value)
        
        message = self.merge(attn.reshape(B, -1, self.d_model))
        message = self.norm1(self.dropout(message))
        
        # FFN
        message = self.mlp(torch.cat([x, message], dim=-1))
        message = self.norm2(self.dropout(message))
        
        return x + message


class PoseRegressorMaRepo(nn.Module):
    """
    借鉴 MaRepo 的位姿回归器
    
    核心设计:
    1. 3D坐标使用 NeRF 位置编码 → 增强高频细节
    2. 2D坐标使用焦距归一化 + 正弦编码 → 相机感知
    3. 深度自注意力（12层 + 每4层 skip connection）
    4. 残差卷积块降维
    5. 分离的旋转/平移头
    
    输入:
        - img_keypoints: (B, N, 2) 2D关键点
        - pcd_keypoints: (B, N, 3) 3D关键点
        - query_features: (B, N, C) 可选的查询特征
        - intrinsics: (B, 3, 3) 相机内参
    
    输出:
        - pose: (B, 4, 4) 位姿矩阵
        - pose_6d: (B, 9) [tx, ty, tz, r1...r6]
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 12,
        feature_dim: int = 256,
        use_linear_attention: bool = True,
        use_nerf_encoding: bool = True,
        nerf_freqs: int = 5,
        coord_range: float = 50.0,  # 坐标范围 (用于归一化)
        dropout: float = 0.1,
        rotation_repr: str = '6D',  # '6D' or '9D'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.coord_range = coord_range
        self.rotation_repr = rotation_repr
        
        # === 位置编码 ===
        self.use_nerf_encoding = use_nerf_encoding
        if use_nerf_encoding:
            self.nerf_embed_fn, nerf_out_dim = get_nerf_embedder(nerf_freqs, input_dims=3)
            self.coord_3d_proj = nn.Linear(nerf_out_dim, d_model)
        else:
            self.coord_3d_proj = nn.Linear(3, d_model)
        
        # 2D坐标编码（使用正弦编码）
        # 编码后的维度: 2 * (1 + 2*num_freqs)
        self.embed_2d_fn, embed_2d_dim = get_nerf_embedder(nerf_freqs, input_dims=2)
        self.coord_2d_proj = nn.Linear(embed_2d_dim, d_model // 2)
        
        # 查询特征投影（如果有）
        self.query_proj = nn.Linear(feature_dim, d_model // 2) if feature_dim > 0 else None
        
        # === Transformer 层 ===
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, use_linear_attention, dropout
        )
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        
        # === 回归头 ===
        # 残差卷积块（借鉴 MaRepo）
        self.res_conv = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, d_model),
        )
        
        # 额外的 MLP
        self.pose_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
        )
        
        # 旋转头
        rot_dim = 9 if rotation_repr == '9D' else 6
        self.rotation_head = nn.Linear(d_model, rot_dim)
        
        # 平移头
        self.translation_head = nn.Linear(d_model, 3)
        
        # 场景均值（可学习或固定）
        self.register_buffer('scene_mean', torch.zeros(3))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def set_scene_mean(self, mean: torch.Tensor):
        """设置场景坐标均值（用于归一化）"""
        self.scene_mean = mean.view(3)
    
    def normalize_3d_coords(self, coords_3d):
        """
        归一化3D坐标到 [-π, π] 范围
        
        借鉴 MaRepo 的 hard_clip 方法
        """
        # 减去场景均值
        coords = coords_3d - self.scene_mean.view(1, 1, 3)
        # 裁剪并缩放到 [-π, π]
        coords = torch.clamp(coords, -self.coord_range, self.coord_range)
        coords = (coords / self.coord_range) * math.pi
        return coords
    
    def normalize_2d_coords(self, coords_2d, intrinsics=None):
        """
        归一化2D坐标（焦距归一化）
        
        借鉴 MaRepo 的 focal_norm 方法
        """
        if intrinsics is not None:
            B = coords_2d.shape[0]
            fx = intrinsics[:, 0, 0].view(B, 1, 1)
            fy = intrinsics[:, 1, 1].view(B, 1, 1)
            cx = intrinsics[:, 0, 2].view(B, 1, 1)
            cy = intrinsics[:, 1, 2].view(B, 1, 1)
            
            # (u - cx) / fx, (v - cy) / fy
            u = (coords_2d[..., 0:1] - cx) / (fx + 1e-8)
            v = (coords_2d[..., 1:2] - cy) / (fy + 1e-8)
            coords = torch.cat([u, v], dim=-1)  # (B, N, 2)
        else:
            # 简单归一化到 [-1, 1]
            coords = coords_2d / 320.0 - 1.0
        
        return coords
    
    def forward(
        self,
        img_keypoints: torch.Tensor,
        pcd_keypoints: torch.Tensor,
        query_features: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        """
        前向传播
        
        Args:
            img_keypoints: (B, N, 2) 2D关键点坐标
            pcd_keypoints: (B, N, 3) 3D关键点坐标
            query_features: (B, N, C) 查询特征（可选）
            intrinsics: (B, 3, 3) 相机内参（可选，用于焦距归一化）
            mask: (B, N) 有效掩码
        
        Returns:
            pose_matrix: (B, 4, 4) 位姿矩阵
            rotation_6d: (B, 6) 6D旋转
            translation: (B, 3) 平移
        """
        B, N, _ = pcd_keypoints.shape
        device = pcd_keypoints.device
        
        # === 1. 坐标编码 ===
        # 3D坐标: 归一化 → NeRF编码 → 投影
        coords_3d = self.normalize_3d_coords(pcd_keypoints)
        if self.use_nerf_encoding:
            coords_3d = self.nerf_embed_fn(coords_3d)
        feat_3d = self.coord_3d_proj(coords_3d)  # (B, N, d_model)
        
        # 2D坐标: 焦距归一化 → 正弦编码 → 投影
        coords_2d = self.normalize_2d_coords(img_keypoints, intrinsics)
        coords_2d = self.embed_2d_fn(coords_2d)
        feat_2d = self.coord_2d_proj(coords_2d)  # (B, N, d_model//2)
        
        # === 2. 特征融合 ===
        if self.query_proj is not None and query_features is not None:
            feat_query = self.query_proj(query_features)  # (B, N, d_model//2)
            feat = feat_3d + torch.cat([feat_2d, feat_query], dim=-1)
        else:
            # 如果没有查询特征，只用坐标特征
            feat = feat_3d + F.pad(feat_2d, (0, self.d_model // 2))
        
        # === 3. Transformer 自注意力 ===
        feat_skip = feat.clone()
        for i, layer in enumerate(self.layers):
            feat = layer(feat, mask)
            
            # 每4层做一次 skip connection（借鉴 MaRepo）
            if (i + 1) % 4 == 0:
                feat = feat + feat_skip
                feat_skip = feat.clone()
        
        # === 4. 全局池化 ===
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)  # (B, N, 1)
            feat_global = (feat * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-8)
        else:
            feat_global = feat.mean(dim=1)  # (B, d_model)
        
        # === 5. 残差块 + MLP ===
        feat_global = feat_global + self.res_conv(feat_global)
        feat_global = self.pose_mlp(feat_global)
        
        # === 6. 回归旋转和平移 ===
        rotation = self.rotation_head(feat_global)  # (B, 6) or (B, 9)
        translation = self.translation_head(feat_global)  # (B, 3)
        
        # 转换旋转表示为矩阵
        if self.rotation_repr == '9D':
            rotation_matrix = self.svd_orthogonalize(rotation)
        else:
            rotation_matrix = rotation_6d_to_matrix(rotation)
        
        # 构建 4x4 位姿矩阵
        pose_matrix = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        pose_matrix[:, :3, :3] = rotation_matrix
        pose_matrix[:, :3, 3] = translation
        
        return pose_matrix, rotation, translation
    
    def svd_orthogonalize(self, m):
        """将9D表示转换为SO(3)（使用SVD正交化）"""
        m = m.reshape(-1, 3, 3)
        m_transpose = torch.transpose(F.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    将6D旋转表示转换为旋转矩阵
    
    参考: Zhou et al. CVPR 2019
    """
    a1 = d6[:, :3]
    a2 = d6[:, 3:]
    
    b1 = F.normalize(a1, dim=1)
    dot = (a2 * b1).sum(dim=1, keepdim=True)
    u2 = a2 - dot * b1
    b2 = F.normalize(u2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    
    return torch.stack([b1, b2, b3], dim=-1)


class PoseRegressorMaRepoIterative(nn.Module):
    """
    迭代式 MaRepo 位姿回归器
    
    借鉴 MaRepo 的 C2F (Coarse-to-Fine) 设计:
    每4层输出一个中间位姿预测，逐步精化
    
    输入:
        - img_keypoints: (B, N, 2) 2D关键点
        - pcd_keypoints: (B, N, 3) 3D关键点  
        - query_features: (B, N, C) 查询特征
        - initial_pose: (B, 4, 4) 初始位姿
    
    输出:
        - pose_list: List of (B, 4, 4) 多阶段位姿预测
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 12,
        feature_dim: int = 256,
        use_linear_attention: bool = True,
        nerf_freqs: int = 5,
        coord_range: float = 50.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.coord_range = coord_range
        self.num_stages = num_layers // 4  # 每4层一个阶段
        
        # 坐标编码
        self.nerf_embed_fn, nerf_out_dim = get_nerf_embedder(nerf_freqs, input_dims=3)
        self.coord_3d_proj = nn.Linear(nerf_out_dim, d_model)
        
        self.embed_2d_fn, embed_2d_dim = get_nerf_embedder(nerf_freqs, input_dims=2)
        self.coord_2d_proj = nn.Linear(embed_2d_dim, d_model // 2)
        
        self.query_proj = nn.Linear(feature_dim, d_model // 2) if feature_dim > 0 else None
        
        # 初始位姿编码（用于条件化）
        self.pose_encoder = nn.Sequential(
            nn.Linear(12, 64),  # 3x4 位姿 → 12维
            nn.ReLU(True),
            nn.Linear(64, d_model),
        )
        
        # Transformer 层
        encoder_layer = TransformerEncoderLayer(d_model, nhead, use_linear_attention, dropout)
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        
        # 每个阶段的回归头（共享权重）
        self.res_conv = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, d_model),
        )
        
        self.pose_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
        )
        
        self.rotation_head = nn.Linear(d_model, 6)
        self.translation_head = nn.Linear(d_model, 3)
        
        self.register_buffer('scene_mean', torch.zeros(3))
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        img_keypoints: torch.Tensor,
        pcd_keypoints: torch.Tensor,
        query_features: torch.Tensor = None,
        initial_pose: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        """
        前向传播
        
        Returns:
            pose_list: List[(B, 4, 4)] 每个阶段的位姿预测
            final_pose: (B, 4, 4) 最终位姿
        """
        B, N, _ = pcd_keypoints.shape
        device = pcd_keypoints.device
        
        # 坐标编码
        coords_3d = pcd_keypoints - self.scene_mean.view(1, 1, 3)
        coords_3d = torch.clamp(coords_3d / self.coord_range, -1, 1) * math.pi
        coords_3d = self.nerf_embed_fn(coords_3d)
        feat_3d = self.coord_3d_proj(coords_3d)
        
        coords_2d = img_keypoints / 320.0 - 1.0  # 简单归一化
        coords_2d = self.embed_2d_fn(coords_2d)
        feat_2d = self.coord_2d_proj(coords_2d)
        
        # 特征融合
        if self.query_proj is not None and query_features is not None:
            feat_query = self.query_proj(query_features)
            feat = feat_3d + torch.cat([feat_2d, feat_query], dim=-1)
        else:
            feat = feat_3d + F.pad(feat_2d, (0, self.d_model // 2))
        
        # 添加初始位姿编码
        if initial_pose is not None:
            pose_enc = self.pose_encoder(initial_pose[:, :3, :4].reshape(B, 12))  # (B, d_model)
            feat = feat + pose_enc.unsqueeze(1)  # 广播到所有token
        
        # Transformer + 多阶段输出
        pose_list = []
        feat_skip = feat.clone()
        current_pose = initial_pose if initial_pose is not None else torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
        
        for i, layer in enumerate(self.layers):
            feat = layer(feat, mask)
            
            # 每4层输出一个位姿预测
            if (i + 1) % 4 == 0:
                feat = feat + feat_skip
                feat_skip = feat.clone()
                
                # 回归当前阶段的位姿
                if mask is not None:
                    mask_float = mask.float().unsqueeze(-1)
                    feat_global = (feat * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-8)
                else:
                    feat_global = feat.mean(dim=1)
                
                feat_global = feat_global + self.res_conv(feat_global)
                feat_global = self.pose_mlp(feat_global)
                
                rotation = self.rotation_head(feat_global)
                translation = self.translation_head(feat_global)
                
                # 构建位姿矩阵
                rot_matrix = rotation_6d_to_matrix(rotation)
                stage_pose = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
                stage_pose[:, :3, :3] = rot_matrix
                stage_pose[:, :3, 3] = translation
                
                # 累积位姿（相对 → 绝对）
                current_pose = current_pose @ stage_pose
                pose_list.append(current_pose.clone())
        
        return pose_list, pose_list[-1] if pose_list else current_pose


if __name__ == '__main__':
    """测试代码"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = PoseRegressorMaRepo(
        d_model=256,
        nhead=8,
        num_layers=12,
        feature_dim=256,
    ).to(device)
    
    # 测试输入
    B, N = 4, 64
    img_kp = torch.randn(B, N, 2).to(device)
    pcd_kp = torch.randn(B, N, 3).to(device)
    query_feat = torch.randn(B, N, 256).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).expand(B, -1, -1).to(device)
    intrinsics[:, 0, 0] = 320
    intrinsics[:, 1, 1] = 320
    intrinsics[:, 0, 2] = 319.5
    intrinsics[:, 1, 2] = 239.5
    
    # 前向传播
    pose_matrix, rotation, translation = model(
        img_kp, pcd_kp, query_feat, intrinsics
    )
    
    print(f"Pose matrix: {pose_matrix.shape}")
    print(f"Rotation 6D: {rotation.shape}")
    print(f"Translation: {translation.shape}")
    
    # 检查旋转矩阵正交性
    R = pose_matrix[:, :3, :3]
    should_be_identity = torch.bmm(R, R.transpose(1, 2))
    print(f"R @ R^T ≈ I: {torch.allclose(should_be_identity, torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1), atol=1e-5)}")
    
    # 测试迭代模型
    print("\n=== 迭代模型测试 ===")
    model_iter = PoseRegressorMaRepoIterative(
        d_model=256,
        nhead=8,
        num_layers=12,
        feature_dim=256,
    ).to(device)
    
    init_pose = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
    pose_list, final_pose = model_iter(
        img_kp, pcd_kp, query_feat, init_pose, intrinsics
    )
    
    print(f"Number of stages: {len(pose_list)}")
    print(f"Final pose: {final_pose.shape}")
    
    # 参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"\nMaRepo Regressor params: {params / 1e6:.2f}M")
    
    params_iter = sum(p.numel() for p in model_iter.parameters())
    print(f"MaRepo Iterative params: {params_iter / 1e6:.2f}M")
