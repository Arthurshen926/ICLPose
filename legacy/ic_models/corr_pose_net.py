"""
CorrPoseNet: Correlation-based Iterative Pose Refinement Network

核心创新（vs 之前失败的 v3-v9b 网络）:
  - 使用 LOCAL CORRELATION VOLUME 代替逐像素 concat/subtract
  - Correlation 提供方向性位移信息（类似 RAFT 的核心思想）
  - 预测 dense flow + confidence，通过可微分几何求解器得到 6-DOF 位姿
  - ConvGRU 迭代精修，保留跨迭代记忆

Pipeline (每次迭代):
  1. 在当前位姿渲染3DGS特征图 → fmap_r (detached, 无梯度)
  2. 编码query和rendered特征: 768d → enc_dim (共享编码器)
  3. 构建局部 correlation volume: <fmap_r, fmap_q> within radius r
  4. ConvGRU 更新 hidden state
  5. Flow head: hidden → dense (Δu, Δv)
  6. Confidence head: hidden → per-pixel weight w ∈ [0,1]
  7. 可微分位姿求解: (flow, w, depth, Image Jacobian) → δξ ∈ se(3)
  8. 位姿更新: T_{k+1} = exp(δξ) · T_k

关键设计选择:
  - correlation(fmap_r, fmap_q): 对每个渲染像素搜索query邻域匹配
  - 几何位姿求解器（非直接回归）: 保证物理正确性，利用超定约束
  - 渲染 detach: 不通过可微渲染反传梯度，仅通过网络预测和几何求解器

数学原理:
  flow 预测 "渲染图中每个像素应移动到query图中的哪里"
  Image Jacobian J 将像素位移映射到相机运动:
    Ju · δξ ≈ Δu,  Jv · δξ ≈ Δv
  加权最小二乘求解:
    (Ju^T W Ju + Jv^T W Jv) δξ = Ju^T W flow_u + Jv^T W flow_v
  本质是 FDA 的 flow-based 版本: 用学习的 flow 代替特征空间梯度隐式 flow
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from modules.lie_algebra import se3_exp
from modules.pose_aware_upsampler import PoseAwareUpsampler


# ==============================================================================
# Building Blocks
# ==============================================================================

class ConvGRU(nn.Module):
    """
    Convolutional Gated Recurrent Unit.
    
    空间感知的循环单元，3×3 卷积在空间维度上聚合邻域信息，
    保持跨迭代的记忆，让网络能逐步精修 flow 预测。
    """
    
    def __init__(self, hidden_dim: int = 128, input_dim: int = 64):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, hidden_dim, H, W) hidden state
            x: (B, input_dim, H, W) input features
        Returns:
            h_new: (B, hidden_dim, H, W) updated hidden state
        """
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))       # update gate
        r = torch.sigmoid(self.convr(hx))       # reset gate
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))  # candidate
        return (1 - z) * h + z * q


def local_correlation(
    fmap1: torch.Tensor, 
    fmap2: torch.Tensor, 
    radius: int = 4,
) -> torch.Tensor:
    """
    计算局部 correlation volume.
    
    对 fmap1 中的每个像素 (i,j)，与 fmap2 中半径 r 内的所有像素做点积。
    这提供了 per-pixel concat/subtract 无法提供的方向性位移信息:
    correlation 的峰值方向直接指示了匹配位置的偏移方向。
    
    对于 radius=4, 搜索范围 ±4 像素。在 35×46 特征分辨率下:
    1 特征像素 ≈ 18 原图像素 ≈ 3.2° 旋转
    所以 ±4 像素 ≈ ±13° 的搜索范围，4 次迭代可处理 ~40° 偏差。
    
    自动选择实现:
      - 小分辨率 (<1GB): F.unfold 向量化, ~5x 加速
      - 大分辨率 (≥1GB): 预分配输出 + narrow 视图, ~2x 加速
    
    Args:
        fmap1: (B, C, H, W) 参考特征图 (L2-normalized)
        fmap2: (B, C, H, W) 搜索特征图 (L2-normalized)
        radius: 搜索半径 (像素)
    
    Returns:
        corr: (B, (2r+1)², H, W) correlation volume
              channel 排列: dy=-r..r, dx=-r..r (row-major)
    """
    B, C, H, W = fmap1.shape
    d = 2 * radius + 1
    
    # 估算 F.unfold 所需内存 (bytes)
    mem_estimate = B * C * d * d * H * W * 4  # float32
    
    # 性能测试 (RTX 3090): fast_loop 在两种分辨率均优于 unfold
    # 35×46@128ch: fast_loop 3.2ms vs unfold 5.1ms  
    # 140×184@64ch: fast_loop 21.3ms (unfold OOM)
    return _local_correlation_fast_loop(fmap1, fmap2, radius)


def _local_correlation_unfold(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """向量化 correlation — 使用 F.unfold, 一次计算所有 (2r+1)² 个相关值."""
    B, C, H, W = fmap1.shape
    d = 2 * radius + 1
    N = H * W
    
    fmap2_pad = F.pad(fmap2, [radius] * 4, mode='constant', value=0)
    # F.unfold: (B, C*(2r+1)², H*W)
    fmap2_unf = F.unfold(fmap2_pad, kernel_size=d, stride=1)
    fmap2_unf = fmap2_unf.view(B, C, d * d, N)         # (B, C, d², N)
    fmap1_flat = fmap1.reshape(B, C, N)                  # (B, C, N)
    
    # Dot product along channel dim: (B, d², N)
    corr = torch.einsum('bcn,bckn->bkn', fmap1_flat, fmap2_unf)
    return corr.view(B, d * d, H, W)


def _local_correlation_fast_loop(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """预分配输出 + narrow 视图 — 内存高效, 比原始 list+cat 快 ~2x."""
    B, C, H, W = fmap1.shape
    d = 2 * radius + 1
    
    fmap2_pad = F.pad(fmap2, [radius] * 4, mode='constant', value=0)
    
    # 预分配输出 (避免 list + torch.cat)
    corr = fmap1.new_empty(B, d * d, H, W)
    idx = 0
    for dy in range(-radius, radius + 1):
        # narrow 返回视图, 零拷贝
        strip = fmap2_pad.narrow(2, radius + dy, H)
        for dx in range(-radius, radius + 1):
            shifted = strip.narrow(3, radius + dx, W)
            corr[:, idx] = (fmap1 * shifted).sum(1)
            idx += 1
    
    return corr


def diff_pose_solve(
    flow: torch.Tensor,
    confidence: torch.Tensor,
    Ju: torch.Tensor,
    Jv: torch.Tensor,
    valid: torch.Tensor,
    damping: float = 1e-3,
) -> torch.Tensor:
    """
    可微分的加权最小二乘位姿求解器.
    
    给定预测的 dense flow (像素位移) 和 Image Jacobian (像素位移→相机运动的映射),
    通过加权正规方程求解 6-DOF 相机运动增量。
    
    数学:
      对每个像素 i, 我们希望: Ju[i]·δξ ≈ flow_u[i], Jv[i]·δξ ≈ flow_v[i]
      加权目标: min Σ w[i] · (||Ju[i]·δξ - flow_u[i]||² + ||Jv[i]·δξ - flow_v[i]||²)
      正规方程: (Σ J^T W J) δξ = Σ J^T W flow
    
    这等价于 FDA 的正规方程，但用学习的 flow 代替特征梯度隐式 flow。
    35×46 = 1610 像素的加权约束求解 6 个未知数 → 极度超定。
    
    Args:
        flow: (B, 2, H, W) 预测的像素位移 [Δu, Δv]
        confidence: (B, 1, H, W) 每像素权重 [0, 1]
        Ju: (B, N, 6) Image Jacobian ∂u/∂ξ
        Jv: (B, N, 6) Image Jacobian ∂v/∂ξ
        valid: (B, N) 有效像素 mask
        damping: LM 阻尼项
    
    Returns:
        delta_xi: (B, 6) se(3) 更新向量 [vx, vy, vz, ωx, ωy, ωz]
    """
    B = flow.shape[0]
    device = flow.device
    
    # Flatten spatial dims
    flow_u = flow[:, 0].reshape(B, -1)              # (B, N)
    flow_v = flow[:, 1].reshape(B, -1)              # (B, N)
    w = confidence[:, 0].reshape(B, -1)             # (B, N)
    w = w * valid.float()                            # zero out invalid pixels
    
    # Weighted Jacobians
    w_unsq = w.unsqueeze(-1)                         # (B, N, 1)
    wJu = Ju * w_unsq                                # (B, N, 6)
    wJv = Jv * w_unsq                                # (B, N, 6)
    
    # JtWJ: (B, 6, 6)
    JtWJ = torch.bmm(Ju.transpose(1, 2), wJu) + \
           torch.bmm(Jv.transpose(1, 2), wJv)
    
    # LM damping
    JtWJ = JtWJ + damping * torch.eye(6, device=device).unsqueeze(0)
    
    # JtWr: (B, 6)
    JtWr = torch.bmm(
        Ju.transpose(1, 2), (w * flow_u).unsqueeze(-1)
    ).squeeze(-1) + torch.bmm(
        Jv.transpose(1, 2), (w * flow_v).unsqueeze(-1)
    ).squeeze(-1)
    
    # Solve 6×6 system
    delta_xi = torch.linalg.solve(JtWJ, JtWr)  # (B, 6)
    
    return delta_xi


# ==============================================================================
# Main Network
# ==============================================================================

class CorrPoseNet(nn.Module):
    """
    Correlation-based Iterative Pose Refinement Network.
    
    通过局部 correlation matching 预测 dense optical flow，
    再经可微分几何求解器得到 6-DOF 位姿更新。
    
    相比之前失败的网络 (v3-v9b):
    - 用 correlation volume 代替 concat/subtract → 有方向性搜索能力
    - 用几何求解器代替直接回归 → 物理正确，利用超定约束
    - 用 ConvGRU 迭代精修 → 跨迭代记忆
    
    Multi-scale (optional):
    - 前 coarse_iters 次迭代在下采样特征上做 correlation → 有效搜索范围翻倍
    - 后续迭代在原分辨率上精修
    - coarse 和 fine 共享编码器和 GRU (通过 interpolate 统一分辨率)
    """
    
    def __init__(
        self,
        feat_dim: int = 768,
        enc_dim: int = 128,
        hidden_dim: int = 128,
        corr_radius: int = 4,
        corr_enc_dim: int = 64,
        num_iters: int = 3,
        damping: float = 1e-3,
        use_multiscale: bool = False,
        coarse_iters: int = 1,
        coarse_scale_factor: float = 0.5,
        use_upsampler: bool = False,
        upsample_dim: int = 64,
        upsample_scale: int = 4,
        upsample_after_iter: int = 2,
        use_motion_input: bool = False,
        use_flow_init: bool = False,
        learnable_damping: bool = False,
    ):
        """
        Args:
            feat_dim: 输入特征维度 (768 for DINO, 640 for SD)
            enc_dim: 编码后特征维度
            hidden_dim: GRU hidden state 维度
            corr_radius: correlation 搜索半径 (4 → 81 channels)
            corr_enc_dim: correlation 编码后维度
            num_iters: 默认迭代次数
            damping: 位姿求解器 LM 阻尼
            use_multiscale: 是否使用多尺度 coarse-to-fine
            coarse_iters: 前几次迭代用 coarse scale
            coarse_scale_factor: coarse 下采样倍率 (0.5 → 18×23)
            use_upsampler: 是否使用 PoseAwareUpsampler (提升平移精度)
            upsample_dim: Upsampler 输出特征维度 (推荐 64)
            upsample_scale: 空间上采样倍率 (2 → 70×92, 4 → 140×184)
            upsample_after_iter: 从第几次迭代开始用高分辨率 (coarse-to-fine)
                                 前 N 次在 35×46 粗定位 (大搜索范围), 后续在高分辨率精修
            use_motion_input: 是否注入深度+flow反馈到GRU (exp013新增)
                             将 inverse depth (1ch) + 上次预测的 flow (2ch) 
                             编码后叠加到 correlation 特征上, 提供 3D 几何先验
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.enc_dim = enc_dim
        self.hidden_dim = hidden_dim
        self.corr_radius = corr_radius
        self.num_iters = num_iters
        self.damping = damping
        self.use_multiscale = use_multiscale
        self.coarse_iters = coarse_iters
        self.coarse_scale_factor = coarse_scale_factor
        self.use_upsampler = use_upsampler
        self.upsample_dim = upsample_dim
        self.upsample_scale = upsample_scale
        self.upsample_after_iter = upsample_after_iter
        self.use_motion_input = use_motion_input
        self.use_flow_init = use_flow_init
        self.learnable_damping = learnable_damping
        self.render_chunk_size = 128  # gsplat channel chunk size (non-parameter, not saved in state_dict)
        
        corr_channels = (2 * corr_radius + 1) ** 2  # 81 for r=4
        
        # --- PoseAwareUpsampler (task-driven feature transform) ---
        # 替代 feat_encoder: 768d@35×46 → upsample_dim@(35*s)×(46*s)
        # 通过 pose loss 端到端训练, 提升平移灵敏度
        if use_upsampler:
            self.upsampler = PoseAwareUpsampler(
                in_dim=feat_dim,
                out_dim=upsample_dim,
                scale=upsample_scale,
            )
            # feat_encoder 仍然保留用于 multiscale 或 non-upsampler 模式
            # 但 upsampler 模式下 correlation 使用 upsample_dim 维特征
        
        # --- Feature encoder (shared for query and rendered) ---
        # 768d → 128d, 1×1 convolutions = per-pixel MLP
        # 使用 GroupNorm 替代 BatchNorm: batch_size=1时BN退化为InstanceNorm (train),
        # 但eval时使用running stats会导致train/val不一致 → 严重过拟合
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 1),
            nn.GroupNorm(16, 256),  # 16 groups of 16 channels
            nn.ReLU(inplace=True),
            nn.Conv2d(256, enc_dim, 1),
            nn.GroupNorm(8, enc_dim),  # 8 groups of 16 channels
            nn.ReLU(inplace=True),
        )
        
        # --- Context encoder (query only → GRU initial hidden state) ---
        # 提取query的全局上下文，初始化GRU的记忆
        self.context_encoder = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, hidden_dim, 1),
            # 无激活，forward 时用 tanh 初始化 GRU
        )
        
        # --- Correlation encoder ---
        # 将 81-channel correlation volume 压缩为紧凑表示
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(corr_channels, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, corr_enc_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # --- ConvGRU ---
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=corr_enc_dim)
        
        # --- Flow head: predict dense pixel displacement (Δu, Δv) ---
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),  # 防止过拟合
            nn.Conv2d(64, 2, 3, padding=1),
        )
        
        # --- Confidence head: per-pixel weight for geometric solver ---
        self.conf_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, 3, padding=1),
        )
        
        # --- Motion encoder: inject depth + flow feedback into GRU (exp013) ---
        # Encodes inverse depth (1ch) + previous flow prediction (2ch) → corr_enc_dim
        # Additive fusion with corr_feat: GRU input = corr_feat + motion_feat
        # This gives the matching network awareness of:
        #   - 3D geometry (depth-dependent flow magnitude)
        #   - prediction history (what was predicted last iteration)
        if use_motion_input:
            self.motion_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),  # 2ch flow + 1ch inv_depth
                nn.GroupNorm(4, 32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, corr_enc_dim, 1),  # → same dim as corr_feat
                nn.ReLU(inplace=True),
            )
        
        # --- Soft-argmax flow initialization (exp015) ---
        # Computes initial flow estimate from correlation peak via soft-argmax.
        # The flow_head then only needs to predict a RESIDUAL correction.
        # This provides a strong inductive bias: flow ≈ location of correlation peak.
        #
        # Gating mechanism for warm-start compatibility:
        #   flow = gate * soft_argmax(corr) + flow_head(hidden)
        #   gate starts at ~0 (sigmoid(-5)≈0.007) so initial behavior matches exp013,
        #   then opens during training to let soft-argmax dominate.
        if use_flow_init:
            # Learnable temperature for softmax (higher = sharper peak)
            self.flow_init_temperature = nn.Parameter(torch.tensor(10.0))
            # Gate: starts closed (~0) for warm-start safety
            self.flow_init_gate = nn.Parameter(torch.tensor(-5.0))
            # Precompute offset grid for soft-argmax (stored as buffer, not parameter)
            r = corr_radius
            dy, dx = torch.meshgrid(
                torch.arange(-r, r + 1, dtype=torch.float32),
                torch.arange(-r, r + 1, dtype=torch.float32),
                indexing='ij',
            )
            self.register_buffer('_corr_dx', dx.flatten())  # (81,)
            self.register_buffer('_corr_dy', dy.flatten())  # (81,)
        
        # --- Learnable per-iteration damping (exp015) ---
        # Instead of fixed damping, learn optimal damping factor per iteration.
        # Parameterized as log-space to ensure positive values.
        # Initialized at current damping value.
        if learnable_damping:
            self.log_damping = nn.Parameter(
                torch.full((num_iters,), math.log(damping))
            )
        
        # --- Initialization ---
        # Flow head: 初始预测 ≈ 零 (identity transform)
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)
        
        # Confidence head: 初始输出 ≈ 0 (sigmoid(0) = 0.5)
        nn.init.zeros_(self.conf_head[-1].weight)
        nn.init.zeros_(self.conf_head[-1].bias)
    
    def encode(self, feats: torch.Tensor, use_upsampler: bool = False) -> torch.Tensor:
        """
        编码原始特征到紧凑表示 + L2 归一化.
        
        Args:
            feats: (B, feat_dim, H, W) 原始特征 (e.g. 768-dim DINO)
            use_upsampler: 是否使用 PoseAwareUpsampler (已包含 L2 norm)
        Returns:
            encoded: (B, dim, H', W') 编码后特征, L2-normalized
                     普通模式: dim=enc_dim, H'=H, W'=W
                     upsampler模式: dim=upsample_dim, H'=H*scale, W'=W*scale
        """
        if use_upsampler and self.use_upsampler:
            return self.upsampler(feats)  # 已包含 L2 norm
        enc = self.feat_encoder(feats)
        return F.normalize(enc, dim=1)
    
    def _soft_argmax_flow(self, corr: torch.Tensor) -> torch.Tensor:
        """
        Compute initial flow estimate from correlation volume via soft-argmax.
        
        The correlation volume has shape (B, (2r+1)^2, H, W) where each of the
        81 channels represents the similarity at a spatial offset (dx, dy) within
        the search radius. Soft-argmax computes a weighted average of offsets,
        giving a continuous (differentiable) flow estimate.
        
        Args:
            corr: (B, C, H, W) local correlation volume, C = (2r+1)^2
            
        Returns:
            flow_init: (B, 2, H, W) initial flow estimate [du, dv]
        """
        T = self.flow_init_temperature.abs() + 1.0  # ensure T ≥ 1.0
        weights = F.softmax(corr * T, dim=1)  # (B, C, H, W)
        
        # Weighted sum of offsets → expected displacement
        dx = self._corr_dx.view(1, -1, 1, 1)  # (1, C, 1, 1)
        dy = self._corr_dy.view(1, -1, 1, 1)  # (1, C, 1, 1)
        flow_u = (weights * dx).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        flow_v = (weights * dy).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        return torch.cat([flow_u, flow_v], dim=1)  # (B, 2, H, W)
    
    def _get_damping(self, k: int) -> float:
        """Get damping factor for iteration k."""
        if self.learnable_damping:
            return torch.exp(self.log_damping[k])
        return self.damping
    
    def forward(
        self,
        query_feats: torch.Tensor,
        initial_pose: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: dict,
        renderer=None,
        scale_name: str = 'fine_dino',
        num_iters: Optional[int] = None,
    ) -> dict:
        """
        完整前向传播: 迭代渲染 + correlation + flow + 位姿更新.
        
        Multi-scale mode:
          前 coarse_iters 次迭代在下采样分辨率(e.g. 18×23)做 correlation，
          有效搜索范围翻倍(~±26°)。然后在原分辨率(35×46)精修。
          GRU hidden state 通过 interpolate 在两个尺度间转换。
        
        Args:
            query_feats: (B, D, H, W) 查询特征
            initial_pose: (B, 4, 4) 初始 w2c 位姿
            depth: (B, H, W) 深度图 (用于 Image Jacobian 和 GT flow)
            intrinsics: {'fx', 'fy', 'cx', 'cy'} 在特征分辨率下
            renderer: MultiScaleRenderer 实例
            scale_name: 渲染哪个尺度
            num_iters: 覆盖默认迭代次数
        
        Returns:
            dict:
                'poses': list of (B, 4, 4) — 每次迭代后的位姿 (len = num_iters + 1)
                'flows': list of (B, 2, H, W) — 预测的 flow
                'confidences': list of (B, 1, H, W) — 预测的置信度
                'delta_xis': list of (B, 6) — se(3) 更新量
        """
        if num_iters is None:
            num_iters = self.num_iters
        
        assert renderer is not None, "renderer is required for forward()"
        
        B, D, H, W = query_feats.shape
        device = query_feats.device
        
        # --- Always compute standard Jacobian + features ---
        from modules.featuremetric import compute_image_jacobian
        Ju, Jv, valid = compute_image_jacobian(depth, intrinsics)
        # Standard encoded query (35×46)
        fmap_q_std = self.encode(query_feats, use_upsampler=False)       # (B, enc_dim, H, W)
        hidden = torch.tanh(self.context_encoder(query_feats))           # (B, hidden_dim, H, W)
        
        # --- Upsampler: also precompute high-res Jacobian + features ---
        if self.use_upsampler:
            s = self.upsample_scale
            uH, uW = H * s, W * s
            up_depth = F.interpolate(
                depth.unsqueeze(1), size=(uH, uW), mode='nearest'
            ).squeeze(1)
            up_intrinsics = {k: v * s for k, v in intrinsics.items()}
            Ju_up, Jv_up, valid_up = compute_image_jacobian(up_depth, up_intrinsics)
            fmap_q_up = self.encode(query_feats, use_upsampler=True)     # (B, upsample_dim, uH, uW)
        
        # Multi-scale: precompute coarse query + coarse Jacobians
        if self.use_multiscale:
            sf = self.coarse_scale_factor
            cH, cW = int(H * sf), int(W * sf)
            # 下采样 query 特征 → coarse 分辨率
            fmap_q_coarse = F.interpolate(fmap_q, size=(cH, cW), mode='bilinear', align_corners=False)
            fmap_q_coarse = F.normalize(fmap_q_coarse, dim=1)
            # Coarse depth + Jacobian
            depth_coarse = F.interpolate(depth.unsqueeze(1), size=(cH, cW), mode='nearest').squeeze(1)
            # Coarse intrinsics
            coarse_intrinsics = {
                'fx': intrinsics['fx'] * sf,
                'fy': intrinsics['fy'] * sf, 
                'cx': intrinsics['cx'] * sf,
                'cy': intrinsics['cy'] * sf,
            }
            Ju_c, Jv_c, valid_c = compute_image_jacobian(depth_coarse, coarse_intrinsics)
        
        pose = initial_pose
        results = {
            'poses': [pose],
            'flows': [],
            'confidences': [],
            'delta_xis': [],
        }
        
        # --- Motion input: initialize previous flow (for flow feedback) ---
        prev_flow = torch.zeros(B, 2, H, W, device=device) if self.use_motion_input else None
        
        # Precompute normalized inverse depth maps for motion encoder
        if self.use_motion_input:
            inv_depth_std = self._normalize_inv_depth(depth)  # (B, 1, H, W)
            if self.use_upsampler:
                inv_depth_up = F.interpolate(inv_depth_std, size=(uH, uW), mode='nearest')
        
        for k in range(num_iters):
            # Determine if this iteration is coarse or fine
            is_coarse = self.use_multiscale and k < self.coarse_iters
            
            # Determine if this iteration uses upsampled resolution
            use_up_this_iter = (self.use_upsampler and k >= self.upsample_after_iter)
            
            # --- 1. Render features at current pose (no gradient) ---
            with torch.no_grad():
                rendered_feats = self._render_batch(
                    renderer, pose.detach(), scale_name, device,
                    chunk_size=self.render_chunk_size,
                )  # (B, D, H, W) always at fine resolution
            
            # --- 2. Encode rendered features ---
            fmap_r = self.encode(rendered_feats, use_upsampler=use_up_this_iter)
            
            if is_coarse:
                # === Coarse iteration: downsample → correlate → upsample ===
                fmap_r_coarse = F.interpolate(fmap_r, size=(cH, cW), mode='bilinear', align_corners=False)
                fmap_r_coarse = F.normalize(fmap_r_coarse, dim=1)
                
                # Downsample hidden state for coarse GRU
                hidden_coarse = F.interpolate(hidden, size=(cH, cW), mode='bilinear', align_corners=False)
                
                # Correlation at coarse resolution
                corr = local_correlation(fmap_r_coarse, fmap_q_coarse, self.corr_radius)
                corr_feat = self.corr_encoder(corr)
                
                # GRU at coarse resolution
                hidden_coarse = self.gru(hidden_coarse, corr_feat)
                
                # Predict flow and confidence at coarse resolution
                flow_coarse = self.flow_head(hidden_coarse)         # (B, 2, cH, cW)
                conf_coarse = torch.sigmoid(self.conf_head(hidden_coarse))
                
                # Upsample hidden back to fine resolution
                hidden = F.interpolate(hidden_coarse, size=(H, W), mode='bilinear', align_corners=False)
                
                # Geometric solve at coarse resolution
                damping_k = self._get_damping(k)
                delta_xi = diff_pose_solve(
                    flow_coarse, conf_coarse, Ju_c, Jv_c, valid_c, damping_k
                )
                
                # Upsample flow/conf for logging (scale flow by 1/sf)
                flow_fine = F.interpolate(flow_coarse, size=(H, W), mode='bilinear', align_corners=False) / self.coarse_scale_factor
                conf_fine = F.interpolate(conf_coarse, size=(H, W), mode='bilinear', align_corners=False)
            else:
                # === Fine or Upsampled iteration ===
                if use_up_this_iter:
                    # --- Upsampled resolution: high translation sensitivity ---
                    # Transition hidden state from 35×46 → uH×uW on first upsampled iter
                    if hidden.shape[-2:] != (uH, uW):
                        hidden = F.interpolate(hidden, size=(uH, uW), mode='bilinear', align_corners=False)
                    
                    corr = local_correlation(fmap_r, fmap_q_up, self.corr_radius)
                    corr_feat = self.corr_encoder(corr)
                    
                    # Motion input: add depth + flow feedback at upsampled resolution
                    if self.use_motion_input:
                        pf = prev_flow if prev_flow.shape[-2:] == (uH, uW) else \
                            F.interpolate(prev_flow, size=(uH, uW), mode='bilinear', align_corners=False)
                        motion_in = torch.cat([pf, inv_depth_up], dim=1)
                        corr_feat = corr_feat + self.motion_encoder(motion_in)
                    
                    hidden = self.gru(hidden, corr_feat)
                    
                    flow_up = self.flow_head(hidden)              # (B, 2, uH, uW)
                    # Soft-argmax flow initialization: flow = gate * corr_peak + residual
                    if self.use_flow_init:
                        gate = torch.sigmoid(self.flow_init_gate)
                        flow_up = gate * self._soft_argmax_flow(corr) + flow_up
                    conf_up = torch.sigmoid(self.conf_head(hidden))
                    
                    damping_k = self._get_damping(k)
                    delta_xi = diff_pose_solve(
                        flow_up, conf_up, Ju_up, Jv_up, valid_up, damping_k
                    )
                    
                    # Store at upsampled resolution (flow loss will use this)
                    flow_fine = flow_up
                    conf_fine = conf_up
                else:
                    # --- Standard resolution ---
                    corr = local_correlation(fmap_r, fmap_q_std, self.corr_radius)
                    corr_feat = self.corr_encoder(corr)
                    
                    # Motion input: add depth + flow feedback
                    if self.use_motion_input:
                        pf = prev_flow if prev_flow.shape[-2:] == (H, W) else \
                            F.interpolate(prev_flow, size=(H, W), mode='bilinear', align_corners=False)
                        motion_in = torch.cat([pf, inv_depth_std], dim=1)  # (B, 3, H, W)
                        corr_feat = corr_feat + self.motion_encoder(motion_in)
                    
                    hidden = self.gru(hidden, corr_feat)
                    
                    flow_fine = self.flow_head(hidden)
                    # Soft-argmax flow initialization: flow = gate * corr_peak + residual
                    if self.use_flow_init:
                        gate = torch.sigmoid(self.flow_init_gate)
                        flow_fine = gate * self._soft_argmax_flow(corr) + flow_fine
                    conf_fine = torch.sigmoid(self.conf_head(hidden))
                    
                    damping_k = self._get_damping(k)
                    delta_xi = diff_pose_solve(
                        flow_fine, conf_fine, Ju, Jv, valid, damping_k
                    )
            
            # --- Update pose: T_{k+1} = exp(δξ) · T_k ---
            delta_T = se3_exp(delta_xi)  # (B, 4, 4)
            pose = delta_T @ pose
            
            # Update previous flow for motion encoder (detach to prevent through-time gradient)
            if self.use_motion_input:
                prev_flow = flow_fine.detach()
            
            results['poses'].append(pose)
            results['flows'].append(flow_fine)
            results['confidences'].append(conf_fine)
            results['delta_xis'].append(delta_xi)
        
        return results
    
    def forward_prerendered(
        self,
        query_feats: torch.Tensor,
        rendered_feats_list: List[torch.Tensor],
        initial_pose: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: dict,
    ) -> dict:
        """
        使用预渲染特征的前向传播 (用于 benchmark / 快速测试).
        
        Args:
            query_feats: (B, D, H, W)
            rendered_feats_list: list of (B, D, H, W), 每次迭代一个
            initial_pose: (B, 4, 4)
            depth: (B, H, W)
            intrinsics: {'fx', 'fy', 'cx', 'cy'}
        """
        B, D, H, W = query_feats.shape
        device = query_feats.device
        
        from modules.featuremetric import compute_image_jacobian
        
        if self.use_upsampler:
            s = self.upsample_scale
            uH, uW = H * s, W * s
            up_depth = F.interpolate(
                depth.unsqueeze(1), size=(uH, uW), mode='nearest'
            ).squeeze(1)
            up_intrinsics = {k: v * s for k, v in intrinsics.items()}
            Ju, Jv, valid = compute_image_jacobian(up_depth, up_intrinsics)
            fmap_q = self.encode(query_feats, use_upsampler=True)
            ctx = self.context_encoder(query_feats)
            hidden = torch.tanh(
                F.interpolate(ctx, size=(uH, uW), mode='bilinear', align_corners=False)
            )
        else:
            Ju, Jv, valid = compute_image_jacobian(depth, intrinsics)
            fmap_q = self.encode(query_feats)
            hidden = torch.tanh(self.context_encoder(query_feats))
        
        pose = initial_pose
        results = {
            'poses': [pose],
            'flows': [],
            'confidences': [],
            'delta_xis': [],
        }
        
        for rendered_feats in rendered_feats_list:
            fmap_r = self.encode(rendered_feats, use_upsampler=self.use_upsampler)
            corr = local_correlation(fmap_r, fmap_q, self.corr_radius)
            corr_feat = self.corr_encoder(corr)
            hidden = self.gru(hidden, corr_feat)
            
            flow = self.flow_head(hidden)
            conf = torch.sigmoid(self.conf_head(hidden))
            delta_xi = diff_pose_solve(flow, conf, Ju, Jv, valid, self.damping)
            
            delta_T = se3_exp(delta_xi)
            pose = delta_T @ pose
            
            results['poses'].append(pose)
            results['flows'].append(flow)
            results['confidences'].append(conf)
            results['delta_xis'].append(delta_xi)
        
        return results
    
    @staticmethod
    def _normalize_inv_depth(depth: torch.Tensor) -> torch.Tensor:
        """Compute normalized inverse depth: 1/Z → [0, 1] per sample.
        
        Args:
            depth: (B, H, W) depth map
        Returns:
            inv_d: (B, 1, H, W) normalized inverse depth
        """
        inv_d = 1.0 / (depth + 0.01)  # avoid div by zero
        # Per-sample normalization to [0, 1]
        B = inv_d.shape[0]
        inv_d_flat = inv_d.view(B, -1)
        d_min = inv_d_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
        d_max = inv_d_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)
        inv_d = (inv_d - d_min) / (d_max - d_min + 1e-8)
        return inv_d.unsqueeze(1)  # (B, 1, H, W)
    
    @staticmethod
    def _render_batch(
        renderer, 
        poses: torch.Tensor, 
        scale_name: str, 
        device: torch.device,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """
        渲染一个 batch 的位姿对应的特征图.
        gsplat v1.5+ 支持原生 batch rendering: viewmats [C, 4, 4]
        """
        from feature_3dgs.feature_renderer import FeatureRenderer
        
        model = renderer.models[scale_name]
        info = renderer.scale_info[scale_name]
        fH, fW = info['resolution']
        
        scale_x = fW / renderer.img_width
        scale_y = fH / renderer.img_height
        render_fx = renderer.fx * scale_x
        render_fy = renderer.fy * scale_y
        render_cx = renderer.cx * scale_x
        render_cy = renderer.cy * scale_y
        
        result = FeatureRenderer.render_features_batch(
            gaussian_model=model,
            viewmats=poses,                # [B, 4, 4] — 原生batch!
            fx=render_fx, fy=render_fy,
            cx=render_cx, cy=render_cy,
            img_height=fH,
            img_width=fW,
            feature_height=fH,
            feature_width=fW,
            norm_feat_before_render=True,
            norm_feat_after_render=True,
            max_channels_per_chunk=chunk_size,
        )
        return result['feature_map'].to(device)  # (B, D, fH, fW)
    
    def num_parameters(self) -> int:
        """返回可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        corr_ch = (2 * self.corr_radius + 1) ** 2
        ms_info = ""
        if self.use_multiscale:
            ms_info = (f"\n  multiscale: coarse_iters={self.coarse_iters}, "
                       f"scale_factor={self.coarse_scale_factor},")
        up_info = ""
        if self.use_upsampler:
            up_info = (f"\n  upsampler: {self.feat_dim}d→{self.upsample_dim}d, "
                       f"scale={self.upsample_scale}x, "
                       f"after_iter={self.upsample_after_iter},")
        motion_info = ""
        if self.use_motion_input:
            motion_info = "\n  motion_input: inv_depth(1ch) + flow_feedback(2ch),"
        flow_init_info = ""
        if self.use_flow_init:
            T = self.flow_init_temperature.item()
            gate = torch.sigmoid(self.flow_init_gate).item()
            flow_init_info = f"\n  flow_init: soft-argmax (T={T:.1f}, gate={gate:.3f}),"
        damping_info = ""
        if self.learnable_damping:
            dampings = torch.exp(self.log_damping).detach().cpu().tolist()
            damping_info = f"\n  learnable_damping: {[f'{d:.1e}' for d in dampings]},"
        return (
            f"CorrPoseNet(\n"
            f"  feat_dim={self.feat_dim}, enc_dim={self.enc_dim},\n"
            f"  hidden_dim={self.hidden_dim}, corr_radius={self.corr_radius} "
            f"({corr_ch} channels),\n"
            f"  num_iters={self.num_iters}, damping={self.damping},"
            f"{ms_info}{up_info}{motion_info}{flow_init_info}{damping_info}\n"
            f"  total_params={self.num_parameters():,}\n"
            f")"
        )
