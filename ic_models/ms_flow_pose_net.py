"""
MSFlowPoseNet: SD-Primary Coarse-to-Fine Flow Pose Network
============================================================
核心思路:
  - 单次渲染 (不迭代) → 多尺度 decode → 全局/局部 correlation →
    coarse→mid→fine 流场预测 → 几何求解

层级结构 (Q/R 共享 decoder, 每层独立 FlowHead):
  coarse: SD s5 1280d @  8×10  → 64d, global all-pairs correlation (80ch)
  mid:    SD s4 1280d @ 16×20  → 64d, warp-guided local corr r=4 (81ch)
  fine:   SD s3 640d @32×40 + DINO 768d @35×46 → dual-branch → 64d @35×46, local corr r=4 (81ch)

Fine 层支持 RAFT 风格多次 GRU 迭代: 每次迭代重新计算 warp-guided correlation,
用更新后的 flow 进行渐进细化. fine_iters 默认 4 次.

最终 flow 在 35×46 分辨率 → Image Jacobian + 加权最小二乘 → 几何求解
35×46 = 1,610 像素约束 vs 6 未知数 (268:1), 无需上采样.

参数量估算:
  Decoder × 4 (shared Q/R) + FlowHead × 3 + misc ≈ 2M
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from modules.geometry_solver import compute_image_jacobian, diff_pose_solve
from modules.localizability_head import LocalizabilityHead


# ==============================================================================
#  Building Blocks
# ==============================================================================

class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for spatial disambiguation in feature correlation.

    Supports two modes:
    - 'add': Add PE to features (weak signal, ~1% contribution — deprecated)
    - 'concat': Concatenate PE channels to features (strong signal, ~33% contribution)

    Concat mode is preferred: local/global correlation output shape depends on
    spatial displacement, not feature channels, so the flow heads need no changes.
    The PE channels add a position-dependent similarity term to the dot-product
    correlation, breaking symmetry between identical-looking textures at different
    positions (e.g. stair steps).
    """

    def __init__(self, feat_dim: int = 64, pe_dim: int = 32,
                 mode: str = 'concat'):
        super().__init__()
        self.feat_dim = feat_dim
        self.pe_dim = pe_dim
        self.mode = mode

        if mode == 'add':
            # Project PE to feat_dim for addition
            if pe_dim != feat_dim:
                self.proj = nn.Conv2d(pe_dim, feat_dim, 1, bias=False)
            else:
                self.proj = nn.Identity()
            self.scale = nn.Parameter(torch.tensor(0.1))
        else:
            # Concat mode: PE channels kept separate, scale starts at 1.0
            self.scale = nn.Parameter(torch.tensor(1.0))

        # Sinusoidal base frequencies: half for y, half for x
        half = pe_dim // 2
        freq = torch.exp(torch.arange(0, half, dtype=torch.float32) *
                         -(math.log(10000.0) / half))
        self.register_buffer('freq', freq)

    def _make_pe(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate sinusoidal PE grid. Returns (1, pe_dim, H, W)."""
        half = self.pe_dim // 2
        y_pos = torch.linspace(0, 1, H, device=device)
        x_pos = torch.linspace(0, 1, W, device=device)

        y_enc = y_pos.unsqueeze(1) * self.freq.unsqueeze(0) * math.pi * 2
        x_enc = x_pos.unsqueeze(1) * self.freq.unsqueeze(0) * math.pi * 2

        q = half // 2
        pe_y = torch.cat([torch.sin(y_enc[:, :q]), torch.cos(y_enc[:, :q])], dim=1)
        pe_x = torch.cat([torch.sin(x_enc[:, :q]), torch.cos(x_enc[:, :q])], dim=1)

        pe = torch.zeros(H, W, self.pe_dim, device=device)
        pe[:, :, :half] = pe_y.unsqueeze(1).expand(-1, W, -1)
        pe[:, :, half:] = pe_x.unsqueeze(0).expand(H, -1, -1)

        return pe.permute(2, 0, 1).unsqueeze(0)

    def forward(self, feat: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        """
        Add or concatenate PE to features.
        - 'add':    (B, C, H, W) → (B, C, H, W)
        - 'concat': (B, C, H, W) → (B, C + pe_dim, H, W)
        depth argument is ignored by 2D PE (accepted for interface compatibility).
        """
        B, C, H, W = feat.shape
        pe = self._make_pe(H, W, feat.device)

        if self.mode == 'add':
            pe = self.proj(pe)
            out = feat + self.scale * pe.expand(B, -1, -1, -1)
            return F.normalize(out, p=2, dim=1)
        else:
            # L2-normalize PE independently so it has comparable magnitude to features
            pe = F.normalize(pe, p=2, dim=1)
            pe = self.scale * pe.expand(B, -1, -1, -1)
            return torch.cat([feat, pe], dim=1)


class PositionalEncoding3D(nn.Module):
    """
    Depth-aware 3D positional encoding: extends 2D PE with per-pixel depth encoding.

    The depth channel adds z-axis awareness to correlation, making it possible to
    distinguish repeated textures at different depths (e.g. parallel walls, floor patterns).

    Output (concat mode): (B, C + xy_pe_dim + depth_pe_dim, H, W)

    Depth is log-normalized before encoding to handle the large dynamic range of depth values.
    When depth is not provided, falls back to pure 2D PE (depth channels are zeros).
    """

    def __init__(self, feat_dim: int = 64, pe_dim: int = 32,
                 depth_pe_dim: int = 8, mode: str = 'concat'):
        super().__init__()
        self.feat_dim = feat_dim
        self.xy_pe_dim = pe_dim
        self.depth_pe_dim = depth_pe_dim
        # Total PE channels = xy + depth
        self.pe_dim = pe_dim + depth_pe_dim
        self.mode = mode

        if mode == 'add':
            proj_dim = pe_dim + depth_pe_dim
            if proj_dim != feat_dim:
                self.proj = nn.Conv2d(proj_dim, feat_dim, 1, bias=False)
            else:
                self.proj = nn.Identity()
            self.scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.scale = nn.Parameter(torch.tensor(1.0))

        # XY frequencies (same as 2D PE)
        half_xy = pe_dim // 2
        freq_xy = torch.exp(torch.arange(0, half_xy, dtype=torch.float32) *
                            -(math.log(10000.0) / half_xy))
        self.register_buffer('freq_xy', freq_xy)

        # Depth frequencies (higher frequency range for finer depth discrimination)
        half_z = depth_pe_dim // 2
        freq_z = torch.exp(torch.arange(0, half_z, dtype=torch.float32) *
                           -(math.log(1000.0) / max(half_z, 1)))
        self.register_buffer('freq_z', freq_z)

        # Learnable depth scale (initialized so that log-depth ∈ [0, 1] maps well)
        self.depth_scale = nn.Parameter(torch.tensor(1.0))

    def _make_xy_pe(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate 2D sinusoidal PE grid. Returns (1, xy_pe_dim, H, W)."""
        half = self.xy_pe_dim // 2
        y_pos = torch.linspace(0, 1, H, device=device)
        x_pos = torch.linspace(0, 1, W, device=device)

        y_enc = y_pos.unsqueeze(1) * self.freq_xy.unsqueeze(0) * math.pi * 2
        x_enc = x_pos.unsqueeze(1) * self.freq_xy.unsqueeze(0) * math.pi * 2

        q = half // 2
        pe_y = torch.cat([torch.sin(y_enc[:, :q]), torch.cos(y_enc[:, :q])], dim=1)
        pe_x = torch.cat([torch.sin(x_enc[:, :q]), torch.cos(x_enc[:, :q])], dim=1)

        pe = torch.zeros(H, W, self.xy_pe_dim, device=device)
        pe[:, :, :half] = pe_y.unsqueeze(1).expand(-1, W, -1)
        pe[:, :, half:] = pe_x.unsqueeze(0).expand(H, -1, -1)
        return pe.permute(2, 0, 1).unsqueeze(0)

    def _make_depth_pe(self, depth: torch.Tensor) -> torch.Tensor:
        """Encode per-pixel depth values. Input (B, 1, H, W) → (B, depth_pe_dim, H, W)."""
        # Log-normalize depth: log(1 + d) / log(1 + d_max)
        # Clamp to avoid log(0) and bound the range
        d = depth.float().clamp(min=0.01)
        d_log = torch.log1p(d)
        # Normalize per-batch to [0, 1] for stable encoding
        d_max = d_log.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        d_norm = d_log / d_max.clamp(min=1e-6) * self.depth_scale

        B, _, H, W = depth.shape
        half_z = self.depth_pe_dim // 2

        # (B, 1, H, W) * (half_z,) → (B, half_z, H, W)
        d_flat = d_norm.squeeze(1)  # (B, H, W)
        # Expand frequencies: (B, H, W, half_z)
        z_enc = d_flat.unsqueeze(-1) * self.freq_z.unsqueeze(0).unsqueeze(0).unsqueeze(0) * math.pi * 2

        # Sin/cos encoding
        z_sin = torch.sin(z_enc[..., :half_z])
        z_cos = torch.cos(z_enc[..., :half_z])
        depth_pe = torch.cat([z_sin, z_cos], dim=-1)  # (B, H, W, depth_pe_dim)

        return depth_pe.permute(0, 3, 1, 2)  # (B, depth_pe_dim, H, W)

    def forward(self, feat: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        """
        Apply 3D positional encoding.
        Args:
            feat:  (B, C, H, W) feature map
            depth: (B, 1, H, W) or (B, H, W) depth map. If None, depth PE is zeros.
        Returns:
            concat mode: (B, C + xy_pe_dim + depth_pe_dim, H, W)
            add mode:    (B, C, H, W)
        """
        B, C, H, W = feat.shape
        xy_pe = self._make_xy_pe(H, W, feat.device).expand(B, -1, -1, -1)

        if depth is not None:
            # Ensure (B, 1, H, W)
            if depth.ndim == 3:
                depth = depth.unsqueeze(1)
            # Resize depth to match feature resolution
            if depth.shape[-2:] != (H, W):
                depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)
            depth_pe = self._make_depth_pe(depth)
        else:
            depth_pe = torch.zeros(B, self.depth_pe_dim, H, W, device=feat.device)

        full_pe = torch.cat([xy_pe, depth_pe], dim=1)  # (B, pe_dim, H, W)

        if self.mode == 'add':
            full_pe = self.proj(full_pe)
            out = feat + self.scale * full_pe
            return F.normalize(out, p=2, dim=1)
        else:
            full_pe = F.normalize(full_pe, p=2, dim=1)
            full_pe = self.scale * full_pe
            return torch.cat([feat, full_pe], dim=1)


class ScaleDecoder(nn.Module):
    """
    单尺度 CNN 解码器: high-dim → 64d, 纯逐像素 (不改分辨率).

    1×1 conv 逐步降维, 保留空间结构.
    Query 和 Render 共享同一实例, 确保映射到相同特征空间.
    """

    def __init__(self, in_dim: int, out_dim: int = 64, mid_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.GroupNorm(16, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, mid_dim, 1),
            nn.GroupNorm(16, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, H, W) → (B, 64, H, W), L2 normalized"""
        out = self.net(x)
        return F.normalize(out, p=2, dim=1)


class FineDualDecoder(nn.Module):
    """
    Fine 层双分支解码器:
      - SD s3 → 64d via 1×1 conv → interpolate to target_hw (如有必要)
      - DINO  → 64d via 1×1 conv
      - concat 128d → 1×1 fuse → 64d

    Query 和 Render 共享同一实例.
    当 SD 和 DINO 分辨率相同时自动跳过插值.
    """

    def __init__(self, sd_dim: int = 640, dino_dim: int = 768,
                 out_dim: int = 64, mid_dim: int = 256,
                 target_hw: Tuple[int, int] = (35, 46)):
        super().__init__()
        self.target_hw = target_hw

        # SD s3 branch
        self.sd_branch = nn.Sequential(
            nn.Conv2d(sd_dim, mid_dim, 1),
            nn.GroupNorm(16, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_dim, 1),
        )

        # DINO branch
        self.dino_branch = nn.Sequential(
            nn.Conv2d(dino_dim, mid_dim, 1),
            nn.GroupNorm(16, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_dim, 1),
        )

        # Fusion: 128d → 64d
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1),
        )

    def forward(
        self,
        sd_feat: torch.Tensor,    # (B, 640, Hs, Ws)
        dino_feat: torch.Tensor,  # (B, 768, Hd, Wd)
    ) -> torch.Tensor:
        """→ (B, 64, tH, tW) L2 normalized"""
        sd_out = self.sd_branch(sd_feat)
        # 仅当 SD 分辨率与目标不同时才 interpolate
        if sd_out.shape[2:] != torch.Size(self.target_hw):
            sd_out = F.interpolate(
                sd_out, size=self.target_hw,
                mode='bilinear', align_corners=False,
            )

        dino_out = self.dino_branch(dino_feat)
        if dino_out.shape[2:] != torch.Size(self.target_hw):
            dino_out = F.interpolate(
                dino_out, size=self.target_hw,
                mode='bilinear', align_corners=False,
            )

        fused = self.fuse(torch.cat([sd_out, dino_out], dim=1))
        return F.normalize(fused, p=2, dim=1)


# ==============================================================================
#  Correlation Functions
# ==============================================================================

def global_correlation(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
) -> torch.Tensor:
    """
    全局 All-Pairs Correlation (coarse 层专用).

    对每个 fmap1 像素, 计算与 fmap2 所有像素的点积相关.
    结果是一个 (H1*W1) 维的相关向量 — 每个查询位置获得全局匹配信息.

    Args:
        fmap1: (B, C, H1, W1) 查询特征
        fmap2: (B, C, H2, W2) 渲染特征

    Returns:
        corr: (B, H2*W2, H1, W1) — channel dim = 所有 render 位置
    """
    B, C, H1, W1 = fmap1.shape
    _, _, H2, W2 = fmap2.shape

    # einsum avoids explicit transpose + permute copies
    f1 = fmap1.reshape(B, C, H1 * W1)   # (B, C, N1)
    f2 = fmap2.reshape(B, C, H2 * W2)   # (B, C, N2)
    corr = torch.einsum('bcn,bcm->bmn', f1, f2)  # (B, N2, N1)
    return corr.reshape(B, H2 * W2, H1, W1)


def local_correlation(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    radius: int = 4,
) -> torch.Tensor:
    """
    局部 Correlation (mid/fine 层).
    在 fmap2 上对每个 fmap1 像素的 (2r+1)² 邻域计算相关.

    使用 unfold 向量化实现 (比 Python loop 快 ~10x on GPU, ~50x on CPU).

    Args:
        fmap1: (B, C, H, W)
        fmap2: (B, C, H, W) — 必须与 fmap1 同分辨率
        radius: 搜索半径

    Returns:
        corr: (B, (2r+1)², H, W)
    """
    B, C, H, W = fmap1.shape
    d = 2 * radius + 1

    # Pad fmap2 and extract all (2r+1)×(2r+1) patches via unfold
    fmap2_pad = F.pad(fmap2, [radius] * 4, mode='constant', value=0)
    # unfold: (B, C, H+2r, W+2r) → (B, C, H, d, W, d) via two unfolds
    fmap2_unfold = fmap2_pad.unfold(2, d, 1).unfold(3, d, 1)  # (B, C, H, W, d, d)
    # Reshape for batch dot product: (B, C, H, W, d*d)
    fmap2_unfold = fmap2_unfold.reshape(B, C, H, W, d * d)

    # fmap1: (B, C, H, W) → (B, C, H, W, 1) for broadcasting
    corr = (fmap1.unsqueeze(-1) * fmap2_unfold).sum(dim=1)  # (B, H, W, d*d)

    # Permute to (B, d*d, H, W) — channel-first convention
    return corr.permute(0, 3, 1, 2).contiguous()


def dilated_local_correlation(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    radius: int = 4,
    dilation: int = 2,
) -> torch.Tensor:
    """
    Dilated local correlation: same (2r+1)² window but sampling every `dilation`
    pixels, covering effective radius = radius * dilation.

    Args:
        fmap1: (B, C, H, W)
        fmap2: (B, C, H, W)
        radius: search radius (number of offsets per direction)
        dilation: spacing between sampled positions

    Returns:
        corr: (B, (2r+1)², H, W)
    """
    B, C, H, W = fmap1.shape
    d = 2 * radius + 1
    effective_r = radius * dilation

    fmap2_pad = F.pad(fmap2, [effective_r] * 4, mode='constant', value=0)

    # Generate dilated offsets: -r*d, ..., 0, ..., r*d with step d
    offsets = torch.arange(-radius, radius + 1, device=fmap1.device) * dilation + effective_r

    # Gather dilated patches: (B, C, H, W, d*d)
    patches = []
    for dy in offsets:
        for dx in offsets:
            patch = fmap2_pad[:, :, dy:dy + H, dx:dx + W]  # (B, C, H, W)
            patches.append(patch)
    fmap2_dilated = torch.stack(patches, dim=-1)  # (B, C, H, W, d*d)

    corr = (fmap1.unsqueeze(-1) * fmap2_dilated).sum(dim=1)  # (B, H, W, d*d)
    return corr.permute(0, 3, 1, 2).contiguous()


# Grid cache for guided_local_correlation (avoids repeated meshgrid calls)
_GRID_CACHE: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_base_grid(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cached (grid_x, grid_y) each shaped (1, H, W)."""
    key = (H, W, device)
    if key not in _GRID_CACHE:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        _GRID_CACHE[key] = (grid_x.unsqueeze(0), grid_y.unsqueeze(0))
    return _GRID_CACHE[key]


def guided_local_correlation(
    fmap_q: torch.Tensor,
    fmap_r: torch.Tensor,
    flow: torch.Tensor,
    radius: int = 4,
) -> torch.Tensor:
    """
    Warp-guided 局部 correlation.

    先用 flow 将 fmap_r 向 fmap_q 方向 warp, 然后做标准局部 correlation.
    这样局部搜索窗口就以上一级预测的位移为中心, 避免大位移被 local window 截断.

    Args:
        fmap_q: (B, C, H, W)
        fmap_r: (B, C, H, W)
        flow:   (B, 2, H, W) 上一级上采样并缩放后的 flow [du, dv]
        radius: 搜索半径

    Returns:
        corr: (B, (2r+1)², H, W)
    """
    B, C, H, W = fmap_q.shape

    # 构建 warp grid: base_grid + flow → normalized [-1, 1]
    # 使用缓存的 base grid 避免每次 meshgrid 调用
    grid_x, grid_y = _get_base_grid(H, W, flow.device)
    # flow[:, 0] = du (x-dir), flow[:, 1] = dv (y-dir)
    grid_x = (grid_x + flow[:, 0]) / (W - 1) * 2.0 - 1.0
    grid_y = (grid_y + flow[:, 1]) / (H - 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)

    warped_r = F.grid_sample(
        fmap_r, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    return local_correlation(fmap_q, warped_r, radius=radius)


def guided_multiscale_correlation(
    fmap_q: torch.Tensor,
    fmap_r: torch.Tensor,
    flow: torch.Tensor,
    radius: int = 4,
    dilations: Tuple[int, ...] = (1, 2),
) -> torch.Tensor:
    """
    Multi-scale warp-guided correlation (RAFT-style correlation pyramid).

    Warp fmap_r using flow, then compute local correlation at multiple dilation
    levels. Concatenates results → richer motion context for the GRU.

    Args:
        fmap_q: (B, C, H, W)
        fmap_r: (B, C, H, W)
        flow:   (B, 2, H, W)
        radius: search radius per level
        dilations: tuple of dilation factors, e.g. (1, 2) or (1, 2, 4)

    Returns:
        corr: (B, len(dilations) * (2r+1)², H, W)
    """
    B, C, H, W = fmap_q.shape

    # Warp fmap_r using flow
    grid_x, grid_y = _get_base_grid(H, W, flow.device)
    grid_x = (grid_x + flow[:, 0]) / (W - 1) * 2.0 - 1.0
    grid_y = (grid_y + flow[:, 1]) / (H - 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    warped_r = F.grid_sample(
        fmap_r, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    corrs = []
    for d in dilations:
        if d == 1:
            corrs.append(local_correlation(fmap_q, warped_r, radius=radius))
        else:
            corrs.append(dilated_local_correlation(fmap_q, warped_r, radius=radius, dilation=d))

    return torch.cat(corrs, dim=1)


# ==============================================================================
#  ConvGRU (from CorrPoseNet)
# ==============================================================================

class ContextAdapter(nn.Module):
    """
    上采样 hidden 与当前尺度 query 特征融合 → 初始化当前尺度的 GRU hidden.

    解决原始设计中 mid/fine 隐状态仅靠双线性上采样、缺乏尺度特异信息的问题.
    用 residual connection 保留上一级信息, 并注入当前尺度的细节.
    """

    def __init__(self, hidden_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim + feat_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, h_up: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_up:   (B, hidden_dim, H, W) upsampled hidden from previous scale
            q_feat: (B, feat_dim, H, W) current scale query features
        Returns:
            (B, hidden_dim, H, W) refined hidden state
        """
        return h_up + self.net(torch.cat([h_up, q_feat], dim=1))


class ConvGRU(nn.Module):
    """Convolutional GRU, 3×3 卷积门控.

    z/r 门共享同一次 concat + 单个 conv (内部 chunk 拆分),
    减少一次 cat + 一次 3×3 conv 的开销.
    """

    def __init__(self, hidden_dim: int = 128, input_dim: int = 64):
        super().__init__()
        # Fused z+r gate: one conv outputs 2*hidden_dim, then chunk
        self.conv_zr = nn.Conv2d(hidden_dim + input_dim, 2 * hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        zr = torch.sigmoid(self.conv_zr(hx))
        z, r = zr.chunk(2, dim=1)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


# ==============================================================================
#  Flow Refinement Head (per-scale)
# ==============================================================================


class _ResBlock(nn.Module):
    """Simple residual block with GroupNorm."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class DeepFlowHead(nn.Module):
    """
    Deeper flow prediction head with residual blocks.
    hidden_dim → 128 → ResBlock → ResBlock → 64 → flow(2) + conf(conf_dim)
    ~4× deeper than the original 2-layer head.
    """
    def __init__(self, hidden_dim: int, mid_dim: int = 128, conf_dim: int = 1):
        super().__init__()
        self.conf_dim = conf_dim
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, mid_dim, 3, padding=1),
            nn.GroupNorm(8, mid_dim),
            nn.GELU(),
            _ResBlock(mid_dim),
            _ResBlock(mid_dim),
            nn.Conv2d(mid_dim, 64, 3, padding=1),
            nn.GELU(),
        )
        self.flow_conv = nn.Conv2d(64, 2, 3, padding=1)
        self.conf_conv = nn.Conv2d(64, conf_dim, 3, padding=1)

        nn.init.normal_(self.flow_conv.weight, std=0.001)
        nn.init.zeros_(self.flow_conv.bias)
        nn.init.zeros_(self.conf_conv.weight)
        nn.init.zeros_(self.conf_conv.bias)

    def forward(self, x):
        feat = self.net(x)
        flow = self.flow_conv(feat)
        conf = self.conf_conv(feat)
        return torch.cat([flow, conf], dim=1)  # (B, 2+conf_dim, H, W)


class CrossScaleContext(nn.Module):
    """
    Encode upsampled coarse+mid decoded features into a compact context vector.
    This provides global and mid-scale disambiguation cues to the fine-level GRU
    iterations, helping resolve ambiguities from repetitive textures and limited
    local correlation windows.

    Uses zero-init on last layer so output starts at zero (no disruption to
    warmstarted models). The model gradually learns to use cross-scale info.
    """

    def __init__(self, decode_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(decode_dim * 2, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, out_dim, 1),
        )
        # Zero-init last conv so cross-scale starts as no-op
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        q_coarse: torch.Tensor,  # (B, D, H_c, W_c)
        q_mid: torch.Tensor,     # (B, D, H_m, W_m)
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """→ (B, out_dim, tH, tW)"""
        q_c_up = F.interpolate(q_coarse, target_hw, mode='bilinear', align_corners=False)
        q_m_up = F.interpolate(q_mid, target_hw, mode='bilinear', align_corners=False)
        return self.net(torch.cat([q_c_up, q_m_up], dim=1))


class PoseRefinementHead(nn.Module):
    """
    Lightweight MLP that refines the analytical geometry solver output.
    Takes the solver's delta_xi plus pooled features from the fine hidden state
    to predict a residual correction, compensating for systematic flow biases.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # Pool hidden features → compact summary
        self.pool_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
        )
        # Combine pooled features + raw delta_xi → refined delta_xi
        self.refine = nn.Sequential(
            nn.Linear(64 + 6, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 6),
        )
        # Init last layer near-zero so refinement starts as identity
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(
        self,
        delta_xi: torch.Tensor,    # (B, 6) from geometry solver
        hidden: torch.Tensor,      # (B, hidden_dim, H, W)
    ) -> torch.Tensor:
        """→ (B, 6) refined delta_xi"""
        feat = self.pool_net(hidden)          # (B, 64)
        combined = torch.cat([feat, delta_xi], dim=1)  # (B, 70)
        correction = self.refine(combined)    # (B, 6)
        return delta_xi + correction


class FlowRefinementHead(nn.Module):
    """
    单尺度 flow 细化头:
      correlation → encoder → ConvGRU 更新 hidden → 输出 flow + confidence

    每个尺度独立一个 Head (不共享权重).

    Args:
        corr_channels: correlation 的 channel 数
          - coarse(global): H2*W2 = 80
          - mid/fine(local r=4): (2*4+1)² = 81
        hidden_dim: GRU 隐状态维度
        flow_dim: flow + confidence 输出维度 (3: du, dv, conf)
        deep_flow_head: use deeper flow head with residual blocks
        cross_scale_dim: extra channels from cross-scale context (0=disabled)
    """

    def __init__(
        self,
        corr_channels: int,
        hidden_dim: int = 128,
        context_dim: int = 64,
        deep_flow_head: bool = False,
        cross_scale_dim: int = 0,
        conf_dim: int = 1,
    ):
        super().__init__()
        self.cross_scale_dim = cross_scale_dim
        self.conf_dim = conf_dim

        # Correlation encoder: corr + flow(2) + conf(conf_dim) → context_dim
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(corr_channels + 2 + conf_dim, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, context_dim, 3, padding=1),
            nn.GELU(),
        )

        # Cross-scale context: separate encoder, additively injected
        # Zero-init first layer so warmstarted model starts unchanged
        if cross_scale_dim > 0:
            self.cross_encoder = nn.Sequential(
                nn.Conv2d(cross_scale_dim, context_dim, 3, padding=1),
                nn.GELU(),
            )
            nn.init.zeros_(self.cross_encoder[0].weight)
            nn.init.zeros_(self.cross_encoder[0].bias)

        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=context_dim)

        if deep_flow_head:
            self.flow_head = DeepFlowHead(hidden_dim, conf_dim=conf_dim)
        else:
            # Original shallow flow head
            self.flow_head = nn.Sequential(
                nn.Conv2d(hidden_dim, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 2 + conf_dim, 3, padding=1),  # du, dv, raw_conf(s)
            )

    def forward(
        self,
        corr: torch.Tensor,     # (B, corr_ch, H, W)
        hidden: torch.Tensor,   # (B, hidden_dim, H, W)
        flow: torch.Tensor,     # (B, 2, H, W) current flow estimate
        confidence: torch.Tensor,  # (B, conf_dim, H, W)
        cross_ctx: Optional[torch.Tensor] = None,  # (B, cross_scale_dim, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_flow: (B, 2, H, W) flow update
            new_conf:   (B, conf_dim, H, W) updated confidence
            new_hidden: (B, hidden_dim, H, W)
            flow_out:   (B, 2, H, W) = flow + delta_flow
        """
        # Encode: corr + current flow + conf
        inp = torch.cat([corr, flow, confidence], dim=1)
        inp_encoded = self.corr_encoder(inp)

        # Additive cross-scale context injection
        if cross_ctx is not None and self.cross_scale_dim > 0:
            inp_encoded = inp_encoded + self.cross_encoder(cross_ctx)

        # GRU update
        new_hidden = self.gru(hidden, inp_encoded)

        # Predict flow delta + conf
        out = self.flow_head(new_hidden)
        delta_flow = out[:, :2]          # (B, 2, H, W)
        raw_conf = out[:, 2:]            # (B, conf_dim, H, W)
        new_conf = torch.sigmoid(raw_conf)

        flow_out = flow + delta_flow

        return delta_flow, new_conf, new_hidden, flow_out


# ==============================================================================
#  Main Network
# ==============================================================================

class MSFlowPoseNet(nn.Module):
    """
    SD-Primary Coarse-to-Fine Multi-Scale Flow Pose Network.

    Architecture:
      1. Decode query/render features at each scale to 64d (shared decoder)
      2. Coarse: global all-pairs correlation → flow head → initial flow
      3. Mid: upsample coarse flow → warp-guided local corr → flow head
      4. Fine: upsample mid flow → dual-branch decode → local corr → flow head
      5. Geometry solver: Image Jacobian + weighted least squares → Δξ

    Resolution 通过构造函数参数配置, 同时支持 v1/v2 特征:
      v1: coarse 7×10, mid 15×20, fine_sd 35×46, fine 35×46
      v2: coarse 8×10, mid 16×20, fine_sd 32×40, fine 35×46

    Args:
        hidden_dim: GRU hidden state dimension
        decode_dim: decoder output dimension (common across scales)
        local_radius: local correlation search radius
        damping: LM solver damping
        coarse_hw: coarse 特征分辨率 (H, W)
        mid_hw: mid 特征分辨率
        fine_hw: fine 输出分辨率 (= DINO patch grid)
        fine_iters: fine 层 GRU 迭代次数 (RAFT 风格, 默认 4)
    """

    # Default intrinsics (Replica room_0 base) — override via constructor
    DEFAULT_INTRINSICS = {'fx': 320.0, 'fy': 320.0, 'cx': 319.5, 'cy': 239.5}
    DEFAULT_IMG_HW = (480, 640)

    def __init__(
        self,
        hidden_dim: int = 128,
        decode_dim: int = 64,
        local_radius: int = 4,
        damping: float = 1e-3,
        coarse_hw: Tuple[int, int] = (7, 10),
        mid_hw: Tuple[int, int] = (15, 20),
        fine_hw: Tuple[int, int] = (35, 46),
        fine_iters: int = 4,
        mid_iters: int = 1,
        corr_temperature: float = 1.0,
        # Learnable per-scale correlation temperature: each scale gets its
        # own temperature parameter initialized to corr_temperature.
        # Softplus ensures positivity; gradients flow through correlation scores.
        learnable_temperature: bool = False,
        intrinsics: Optional[Dict[str, float]] = None,
        img_hw: Optional[Tuple[int, int]] = None,
        # ODISE backbone projects SD features to 512d (not raw 1280/640)
        coarse_in_dim: int = 512,
        mid_in_dim: int = 512,
        fine_sd_in_dim: int = 512,
        fine_dino_in_dim: int = 768,
        # IRLS robust estimation in geometry solver
        irls_iters: int = 0,
        irls_huber_k: float = 1.345,
        # Deeper flow head with residual blocks
        deep_flow_head: bool = False,
        # Cross-scale context: inject coarse+mid features into fine iterations
        cross_scale_context: bool = False,
        cross_scale_dim: int = 32,
        # Pose refinement: learnable correction after geometry solver
        pose_refinement: bool = False,
        # Multi-scale correlation: dilated lookups at fine stage
        # e.g. (1, 2) → standard + dilation=2, (1, 2, 4) → three levels
        corr_dilations: Optional[Tuple[int, ...]] = None,
        # Geometry upsampling: solve geometry at higher resolution than flow
        # e.g. 2 → upsample flow/depth 2× before geometry solver
        geometry_upsample: int = 1,
        # Multi-scale consistency: use coarse+mid flow agreement to reweight
        # confidence in geometry solver (helps with repetitive textures)
        multiscale_consistency: bool = False,
        ms_consistency_sigma: float = 1.0,
        # Pixel stride for geometry solver subsampling
        pixel_stride: int = 1,
        # Adaptive damping: scale LM damping by condition number of J^T W J
        adaptive_damping: bool = False,
        adaptive_damping_max: float = 0.1,
        adaptive_damping_cond_thresh: float = 1e4,
        # Positional encoding: add spatial awareness to decoded features
        # before correlation (helps disambiguate repetitive textures)
        positional_encoding: bool = False,
        pe_mode: str = 'concat',   # 'concat' (strong) or 'add' (weak)
        pe_dim: int = 32,          # PE channel count (concat only)
        depth_pe_dim: int = 0,     # >0 enables 3D PE with depth encoding channels
        # Skip coarse flow: initialize mid from zero instead of coarse prediction.
        # Useful when coarse features are too similar for reliable global correlation.
        skip_coarse_flow: bool = False,
        # Directional confidence: output 2-channel (u,v) confidence
        # instead of scalar, enabling direction-specific weighting in
        # the geometry solver. Backward compatible — solver auto-detects.
        directional_confidence: bool = False,
        # DINOv2 at all scales: downsample fine_dino to coarse/mid resolutions
        # and fuse with SD features via learned gates (warmstart safe: gates start at 0)
        dino_all_scales: bool = False,
        # DINOv2 replace SD: completely replace SD features at coarse/mid with DINOv2
        # More aggressive than dino_all_scales (which blends). Directly addresses
        # the OldHospital structural floor caused by flat SD correlations.
        dino_replace_sd: bool = False,
        # Localizability prior: per-pixel scoring of how localizable each pixel is,
        # based on correlation volume statistics (peak, sharpness, entropy).
        # Modulates confidence before geometry solver. Supervised by flow error.
        localizability_prior: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decode_dim = decode_dim
        self.local_radius = local_radius
        self.damping = damping
        self.fine_iters = fine_iters
        self.mid_iters = mid_iters
        self.corr_temperature = corr_temperature
        self.learnable_temperature = learnable_temperature
        self.directional_confidence = directional_confidence
        self.irls_iters = irls_iters
        self.irls_huber_k = irls_huber_k
        self.use_cross_scale = cross_scale_context
        self.use_pose_refinement = pose_refinement
        self.corr_dilations = corr_dilations
        self.geometry_upsample = geometry_upsample
        self.multiscale_consistency = multiscale_consistency
        self.ms_consistency_sigma = ms_consistency_sigma
        self.pixel_stride = pixel_stride
        self.adaptive_damping = adaptive_damping
        self.adaptive_damping_max = adaptive_damping_max
        self.adaptive_damping_cond_thresh = adaptive_damping_cond_thresh
        self.use_positional_encoding = positional_encoding
        self.pe_mode = pe_mode
        self.depth_pe_dim = depth_pe_dim
        self.skip_coarse_flow = skip_coarse_flow
        self.dino_all_scales = dino_all_scales
        self.dino_replace_sd = dino_replace_sd
        self.use_localizability = localizability_prior

        # Configurable intrinsics and image resolution
        self.BASE_INTRINSICS = intrinsics if intrinsics is not None else self.DEFAULT_INTRINSICS.copy()
        self.IMG_HW = img_hw if img_hw is not None else self.DEFAULT_IMG_HW

        # 分辨率配置 (实例属性, 不再是类常量)
        self.COARSE_HW = tuple(coarse_hw)
        self.MID_HW = tuple(mid_hw)
        self.FINE_HW = tuple(fine_hw)

        d = 2 * local_radius + 1
        local_corr_ch = d * d  # 81

        # Multi-scale correlation: multiply corr channels by number of dilation levels
        if corr_dilations is not None and len(corr_dilations) > 1:
            fine_corr_ch = local_corr_ch * len(corr_dilations)
        else:
            fine_corr_ch = local_corr_ch

        # ── Decoders (Q/R shared) ──
        self.coarse_dec = ScaleDecoder(coarse_in_dim, decode_dim)
        self.mid_dec = ScaleDecoder(mid_in_dim, decode_dim)
        self.fine_dec = FineDualDecoder(
            fine_sd_in_dim, fine_dino_in_dim, decode_dim, target_hw=self.FINE_HW)

        # ── DINOv2 at all scales: parallel decoders + learned fusion gates ──
        if dino_all_scales:
            self.coarse_dino_dec = ScaleDecoder(fine_dino_in_dim, decode_dim)
            self.mid_dino_dec = ScaleDecoder(fine_dino_in_dim, decode_dim)
            # Gates start at 0 → DINOv2 contributes nothing at warmstart
            self.coarse_dino_gate = nn.Parameter(torch.tensor(0.0))
            self.mid_dino_gate = nn.Parameter(torch.tensor(0.0))

        # ── DINOv2 replace SD at coarse/mid: dedicated decoders, no blending ──
        if dino_replace_sd:
            self.coarse_dino_dec = ScaleDecoder(fine_dino_in_dim, decode_dim)
            self.mid_dino_dec = ScaleDecoder(fine_dino_in_dim, decode_dim)

        # ── Learnable per-scale correlation temperature ──
        # Softplus(raw) ensures positivity; initialized so softplus(raw)≈corr_temperature
        if learnable_temperature:
            # inverse softplus: raw = log(exp(t) - 1)
            init_raw = math.log(math.exp(corr_temperature) - 1.0) if corr_temperature > 0 else 0.0
            self.temp_coarse_raw = nn.Parameter(torch.tensor(init_raw))
            self.temp_mid_raw = nn.Parameter(torch.tensor(init_raw))
            self.temp_fine_raw = nn.Parameter(torch.tensor(init_raw))

        # ── Flow Heads ──
        coarse_global_ch = self.COARSE_HW[0] * self.COARSE_HW[1]
        conf_dim = 2 if directional_confidence else 1
        self.conf_dim = conf_dim
        self.coarse_head = FlowRefinementHead(
            corr_channels=coarse_global_ch,
            hidden_dim=hidden_dim,
            deep_flow_head=deep_flow_head,
            conf_dim=conf_dim,
        )
        self.mid_head = FlowRefinementHead(
            corr_channels=local_corr_ch,
            hidden_dim=hidden_dim,
            deep_flow_head=deep_flow_head,
            conf_dim=conf_dim,
        )
        self.fine_head = FlowRefinementHead(
            corr_channels=fine_corr_ch,
            hidden_dim=hidden_dim,
            deep_flow_head=deep_flow_head,
            cross_scale_dim=cross_scale_dim if cross_scale_context else 0,
            conf_dim=conf_dim,
        )

        # ── Positional encoding: add spatial identity to decoded features ──
        if positional_encoding:
            if depth_pe_dim > 0:
                # 3D PE: xy + depth sinusoidal encoding
                PE_cls = lambda: PositionalEncoding3D(
                    feat_dim=decode_dim, pe_dim=pe_dim,
                    depth_pe_dim=depth_pe_dim, mode=pe_mode)
            else:
                PE_cls = lambda: PositionalEncoding2D(
                    feat_dim=decode_dim, pe_dim=pe_dim, mode=pe_mode)
            self.fine_pe = PE_cls()
            self.mid_pe = PE_cls()
            self.coarse_pe = PE_cls()

        # ── Cross-scale context: coarse+mid → compact vector for fine iterations ──
        if cross_scale_context:
            self.cross_scale_ctx = CrossScaleContext(
                decode_dim=decode_dim, out_dim=cross_scale_dim)

        # ── Pose refinement head: refine geometry solver output ──
        if pose_refinement:
            self.pose_refine_head = PoseRefinementHead(hidden_dim=hidden_dim)

        # ── Context encoder: query feature → initial hidden state ──
        self.context_net = nn.Sequential(
            nn.Conv2d(decode_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # ── Per-scale context adapters: inject scale-specific query features ──
        self.mid_context = ContextAdapter(hidden_dim, decode_dim)
        self.fine_context = ContextAdapter(hidden_dim, decode_dim)

        # ── Localizability prior: score how localizable each fine pixel is ──
        if localizability_prior:
            self.loc_head = LocalizabilityHead(
                corr_channels=fine_corr_ch, hidden_dim=32)

        # Intrinsics at fine resolution for geometry solving
        self.fine_intrinsics = self._scale_intrinsics(*self.FINE_HW)
        # If geometry_upsample > 1, compute upsampled intrinsics
        if self.geometry_upsample > 1:
            s = self.geometry_upsample
            uH, uW = self.FINE_HW[0] * s, self.FINE_HW[1] * s
            self.geo_intrinsics = self._scale_intrinsics(uH, uW)
        else:
            self.geo_intrinsics = self.fine_intrinsics

    def _scale_intrinsics(self, tH: int, tW: int) -> Dict[str, float]:
        """从 640×480 基准缩放内参到目标分辨率."""
        return {
            'fx': self.BASE_INTRINSICS['fx'] * tW / self.IMG_HW[1],
            'fy': self.BASE_INTRINSICS['fy'] * tH / self.IMG_HW[0],
            'cx': self.BASE_INTRINSICS['cx'] * tW / self.IMG_HW[1],
            'cy': self.BASE_INTRINSICS['cy'] * tH / self.IMG_HW[0],
        }

    def _get_temperature(self, scale: str) -> float:
        """Get correlation temperature for a given scale."""
        if self.learnable_temperature:
            raw = getattr(self, f'temp_{scale}_raw')
            return F.softplus(raw)
        return self.corr_temperature

    def _init_flow(self, B: int, H: int, W: int,
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化零 flow 和均匀 confidence."""
        flow = torch.zeros(B, 2, H, W, device=device)
        conf = torch.ones(B, self.conf_dim, H, W, device=device) * 0.5
        return flow, conf

    def _upsample_flow(
        self, flow: torch.Tensor, conf: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        双线性上采样 flow 和 confidence 到目标分辨率, 并缩放 flow 值.
        """
        _, _, sH, sW = flow.shape
        tH, tW = target_hw
        scale_x = tW / sW
        scale_y = tH / sH

        flow_up = F.interpolate(flow, size=target_hw, mode='bilinear',
                                 align_corners=False)
        flow_up[:, 0] *= scale_x
        flow_up[:, 1] *= scale_y

        conf_up = F.interpolate(conf, size=target_hw, mode='bilinear',
                                 align_corners=False)
        return flow_up, conf_up

    @staticmethod
    def _match_spatial(x, target_hw):
        """Resize tensor to target_hw if spatial dims don't match."""
        if x.shape[-2:] != tuple(target_hw):
            return F.interpolate(x, size=target_hw, mode='bilinear',
                                 align_corners=False)
        return x

    def forward(
        self,
        query_feats: Dict[str, torch.Tensor],
        render_feats: Dict[str, torch.Tensor],
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            query_feats: {
                'coarse':    (B, 1280, 8, 10),
                'mid':       (B, 1280, 16, 20),
                'fine_sd':   (B, 640, 32, 40),
                'fine_dino': (B, 768, 35, 46),
            }
            render_feats: same structure (from 3DGS rendering)
            depth: (B, 35, 46) rendered depth at fine resolution

        Returns:
            dict:
                'delta_xi': (B, 6) se(3) update — only if depth provided
                'flow_fine': (B, 2, 35, 46) final fine-resolution flow
                'conf_fine': (B, 1, 35, 46)
                'flow_coarse': (B, 2, 8, 10)
                'flow_mid':    (B, 2, 16, 20)
                'hidden_fine': (B, hidden_dim, 35, 46)
        """
        B = query_feats['coarse'].shape[0]
        device = query_feats['coarse'].device

        # ══════════════════════════════════════════════
        #  1. Decode all scales to 64d (shared Q/R decoders)
        # ══════════════════════════════════════════════
        q_coarse = self._match_spatial(
            self.coarse_dec(query_feats['coarse']), self.COARSE_HW)
        r_coarse = self.coarse_dec(render_feats['coarse'])

        q_mid = self._match_spatial(
            self.mid_dec(query_feats['mid']), self.MID_HW)
        r_mid = self._match_spatial(
            self.mid_dec(render_feats['mid']), self.MID_HW)

        q_fine = self._match_spatial(
            self.fine_dec(query_feats['fine_sd'], query_feats['fine_dino']),
            self.FINE_HW)
        r_fine = self._match_spatial(
            self.fine_dec(render_feats['fine_sd'], render_feats['fine_dino']),
            self.FINE_HW)

        # ── DINOv2 at all scales: fuse downsampled DINOv2 with SD at coarse/mid ──
        if self.dino_all_scales:
            # Downsample fine_dino to coarse/mid resolutions
            q_dino_c = self.coarse_dino_dec(self._match_spatial(
                query_feats['fine_dino'], self.COARSE_HW))
            r_dino_c = self.coarse_dino_dec(self._match_spatial(
                render_feats['fine_dino'], self.COARSE_HW))
            q_dino_m = self.mid_dino_dec(self._match_spatial(
                query_feats['fine_dino'], self.MID_HW))
            r_dino_m = self.mid_dino_dec(self._match_spatial(
                render_feats['fine_dino'], self.MID_HW))
            # Additive fusion with learned gates (warmstart safe: gates start at 0)
            gc = self.coarse_dino_gate
            gm = self.mid_dino_gate
            q_coarse = F.normalize(q_coarse + gc * q_dino_c, p=2, dim=1)
            r_coarse = F.normalize(r_coarse + gc * r_dino_c, p=2, dim=1)
            q_mid = F.normalize(q_mid + gm * q_dino_m, p=2, dim=1)
            r_mid = F.normalize(r_mid + gm * r_dino_m, p=2, dim=1)

        # ── DINOv2 replace SD at coarse/mid: complete replacement ──
        if self.dino_replace_sd:
            q_coarse = self.coarse_dino_dec(self._match_spatial(
                query_feats['fine_dino'], self.COARSE_HW))
            r_coarse = self.coarse_dino_dec(self._match_spatial(
                render_feats['fine_dino'], self.COARSE_HW))
            q_mid = self.mid_dino_dec(self._match_spatial(
                query_feats['fine_dino'], self.MID_HW))
            r_mid = self.mid_dino_dec(self._match_spatial(
                render_feats['fine_dino'], self.MID_HW))

        # Apply positional encoding to disambiguate repetitive textures
        # PE is applied to ALL scales' correlation inputs (coarse/mid/fine)
        # NOT context adapter inputs, because concat PE changes channel count
        # For 3D PE (depth_pe_dim > 0), depth is passed for z-axis encoding
        if self.use_positional_encoding:
            pe_depth = depth if self.depth_pe_dim > 0 else None
            q_coarse_corr = self.coarse_pe(q_coarse, pe_depth)
            r_coarse_corr = self.coarse_pe(r_coarse, pe_depth)
            q_mid_corr = self.mid_pe(q_mid, pe_depth)
            r_mid_corr = self.mid_pe(r_mid, pe_depth)
            q_fine_corr = self.fine_pe(q_fine, pe_depth)
            r_fine_corr = self.fine_pe(r_fine, pe_depth)
        else:
            q_coarse_corr = q_coarse
            r_coarse_corr = r_coarse
            q_mid_corr = q_mid
            r_mid_corr = r_mid
            q_fine_corr = q_fine
            r_fine_corr = r_fine

        # ══════════════════════════════════════════════
        #  2. Coarse: Global correlation → initial flow
        # ══════════════════════════════════════════════
        coarse_corr = global_correlation(q_coarse_corr, r_coarse_corr)
        temp_coarse = self._get_temperature('coarse')
        if self.learnable_temperature or temp_coarse != 1.0:
            coarse_corr = coarse_corr / temp_coarse

        # Context → initial hidden (at coarse resolution)
        h_coarse = self.context_net(q_coarse)  # (B, hidden_dim, 8, 10)

        flow_c, conf_c = self._init_flow(
            B, *self.COARSE_HW, device)

        _, conf_c, h_coarse, flow_c = self.coarse_head(
            coarse_corr, h_coarse, flow_c, conf_c)

        # ══════════════════════════════════════════════
        #  3. Mid: Warp-guided local correlation (iterative)
        # ══════════════════════════════════════════════
        if self.skip_coarse_flow:
            # Start mid from zero flow — coarse correlation too flat to be useful
            flow_m, conf_m = self._init_flow(B, *self.MID_HW, device)
        else:
            flow_m, conf_m = self._upsample_flow(
                flow_c, conf_c, self.MID_HW)

        # Initialize mid hidden: upsample coarse hidden + inject mid query context
        h_mid = F.interpolate(
            h_coarse, size=self.MID_HW, mode='bilinear', align_corners=False)
        h_mid = self.mid_context(h_mid, q_mid)

        mid_flow_preds = []  # 存储 mid 迭代的 flow, 用于 sequence loss

        for _iter in range(self.mid_iters):
            mid_corr = guided_local_correlation(
                q_mid_corr, r_mid_corr, flow_m, radius=self.local_radius)
            temp_mid = self._get_temperature('mid')
            if self.learnable_temperature or temp_mid != 1.0:
                mid_corr = mid_corr / temp_mid

            _, conf_m, h_mid, flow_m = self.mid_head(
                mid_corr, h_mid, flow_m, conf_m)

            mid_flow_preds.append(flow_m)

        # ══════════════════════════════════════════════
        #  4. Fine: RAFT-style iterative refinement
        #     多次 GRU 迭代, 每次重新计算 warp-guided correlation
        # ══════════════════════════════════════════════
        flow_f, conf_f = self._upsample_flow(
            flow_m, conf_m, self.FINE_HW)

        h_fine = F.interpolate(
            h_mid, size=self.FINE_HW, mode='bilinear', align_corners=False)
        h_fine = self.fine_context(h_fine, q_fine)

        # Pre-compute cross-scale context (reused across iterations)
        cross_ctx = None
        if self.use_cross_scale:
            cross_ctx = self.cross_scale_ctx(q_coarse, q_mid, self.FINE_HW)

        fine_flow_preds = []   # 存储每次迭代的 flow, 用于 sequence loss

        for _iter in range(self.fine_iters):
            if self.corr_dilations is not None and len(self.corr_dilations) > 1:
                fine_corr = guided_multiscale_correlation(
                    q_fine_corr, r_fine_corr, flow_f,
                    radius=self.local_radius,
                    dilations=self.corr_dilations)
            else:
                fine_corr = guided_local_correlation(
                    q_fine_corr, r_fine_corr, flow_f, radius=self.local_radius)
            temp_fine = self._get_temperature('fine')
            if self.learnable_temperature or temp_fine != 1.0:
                fine_corr = fine_corr / temp_fine

            _, conf_f, h_fine, flow_f = self.fine_head(
                fine_corr, h_fine, flow_f, conf_f, cross_ctx=cross_ctx)

            fine_flow_preds.append(flow_f)

        # ── Localizability prior: modulate confidence with per-pixel score ──
        loc_score = None
        if self.use_localizability:
            loc_score = self.loc_head(fine_corr)  # (B, 1, H, W)
            conf_f = conf_f * loc_score  # modulate: low localizability → low weight

        # ══════════════════════════════════════════════
        #  5. Geometry Solver (if depth available)
        #     直接在 35×46 分辨率求解, 无需上采样
        # ══════════════════════════════════════════════
        result = {
            'flow_coarse': flow_c,
            'conf_coarse': conf_c,
            'flow_mid': flow_m,
            'conf_mid': conf_m,
            'flow_fine': flow_f,
            'conf_fine': conf_f,
            'hidden_fine': h_fine,
            'fine_flow_preds': fine_flow_preds,  # 每次迭代的 flow, 用于 sequence loss
            'mid_flow_preds': mid_flow_preds,    # mid 迭代的 flow
            # Decoded features for diversity regularization
            'decoded_q_coarse': q_coarse,
            'decoded_q_mid': q_mid,
            'decoded_q_fine': q_fine,
        }
        if loc_score is not None:
            result['loc_score'] = loc_score

        if depth is not None:
            # Geometry solver must run in fp32 (linalg.solve doesn't support fp16)
            with torch.amp.autocast('cuda', enabled=False):
                depth_f32 = depth.float()
                flow_f32 = flow_f.float()
                conf_f32 = conf_f.float()

                # Multi-scale consistency: downweight pixels where coarse/mid
                # flows disagree with fine flow (catches repetitive texture ambiguity)
                if self.multiscale_consistency:
                    fH, fW = self.FINE_HW
                    cH, cW = self.COARSE_HW
                    mH, mW = self.MID_HW

                    # Upsample coarse flow to fine resolution, rescaling pixel values
                    flow_c_up = F.interpolate(
                        flow_c.float(), size=(fH, fW),
                        mode='bilinear', align_corners=False)
                    flow_c_up[:, 0] *= fW / cW
                    flow_c_up[:, 1] *= fH / cH

                    # Upsample mid flow to fine resolution
                    flow_m_up = F.interpolate(
                        flow_m.float(), size=(fH, fW),
                        mode='bilinear', align_corners=False)
                    flow_m_up[:, 0] *= fW / mW
                    flow_m_up[:, 1] *= fH / mH

                    # Compute cross-scale disagreement (L2 distance)
                    sigma2 = self.ms_consistency_sigma ** 2
                    diff_cf = (flow_f32 - flow_c_up).pow(2).sum(1, keepdim=True)
                    diff_mf = (flow_f32 - flow_m_up).pow(2).sum(1, keepdim=True)

                    # Consistency weight: high where all scales agree
                    consistency = torch.exp(-(diff_cf + diff_mf) / (2.0 * sigma2))
                    conf_f32 = conf_f32 * consistency

                # Resize depth to match flow resolution if they differ
                # (e.g., when fine_hw differs from renderer depth_res)
                if depth_f32.shape[-2:] != flow_f32.shape[-2:]:
                    need_squeeze = (depth_f32.ndim == 3)
                    if need_squeeze:
                        depth_f32 = depth_f32.unsqueeze(1)
                    depth_f32 = F.interpolate(
                        depth_f32, size=flow_f32.shape[-2:],
                        mode='bilinear', align_corners=False)
                    if need_squeeze:
                        depth_f32 = depth_f32.squeeze(1)

                # Upsample flow/depth/conf for finer geometry solving
                s = self.geometry_upsample
                if s > 1:
                    tgt_size = (flow_f32.shape[-2] * s, flow_f32.shape[-1] * s)
                    flow_f32 = F.interpolate(
                        flow_f32, size=tgt_size,
                        mode='bilinear', align_corners=False) * s
                    conf_f32 = F.interpolate(
                        conf_f32, size=tgt_size,
                        mode='bilinear', align_corners=False)
                    need_squeeze = (depth_f32.ndim == 3)
                    if need_squeeze:
                        depth_f32 = depth_f32.unsqueeze(1)
                    depth_f32 = F.interpolate(
                        depth_f32, size=tgt_size,
                        mode='bilinear', align_corners=False)
                    if need_squeeze:
                        depth_f32 = depth_f32.squeeze(1)

                Ju, Jv, valid = compute_image_jacobian(
                    depth_f32, self.geo_intrinsics)

                delta_xi = diff_pose_solve(
                    flow_f32, conf_f32, Ju, Jv, valid,
                    damping=self.damping,
                    irls_iters=self.irls_iters,
                    irls_huber_k=self.irls_huber_k,
                    pixel_stride=self.pixel_stride,
                    adaptive_damping=self.adaptive_damping,
                    adaptive_damping_max=self.adaptive_damping_max,
                    adaptive_damping_cond_thresh=self.adaptive_damping_cond_thresh,
                )

                # Learnable pose refinement (if enabled)
                if self.use_pose_refinement:
                    delta_xi = self.pose_refine_head(
                        delta_xi, h_fine.float())

            result['delta_xi'] = delta_xi

        return result

    def compute_gt_flow(
        self,
        pose_init: torch.Tensor,
        pose_gt: torch.Tensor,
        depth: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据位姿差异和深度, 计算 GT flow (用于 flow supervision).

        将 init 位姿下的 3D 点投影到 GT 视角, 得到像素位移.
        同时返回有效性 mask: 排除 depth 无效、Z_gt 太小或投影出界的像素.

        Args:
            pose_init: (B, 4, 4) 初始 w2c
            pose_gt:   (B, 4, 4) GT w2c
            depth:     (B, H, W) 在 init 视角下的深度
            resolution: (tH, tW) 目标 flow 分辨率

        Returns:
            flow_gt: (B, 2, tH, tW) GT 像素位移 (无效像素设为 0)
            valid:   (B, 1, tH, tW) 有效像素 mask (0/1 float)
        """
        B, H, W = depth.shape
        device = depth.device
        intrinsics = self._scale_intrinsics(H, W)
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        # 像素坐标 → 3D (init camera frame)
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        u_coords = u_coords.unsqueeze(0).expand(B, -1, -1)
        v_coords = v_coords.unsqueeze(0).expand(B, -1, -1)

        X = (u_coords - cx) / fx * depth
        Y = (v_coords - cy) / fy * depth
        Z = depth

        pts_init = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)  # (B, H, W, 4)
        pts_flat = pts_init.reshape(B, -1, 4).permute(0, 2, 1)  # (B, 4, N)

        # init_cam → world → gt_cam
        T_rel = pose_gt @ torch.linalg.inv(pose_init)  # (B, 4, 4)
        pts_gt = torch.bmm(T_rel[:, :3, :], pts_flat)  # (B, 3, N)
        pts_gt = pts_gt.reshape(B, 3, H, W)

        # Project
        Z_gt_raw = pts_gt[:, 2:3]
        Z_gt = Z_gt_raw.clamp(min=0.01)
        u_gt = fx * pts_gt[:, 0:1] / Z_gt + cx
        v_gt = fy * pts_gt[:, 1:2] / Z_gt + cy

        flow_gt = torch.cat([u_gt - u_coords.unsqueeze(1),
                              v_gt - v_coords.unsqueeze(1)], dim=1)  # (B, 2, H, W)

        # Validity mask: depth > 0.05, Z_gt > 0.1, projected coords within image bounds
        valid = (
            (depth.unsqueeze(1) > 0.05) &           # 原始深度有效
            (Z_gt_raw > 0.1) &                       # GT 视角下在相机前方
            (u_gt > -0.5) & (u_gt < W - 0.5) &      # 投影在图像内
            (v_gt > -0.5) & (v_gt < H - 0.5)
        ).float()  # (B, 1, H, W)

        # 将无效像素的 flow 设为 0 (避免 NaN/Inf 传播)
        flow_gt = flow_gt * valid

        # Resize to target
        tH, tW = resolution
        if tH != H or tW != W:
            scale_x = tW / W
            scale_y = tH / H
            flow_gt = F.interpolate(flow_gt, size=(tH, tW), mode='bilinear',
                                     align_corners=False)
            flow_gt[:, 0] *= scale_x
            flow_gt[:, 1] *= scale_y
            # mask 用 nearest 以避免边缘模糊
            valid = F.interpolate(valid, size=(tH, tW), mode='nearest')

        return flow_gt, valid
