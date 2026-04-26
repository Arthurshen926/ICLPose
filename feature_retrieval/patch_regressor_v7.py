#!/usr/bin/env python3
"""
Patch Token Pose Regression (exp14 series)
==========================================
Uses RADIO patch tokens (64d PCA, 68×120 spatial) for pose regression.
Supports multiple pooling strategies and feature combinations.

Pooling modes:
  gap       - Global Average Pooling → 64d per feature type
  gem       - Generalized Mean Pooling (learnable p) → 64d
  spp       - Spatial Pyramid Pooling (1×1, 2×2, 4×4) → 64×21 = 1344d
  attn      - Attention Pooling (learned spatial weights) → 64d × num_heads
  conv      - Conv encoder to reduce spatial dims → flatten

Feature modes:
  fine      - fine_geo only
  coarse    - coarse_sem only
  both      - concat fine_geo + coarse_sem (after pooling)
  both+sum  - concat fine_geo + coarse_sem + summary token

Usage:
  python feature_retrieval/patch_regressor_v7.py --pool gap --feat fine --exp_name exp14a_gap_fine
  python feature_retrieval/patch_regressor_v7.py --pool spp --feat both+sum --exp_name exp14f_spp_both_sum
"""

import argparse
import glob as glob_module
import json
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_field.utils.loc_reporting import save_experiment_bundle


# ============================================================
# Utility: Rotation representations
# ============================================================

def quaternion_to_matrix(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(-1)
    B = q.shape[:-1]
    mat = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*B, 3, 3)
    return mat


def gram_schmidt_6d_to_matrix(v):
    a1 = v[..., :3]
    a2 = v[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def geodesic_distance(R1, R2):
    Rd = torch.bmm(R1.reshape(-1, 3, 3), R2.reshape(-1, 3, 3).transpose(-1, -2))
    trace = Rd[:, 0, 0] + Rd[:, 1, 1] + Rd[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle).reshape(R1.shape[:-2])


def center_rot_to_w2c(center, rot_w2c):
    pose = torch.eye(4, device=center.device, dtype=center.dtype).unsqueeze(0).repeat(center.shape[0], 1, 1)
    pose[:, :3, :3] = rot_w2c
    pose[:, :3, 3] = -torch.bmm(rot_w2c, center.unsqueeze(-1)).squeeze(-1)
    return pose


# ============================================================
# Loss functions
# ============================================================

def log_cosh_loss(pred, target):
    """Log-cosh loss: smooth approximation of L1, differentiable everywhere."""
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff)))


def wing_loss(pred, target, w=10.0, eps=2.0):
    """Wing loss: better for small errors (sub-meter), log for large errors.
    w controls the non-linear range, eps controls curvature.
    """
    diff = (pred - target).abs()
    C = w - w * math.log(1 + w / eps)
    loss = torch.where(
        diff < w,
        w * torch.log(1 + diff / eps),
        diff - C
    )
    return loss.mean()


TRANS_LOSS_REGISTRY = {
    'smooth_l1': F.smooth_l1_loss,
    'log_cosh': log_cosh_loss,
    'wing': wing_loss,
    'mse': F.mse_loss,
}


# ============================================================
# Pooling modules
# ============================================================

class GAPPooling(nn.Module):
    """Global Average Pooling: (B, C, H, W) → (B, C)"""
    def __init__(self, C, **kwargs):
        super().__init__()
        self.output_dim = C
    
    def forward(self, x):
        return x.mean(dim=(-2, -1))  # (B, C)


class GeMPooling(nn.Module):
    """Generalized Mean Pooling: (B, C, H, W) → (B, C)
    GeM(x, p) = (mean(x^p))^(1/p), p is learnable.
    """
    def __init__(self, C, p_init=3.0, eps=1e-6, **kwargs):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps
        self.output_dim = C
    
    def forward(self, x):
        # x: (B, C, H, W), clamp to avoid negative values for fractional powers
        x_clamped = x.clamp(min=self.eps)
        return (x_clamped.pow(self.p).mean(dim=(-2, -1))).pow(1.0 / self.p)  # (B, C)


class SPPPooling(nn.Module):
    """Spatial Pyramid Pooling: (B, C, H, W) → (B, C * sum(level^2))
    Levels: [1, 2, 4] → 1+4+16 = 21 regions → C*21
    """
    def __init__(self, C, levels=(1, 2, 4), spp_levels=None, **kwargs):
        super().__init__()
        if spp_levels is not None:
            self.levels = spp_levels
        else:
            self.levels = levels
        self.output_dim = C * sum(l * l for l in self.levels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        pooled = []
        for level in self.levels:
            # Use adaptive average pooling to get level×level spatial grid
            out = F.adaptive_avg_pool2d(x, (level, level))  # (B, C, level, level)
            pooled.append(out.reshape(B, -1))  # (B, C*level*level)
        return torch.cat(pooled, dim=-1)  # (B, C * sum(l^2))


class AttentionPooling(nn.Module):
    """Attention Pooling: learns spatial attention weights.
    (B, C, H, W) → (B, C * num_heads) or (B, C) if num_heads=1.
    """
    def __init__(self, C, num_heads=4, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attn_conv = nn.Sequential(
            nn.Conv2d(C, C // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, num_heads, 1),
        )
        self.output_dim = C * num_heads
    
    def forward(self, x):
        B, C, H, W = x.shape
        # Compute attention maps: (B, num_heads, H, W)
        attn = self.attn_conv(x)
        attn = attn.reshape(B, self.num_heads, H * W)
        attn = F.softmax(attn, dim=-1)  # (B, num_heads, H*W)
        
        # Weighted average
        x_flat = x.reshape(B, C, H * W)  # (B, C, H*W)
        # For each head, compute weighted avg: (B, C)
        out = torch.bmm(x_flat, attn.transpose(1, 2))  # (B, C, num_heads)
        return out.reshape(B, -1)  # (B, C * num_heads)


class ConvEncoder(nn.Module):
    """Conv encoder to reduce spatial dims: (B, C, 68, 120) → (B, out_dim)"""
    def __init__(self, C, out_dim=512, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 128, 3, stride=2, padding=1),  # → 128×34×60
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # → 256×17×30
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # → 256×9×15
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 4)),  # → 256×2×4
        )
        self.fc = nn.Linear(256 * 2 * 4, out_dim)
        self.output_dim = out_dim
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.reshape(h.shape[0], -1)
        return self.fc(h)


POOL_REGISTRY = {
    'gap': GAPPooling,
    'gem': GeMPooling,
    'spp': SPPPooling,
    'attn': AttentionPooling,
    'conv': ConvEncoder,
}


# ============================================================
# Dataset
# ============================================================

class PatchPoseDataset:
    """Dataset with patch tokens and optional summary tokens."""
    
    def __init__(self, feature_dir, dataset_dir, split='train', device='cpu',
                 use_fine=True, use_coarse=True, use_summary=True):
        self.device = device
        
        # Load features based on configuration
        self.patch_fine = None
        self.patch_coarse = None
        self.summary = None
        
        if use_fine:
            path = os.path.join(feature_dir, f'fine_geo_{split}.pt')
            self.patch_fine = torch.load(path, map_location='cpu').float()
            print(f"  Loaded fine_geo: {self.patch_fine.shape}")
        
        if use_coarse:
            path = os.path.join(feature_dir, f'coarse_sem_{split}.pt')
            self.patch_coarse = torch.load(path, map_location='cpu').float()
            print(f"  Loaded coarse_sem: {self.patch_coarse.shape}")
        
        if use_summary:
            summary_matrix = torch.load(
                os.path.join(feature_dir, 'summary_matrix.pt'), map_location='cpu').float()
            # Need to index by split
            image_paths = sorted(glob_module.glob(os.path.join(dataset_dir, 'seq*/*.png')))
            name_to_idx = {os.path.relpath(p, dataset_dir): i for i, p in enumerate(image_paths)}
            
            split_file = os.path.join(dataset_dir, f'dataset_{split}.txt')
            with open(split_file) as f:
                lines = f.readlines()
            
            indices = []
            for line in lines[3:]:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                img_name = parts[0]
                idx = name_to_idx.get(img_name)
                if idx is not None:
                    indices.append(idx)
            
            self.summary = summary_matrix[indices]
            print(f"  Loaded summary: {self.summary.shape}")
        
        # Load poses
        split_file = os.path.join(dataset_dir, f'dataset_{split}.txt')
        with open(split_file) as f:
            lines = f.readlines()
        
        translations_list = []
        rotations_list = []
        names_list = []
        
        for line in lines[3:]:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            img_name = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            w, p, q, r = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            translations_list.append(torch.tensor([x, y, z], dtype=torch.float32))
            rotations_list.append(torch.tensor([w, p, q, r], dtype=torch.float32))
            names_list.append(img_name)
        
        self.translations = torch.stack(translations_list)
        quat = torch.stack(rotations_list)
        self.rotations = quaternion_to_matrix(quat)
        self.names = names_list
        self.N = len(names_list)
        
        # Verify shapes match
        if self.patch_fine is not None:
            assert self.patch_fine.shape[0] == self.N, \
                f"fine_geo has {self.patch_fine.shape[0]} samples but expected {self.N}"
        if self.patch_coarse is not None:
            assert self.patch_coarse.shape[0] == self.N
        if self.summary is not None:
            assert self.summary.shape[0] == self.N
        
        print(f"  [{split}] {self.N} samples loaded")
    
    def to(self, device):
        """Move all data to device."""
        self.device = device
        self.translations = self.translations.to(device)
        self.rotations = self.rotations.to(device)
        if self.patch_fine is not None:
            self.patch_fine = self.patch_fine.to(device)
        if self.patch_coarse is not None:
            self.patch_coarse = self.patch_coarse.to(device)
        if self.summary is not None:
            self.summary = self.summary.to(device)
        return self
    
    def compute_normalization(self):
        self.trans_mean = self.translations.mean(dim=0)
        self.trans_std = self.translations.std(dim=0).clamp(min=1e-6)
        return self.trans_mean, self.trans_std
    
    def normalize_translations(self, mean, std):
        self.translations_norm = (self.translations - mean) / std
    
    def denormalize(self, trans_norm, mean, std):
        return trans_norm * std + mean


# ============================================================
# Model
# ============================================================

class PatchPoseRegressor(nn.Module):
    """Patch token pose regressor with configurable pooling and feature combinations."""
    
    def __init__(self, pool_type='gap', feat_mode='both+sum',
                 patch_dim=64, hidden_dims=(1024, 512, 256), dropout=0.1,
                 attn_heads=4, conv_out_dim=512, spp_levels=None):
        super().__init__()
        self.feat_mode = feat_mode
        self.pool_type = pool_type
        
        # Build pooling modules
        pool_cls = POOL_REGISTRY[pool_type]
        pool_kwargs = {'num_heads': attn_heads, 'out_dim': conv_out_dim,
                       'spp_levels': tuple(spp_levels) if spp_levels else None}
        
        self.pool_fine = None
        self.pool_coarse = None
        
        use_fine = 'fine' in feat_mode or 'both' in feat_mode
        use_coarse = 'coarse' in feat_mode or 'both' in feat_mode
        
        if use_fine:
            self.pool_fine = pool_cls(patch_dim, **pool_kwargs)
        if use_coarse:
            self.pool_coarse = pool_cls(patch_dim, **pool_kwargs)
        
        # Compute total input dimension
        total_dim = 0
        if use_fine:
            total_dim += self.pool_fine.output_dim
        if use_coarse:
            total_dim += self.pool_coarse.output_dim
        if '+sum' in feat_mode:
            total_dim += 2560  # summary token
        
        self.total_dim = total_dim
        print(f"  Feature dim after pooling: {total_dim}")
        
        # Optional batch normalization on concatenated features
        self.input_norm = nn.LayerNorm(total_dim)
        
        # MLP backbone
        layers = []
        in_dim = total_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        self.backbone = nn.Sequential(*layers)
        self.trans_head = nn.Linear(hidden_dims[-1], 3)
        self.rot_head = nn.Linear(hidden_dims[-1], 6)
        self.s_rot = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def pool_features(self, patch_fine=None, patch_coarse=None, summary=None):
        """Pool spatial features and concatenate into a single vector.
        Returns: (B, total_dim) pooled feature vector.
        """
        parts = []
        if self.pool_fine is not None and patch_fine is not None:
            parts.append(self.pool_fine(patch_fine))
        if self.pool_coarse is not None and patch_coarse is not None:
            parts.append(self.pool_coarse(patch_coarse))
        if '+sum' in self.feat_mode and summary is not None:
            parts.append(summary)
        return torch.cat(parts, dim=-1)
    
    def forward_from_pooled(self, pooled):
        """Forward pass from pre-pooled features. (B, total_dim) -> trans, rot_mat."""
        x = self.input_norm(pooled)
        h = self.backbone(x)
        trans = self.trans_head(h)
        rot_6d = self.rot_head(h)
        rot_mat = gram_schmidt_6d_to_matrix(rot_6d)
        return trans, rot_mat

    def forward(self, patch_fine=None, patch_coarse=None, summary=None):
        """
        Args:
            patch_fine: (B, 64, 68, 120) or None
            patch_coarse: (B, 64, 68, 120) or None
            summary: (B, 2560) or None
        Returns:
            trans: (B, 3), rot_mat: (B, 3, 3)
        """
        pooled = self.pool_features(patch_fine, patch_coarse, summary)
        return self.forward_from_pooled(pooled)


class ResidualBlock(nn.Module):
    """Residual block with pre-norm: LN -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.net(self.norm(x))


class ResidualPatchPoseRegressor(PatchPoseRegressor):
    """PatchPoseRegressor with residual MLP backbone.
    
    Architecture: input_proj -> N residual blocks -> output heads
    The input projection maps total_dim to hidden_dim, then residual blocks
    refine the representation with skip connections.
    """
    
    def __init__(self, pool_type='gap', feat_mode='both+sum',
                 patch_dim=64, hidden_dims=(1024, 512, 256), dropout=0.1,
                 attn_heads=4, conv_out_dim=512, spp_levels=None,
                 n_res_blocks=4):
        # Call grandparent __init__ to skip PatchPoseRegressor's MLP building
        nn.Module.__init__(self)
        self.feat_mode = feat_mode
        self.pool_type = pool_type
        
        # Build pooling (same as parent)
        pool_cls = POOL_REGISTRY[pool_type]
        pool_kwargs = {'num_heads': attn_heads, 'out_dim': conv_out_dim,
                       'spp_levels': tuple(spp_levels) if spp_levels else None}
        
        self.pool_fine = None
        self.pool_coarse = None
        
        use_fine = 'fine' in feat_mode or 'both' in feat_mode
        use_coarse = 'coarse' in feat_mode or 'both' in feat_mode
        
        if use_fine:
            self.pool_fine = pool_cls(patch_dim, **pool_kwargs)
        if use_coarse:
            self.pool_coarse = pool_cls(patch_dim, **pool_kwargs)
        
        total_dim = 0
        if use_fine:
            total_dim += self.pool_fine.output_dim
        if use_coarse:
            total_dim += self.pool_coarse.output_dim
        if '+sum' in feat_mode:
            total_dim += 2560
        
        self.total_dim = total_dim
        hidden_dim = hidden_dims[0]  # Use first hidden_dim as the residual dim
        print(f"  Feature dim after pooling: {total_dim}")
        print(f"  Residual backbone: {total_dim} -> {hidden_dim} x {n_res_blocks} blocks")
        
        self.input_norm = nn.LayerNorm(total_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(n_res_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        self.trans_head = nn.Linear(hidden_dim, 3)
        self.rot_head = nn.Linear(hidden_dim, 6)
        self.s_rot = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def forward_from_pooled(self, pooled):
        x = self.input_norm(pooled)
        h = self.input_proj(x)
        h = self.res_blocks(h)
        h = self.final_norm(h)
        trans = self.trans_head(h)
        rot_6d = self.rot_head(h)
        rot_mat = gram_schmidt_6d_to_matrix(rot_6d)
        return trans, rot_mat
    
    def forward(self, patch_fine=None, patch_coarse=None, summary=None):
        pooled = self.pool_features(patch_fine, patch_coarse, summary)
        return self.forward_from_pooled(pooled)


# ============================================================
# Training
# ============================================================

def precompute_pooled_features(model, dataset, device, batch_size=32):
    """Pre-compute pooled features for all samples in mini-batches.
    
    This avoids OOM by processing raw patches in small batches,
    only keeping the compact pooled vectors (total_dim) on GPU.
    
    Returns: (N, total_dim) tensor on device.
    """
    model.eval()
    N = dataset.N
    pooled_list = []
    
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            pf = dataset.patch_fine[start:end].to(device) if dataset.patch_fine is not None else None
            pc = dataset.patch_coarse[start:end].to(device) if dataset.patch_coarse is not None else None
            sm = dataset.summary[start:end].to(device) if dataset.summary is not None else None
            
            pooled = model.pool_features(pf, pc, sm)
            pooled_list.append(pooled.cpu())
            
            # Free GPU memory
            del pf, pc, sm, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    result = torch.cat(pooled_list, dim=0).to(device)
    print(f"  Pre-computed pooled features: {result.shape} ({result.numel() * 4 / 1e6:.1f} MB)")
    return result


def train(args):
    # Set random seed if specified
    if getattr(args, 'seed', None) is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse feat mode
    use_fine = 'fine' in args.feat or 'both' in args.feat
    use_coarse = 'coarse' in args.feat or 'both' in args.feat
    use_summary = '+sum' in args.feat
    
    print(f"\nLoading data (fine={use_fine}, coarse={use_coarse}, summary={use_summary})...")
    train_data = PatchPoseDataset(
        args.feature_dir, args.dataset_dir, 'train', 'cpu',
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    test_data = PatchPoseDataset(
        args.feature_dir, args.dataset_dir, 'test', 'cpu',
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    
    # Move poses to device (features moved later depending on precompute mode)
    train_data.translations = train_data.translations.to(device)
    train_data.rotations = train_data.rotations.to(device)
    test_data.translations = test_data.translations.to(device)
    test_data.rotations = test_data.rotations.to(device)
    
    # Normalize translations
    trans_mean, trans_std = train_data.compute_normalization()
    train_data.normalize_translations(trans_mean, trans_std)
    test_data.normalize_translations(trans_mean, trans_std)
    print(f"Translation norm: mean={trans_mean.cpu().numpy()}, std={trans_std.cpu().numpy()}")
    
    torch.save({'mean': trans_mean, 'std': trans_std}, out_dir / 'norm_params.pt')
    
    # Model
    model_type = getattr(args, 'model_type', 'mlp')
    if model_type == 'residual':
        model = ResidualPatchPoseRegressor(
            pool_type=args.pool,
            feat_mode=args.feat,
            patch_dim=args.patch_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            attn_heads=args.attn_heads,
            conv_out_dim=args.conv_out_dim,
            spp_levels=args.spp_levels,
            n_res_blocks=getattr(args, 'n_res_blocks', 4),
        ).to(device)
    else:
        model = PatchPoseRegressor(
            pool_type=args.pool,
            feat_mode=args.feat,
            patch_dim=args.patch_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            attn_heads=args.attn_heads,
            conv_out_dim=args.conv_out_dim,
            spp_levels=args.spp_levels,
        ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Pre-compute pooled features if requested (avoids OOM for large patches)
    use_precompute = getattr(args, 'precompute_pool', False)
    train_pooled = None
    test_pooled = None
    
    if use_precompute:
        print("\nPre-computing pooled features (OOM-safe mode)...")
        # Keep raw features on CPU, pool in mini-batches
        train_pooled = precompute_pooled_features(model, train_data, device, batch_size=16)
        test_pooled = precompute_pooled_features(model, test_data, device, batch_size=16)
        # Free raw patch features from memory (keep summary for retrieval eval)
        train_data.patch_fine = None
        train_data.patch_coarse = None
        test_data.patch_fine = None
        test_data.patch_coarse = None
        if train_data.summary is not None:
            train_data.summary = train_data.summary.to(device)
        if test_data.summary is not None:
            test_data.summary = test_data.summary.to(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  GPU memory freed. Training with pre-pooled features ({train_pooled.shape[1]}d)")
    
    # Feature whitening (PCA whitening on pooled features)
    whiten_transform = None
    if getattr(args, 'whiten', False) and train_pooled is not None:
        print("\nApplying PCA whitening to pooled features...")
        N, D = train_pooled.shape
        # With N=895 and D=7936, cov matrix is rank-deficient (rank<=N-1=894)
        # Use SVD on centered data directly (more numerically stable for rank-deficient case)
        feat_mean = train_pooled.mean(dim=0, keepdim=True)  # (1, D)
        centered = train_pooled - feat_mean  # (N, D)
        
        # SVD: centered = U @ S @ V^T, where V columns are eigenvectors of cov
        # Only keep top-k components where k = min(N-1, D)
        k = min(N - 1, D)
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # S are singular values, eigenvalues = S^2 / (N-1)
        eigenvalues = (S[:k] ** 2) / (N - 1)
        eigenvectors = Vh[:k].T  # (D, k) — top-k principal components
        
        # Only keep components with significant eigenvalue (>1% of max)
        eig_threshold = eigenvalues[0] * 0.001
        n_keep = (eigenvalues > eig_threshold).sum().item()
        n_keep = max(n_keep, 128)  # Keep at least 128 components
        n_keep = min(n_keep, k)
        print(f"  Keeping {n_keep}/{D} components (threshold={eig_threshold:.4f})")
        
        eigenvalues = eigenvalues[:n_keep]
        eigenvectors = eigenvectors[:, :n_keep]
        
        # Whitening matrix: project onto PCs then scale by 1/sqrt(eigenvalue)
        # W = V_k @ diag(1/sqrt(lambda_k))  →  output is (N, n_keep)
        eps_whiten = 1e-5
        whitening_matrix = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues + eps_whiten))
        
        # Apply whitening (reduces dimensionality from D to n_keep)
        train_pooled_w = (train_pooled - feat_mean) @ whitening_matrix  # (N, n_keep)
        test_pooled_w = (test_pooled - feat_mean) @ whitening_matrix
        
        whiten_transform = {'mean': feat_mean.cpu(), 'matrix': whitening_matrix.cpu()}
        torch.save(whiten_transform, out_dir / 'whiten_transform.pt')
        
        # Report statistics
        total_var = eigenvalues.sum()
        var_ratio = eigenvalues / total_var
        eff_rank = torch.exp(-torch.sum(var_ratio * torch.log(var_ratio + 1e-10))).item()
        print(f"  Effective rank: {eff_rank:.1f}/{n_keep}")
        print(f"  Top-10 eigenvalue ratio: {var_ratio[:10].sum().item()*100:.1f}%")
        print(f"  Whitened features: dim={train_pooled_w.shape[1]}, mean={train_pooled_w.mean().item():.4f}, std={train_pooled_w.std().item():.4f}")
        
        # Replace pooled features with whitened ones
        # Build a simple MLP directly (bypass pool_features since already pooled+whitened)
        new_input_dim = n_keep
        print(f"  Rebuilding model with input_dim={new_input_dim} (was {D})")
        
        # Build a standalone MLP model for whitened features
        hidden_dims = args.hidden_dims
        layers = []
        in_dim = new_input_dim
        layer_norm_input = nn.LayerNorm(new_input_dim)
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(args.dropout),
            ])
            in_dim = h_dim
        
        class WhitenedPoseRegressor(nn.Module):
            def __init__(self, input_norm, backbone, hidden_dim):
                super().__init__()
                self.input_norm = input_norm
                self.backbone = backbone
                self.trans_head = nn.Linear(hidden_dim, 3)
                self.rot_head = nn.Linear(hidden_dim, 6)
                self.s_rot = nn.Parameter(torch.zeros(1))
                # Initialize weights
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                # Feature dropout placeholder  
                self.feature_dropout = nn.Dropout(getattr(args, 'feature_dropout', 0.0))
                self.total_dim = new_input_dim
            
            def forward_from_pooled(self, x):
                x = self.feature_dropout(x)
                x = self.input_norm(x)
                h = self.backbone(x)
                trans = self.trans_head(h)
                rot_6d = self.rot_head(h)
                from feature_retrieval.patch_regressor_v7 import gram_schmidt_6d_to_matrix
                rot = gram_schmidt_6d_to_matrix(rot_6d)
                return trans, rot
            
            def forward(self, patch_fine=None, patch_coarse=None, summary=None):
                raise RuntimeError("WhitenedPoseRegressor only supports forward_from_pooled")
            
            def pool_features(self, patch_fine=None, patch_coarse=None, summary=None):
                raise RuntimeError("WhitenedPoseRegressor uses pre-whitened features")
        
        new_model = WhitenedPoseRegressor(
            layer_norm_input, nn.Sequential(*layers), hidden_dims[-1]
        )
        new_model.to(device)
        n_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
        print(f"  New model parameters: {n_params:,}")
        
        # Replace model and pooled features
        model = new_model
        train_pooled = train_pooled_w
        test_pooled = test_pooled_w
    
    else:
        # Original path: move everything to GPU
        if train_data.patch_fine is not None:
            train_data.patch_fine = train_data.patch_fine.to(device)
        if train_data.patch_coarse is not None:
            train_data.patch_coarse = train_data.patch_coarse.to(device)
        if test_data.patch_fine is not None:
            test_data.patch_fine = test_data.patch_fine.to(device)
        if test_data.patch_coarse is not None:
            test_data.patch_coarse = test_data.patch_coarse.to(device)
        if train_data.summary is not None:
            train_data.summary = train_data.summary.to(device)
        if test_data.summary is not None:
            test_data.summary = test_data.summary.to(device)
    print(f"Model parameters: {n_params:,}")
    
    # Pre-compute spatial augmentation neighbors
    spatial_nn_indices = None
    spatial_aug_k = getattr(args, 'spatial_aug_k', 0)
    if spatial_aug_k > 0 and use_precompute:
        from scipy.spatial import cKDTree
        train_pos_np = train_data.translations.cpu().numpy()
        tree = cKDTree(train_pos_np)
        max_dist = getattr(args, 'spatial_aug_dist', 5.0)
        # Query k+1 nearest neighbors (first is self)
        dists, nn_idx = tree.query(train_pos_np, k=spatial_aug_k + 1)
        # Remove self (first column) and filter by distance
        nn_idx = nn_idx[:, 1:]  # (N, k)
        nn_dists = dists[:, 1:]  # (N, k)
        # Mask neighbors beyond max_dist
        nn_mask = nn_dists < max_dist  # (N, k) bool
        spatial_nn_indices = torch.tensor(nn_idx, device=device, dtype=torch.long)
        spatial_nn_mask = torch.tensor(nn_mask, device=device, dtype=torch.bool)
        avg_valid_neighbors = nn_mask.sum(axis=1).mean()
        print(f"\nSpatial augmentation: k={spatial_aug_k}, max_dist={max_dist}m")
        print(f"  Avg valid neighbors per sample: {avg_valid_neighbors:.1f}")
        print(f"  Samples with 0 valid neighbors: {(nn_mask.sum(axis=1) == 0).sum()}")
    
    # Save config
    config = vars(args)
    config['n_params'] = n_params
    config['total_feature_dim'] = model.total_dim
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # For mini-batch training (patch tokens are large)
    batch_size = args.batch_size
    n_train = train_data.N
    
    history = {
        'epoch': [], 'loss': [], 'loss_trans': [], 'loss_rot': [],
        'val_rot_median': [], 'val_trans_median': [], 'beta_rot': [],
    }
    best_val_score = float('inf')
    best_epoch = 0
    
    # SWA state
    swa_state = None
    swa_count = 0
    swa_start = getattr(args, 'swa_start', 0)
    swa_freq = getattr(args, 'swa_freq', 50)
    
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        if batch_size >= n_train:
            # Full batch
            indices = torch.arange(n_train, device=device)
        else:
            # Mini-batch: random sample
            indices = torch.randperm(n_train, device=device)[:batch_size]
        
        # Get batch
        target_trans = train_data.translations_norm[indices]
        target_rot = train_data.rotations[indices]
        
        # Translation label noise (regularizer)
        trans_noise_std = getattr(args, 'trans_noise', 0.0)
        if trans_noise_std > 0:
            # Noise in normalized space: divide by trans_std to match normalization
            noise = torch.randn_like(target_trans) * trans_noise_std / trans_std
            target_trans = target_trans + noise
        
        if use_precompute:
            # Pre-computed path: just index into pooled features
            pooled_batch = train_pooled[indices]
            
            # Spatial interpolation augmentation
            if spatial_aug_k > 0 and spatial_nn_indices is not None:
                aug_prob = getattr(args, 'spatial_aug_prob', 0.5)
                aug_alpha_max = getattr(args, 'spatial_aug_alpha', 0.3)
                B = len(indices)
                
                # Decide which samples to augment
                aug_mask = torch.rand(B, device=device) < aug_prob
                n_aug = aug_mask.sum().item()
                
                if n_aug > 0:
                    aug_indices = indices[aug_mask]  # original dataset indices
                    # For each augmented sample, pick a random valid neighbor
                    nn_for_aug = spatial_nn_indices[aug_indices]  # (n_aug, k)
                    mask_for_aug = spatial_nn_mask[aug_indices]   # (n_aug, k) bool
                    
                    # Random neighbor selection: pick uniformly from valid neighbors
                    # For samples with no valid neighbors, skip augmentation
                    rand_k = torch.randint(0, spatial_aug_k, (n_aug,), device=device)
                    neighbor_idx = nn_for_aug[torch.arange(n_aug, device=device), rand_k]
                    valid = mask_for_aug[torch.arange(n_aug, device=device), rand_k]
                    
                    if valid.any():
                        # Get neighbor features and targets
                        valid_aug_local = aug_mask.clone()
                        # Map back: only augment where we have valid neighbors
                        valid_global = torch.zeros(B, dtype=torch.bool, device=device)
                        valid_global[aug_mask] = valid
                        
                        valid_neighbor_idx = neighbor_idx[valid]
                        neighbor_pooled = train_pooled[valid_neighbor_idx]
                        neighbor_trans = train_data.translations_norm[valid_neighbor_idx]
                        neighbor_rot = train_data.rotations[valid_neighbor_idx]
                        
                        # Random interpolation weight
                        alpha = torch.rand(valid.sum().item(), 1, device=device) * aug_alpha_max
                        
                        # Interpolate features
                        pooled_batch[valid_global] = (1 - alpha) * pooled_batch[valid_global] + \
                                                      alpha * neighbor_pooled
                        
                        # Interpolate translation targets
                        alpha_t = alpha  # same weight for translation
                        target_trans[valid_global] = (1 - alpha_t) * target_trans[valid_global] + \
                                                      alpha_t * neighbor_trans
                        
                        # Interpolate rotation via SVD projection
                        alpha_r = alpha.unsqueeze(-1)  # (n_valid, 1, 1)
                        orig_rot = target_rot[valid_global]
                        interp_rot_raw = (1 - alpha_r) * orig_rot + alpha_r * neighbor_rot
                        U, S, Vh = torch.linalg.svd(interp_rot_raw)
                        interp_rot = torch.bmm(U, Vh)
                        det = torch.det(interp_rot)
                        fix_mat = torch.ones_like(Vh)
                        fix_mat[:, -1, :] *= det.sign().unsqueeze(-1)
                        target_rot[valid_global] = torch.bmm(U, fix_mat * Vh)
            
            # Feature dropout on SUMMARY portion only (matches non-precompute path)
            # Pooled layout: [SPP_fine | SPP_coarse | summary]
            # Only the summary (last summary_dim dimensions) should have dropout
            if args.feature_dropout > 0:
                summary_dim = train_data.summary.shape[1] if train_data.summary is not None else 0
                if summary_dim > 0 and '+sum' in args.feat:
                    spp_part = pooled_batch[:, :-summary_dim]
                    sum_part = F.dropout(pooled_batch[:, -summary_dim:],
                                         p=args.feature_dropout, training=True)
                    pooled_batch = torch.cat([spp_part, sum_part], dim=-1)
                # else: no summary in pooled, skip dropout entirely (matches non-precompute)
            
            trans_pred, rot_pred = model.forward_from_pooled(pooled_batch)
        else:
            # Original path: raw patches -> pool -> MLP
            pf = train_data.patch_fine[indices] if train_data.patch_fine is not None else None
            pc = train_data.patch_coarse[indices] if train_data.patch_coarse is not None else None
            sm = train_data.summary[indices] if train_data.summary is not None else None
            
            # Mixup augmentation: interpolate between pairs of samples
            if getattr(args, 'mixup_alpha', 0) > 0:
                lam = torch.distributions.Beta(args.mixup_alpha, args.mixup_alpha).sample(
                    (len(indices),)).to(device)
                perm = torch.randperm(len(indices), device=device)
                lam_feat = lam.reshape(-1, 1, 1, 1)
                lam_vec = lam.reshape(-1, 1)
                if pf is not None:
                    pf = lam_feat * pf + (1 - lam_feat) * pf[perm]
                if pc is not None:
                    pc = lam_feat * pc + (1 - lam_feat) * pc[perm]
                if sm is not None:
                    sm = lam.unsqueeze(-1) * sm + (1 - lam.unsqueeze(-1)) * sm[perm]
                target_trans = lam_vec * target_trans + (1 - lam_vec) * target_trans[perm]
            
            # Patch spatial dropout
            if getattr(args, 'patch_dropout', 0) > 0:
                if pf is not None:
                    mask = (torch.rand(pf.shape[0], 1, pf.shape[2], pf.shape[3], 
                                       device=device) > args.patch_dropout).float()
                    pf = pf * mask
                if pc is not None:
                    mask = (torch.rand(pc.shape[0], 1, pc.shape[2], pc.shape[3],
                                       device=device) > args.patch_dropout).float()
                    pc = pc * mask
            
            # Feature dropout on summary
            if sm is not None and args.feature_dropout > 0:
                sm = F.dropout(sm, p=args.feature_dropout, training=True)
            
            trans_pred, rot_pred = model(pf, pc, sm)
        
        # Translation loss (configurable)
        use_hard_mining = (getattr(args, 'ohem_ratio', 0) > 0 or
                           getattr(args, 'focal_gamma', 0) > 0)
        
        if use_hard_mining:
            # ---- Per-sample loss for OHEM / Focal weighting ----
            # Translation: per-sample smooth_l1 (works for all loss types)
            per_trans = F.smooth_l1_loss(
                trans_pred, target_trans, reduction='none').mean(dim=-1)  # (B,)
            
            if args.trans_only:
                per_total = per_trans
                per_rot = torch.zeros_like(per_trans)
                beta_rot = torch.zeros(1, device=device)
            else:
                per_rot = geodesic_distance(rot_pred, target_rot)  # (B,)
                if args.rot_weight is not None:
                    beta_rot = torch.tensor(args.rot_weight, device=device)
                else:
                    beta_rot = torch.exp(model.s_rot)
                per_total = per_trans + beta_rot * per_rot
            
            if getattr(args, 'ohem_ratio', 0) > 0:
                # OHEM: keep only the hardest K% of samples
                k = max(1, int(per_total.shape[0] * args.ohem_ratio))
                topk_vals, _ = torch.topk(per_total, k)
                loss = topk_vals.mean()
            elif getattr(args, 'focal_gamma', 0) > 0:
                # Focal-style weighting: upweight hard samples
                with torch.no_grad():
                    # Normalize losses to [0,1] range for stable weighting
                    loss_max = per_total.max().clamp(min=1e-6)
                    normalized = per_total / loss_max
                    weights = (1.0 + normalized) ** args.focal_gamma
                    weights = weights / weights.sum()  # sum to 1
                loss = (weights * per_total).sum()
            
            # For logging (scalar means)
            loss_trans = per_trans.mean()
            loss_rot = per_rot.mean()
        else:
            # ---- Original scalar loss path ----
            trans_loss_fn = TRANS_LOSS_REGISTRY[args.trans_loss]
            loss_trans = trans_loss_fn(trans_pred, target_trans)
            
            if args.trans_only:
                loss = loss_trans
                loss_rot = torch.zeros(1, device=device)
                beta_rot = torch.zeros(1, device=device)
            else:
                loss_rot = geodesic_distance(rot_pred, target_rot).mean()
                if args.rot_weight is not None:
                    beta_rot = torch.tensor(args.rot_weight, device=device)
                else:
                    beta_rot = torch.exp(model.s_rot)
                loss = loss_trans + beta_rot * loss_rot
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                if use_precompute:
                    vt_pred, vr_pred = model.forward_from_pooled(test_pooled)
                else:
                    vt_pred, vr_pred = model(
                        test_data.patch_fine, test_data.patch_coarse, test_data.summary)
                vt_pred_real = test_data.denormalize(vt_pred, trans_mean, trans_std)
                trans_errors = (vt_pred_real - test_data.translations).norm(dim=-1)
                rot_errors = geodesic_distance(vr_pred, test_data.rotations) * 180 / math.pi
                
                val_trans_med = trans_errors.median().item()
                val_rot_med = rot_errors.median().item()
                val_score = val_trans_med + val_rot_med * 0.1
            
            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['loss_trans'].append(loss_trans.item())
            history['loss_rot'].append(loss_rot.item())
            history['val_rot_median'].append(val_rot_med)
            history['val_trans_median'].append(val_trans_med)
            history['beta_rot'].append(beta_rot.item())
            
            if epoch % (args.log_every * 10) == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                      f"(t={loss_trans.item():.4f} r={loss_rot.item():.4f} beta={beta_rot.item():.2f}) "
                      f"| val: {val_rot_med:.2f}deg/{val_trans_med*1000:.0f}mm "
                      f"| {elapsed:.1f}s")
            
            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_rot_median': val_rot_med,
                    'val_trans_median': val_trans_med,
                    'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
                    'config': config,
                }, out_dir / 'model_best.pt')
            
            # SWA: accumulate weight snapshots
            if swa_start > 0 and epoch >= swa_start and epoch % swa_freq == 0:
                current_state = {k: v.clone() for k, v in model.state_dict().items()}
                if swa_state is None:
                    swa_state = current_state
                    swa_count = 1
                else:
                    for k in swa_state:
                        swa_state[k] = (swa_state[k] * swa_count + current_state[k]) / (swa_count + 1)
                    swa_count += 1
    
    print(f"\nTraining done. Best epoch: {best_epoch}, score: {best_val_score:.4f}")
    
    # Save SWA model if available
    if swa_state is not None and swa_count > 1:
        print(f"SWA: averaged {swa_count} snapshots (start={swa_start}, freq={swa_freq})")
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': swa_state,
            'optimizer_state_dict': None,
            'val_rot_median': None,
            'val_trans_median': None,
            'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
            'config': config,
            'swa_count': swa_count,
        }, out_dir / 'model_swa.pt')
        print(f"  Saved SWA model to {out_dir / 'model_swa.pt'}")
    
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, train_data, test_data, trans_mean, trans_std, history, test_pooled


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, test_data, train_data, trans_mean, trans_std, out_dir,
             test_pooled=None):
    out_dir = Path(out_dir)
    model.eval()
    
    with torch.no_grad():
        if test_pooled is not None:
            trans_pred, rot_pred = model.forward_from_pooled(test_pooled)
        else:
            trans_pred, rot_pred = model(
                test_data.patch_fine, test_data.patch_coarse, test_data.summary)
        trans_pred_real = test_data.denormalize(trans_pred, trans_mean, trans_std)
        pred_pose_w2c = center_rot_to_w2c(trans_pred_real, rot_pred)
    
    trans_errors = (trans_pred_real - test_data.translations).norm(dim=-1).cpu().numpy()
    rot_errors = (geodesic_distance(rot_pred, test_data.rotations) * 180 / math.pi).cpu().numpy()
    
    results = {
        'rotation_deg': {
            'median': float(np.median(rot_errors)),
            'mean': float(np.mean(rot_errors)),
        },
        'translation_m': {
            'median': float(np.median(trans_errors)),
            'mean': float(np.mean(trans_errors)),
        },
        'translation_mm': {
            'median': float(np.median(trans_errors) * 1000),
            'mean': float(np.mean(trans_errors) * 1000),
        },
        'recall': {},
        'individual_pass': {},
    }
    
    thresholds = [
        ('5deg_1m', 5, 1.0),
        ('10deg_2m', 10, 2.0),
        ('15deg_5m', 15, 5.0),
        ('25deg_5m', 25, 5.0),
    ]
    for name, rot_th, trans_th in thresholds:
        rot_pass = (rot_errors < rot_th).mean() * 100
        trans_pass = (trans_errors < trans_th).mean() * 100
        combined = ((rot_errors < rot_th) & (trans_errors < trans_th)).mean() * 100
        results['recall'][name] = float(combined)
        results['individual_pass'][name] = {
            'rot_pass': float(rot_pass),
            'trans_pass': float(trans_pass),
            'combined': float(combined),
        }
    
    # Retrieval evaluation: use predicted center to find nearest training image
    if train_data is not None:
        train_positions = train_data.translations.cpu().numpy()  # (N_train, 3)
        pred_positions = trans_pred_real.cpu().numpy()  # (N_test, 3)
        
        # For each test image, find nearest training image by predicted position
        from scipy.spatial import cKDTree
        tree = cKDTree(train_positions)
        dists, nn_indices = tree.query(pred_positions, k=1)
        
        # Compute pose error of retrieved training image
        train_rot = train_data.rotations.cpu()
        test_rot = test_data.rotations.cpu()
        test_trans_np = test_data.translations.cpu().numpy()
        
        retr_trans_errors = np.linalg.norm(train_positions[nn_indices] - test_trans_np, axis=-1)
        
        retr_rot_pred = train_rot[nn_indices]
        retr_rot_errors = (geodesic_distance(retr_rot_pred, test_rot) * 180 / math.pi).numpy()
        
        results['retrieval'] = {}
        for name, rot_th, trans_th in thresholds:
            combined = ((retr_rot_errors < rot_th) & (retr_trans_errors < trans_th)).mean() * 100
            results['retrieval'][name] = {
                'combined': float(combined),
                'rot_pass': float((retr_rot_errors < rot_th).mean() * 100),
                'trans_pass': float((retr_trans_errors < trans_th).mean() * 100),
            }
        results['retrieval']['median_rot_deg'] = float(np.median(retr_rot_errors))
        results['retrieval']['median_trans_mm'] = float(np.median(retr_trans_errors) * 1000)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Direct Regression)")
    print("=" * 70)
    print(f"Rotation  (median): {results['rotation_deg']['median']:.2f}deg")
    print(f"Translation (median): {results['translation_mm']['median']:.0f}mm")
    print()
    for name, rot_th, trans_th in thresholds:
        ip = results['individual_pass'][name]
        print(f"  R@{name}: {results['recall'][name]:.1f}%  "
              f"(rot_pass={ip['rot_pass']:.1f}%, trans_pass={ip['trans_pass']:.1f}%)")
    
    if 'retrieval' in results:
        print(f"\n{'=' * 70}")
        print("EVALUATION RESULTS (Retrieval Mode)")
        print(f"{'=' * 70}")
        print(f"Retrieval median: {results['retrieval']['median_rot_deg']:.2f}deg / "
              f"{results['retrieval']['median_trans_mm']:.0f}mm")
        for name, rot_th, trans_th in thresholds:
            r = results['retrieval'][name]
            print(f"  R@{name}: {r['combined']:.1f}%  "
                  f"(rot={r['rot_pass']:.1f}%, trans={r['trans_pass']:.1f}%)")
    print("=" * 70)
    
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    np.savez(
        out_dir / 'test_poses_w2c.npz',
        img_names=np.array(test_data.names),
        camera_centers=trans_pred_real.cpu().numpy().astype(np.float32),
        poses_w2c=pred_pose_w2c.cpu().numpy().astype(np.float32),
    )

    return results, trans_errors, rot_errors, trans_pred_real.cpu().numpy()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Patch Token Pose Regression')
    parser.add_argument('--pool', type=str, default='gap',
                        choices=['gap', 'gem', 'spp', 'attn', 'conv'])
    parser.add_argument('--feat', type=str, default='both+sum',
                        choices=['fine', 'coarse', 'both', 'fine+sum', 'coarse+sum', 'both+sum'])
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--feature_dir', type=str,
                        default='output/feature_extract/features_radio_dual/OldHospital_pilot')
    parser.add_argument('--dataset_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--output_base', type=str,
                        default='output/feature_retrieval/pose_regression')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--feature_dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=895,
                        help='Batch size (895 = full batch for OldHospital train)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[1024, 512, 256])
    parser.add_argument('--log_every', type=int, default=10)
    
    # Pooling-specific
    parser.add_argument('--attn_heads', type=int, default=4)
    parser.add_argument('--conv_out_dim', type=int, default=512)
    parser.add_argument('--spp_levels', type=int, nargs='+', default=None,
                        help='Custom SPP levels (default: 1 2 4). E.g. --spp_levels 1 2 4 8')
    
    # Loss configuration
    parser.add_argument('--trans_only', action='store_true',
                        help='Train translation head only (freeze rotation loss)')
    parser.add_argument('--trans_loss', type=str, default='smooth_l1',
                        choices=['smooth_l1', 'log_cosh', 'wing', 'mse'],
                        help='Translation loss function')
    parser.add_argument('--rot_weight', type=float, default=None,
                        help='Fixed rotation loss weight (overrides learned beta)')
    
    # Data augmentation
    parser.add_argument('--mixup_alpha', type=float, default=0.0,
                        help='Mixup alpha (0 = disabled, 0.2-0.4 typical)')
    parser.add_argument('--patch_dropout', type=float, default=0.0,
                        help='Spatial patch dropout rate (0 = disabled)')
    parser.add_argument('--patch_dim', type=int, default=64,
                        help='Patch token dimension (64 for default PCA, 128/256 for higher)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--swa_start', type=int, default=0,
                        help='Epoch to start SWA averaging (0 = disabled)')
    parser.add_argument('--swa_freq', type=int, default=50,
                        help='Frequency of SWA snapshots (epochs)')
    parser.add_argument('--precompute_pool', action='store_true',
                        help='Pre-compute pooled features to avoid OOM with large patches')
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'residual'],
                        help='Model architecture: mlp (default) or residual')
    parser.add_argument('--n_res_blocks', type=int, default=4,
                        help='Number of residual blocks (for --model_type residual)')
    parser.add_argument('--trans_noise', type=float, default=0.0,
                        help='Gaussian noise std (meters) added to translation targets during training')
    parser.add_argument('--whiten', action='store_true',
                        help='Apply PCA whitening to pooled features (requires --precompute_pool)')
    parser.add_argument('--spatial_aug_k', type=int, default=0,
                        help='Spatial interpolation augmentation: number of nearest neighbors (0=disabled)')
    parser.add_argument('--spatial_aug_alpha', type=float, default=0.3,
                        help='Max interpolation weight for spatial augmentation')
    parser.add_argument('--spatial_aug_dist', type=float, default=5.0,
                        help='Max distance (meters) for spatial augmentation neighbors')
    parser.add_argument('--spatial_aug_prob', type=float, default=0.5,
                        help='Probability of applying spatial augmentation to each sample')
    # Hard example mining
    parser.add_argument('--ohem_ratio', type=float, default=0.0,
                        help='OHEM: keep only top-k%% hardest samples (0=disabled, 0.5=top 50%%)')
    parser.add_argument('--focal_gamma', type=float, default=0.0,
                        help='Focal loss gamma: weight samples by (1+loss)^gamma (0=disabled)')
    parser.add_argument('--curriculum_start', type=int, default=0,
                        help='Curriculum learning: start with easiest N%% of samples, increase gradually')
    
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_base, args.exp_name)
    
    print(f"{'=' * 70}")
    print(f"Experiment: {args.exp_name}")
    print(f"Pool: {args.pool}, Features: {args.feat}")
    print(f"{'=' * 70}")
    
    model, train_data, test_data, trans_mean, trans_std, history, test_pooled = train(args)
    
    # Load best model
    ckpt = torch.load(Path(args.output_dir) / 'model_best.pt',
                      map_location=f'cuda:{args.gpu}')
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"\nLoaded best model from epoch {ckpt['epoch']}")
    
    # Re-compute pooled features with best model weights if using precompute
    if getattr(args, 'precompute_pool', False) and test_pooled is not None:
        # Need to re-pool with best model's pooling layers (they don't have learnable params
        # for SPP/GAP, but AttentionPooling has learned params)
        if args.pool in ('attn', 'conv'):
            print("  Re-computing pooled features with best model weights...")
            # Reload raw features temporarily
            use_fine = 'fine' in args.feat or 'both' in args.feat
            use_coarse = 'coarse' in args.feat or 'both' in args.feat
            use_summary = '+sum' in args.feat
            test_data_reload = PatchPoseDataset(
                args.feature_dir, args.dataset_dir, 'test', 'cpu',
                use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
            device = next(model.parameters()).device
            test_pooled = precompute_pooled_features(model, test_data_reload, device, batch_size=16)
            del test_data_reload
    
    results, trans_errors, rot_errors, trans_pred = evaluate(
        model, test_data, train_data, trans_mean, trans_std, args.output_dir,
        test_pooled=test_pooled)

    bundle_metrics = {
        'experiment': args.exp_name,
        'best_epoch': int(ckpt['epoch']),
        'feature_dir': args.feature_dir,
        'dataset_dir': args.dataset_dir,
        'config': json.loads(Path(args.output_dir, 'config.json').read_text()),
        'results': results,
    }
    save_experiment_bundle(
        exp_name=args.exp_name,
        output_dir=args.output_dir,
        metrics=bundle_metrics,
        summary_lines=[
            f"direct median: {results['rotation_deg']['median']:.3f} deg / {results['translation_mm']['median']:.1f} mm",
            f"R@5deg/1m: {results['recall']['5deg_1m']:.2f}%",
            f"R@10deg/2m: {results['recall']['10deg_2m']:.2f}%",
            f"retrieval median: {results.get('retrieval', {}).get('median_rot_deg', float('nan')):.3f} deg / {results.get('retrieval', {}).get('median_trans_mm', float('nan')):.1f} mm",
        ],
        notes=[
            f"best_checkpoint_epoch={int(ckpt['epoch'])}",
            f"test_pose_cache={Path(args.output_dir) / 'test_poses_w2c.npz'}",
        ],
        artifact_paths=[
            Path(args.output_dir) / 'config.json',
            Path(args.output_dir) / 'training_log.json',
            Path(args.output_dir) / 'model_best.pt',
            Path(args.output_dir) / 'test_poses_w2c.npz',
        ],
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
