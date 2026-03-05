"""
Gaussian Feature Model
======================
加载预训练3DGS，冻结几何/外观参数，添加可学习的特征嵌入。
参考STDLoc的GaussianModel设计。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


class GaussianFeatureModel(nn.Module):
    """
    带特征嵌入的3DGS/2DGS模型。
    
    冻结几何参数 (xyz, rotation, scaling, opacity, SH)，
    仅优化每个Gaussian的特征嵌入向量。
    支持 3DGS (3 scales) 和 2DGS (2 scales, surfel) 两种模式。
    
    Attributes:
        _xyz: [N, 3] Gaussian中心位置 (frozen)
        _rotation: [N, 4] 旋转四元数 (frozen)
        _scaling: [N, 2或3] 缩放 log空间 (frozen, 2DGS为2维)
        _opacity: [N, 1] 不透明度 logit空间 (frozen)
        _features_dc: [N, 1, 3] SH DC系数 (frozen)
        _loc_feature: [N, D] 可学习的特征嵌入 (trainable)
        is_2dgs: bool, 是否为2DGS surfel模型
    """
    
    def __init__(self, feature_dim: int = 256):
        """
        Args:
            feature_dim: 每个Gaussian的特征嵌入维度
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.is_2dgs = False  # 加载PLY时根据scale维数自动设置
        
        # 冻结的3DGS几何/外观参数 (register_buffer不参与梯度)
        self.register_buffer('_xyz', torch.empty(0))
        self.register_buffer('_rotation', torch.empty(0))
        self.register_buffer('_scaling', torch.empty(0))
        self.register_buffer('_opacity', torch.empty(0))
        self.register_buffer('_features_dc', torch.empty(0))
        
        # 可学习的特征嵌入
        self._loc_feature = nn.Parameter(torch.empty(0))
        
        # 激活函数
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = F.normalize
        
    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]
    
    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz
    
    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation, dim=-1)
    
    @property
    def get_scaling(self) -> torch.Tensor:
        """返回激活后的缩放 [N, 2] (2DGS) 或 [N, 3] (3DGS)"""
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_for_render(self) -> torch.Tensor:
        """
        返回渲染用的 [N, 3] 缩放。
        2DGS: 在激活后 pad 第3维为 1.0 (与 STDLoc 一致)
        3DGS: 直接返回 [N, 3]
        """
        scales = self.scaling_activation(self._scaling)
        if self.is_2dgs:
            # 参考 STDLoc: pad ones 作为第3维 (surfel 不关心厚度)
            ones = torch.ones(scales.shape[0], 1, device=scales.device, dtype=scales.dtype)
            return torch.cat([scales, ones], dim=-1)
        return scales
    
    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)
    
    @property
    def get_loc_feature(self) -> torch.Tensor:
        """返回L2归一化后的特征嵌入 [N, D]"""
        return F.normalize(self._loc_feature, p=2, dim=-1)
    
    def load_ply(self, ply_path: str):
        """
        从PLY文件加载预训练的3DGS，冻结所有几何/外观参数。
        
        Args:
            ply_path: PLY文件路径
        """
        print(f"[GaussianFeatureModel] 加载PLY: {ply_path}")
        plydata = PlyData.read(ply_path)
        vertex = plydata.elements[0]
        
        N = vertex.count
        print(f"  Gaussian数量: {N}")
        
        # 加载xyz
        xyz = np.stack([
            np.asarray(vertex['x']),
            np.asarray(vertex['y']),
            np.asarray(vertex['z']),
        ], axis=1)  # [N, 3]
        
        # 加载opacity
        opacity = np.asarray(vertex['opacity'])[..., np.newaxis]  # [N, 1]
        
        # 加载SH DC (sh_degree=0)
        features_dc = np.zeros((N, 3))
        features_dc[:, 0] = np.asarray(vertex['f_dc_0'])
        features_dc[:, 1] = np.asarray(vertex['f_dc_1'])
        features_dc[:, 2] = np.asarray(vertex['f_dc_2'])
        
        # 加载scaling
        scale_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith('scale_')],
            key=lambda x: int(x.split('_')[-1])
        )
        scales = np.zeros((N, len(scale_names)))
        for idx, name in enumerate(scale_names):
            scales[:, idx] = np.asarray(vertex[name])
        
        # 检测 2DGS: 2D surfel 只有 2 个 scale
        if scales.shape[1] == 2:
            self.is_2dgs = True
            print(f"  [2DGS] 检测到 2 个 scale，使用 rasterization_2dgs 渲染")
        else:
            self.is_2dgs = False
        
        # 加载rotation
        rot_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith('rot')],
            key=lambda x: int(x.split('_')[-1])
        )
        rots = np.zeros((N, len(rot_names)))
        for idx, name in enumerate(rot_names):
            rots[:, idx] = np.asarray(vertex[name])
        
        # 存入buffer (frozen, 不参与梯度计算)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._xyz = torch.tensor(xyz, dtype=torch.float32, device=device)
        self._rotation = torch.tensor(rots, dtype=torch.float32, device=device)
        self._scaling = torch.tensor(scales, dtype=torch.float32, device=device)
        self._opacity = torch.tensor(opacity, dtype=torch.float32, device=device)
        self._features_dc = torch.tensor(features_dc, dtype=torch.float32, device=device)
        
        # 初始化可学习的特征嵌入: 随机初始化 + L2归一化 (参考STDLoc)
        loc_feature = torch.randn(N, self.feature_dim, device=device)
        loc_feature = F.normalize(loc_feature, p=2, dim=-1)
        self._loc_feature = nn.Parameter(loc_feature)
        
        print(f"  特征嵌入维度: {self.feature_dim}")
        print(f"  冻结参数: xyz, rotation, scaling, opacity, SH")
        print(f"  可训练参数: loc_feature [{N} x {self.feature_dim}]")
        print(f"  可训练参数量: {N * self.feature_dim:,}")
        
    def save_ply_with_features(self, path: str):
        """
        保存带特征嵌入的PLY文件。
        
        在原有3DGS属性基础上，追加 loc_0, loc_1, ..., loc_{D-1} 属性。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()
        loc_feat = self._loc_feature.detach().cpu().numpy()  # [N, D]
        
        # 构建属性列表
        attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity']
        attrs += [f'scale_{i}' for i in range(scales.shape[1])]
        attrs += [f'rot_{i}' for i in range(rots.shape[1])]
        attrs += [f'loc_{i}' for i in range(self.feature_dim)]
        
        dtype_full = [(a, 'f4') for a in attrs]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        data = np.concatenate([xyz, normals, f_dc, opacity, scales, rots, loc_feat], axis=1)
        elements[:] = list(map(tuple, data))
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"[GaussianFeatureModel] 保存PLY (含特征嵌入): {path}")
    
    def load_ply_with_features(self, ply_path: str):
        """
        加载带特征嵌入的PLY文件 (用于推理或继续训练)。
        """
        print(f"[GaussianFeatureModel] 加载PLY (含特征嵌入): {ply_path}")
        plydata = PlyData.read(ply_path)
        vertex = plydata.elements[0]
        N = vertex.count
        
        # 加载Gaussian属性 (同load_ply)
        xyz = np.stack([np.asarray(vertex['x']), np.asarray(vertex['y']), np.asarray(vertex['z'])], axis=1)
        opacity = np.asarray(vertex['opacity'])[..., np.newaxis]
        features_dc = np.zeros((N, 3))
        for i in range(3):
            features_dc[:, i] = np.asarray(vertex[f'f_dc_{i}'])
        
        scale_names = sorted([p.name for p in vertex.properties if p.name.startswith('scale_')], key=lambda x: int(x.split('_')[-1]))
        scales = np.stack([np.asarray(vertex[n]) for n in scale_names], axis=1)
        
        # 检测 2DGS: 2D surfel 只有 2 个 scale
        if scales.shape[1] == 2:
            self.is_2dgs = True
        else:
            self.is_2dgs = False
        
        rot_names = sorted([p.name for p in vertex.properties if p.name.startswith('rot')], key=lambda x: int(x.split('_')[-1]))
        rots = np.stack([np.asarray(vertex[n]) for n in rot_names], axis=1)
        
        # 加载特征嵌入
        loc_names = sorted([p.name for p in vertex.properties if p.name.startswith('loc_')], key=lambda x: int(x.split('_')[-1]))
        if loc_names:
            self.feature_dim = len(loc_names)
            loc_feat = np.stack([np.asarray(vertex[n]) for n in loc_names], axis=1)
        else:
            raise ValueError("PLY文件中未找到loc_*属性")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._xyz = torch.tensor(xyz, dtype=torch.float32, device=device)
        self._rotation = torch.tensor(rots, dtype=torch.float32, device=device)
        self._scaling = torch.tensor(scales, dtype=torch.float32, device=device)
        self._opacity = torch.tensor(opacity, dtype=torch.float32, device=device)
        self._features_dc = torch.tensor(features_dc, dtype=torch.float32, device=device)
        self._loc_feature = nn.Parameter(torch.tensor(loc_feat, dtype=torch.float32, device=device))
        
        print(f"  Gaussians: {N}, Feature dim: {self.feature_dim}")
